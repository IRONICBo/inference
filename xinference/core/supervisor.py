# Copyright 2022-2023 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import itertools
import os
import signal
import time
import typing
from dataclasses import dataclass
from logging import getLogger
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import xoscar as xo

from ..constants import (
    XINFERENCE_DEFAULT_CANCEL_BLOCK_DURATION,
    XINFERENCE_DISABLE_HEALTH_CHECK,
    XINFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD,
    XINFERENCE_HEALTH_CHECK_INTERVAL,
    XINFERENCE_HEALTH_CHECK_TIMEOUT,
)
from ..core.model import ModelActor, PDModelActor
from ..core.status_guard import InstanceInfo, LaunchStatus
from ..types import PeftModelConfig
from .metrics import record_metrics
from .resource import GPUStatus, ResourceStatus, Labels
from .utils import (
    assign_replica_gpu,
    build_replica_model_uid,
    gen_random_string,
    is_valid_model_uid,
    iter_replica_model_uid,
    log_async,
    log_sync,
    parse_model_version,
    parse_replica_model_uid,
)

if TYPE_CHECKING:
    from ..model.audio import AudioModelFamilyV1
    from ..model.embedding import EmbeddingModelSpec
    from ..model.flexible import FlexibleModelSpec
    from ..model.image import ImageModelFamilyV1
    from ..model.llm import LLMFamilyV1
    from ..model.rerank import RerankModelSpec
    from ..model.video import VideoModelFamilyV1
    from .worker import WorkerActor


logger = getLogger(__name__)


ASYNC_LAUNCH_TASKS = {}  # type: ignore


def callback_for_async_launch(model_uid: str):
    ASYNC_LAUNCH_TASKS.pop(model_uid, None)
    logger.debug(f"Model uid: {model_uid} async launch completes.")


@dataclass
class WorkerStatus:
    update_time: float
    failure_remaining_count: int
    status: Dict[str, Union[ResourceStatus, GPUStatus, Labels]]


@dataclass
class ReplicaInfo:
    replica: int
    scheduler: Iterator


class SupervisorActor(xo.StatelessActor):
    def __init__(self, role: Optional[str]="normal"):
        super().__init__()
        self._worker_address_to_worker: Dict[str, xo.ActorRefType["WorkerActor"]] = {}  # type: ignore
        self._worker_status: Dict[str, WorkerStatus] = {}  # type: ignore
        self._replica_model_uid_to_worker: Dict[  # type: ignore
            str, xo.ActorRefType["WorkerActor"]
        ] = {}
        self._model_uid_to_replica_info: Dict[str, ReplicaInfo] = {}  # type: ignore
        self._uptime = None
        self._lock = asyncio.Lock()
        self._role = role

    @classmethod
    def default_uid(cls) -> str:
        return "supervisor"

    def _get_worker_ref_by_ip(
        self, ip: str
    ) -> Optional[xo.ActorRefType["WorkerActor"]]:
        for addr, ref in self._worker_address_to_worker.items():
            existing_ip = addr.split(":")[0]
            if existing_ip == ip:
                return ref
        return None

    async def __post_create__(self):
        self._uptime = time.time()
        if not XINFERENCE_DISABLE_HEALTH_CHECK:
            # Run _check_dead_nodes() in a dedicated thread.
            from ..isolation import Isolation

            self._isolation = Isolation(asyncio.new_event_loop(), threaded=True)
            self._isolation.start()
            asyncio.run_coroutine_threadsafe(
                self._check_dead_nodes(), loop=self._isolation.loop
            )
        logger.info(f"Xinference supervisor {self.address} started with role: {self._role}.")
        from .cache_tracker import CacheTrackerActor
        from .progress_tracker import ProgressTrackerActor
        from .status_guard import StatusGuardActor

        self._status_guard_ref: xo.ActorRefType[  # type: ignore
            "StatusGuardActor"
        ] = await xo.create_actor(
            StatusGuardActor, address=self.address, uid=StatusGuardActor.default_uid()
        )
        self._cache_tracker_ref: xo.ActorRefType[  # type: ignore
            "CacheTrackerActor"
        ] = await xo.create_actor(
            CacheTrackerActor, address=self.address, uid=CacheTrackerActor.default_uid()
        )
        self._progress_tracker: xo.ActorRefType[  # type: ignore
            "ProgressTrackerActor"
        ] = await xo.create_actor(
            ProgressTrackerActor,
            address=self.address,
            uid=ProgressTrackerActor.default_uid(),
        )

        from .event import EventCollectorActor

        self._event_collector_ref: xo.ActorRefType[  # type: ignore
            EventCollectorActor
        ] = await xo.create_actor(
            EventCollectorActor,
            address=self.address,
            uid=EventCollectorActor.default_uid(),
        )

        from ..model.audio import (
            CustomAudioModelFamilyV1,
            generate_audio_description,
            get_audio_model_descriptions,
            register_audio,
            unregister_audio,
        )
        from ..model.embedding import (
            CustomEmbeddingModelSpec,
            generate_embedding_description,
            get_embedding_model_descriptions,
            register_embedding,
            unregister_embedding,
        )
        from ..model.flexible import (
            FlexibleModelSpec,
            generate_flexible_model_description,
            get_flexible_model_descriptions,
            register_flexible_model,
            unregister_flexible_model,
        )
        from ..model.image import (
            CustomImageModelFamilyV1,
            generate_image_description,
            get_image_model_descriptions,
            register_image,
            unregister_image,
        )
        from ..model.llm import (
            CustomLLMFamilyV1,
            generate_llm_description,
            get_llm_model_descriptions,
            register_llm,
            unregister_llm,
        )
        from ..model.rerank import (
            CustomRerankModelSpec,
            generate_rerank_description,
            get_rerank_model_descriptions,
            register_rerank,
            unregister_rerank,
        )

        self._custom_register_type_to_cls: Dict[str, Tuple] = {  # type: ignore
            "LLM": (
                CustomLLMFamilyV1,
                register_llm,
                unregister_llm,
                generate_llm_description,
            ),
            "embedding": (
                CustomEmbeddingModelSpec,
                register_embedding,
                unregister_embedding,
                generate_embedding_description,
            ),
            "rerank": (
                CustomRerankModelSpec,
                register_rerank,
                unregister_rerank,
                generate_rerank_description,
            ),
            "image": (
                CustomImageModelFamilyV1,
                register_image,
                unregister_image,
                generate_image_description,
            ),
            "audio": (
                CustomAudioModelFamilyV1,
                register_audio,
                unregister_audio,
                generate_audio_description,
            ),
            "flexible": (
                FlexibleModelSpec,
                register_flexible_model,
                unregister_flexible_model,
                generate_flexible_model_description,
            ),
        }

        # record model version
        model_version_infos: Dict[str, List[Dict]] = {}  # type: ignore
        model_version_infos.update(get_llm_model_descriptions())
        model_version_infos.update(get_embedding_model_descriptions())
        model_version_infos.update(get_rerank_model_descriptions())
        model_version_infos.update(get_image_model_descriptions())
        model_version_infos.update(get_audio_model_descriptions())
        model_version_infos.update(get_flexible_model_descriptions())
        await self._cache_tracker_ref.record_model_version(
            model_version_infos, self.address
        )

        # Windows does not have signal handler
        if os.name != "nt":

            async def signal_handler():
                os._exit(0)

            loop = asyncio.get_running_loop()
            loop.add_signal_handler(
                signal.SIGTERM, lambda: asyncio.create_task(signal_handler())
            )

        from ..model.llm.vllm.xavier.block_tracker import VLLMBlockTracker
        from ..model.llm.vllm.xavier.collective_manager import CollectiveManager

        self._block_tracker_mapping: Dict[str, xo.ActorRefType[VLLMBlockTracker]] = {}
        self._collective_manager_mapping: Dict[
            str, xo.ActorRefType[CollectiveManager]
        ] = {}

    @typing.no_type_check
    async def get_cluster_device_info(self, detailed: bool = False) -> List:
        import psutil

        supervisor_device_info = {
            "ip_address": self.address.split(":")[0],
            "gpu_count": 0,
            "gpu_vram_total": 0,
            # Cluster only contains one role.
            "role": self._role,
        }
        if detailed:
            supervisor_device_info["gpu_vram_total"] = 0
            supervisor_device_info["gpu_vram_available"] = 0
            supervisor_device_info["cpu_available"] = psutil.cpu_count() * (
                1 - psutil.cpu_percent() / 100.0
            )
            supervisor_device_info["cpu_count"] = psutil.cpu_count()
            mem_info = psutil.virtual_memory()
            supervisor_device_info["mem_used"] = mem_info.used
            supervisor_device_info["mem_available"] = mem_info.available
            supervisor_device_info["mem_total"] = mem_info.total
        res = [{"node_type": "Supervisor", **supervisor_device_info}]
        for worker_addr, worker_status in self._worker_status.items():
            vram_total: float = sum(
                [v.mem_total for k, v in worker_status.status.items() if k != "cpu" and k != "labels"]  # type: ignore
            )
            total = (
                vram_total if vram_total == 0 else f"{int(vram_total / 1024 / 1024)}MiB"
            )
            info = {
                "node_type": "Worker",
                "ip_address": worker_addr.split(":")[0],
                "gpu_count": len(worker_status.status) - 2,
                "gpu_vram_total": total,
            }
            if detailed:
                cpu_info = worker_status.status["cpu"]
                info["cpu_available"] = cpu_info.total * (1 - cpu_info.usage)
                info["cpu_count"] = cpu_info.total
                info["mem_used"] = cpu_info.memory_used
                info["mem_available"] = cpu_info.memory_available
                info["mem_total"] = cpu_info.memory_total
                info["gpu_vram_total"] = vram_total
                info["role"] = worker_status.status["labels"].role
                info["gpu_vram_available"] = sum(
                    [v.mem_free for k, v in worker_status.status.items() if k != "cpu" and k != "labels"]
                )
            res.append(info)
        return res

    @staticmethod
    async def get_builtin_prompts() -> Dict[str, Any]:
        from ..model.llm.llm_family import BUILTIN_LLM_PROMPT_STYLE

        return {k: v for k, v in BUILTIN_LLM_PROMPT_STYLE.items()}

    @staticmethod
    async def get_builtin_families() -> Dict[str, List[str]]:
        from ..model.llm.llm_family import (
            BUILTIN_LLM_FAMILIES,
            BUILTIN_LLM_MODEL_CHAT_FAMILIES,
            BUILTIN_LLM_MODEL_GENERATE_FAMILIES,
            BUILTIN_LLM_MODEL_TOOL_CALL_FAMILIES,
        )

        return {
            "chat": list(BUILTIN_LLM_MODEL_CHAT_FAMILIES),
            "generate": list(BUILTIN_LLM_MODEL_GENERATE_FAMILIES),
            "tools": list(BUILTIN_LLM_MODEL_TOOL_CALL_FAMILIES),
            "vision": [
                family.model_name
                for family in BUILTIN_LLM_FAMILIES
                if "vision" in family.model_ability
            ],
        }

    async def get_devices_count(self) -> int:
        from ..device_utils import gpu_count

        if self.is_local_deployment():
            return gpu_count()
        # distributed deployment, choose a worker and return its device_count.
        # Assume that each worker has the same count of cards.
        worker_ref = await self._choose_worker()
        return await worker_ref.get_devices_count()

    async def _choose_worker(self, role: Optional[str] = None) -> xo.ActorRefType["WorkerActor"]:
        # TODO: better allocation strategy.
        min_running_model_count = None
        target_worker = None

        workers = list(self._worker_address_to_worker.values())
        logger.debug(f"Choose worker: {workers}, status: {self._worker_status}")
        for worker in workers:
            if role is not None:
                worker_status = self._worker_status.get(worker.address)
                if worker_status.status['labels'].role == role:
                    target_worker = worker
            else:
                running_model_count = await worker.get_model_count()
                if (
                    min_running_model_count is None
                    or running_model_count < min_running_model_count
                ):
                    min_running_model_count = running_model_count
                    target_worker = worker

        if target_worker:
            logger.debug(f"Found available worker {target_worker.address} role {role}")
            return target_worker

        raise RuntimeError("No available worker found")

    @log_sync(logger=logger)
    def get_status(self) -> Dict:
        return {
            "uptime": int(time.time() - self._uptime),
            "workers": self._worker_status,
        }

    async def _to_llm_reg(
        self, llm_family: "LLMFamilyV1", is_builtin: bool
    ) -> Dict[str, Any]:
        from ..model.llm import get_cache_status

        instance_cnt = await self.get_instance_count(llm_family.model_name)
        version_cnt = await self.get_model_version_count(llm_family.model_name)

        if self.is_local_deployment():
            specs = []
            # TODO: does not work when the supervisor and worker are running on separate nodes.
            for spec in llm_family.model_specs:
                cache_status = get_cache_status(llm_family, spec)
                specs.append({**spec.dict(), "cache_status": cache_status})
            res = {**llm_family.dict(), "is_builtin": is_builtin, "model_specs": specs}
        else:
            res = {**llm_family.dict(), "is_builtin": is_builtin}
        res["model_version_count"] = version_cnt
        res["model_instance_count"] = instance_cnt
        return res

    async def _to_embedding_model_reg(
        self, model_spec: "EmbeddingModelSpec", is_builtin: bool
    ) -> Dict[str, Any]:
        from ..model.embedding import get_cache_status

        instance_cnt = await self.get_instance_count(model_spec.model_name)
        version_cnt = await self.get_model_version_count(model_spec.model_name)

        if self.is_local_deployment():
            # TODO: does not work when the supervisor and worker are running on separate nodes.
            cache_status = get_cache_status(model_spec)
            res = {
                **model_spec.dict(),
                "cache_status": cache_status,
                "is_builtin": is_builtin,
            }
        else:
            res = {
                **model_spec.dict(),
                "is_builtin": is_builtin,
            }
        res["model_version_count"] = version_cnt
        res["model_instance_count"] = instance_cnt
        return res

    async def _to_rerank_model_reg(
        self, model_spec: "RerankModelSpec", is_builtin: bool
    ) -> Dict[str, Any]:
        from ..model.rerank import get_cache_status

        instance_cnt = await self.get_instance_count(model_spec.model_name)
        version_cnt = await self.get_model_version_count(model_spec.model_name)

        if self.is_local_deployment():
            # TODO: does not work when the supervisor and worker are running on separate nodes.
            cache_status = get_cache_status(model_spec)
            res = {
                **model_spec.dict(),
                "cache_status": cache_status,
                "is_builtin": is_builtin,
            }
        else:
            res = {
                **model_spec.dict(),
                "is_builtin": is_builtin,
            }
        res["model_version_count"] = version_cnt
        res["model_instance_count"] = instance_cnt
        return res

    async def _to_image_model_reg(
        self, model_family: "ImageModelFamilyV1", is_builtin: bool
    ) -> Dict[str, Any]:
        from ..model.image import get_cache_status

        instance_cnt = await self.get_instance_count(model_family.model_name)
        version_cnt = await self.get_model_version_count(model_family.model_name)

        if self.is_local_deployment():
            # TODO: does not work when the supervisor and worker are running on separate nodes.
            cache_status = get_cache_status(model_family)
            res = {
                **model_family.dict(),
                "cache_status": cache_status,
                "is_builtin": is_builtin,
            }
        else:
            res = {
                **model_family.dict(),
                "is_builtin": is_builtin,
            }
        res["model_version_count"] = version_cnt
        res["model_instance_count"] = instance_cnt
        return res

    async def _to_audio_model_reg(
        self, model_family: "AudioModelFamilyV1", is_builtin: bool
    ) -> Dict[str, Any]:
        from ..model.audio import get_cache_status

        instance_cnt = await self.get_instance_count(model_family.model_name)
        version_cnt = await self.get_model_version_count(model_family.model_name)

        if self.is_local_deployment():
            # TODO: does not work when the supervisor and worker are running on separate nodes.
            cache_status = get_cache_status(model_family)
            res = {
                **model_family.dict(),
                "cache_status": cache_status,
                "is_builtin": is_builtin,
            }
        else:
            res = {
                **model_family.dict(),
                "is_builtin": is_builtin,
            }
        res["model_version_count"] = version_cnt
        res["model_instance_count"] = instance_cnt
        return res

    async def _to_video_model_reg(
        self, model_family: "VideoModelFamilyV1", is_builtin: bool
    ) -> Dict[str, Any]:
        from ..model.video import get_cache_status

        instance_cnt = await self.get_instance_count(model_family.model_name)
        version_cnt = await self.get_model_version_count(model_family.model_name)

        if self.is_local_deployment():
            # TODO: does not work when the supervisor and worker are running on separate nodes.
            cache_status = get_cache_status(model_family)
            res = {
                **model_family.dict(),
                "cache_status": cache_status,
                "is_builtin": is_builtin,
            }
        else:
            res = {
                **model_family.dict(),
                "is_builtin": is_builtin,
            }
        res["model_version_count"] = version_cnt
        res["model_instance_count"] = instance_cnt
        return res

    async def _to_flexible_model_reg(
        self, model_spec: "FlexibleModelSpec", is_builtin: bool
    ) -> Dict[str, Any]:
        instance_cnt = await self.get_instance_count(model_spec.model_name)
        version_cnt = await self.get_model_version_count(model_spec.model_name)

        if self.is_local_deployment():
            res = {
                **model_spec.dict(),
                "cache_status": True,
                "is_builtin": is_builtin,
            }
        else:
            res = {
                **model_spec.dict(),
                "is_builtin": is_builtin,
            }
        res["model_version_count"] = version_cnt
        res["model_instance_count"] = instance_cnt
        return res

    @log_async(logger=logger)
    async def list_model_registrations(
        self, model_type: str, detailed: bool = False
    ) -> List[Dict[str, Any]]:
        def sort_helper(item):
            assert isinstance(item["model_name"], str)
            return item.get("model_name").lower()

        ret = []
        if not self.is_local_deployment():
            workers = list(self._worker_address_to_worker.values())
            for worker in workers:
                ret.extend(await worker.list_model_registrations(model_type, detailed))

        if model_type == "LLM":
            from ..model.llm import BUILTIN_LLM_FAMILIES, get_user_defined_llm_families

            for family in BUILTIN_LLM_FAMILIES:
                if detailed:
                    ret.append(await self._to_llm_reg(family, True))
                else:
                    ret.append({"model_name": family.model_name, "is_builtin": True})

            for family in get_user_defined_llm_families():
                if detailed:
                    ret.append(await self._to_llm_reg(family, False))
                else:
                    ret.append({"model_name": family.model_name, "is_builtin": False})

            ret.sort(key=sort_helper)
            return ret
        elif model_type == "embedding":
            from ..model.embedding import BUILTIN_EMBEDDING_MODELS
            from ..model.embedding.custom import get_user_defined_embeddings

            for model_name, family in BUILTIN_EMBEDDING_MODELS.items():
                if detailed:
                    ret.append(
                        await self._to_embedding_model_reg(family, is_builtin=True)
                    )
                else:
                    ret.append({"model_name": model_name, "is_builtin": True})

            for model_spec in get_user_defined_embeddings():
                if detailed:
                    ret.append(
                        await self._to_embedding_model_reg(model_spec, is_builtin=False)
                    )
                else:
                    ret.append(
                        {"model_name": model_spec.model_name, "is_builtin": False}
                    )

            ret.sort(key=sort_helper)
            return ret
        elif model_type == "image":
            from ..model.image import BUILTIN_IMAGE_MODELS
            from ..model.image.custom import get_user_defined_images

            for model_name, family in BUILTIN_IMAGE_MODELS.items():
                if detailed:
                    ret.append(await self._to_image_model_reg(family, is_builtin=True))
                else:
                    ret.append({"model_name": model_name, "is_builtin": True})

            for model_spec in get_user_defined_images():
                if detailed:
                    ret.append(
                        await self._to_image_model_reg(model_spec, is_builtin=False)
                    )
                else:
                    ret.append(
                        {"model_name": model_spec.model_name, "is_builtin": False}
                    )

            ret.sort(key=sort_helper)
            return ret
        elif model_type == "audio":
            from ..model.audio import BUILTIN_AUDIO_MODELS
            from ..model.audio.custom import get_user_defined_audios

            for model_name, family in BUILTIN_AUDIO_MODELS.items():
                if detailed:
                    ret.append(await self._to_audio_model_reg(family, is_builtin=True))
                else:
                    ret.append({"model_name": model_name, "is_builtin": True})

            for model_spec in get_user_defined_audios():
                if detailed:
                    ret.append(
                        await self._to_audio_model_reg(model_spec, is_builtin=False)
                    )
                else:
                    ret.append(
                        {"model_name": model_spec.model_name, "is_builtin": False}
                    )

            ret.sort(key=sort_helper)
            return ret
        elif model_type == "video":
            from ..model.video import BUILTIN_VIDEO_MODELS

            for model_name, family in BUILTIN_VIDEO_MODELS.items():
                if detailed:
                    ret.append(await self._to_video_model_reg(family, is_builtin=True))
                else:
                    ret.append({"model_name": model_name, "is_builtin": True})

            ret.sort(key=sort_helper)
            return ret
        elif model_type == "rerank":
            from ..model.rerank import BUILTIN_RERANK_MODELS
            from ..model.rerank.custom import get_user_defined_reranks

            for model_name, family in BUILTIN_RERANK_MODELS.items():
                if detailed:
                    ret.append(await self._to_rerank_model_reg(family, is_builtin=True))
                else:
                    ret.append({"model_name": model_name, "is_builtin": True})

            for model_spec in get_user_defined_reranks():
                if detailed:
                    ret.append(
                        await self._to_rerank_model_reg(model_spec, is_builtin=False)
                    )
                else:
                    ret.append(
                        {"model_name": model_spec.model_name, "is_builtin": False}
                    )

            ret.sort(key=sort_helper)
            return ret
        elif model_type == "flexible":
            from ..model.flexible import get_flexible_models

            ret = []

            for model_spec in get_flexible_models():
                if detailed:
                    ret.append(
                        await self._to_flexible_model_reg(model_spec, is_builtin=False)
                    )
                else:
                    ret.append(
                        {"model_name": model_spec.model_name, "is_builtin": False}
                    )

            ret.sort(key=sort_helper)
            return ret
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @log_sync(logger=logger)
    async def get_model_registration(self, model_type: str, model_name: str) -> Any:
        # search in worker first
        if not self.is_local_deployment():
            workers = list(self._worker_address_to_worker.values())
            for worker in workers:
                f = await worker.get_model_registration(model_type, model_name)
                if f is not None:
                    return f

        if model_type == "LLM":
            from ..model.llm import BUILTIN_LLM_FAMILIES, get_user_defined_llm_families

            for f in BUILTIN_LLM_FAMILIES + get_user_defined_llm_families():
                if f.model_name == model_name:
                    return f

            raise ValueError(f"Model {model_name} not found")
        elif model_type == "embedding":
            from ..model.embedding import BUILTIN_EMBEDDING_MODELS
            from ..model.embedding.custom import get_user_defined_embeddings

            for f in (
                list(BUILTIN_EMBEDDING_MODELS.values()) + get_user_defined_embeddings()
            ):
                if f.model_name == model_name:
                    return f
            raise ValueError(f"Model {model_name} not found")
        elif model_type == "image":
            from ..model.image import BUILTIN_IMAGE_MODELS
            from ..model.image.custom import get_user_defined_images

            for f in list(BUILTIN_IMAGE_MODELS.values()) + get_user_defined_images():
                if f.model_name == model_name:
                    return f
            raise ValueError(f"Model {model_name} not found")
        elif model_type == "audio":
            from ..model.audio import BUILTIN_AUDIO_MODELS
            from ..model.audio.custom import get_user_defined_audios

            for f in list(BUILTIN_AUDIO_MODELS.values()) + get_user_defined_audios():
                if f.model_name == model_name:
                    return f
            raise ValueError(f"Model {model_name} not found")
        elif model_type == "rerank":
            from ..model.rerank import BUILTIN_RERANK_MODELS
            from ..model.rerank.custom import get_user_defined_reranks

            for f in list(BUILTIN_RERANK_MODELS.values()) + get_user_defined_reranks():
                if f.model_name == model_name:
                    return f
            raise ValueError(f"Model {model_name} not found")
        elif model_type == "flexible":
            from ..model.flexible import get_flexible_models

            for f in get_flexible_models():
                if f.model_name == model_name:
                    return f
            raise ValueError(f"Model {model_name} not found")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @log_async(logger=logger)
    async def query_engines_by_model_name(self, model_name: str):
        from copy import deepcopy

        from ..model.llm.llm_family import LLM_ENGINES

        # search in worker first
        workers = list(self._worker_address_to_worker.values())
        for worker in workers:
            res = await worker.query_engines_by_model_name(model_name)
            if res is not None:
                return res

        if model_name not in LLM_ENGINES:
            raise ValueError(f"Model {model_name} not found")

        # filter llm_class
        engine_params = deepcopy(LLM_ENGINES[model_name])
        for engine in engine_params:
            params = engine_params[engine]
            for param in params:
                del param["llm_class"]

        return engine_params

    @log_async(logger=logger)
    async def register_model(
        self,
        model_type: str,
        model: str,
        persist: bool,
        worker_ip: Optional[str] = None,
    ):
        if model_type in self._custom_register_type_to_cls:
            (
                model_spec_cls,
                register_fn,
                unregister_fn,
                generate_fn,
            ) = self._custom_register_type_to_cls[model_type]

            target_ip_worker_ref = (
                self._get_worker_ref_by_ip(worker_ip) if worker_ip is not None else None
            )
            if (
                worker_ip is not None
                and not self.is_local_deployment()
                and target_ip_worker_ref is None
            ):
                raise ValueError(
                    f"Worker ip address {worker_ip} is not in the cluster."
                )

            if target_ip_worker_ref:
                await target_ip_worker_ref.register_model(model_type, model, persist)
                return

            model_spec = model_spec_cls.parse_raw(model)
            try:
                register_fn(model_spec, persist)
                await self._cache_tracker_ref.record_model_version(
                    generate_fn(model_spec), self.address
                )
            except ValueError as e:
                raise e
            except Exception as e:
                unregister_fn(model_spec.model_name, raise_error=False)
                raise e
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @log_async(logger=logger)
    async def unregister_model(self, model_type: str, model_name: str):
        if model_type in self._custom_register_type_to_cls:
            _, _, unregister_fn, _ = self._custom_register_type_to_cls[model_type]
            unregister_fn(model_name, False)

            if not self.is_local_deployment():
                workers = list(self._worker_address_to_worker.values())
                for worker in workers:
                    await worker.unregister_model(model_type, model_name)

            await self._cache_tracker_ref.unregister_model_version(model_name)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _gen_model_uid(self, model_name: str) -> str:
        if model_name not in self._model_uid_to_replica_info:
            return model_name
        logger.debug(
            f"{model_name} exists in xinference. Generate suffix to {model_name} for model_uid."
        )
        return f"{model_name}-{gen_random_string(8)}"

    async def get_model_versions(self, model_type: str, model_name: str) -> List[Dict]:
        return await self._cache_tracker_ref.get_model_versions(model_name)

    async def get_model_version_count(self, model_name: str) -> int:
        return await self._cache_tracker_ref.get_model_version_count(model_name)

    @log_async(logger=logger)
    async def launch_model_by_version(
        self,
        model_uid: Optional[str],
        model_type: str,
        model_engine: Optional[str],
        model_version: str,
        replica: int = 1,
        prefill_replica: Optional[int] = None,
        decode_replica: Optional[int] = None,
        n_gpu: Optional[Union[int, str]] = "auto",
        wait_ready: bool = True,
    ):
        parse_results = parse_model_version(model_version, model_type)

        if model_type == "image" and len(parse_results) == 2:
            kwargs = {"controlnet": parse_results[1]}
        else:
            kwargs = {}

        return await self.launch_builtin_model(
            model_uid=model_uid,
            model_name=parse_results[0],
            model_engine=model_engine,
            model_size_in_billions=parse_results[1] if model_type == "LLM" else None,
            model_format=parse_results[2] if model_type == "LLM" else None,
            quantization=parse_results[3] if model_type == "LLM" else None,
            model_type=model_type,
            replica=replica,
            n_gpu=n_gpu,
            prefill_replica=prefill_replica,
            decode_replica=decode_replica,
            wait_ready=wait_ready,
            model_version=model_version,
            **kwargs,
        )

    async def launch_builtin_model(
        self,
        model_uid: Optional[str],
        model_name: str,
        model_size_in_billions: Optional[Union[int, str]],
        model_format: Optional[str],
        quantization: Optional[str],
        model_engine: Optional[str],
        model_type: Optional[str],
        replica: int = 1,
        prefill_replica: Optional[int] = None,
        decode_replica: Optional[int] = None,
        n_gpu: Optional[Union[int, str]] = "auto",
        request_limits: Optional[int] = None,
        wait_ready: bool = True,
        model_version: Optional[str] = None,
        peft_model_config: Optional[PeftModelConfig] = None,
        worker_ip: Optional[str] = None,
        gpu_idx: Optional[Union[int, List[int]]] = None,
        download_hub: Optional[Literal["huggingface", "modelscope", "csghub"]] = None,
        model_path: Optional[str] = None,
        **kwargs,
    ) -> str:
        # search in worker first
        if not self.is_local_deployment():
            workers = list(self._worker_address_to_worker.values())
            logger.debug(
                f"Searching model {model_name} in worker {worker_ip} if exists."
            )
            for worker in workers:
                res = await worker.get_model_registration(model_type, model_name)
                if res is not None:
                    worker_ip = worker.address.split(":")[0]

        target_ip_worker_ref = (
            self._get_worker_ref_by_ip(worker_ip) if worker_ip is not None else None
        )
        if (
            worker_ip is not None
            and not self.is_local_deployment()
            and target_ip_worker_ref is None
        ):
            raise ValueError(f"Worker ip address {worker_ip} is not in the cluster.")
        if worker_ip is not None and self.is_local_deployment():
            logger.warning(
                f"You specified the worker ip: {worker_ip} in local mode, "
                f"xinference will ignore this option."
            )

        if kwargs.get("enable_tensorizer", None) and (
            (
                model_engine is None
                or model_engine.lower() != "transformers"
                or model_format != "pytorch"
                or quantization != "none"
                or model_type != "LLM"
            )
        ):
            raise ValueError(
                "Tensorizer can only be enabled for LLM models with Transformers engine, PyTorch format, and none quantization."
            )

        if kwargs.get("enable_tensorizer", None) and model_name in [
            "OmniLMM",
            "yi-vl-chat",
            "deepseek-vl-chat",
        ]:
            raise ValueError("Tensorizer is not supported for %s." % model_name)

        if model_uid is None:
            model_uid = self._gen_model_uid(model_name)

        # Xavier-related
        enable_xavier: bool = (
            bool(kwargs.pop("enable_xavier", False))
            and model_engine is not None
            and model_engine.lower() == "vllm"
        )

        # Disaggregated-related
        enable_disagg: bool = (
            self._role == "disaggregated"
            and prefill_replica is not None
            and decode_replica is not None
            and enable_xavier
        )

        store_address = None
        store_port = None
        world_size = None
        if enable_xavier:
            if replica <= 1 and self._role != "disaggregated":
                logger.warning(f"Enabling xavier when `replica<=1` is meaningless.")
                enable_xavier = False
            else:
                from ..model.llm.vllm.xavier.block_tracker import VLLMBlockTracker
                from ..model.llm.vllm.xavier.collective_manager import CollectiveManager

                self._block_tracker_mapping[model_uid] = await xo.create_actor(
                    VLLMBlockTracker,
                    address=self.address,
                    uid=f"{VLLMBlockTracker.default_uid()}-{model_uid}",
                )

                #  TODO: this will be removed and replaced by a more general solution,
                # current default is 3 for 2 worker with 1p1d.
                if self._role == "disaggregated":
                    # for 1 p and 1 d, in each one replica
                    world_size = prefill_replica + decode_replica + 1
                else:
                    world_size = replica + 1
                logger.info(f"Going to start xavier with world size: {world_size}")
                self._collective_manager_mapping[model_uid] = await xo.create_actor(
                    CollectiveManager,
                    address=self.address,
                    uid=f"{CollectiveManager.default_uid()}-{model_uid}",
                    model_uid=model_uid,
                )
                logger.info(f"Start collective manager for {model_uid} done.")

        model_size = str(model_size_in_billions) if model_size_in_billions else ""
        logger.debug(
            f"Enter launch_builtin_model, model_uid: {model_uid}, model_name: {model_name}, model_size: {model_size}, "
            f"model_format: {model_format}, quantization: {quantization}, replica: {replica}, enable_xavier: {enable_xavier}, "
            f"enable_disagg: {enable_disagg} kwargs: {kwargs}"
        )

        async def _launch_one_model(worker_ref, _replica_model_uid, rank: int):
            if _replica_model_uid in self._replica_model_uid_to_worker:
                raise ValueError(
                    f"Model is already in the model list, uid: {_replica_model_uid}"
                )

            nonlocal store_address
            nonlocal store_port
            xavier_config = (
                {
                    "block_tracker_uid": self._block_tracker_mapping[model_uid].uid,
                    "block_tracker_address": self._block_tracker_mapping[
                        model_uid
                    ].address,
                    "rank": rank,
                    "world_size": world_size,
                    "store_address": store_address,
                    "store_port": store_port,
                }
                if enable_xavier
                else None
            )

            if enable_xavier and rank == 0:
                rank0_address, _port = await worker_ref.launch_rank0_model(
                    _replica_model_uid, xavier_config
                )
                self._replica_model_uid_to_worker[_replica_model_uid] = worker_ref
                store_address = rank0_address.split(":")[0]
                store_port = _port
                return rank0_address

            replica_gpu_idx = assign_replica_gpu(_replica_model_uid, replica, gpu_idx)
            nonlocal model_type

            # LLM as default for compatibility
            model_type = model_type or "LLM"
            subpool_address = await worker_ref.launch_builtin_model(
                model_uid=_replica_model_uid,
                model_name=model_name,
                model_size_in_billions=model_size_in_billions,
                model_format=model_format,
                quantization=quantization,
                model_engine=model_engine,
                model_type=model_type,
                n_gpu=n_gpu,
                request_limits=request_limits,
                peft_model_config=peft_model_config,
                gpu_idx=replica_gpu_idx,
                download_hub=download_hub,
                model_path=model_path,
                xavier_config=xavier_config,
                **kwargs,
            )
            self._replica_model_uid_to_worker[_replica_model_uid] = worker_ref
            logger.info(
                f"Launch builtin model {self._replica_model_uid_to_worker}."
            )
            return subpool_address

        async def _launch_model():
            try:
                worker_refs = []
                rank_addresses = []
                if enable_disagg:
                    logger.info(f"Starting disagg models for {model_uid} prefill replica num: {prefill_replica} decode replica num: {decode_replica}")
                    # Start prefill models, with extra replica for collective communication.
                    # The rank 0 replica is used for collective communication.
                    # For prefill and decode replicas, we need to count the index by replica number,
                    # and allocate the correct gpu index for each replica.
                    # For example, if prefill replica is 2, decode replica is 2, total replica is 4.
                    for rank, rep_model_uid in enumerate(
                        iter_replica_model_uid(model_uid, prefill_replica, start=0),
                        start=0
                    ):
                        worker_ref = (
                            target_ip_worker_ref
                            if target_ip_worker_ref is not None
                            else await self._choose_worker(role="prefill")
                        )
                        """
                        Start the rank 0 model actor on the supervisor that holds the rank 1 replica,
                        solely for constructing the collective communication world.
                        """
                        if rank == 0:
                            _uid = model_uid + "-rank0"
                            rank0_address = await _launch_one_model(worker_ref, _uid, rank)
                            worker_refs.append((worker_ref, _uid))
                            rank_addresses.append(rank0_address)
                            logger.debug(
                                f"Starting rank 0 model {_uid} on worker {worker_ref.address} with rank {rank}."
                            )

                        subpool_address = await _launch_one_model(
                            worker_ref, rep_model_uid, rank+1
                        )
                        worker_refs.append((worker_ref, rep_model_uid))
                        rank_addresses.append(subpool_address)
                        logger.debug(
                            f"Starting prefill model {rep_model_uid} on worker {worker_ref.address} with rank {rank}."
                        )

                    # Start decode models
                    for rank, rep_model_uid in enumerate(
                        # support xPyD replica id here
                        iter_replica_model_uid(model_uid, prefill_replica+decode_replica, start=prefill_replica),
                        start=prefill_replica
                    ):
                        worker_ref = (
                            target_ip_worker_ref
                            if target_ip_worker_ref is not None
                            else await self._choose_worker(role="decode")
                        )
                        subpool_address = await _launch_one_model(
                            worker_ref, rep_model_uid, rank+1
                        )
                        worker_refs.append((worker_ref, rep_model_uid))
                        rank_addresses.append(subpool_address)
                        logger.debug(
                            f"Starting decode model {rep_model_uid} on worker {worker_ref.address} with rank {rank}."
                        )
                else:
                    for _idx, rep_model_uid in enumerate(
                        iter_replica_model_uid(model_uid, replica)
                    ):
                        worker_ref = (
                            target_ip_worker_ref
                            if target_ip_worker_ref is not None
                            else await self._choose_worker()
                        )
                        if enable_xavier and _idx == 0:
                            """
                            Start the rank 0 model actor on the worker that holds the rank 1 replica,
                            solely for constructing the collective communication world.
                            """
                            _uid = model_uid + "-rank0"
                            rank0_address = await _launch_one_model(worker_ref, _uid, 0)
                            worker_refs.append((worker_ref, _uid))
                            rank_addresses.append(rank0_address)

                        subpool_address = await _launch_one_model(
                            worker_ref, rep_model_uid, _idx + 1
                        )
                        worker_refs.append((worker_ref, rep_model_uid))
                        rank_addresses.append(subpool_address)


                # For xavier, start all the vllm instances first,
                # and then start the transfer component,
                # because the transfer actor needs all the rank addresses used for collective communication
                if enable_xavier:
                    logger.debug(f"Init transfer component for xavier...")
                    collective_manager_ref = self._collective_manager_mapping[model_uid]
                    tasks = []
                    for worker_ref, rep_model_uid in worker_refs:
                        tasks.append(
                            worker_ref.start_transfer_for_vllm(
                                rep_model_uid, rank_addresses
                            )
                        )
                    # Here you must use asyncio.gather, not a for loop,
                    # or you will get stuck.
                    await asyncio.gather(*tasks)

                    # init collective_manager
                    for idx, addr in enumerate(rank_addresses):
                        await collective_manager_ref.register_rank(
                            idx, addr, update=False
                        )

                    logger.debug(f"Init transfer component for xavier done.")
            except Exception:
                # terminate_model will remove the replica info.
                await self.terminate_model(model_uid, suppress_exception=True)
                await self._status_guard_ref.update_instance_info(
                    model_uid, {"status": LaunchStatus.ERROR.name}
                )
                raise

        if not is_valid_model_uid(model_uid):
            raise ValueError(
                "The model UID is invalid. Please specify the model UID by 0 < length <= 100."
            )

        if request_limits is not None and request_limits < 0:
            raise ValueError(
                "The `request_limits` parameter must be greater or equal than 0."
            )

        if model_uid in self._model_uid_to_replica_info:
            raise ValueError(f"Model is already in the model list, uid: {model_uid}")
        # Update real replica number
        if enable_disagg:
            prefill_replica_num = prefill_replica if prefill_replica is not None else 0
            decode_replica_num = decode_replica if decode_replica is not None else 0
            replica = prefill_replica_num + decode_replica_num

        # Set replica info first for exception handler to terminate model.
        self._model_uid_to_replica_info[model_uid] = ReplicaInfo(
            replica=replica, scheduler=itertools.cycle(range(replica))
        )
        instance_info = InstanceInfo(
            model_name=model_name,
            model_uid=model_uid,
            model_version=model_version,
            model_ability=[],
            replica=replica,
            status=LaunchStatus.CREATING.name,
            instance_created_ts=int(time.time()),
        )
        await self._status_guard_ref.set_instance_info(model_uid, instance_info)
        if wait_ready:
            await _launch_model()
        else:
            task = asyncio.create_task(_launch_model())
            ASYNC_LAUNCH_TASKS[model_uid] = task
            task.add_done_callback(lambda _: callback_for_async_launch(model_uid))  # type: ignore
        return model_uid

    async def get_instance_info(
        self, model_name: Optional[str], model_uid: Optional[str]
    ) -> List[Dict]:
        infos = await self._status_guard_ref.get_instance_info(
            model_name=model_name, model_uid=model_uid
        )
        return [info.dict() for info in sorted(infos, key=lambda info: info.model_uid)]

    async def get_instance_count(self, model_name: str) -> int:
        return await self._status_guard_ref.get_instance_count(model_name)

    async def _check_dead_nodes(self):
        while True:
            try:
                dead_nodes = []
                for address, status in self._worker_status.items():
                    if (
                        time.time() - status.update_time
                        > XINFERENCE_HEALTH_CHECK_TIMEOUT
                    ):
                        status.failure_remaining_count -= 1
                    else:
                        status.failure_remaining_count = (
                            XINFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD
                        )

                    if status.failure_remaining_count <= 0:
                        dead_models = []
                        for model_uid in self._replica_model_uid_to_worker:
                            if (
                                self._replica_model_uid_to_worker[model_uid].address
                                == address
                            ):
                                dead_models.append(model_uid)
                        logger.error(
                            "Worker dead. address: %s, influenced models: %s",
                            address,
                            dead_models,
                        )
                        for replica_model_uid in dead_models:
                            model_uid, _ = parse_replica_model_uid(replica_model_uid)
                            self._model_uid_to_replica_info.pop(model_uid, None)
                            self._replica_model_uid_to_worker.pop(
                                replica_model_uid, None
                            )
                        dead_nodes.append(address)
                    elif (
                        status.failure_remaining_count
                        != XINFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD
                    ):
                        logger.error(
                            "Worker timeout. address: %s, check count remaining %s...",
                            address,
                            status.failure_remaining_count,
                        )

                for address in dead_nodes:
                    self._worker_status.pop(address, None)
                    self._worker_address_to_worker.pop(address, None)
            finally:
                await asyncio.sleep(XINFERENCE_HEALTH_CHECK_INTERVAL)

    @log_async(logger=logger)
    async def terminate_model(self, model_uid: str, suppress_exception=False):
        async def _terminate_one_model(_replica_model_uid):
            worker_ref = self._replica_model_uid_to_worker.get(_replica_model_uid, None)

            if worker_ref is None:
                raise ValueError(
                    f"Model not found in the model list, uid: {_replica_model_uid}"
                )
            await worker_ref.terminate_model(model_uid=_replica_model_uid)
            del self._replica_model_uid_to_worker[_replica_model_uid]

        replica_info = self._model_uid_to_replica_info.get(model_uid, None)
        if replica_info is None:
            raise ValueError(f"Model not found in the model list, uid: {model_uid}")

        for rep_model_uid in iter_replica_model_uid(model_uid, replica_info.replica):
            try:
                await _terminate_one_model(rep_model_uid)
            except Exception:
                if not suppress_exception:
                    raise
        self._model_uid_to_replica_info.pop(model_uid, None)

        # clear for xavier
        rank0_uid = model_uid + "-rank0"
        if rank0_uid in self._replica_model_uid_to_worker:
            await _terminate_one_model(rank0_uid)

        collective_manager_ref = self._collective_manager_mapping.pop(model_uid, None)
        if collective_manager_ref is not None:
            try:
                await xo.destroy_actor(collective_manager_ref)
            except Exception as e:
                logger.debug(
                    "Destroy collective_manager_ref failed, model uid: %s, error: %s",
                    model_uid,
                    e,
                )
            finally:
                logger.debug(
                    f"Destroy collective_manager_ref done. model uid: {model_uid}"
                )
        block_tracker_ref = self._block_tracker_mapping.pop(model_uid, None)
        if block_tracker_ref is not None:
            try:
                await xo.destroy_actor(block_tracker_ref)
            except Exception as e:
                logger.debug(
                    "Destroy block_tracker_ref failed, model uid: %s, error: %s",
                    model_uid,
                    e,
                )
            finally:
                logger.debug(f"Destroy block_tracker_ref done. model uid: {model_uid}")

    @log_async(logger=logger)
    async def get_model(self, model_uid: str) -> xo.ActorRefType["ModelActor"]:
        # Get model from decode worker
        replica_info = self._model_uid_to_replica_info.get(model_uid, None)
        if replica_info is None:
            raise ValueError(f"Model not found in the model list, uid: {model_uid}")

        replica_model_uid = build_replica_model_uid(
            model_uid, next(replica_info.scheduler)
        )

        worker_ref = self._replica_model_uid_to_worker.get(replica_model_uid, None)
        if worker_ref is None:
            raise ValueError(
                f"Model not found in the model list, uid: {replica_model_uid}"
            )
        return await worker_ref.get_model(model_uid=replica_model_uid)

    @log_async(logger=logger)
    async def get_role(self) -> str:
        return self._role

    @log_async(logger=logger)
    async def get_disagg_model(self, model_uid: str) -> xo.ActorRefType["PDModelActor"]:
        # Make sure the model name is raw
        model_uid = parse_replica_model_uid(model_uid)[0]
        # Get model from decode worker
        replica_info = self._model_uid_to_replica_info.get(model_uid, None)
        if replica_info is None:
            raise ValueError(f"Model not found in the model list, uid: {model_uid}")

        # get prefill and decode worker
        # TODO: add prefill and decode models into mapping
        prefill_model_ref = None
        for idx in range(replica_info.replica*2):
            replica_model_uid = build_replica_model_uid(
                model_uid, idx
            )
            logger.debug(f"Get model from worker with _replica_model_uid_to_worker: {self._replica_model_uid_to_worker}")

            worker_ref = self._replica_model_uid_to_worker.get(replica_model_uid, None)
            if worker_ref is None:
                continue

            worker_status = self._worker_status.get(worker_ref.address)
            if worker_status.status['labels'].role == "prefill":
                prefill_model_ref = await worker_ref.get_model(model_uid=replica_model_uid)
                break

        if prefill_model_ref is None:
            raise ValueError(
                f"Prefill model not found in the model list, uid: {model_uid}"
            )

        decode_model_ref = None
        for idx in range(replica_info.replica*2):
            replica_model_uid = build_replica_model_uid(
                model_uid, idx
            )
            logger.debug(f"Get model from worker with _replica_model_uid_to_worker: {self._replica_model_uid_to_worker}")

            worker_ref = self._replica_model_uid_to_worker.get(replica_model_uid, None)
            if worker_ref is None:
                continue

            worker_status = self._worker_status.get(worker_ref.address)
            if worker_status.status['labels'].role == "decode":
                decode_model_ref = await worker_ref.get_model(model_uid=replica_model_uid)
                break

        if decode_model_ref is None:
            raise ValueError(
                f"Decode model not found in the model list, uid: {model_uid}"
            )

        pd_model_ref = await xo.create_actor(
            PDModelActor,
            uid=gen_random_string(8),
            address=self.address,
            prefill_model=prefill_model_ref,
            decode_model=decode_model_ref,
        )

        return pd_model_ref

    @log_async(logger=logger)
    async def get_model_status(self, replica_model_uid: str):
        worker_ref = self._replica_model_uid_to_worker.get(replica_model_uid, None)
        if worker_ref is None:
            raise ValueError(
                f"Model not found in the model list, uid: {replica_model_uid}"
            )
        return await worker_ref.get_model_status(replica_model_uid)

    @log_async(logger=logger)
    async def describe_model(self, model_uid: str) -> Dict[str, Any]:
        logger.info(f"Describe model, model_uid: {model_uid} _model_uid_to_replica_info: {self._model_uid_to_replica_info} _replica_model_uid_to_worker: {self._replica_model_uid_to_worker}")
        replica_info = self._model_uid_to_replica_info.get(model_uid, None)

        if replica_info is None:
            raise ValueError(f"Model not found in the model list, uid: {model_uid}")
        # Use rep id 0 to instead of next(replica_info.scheduler) to avoid
        # consuming the generator.
        replica_id = 1
        if self._role == "disaggregated":
            replica_id = 1
        else:
            # replica = 0
            # set default is 1
            replica_id = 1
        replica_model_uid = build_replica_model_uid(model_uid, replica_id)
        logger.debug(f"Describe model, model_uid: {model_uid} replica_model_uid: {replica_model_uid}")
        worker_ref = self._replica_model_uid_to_worker.get(replica_model_uid, None)
        if worker_ref is None:
            raise ValueError(
                f"Model not found in the model list, uid: {replica_model_uid}"
            )
        info = await worker_ref.describe_model(model_uid=replica_model_uid)
        info["replica"] = replica_info.replica
        return info

    @log_async(logger=logger)
    async def list_models(self) -> Dict[str, Dict[str, Any]]:
        ret = {}

        workers = list(self._worker_address_to_worker.values())
        for worker in workers:
            ret.update(await worker.list_models())
        running_model_info = {parse_replica_model_uid(k)[0]: v for k, v in ret.items()}
        logger.info(f"running_model_info: {running_model_info}")
        logger.info(f"_model_uid_to_replica_info: {self._model_uid_to_replica_info}")
        # add replica count
        for k, v in running_model_info.items():
            v["replica"] = self._model_uid_to_replica_info[k].replica
        return running_model_info

    def is_local_deployment(self) -> bool:
        # TODO: temporary.
        return (
            len(self._worker_address_to_worker) == 1
            and list(self._worker_address_to_worker)[0] == self.address
        )

    @log_async(logger=logger)
    async def list_cached_models(
        self, model_name: Optional[str] = None, worker_ip: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        target_ip_worker_ref = (
            self._get_worker_ref_by_ip(worker_ip) if worker_ip is not None else None
        )
        if (
            worker_ip is not None
            and not self.is_local_deployment()
            and target_ip_worker_ref is None
        ):
            raise ValueError(f"Worker ip address {worker_ip} is not in the cluster.")

        # search assigned worker and return
        if target_ip_worker_ref:
            cached_models = await target_ip_worker_ref.list_cached_models(model_name)
            cached_models = sorted(cached_models, key=lambda x: x["model_name"])
            return cached_models

        # search all worker
        cached_models = []
        for worker in self._worker_address_to_worker.values():
            res = await worker.list_cached_models(model_name)
            cached_models.extend(res)
        cached_models = sorted(cached_models, key=lambda x: x["model_name"])
        return cached_models

    @log_async(logger=logger)
    async def abort_request(
        self,
        model_uid: str,
        request_id: str,
        block_duration: int = XINFERENCE_DEFAULT_CANCEL_BLOCK_DURATION,
    ) -> Dict:
        from .scheduler import AbortRequestMessage

        res = {"msg": AbortRequestMessage.NO_OP.name}
        replica_info = self._model_uid_to_replica_info.get(model_uid, None)
        if not replica_info:
            return res
        replica_cnt = replica_info.replica

        # Query all replicas
        for rep_mid in iter_replica_model_uid(model_uid, replica_cnt):
            worker_ref = self._replica_model_uid_to_worker.get(rep_mid, None)
            if worker_ref is None:
                continue
            model_ref = await worker_ref.get_model(model_uid=rep_mid)
            result_info = await model_ref.abort_request(request_id, block_duration)
            res["msg"] = result_info
            if result_info == AbortRequestMessage.DONE.name:
                break
            elif result_info == AbortRequestMessage.NOT_FOUND.name:
                logger.debug(f"Request id: {request_id} not found for model {rep_mid}")
            else:
                logger.debug(f"No-op for model {rep_mid}")
        return res

    @log_async(logger=logger)
    async def add_worker(self, worker_address: str):
        from .worker import WorkerActor

        assert (
            worker_address not in self._worker_address_to_worker
        ), f"Worker {worker_address} exists"

        worker_ref = await xo.actor_ref(
            address=worker_address, uid=WorkerActor.default_uid()
        )
        self._worker_address_to_worker[worker_address] = worker_ref
        logger.debug("Worker %s has been added successfully", worker_address)

    @log_async(logger=logger)
    async def remove_worker(self, worker_address: str):
        uids_to_remove = []
        for model_uid in self._replica_model_uid_to_worker:
            if self._replica_model_uid_to_worker[model_uid].address == worker_address:
                uids_to_remove.append(model_uid)

        for replica_model_uid in uids_to_remove:
            model_uid, _ = parse_replica_model_uid(replica_model_uid)
            self._model_uid_to_replica_info.pop(model_uid, None)
            self._replica_model_uid_to_worker.pop(replica_model_uid, None)

        if worker_address in self._worker_address_to_worker:
            del self._worker_address_to_worker[worker_address]
            logger.debug("Worker %s has been removed successfully", worker_address)
        else:
            logger.warning(
                f"Worker {worker_address} cannot be removed since it is not registered to supervisor."
            )

    async def report_worker_status(
        self, worker_address: str, status: Dict[str, Union[ResourceStatus, GPUStatus, Labels]],
    ):
        if worker_address not in self._worker_status:
            logger.info("Worker %s resources: %s", worker_address, status)
            self._worker_status[worker_address] = WorkerStatus(
                update_time=time.time(),
                failure_remaining_count=XINFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD,
                status=status,
            )
        else:
            worker_status = self._worker_status[worker_address]
            worker_status.update_time = time.time()
            worker_status.status = status

    async def list_deletable_models(
        self, model_version: str, worker_ip: Optional[str] = None
    ) -> List[str]:
        target_ip_worker_ref = (
            self._get_worker_ref_by_ip(worker_ip) if worker_ip is not None else None
        )
        if (
            worker_ip is not None
            and not self.is_local_deployment()
            and target_ip_worker_ref is None
        ):
            raise ValueError(f"Worker ip address {worker_ip} is not in the cluster.")

        ret = []
        if target_ip_worker_ref:
            ret = await target_ip_worker_ref.list_deletable_models(
                model_version=model_version,
            )
            return ret

        for worker in self._worker_address_to_worker.values():
            path = await worker.list_deletable_models(model_version=model_version)
            ret.extend(path)
        return ret

    async def confirm_and_remove_model(
        self, model_version: str, worker_ip: Optional[str] = None
    ) -> bool:
        target_ip_worker_ref = (
            self._get_worker_ref_by_ip(worker_ip) if worker_ip is not None else None
        )
        if (
            worker_ip is not None
            and not self.is_local_deployment()
            and target_ip_worker_ref is None
        ):
            raise ValueError(f"Worker ip address {worker_ip} is not in the cluster.")

        if target_ip_worker_ref:
            ret = await target_ip_worker_ref.confirm_and_remove_model(
                model_version=model_version,
            )
            return ret
        ret = True
        for worker in self._worker_address_to_worker.values():
            ret = ret and await worker.confirm_and_remove_model(
                model_version=model_version,
            )
        return ret

    async def get_workers_info(self) -> List[Dict[str, Any]]:
        ret = []
        for worker in self._worker_address_to_worker.values():
            ret.append(await worker.get_workers_info())
        return ret

    async def get_supervisor_info(self) -> Dict[str, Any]:
        ret = {
            "supervisor_ip": self.address,
        }
        return ret

    async def trigger_exit(self) -> bool:
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except Exception as e:
            logger.info(f"trigger exit error: {e}")
            return False
        return True

    async def abort_cluster(self) -> bool:
        ret = True
        for worker in self._worker_address_to_worker.values():
            ret = ret and await worker.trigger_exit()

        ret = ret and await self.trigger_exit()
        return ret

    @staticmethod
    def record_metrics(name, op, kwargs):
        record_metrics(name, op, kwargs)

    async def get_progress(self, request_id: str) -> float:
        return await self._progress_tracker.get_progress(request_id)

    async def call_collective_manager(
        self, model_uid: str, func_name: str, *args, **kwargs
    ):
        """
        Used by worker.
        """
        collective_manager_ref = self._collective_manager_mapping[model_uid]
        await getattr(collective_manager_ref, func_name)(*args, **kwargs)
