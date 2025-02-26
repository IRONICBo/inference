# Copyright 2022-2025 XProbe Inc.
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
import random
from logging import getLogger
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import xoscar as xo
from vllm.core.scheduler import Scheduler
from vllm.utils import TORCH_DTYPE_TO_NUMPY_DTYPE, Device
from vllm.worker.cache_engine import CacheEngine

from .transfer import TransferActor
from .executor import XavierExecutor
from .remote_kvcache_manager import RemoteKVCacheManager

logger = getLogger(__name__)


class XavierRemoteKVCacheManager(RemoteKVCacheManager):
    @classmethod
    def default_uid(cls):
        return f"kvcache-manager-actor"

    def __init__(self):
        super().__init__()

        self._transfer_ref: Optional[xo.ActorRefType["TransferActor"]] = None

    async def setup(
        self,
        xavier_config: Dict[str, Any],
        transfer_metadata: Dict[str, Any],
    ):
        """
        Setup current transfer metadata to the cache manager.
        """
        from .transfer import TransferActor

        if self._transfer_ref is None:
            transfer_address = xavier_config.get("rank_address")
            rank = xavier_config.get("rank")
            self._transfer_ref = await xo.actor_ref(
                address=transfer_address, uid=f"{TransferActor.default_uid()}-{rank}"
            )

        cache_engine = transfer_metadata.get("cache_engine")
        scheduler = transfer_metadata.get("scheduler")
        num_buffer = transfer_metadata.get("num_buffer")
        buffer_shape = transfer_metadata.get("buffer_shape")
        buffer_dtype = transfer_metadata.get("buffer_dtype")
        buffer_device = transfer_metadata.get("buffer_device")
        pin_memory = transfer_metadata.get("pin_memory")

        self._transfer_ref.setup(
            cache_engine,
            scheduler,
            num_buffer=num_buffer,
            buffer_shape=buffer_shape,
            buffer_dtype=buffer_dtype,
            buffer_device=buffer_device,
            pin_memory=pin_memory,
        )

    def register_blocks(
        self, engine_metadata: Dict[str, Union[str, int]], cache_metadata: List[Dict[str, Union[str, int]]]
    ):
        """
        Used to register metadata in the cache manager.

        engine_metadata: virtual engine for llm backend, used to choose engine by
        cache_metadata: key value for this kvcache metadata, maybe contains hash_content, prefix promopt and so on.
        """
        pass

    def write_blocks(
        self, engine_metadata: Dict[str, Union[str, int]], cache_metadata: List[Dict[str, Union[str, int]]], cache_data: List[torch.Tensor]
    ):
        """
        Used to write cache data to the storage.

        engine_metadata: virtual engine for llm backend, used to choose engine by
        cache_metadata: key value for this kvcache metadata, maybe contains hash_content, prefix promopt and so on.
        cache_data: a list of kvcache data, espically for decoder llm each layer.
        """
        pass

    def query_blocks(
        self, engine_metadata: Dict[str, Union[str, int]], cache_metadata: List[Dict[str, Union[str, int]]]
    ) -> List[Dict[str, Union[str, int]]]:
        """
        Used to query cache metadata from remote storage.

        engine_metadata: virtual engine for llm backend, used to choose engine by
        cache_metadata: key value for this kvcache metadata, maybe contains hash_content, prefix promopt and so on.

        return:
        remote: a dict of remote cache metadata, .
        """
        pass

    def read_blocks(
        self, engine_metadata: Dict[str, Union[str, int]], cache_metadata: List[Dict[str, Union[str, int]]]
    ) -> Tuple[torch.Tensor, Dict[int, int], Dict[str, int]]:
        """
        Used to read cache metadata from remote storage, these data will be read at the buffer in self._buffer

        engine_metadata: virtual engine for llm backend, used to choose engine by
        cache_metadata: key value for this kvcache metadata, maybe contains hash_content, prefix promopt and so on.

        return:
        1. A full buffer reference.
        2. a dict of block id to swap in index.
        3. A full buffer reference's metadata.
        """
        self._transfer_ref.read_blocks(
            engine_metadata, cache_metadata
        )

    def free_blocks(
        self, buffer_metadata:  Dict[str, int]
    ):
        """
        Used to free buffer metadata from current storage
        """
        pass


    def unregister_blocks(
        self, engine_metadata: Dict[str, Union[str, int]], cache_metadata: List[Dict[str, Union[str, int]]]
    ):
        """
        Used to remove metadata from remote storage

        engine_metadata: virtual engine for llm backend, used to choose engine by
        cache_metadata: key value for this kvcache metadata, maybe contains hash_content, prefix promopt and so on.
        """
        pass


    def remove_blocks(
        self, engine_metadata: Dict[str, Union[str, int]], cache_metadata: List[Dict[str, Union[str, int]]]
    ):
        """
        Used to remove cache metadata from remote storage

        engine_metadata: virtual engine for llm backend, used to choose engine by
        cache_metadata: key value for this kvcache metadata, maybe contains hash_content, prefix promopt and so on.
        """
        pass

    def unregister_rank(self, rank_metada: Dict[str, Union[str, int]]):
        """
        Used to unregister p2p components.

        rank_metada: rank metadata, used to specify the rank to be unregistered.
        """
        pass

    def register_rank(self, rank_metada: Dict[str, Union[str, int]]):
        """
        Used to register p2p components.

        rank_metada: rank metadata, used to specify the rank to be unregistered.
        """
        pass
