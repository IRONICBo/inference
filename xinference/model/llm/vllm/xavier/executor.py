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
from typing import TYPE_CHECKING, List, Optional, Set, Tuple, Union
from logging import getLogger

import xoscar as xo
from vllm.executor.gpu_executor import GPUExecutorAsync
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest, PoolerOutput
from vllm.utils import is_pin_memory_available
from vllm.worker.cache_engine import CacheEngine

if TYPE_CHECKING:
    from .scheduler import XavierScheduler


logger = getLogger(__name__)

class XavierExecutor(GPUExecutorAsync):
    scheduler: Optional[List["XavierScheduler"]] = None

    def _init_executor(self) -> None:
        super()._init_executor()
        self._transfer_ref = None
        self._block_tracker_ref = None

    async def init_transfer(self):
        backend_type = self.vllm_config.xavier_config.get("backend_type")
        if backend_type == "xavier":
            from .xavier_scheduler_hook import XavierEngineHook
            self._engine_hook: XavierEngineHook = XavierEngineHook()
            await self._engine_hook.post_execute_init(self)
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")

    async def execute_model_async(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> List[Union[SamplerOutput, PoolerOutput]]:
        """
        Before execute model, we need to do some pre-processing.
        """
        await self._engine_hook.pre_execute(self, execute_model_req)

        res = await super().execute_model_async(execute_model_req)

        # logger.debug(f"Execute model async, virtual_engine: {virtual_engine} executed_blocks_details: {executed_blocks_details}")
        """
        After execute model, we need to do some post-processing.
        """
        await self._engine_hook.post_execute(self, execute_model_req)

        return res
