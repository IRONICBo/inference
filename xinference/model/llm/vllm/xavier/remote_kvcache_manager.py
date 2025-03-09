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
from typing import Any, Dict, List, Tuple, Union

import torch


class RemoteKVCacheManager:
    async def setup(
        self,
        xavier_config: Dict[str, Any],
        transfer_metadata: Dict[str, Any] = None,
        block_tracker_metadata: Dict[str, Any] = None,
    ):
        pass

    async def register_blocks(
        self, engine_metadata: Dict[str, Union[str, int]], cache_metadatas: List[Dict[str, Union[str, int]]]
    ):
        """
        Used to register metadata in the cache manager.

        engine_metadata: virtual engine for llm backend, used to choose engine by
        cache_metadata: key value for this kvcache metadata, maybe contains hash_content, prefix promopt and so on.
        """
        pass

    async def write_blocks(
        self, engine_metadata: Dict[str, Union[str, int]], cache_metadata: List[Dict[str, Union[str, int]]], cache_data: List[List[torch.Tensor]]
    ):
        """
        Used to write cache data to the storage.

        engine_metadata: virtual engine for llm backend, used to choose engine by
        cache_metadata: key value for this kvcache metadata, maybe contains hash_content, prefix promopt and so on.
        cache_data: a list of kvcache data, espically for decoder llm each layer, blockid -> layer -> tensor
        """
        pass

    async def query_blocks(
        self, engine_metadata: Dict[str, Union[str, int]], cache_metadatas: List[Dict[str, Union[str, int]]]
    ) -> List[Dict[str, Union[str, int]]]:
        """
        Used to query cache metadata from remote storage.

        engine_metadata: virtual engine for llm backend, used to choose engine by
        cache_metadata: key value for this kvcache metadata, maybe contains hash_content, prefix promopt and so on.

        return:
        remote: a dict of remote cache metadata, .
        """
        pass

    async def read_blocks(
        self, engine_metadata: Dict[str, Union[str, int]], cache_metadata: List[Dict[str, Union[str, int]]]
    ) -> Tuple[torch.Tensor, Dict[int, int], Dict[str, int]]:
        """
        Used to read cache metadata from remote storage, these data will be read at the buffer in self._buffer

        engine_metadata: virtual engine for llm backend, used to choose engine by
        cache_metadata: key value for this kvcache metadata, maybe contains hash_content, prefix promopt and so on.

        return:
        remote: a list of kvcache data, espically for decoder llm each layer.
        """
        pass

    async def free_blocks(
        self, buffer_metadata:  Dict[str, int]
    ):
        """
        Used to free buffer metadata from current storage

        buffer_metadata: a dict of buffer metadata, maybe contains cpu_buf_index and so on.
        """
        pass

    async def unregister_blocks(
        self, engine_metadata: Dict[str, Union[str, int]], cache_metadatas: List[Dict[str, Union[str, int]]]
    ):
        """
        Used to remove metadata from remote storage

        engine_metadata: virtual engine for llm backend, used to choose engine by
        cache_metadata: key value for this kvcache metadata, maybe contains hash_content, prefix promopt and so on.
        """
        pass

    async def remove_blocks(
        self, engine_metadata: Dict[str, Union[str, int]], cache_metadatas: List[Dict[str, Union[str, int]]]
    ):
        """
        Used to remove cache metadata from remote storage

        engine_metadata: virtual engine for llm backend, used to choose engine by
        cache_metadata: key value for this kvcache metadata, maybe contains hash_content, prefix promopt and so on.
        """
        pass
