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
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import xoscar as xo

logger = getLogger(__name__)


class RemoteKVCacheManager(xo.StatelessActor):
    @classmethod
    def default_uid(cls):
        return f"kvcache-manager-actor"

    def __init__(self):
        super().__init__()

    def register_blocks(
        self, engine_metadata: List[Dict[str, Union[str, int]]], cache_metadata: List[Dict[str, Union[str, int]]]
    ):
        """
        Used to register metadata in the cache manager.

        engine_metadata: virtual engine for llm backend, used to choose engine by
        cache_metadata: key value for this kvcache metadata, maybe contains hash_content, prefix promopt and so on.
        """
        pass

    def write_blocks(
        self, engine_metadata: List[Dict[str, Union[str, int]]], cache_metadata: List[Dict[str, Union[str, int]]], cache_data: List[torch.Tensor]
    ):
        """
        Used to write cache data to the storage.

        engine_metadata: virtual engine for llm backend, used to choose engine by
        cache_metadata: key value for this kvcache metadata, maybe contains hash_content, prefix promopt and so on.
        cache_data: a list of kvcache data, espically for decoder llm each layer.
        """
        pass

    def query_blocks(
        self, engine_metadata: List[Dict[str, Union[str, int]]], cache_metadata: List[Dict[str, Union[str, int]]]
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
        self, engine_metadata: List[Dict[str, Union[str, int]]], cache_metadata: List[Dict[str, Union[str, int]]]
    ) -> List[torch.Tensor]:
        """
        Used to read cache metadata from remote storage, these data will be read at the buffer in self._buffer

        engine_metadata: virtual engine for llm backend, used to choose engine by
        cache_metadata: key value for this kvcache metadata, maybe contains hash_content, prefix promopt and so on.

        return:
        remote: a list of kvcache data, espically for decoder llm each layer.
        """
        pass

    def unregister_blocks(
        self, engine_metadata: List[Dict[str, Union[str, int]]], cache_metadata: List[Dict[str, Union[str, int]]]
    ):
        """
        Used to remove metadata from remote storage

        engine_metadata: virtual engine for llm backend, used to choose engine by
        cache_metadata: key value for this kvcache metadata, maybe contains hash_content, prefix promopt and so on.
        """
        pass


    def remove_blocks(
        self, engine_metadata: List[Dict[str, Union[str, int]]], cache_metadata: List[Dict[str, Union[str, int]]]
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

    def register_ranks(self, rank_metada: Dict[str, Union[str, int]]):
        """
        Used to register p2p components.

        rank_metada: rank metadata, used to specify the rank to be unregistered.
        """
        pass
