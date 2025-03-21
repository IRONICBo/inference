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
from datenlordsdk import DatenLordSDK
from vllm.core.scheduler import Scheduler
from vllm.utils import TORCH_DTYPE_TO_NUMPY_DTYPE, Device
from vllm.worker.cache_engine import CacheEngine

from .block_tracker import VLLMBlockTracker
from .transfer import TransferActor
from .remote_kvcache_manager import RemoteKVCacheManager

logger = getLogger(__name__)


class DatenlordRemoteKVCacheManager(RemoteKVCacheManager):
    def __init__(self):
        super().__init__()

        self._datenlord_sdk: Optional[DatenLordSDK] = None

    async def setup(
        self,
        xavier_config: Dict[str, Any],
        transfer_metadata: Dict[str, Any] = None,
        block_tracker_metadata: Dict[str, Any] = None,
    ):
        """
        Lazy setup actor reference with transfer actor and tracker actor.
        """
        if self._datenlord_sdk is not None:
            return

        datenlord_block_size = xavier_config.get("datenlord_block_size")
        datenlord_kv_engine_address = xavier_config.get("datenlord_kv_engine_address")
        datenlord_log_level = xavier_config.get("datenlord_log_level")
        self._datenlord_sdk = DatenLordSDK(
            block_size=datenlord_block_size,
            kv_engine_address=[datenlord_kv_engine_address],
            log_level=datenlord_log_level,
        )
        logger.debug(f"DatenlordRemoteKVCacheManager setup done with xavier_config: {xavier_config}")

    async def register_blocks(
        self, engine_metadata: Dict[str, Union[str, int]], cache_metadatas: List[Dict[str, Union[str, int]]]
    ):
        """
        Used to register metadata in the cache manager.

        engine_metadata: virtual engine for llm backend, used to choose engine by
        cache_metadata: key value for this kvcache metadata, maybe contains hash_content, prefix promopt and so on.
        """
        # We need to register block in write_blocks func for consistency.
        logger.debug(f"Register blocks in DatenlordRemoteKVCacheManager with engine_metadata: {engine_metadata} and cache_metadata: {cache_metadatas}")
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
        logger.debug(f"Write blocks in DatenlordRemoteKVCacheManager with engine_metadata: {engine_metadata} and cache_metadata: {cache_metadata}")
        if len(cache_metadata) != len(cache_data):
            logger.error(f"cache_metadata and cache_data should have the same length, but got {len(cache_metadata)} and {len(cache_data)}")
            return

        layer_num = engine_metadata.get("layer_num", None)
        if layer_num is None:
            logger.error("layer_num should not be None")
            return

        # Write data to remote storage.
        for idx, metadata in enumerate(cache_metadata):
            import io
            buf = io.BytesIO()
            for _, layer_data in enumerate(cache_data[idx]):
                torch.save(layer_data, buf)
            # TODO: replace with promot token ids
            content_hash = metadata.get("content_hash", None)
            if content_hash is None:
                logger.error("key should not be None")
                return

            content_hash = str(content_hash)
            content_hash = [ord(char) for char in content_hash]
            kv_cache_block = buf.getvalue()

            logger.debug(f"Write data idx to remote storage, metadata: {metadata} kv_cache_block size: {len(kv_cache_block)}")
            await self._datenlord_sdk.insert(
                content_hash,
                kv_cache_block,
            )
            logger.debug(f"Write data idx to remote storage success, metadata: {metadata} kv_cache_block size: {len(kv_cache_block)}")

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
        logger.debug(f"Query blocks in DatenlordRemoteKVCacheManager with engine_metadata: {engine_metadata} and cache_metadata: {cache_metadatas}")
        virtual_engine = engine_metadata.get("virtual_engine")
        executed_blocks_details = [
            (metadata['content_hash'], metadata['block_id'])
            for metadata in cache_metadatas
        ]
        logger.debug(f"Query blocks in DatenlordRemoteKVCacheManager with content hash {executed_blocks_details}")

        res = []
        for content_hash, local_block_id in executed_blocks_details:
            content_hash = str(content_hash)
            content_hash = [ord(char) for char in content_hash]
            prefx = await self._datenlord_sdk.match_prefix(content_hash)
            # Make sure current size is not 0
            if prefx is not None and len(prefx) != 0:
                # Append to data, returned prefix and -1 is not used.
                res.append((prefx, -1 ,local_block_id))

        if len(res) == 0:
            return {}

        # Default rank is datenlord
        return {'datenlord': res}

    async def read_blocks(
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
        logger.debug(f"Read blocks in DatenlordRemoteKVCacheManager with engine_metadata: {engine_metadata} and cache_metadata: {cache_metadata}")
        from_rank = cache_metadata.get("from_rank")
        layer_num = engine_metadata.get("layer_num", None)
        if layer_num is None:
            logger.error("layer_num should not be None")
            return
        remote_block_metadata = cache_metadata.get("remote_block_metadata")

        recvbuf = [[] for _ in range(layer_num)]
        dst_to_block_hash: Dict[int, int] = {x[2]: x[0] for x in remote_block_metadata}
        # block_hash_to_dst: Dict[int, int] = {x[0]: x[2] for x in remote_block_metadata}
        recv_block_ids = []
        # for block_hash, dst in block_hash_to_dst.items():
        for dst, block_hash in dst_to_block_hash.items():
            logger.debug(f"Read blocks in DatenlordRemoteKVCacheManager with block_hash: {block_hash} and dst: {dst}")
            # content_hash = str(content_hash)
            # content_hash = [ord(char) for char in content_hash]
            content_hash = block_hash
            logger.debug(f"Read blocks in DatenlordRemoteKVCacheManager with content hash {content_hash}")
            matched_key, data = await self._datenlord_sdk.try_load(content_hash)
            data = memoryview(data).tobytes()
            logger.debug(f"Read blocks in DatenlordRemoteKVCacheManager with matched_key: {matched_key} and data size: {len(data)}")
            # TODO
            layer_size = len(data) // layer_num
            if matched_key is not None and matched_key == content_hash:
                import io
                for i in range(layer_num):
                    buf = io.BytesIO()
                    buf.write(data[layer_size*i:layer_size*(i+1)])
                    buf.seek(0)
                    kv_cache = torch.load(buf, weights_only=True)
                    recvbuf[i].append(kv_cache)
                recv_block_ids.append(dst)
            else:
                logger.error(f"Currrent bloc is not valid, skip read blocks in DatenlordRemoteKVCacheManager with block_hash: {block_hash} and dst: {dst}")

        # cpu_buf_index is -1, we don't need to free buffer in datenlord.
        return recvbuf, recv_block_ids, -1

    async def free_blocks(
        self, buffer_metadata:  Dict[str, int]
    ):
        """
        Used to free buffer metadata from current storage

        buffer_metadata: a dict of buffer metadata, maybe contains cpu_buf_index and so on.
        """
        logger.debug(f"[skip]Free blocks in DatenlordRemoteKVCacheManager with buffer_metadata: {buffer_metadata}")
        pass

    async def unregister_blocks(
        self, engine_metadata: Dict[str, Union[str, int]], cache_metadatas: List[Dict[str, Union[str, int]]]
    ):
        """
        Used to remove metadata from remote storage

        engine_metadata: virtual engine for llm backend, used to choose engine by
        cache_metadata: key value for this kvcache metadata, maybe contains hash_content, prefix promopt and so on.
        """
        logger.debug(f"[skip]Unregister blocks in DatenlordRemoteKVCacheManager with engine_metadata: {engine_metadata} and cache_metadata: {cache_metadatas}")
        pass

    async def remove_blocks(
        self, engine_metadata: Dict[str, Union[str, int]], cache_metadatas: List[Dict[str, Union[str, int]]]
    ):
        """
        Used to remove cache metadata from remote storage

        engine_metadata: virtual engine for llm backend, used to choose engine by
        cache_metadata: key value for this kvcache metadata, maybe contains hash_content, prefix promopt and so on.
        """
        # In datenlord backend, we do not need to remove cache from anywhere.
        logger.debug(f"[skip]Remove blocks in DatenlordRemoteKVCacheManager with engine_metadata: {engine_metadata} and cache_metadata: {cache_metadatas}")
        pass
