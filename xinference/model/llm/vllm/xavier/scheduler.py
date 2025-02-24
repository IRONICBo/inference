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
import asyncio
import logging
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Set, Tuple, no_type_check

import xoscar as xo
from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.block.interfaces import Block
from vllm.core.interfaces import BlockSpaceManager
from vllm.core.scheduler import Scheduler, SchedulerOutputs
from vllm.sequence import (
    SequenceData,
    SequenceGroup,
    SequenceGroupMetadata,
    SequenceGroupMetadataDelta,
    SequenceStage,
    SequenceStatus,
)

from .....core.model import PDModelActor
from .block_manager import XavierBlockManager

logger = logging.getLogger(__name__)


class XavierScheduler(Scheduler):
    @staticmethod
    def _get_block_space_manager_class(version: str):
        logger.debug("Init xavier block manager.")
        return XavierBlockManager

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
        pipeline_parallel_size: int = 1,
        output_proc_callback: Optional[Callable] = None,
        xavier_config: Optional[Dict] = None,
        virtual_engine: Optional[int] = 0,
        role: Optional[str] = "decode",
    ) -> None:
        # Monkey patch for free seq
        Scheduler.free_finished_seq_groups = XavierScheduler.free_finished_seq_groups
        BlockSpaceManager.get_block_space_manager_class = (
            self._get_block_space_manager_class
        )
        super().__init__(
            scheduler_config,
            cache_config,
            lora_config,
            pipeline_parallel_size,
            output_proc_callback,
        )
        xavier_config["virtual_engine"] = virtual_engine  # type: ignore
        self.block_manager.xavier_config = xavier_config
        self._xavier_config = xavier_config

        backend_type = self._xavier_config.get("backend_type")
        if backend_type == "xavier":
            from .xavier_scheduler_hook import XavierEngineHook
            self._scheduler_hook: XavierEngineHook = XavierEngineHook()
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")

        self._scheduler_hook.post_scheduler_init(self)
        self._virtual_engine = virtual_engine
        # Xavier Transfer related
        self._block_tracker_ref = None
        self._transfer_ref = None
        self._transferring: Deque[SequenceGroup] = deque()
        self._transfer_status: Dict[SequenceGroup, Set[int]] = {}
        self._role = role
        self._unpin_handles: Dict[str, xo.ActorRefType["PDModelActor"]] = {}

    @no_type_check
    async def schedule(
        self,
    ) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs, bool]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        scheduler_start_time = time.perf_counter()

        scheduler_outputs: SchedulerOutputs = self._schedule()
        now = time.time()

        if not self.cache_config.enable_prefix_caching:
            common_computed_block_nums = []

        allow_async_output_proc: bool = self.use_async_output_proc

        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for i, scheduled_seq_group in enumerate(scheduler_outputs.scheduled_seq_groups):
            seq_group = scheduled_seq_group.seq_group
            token_chunk_size = scheduled_seq_group.token_chunk_size
            seq_group.maybe_set_first_scheduled_time(now)

            seq_group_metadata = self._seq_group_metadata_cache[
                self.cache_id
            ].get_object()
            seq_group_metadata.seq_data.clear()
            seq_group_metadata.block_tables.clear()

            # seq_id -> SequenceData
            seq_data: Dict[int, SequenceData] = {}
            # seq_id -> physical block numbers
            block_tables: Dict[int, List[int]] = {}

            if seq_group.is_encoder_decoder():
                # Encoder associated with SequenceGroup
                encoder_seq = seq_group.get_encoder_seq()
                assert encoder_seq is not None
                encoder_seq_data = encoder_seq.data
                # Block table for cross-attention
                # Also managed at SequenceGroup level
                cross_block_table = self.block_manager.get_cross_block_table(seq_group)
            else:
                encoder_seq_data = None
                cross_block_table = None

            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_block_table(seq)
                self.block_manager.access_all_blocks_in_seq(seq, now)

            """Xinference Change!!!
            After completing the scheduling, the blocks have been allocated.
            Therefore, it is possible to check whether some blocks have already been computed on other replicas based on this information,
            and subsequently initiate the transfer.
            According to the internal code comments in vllm,
            whether `token_chunk_size` is 1 can indicate whether the `seq_group` is in the decode or prefill stage.
            It is noted that data transmission is only applied during the prefill stage.
            In the decode stage, it only applies to the last token of the block, which can negatively impact throughput.
            """
            # TODO: add params
            if await self._scheduler_hook.pre_scheduler_prefill(self, scheduled_seq_group, block_tables):
                continue
            if await self._scheduler_hook.pre_scheduler_decode(self, scheduled_seq_group, block_tables):
                continue

            if self.cache_config.enable_prefix_caching:
                common_computed_block_nums = (
                    self.block_manager.get_common_computed_block_ids(
                        seq_group.get_seqs(status=SequenceStatus.RUNNING)
                    )
                )

            do_sample = True
            is_prompt = seq_group.is_prefill()
            # We should send the metadata to workers when the first prefill
            # is sent. Subsequent requests could be chunked prefill or decode.
            is_first_prefill = False
            if is_prompt:
                seqs = seq_group.get_seqs()
                # Prefill has only 1 sequence.
                assert len(seqs) == 1
                num_computed_tokens = seqs[0].data.get_num_computed_tokens()
                is_first_prefill = num_computed_tokens == 0
                # In the next iteration, all prompt tokens are not computed.
                # It means the prefill is chunked, and we don't need sampling.
                # NOTE: We use get_len instead of get_prompt_len because when
                # a sequence is preempted, prefill includes previous generated
                # output tokens.
                if token_chunk_size + num_computed_tokens < seqs[0].data.get_len():
                    do_sample = False

            # It assumes the scheduled_seq_groups is ordered by
            # prefill < decoding.
            if is_first_prefill or not self.scheduler_config.send_delta_data:
                seq_group_metadata = SequenceGroupMetadata(
                    request_id=seq_group.request_id,
                    is_prompt=is_prompt,
                    seq_data=seq_data,
                    sampling_params=seq_group.sampling_params,
                    block_tables=block_tables,
                    do_sample=do_sample,
                    pooling_params=seq_group.pooling_params,
                    token_chunk_size=token_chunk_size,
                    lora_request=seq_group.lora_request,
                    computed_block_nums=common_computed_block_nums,
                    encoder_seq_data=encoder_seq_data,
                    cross_block_table=cross_block_table,
                    state=seq_group.state,
                    token_type_ids=seq_group.token_type_ids,
                    # `multi_modal_data` will only be present for the 1st comm
                    # between engine and worker.
                    # the subsequent comms can still use delta, but
                    # `multi_modal_data` will be None.
                    multi_modal_data=seq_group.multi_modal_data
                    if scheduler_outputs.num_prefill_groups > 0
                    else None,
                    multi_modal_placeholders=seq_group.multi_modal_placeholders
                    if scheduler_outputs.num_prefill_groups > 0
                    else None,
                    mm_processor_kwargs=seq_group.mm_processor_kwargs,
                    prompt_adapter_request=seq_group.prompt_adapter_request,
                )
            else:
                # When SPMD mode is enabled, we only send delta data except for
                # the first request to reduce serialization cost.
                seq_data_delta = {}
                for id, data in seq_data.items():
                    seq_data_delta[id] = data.get_delta_and_reset()
                seq_group_metadata = SequenceGroupMetadataDelta(
                    seq_data_delta,
                    seq_group.request_id,
                    block_tables,
                    is_prompt,
                    do_sample=do_sample,
                    token_chunk_size=token_chunk_size,
                    computed_block_nums=common_computed_block_nums,
                )
            seq_group_metadata_list.append(seq_group_metadata)

            if allow_async_output_proc:
                allow_async_output_proc = self._allow_async_output_proc(seq_group)

        """Xinference Change!!!
        If the `seq_group` in this scheduling triggers a transfer,
        it needs to be removed from the running queue (as it is already in the transferring queue).
        It should remain in the transferring queue until the transfer is complete,
        and then it can be placed back into the appropriate queue for scheduling.
        """
        await self._scheduler_hook.post_scheduler_prefill(self, scheduler_outputs, scheduler_outputs.scheduled_seq_groups)
        await self._scheduler_hook.post_scheduler_decode(self, scheduler_outputs, scheduler_outputs.scheduled_seq_groups)

        # logger.error(f"Scheduler outputs: {scheduler_outputs.scheduled_seq_groups}")

        # Now that the batch has been created, we can assume all blocks in the
        # batch will have been computed before the next scheduling invocation.
        # This is because the engine assumes that a failure in model execution
        # will crash the vLLM instance / will not retry.
        for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
            logger.error(f"Scheduler outputs type: {type(scheduled_seq_group)}")
            # Check current seq_group is existing.
            if hasattr(scheduled_seq_group, 'seq_group') and scheduled_seq_group.seq_group is not None:
                self.block_manager.mark_blocks_as_computed(
                    scheduled_seq_group.seq_group, scheduled_seq_group.token_chunk_size
                )

        self._seq_group_metadata_cache[self.next_cache_id].reset()

        scheduler_time = time.perf_counter() - scheduler_start_time
        # Add this to scheduler time to all the sequences that are currently
        # running. This will help estimate if the scheduler is a significant
        # component in the e2e latency.
        for seq_group in self.running:
            if seq_group is not None and seq_group.metrics is not None:
                if seq_group.metrics.scheduler_time is not None:
                    seq_group.metrics.scheduler_time += scheduler_time
                else:
                    seq_group.metrics.scheduler_time = scheduler_time

        # Move to next cache (if exists)
        self.cache_id = self.next_cache_id

        # Return results
        return (seq_group_metadata_list, scheduler_outputs, allow_async_output_proc)

    def has_unfinished_seqs(self) -> bool:
        """
        This interface is used to determine whether the scheduling process should stop,
        so it needs to include information about the transferring queue.
        """
        res = super().has_unfinished_seqs()
        return res or len(self._transferring) != 0

    def get_num_unfinished_seq_groups(self) -> int:
        """
        When retrieving information from this interface,
        the information from the transferring queue needs to be taken into account.
        """
        res = super().get_num_unfinished_seq_groups()
        return res + len(self._transferring)

    def free_finished_seq_groups(self) -> None:
        # Only decode instance will auto free seq_group,
        # For prefill instance, we will defer the free operation
        # to the decode instance is reached.
        if self._role == "decode":
            remaining: Deque[SequenceGroup] = deque()
            for seq_group in self.running:
                self._free_finished_seq_group(seq_group)
                if not seq_group.is_finished():
                    remaining.append(seq_group)

            self.running = remaining

            # Handle async stopped sequence groups
            # (ones that reached max model len)
            if self._async_stopped:
                for seq_group in self._async_stopped:
                    self._free_seq_group_cross_attn_blocks(seq_group)
                    self._finished_requests_ids.append(seq_group.request_id)

                    # Free finished seqs
                    self._free_finished_seqs(seq_group)

                self._async_stopped.clear()

    def free_seq_cache(self, request_id: str):
        """
        This interface is used to free the kvcache reference count in inference.
        """
        logger.debug("Free seq cache for request_id: {}".format(request_id))
        for seq_group in self.running:
            if seq_group is not None and seq_group.request_id != request_id:
                self._free_finished_seq_group(seq_group)
                logger.debug(
                    "Free running seq cache for request_id: {}".format(seq_group.request_id)
                )

        for seq_group in self._transferring:
            if seq_group is not None and seq_group.request_id != request_id:
                self._free_finished_seq_group(seq_group)
                logger.debug(
                    "Free transferring seq cache for request_id: {}".format(seq_group.request_id)
                )

        for seq_group in self.waiting:
            if seq_group is not None and seq_group.request_id != request_id:
                self._free_finished_seq_group(seq_group)
                logger.debug(
                    "Free waiting seq cache for request_id: {}".format(seq_group.request_id)
                )

        for seq_group in self.swapped:
            if seq_group is not None and seq_group.request_id != request_id:
                self._free_finished_seq_group(seq_group)
                logger.debug(
                    "Free swapped seq cache for request_id: {}".format(seq_group.request_id)
                )

    async def set_unpin_handle(self, model_uid: str, request_id: str, pd_model_actor_address: str):
        self._unpin_handles[request_id] = await xo.actor_ref(
                address=pd_model_actor_address, uid=f"{PDModelActor.default_uid()}-{model_uid}"
            )
        logger.debug(f"[XaiverScheduler] Set unpin handle for request_id: {request_id}")

    def remove_unpin_handle(self, request_id: str):
        if request_id in self._unpin_handles:
            del self._unpin_handles[request_id]
            logger.debug(f"[XaiverScheduler] Remove unpin handle for request_id: {request_id}")