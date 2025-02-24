import asyncio
from collections import deque
import logging
from typing import Any, Deque, Dict, List, Set, Tuple

import xoscar as xo
from vllm.executor.gpu_executor import GPUExecutorAsync
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest, PoolerOutput
from vllm.utils import is_pin_memory_available
from vllm.core.block.interfaces import Block
from vllm.core.scheduler import SchedulerOutputs, ScheduledSequenceGroup
from vllm.worker.cache_engine import CacheEngine
from vllm.sequence import (
    SequenceData,
    SequenceGroup,
    SequenceGroupMetadata,
    SequenceGroupMetadataDelta,
    SequenceStage,
    SequenceStatus,
)

from .executor import XavierExecutor
from .scheduler import XavierScheduler
from .scheduler_hook import EngineHook

logger = logging.getLogger(__name__)


class XavierEngineHook(EngineHook):
    def __init__(self):
        super().__init__()
        # Used to store the information collected during the hook.
        self._executor_context: Dict[str, Any] = {}
        self._scheduler_context: Dict[str, Any] = {}

    def post_scheduler_init(self, scheduler: XavierScheduler):
        scheduler._block_tracker_ref = None
        scheduler._transfer_ref = None
        # scheduler._transferring: Deque[SequenceGroup] = deque()
        scheduler._transferring = deque()
        # scheduler._transfer_status: Dict[SequenceGroup, Set[int]] = {}
        scheduler._transfer_status = {}

        self._scheduler_context = {
            "scheduled_seq_groups": [],
            "has_transferring": False,
        }

    async def _get_block_tracker_ref(self, scheduler: XavierScheduler):
        if self._block_tracker_ref is None:
            block_tracker_address = self._xavier_config.get("block_tracker_address")
            block_tracker_uid = self._xavier_config.get("block_tracker_uid")
            self._block_tracker_ref = await xo.actor_ref(
                address=block_tracker_address, uid=block_tracker_uid
            )
        return self._block_tracker_ref

    async def _get_transfer_ref(self, scheduler: XavierScheduler):
        from .transfer import TransferActor

        if self._transfer_ref is None:
            transfer_address = self._xavier_config.get("rank_address")
            rank = self._xavier_config.get("rank")
            self._transfer_ref = await xo.actor_ref(
                address=transfer_address, uid=f"{TransferActor.default_uid()}-{rank}"
            )
        return self._transfer_ref

    async def _get_transfer_details(
        self,
        scheduler: XavierScheduler,
        virtual_engine: int,
        block_tables: Dict[int, List[int]],
        seq_group: SequenceGroup,
    ) -> Tuple[Set[int], Dict[int, Set[Tuple[int, int, int]]]]:
        # If the `seq_group` has the `force_calculation` attribute set to `True`,
        # it indicates that there were issues during the transmission process.
        # In this case, force the computation and exclude it from the Xavier process.
        if getattr(seq_group, "force_calculation", False):
            return set(), dict()
        """
        Retrieve information from other replicas to check if any blocks have already been computed,
        for the purpose of data transfer.
        """
        details: Set[Tuple[int, int]] = set()
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            block_ids = block_tables[seq.seq_id]
            for _id in block_ids:
                block: Block = self.block_manager.get_block_by_block_id(seq.seq_id, _id)
                detail = (block.content_hash, _id)
                """
                1. `block.content_hash is not None` means that the block has been filled with tokens.
                Unless it is evicted from the cache, the computation result of this block is constant.
                2. Check the `transferred` status of the block.
                If it is `True`, it means the block has already been transferred locally
                and does not need to be transferred again.
                3. Check the `executed` status of the block.
                If it is `True`, it means the block has already been computed locally
                and does not need to be transferred.
                """
                if (
                    (block.content_hash is not None)
                    and (
                        not self.block_manager.get_block_status_by_block_id(
                            "transferred", block.block_id
                        )
                    )
                    and (
                        not self.block_manager.get_block_status_by_block_id(
                            "executed", block.block_id
                        )
                    )
                ):
                    details.add(detail)

        logger.debug(f"Xaiver scheduler details: {details}")
        if details:
            tracker_ref = await self._get_block_tracker_ref()
            remote = await tracker_ref.query_blocks(virtual_engine, list(details))
            # Not all queried blocks have corresponding results in other replicas.
            # Therefore, it is necessary to record which local block data was actually transferred.
            local: Set[int] = set()
            for _, remote_details in remote.items():
                for _, _, local_block_id in remote_details:
                    local.add(local_block_id)
            if local:
                logger.debug(
                    f"Data in local blocks: {local} will be transmitted from the remote."
                )
            return local, remote
        else:
            return set(), dict()

    async def _do_transfer_inner(
        self, scheduler: XavierScheduler, virtual_engine: int, remote: Dict[int, Set[Tuple[int, int, int]]]
    ):
        transfer_ref = await self._get_transfer_ref()
        for from_rank, hash_and_block_id in remote.items():
            src_to_dst: Dict[int, int] = {x[1]: x[2] for x in hash_and_block_id}
            await transfer_ref.recv(virtual_engine, from_rank, src_to_dst)

    async def _do_transfer(
        self,
        scheduler: XavierScheduler,
        virtual_engine: int,
        local: Set[int],
        remote: Dict[int, Set[Tuple[int, int, int]]],
        seq_group: SequenceGroup,
    ):
        try:
            await self._do_transfer_inner(virtual_engine, remote)
        except Exception as e:
            """
            The exception here is most likely due to the sender triggering recovery during the transmission process.
            In this case, fallback to performing computation during the prefill stage.
            """
            logger.error(f"Transfer failed: {e}")
            # Force this `seq_group` to perform computation.
            seq_group.force_calculation = True
            self._transfer_status.pop(seq_group, None)
            self.waiting.appendleft(seq_group)
            self._transferring.remove(seq_group)

            # Unpin prefill instance kvcache
            unpin_handle = self._unpin_handles.get(seq_group.request_id, None)
            if unpin_handle is not None:
                await unpin_handle.free_prefill_model_cache(seq_group.request_id)
            self.remove_unpin_handle(seq_group.request_id)

        else:
            # After the transfer is completed, update the corresponding metadata.
            self._transfer_status[seq_group] = local
            for _id in local:
                self.block_manager.set_block_status_by_block_id(
                    "transferred", _id, True
                )
            # After the transfer, place the `seq_group` back into the `waiting` queue to
            # wait for the next scheduling execution.
            self.waiting.appendleft(seq_group)
            self._transferring.remove(seq_group)

            # Unpin prefill instance kvcache
            unpin_handle = self._unpin_handles.get(seq_group.request_id, None)
            if unpin_handle is not None:
                await unpin_handle.free_prefill_model_cache(seq_group.request_id)
            self.remove_unpin_handle(seq_group.request_id)

    async def pre_scheduler_prefill(
        self,
        scheduler: XavierScheduler,
        scheduled_seq_group: ScheduledSequenceGroup,
        block_tables: Dict[int, List[int]],
    ) -> bool:
        """
        After completing the scheduling, the blocks have been allocated.
        Therefore, it is possible to check whether some blocks have already been computed on other replicas based on this information,
        and subsequently initiate the transfer.
        According to the internal code comments in vllm,
        whether `token_chunk_size` is 1 can indicate whether the `seq_group` is in the decode or prefill stage.
        It is noted that data transmission is only applied during the prefill stage.
        In the decode stage, it only applies to the last token of the block, which can negatively impact throughput.
        """
        virtual_engine = scheduler._virtual_engine
        block_tables = scheduled_seq_group.block_tables
        seq_group = scheduled_seq_group.seq_group
        token_chunk_size = scheduled_seq_group.token_chunk_size
        is_prefill: bool = token_chunk_size != 1
        # must query remote in decode
        if is_prefill:
            local, remote = await self._get_transfer_details(
                scheduler, virtual_engine, block_tables, seq_group
            )
            if remote:
                running_seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
                for seq in running_seqs:
                    seq.status = SequenceStatus.WAITING
                    # Additional attribute `transferred` to mark that this `seq_group` involves a transfer process.
                    # During the next scheduling, block allocation will no longer be required
                    # since it has already been completed.
                    seq.transferred = True
                    seq.data._stage = SequenceStage.PREFILL
                scheduler._transfer_status[seq_group] = set()
                # Use `create_task` to avoid blocking subsequent scheduling.
                asyncio.create_task(
                    self._do_transfer(
                        scheduler,
                        virtual_engine,
                        local,
                        remote,
                        seq_group
                    )
                )
                # The `seq_group` that is currently being transferred enters a new queue.
                scheduler._transferring.append(seq_group)
                self._scheduler_context["has_transferring"] = True
                return True
            else:
                import vllm.core.scheduler
                self._scheduler_context["scheduled_seq_groups"].append(vllm.core.scheduler.ScheduledSequenceGroup(seq_group, token_chunk_size))
                return False

        if scheduler.cache_config.enable_prefix_caching:
            common_computed_block_nums = (
                self.block_manager.get_common_computed_block_ids(
                    seq_group.get_seqs(status=SequenceStatus.RUNNING)
                )
            )
            """Xinference Change!!!
            This is very important and is the core of Xavier.
            `computed_block_nums` is the key attribute that determines which blocks do not need to be computed,
            as decided by the `model_runner`.
            Therefore, after the transfer is completed, this attribute needs to be updated.
            """
            if seq_group in scheduler._transfer_status:
                transferred_blocks = scheduler._transfer_status[seq_group]
                if transferred_blocks:
                    common_computed_block_nums.extend(transferred_blocks)
                    common_computed_block_nums = list(
                        sorted(common_computed_block_nums)
                    )
                    del scheduler._transfer_status[seq_group]

    async def post_scheduler_prefill(
        self,
        scheduler: XavierScheduler,
        scheduler_outputs: SchedulerOutputs,
        scheduled_seq_groups: List[SequenceGroup],
    ):
        """Xinference Change!!!
        If the `seq_group` in this scheduling triggers a transfer,
        it needs to be removed from the running queue (as it is already in the transferring queue).
        It should remain in the transferring queue until the transfer is complete,
        and then it can be placed back into the appropriate queue for scheduling.
        """
        has_transferring = self._scheduler_context.get("has_transferring", False)
        if has_transferring:
            scheduler_outputs.scheduled_seq_groups = scheduled_seq_groups
            for seq_group in scheduler.running.copy():
                if seq_group in scheduler._transfer_status:
                    scheduler.running.remove(seq_group)

    async def pre_scheduler_decode(
        self,
        scheduler: XavierScheduler,
        scheduled_seq_group: ScheduledSequenceGroup,
        block_tables: Dict[int, List[int]],
    ):
        """
        Before the _schedule() is ready, we need to do something before decode stage.
        """
        pass

    async def post_scheduler_decode(
        self,
        scheduler: XavierScheduler,
        scheduler_outputs: SchedulerOutputs,
        scheduled_seq_groups: List[SequenceGroup],
    ):
        """
        Before the _schedule() is ready, we need to do something after decode stage.
        """
        pass

    async def _get_block_tracker_ref(self, executor: XavierExecutor):
        if executor._block_tracker_ref is None:
            block_tracker_address = executor.vllm_config.xavier_config.get(
                "block_tracker_address"
            )
            block_tracker_uid = executor.vllm_config.xavier_config.get("block_tracker_uid")
            executor._block_tracker_ref = await xo.actor_ref(
                address=block_tracker_address, uid=block_tracker_uid
            )
        return executor._block_tracker_ref

    async def _get_transfer_ref(self, executor: XavierExecutor):
        from .transfer import TransferActor

        if executor._transfer_ref is None:
            transfer_address = executor.vllm_config.xavier_config.get("rank_address")
            rank = executor.vllm_config.xavier_config.get("rank")
            executor._transfer_ref = await xo.actor_ref(
                address=transfer_address, uid=f"{TransferActor.default_uid()}-{rank}"
            )
        return executor._transfer_ref

    async def post_execute_init(self, executor: XavierExecutor):
        """
        In vllm, the `cache_engine` is the entity that truly manages the KV cache tensors.
        Retrieve the necessary transmission information from the `cache_engine`.
        """
        transfer_ref = await self._get_transfer_ref(executor)
        ref_cache_engine: CacheEngine = executor.driver_worker.cache_engine[0]
        buffer_dtype = ref_cache_engine.dtype
        buffer_device = "cpu"
        buffer_pin_memory = is_pin_memory_available()
        num_attn_layers = ref_cache_engine.num_attention_layers
        kv_cache_shape = ref_cache_engine.gpu_cache[0].shape
        assert kv_cache_shape[0] == 2
        buffer_num = 2
        transfer_block_num = executor.vllm_config.xavier_config.get("transfer_block_num")
        buffer_shape = (
            transfer_block_num,
            num_attn_layers,
            kv_cache_shape[0],
            *kv_cache_shape[2:],
        )
        await transfer_ref.setup(
            self.driver_worker.cache_engine,
            self.scheduler,
            num_buffer=buffer_num,
            buffer_shape=buffer_shape,
            buffer_dtype=buffer_dtype,
            buffer_device=buffer_device,
            pin_memory=buffer_pin_memory,
        )

    def _get_rank(self, executor: XavierExecutor) -> int:
        return executor.vllm_config.xavier_config.get("rank")

    async def pre_execute(
        self,
        executor: XavierExecutor,
        execute_model_req: ExecuteModelRequest):
        """
        Collect information about the blocks involved in the execution before the vllm `ModelRunner` executes.
        This information will be used by the tracker after execution to register the locally computed blocks.
        """
        virtual_engine = execute_model_req.virtual_engine
        # logger.debug(f"Execute model async, virtual_engine: {virtual_engine}")
        scheduler = executor.scheduler[virtual_engine]  # type: ignore
        executed_blocks_details: Set[Tuple[int, int]] = set()
        for meta in execute_model_req.seq_group_metadata_list:
            block_tables = meta.block_tables
            for seq_id, block_ids in block_tables.items():
                for _id in block_ids:
                    b = scheduler.block_manager.get_block_by_block_id(seq_id, _id)
                    # The `executed` attribute is used to prevent duplicate registration of the block.
                    executed = scheduler.block_manager.get_block_status_by_block_id(
                        "executed", _id
                    )
                    detail = (b.content_hash, b.block_id)
                    if (b.content_hash is not None) and (not executed):
                        executed_blocks_details.add(detail)

        # Add to hook context
        self._executor_context["executed_blocks_details"] = executed_blocks_details


    async def post_execute(self, executor: XavierExecutor, execute_model_req: ExecuteModelRequest):
        executed_blocks_details = self._executor_context.get("executed_blocks_details", None)
        rank = self._get_rank()
        block_tracker_ref = await self._get_block_tracker_ref()
        virtual_engine = execute_model_req.virtual_engine
        scheduler = executor.scheduler[virtual_engine]

        if executed_blocks_details:
            """
            Why not collect and register the information after execution?
            Because after execution, the model's execution callback hook will release the block_id,
            causing the block manager to lose access to the correct information.
            """
            await block_tracker_ref.register_blocks(
                virtual_engine, list(executed_blocks_details), rank
            )

            for _, _id in executed_blocks_details:
                scheduler.block_manager.set_block_status_by_block_id(
                    "executed", _id, True
                )

        # Clear the context
        self._executor_context.pop("executed_blocks_details", None)