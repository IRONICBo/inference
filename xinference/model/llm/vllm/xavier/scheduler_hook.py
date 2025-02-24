from vllm.sequence import ExecuteModelRequest
from vllm.core.scheduler import SchedulerOutputs

from .executor import XavierExecutor
from .scheduler import XavierScheduler


class EngineHook:
    async def post_scheduler_init(self, scheduler: XavierScheduler):
        """
        Before the _schedule() is ready, we need to do something before init stage.
        """
        pass

    async def pre_scheduler_prefill(self, scheduler: XavierScheduler, scheduled_seq_group: SchedulerOutputs):
        """
        Before the _schedule() is ready, we need to do something before prefill stage.
        """
        pass

    async def post_scheduler_prefill(self, scheduler: XavierScheduler):
        """
        Before the _schedule() is ready, we need to do something after prefill stage.
        """
        pass

    async def pre_scheduler_decode(self, scheduler: XavierScheduler):
        """
        Before the _schedule() is ready, we need to do something before decode stage.
        """
        pass

    async def post_scheduler_decode(self, scheduler: XavierScheduler):
        """
        Before the _schedule() is ready, we need to do something after decode stage.
        """
        pass

    async def post_execute_init(self, executor: XavierExecutor):
        """
        Before the _execute() is ready, we need to do something.
        """
        pass

    async def pre_execute(self, executor: XavierExecutor, execute_model_req: ExecuteModelRequest):
        """
        Before the _execute() is ready, we need to do something.
        """
        pass

    async def post_execute(self, executor: XavierExecutor, execute_model_req: ExecuteModelRequest):
        """
        After the _execute() is ready, we need to do something.
        """
        pass