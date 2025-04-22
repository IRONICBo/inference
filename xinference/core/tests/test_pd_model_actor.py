import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock
import xoscar as xo

from ..model import PDModelActor, RoundRobinSchedulingPolicy


class MockModelActor:
    def __init__(self, id):
        self.id = id
        self.generate = AsyncMock(return_value="mock_generated_output")


@pytest_asyncio.fixture
async def setup_pool():
    pool = await xo.create_actor_pool(
        f"test://127.0.0.1:{xo.utils.get_next_port()}", n_process=0
    )
    async with pool:
        yield pool


@pytest.mark.asyncio
async def test_generate_with_round_robin(setup_pool):
    pool = setup_pool
    addr = pool.external_address

    prefill_actors = [MockModelActor(i) for i in range(3)]
    decode_actors = [MockModelActor(i) for i in range(3)]


    pd_model_actor = await xo.create_actor(
        PDModelActor,
        address=addr,
        uid=PDModelActor.default_uid(),
        prefill_model_actors=prefill_actors,
        decode_model_actors=decode_actors
    )

    await pd_model_actor.generate("test prompt")
    await pd_model_actor.generate("test prompt")
    await pd_model_actor.generate("test prompt")

    assert prefill_actors[0].generate.call_count == 1
    assert prefill_actors[1].generate.call_count == 1
    assert prefill_actors[2].generate.call_count == 1

    assert decode_actors[0].generate.call_count == 1
    assert decode_actors[1].generate.call_count == 1
    assert decode_actors[2].generate.call_count == 1


if __name__ == "__main__":
    pytest.main()
