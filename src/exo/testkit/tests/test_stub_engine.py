"""The stub engine on its own: deterministic replies, rank agreement, and losing a peer."""

import socket
import threading
import time

import pytest

from exo.shared.models.model_cards import ModelId
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.common import CommandId, Host, NodeId
from exo.shared.types.tasks import TaskId, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import (
    BoundInstance,
    InstanceId,
    MlxRingInstance,
)
from exo.shared.types.worker.runner_response import CancelledResponse, FinishedResponse
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.testkit.cluster import StubModel
from exo.testkit.stub.directives import (
    Directives,
    expected_reply,
    parse_directives,
)
from exo.testkit.stub.engine import StubEngine
from exo.testkit.stub.ring import Ring, RingError
from exo.utils.channels import MpReceiver, mp_channel

MODEL = ModelId("exo-testkit/stub-8l")


def _task(
    prompt: str, max_tokens: int | None = None, name: str = "a"
) -> TextGeneration:
    return TextGeneration(
        task_id=TaskId(f"task-{name}"),
        instance_id=InstanceId("instance"),
        command_id=CommandId(f"command-{name}"),
        task_params=TextGenerationTaskParams(
            model=MODEL,
            input=[InputMessage(role="user", content=InputMessageContent(prompt))],
            stream=True,
            max_output_tokens=max_tokens,
        ),
    )


def _cancels() -> MpReceiver[TaskId]:
    _, receiver = mp_channel[TaskId]()
    return receiver


def _run(engine: StubEngine, task: TextGeneration, steps: int = 10_000) -> list[object]:
    engine.submit(task)
    results: list[object] = []
    for _ in range(steps):
        for task_id, result in engine.step():
            assert task_id == task.task_id
            results.append(result)
            if isinstance(result, (FinishedResponse, CancelledResponse)):
                return results
    raise AssertionError("the engine never finished the request")


def test_directives_round_trip_through_their_tag() -> None:
    directives = Directives(
        tokens=40, rate=20, fault="stall", fault_at=3, fault_rank=1, stall_seconds=2
    )
    assert parse_directives(f"hello {directives.tag()} world") == directives
    assert parse_directives("no tag here") == Directives()


@pytest.mark.parametrize(
    "prompt", ["[stub tokns=3]", "[stub fault=explode]", "[stub tokens]"]
)
def test_a_directive_the_stub_does_not_know_is_an_error(prompt: str) -> None:
    with pytest.raises(ValueError):
        parse_directives(prompt)


def test_the_reply_is_a_function_of_the_prompt() -> None:
    assert expected_reply("p [stub tokens=3]") == expected_reply("p [stub tokens=3]")
    assert (
        expected_reply("p [stub tokens=3]").text
        != expected_reply("q [stub tokens=3]").text
    )
    short = expected_reply("p [stub tokens=5]", max_tokens=2)
    assert len(short.tokens) == 2
    assert short.finish_reason == "length"


def test_one_rank_generates_exactly_the_expected_tokens() -> None:
    prompt = "count [stub tokens=7 rate=1000]"
    engine = StubEngine(MODEL, 0, 1, None, _cancels())
    results = _run(engine, _task(prompt))
    chunks = [r for r in results if isinstance(r, TokenChunk)]
    assert "".join(chunk.text for chunk in chunks) == expected_reply(prompt).text
    assert [chunk.finish_reason for chunk in chunks] == [None] * 6 + ["stop"]
    assert chunks[-1].usage is not None and chunks[-1].usage.completion_tokens == 7
    assert isinstance(results[-1], FinishedResponse)


def test_the_rate_holds_when_sleeps_run_late(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under load macOS stretches short sleeps (a 20 ms sleep took 127 ms on a busy Mac): the
    stub makes up for late steps instead of falling behind."""
    real_sleep = time.sleep

    def late_sleep(seconds: float) -> None:
        real_sleep(seconds + 0.1)

    monkeypatch.setattr(time, "sleep", late_sleep)
    prompt = "steady [stub tokens=50 rate=50]"
    engine = StubEngine(MODEL, 0, 1, None, _cancels())
    started = time.monotonic()
    results = _run(engine, _task(prompt))
    took = time.monotonic() - started
    assert "".join(r.text for r in results if isinstance(r, TokenChunk)) == (
        expected_reply(prompt).text
    )
    # 50 tokens at 50 a second: the last is due after 0.98 s
    assert 0.9 < took < 1.3, f"{took:.2f}s"


def test_max_tokens_cuts_the_reply_short() -> None:
    prompt = "count [stub tokens=7 rate=1000]"
    engine = StubEngine(MODEL, 0, 1, None, _cancels())
    results = _run(engine, _task(prompt, max_tokens=3))
    text = "".join(r.text for r in results if isinstance(r, TokenChunk))
    assert text == expected_reply(prompt, max_tokens=3).text
    last = [r for r in results if isinstance(r, TokenChunk)][-1]
    assert last.finish_reason == "length"


def test_a_cancelled_request_ends_with_a_cancellation() -> None:
    sender, receiver = mp_channel[TaskId]()
    engine = StubEngine(MODEL, 0, 1, None, receiver)
    task = _task("long [stub tokens=1000 rate=1000]")
    engine.submit(task)
    for _ in range(3):
        list(engine.step())
    sender.send(task.task_id)
    results = [result for _ in range(5) for _, result in engine.step()]
    assert any(isinstance(result, CancelledResponse) for result in results)


def _free_port_pair() -> int:
    for _ in range(50):
        with socket.socket() as first, socket.socket() as second:
            first.bind(("0.0.0.0", 0))
            port = int(first.getsockname()[1])  # pyright: ignore[reportAny]
            try:
                second.bind(("0.0.0.0", port + 1))
            except OSError:
                continue
            return port
    raise AssertionError("no two free ports in a row")


def _two_rank_instance() -> tuple[BoundInstance, BoundInstance]:
    port = _free_port_pair()
    nodes = (NodeId("node-0"), NodeId("node-1"))
    runners = (RunnerId("runner-0"), RunnerId("runner-1"))
    card = StubModel().card()
    instance = MlxRingInstance(
        instance_id=InstanceId("instance"),
        shard_assignments=ShardAssignments(
            model_id=MODEL,
            runner_to_shard={
                runners[rank]: PipelineShardMetadata(
                    model_card=card,
                    device_rank=rank,
                    world_size=2,
                    start_layer=4 * rank,
                    end_layer=4 * rank + 4,
                    n_layers=8,
                )
                for rank in (0, 1)
            },
            node_to_runner=dict(zip(nodes, runners, strict=True)),
        ),
        hosts_by_node={
            nodes[0]: [Host(ip="0.0.0.0", port=port), Host(ip="127.0.0.1", port=port)],
            nodes[1]: [Host(ip="127.0.0.1", port=port), Host(ip="0.0.0.0", port=port)],
        },
        ephemeral_port=port,
    )
    first, second = (
        BoundInstance(
            instance=instance, bound_runner_id=runners[rank], bound_node_id=nodes[rank]
        )
        for rank in (0, 1)
    )
    return first, second


def _connect_both() -> tuple[Ring, Ring]:
    first, second = _two_rank_instance()
    rings: dict[int, Ring] = {}

    def join(rank: int, bound: BoundInstance) -> None:
        rings[rank] = Ring.connect(bound, proxy_port=None)

    threads = [
        threading.Thread(target=join, args=(rank, bound))
        for rank, bound in enumerate((first, second))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert set(rings) == {0, 1}, "the ranks didn't connect"
    return rings[0], rings[1]


def test_two_ranks_agree_and_only_rank_zero_reports_tokens() -> None:
    rings = _connect_both()
    prompt = "together [stub tokens=12 rate=1000]"
    results: dict[int, list[object]] = {}

    def run(rank: int) -> None:
        engine = StubEngine(MODEL, rank, 2, rings[rank], _cancels())
        results[rank] = _run(engine, _task(prompt))

    threads = [threading.Thread(target=run, args=(rank,)) for rank in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(20)
    assert set(results) == {0, 1}
    text = "".join(r.text for r in results[0] if isinstance(r, TokenChunk))
    assert text == expected_reply(prompt).text
    assert not any(isinstance(r, TokenChunk) for r in results[1])
    assert isinstance(results[0][-1], FinishedResponse)
    assert isinstance(results[1][-1], FinishedResponse)
    for ring in rings:
        ring.close()


def test_a_rank_that_loses_its_peer_sees_a_failed_receive() -> None:
    first, second = _connect_both()
    second.close()
    with pytest.raises(RingError):
        first.all_gather(b"anyone there?")
    first.close()
