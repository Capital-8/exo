"""Cluster scenarios on real exo nodes (exo.testkit), with main's behaviour as the expectation.

Tests marked xfail(strict=True) are failures exo has today. Each cites its evidence: the
chaos runs on real Macs (CLUSTER = chaos-nov3/cluster/CLUSTER-RESULTS.md "Ranked issues",
NETWORK = NETWORK-RESULTS.md) and the architecture review's failure ids (A1, B1, ...). The step
of the redesign that fixes one removes its marker; strict means an unexpected pass fails the run,
so a fix can't go unnoticed.

Timing bounds are what these scenarios measured on main, with a margin, and say so.
Unmarked tests run in CI's default (fast) run; `slow` ones run nightly (`pytest -m slow`).
"""

import threading
import time

import pytest

from exo.shared.types.worker.instances import InstanceId
from exo.testkit.cluster import ChatHandle, ChatResult, Cluster, HarnessError
from exo.testkit.tests.conftest import MakeCluster

pytestmark = pytest.mark.cluster

# A reply long enough to still be streaming whatever happens to the cluster (60 s)
LONG = "[stub tokens=3000 rate=50]"
# Seconds a running chat streams before the scenario's fault
STREAM_FIRST = 3.0


def _ok(result: ChatResult) -> None:
    assert result.ok, result.summary()
    assert not result.problems(), result.problems()


def _clean_error(result: ChatResult) -> None:
    """The chat ended with an error, and every token it got before was right."""
    assert not result.ok, f"expected an error: {result.summary()}"
    assert result.error, result.summary()
    assert not result.problems(), result.problems()


def _streaming(cluster: Cluster, prompt: str, via: int) -> ChatHandle:
    handle = cluster.chat_in_background(prompt, via=via, timeout=150)
    time.sleep(STREAM_FIRST)
    assert not handle.done(), "the chat ended before the fault"
    return handle


def _instance_gone(cluster: Cluster, instance_id: InstanceId, via: int) -> bool:
    state = cluster.state(via)
    return state is not None and instance_id not in state.instances


# --- serving ---------------------------------------------------------------------------


def test_every_node_serves_every_token(make_cluster: MakeCluster) -> None:
    """A single-node and a two-node (pipeline, ring) instance serve concurrent chats through
    every node's API, streamed and not, and every token arrives, in order."""
    cluster = make_cluster(3)
    assert cluster.master() == 0
    cluster.launch(on=(1,))
    cluster.launch(on=(0, 2))
    handles = [
        cluster.chat_in_background(
            f"serve {via} {stream} [stub tokens=60 rate=200]", via=via, stream=stream
        )
        for via in range(3)
        for stream in (True, False)
    ]
    for handle in handles:
        _ok(handle.result(60))
    assert cluster.violations() == []


# --- a follower dies -------------------------------------------------------------------


def test_killing_a_follower_ends_its_requests_with_a_clean_error(
    make_cluster: MakeCluster,
) -> None:
    """kill -9 of a follower holding half of a two-node instance. Main: the in-flight chat
    ends with an error (measured 3.6 s; bound 15 s: the supervisor checks every 5 s); the
    master removes the node and deletes the instance (measured 16 s; bound 35 s) and nothing
    places it again; a new chat for the model fails at once. CLUSTER "kill -9 follower"."""
    cluster = make_cluster(3)
    instance = cluster.launch(on=(1, 2))
    chat = _streaming(cluster, "follower dies " + LONG, via=0)
    killed_at = time.monotonic()
    cluster.kill(2)

    result = chat.result(60)
    _clean_error(result)
    assert result.ended - killed_at < 15, result.summary()

    cluster.wait_until_consistent(timeout=35)
    assert _instance_gone(cluster, instance, via=0)
    after = cluster.chat("after the loss", via=0, timeout=10)
    assert after.status_code == 404, after.summary()


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason="C1: a lost model is never placed again (CLUSTER #7: 48 manual relaunches in 2 h)",
)
def test_a_model_that_lost_a_node_is_placed_again(make_cluster: MakeCluster) -> None:
    cluster = make_cluster(3)
    cluster.launch(on=(1, 2))
    cluster.kill(2)
    cluster.wait_for(
        lambda: cluster.chat("anyone serving?", via=0, timeout=10).ok,
        timeout=90,
        what="the model to serve again on the nodes that are left",
    )


# --- the master dies -------------------------------------------------------------------


@pytest.mark.slow
def test_killing_the_master_elects_a_new_one_and_fails_in_flight_chats(
    make_cluster: MakeCluster,
) -> None:
    """kill -9 of the master while a chat streams from a healthy instance on another node.
    Main: the followers elect a new master only after the 30 s master-silence timeout
    (measured 32.4 s; bound 45 s), the chat fails when they do, and the dead master stays in
    every view until it times out (bound 90 s). CLUSTER #1, #2; A1, A2, B3."""
    cluster = make_cluster(3)
    cluster.launch(on=(1,))
    chat = _streaming(cluster, "master dies " + LONG, via=2)
    killed_at = time.monotonic()
    cluster.kill(0)

    new_master = cluster.wait_for_master(timeout=45, other_than=0)
    assert new_master in (1, 2)
    _clean_error(chat.result(60))
    cluster.wait_until_consistent(timeout=90)
    _ok(cluster.chat("after the new master", via=2, timeout=30))
    assert time.monotonic() - killed_at < 140


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason="A1/B3: followers wait out the 30 s master-silence timeout (CLUSTER #1: "
    "'30 s freeze in 2 of 3 cases'; the zenoh liveliness token of a crashed router "
    "isn't deleted in 3+ node meshes)",
)
def test_a_crashed_master_is_replaced_within_5s(make_cluster: MakeCluster) -> None:
    cluster = make_cluster(3)
    cluster.kill(0)
    cluster.wait_for_master(timeout=5, other_than=0)


@pytest.mark.xfail(
    strict=True,
    reason="A2: every change of master ends every in-flight request, graceful quit "
    "included (CLUSTER #2: 6/6 failed; 16-22% of soak failures were on instances that "
    "lost no node)",
)
def test_a_master_change_keeps_chats_on_healthy_instances(
    make_cluster: MakeCluster,
) -> None:
    """The master quits while a chat streams from an instance on another node."""
    cluster = make_cluster(3)
    cluster.launch(on=(1,))
    chat = _streaming(cluster, "master quits [stub tokens=500 rate=50]", via=2)
    cluster.quit(0)
    _ok(chat.result(60))


@pytest.mark.slow
def test_quitting_the_master_elects_a_new_one_quickly(
    make_cluster: MakeCluster,
) -> None:
    """Main: a master that quits is replaced within seconds (measured 4.4 s; bound 15 s),
    but it stays in every view until it times out (measured 19.7 s; bound 40 s). CLUSTER
    "Graceful quit, master"; B2."""
    cluster = make_cluster(3)
    quit_at = time.monotonic()
    cluster.quit(0)
    cluster.wait_for_master(timeout=15, other_than=0)
    cluster.wait_until_consistent(timeout=40)
    assert time.monotonic() - quit_at < 60


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason="C3: commands accepted while the master dies are dropped, never resent "
    "(CLUSTER #8; membership REPORT section 2: 2/2 launches never appeared on main)",
)
def test_a_launch_sent_while_the_master_dies_is_placed(
    make_cluster: MakeCluster,
) -> None:
    cluster = make_cluster(3)
    cluster.kill(0)
    # The follower's API accepts the launch ("Command received.") and only the dead master
    # hears it; the master elected after the 30 s silence timeout never does
    cluster.place(on=(1,), via=1, timeout=90)


# --- a follower quits ------------------------------------------------------------------


@pytest.mark.slow
def test_quitting_a_follower_ends_its_requests_and_removes_it(
    make_cluster: MakeCluster,
) -> None:
    """A follower holding half of a two-node instance quits (SIGINT). Main: it exits in a few
    seconds; the in-flight chat and a chat sent after it left both end with an error
    (measured 8.6 s; bound 20 s); the master removes the node (measured 14.5 s; bound 35 s).
    CLUSTER "Graceful quit, follower"; B2."""
    cluster = make_cluster(3)
    cluster.launch(on=(1, 2))
    chat = _streaming(cluster, "follower quits " + LONG, via=0)
    quit_at = time.monotonic()
    assert cluster.quit(2) < 15
    late = cluster.chat_in_background("sent after the quit " + LONG, via=0, timeout=60)
    for result in (chat.result(60), late.result(60)):
        _clean_error(result)
        assert result.ended - quit_at < 20, result.summary()
    cluster.wait_until_consistent(timeout=35)


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason="B2: a graceful quit isn't announced; the node and its models stay listed "
    "for the 15 s silence timeout (CLUSTER #9)",
)
def test_a_node_that_quits_leaves_every_view_at_once(make_cluster: MakeCluster) -> None:
    cluster = make_cluster(3)
    cluster.quit(2)
    cluster.wait_until_consistent(timeout=3)


# --- a follower sleeps -----------------------------------------------------------------


def _pause_holding_half_an_instance(
    cluster: Cluster, seconds: float
) -> tuple[InstanceId, ChatResult]:
    instance = cluster.launch(on=(1, 2))
    chat = _streaming(cluster, f"sleeps {seconds:g}s " + LONG, via=0)
    cluster.pause(2)
    time.sleep(seconds)
    cluster.resume(2)
    return instance, chat.result(120)


@pytest.mark.xfail(
    strict=True,
    reason="B1: a pause of about 11 s or more removes the node and deletes its models "
    "(CLUSTER #5: a 10 s sleep failed 13 of 19 requests; the harness measured removal "
    "at +16 s)",
)
def test_a_10s_pause_keeps_the_model(make_cluster: MakeCluster) -> None:
    cluster = make_cluster(3)
    instance, _ = _pause_holding_half_an_instance(cluster, 10)
    time.sleep(20)
    assert not _instance_gone(cluster, instance, via=0), "the model was deleted"
    _ok(cluster.chat("after the pause", via=0, timeout=30))


@pytest.mark.slow
def test_a_40s_pause_removes_the_node_and_it_rejoins(make_cluster: MakeCluster) -> None:
    """SIGSTOP of a follower holding half of an instance for 40 s. Main: the in-flight chat
    fails, the node is removed and its instance deleted (~16 s); on waking it briefly runs
    as its own master (D1), then merges back (measured +56 s; bound 60 s after waking) with
    no ghosts. CLUSTER "60 s / 300 s sleep, follower"."""
    cluster = make_cluster(3)
    instance, result = _pause_holding_half_an_instance(cluster, 40)
    _clean_error(result)
    cluster.wait_until_consistent(timeout=60)
    assert _instance_gone(cluster, instance, via=0)


# --- runner faults -------------------------------------------------------------------


def _runner_statuses(cluster: Cluster, instance: InstanceId, via: int) -> list[str]:
    state = cluster.state(via)
    if state is None or instance not in state.instances:
        return []
    runners = state.instances[instance].shard_assignments.runner_to_shard
    return [type(state.runners.get(runner)).__name__ for runner in runners]


@pytest.mark.parametrize("rank", [0, 1])
def test_a_runner_that_crashes_mid_reply_ends_it_and_its_instance_recovers(
    make_cluster: MakeCluster, rank: int
) -> None:
    """One rank of a two-node instance dies (SIGKILL, as a segfault or the OS would) before
    token 10. Main: the chat ends with an error (measured 2.8-3.5 s; bound 15 s) and the
    workers rebuild the instance's runners, which serve again (measured 7.5-8.6 s; bound
    30 s)."""
    cluster = make_cluster(3)
    instance = cluster.launch(on=(1, 2))
    started = time.monotonic()
    result = cluster.chat(
        f"crash {rank} [stub tokens=200 rate=50 fault=crash at=10 rank={rank}]",
        via=0,
        timeout=60,
    )
    _clean_error(result)
    assert result.ended - started < 15, result.summary()
    cluster.wait_ready(instance, timeout=30)
    _ok(cluster.chat("after the crash", via=0, timeout=30))


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason="Found by this harness: after rank 0 raises mid-reply, the replacement runners "
    "pair with the stopping ones; one ends RunnerConnected to a dead peer and the other "
    "RunnerIdle, which the worker's plan never moves on (connect needs every runner Idle "
    "or Connecting, load needs every runner Connected), so the instance never serves "
    "again, its chats hang, and the stopped runners' ConnectToGroup tasks stay Running",
)
def test_a_runner_that_raises_mid_reply_leaves_its_instance_serving_again(
    make_cluster: MakeCluster,
) -> None:
    cluster = make_cluster(3)
    instance = cluster.launch(on=(1, 2))
    result = cluster.chat(
        "raise [stub tokens=200 rate=50 fault=raise at=10 rank=0]", via=0, timeout=60
    )
    _clean_error(result)
    try:
        cluster.wait_ready(instance, timeout=60)
    except HarnessError as error:
        pytest.fail(f"{error}; runners {_runner_statuses(cluster, instance, 0)}")
    _ok(cluster.chat("after the exception", via=0, timeout=30))
    cluster.wait_until_consistent(timeout=30)


# --- restarts --------------------------------------------------------------------------


def test_a_restarted_node_leaves_no_ghost_and_no_instance(
    make_cluster: MakeCluster,
) -> None:
    """kill -9 of a node holding an instance, restarted at once: exo gives it a new id, so it
    comes back as a new device. Main: the old id lingers as a ghost (measured 15.6 s) and its
    instance is deleted when it times out (16.1 s); bound 45 s. JOURNEYS #10; E1."""
    cluster = make_cluster(3)
    instance = cluster.launch(on=(2,))
    old_id = cluster.nodes[2].node_id
    cluster.kill(2)
    cluster.restart(2)
    assert cluster.nodes[2].node_id != old_id
    cluster.wait_until_consistent(timeout=45)
    assert _instance_gone(cluster, instance, via=0)


# --- races -----------------------------------------------------------------------------


def test_a_chat_sent_as_its_instance_is_deleted_never_hangs(
    make_cluster: MakeCluster,
) -> None:
    """DELETE /instance and, 0-12 ms later, a chat for the same model, through a follower whose
    view still holds the instance. Before #2560 some of these hung forever (CLUSTER #3:
    4/12 on main 63760522, 12/12 on the candidate); each must now end within 15 s."""
    cluster = make_cluster(2)
    for trial in range(12):
        instance = cluster.launch(on=(1,), via=1)
        results: list[ChatResult] = []

        def chat(trial: int, results: list[ChatResult]) -> None:
            results.append(cluster.chat(f"race {trial}", via=1, timeout=15))

        sender = threading.Timer(trial / 1000, chat, args=(trial, results))
        sender.start()
        assert cluster.delete(instance, via=1).status_code == 200
        sender.join(20)
        assert results, f"trial {trial}: the chat didn't return"
        (result,) = results
        assert "within" not in (result.error or ""), (
            f"trial {trial}: {result.summary()}"
        )
        assert not result.problems(), result.problems()
        cluster.wait_for(
            lambda deleted=instance: _instance_gone(cluster, deleted, via=1),
            timeout=10,
            what="the deletion",
        )


# --- network cuts ----------------------------------------------------------------------


@pytest.mark.slow
def test_a_cut_node_crowns_itself_and_rejoins_after_the_heal(
    make_cluster: MakeCluster,
) -> None:
    """A follower holding half of an instance is cut off from both others (blackhole) for
    20 s. Main: the cut node makes itself master of a one-node cluster (measured +13.4 s;
    NETWORK N2: +13.6-13.9 s on real hosts), the majority removes it and deletes the
    instance (~16 s); after the heal it merges back with one master and no ghosts (measured
    3.5 s; bound 30 s). NETWORK N1, N2; D2."""
    cluster = make_cluster(3)
    instance = cluster.launch(on=(1, 2))
    chat = _streaming(cluster, "cut off " + LONG, via=0)
    cluster.isolate(2)
    cluster.wait_for(
        lambda: cluster.nodes[2].master() == cluster.nodes[2].node_id,
        timeout=25,
        what="the cut node to make itself master",
    )
    _clean_error(chat.result(60))
    time.sleep(max(0.0, 20 - STREAM_FIRST))
    cluster.heal()
    cluster.wait_until_consistent(timeout=30)
    assert _instance_gone(cluster, instance, via=0)


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason="B1/D2: a node cut off for longer than ~11 s loses its models, and the cut "
    "node deletes every model in its own view (NETWORK N1, N2; membership REPORT "
    "section 2: a 20 s cut deleted at +16 s on main)",
)
def test_a_20s_cut_keeps_the_model(make_cluster: MakeCluster) -> None:
    cluster = make_cluster(3)
    instance = cluster.launch(on=(1, 2))
    cluster.isolate(2)
    time.sleep(20)
    cluster.heal()
    cluster.wait_until_consistent(timeout=30)
    assert not _instance_gone(cluster, instance, via=0), "the model was deleted"


# --- configuration ---------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason="Found by this harness: each node probes its peers' /node_id at its own "
    "--api-port, so nodes started with different API ports never get a topology edge, and "
    "every placement across them is refused",
)
def test_nodes_with_different_api_ports_can_run_a_model_together(
    make_cluster: MakeCluster,
) -> None:
    cluster = make_cluster(2, probe_own_port=True, require_edges=False)
    try:
        cluster.launch(on=(0, 1), timeout=30)
    except HarnessError as error:
        pytest.fail(str(error))
