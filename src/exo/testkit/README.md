# exo.testkit: cluster tests on one machine

`exo.testkit` starts N real exo nodes as separate processes on one machine, with no MLX and
no GPU, and lets a test break them: kill -9, SIGSTOP/SIGCONT, quit, restart, and network
cuts. Everything is exo's production code except the model, which a stub engine replaces
with deterministic tokens, so a test can check every token a client receives.

## Run the tests

```bash
uv run pytest src/exo/testkit                  # fast set (also part of the default run): ~3 min
uv run pytest src/exo/testkit -m slow          # nightly set: ~10 min
uv run pytest -m "not slow and not cluster"    # everything except cluster tests
```

A failing test's report includes each cluster's timeline and the end of every node's log.
The files stay in pytest's `tmp_path` (`<cluster>/n0/exo_log/exo.log`, `n0/output.log`,
`timeline.log`).

## Write a scenario

```python
import time

import pytest

from exo.testkit.tests.conftest import MakeCluster

pytestmark = pytest.mark.cluster


def test_killing_a_follower_ends_its_chat(make_cluster: MakeCluster) -> None:
    cluster = make_cluster(3)                    # started, one master, edges: ready
    instance = cluster.launch(on=(1, 2))         # a 2-node pipeline ring instance, ready
    chat = cluster.chat_in_background("story [stub tokens=3000 rate=50]", via=0)
    time.sleep(3)
    cluster.kill(2)                              # kill -9 of n2 and its runners
    result = chat.result(timeout=60)
    assert not result.ok and not result.problems()   # clean error, no corrupted tokens
    cluster.wait_until_consistent(timeout=35)    # one master, no ghosts, no leaked tasks
```

Write the expectation as what main does today. For a failure main still has, write the
behaviour you want and mark the test `@pytest.mark.xfail(strict=True, reason=...)`, citing
the evidence. The PR that fixes it deletes the marker; strict means an unexpected pass fails
the run. Give each timing bound the value you measured and the margin you added, in the
docstring. Mark tests that take more than about 30 s `@pytest.mark.slow`; they run nightly.

## The `Cluster` API (exo/testkit/cluster.py)

Nodes are addressed by index; `n0` has the largest id, so it is the first master.

| what | call |
|---|---|
| start and stop | `Cluster(size, root)`, `.start()`, `.close()` (or the `make_cluster` fixture) |
| faults | `.kill(i)` (SIGKILL to the node's process group), `.pause(i)` / `.resume(i)` (SIGSTOP / SIGCONT), `.quit(i)` (SIGINT; returns seconds to exit), `.restart(i, new_identity=True)` |
| network | `.cut(i, j, mode)`, `.isolate(i, mode)`, `.heal()`. `mode="blackhole"` holds every byte, as when a cable is pulled; `"reset"` tears connections down at once |
| models | `.launch(on=(1, 2))` (placed on exactly those nodes, then waits until ready), `.launch()` (exo places it), `.place(...)` (doesn't wait for ready), `.wait_ready(id)`, `.delete(id)` |
| chats | `.chat(prompt, via=i, stream=True, max_tokens=None)` returns a `ChatResult`; `.chat_in_background(...)` returns a handle with `.result(timeout)` |
| results | `ChatResult.ok`, `.error`, `.content`, `.finish_reason`, `.problems()` (empty when every token is right; a reply cut short by an error must be an exact prefix), `.longest_gap`, `.summary()` |
| state | `.state(i)` (the node's `/state` as exo's `State`), `.master()`, `.masters()`, `.wait_for_master(timeout, other_than=)`, `.violations()`, `.wait_until_consistent(timeout)`, `.instance_nodes(id)` |
| helpers | `.wait_for(condition, timeout=, what=)` polls every 0.25 s and returns the condition's first truthy value: None, False, 0 and empty values mean "not yet" (it raises `HarnessError` on timeout, and at once if a node died by itself); `.note(message)` (adds a line to the timeline) |

`violations()` checks the invariants: one master that every running node follows; every
running node sees exactly the running nodes (no ghosts, none missing) and the same
instances; no instance on a node outside the cluster; no task left Pending or Running
(`allow_tasks=True` while requests are in flight).

## The stub engine's prompts

The reply to a prompt is a function of the prompt: token i is `"<key>.<i> "`, where key is the
first 8 hex digits of the prompt's sha256. Give each request its own prompt. A `[stub ...]` tag
in the prompt sets the reply and its faults:

| directive | meaning (default) |
|---|---|
| `tokens=N` | reply length (16); `max_tokens` below it ends early with `length` |
| `rate=R` | tokens per second per request (200) |
| `prefill=S` | seconds before the first token (0) |
| `fault=crash\|raise\|hang\|stall at=K rank=R stall=S` | rank R's runner, just before token K: dies by SIGKILL, raises, stops forever (the supervisor's 90 s stall check ends it), or stops S seconds (10) |

Faults on the way to serving (every runner of a node, each time) go in the node's stub config:
`exo.testkit.stub.config.StubConfig(connect_seconds=, load_seconds=, warmup_seconds=,
fault_phase="connect"|"load"|"warmup", fault=...)`, written with `write_config` to
`cluster.nodes[i].stub_config_path`. It is read at the start of each phase.

## What is real and what isn't

Real, and unchanged: discovery's Hello/WhatsUp protocol and dial rule, zenoh, the election,
the master, workers, the API, downloads (models are pre-installed and marked complete), the
runner supervisor, and the runner process with its state machine and its error handling.

Replaced or simulated:

- **The model.** `StubBuilder`/`StubEngine` (stub/engine.py) are run by exo's own bootstrap
  in place of MLX (stub/entrypoint.py). The ranks of a multi-node instance agree on every
  step's requests and cancellations over a TCP ring built from the instance's placement
  (stub/ring.py), as the real generators agree through MLX collectives. A rank that loses
  its ring prints MLX's ring error lines and stops, which is what exo's supervisor watches
  for. Each rank listens on the placement's port plus its rank, because every node is on
  one machine.
- **Discovery's transport.** Each node sends its Hellos to the harness's relay on `[::1]`
  (`EXO_DISCOVERY_ANNOUNCE_TO`, the one hook in production code,
  rust/networking/src/discovery.rs). On one machine multicast can't work: nodes share the
  discovery port and a unicast reply reaches only one of them, and on macOS a process
  without Local Network access can't send multicast. The relay passes Hellos between nodes
  and rewrites each reply's zenoh port to a per-pair proxy.
- **The network.** Every zenoh link and every stub ring link passes through a proxy in the
  harness (netsim.py), which is what lets a test cut a pair without root.
- **Three things in each node process** (node.py): its node id is chosen by the harness;
  runners use the stub; and the worker's reachability probe asks each peer at that peer's
  own API port, only on 127.0.0.1, and respects cuts. exo itself probes every peer at its
  own API port (see `test_nodes_with_different_api_ports_can_run_a_model_together`).

What this can't show: anything MLX or Metal does (GPU timeouts, real load and warm-up times,
memory), Thunderbolt and RDMA, real sleep (a SIGSTOP keeps sockets open and the clock
running), and real Wi-Fi loss, latency or packet loss. Use real hosts for those.
