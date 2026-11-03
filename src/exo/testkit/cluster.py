"""Start N real exo nodes on this machine and drive them: faults, cuts, requests, state checks.

Every node is a separate `python -m exo.testkit.node` process with its own ports, EXO_HOME and
process group, in a namespace no other cluster uses. Nodes find each other with exo's own
discovery and talk over zenoh, both through the simulated network (exo.testkit.netsim), and
run models with the stub engine. See README.md for how to write a scenario with it.
"""

import atexit
import contextlib
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Self, final

import httpx
import tomlkit
from pydantic import BaseModel, ConfigDict, ValidationError

from exo.api.types import (
    CreateInstanceParams,
    PlacementPreviewResponse,
    PlaceInstanceParams,
)
from exo.download.hub import RepoFile
from exo.download.store import Manifest, write_manifest
from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import ModelId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.tasks import TaskStatus
from exo.shared.types.worker.instances import Instance, InstanceId, InstanceMeta
from exo.shared.types.worker.runners import RunnerReady, RunnerRunning
from exo.shared.types.worker.shards import Sharding
from exo.testkit.netsim import ANNOUNCE_TO_ENV, NetSim
from exo.testkit.network_table import (
    NETWORK_TABLE_ENV,
    Cut,
    CutMode,
    NetworkTable,
    write_table,
)
from exo.testkit.stub.config import (
    RING_PROXY_ENV,
    STUB_CONFIG_ENV,
    StubConfig,
    write_config,
)
from exo.testkit.stub.directives import expected_reply

STUB_MODEL = ModelId("exo-testkit/stub-8l")
_MASTER_LINE = re.compile(
    r"Node elected Master - (?:promoting|maintaining) self|Node (?P<other>[0-9a-f]+) elected master"
)
_POLL_SECONDS = 0.25
_HTTP_TIMEOUT = 5.0
_QUIT_TIMEOUT = 30.0
# The environment a node runs with is only what the harness sets: none of the caller's exo
# settings (models dirs, namespace, offline, ...) leak into a test cluster
_INHERITED_ENV_PREFIXES = ("EXO_", "HF_")
# ...except where exo finds its own files, when it runs from a bundle or a CI checkout
_PASSED_THROUGH_ENV = ("EXO_RESOURCES_DIR",)


class HarnessError(Exception):
    pass


@final
@dataclass(frozen=True)
class StubModel:
    """A model card the stub engine serves; every node has its files, already downloaded."""

    model_id: ModelId = STUB_MODEL
    n_layers: int = 8
    storage_bytes: int = 64 * 1024 * 1024
    supports_tensor: bool = True

    def card(self) -> ModelCard:
        return ModelCard(
            model_id=self.model_id,
            storage_size=Memory.from_bytes(self.storage_bytes),
            n_layers=self.n_layers,
            hidden_size=1024,
            supports_tensor=self.supports_tensor,
            num_key_value_heads=8,
            tasks=[ModelTask.TextGeneration],
            backends=[Backend.MlxMetal, Backend.MlxCpu],
            context_length=8192,
            trust_remote_code=False,
        )


def _install_model(home: Path, model: StubModel) -> None:
    cards = home / "custom_model_cards"
    cards.mkdir(parents=True, exist_ok=True)
    card = model.card()
    document = card.model_dump(mode="json", exclude_none=True, exclude={"is_custom"})
    (cards / f"{model.model_id.normalize()}.toml").write_text(tomlkit.dumps(document))  # pyright: ignore[reportUnknownMemberType]
    model_dir = home / "models" / model.model_id.normalize()
    model_dir.mkdir(parents=True, exist_ok=True)
    weights = model_dir / "model.safetensors"
    with weights.open("wb") as file:
        file.truncate(model.storage_bytes)
    write_manifest(
        model_dir,
        Manifest(
            repo=model.model_id,
            commit="testkit",
            files=[RepoFile(path=weights.name, size=model.storage_bytes)],
            complete=True,
        ),
    )


def _free_port(kind: int, family: int, host: str) -> int:
    with socket.socket(family, kind) as probe:
        probe.bind((host, 0))
        port = probe.getsockname()[1]  # pyright: ignore[reportAny]
        assert isinstance(port, int)
        return port


def _free_ports() -> tuple[int, int, int]:
    """An API, a zenoh and a discovery port that nothing is using."""
    return (
        _free_port(socket.SOCK_STREAM, socket.AF_INET, "0.0.0.0"),
        _free_port(socket.SOCK_STREAM, socket.AF_INET6, "::"),
        _free_port(socket.SOCK_DGRAM, socket.AF_INET6, "::"),
    )


NodeStatus = Literal["running", "paused", "stopped"]


@final
@dataclass(eq=False)
class Node:
    """One node slot. Its ports and EXO_HOME stay the same across restarts; its id may not."""

    index: int
    name: str
    api_port: int
    zenoh_port: int
    discovery_port: int
    home: Path
    run_token: str
    generation: int = 0
    process: subprocess.Popen[bytes] | None = None
    status: NodeStatus = "stopped"
    started_at: float = 0.0

    @property
    def node_id(self) -> NodeId:
        # The first digit orders the nodes for elections: with nothing else to choose by,
        # exo's election picks the largest id, so n0 is the first master
        return NodeId(f"{0xF - self.index:x}{self.generation:02x}{self.run_token}")

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.api_port}"

    @property
    def log_path(self) -> Path:
        return self.home / "exo_log" / "exo.log"

    @property
    def output_path(self) -> Path:
        return self.home / "output.log"

    @property
    def stub_config_path(self) -> Path:
        return self.home / "stub.json"

    def log(self) -> str:
        try:
            return self.log_path.read_text(errors="replace")
        except FileNotFoundError:
            return ""

    def master(self) -> NodeId | None:
        """The master this node last said it follows, from its log; None before any election."""
        found: NodeId | None = None
        for match in _MASTER_LINE.finditer(self.log()):
            other = match.group("other")
            found = NodeId(other) if other else self.node_id
        return found


# --- requests ----------------------------------------------------------------------------


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="ignore")


class _ErrorInfo(_Lenient):
    message: str = ""
    code: int | str | None = None


class _Delta(_Lenient):
    content: str | None = None


class _StreamChoice(_Lenient):
    delta: _Delta | None = None
    finish_reason: str | None = None


class _StreamEvent(_Lenient):
    choices: list[_StreamChoice] = []
    error: _ErrorInfo | None = None


class _Message(_Lenient):
    content: str | None = None


class _Choice(_Lenient):
    message: _Message
    finish_reason: str | None = None


class _Completion(_Lenient):
    choices: list[_Choice] = []
    error: _ErrorInfo | None = None


class _Detail(_Lenient):
    detail: str | None = None
    error: _ErrorInfo | None = None


@final
@dataclass
class ChatResult:
    """What a client saw of one chat request."""

    prompt: str
    via: str
    stream: bool
    max_tokens: int | None
    started: float
    ended: float = 0.0
    status_code: int | None = None
    content: str = ""
    finish_reason: str | None = None
    error: str | None = None
    # When each piece of content arrived (streaming only), seconds since `started`
    arrivals: list[float] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None and self.finish_reason in ("stop", "length")

    @property
    def duration(self) -> float:
        return self.ended - self.started

    @property
    def longest_gap(self) -> float:
        """The longest wait between two pieces of the reply (or for the first one)."""
        times = [0.0, *self.arrivals]
        return max((b - a for a, b in zip(times, times[1:], strict=False)), default=0.0)

    def problems(self) -> list[str]:
        """Everything wrong with the reply, checked against the stub's deterministic tokens.

        A complete reply must be exactly the expected tokens. A reply that ended in an error
        must be an exact prefix of them: an error may cut a reply short, but never corrupt,
        reorder, repeat or drop tokens before it."""
        expected = expected_reply(self.prompt, self.max_tokens)
        problems: list[str] = []
        if self.ok:
            if self.content != expected.text:
                problems.append(_describe_mismatch(self.content, expected.tokens))
            if self.finish_reason != expected.finish_reason:
                problems.append(
                    f"finish_reason {self.finish_reason}, expected {expected.finish_reason}"
                )
        elif not expected.text.startswith(self.content):
            problems.append(
                "an interrupted reply isn't a prefix of the expected one: "
                + _describe_mismatch(self.content, expected.tokens)
            )
        return problems

    def summary(self) -> str:
        outcome = (
            f"ok ({self.finish_reason})"
            if self.ok
            else f"error {self.status_code}: {self.error}"
        )
        return (
            f"chat via {self.via} {'stream' if self.stream else 'whole'}: {outcome} after "
            f"{self.duration:.1f}s, {len(self.content.split())} tokens, longest gap "
            f"{self.longest_gap:.1f}s"
        )


def _describe_mismatch(content: str, expected: tuple[str, ...]) -> str:
    got = content.split(" ")
    if got and got[-1] == "":
        got = got[:-1]
    want = [token.strip() for token in expected]
    for index, (a, b) in enumerate(zip(got, want, strict=False)):
        if a != b:
            return f"token {index} is {a!r}, expected {b!r}"
    if len(got) != len(want):
        return f"{len(got)} tokens, expected {len(want)}"
    return "content differs in whitespace"


@final
class ChatHandle:
    """A chat running in the background."""

    def __init__(self, run: Callable[[], ChatResult]) -> None:
        self._result: ChatResult | None = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, args=(run,), daemon=True)
        self._thread.start()

    def _run(self, run: Callable[[], ChatResult]) -> None:
        try:
            self._result = run()
        except BaseException as error:  # handed to whoever waits for the result
            self._error = error

    def done(self) -> bool:
        return not self._thread.is_alive()

    def result(self, timeout: float) -> ChatResult:
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise HarnessError(f"the chat didn't end within {timeout:.0f}s")
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


# --- the cluster -------------------------------------------------------------------------


@final
class Cluster:
    """N exo nodes on this machine. Use it as a context manager; `start()` waits for a cluster.

    Nodes are addressed by index (0 is n0). Unless a test arranges otherwise, n0 becomes the
    first master (it has the largest id).
    """

    def __init__(
        self,
        size: int,
        root: Path,
        *,
        models: tuple[StubModel, ...] = (StubModel(),),
        stub: StubConfig | None = None,
        verbose: bool = False,
    ) -> None:
        if not 1 <= size <= 15:
            raise ValueError("a test cluster has 1 to 15 nodes")
        self.root = root
        self.models = models
        self.verbose = verbose
        self.namespace = f"exo-testkit-{uuid.uuid4().hex[:12]}"
        self.network = NetSim()
        self.network_table_path = root / "network.json"
        self.timeline_path = root / "timeline.log"
        self._cuts: dict[frozenset[int], CutMode] = {}
        self._t0 = time.monotonic()
        run_token = secrets.token_hex(7)
        self.nodes: list[Node] = []
        for index in range(size):
            api_port, zenoh_port, discovery_port = _free_ports()
            self.nodes.append(
                Node(
                    index=index,
                    name=f"n{index}",
                    api_port=api_port,
                    zenoh_port=zenoh_port,
                    discovery_port=discovery_port,
                    home=root / f"n{index}",
                    run_token=run_token,
                )
            )
        self._stub = stub or StubConfig()

    # --- lifecycle -----------------------------------------------------------------------

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def start(self, timeout: float = 90.0) -> None:
        """Start every node and wait until they form one cluster with one master."""
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "dashboard").mkdir(exist_ok=True)
        self.network.start()
        for node in self.nodes:
            self.network.add_node(node.name, node.discovery_port, node.zenoh_port)
            self._prepare(node)
        atexit.register(self.close)
        for node in self.nodes:
            self._spawn(node)
        self.wait_for_cluster(timeout=timeout)

    def close(self) -> None:
        """Stop every node (gracefully, then by force) and the network."""
        for node in self.nodes:
            if node.process is not None and node.status == "paused":
                self._signal_group(node, signal.SIGCONT)
            if node.process is not None and node.process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    node.process.send_signal(signal.SIGINT)
        deadline = time.monotonic() + 10
        for node in self.nodes:
            if node.process is None:
                continue
            with contextlib.suppress(subprocess.TimeoutExpired):
                node.process.wait(max(0.1, deadline - time.monotonic()))
            self._signal_group(node, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                node.process.wait(5)
            node.status = "stopped"
        self.network.stop()
        with contextlib.suppress(Exception):
            atexit.unregister(self.close)

    def _prepare(self, node: Node) -> None:
        node.home.mkdir(parents=True, exist_ok=True)
        # The first start of exo opens the dashboard in a browser (and blocks while it does)
        (node.home / ".dashboard_opened").touch()
        for model in self.models:
            _install_model(node.home, model)
        write_config(node.stub_config_path, self._stub)

    def _environment(self, node: Node) -> dict[str, str]:
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(_INHERITED_ENV_PREFIXES)
            or key in _PASSED_THROUGH_ENV
        }
        environment.update(
            {
                "EXO_HOME": str(node.home),
                # The API serves the dashboard from here; tests don't need it built
                "EXO_DASHBOARD_DIR": str(self.root / "dashboard"),
                "EXO_DEFAULT_MODELS_DIR": str(node.home / "models"),
                "HF_HOME": str(node.home / "hf"),
                "HF_HUB_OFFLINE": "1",
                "PYTHONUNBUFFERED": "1",
                ANNOUNCE_TO_ENV: f"[::1]:{self.network.relay_port}",
                RING_PROXY_ENV: str(self.network.ring_proxy_port),
                STUB_CONFIG_ENV: str(node.stub_config_path),
                NETWORK_TABLE_ENV: str(self.network_table_path),
            }
        )
        return environment

    def _spawn(self, node: Node) -> None:
        self.network.set_node_id(node.name, node.node_id)
        self._write_network_table()
        arguments = [
            sys.executable,
            "-m",
            "exo.testkit.node",
            node.node_id,
            "--namespace",
            self.namespace,
            "--api-port",
            str(node.api_port),
            "--zenoh-port",
            str(node.zenoh_port),
            "--discovery-port",
            str(node.discovery_port),
            "--offline",
        ]
        if self.verbose:
            arguments.append("-v")
        with node.output_path.open("ab") as output:
            output.write(
                f"\n=== {node.name} generation {node.generation} id {node.node_id}\n".encode()
            )
            output.flush()
            node.process = subprocess.Popen(
                arguments,
                env=self._environment(node),
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=node.home,
            )
        node.status = "running"
        node.started_at = time.monotonic()
        self.note(f"started {node.name} id {node.node_id} pid {node.process.pid}")

    def _write_network_table(self) -> None:
        write_table(
            self.network_table_path,
            NetworkTable(
                api_ports={node.node_id: node.api_port for node in self.nodes},
                cuts=[
                    Cut(
                        between=(self.nodes[a].node_id, self.nodes[b].node_id),
                        mode=mode,
                    )
                    for a, b, mode in self._cut_list()
                ],
            ),
        )

    def _cut_list(self) -> list[tuple[int, int, CutMode]]:
        cuts: list[tuple[int, int, CutMode]] = []
        for pair, mode in self._cuts.items():
            a, b = sorted(pair)
            cuts.append((a, b, mode))
        return cuts

    # --- timeline ------------------------------------------------------------------------

    def now(self) -> float:
        """Seconds since the cluster object was made: the timeline's clock."""
        return time.monotonic() - self._t0

    def note(self, message: str) -> None:
        line = f"{self.now():8.2f} {message}\n"
        with contextlib.suppress(OSError):
            with self.timeline_path.open("a") as timeline:
                timeline.write(line)

    # --- node faults ---------------------------------------------------------------------

    def running(self) -> list[Node]:
        return [node for node in self.nodes if node.status == "running"]

    def _signal_group(self, node: Node, signal_number: signal.Signals) -> None:
        if node.process is None:
            return
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(node.process.pid, signal_number)

    def kill(self, index: int) -> None:
        """kill -9 the node and its runners at once: a power cut, or the OS killing exo."""
        node = self.nodes[index]
        self._signal_group(node, signal.SIGKILL)
        if node.process is not None:
            node.process.wait(10)
        node.status = "stopped"
        self.note(f"kill -9 {node.name}")

    def pause(self, index: int) -> None:
        """SIGSTOP the node and its runners: a laptop going to sleep (its sockets stay open)."""
        node = self.nodes[index]
        self._signal_group(node, signal.SIGSTOP)
        node.status = "paused"
        self.note(f"SIGSTOP {node.name}")

    def resume(self, index: int) -> None:
        node = self.nodes[index]
        self._signal_group(node, signal.SIGCONT)
        node.status = "running"
        self.note(f"SIGCONT {node.name}")

    def quit(self, index: int, timeout: float = _QUIT_TIMEOUT) -> float:
        """SIGINT exo, as quitting it does, and wait for it to exit. Returns how long it took.

        Raises HarnessError if it doesn't exit in time (the node is then killed)."""
        node = self.nodes[index]
        assert node.process is not None
        started = time.monotonic()
        self.note(f"SIGINT {node.name}")
        node.process.send_signal(signal.SIGINT)
        try:
            node.process.wait(timeout)
        except subprocess.TimeoutExpired as error:
            self.kill(index)
            raise HarnessError(f"{node.name} didn't exit within {timeout:.0f}s") from error
        took = time.monotonic() - started
        # Anything of its group still alive (a runner) outlived the node: end it
        self._signal_group(node, signal.SIGKILL)
        node.status = "stopped"
        self.note(f"{node.name} exited after {took:.1f}s, code {node.process.returncode}")
        return took

    def restart(self, index: int, *, new_identity: bool = True) -> None:
        """Start a stopped node again. exo picks a new node id on every start, so by default
        it comes back as a new device; `new_identity=False` keeps the old id."""
        node = self.nodes[index]
        if node.status != "stopped":
            raise HarnessError(f"{node.name} is {node.status}; stop it before restarting")
        if new_identity:
            node.generation += 1
        self._spawn(node)

    # --- network cuts --------------------------------------------------------------------

    def cut(self, a: int, b: int, mode: CutMode = "blackhole") -> None:
        """Cut the link between two nodes in both directions (see exo.testkit.netsim)."""
        self._cuts[frozenset((a, b))] = mode
        self._write_network_table()
        self.network.cut(self.nodes[a].name, self.nodes[b].name, mode)
        self.note(f"cut {self.nodes[a].name}-{self.nodes[b].name} ({mode})")

    def isolate(self, index: int, mode: CutMode = "blackhole") -> None:
        for other in range(len(self.nodes)):
            if other != index:
                self.cut(index, other, mode)

    def heal(self) -> None:
        for pair in list(self._cuts):
            a, b = sorted(pair)
            self.network.heal(self.nodes[a].name, self.nodes[b].name)
        self._cuts.clear()
        self._write_network_table()
        self.note("healed every cut")

    # --- API -----------------------------------------------------------------------------

    def get(self, index: int, path: str, timeout: float = _HTTP_TIMEOUT) -> httpx.Response:
        return httpx.get(self.nodes[index].url + path, timeout=timeout)

    def state(self, index: int, timeout: float = _HTTP_TIMEOUT) -> State | None:
        """The node's view of the cluster; None if it doesn't answer."""
        try:
            response = self.get(index, "/state", timeout=timeout)
            response.raise_for_status()
            return State.model_validate_json(response.content)
        except (httpx.HTTPError, ValidationError):
            return None

    def place(
        self,
        *,
        on: tuple[int, ...] | None = None,
        model: ModelId = STUB_MODEL,
        sharding: Sharding = Sharding.Pipeline,
        via: int = 0,
        min_nodes: int = 1,
    ) -> InstanceId:
        """Launch a model and return its instance id once the master has created it.

        With `on`, the instance runs on exactly those nodes (as when the dashboard's user picks
        devices); otherwise exo places it."""
        before = set(self._instances(via))
        if on is None:
            response = httpx.post(
                self.nodes[via].url + "/place_instance",
                content=PlaceInstanceParams(
                    model_id=model, sharding=sharding, min_nodes=min_nodes
                ).model_dump_json(),
                headers={"content-type": "application/json"},
                timeout=_HTTP_TIMEOUT,
            )
        else:
            instance = self.preview(on, model=model, sharding=sharding, via=via)
            response = httpx.post(
                self.nodes[via].url + "/instance",
                content=CreateInstanceParams(instance=instance).model_dump_json(),
                headers={"content-type": "application/json"},
                timeout=_HTTP_TIMEOUT,
            )
        if response.status_code != 200:
            raise HarnessError(f"launch refused ({response.status_code}): {response.text}")
        self.note(f"launch {model} on {on or 'any'} via n{via}")
        return self.wait_for(
            lambda: next(iter(set(self._instances(via)) - before), None),
            timeout=30,
            what="the new instance to appear",
        )

    def preview(
        self,
        on: tuple[int, ...],
        *,
        model: ModelId = STUB_MODEL,
        sharding: Sharding = Sharding.Pipeline,
        via: int = 0,
    ) -> Instance:
        wanted = {self.nodes[i].node_id for i in on}
        response = httpx.get(
            self.nodes[via].url + "/instance/previews",
            params=[("model_id", model), *(("node_ids", node_id) for node_id in wanted)],
            timeout=_HTTP_TIMEOUT,
        )
        response.raise_for_status()
        previews = PlacementPreviewResponse.model_validate_json(response.content)
        errors: list[str] = []
        for preview in previews.previews:
            instance = preview.instance
            if preview.error is not None:
                errors.append(preview.error)
            if (
                instance is not None
                and preview.sharding == sharding
                and preview.instance_meta == InstanceMeta.MlxRing
                and set(instance.shard_assignments.node_to_runner) == wanted
            ):
                return instance
        raise HarnessError(
            f"no {sharding} ring placement on {sorted(wanted)}: {sorted(set(errors))}"
        )

    def _instances(self, via: int) -> Mapping[InstanceId, Instance]:
        state = self.state(via)
        return {} if state is None else state.instances

    def wait_ready(self, instance_id: InstanceId, via: int = 0, timeout: float = 60) -> None:
        """Wait until every runner of the instance is ready to serve, in `via`'s view."""

        def ready() -> bool:
            state = self.state(via)
            if state is None or instance_id not in state.instances:
                return False
            runners = state.instances[instance_id].shard_assignments.runner_to_shard
            return all(
                isinstance(state.runners.get(runner), (RunnerReady, RunnerRunning))
                for runner in runners
            )

        self.wait_for(ready, timeout=timeout, what=f"instance {instance_id} to be ready")
        self.note(f"instance {instance_id} ready")

    def launch(
        self,
        on: tuple[int, ...] | None = None,
        *,
        model: ModelId = STUB_MODEL,
        sharding: Sharding = Sharding.Pipeline,
        via: int = 0,
        timeout: float = 60,
    ) -> InstanceId:
        """Place a model and wait until it serves."""
        instance_id = self.place(on=on, model=model, sharding=sharding, via=via)
        self.wait_ready(instance_id, via=via, timeout=timeout)
        return instance_id

    def delete(self, instance_id: InstanceId, via: int = 0) -> httpx.Response:
        self.note(f"delete {instance_id} via n{via}")
        return httpx.delete(
            f"{self.nodes[via].url}/instance/{instance_id}", timeout=_HTTP_TIMEOUT
        )

    def chat(
        self,
        prompt: str,
        *,
        via: int = 0,
        model: ModelId = STUB_MODEL,
        stream: bool = True,
        max_tokens: int | None = None,
        timeout: float = 120,
    ) -> ChatResult:
        """Send one chat completion and record exactly what came back."""
        node = self.nodes[via]
        result = ChatResult(
            prompt=prompt,
            via=node.name,
            stream=stream,
            max_tokens=max_tokens,
            started=time.monotonic(),
        )
        body: dict[str, object] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        try:
            if stream:
                self._stream_chat(node, body, result, timeout)
            else:
                self._whole_chat(node, body, result, timeout)
        except httpx.TimeoutException:
            result.error = f"no complete reply within {timeout:.0f}s"
        except httpx.HTTPError as error:
            result.error = f"{type(error).__name__}: {error}"
        result.ended = time.monotonic()
        self.note(f"{result.summary()} [{prompt[:40]}]")
        return result

    def chat_in_background(self, prompt: str, **options: object) -> ChatHandle:
        def run() -> ChatResult:
            return self.chat(prompt, **options)  # pyright: ignore[reportArgumentType]

        return ChatHandle(run)

    def _stream_chat(
        self, node: Node, body: dict[str, object], result: ChatResult, timeout: float
    ) -> None:
        deadline = time.monotonic() + timeout
        with httpx.stream(
            "POST",
            node.url + "/v1/chat/completions",
            json=body,
            timeout=httpx.Timeout(_HTTP_TIMEOUT, read=timeout),
        ) as response:
            result.status_code = response.status_code
            if response.status_code != 200:
                response.read()
                result.error = _error_text(response)
                return
            for line in response.iter_lines():
                if time.monotonic() > deadline:
                    result.error = f"no complete reply within {timeout:.0f}s"
                    return
                if not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ")
                if data == "[DONE]":
                    break
                event = _StreamEvent.model_validate_json(data)
                if event.error is not None:
                    result.error = event.error.message or "error"
                    continue
                for choice in event.choices:
                    if choice.delta is not None and choice.delta.content:
                        result.content += choice.delta.content
                        result.arrivals.append(time.monotonic() - result.started)
                    if choice.finish_reason is not None:
                        result.finish_reason = choice.finish_reason
        if result.error is None and result.finish_reason is None:
            result.error = "the stream ended without a finish reason"

    def _whole_chat(
        self, node: Node, body: dict[str, object], result: ChatResult, timeout: float
    ) -> None:
        response = httpx.post(
            node.url + "/v1/chat/completions",
            json=body,
            timeout=httpx.Timeout(_HTTP_TIMEOUT, read=timeout),
        )
        result.status_code = response.status_code
        if response.status_code != 200:
            result.error = _error_text(response)
            return
        completion = _Completion.model_validate_json(response.content)
        if completion.error is not None:
            result.error = completion.error.message or "error"
            return
        if completion.choices:
            result.content = completion.choices[0].message.content or ""
            result.finish_reason = completion.choices[0].finish_reason

    # --- waiting and checking ------------------------------------------------------------

    def wait_for[T](
        self, condition: Callable[[], T | None], *, timeout: float, what: str
    ) -> T:
        """Poll until `condition` returns something other than None or False. Raises at once
        if a node the harness believes is running has exited on its own."""
        deadline = time.monotonic() + timeout
        while True:
            self.check_alive()
            value = condition()
            if value is not None and value is not False:
                return value
            if time.monotonic() > deadline:
                raise HarnessError(f"timed out after {timeout:.0f}s waiting for {what}")
            time.sleep(_POLL_SECONDS)

    def check_alive(self) -> None:
        """Raise if a node that should be running or paused has exited by itself."""
        for node in self.nodes:
            if node.status == "stopped" or node.process is None:
                continue
            code = node.process.poll()
            if code is not None:
                node.status = "stopped"
                tail = node.output_path.read_text(errors="replace")[-2000:]
                self.note(f"{node.name} exited by itself with code {code}")
                raise HarnessError(f"{node.name} exited with code {code}:\n{tail}")

    def masters(self) -> dict[str, NodeId | None]:
        """Which master each running node follows, by node name."""
        return {node.name: node.master() for node in self.running()}

    def master(self) -> int | None:
        """The index of the master every running node follows, or None if they disagree."""
        followed = set(self.masters().values())
        if len(followed) != 1:
            return None
        (master_id,) = followed
        return next(
            (node.index for node in self.running() if node.node_id == master_id), None
        )

    def wait_for_master(self, timeout: float = 60, *, other_than: int | None = None) -> int:
        """Wait until every running node follows one running master; return its index."""

        def agreed() -> int | None:
            master = self.master()
            if master is None or master == other_than:
                return None
            return master

        master = self.wait_for(agreed, timeout=timeout, what="one master")
        self.note(f"master is {self.nodes[master].name}")
        return master

    def wait_for_cluster(self, timeout: float = 90) -> None:
        """Wait until the running nodes form one cluster: one master, every node seeing
        exactly the running nodes, and every pair able to reach the other (topology edges,
        which placement needs)."""
        self.wait_for(
            lambda: not self.violations(require_edges=True),
            timeout=timeout,
            what="the running nodes to form one cluster",
        )
        self.note(f"cluster formed: {[node.name for node in self.running()]}")

    def violations(
        self,
        *,
        require_edges: bool = False,
        allow_tasks: bool = False,
    ) -> list[str]:
        """Every invariant the running nodes break right now (empty when all hold):

        - one master, followed by every running node, and itself running;
        - every running node answers and sees exactly the running nodes (no ghosts, none
          missing) with the same instances;
        - no instance uses a node that isn't in the cluster;
        - no leaked tasks: none left Pending or Running (unless `allow_tasks`, for checks
          while requests are in flight);
        - with `require_edges`, every pair of running nodes has a topology edge.
        """
        problems: list[str] = []
        running = self.running()
        running_ids = {node.node_id for node in running}
        if self.master() is None:
            problems.append(f"no single master: {self.masters()}")
        views: dict[str, State] = {}
        for node in running:
            state = self.state(node.index)
            if state is None:
                problems.append(f"{node.name} doesn't answer /state")
            else:
                views[node.name] = state
        for name, state in views.items():
            seen = set(state.topology.list_nodes())
            if seen != running_ids:
                ghosts = {self._name_of(i) for i in seen - running_ids}
                missing = {self._name_of(i) for i in running_ids - seen}
                problems.append(f"{name} sees ghosts {ghosts} and misses {missing}")
            for instance_id, instance in state.instances.items():
                outside = set(instance.shard_assignments.node_to_runner) - running_ids
                if outside:
                    problems.append(
                        f"{name}: instance {instance_id} is on nodes outside the "
                        f"cluster {[self._name_of(i) for i in outside]}"
                    )
            if not allow_tasks:
                for task_id, task in state.tasks.items():
                    if task.task_status in (TaskStatus.Pending, TaskStatus.Running):
                        problems.append(
                            f"{name}: task {type(task).__name__} {task_id} is still "
                            f"{task.task_status}"
                        )
            if require_edges:
                for a in running:
                    for b in running:
                        if a is not b and not any(
                            True for _ in state.topology.get_all_connections_between(a.node_id, b.node_id)
                        ):
                            problems.append(f"{name}: no edge {a.name}->{b.name}")
        instance_views = {
            name: frozenset(state.instances) for name, state in views.items()
        }
        if len(set(instance_views.values())) > 1:
            problems.append(f"views disagree on instances: {instance_views}")
        return problems

    def wait_until_consistent(
        self, timeout: float, *, allow_tasks: bool = False
    ) -> float:
        """Wait until every invariant holds; return how long it took. Raises with the last
        violations if they don't all hold within `timeout`."""
        started = time.monotonic()
        last: list[str] = []

        def consistent() -> bool:
            nonlocal last
            last = self.violations(allow_tasks=allow_tasks)
            return not last

        try:
            self.wait_for(consistent, timeout=timeout, what="the cluster to agree")
        except HarnessError as error:
            raise HarnessError(f"{error}: {last}") from error
        took = time.monotonic() - started
        self.note(f"consistent after {took:.1f}s")
        return took

    def _name_of(self, node_id: NodeId) -> str:
        for node in self.nodes:
            if node.node_id == node_id:
                return node.name
        return f"unknown {node_id[:6]}"

    def instance_nodes(self, instance_id: InstanceId, via: int = 0) -> set[int]:
        state = self.state(via)
        if state is None or instance_id not in state.instances:
            return set()
        ids = set(state.instances[instance_id].shard_assignments.node_to_runner)
        return {node.index for node in self.nodes if node.node_id in ids}

    def logs(self) -> Iterator[tuple[str, str]]:
        for node in self.nodes:
            yield node.name, node.log()


def _error_text(response: httpx.Response) -> str:
    try:
        detail = _Detail.model_validate_json(response.content)
    except ValidationError:
        return response.text[:300]
    if detail.error is not None:
        return detail.error.message
    return detail.detail or response.text[:300]
