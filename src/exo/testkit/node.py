"""Run one exo node for the harness: `python -m exo.testkit.node NODE_ID [exo arguments...]`.

This is exo's own `exo.main.main()`, with three changes made in this process before it starts:

1. The node id is NODE_ID instead of a random one, so the harness knows every node's id (and so
   which one an election picks) before it starts. A restart "as a new device" passes a new id.
2. Runners use the stub engine (exo.testkit.stub) instead of MLX.
3. The worker's reachability probe asks each node at that node's own API port (every node of
   a real cluster listens on the same port, here each has its own), only on 127.0.0.1, and not
   across a pair the harness has cut.
4. Where exo has a data plane (exo.shared.serving, not on main), a node reaches a peer's API
   through the simulated network's proxy for the pair, at the peer's own port, so a cut
   applies to it too. Without that module this does nothing.

Everything else - discovery, zenoh, election, master, worker, API, downloads, runners - is the
production code, unchanged.
"""

import importlib
import importlib.util
import os
import sys

import anyio
import httpx

import exo.main
import exo.utils.info_gatherer.net_profile as net_profile
import exo.worker.runner.supervisor as supervisor
from exo.shared.types.common import NodeId
from exo.testkit.network_table import PROBE_OWN_PORT_ENV, read_table
from exo.testkit.stub.entrypoint import entrypoint as stub_entrypoint

# A probe of a blackholed address fails only after every attempt times out: httpx's 5 s
# timeout plus net_profile's 1 s pause, REACHABILITY_ATTEMPTS times
BLACKHOLED_PROBE_SECONDS = net_profile.REACHABILITY_ATTEMPTS * (5.0 + 1.0)

_real_check_reachability = net_profile.check_reachability


def _probe_for(self_id: NodeId):
    async def check_reachability(
        target_ip: str,
        expected_node_id: NodeId,
        out: dict[NodeId, set[str]],
        client: httpx.AsyncClient,
        api_port: int,
    ) -> None:
        if target_ip != "127.0.0.1":
            return
        table = read_table()
        port = (
            api_port
            if os.environ.get(PROBE_OWN_PORT_ENV) == "1"
            else table.api_ports.get(expected_node_id)
        )
        if port is None:
            return
        match table.cut_between(self_id, expected_node_id):
            case "blackhole":
                await anyio.sleep(BLACKHOLED_PROBE_SECONDS)
                return
            case "reset":
                return
            case None:
                pass
        await _real_check_reachability(
            target_ip, expected_node_id, out, client, api_port=port
        )

    return check_reachability


# The data plane's module and the function that gives the URLs one node uses to reach
# another's API: exo's assumes every node serves its API on the same port
SERVING_MODULE = "exo.shared.serving"


def node_urls(
    _state: object, source: NodeId, sink: NodeId, _api_port: int
) -> list[str]:
    """Where `source` reaches `sink`'s API: the pair's proxy in the simulated network."""
    port = read_table().api_routes.get(source, {}).get(sink)
    return [] if port is None else [f"http://[::1]:{port}"]


def route_node_urls() -> bool:
    """Point the data plane's node_urls at the simulated network; False where there is none."""
    if importlib.util.find_spec(SERVING_MODULE) is None:
        return False
    serving = importlib.import_module(SERVING_MODULE)
    if "node_urls" not in serving.__dict__:
        return False
    serving.__dict__["node_urls"] = node_urls
    return True


def install(node_id: NodeId) -> None:
    exo.main.get_node_zid = lambda: node_id
    supervisor.entrypoint = stub_entrypoint
    net_profile.check_reachability = _probe_for(node_id)
    route_node_urls()


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m exo.testkit.node NODE_ID [exo arguments...]")
    node_id = NodeId(sys.argv[1])
    install(node_id)
    sys.argv = ["exo", *sys.argv[2:]]
    exo.main.main()


if __name__ == "__main__":
    main()
