"""Two nodes in separate processes, with real discovery and zenoh: they connect only when they
run the same wire version.

Marked slow: each case waits out a fixed window, and discovery needs to send IPv6 multicast,
which macOS allows only to a process with Local Network access (for one started over ssh, or
from a terminal app that has it).
"""

import json
import socket
import subprocess
import sys
import uuid
from typing import cast

import pytest

from exo.routing.wire import WIRE_VERSION

# Matched nodes heard each other within 1.2 s on a laptop (a Hello goes out every second, then
# a zenoh session and the subscriptions form): 10 s is a wide margin, and is the window in
# which mismatched nodes must not connect.
_WINDOW_SECONDS = 10.0
# Two nodes on one machine share discovery's port. macOS hands a reply sent to that port to
# the socket bound first, so only the first node discovers the other, and a node dials only
# a peer whose id is smaller than its own: the node with the greater id starts first.
_FIRST_ID = "f" * 32
_SECOND_ID = "1" * 32


def _free_port(kind: socket.SocketKind) -> int:
    with socket.socket(socket.AF_INET6, kind) as sock:
        sock.bind(("::", 0))
        return cast(int, sock.getsockname()[1])


def _run_pair(first_version: int, second_version: int) -> list[dict[str, object]]:
    """Run two nodes in one fresh namespace until the window ends; return their reports."""
    namespace = f"exo-wire-gate-test-{uuid.uuid4().hex[:12]}"
    discovery_port = _free_port(socket.SOCK_DGRAM)

    def start(identity: str, version: int) -> subprocess.Popen[str]:
        peer = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "exo.routing.tests.wire_gate_peer",
                "--identity",
                identity,
                "--namespace",
                namespace,
                "--wire-version",
                str(version),
                "--zenoh-port",
                str(_free_port(socket.SOCK_STREAM)),
                "--discovery-port",
                str(discovery_port),
                "--seconds",
                str(_WINDOW_SECONDS),
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        assert peer.stdout is not None
        assert peer.stdout.readline().strip() == "ready"
        return peer

    peers = [start(_FIRST_ID, first_version), start(_SECOND_ID, second_version)]
    reports: list[dict[str, object]] = []
    for peer in peers:
        output, _ = peer.communicate(timeout=_WINDOW_SECONDS + 30)
        assert peer.returncode == 0, output
        reports.append(
            cast(dict[str, object], json.loads(output.strip().splitlines()[-1]))
        )
    return reports


@pytest.mark.slow
def test_nodes_on_the_same_wire_version_connect() -> None:
    reports = _run_pair(WIRE_VERSION, WIRE_VERSION)
    for report in reports:
        assert report["heard_after"] is not None, reports
        assert report["other_versions"] == [], reports


@pytest.mark.slow
def test_nodes_on_different_wire_versions_never_connect() -> None:
    newer = WIRE_VERSION + 1
    older_report, newer_report = _run_pair(WIRE_VERSION, newer)
    assert older_report["heard_after"] is None, (older_report, newer_report)
    assert newer_report["heard_after"] is None, (older_report, newer_report)
    # Each says a device in its namespace runs another version, and which
    older_heard = cast(list[list[object]], older_report["other_versions"])
    newer_heard = cast(list[list[object]], newer_report["other_versions"])
    assert older_heard and {version for _, version in older_heard} == {newer}
    assert newer_heard and {version for _, version in newer_heard} == {WIRE_VERSION}
