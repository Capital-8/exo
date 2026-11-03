"""One node for test_wire_gate.py, in its own process: real discovery and zenoh, no exo.

It joins ``--namespace`` with ``--wire-version`` and ``--mlx-release``, publishes on a test topic for ``--seconds``,
and prints one JSON line: how long until it heard another node on the topic (null if it
never did), and the devices on other wire versions its discovery heard.
"""

import argparse
import asyncio
import json
import os
import time
from typing import cast

from exo_rs import FromSwarm, NetworkingHandle

TOPIC = "wire_gate_test"


async def run(
    identity: str,
    namespace: str,
    wire_version: int,
    mlx: str | None,
    zenoh_port: int,
    discovery_port: int,
    seconds: float,
) -> None:
    handle = NetworkingHandle.new(
        identity,
        namespace,
        wire_version,
        mlx,
        zenoh_port,
        discovery_port,
    )
    # Discovery's socket is bound: the next node may start
    print("ready", flush=True)
    start = time.monotonic()
    heard_after: float | None = None

    async def receive() -> None:
        nonlocal heard_after
        while True:
            message = await handle.recv()
            # Only messages from other nodes arrive: zenoh doesn't loop our own back
            if isinstance(message, FromSwarm.Message) and message.topic == TOPIC:
                heard_after = heard_after or time.monotonic() - start

    # The networking handles requests only while something receives from it
    receiving = asyncio.create_task(receive())
    await handle.gossipsub_subscribe(TOPIC)
    while time.monotonic() - start < seconds:
        await handle.gossipsub_publish(TOPIC, b"hello")
        await asyncio.sleep(0.2)
    receiving.cancel()
    report = {
        "heard_after": heard_after,
        "other_versions": handle.devices_on_other_versions(),
    }
    print(json.dumps(report), flush=True)
    # Exit at once: a pending receive from the networking runtime doesn't cancel, and its
    # threads don't stop by themselves
    os._exit(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--wire-version", type=int, required=True)
    parser.add_argument(
        "--mlx-release", help="the MLX release it claims; none if left out"
    )
    parser.add_argument("--zenoh-port", type=int, required=True)
    parser.add_argument("--discovery-port", type=int, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    args = parser.parse_args()
    asyncio.run(
        run(
            cast(str, args.identity),
            cast(str, args.namespace),
            cast(int, args.wire_version),
            cast(str | None, args.mlx_release),
            cast(int, args.zenoh_port),
            cast(int, args.discovery_port),
            cast(float, args.seconds),
        )
    )


if __name__ == "__main__":
    main()
