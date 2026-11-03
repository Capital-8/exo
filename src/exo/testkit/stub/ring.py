"""The stub ranks' collective: a TCP ring between an instance's runners, as MLX's ring backend has.

It couples the ranks the way the real model does: each rank waits for the others at every
step, a stopped rank stops the others, and a lost rank shows up as a failed receive. The
addresses come from the instance's placement (`hosts_by_node`), the same ones MLX would use.

One deviation, because every node is on one machine: MLX gives every rank the same port, and
here each rank listens on that port plus its rank.
"""

import errno
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import cast, final

from exo.shared.types.common import NodeId
from exo.shared.types.worker.instances import BoundInstance, MlxRingInstance

_FRAME_HEADER = struct.Struct("!IH")
_CONNECT_RETRY_SECONDS = 0.2
# A runner answers a hello at once; one that doesn't is stale (stopped, or about to be)
_HANDSHAKE_TIMEOUT = 2.0
_READY = b"READY\n"


class RingError(Exception):
    """A receive or send to a neighbouring rank failed: the peer is gone or the link is cut."""

    def __init__(self, error_number: int, file_descriptor: int, detail: str) -> None:
        super().__init__(detail)
        self.error_number = error_number
        self.file_descriptor = file_descriptor


def rank_to_node(bound_instance: BoundInstance) -> dict[int, NodeId]:
    assignments = bound_instance.instance.shard_assignments
    return {
        assignments.runner_to_shard[runner_id].device_rank: node_id
        for node_id, runner_id in assignments.node_to_runner.items()
    }


@final
@dataclass
class Ring:
    rank: int
    size: int
    _send_socket: socket.socket
    _receive_socket: socket.socket
    _listener: socket.socket

    @classmethod
    def connect(cls, bound_instance: BoundInstance, proxy_port: int | None) -> "Ring":
        """Join the instance's ring, waiting as long as it takes for the neighbours (exo's
        supervisor stops a runner that takes too long to connect)."""
        instance = bound_instance.instance
        if not isinstance(instance, MlxRingInstance):
            raise NotImplementedError("the stub engine only runs ring instances")
        shard = bound_instance.bound_shard
        rank, size = shard.device_rank, shard.world_size
        right, left = (rank + 1) % size, (rank - 1) % size
        hosts = instance.hosts_by_node[bound_instance.bound_node_id]
        nodes = rank_to_node(bound_instance)
        instance_id = str(instance.instance_id)

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("0.0.0.0", instance.ephemeral_port + rank))
        listener.listen(4)

        accepted: list[socket.socket] = []
        acceptor = threading.Thread(
            target=_accept,
            args=(listener, f"{instance_id} {left}", accepted),
            name="stub-ring-accept",
            daemon=True,
        )
        acceptor.start()
        send_socket = _dial(
            host=hosts[right].ip,
            port=instance.ephemeral_port + right,
            proxy_port=proxy_port,
            source=bound_instance.bound_node_id,
            destination=nodes[right],
            hello=f"{instance_id} {rank}",
        )
        acceptor.join()
        (receive_socket,) = accepted
        for each in (send_socket, receive_socket):
            each.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return cls(rank, size, send_socket, receive_socket, listener)

    def all_gather(self, payload: bytes) -> list[bytes]:
        """Every rank's payload, indexed by rank."""
        gathered: list[bytes | None] = [None] * self.size
        gathered[self.rank] = payload
        current_rank, current = self.rank, payload
        for _ in range(self.size - 1):
            self._send(current_rank, current)
            current_rank, current = self._receive()
            gathered[current_rank] = current
        return [item if item is not None else b"" for item in gathered]

    def barrier(self) -> None:
        self.all_gather(b"")

    def close(self) -> None:
        for each in (self._send_socket, self._receive_socket, self._listener):
            each.close()

    def _send(self, origin_rank: int, payload: bytes) -> None:
        try:
            self._send_socket.sendall(
                _FRAME_HEADER.pack(len(payload), origin_rank) + payload
            )
        except OSError as error:
            raise RingError(
                error.errno or errno.EPIPE,
                self._send_socket.fileno(),
                f"send to rank {(self.rank + 1) % self.size} failed: {error}",
            ) from error

    def _receive(self) -> tuple[int, bytes]:
        header = self._receive_exactly(_FRAME_HEADER.size)
        length, origin_rank = cast(tuple[int, int], _FRAME_HEADER.unpack(header))
        return origin_rank, self._receive_exactly(length)

    def _receive_exactly(self, length: int) -> bytes:
        received = bytearray()
        while len(received) < length:
            try:
                chunk = self._receive_socket.recv(length - len(received))
            except OSError as error:
                raise RingError(
                    error.errno or errno.ECONNRESET,
                    self._receive_socket.fileno(),
                    f"receive failed: {error}",
                ) from error
            if not chunk:
                # MLX reports a peer that went away as a failed receive
                raise RingError(
                    errno.ECONNRESET,
                    self._receive_socket.fileno(),
                    "the previous rank closed the connection",
                )
            received += chunk
        return bytes(received)


def _accept(
    listener: socket.socket, expected_hello: str, accepted: list[socket.socket]
) -> None:
    """Accept the previous rank's connection, and say so, so that it knows it reached this
    runner and not a stale one (an earlier runner of this instance, still alive, still
    listening on the port)."""
    while True:
        connection = listener.accept()[0]
        connection.settimeout(_HANDSHAKE_TIMEOUT)
        try:
            hello = _read_line(connection)
            if hello == expected_hello:
                connection.sendall(_READY)
                connection.settimeout(None)
                accepted.append(connection)
                return
        except OSError:
            pass
        connection.close()


def _read_line(connection: socket.socket) -> str:
    line = bytearray()
    while not line.endswith(b"\n"):
        chunk = connection.recv(1)
        if not chunk:
            break
        line += chunk
    return line.decode(errors="replace").strip()


def _dial(
    *,
    host: str,
    port: int,
    proxy_port: int | None,
    source: NodeId,
    destination: NodeId,
    hello: str,
) -> socket.socket:
    """Connect to the next rank, retrying until that rank's runner answers the hello."""
    while True:
        connection: socket.socket | None = None
        try:
            if proxy_port is None:
                connection = socket.create_connection((host, port))
            else:
                connection = socket.create_connection(("::1", proxy_port))
                connection.sendall(
                    f"RING {source} {destination} {host} {port}\n".encode()
                )
                if _read_line(connection) != "OK":
                    raise ConnectionError("the harness's proxy couldn't connect")
            connection.settimeout(_HANDSHAKE_TIMEOUT)
            connection.sendall(f"{hello}\n".encode())
            if _read_line(connection).encode() + b"\n" == _READY:
                connection.settimeout(None)
                return connection
        except OSError:
            pass
        if connection is not None:
            connection.close()
        time.sleep(_CONNECT_RETRY_SECONDS)
