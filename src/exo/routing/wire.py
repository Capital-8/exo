"""What decides whether two exo nodes can talk: the wire version and the MLX release.

Nodes connect only when they run the same ``WIRE_VERSION`` and compatible MLX releases (see
``rust/networking/src/wire.rs``). Every message model is strict (``extra="forbid"``), so nodes
whose schemas differ can't read each other: a node that meets an event it can't read stalls
at that index until it restarts, and a node can't read a state snapshot that has a field it
doesn't know. The ranks of a model on different MLX releases can hang part way through it.
The package version isn't bumped for every build, so it can't tell builds apart.

Bump ``WIRE_VERSION`` when what nodes send each other changes. ``wire_schema.py`` keeps a
snapshot of the schema and a ledger of versions, and a test fails when the code no longer
matches them; its docstring says what to do. A bump splits clusters: devices with the new
version join only each other.
"""

import re
from importlib.metadata import PackageNotFoundError, version
from typing import Final, Literal, final

from exo.utils.pydantic_ext import FrozenModel

WIRE_VERSION: Final = 1

# The release part of a version: "0.32.4" of "0.32.4.dev20261030+36969c09"
_RELEASE = re.compile(r"\d+(?:\.\d+)*")


def mlx_release(installed_version: str | None = None) -> str | None:
    """The MLX release this node runs, such as "0.32.4", or None without MLX.

    Only the release counts: one exo build installs a development build of MLX on macOS
    ("0.32.4.dev20261030+36969c09") and the published release on Linux ("0.32.4"), and its
    Macs and Linux machines must still connect.
    """
    if installed_version is None:
        try:
            installed_version = version("mlx")
        except PackageNotFoundError:
            return None
    match = _RELEASE.match(installed_version)
    return match.group(0) if match else installed_version


@final
class DeviceOnOtherVersion(FrozenModel):
    """A device discovery heard under this cluster's name that can't join it."""

    # Every address it was heard on: one per network interface
    addresses: list[str]
    reason: Literal["older_exo", "wire_version", "mlx_release"]
    # Its wire version, when it runs another one
    wire_version: int | None
