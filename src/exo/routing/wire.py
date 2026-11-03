"""exo's wire version: nodes connect only when they run the same one.

Discovery hashes it into the namespace (``rust/networking/src/wire.rs``). Every message
model is strict (``extra="forbid"``), so nodes whose schemas differ can't read each other:
a node that meets an event it can't read stalls at that index until it restarts, and a node
can't read a state snapshot that has a field it doesn't know. The package version isn't
bumped for every build, so it can't tell builds apart. ``WIRE_VERSION`` can.

Bump it when what nodes send each other changes. ``wire_schema.py`` keeps a snapshot of
the schema, and a test fails when the code no longer matches it; its docstring says what to
do. A bump splits clusters: devices with the new version join only each other.
"""

from typing import Final, final

from exo.utils.pydantic_ext import FrozenModel

WIRE_VERSION: Final = 1


@final
class DeviceOnOtherVersion(FrozenModel):
    """A device discovery heard on the network that can't join: it runs another wire version."""

    address: str
    # None: an exo from before wire versions
    wire_version: int | None
