"""The schema of every message exo nodes send each other, its snapshot, and its ledger.

``wire_schema.json`` is the schema at ``WIRE_VERSION`` (``wire.py``), and
``wire_versions.txt`` lists every wire version with a hash of its schema, one line each,
appended to and never rewritten. ``tests/test_wire_schema.py`` fails when the schema
generated from the code differs from the snapshot, or its hash from the ledger's last line.
When it does:

1. bump ``WIRE_VERSION`` in ``src/exo/routing/wire.py`` to the ledger's last version + 1;
2. run ``uv run python -m exo.routing.wire_schema --write``, which adds that version's line.

Two changes that each take the next version both add a line at the end of the ledger, so git
makes whichever merges second take a new one, instead of letting two different schemas share
a version.

Write the snapshot with ``--keep-version`` only when the generated schema changed but the
bytes nodes send did not (a pydantic upgrade renders schemas differently, or a model that is
never sent by name was renamed), and say why in the PR. It rewrites the hash on the ledger's
last line.

Not in the snapshot, so bump by hand when they change: discovery's packets (a test in
``rust/networking/src/discovery.rs`` pins them), the zenoh key expressions in
``rust/networking/src/swarm.rs``, and ``GET /node_id``, which nodes call on each other to
check reachability.
"""

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast, final, override

from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue, models_json_schema
from pydantic_core import core_schema

from exo.routing import topics
from exo.routing.topics import PublishPolicy, TypedTopic
from exo.routing.wire import WIRE_VERSION
from exo.utils.pydantic_ext import FrozenModel, TaggedModel

SNAPSHOT_PATH: Final = Path(__file__).with_name("wire_schema.json")
LEDGER_PATH: Final = Path(__file__).with_name("wire_versions.txt")
_LEDGER_HEADER: Final = """\
# exo's wire versions: one line each, the version, then a hash of its wire schema
# (src/exo/routing/wire_schema.py). Append only: two changes that each take the next version
# both add a line here, so git makes the second to merge take a new one.
"""

# What a schema says that doesn't change which messages a node accepts. Defaults don't
# matter because every field is always sent (TypedTopic.serialize).
_NOT_ON_THE_WIRE = frozenset({"title", "description", "default", "examples"})
# Schema keywords whose value maps names (of fields, or of definitions) to schemas
_NAMED_SCHEMAS = frozenset({"properties", "$defs", "patternProperties"})


@final
class _WireSchemaGenerator(GenerateJsonSchema):
    """The JSON schema of messages as they are on the wire: a tagged model is sent as
    ``{"ClassName": {...}}``, so its class name is part of what a receiver must know."""

    @override
    def model_schema(self, schema: core_schema.ModelSchema) -> JsonSchemaValue:
        inner = super().model_schema(schema)
        model = schema["cls"]
        if not issubclass(model, TaggedModel):
            return inner
        return {
            "type": "object",
            "properties": {model.__name__: inner},
            "required": [model.__name__],
            "additionalProperties": False,
        }


def network_topics() -> dict[str, type[BaseModel]]:
    """Every topic that nodes send each other, with its message model."""
    found: dict[str, type[BaseModel]] = {}
    for value in list[object](vars(topics).values()):
        if not isinstance(value, TypedTopic):
            continue
        topic = cast(TypedTopic[FrozenModel], value)
        if topic.publish_policy is not PublishPolicy.Never:
            found[topic.topic] = topic.model_type
    return found


def schema_of(models_by_topic: Mapping[str, type[BaseModel]]) -> dict[str, object]:
    """The wire schema of these topics: what each topic carries, and every model in it."""
    # By field name, as TypedTopic.serialize sends them (aliases are only accepted)
    roots, schema = models_json_schema(
        [(model, "serialization") for model in models_by_topic.values()],
        by_alias=False,
        schema_generator=_WireSchemaGenerator,
    )
    return {
        "topics": {
            topic: cast(str, roots[(model, "serialization")]["$ref"])
            for topic, model in sorted(models_by_topic.items())
        },
        "$defs": _without_annotations(cast(object, schema.get("$defs", {}))),
    }


def wire_schema() -> dict[str, object]:
    return schema_of(network_topics())


def _without_annotations(node: object, names: bool = False) -> object:
    """``node`` without what doesn't change which messages are accepted. ``names``: its
    keys are field or definition names, kept even when one is called "title"."""
    if isinstance(node, dict):
        return {
            key: _without_annotations(value, names=key in _NAMED_SCHEMAS and not names)
            for key, value in cast(dict[str, object], node).items()
            if names or key not in _NOT_ON_THE_WIRE
        }
    if isinstance(node, list):
        return [_without_annotations(item) for item in cast(list[object], node)]
    return node


def schema_hash(schema: Mapping[str, object]) -> str:
    """A short hash of a wire schema, as the ledger records it."""
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def load_snapshot(path: Path = SNAPSHOT_PATH) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text()))


def dump_snapshot(schema: Mapping[str, object]) -> str:
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


@final
@dataclass(frozen=True)
class LedgerEntry:
    wire_version: int
    schema_hash: str


def load_ledger(path: Path = LEDGER_PATH) -> list[LedgerEntry]:
    entries: list[LedgerEntry] = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        version, hash_ = line.split()
        entries.append(LedgerEntry(int(version), hash_))
    return entries


def dump_ledger(entries: list[LedgerEntry]) -> str:
    return _LEDGER_HEADER + "".join(
        f"{entry.wire_version} {entry.schema_hash}\n" for entry in entries
    )


def changes(old: Mapping[str, object], new: Mapping[str, object]) -> list[str]:
    """What differs between two wire schemas, one line per topic or model."""
    lines: list[str] = []
    for section in ("topics", "$defs"):
        before = cast(Mapping[str, object], old.get(section, {}))
        after = cast(Mapping[str, object], new.get(section, {}))
        for name in sorted(set(before) | set(after)):
            if name not in after:
                lines.append(f"removed {name}")
            elif name not in before:
                lines.append(f"added {name}")
            elif before[name] != after[name]:
                lines.append(f"changed {name}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m exo.routing.wire_schema",
        description="Check or write the snapshot and ledger of exo's wire schema.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help=f"write {SNAPSHOT_PATH.name}, and the new version's line in {LEDGER_PATH.name}",
    )
    parser.add_argument(
        "--keep-version",
        action="store_true",
        help="write a changed schema without a new WIRE_VERSION (only when the bytes "
        "nodes send are the same)",
    )
    args = parser.parse_args(argv)
    write = cast(bool, args.write)
    keep_version = cast(bool, args.keep_version)

    current = wire_schema()
    current_hash = schema_hash(current)
    ledger = load_ledger(LEDGER_PATH)
    last = ledger[-1]
    saved = load_snapshot(SNAPSHOT_PATH) if SNAPSHOT_PATH.exists() else None
    if (
        saved == current
        and last.wire_version == WIRE_VERSION
        and last.schema_hash == current_hash
    ):
        print(f"{SNAPSHOT_PATH.name} is up to date (wire version {WIRE_VERSION}).")
        return 0
    if saved is not None:
        for line in changes(saved, current):
            print(f"  {line}")
    if not write:
        print(f"The wire schema or its version changed; see {__name__}'s docstring.")
        return 1

    if last.wire_version + 1 == WIRE_VERSION:
        ledger.append(LedgerEntry(WIRE_VERSION, current_hash))
    elif last.wire_version == WIRE_VERSION and (
        current_hash == last.schema_hash or keep_version
    ):
        ledger[-1] = LedgerEntry(WIRE_VERSION, current_hash)
    else:
        print(
            f"WIRE_VERSION is {WIRE_VERSION} and the ledger's last version is "
            f"{last.wire_version}. Set WIRE_VERSION to {last.wire_version + 1} in "
            "src/exo/routing/wire.py, or pass --keep-version if the bytes nodes send "
            "didn't change.",
            file=sys.stderr,
        )
        return 1
    SNAPSHOT_PATH.write_text(dump_snapshot(current))
    LEDGER_PATH.write_text(dump_ledger(ledger))
    print(
        f"Wrote {SNAPSHOT_PATH.name} and {LEDGER_PATH.name} for wire version {WIRE_VERSION}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
