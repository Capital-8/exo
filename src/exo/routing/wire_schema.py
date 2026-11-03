"""The schema of every message exo nodes send each other, and its checked-in snapshot.

``wire_schema.json`` is the schema at ``WIRE_VERSION`` (``wire.py``).
``tests/test_wire_schema.py`` fails when the schema generated from the code differs from it.
When it does:

1. bump ``WIRE_VERSION`` in ``src/exo/routing/wire.py``;
2. run ``uv run python -m exo.routing.wire_schema --write``.

Write the snapshot with ``--keep-version`` only when the generated schema changed but the
bytes nodes send did not (a pydantic upgrade renders schemas differently, or a model that is
never sent by name was renamed), and say why in the PR.

Not in the snapshot, so bump by hand when they change: discovery's packets (a test in
``rust/networking/src/discovery.rs`` pins them), the zenoh key expressions in
``rust/networking/src/swarm.rs``, and ``GET /node_id``, which nodes call on each other to
check reachability.
"""

import argparse
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


@final
@dataclass(frozen=True)
class Snapshot:
    """A wire schema and the wire version it belongs to, as saved in ``wire_schema.json``."""

    wire_version: int
    schema: Mapping[str, object]

    @classmethod
    def load(cls, path: Path = SNAPSHOT_PATH) -> "Snapshot":
        document = cast(dict[str, object], json.loads(path.read_text()))
        version = document.pop("wire_version")
        assert isinstance(version, int), f"{path.name}: wire_version isn't a number"
        return cls(wire_version=version, schema=document)

    def dumps(self) -> str:
        document = {"wire_version": self.wire_version, **self.schema}
        return json.dumps(document, indent=2, sort_keys=True) + "\n"


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
        description="Check or write the snapshot of exo's wire schema.",
    )
    parser.add_argument(
        "--write", action="store_true", help=f"write {SNAPSHOT_PATH.name}"
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

    current = Snapshot(wire_version=WIRE_VERSION, schema=wire_schema())
    saved = Snapshot.load() if SNAPSHOT_PATH.exists() else None
    if saved == current:
        print(f"{SNAPSHOT_PATH.name} is up to date (wire version {WIRE_VERSION}).")
        return 0
    if saved is not None:
        for line in changes(saved.schema, current.schema):
            print(f"  {line}")
    if not write:
        print(f"{SNAPSHOT_PATH.name} is out of date; see this module's docstring.")
        return 1
    if (
        saved is not None
        and saved.schema != current.schema
        and saved.wire_version == WIRE_VERSION
        and not keep_version
    ):
        print(
            f"The wire schema changed but WIRE_VERSION is still {WIRE_VERSION}. Bump it "
            "in src/exo/routing/wire.py first, or pass --keep-version if the bytes nodes "
            "send didn't change.",
            file=sys.stderr,
        )
        return 1
    SNAPSHOT_PATH.write_text(current.dumps())
    print(f"Wrote {SNAPSHOT_PATH.name} for wire version {WIRE_VERSION}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
