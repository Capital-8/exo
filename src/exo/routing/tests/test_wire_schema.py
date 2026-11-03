"""What nodes send each other may only change together with WIRE_VERSION (exo/routing/wire.py)."""

import pytest
from pydantic import BaseModel, Field, create_model

from exo.routing import topics
from exo.routing.wire import WIRE_VERSION
from exo.routing.wire_schema import (
    Snapshot,
    changes,
    network_topics,
    schema_of,
    wire_schema,
)
from exo.utils.pydantic_ext import FrozenModel, TaggedModel


def test_the_wire_schema_matches_its_snapshot() -> None:
    saved = Snapshot.load()
    current = wire_schema()
    if current != saved.schema:
        pytest.fail(
            "What exo nodes send each other changed:\n"
            + "".join(f"  {line}\n" for line in changes(saved.schema, current))
            + "Nodes on the old schema can't read the new one, so they must not connect. "
            f"Bump WIRE_VERSION in src/exo/routing/wire.py (now {WIRE_VERSION}), then run\n"
            "  uv run python -m exo.routing.wire_schema --write\n"
            "Devices then join only devices on the same version. If the bytes on the wire "
            "didn't change (a pydantic upgrade renders schemas differently, a model that is "
            "never sent by name was renamed), pass --keep-version instead of bumping, and "
            "say why in the PR.",
            pytrace=False,
        )
    assert saved.wire_version == WIRE_VERSION, (
        f"WIRE_VERSION is {WIRE_VERSION} but wire_schema.json was written for "
        f"{saved.wire_version}: run uv run python -m exo.routing.wire_schema --write"
    )


def test_every_topic_nodes_send_is_in_the_schema() -> None:
    assert set(network_topics()) == {
        topics.GLOBAL_EVENTS.topic,
        topics.LOCAL_EVENTS.topic,
        topics.STATE_SNAPSHOTS.topic,
        topics.COMMANDS.topic,
        topics.ELECTION_MESSAGES.topic,
        topics.DOWNLOAD_COMMANDS.topic,
    }
    # Connection messages come from this node's own networking, never from the network
    assert topics.CONNECTION_MESSAGES.topic not in network_topics()


def _model(
    name: str, base: type[BaseModel], doc: str = "", **fields: object
) -> type[BaseModel]:
    return create_model(  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
        name,
        __base__=base,
        __doc__=doc,
        **fields,  # pyright: ignore[reportArgumentType]
    )


def _message(doc: str, **fields: object) -> type[BaseModel]:
    return _model("Message", FrozenModel, doc, **fields)


def test_documentation_is_not_part_of_the_wire() -> None:
    before = _message("One thing.", count=(int, Field(description="how many")))
    after = _message("Another thing.", count=(int, Field(description="how much")))
    assert schema_of({"t": before}) == schema_of({"t": after})


def test_a_new_field_changes_the_wire_even_with_a_default() -> None:
    # Old nodes forbid fields they don't know, and every field is always sent
    before = _message("", count=(int, ...))
    after = _message("", count=(int, ...), label=(str, ""))
    assert schema_of({"t": before}) != schema_of({"t": after})


def test_a_new_constraint_changes_the_wire() -> None:
    before = _message("", count=(int, ...))
    after = _message("", count=(int, Field(ge=0)))
    assert schema_of({"t": before}) != schema_of({"t": after})


def test_fields_are_named_as_they_are_sent() -> None:
    message = _message("", origin_index=(int, ...), title=(str, ...))
    sent = message.model_validate({"origin_index": 1, "title": "x"}).model_dump_json()
    assert '"origin_index"' in sent
    properties = schema_of({"t": message})["$defs"]["Message"]["properties"]  # pyright: ignore[reportIndexIssue, reportUnknownVariableType]
    # A field called like a schema keyword is still a field
    assert set(properties) == {"origin_index", "title"}  # pyright: ignore[reportUnknownArgumentType]


def _event_holder(tag: str) -> type[BaseModel]:
    event = _model(tag, TaggedModel, value=(int, ...))
    return _message("", event=(event, ...))


def test_a_tagged_model_is_sent_under_its_class_name() -> None:
    holder = _event_holder("Ping")
    sent = holder.model_validate({"event": {"Ping": {"value": 1}}}).model_dump_json()
    assert sent == '{"event":{"Ping":{"value":1}}}'
    schema = schema_of({"t": holder})
    assert schema["$defs"]["Ping"]["required"] == ["Ping"]  # pyright: ignore[reportIndexIssue]
    # So renaming it changes the wire
    assert schema_of({"t": _event_holder("Pong")}) != schema


def test_changes_name_what_differs() -> None:
    before = schema_of({"t": _message("", count=(int, ...))})
    after = schema_of({"t": _message("", count=(str, ...))})
    assert changes(before, after) == ["changed Message"]
    assert changes(before, before) == []
