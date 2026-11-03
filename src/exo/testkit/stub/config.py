"""How a node's stub runners behave on their way to serving, set by the harness for each node.

The harness writes the file named by EXO_TESTKIT_STUB_CONFIG; the stub reads it at the start
of every phase (connect, load, warm-up), so a test can change it between launches.
"""

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from exo.testkit.stub.directives import FaultKind

STUB_CONFIG_ENV = "EXO_TESTKIT_STUB_CONFIG"
# Where the stub's ring connections go through the harness, so the harness can cut them
RING_PROXY_ENV = "EXO_TESTKIT_RING_PROXY"

Phase = Literal["connect", "load", "warmup"]


class StubConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    connect_seconds: float = 0.0
    load_seconds: float = 0.0
    warmup_seconds: float = 0.0
    # A fault the runner meets in one phase, every time it gets there
    fault_phase: Phase | None = None
    fault: FaultKind = "none"


def config_path() -> Path | None:
    path = os.environ.get(STUB_CONFIG_ENV)
    return Path(path) if path else None


def read_config() -> StubConfig:
    path = config_path()
    if path is None:
        return StubConfig()
    try:
        return StubConfig.model_validate_json(path.read_bytes())
    except FileNotFoundError:
        return StubConfig()
    except ValidationError as error:
        raise ValueError(f"bad stub config in {path}: {error}") from error


def write_config(path: Path, config: StubConfig) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(config.model_dump_json())
    os.replace(temporary, path)


def ring_proxy_port() -> int | None:
    port = os.environ.get(RING_PROXY_ENV)
    return int(port) if port else None
