"""The harness's own helpers, without starting nodes."""

from pathlib import Path

import pytest

from exo.testkit.cluster import ChatResult, Cluster, HarnessError
from exo.testkit.stub.directives import expected_reply


@pytest.mark.parametrize("falsy", [None, False, 0, "", [], {}])
def test_wait_for_keeps_waiting_while_the_condition_is_falsy(
    tmp_path: Path, falsy: object
) -> None:
    cluster = Cluster(1, tmp_path)
    answers: list[object] = [falsy, falsy, ["done"]]
    assert cluster.wait_for(lambda: answers.pop(0), timeout=5, what="it") == ["done"]
    with pytest.raises(HarnessError, match="waiting for nothing"):
        cluster.wait_for(lambda: falsy, timeout=0.3, what="nothing")


def _result(content: str, finish: str | None, error: str | None = None) -> ChatResult:
    return ChatResult(
        prompt="p [stub tokens=4]",
        via="n0",
        stream=True,
        max_tokens=None,
        started=0.0,
        content=content,
        finish_reason=finish,
        error=error,
    )


def test_a_reply_is_checked_token_by_token() -> None:
    tokens = expected_reply("p [stub tokens=4]").tokens
    assert _result("".join(tokens), "stop").problems() == []
    swapped = tokens[1] + tokens[0] + tokens[2] + tokens[3]
    assert "token 0" in _result(swapped, "stop").problems()[0]
    assert _result("".join(tokens[:3]), "stop").problems() != []
    # Cut short by an error: fine if it's an exact prefix, not if anything is out of place
    assert _result("".join(tokens[:2]), None, error="gone").problems() == []
    assert _result(tokens[1], None, error="gone").problems() != []
