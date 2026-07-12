"""Contract test: scene_classifier.classify() must return expected_scene for golden cases.

This test GUARDS the implicit dependency between golden_aftersale.py and
scene_classifier.py. If the classification logic changes, this test breaks.
"""

import asyncio

import pytest

from app.orchestrator.scene_classifier import SceneClassifier
from tests.golden_aftersale import DATASET


@pytest.fixture(autouse=True)
def _clear_scene_cache() -> None:
    SceneClassifier.clear_cache()


def _classify_sync(query) -> str:
    """Run async classifier and return the classified scene."""
    classifier = SceneClassifier()
    return asyncio.run(
        classifier.classify(
            dependencies={"customer_uid": query.user_id, "user_id": query.user_id},
            question=query.question,
            session_id=f"contract-{query.query_id}",
        )
    )


def test_all_golden_aftersale_classify_correctly() -> None:
    """Every golden test case must classify as 'aftersale' via real order data."""
    for query in DATASET:
        scene = _classify_sync(query)
        assert scene == "aftersale", (
            f"Scene mismatch for {query.query_id}: got '{scene}', "
            f"expected 'aftersale' (user_id={query.user_id})"
        )


def test_presale_user_classifies_presale() -> None:
    """User with no orders in DB → presale."""
    classifier = SceneClassifier()
    scene = asyncio.run(
        classifier.classify(
            dependencies={"customer_uid": "user_presale_1", "user_id": "user_presale_1"},
            question="iPhone 17 的参数",
            session_id="contract-presale",
        )
    )
    assert scene == "presale"


def test_insale_user_classifies_insale() -> None:
    """User with unsigned orders → insale (all_signed=False)."""
    classifier = SceneClassifier()
    scene = asyncio.run(
        classifier.classify(
            dependencies={"customer_uid": "user_i01", "user_id": "user_i01"},
            question="我的快递到哪了",
            session_id="contract-insale",
        )
    )
    assert scene == "insale"


def test_aftersale_user_classifies_aftersale() -> None:
    """User with signed orders → aftersale (all_signed=True)."""
    classifier = SceneClassifier()
    scene = asyncio.run(
        classifier.classify(
            dependencies={"customer_uid": "user_a01", "user_id": "user_a01"},
            question="iPhone 17 退货流程",
            session_id="contract-aftersale",
        )
    )
    assert scene == "aftersale"


def test_mixed_user_classifies_insale() -> None:
    """User with both signed and unsigned orders → insale (scene_hint=mixed_orders)."""
    classifier = SceneClassifier()
    scene = asyncio.run(
        classifier.classify(
            dependencies={"customer_uid": "user_m01", "user_id": "user_m01"},
            question="我的快递到哪了",
            session_id="contract-mixed",
        )
    )
    assert scene == "insale"
    assert classifier.last_scene_hint == "mixed_orders"


def test_mixed_scene_prompt_file_exists() -> None:
    """The mixed.md prompt must exist for clarification flow."""
    from pathlib import Path
    from app.config import get_settings
    settings = get_settings()
    p = Path(settings.prompt_dir) / "mixed.md"
    assert p.exists(), f"Missing prompt: {p}"
    content = p.read_text()
    assert "订单号" in content


def test_no_user_id_defaults_to_presale() -> None:
    """Without user_id, classifier returns presale (no order context)."""
    classifier = SceneClassifier()
    scene = asyncio.run(
        classifier.classify(
            dependencies={},
            question="随便问点什么",
            session_id="contract-anonymous",
        )
    )
    assert scene == "presale"