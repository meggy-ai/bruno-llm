"""Tests for cost tracking."""

import pytest

from bruno_llm.base.cost_tracker import (
    PRICING_OLLAMA,
    PRICING_OPENAI,
    CostTracker,
    UsageRecord,
)


def test_usage_record():
    """Test UsageRecord creation."""
    import time

    record = UsageRecord(
        timestamp=time.time(),
        model="gpt-4",
        input_tokens=100,
        output_tokens=50,
        input_cost=3.0,
        output_cost=3.0,
        total_cost=6.0,
    )

    assert record.model == "gpt-4"
    assert record.total_tokens == 150
    assert record.datetime is not None


def test_cost_tracker_init():
    """Test CostTracker initialization."""
    tracker = CostTracker(
        provider_name="openai",
        pricing=PRICING_OPENAI,
    )

    assert tracker.provider_name == "openai"
    assert len(tracker.usage_history) == 0


def test_cost_tracker_track_request():
    """Test tracking a single request."""
    tracker = CostTracker(
        provider_name="openai",
        pricing=PRICING_OPENAI,
    )

    record = tracker.track_request(
        model="gpt-4",
        input_tokens=100,
        output_tokens=50,
    )

    assert record.model == "gpt-4"
    assert record.input_tokens == 100
    assert record.output_tokens == 50
    # GPT-4: $0.03 per 1K input, $0.06 per 1K output
    # 100 input = 0.1K * 0.03 = 0.003
    # 50 output = 0.05K * 0.06 = 0.003
    assert record.total_cost == pytest.approx(0.006, rel=1e-5)


def test_cost_tracker_get_total_cost():
    """Test getting total cost."""
    tracker = CostTracker(
        provider_name="openai",
        pricing=PRICING_OPENAI,
    )

    # Track multiple requests
    tracker.track_request(model="gpt-4", input_tokens=100, output_tokens=50)
    tracker.track_request(model="gpt-4", input_tokens=200, output_tokens=100)

    total = tracker.get_total_cost()
    # (100*0.03 + 50*0.06)/1000 + (200*0.03 + 100*0.06)/1000
    # = 0.006 + 0.012 = 0.018
    assert total == pytest.approx(0.018, rel=1e-5)


def test_cost_tracker_get_total_cost_by_model():
    """Test getting total cost filtered by model."""
    tracker = CostTracker(
        provider_name="openai",
        pricing=PRICING_OPENAI,
    )

    tracker.track_request(model="gpt-4", input_tokens=100, output_tokens=50)
    tracker.track_request(model="gpt-3.5-turbo", input_tokens=100, output_tokens=50)

    gpt4_cost = tracker.get_total_cost(model="gpt-4")
    gpt35_cost = tracker.get_total_cost(model="gpt-3.5-turbo")

    # GPT-4 is more expensive
    assert gpt4_cost > gpt35_cost


def test_cost_tracker_get_total_tokens():
    """Test getting total tokens."""
    tracker = CostTracker(
        provider_name="openai",
        pricing=PRICING_OPENAI,
    )

    tracker.track_request(model="gpt-4", input_tokens=100, output_tokens=50)
    tracker.track_request(model="gpt-4", input_tokens=200, output_tokens=100)

    tokens = tracker.get_total_tokens()

    assert tokens["input"] == 300
    assert tokens["output"] == 150
    assert tokens["total"] == 450


def test_cost_tracker_get_request_count():
    """Test getting request count."""
    tracker = CostTracker(
        provider_name="openai",
        pricing=PRICING_OPENAI,
    )

    tracker.track_request(model="gpt-4", input_tokens=100, output_tokens=50)
    tracker.track_request(model="gpt-4", input_tokens=200, output_tokens=100)
    tracker.track_request(model="gpt-3.5-turbo", input_tokens=100, output_tokens=50)

    assert tracker.get_request_count() == 3
    assert tracker.get_request_count(model="gpt-4") == 2
    assert tracker.get_request_count(model="gpt-3.5-turbo") == 1


def test_cost_tracker_get_model_breakdown():
    """Test getting cost breakdown by model."""
    tracker = CostTracker(
        provider_name="openai",
        pricing=PRICING_OPENAI,
    )

    tracker.track_request(model="gpt-4", input_tokens=100, output_tokens=50)
    tracker.track_request(model="gpt-4", input_tokens=200, output_tokens=100)
    tracker.track_request(model="gpt-3.5-turbo", input_tokens=100, output_tokens=50)

    breakdown = tracker.get_model_breakdown()

    assert "gpt-4" in breakdown
    assert "gpt-3.5-turbo" in breakdown
    assert breakdown["gpt-4"]["requests"] == 2
    assert breakdown["gpt-3.5-turbo"]["requests"] == 1


def test_cost_tracker_get_usage_report():
    """Test getting comprehensive usage report."""
    tracker = CostTracker(
        provider_name="openai",
        pricing=PRICING_OPENAI,
    )

    tracker.track_request(model="gpt-4", input_tokens=100, output_tokens=50)

    report = tracker.get_usage_report()

    assert report["provider"] == "openai"
    assert report["total_requests"] == 1
    assert "total_cost" in report
    assert "model_breakdown" in report
    assert report["first_request"] is not None


def test_cost_tracker_clear_history():
    """Test clearing usage history."""
    tracker = CostTracker(
        provider_name="openai",
        pricing=PRICING_OPENAI,
    )

    tracker.track_request(model="gpt-4", input_tokens=100, output_tokens=50)
    assert len(tracker.usage_history) == 1

    tracker.clear_history()
    assert len(tracker.usage_history) == 0


def test_cost_tracker_export_history():
    """Test exporting usage history."""
    tracker = CostTracker(
        provider_name="openai",
        pricing=PRICING_OPENAI,
    )

    tracker.track_request(
        model="gpt-4", input_tokens=100, output_tokens=50, metadata={"user": "test"}
    )

    history = tracker.export_history()

    assert len(history) == 1
    assert history[0]["model"] == "gpt-4"
    assert history[0]["metadata"]["user"] == "test"


def test_ollama_pricing():
    """Test Ollama has zero pricing."""
    tracker = CostTracker(
        provider_name="ollama",
        pricing=PRICING_OLLAMA,
    )

    record = tracker.track_request(
        model="llama2",
        input_tokens=1000,
        output_tokens=500,
    )

    assert record.total_cost == 0.0
