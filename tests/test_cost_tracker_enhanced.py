"""
Additional tests for enhanced cost tracking features (Phase 5).
"""

import json
import os
import tempfile
import time

import pytest

from bruno_llm.base.cost_tracker import CostTracker, PRICING_OPENAI


@pytest.fixture
def tracker():
    """Create a cost tracker for testing."""
    return CostTracker(
        provider_name="test_provider",
        pricing=PRICING_OPENAI,
    )


def test_export_to_csv(tracker, tmp_path):
    """Test exporting usage history to CSV."""
    # Track some requests
    tracker.track_request("gpt-4", input_tokens=100, output_tokens=50)
    tracker.track_request("gpt-3.5-turbo", input_tokens=200, output_tokens=100)
    
    # Export to CSV
    csv_file = tmp_path / "usage.csv"
    tracker.export_to_csv(str(csv_file))
    
    # Read and verify
    assert csv_file.exists()
    content = csv_file.read_text()
    
    assert "Timestamp" in content
    assert "Model" in content
    assert "Input Tokens" in content
    assert "gpt-4" in content
    assert "gpt-3.5-turbo" in content


def test_export_to_csv_empty(tracker, tmp_path):
    """Test exporting empty history to CSV."""
    csv_file = tmp_path / "empty.csv"
    tracker.export_to_csv(str(csv_file))
    
    # Should not create file for empty history
    assert not csv_file.exists()


def test_export_to_json(tracker, tmp_path):
    """Test exporting usage history to JSON."""
    # Track some requests
    tracker.track_request("gpt-4", input_tokens=100, output_tokens=50)
    tracker.track_request("gpt-3.5-turbo", input_tokens=200, output_tokens=100)
    
    # Export to JSON
    json_file = tmp_path / "usage.json"
    tracker.export_to_json(str(json_file))
    
    # Read and verify
    assert json_file.exists()
    with open(json_file) as f:
        data = json.load(f)
    
    assert data["provider"] == "test_provider"
    assert data["currency"] == "USD"
    assert "export_date" in data
    assert "summary" in data
    assert "history" in data
    assert len(data["history"]) == 2


def test_get_time_range_report(tracker):
    """Test getting usage report for specific time range."""
    start_time = time.time()
    
    # Track requests at different times
    tracker.track_request("gpt-4", input_tokens=100, output_tokens=50)
    time.sleep(0.1)
    mid_time = time.time()
    time.sleep(0.1)
    tracker.track_request("gpt-3.5-turbo", input_tokens=200, output_tokens=100)
    end_time = time.time()
    
    # Get report for full range
    report = tracker.get_time_range_report(start_time, end_time)
    assert report["total_requests"] == 2
    assert report["total_tokens"] == 450  # 100+50+200+100
    
    # Get report for first half
    report = tracker.get_time_range_report(start_time, mid_time)
    assert report["total_requests"] == 1
    
    # Get report for second half
    report = tracker.get_time_range_report(mid_time, end_time)
    assert report["total_requests"] == 1


def test_get_time_range_report_empty(tracker):
    """Test time range report with no data."""
    start_time = time.time()
    end_time = start_time + 100
    
    report = tracker.get_time_range_report(start_time, end_time)
    
    assert report["total_cost"] == 0.0
    assert report["total_tokens"] == 0
    assert report["total_requests"] == 0
    assert report["model_breakdown"] == {}


def test_get_time_range_report_partial(tracker):
    """Test time range report with partial filtering."""
    # Track 3 requests
    tracker.track_request("gpt-4", input_tokens=100, output_tokens=50)
    time.sleep(0.1)
    mid_time = time.time()
    time.sleep(0.1)
    tracker.track_request("gpt-4", input_tokens=100, output_tokens=50)
    time.sleep(0.1)
    tracker.track_request("gpt-4", input_tokens=100, output_tokens=50)
    
    # Only start time filter
    report = tracker.get_time_range_report(start_time=mid_time)
    assert report["total_requests"] == 2
    
    # Only end time filter
    report = tracker.get_time_range_report(end_time=mid_time)
    assert report["total_requests"] == 1


def test_check_budget_within_limit(tracker):
    """Test budget check when within limit."""
    # Track requests totaling ~$4.50
    tracker.track_request("gpt-4", input_tokens=100_000, output_tokens=50_000)
    
    status = tracker.check_budget(budget_limit=10.0)
    
    assert status["budget_limit"] == 10.0
    assert status["total_spent"] > 0
    assert status["remaining"] > 0
    assert status["within_budget"] is True
    assert status["near_limit"] is False


def test_check_budget_exceeded(tracker):
    """Test budget check when limit exceeded."""
    # Track expensive requests
    tracker.track_request("gpt-4", input_tokens=100_000, output_tokens=50_000)
    tracker.track_request("gpt-4", input_tokens=100_000, output_tokens=50_000)
    
    status = tracker.check_budget(budget_limit=5.0)
    
    assert status["within_budget"] is False
    assert status["remaining"] < 0


def test_check_budget_near_limit(tracker):
    """Test budget check when near limit (>90%)."""
    # Track requests to get close to limit
    tracker.track_request("gpt-4", input_tokens=100_000, output_tokens=50_000)
    
    # Set budget just above current spending
    current_cost = tracker.get_total_cost()
    budget = current_cost * 1.05  # 95% usage
    
    status = tracker.check_budget(budget_limit=budget)
    
    assert status["within_budget"] is True
    assert status["near_limit"] is True  # Should warn at >90%
    assert status["percent_used"] > 90


def test_check_budget_zero_limit(tracker):
    """Test budget check with zero limit."""
    status = tracker.check_budget(budget_limit=0)
    
    assert status["within_budget"] is True  # No spending yet
    assert status["percent_used"] == 0


def test_check_budget_with_spending(tracker):
    """Test budget check calculates percentages correctly."""
    tracker.track_request("gpt-4", input_tokens=10_000, output_tokens=5_000)
    
    cost = tracker.get_total_cost()
    status = tracker.check_budget(budget_limit=cost * 2)
    
    # Should be at 50% of budget
    assert abs(status["percent_used"] - 50.0) < 1.0


def test_time_range_model_breakdown(tracker):
    """Test model breakdown in time range report."""
    # Track different models
    tracker.track_request("gpt-4", input_tokens=100, output_tokens=50)
    tracker.track_request("gpt-3.5-turbo", input_tokens=200, output_tokens=100)
    tracker.track_request("gpt-4", input_tokens=150, output_tokens=75)
    
    report = tracker.get_time_range_report()
    
    breakdown = report["model_breakdown"]
    assert "gpt-4" in breakdown
    assert "gpt-3.5-turbo" in breakdown
    
    assert breakdown["gpt-4"]["requests"] == 2
    assert breakdown["gpt-4"]["input_tokens"] == 250
    assert breakdown["gpt-4"]["output_tokens"] == 125
    
    assert breakdown["gpt-3.5-turbo"]["requests"] == 1
    assert breakdown["gpt-3.5-turbo"]["input_tokens"] == 200


def test_export_json_structure(tracker, tmp_path):
    """Test JSON export has correct structure."""
    tracker.track_request("gpt-4", input_tokens=100, output_tokens=50)
    
    json_file = tmp_path / "test.json"
    tracker.export_to_json(str(json_file))
    
    with open(json_file) as f:
        data = json.load(f)
    
    # Verify structure
    assert "provider" in data
    assert "currency" in data
    assert "export_date" in data
    assert "summary" in data
    assert "history" in data
    
    # Verify summary structure
    summary = data["summary"]
    assert "provider" in summary
    assert "total_cost" in summary
    assert "total_requests" in summary
    assert "model_breakdown" in summary
    
    # Verify history structure
    assert len(data["history"]) == 1
    record = data["history"][0]
    assert "timestamp" in record
    assert "model" in record
    assert "input_tokens" in record
    assert "output_tokens" in record
    assert "total_cost" in record


def test_csv_export_formatting(tracker, tmp_path):
    """Test CSV export has proper formatting."""
    tracker.track_request("gpt-4", input_tokens=1000, output_tokens=500)
    
    csv_file = tmp_path / "test.csv"
    tracker.export_to_csv(str(csv_file))
    
    content = csv_file.read_text()
    lines = content.strip().split('\n')
    
    # Should have header + 1 data row
    assert len(lines) == 2
    
    # Check header
    header = lines[0]
    assert all(col in header for col in [
        "Timestamp", "Model", "Input Tokens", "Output Tokens",
        "Total Tokens", "Input Cost", "Output Cost", "Total Cost"
    ])
    
    # Check data row has all fields
    data_row = lines[1]
    fields = data_row.split(',')
    assert len(fields) == 8
