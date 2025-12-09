"""
Cost tracking for LLM API usage.

Tracks token usage and calculates costs for different providers
based on their pricing models.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class UsageRecord:
    """
    Record of a single API usage event.

    Attributes:
        timestamp: When the request was made
        model: Model name used
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        input_cost: Cost for input tokens
        output_cost: Cost for output tokens
        total_cost: Total cost for this request
    """

    timestamp: float
    model: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def datetime(self) -> datetime:
        """Get datetime from timestamp."""
        return datetime.fromtimestamp(self.timestamp)

    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return self.input_tokens + self.output_tokens


class CostTracker:
    """
    Track API usage costs across requests.

    Maintains history of API calls with token usage and costs.
    Supports multiple models with different pricing.

    Example:
        >>> tracker = CostTracker(
        ...     provider_name="openai",
        ...     pricing={
        ...         "gpt-4": {"input": 0.03, "output": 0.06},
        ...         "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
        ...     }
        ... )
        >>> tracker.track_request(
        ...     model="gpt-4",
        ...     input_tokens=100,
        ...     output_tokens=50
        ... )
        >>> print(tracker.get_total_cost())
        6.0  # $0.06 (in cents)
    """

    def __init__(
        self,
        provider_name: str,
        pricing: Dict[str, Dict[str, float]],
        currency: str = "USD",
    ):
        """
        Initialize cost tracker.

        Args:
            provider_name: Name of the provider
            pricing: Pricing per model (per 1K tokens)
                Format: {"model_name": {"input": price, "output": price}}
            currency: Currency code (default: "USD")
        """
        self.provider_name = provider_name
        self.pricing = pricing
        self.currency = currency
        self.usage_history: List[UsageRecord] = []

    def track_request(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        metadata: Optional[Dict[str, str]] = None,
    ) -> UsageRecord:
        """
        Track a single API request.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            metadata: Optional metadata about the request

        Returns:
            UsageRecord with calculated costs
        """
        # Get pricing for model (with fallback to default if available)
        model_pricing = self.pricing.get(model, self.pricing.get("default", {}))

        # Calculate costs (pricing is per 1K tokens)
        input_cost = (input_tokens / 1000.0) * model_pricing.get("input", 0.0)
        output_cost = (output_tokens / 1000.0) * model_pricing.get("output", 0.0)
        total_cost = input_cost + output_cost

        # Create record
        record = UsageRecord(
            timestamp=time.time(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            metadata=metadata or {},
        )

        self.usage_history.append(record)
        return record

    def get_total_cost(self, model: Optional[str] = None) -> float:
        """
        Get total cost across all requests.

        Args:
            model: Optional model filter (None = all models)

        Returns:
            Total cost in the configured currency
        """
        total = 0.0
        for record in self.usage_history:
            if model is None or record.model == model:
                total += record.total_cost
        return total

    def get_total_tokens(self, model: Optional[str] = None) -> Dict[str, int]:
        """
        Get total tokens used.

        Args:
            model: Optional model filter (None = all models)

        Returns:
            Dict with input, output, and total token counts
        """
        input_tokens = 0
        output_tokens = 0

        for record in self.usage_history:
            if model is None or record.model == model:
                input_tokens += record.input_tokens
                output_tokens += record.output_tokens

        return {
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens,
        }

    def get_request_count(self, model: Optional[str] = None) -> int:
        """
        Get number of requests made.

        Args:
            model: Optional model filter (None = all models)

        Returns:
            Number of requests
        """
        if model is None:
            return len(self.usage_history)
        return sum(1 for r in self.usage_history if r.model == model)

    def get_model_breakdown(self) -> Dict[str, Dict[str, float]]:
        """
        Get cost breakdown by model.

        Returns:
            Dict mapping model names to their usage statistics
        """
        breakdown: Dict[str, Dict[str, float]] = {}

        for record in self.usage_history:
            if record.model not in breakdown:
                breakdown[record.model] = {
                    "cost": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "requests": 0,
                }

            breakdown[record.model]["cost"] += record.total_cost
            breakdown[record.model]["input_tokens"] += record.input_tokens
            breakdown[record.model]["output_tokens"] += record.output_tokens
            breakdown[record.model]["requests"] += 1

        return breakdown

    def get_usage_report(self) -> Dict:
        """
        Get comprehensive usage report.

        Returns:
            Dict with complete usage statistics
        """
        return {
            "provider": self.provider_name,
            "currency": self.currency,
            "total_cost": self.get_total_cost(),
            "total_requests": self.get_request_count(),
            "total_tokens": self.get_total_tokens(),
            "model_breakdown": self.get_model_breakdown(),
            "first_request": (
                self.usage_history[0].datetime.isoformat() if self.usage_history else None
            ),
            "last_request": (
                self.usage_history[-1].datetime.isoformat() if self.usage_history else None
            ),
        }

    def clear_history(self) -> None:
        """Clear all usage history."""
        self.usage_history.clear()

    def export_history(self) -> List[Dict]:
        """
        Export usage history as list of dicts.

        Returns:
            List of usage records as dictionaries
        """
        return [
            {
                "timestamp": record.datetime.isoformat(),
                "model": record.model,
                "input_tokens": record.input_tokens,
                "output_tokens": record.output_tokens,
                "input_cost": record.input_cost,
                "output_cost": record.output_cost,
                "total_cost": record.total_cost,
                "metadata": record.metadata,
            }
            for record in self.usage_history
        ]

    def export_to_csv(self, filepath: str) -> None:
        """
        Export usage history to CSV file.

        Args:
            filepath: Path to output CSV file
        """
        import csv

        if not self.usage_history:
            return

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Timestamp",
                    "Model",
                    "Input Tokens",
                    "Output Tokens",
                    "Total Tokens",
                    "Input Cost",
                    "Output Cost",
                    "Total Cost",
                ]
            )

            for record in self.usage_history:
                writer.writerow(
                    [
                        record.datetime.isoformat(),
                        record.model,
                        record.input_tokens,
                        record.output_tokens,
                        record.total_tokens,
                        f"{record.input_cost:.6f}",
                        f"{record.output_cost:.6f}",
                        f"{record.total_cost:.6f}",
                    ]
                )

    def export_to_json(self, filepath: str) -> None:
        """
        Export usage history to JSON file.

        Args:
            filepath: Path to output JSON file
        """
        import json

        data = {
            "provider": self.provider_name,
            "currency": self.currency,
            "export_date": datetime.now().isoformat(),
            "summary": self.get_usage_report(),
            "history": self.export_history(),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_time_range_report(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Dict:
        """
        Get usage report for a specific time range.

        Args:
            start_time: Start timestamp (inclusive)
            end_time: End timestamp (inclusive)

        Returns:
            Usage report for the time range
        """
        filtered_records = self.usage_history

        if start_time:
            filtered_records = [r for r in filtered_records if r.timestamp >= start_time]

        if end_time:
            filtered_records = [r for r in filtered_records if r.timestamp <= end_time]

        if not filtered_records:
            return {
                "total_cost": 0.0,
                "total_tokens": 0,
                "total_requests": 0,
                "model_breakdown": {},
            }

        total_cost = sum(r.total_cost for r in filtered_records)
        total_tokens = sum(r.total_tokens for r in filtered_records)

        # Model breakdown
        breakdown = {}
        for record in filtered_records:
            if record.model not in breakdown:
                breakdown[record.model] = {
                    "cost": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "requests": 0,
                }

            breakdown[record.model]["cost"] += record.total_cost
            breakdown[record.model]["input_tokens"] += record.input_tokens
            breakdown[record.model]["output_tokens"] += record.output_tokens
            breakdown[record.model]["requests"] += 1

        return {
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "total_requests": len(filtered_records),
            "model_breakdown": breakdown,
            "start_time": (filtered_records[0].datetime.isoformat() if filtered_records else None),
            "end_time": (filtered_records[-1].datetime.isoformat() if filtered_records else None),
        }

    def check_budget(self, budget_limit: float) -> Dict:
        """
        Check if spending is within budget.

        Args:
            budget_limit: Budget limit in currency units

        Returns:
            Budget status information
        """
        total_cost = self.get_total_cost()
        remaining = budget_limit - total_cost
        percent_used = (total_cost / budget_limit * 100) if budget_limit > 0 else 0

        return {
            "budget_limit": budget_limit,
            "total_spent": total_cost,
            "remaining": remaining,
            "percent_used": percent_used,
            "within_budget": total_cost <= budget_limit,
            "near_limit": percent_used >= 90,  # Warning at 90%
        }


# Common pricing configurations (prices in USD per 1K tokens)
PRICING_OPENAI = {
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
}

PRICING_CLAUDE = {
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "claude-2.1": {"input": 0.008, "output": 0.024},
    "claude-2": {"input": 0.008, "output": 0.024},
}

# Ollama is typically free (local inference)
PRICING_OLLAMA = {
    "default": {"input": 0.0, "output": 0.0},
}
