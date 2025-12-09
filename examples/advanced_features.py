"""
Advanced features example for bruno-llm.

Demonstrates:
- Response caching
- Context window management
- Stream aggregation
- Cost tracking
- Middleware usage
"""

import asyncio

from bruno_core.models import Message, MessageRole
from bruno_llm import LLMFactory
from bruno_llm.base import (
    ContextLimits,
    ContextWindowManager,
    ResponseCache,
    StreamAggregator,
    TruncationStrategy,
)


async def caching_example():
    """Response caching example."""
    print("=== Response Caching Example ===\n")

    provider = LLMFactory.create("ollama", {"model": "llama2"})

    if not await provider.check_connection():
        print("❌ Ollama not available\n")
        return

    # Create cache
    cache = ResponseCache(max_size=100, ttl=300)  # 5 minute TTL

    messages = [
        Message(role=MessageRole.USER, content="What is 2+2?"),
    ]

    # First request - cache miss
    print("First request (cache miss)...")
    response1 = await provider.generate(messages, temperature=0.0)
    cache.set(messages, response1, temperature=0.0)
    print(f"Response: {response1}\n")

    # Second request - cache hit
    print("Second request (cache hit)...")
    cached_response = cache.get(messages, temperature=0.0)
    if cached_response:
        print(f"Cached response: {cached_response}")
        print("✅ Retrieved from cache (no API call!)\n")

    # Show cache stats
    stats = cache.get_stats()
    print("Cache stats:")
    print(f"  - Size: {stats['size']}/{stats['max_size']}")
    print(f"  - Hits: {stats['hits']}")
    print(f"  - Misses: {stats['misses']}")
    print(f"  - Hit rate: {stats['hit_rate']:.1%}\n")

    await provider.close()


async def context_management_example():
    """Context window management example."""
    print("=== Context Window Management Example ===\n")

    provider = LLMFactory.create("ollama", {"model": "llama2"})

    if not await provider.check_connection():
        print("❌ Ollama not available\n")
        return

    # Create context manager with small limit
    context_manager = ContextWindowManager(
        model="llama2",
        limits=ContextLimits(max_tokens=100, max_output_tokens=20),
        strategy=TruncationStrategy.SMART,
    )

    # Create conversation with many messages
    messages = [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="Tell me about Python programming."),
        Message(
            role=MessageRole.ASSISTANT, content="Python is a high-level programming language..."
        ),
        Message(role=MessageRole.USER, content="What about its history?"),
        Message(role=MessageRole.ASSISTANT, content="Python was created by Guido van Rossum..."),
        Message(role=MessageRole.USER, content="What is a list comprehension?"),
    ]

    # Check if within limit
    print(f"Original message count: {len(messages)}")

    if not context_manager.check_limit(messages):
        print("⚠️  Context limit exceeded, truncating...\n")
        messages = context_manager.truncate(messages)
        print(f"Truncated to: {len(messages)} messages\n")

    # Show stats
    stats = context_manager.get_stats(messages)
    print("Context stats:")
    print(f"  - Input tokens: {stats['input_tokens']}")
    print(f"  - Available output tokens: {stats['available_output_tokens']}")
    print(f"  - Usage: {stats['usage_percent']:.1f}%\n")

    # Generate with truncated messages
    response = await provider.generate(messages, max_tokens=20)
    print(f"Response: {response}\n")

    await provider.close()


async def stream_aggregation_example():
    """Stream aggregation example."""
    print("=== Stream Aggregation Example ===\n")

    provider = LLMFactory.create("ollama", {"model": "llama2"})

    if not await provider.check_connection():
        print("❌ Ollama not available\n")
        return

    messages = [
        Message(role=MessageRole.USER, content="Count from 1 to 10."),
    ]

    # Word-by-word aggregation
    print("Word-by-word streaming:")
    aggregator = StreamAggregator(strategy="word")
    stream = provider.stream(messages, max_tokens=50)

    async for word in aggregator.aggregate(stream):
        print(f"  [{word.strip()}]", end=" ", flush=True)

    print("\n\n✅ Streaming complete!\n")

    await provider.close()


async def cost_tracking_example():
    """Cost tracking example."""
    print("=== Cost Tracking Example ===\n")

    try:
        provider = LLMFactory.create_from_env("openai")

        if not await provider.check_connection():
            print("❌ OpenAI not available\n")
            return

        messages = [
            Message(role=MessageRole.USER, content="Explain AI in one sentence."),
        ]

        # Make several requests
        print("Making 3 requests...\n")
        for i in range(3):
            await provider.generate(messages, max_tokens=30)
            print(f"  Request {i + 1} complete")

        print()

        # Show cost report
        report = provider.cost_tracker.get_usage_report()
        print("Cost Report:")
        print(f"  - Total requests: {report['total_requests']}")
        print(f"  - Total tokens: {report['total_tokens']}")
        print(f"  - Total cost: ${report['total_cost']:.6f}")
        print()

        # Model breakdown
        if report["model_breakdown"]:
            print("By model:")
            for model, stats in report["model_breakdown"].items():
                print(f"  - {model}:")
                print(f"    * Requests: {stats['requests']}")
                print(f"    * Cost: ${stats['cost']:.6f}")

        print()

        # Check budget
        budget_status = provider.cost_tracker.check_budget(budget_limit=1.0)
        print("Budget check (limit: $1.00):")
        print(f"  - Spent: ${budget_status['total_spent']:.6f}")
        print(f"  - Remaining: ${budget_status['remaining']:.6f}")
        print(f"  - Usage: {budget_status['percent_used']:.1f}%")
        print(f"  - Within budget: {'✅' if budget_status['within_budget'] else '❌'}\n")

        # Export to CSV
        provider.cost_tracker.export_to_csv("cost_report.csv")
        print("✅ Exported cost report to cost_report.csv\n")

        await provider.close()

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure OPENAI_API_KEY is set\n")


async def middleware_example():
    """Middleware usage example."""
    print("=== Middleware Example ===\n")

    print("(Middleware is integrated into providers)")
    print("You can create custom middleware by extending Middleware class")
    print("Built-in middleware: LoggingMiddleware, CachingMiddleware, ValidationMiddleware\n")

    # Example would require provider modifications
    print("See documentation for middleware integration examples\n")


async def main():
    """Run all advanced examples."""
    print("╔══════════════════════════════════════╗")
    print("║ bruno-llm Advanced Features Examples ║")
    print("╚══════════════════════════════════════╝\n")

    try:
        await caching_example()
        await context_management_example()
        await stream_aggregation_example()
        await cost_tracking_example()
        await middleware_example()
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
