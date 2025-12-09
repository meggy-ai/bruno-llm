"""
Basic usage example for bruno-llm.

Demonstrates:
- Creating providers with the factory
- Simple generation
- Streaming responses
- Error handling
"""

import asyncio

from bruno_core.models import Message, MessageRole
from bruno_llm import LLMFactory


async def basic_ollama_example():
    """Basic Ollama usage example."""
    print("=== Basic Ollama Example ===\n")
    
    # Create provider using factory
    provider = LLMFactory.create(
        provider="ollama",
        config={"model": "llama2", "base_url": "http://localhost:11434"}
    )
    
    # Check connection
    if not await provider.check_connection():
        print("❌ Cannot connect to Ollama. Make sure it's running!")
        return
    
    print("✅ Connected to Ollama\n")
    
    # Create messages
    messages = [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="What is Python? Answer in one sentence."),
    ]
    
    # Generate response
    print("Generating response...")
    response = await provider.generate(messages, max_tokens=50)
    print(f"Response: {response}\n")
    
    # Close provider
    await provider.close()
    print("✅ Done!")


async def streaming_example():
    """Streaming response example."""
    print("\n=== Streaming Example ===\n")
    
    provider = LLMFactory.create("ollama", {"model": "llama2"})
    
    if not await provider.check_connection():
        print("❌ Cannot connect to Ollama")
        return
    
    messages = [
        Message(role=MessageRole.USER, content="Count from 1 to 5."),
    ]
    
    print("Streaming response: ", end="", flush=True)
    
    async for chunk in provider.stream(messages, max_tokens=50):
        print(chunk, end="", flush=True)
    
    print("\n\n✅ Streaming complete!")
    await provider.close()


async def openai_example():
    """OpenAI usage example (requires API key)."""
    print("\n=== OpenAI Example ===\n")
    
    try:
        # Create from environment (requires OPENAI_API_KEY)
        provider = LLMFactory.create_from_env("openai")
        
        # Check connection
        if not await provider.check_connection():
            print("❌ Cannot connect to OpenAI")
            return
        
        print("✅ Connected to OpenAI\n")
        
        messages = [
            Message(role=MessageRole.USER, content="What is 2+2? Be concise."),
        ]
        
        response = await provider.generate(messages, max_tokens=20)
        print(f"Response: {response}\n")
        
        # Check cost
        cost = provider.cost_tracker.get_total_cost()
        print(f"💰 Cost: ${cost:.6f}\n")
        
        await provider.close()
        print("✅ Done!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure OPENAI_API_KEY environment variable is set")


async def fallback_example():
    """Provider fallback example."""
    print("\n=== Fallback Example ===\n")
    
    print("Trying OpenAI first, then Ollama as fallback...\n")
    
    try:
        provider = await LLMFactory.create_with_fallback(
            providers=["openai", "ollama"],
            configs=[
                {"api_key": "invalid"},  # Will fail
                {"base_url": "http://localhost:11434", "model": "llama2"},
            ]
        )
        
        print(f"✅ Using provider: {provider.get_model_info()['provider']}\n")
        
        messages = [Message(role=MessageRole.USER, content="Hello!")]
        response = await provider.generate(messages, max_tokens=20)
        print(f"Response: {response}\n")
        
        await provider.close()
        
    except Exception as e:
        print(f"❌ All providers failed: {e}")


async def main():
    """Run all examples."""
    print("╔══════════════════════════════════════╗")
    print("║  bruno-llm Basic Usage Examples     ║")
    print("╚══════════════════════════════════════╝\n")
    
    try:
        await basic_ollama_example()
        await streaming_example()
        await openai_example()
        await fallback_example()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
