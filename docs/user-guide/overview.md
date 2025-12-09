# User Guide Overview

Complete guide to using bruno-llm with bruno-core framework.

## What is bruno-llm?

bruno-llm provides production-ready LLM provider implementations that integrate with the bruno-core framework. It enables you to:

- Use multiple LLM providers with a unified interface
- Switch between local and cloud models
- Leverage advanced features like caching, streaming, and cost tracking

## Providers

- **[Ollama](providers/ollama.md)** - Local LLM inference
- **[OpenAI](providers/openai.md)** - Cloud GPT models

## Advanced Features

- **[Response Caching](advanced/caching.md)** - Cache LLM responses to reduce costs
- **[Context Management](advanced/context.md)** - Handle context window limits
- **[Streaming](advanced/streaming.md)** - Stream tokens as they're generated
- **[Cost Tracking](advanced/cost-tracking.md)** - Track API costs per request
- **[Middleware](advanced/middleware.md)** - Add logging, validation, and custom processing

## See Also

- [API Reference](../api/factory.md)
- [Examples](https://github.com/meggy-ai/bruno-llm/tree/main/examples)
