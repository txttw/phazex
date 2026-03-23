# PhaseX

A framework for building reliable, stateful conversational flows with structured information extraction using typed objectives, dependency graphs, and configurable routing rules. Designed for production workflows where correctness, auditability, real-world side effects are first-class design goals.

**The model speaks. The orchestrator manages. The graph governs.**

## Next steps

Read the [PhaseX Overview](/docs/overview.md) for a general understanding of typical use cases, core concepts and a high level description of how the system works.

For detailed understanding read the [PhaseX Architecture & Developer Guide](/docs/dev_guide.md). 

##  How to use

This is an **open design**, feel free to use the [PhaseX Architecture & Developer Guide](/docs/dev_guide.md) to implement your version. 

If you are interested to try and use my implementation, visit this page for status updates and [get in touch](https://txttw.online) for more information.


## Recommended Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — graph execution, state management, checkpointing
- [LangChain](https://github.com/langchain-ai/langchain) — LLM abstraction, tool calling, ReAct agent
- [Anthropic Claude](https://docs.anthropic.com) — for low-medium complexity: `claude-haiku`; for complex scenarios: `claude-sonnet`
- [OpenAI](https://openai.com/) — alternative LLM **\***
- [Model Context Protocol](https://modelcontextprotocol.io) — external tool integration

*\* LangChain supports a wide range of LLMs, it is an easy switch, but the orchestrator's caching behavior might need to adjust for cost efficiency and low latency.*

## Status

Core loops, state model, caching and side effect sungraph are implemented and are under alpha-testing. 