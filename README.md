# Objective-Driven Conversation Orchestration

A framework for building structured conversational flows where routing, state, and side effects are handled deterministically — not by the model.

**The model speaks. The orchestrator decides. The graph governs.**

---

## Goals

LLM agents that do real structured work — multi-phase intake, booking, onboarding, triage — fail in predictable ways. They fail not because the model is bad at conversation, but because routing, state management, and side effect execution are the wrong jobs for a language model.

- Agent-based routing is probabilistic — errors are silent and compound
- State lives in the context window: cannot be audited, replayed, or resumed
- Side effects have no safety boundary — a model calling a booking API directly has no idempotency guarantees

This framework addresses each at the structural level:

- Routing is a **pure function of truth state** evaluated by the orchestrator
- State is a **typed, persisted truth store** — mutated only through validated tool calls, fully auditable and resumable
- Side effects are **declared by the model, executed by the orchestrator** — with idempotency keys, retries, and checkpointing

These aren't conventions the model is asked to follow. They are architectural constraints it cannot violate.

---

## How It Works

The system runs three nested control loops, each handling a distinct level of concern:

**Inner — ReAct loop:** Handles model invocation, multi-step MCP tool calls, and validated objective and side effect declaration.

**Middle — Orchestrator loop:**
- Handles multi-phase conversation with deterministic routing
- Processes extracted data against typed objective schemas
- Manages multi-phase extraction from a single user input
- Propagates cascade invalidations through the objective dependency graph
- Ensures correct conversation phase after invalidation
- Hands off declared side effects to the intent execution pipeline
- Provides phase-specific error handling for data extraction and MCP tool failures

**Outer — Two interlocking DAGs define the conversation structure:**
- **Phase graph** — user-defined phases connected by JsonLogic routing rules, evaluated deterministically over truth state
- **Objective dependency graph** — declares which objectives depend on others, driving cascade invalidation across phases

See the [Overview](/docs/overview.md) for the full architectural picture, design decisions, and typical use cases.

---

## Documents

| Document | Audience | Purpose |
|---|---|---|
| [Overview](/docs/overview.md) | Everyone | Architecture, design decisions, typical use cases |
| [Orchestrator Dev Guide](/docs/dev_guide.md) | Platform engineers | State model, execution loops, LangGraph wiring, caching, retry |
| [Config & Phase Design Guide](/docs/config_phase_design_guide.md) | Application authors | Phase design, objectives, routing, prompt quality, debugging |
| [Config Schema](/schema/config_schema.py) | Application authors | Config schema (JSON schema) |

---

## Example

**Coming soon:**

[Travel booking](/examples/travel_booking/) — flight and hotel booking with live Amadeus MCP tools, conditional routing across leisure / business / group trip types, typed booking intents, and structured failure handling.

---

## Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — graph execution, state management, checkpointing
- [LangChain](https://github.com/langchain-ai/langchain) — LLM abstraction, tool calling, ReAct agent
- [Anthropic Claude](https://docs.anthropic.com) — for routine or real-time: `claude-haiku-4-5`; for complex tasks: `claude-sonnet-4-6`
- [Model Context Protocol](https://modelcontextprotocol.io) — external tool integration

---

## Status

Active development. Core orchestration loop, state model, caching, and MCP use are stable. See [Overview](docs/overview.md) for what this is designed to do — and what it deliberately is not.