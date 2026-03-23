# PhaseX: Overview

A framework for building reliable, stateful conversational flows with structured information extraction, — using typed objectives, validators, dependency graphs — configurable conversation paths with routing rules and safe side-effect execution. Designed for production workflows where correctness, auditability, real-world side effects are first-class design goals.

## The Problem

Building conversational AI that does real work — not just chatting, but reliably extracting structured information, routing based on what users say, and triggering real-world actions like bookings or payments — is harder than it looks.

The obvious approach is to let the model do everything: ask questions, decide what to do next, call APIs directly. This breaks in production for several reasons:

**Routing is non-deterministic.** An LLM deciding which phase of a conversation to enter next will occasionally make the wrong call. These errors are silent, hard to reproduce, and compound over turns.

**State is invisible.** A model that tracks what it knows in its context window cannot be audited, replayed, or resumed. If something goes wrong, there is no ground truth to inspect.

**Side effects are unsafe.** A model that calls a booking API directly has no idempotency guarantees. On retry it books twice. On crash mid-turn the result is unknown.

**Trust boundaries are wrong.** Letting the model decide what is true — not just what the user said — conflates conversation with data integrity.

These problems appear across many domains wherever structured data collection, real-world actions, or complex routing are involved.


## Typical Use Cases

This platform addresses three categories of conversational flow, each with a different primary goal but the same underlying reliability requirements.

**Where reliability and auditability is a compliance requirement, not a nice-to-have:**

Every extracted fact, routing decision, and side effect must be replayable and inspectable. The trust boundary problem is acute: e.g. a model deciding what counts as a valid KYC answer is not acceptable. The orchestrator enforces this — the model extracts, the orchestrator (potentially calling the application) validates, the tool commits or rejects.

### Consultative Conversations

Sales, support, coaching, customer success and advisory flows where the goal is not just collecting information but revealing real user needs and addressing them with tailored recommendations.

The model uses MCP tools to access knowledge bases, product catalogues, case studies, ratings, and real-time availability — becoming a domain expert. It asks open questions, interprets intent, and narrows toward a recommendation or action through structured conversation.

The objective model and phase structure ground the model minimizing hallucination and irrelevant questions.

**Why reliability matters here:** The model's judgment about what the user needs influences real actions — routing to a sales motion, scheduling a demo, initiating a return. These must be deterministic functions of extracted facts and configured thresholds, not unconstrained model decisions.

**Typical application:**
- consultative product search and sales
- conversational booking
- loan applications
- insurance enrollment
- account opening
- benefits selection
- auditable customer service with routing to appropriate function (e.g. billing, tech support, etc-)
- IT service desk triage
- sales discovery and qualification
- product configuration wizards
- healthcare patient intake
- employee onboarding

### Conversational structured data collection

Any structured data collection that currently lives in a multi-step form — onboarding, applications, intake, configuration wizards — can be replaced with a conversational flow that is more helpful and equally reliable.

**Where forms fail:**

- Fields with complex dependencies require custom conditional logic that accumulates into unmaintainable trees.
- Users don't understand what's being asked. Technical labels and context-free questions cause confusion.
- Validation errors are unhelpful, domain specific or too complex. Users abandon rather than debug.
- Progress is fragile. Navigating back to change an answer can invalidate downstream fields silently or force a restart.

**How are these challenges approached:**

Objective dependencies are declared in configuration and enforced structurally — a field is only asked when its prerequisites are satisfied, and changing an upstream answer automatically re-collects what is now stale. Instead of a terse error, the model explains what is needed and asks a targeted follow-up. MCP tools can query knowledge bases to answer "what does this mean?" in real time, and query systems (Structured database, CRM, analytics, preferences etc.) to fetch available options or pre-populate known values. Changing a previous answer triggers cascade invalidation — unaffected answers stay committed, only stale ones are re-collected.

Example:

> *"I'm not sure what counts as 'primary residence'. Does a flat I own but rent out count?"*  
> *"No, primary residence means where you currently live. Do you own or rent your current home?"*

**Typical application:**
- contact forms
- loan applications
- insurance enrollment
- account opening
- benefits selection
- equipment provisioning
- compliance questionnaires
- product configuration wizards
- and more, where currently a form collects data


### Scenario-Based Training

The model takes on a role: a sceptical prospect, a job interviewer, a patient describing symptoms, a difficult customer. The trainee responds as they would in the real situation. Objectives are assessments rather than extractions — the model evaluates whether the trainee demonstrated a required competency (acknowledged the objection before responding, used the correct tense, asked a clarifying question before pitching) and commits the result through the same resolve tool. Truth state accumulates a structured record of what has been demonstrated. Routing advances the scenario only when the trainee has satisfied the phase's requirements.

Routing is the most acute of the four problems here. Advancing before the trainee has demonstrated the required competency produces a worse training outcome and false confidence. Deterministic routing over assessed truth state enforces progression the same way it enforces data collection in other domains — the model assesses, the orchestrator decides whether to advance.

State continuity matters for extended programmes. A trainee working through preparation across several sessions. Resumes from the correct phase with prior assessments committed — no re-demonstration of already-assessed competencies, no loss of progress. MCP can supply domain material in real time: product knowledge bases during sales practice, vocabulary and grammar references during language training, job descriptions and company information during interview prep, clinical protocols during healthcare scenario work. A side effect at completion can log results to an LMS or generate a performance summary.

The model's assessments are committed as structured facts rather than floating in a context window. An instructor reviewing a session sees exactly what was assessed, in which phase, and when — auditable and replayable in the same way a compliance-critical intake flow is.

**Training applies to multiple domains:**
- sales training
- clinical skills practice
- customer service coaching
- language learning
- interview preparation
- improving conversational skills
- anywhere a trainee needs to practice a high-stakes interaction before facing it for real


## The Core Idea

Use what the model can do reliably and add scaffolding where it can fail to prevent incremental state corruption.

**The model is good at:** natural language, reading user intent, asking the right follow-up question, handling ambiguity, bridging topics conversationally.

**The model needs scaffolding:** structured state management, enforcing data constraints, deterministic routing decisions, executing side effects, safe retry-ability.

The system enforces this split structurally. The model converses and extracts. The orchestrator decides and executes. Neither can do the other's job — not by convention, but by architecture.


## How It Works

### Phases and Objectives

A conversation is decomposed into **phases** — logical stages with a defined purpose (collect travel intent, gather booking details, confirm payment). Each phase has a set of **objectives** — typed, structured pieces of information to extract from the user (destination, travel dates, budget).

The model's job within a phase is to extract those objectives through natural conversation. It uses dedicated tools to commit extractions. It cannot route to the next phase, cannot mutate state directly, and cannot execute side effects.

### Deterministic Routing

When all objectives in a phase are resolved, the orchestrator evaluates **transition rules** — configurable conditions evaluated against the current truth state. Rules determine which phase comes next. The model is never involved in this decision.

Routing is a pure function of state. It is deterministic, auditable, and fully replayable from any checkpoint.

### Truth State

All extracted information lives in **truth state** — an object store of resolved objectives. It is the authoritative record of what is currently believed to be true.

Truth state is mutated only through explicit tool calls. It supports **invalidation** — if a user changes a previously resolved value, the objective is marked unresolved and its dependents are recursively invalidated. The orchestrator re-walks the phase graph from the entry point using current truth state to determine where to resume, advancing as far as the remaining state allows.

Dependencies between objectives are defined in configuration, not code. Cross-phase dependencies are supported.

### Side Effects

Real-world actions (booking, payment, cancellation) are never executed by the model directly. The orchestrator wraps side-effect bearing MCP tools in SafeTool. The tool usage invokes a subgraph providing semantic and API level idempotency, overwrite safety, execution history and checkpointing with safe retry-ability. Subgraph executes synchronously and returns the outcome as a tool response in the same turn. The model reads the result and responds naturally.

### State Injection

The model sees state through structured JSON blocks prepended to messages — not by scanning conversation history. Completed phases appear in the system prompt. Current-phase extractions appear in the message history. The model never needs to re-derive what is already known.

### Caching

This design enables prompt caching: the tool list and general prompt cache per phase are shared across all users in that phase within the same organisation — under load, every user hitting the same phase resets the TTL and pays cache-read price. The phase prompt caches per phase transition. Conversation history caches as a rolling per-user window. Latency is low within a phase; phase transitions cause a single cache miss.

## Three Nested Loops

The architecture is built from three nested control loops. These separate concerns cleanly and are complementary. The ReAct loop handles tool orchestration and validation without knowing anything about phases or routing. The orchestrator loop handles multi-phase progression and invalidation without knowing anything about conversation content. The phase graph defines the structure of the domain without knowing anything about execution mechanics. Together they produce a system that is conversationally natural, structurally reliable, and straightforwardly debuggable — each loop has a clear owner, clear failure modes, and clear invariants.

**Loop 1 — ReAct (innermost): tool use and validation**

Within a single model call, LangChain's ReAct loop handles multi-step MCP tool use. The model calls a search tool, reasons over the results, calls another tool if needed, then produces a final response. Objective tools — `resolve_*`, `invalidate` and `SafeTool` — participate in the same loop. The orchestrator enforces schema and validation on each tool call. Objective tools that fail validation return immediately so the model can correct and retry within the same invocation. For `SafeTool` tools, the Orchestrator manages execution, it is opaque to the model. 

This loop is entirely invisible to the user. It runs inside a single turn, within a single phase.


**Loop 2 — Turn loop: multi-phase extraction from a single user turn**

The turn loop runs across phases within a single user turn:

```
                       ┌──────────────────────────────────────────────────────────────────────┐
                       ▼                                                                      │
PreparePhase → ExtractAndGenerate → ApplyTool → Route ── [ same phase ] ──► RespondAndInput ──┘
     ▲                                  │
     │                             [ new phase ]
     │                                  │
     └──────────────────────────────────┘
```

A single user message may satisfy objectives in one phase and immediately trigger a transition to the next. The orchestrator detects this, prepares the new phase context, and runs extraction again — all before returning a response to the user. The user always sees exactly one response, always from the correct final phase.

This loop also makes invalidation reliable. When a user changes a previously resolved value, the orchestrator re-walks the phase graph from the entry point over current truth state, advancing as far as it can before stopping to collect what is now stale. Each phase in the re-walk sees a focused context — its own system prompt, only its own objectives, and only the MCP tools it needs. **This scope reduction is a primary reason that vast conversation scenarios with complex routing can be handled with precision and even cheaper models remain viable.** The model never sees all the possible conversation paths at once — only the focused context of the current phase.


**Loop 3 — Phase graph (outermost): the user-defined conversation structure**

The phase graph is user-defined. It governs the overall shape of the conversation across multiple user turns. It is not a loop in the traditional sense (it is a DAG) but it has safe looping behaviour built into by the orchestrator:

- **Stay in phase** — a phase with unsatisfied required objectives does not route forward. The model stays, asks again differently, escalates, or eventually calls `resolve(fail)`.
- **Invalidation re-walk** — changing a resolved value causes the orchestrator to re-walk from ENTRY, which may revisit earlier phases to re-collect stale information.
- **Fail handler** — a phase with a configured `fallback_phase_id` can route to a recovery phase on failure instead of halting, enabling remediation flows without back-edges in the graph.

The graph itself contains no cycles — these are rejected at config time. Apparent "looping" is entirely an emergent property of the orchestrator and invalidation mechanisms, not of the graph structure.


## What This Looks Like in Practice

A user interacts with what feels like a single coherent conversation. Behind the scenes:

- Each turn is handled by the current phase's LLM node
- Extractions are committed to truth state through typed tools
- When a phase completes, the orchestrator evaluates rules and advances — the model introduces the new phase naturally if needed
- If the user changes something already resolved, the system invalidates, re-routes, and re-collects only what is now stale — the rest stays committed
- If the user requests a real-world action, the model declares intent, the orchestrator executes safely, and the result comes back in the same turn

The user experiences a natural conversation. The system maintains a reliable, auditable record of the conversation.


## Key Properties

| Property | How it is achieved |
|---|---|
| Routing correctness | Deterministic rules over truth state — no model involvement |
| State auditability | Explicit tool-only mutations; fully replayable from checkpoint |
| Side effect safety | Safe execution with idempotency guarantee, overwrite protection, and execution history; result returned synchronously |
| Invalidation correctness | Dependency graph propagation; re-walk from ENTRY over current truth state |
| Latency | Prompt caching at three breakpoints |
| Retry safety | All nodes retriable, safe state mutation, checkpointing prevents re-execution |
| Conversational quality | Model owns all natural language; orchestrator never speaks |

This is not a general-purpose agent framework. The model does not plan, does not decide what to do next, and does not self-direct. Every path through the conversation is a function of pre-defined phases, objectives, dependencies, and routing rules configured by the author.

Flexibility comes from the configuration — the graph of phases and rules, the dependency structure of objectives, the routing conditions. When model judgement is required it is explicitly expressed through validated, auditable objectives. 

The platform enforces the invariants; authors define the behaviour.

## Where to Go Next

- **[Developer Guide](/docs/dev_guide.md)** — architecture, state model, execution loop, caching, MCP integration, retry model, graph authoring