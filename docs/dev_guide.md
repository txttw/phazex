# PhazeX: Architecture & Developer Guide

This document defines the runtime architecture, core components, and developer-facing responsibilities for building, extending, and operating the conversational orchestrator using LangGraph and LangChain.

It is intended for:
- Orchestrator / platform developers
- Infra & reliability engineers
- Advanced conversation designers integrating custom tools

This document does not contain LLM-facing content. It complements the Config and Phase Design Guide and the General System Prompt.


## 1. Architectural Overview

The system is composed of three strictly separated layers:

```
┌───────────────────────────────────────────────┐
│                 Conversation UI               │
│        (Web, Voice, Streaming, TTS)           │
└───────────────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────┐
│              Phase Orchestrator               │
│  (LangGraph runtime + Orchestrator logic)     │
└───────────────────────────────────────────────┘
        │                         │
        ▼                         ▼
┌─────────────────────┐   ┌────────────────────┐
│      LLM Runtime    │   │  External Services │
│ (Conversation and   │   │ (MCP, APIs, Tools) │             
│    system tools)    │   └────────────────────┘
└─────────────────────┘   
```

**Core design principle**
The LLM converses and declares facts. The orchestrator decides and executes.
No routing, no state mutation, and no side effects are entrusted to the model.

**What "the orchestrator decides" means:**

The orchestrator decides when the answer is structured. It is derivable from typed, validated truth state having known shape at config time — destination, budget, traveler_count, request_type (billing/tech support) etc.. Routing on them is correct because the config author has full information about the structure, therefore rules can be formulated.

The orchestrator cannot decide when the answer requires understanding the semantics of the user's situation — i.e. "Are all the bookings the user needed done?" That depends on what the user needed, which isn't known until runtime. Encoding it as an orchestrator rule would require the config author to anticipate every possible combination, which is impossible.

Objectives (See Section 3) can be used to resolve these situations This is the same mechanism as any other extraction, but what's being extracted is a model judgment, not a user fact.
e.g. `user_finished_booking` or `loyalty_program_offered` are explicit semantic gates — the model must make a deliberate judgment and commit it through the validated tool call. 

### 1.1 Three Nested Control Loops

The Orchestrator is governed by three nested loops with non-overlapping responsibilities. Understanding their boundaries is the most important prerequisite for working on the orchestrator.

**1.1.1 Loop 1 — ReAct (innermost): tool use and validation**

Scope: Model invocation with tools, inside a conversation phase.

LangChain's ReAct loop drives multi-step tool use within a turn. The model calls a tool, receives results, reasons, calls another if needed, then produces prose. Objective tools (resolve_\*, invalidate) and MCP tools (See Section 7) run in the same loop. The orchestrator validates every objective tool call at the boundary — a rejected call returns `status: error` immediately so the model can correct and retry within the same invocation.

For side-effects (SafeTools — see Section 8), the orchestrator invokes a subgraph synchronously — schema validation, idempotency check, MCP execution, and result serialisation all happen within tool call boundary. The model receives the outcome as ToolResponse and responds in the same turn.

The ReAct loop has no knowledge of phases, routing, or state persistence. Its goal: given a context and a tool set, produce a response. Schema is always validated and custom objective validators (See in Section 3.5) will execute when available — they run at the same tool boundary, before a value is committed to truth state.

**Loop 2 — Turn loop (middle): multi-phase extraction within a turn**

Scope: a single user turn, across one or more phases.

```
                       ┌──────────────────────────────────────────────────────────────────────┐
                       ▼                                                                      │
PreparePhase → ExtractAndGenerate → ApplyTool → Route ── [ same phase ] ──► RespondAndInput ──┘
     ▲                                  │
     │                             [ new phase ]
     │                                  │
     └──────────────────────────────────┘
```

A single user message can satisfy objectives in the current phase and immediately trigger transition to the next. The orchestrator detects completion at Route, prepares the new phase context, and runs another Extract → Apply → Rpute cycle — all before returning to the user. The user sees exactly one response, always from the correct terminal phase of that turn.

This loop also owns invalidation. When ApplyTool detects an invalidated objective, the router re-walks the graph from ENTRY over current truth state, advancing as far as it can before stopping where re-collection is needed. 

Each ExtractAndGenerate step runs in the focused context of one phase — its own prompt, objectives, and tools only. **This scope reduction is a primary reason that vast conversation scenarios with complex routing can be handled with precision and even cheaper models remain viable.** The model never sees all the possible conversation paths at once — only the focused context of the current phase.

**Loop 3 — Phase graph (outermost): conversation structure across turns**

Scope: the full conversation, across multiple user turns.

The phase graph is a user-defined DAG evaluated by Route node. It is acyclic by construction — routing cycles are rejected at config compile time. Apparent looping behaviour emerges from three mechanisms that require no back-edges:

- **Stay in phase** — a phase with unsatisfied required objectives never routes forward. The model re-engages, escalates, or calls `resolve(fail)`.
- **Invalidation re-walk** — changing a resolved value triggers Loop 2's re-walk from ENTRY, which re-enters earlier phases without any back-edge in the graph.
- **Fail handler with `fallback_phase_id`** — a phase can route to a recovery phase on failure, forming a diamond rather than a cycle.

Routing is a pure function of truth state evaluated by the orchestrator. No model involvement. No implicit fallthrough.

**Responsibility summary:**

| Concern | Loop | Owner |
|---|---|---|
| Prose generation | ReAct | LLM |
| Objective schema validation | ReAct | Tool boundary (orchestrator) |
| SafeTool validation and execution | ReAct | wrapper tool + subgraph |
| Multi-step MCP tool use | ReAct | LangChain agent |
| Multi-phase extraction in one turn | Turn loop | Extract + Apply + Route |
| Invalidation and re-walk | Turn loop | Apply + Route |
| Conversation structure and routing | Phase graph | Config + Route |


**SafeTools** usually represent side effects. They execute in their separate subgraph with their own idempotency and retry guarantees (See Section 8).


## 2. State Model

The orchestrator maintains multiple independent but coordinated state layers.

**Separation guarantees:**
- Only validated objective tools change truth state with commit once guarantee (extraction).
- Only explicit invalidataion (validated invalidation tool) can erase a resolved/extracted objective.  
- Truth invalidation cascade and may cause phase re-routing.
- SafeTools are executed by a subgraph providing idempotency and retry safety.

### 2.1 Truth State (Semantic Data)

Represents what is currently believed to be true.

Characteristics:
- Built from resolved objectives. e.g. travel_dates, destination, travelers, etc.
- Phase membership is defined in objective config, not stored in truth state
- Mutation triggered only via explicit tool calls (resolve tools, invalidation tool)
- Mutated by an update object and custom reducer
- Fully replayable and auditable 
- Subject to invalidation propagation via the dependency graph (config-defined)

Example (internal storage format — includes status/metadata/value envelope):

```json
{
  "destination": { 
    "status": "success", 
	"phase": {
	  "phase_id": "travel_data",
	  "phase_execution_id": "f73f6e43-4ffa-4e4f-9973-8b05f963f96b"
	},
	"turn_id": 3,
	"value": { "city": "Rome", "country": "Italy" } 
  }
}
```

Note: the injection block (`StateBlock`) strips the envelope — the model sees only the value fields (See Section 6.1).

### 2.2 Phase State

Represents where we are in the conversation.

Characteristics:
- Tracks current phase
- Phases can be re-entered natutrally
- phase_execution_id is a unique ID
- Checkpointing keeps phase histroy, no need to include it in the state

Example:
```json
{
  "current_phase": {
    "phase_id": "travel_data",
    "phase_execution_id": "f73f6e43-4ffa-4e4f-9973-8b05f963f96b"
  }
}
```

### 2.3 Message history

Human and AI conversation history.  

Characteristics:
- Stores prose messages
- Objective tools (resolve_\*, invalidate) are stripped
- SafeTool calls and ToolResponse stored for further reference
- MCP tools can be stored or stripped depending on operational needs
- All relevant state information is injected as JSON blocks by assembly_context. They are authoritative over message history.

### 2.4 SafeTool State

Tracks calls and results of all SafeTools 

Example:
```json
{
  "safe_tools": [
    {
      "tool_id": "reserve_room",  // configured
      "semantic_id": "book_room_1", // model generated
      "status": "failed",
      "idempotency_key": "<uuid_v4>", // orchestrator generates before execution
      "phase": {
	    "phase_id": "accomodation_booking",
	    "phase_execution_id": "6927d95b-83d6-4775-a0e8-9cf2c271b7e5"
	  },
	  "turn_id": 13,
      "params": { "offer_id": "offer_48545" }
      "result": {
        "reason": "Room type no longer available"
      } 
    },
    {
      "tool_id": "reserve_room",
      "semantic_id": "book_room_1", // model generated, same semantic_id as above if it is the same semantic intent (fail retry with new params)
      "status": "succeeded",
      "idempotency_key": "<uuid_v4>", // different than above, this is a new execution
      "phase": {
	    "phase_id": "accomodation_booking",
	    "phase_execution_id": "139b62f8-cb8d-479d-b4c5-c657753f1cbe"
	  },
	  "turn_id": 16,
	  "params": { "offer_id": "offer_17965" }
      "result": {
        "cancellation_id": "cxl_abc123"
      } 
    },
    {
      "tool_id": "reserve_room",
      "semantic_id": "book_room_2", // model generates new semantic_id, this is a 2nd reservation
      "status": "succeeded",
      "idempotency_key": "<uuid_v4>", // orchestrator generates before execution
      "phase": {
	    "phase_id": "accomodation_booking",
	    "phase_execution_id": "139b62f8-cb8d-479d-b4c5-c657753f1cbe"
	  },
	  "turn_id": 20,
	  "params": { "offer_id": "offer_1234" }
      "result": {
        "cancellation_id": "cxl_abc456"
      } 
    }
  ]
}
```

### 2.5 Current phase extractions

Stores what objectives were extracted in the current phase.

Used by assembly_context to reconstruct the injected state. 

Why this is needed?

State from previous phases arre injected to system prompt, but phase internal state injected to HumanMessage as state block.
The reason is to make caching effective. Thruth state change (objective resolution or state local invalidation) won't invalidate the whole cache.

## 3. Objectives and Lifecycle

### 3.1 Objective Definition

An objective is a **typed data contract or model judgement** the model **resolves** through conversation. It is the fundamental unit of truth state — what has been established, by whom, and when.

Objectives are **not** workflow steps.

Each objective has:

- **`id`** — stable identifier, used as the routing variable and truth state key
- **`description`** — what the value represents semantically; injected everywhere the objective is visible
- **`resultSchema`** — typed fields committed to truth state on `success`; the resolve tool's parameter signature
- **`required`** — `required`, `optional`, `passive`
- **`phase_id`** — owning phase; where the resolve tool is available and requirement tier is enforced
- **`available_in`** — where the resolve tool is available and requirement tier is not enforced (equivalent to passive)
- **`dependencies`** — objective IDs that this one depends on (e.g. accomodation depends on destination). Drives cascade invalidation
- **`extraction_instructions`** — optional; injected only in the owning phase alongside the resolve tool
- **`invalidation_instructions`** — optional; injected with the resolved value in every phase


**Requirement model:**

- **`required`** — the model must extract this objective. Phase cannot advance without `success` or `fail`. The model drives the conversation until it is resolved.

- **`optional`** — the model must ask, but can skip if the user rejects to answer (or ignores the question). Phase cannot advance without `success` or `skipped`. The model drives the conversation until it is resolved.

- **`passive`** — the model never asks proactively. If the user volunteers the information, the model extracts it. If unmentioned the orchestrator auto-skips it — no tool call required from the model. Use for preferences or constraints the model should capture opportunistically but never drive a question e.g. loyalty program numbers, special requests, accessibility needs.

**Owning phase vs. Available in (phase):** 

Required objectives are not available outside of their owning phase. The correct update mechanism is explicit invalidation then resolve in the owning phase (router re-walk and extraction loop will ensure owning state re-entrance. See Section 3.4 and Section 5 for details.)

Objectives are always available in their owning phase. Optional and passive objectives in `available_in` phases. For updates, invalidation is the correct mechanism regardles of requirement model (required / optional / passsive). The design intentionally makes invalidation explicit. 

**Usage notes for `available_in`:**
- Don't use for global invalidation → invalidation is always global, nothing needed
- Don't use for update → no such feature, model uses invalidation then resolves in owning or available in phase
- Don't use for phase-specific objectives → just don't set available_in
- Use when the objective is genuinely conversational in multiple phases → loyalty_number, dietary_restrictions, language_preference

### 3.2 Resolution Rules

- An objective resolves only via its tool
- Resolution is final unless explicitly invalidated
- Low-confidence values must not be extracted — the model must not emit a resolve tool call without high confidence

### 3.3 Optional and Passive Objective Completion

**Optional objectives** must be either resolved (`success`) or explicitly skipped (`skipped`) before routing evaluates phase eligibility. The model should make at least a light attempt, then call `resolve_*(status="skipped")` if the user does not engage. Silent abandonment is not permitted — the orchestrator cannot advance past an unresolved optional objective.

**Passive objectives** require no action from the model. If still unresolved when Router evaluates eligibility, the orchestrator treats them as auto-skipped. The model extracts them if the user volunteers the information; otherwise they contribute nothing to routing decisions.

This distinction eliminates the "N skip calls" problem that occurs when a phase has many soft-optional objectives the user has not engaged with. Passive objectives have zero routing effect when unanswered.

### 3.4 Invalidation Propagation

Invalidation operates at the **objective dependency level**, not the phase level. Dependencies may cross phase boundaries.

When an objective is invalidated:
1. The objective is removed from truth
2. All dependent objectives are recursively invalidated, regardless of which phase owns them
3. All same-turn resolve calls are applied
4. The phase graph is re-walked from ENTRY using current truth state to determine correct phase

**Phase re-walk after invalidation:**

```
procedure re_walk(current, truth_state):
  if any required objective of current is unresolved → return current
  if any optional objective of current is unresolved (not success or skipped) → return current
  // At this point all required and skippable objectives are extracted, we can not STOP here
  if current is Terminal → Natural end of conversation → Execute terminal action (if configured)
  evaluate routing rules of current phase against truth_state
  highest priority rule matches → advance current to matched_phase, return re_walk(matched_phase, truth_state)
  if no rule matches → error handler when all objectives are extracted but no phase matched
```
First call:

```
re_walk(ENTRY, truth_state)
```

It is O(N) over a deterministic computation against truth state alone — no history involved.

**Important:** Phase graph must be a DAG and has to be validated before execution, otherwise infinite recursion can crash the system. (See Section 10)  


**Why ENTRY and not the owning phase of the earliest invalidated objective:**

After complex same-turn invalidation and re-resolution, the current truth state may represent a phase never reached by the actual traversal. Path changes are rare but practically valid. Re-walking from ENTRY is the safe and deterministic approach. It discovers exactly how far the current truth state can advance from the start.

**Re-walk example (simplified):**

```
                               luxury_options 
                             ↗                ↘
Phases: welcome → basic_info                    booking
                             ↘                ↗
                               budget_only ──

User: "Changed my mind, I will go to Rome."
Model invalidates destination (resolved in basic_info) and resolves destination(Rome) in same turn.
Cascade: accommodation (depending on destination) also invalidated.

After invalidation and re-resolve, truth state:
  destination: Rome ✓   travel_dates: ✓   budget: ✓   hotel_chain: ✓
  accommodation: unresolved

Re-walk from ENTRY:
  welcome      → all objectives satisfied → routing rule (true) → advance to basic_info
  basic_info   → all objectives satisfied → routing rule (budget > 5000) → advance to luxury_options
  luxury_options → accommodation unresolved → STOP

Result: route to luxury_options. basic_info was never re-entered despite being the owning phase
of the invalidated objective — the same-turn resolution made it unnecessary.
```

**Re-entry behaviour:**

The target phase from re-walk receives a new `PhaseExecution` (new `phase_execution_id`, set `phase_id`, empty `phase_extractions`). Only unresolved objectives in the target phase are addressed. Objectives that survived the cascade remain committed and are skipped by the model via resolve-once semantics.

**Robustness under real-world conditions:**

The combination of deterministic, idempotent invalidation, truth state resolution, pure-function routing and internal extraction loop makes the system robust to common real-world conversation behaviors and even some model errors:

- **Redundant resolve an already resolved objective** — The orchestrator enforces a resolve once policy. Secondary resolves are silently dropped. This increases stability around update. Deliberate update requires a specific action, invalidation (makes objective unresolved), then the objective can be updated (resolved). 
- **Redundant invalidate on an already-absent value** — nothing to cascade, no-op effectively. Even if a re-walk is triggered on an already-correct truth state returns the same phase. One extra routing evaluation, negligible cost.
- **Invalidate + re-resolve in the same call** — the invalidate clears the old value, the resolve commits the new one. Cascade runs before resolve so a same-call re-resolve is not affected by its own invalidation.
- **User input contains objective resolutions from multiple phases** — Internal extraction loop can handle multi-phase resolutions from the same user input.
- **Model in extraction loop invalidates already correctly updated objective** — Invalidation validation prevents the model to make this pattern: invalidate → resolve → invalidate — in one turn. This is always a model error (mostly originates from unclear prompting) as no new user information was presented before the second invalidation.

These properties mean the routing outcome depends only on what ends up in truth state, not on how the model arrived there.

**Design guidance for authors:**
- Dependencies should reflect genuine data relationships, not conversational ordering
- Invalidation cascades with same-turn recovery can route to phases never previously visited — test these paths explicitly
- Routing conditions must be well-defined for all reachable truth state combinations, not just the happy path
- Double check complex branching where exists an exit path, otherwise mark the phase as terminal or add a catch all route (rule: true) and route to a specific phase. 

### 3.5 Objective Validators

The orchestrator enforces schema validation on every `resolve_*` tool call — the value must match the objective's `resultSchema` type constraints. Objective validators extend this with domain-specific validation that the schema cannot express: checking that a city code exists in the airline's route network, that a date range is within policy limits, that a loyalty number passes a checksum.

Validators run synchronously at the tool call boundary, before the value is committed to truth state. A rejected call returns an error reason to the model within the same ReAct loop invocation. The model reads the reason and corrects — either by asking the user or by re-calling with a valid value. The user sees only the model's natural response, not the validation logic.

Params passed to the validator: 
- objective fields under validation
- ProjectedTruthState (snapshot projected based on current ReAct loop)

#### 3.5.1 Projected Truth State

Before each validation, the orchestrator computes a read-only snapshot.

ProjectedTruthState build order mirrors ApplyTools: the base is the committed TruthState, then invalidations with cascade first, then resolutions.

ProjectedTruthState is **not stored** and **not incremental**, it is recomputed fresh before each validation call as a pure function. Not authoritative  as the ReAct loop has not finished. 

#### 3.5.2 Consistency, not order

Enforcing resolution order in the conversation can make it mechanical. The model converses freely and commits values through `resolve_*` when it has sufficient context to pass validation. Resolution is a commitment, not a transcription step. The model may hear a value early in the conversation, hold it, and resolve it later once prerequisites are in place. Validation provides consistency on commitment while preserving conversational freedom.

A failing validator returns a targeted message to the model. The model should read the message and determine what to resolve next — not ask the user again for information already in the conversation.


#### 3.5.3 The invalidate-once in a turn rule is not affected

Validators operate on ProjectedTruthState, which is built from committed truth plus in-flight accepted changes. The invalidate-once rule operates on committed truth. There is no interaction between the two.


#### 3.5.4 Running validators — JsonLogic

The configuration way to define validators is JsonLogic.

Each objective may declare an ordered array of JsonLogic rules in objective config. All must pass; first failure short-circuits and returns that rule's message.

```json
{
  "id": "destination",
  "validators": [
    {
      "rule": {"!=": [{"var": "travel_type.type"}, null]},
      "message": "Resolve travel_type before destination."
    },
    {
      "rule": {
        "?:": [
          {"==": [{"var": "travel_type.type"}, "business"]},
          {"in": [{"var": "destination.city_code"}, {"var": "const.business_codes"}]},
          true
        ]
      },
      "message": "city_code not allowed for business travel. Use the MCP tool to query available cities."
    }
  ]
}
```

#### 3.5.5 Library distribution — Callable

When the orchestrator is distributed as a library, validators can be Python Callable-s registered at startup:

```python
def validate_destination(
    fields: dict,
    projected: ProjectedTruthState
) -> ValidationResult:
    travel_type = projected.get("travel_type")
    if travel_type is None:
        raise ObjectiveValidationError(reason="Resolve travel_type before destination.")
    if travel_type.get("type") == "business":
        city_code = fields.get("city_code")
        if city_code not in BUSINESS_CODES:
            raise ObjectiveValidationError(
                reason=f"city_code {city_code} not allowed for business travel."
            )
    return fields
```

The validator receives the resolved field dict and returns the validated data. Raising a `ObjectiveValidationError` exception is treated as failure with the provided reason. Validators should catch their own exceptions and return structured reasons so the model has actionable feedback.


#### 3.5.6 Service distribution — MCP tool

When the orchestrator is distributed as a service, validators are MCP tools named `validate_<objective_id>` available in the application's MCP server. The orchestrator calls them internally after schema validation passes and before committing to truth state. The orchestrator will call the tool before resolution.

```json
{
  "mcp_config": {
    "connections": {
      "validation_server": {
        "transport": "http",
        "url": "https://mcp.example.com",
        "headers": {
          "Authorization": "Bearer {VALIDATION_API_KEY}"
        }
      }
    }
  }
}
```

The MCP tool receives the same data as the Callable would:

```json
{
  "fields": { "city_code": "IBZ", "country": "Spain" },
  "projected_truth_state": {
    "travel_type": { "type": "business" }
  }
}
```
The orchestrator expects:

`{ "valid": true }` or `{ "valid": false, "reason": "..." }`.

**Precedence:** If a Callable or MCP tool is registered, the config array is ignored.


Any other response shape or tool error is treated as failed validation with a generic reason.

#### 3.5.7 What validators are not for

Validators check that a value is correct given current state snapshot. 
They do not:
- Replace objective dependencies. They are required for invalidation.
- Replace phase design and routing to enforce business rules.
- Replace the model's conversational judgment — if a value is plausible but the model should ask a clarifying question, that belongs in the phase prompt, not a validator.
- Perform side effects — validators may read external state (DB, API) but must not write it. Writing external state in a validator creates a coupling between extraction and execution that belongs in a SafeTool.

## 4. Phase Graph Model

### 4.1 Phases as Nodes (overview)

Each phase is a logical unit implemented at runtime using the following LangGraph nodes.

**PreparePhase Node:** Prepares phase execution

**ExtractAndGenerate Node**: Runs extraction and prose generation in a ReAct looop

**ApplyTools Node**: Safely applies extraction results

**Router Node**: Evaluates routing rules and handles termination and error routing.
   
**RespondAndInput Node**: Responds to the user and waits for new input

Note: See Section 5.2 for all nodes and their detailed behavior

### 4.2 Phase Transition

When Router determines a transition, it advances immediately to the new phase. Phase transition does not automatically trigger a new user turn.
This is useful because the system can resolve objectives from multiple phases from one user input.

If a phase transition occurs during a turn, the prose generated in the old phase context is discarded (it is not the correct phase anymore). The orchestrator calls ExtractAndGenerate again in the new phase context — fresh `assemble_context`, new system prompt, `phase_started` block in the HumanMessage. The model generates prose from the correct phase. The user receives only the last (correct) response.

The model handles bridging as part of normal conversation — it is responding to the first turn of a new phase indicated by a `phase_started` signal.

**Terminal phases:**

A phase is terminal when its phase definition (config) has `"terminal": true`. Router detects and halts cleanly. When configured, executes terminal action e.g. POST request to configured service.

```json
{ "id": "booking_confirmation", "name": "Booking Confirmation", "terminal": true, ... }
```

Router logic:
- Phase is terminal AND all objectives satisfied → **HALT** (clean, intentional end)
- Phase is not terminal AND all objectives satisfied AND no rule matches → **`NoRouteHandler`**

`NoRouteHandler` always signals a config gap — incomplete rule coverage that needs fixing. It is never a valid terminal state.


## 5. Turm Execution Loop

Each user interaction advances the system through the following loop.

### 5.1 Main graph paths per turn (conceptual)

```
Phase transition path:
PreparePhase → ExtractAndGenerate → ApplyTools → Route  → new phase (PreparePhase — Phase transition path)
                                                        → or stay (RespondAndInput — Stay-in-phase path)
                                                        → HALT or handle error
Stay-in-phase path:
RespondAndInput → ExtractAndGenerate → ApplyTools → Route → new phase (PreparePhase — Phase transition path)
                                                          → or stay (RespondAndInput — Stay-in-phase path)
                                                          → HALT or handle error
SafeTool execution path:
ExtractAndGenerate → SafeToolExecutorNode (subgraph) → ResumeExtractAndGenerate →
→ ApplyTools → Route → new phase (PreparePhase — Phase transition path)
                     → or stay (RespondAndInput — Stay-in-phase path)
                     → HALT or handle error
```

The LLM call is in `ExtractAndGenerate` or `ResumeExtractAndGenerate` — tool calls are processed in `ApplyTools`, prose is held in `last_ai_prose` but not committed to history. The graph may traverse multiple phases before reaching an interrupt (user input).

The user always sees exactly one AIMessage per turn — always from the correct final phase.

### 5.2 Detailed behavior

── PHASE TRANSITION PATH ──

1. **PreparePhase**
   - Generate typed `resolve_<objective_id>` tools for all objectives based on `objectives.resultSchema`
   - Generate invalidation tool.
   - Generate `SafeTool` wrapper tools
   - Fetch stateless MCP tools
   - Build truth state injection block for system prompt
   - Build and cache agent with:
     * MCP tools
     * declare tools
     * objective tools (resolve_*, invalidate)
     * general system prompt
     * deployment prompt
     * phase prompt
     * objective definitions
     * state block (system prompt)
   - Note: Stateful tools and Agent are not cached. They are created in `ExtractAndGenerate`
   - Is this the 0. phase (first invocation)?
     * Yes: `RespondAndInput` (5)
     * No: `ExtractAndGenerate` (2)

3. **ExtractAndGenerate**
   - Message history assembled by `assemble_context`
   - Execute ReAct loop (LLM call — Produce Objective resolution / invalidation tool calls + prose in one call)
   - Store `AIMessage` prose in `last_ai_prose`
   - If a `SafeTool` is called: tool raises an interrupt. Continue in `SafeToolExecutor` (6)
   - Was any invalidate or `resolve_*` tool calls?
     * Yes: Continue in `ApplyTools` (3)
     * No: Commit `AIMessage` from `last_ai_prose` and continue to `RespondAndInput` (5) for the next turn 

4. **ApplyTools** 
   - Applie all `invalidate` tool calls
   - Propagate objective invalidation via dependency graph (cross-phase)
   - Applie all `resolve_\*` tool calls
   - Enforce runtime guards: resolve-once semantics, explicit invalidation only
   - Return a truth state mutation object (LangGraph mutates state via a reducer — ensures retry safety)
   - Return phase_extractions mutation object used by `assembly_context` (safe update via reducer)

6. **Route**
   - If current phase is terminal and all objectives satisfied → `TerminationHandler`
   - If required objective resolved as `fail` → `FailHandler`
   - If `last_mcp_errors` or `last_intent_errors` non-empty → `MCPErrorHandler` or `IntentErrorHandler`
   - Did invalidation occurre in ApplyTools? 
     * Yes: re-walk phase graph from ENTRY (see Section 3.4)
	 * No: Evaluate phase eligibility and transition rules
   - Phase transition → loop back to PreparePhase with new phase
   - Stay in phase → `RespondAndInput` (5)
   - If no rule matches → `NoRouteHandler`

── STAY-IN-PHASE PATH ──

5. **RespondAndInput**
   - Append `last_ai_prose` as `AIMessage` to history, clear `last_ai_prose`
   - Respond to user (last `AIMessage`)
   - Increment `turn_id`
   - Wait for user input (interrupt)
   - Receive user input
   - Append user input to conversation history as HumanMessageWrapper

── EXECUTE SafeTool PATH ──

6. **SafeToolExecutor**
   - Receive `safe_tool_interrupt` payload from orchestrator state
   - Run the intent execution subgraph inside the orchestrator's checkpoint namespace
   - Every execution step (idempotency key assignment, side-effect tool invocation, result storage) is checkpointed in the orchestrator graph
   - Write result to orchestrator state
   - Proceed to `ResumeExtractAndGenerate`

7. **ResumeExtractAndGenerate**
   - Recreate the ReAct agent if the MCP session was dropped during the SafeTool interrupt
   - Issue `Command(resume={ result })` into the agent's internal ReAct loop
   - Agent receives the `ToolMessage`, continues naturally
   - Post-turn state updates merged
   - Prose appended to `last_ai_prose`
   - Proceed to `ApplyTools`

### 5.3 Handler nodes

Handler nodes sit outside the main loops. Each is a simple node now and can be replaced with a subgraph independently without touching the main loop wiring.

**`FailHandler`**
- Deliver `fail_handler.message` to user (phase-level if configured, else `global_fail_handler`)
- Fire `fail_handler.notify_url` POST if configured: `{ thread_id, phase_id, objective_id, user_id? }`
- If `fallback_phase_id` configured: `Command(goto="PreparePhase")` with `fallback_phase_id` as new phase
- Otherwise: halt

**`TerminationHandler`**
- Fire `termination_notification.notify_url` POST if configured: `{ thread_id, phase_id, user_id? }`
- Halt cleanly

**`MCPErrorHandler`**
- Fire `error_notification.notify_url` POST if configured: `{ thread_id, phase_id, tool, error_code, error_message, user_id? }`
- Temporary Halt (might resume later)

**`NoRouteHandler`**
- Truth state reached a combination not covered by any routing rule
- Config error — halt


### 5.4 Idempotency guarantee

- LLM history is assembled before each LLM call, safely retriable
- ApplyTools returns an update object instead of mutating the state (safe to retry)
- Routter is a pure function of current truth state
- `last_ai_prose` always comitted to history once (no duplicate AIMessage) 
- SafeTool Execution Pipeline ensures safe execution. Each step is safely retriable

**Single AIMessage per turn:** the user always receives exactly one response, always from the final settled phase.


### 5.5. Phase transition eligibility and rule evaluation

Phase transition eligibility is evaluated in the Router Node.

**Evaluation sequence:**

1. Did any required objective resolve as `fail`?
   → execute configured fail handler → HALT
2. Are all owned `required` objectives resolved as `success`
   AND all owned `optional` objectives resolved as `success` or `skipped`?
   - No → remain in current phase
   - Yes → proceed to rule evaluation
3. Evaluate transition rules in priority order (lowest number = highest priority)
4. First matching rule → route to that phase
5. No match: Is phase terminal?
   - Yes → execute terminal action and finish conversation.
   - No → NoRouteHandler

**Rules:**
- Evaluated using only current truth state
- No model involvement
- No implicit fallthrough — all reachable truth state combinations must be covered by rules, either exhaustively or via a `rule: true` catch-all


## 5.6. Required Objective Failure

When a required objective resolves as `fail`, a configured fail response is executed by the orchestrator (not the model).

Authors must configure fail-handlers either at the phase level or at global level. 

Can be **phase-configurable**:
- Fail message delivered to the user
- Action executed e.g. request sent to a configured endpoint
- Fallback phase: appropriate for educational conversations or handle when the user returns with new information

If phase-specific fail handler is not present the system uses the **global fail handler**:
- Fail message delivered to the user
- Action executed e.g. request sent to a configured endpoints

Examples:
- *"Let me connect you with one of our team members who can help."*
- *"A colleague will reach out to you shortly."*
- *"I need more information to proceed, contact us when all the required information is available"*

After the fail handler executes, the system:
- Delivers the configured message to the user
- POSTs to `fail_handler.notify_url` if configured otherwise to `global_fail_handler.notify_url`: `{ thread_id, phase_id, objective_id, user_id? }`
- Routes to `fallback_phase_id` if configured, otherwise terminates

See Config and Phase Design Guide for full resolution order and global handler configuration.

## 6. Message Structure, State Injection & Caching

### 6.1 Block Schema

The orchestrator communicates important information (e.g. state, definitions) to the model using JSON blocks. Each block has a type. All blocks are read-only for the model.

Blocks are injected as text either to system prompt and/or to user prompt.


#### 6.1.1 State injection

`state_delta` blocks are used to inject state for the model. Keys are omitted when empty.

```json
{
  "block": "state_delta",
  "resolved": {
    "<objective_id>": {
      "definition": "<string>", // needed for invalidation
      "invalidation_instructions": "<string>",   // omitted if not set
      "value": { ...resolved fields... }
    }
  },
  "phase_started": true
}
```

**`resolved`** — objective truth. Omitted if no resolutions to report at this position. Assembly guarantees each objective_id appears exactly once across all blocks visible to the model. Phase membership is in config, not in the block. Definition and invalidation_instructions are injected because invalidation can happen in any phase regardles what owns the objective.

**`phase_started`** — present and `true` on the first turn of a new phase only. The new phase prompt is already in the system prompt; this signal gives the model explicit confirmation that a transition occurred.

**Invariant:** Each objective ID appears exactly once across all state_delta blocks visible to the model at any turn. The model doesn't have to reason about invalidation and re-resolve.


#### 6.1.2 `objective_definitions` block — phase objective metadata

Baked into the phase system prompt at PreparePhase time. Stable for the phase lifetime.

```json
{
  "block": "objective_definitions",
  "objectives": [
    {
      "id": "<string>",
      "name": "<string>",
      "description": "<string>",
      "extraction_instructions": "<string>",   // omitted if not set
      "required": "required | skippable | passive"
    }
  ]
}
```

For simple cases use the description only. 

```json
{
  "id": "adult_travelers",
  "description": "Travelers above the age of 14.",
  ...
}
```

If detailed extraction or invalidation instructions are needed use the dedicated fields instead of the description. This allows optimizations for the orchestrator. 

```json
{
  "id": "destination_airport",
  "description": "Destination airport code.",
  "extraction_instructions": "Use the find_airport MCP tool to retrieve airport codes. Provide a city and a search radius in km ...",
  ...
}
```
**Important:** Do not add objective description, extraction or invalidation instructions to the phase prompt. 


### 6.2 Message Structure

**Storage vs assembled view:**

Message history (See Section 2.3) stores prose conversation history `HumanMessage`/`AIMessage` pairs, SafeTool calls, results and optionally MCP tool calls. Objective tool calls (`resolve_*`, `invalidate`) are **not** stored — their outcomes are captured in truth state. SafeTool calls and their response `ToolMessage` are kept in history.

Injection blocks `{ "block": "state_delta" }` created by `assemble_context`. It is a pure function called 
- in `PreparePhase` to assembly state from previous phases and inject into system prompt
- immediately before each LLM (ReAct loop) invocation to inject current phase state deltas to HumanMessage

It reads current state and constructs the full LLM input on the fly. Nothing is stripped on phase change or invalidation — the assembled view is simply rebuilt from the source of truth.


### 6.3 Assemble context output

**SystemMessage**
```
[general_prompt]
[deployment_prompt] ← deployment-authored, optional
[phase_prompt] ← phase-authored, optional
{ "block": "objective_definitions", "objectives": [...] } ← baked at PreparePhase
[cache_control: ephemeral] ← cache breakpoint (1)                      
{ "block": "state_delta", "resolved": { "travel_dates": ..., "destination": ..., ... } }
[cache_control: ephemeral] ← cache breakpoint (2)                                        
```

**Cache breakpoint (1) Tools + general prompt + deployment prompt + objective definitions + phase prompt breakpoint** — stable within a phase. Written before first turn of each phase. Re-cached on phase transition (new tool set → new cache key). This breakpoint is valid cross users (See Section 6.4).

**Cache breakpoint(2) Completed-phase state breakpoint** — changes on phase transitions. Re-cached once before the first turn of each new phase.

**HumanMessage / AIMessage alternating pairs:**

Each `HumanMessage` is assembled to contain **current-phase** objective deltas, and the phase_started signal — followed by a blank line and stored prose. Keys are omitted when empty.

```
Turn 10:
  HumanMessage:
    I'm thinking Four Seasons actually.

  AIMessage:
    resolve_preferred_chain("FS") ← resolve tool call not stored in history
    Great choice — Four Seasons has excellent properties in Rome. ...

Turn 11:
  HumanMessage:
    { "block": "state", "resolved": { "preferred_chain": { "chain_code": "FS" } } }  ← previous resolution injected as state delta

    Can we make sure there's a spa?

  AIMessage:
    resolve_amenities(["spa", "pool"]) ← resolve tool call not stored in history
    Noted — I'll filter for properties with spa access. ...

Turn 12:
  HumanMessage:
    { "block": "state", "resolved": { "amenities": { "amenities": ["spa", "pool"] } } }  ← previous resolution injected as state delta
	
	Ok, let's see what is available.
...

Turn N-1:
  HumanMessage:
    Ok, reserve the hotel.
  
  AIMessage:
    safe_tool_wrapper_book_hotel(<semantic_id>, <offer_id>) → SafeTool executor subgraph runs
    ToolResult({"status"="succeeded", "cancellation_id": "cxl_abc123"}, ...) ← Not modified by the orchestrator
    
  AIMessage:
    I made the reservation ...

← cache_control: ephemeral (rolling, on second-to-last turn)	

Turn N:
	
  HumanMessage:
    Great, what's next?
  
  AIMessage:
    You are all set. If plans change, continue this conversation, I can cancel the reservation and find a new accomodation for you.  
```

**Note:** The SafeTool executor subgraph is stateful, it won't execute the same `semantic_id` twice after it succeeded. Model can extract objectives from tool response as usual when configured.

**How `assemble_context` determines which state to prepend to each turn:**

Each `HumanMessage` is stored as a `HumanMessageWrapper` carrying `phase_execution_id` and `turn_id`. `assemble_context` checks each wrapper against `state.current_phase.phase_execution_id` and `state.turn_id` (See Section 2.2):
- `wrapper.phase_execution_id != current_phase.phase_execution_id`
  → prior or completed execution; state for that phase is in the system prompt; no block prepended
- Same phase execution, `wrapper.turn_id - 1 in state.phase_extractions`
  → emit state block if non-empty

The model always sees each objective exactly once — the invariant holds by construction.

**Why can't inject the full state into the system prompt:**
Because every objective resolution and invalidation would cause a cache miss. It means higher cost and latency.

### 6.3 Invalidation & State Coherence

After invalidation the invariant objective ID appears exactly once across all assembled state blocks still need to hold.

**Normal turns (no invalidation) for reference:**
 `RespondAndInput` increments `turn_id`. ApplyTools writes resolved objectives to `state.phase_extractions[turn_id]`. `assemble_context` looks up `phase_extractions[wrapper.turn_id - 1]` for each wrapper and emits a state block if non-empty. 

```
PhaseExecution after 2 turns:
  phase_execution_id: "exec-abc"
  state.history = [
    HumanMessageWrapper(
        content="Hi, I'd like to travel to Rome",
        phase_execution_id="exec-abc"
        turn_id=1),
	AIMessage("Hi, ... Do you have dates in mind? ..."),
    HumanMessageWrapper(
        content="july 12 to july 20e",
        phase_execution_id="exec-abc"
        turn_id=2)
  ]
  state.phase_extractions: {
    1: { "destination": { "status": "success", "value": { "city": "Rome" } } },
    2: { "travel_dates": { "status": "success", "value": { ... } } },
  }

assemble_context for phase_execution_id="exec-abc", turn_id=2:
  for the 2nd turn's HumanMessage assemble_context injecta objectives extracted in 1st turn:
  { "destination": { "value": { "city": "Rome" } }

assemble_context for phase_execution_id="exec-abc", turn_id=3:
  for the 3rd turn's HumanMessage assemble_context injects objectives extracted in 2nd turn:
  { "travel_dates": { "value": { "check_in": "2026-07-12", "check_out: "2026-07-20" } }
```

**On invalidation within the current phase:**
Invalidated objective is removed from `phase_extractions`. The current turn's `assemble_context` won't add it to HumanMessage. Cache misses from this turn affect messages after the extracton turn of the invalidated objective.

```
state.history = [
    ...
    HumanMessageWrapper(
        content="I was wrong we can not travel in july ...",
        phase_execution_id="exec-abc"
        turn_id=3)
  ]
phase_extractions after 3 turns (travel_dates removed):
  state.phase_extractions: {
    1: { "destination": { "status": "success", "value": { "city": "Rome" } } },
  }

assemble_context for phase_execution_id="exec-abc", turn_id=3:
  for the 3rd turn's HumanMessage nothing to inject as there is nothing for key=2 :
```

**On phase transition after invalidation:**
Router advances the phase. PreparePhase for the new phase generates a fresh `phase_execution_id` and empty `phase_extractions`. `assemble_context` uses the completed phase's objectives from `state.truth` and injects into the SystemMessage as a state block. Prior-phase `HumanMessageWrapper`s have a different `phase_execution_id` — no current-phase blocks prepended for them.

```
Update phase execution id
  phase_execution_id="exec-cdf"

PhaseExecution after 2 turns:
  state.phase_extractions: {}

assemble_context assemblies the full state in the system prompt and no state is injected into HumanMessages for phase_execution_id="exec-abc"
```

### 6.4 Cache Strategy

This system is designed to work with Anthropic style caching that caches content as a linear token stream keyed by a cryptographic hash of the prompt prefix up to each `cache_control` breakpoint. Caches are **organisation-scoped** — not user- or session-scoped. The TTL is 5 minutes by default (1 hour available), and **resets on every cache hit**. There is no explicit invalidation — cache entries simply expire when the TTL elapses without a hit.

**Cache evaluation order: tool definitions → system messages → conversatoin.** The cache prefix is built in this order, so a change to the tool list produces a different prefix hash and therefore a different cache entry. The old entry continues to exist independently until its TTL expires.

**Cross-user cache sharing:** Because no user or session identifiers are baked into the tool list or system prompt — tools are phase-scoped, and both the general prompt and phase prompt are static per phase — any two users in the same phase within the same organisation share the same cache entry. Under load, each request hitting that entry resets the TTL, keeping the cache. This is the primary scaling property: cache efficiency increases with load, exactly when it matters most.

| Content | Breakpoint | Cache behaviour |
|---|---|---|
| Tools + general prompt + deployment prompt + objective definitions + phase prompt | 1 | Shared across all users in this phase. Stable until deployment prompt or phase prompt changes. Miss on phase transition (new tool set → new cache key). |
| Completed-phase state blocks in system prompt | 2 | Shared across messages within the same phase. Miss on phase transition. |
| Prior conversation turns (assembled from HumanMessageWrapper) | Rolling | Shared across messages within the same phase. Miss on phase transition. |
| Previous user prose | None | Always fresh. |

**Phase transitions** produce a cache miss at breakpoints 1 (if no cross-user entry is available) and 2 — the new phase agent has a different tool set, yielding a different prefix hash. The new entry is written on the first turn of the new phase.

**In-phase state changes (objective resolve)** changes only the last assembled HumanMessage content — rolling cache set to before last so cache unaffected.

**In-phase state changes (objective invalidation)** change the assembled HumanMessage content — rolling cache misses from the affected turn forward. Breakpoints 1 and 2 are unaffected.

**Low-load behaviour.** When fewer than one user per 5 minutes is in a given phase, the cache expires between requests and every user pays full write cost. For phases that are used infrequently but have expensive prompts, consider the 1-hour TTL.


### 6.5 Summarization

When conversation history exceeds a configurable size threshold, a summarization model condenses it:
- Has access to full history and all resolved objectives
- Condenses prose turn content only — MCP tool call/result messages may be summarised into prose; state blocks are recomputed by `assemble_context` and are never part of what gets summarised.
- Output must be conservative: preserve ambiguous or potentially relevant user statements.


### 6.6 Session Continuity

All state and conversation history are persisted across sessions. Users may pause and resume at any point. On re-entry, resolved objectives render as state blocks in the SystemMessage; current-phase objectives render correctly from the persisted `HumanMessageWrapper` identity and `state.phase_extractions` — `assemble_context` reconstructs every prior turn's state block exactly as it was. No re-confirmation of resolved objectives is required unless the user explicitly changes them.

### 6.7 Streaming

#### 6.7.1 cross-phase extraction with last AI prose (default)

Token streaming is not compatible with cross-phase extraction in a single user turn — which is a deliberate feature, not an edge case.

The reason is structural: the model cannot reliably predict whether its extraction will trigger a phase transition. A model that resolves objectives in a phase responds naturally within its context — it has no way to know the router is about to advance.

Streaming without knowing routing decisions would deliver a phase-inapropriate response to the user before the correct one, producing a visible double-message glitch.

The `last_ai_prose` pattern resolves this correctly: in `ExtractAndGenerate` Node, prose is held, not streamed; the final phase commits exactly one AIMessage; the user sees this one, always from the correct phase. The trade-off is no token streaming — the client displays a loading indicator while the turn processes.

#### 6.7.2 Real time use with streaming (experimental)

For real time cases, AI prose is visible from multiple phases. It requires careful phase design and phase specific prompt engineering to minimize glitchy phase transitions. 

## 7. External Tools (MCP) Architecture

### 7.1 Tool Scope

External tools:
- Are attached to specific phases
- Can be attached to multiple phases
- Are injected only when that phase is active
- Never mutate truth directly

### 7.2 Tool Safety Model

All MCP tools injected into a phase are safe for the model to call freely (idempotent). Tools with real-world side effects (booking, payment, cancellation) should be exposed as SafeTools.

This means:
- Every tool in the model's tool list is safe to call; the model does not need to reason about idempotency
- Authors ensure safety by what they do and do not expose as plain MCP tools
- The SafeTool boundary is the trust boundary — params are validated, tool is executed by the orchestrator with idempotency guarantees

### 7.3 Responsibility Split

| Concern | Owner |
|---|---|
| MCP tool invocation | LLM (ReAct loop) |
| SafeTool execution | Orchestrator (via executor subgraph) |
| Result interpretation | LLM |
| Truth mutation | Objective tools only |

### 7.4 MCP Tool Handling

The agent is stateless beyond its ReAct loop — message history is owned by the orchestrator, assembled from state before each call, and passed in. The agent retains no internal state between node executions or across retries.

The inner ReAct loop allows the model to make multi-step MCP chaining (search → filter → availability check) where the number of calls is not fixed in advance.


**Agent built on phase entry, cached for reuse**

Agents are built lazily on first entry to each phase. Stateless MCP tool list can be cached cross users to scale better. Stateful tools must be attached to a session, caching can not work here.

**Stateful MCP phases:**

For phases requiring a stateful MCP session, the agent is built per ReAct loop invocations inside the ExtractAndGenerate Node.

Agents are stateless across invocations. On retry the outer graph reconstructs message history from `state.messages` and passes fresh context.

**Message persistence after agent run:**

Not all agent output message is stored in `state.history`. By default, MCP Tools and results are filtered out. The agent either has to resolve related objectives or note results in prose. Objectives can be invalidated and are the primary way to manage state.
This behavior optimizes token usage and removes obsolate information e.g. a previously called search_flight(...) tool result is obsolate later.


### 7.5 MCP Error Handler Tool

All phases include a `report_mcp_error(tool, code, message)` tool in addition to objective and declare tools. This gives the model a structured exit for situations it cannot resolve conversationally — persistent MCP failures, validation deadlocks, unexpected service responses.

```python
report_mcp_error(
    tool="search_flight",
    error_code="xyz",
    message="Flight search tool returned failure after 3 retries"
)
```

**MCP errors can be used in routing:**

```python
class MCPErrorEntry(TypedDict):
    tool: str
    error_code: str
    message: str
    
last_mcp_errors: dict[str, MCPErrorEntry] # tool_name → entry
```
Example jsonLogic rule:

```json
{"!=": [{"var": ["last_mcp_errors.search_room.error_code"]}, 3456]}
```

**The model should call `report_mcp_error` when:**
- An MCP tool returns persistent errors (not transient; after retry)
- A required value cannot be obtained and the failure is not user-caused
- A service returns an unexpected response that prevents progress

The model must not call `report_mcp_error` for normal conversation difficulties — a user who refuses to provide a required value should be handled via `resolve_*(status="failed")`, not as a tool error.

## 8 Side effect execution (SafeTools)

Some MCP tools have real-world side effects: they lock inventory, initiate transactions, or modify external state. A plain MCP tool call inside the ReAct loop provides no idempotency guarantee — if the node is retried after an error, the tool may execute twice. **SafeTools** solve this by wrapping an existing MCP tool with semantic-based idempotency and safe checkpointing, without requiring a separate HTTP executor.


### 8.1 What a SafeTool is

A SafeTool is configured in `side_effect_intents` by referencing an MCP tool that already exists in the phase's `mcp_config`:
```json
{
  "id": "reserve_room",
  "description": "...",
  "on_success_updates": "<UpdateObject>",
  "on_fail_updates": "<UpdateObject>",
}
```

See `UpdateObject` structure in Section 8.5

`id` must match an MCP tool name. The orchestrator wraps that LangChain Tool object internally. The model sees the wrapped tool with the configured `description` replacing the original tool description. The `semantic_id` parameter is added automatically. The description should explain what `semantic_id` means in the tool's context.

The underlying tool is called via `tool.invoke(params)`. Success is determined by outcome: normal return → `succeeded`; exception → `failed`. The raw return value is stored and returned to the model as a ToolMessage within the same ReAct loop turn.


### 8.2 Idempotency guarantee

Call safety is enforced by `semantic_id`: if an entry with the same `semantic_id` already has status `succeeded`, a new declaration is blocked. The model must generate a new `semantic_id` for any logically new action. Same `semantic_id` on retry is strictly for `failed → retry with different params`.

For API execution safety, the subgraph generates an `idempotency_key` that is checkpointed before execution attempt. If the LangGraph node is retried after failure, the same `idempotency_key` will be used.


### 8.3 What SafeTools do not manage

**High-Stakes Actions Requiring Explicit Confirmation**: 

Some intents cannot be executed by the orchestrator unilaterally. Regulatory requirements, financial risk controls, or UX policy may require a deliberate, explicit human action outside the conversation — a checkbox, a button click, a digital signature — as the authoritative record of consent.

The model reads the ToolMessage result, extracts confirmation context (transaction ID, confirmation URL) into objectives, and uses MCP query tools to request status. The designer controls the flow through objectives and routing rules.

**Cancellation**: the orchestrator has no built-in concept of cancellation. A cancellation tool is just another SafeTool with user-defined semantics. 

**How to handle tool results:**

The model extracting  from `ToolMessage` or post-turn state updates can encode the semantic consequence — which objectives to invalidate or update. The router determines the correct next phase from those objective changes.

### 8.4 SafeTool Execution

Side effects are handled via a three-stage process.

**Stage 1: SafeTool call (inside ReAct loop)**
- LLM calls generated tools e.g. `book_flight(...)`
- Purely declarative — no side effect execution at this stage
- interrupts tool call and the orchestrator routes to SafeToolExecutorNode (subgraph) 

**Stage 2: Preparation node**
- Executor treats `semantic_id` as execution level idempotency key and prevents double call after success.
- Assigns an API/Service level idempotency key to the intent
- No external calls, no commitments
- Safe to retry

**Stage 3: Executor node**
- Executes: `tool.invoke(<params>)`
- Tool raise means `failed` otherwise `succeeded` (Consistent with LangChain)
- The executor does not read the payload
- Checkpoints execution
- Routes to `ResumeExtractAndGenerate` to resume tool execution inside the ReAct loop with the tool payload.
- API/Service level idempotency keys protect against duplicate execution due to response loss, and crash before persistence

All three stages are retriable by LangGraph. Checkpointing prevents re-execution of completed stages.
Idempotency keys prevent re-execution on DB and network errors.

### 8.5 Post-turn state updates

SafeTools results are not injected into the state block. The ToolMessage is visible in the conversation history. Designers use objectives to capture whatever slice of the tool result the conversation needs for routing or downstream phases. 

For simple cases the model can handle objective resolutions from ToolMessage, but for complex cases or deterministic, guaranteed update the following options are available:

**UpdateObject:**

```json
{
  "default": {
    "booking_reference": "HTL-8821"
  },
  "override": {
    "booking_status": "confirmed",
    "invalidated_field": null
  }
}
```

Use `null` for invalidation. 

**Merge sequence** (after ReAct loop, before ApplyTools):
```
step1 = merge(default, model_extractions + model_invalidations)   ← model wins on conflict
step2 = merge(step1, override)                                    ← override wins on conflict
```

`default` and `override` are both optional and independent. Omit either if not needed.


**`on_success_updates | on_fail_updates`** — static update object applied after a turn where the SafeTool succeeded / failed. On multiple SafeTool calls merge sequence applied in tool call order. Use for simple, predictable postconditions that do not depend on the tool's return value.

```json
"on_success_updates": "<UpdateObject>",
"on_fail_updates": "<UpdateObject>",
```

Note: if the Callable (in case of library distribution) or webhook (service distribution) is configured it provides the state update. `on_success_updates | on_fail_updates` are ignored. 

**`Callable` or `on_safe_tool_webhook`:** if configured it is called after the ReAct loop completes for turns containing SafeTool activity. Receive a `WebhookPayload`:

```json
{
  "thread_id": "...",
  "phase_id": "accommodation_booking",
  "phase_execution_id": "...",
  "turn_id": 42,
  "safe_tool_calls": [
    {
      "tool": "reserve_room",
      "semantic_id": "book_room_1",
      "params": { "offer_id": "offer_1234" },
      "status": "succeeded",
      "result": { "cancellation_id": "cxl_abc123", "confirmation_number": "HTL-8821" },
      "error_message": null
    }
  ],
  "prose": "Your room is confirmed. Confirmation: HTL-8821.",
  "extracted": {
    "rooms_booked_count": { "count": 1 }
  },
  "invalidated": []
}
```

Both succeeded and failed SafeTool calls are included — the webhook handles partial success cases.

The webhook returns an `UpdateObject` or `{}`:

**Synthetic state update**

1. For each `objective_id` in the update object:
   - Validate the ID exists in config — if unknown (config error) halt
   - If `id in state.truth` → append to invalidation list (invalidate first semantics still holds)
   - If fields provided → call `resolve_tool.args_schema.model_validate(fields)` for schema validation then `resolve_tool.func(**params)`
     for business logic validation — `ValidationError` is a hard error (config or webhook), halt
   - Append to resolved dict if valid
2. Follow graph execution (ApplyTools → Router). Invalidation trigger a re-walk and potential phase change.

## 9. Retry & Durability Model

### 9.1 What is Retriable

All LangGraph nodes are retriable by design. LangGraph retries on failure and resumes from the last checkpoint:
- Model execution
- Tool Application
- Routing Evaluation
- SafeTool call and execution

**Guarantees on retry:**
- No duplicated truth mutation
- Routing depends only on truth state
- No duplicated SafeTools for the same `semantic_id` and `idempotency_key`


### 9.2 LLM Call Failure

If any LLM invocation fails, LangGraph retries from the last checkpoint. No direct state mutation happens inside a Node. State is updated by the LangGraph runtime using reducers. Retry is clean and idempotent.


## 10 Constants and the JsonLogic Data Object

All JsonLogic expressions in the system — routing rules and objective validators — are evaluated against a unified data object with a consistent structure.

### 10.1 Data object structure

**Objective fields** use dot notation — `objective_id.field_id`:
```json
{"var": "destination.city_code"}
{"var": "travelers.adults"}
{"var": ["travel_type.type", "leisure"]}    ← two-element form for optional/passive fields
```

**Constants** are accessed under the `const` key:
```json
{"var": "const.business_travel_cities"}
{"var": "const.budget_tiers.0.max"}    ← array indexing with constant index
```

**Example:**

```json
{
  "destination": { "city_code": "LON", "country": "GB" },
  "travel_type": { "type": "business" },
  "travel_dates": { "check_in": "2026-07-08", "check_out": "2026-07-12" },
  "const": {
    "business_travel_cities": ["LON", "NYC", "PAR", "FRA"],
    "max_group_size": 8,
    "budget_tiers": [
      { "name": "standard", "max": 2000 },
      { "name": "premium", "max": 5000 }
    ]
  }
}
```

### 10.2 Constants definition

Static constants are defined in a `constants.json` file alongside the graph config. Loaded at service start, immutable at runtime, available in all JsonLogic expressions.

Example `constants.json`:

```json
{
  "business_travel_cities": ["LON", "NYC", "PAR", "FRA"],
  "max_group_size": 8,
  "budget_tiers": [
    { "name": "standard", "max": 2000 },
    { "name": "premium", "max": 5000 }
  ]
}
```

The orchestrator merges this under the `const` key before evaluating any rule.

For values that change frequently or depending on runtime variables — available destinations, current pricing, policy limits — use MCP tools or an objective resolved from an MCP lookup rather than constants. Constants are for static values that are stable for a longer perid.


### 10.3 Usage in routing rules

```json
{"==": [{"var": "travel_type.type"}, "business"]}

{">=": [{"var": "travelers.adults"}, {"var": "const.max_group_size"}]}

{"in": [{"var": "destination.city_code"}, {"var": "const.business_travel_cities"}]}
```

Error routing variables follow the same dot notation, under their own namespaces:

```json
{"!=": [{"var": "last_mcp_errors.search_flight.error_code"}, null]}
{"==": [{"var": "last_intent_errors.book_room.http_status_code"}, 503]}
```

### 10.4 Usage in objective validators

```json
{
  "validator": {
    "rule": {
      "in": [{"var": "destination.city_code"}, {"var": "const.business_travel_cities"}]
    },
    "defer_until": ["travel_type"],
    "message": "This destination is not available for business travel."
  }
}
```

Inline lists can be used for short enumerations — no `constants.json` entry needed:

```json
{
  "rule": {"in": [{"var": "destination.city_code"}, ["LON", "NYC", "PAR"]]},
  "defer_until": ["travel_type"],
  "message": "This destination is not available for business travel."
}
```

## 11. Config Validation — Hard Errors

These conditions are validated at graph compile/save time and must be rejected before runtime.

**Structural hard errors:**
- Unreachable phases are NOT allowed (every non-entry phase must have at least one inbound edge reachable from ENTRY)
- Objective dependency cycles are NOT allowed
- A phase with no incoming edges is NOT allowed unless it is the entry phase
- Routing cycles in the phase graph are NOT allowed (see below)
- An `Objective` with a `phase_id` that does not exist in the graph is NOT allowed
- `SafeTool` referencing non-exisitng MCP tool

**Safe cycles are built into the architecture:**
- Staying in a phase is the default — a phase with unsatisfied required objectives does not route forward. No self-loop needed.
- Returning to a prior phase after invalidation is handled by the re-walk — routing edges back are not needed and would be ignored by the re-walk anyway.
- Retry or error recovery is handled by `fail_handler.fallback_phase_id` or a dedicated error phase with a forward edge

*Configured phase cycles (backward edges) are not needed and allowed*

Detect routing cycles using DFS on the directed edge graph at compile time. Reject any graph containing a cycle. Self-loops must be caught as a special case before the general DFS (they may not surface depending on the DFS implementation)

**Dependency dominance constraint:**

If objective B depends on objective A, then `phase(A)` must **dominate** `phase(B)` in the phase graph from the entry node. A phase X dominates phase Y if every path from ENTRY to Y passes through X.

If this does not hold, there exists a path to `phase(B)` that bypasses `phase(A)` entirely — B could be reached before A is resolvable, making the dependency unenforceable.

```
ENTRY → welcome → basic_info → special_options
                ↘                ↗
                  bypassing_phase

hotel_chain (special_options) depends on destination (basic_info):
  Path: ENTRY → welcome → bypassing_phase → special_options
  → reaches special_options without passing through basic_info
  → basic_info does NOT dominate special_options
  → INVALID — hard error at config time
```

Compute dominators using standard graph algorithms (e.g. Lengauer-Tarjan) at graph compile time. Validate all objective dependency edges against the dominator relation. Reject the graph if any violation is found.

**Allowed structures:**
- Objectives may have no dependencies
- `"terminal": true` on a phase is the correct way to mark a terminal phase

**Routing rule completeness:**
`NoRouteHandler` at runtime indicates a config error — truth state reached a combination not covered by any routing rule. Validate rule coverage against known objective combinations during config review. See the Config and Phase Design Guide for guidance.

Design recommendations, phase authoring guidance, debugging tips, and MCP best practices are in the **Config and Phase Design Guide** — that document is the primary reference for application authors.


## 12. Non-Goals

The orchestrator does NOT attempt to:
- Infer user intent implicitly
- Perform autonomous task planning
- Guarantee optimal conversational paths
- Enforce business correctness via the model

Those concerns belong to pre-defined and validated orchestration logic. If adaptation is required, it must be encoded there.
