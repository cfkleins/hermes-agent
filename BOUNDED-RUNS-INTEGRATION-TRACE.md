# Bounded Runs Integration Trace (rt3c)

Status: read-only architecture trace; no implementation, imports, tests, network access, service activation, or credential/config reads were performed.

## Decision

Use a **distinct bounded HTTP surface**, not a body/header discriminator inside generic `POST /v1/runs`:

- `POST /v1/bounded-runs`
- `GET /v1/bounded-runs/{run_id}`
- `POST /v1/bounded-runs/{run_id}/stop`

The bounded submission handler should reuse the native run lifecycle machinery after admission, but it must have separate ingress authentication, exact raw-body handling, ownership checks, capability advertisement, and a sealed agent factory. Do not add bounded-only keys to the generic `/v1/runs` body and do not infer bounded mode from `case`, `model`, `Idempotency-Key`, or a client-supplied boolean/header.

Why this is safest:

1. Generic `/v1/runs` accepts broad OpenAI/Hermes-native input, history, session, model, provider, route, and options. Bounded admission accepts exactly two JSON keys (`input`, `case`) and requires an idempotency key. A shared handler necessarily creates an allow/deny discriminator at the security boundary.
2. Generic admission authenticates with `API_SERVER_KEY` or room grants in `_admit_api_agent_request`; rt3b authenticates against an immutable bounded registry. A distinct path lets authentication complete before any body read without changing generic auth semantics.
3. Existing status/stop ownership is derived from generic API-key/room-grant scope. Reusing those HTTP endpoints would either deny the bounded principal or widen generic ownership. Distinct control routes keep the two authorization domains separate while reusing the same internal run maps/store.
4. Capability discovery can advertise bounded availability independently. Existing `run_submission` and `runs_idempotency` remain unchanged.
5. `/p/{profile}` mirrors can still be registered by the existing route-table loop, so profile scoping remains server-derived.

A fail-closed discriminator remains appropriate **inside the private launch object/executor**, where a frozen rt3b admission selects the bounded factory. It is not appropriate at the public generic route.

## Evidence read

- `AGENTS.md`: preserve generic behavior, prompt caching, narrow waist, behavior tests, no source-shape tests.
- `gateway/AGENTS.md`: gateway facade/sibling layout and fail-closed profile identity rules.
- `agent/AGENTS.md`: constructor/turn path, byte-stable prompt, model/fallback sensitivity, durable history.
- `gateway/platforms/api_server.py`:
  - global request cap is 10,000,000 bytes; `body_limit_middleware` maps `HTTPRequestEntityTooLarge` to 413;
  - `_admit_api_agent_request` authenticates, checks drain, and reserves pending work before the handler's first await;
  - native/multiplex routes are generated from `_http_route_table`;
  - `_create_agent` performs generic runtime/default/model/session/fallback/toolset resolution and is therefore not a bounded factory;
  - capabilities currently advertise only generic runs;
  - `web.Application(..., client_max_size=MAX_REQUEST_BYTES)` is much larger than rt3b's 65,536-byte cap.
- `gateway/platforms/api_server_runs.py`:
  - generic request JSON/history/model/session selection precedes native launch construction;
  - replay lookup precedes concurrency, then atomic reserve closes the race;
  - `_RunLaunch.agent_kwargs` currently feeds generic `_create_agent`;
  - `_run_agent_sync` does not yet pass `binding_identity`;
  - native status/stop use authenticated scope ownership;
  - initial/live/terminal status persistence is best-effort and exceptions are swallowed in `_set_run_status`.
- `gateway/platforms/api_server_run_idempotency.py`:
  - unique `(scope, idempotency_key)` reservation is serialized by `BEGIN IMMEDIATE`;
  - initialization silently falls back to process memory;
  - no conditional reservation rollback primitive exists;
  - status updates are durable when the database is available.
- `gateway/platforms/api_server_bound_runtime.py` (rt3a): exact Nous/static-key/chat-completions lane; returns frozen `BoundRuntime`; explicitly forbids generic resolution, pools, request overrides, and fallback recovery.
- `gateway/platforms/api_server_bound_admission.py` (rt3b work observed during the trace): frozen `BoundAdmission` with principal, validated input/key, native bound session id, immutable binding identity, exact context bytes, required skills, deny-all policy, and `BoundRuntime`; admission authenticates before JSON parsing and mutates native session state only after request/policy/runtime validation.
- `agent/turn_facade.py` and `agent/turn_facade_lease.py`: `run_conversation(..., binding_identity=...)` validates the durable binding before lease/history, and `require_durable_history` replaces caller history with native durable history.
- `agent/agent_init.py` and `agent/tool_execution_policy.py`: bound sessions force deny-all/durable-history; deny-all is enforced at schema, dispatch, and final request publication for verified `chat_completions` backends.
- Existing run, idempotency, drain, bound-runtime, bound-admission, and bounded-conversation tests.

`BOUNDED-BRIDGE-TRACE.md` was requested as an input but was not present under `C:/Users/cfkle`, this checkout, any listed worktree, or any local git ref. Conclusions that depend on its unstated contract are explicitly called out below.

## Required integration order

The bounded route must use this order. A later step must not occur if an earlier step fails.

1. **Resolve profile and bounded registry from server state.** The existing profile-prefix middleware establishes the profile scope. Do not read credentials/config from the request body.
2. **Authenticate before reading the body.** Validate the single Authorization value against the bounded registry in a non-awaiting wrapper. Unauthorized requests return the rt3b sanitized 401 and leave `request.content` unread, `_pending_agent_requests == 0`, SessionDB untouched, and RunIdempotencyStore untouched. Current `admit_bound_run` authenticates before parsing already-supplied bytes, but the HTTP adapter still needs a public rt3b authentication seam (or an admission API split) so it does not have to call private `_authenticate` or authenticate twice.
3. **Check drain and reserve pending request work.** After authentication, before the first await, apply the same mutable reservation protocol as `_admit_api_agent_request`. The bounded route needs its own decorator because the generic decorator invokes `_check_auth`/room-grant auth.
4. **Require a durable run store before body/state work.** If `self._run_idempotency_store.durable is not True`, return retryable 503 `bounded_idempotency_unavailable`. Do not accept bounded execution on the store's current in-memory fallback.
5. **Apply the 65,536-byte raw limit while reading exactly once.** Check valid `Content-Length` only after auth, then incrementally read `request.content`, stopping at `MAX_RAW_BODY_BYTES + 1`. Do not call `request.json()`, `request.text()`, or `request.read()` and then parse again. Pass the exact collected `bytes` to rt3b. This preserves duplicate-key rejection and avoids allowing aiohttp's 10 MB global cap to become the bounded cap. Chunked and declared-length bodies must follow the same 65,536-byte rule. Map bounded overflow to stable 413 without including body data.
6. **Obtain one frozen rt3b admission.** Map only sanitized `BoundAdmissionError.code` values. No raw exception, authorization value, API key, context bytes, digest, or body text enters logs/errors/repr.
7. **Derive idempotency scope and fingerprint from accepted authority, not client authority.**
   - Scope: `SHA256("bounded-run-v1\0" || resolved_profile || "\0" || admission.principal_id)`. This makes the key namespace per authenticated principal/profile. Do not persist the bearer secret.
   - Fingerprint: SHA-256 over a versioned, length-prefixed binary encoding of: exact raw-body SHA-256; native session id; all binding-identity fields in fixed order; context digest; runtime public tuple `(provider, model, api_mode, base_url)`; required skill names in fixed order; and tool policy. Do not include the model API key or raw context bytes. Do not use ambiguous string joining or a dict whose future defaults can silently alter meaning.
   - Principal-wide scope intentionally makes reuse of one key for another case/input a 409 conflict rather than a second run. If product policy wants per-case key reuse, that is a separate owner-approved contract change, not an implementation default.
8. **Replay lookup before capacity.** After complete admission/fingerprint, lookup `(scope,key,fingerprint)`. A matching durable record returns the original run/status even at capacity. A different fingerprint returns 409. Neither starts an agent nor consumes capacity.
9. **Concurrency check for a genuinely new run.** Use the existing shared `max_concurrent_runs` calculation. The current pending reservation must be excluded exactly as in the generic admission test. Do not reserve the idempotency key on a 429.
10. **Build local launch material without side effects.** Generate the run id, queue, created timestamp, initial public status, and `_RunLaunch` in locals. Do not publish owner/status/stream/approval maps yet.
11. **Atomically reserve before launch.** Call durable `RunIdempotencyStore.reserve`. A concurrent winner replays/conflicts. A persistence exception returns 503 and publishes no native run state/task. Only `outcome == "created"` may continue.
12. **Publish native ownership/control state, register the task, then transfer the drain reservation.** Insert `_run_owners`, `_run_streams`, `_run_streams_created`, `_run_statuses`, `_run_idempotency_ids`, and `_active_run_tasks`. Call `_activate_admitted_request()` only after the task is visible in `_active_run_tasks`; the current generic path releases just before `create_task`, leaving a small drain-accounting gap worth correcting for both paths.
13. **Execute through a sealed bounded factory.** The task persists `running` before constructing/calling the model. Factory or persistence failure produces a sanitized terminal failure if that terminal state can be durably committed.
14. **Run exactly the admitted input with native history.** Call `agent.run_conversation(user_message=admission.input_text, conversation_history=None, task_id=admission.session_id, binding_identity=dict(admission.binding_identity))`. `None` is required: do not invoke `_resolve_conversation_history`, `previous_response_id`, body `session_id`, `X-Hermes-Session-Key`, or `_conversation_history_for_session`. Durable lease admission owns history loading.
15. **Finalize status before terminal event/cleanup.** Persist terminal status first, then emit the event/sentinel, retire live refs, and retain ownership for the existing status-retention window.

## `_RunLaunch` contract

Do not put bounded secrets/authority into the mutable generic `agent_kwargs` dict. The least invasive safe extension is:

- keep all existing generic fields unchanged;
- change `conversation_history` to optional so bounded launches carry exactly `None`;
- add one nullable `bound_admission` field containing the frozen rt3b object;
- add durable reservation identity needed for strict cleanup: `idempotency_scope`, `idempotency_key`, `idempotency_fingerprint` (or a frozen reservation token returned by the store).

Executor invariants when `bound_admission is not None`:

- `session_id == admission.session_id`;
- `user_message == admission.input_text`;
- `conversation_history is None`;
- no generic route/model/provider/session/history fields are present in `agent_kwargs`;
- factory is `_create_bound_agent`, never `_create_agent`;
- `run_conversation` receives the immutable admission binding identity;
- approval/steer are not advertised for bounded runs (deny-all cannot legitimately ask for tool approval, and steer would introduce unadmitted prompt bytes).

If any invariant is false, fail before agent construction. This is the correct internal fail-closed discriminator.

## Bounded factory

Add a narrow `_create_bound_agent(admission, callbacks, session_db)` in the bounded-runs sibling. It should construct `AIAgent` directly with explicit values only:

- `model`, `provider`, `requested_provider`, `api_mode`, `base_url`, `api_key` from `admission.runtime`;
- `credential_pool=None`, `fallback_model=None`, `request_overrides=None`;
- `enabled_toolsets=[]`, `deny_all_tools=True`, `require_durable_history=True`;
- `session_id=admission.session_id`, exact profile-scoped native `session_db`, `platform="api_server"`;
- `skip_context_files=True`, `load_soul_identity=False`, `skip_memory=True`, `skip_background_review=True`;
- fixed server-owned `max_iterations`, `run_budget_seconds`, output/token ceiling, reasoning policy, quiet/log settings, and callbacks. These limits must be code/registry policy, never generic gateway defaults or body values.

Decode `admission.context_bytes` once with strict UTF-8 and verify round-trip equality before construction. Pass that exact text as the approved prompt component. Do not `.strip()`, normalize newlines/Unicode, load context files, load memory, or resolve skill files during launch.

After construction, compare the effective agent `model`, `provider`, `requested_provider`, `api_mode`, and `base_url` to the frozen runtime and assert no credential pool/fallback/request overrides exist before the task can call `run_conversation`. Constructor drift fails closed; it does not retry generic resolution.

### Prompt/skill contract gap

The observed rt3b object binds `context_bytes` and the tuple `("core-interview", "project-issue-interview")`, but it does not bind the bytes of those skill documents. Loading ambient skills by name during rt3c would violate exact-byte approval and permit post-approval skill drift. Therefore one of these must be made explicit before implementation:

1. Preferred: rt3b's `context_bytes/context_digest` cover the already assembled full bounded prompt, including reviewed immutable copies of both required interview skill bodies; `skills` are attestations and rt3c does not load them.
2. Alternative: the server registry also pins each skill's approved content digest and rt3b assembles/verifies the exact prompt bytes before returning admission.

Names alone are insufficient. Until this is resolved, a provider-message assertion that the required skill instructions are present and digest-covered must remain RED. The missing `BOUNDED-BRIDGE-TRACE.md` may have defined this contract, but it was unavailable.

Also note that current generic prompt assembly prepends Hermes base identity and later combines/strips `ephemeral_system_prompt`. If `context_digest` is intended to cover the **entire provider system-message bytes**, not just the bounded prompt component, a narrow explicit-system-prompt constructor seam is required in `run_agent.py`/`agent_init.py`/prompt assembly; mutating `_cached_system_prompt` after construction is not acceptable. This question must be settled before GREEN.

## Ownership, status, read, and stop

- Bounded creation stamps `_run_owners[run_id]` with the bounded principal/profile scope before any control-visible state is published.
- Bounded GET/stop authenticate against the same bounded registry before any body read (stop should require no body), derive the caller's scope, and return 404 for foreign/unknown runs. Never fall through to `_check_auth` and never let possession of a generic API key control a bounded run.
- Durable status lookup uses the same scope/run id in `RunIdempotencyStore`. In-memory owner absence remains fail-closed unless a durable row proves ownership.
- `GET /v1/runs/{id}` and generic stop/steer/approval must not see/control bounded runs. Conversely bounded GET/stop must not see generic or room runs.
- Stop keeps existing native process ownership/reaping behavior, status transition (`stopping`), retained agent/task refs, and terminal race rules. It must only target the exact bounded run id.
- Do not expose `binding_identity`, context bytes/digest, model API key, bearer token, or full factory spec in public status/events. If model is retained in status, it comes from the approved public runtime field only.
- Do not expose steer or approval endpoints for the bounded lane. Any future bounded follow-up input requires a fresh rt3b admission/idempotency contract.

## Persistence failure and rollback

Current generic status persistence logs and continues, and `RunIdempotencyStore` can become in-memory. That is insufficient for bounded admission.

Required behavior:

- Before admission: non-durable store => 503, no body read/state write.
- Initial reserve exception => 503, no live maps/task, pending reservation released.
- Atomic reserve loses race => discard only local unpublished launch material and replay/conflict.
- Failure after reserve but before task registration => compare-and-delete exactly `(scope,key,fingerprint,run_id)` and remove all published maps. The store needs a conditional rollback method; never delete by key or run id alone.
- If rollback itself fails, fail closed, log only identifiers safe for operations (run id and exception type), do not start work, and leave the durable reservation as a nonterminal owner record so replay cannot duplicate execution.
- `running` persistence failure occurs before model construction/call and must abort execution.
- Terminal persistence failure after model execution cannot be rolled back. Retry the durable update within a small fixed bound; do not emit `run.completed` until persistence succeeds. If it never succeeds, keep the in-memory run failed/degraded, close transport, and leave the last durable nonterminal status; restart logic will conservatively mark it `interrupted`. Never claim durable completion that was not stored.
- Native bound-session creation from rt3b is deterministic and may remain after a later run reservation/capacity failure, but it must contain no messages. A retry reuses it. Transcript writes begin only inside the admitted durable turn.

## Route registration and capabilities

In `APIServerAdapter._http_route_table`, extend a new bounded-runs sibling's route rows. The existing loop will create profile-prefixed mirrors. Add explicit capability endpoint entries for bounded submit/status/stop and a feature object such as:

- `enabled`: bounded registry configured **and** idempotency store durable;
- `auth`: `bounded_bearer`;
- `idempotency.required`: true;
- `idempotency.durable`: true;
- `history`: `native_bound_session`;
- `tools`: `deny_all`;
- `steer`: false;
- `approval`: false.

Do not alter existing `run_submission`, `runs_idempotency`, or generic endpoint values. Disabled/unconfigured bounded support should be absent or explicitly `enabled:false` and return 404/503 according to the agreed exposure policy; it must never downgrade to generic execution.

## Minimum production files

Existing files that must change:

1. `gateway/platforms/api_server.py` — bounded auth/admission decorator plumbing, registry ownership, route-table/capability advertisement, and thin class delegators; also correct task-registration-before-reservation-transfer ordering if kept in the shared path.
2. `gateway/platforms/api_server_runs.py` — shared `_RunLaunch` contract, exact `binding_identity` propagation, native executor/status/control reuse, and generic-behavior preservation.
3. `gateway/platforms/api_server_run_idempotency.py` — conditional exact-reservation rollback and/or strict persistence API required by bounded launch.

Preferred new implementation sibling (keeps the 3,964-line facade and 849-line runs module from growing into another mixed security boundary):

4. `gateway/platforms/api_server_bounded_runs.py` — distinct routes, one-read raw-body cap, scope/fingerprint, bounded factory, strict sequencing, bounded ownership/control.

Existing tests that must change or receive focused additions:

5. `tests/gateway/test_api_server_runs.py` — generic regression plus shared launch/binding/status/ownership behavior.
6. `tests/gateway/test_api_server_active_work_drain.py` — authenticated bounded pending admission, task visibility, drain, and shared concurrency ordering.
7. `tests/gateway/test_api_server_run_idempotency.py` — exact conditional rollback and persistence failures.

Preferred new integration test:

8. `tests/gateway/test_api_server_bounded_runs.py` — end-to-end aiohttp-to-native-run RED suite.

Potential additional production files only if the prompt digest is defined over the full provider system message:

- `run_agent.py`
- `agent/agent_init.py`
- the relevant prompt assembly sibling

Do not modify those merely to set private prompt state; first settle the prompt/skill contract.

The rt3a/rt3b helper files and their tests should remain independently testable. rt3b needs only the small public pre-body authentication split if the accepted version does not provide one.

## Exact RED acceptance cases

Add these as behavior tests, not source-text assertions.

### Seam and generic non-regression

1. `POST /v1/runs` with the same generic payload/model/provider/history/idempotency cases produces the existing status/body/headers and calls generic `_create_agent`; bounded registry/factory is never touched.
2. Adding `case`, a bounded-looking key, or a bounded-looking Authorization header to generic `/v1/runs` does not select bounded execution; it follows existing generic validation/auth behavior.
3. `/v1/capabilities` retains every existing generic run field and advertises distinct bounded endpoints only; multiplex route table contains exact `/p/{profile}` mirrors without collisions.

### Authentication and one-read body

4. Invalid/missing/ambiguous bounded Authorization plus malformed, oversized, or never-ending body returns sanitized 401 before any request-body read, pending-work reservation, SessionDB call, idempotency call, or runtime/factory call.
5. Authenticated declared `Content-Length > 65_536` returns stable 413 without body read or state/idempotency mutation.
6. Authenticated chunked body at exactly 65,536 bytes is passed once as identical bytes to rt3b; 65,537 bytes returns 413 after bounded reading and before rt3b/state/idempotency.
7. Duplicate JSON keys, invalid UTF-8, NaN/Infinity, extra authority fields, and non-object bodies reach rt3b as raw bytes and fail; no prior `request.json()` normalization occurs.

### Durable admission/idempotency/concurrency

8. Non-durable/fallback `RunIdempotencyStore` returns retryable 503 before body read and before rt3b creates/requires a bound session.
9. Same authenticated principal/profile + same key + exact admission fingerprint, sequentially and concurrently, yields one run id, one task, one model call, and replay headers even while capacity is full.
10. Same principal/profile + same key with any changed raw input/case/session binding/context digest/public runtime/required skills/tool policy returns 409 and starts nothing.
11. Same key/fingerprint under another authenticated principal or profile neither replays nor reveals the first run; it creates an independently owned run (subject to capacity).
12. New request at capacity returns 429 without reserve/task; a retry after capacity clears can create. A replay at capacity bypasses 429.
13. Initial reserve exception returns 503 and leaves no owner/status/stream/approval/task/idempotency-id maps and no transcript messages.
14. Failure after reserve before task registration conditionally rolls back only its exact row/maps; a changed fingerprint or replacement row cannot be deleted by stale cleanup.
15. Two concurrent new submissions at the last slot cannot both launch. Same-key requests converge through atomic reserve; different keys obey the shared concurrency limit without oversubscription.

### Launch and factory

16. Captured bounded `_RunLaunch` contains exactly the rt3b user message/session/admission, `conversation_history is None`, and no generic model/provider/history/session overrides.
17. Factory receives exact explicit runtime values, API key only in the constructor call, `fallback_model=None`, `credential_pool=None`, no request overrides, no ambient route/default/model recovery call, no fallback loader, no provider pool, no local model discovery, and no generic `_create_agent` call.
18. Post-construction drift in model/provider/requested-provider/api-mode/base-url, or any credential pool/fallback/request override, fails before `run_conversation` and persists sanitized failure.
19. Agent construction has deny-all, durable history, zero enabled toolsets, skipped context/memory/soul/review, exact server-owned iteration/time/token limits, and the approved prompt component round-trips byte-for-byte.
20. Provider request contains no tool schema/tool-choice/function fields and cannot execute a forged model tool call (existing deny-all tests remain green).

### Prompt, skills, and history

21. The actual first provider request contains the approved digest-covered prompt bytes and both required interview skill instructions; changing an ambient installed skill after rt3b admission cannot change the request. This remains RED until the prompt/skill contract gap is resolved.
22. Body attempts to supply history, instructions, session id, previous response id, model/provider, tools/skills, context, or policy are rejected by rt3b before launch.
23. A forged `conversation_history` cannot be supplied through the integration. `run_conversation` receives `conversation_history=None` and exact `binding_identity`; durable prior messages are loaded under the native session lease before the new user message.
24. Missing/changed/incomplete binding identity, changed durable binding, missing durable SessionDB/lease support, or persistence-disabled agent fails before history/model access.
25. Two bounded runs for the same native session serialize on the durable turn lease and the second sees the first durable result; no stale caller history bypass exists.

### Ownership, control, and persistence

26. Creating bounded principal can read and stop its run through bounded routes. Another bounded principal/profile, a generic API key, room grant, or ownerless in-memory state receives 404 and cannot interrupt it.
27. Bounded GET/stop cannot access generic/room runs; generic GET/stop/steer/approval cannot access bounded runs. Bounded steer/approval routes are absent.
28. Stop interrupts/reaps only the exact run, preserves the stopping-versus-completion race semantics, and terminal cleanup retains owner/status for the normal retention window.
29. `running` persistence failure prevents factory/model execution. Terminal persistence failure suppresses `run.completed`, never claims durable completion, performs bounded retries, closes live transport safely, and restarts as conservative `interrupted` from the last durable nonterminal row.
30. Public responses/status/events/logs/repr never contain bearer/model API keys, raw context, binding identity, context digest, raw request body, or unsanitized exceptions.
31. Pending-work count is nonzero from post-auth/pre-body reservation through task registration, then transfers without a zero-count gap; drain refuses new bounded work and waits for a queued-before-factory bounded task.

## Conclusion

A distinct bounded route family with shared native run internals gives the smallest auditable security boundary and preserves generic API behavior. The launch must be durably reserved before work, visible to drain before reservation transfer, constructed only from the frozen rt3b admission, and invoked with explicit binding identity plus native-history semantics. The two unresolved prerequisites are a public rt3b pre-body authentication seam and a precise definition of whether the context digest covers the full provider system prompt—including immutable required skill content—or only one prompt component.
