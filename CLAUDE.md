# niri (gajdusek fork) — coding conventions

Personal fork of niri. Branch: `feature/activities`. Upstream: `niri-wm/niri`.

For the workspace-level DD-driven phase loop and ecosystem tooling, see the parent workspace's `CLAUDE.md` — specifically the "Active work" section that pins the current DD and sub-phase. For upstream's own build/test/architecture reference, see `../upstream/CLAUDE.md`.

This file covers **fork-specific coding conventions** that are enforced in review: error discipline, invariant enforcement, borrow-order, commit hygiene. These are the conventions the `feature/activities` branch has built up over Phases 0–1 and that Phase 1a+ continues to enforce.

## Build & test

- `cargo check --workspace` — fast compile feedback.
- `cargo test --all --exclude niri-visual-tests` — full suite.
  - **Expected pass count: 284 + 21 + 14 + 1.** If this number changes, the active DD must be updated in the same commit.
- `RUN_SLOW_TESTS=1 cargo test random_operations_dont_panic` — proptest corpus (slow; off by default).
- `cargo clippy --all --all-targets` — **baseline: 5 pre-existing warnings at `src/layout/floating.rs:492`.** Any new warning must be fixed before commit.
- `cargo build --release` — sanity-check release profile after any `#[cfg(debug_assertions)]` edit.
- `cargo +nightly fmt --all` — only when explicitly asked; never as a drive-by (it pollutes diffs).

## Code conventions (enforced in review)

### 1. Errors from invariants panic loudly

- `.unwrap()` is banned outside test code. Use `.expect("why this is impossible")` with a message that names the invariant (e.g. `pool.get(&id).expect("view id must be a key in the pool")`).
- Statically unreachable match arms use `unreachable!("<the invariant that guarantees this>")`. Silent `return` or `_ => ()` in such arms is a review-stop bug — precedent at `mod.rs:936` and `mod.rs:4780`.

### 2. Comment hygiene — WHY, not WHAT

- Rustdoc describes the contract. Inline comments explain non-obvious decisions. Bare one-liners that restate the code are deleted on sight.
- **No phase markers in committed source.** `// Phase 0b-2 sub-step 3c` belongs in the DD, not in `mod.rs`. The only exception is a deliberate `// Phase Na:` breadcrumb where the code has a known upcoming reshape; if you add one, mention it in the DD too.

### 3. Invariant enforcement

- Cross-field invariants on `Layout` are checked by `Layout::verify_invariants` and chained through `Workspace` / `ScrollingSpace` / `FloatingSpace` / `Tile` / `Column`.
- `verify_invariants` is `#[cfg(debug_assertions)]` (not `#[cfg(test)]`) and runs at the end of every `Layout::refresh` in debug builds (Phase 1a prerequisite, landed in `5824f13b`). Release builds skip the chain at zero cost.
- Keep invariant checks cheap enough not to regress interactive debug-session latency. Wrap hot paths with `tracy_client::span!` so the cost is measurable (example: `d2ae6cd9`).
- Adding a new cross-field field on `Layout` / `Monitor` / `Workspace` implies adding its invariant assertion in the same commit.

### 4. `#[cfg(debug_assertions)]` vs `#[cfg(test)]`

- `verify_invariants` and its helper accessors are **debug-gated**, not test-gated. Release builds must not see the accessors or any `approx`-using code.
- Test modules themselves stay `#[cfg(test)]`.

### 5. Borrow-order discipline

- When a method needs both `&mut self.workspaces` (the pool) and `&mut self.monitors[i]`, hoist `let pool = &mut self.workspaces;` before the match/destructure. This is the standard borrow-split recipe in this codebase (applied consistently across `Layout::active_workspace_mut` and all action-handler sites in sub-step 3a Part 2).
- NLL-fragile call orders — where reordering breaks the build — are a smell. Split into narrow blocks with explicit shared/exclusive borrow shapes so a future reader can't accidentally break the build.

### 6. Bind/unbind symmetry

- `Workspace::bind_output` only fires `output_enter`; `unbind_output` only fires `output_leave`. Moving a bound workspace between monitors requires `unbind_output(&old)` **then** `bind_output(&new)`.
- The `verify_output_bindings` harness in `src/layout/tests.rs` enforces this after every proptest Op. See the sub-step 2 "Bind/unbind symmetry (caller contract)" paragraph in the DD for the three in-tree transfer scenarios.

## Commit conventions

- Subject form: `<module>: <imperative short summary> (<phase marker>)`. Example: `layout: tracy span + soften block comment around per-refresh verify_invariants (Phase 1a prerequisite follow-up)`.
- Module prefix is the directory the primary change lives in (`layout`, `tests/layout`, `niri-ipc`, etc.).
- Phase marker in parentheses at the end for fork-branch refactor commits; strip before upstream PR. Detailed context goes in the DD, not the commit body.
- Post-review fixes typically **squash** into the commit they fix (`git commit --amend`). The exception: a fix that doesn't belong in the reviewed commit's subject — a regression test pinning the refactor, post-main polish, or a surfaced pre-existing bug — lands as a **separate follow-up commit** instead. The fixer decides based on the *"would the reviewed commit's subject still describe all its changes after squash?"* test. The DD `Reviewed: YYYY-MM-DD (<hash1>, <hash2>, ...)` entry cites all commits covered. If amend changes the hash, update the DD's hash reference in the review-scribing commit.
- Trailers per global `~/CLAUDE.md`. The `<mode>` in `AI-Assisted: <mode> (<model>)` extends to these niri-specific values for fork work:
  - `full-loop` — fork commit that went through `/land-subphase` (architect → implementer → review → fixer).
  - `implementer` — ad-hoc code commit outside the loop (direct `/implement` call, no review step).
  - `scribe` — DD commit in the workspace repo.
  - `one-shot` — any other Claude-touched commit (manual tweak, drive-by).

  Strip `Review-Needed:` before upstream PR; `AI-Assisted:` handling per upstream's policy.

## Architecture (short reference)

Don't duplicate the upstream architecture doc — see `../upstream/CLAUDE.md` for crate structure, build flags, and the calloop-driven single-threaded design.

Fork-specific structural additions the upstream doc doesn't cover:

- **`Layout<W>.workspaces: HashMap<WorkspaceId, Workspace<W>>`** — canonical workspace pool (Phase 0b-2 sub-step 3a Part 1, `ddba4bf8`).
- **`Layout.monitors` / `primary_idx` / `active_monitor_idx` / `disconnected_workspace_ids`** as flat fields on `Layout` (previously wrapped in `MonitorSet::Normal` / `::NoOutputs`; flattened in sub-step 3d, `027ce682`).
- **`Activity.views: HashMap<OutputId, WorkspaceView>`** — per-output ordered `Vec<WorkspaceId>` + navigation methods; for the active activity this is the authoritative storage read via `Layout::active_view(&output_id)` (Phase 1a Part 2, `41df7cbc`).
- **`LayoutCtx<'_, W>`** — borrow-bundle (`&pool`, `&view`) for read-path methods that need both (sub-step 3b, `d7d1402b` / `faef2fe1`).
- **`Workspace::bind_output` / `unbind_output`** split — previously a single `set_output`; split in sub-step 2 to keep `output_enter` / `output_leave` firing disciplines distinct.

For deeper context, see `docs/activities/design.md` in the workspace repo.
