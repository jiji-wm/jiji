# Contributing to jiji

Issues and PRs are welcome on the project's GitHub repos under the [`jiji-wm`](https://github.com/jiji-wm) organization.

One ask, in the spirit of being a good downstream: if your fix applies equally to [niri](https://github.com/niri-wm/niri) (a bug in the shared tiling core, a new protocol, a performance win), consider sending it upstream as well — it benefits both projects, and jiji picks it up through periodic rebases.

Before opening a PR for feature work, skim the feature overview in the [README](README.md) to understand the current model. Opening an issue first to discuss the approach is appreciated.

## Coding conventions

Fork-specific conventions — error discipline, invariant enforcement, borrow-order rules, commit hygiene — are documented in [CLAUDE.md](CLAUDE.md). The most mechanical rules (rustfmt, no internal design-doc references) are enforced by versioned git hooks that live in the [jiji-workspace](https://github.com/jiji-wm/jiji-workspace) repo's shared `.githooks/` and are wired in automatically by that repo's `scripts/clone.sh` (optional: CI and review enforce the same rules). If you're working from a standalone clone of this repo rather than the full workspace, those local hooks aren't available — CI and review still catch the same issues; the rest is checked in review.

The short version:

- `.unwrap()` is banned outside test code; use `.expect("invariant description")`.
- `verify_invariants` runs at the end of every `Layout::refresh` in debug builds. New cross-field invariants need a matching assertion.
- Keep internal architecture-doc references out of commit messages and source code.
- Commit subject form: `<module>: <imperative short summary>`.

The quality bar is the existing code. Match the conventions and rigor of the code around your change — you won't be held to a higher standard than the codebase itself meets, and its current limitations are fair game to point out, or fix.

## AI-assisted contributions

AI-assisted PRs are welcome. jiji itself is developed with heavy AI assistance under a structured plan → implement → review → fix loop — the tooling and process docs live in the [jiji-workspace repo](https://github.com/jiji-wm/jiji-workspace), and you are welcome to use them yourself — so this project will not reject your contribution for how it was written, only for whether it holds up.

What we ask:

- **You are the author and accountable for every line.** Submit only changes you have read, understood, and can defend in review.
- **Verify before you submit.** Build it, run the test suite (`cargo test --all --exclude jiji-visual-tests`), and exercise the changed behavior in a running compositor when it has a runtime surface. "The model said it works" is not verification.
- **Disclose substantial AI involvement with a commit trailer.** Use the Linux-kernel convention `Assisted-by: AGENT:MODEL`; this project's own commits carry the equivalent house trailer `AI-Assisted: <mode> (<model-id>)`, which is also accepted. Trivial assistance (autocomplete-level) needs no disclosure.
- **Follow the same conventions as any other PR** — the coding rules above, commit-message hygiene, one coherent change per commit.
- **Planning something bigger?** You're always welcome to open an issue and chat about it first — I'm happy to point at what's already planned or in flight and help find the approach most likely to land. Optional, but it can spare you wasted effort: when a big PR dies, it's usually not for code quality but for things visible up front — the feature was already designed differently, collides with an invariant you couldn't see from outside, or sits outside the project's scope. An issue is also a great place to announce what you're building: it prevents duplicate work and can draw in ideas, interest, and collaborators.
- **You must have the right to contribute the code** under this project's license — don't submit verbatim reproductions of incompatible-licensed code, whatever produced them.

If a contribution looks unreviewed by its author, expect it to be returned with a request to review it first rather than a detailed code review; if there's no follow-up after a while, it may be closed — reopening with the review done is always welcome.
