# Contributing to jiji

jiji is a personal fork of [niri](https://github.com/niri-wm/niri). For general compositor improvements — bug fixes, new protocols, performance — please send PRs to upstream niri. Those changes benefit everyone and will reach jiji through periodic rebases.

## jiji-specific contributions

The Activities feature and anything else that is jiji-only will not go upstream. Issues and PRs for jiji-specific work are welcome on the project's GitHub repos under the [`jiji-wm`](https://github.com/jiji-wm) organization.

Before opening a PR, skim the architecture notes under `docs/activities/` to understand the current model and what is already planned or explicitly out of scope. Opening an issue first to discuss the approach is appreciated.

## Coding conventions

Fork-specific conventions — error discipline, invariant enforcement, borrow-order rules, commit hygiene — are documented in [CLAUDE.md](CLAUDE.md). The pre-commit hooks enforce the most mechanical rules; the rest are checked in review.

The short version:

- `.unwrap()` is banned outside test code; use `.expect("invariant description")`.
- `verify_invariants` runs at the end of every `Layout::refresh` in debug builds. New cross-field invariants need a matching assertion.
- Keep internal architecture-doc references out of commit messages and source code.
- Commit subject form: `<module>: <imperative short summary>`.
