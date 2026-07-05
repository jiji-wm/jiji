# Contributing to jiji

Issues and PRs are welcome on the project's GitHub repos under the [`jiji-wm`](https://github.com/jiji-wm) organization.

One ask, in the spirit of being a good downstream: if your fix applies equally to [niri](https://github.com/niri-wm/niri) (a bug in the shared tiling core, a new protocol, a performance win), consider sending it upstream as well — it benefits both projects, and jiji picks it up through periodic rebases.

Before opening a PR for feature work, skim the feature overview in the [README](README.md) to understand the current model. Opening an issue first to discuss the approach is appreciated.

## Coding conventions

Fork-specific conventions — error discipline, invariant enforcement, borrow-order rules, commit hygiene — are documented in [CLAUDE.md](CLAUDE.md). The pre-commit hooks enforce the most mechanical rules; the rest are checked in review.

The short version:

- `.unwrap()` is banned outside test code; use `.expect("invariant description")`.
- `verify_invariants` runs at the end of every `Layout::refresh` in debug builds. New cross-field invariants need a matching assertion.
- Keep internal architecture-doc references out of commit messages and source code.
- Commit subject form: `<module>: <imperative short summary>`.
