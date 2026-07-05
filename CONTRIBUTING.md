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

The quality bar is the existing code. Match the conventions and rigor of the code around your change — you won't be held to a higher standard than the codebase itself meets, and its current limitations are fair game to point out, or fix.

## AI-assisted contributions

AI-assisted PRs are welcome. jiji itself is developed with heavy AI assistance under a structured plan → implement → review → fix loop, so this project will not reject your contribution for how it was written — only for whether it holds up.

What we ask:

- **You are the author and accountable for every line.** Submit only changes you have read, understood, and can defend in review.
- **Verify before you submit.** Build it, run the test suite (`cargo test --all --exclude jiji-visual-tests`), and exercise the changed behavior in a running compositor when it has a runtime surface. "The model said it works" is not verification.
- **Disclose substantial AI involvement with a commit trailer.** Use the Linux-kernel convention `Assisted-by: AGENT:MODEL`; this project's own commits carry the equivalent house trailer `AI-Assisted: <mode> (<model-id>)`, which is also accepted. Trivial assistance (autocomplete-level) needs no disclosure.
- **Follow the same conventions as any other PR** — the coding rules above, commit-message hygiene, one coherent change per commit. For larger or architectural work, open an issue first; AI makes it cheap to produce big diffs, and unsolicited big diffs are expensive to review.
- **You must have the right to contribute the code** under this project's license — don't submit verbatim reproductions of incompatible-licensed code, whatever produced them.

If a contribution looks unreviewed by its author, expect it to be returned with a request to review it first rather than a detailed code review; if there's no follow-up after a while, it may be closed — reopening with the review done is always welcome.
