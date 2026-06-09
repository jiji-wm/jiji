# jiji — A scrollable-tiling Wayland compositor (hard fork of niri)

## About

jiji is a hard fork of [niri](https://github.com/niri-wm/niri), periodically rebased against upstream.
It exists primarily as the author's daily-driver compositor and as the vehicle for the Activities feature described below.

jiji is identity-distinct from upstream niri at every surface it controls: it ships as the `jiji` binary, uses `$JIJI_SOCKET`, reads config from `~/.config/jiji/`, and provides the `jiji-ipc` crate rather than `niri-ipc`. Third-party tools written against upstream niri's socket and binary name will not work with jiji without porting.

The fork does not make compatibility promises and is not intended as a general-purpose niri replacement.

## What jiji adds: Activities

jiji implements KDE-style Activities — named, switchable contexts that each own their own ordered set of workspaces and windows. Switching an activity swaps the entire visible workspace set on every connected output, so completely separate workflows (e.g. "Work", "Personal", "Gaming") never bleed into each other.

The workspace-as-atom model means workspaces can belong to multiple activities; windows inherit visibility from their workspace.

Architecture and rationale live under [`docs/activities/`](docs/activities/) in this repo.

Companion CLI: `jiji-activities` — drives activity switching, workspace assignment, and the fuzzy picker. Lives in a sibling repo: [`jiji-wm/jiji-activities`](https://github.com/jiji-wm/jiji-activities).

## Everything else

For documentation on the underlying scrollable-tiling features, configuration, and Wayland protocol support, see [upstream niri's wiki](https://niri-wm.github.io/niri/Getting-Started.html). All of upstream niri's behavior applies unless jiji has explicitly diverged (binary name, socket env var, config directory, and the Activities feature are the current divergences).

## Status

jiji is the author's daily-driver compositor. It is a personal project — not a community fork, not an upstream-replacement. There is no Matrix channel or community Discord for jiji specifically.

For questions about the underlying niri features, upstream's channels are the right place:

- Matrix: https://matrix.to/#/#niri:matrix.org
- Upstream repo: https://github.com/niri-wm/niri

jiji-specific issues belong on the project's GitHub repos under the [`jiji-wm`](https://github.com/jiji-wm) organization.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
