# jiji — A scrollable-tiling Wayland compositor (hard fork of niri)

## About

jiji is a hard fork of [niri](https://github.com/niri-wm/niri), periodically rebased against upstream.
It exists primarily as the author's daily-driver compositor and as the vehicle for the Activities feature described below.

jiji is identity-distinct from upstream niri at every surface it controls: it ships as the `jiji` binary, uses `$JIJI_SOCKET`, reads config from `~/.config/jiji/`, and provides the `jiji-ipc` crate rather than `niri-ipc`. Third-party tools written against upstream niri's socket and binary name will not work with jiji without porting.

The fork does not make compatibility promises and is not intended as a general-purpose niri replacement.

## What jiji adds

### Activities

jiji implements KDE-style Activities — named, switchable contexts that each own their own ordered set of workspaces and windows. Switching an activity swaps the entire visible workspace set on every connected output, so completely separate workflows (e.g. "Work", "Personal", "Gaming") never bleed into each other.

The workspace-as-atom model means workspaces can belong to multiple activities; windows inherit visibility from their workspace. Every activity keeps at least one empty workspace per output, `focus-workspace` can be scoped to an activity, and activity switches have their own configurable animation.

Companion CLI: `jiji-activities` — drives activity switching, workspace assignment, and the fuzzy picker. Lives in a sibling repo: [`jiji-wm/jiji-activities`](https://github.com/jiji-wm/jiji-activities).

### Bookmarks

Windows and workspaces can be bookmarked and recalled by key. The subsystem covers:

- a leader-key **bookmark mode** (`enter-bookmark-mode`) with per-bookmark key assignment, interactive key capture, and hotkey-overlay integration, so assigned keys show up in the compositor's built-in key hints;
- **bookmark walk** — cycle forward/backward through bookmarks, re-entering at the last-visited one;
- a **bookmark switcher** UI, jump/move/rename/remove actions, and rule-based bookmarks that anchor to whatever window currently matches an app-id/title regex.

### Viewport paging

`move-view-left` / `move-view-right` page the scrolling view by the visible viewport while preserving which column slot has focus — single-screen paging through a long strip of columns instead of column-by-column scrolling.

### Workspace insertion

`add-workspace-up` / `add-workspace-down` insert a fresh workspace adjacent to the focused one, and `move-window-to-new-workspace-up` / `-down` carry the focused window into it — no round-trip through the workspace list's tail.

### IPC additions

The `jiji-ipc` crate carries the surface backing all of the above (activities, bookmarks, view paging, workspace insertion) plus quality-of-life changes: explicit no-op/error responses where upstream silently ignores unreachable targets (e.g. focusing an unknown workspace name or moving a window to a workspace in another activity), `id:N` workspace references, per-window application tags, and an activity-aware event stream for bars and external tools.

## Everything else

For documentation on the underlying scrollable-tiling features, configuration, and Wayland protocol support, see [upstream niri's wiki](https://niri-wm.github.io/niri/Getting-Started.html). All of upstream niri's behavior applies unless jiji has explicitly diverged (binary name, socket env var, config directory, and the features above are the current divergences). A vendored snapshot of the upstream wiki is kept under [`docs/wiki/`](docs/wiki/) for rebase reference — it documents upstream niri, not jiji.

## Building

Same toolchain and system dependencies as upstream niri (see the [upstream build docs](https://github.com/niri-wm/niri/wiki/Getting-Started)):

```
cargo build --release
cargo test --all --exclude jiji-visual-tests
```

The binary is `target/release/jiji`; session files (`jiji.desktop`, `jiji.service`, `jiji-portals.conf`, `jiji-session`) live in `resources/`. Nix and RPM packaging from upstream are not maintained in this fork.

## Companion projects

The jiji ecosystem lives in sibling repos under the [`jiji-wm`](https://github.com/jiji-wm) organization:

- [`jiji-activities`](https://github.com/jiji-wm/jiji-activities) — CLI for activity switching, workspace assignment, and pickers.
- [`jiji-do`](https://github.com/jiji-wm/jiji-do) — Helix-style command launcher exposing compositor and activity verbs through fuzzel.
- [`jiji-waybar`](https://github.com/jiji-wm/jiji-waybar) — Waybar fork with activity-aware `jiji/*` modules and an activities indicator/picker.
- [`jiji-firefox-workspaces`](https://github.com/jiji-wm/jiji-firefox-workspaces) — restores Firefox windows to their workspaces across sessions (WebExtension + native-messaging host).
- [`jiji-hamster`](https://github.com/jiji-wm/jiji-hamster) / [`jiji-hamster-bridge`](https://github.com/jiji-wm/jiji-hamster-bridge) — Hamster time-tracker fork and a daemon that pauses/resumes tracking based on jiji activity focus.

## Status

jiji is the author's daily-driver compositor. It is a personal project — not a community fork, not an upstream-replacement. There is no Matrix channel or community Discord for jiji specifically.

For questions about the underlying niri features, upstream's channels are the right place:

- Matrix: https://matrix.to/#/#niri:matrix.org
- Upstream repo: https://github.com/niri-wm/niri

jiji-specific issues belong on the project's GitHub repos under the [`jiji-wm`](https://github.com/jiji-wm) organization.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
