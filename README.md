# jiji — a scrollable-tiling Wayland compositor with Activities

## About

jiji is a scrollable-tiling Wayland compositor built around **Activities** — named, switchable desktop contexts that each own their own ordered set of workspaces and windows. It ships as the `jiji` binary, uses `$JIJI_SOCKET`, reads config from `~/.config/jiji/`, and provides the `jiji-ipc` crate.

jiji is an independent project. It began as a fork of the excellent [niri](https://github.com/niri-wm/niri) compositor and periodically rebases on it to inherit upstream improvements — see [Credits](#credits) — but it has its own identity, IPC surface, tooling ecosystem, and release cadence, and is not a drop-in niri replacement: tools written against niri's socket and binary name need porting.

It is primarily the author's daily-driver compositor; expect a personal project's pace and priorities.

## Features

### Activities

KDE-style Activities — named, switchable contexts that each own their own ordered set of workspaces and windows. Switching an activity swaps the entire visible workspace set on every connected output, so completely separate workflows (e.g. "Work", "Personal", "Gaming") never bleed into each other.

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

### IPC

The `jiji-ipc` crate carries the surface backing all of the above (activities, bookmarks, view paging, workspace insertion) plus quality-of-life changes: explicit no-op/error responses where a target is unreachable (e.g. focusing an unknown workspace name or moving a window to a workspace in another activity), `id:N` workspace references, per-window application tags, and an activity-aware event stream for bars and external tools.

### Scrollable tiling

Underneath the jiji-specific features sits the full scrollable-tiling model jiji inherited and keeps rebasing on: windows arranged in columns on an infinite horizontal strip per workspace, dynamic workspaces per monitor, built-in screenshot UI, overview, monitor hot-plug handling, fractional scaling, and wide Wayland protocol support.

## Documentation

Reference documentation lives under [`docs/wiki/`](docs/wiki/) in this repo. It is inherited from the pre-fork codebase and is being adapted; where a page still says `niri`, `~/.config/niri/`, or `$NIRI_SOCKET`, read `jiji`, `~/.config/jiji/`, and `$JIJI_SOCKET`. The default config with inline documentation is [`resources/default-config.kdl`](resources/default-config.kdl).

## Building

```
cargo build --release
cargo test --all --exclude jiji-visual-tests
```

The binary is `target/release/jiji`; session files (`jiji.desktop`, `jiji.service`, `jiji-portals.conf`, `jiji-session`) live in `resources/`. Build dependencies match what [`.github/workflows/ci.yml`](.github/workflows/ci.yml) installs. Nix and RPM packaging are not maintained.

## Bugs and questions

jiji issues — including issues in the inherited tiling behavior as it behaves *in jiji* — belong on this repo's [issue tracker](https://github.com/jiji-wm/jiji/issues) or the other repos under the [`jiji-wm`](https://github.com/jiji-wm) organization. Please don't take jiji problems to the niri project; they rightly won't support a fork.

## Companion projects

The jiji ecosystem lives in sibling repos under the [`jiji-wm`](https://github.com/jiji-wm) organization:

- [`jiji-activities`](https://github.com/jiji-wm/jiji-activities) — CLI for activity switching, workspace assignment, and pickers.
- [`jiji-do`](https://github.com/jiji-wm/jiji-do) — Helix-style command launcher exposing compositor and activity verbs through fuzzel.
- [`jiji-waybar`](https://github.com/jiji-wm/jiji-waybar) — Waybar fork with activity-aware `jiji/*` modules and an activities indicator/picker.
- [`jiji-firefox-workspaces`](https://github.com/jiji-wm/jiji-firefox-workspaces) — restores Firefox windows to their workspaces across sessions (WebExtension + native-messaging host).
- [`jiji-hamster`](https://github.com/jiji-wm/jiji-hamster) / [`jiji-hamster-bridge`](https://github.com/jiji-wm/jiji-hamster-bridge) — Hamster time-tracker fork and a daemon that pauses/resumes tracking based on jiji activity focus.

The whole ecosystem — cross-repo docs, build/install scripts, and the multi-agent development loop — is coordinated from the [jiji-workspace repo](https://github.com/jiji-wm/jiji-workspace).

## Credits

jiji stands on the shoulders of [niri](https://github.com/niri-wm/niri) by Ivan Molodetskikh ([@YaLTeR](https://github.com/YaLTeR)) and its contributors. The scrollable-tiling core, the Smithay-based architecture, and much of what makes jiji pleasant to use every day is their work — thank you. jiji periodically rebases on niri so upstream fixes and features keep flowing in; if you want the original, actively supported compositor with a real community, use niri.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
