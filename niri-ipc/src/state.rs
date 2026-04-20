//! Helpers for keeping track of the event stream state.
//!
//! 1. Create an [`EventStreamState`] using `Default::default()`, or any individual state part if
//!    you only care about part of the state.
//! 2. Connect to the niri socket and request an event stream.
//! 3. Pass every [`Event`] to [`EventStreamStatePart::apply`] on your state.
//! 4. Read the fields of the state as needed.

use std::collections::hash_map::Entry;
use std::collections::HashMap;

use crate::{Cast, Event, KeyboardLayouts, Window, Workspace};

/// Part of the state communicated via the event stream.
pub trait EventStreamStatePart {
    /// Returns a sequence of events that replicates this state from default initialization.
    fn replicate(&self) -> Vec<Event>;

    /// Applies the event to this state.
    ///
    /// Returns `None` after applying the event, and `Some(event)` if the event is ignored by this
    /// part of the state.
    fn apply(&mut self, event: Event) -> Option<Event>;
}

/// The full state communicated over the event stream.
///
/// Different parts of the state are not guaranteed to be consistent across every single event
/// sent by niri. For example, you may receive the first [`Event::WindowOpenedOrChanged`] for a
/// just-opened window *after* an [`Event::WorkspaceActiveWindowChanged`] for that window. Between
/// these two events, the workspace active window id refers to a window that does not yet exist in
/// the windows state part.
#[derive(Debug, Default)]
pub struct EventStreamState {
    /// State of workspaces.
    pub workspaces: WorkspacesState,

    /// State of workspaces.
    pub windows: WindowsState,

    /// State of the keyboard layouts.
    pub keyboard_layouts: KeyboardLayoutsState,

    /// State of the overview.
    pub overview: OverviewState,

    /// State of the config.
    pub config: ConfigState,

    /// State of screencasts.
    pub casts: CastsState,
}

/// The workspaces state communicated over the event stream.
#[derive(Debug, Default)]
pub struct WorkspacesState {
    /// Map from a workspace id to the workspace.
    pub workspaces: HashMap<u64, Workspace>,
}

/// The windows state communicated over the event stream.
#[derive(Debug, Default)]
pub struct WindowsState {
    /// Map from a window id to the window.
    pub windows: HashMap<u64, Window>,
}

/// The keyboard layout state communicated over the event stream.
#[derive(Debug, Default)]
pub struct KeyboardLayoutsState {
    /// Configured keyboard layouts.
    pub keyboard_layouts: Option<KeyboardLayouts>,
}

/// The overview state communicated over the event stream.
#[derive(Debug, Default)]
pub struct OverviewState {
    /// Whether the overview is currently open.
    pub is_open: bool,
}

/// The config state communicated over the event stream.
#[derive(Debug, Default)]
pub struct ConfigState {
    /// Whether the last config load attempt had failed.
    pub failed: bool,
}

/// The casts state communicated over the event stream.
#[derive(Debug, Default)]
pub struct CastsState {
    /// Map from a stream id to the screencast.
    pub casts: HashMap<u64, Cast>,
}

impl EventStreamStatePart for EventStreamState {
    fn replicate(&self) -> Vec<Event> {
        let mut events = Vec::new();
        events.extend(self.workspaces.replicate());
        events.extend(self.windows.replicate());
        events.extend(self.keyboard_layouts.replicate());
        events.extend(self.overview.replicate());
        events.extend(self.config.replicate());
        events.extend(self.casts.replicate());
        events
    }

    fn apply(&mut self, event: Event) -> Option<Event> {
        let event = self.workspaces.apply(event)?;
        let event = self.windows.apply(event)?;
        let event = self.keyboard_layouts.apply(event)?;
        let event = self.overview.apply(event)?;
        let event = self.config.apply(event)?;
        let event = self.casts.apply(event)?;
        Some(event)
    }
}

impl EventStreamStatePart for WorkspacesState {
    fn replicate(&self) -> Vec<Event> {
        let workspaces: Vec<_> = self.workspaces.values().cloned().collect();

        // Emit per-workspace deltas first, then the full-list event for
        // backwards-compatible consumers. Consumers that handle the delta events should
        // ignore `WorkspacesChanged`.
        let mut events: Vec<Event> = workspaces
            .iter()
            .cloned()
            .map(|workspace| Event::WorkspaceOpenedOrChanged { workspace })
            .collect();
        events.push(Event::WorkspacesChanged { workspaces });
        events
    }

    fn apply(&mut self, event: Event) -> Option<Event> {
        match event {
            Event::WorkspacesChanged { workspaces } => {
                self.workspaces = workspaces.into_iter().map(|ws| (ws.id, ws)).collect();
            }
            Event::WorkspaceOpenedOrChanged { workspace } => {
                self.workspaces.insert(workspace.id, workspace);
            }
            Event::WorkspaceClosed { id } => {
                // Tolerant of unknown ids (e.g. a late-connecting client that
                // never saw the corresponding WorkspaceOpenedOrChanged). The
                // matching WindowClosed path intentionally panics instead.
                self.workspaces.remove(&id);
            }
            Event::WorkspaceUrgencyChanged { id, urgent } => {
                for ws in self.workspaces.values_mut() {
                    if ws.id == id {
                        ws.is_urgent = urgent;
                    }
                }
            }
            Event::WorkspaceActivated { id, focused } => {
                let ws = self.workspaces.get(&id);
                let ws = ws.expect("activated workspace was missing from the map");
                let output = ws.output.clone();

                for ws in self.workspaces.values_mut() {
                    let got_activated = ws.id == id;
                    if ws.output == output {
                        ws.is_active = got_activated;
                    }

                    if focused {
                        ws.is_focused = got_activated;
                    }
                }
            }
            Event::WorkspaceActiveWindowChanged {
                workspace_id,
                active_window_id,
            } => {
                let ws = self.workspaces.get_mut(&workspace_id);
                let ws = ws.expect("changed workspace was missing from the map");
                ws.active_window_id = active_window_id;
            }
            event => return Some(event),
        }
        None
    }
}

impl EventStreamStatePart for WindowsState {
    fn replicate(&self) -> Vec<Event> {
        let windows = self.windows.values().cloned().collect();
        vec![Event::WindowsChanged { windows }]
    }

    fn apply(&mut self, event: Event) -> Option<Event> {
        match event {
            Event::WindowsChanged { windows } => {
                self.windows = windows.into_iter().map(|win| (win.id, win)).collect();
            }
            Event::WindowOpenedOrChanged { window } => {
                let (id, is_focused) = match self.windows.entry(window.id) {
                    Entry::Occupied(mut entry) => {
                        let entry = entry.get_mut();
                        *entry = window;
                        (entry.id, entry.is_focused)
                    }
                    Entry::Vacant(entry) => {
                        let entry = entry.insert(window);
                        (entry.id, entry.is_focused)
                    }
                };

                if is_focused {
                    for win in self.windows.values_mut() {
                        if win.id != id {
                            win.is_focused = false;
                        }
                    }
                }
            }
            Event::WindowClosed { id } => {
                let win = self.windows.remove(&id);
                win.expect("closed window was missing from the map");
            }
            Event::WindowFocusChanged { id } => {
                for win in self.windows.values_mut() {
                    win.is_focused = Some(win.id) == id;
                }
            }
            Event::WindowFocusTimestampChanged {
                id,
                focus_timestamp,
            } => {
                for win in self.windows.values_mut() {
                    if win.id == id {
                        win.focus_timestamp = focus_timestamp;
                        break;
                    }
                }
            }
            Event::WindowUrgencyChanged { id, urgent } => {
                for win in self.windows.values_mut() {
                    if win.id == id {
                        win.is_urgent = urgent;
                        break;
                    }
                }
            }
            Event::WindowLayoutsChanged { changes } => {
                for (id, update) in changes {
                    let win = self.windows.get_mut(&id);
                    let win = win.expect("changed window was missing from the map");
                    win.layout = update;
                }
            }
            event => return Some(event),
        }
        None
    }
}

impl EventStreamStatePart for KeyboardLayoutsState {
    fn replicate(&self) -> Vec<Event> {
        if let Some(keyboard_layouts) = self.keyboard_layouts.clone() {
            vec![Event::KeyboardLayoutsChanged { keyboard_layouts }]
        } else {
            vec![]
        }
    }

    fn apply(&mut self, event: Event) -> Option<Event> {
        match event {
            Event::KeyboardLayoutsChanged { keyboard_layouts } => {
                self.keyboard_layouts = Some(keyboard_layouts);
            }
            Event::KeyboardLayoutSwitched { idx } => {
                let kb = self.keyboard_layouts.as_mut();
                let kb = kb.expect("keyboard layouts must be set before a layout can be switched");
                kb.current_idx = idx;
            }
            event => return Some(event),
        }
        None
    }
}

impl EventStreamStatePart for OverviewState {
    fn replicate(&self) -> Vec<Event> {
        vec![Event::OverviewOpenedOrClosed {
            is_open: self.is_open,
        }]
    }

    fn apply(&mut self, event: Event) -> Option<Event> {
        match event {
            Event::OverviewOpenedOrClosed { is_open } => {
                self.is_open = is_open;
            }
            event => return Some(event),
        }
        None
    }
}

impl EventStreamStatePart for ConfigState {
    fn replicate(&self) -> Vec<Event> {
        vec![Event::ConfigLoaded {
            failed: self.failed,
        }]
    }

    fn apply(&mut self, event: Event) -> Option<Event> {
        match event {
            Event::ConfigLoaded { failed } => {
                self.failed = failed;
            }
            event => return Some(event),
        }
        None
    }
}

impl EventStreamStatePart for CastsState {
    fn replicate(&self) -> Vec<Event> {
        let casts = self.casts.values().cloned().collect();
        vec![Event::CastsChanged { casts }]
    }

    fn apply(&mut self, event: Event) -> Option<Event> {
        match event {
            Event::CastsChanged { casts } => {
                self.casts = casts.into_iter().map(|c| (c.stream_id, c)).collect();
            }
            Event::CastStartedOrChanged { cast } => {
                self.casts.insert(cast.stream_id, cast);
            }
            Event::CastStopped { stream_id } => {
                let cast = self.casts.remove(&stream_id);
                cast.expect("stopped cast was missing from the map");
            }
            event => return Some(event),
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ws(id: u64, idx: u8, output: &str) -> Workspace {
        Workspace {
            id,
            idx,
            name: None,
            output: Some(output.to_owned()),
            is_urgent: false,
            is_active: false,
            is_focused: false,
            active_window_id: None,
            activities: vec![],
            is_sticky: false,
            is_in_active_activity: true,
        }
    }

    #[test]
    fn workspace_opened_or_changed_inserts_new() {
        let mut state = WorkspacesState::default();
        assert!(state
            .apply(Event::WorkspaceOpenedOrChanged {
                workspace: ws(1, 1, "HDMI-1")
            })
            .is_none());
        assert_eq!(state.workspaces.len(), 1);
        assert_eq!(state.workspaces[&1].idx, 1);
    }

    #[test]
    fn workspace_opened_or_changed_updates_existing() {
        let mut state = WorkspacesState::default();
        state.apply(Event::WorkspaceOpenedOrChanged {
            workspace: ws(1, 1, "HDMI-1"),
        });
        // Same id, different idx — should replace.
        state.apply(Event::WorkspaceOpenedOrChanged {
            workspace: ws(1, 3, "HDMI-1"),
        });
        assert_eq!(state.workspaces.len(), 1);
        assert_eq!(state.workspaces[&1].idx, 3);
    }

    #[test]
    fn workspace_closed_removes() {
        let mut state = WorkspacesState::default();
        state.apply(Event::WorkspaceOpenedOrChanged {
            workspace: ws(1, 1, "HDMI-1"),
        });
        state.apply(Event::WorkspaceOpenedOrChanged {
            workspace: ws(2, 2, "HDMI-1"),
        });
        assert_eq!(state.workspaces.len(), 2);

        assert!(state.apply(Event::WorkspaceClosed { id: 1 }).is_none());
        assert_eq!(state.workspaces.len(), 1);
        assert!(!state.workspaces.contains_key(&1));
    }

    #[test]
    fn workspace_closed_for_missing_id_is_noop() {
        // Closed events for ids we never saw must be tolerated (e.g. a
        // client that connected after the workspace was already gone).
        let mut state = WorkspacesState::default();
        assert!(state.apply(Event::WorkspaceClosed { id: 99 }).is_none());
        assert!(state.workspaces.is_empty());
    }

    #[test]
    fn replicate_emits_deltas_then_full_list() {
        let mut source = WorkspacesState::default();
        source.apply(Event::WorkspaceOpenedOrChanged {
            workspace: ws(1, 1, "HDMI-1"),
        });
        source.apply(Event::WorkspaceOpenedOrChanged {
            workspace: ws(2, 2, "DP-1"),
        });

        let events = source.replicate();
        assert_eq!(events.len(), 3); // 2 deltas + 1 full list

        // All deltas come first; the full-list event is last.
        let (last, deltas) = events.split_last().unwrap();
        for event in deltas {
            assert!(matches!(event, Event::WorkspaceOpenedOrChanged { .. }));
        }
        assert!(matches!(last, Event::WorkspacesChanged { .. }));
    }

    #[test]
    fn delta_events_alone_reconstruct_full_state() {
        // Consumers that ignore WorkspacesChanged and only process
        // WorkspaceOpenedOrChanged / WorkspaceClosed still get the correct view
        // of the world.
        let mut source = WorkspacesState::default();
        source.apply(Event::WorkspaceOpenedOrChanged {
            workspace: ws(1, 1, "HDMI-1"),
        });
        source.apply(Event::WorkspaceOpenedOrChanged {
            workspace: ws(2, 2, "DP-1"),
        });

        let mut replica = WorkspacesState::default();
        for event in source.replicate() {
            if matches!(event, Event::WorkspacesChanged { .. }) {
                continue;
            }
            replica.apply(event);
        }
        assert_eq!(replica.workspaces, source.workspaces);
    }

    #[test]
    fn full_list_alone_reconstructs_full_state() {
        // Backwards compatibility: a client that only processes the legacy
        // WorkspacesChanged event still ends up with the right state.
        let mut source = WorkspacesState::default();
        source.apply(Event::WorkspaceOpenedOrChanged {
            workspace: ws(1, 1, "HDMI-1"),
        });
        source.apply(Event::WorkspaceOpenedOrChanged {
            workspace: ws(2, 2, "DP-1"),
        });

        let mut replica = WorkspacesState::default();
        for event in source.replicate() {
            if !matches!(event, Event::WorkspacesChanged { .. }) {
                continue;
            }
            replica.apply(event);
        }
        assert_eq!(replica.workspaces, source.workspaces);
    }

    #[test]
    fn applying_both_paths_is_idempotent() {
        // Dual-event-path consumers (§4.6): ones that process both the deltas
        // and the full-list event end up with the correct state regardless.
        let mut source = WorkspacesState::default();
        source.apply(Event::WorkspaceOpenedOrChanged {
            workspace: ws(1, 1, "HDMI-1"),
        });
        source.apply(Event::WorkspaceOpenedOrChanged {
            workspace: ws(2, 2, "DP-1"),
        });

        let mut replica = WorkspacesState::default();
        for event in source.replicate() {
            replica.apply(event);
        }
        assert_eq!(replica.workspaces, source.workspaces);
    }
}
