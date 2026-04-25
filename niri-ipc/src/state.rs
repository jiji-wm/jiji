//! Helpers for keeping track of the event stream state.
//!
//! 1. Create an [`EventStreamState`] using `Default::default()`, or any individual state part if
//!    you only care about part of the state.
//! 2. Connect to the niri socket and request an event stream.
//! 3. Pass every [`Event`] to [`EventStreamStatePart::apply`] on your state.
//! 4. Read the fields of the state as needed.

use std::collections::hash_map::Entry;
use std::collections::HashMap;

use crate::{Activity, Cast, Event, KeyboardLayouts, Window, Workspace};

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
    /// State of activities.
    ///
    /// Placed before [`Self::workspaces`] to mirror the structure-before-state
    /// emission order established by `ipc_refresh_layout`: activity lifecycle
    /// events fire before any per-workspace event whose `is_in_active_activity`
    /// might flip, so clients that apply events in stream order always see the
    /// activity layer settled before the workspace layer reacts. See the
    /// `Event::ActivityCreated` rustdoc in [`crate`] for the normative
    /// statement.
    pub activities: ActivitiesState,

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

/// The activities state communicated over the event stream.
#[derive(Debug, Default)]
pub struct ActivitiesState {
    /// Map from an activity id to the activity.
    pub activities: HashMap<u64, Activity>,
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
        // Field ordering is load-bearing for initial burst and the
        // structure-before-state emission contract; see rustdoc on Self::activities.
        let mut events = Vec::new();
        events.extend(self.activities.replicate());
        events.extend(self.workspaces.replicate());
        events.extend(self.windows.replicate());
        events.extend(self.keyboard_layouts.replicate());
        events.extend(self.overview.replicate());
        events.extend(self.config.replicate());
        events.extend(self.casts.replicate());
        events
    }

    fn apply(&mut self, event: Event) -> Option<Event> {
        // Field ordering is load-bearing for initial burst and the
        // structure-before-state emission contract; see rustdoc on Self::activities.
        let event = self.activities.apply(event)?;
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
        // backwards-compatible consumers.
        // Consumers that handle the delta events should ignore `WorkspacesChanged`.
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

impl EventStreamStatePart for ActivitiesState {
    fn replicate(&self) -> Vec<Event> {
        // Unlike `WorkspacesState` which replays per-workspace deltas *and*
        // the full-list event, activities replicate as a single
        // `ActivitiesChanged` snapshot. Per-activity `ActivityCreated` events
        // are reserved for true lifecycle transitions (runtime creation,
        // config-reload additions): emitting them on every stream-replicate
        // tick would invite clients to treat every reconnect as a create
        // burst. Bulk state lands via `ActivitiesChanged` (see the rustdoc on
        // `Event::ActivitiesChanged`).
        let activities = self.activities.values().cloned().collect();
        vec![Event::ActivitiesChanged { activities }]
    }

    fn apply(&mut self, event: Event) -> Option<Event> {
        match event {
            Event::ActivitiesChanged { activities } => {
                // Tolerant full replacement, mirrors `WorkspacesChanged`.
                self.activities = activities.into_iter().map(|a| (a.id, a)).collect();
            }
            Event::ActivityCreated { activity } => {
                // Tolerant of id re-insertion (mirrors
                // `WindowOpenedOrChanged`'s `Entry::Occupied` overwrite):
                // a late-connecting client may apply the initial
                // `ActivitiesChanged` payload and then see a subsequent
                // `ActivityCreated` for an id already present; overwrite
                // rather than panic.
                //
                // Cross-entry flip on `is_active`: per the `Event::ActivityCreated`
                // rustdoc, a create-and-activate in one tick carries
                // `is_active: true` in the payload and emits no separate
                // `ActivitySwitched`. To honor that contract, when the new
                // payload is active we must clear `is_active` on every other
                // entry — otherwise a client that missed only the final
                // `ActivitySwitched` would keep a stale `is_active=true` on
                // the old cursor forever. Mirrors the `WindowOpenedOrChanged`
                // focus-flip shape above.
                let id = activity.id;
                let promoted_active = activity.is_active;
                self.activities.insert(id, activity);
                if promoted_active {
                    for a in self.activities.values_mut() {
                        if a.id != id {
                            a.is_active = false;
                        }
                    }
                }
            }
            Event::ActivityRemoved { id } => {
                // Tolerant of unknown ids (e.g. a late-connecting client that
                // never saw the corresponding `ActivityCreated`). Matches
                // `WorkspaceClosed`'s tolerance posture.
                self.activities.remove(&id);
            }
            Event::ActivityRenamed { id, name } => {
                let a = self.activities.get_mut(&id);
                // Strict: rename follows a seen Created per lifecycle ordering.
                let a = a.expect("renamed activity was missing from the map");
                a.name = name;
            }
            Event::ActivitySwitched { id, previous_id: _ } => {
                // `previous_id` is purely informational (derived from `id`
                // alone); ignore it and drive the flip from `id` only.
                //
                // Tolerant of an unknown `id`: when `id` is absent from the
                // map (e.g. a client that hasn't seen the corresponding
                // `ActivityCreated`), skip the flip entirely so the previous
                // `is_active` signal is preserved rather than clearing all
                // entries to `false`.
                if !self.activities.contains_key(&id) {
                    return None;
                }
                for a in self.activities.values_mut() {
                    a.is_active = a.id == id;
                }
            }
            Event::ActivityUrgencyChanged { id, urgent } => {
                // Tolerant for-loop mirrors `WorkspaceUrgencyChanged`.
                for a in self.activities.values_mut() {
                    if a.id == id {
                        a.is_urgent = urgent;
                    }
                }
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
        // consumers that ignore WorkspacesChanged and only
        // process WorkspaceOpenedOrChanged / WorkspaceClosed still get the
        // correct view of the world.
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
        // Dual-event-path consumers: ones that process both the deltas
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

    fn activity(
        id: u64,
        name: &str,
        is_active: bool,
        is_urgent: bool,
        is_config_declared: bool,
    ) -> Activity {
        Activity {
            id,
            name: name.to_owned(),
            is_config_declared,
            is_active,
            is_urgent,
        }
    }

    #[test]
    fn activities_state_activities_changed_replaces_map() {
        let mut state = ActivitiesState::default();
        state.apply(Event::ActivityCreated {
            activity: activity(1, "old", true, false, false),
        });
        // Full replacement: prior ids must be dropped, new ids installed.
        assert!(state
            .apply(Event::ActivitiesChanged {
                activities: vec![
                    activity(2, "a", true, false, false),
                    activity(3, "b", false, true, true),
                ],
            })
            .is_none());
        assert_eq!(state.activities.len(), 2);
        assert!(!state.activities.contains_key(&1));
        assert_eq!(state.activities[&2].name, "a");
        assert!(state.activities[&2].is_active);
        assert!(state.activities[&3].is_urgent);
    }

    #[test]
    fn activities_state_activity_created_inserts_new() {
        let mut state = ActivitiesState::default();
        assert!(state
            .apply(Event::ActivityCreated {
                activity: activity(7, "Work", false, false, false),
            })
            .is_none());
        assert_eq!(state.activities.len(), 1);
        assert_eq!(state.activities[&7].name, "Work");
    }

    #[test]
    fn activities_state_activity_created_overwrites_existing_id() {
        // Tolerant of id re-insertion (mirrors `WindowOpenedOrChanged`'s
        // `Entry::Occupied` precedent): a client that applied an earlier
        // `ActivitiesChanged` containing the same id and then receives
        // `ActivityCreated` for that id overwrites rather than panicking.
        // The cross-entry `is_active` flip covers the re-insertion path
        // too: seed entry 1 (active), then re-insert 7 with active:true
        // must also clear 1's is_active.
        let mut state = ActivitiesState::default();
        state.apply(Event::ActivityCreated {
            activity: activity(1, "Other", true, false, false),
        });
        state.apply(Event::ActivityCreated {
            activity: activity(7, "Old", false, false, false),
        });
        state.apply(Event::ActivityCreated {
            activity: activity(7, "New", true, true, true),
        });
        assert_eq!(state.activities.len(), 2);
        assert_eq!(state.activities[&7].name, "New");
        assert!(state.activities[&7].is_active);
        assert!(state.activities[&7].is_urgent);
        assert!(state.activities[&7].is_config_declared);
        assert!(
            !state.activities[&1].is_active,
            "overwrite with is_active:true must also clear is_active on other entries",
        );
    }

    #[test]
    fn activities_state_activity_created_with_is_active_clears_others() {
        // Cross-entry flip contract: an `ActivityCreated` whose payload is
        // `is_active: true` (create-and-activate in one tick) must clear
        // `is_active` on every other entry in the map. Without this, a
        // client that receives `ActivityCreated{is_active:true}` and no
        // trailing `ActivitySwitched` keeps the old cursor's stale
        // `is_active=true` forever. Mirrors `WindowOpenedOrChanged`'s
        // focus-flip shape. Pins the interaction between
        // `EventStreamState::activities` and the server's emission
        // contract in `Event::ActivityCreated`'s rustdoc.
        let mut state = ActivitiesState::default();
        state.apply(Event::ActivityCreated {
            activity: activity(1, "alpha", true, false, false),
        });
        assert!(state.activities[&1].is_active);
        state.apply(Event::ActivityCreated {
            activity: activity(2, "beta", true, false, false),
        });
        assert!(
            !state.activities[&1].is_active,
            "alpha's is_active must be cleared when beta is created as active",
        );
        assert!(state.activities[&2].is_active);
    }

    #[test]
    fn activities_state_activity_created_with_is_active_false_leaves_others_alone() {
        // Create-only (no active promotion): existing is_active must be
        // preserved. The cross-entry flip is gated on the payload's
        // `is_active: true`.
        let mut state = ActivitiesState::default();
        state.apply(Event::ActivityCreated {
            activity: activity(1, "alpha", true, false, false),
        });
        state.apply(Event::ActivityCreated {
            activity: activity(2, "beta", false, false, false),
        });
        assert!(
            state.activities[&1].is_active,
            "creating beta non-active must leave alpha's is_active intact",
        );
        assert!(!state.activities[&2].is_active);
    }

    #[test]
    fn activities_state_activity_removed_removes() {
        let mut state = ActivitiesState::default();
        state.apply(Event::ActivityCreated {
            activity: activity(7, "Work", false, false, false),
        });
        assert!(state.apply(Event::ActivityRemoved { id: 7 }).is_none());
        assert!(state.activities.is_empty());
    }

    #[test]
    fn activities_state_activity_removed_for_missing_id_is_noop() {
        // `WorkspaceClosed`-style tolerance: a client that connected after the
        // activity was already gone silently drops the event.
        let mut state = ActivitiesState::default();
        assert!(state.apply(Event::ActivityRemoved { id: 99 }).is_none());
        assert!(state.activities.is_empty());
    }

    #[test]
    fn activities_state_activity_renamed_mutates_name() {
        let mut state = ActivitiesState::default();
        state.apply(Event::ActivityCreated {
            activity: activity(7, "Work", false, false, false),
        });
        assert!(state
            .apply(Event::ActivityRenamed {
                id: 7,
                name: "Office".to_owned(),
            })
            .is_none());
        assert_eq!(state.activities[&7].name, "Office");
    }

    #[test]
    #[should_panic(expected = "renamed activity was missing from the map")]
    fn activities_state_activity_renamed_for_missing_id_panics() {
        // Strict posture mirrors `WorkspaceActiveWindowChanged`'s `expect`: a
        // rename for an id the client never saw is a protocol violation on
        // the server side, not a tolerable race.
        let mut state = ActivitiesState::default();
        state.apply(Event::ActivityRenamed {
            id: 7,
            name: "Office".to_owned(),
        });
    }

    #[test]
    fn activities_state_activity_switched_flips_is_active_two_entries() {
        // Two-entry flip: the old active goes from true→false and the new
        // active goes from false→true. Driven by `id`, not `previous_id`.
        let mut state = ActivitiesState::default();
        state.apply(Event::ActivityCreated {
            activity: activity(1, "A", true, false, false),
        });
        state.apply(Event::ActivityCreated {
            activity: activity(2, "B", false, false, false),
        });
        assert!(state
            .apply(Event::ActivitySwitched {
                id: 2,
                previous_id: Some(1),
            })
            .is_none());
        assert!(!state.activities[&1].is_active);
        assert!(state.activities[&2].is_active);
    }

    #[test]
    fn activities_state_activity_urgency_changed_mutates_is_urgent() {
        let mut state = ActivitiesState::default();
        state.apply(Event::ActivityCreated {
            activity: activity(7, "Work", false, false, false),
        });
        assert!(state
            .apply(Event::ActivityUrgencyChanged {
                id: 7,
                urgent: true,
            })
            .is_none());
        assert!(state.activities[&7].is_urgent);
        state.apply(Event::ActivityUrgencyChanged {
            id: 7,
            urgent: false,
        });
        assert!(!state.activities[&7].is_urgent);
    }

    #[test]
    fn activities_state_replicate_emits_activities_changed_only() {
        // Unlike `WorkspacesState::replicate`, we do NOT emit per-activity
        // `ActivityCreated` events during replicate — those are reserved for
        // true lifecycle transitions. A client reconnecting should see a
        // single `ActivitiesChanged` carrying the full snapshot, not a flood
        // of synthetic "creates".
        let mut source = ActivitiesState::default();
        source.apply(Event::ActivityCreated {
            activity: activity(1, "a", true, false, false),
        });
        source.apply(Event::ActivityCreated {
            activity: activity(2, "b", false, true, false),
        });

        let events = source.replicate();
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], Event::ActivitiesChanged { .. }));
    }

    #[test]
    fn activities_state_replicate_roundtrips_to_equivalent_state() {
        // Apply-replicate invariance: `replica.apply(source.replicate())`
        // yields a map equal to `source.activities`.
        let mut source = ActivitiesState::default();
        source.apply(Event::ActivityCreated {
            activity: activity(1, "a", true, false, false),
        });
        source.apply(Event::ActivityCreated {
            activity: activity(2, "b", false, true, true),
        });

        let mut replica = ActivitiesState::default();
        for event in source.replicate() {
            replica.apply(event);
        }
        assert_eq!(replica.activities, source.activities);
    }

    #[test]
    fn activities_state_applying_activity_switched_then_activities_changed_preserves_final_state() {
        // Dual-path consumer equivalence (mirrors `applying_both_paths_is_idempotent`):
        // a client that applies both the incremental `ActivitySwitched` and
        // the subsequent `ActivitiesChanged` ends up in the same final state
        // as one that applies only `ActivitiesChanged`.
        let mut dual = ActivitiesState::default();
        dual.apply(Event::ActivityCreated {
            activity: activity(1, "A", true, false, false),
        });
        dual.apply(Event::ActivityCreated {
            activity: activity(2, "B", false, false, false),
        });
        dual.apply(Event::ActivitySwitched {
            id: 2,
            previous_id: Some(1),
        });
        dual.apply(Event::ActivitiesChanged {
            activities: vec![
                activity(1, "A", false, false, false),
                activity(2, "B", true, false, false),
            ],
        });

        let mut only_changed = ActivitiesState::default();
        only_changed.apply(Event::ActivitiesChanged {
            activities: vec![
                activity(1, "A", false, false, false),
                activity(2, "B", true, false, false),
            ],
        });

        assert_eq!(dual.activities, only_changed.activities);
    }

    #[test]
    fn activities_state_activity_switched_with_unknown_previous_id_is_tolerated() {
        // `previous_id` is purely informational; an unknown value must not
        // cause a panic or corrupt state. The flip proceeds normally driven
        // by `id` alone.
        let mut state = ActivitiesState::default();
        state.apply(Event::ActivityCreated {
            activity: activity(1, "A", true, false, false),
        });
        state.apply(Event::ActivityCreated {
            activity: activity(2, "B", false, false, false),
        });
        assert!(state
            .apply(Event::ActivitySwitched {
                id: 2,
                previous_id: Some(999), // unknown previous
            })
            .is_none());
        assert!(!state.activities[&1].is_active);
        assert!(state.activities[&2].is_active);
    }

    #[test]
    fn activities_state_activity_urgency_changed_for_missing_id_is_noop() {
        // Tolerant for-loop: an `ActivityUrgencyChanged` for an id that is
        // absent from the map (e.g. a late-connecting client that never saw
        // the corresponding `ActivityCreated`) is silently ignored. Mirrors
        // `activities_state_activity_removed_for_missing_id_is_noop`.
        let mut state = ActivitiesState::default();
        assert!(state
            .apply(Event::ActivityUrgencyChanged {
                id: 99,
                urgent: true,
            })
            .is_none());
        assert!(state.activities.is_empty());
    }

    #[test]
    fn activities_state_activity_switched_with_unknown_id_preserves_prior_actives() {
        // Tolerance for unknown `id`: when the new active id is absent from
        // the map, skip the flip entirely so the previous `is_active` signal
        // is preserved rather than clearing all entries to `false`. A
        // subsequent `ActivitiesChanged` will repair state.
        let mut state = ActivitiesState::default();
        state.apply(Event::ActivityCreated {
            activity: activity(1, "A", true, false, false),
        });
        state.apply(Event::ActivityCreated {
            activity: activity(2, "B", false, false, false),
        });
        // Switch to unknown id 999 — prior actives must survive intact.
        assert!(state
            .apply(Event::ActivitySwitched {
                id: 999,
                previous_id: Some(1),
            })
            .is_none());
        assert!(
            state.activities[&1].is_active,
            "prior is_active must be preserved when switch target is unknown",
        );
        assert!(!state.activities[&2].is_active);
    }
}
