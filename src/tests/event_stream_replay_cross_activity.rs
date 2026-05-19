//! Pins the `Event` wire-stream replay contract across an activity-modifying
//! flow.
//!
//! Three layers of assertion ride a single `#[test]`:
//!
//! 1. **Replicate-fidelity** — given a fresh `EventStreamState` seeded with the server's pre-flow
//!    `replicate()` snapshot and then walked through the captured deltas, the resulting state must
//!    equal a fresh state seeded with the server's post-flow `replicate()` snapshot. The wire
//!    deltas faithfully transform the seed into the post-state.
//!
//! 2. **Independent oracle** — the same delta-walked replay state is cross-checked against the live
//!    `Layout` (the source of truth the refresh layer projects from). Activities, workspaces, and
//!    windows in the replay state must match the live layout's keyset.
//!
//! 3. **Emission-ordering pins** — `ActivityCreated` precedes any `WorkspaceOpenedOrChanged` in the
//!    captured stream, and within the cascade tick `ActivityRemoved` precedes `ActivitySwitched`.
//!    These encode the structure-before-state contract documented on `Event::ActivityCreated` /
//!    `Event::ActivitiesChanged`.

use jiji_ipc::state::{EventStreamState, EventStreamStatePart as _};
use jiji_ipc::Event;
use niri_config::{Action, ActivityReference, Config, WorkspaceReference};

use super::client::ClientId;
use super::fixture::Fixture;

/// Drive a full initial-commit → ack-buffer roundtrip so the window is
/// mapped and present in the layout. Mirrors the helpers in
/// `find_window_cross_activity.rs` and `window_opening_cross_activity.rs`.
fn map_window(f: &mut Fixture, id: ClientId, w: u16, h: u16) {
    let window = f.client(id).create_window();
    let surface = window.surface.clone();
    window.commit();
    f.roundtrip(id);

    let window = f.client(id).window(&surface);
    window.attach_new_buffer();
    window.set_size(w, h);
    window.ack_last_and_commit();
    f.roundtrip(id);
}

#[test]
fn event_stream_replay_round_trip_matches_live_state() {
    // Config: alpha (active) + beta, one workspace each. No
    // `open-on-activity` rule on the test window — the window maps onto
    // alpha, so step 4's `SwitchActivity(gamma)` lands on a brand-new
    // empty gamma view (auto-spawned trailing empty workspace). Routing
    // the window directly into gamma would leave gamma's view ending in a
    // non-empty workspace and violate the monitor "trailing empty
    // workspace" invariant. The cross-activity coverage we care about
    // (lifecycle + workspace-activities mutation + active-cursor change +
    // cascade-remove) is preserved either way.
    let kdl = "\
        activity \"alpha\"\n\
        activity \"beta\"\n\
        workspace \"ws_a\" { activity \"alpha\"; }\n\
        workspace \"ws_b\" { activity \"beta\"; }\n\
    ";
    let config = Config::parse_mem(kdl).expect("test KDL must parse");
    let mut f = Fixture::with_config(config);

    f.add_output(1, (1920, 1080));
    f.niri_state().refresh_and_flush_clients();

    // Snapshot the server's event-stream state BEFORE the tap is installed.
    // This is the seed the second fresh state will be initialised from in
    // Layer 1 (so its delta replay starts from the same reference frame the
    // captured deltas were emitted against).
    let pre_flow_replicate: Vec<Event> = f.replicate_event_stream_state();

    // Install the tap AFTER the seed-burst has settled so we capture only
    // the test's flow deltas.
    f.install_event_tap();

    // 1. Create a fresh activity "gamma". Lifecycle event: `ActivityCreated`.
    f.niri_state()
        .do_action_inner(Action::CreateActivity("gamma".to_string()), false)
        .expect("CreateActivity(gamma) must succeed on a unique name");
    f.niri_state().refresh_and_flush_clients();

    // 2. Map a window — lands on alpha's active workspace (the seed). Lifecycle:
    //    `WindowOpenedOrChanged`, plus a `WorkspaceOpenedOrChanged` for the trailing-empty
    //    workspace alpha auto-spawns once its active workspace becomes non-empty.
    let client_id = f.add_client();
    map_window(&mut f, client_id, 100, 100);
    f.niri_state().refresh_and_flush_clients();

    // Capture an existing workspace id for the multi-activity assignment in
    // step 3 — the `ws_b` config-declared workspace lives in beta. We
    // mutate `ws_b` (not `ws_a`) so the ordering invariants of the active
    // monitor's view (gamma's view, after step 4 makes gamma active) are
    // not perturbed by inserting an extra workspace at the end of an
    // already-non-empty view chain.
    let ws_b_id = {
        let layout = &f.niri().layout;
        layout
            .workspaces_all()
            .find(|(_, ws)| ws.name() == Some(&"ws_b".to_owned()))
            .expect("ws_b must exist in the workspace pool from the config")
            .1
            .id()
    };

    // 3. Multi-activity workspace assignment: ws_b now belongs to beta and alpha. Structural delta:
    //    `WorkspaceOpenedOrChanged` (activities field changed). Targets `ws_b` (not `ws_a`) — same
    //    trailing-empty-workspace rationale as the config block.
    f.niri_state()
        .do_action_inner(
            Action::SetWorkspaceActivities(
                Some(WorkspaceReference::Id(ws_b_id.get())),
                vec![
                    ActivityReference::Name("beta".into()),
                    ActivityReference::Name("alpha".into()),
                ],
            ),
            false,
        )
        .expect("SetWorkspaceActivities with valid refs must succeed");
    f.niri_state().refresh_and_flush_clients();

    // 4. Switch active activity to gamma. Lifecycle: `ActivitySwitched`.
    f.niri_state()
        .do_action_inner(
            Action::SwitchActivity(ActivityReference::Name("gamma".into())),
            false,
        )
        .expect("SwitchActivity(gamma) must succeed");
    f.niri_state().refresh_and_flush_clients();

    // 5. Cascade tick: remove the active activity (gamma). The cascade repoints the active cursor;
    //    on the same refresh tick the stream must emit `ActivityRemoved { gamma }` BEFORE
    //    `ActivitySwitched { previous_id: Some(gamma) }`.
    f.niri_state()
        .do_action_inner(
            Action::RemoveActivity(ActivityReference::Name("gamma".into())),
            false,
        )
        .expect("RemoveActivity(gamma) must succeed");
    f.niri_state().refresh_and_flush_clients();

    let captured = f.drain_events();
    assert!(
        !captured.is_empty(),
        "the flow must produce at least one captured event \
         (ActivityCreated for gamma)",
    );

    // === Layer 1: replicate-fidelity ===
    //
    // Build state X = fresh + pre_flow_replicate (snapshot before install)
    //              + captured deltas (during flow).
    // Build state Y = fresh + post_flow_replicate (snapshot after flow).
    // The wire contract says X and Y must agree on the structural maps.
    let post_flow_replicate: Vec<Event> = f.replicate_event_stream_state();

    let mut state_x = EventStreamState::default();
    for ev in pre_flow_replicate.iter().cloned() {
        let _ = state_x.apply(ev);
    }
    for ev in captured.iter().cloned() {
        let _ = state_x.apply(ev);
    }

    let mut state_y = EventStreamState::default();
    for ev in post_flow_replicate.iter().cloned() {
        let _ = state_y.apply(ev);
    }

    assert!(
        !state_y.activities.activities.is_empty(),
        "Layer 1 precondition: post-flow replicate must have non-empty activities map \
         (else equality is vacuous)",
    );
    assert_eq!(
        state_x.activities.activities, state_y.activities.activities,
        "Layer 1: captured deltas must transform pre-flow replicate into the \
         post-flow activities map (replicate-fidelity broken)",
    );
    assert!(
        !state_y.workspaces.workspaces.is_empty(),
        "Layer 1 precondition: post-flow replicate must have non-empty workspaces map \
         (else equality is vacuous)",
    );
    assert_eq!(
        state_x.workspaces.workspaces, state_y.workspaces.workspaces,
        "Layer 1: captured deltas must transform pre-flow replicate into the \
         post-flow workspaces map (replicate-fidelity broken)",
    );
    // `jiji_ipc::Window` does not implement `PartialEq`, so compare keysets
    // here at Layer 1 — Layer 2 cross-checks the keyset against the live
    // layout, which is the stronger property anyway.
    let x_win_ids: std::collections::HashSet<u64> =
        state_x.windows.windows.keys().copied().collect();
    let y_win_ids: std::collections::HashSet<u64> =
        state_y.windows.windows.keys().copied().collect();
    assert_eq!(
        x_win_ids, y_win_ids,
        "Layer 1: captured deltas must transform pre-flow replicate into the \
         post-flow windows keyset (replicate-fidelity broken; keyset proxy \
         since jiji_ipc::Window: !PartialEq)",
    );

    // === Layer 2: independent oracle ===
    //
    // Cross-check `state_x` against the live `Layout` — the projection
    // source the refresh layer reads from. Mismatch here means the wire
    // protocol diverged from layout truth somewhere in the flow.
    let layout = &f.niri().layout;

    // Activities: keyset and the active-flag invariants.
    let live_activity_ids: std::collections::HashSet<u64> =
        layout.activities().iter().map(|a| a.id().get()).collect();
    let replay_activity_ids: std::collections::HashSet<u64> =
        state_x.activities.activities.keys().copied().collect();
    assert_eq!(
        replay_activity_ids, live_activity_ids,
        "Layer 2: activity keyset in replay state must match live layout",
    );

    let active_id = layout.active_activity_id().get();
    let active_entry = state_x
        .activities
        .activities
        .get(&active_id)
        .expect("Layer 2: active activity id must be present in replay state");
    assert!(
        active_entry.is_active,
        "Layer 2: live active activity must be marked is_active=true in replay",
    );
    // Layer 2's `active_count == 1` cardinality pin is the only check that
    // catches a regression flipping `is_active` onto the wrong activity:
    // Layer 1's HashMap equality would pass even if every activity in both
    // replicate snapshots carried `is_active = true`.
    let active_count = state_x
        .activities
        .activities
        .values()
        .filter(|a| a.is_active)
        .count();
    assert_eq!(
        active_count, 1,
        "Layer 2: exactly one activity must carry is_active=true in replay",
    );

    // Workspaces: keyset.
    let live_ws_ids: std::collections::HashSet<u64> = layout
        .workspaces_all()
        .map(|(_, ws)| ws.id().get())
        .collect();
    let replay_ws_ids: std::collections::HashSet<u64> =
        state_x.workspaces.workspaces.keys().copied().collect();
    assert_eq!(
        replay_ws_ids, live_ws_ids,
        "Layer 2: workspace keyset in replay state must match live layout",
    );

    // Windows: keyset. `Mapped::id()` (inherent) returns the `MappedId` that
    // the IPC layer projects as `Window::id`; `LayoutElement::id` is a
    // different "id" (the smithay `Window` handle) and is not the IPC id.
    let mut live_win_ids: std::collections::HashSet<u64> = std::collections::HashSet::new();
    layout.with_windows_all(|win, _, _, _| {
        live_win_ids.insert(win.id().get());
    });
    let replay_win_ids: std::collections::HashSet<u64> =
        state_x.windows.windows.keys().copied().collect();
    assert_eq!(
        replay_win_ids, live_win_ids,
        "Layer 2: window keyset in replay state must match live layout",
    );

    // === Layer 3: emission-ordering pins ===
    //
    // Pin the structure-before-state contract on the captured sequence:
    //   - max(ActivityCreated idx) < min(WorkspaceOpenedOrChanged idx)
    //   - within the cascade tick, ActivityRemoved precedes ActivitySwitched.
    let activity_created_idxs: Vec<usize> = captured
        .iter()
        .enumerate()
        .filter_map(|(i, e)| matches!(e, Event::ActivityCreated { .. }).then_some(i))
        .collect();
    let workspace_opened_idxs: Vec<usize> = captured
        .iter()
        .enumerate()
        .filter_map(|(i, e)| matches!(e, Event::WorkspaceOpenedOrChanged { .. }).then_some(i))
        .collect();
    assert!(
        !activity_created_idxs.is_empty(),
        "Layer 3 precondition: captured stream must contain at least one ActivityCreated \
         (CreateActivity(gamma))",
    );
    assert!(
        !workspace_opened_idxs.is_empty(),
        "Layer 3 precondition: captured stream must contain at least one \
         WorkspaceOpenedOrChanged (window mapped + activity assignment)",
    );
    let max_activity_created = *activity_created_idxs.iter().max().expect("non-empty above");
    let min_workspace_opened = *workspace_opened_idxs.iter().min().expect("non-empty above");
    assert!(
        max_activity_created < min_workspace_opened,
        "Layer 3: every ActivityCreated index ({max_activity_created}) must precede \
         every WorkspaceOpenedOrChanged index ({min_workspace_opened}) — \
         structure-before-state contract on Event::ActivityCreated rustdoc",
    );

    // Within the cascade tick: find the (only) ActivityRemoved for gamma
    // and the (only) ActivitySwitched and assert ordering. Uses positional
    // search rather than per-tick partitioning because the captured buffer
    // does not encode tick boundaries; the cascade emits both events in the
    // same `ipc_refresh_layout` call so positions are dense.
    // gamma was successfully removed, so it's gone from both state_y and
    // live_activity_ids; recover its id from the captured ActivityRemoved event.
    let gamma_id = captured
        .iter()
        .find_map(|e| match e {
            Event::ActivityRemoved { id } => Some(*id),
            _ => None,
        })
        .expect("gamma's id must be derivable from the captured ActivityRemoved event");

    let removed_idx = captured
        .iter()
        .enumerate()
        .find_map(|(i, e)| match e {
            Event::ActivityRemoved { id } if *id == gamma_id => Some(i),
            _ => None,
        })
        .expect("Layer 3 precondition: captured stream must contain ActivityRemoved { gamma }");
    let switched_idx = captured
        .iter()
        .enumerate()
        .find_map(|(i, e)| match e {
            Event::ActivitySwitched {
                previous_id: Some(pid),
                ..
            } if *pid == gamma_id => Some(i),
            _ => None,
        })
        .expect(
            "Layer 3 precondition: captured stream must contain \
             ActivitySwitched { previous_id: Some(gamma) }",
        );
    assert!(
        removed_idx < switched_idx,
        "Layer 3: ActivityRemoved {{ gamma }} (idx {removed_idx}) must precede \
         ActivitySwitched {{ previous_id: Some(gamma) }} (idx {switched_idx}) — \
         lifecycle-before-active-cursor on the cascade tick",
    );
}
