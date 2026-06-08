//! Pins the workspace-miss harmonization contract for the activity-action
//! cohort.
//!
//! Previously, the dispatch arms for `MoveWorkspaceToActivity`,
//! `SetWorkspaceActivities`, and the three sticky helpers
//! (`ToggleWorkspaceSticky`, `SetWorkspaceSticky`, `UnsetWorkspaceSticky`)
//! intercepted a `WorkspaceNotFound` outcome from the layout layer and
//! returned `Ok(())` — silent no-op on the wire. Those intercepts were
//! dropped to harmonize the contract: all five actions now return
//! `Err(DoActionError::{Outer}(InnerError::WorkspaceNotFound))` on a bogus
//! workspace reference, matching `Add` / `Remove`'s long-standing contract.
//!
//! These tests dispatch through `do_action_inner` to pin the wire-visible
//! surface (the keybinding loop silently drops `Err` arms; only the IPC
//! envelope path surfaces them).

use jiji_config::{Action, ActivityReference, WorkspaceReference};

use super::client::ClientId;
use super::fixture::{config_with_two_activities, Fixture};
use crate::layout::{
    DoActionError, DoActionOutcome, FocusWorkspaceInActivityError, LayoutElement,
    MoveWorkspaceToActivityError, SetWorkspaceActivitiesError, SetWorkspaceStickyError,
    SwitchActivityError, ToggleWorkspaceStickyError, UnsetWorkspaceStickyError,
};

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

// --- Plain focus-workspace (no --activity) dispatch coverage ---

#[test]
fn focus_workspace_unknown_name_returns_err() {
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));

    let result = f.niri_state().do_action_inner(
        Action::FocusWorkspace(
            WorkspaceReference::Name("no-such-workspace".to_owned()),
            None,
        ),
        false,
    );

    assert_eq!(
        result,
        Err(DoActionError::FocusWorkspaceTargetUnknown {
            reference: "no-such-workspace".to_owned()
        }),
        "FocusWorkspace with an unknown name and no activity must surface \
         Err(FocusWorkspaceTargetUnknown)",
    );
}

#[test]
fn focus_workspace_unknown_id_returns_err() {
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));

    let result = f.niri_state().do_action_inner(
        Action::FocusWorkspace(WorkspaceReference::Id(BOGUS_WS_ID), None),
        false,
    );

    assert_eq!(
        result,
        Err(DoActionError::FocusWorkspaceTargetUnknown {
            reference: format!("id:{BOGUS_WS_ID}")
        }),
        "FocusWorkspace with a bogus id and no activity must surface \
         Err(FocusWorkspaceTargetUnknown)",
    );
}

#[test]
fn focus_workspace_valid_name_still_focuses() {
    // Positive control: a valid named workspace must still return Ok(Handled)
    // after the dispatch arm reshape. Uses a config-seeded named workspace so
    // the name resolves without creating windows first.
    let mut f = Fixture::with_config(config_with_two_activities(&["my-ws"], &[]));
    f.add_output(1, (1920, 1080));

    let result = f.niri_state().do_action_inner(
        Action::FocusWorkspace(WorkspaceReference::Name("my-ws".to_owned()), None),
        false,
    );

    assert_eq!(
        result,
        Ok(DoActionOutcome::Handled),
        "FocusWorkspace with a valid named workspace must return Ok(Handled)",
    );
}

/// A workspace id that is guaranteed not to resolve in any test fixture.
/// `WorkspaceId` is a monotonic global counter — `u64::MAX` will not be
/// minted before the test exits.
const BOGUS_WS_ID: u64 = u64::MAX;

#[test]
fn focus_workspace_index_out_of_range_clamps_not_errors() {
    // Pins the explicit decision that `WorkspaceReference::Index` keeps its
    // upstream clamp behaviour and never reaches `FocusWorkspaceTargetUnknown`.
    // `find_output_and_workspace_index` returns Some unconditionally for Index
    // (saturating_sub clamp, early return) — this test guards against a future
    // change to that arm silently flipping dispatch to Err.
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));

    let result = f.niri_state().do_action_inner(
        Action::FocusWorkspace(WorkspaceReference::Index(u8::MAX), None),
        false,
    );

    assert_eq!(
        result,
        Ok(DoActionOutcome::Handled),
        "FocusWorkspace with an out-of-range Index must clamp to Ok(Handled), \
         never reach FocusWorkspaceTargetUnknown",
    );
}

// --- move-window-to-workspace Name-miss dispatch coverage ---

#[test]
fn move_window_to_workspace_unknown_name_returns_err() {
    // `MoveWindowToWorkspace` Name-miss: an unknown workspace name must surface
    // as `Err(MoveWindowTargetUnknownName)`. Pre-fix the dispatch silently
    // fell through with no error signal, making typos and stale completions
    // indistinguishable from a successful move.
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();
    map_window(&mut f, client_id, 100, 100);
    f.niri_state().refresh_and_flush_clients();

    let result = f.niri_state().do_action_inner(
        Action::MoveWindowToWorkspace(
            WorkspaceReference::Name("no-such-workspace".to_owned()),
            false,
        ),
        false,
    );

    assert_eq!(
        result,
        Err(DoActionError::MoveWindowTargetUnknownName {
            name: "no-such-workspace".to_owned()
        }),
        "MoveWindowToWorkspace with an unknown name must surface \
         Err(MoveWindowTargetUnknownName)",
    );
}

#[test]
fn move_window_to_workspace_by_id_unknown_name_returns_err() {
    // `MoveWindowToWorkspaceById` Name-miss with an active-view source window.
    // Mirrors the plain-move-window pin above across the `window_id: Some(_)`
    // arm (`move-window --window <id>` IPC path).
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();
    map_window(&mut f, client_id, 100, 100);
    f.niri_state().refresh_and_flush_clients();

    let window_id = {
        let layout = &f.niri().layout;
        let (_out, mapped) = layout
            .windows_all()
            .next()
            .expect("mapped window must be in pool");
        mapped.id().get()
    };

    let result = f.niri_state().do_action_inner(
        Action::MoveWindowToWorkspaceById {
            window_id,
            reference: WorkspaceReference::Name("no-such-workspace".to_owned()),
            focus: false,
        },
        false,
    );

    assert_eq!(
        result,
        Err(DoActionError::MoveWindowTargetUnknownName {
            name: "no-such-workspace".to_owned()
        }),
        "MoveWindowToWorkspaceById with an unknown name and an active-view source \
         must surface Err(MoveWindowTargetUnknownName)",
    );
}

#[test]
fn move_window_to_workspace_by_id_dormant_source_unknown_name_returns_err() {
    // Dormant source + unresolvable Name target via MoveWindowToWorkspaceById.
    // The dormant-source fall-through path was previously only tested with an
    // Index reference (which resolves and yields Ok); this test exercises the
    // same fall-through with an unknown Name, verifying that the new `else`
    // arm below `find_output_and_workspace_index` fires with
    // `Err(MoveWindowTargetUnknownName)` regardless of source dormancy.
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();

    // Map the window under alpha (the default-active activity).
    map_window(&mut f, client_id, 100, 100);
    f.niri_state().refresh_and_flush_clients();

    // Capture the window id while alpha is still active.
    let window_id = {
        let layout = &f.niri().layout;
        let (_out, mapped) = layout
            .windows_all()
            .next()
            .expect("mapped window must be in pool");
        mapped.id().get()
    };

    // Switch to beta. The window is now on a dormant-activity workspace;
    // the `monitors × active-view` walk in `move_to_workspace` cannot
    // reach it from the active (beta) view.
    let beta_id = f
        .niri()
        .layout
        .activities()
        .iter()
        .find(|a| a.name() == "beta")
        .expect("beta must be present in the config-seeded activity pool")
        .id();
    f.niri().layout.switch_activity(beta_id);
    f.niri_state().refresh_and_flush_clients();

    // Dispatch with the dormant-source window id but an unresolvable Name
    // target. The new error path must fire even though the source is dormant.
    let result = f.niri_state().do_action_inner(
        Action::MoveWindowToWorkspaceById {
            window_id,
            reference: WorkspaceReference::Name("no-such-workspace".to_owned()),
            focus: false,
        },
        false,
    );

    assert_eq!(
        result,
        Err(DoActionError::MoveWindowTargetUnknownName {
            name: "no-such-workspace".to_owned()
        }),
        "MoveWindowToWorkspaceById with a dormant source window and an unresolvable \
         Name target must surface Err(MoveWindowTargetUnknownName)",
    );
}

#[test]
fn move_window_to_workspace_index_out_of_range_clamps_not_errors() {
    // Pins the explicit decision that `WorkspaceReference::Index` keeps its
    // upstream clamp behaviour and never reaches `MoveWindowTargetUnknownName`.
    // `find_output_and_workspace_index` returns Some unconditionally for Index
    // (saturating_sub clamp, early return) — this test guards against a future
    // change to that arm silently flipping dispatch to Err.
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();
    map_window(&mut f, client_id, 100, 100);
    f.niri_state().refresh_and_flush_clients();

    let result = f.niri_state().do_action_inner(
        Action::MoveWindowToWorkspace(WorkspaceReference::Index(u8::MAX), false),
        false,
    );

    assert_eq!(
        result,
        Ok(DoActionOutcome::Handled),
        "MoveWindowToWorkspace with an out-of-range Index must clamp to Ok(Handled), \
         never reach MoveWindowTargetUnknownName",
    );
}

#[test]
fn move_window_to_workspace_valid_name_still_moves() {
    // Positive control: a valid named workspace must still return Ok(Handled)
    // after the dispatch arm reshape AND the window must actually land on the
    // target workspace (not just return Handled from a silent self-move).
    // Two config-seeded named workspaces in alpha: the window opens on
    // "source-ws" (first in config, therefore the initial active workspace);
    // we move it to "target-ws" which is a distinct named slot — the self-move
    // short-circuit does not fire.
    let mut f = Fixture::with_config(config_with_two_activities(&["source-ws", "target-ws"], &[]));
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();
    map_window(&mut f, client_id, 100, 100);
    f.niri_state().refresh_and_flush_clients();

    // Capture the window id and its current (source) workspace id before dispatch.
    let (window_id, source_ws_id, target_ws_id) = {
        let layout = &f.niri().layout;
        let (_out, mapped) = layout
            .windows_all()
            .next()
            .expect("mapped window must be in pool");
        let window = LayoutElement::id(mapped);
        let window_id = mapped.id().get();
        let (_out, src_ws) = layout
            .workspaces_all()
            .find(|(_, ws)| ws.has_window(window))
            .expect("window must be on some workspace before dispatch");
        let src_id = src_ws.id();
        let (_out, tgt_ws) = layout
            .workspaces_all()
            .find(|(_, ws)| ws.name().map(|s| s.as_str()) == Some("target-ws"))
            .expect("target-ws must be in pool");
        (window_id, src_id, tgt_ws.id())
    };
    assert_ne!(
        source_ws_id, target_ws_id,
        "source-ws and target-ws must be distinct workspaces for this positive-control test",
    );

    let result = f.niri_state().do_action_inner(
        Action::MoveWindowToWorkspace(WorkspaceReference::Name("target-ws".to_owned()), false),
        false,
    );

    assert_eq!(
        result,
        Ok(DoActionOutcome::Handled),
        "MoveWindowToWorkspace with a valid named workspace must return Ok(Handled)",
    );

    // Verify the window actually landed on target-ws — a silent self-move
    // (or any incorrect no-op) returning Handled would fail here.
    let layout = &f.niri().layout;
    let (_out, mapped) = layout
        .windows_all()
        .find(|(_, m)| m.id().get() == window_id)
        .expect("window must still be in pool after successful move");
    let window = LayoutElement::id(mapped);
    let (_out, current_ws) = layout
        .workspaces_all()
        .find(|(_, ws)| ws.has_window(window))
        .expect("window must be on some workspace after dispatch");
    assert_eq!(
        current_ws.id(),
        target_ws_id,
        "window must have moved to target-ws after a successful MoveWindowToWorkspace",
    );
}

#[test]
fn move_workspace_to_activity_workspace_not_found_returns_err() {
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));

    let beta_id = f
        .niri()
        .layout
        .activities()
        .iter()
        .find(|a| a.name() == "beta")
        .expect("beta must be present in the config-seeded activity pool")
        .id();

    let result = f.niri_state().do_action_inner(
        Action::MoveWorkspaceToActivity(
            Some(WorkspaceReference::Id(BOGUS_WS_ID)),
            ActivityReference::Id(beta_id.get()),
            false,
        ),
        false,
    );

    assert_eq!(
        result,
        Err(DoActionError::MoveWorkspaceToActivity(
            MoveWorkspaceToActivityError::WorkspaceNotFound,
        )),
        "MoveWorkspaceToActivity with a bogus workspace ref must surface \
         Err(MoveWorkspaceToActivity(WorkspaceNotFound))",
    );
}

#[test]
fn set_workspace_activities_workspace_not_found_returns_err() {
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));

    let alpha_id = f.niri().layout.active_activity_id();

    let result = f.niri_state().do_action_inner(
        Action::SetWorkspaceActivities(
            Some(WorkspaceReference::Id(BOGUS_WS_ID)),
            vec![ActivityReference::Id(alpha_id.get())],
        ),
        false,
    );

    assert_eq!(
        result,
        Err(DoActionError::SetWorkspaceActivities(
            SetWorkspaceActivitiesError::WorkspaceNotFound,
        )),
        "SetWorkspaceActivities with a bogus workspace ref must surface \
         Err(SetWorkspaceActivities(WorkspaceNotFound))",
    );
}

#[test]
fn toggle_workspace_sticky_workspace_not_found_returns_err() {
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));

    let result = f.niri_state().do_action_inner(
        Action::ToggleWorkspaceStickyByRef(WorkspaceReference::Id(BOGUS_WS_ID)),
        false,
    );

    assert_eq!(
        result,
        Err(DoActionError::ToggleWorkspaceSticky(
            ToggleWorkspaceStickyError::WorkspaceNotFound,
        )),
        "ToggleWorkspaceStickyByRef with a bogus workspace ref must surface \
         Err(ToggleWorkspaceSticky(WorkspaceNotFound))",
    );
}

#[test]
fn set_workspace_sticky_workspace_not_found_returns_err() {
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));

    let result = f.niri_state().do_action_inner(
        Action::SetWorkspaceStickyByRef(WorkspaceReference::Id(BOGUS_WS_ID)),
        false,
    );

    assert_eq!(
        result,
        Err(DoActionError::SetWorkspaceSticky(
            SetWorkspaceStickyError::WorkspaceNotFound,
        )),
        "SetWorkspaceStickyByRef with a bogus workspace ref must surface \
         Err(SetWorkspaceSticky(WorkspaceNotFound))",
    );
}

#[test]
fn unset_workspace_sticky_workspace_not_found_returns_err() {
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));

    let result = f.niri_state().do_action_inner(
        Action::UnsetWorkspaceStickyByRef(WorkspaceReference::Id(BOGUS_WS_ID)),
        false,
    );

    assert_eq!(
        result,
        Err(DoActionError::UnsetWorkspaceSticky(
            UnsetWorkspaceStickyError::WorkspaceNotFound,
        )),
        "UnsetWorkspaceStickyByRef with a bogus workspace ref must surface \
         Err(UnsetWorkspaceSticky(WorkspaceNotFound))",
    );
}

// --- Positive coverage: valid workspace refs must return Ok(()) ---

#[test]
fn move_workspace_to_activity_valid_ref_returns_ok() {
    // Two named workspaces in alpha so moving one to beta leaves alpha non-empty
    // (the invariant requires every live activity to own at least one workspace).
    let mut f = Fixture::with_config(config_with_two_activities(&["ws1", "ws2"], &[]));
    f.add_output(1, (1920, 1080));

    let active_ws_id = f
        .niri()
        .layout
        .active_workspace()
        .expect("fixture must have an active workspace after add_output")
        .id();
    let beta_id = f
        .niri()
        .layout
        .activities()
        .iter()
        .find(|a| a.name() == "beta")
        .expect("beta must be present in the config-seeded activity pool")
        .id();

    let result = f.niri_state().do_action_inner(
        Action::MoveWorkspaceToActivity(
            Some(WorkspaceReference::Id(active_ws_id.get())),
            ActivityReference::Id(beta_id.get()),
            false,
        ),
        false,
    );

    assert_eq!(
        result,
        Ok(DoActionOutcome::Handled),
        "MoveWorkspaceToActivity with a valid workspace ref must return Ok(Handled)",
    );
}

#[test]
fn set_workspace_activities_valid_ref_returns_ok() {
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));

    let layout = &f.niri().layout;
    let active_ws_id = layout
        .active_workspace()
        .expect("fixture must have an active workspace after add_output")
        .id();
    let alpha_id = layout.active_activity_id();

    let result = f.niri_state().do_action_inner(
        Action::SetWorkspaceActivities(
            Some(WorkspaceReference::Id(active_ws_id.get())),
            vec![ActivityReference::Id(alpha_id.get())],
        ),
        false,
    );

    assert_eq!(
        result,
        Ok(DoActionOutcome::Handled),
        "SetWorkspaceActivities with a valid workspace ref must return Ok(Handled)",
    );
}

#[test]
fn toggle_workspace_sticky_valid_ref_returns_ok() {
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));

    let active_ws_id = f
        .niri()
        .layout
        .active_workspace()
        .expect("fixture must have an active workspace after add_output")
        .id();

    let result = f.niri_state().do_action_inner(
        Action::ToggleWorkspaceStickyByRef(WorkspaceReference::Id(active_ws_id.get())),
        false,
    );

    assert_eq!(
        result,
        Ok(DoActionOutcome::Handled),
        "ToggleWorkspaceStickyByRef with a valid workspace ref must return Ok(Handled)",
    );
}

#[test]
fn set_workspace_sticky_valid_ref_returns_ok() {
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));

    let active_ws_id = f
        .niri()
        .layout
        .active_workspace()
        .expect("fixture must have an active workspace after add_output")
        .id();

    let result = f.niri_state().do_action_inner(
        Action::SetWorkspaceStickyByRef(WorkspaceReference::Id(active_ws_id.get())),
        false,
    );

    assert_eq!(
        result,
        Ok(DoActionOutcome::Handled),
        "SetWorkspaceStickyByRef with a valid workspace ref must return Ok(Handled)",
    );
}

#[test]
fn unset_workspace_sticky_valid_ref_returns_ok() {
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));

    let active_ws_id = f
        .niri()
        .layout
        .active_workspace()
        .expect("fixture must have an active workspace after add_output")
        .id();

    let result = f.niri_state().do_action_inner(
        Action::UnsetWorkspaceStickyByRef(WorkspaceReference::Id(active_ws_id.get())),
        false,
    );

    assert_eq!(
        result,
        Ok(DoActionOutcome::Handled),
        "UnsetWorkspaceStickyByRef with a valid workspace ref must return Ok(Handled)",
    );
}

// --- MoveColumnToWorkspace deliberate silent no-op carve-out ---

#[test]
fn move_column_to_workspace_unknown_name_is_silent_ok() {
    // Pins the deliberate carve-out: MoveColumnToWorkspace has no by-name/by-id
    // IPC consumer (only the Index keybind, which saturating-clamps), so a Name
    // miss keeps the pre-existing silent Ok(Handled) rather than erroring.
    // This test guards against a future harmonization pass silently flipping
    // this arm to Err without an explicit decision.
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();
    map_window(&mut f, client_id, 100, 100);
    f.niri_state().refresh_and_flush_clients();

    let result = f.niri_state().do_action_inner(
        Action::MoveColumnToWorkspace(
            WorkspaceReference::Name("no-such-workspace".to_owned()),
            false,
        ),
        false,
    );

    assert_eq!(
        result,
        Ok(DoActionOutcome::Handled),
        "MoveColumnToWorkspace with an unknown name must remain a silent Ok(Handled) \
         — the carve-out from move-window error behaviour is deliberate",
    );
}

// --- I1: SwitchActivity dispatch coverage ---

#[test]
fn switch_activity_not_found_returns_err() {
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));

    // ActivityId is a monotonic counter — u64::MAX will not be minted before exit.
    let result = f.niri_state().do_action_inner(
        Action::SwitchActivity(ActivityReference::Id(u64::MAX)),
        false,
    );

    assert_eq!(
        result,
        Err(DoActionError::SwitchActivity(SwitchActivityError::NotFound)),
        "SwitchActivity with a bogus activity id must surface \
         Err(SwitchActivity(NotFound))",
    );
}

// --- I1: FocusWorkspace-with-activity dispatch coverage ---

#[test]
fn focus_workspace_in_activity_workspace_not_in_activity_returns_err() {
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));

    // Resolve an existing activity name but supply a bogus workspace id that
    // belongs to no activity — exercises the workspace-resolve fail path.
    let result = f.niri_state().do_action_inner(
        Action::FocusWorkspace(
            WorkspaceReference::Id(BOGUS_WS_ID),
            Some(ActivityReference::Name("alpha".to_owned())),
        ),
        false,
    );

    assert_eq!(
        result,
        Err(DoActionError::FocusWorkspaceInActivity(
            FocusWorkspaceInActivityError::WorkspaceNotInActivity,
        )),
        "FocusWorkspace with a valid activity but a bogus workspace id must surface \
         Err(FocusWorkspaceInActivity(WorkspaceNotInActivity))",
    );
}

#[test]
fn focus_workspace_in_activity_activity_not_found_returns_err() {
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));

    // ActivityId is a monotonic counter — u64::MAX will not be minted before exit.
    let result = f.niri_state().do_action_inner(
        Action::FocusWorkspace(
            WorkspaceReference::Index(1),
            Some(ActivityReference::Id(u64::MAX)),
        ),
        false,
    );

    assert_eq!(
        result,
        Err(DoActionError::SwitchActivity(SwitchActivityError::NotFound)),
        "FocusWorkspace with a bogus activity id must surface \
         Err(SwitchActivity(NotFound))",
    );
}
