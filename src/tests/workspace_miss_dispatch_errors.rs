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

use niri_config::{Action, ActivityReference, WorkspaceReference};

use super::fixture::{config_with_two_activities, Fixture};
use crate::layout::{
    DoActionError, MoveWorkspaceToActivityError, SetWorkspaceActivitiesError,
    SetWorkspaceStickyError, SwitchActivityError, ToggleWorkspaceStickyError,
    UnsetWorkspaceStickyError,
};

/// A workspace id that is guaranteed not to resolve in any test fixture.
/// `WorkspaceId` is a monotonic global counter — `u64::MAX` will not be
/// minted before the test exits.
const BOGUS_WS_ID: u64 = u64::MAX;

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
        Ok(()),
        "MoveWorkspaceToActivity with a valid workspace ref must return Ok(())",
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
        Ok(()),
        "SetWorkspaceActivities with a valid workspace ref must return Ok(())",
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
        Ok(()),
        "ToggleWorkspaceStickyByRef with a valid workspace ref must return Ok(())",
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
        Ok(()),
        "SetWorkspaceStickyByRef with a valid workspace ref must return Ok(())",
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
        Ok(()),
        "UnsetWorkspaceStickyByRef with a valid workspace ref must return Ok(())",
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
