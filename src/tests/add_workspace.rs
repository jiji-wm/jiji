//! Pins the `Action::AddWorkspaceDown` / `Action::AddWorkspaceUp` dispatch
//! path through `do_action_inner`.
//!
//! Each action must return `Ok(DoActionOutcome::Handled)` and focus a
//! workspace whose id was not in the pre-dispatch id set. A third test checks
//! that a fresh workspace left empty is pruned once animations complete.

use std::collections::HashSet;

use jiji_config::Action;

use super::client::ClientId;
use super::fixture::{config_with_two_activities, Fixture};
use crate::layout::DoActionOutcome;

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

/// Collect the set of workspace ids currently visible in the active view
/// of the first connected monitor.
fn active_view_ws_ids(f: &mut Fixture) -> HashSet<u64> {
    let layout = &f.niri().layout;
    let mon = layout
        .monitors()
        .next()
        .expect("fixture must have at least one monitor");
    let view = layout.active_view(&mon.output_id());
    view.ids().iter().map(|id| id.get()).collect()
}

#[test]
fn add_workspace_down_dispatch_focuses_fresh_empty_workspace() {
    // Two populated workspaces. Dispatch AddWorkspaceDown from the first:
    // a fresh empty workspace is inserted between it and the second. The
    // active workspace id after dispatch must not appear in the pre-dispatch
    // id set, and the active workspace must have no windows.
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();

    // Map window on workspace 0.
    map_window(&mut f, client_id, 100, 100);
    f.niri_state().refresh_and_flush_clients();

    // Switch to a new workspace and map a second window there.
    f.niri().layout.switch_workspace_down();
    map_window(&mut f, client_id, 100, 100);
    f.niri_state().refresh_and_flush_clients();

    // Return to the first workspace.
    f.niri().layout.switch_workspace_up();
    f.niri_state().refresh_and_flush_clients();

    let pre_ws_ids = active_view_ws_ids(&mut f);

    let result = f
        .niri_state()
        .do_action_inner(Action::AddWorkspaceDown, false);

    assert_eq!(
        result,
        Ok(DoActionOutcome::Handled),
        "AddWorkspaceDown must return Ok(Handled)",
    );
    f.niri_state().refresh_and_flush_clients();

    let layout = &f.niri().layout;
    let active_ws = layout
        .active_workspace()
        .expect("must have an active workspace after dispatch");
    let post_ws_id = active_ws.id().get();

    assert!(
        !pre_ws_ids.contains(&post_ws_id),
        "active workspace (id {post_ws_id}) must be freshly minted \
         (not in pre-dispatch set {pre_ws_ids:?})",
    );
    assert!(
        !active_ws.has_windows_or_name(),
        "the focused workspace after AddWorkspaceDown must be empty and unnamed",
    );
}

#[test]
fn add_workspace_up_dispatch_focuses_fresh_empty_workspace() {
    // Single populated workspace. Dispatch AddWorkspaceUp from it: a fresh
    // empty workspace is inserted above. The active workspace id must not
    // appear in the pre-dispatch id set and must have no windows.
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();

    map_window(&mut f, client_id, 100, 100);
    f.niri_state().refresh_and_flush_clients();

    let pre_ws_ids = active_view_ws_ids(&mut f);

    let result = f
        .niri_state()
        .do_action_inner(Action::AddWorkspaceUp, false);

    assert_eq!(
        result,
        Ok(DoActionOutcome::Handled),
        "AddWorkspaceUp must return Ok(Handled)",
    );
    f.niri_state().refresh_and_flush_clients();

    let layout = &f.niri().layout;
    let active_ws = layout
        .active_workspace()
        .expect("must have an active workspace after dispatch");
    let post_ws_id = active_ws.id().get();

    assert!(
        !pre_ws_ids.contains(&post_ws_id),
        "active workspace (id {post_ws_id}) must be freshly minted \
         (not in pre-dispatch set {pre_ws_ids:?})",
    );
    assert!(
        !active_ws.has_windows_or_name(),
        "the focused workspace after AddWorkspaceUp must be empty and unnamed",
    );
}

#[test]
fn add_workspace_left_empty_prunes_after_focus_leaves() {
    // Dispatch AddWorkspaceDown, then switch away without populating the fresh
    // workspace. After animations complete the fresh workspace must no longer
    // appear in the active view.
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();

    map_window(&mut f, client_id, 100, 100);
    f.niri_state().refresh_and_flush_clients();

    // Switch down and map a second window so the view has two content slots.
    f.niri().layout.switch_workspace_down();
    map_window(&mut f, client_id, 100, 100);
    f.niri_state().refresh_and_flush_clients();
    f.niri().layout.switch_workspace_up();
    f.niri_state().refresh_and_flush_clients();

    // Dispatch AddWorkspaceDown to insert a fresh empty workspace at slot 1.
    let result = f
        .niri_state()
        .do_action_inner(Action::AddWorkspaceDown, false);
    assert_eq!(result, Ok(DoActionOutcome::Handled));

    // Capture the fresh id.
    let fresh_ws_id = f
        .niri()
        .layout
        .active_workspace()
        .expect("must have an active workspace")
        .id()
        .get();

    // Switch away without populating it.
    f.niri().layout.switch_workspace_down();
    f.niri_state().refresh_and_flush_clients();

    // Complete animations to trigger the cleanup hook.
    f.niri_complete_animations();
    f.niri_state().refresh_and_flush_clients();

    let post_ws_ids = active_view_ws_ids(&mut f);
    assert!(
        !post_ws_ids.contains(&fresh_ws_id),
        "fresh empty workspace (id {fresh_ws_id}) must be pruned after focus leaves; \
         remaining ids: {post_ws_ids:?}",
    );
}
