//! Pins the `Action::MoveWindowToNewWorkspaceDown` /
//! `Action::MoveWindowToNewWorkspaceUp` dispatch path through
//! `do_action_inner`.
//!
//! Each action must return `Ok(DoActionOutcome::Handled)` and place the
//! window on a workspace that did not exist before the dispatch. A third
//! test checks that the emptied source workspace is pruned once
//! `refresh_and_flush_clients` advances the animations.

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
fn move_window_to_new_workspace_down_dispatch_lands_on_fresh_adjacent_workspace() {
    // Two windows on two separate workspaces. Dispatch MoveWindowToNewWorkspaceDown
    // from the first workspace: this inserts a new workspace between it and the
    // second workspace (the trailing bookend is NOT adjacent to the source, so
    // the reuse arm does not fire). The window must land on a workspace whose id
    // was not in the pre-dispatch id set.
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();

    // Map first window on workspace 0.
    map_window(&mut f, client_id, 100, 100);
    f.niri_state().refresh_and_flush_clients();

    // Switch to a new workspace and map a second window there.
    f.niri().layout.switch_workspace_down();
    map_window(&mut f, client_id, 100, 100);
    f.niri_state().refresh_and_flush_clients();

    // Focus back to the first workspace (holds window 1).
    f.niri().layout.switch_workspace_up();
    f.niri_state().refresh_and_flush_clients();

    // Capture the pre-dispatch view's workspace ids.
    let pre_ws_ids = active_view_ws_ids(&mut f);

    let result = f
        .niri_state()
        .do_action_inner(Action::MoveWindowToNewWorkspaceDown(true), false);

    assert_eq!(
        result,
        Ok(DoActionOutcome::Handled),
        "MoveWindowToNewWorkspaceDown must return Ok(Handled)",
    );
    f.niri_state().refresh_and_flush_clients();

    // The window must now live on a workspace that did not exist before dispatch.
    let layout = &f.niri().layout;
    // Find the window that was originally on workspace 0 (the focused one before dispatch).
    // After dispatch with focus=true, the layout's active workspace holds that window.
    let active_ws = layout
        .active_workspace()
        .expect("must have an active workspace after dispatch");
    let post_ws_id = active_ws.id().get();

    assert!(
        !pre_ws_ids.contains(&post_ws_id),
        "window must land on a freshly minted workspace (id {post_ws_id} must not be \
         in the pre-dispatch id set {pre_ws_ids:?})",
    );
}

#[test]
fn move_window_to_new_workspace_up_dispatch_lands_on_fresh_adjacent_workspace() {
    // Same as the down test; exercises the up direction.
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();
    map_window(&mut f, client_id, 100, 100);
    f.niri_state().refresh_and_flush_clients();

    let pre_ws_ids = active_view_ws_ids(&mut f);

    let result = f
        .niri_state()
        .do_action_inner(Action::MoveWindowToNewWorkspaceUp(true), false);

    assert_eq!(
        result,
        Ok(DoActionOutcome::Handled),
        "MoveWindowToNewWorkspaceUp must return Ok(Handled)",
    );
    f.niri_state().refresh_and_flush_clients();

    let layout = &f.niri().layout;
    let (_, mapped) = layout
        .windows_all()
        .next()
        .expect("window must still be in pool after dispatch");
    let win = mapped.window.clone();
    let post_ws_id = layout
        .workspaces_all()
        .find(|(_, ws)| ws.windows().any(|w| w.window == win))
        .expect("window must be on some workspace after dispatch")
        .1
        .id()
        .get();

    assert!(
        !pre_ws_ids.contains(&post_ws_id),
        "window must land on a freshly minted workspace (id {post_ws_id} must not be \
         in the pre-dispatch id set {pre_ws_ids:?})",
    );
}

#[test]
fn move_window_to_new_workspace_emptied_source_prunes_and_view_order_updates() {
    // After the action and completed animations, the emptied source workspace
    // must no longer appear in the active view (pruned by the animation-finish
    // cleanup hook in `advance_animations`).
    //
    // Setup mirrors the down-dispatch test: two windows on two workspaces, focus
    // on the first. After move-down, the source (ws0) becomes empty and must be
    // cleaned up once the workspace-switch animation completes.
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();

    // Map first window on workspace 0.
    map_window(&mut f, client_id, 100, 100);
    f.niri_state().refresh_and_flush_clients();

    // Switch to a new workspace and map a second window there.
    f.niri().layout.switch_workspace_down();
    map_window(&mut f, client_id, 100, 100);
    f.niri_state().refresh_and_flush_clients();

    // Focus back to the first workspace.
    f.niri().layout.switch_workspace_up();
    f.niri_state().refresh_and_flush_clients();

    // Capture source workspace id before the dispatch.
    let source_ws_id = {
        let layout = &f.niri().layout;
        layout
            .active_workspace()
            .expect("fixture must have an active workspace")
            .id()
            .get()
    };

    let result = f
        .niri_state()
        .do_action_inner(Action::MoveWindowToNewWorkspaceDown(true), false);
    assert_eq!(result, Ok(DoActionOutcome::Handled));

    // Complete the workspace-switch animation to trigger the cleanup hook.
    f.niri_complete_animations();
    f.niri_state().refresh_and_flush_clients();

    // The source workspace id must not appear in the active view after pruning.
    let post_ws_ids = active_view_ws_ids(&mut f);
    assert!(
        !post_ws_ids.contains(&source_ws_id),
        "emptied source workspace (id {source_ws_id}) must be pruned from the \
         active view after animation completion; remaining ids: {post_ws_ids:?}",
    );
}
