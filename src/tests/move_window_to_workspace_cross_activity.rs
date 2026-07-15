//! Pins the cross-activity `Action::MoveWindowToWorkspace` dispatch:
//! moving the focused tile to a workspace that belongs to a *different*
//! activity must actually move the window (D1), and an unknown target id
//! must return `Err(MoveWindowTargetUnreachable)` instead of the pre-fix
//! silent `Ok(Handled)` (D2 — the regression that surfaced as a CLI
//! false-success line).
//!
//! Covers both internal arms the wire action converts to:
//! - `window_id: None` → `Action::MoveWindowToWorkspace(reference, focus)` (plain `move-window`,
//!   the user's repro path).
//! - `window_id: Some(_)` → `Action::MoveWindowToWorkspaceById { .. }` (`move-window --follow`).
//!
//! The disconnected-output arm (arm 3 of `move_window_to_pool_workspace`) is
//! pinned at the Layout level in
//! `layout::tests::move_window_to_pool_workspace_disconnected_output_returns_err`
//! because that arm is only reachable after a full-disconnect (all monitors
//! removed); the State-level path with remaining monitors always migrates
//! dormant workspaces to remaining outputs via the partial-disconnect walk,
//! making arm 3 unreachable here.
//!
//! The dormant-*source* case (the `--window <id>` flag naming a window on a
//! non-active activity) is out of scope and remains pinned by
//! `move_window_to_workspace_by_id_cross_activity` — that test must stay
//! green here.

use jiji_config::{Action, WorkspaceReference};

use super::client::ClientId;
use super::fixture::{config_with_two_activities, Fixture};
use crate::layout::{DoActionError, DoActionOutcome};

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

/// Helper: find beta's trailing-empty workspace id (the bookend) while alpha
/// is the active activity. Resolves via the workspace pool's per-activity
/// view of the connected output — the bookend is the last id in that view.
fn beta_trailing_empty_ws_id(f: &mut Fixture) -> u64 {
    let layout = &f.niri().layout;
    let beta_id = layout
        .activities()
        .iter()
        .find(|a| a.name() == "beta")
        .expect("beta must be present in the config-seeded activity pool")
        .id();
    let beta = layout
        .activities()
        .iter()
        .find(|a| a.id() == beta_id)
        .expect("beta resolves");
    let (_, view) = beta
        .views()
        .iter()
        .next()
        .expect("beta must have a view for the lone connected output");
    let trailing = *view
        .ids()
        .last()
        .expect("per-activity view always carries at least one bookend slot");
    trailing.get()
}

#[test]
fn move_window_to_workspace_id_into_dormant_activity_moves_the_window() {
    // Two activities: alpha (default-active) and beta. Single monitor.
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();
    map_window(&mut f, client_id, 100, 100);
    f.niri_state().refresh_and_flush_clients();

    // Alpha is active; beta's trailing-empty workspace is dormant from
    // alpha's vantage point (not in any active view). This is the cross-
    // activity target.
    let beta_target = beta_trailing_empty_ws_id(&mut f);

    // Capture the window's source workspace id (lives in alpha's view).
    let source_ws_id = {
        let layout = &f.niri().layout;
        let (_out, mapped) = layout
            .windows_all()
            .next()
            .expect("the lone mapped window must be in the pool");
        let win = mapped.window.clone();
        layout
            .workspaces_all()
            .find(|(_, ws)| ws.windows().any(|w| w.window == win))
            .expect("window must live on some workspace")
            .1
            .id()
            .get()
    };
    assert_ne!(
        source_ws_id, beta_target,
        "fixture sanity: source and dormant target must differ",
    );

    // Dispatch plain `move-window` (window_id:None) to beta's dormant
    // workspace. Pre-fix this silently returned `Ok(Handled)`; post-fix it
    // must move the window and return `Ok(Handled)` honestly.
    let outcome = f
        .niri_state()
        .do_action_inner(
            Action::MoveWindowToWorkspace(WorkspaceReference::Id(beta_target), false),
            false,
        )
        .expect("dispatch must succeed against a reachable dormant target");
    assert_eq!(outcome, DoActionOutcome::Handled);
    f.niri_state().refresh_and_flush_clients();

    // The window now lives on beta's workspace.
    let layout = &f.niri().layout;
    let window = {
        let (_out, mapped) = layout.windows_all().next().expect("window still in pool");
        mapped.window.clone()
    };
    let owning_ws = layout
        .workspaces_all()
        .find(|(_, ws)| ws.windows().any(|w| w.window == window))
        .expect("window must live on some workspace post-move")
        .1
        .id()
        .get();
    assert_eq!(
        owning_ws, beta_target,
        "the moved window must live on beta's dormant workspace after the dispatch \
         — pre-fix this assertion failed (window stayed on alpha)",
    );

    // Source workspace is now empty (move, not copy).
    let source_ws_window_count = layout
        .workspaces_all()
        .find(|(_, ws)| ws.id().get() == source_ws_id)
        .map(|(_, ws)| ws.windows().count())
        .unwrap_or(0);
    assert_eq!(
        source_ws_window_count, 0,
        "source workspace must be empty after the move — a copy-instead-of-move \
         regression would fail here",
    );

    // `focus:false` contract: the user stays put on alpha.
    assert_eq!(
        layout
            .activities()
            .iter()
            .find(|a| a.name() == "alpha")
            .map(|a| a.id()),
        Some(layout.active_activity_id()),
        "focus:false must NOT switch activities",
    );
}

#[test]
fn move_window_to_workspace_id_unknown_target_returns_err() {
    // D2 headline: an unknown ws id must surface as
    // `Err(MoveWindowTargetUnreachable)`, NOT silent `Ok(Handled)`.
    // The CLI surfaces this as a non-zero exit with no false-success line.
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();
    map_window(&mut f, client_id, 100, 100);
    f.niri_state().refresh_and_flush_clients();

    // Pick an id guaranteed to be absent from the pool.
    let bogus_id: u64 = f
        .niri()
        .layout
        .workspaces_all()
        .map(|(_, ws)| ws.id().get())
        .max()
        .unwrap_or(0)
        + 1000;

    let result = f.niri_state().do_action_inner(
        Action::MoveWindowToWorkspace(WorkspaceReference::Id(bogus_id), false),
        false,
    );
    assert_eq!(
        result,
        Err(DoActionError::MoveWindowTargetUnreachable { ws_id: bogus_id }),
        "an unknown ws id must surface as a typed error, not silent success — \
         this is the D2 regression pin against the CLI false-success line",
    );
}

#[test]
fn move_window_to_workspace_by_id_unknown_target_returns_err() {
    // D2 across the `Some(window_id)` arm — `move-window --follow`'s path.
    // The dormant-*source* fall-through must not be reached when the source
    // is live; the typed `Err` is the expected outcome.
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

    let bogus_id: u64 = f
        .niri()
        .layout
        .workspaces_all()
        .map(|(_, ws)| ws.id().get())
        .max()
        .unwrap_or(0)
        + 1000;

    let result = f.niri_state().do_action_inner(
        Action::MoveWindowToWorkspaceById {
            window_id,
            reference: WorkspaceReference::Id(bogus_id),
            focus: false,
        },
        false,
    );
    assert_eq!(
        result,
        Err(DoActionError::MoveWindowTargetUnreachable { ws_id: bogus_id }),
        "an unknown target must surface as a typed error on the by-id arm too \
         (mirrors the `None`-arm pin) — the dormant-source fall-through must \
         not swallow this case when the source is live",
    );
}

#[test]
fn move_window_to_workspace_with_focus_true_activates_target_activity() {
    // `focus:true` mirrors `Action::FocusWindow` after the move: switch
    // into a target activity that hosts the destination workspace, then
    // focus the moved window. Pinned end-state: active activity flips from
    // alpha to beta.
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();
    map_window(&mut f, client_id, 100, 100);
    f.niri_state().refresh_and_flush_clients();

    let alpha_id = f.niri().layout.active_activity_id();
    let beta_id = f
        .niri()
        .layout
        .activities()
        .iter()
        .find(|a| a.name() == "beta")
        .expect("beta must be present")
        .id();
    assert_ne!(alpha_id, beta_id);

    let beta_target = beta_trailing_empty_ws_id(&mut f);

    f.niri_state()
        .do_action_inner(
            Action::MoveWindowToWorkspace(WorkspaceReference::Id(beta_target), true),
            false,
        )
        .expect("dispatch must succeed");
    f.niri_state().refresh_and_flush_clients();

    assert_eq!(
        f.niri().layout.active_activity_id(),
        beta_id,
        "focus:true must activate the target activity",
    );
}

#[test]
fn move_window_to_workspace_id_self_move_returns_no_op() {
    // Cross-activity self-move short-circuit: the resolver is now pool-wide,
    // so naming the source workspace by id (irrespective of which activity
    // is active) yields `NoOp(AlreadyOnTarget)`.
    use jiji_ipc::NoOpReason;

    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();
    map_window(&mut f, client_id, 100, 100);
    f.niri_state().refresh_and_flush_clients();

    let source_ws_id = f
        .niri()
        .layout
        .active_workspace()
        .expect("alpha-active monitor must have an active workspace")
        .id()
        .get();

    let result = f.niri_state().do_action_inner(
        Action::MoveWindowToWorkspace(WorkspaceReference::Id(source_ws_id), false),
        false,
    );
    assert_eq!(
        result,
        Ok(DoActionOutcome::NoOp(NoOpReason::AlreadyOnTarget {
            workspace_id: source_ws_id,
        })),
        "naming the source workspace by id must still short-circuit as NoOp, \
         even after the pool-wide resolver widen",
    );
}

#[test]
fn move_window_to_workspace_id_dormant_landing_mints_fresh_trailing_empty() {
    // Per-activity bookend invariant: if the moved tile lands on beta's
    // trailing-empty bookend, beta's view of that monitor must grow a
    // fresh empty bookend. The cross-activity mover runs the normalize
    // sweep post-attach, same as the in-activity column movers.
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();
    map_window(&mut f, client_id, 100, 100);
    f.niri_state().refresh_and_flush_clients();

    let beta_target = beta_trailing_empty_ws_id(&mut f);
    let beta_view_len_before = {
        let layout = &f.niri().layout;
        let beta = layout
            .activities()
            .iter()
            .find(|a| a.name() == "beta")
            .expect("beta present");
        beta.views().iter().next().expect("beta has a view").1.len()
    };

    f.niri_state()
        .do_action_inner(
            Action::MoveWindowToWorkspace(WorkspaceReference::Id(beta_target), false),
            false,
        )
        .expect("dispatch must succeed");
    f.niri_state().refresh_and_flush_clients();

    let layout = &f.niri().layout;
    let beta = layout
        .activities()
        .iter()
        .find(|a| a.name() == "beta")
        .expect("beta present");
    let (_, beta_view) = beta.views().iter().next().expect("beta has a view");

    assert_eq!(
        beta_view.len(),
        beta_view_len_before + 1,
        "the bookend sweep must mint a fresh trailing empty after the moved \
         tile lands on what was beta's trailing slot",
    );
    // The previously-trailing id is no longer at the trailing slot.
    let trailing_after = *beta_view.ids().last().expect("post-sweep view non-empty");
    assert_ne!(
        trailing_after.get(),
        beta_target,
        "the fresh empty must be the new trailing slot — the moved-into \
         workspace is now an interior slot",
    );
}

#[test]
fn move_window_to_workspace_by_id_cross_activity_happy_path() {
    // Fix-5: happy-path cross-activity move through the `Some(window_id)` arm
    // (`Action::MoveWindowToWorkspaceById`). The pre-fix code would silently
    // leave the window in place; this pin ensures the move actually executes.
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();
    map_window(&mut f, client_id, 100, 100);
    f.niri_state().refresh_and_flush_clients();

    let (window_id, source_ws_id) = {
        let layout = &f.niri().layout;
        let (_out, mapped) = layout
            .windows_all()
            .next()
            .expect("mapped window must be in pool");
        let win_id = crate::layout::LayoutElement::id(mapped);
        let ws_id = layout
            .workspaces_all()
            .find(|(_, ws)| ws.has_window(win_id))
            .expect("window must be on a workspace")
            .1
            .id()
            .get();
        (mapped.id().get(), ws_id)
    };

    let beta_target = beta_trailing_empty_ws_id(&mut f);
    assert_ne!(source_ws_id, beta_target, "fixture: source != target");

    let outcome = f
        .niri_state()
        .do_action_inner(
            Action::MoveWindowToWorkspaceById {
                window_id,
                reference: WorkspaceReference::Id(beta_target),
                focus: false,
            },
            false,
        )
        .expect("dispatch must succeed for a reachable dormant target");
    assert_eq!(outcome, DoActionOutcome::Handled);
    f.niri_state().refresh_and_flush_clients();

    // The window now lives on beta's workspace.
    let layout = &f.niri().layout;
    let window = {
        let (_out, mapped) = layout.windows_all().next().expect("window still in pool");
        mapped.window.clone()
    };
    let owning_ws = layout
        .workspaces_all()
        .find(|(_, ws)| ws.windows().any(|w| w.window == window))
        .expect("window must live on some workspace post-move")
        .1
        .id()
        .get();
    assert_eq!(
        owning_ws, beta_target,
        "MoveWindowToWorkspaceById must move the window to the dormant beta workspace — \
         pre-fix this would have left the window on alpha (Some-arm regression pin)",
    );

    // Source workspace is empty (move, not copy).
    let source_ws_window_count = layout
        .workspaces_all()
        .find(|(_, ws)| ws.id().get() == source_ws_id)
        .map(|(_, ws)| ws.windows().count())
        .unwrap_or(0);
    assert_eq!(
        source_ws_window_count, 0,
        "source workspace must be empty after the by-id move",
    );

    // focus:false — alpha still active.
    assert_eq!(
        layout
            .activities()
            .iter()
            .find(|a| a.name() == "alpha")
            .map(|a| a.id()),
        Some(layout.active_activity_id()),
        "focus:false on the by-id arm must not switch activities",
    );
}

#[test]
fn move_window_on_empty_bookend_is_noop_does_not_switch_activity() {
    // Pin for the `NothingToMove` variant: when the active workspace's focused
    // slot is an empty bookend, the cross-activity move must be a clean no-op —
    // no activity switch, no focus change, reply is `Ok(Handled)`.
    //
    // To get a focused-empty-bookend state: alpha has one window; we first move
    // it to beta (so alpha's slot is the empty bookend), then attempt a second
    // cross-activity move. The second dispatch must see `NothingToMove` and
    // return without switching.
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();
    map_window(&mut f, client_id, 100, 100);
    f.niri_state().refresh_and_flush_clients();

    let alpha_id = f.niri().layout.active_activity_id();
    let beta_target = beta_trailing_empty_ws_id(&mut f);

    // First move: window → beta's trailing empty. Alpha now has only the empty bookend.
    f.niri_state()
        .do_action_inner(
            Action::MoveWindowToWorkspace(WorkspaceReference::Id(beta_target), false),
            false,
        )
        .expect("first move must succeed");
    f.niri_state().refresh_and_flush_clients();

    // Verify alpha still active and there are no windows in alpha's view
    // (focused slot = empty bookend after the first move drained the only tile).
    {
        let layout = &f.niri().layout;
        assert_eq!(
            layout.active_activity_id(),
            alpha_id,
            "alpha must still be active after focus:false move"
        );
        let alpha = layout
            .activities()
            .iter()
            .find(|a| a.id() == alpha_id)
            .expect("alpha must still be in the activity pool");
        let alpha_ids: Vec<_> = alpha
            .views()
            .values()
            .flat_map(|v| v.ids().iter().copied())
            .collect();
        let alpha_window_count: usize = layout
            .workspaces_all()
            .filter(|(_, ws)| alpha_ids.contains(&ws.id()))
            .map(|(_, ws)| ws.windows().count())
            .sum();
        assert_eq!(
            alpha_window_count, 0,
            "alpha view must be empty before the NothingToMove dispatch"
        );
    }

    // Get a fresh beta target (the view grew by 1 after the first move).
    let beta_target2 = beta_trailing_empty_ws_id(&mut f);

    // Capture focused window before the no-op dispatch.
    let pre_focus = f.niri().layout.focus().map(|w| w.id().get());

    // Second move: focused slot is the empty bookend — must be a no-op.
    let outcome = f
        .niri_state()
        .do_action_inner(
            Action::MoveWindowToWorkspace(WorkspaceReference::Id(beta_target2), false),
            false,
        )
        .expect("NothingToMove must return Ok, not Err");
    assert_eq!(
        outcome,
        DoActionOutcome::Handled,
        "empty-bookend move must return Ok(Handled) — not an error",
    );
    f.niri_state().refresh_and_flush_clients();

    // Active activity still alpha — no switch must have occurred.
    assert_eq!(
        f.niri().layout.active_activity_id(),
        alpha_id,
        "NothingToMove must not switch the active activity",
    );

    // Focused window must be unchanged.
    let post_focus = f.niri().layout.focus().map(|w| w.id().get());
    assert_eq!(
        post_focus, pre_focus,
        "NothingToMove must not change the focused window",
    );
}

#[test]
fn move_window_on_empty_bookend_focus_true_is_noop_does_not_switch_activity() {
    // Variant of the `NothingToMove` pin above exercising `focus:true`. Even
    // with `focus:true`, an empty-bookend source must be a clean no-op — no
    // activity switch, no focus change, reply is `Ok(Handled)`.
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();
    map_window(&mut f, client_id, 100, 100);
    f.niri_state().refresh_and_flush_clients();

    let alpha_id = f.niri().layout.active_activity_id();
    let beta_target = beta_trailing_empty_ws_id(&mut f);

    // First move (focus:false): window → beta. Alpha now has only the empty bookend.
    f.niri_state()
        .do_action_inner(
            Action::MoveWindowToWorkspace(WorkspaceReference::Id(beta_target), false),
            false,
        )
        .expect("first move must succeed");
    f.niri_state().refresh_and_flush_clients();

    // Get a fresh beta target (view grew by 1 after the first move).
    let beta_target2 = beta_trailing_empty_ws_id(&mut f);

    // Capture focused window before the no-op dispatch.
    let pre_focus = f.niri().layout.focus().map(|w| w.id().get());

    // Second move with focus:true against an empty-bookend source — must be a no-op.
    let outcome = f
        .niri_state()
        .do_action_inner(
            Action::MoveWindowToWorkspace(WorkspaceReference::Id(beta_target2), true),
            false,
        )
        .expect("NothingToMove must return Ok, not Err");
    assert_eq!(
        outcome,
        DoActionOutcome::Handled,
        "empty-bookend move (focus:true) must return Ok(Handled) — not an error",
    );
    f.niri_state().refresh_and_flush_clients();

    // Active activity must still be alpha — focus:true must not switch when
    // there is nothing to move.
    assert_eq!(
        f.niri().layout.active_activity_id(),
        alpha_id,
        "NothingToMove with focus:true must not switch the active activity",
    );

    // Focused window must be unchanged.
    let post_focus = f.niri().layout.focus().map(|w| w.id().get());
    assert_eq!(
        post_focus, pre_focus,
        "NothingToMove must not change the focused window",
    );
}
