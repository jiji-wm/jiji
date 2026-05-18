//! Pins the typed `NoOp(AlreadyOnTarget)` reply emitted when
//! `MoveWindowToWorkspace` / `MoveWindowToWorkspaceById` resolves to the
//! workspace the targeted window already lives on.
//!
//! The dispatch arms detect the move-to-self case *before* delegating to the
//! layout mutator, so the wire-visible result is
//! `Ok(DoActionOutcome::NoOp(NoOpReason::AlreadyOnTarget { workspace_id }))`
//! instead of falling through to the silent equality short-circuit at the
//! layout layer.
//!
//! A regression that emits `Handled` for a move-to-self (collapsing the new
//! detection back into the silent path) fails tests 1 and 2. A regression
//! that emits `NoOp` for a real cross-workspace move (over-aggressive
//! short-circuit) fails test 3.

use niri_config::{Action, Config, WorkspaceReference};
use niri_ipc::NoOpReason;

use super::client::ClientId;
use super::fixture::Fixture;
use crate::layout::{DoActionOutcome, LayoutElement};

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
fn move_window_to_workspace_by_id_same_as_source_returns_no_op() {
    let mut f = Fixture::new();
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();

    map_window(&mut f, client_id, 100, 100);

    // Capture the server-side window id and its owning workspace id.
    let (window_id, source_ws_raw_id) = {
        let layout = &f.niri().layout;
        let (_out, mapped) = layout
            .windows_all()
            .next()
            .expect("mapping a window via Fixture must land it in the pool");
        let win_id = LayoutElement::id(mapped);
        let window_id = mapped.id().get();
        let (_out, ws) = layout
            .workspaces_all()
            .find(|(_, ws)| ws.has_window(win_id))
            .expect("window must be on some workspace in the pool");
        (window_id, ws.id().get())
    };

    let result = f.niri_state().do_action_inner(
        Action::MoveWindowToWorkspaceById {
            window_id,
            reference: WorkspaceReference::Id(source_ws_raw_id),
            focus: false,
        },
        false,
    );

    assert_eq!(
        result,
        Ok(DoActionOutcome::NoOp(NoOpReason::AlreadyOnTarget {
            workspace_id: source_ws_raw_id,
        })),
        "MoveWindowToWorkspaceById against the window's own workspace must \
         emit Ok(NoOp(AlreadyOnTarget {{ workspace_id: source }}))",
    );

    // Invariant: window's owning workspace is unchanged after the no-op.
    let layout = &f.niri().layout;
    let (_out, mapped) = layout
        .windows_all()
        .next()
        .expect("window must still be in pool after the no-op");
    let win_id = LayoutElement::id(mapped);
    let (_out, ws) = layout
        .workspaces_all()
        .find(|(_, ws)| ws.has_window(win_id))
        .expect("window must still be on some workspace");
    assert_eq!(
        ws.id().get(),
        source_ws_raw_id,
        "move-to-self must not change the window's owning workspace",
    );
}

#[test]
fn move_window_to_workspace_by_reference_same_as_source_returns_no_op() {
    // Single activity with one named workspace. The window maps onto the
    // named workspace; we then dispatch the no-id `MoveWindowToWorkspace`
    // arm with a `WorkspaceReference::Name` matching the source workspace.
    let config = Config::parse_mem("workspace \"named-ws\"\n")
        .expect("parse_mem must succeed on a single-named-workspace config");
    let mut f = Fixture::with_config(config);
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();

    map_window(&mut f, client_id, 100, 100);

    let source_ws_raw_id = {
        let layout = &f.niri().layout;
        let (_out, mapped) = layout
            .windows_all()
            .next()
            .expect("mapping a window via Fixture must land it in the pool");
        let win_id = LayoutElement::id(mapped);
        let (_out, ws) = layout
            .workspaces_all()
            .find(|(_, ws)| ws.has_window(win_id))
            .expect("window must be on some workspace in the pool");
        assert!(
            ws.name().is_some_and(|n| n == "named-ws"),
            "fixture sanity: window must land on the named workspace seeded \
             by the config (config declares `named-ws` first, so it's the \
             primary entry the focused tile is mapped onto); got name={:?}",
            ws.name(),
        );
        ws.id().get()
    };

    let result = f.niri_state().do_action_inner(
        Action::MoveWindowToWorkspace(WorkspaceReference::Name("named-ws".into()), false),
        false,
    );

    assert_eq!(
        result,
        Ok(DoActionOutcome::NoOp(NoOpReason::AlreadyOnTarget {
            workspace_id: source_ws_raw_id,
        })),
        "MoveWindowToWorkspace by Name against the focused tile's own \
         workspace must emit Ok(NoOp(AlreadyOnTarget {{ workspace_id: source }}))",
    );

    let layout = &f.niri().layout;
    let (_out, mapped) = layout
        .windows_all()
        .next()
        .expect("window must still be in pool after the no-op");
    let win_id = LayoutElement::id(mapped);
    let (_out, ws) = layout
        .workspaces_all()
        .find(|(_, ws)| ws.has_window(win_id))
        .expect("window must still be on some workspace");
    assert_eq!(
        ws.id().get(),
        source_ws_raw_id,
        "move-to-self by Name must not change the window's owning workspace",
    );
}

#[test]
fn move_window_to_workspace_to_different_workspace_returns_handled() {
    // Single activity with two named workspaces. The window maps onto the
    // first (`ws-a`); we then dispatch a real cross-workspace move targeting
    // `ws-b`. This pins the regression guard that the move-to-self check
    // does NOT collapse a real move into a NoOp.
    let config = Config::parse_mem("workspace \"ws-a\"\nworkspace \"ws-b\"\n")
        .expect("parse_mem must succeed on a two-named-workspace config");
    let mut f = Fixture::with_config(config);
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();

    map_window(&mut f, client_id, 100, 100);

    let (window_id, source_ws_raw_id, target_ws_raw_id) = {
        let layout = &f.niri().layout;
        let (_out, mapped) = layout
            .windows_all()
            .next()
            .expect("mapping a window via Fixture must land it in the pool");
        let win_id = LayoutElement::id(mapped);
        let window_id = mapped.id().get();
        let (_out, source_ws) = layout
            .workspaces_all()
            .find(|(_, ws)| ws.has_window(win_id))
            .expect("window must be on some workspace in the pool");
        let source_id = source_ws.id().get();
        // Pick the *other* named workspace as the target.
        let (_out, target_ws) = layout
            .workspaces_all()
            .find(|(_, ws)| ws.name().is_some_and(|n| n == "ws-b"))
            .expect("config-seeded `ws-b` must exist in the pool");
        let target_id = target_ws.id().get();
        assert_ne!(
            source_id, target_id,
            "fixture sanity: source and target workspaces must differ for \
             the regression-guard branch to be meaningful",
        );
        (window_id, source_id, target_id)
    };

    let result = f.niri_state().do_action_inner(
        Action::MoveWindowToWorkspaceById {
            window_id,
            reference: WorkspaceReference::Id(target_ws_raw_id),
            focus: false,
        },
        false,
    );

    assert_eq!(
        result,
        Ok(DoActionOutcome::Handled),
        "a real cross-workspace move must return Ok(Handled) (NOT NoOp); \
         a regression that flips this would over-fire the move-to-self \
         short-circuit on any reference that resolves",
    );

    // Invariant: window now lives on the target workspace.
    let layout = &f.niri().layout;
    let (_out, mapped) = layout
        .windows_all()
        .next()
        .expect("window must still be in pool after the real move");
    let win_id = LayoutElement::id(mapped);
    let (_out, ws) = layout
        .workspaces_all()
        .find(|(_, ws)| ws.has_window(win_id))
        .expect("window must be on some workspace after the move");
    assert_eq!(
        ws.id().get(),
        target_ws_raw_id,
        "real move must reassign the window's owning workspace to the target",
    );
    assert_ne!(
        ws.id().get(),
        source_ws_raw_id,
        "real move must move the window off the source workspace",
    );
}

#[test]
fn move_window_to_workspace_by_id_unknown_window_id_returns_handled() {
    // An unknown window id must preserve the pre-existing silent exit-0 as
    // `Ok(Handled)`. A future refactor that flips this to `NoOp(AlreadyOnTarget)`
    // (reasoning: "no observable change happened") would silently widen the
    // wire contract — this test pins it.
    let mut f = Fixture::new();
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();

    map_window(&mut f, client_id, 100, 100);

    let known_ws_id = {
        let layout = &f.niri().layout;
        layout
            .workspaces_all()
            .next()
            .expect("at least one workspace must exist after adding an output")
            .1
            .id()
            .get()
    };

    let result = f.niri_state().do_action_inner(
        Action::MoveWindowToWorkspaceById {
            window_id: u64::MAX,
            reference: WorkspaceReference::Id(known_ws_id),
            focus: false,
        },
        false,
    );

    assert_eq!(
        result,
        Ok(DoActionOutcome::Handled),
        "MoveWindowToWorkspaceById with an unknown window_id must return \
         Ok(Handled) (silent exit-0), not NoOp",
    );

    // Invariant: the existing window is unchanged.
    let layout = &f.niri().layout;
    assert_eq!(
        layout.windows_all().count(),
        1,
        "the mapped window must still be in the pool after the no-op",
    );
}

#[test]
fn move_window_to_workspace_by_index_same_as_source_returns_no_op() {
    // Single activity with a single named workspace. The window maps onto that
    // workspace (index 1 in the active view). Dispatching
    // `WorkspaceReference::Index(1)` resolves to the same workspace — must
    // produce `NoOp(AlreadyOnTarget)`. Pins the `Index` arm of
    // `resolve_workspace_reference_to_id`.
    let config = Config::parse_mem("workspace \"idx-ws\"\n")
        .expect("parse_mem must succeed on a single-named-workspace config");
    let mut f = Fixture::with_config(config);
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();

    map_window(&mut f, client_id, 100, 100);

    let source_ws_raw_id = {
        let layout = &f.niri().layout;
        let (_out, mapped) = layout
            .windows_all()
            .next()
            .expect("mapping a window via Fixture must land it in the pool");
        let win_id = LayoutElement::id(mapped);
        let (_out, ws) = layout
            .workspaces_all()
            .find(|(_, ws)| ws.has_window(win_id))
            .expect("window must be on some workspace in the pool");
        ws.id().get()
    };

    // Index 1 = first workspace in the active view (1-based per
    // `WorkspaceReference::Index` convention).
    let result = f.niri_state().do_action_inner(
        Action::MoveWindowToWorkspace(WorkspaceReference::Index(1), false),
        false,
    );

    assert_eq!(
        result,
        Ok(DoActionOutcome::NoOp(NoOpReason::AlreadyOnTarget {
            workspace_id: source_ws_raw_id,
        })),
        "MoveWindowToWorkspace by Index(1) against the focused tile's own \
         workspace must emit Ok(NoOp(AlreadyOnTarget {{ workspace_id: source }}))",
    );

    // Invariant: window unchanged.
    let layout = &f.niri().layout;
    let (_out, mapped) = layout
        .windows_all()
        .next()
        .expect("window must still be in pool after the no-op");
    let win_id = LayoutElement::id(mapped);
    let (_out, ws) = layout
        .workspaces_all()
        .find(|(_, ws)| ws.has_window(win_id))
        .expect("window must still be on some workspace");
    assert_eq!(
        ws.id().get(),
        source_ws_raw_id,
        "move-to-self by Index must not change the window's owning workspace",
    );
}
