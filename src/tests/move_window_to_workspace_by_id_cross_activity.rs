//! Pins the hidden-activity panic gate on
//! [`Layout::move_to_workspace`](crate::layout::Layout::move_to_workspace)
//!
//! The id-based action lookup in `input/mod.rs` uses the pool-spanning
//! `windows_all()` iterator, so a window mapped on a
//! workspace bound to a dormant activity is successfully resolved by id.
//! Before this gate, the `monitors × active-view` walk inside
//! `move_to_workspace` panicked on `.unwrap()` when the resolved window
//! wasn't present in any active-activity view. Real cross-activity
//! move-by-id semantics are not yet implemented; the gate logs a `warn!`
//! and returns cleanly, leaving the window where it was.
//!
//! A regression that collapses the gate back to `.unwrap()` or to
//! `.expect(...)` panics the compositor on dispatch; a regression that
//! silently implements cross-activity move semantics here (policy (a) —
//! deferred) is caught by the "same workspace before/after" invariant
//! below.

use niri_config::{Action, WorkspaceReference};

use super::client::ClientId;
use super::fixture::{config_with_two_activities, Fixture};
use crate::layout::LayoutElement;

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
fn move_window_to_workspace_by_id_reaches_hidden_activity_window_without_panic() {
    // Two activities: alpha (default-active) and beta.
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();

    // Map the window under alpha.
    map_window(&mut f, client_id, 100, 100);

    // Capture the server-side window id and its owning workspace id
    // while alpha is still active.
    let (window_id, original_ws_id) = {
        let layout = &f.niri().layout;
        let mut it = layout.windows_all();
        let (_out, mapped) = it
            .next()
            .expect("mapping a window via Fixture must land it in the pool");
        assert!(
            it.next().is_none(),
            "fixture sanity: exactly one window must be mapped",
        );
        let window = LayoutElement::id(mapped);
        let window_id = mapped.id().get();
        let (_out, ws) = layout
            .workspaces_all()
            .find(|(_, ws)| ws.has_window(window))
            .expect("window must be on some workspace in the pool");
        (window_id, ws.id())
    };

    // Switch to beta. The window stays on alpha's workspace — the
    // `monitors × active-view` walk in `move_to_workspace` cannot
    // reach it. Pre-gate this was a `.unwrap()` panic; post-gate the
    // mutator must `warn!` and return cleanly.
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

    // Dispatch `Action::MoveWindowToWorkspaceById` under beta-active.
    // The hidden-activity gate must silently drop (with `warn!`).
    f.niri_state()
        .do_action_inner(
            Action::MoveWindowToWorkspaceById {
                window_id,
                reference: WorkspaceReference::Index(0),
                focus: false,
            },
            false,
        )
        .expect(
            "MoveWindowToWorkspaceById dispatch must return Ok(()) — the \
             cross-activity drop is a layout-level silent no-op, not a DoActionError",
        );
    f.niri_state().refresh_and_flush_clients();

    // Invariant: window still exists and still lives on its original
    // workspace. A regression that implements cross-activity move
    // semantics here (policy (a)) would reassign it.
    let layout = &f.niri().layout;
    let mut it = layout.windows_all();
    let (_out, mapped) = it
        .next()
        .expect("window must still be in pool after the dropped action");
    assert!(
        it.next().is_none(),
        "exactly one window must still be mapped after the dropped action",
    );
    assert_eq!(mapped.id().get(), window_id, "window id preserved");
    let window = LayoutElement::id(mapped);
    let (_out, ws) = layout
        .workspaces_all()
        .find(|(_, ws)| ws.has_window(window))
        .expect("window must still be on some workspace");
    assert_eq!(
        ws.id(),
        original_ws_id,
        "hidden-activity gate must not silently reassign the window's workspace \
         (cross-activity move-by-id semantics are deferred)",
    );
}
