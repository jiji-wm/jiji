//! Pins the hidden-activity panic gate on
//! [`Layout::move_to_output`](crate::layout::Layout::move_to_output)
//! (Phase 1b Part 2b tail).
//!
//! Parallels `move_window_to_workspace_by_id_cross_activity`: the
//! `monitors × active-view` walk inside `move_to_output` cannot reach a
//! window on a dormant activity. Before this gate, the resolver's
//! `.unwrap()` panicked on such a dispatch. Cross-activity move-by-id
//! semantics are deferred to Phase 2 (DD §5.18); under Phase 1b the
//! gate logs a `warn!` and returns cleanly.
//!
//! Two monitors are required so the policy-(a) invariant assertion has
//! teeth: under a silent cross-activity move implementation, the window
//! would be reassigned to a workspace bound to `headless-2` (a distinct
//! `WorkspaceId` from the alpha workspace on `headless-1`). With a
//! single output the assertion would be trivially satisfied.

use niri_config::Action;

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
fn move_window_to_monitor_by_id_reaches_hidden_activity_window_without_panic() {
    // Two activities: alpha (default-active) and beta.
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    // Two monitors: "headless-1" (primary) and "headless-2".
    f.add_output(1, (1920, 1080));
    f.add_output(2, (1920, 1080));
    let client_id = f.add_client();

    // Map the window under alpha (lands on the active monitor, which
    // is "headless-1" by default).
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

    // Switch to beta. The window stays on alpha's workspace — a
    // regression back to `.unwrap()` panics when the `monitors ×
    // active-view` walk fails to locate the window.
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

    // Dispatch `Action::MoveWindowToMonitorById` targeting the second
    // monitor. The hidden-activity gate must silently drop.
    f.niri_state().do_action(
        Action::MoveWindowToMonitorById {
            id: window_id,
            output: "headless-2".to_string(),
        },
        false,
    );
    f.niri_state().refresh_and_flush_clients();

    // Invariant: window still exists and still lives on its original
    // workspace. A regression implementing cross-activity move
    // semantics here (policy (a)) would reassign it to a workspace on
    // the second monitor.
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
         (cross-activity move-by-id semantics are deferred, DD §5.18)",
    );
}
