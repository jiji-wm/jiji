//! Pins the mechanical pool-wide widen of
//! [`Action::CloseWindowById`](niri_config::Action::CloseWindowById)'s
//! id lookup against a regression back to the
//! active-activity-only `Layout::windows` iterator.
//!
//! The 21 id-based Action arms in `src/input/mod.rs` share identical
//! destructure shape: they resolve the target window by id, then hand
//! it to a Layout mutator. If the lookup is scoped to the active
//! activity's view, a window mapped on a dormant activity silently
//! falls off every such Action — no close, no fullscreen, no move.
//! One representative Action is enough: the other 20 swaps are
//! structurally identical and reviewer-visible.
//!
//! This test maps a window on alpha, switches to beta, dispatches
//! `Action::CloseWindowById(id)`, and asserts the client received the
//! `xdg_toplevel.close` event. A regression that walks
//! `monitors × active view.ids` would miss the window entirely under
//! beta-active and send nothing.

use niri_config::Action;
use wayland_client::protocol::wl_surface::WlSurface as ClientWlSurface;

use super::client::ClientId;
use super::fixture::{config_with_two_activities, Fixture};

/// Drives a full initial-commit + attach-buffer roundtrip so the window
/// is mapped and present in the layout. Returns the client-side
/// `WlSurface` so the caller can query the window's `close_requested`
/// flag after the action dispatches.
fn map_window(f: &mut Fixture, id: ClientId, w: u16, h: u16) -> ClientWlSurface {
    let window = f.client(id).create_window();
    let surface = window.surface.clone();
    window.commit();
    f.roundtrip(id);

    let window = f.client(id).window(&surface);
    window.attach_new_buffer();
    window.set_size(w, h);
    window.ack_last_and_commit();
    f.roundtrip(id);

    surface
}

#[test]
fn close_window_by_id_delivers_close_to_hidden_activity_window() {
    // Two activities: alpha (default-active) and beta.
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();

    // Map the window under alpha.
    let client_surface = map_window(&mut f, client_id, 100, 100);

    // Capture the server-side window id while alpha is still active.
    // The lone mapped window is the one we just created; there is no
    // other client in this fixture.
    let window_id = {
        let layout = &f.niri().layout;
        let mut it = layout.windows_all();
        let (_out, mapped) = it
            .next()
            .expect("mapping a window via Fixture must land it in the pool");
        assert!(
            it.next().is_none(),
            "fixture sanity: exactly one window must be mapped",
        );
        mapped.id().get()
    };

    // Switch to beta. The window stays on alpha's workspace (workspace
    // pool is shared across activities; only the active activity's view
    // drives visibility) — a regression that walks `monitors × active
    // view.ids` would now miss the window on lookup.
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

    // Baseline sanity: the client has not yet been asked to close.
    assert!(
        !f.client(client_id).window(&client_surface).close_requested,
        "baseline: no close event should be pending before the action dispatches",
    );

    // Dispatch `Action::CloseWindowById` under beta-active. The widened
    // `windows_all()` lookup must resolve the dormant-activity window
    // and hand its toplevel a close request.
    f.niri_state()
        .do_action(Action::CloseWindowById(window_id), false);
    f.niri_state().refresh_and_flush_clients();
    f.roundtrip(client_id);

    assert!(
        f.client(client_id).window(&client_surface).close_requested,
        "CloseWindowById must deliver xdg_toplevel.close to a hidden-activity window \
         (pool-spanning widen pins cross-activity id-based Action routing)",
    );
}
