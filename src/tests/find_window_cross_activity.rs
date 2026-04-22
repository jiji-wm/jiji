//! Pins the pool-spanning widen of
//! [`Layout::find_window_and_output`](crate::layout::Layout::find_window_and_output)
//! against a regression back to the `monitors × active_view.ids` scan.
//!
//! A hidden-activity window's surface commits, ack_configures, and popup
//! unconstraining all flow through `find_window_and_output` in
//! `handlers/*`. Before Phase 1b Part 1, the lookup only visited the
//! active activity's views — a window mapped on one activity and left
//! behind when the user switched activities would silently drop every
//! surface event until the activity was re-entered. This test maps a
//! window on alpha, switches to beta, and asserts the same server-side
//! `WlSurface` still resolves to the same window via
//! `find_window_and_output` under the beta-active snapshot.

use super::client::ClientId;
use super::fixture::{config_with_two_activities, Fixture};

/// Drives a full initial-commit + attach-buffer roundtrip so the window
/// is mapped and present in the layout. Mirrors the private helper in
/// `transactions.rs` — kept inline here to avoid cross-module `pub(super)`
/// plumbing for a single use site.
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
fn find_window_and_output_resolves_hidden_activity_window() {
    // Two activities: alpha (default-active) and beta.
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();

    // Map the window under alpha.
    map_window(&mut f, client_id, 100, 100);

    // Capture the server-side WlSurface and window id from the mapped
    // window while alpha is still active. `find_window_and_output`
    // compares against `Mapped::toplevel().wl_surface()` via
    // `LayoutElement::is_wl_surface`, so we must use the server-side
    // reference, not the client-side one returned by `create_window`.
    let (server_surface, expected_id) = {
        let layout = &f.niri().layout;
        let (_out, mapped) = layout
            .windows_all()
            .next()
            .expect("mapping a window via Fixture must land it in the pool");
        (mapped.toplevel().wl_surface().clone(), mapped.id())
    };

    // Baseline sanity: before the activity switch, the lookup resolves.
    {
        let (found, _out) = f
            .niri()
            .layout
            .find_window_and_output(&server_surface)
            .expect("baseline: alpha-active lookup must resolve the mapped window");
        assert_eq!(found.id(), expected_id);
    }

    // Switch to beta. The window stays on alpha's workspace (workspace
    // pool is shared across activities; only the active activity's view
    // drives visibility) — a regression that walks `monitors × active
    // view.ids` would now miss the window.
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

    // The behavioral pin: under beta-active, the same server-side
    // WlSurface must still resolve to the same window via the widened
    // `find_window_and_output` body.
    let (found, _out) = f
        .niri()
        .layout
        .find_window_and_output(&server_surface)
        .expect(
            "find_window_and_output must resolve a window on a dormant activity \
             (pool-spanning widen pins cross-activity surface routing)",
        );
    assert_eq!(
        found.id(),
        expected_id,
        "resolved window id must match the alpha-mapped window's id",
    );
}
