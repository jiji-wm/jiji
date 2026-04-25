//! Pins the `with_windows_all_mut` half of
//! [`Layout::toggle_windowed_fullscreen`](crate::layout::Layout::toggle_windowed_fullscreen)
//! against a regression back to the active-activity-only pool walk.
//!
//! The fix widened `toggle_windowed_fullscreen` to use
//! `with_windows_all_mut` (layout/mod.rs:6049) so the windowed-fullscreen
//! flip reaches windows on dormant activities. A regression that reverts
//! that line to `with_windows_mut` would pass all existing fullscreen tests
//! (none set up two activities) and pass the `close_window_cross_activity`
//! test (different action). This test maps a window on alpha, switches to
//! beta, and asserts `Action::ToggleWindowedFullscreenById` flips
//! `is_pending_windowed_fullscreen` to `true` on the dormant window.

use niri_config::Action;

use super::client::ClientId;
use super::fixture::{config_with_two_activities, Fixture};
use crate::layout::LayoutElement as _;

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
fn toggle_windowed_fullscreen_by_id_reaches_hidden_activity_window() {
    // Two activities: alpha (default-active) and beta.
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();

    // Map the window under alpha.
    map_window(&mut f, client_id, 100, 100);

    // Capture the server-side window id while alpha is still active.
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

    // Baseline sanity: window is not yet pending windowed-fullscreen.
    {
        let (_out, mapped) = f
            .niri()
            .layout
            .windows_all()
            .next()
            .expect("window must still be in pool");
        assert!(
            !mapped.is_pending_windowed_fullscreen(),
            "baseline: is_pending_windowed_fullscreen must be false before the action dispatches",
        );
    }

    // Switch to beta. The window stays on alpha's workspace — a regression
    // that walks `with_windows_mut` (active view only) would now miss the
    // window and leave is_pending_windowed_fullscreen unchanged.
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

    // Dispatch `Action::ToggleWindowedFullscreenById` under beta-active.
    // The `with_windows_all_mut` path must reach the dormant-activity
    // window and flip is_pending_windowed_fullscreen to true.
    f.niri_state()
        .do_action(Action::ToggleWindowedFullscreenById(window_id), false);

    let (_out, mapped) = f
        .niri()
        .layout
        .windows_all()
        .next()
        .expect("window must still be in pool after action");
    assert!(
        mapped.is_pending_windowed_fullscreen(),
        "ToggleWindowedFullscreenById must flip is_pending_windowed_fullscreen on a \
         hidden-activity window (with_windows_all_mut widen pins cross-activity toggle)",
    );
}
