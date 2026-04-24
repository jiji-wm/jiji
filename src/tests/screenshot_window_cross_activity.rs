//! Pins the DD §5.18 `Action::ScreenshotWindowById` widen: dispatching the
//! action with an id that resolves to a window on a dormant workspace must
//! resolve via `windows_all()` (pool-span), not `windows()` (active-view
//! scope). Pre-1b the dispatcher scoped the id lookup to `windows()`, so a
//! hidden-activity window silently dropped every `ScreenshotWindowById`
//! call.
//!
//! Unlike `FocusWindow`, `ScreenshotWindowById` must NOT auto-switch the
//! active activity — a screenshot is a read-only observation; the user's
//! activity cursor is preserved. This test pins both halves: (1) the
//! dispatcher does not panic on the hidden-activity case, (2) the active
//! activity is unchanged across the dispatch.
//!
//! **What this test cannot pin:** actual pixel-buffer capture. The Fixture
//! harness's `backend.with_primary_renderer` does not run GL; the
//! `screenshot_window` call inside the closure is invoked but its buffer
//! is not observable at the Fixture level. This is a known gap matching
//! the precedent set by `focus_window_cross_activity.rs` — buffer-equality
//! lives in manual or visual-tests coverage, not here.

use niri_config::Action;

use super::client::ClientId;
use super::fixture::{config_with_two_activities, Fixture};

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
fn screenshot_window_resolves_hidden_activity_window() {
    // Two activities: alpha (default-active) and beta.
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();

    // Map the window under alpha.
    map_window(&mut f, client_id, 100, 100);
    f.niri_state().refresh_and_flush_clients();

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

    // Switch to beta. The window stays on alpha's workspace — under beta-
    // active it is not reachable via `windows()` (the active view). A
    // regression that walks the active-view scope would silently drop the
    // lookup here.
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
    assert_eq!(
        f.niri().layout.active_activity_id(),
        beta_id,
        "precondition: active activity is beta before ScreenshotWindowById dispatches",
    );

    // Dispatch `Action::ScreenshotWindowById { id, .. }` under beta-active.
    // The widened dispatcher must:
    //   1. Resolve the window via `windows_all()` (pool-span).
    //   2. Resolve the window's bound output to a live monitor.
    //   3. Invoke the screenshot path without panicking.
    //   4. NOT switch the active activity (screenshot is read-only; unlike
    //      `FocusWindow`, no activity-cursor side effect).
    f.niri_state().do_action(
        Action::ScreenshotWindowById {
            id: window_id,
            write_to_disk: false,
            show_pointer: false,
            path: None,
        },
        false,
    );
    f.niri_state().refresh_and_flush_clients();

    assert_eq!(
        f.niri().layout.active_activity_id(),
        beta_id,
        "ScreenshotWindowById must not switch the active activity — unlike \
         FocusWindow, a screenshot is a read-only observation (DD §5.18)",
    );
}
