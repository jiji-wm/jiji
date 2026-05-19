//! Pins the `Action::FocusWindow { id }` auto-switch contract:
//! dispatching the action with an id that resolves to a window on a
//! dormant workspace must auto-switch the active activity to one that
//! includes the workspace, then focus the window.
//!
//! Pre-1b the dispatcher scoped the id lookup to `windows()` (the active
//! view), so a hidden-activity window silently dropped every
//! `FocusWindow { id }` call. The fix widened the lookup to
//! `windows_all()` and, when the resolved window is not visible under
//! the current activity cursor, calls `Layout::switch_activity` to a
//! target chosen by `Layout::pick_activity_for_hidden_window`.
//!
//! The second test pins the "NOT updated on activity switch"
//! contract: `Mapped::last_focused_activity` is set on focus commit, not
//! when the user flips activities. A naive implementation that bumped
//! the hint on every `switch_activity` would make every cross-activity
//! `FocusWindow { id }` resolve to the current active (tier 1 always
//! firing), defeating the hint semantics.

use jiji_config::Action;

use super::client::ClientId;
use super::fixture::{config_with_two_activities, Fixture};
use crate::layout::DoActionError;

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
fn focus_window_on_hidden_activity_switches_then_focuses() {
    // Two activities: alpha (default-active) and beta.
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();

    // Map the window under alpha. The focus-commit path in `State::refresh`
    // fires `set_focus_timestamp` + `set_last_focused_activity` on this
    // map, so the window's hint is set to alpha.
    map_window(&mut f, client_id, 100, 100);
    f.niri_state().refresh_and_flush_clients();

    // Capture ids while alpha is active.
    let (window_id, alpha_id) = {
        let layout = &f.niri().layout;
        let mut it = layout.windows_all();
        let (_out, mapped) = it
            .next()
            .expect("mapping a window via Fixture must land it in the pool");
        assert!(
            it.next().is_none(),
            "fixture sanity: exactly one window must be mapped",
        );
        (mapped.id().get(), layout.active_activity_id())
    };

    // Switch to beta. The window stays on alpha's workspace — under beta-
    // active it is not reachable via `windows()` (the active view).
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
        "precondition: active activity is beta before FocusWindow dispatches",
    );

    // Dispatch `Action::FocusWindow(id)` under beta-active. The widened
    // dispatcher must:
    //   1. Resolve the window via `windows_all()`.
    //   2. Detect it is hidden.
    //   3. Pick alpha (tier 1: `last_focused_activity` hint was set at map).
    //   4. `switch_activity(alpha)` + `focus_window`.
    f.niri_state()
        .do_action_inner(Action::FocusWindow(window_id), false)
        .expect("FocusWindow(window_id) must succeed: window id resolved before dispatch");
    f.niri_state().refresh_and_flush_clients();

    assert_eq!(
        f.niri().layout.active_activity_id(),
        alpha_id,
        "FocusWindow on a hidden-activity window must auto-switch to the \
         activity that hosts the window",
    );

    assert_eq!(
        f.niri().layout.focus().map(|m| m.id().get()),
        Some(window_id),
        "FocusWindow must focus the target window after the activity switch",
    );
}

#[test]
fn focus_window_does_not_bump_last_focused_activity_on_activity_switch() {
    // Pin the "NOT updated on activity switch" contract: the hint
    // on a `Mapped` is refreshed on focus commit, never on
    // `switch_activity`. A naive bump on switch would break tier-1
    // hint semantics.
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();

    // Map under alpha — set hint to alpha.
    map_window(&mut f, client_id, 100, 100);
    f.niri_state().refresh_and_flush_clients();

    let alpha_id = f.niri().layout.active_activity_id();
    let hint_after_map = {
        let layout = &f.niri().layout;
        let (_out, mapped) = layout
            .windows_all()
            .next()
            .expect("window must be in pool after map");
        mapped.get_last_focused_activity()
    };
    assert_eq!(
        hint_after_map,
        Some(alpha_id),
        "precondition: initial map under alpha sets hint to alpha",
    );

    // Switch to beta. The window is NOT focused under beta (it's on
    // alpha's workspace). The hint must stay at alpha.
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

    let hint_after_switch = {
        let layout = &f.niri().layout;
        let (_out, mapped) = layout
            .windows_all()
            .next()
            .expect("window must still be in pool after switch");
        mapped.get_last_focused_activity()
    };
    assert_eq!(
        hint_after_switch,
        Some(alpha_id),
        "last_focused_activity must NOT be updated on activity switch",
    );
}

#[test]
fn focus_window_unknown_id_returns_err_on_wire() {
    // Pin `Action::FocusWindow { id }` with an id that does not resolve
    // to any pool-owned window: must return
    // `Err(DoActionError::WindowNotFound { id })` from `do_action_inner`.
    // The IPC dispatch path flattens this to the wire envelope
    // `"window not found: id={id}"`.
    //
    // Dispatch via `do_action_inner` directly rather than `do_action`
    // (which silently drops both error arms per the keybind contract).
    // This pins the wire-visible surface, not the keybind silent-drop path.
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));

    // No windows mapped: any id is guaranteed stale.
    const BOGUS_ID: u64 = 999_999;
    let result = f
        .niri_state()
        .do_action_inner(Action::FocusWindow(BOGUS_ID), false);

    assert_eq!(
        result,
        Err(DoActionError::WindowNotFound { id: BOGUS_ID }),
        "FocusWindow with an unknown id must return \
         Err(DoActionError::WindowNotFound) on the wire",
    );
}
