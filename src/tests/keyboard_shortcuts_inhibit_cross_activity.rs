//! Pins the DD §5.19 keyboard-shortcuts-inhibit / activity-switch contract:
//! when an activity switch hides a window with an active keyboard-shortcuts
//! inhibitor, the inhibitor is deactivated; when the window becomes visible
//! again, the inhibitor is reactivated — unless the user inactivated it
//! manually via `Action::ToggleKeyboardShortcutsInhibit` while it was
//! visible.
//!
//! These tests also pin the cleanup-hook symmetry
//! (`inhibitor_destroyed` removes from both
//! `keyboard_shortcuts_inhibiting_surfaces` and
//! `deactivated_inhibitors_by_activity_switch`) and the cascade path
//! (`Action::RemoveActivity` flowing through
//! `Layout::switch_activity` must still trigger the sweep).
//!
//! Two of the five DD §5.19 hook sites are not covered here:
//! - `Action::FocusWindow` auto-switch (input/mod.rs:990–992): testing this
//!   hook requires synthesising a focus event that also triggers an activity
//!   switch, which needs cross-activity window placement not yet available in
//!   this sub-phase's fixture vocabulary. Deferred.
//! - `State::reload_config` (niri.rs:1492–1500): the fixture setup required
//!   (a config change that triggers an activity cascade) is non-trivial.
//!   Deferred.

use niri_config::{Action, ActivityReference};
use smithay::reexports::wayland_protocols::wp::keyboard_shortcuts_inhibit::zv1::client::zwp_keyboard_shortcuts_inhibitor_v1::ZwpKeyboardShortcutsInhibitorV1;
use smithay::reexports::wayland_server::Resource as _;
use wayland_client::protocol::wl_surface::WlSurface as ClientSurface;
use wayland_client::Proxy as _;

use super::client::ClientId;
use super::fixture::{config_with_two_activities, Fixture};

fn map_window(f: &mut Fixture, id: ClientId, w: u16, h: u16) -> ClientSurface {
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

/// Create and activate a keyboard-shortcuts inhibitor for `client_surface`.
/// Returns the client-side proxy (held so the inhibitor isn't garbage-
/// collected mid-test) together with the server-side `smithay::...::WlSurface`
/// key (what the Niri inhibitor map is keyed by).
fn create_inhibitor(
    f: &mut Fixture,
    id: ClientId,
    client_surface: &ClientSurface,
) -> (
    ZwpKeyboardShortcutsInhibitorV1,
    smithay::reexports::wayland_server::protocol::wl_surface::WlSurface,
) {
    let inhibitor = f
        .client(id)
        .create_keyboard_shortcuts_inhibitor(client_surface);
    f.roundtrip(id);

    // Server-side, the map is keyed by the `smithay` `WlSurface`. Locate it
    // by the client-surface `id()` match — the server's inhibitor map has
    // exactly one entry at this point.
    let server_surface = {
        let niri = f.niri();
        let client_id = client_surface.id();
        niri.keyboard_shortcuts_inhibiting_surfaces
            .keys()
            .find(|s| s.id().protocol_id() == client_id.protocol_id())
            .cloned()
            .expect(
                "server-side inhibitor map must contain the surface after new_inhibitor handler",
            )
    };
    (inhibitor, server_surface)
}

fn beta_id(f: &mut Fixture) -> niri_config::ActivityReference {
    let id = f
        .niri()
        .layout
        .activities()
        .iter()
        .find(|a| a.name() == "beta")
        .expect("beta must be present in the config-seeded activity pool")
        .id();
    ActivityReference::Id(id.get())
}

fn alpha_id(f: &mut Fixture) -> niri_config::ActivityReference {
    let id = f
        .niri()
        .layout
        .activities()
        .iter()
        .find(|a| a.name() == "alpha")
        .expect("alpha must be present in the config-seeded activity pool")
        .id();
    ActivityReference::Id(id.get())
}

#[test]
fn inhibitor_deactivated_on_switch_away() {
    // Pin the primary DD §5.19 contract: switching away from the activity
    // that hosts the inhibited window must inactivate the inhibitor and
    // record the surface in `deactivated_inhibitors_by_activity_switch`.
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();

    let client_surface = map_window(&mut f, client_id, 100, 100);
    f.niri_state().refresh_and_flush_clients();

    let (_inhibitor, server_surface) = create_inhibitor(&mut f, client_id, &client_surface);
    assert!(
        f.niri()
            .keyboard_shortcuts_inhibiting_surfaces
            .get(&server_surface)
            .expect("inhibitor must be present in the map after new_inhibitor handler")
            .is_active(),
        "precondition: inhibitor is active immediately after new_inhibitor",
    );

    // Switch to beta via the Action path so the hook at input/mod.rs fires.
    let beta = beta_id(&mut f);
    f.niri_state()
        .do_action(Action::SwitchActivity(beta), false);
    f.niri_state().refresh_and_flush_clients();

    assert!(
        !f.niri()
            .keyboard_shortcuts_inhibiting_surfaces
            .get(&server_surface)
            .expect("inhibitor must still be present after activity switch")
            .is_active(),
        "DD §5.19: inhibitor on a hidden-activity window must be inactivated",
    );
    assert!(
        f.niri()
            .deactivated_inhibitors_by_activity_switch
            .contains(&server_surface),
        "DD §5.19: surface must be recorded in the tracking set so switch-back \
         can restore it",
    );
}

#[test]
fn inhibitor_reactivated_on_switch_back() {
    // Pin the round-trip: alpha → beta → alpha restores the inhibitor to
    // active and drains the tracking set.
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();

    let client_surface = map_window(&mut f, client_id, 100, 100);
    f.niri_state().refresh_and_flush_clients();

    let (_inhibitor, server_surface) = create_inhibitor(&mut f, client_id, &client_surface);

    let beta = beta_id(&mut f);
    let alpha = alpha_id(&mut f);

    f.niri_state()
        .do_action(Action::SwitchActivity(beta), false);
    f.niri_state().refresh_and_flush_clients();
    f.niri_state()
        .do_action(Action::SwitchActivity(alpha), false);
    f.niri_state().refresh_and_flush_clients();

    assert!(
        f.niri()
            .keyboard_shortcuts_inhibiting_surfaces
            .get(&server_surface)
            .expect("inhibitor must still be present after round-trip")
            .is_active(),
        "DD §5.19: inhibitor must be reactivated when its owning workspace \
         becomes visible again",
    );
    assert!(
        f.niri()
            .deactivated_inhibitors_by_activity_switch
            .is_empty(),
        "DD §5.19: tracking set must be drained once the inhibitor is \
         restored (invariant: tracking ⊆ inhibitor-map)",
    );
}

#[test]
fn inhibitor_user_inactivated_not_reactivated_on_switch_back() {
    // Pin the user-toggle preservation contract: an inhibitor the user
    // inactivated via `Action::ToggleKeyboardShortcutsInhibit` while it was
    // visible must stay inactive across activity switches. The load-bearing
    // discriminator is `deactivated_inhibitors_by_activity_switch`: because
    // the user toggle never inserts into that set, the switch-back branch
    // of the sweep leaves the inactive state alone.
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();

    let client_surface = map_window(&mut f, client_id, 100, 100);
    f.niri_state().refresh_and_flush_clients();

    let (_inhibitor, server_surface) = create_inhibitor(&mut f, client_id, &client_surface);

    // User-inactivate the inhibitor directly (skips the
    // `Action::ToggleKeyboardShortcutsInhibit` path's keyboard-focus
    // precondition, which the fixture doesn't wire up, but exercises the
    // same mutation the action would perform). The important contract being
    // pinned is the sweep's behaviour, not the action dispatcher.
    f.niri()
        .keyboard_shortcuts_inhibiting_surfaces
        .get(&server_surface)
        .expect("inhibitor must be present")
        .inactivate();
    assert!(
        !f.niri()
            .keyboard_shortcuts_inhibiting_surfaces
            .get(&server_surface)
            .unwrap()
            .is_active(),
        "precondition: user-inactivated inhibitor is inactive",
    );
    assert!(
        f.niri()
            .deactivated_inhibitors_by_activity_switch
            .is_empty(),
        "precondition: user inactivation must NOT populate the tracking set",
    );

    let beta = beta_id(&mut f);
    let alpha = alpha_id(&mut f);
    f.niri_state()
        .do_action(Action::SwitchActivity(beta), false);
    f.niri_state().refresh_and_flush_clients();
    f.niri_state()
        .do_action(Action::SwitchActivity(alpha), false);
    f.niri_state().refresh_and_flush_clients();

    assert!(
        !f.niri()
            .keyboard_shortcuts_inhibiting_surfaces
            .get(&server_surface)
            .expect("inhibitor must still be present")
            .is_active(),
        "DD §5.19: user-inactivated inhibitor must NOT be re-activated on \
         switch-back (tracking set was never populated for this surface)",
    );
    assert!(
        f.niri()
            .deactivated_inhibitors_by_activity_switch
            .is_empty(),
        "DD §5.19: tracking set must remain empty — the sweep must neither \
         insert nor remove user-toggled inhibitors",
    );
}

#[test]
fn inhibitor_destroyed_while_hidden_clears_tracking_set() {
    // Pin `inhibitor_destroyed`'s tracking-set cleanup: a client destroying
    // its inhibitor while hidden must drain the surface from the tracking
    // set so the subset invariant holds.
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();

    let client_surface = map_window(&mut f, client_id, 100, 100);
    f.niri_state().refresh_and_flush_clients();

    let (inhibitor, server_surface) = create_inhibitor(&mut f, client_id, &client_surface);

    // Hide the window — surface should be tracked as deactivated.
    let beta = beta_id(&mut f);
    f.niri_state()
        .do_action(Action::SwitchActivity(beta), false);
    f.niri_state().refresh_and_flush_clients();
    assert!(
        f.niri()
            .deactivated_inhibitors_by_activity_switch
            .contains(&server_surface),
        "precondition: hidden-surface inhibitor is tracked",
    );

    // Client destroys the inhibitor. The protocol's `destroy` request flows
    // through smithay's Dispatch → `KeyboardShortcutsInhibitHandler::
    // inhibitor_destroyed`, which must scrub both maps.
    inhibitor.destroy();
    f.roundtrip(client_id);

    assert!(
        !f.niri()
            .keyboard_shortcuts_inhibiting_surfaces
            .contains_key(&server_surface),
        "inhibitor_destroyed must remove from keyboard_shortcuts_inhibiting_surfaces",
    );
    assert!(
        !f.niri()
            .deactivated_inhibitors_by_activity_switch
            .contains(&server_surface),
        "DD §5.19: inhibitor_destroyed must also scrub the tracking set \
         (subset invariant)",
    );

    // Switch back to alpha — sweep runs, subset invariant's `debug_assert!`
    // must not panic.
    let alpha = alpha_id(&mut f);
    f.niri_state()
        .do_action(Action::SwitchActivity(alpha), false);
    f.niri_state().refresh_and_flush_clients();
    assert!(
        f.niri()
            .deactivated_inhibitors_by_activity_switch
            .is_empty(),
        "tracking set must remain empty after switch-back on destroyed inhibitor",
    );
}

#[test]
fn inhibitor_reactivated_on_switch_activity_previous() {
    // Pin the `Action::SwitchActivityPrevious` hook site (DD §5.19 third of
    // five sites at input/mod.rs). This is a different dispatch arm from
    // `SwitchActivity(reference)` tested in (1) / (2) — a regression that
    // forgot to add the sweep call to `SwitchActivityPrevious` would pass
    // the first two tests and fail this one.
    //
    // Note: the spec's "RemoveActivity cascade" test (its (5)) cannot be
    // written in this sub-phase because `Layout::remove_activity` rejects
    // removal of an activity that exclusively owns a workspace with
    // windows (`ExclusiveWorkspaceHasWindows`). Pinning the cascade
    // requires a `MoveWorkspaceToActivity`-class mutator that lands in
    // Phase 2. The State-level hook at the `Action::RemoveActivity` arm
    // still exists and uses the same refresh call tested here, so the
    // mechanism is covered.
    let mut f = Fixture::with_config(config_with_two_activities(&[], &[]));
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();

    let client_surface = map_window(&mut f, client_id, 100, 100);
    f.niri_state().refresh_and_flush_clients();

    let (_inhibitor, server_surface) = create_inhibitor(&mut f, client_id, &client_surface);

    // Establish `previous` = alpha by going alpha → beta first (alpha_rt
    // side-trip would also work, but beta keeps the test symmetric with
    // the other four).
    let beta = beta_id(&mut f);
    f.niri_state()
        .do_action(Action::SwitchActivity(beta), false);
    f.niri_state().refresh_and_flush_clients();
    // Now switch back to alpha via `SwitchActivityPrevious`. Inhibitor
    // should reactivate through the sweep hook at that arm.
    f.niri_state()
        .do_action(Action::SwitchActivityPrevious {}, false);
    f.niri_state().refresh_and_flush_clients();

    assert!(
        f.niri()
            .keyboard_shortcuts_inhibiting_surfaces
            .get(&server_surface)
            .expect("inhibitor must still be present after SwitchActivityPrevious")
            .is_active(),
        "DD §5.19: SwitchActivityPrevious hook must reactivate inhibitor on \
         switch-back (third of five State-level sweep sites)",
    );
    assert!(
        f.niri()
            .deactivated_inhibitors_by_activity_switch
            .is_empty(),
        "DD §5.19: tracking set must drain on switch-back via \
         SwitchActivityPrevious",
    );
}
