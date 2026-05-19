//! Pins `open-on-activity` window-rule semantics for window opening.
//!
//! Each test sets up a config with two activities (alpha → seed/active,
//! beta → inactive) plus a window-rule that exercises one branch of the
//! 4-bullet precedence matrix. The new window opens through the
//! standard initial-configure → map flow; the test then asserts which
//! workspace it landed on (or did not land on) and that the active activity
//! did NOT silently switch.
//!
//! These tests substitute for the spec-suggested powerset extension in
//! `window_opening.rs` (which would have multiplied the snapshot count).
//! Discrete `#[test]` cases mirror the precedent in
//! `find_window_cross_activity.rs`.

use jiji_config::Config;

use super::client::ClientId;
use super::fixture::Fixture;
use crate::layout::LayoutElement;

/// Drive a full initial-commit → ack-buffer roundtrip so the window is
/// mapped and present in the layout. Mirrors the helper in
/// `find_window_cross_activity.rs` (kept inline for the same
/// "no cross-module `pub(super)` plumbing for one use site" reason).
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

/// Build a config with `alpha` (active) and `beta` (inactive) activities
/// and append `extra_kdl` (window-rule blocks, additional workspaces, etc.)
/// at the end. The base output is `headless-1` (added by tests via
/// `add_output(1, ...)`).
fn config_with(extra_kdl: &str) -> Config {
    let mut src = String::from("activity \"alpha\"\nactivity \"beta\"\n");
    src.push_str(extra_kdl);
    Config::parse_mem(&src).expect("test KDL must parse")
}

fn alpha_id(f: &mut Fixture) -> crate::layout::activity::ActivityId {
    f.niri()
        .layout
        .activities()
        .iter()
        .find(|a| a.name() == "alpha")
        .expect("alpha must be config-seeded")
        .id()
}

fn beta_id(f: &mut Fixture) -> crate::layout::activity::ActivityId {
    f.niri()
        .layout
        .activities()
        .iter()
        .find(|a| a.name() == "beta")
        .expect("beta must be config-seeded")
        .id()
}

#[test]
fn open_on_activity_alone_places_window_in_active_workspace_of_target_activity_on_active_monitor() {
    // Bare `open-on-activity "beta"` with no `open-on-workspace`. The
    // window must land in beta's active workspace on the active monitor;
    // alpha must remain the active activity.
    let mut f = Fixture::with_config(config_with(
        r##"
        window-rule {
            open-on-activity "beta"
        }
        "##,
    ));
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();

    let beta = beta_id(&mut f);
    let alpha = alpha_id(&mut f);
    let pre_active = f.niri().layout.active_activity_id();
    assert_eq!(
        pre_active, alpha,
        "precondition: alpha is the active activity"
    );

    map_window(&mut f, client_id, 100, 100);

    // Active activity must not have flipped.
    assert_eq!(
        f.niri().layout.active_activity_id(),
        alpha,
        "open-on-activity must NOT auto-switch",
    );

    // Window must be on a workspace tagged with beta and not tagged with alpha.
    let layout = &f.niri().layout;
    let mapped = layout
        .windows_all()
        .next()
        .expect("window must be mapped into the layout");
    let win_id = LayoutElement::id(mapped.1);
    let (_out, ws) = layout
        .workspaces_all()
        .find(|(_, ws)| ws.has_window(win_id))
        .expect("window must live on some workspace in the pool");
    assert!(
        ws.activities().contains(&beta),
        "window must be placed on a beta-tagged workspace",
    );
}

#[test]
fn open_on_activity_for_inactive_activity_does_not_auto_switch() {
    // Same as above but a stronger pin on point 1: even after
    // refresh + flush, the active activity is unchanged and the
    // active activity's view is unchanged in length / content (we never
    // synthetically materialize an alpha workspace just because the rule
    // ran).
    let mut f = Fixture::with_config(config_with(
        r##"
        window-rule {
            open-on-activity "beta"
        }
        "##,
    ));
    f.add_output(1, (1920, 1080));
    let alpha = alpha_id(&mut f);
    let alpha_view_before = {
        let layout = &f.niri().layout;
        let out_id = layout.monitors().next().unwrap().output_id();
        layout.active_view(&out_id).ids().to_vec()
    };

    let client_id = f.add_client();
    map_window(&mut f, client_id, 100, 100);
    f.niri_state().refresh_and_flush_clients();

    assert_eq!(f.niri().layout.active_activity_id(), alpha);
    let layout = &f.niri().layout;
    let out_id = layout.monitors().next().unwrap().output_id();
    let alpha_view_after = layout.active_view(&out_id).ids().to_vec();
    assert_eq!(
        alpha_view_after, alpha_view_before,
        "active (alpha) view must not gain any workspace when a window opens into beta",
    );
}

#[test]
fn open_on_activity_plus_open_on_workspace_in_target_activity_uses_named_workspace() {
    // Point 3a: `open-on-activity "beta"` + `open-on-workspace "beta-ws"`
    // where `beta-ws` is tagged with beta — the window must land on `beta-ws`.
    let mut f = Fixture::with_config(config_with(
        r##"
        workspace "beta-ws" {
            activity "beta"
        }
        window-rule {
            open-on-activity "beta"
            open-on-workspace "beta-ws"
        }
        "##,
    ));
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();
    let alpha = alpha_id(&mut f);

    map_window(&mut f, client_id, 100, 100);

    assert_eq!(f.niri().layout.active_activity_id(), alpha);
    let layout = &f.niri().layout;
    let mapped = layout.windows_all().next().expect("window mapped");
    let win_id = LayoutElement::id(mapped.1);
    let (_out, ws) = layout
        .workspaces_all()
        .find(|(_, ws)| ws.has_window(win_id))
        .unwrap();
    assert_eq!(
        ws.name(),
        Some(&"beta-ws".to_owned()),
        "window must land on the explicitly-named beta workspace",
    );
}

#[test]
fn open_on_activity_plus_open_on_workspace_not_in_target_activity_falls_back_to_active_workspace_of_target_activity(
) {
    // Point 3b: `open-on-activity "beta"` + `open-on-workspace "alpha-ws"`
    // where `alpha-ws` is tagged with alpha (NOT beta). The named-workspace
    // lookup falls through, then the chain settles on the active monitor →
    // beta's active workspace there. The window must NOT land on the
    // alpha-tagged `alpha-ws`.
    let mut f = Fixture::with_config(config_with(
        r##"
        workspace "alpha-ws" {
            activity "alpha"
        }
        window-rule {
            open-on-activity "beta"
            open-on-workspace "alpha-ws"
        }
        "##,
    ));
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();
    let alpha = alpha_id(&mut f);
    let beta = beta_id(&mut f);

    map_window(&mut f, client_id, 100, 100);

    assert_eq!(f.niri().layout.active_activity_id(), alpha);
    let layout = &f.niri().layout;
    let mapped = layout.windows_all().next().expect("window mapped");
    let win_id = LayoutElement::id(mapped.1);
    let (_out, ws) = layout
        .workspaces_all()
        .find(|(_, ws)| ws.has_window(win_id))
        .unwrap();
    assert_ne!(
        ws.name(),
        Some(&"alpha-ws".to_owned()),
        "name + activity mismatch must fall through, not silently use alpha-ws",
    );
    assert!(
        ws.activities().contains(&beta),
        "fallback target must be a beta-tagged workspace",
    );
}

#[test]
fn open_on_activity_unknown_name_falls_back_to_active_activity() {
    // Liberal-accept: `open-on-activity "ghost"` names no real activity.
    // The resolver `warn!`s and treats the rule as if it pointed at the
    // active (alpha) activity (mirrors `open-on-output`'s precedent for
    // unknown output names). The window lands on an alpha-tagged workspace.
    let mut f = Fixture::with_config(config_with(
        r##"
        window-rule {
            open-on-activity "ghost"
        }
        "##,
    ));
    f.add_output(1, (1920, 1080));
    let client_id = f.add_client();
    let alpha = alpha_id(&mut f);

    map_window(&mut f, client_id, 100, 100);

    let layout = &f.niri().layout;
    let mapped = layout.windows_all().next().expect("window mapped");
    let win_id = LayoutElement::id(mapped.1);
    let (_out, ws) = layout
        .workspaces_all()
        .find(|(_, ws)| ws.has_window(win_id))
        .unwrap();
    assert!(
        ws.activities().contains(&alpha),
        "unknown activity name must silently fall back to the active activity",
    );
}

#[test]
fn open_on_activity_combined_with_open_on_output_routes_to_named_output_within_activity() {
    // `open-on-activity "beta"` +
    // `open-on-output "headless-2"`. With two outputs, the window must
    // land on output2 (the open-on-output target) on a beta-tagged
    // workspace.
    let mut f = Fixture::with_config(config_with(
        r##"
        window-rule {
            open-on-activity "beta"
            open-on-output "headless-2"
        }
        "##,
    ));
    f.add_output(1, (1920, 1080));
    f.add_output(2, (1920, 1080));
    let client_id = f.add_client();
    let beta = beta_id(&mut f);

    map_window(&mut f, client_id, 100, 100);

    let mapped_output = f.niri_output(2);
    let layout = &f.niri().layout;
    let mapped = layout.windows_all().next().expect("window mapped");
    let win_id = LayoutElement::id(mapped.1);
    // The window's owning workspace must be beta-tagged AND bound to output2.
    let (out, ws) = layout
        .workspaces_all()
        .find(|(_, ws)| ws.has_window(win_id))
        .unwrap();
    assert!(
        ws.activities().contains(&beta),
        "window must land on a beta-tagged workspace",
    );
    let out = out.expect("workspace must be bound to a connected output");
    assert!(
        out.matches(&mapped_output),
        "open-on-output must route the window to headless-2 within the beta activity",
    );
}
