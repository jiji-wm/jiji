//! Pins the configure-vs-map race in the `open-on-activity` window-rule
//! handler when the target workspace is destroyed between
//! `send_initial_configure` (which captures `target_workspace_id` into
//! `InitialConfigureState::Configured`) and the post-buffer-attach commit
//! (which consumes that captured id in
//! `handlers::compositor::CompositorHandler::commit`).
//!
//! The chain in `compositor.rs` settles `workspace_id` via:
//!   (1) `target_workspace_id.filter(contains_key)` — trust the
//!       configure-time pin if the workspace is still in the pool;
//!   (2) name-based re-resolution scoped to `target_activity` (or the
//!       global pool when there is no target activity); else
//!   (3) `None`, falling through to `AddWindowTarget::Output`.
//!
//! This test wedges a `RemoveActivity(gamma)` between the configure and the
//! map so tier (1)'s `.filter(contains_key)` returns `None`. Gamma is
//! materialized at runtime and never named, so its auto-materialized empty
//! workspace has `workspace_name = None` — tier (2) short-circuits. Net
//! `workspace_id = None`; with `output1` still alive the chain falls to
//! `AddWindowTarget::Output(output1)`, which routes the window to the
//! active activity's (alpha's) view on output1.
//!
//! Discriminating regressions:
//!   * Removing the `.filter(contains_key)` guard at `compositor.rs:146-147` (such that a stale
//!     `target_workspace_id` reaches `Layout::add_window`) would either panic on the dead
//!     `WorkspaceId` or silently drop the window — caught by the explicit
//!     `windows_all().next().expect(...)` map-success assert.
//!   * Without the mid-flow `!contains_key` oracle, the test could pass without the destruction
//!     step actually running — e.g. if a future refactor short-circuited the destruction such that
//!     the workspace was never removed from the pool, the test would land on alpha for the wrong
//!     reason (no race occurred). The oracle pins "destruction occurred between configure and map,"
//!     NOT "the race-resolution chain was actually exercised."

use jiji_config::{Action, ActivityReference, Config};
use wayland_client::protocol::wl_surface::WlSurface;

use super::client::ClientId;
use super::fixture::Fixture;
use crate::layout::LayoutElement;

/// First half of the map flow: create a surface and commit it (no buffer
/// attached yet). After driving a roundtrip, the server has run
/// `send_initial_configure`, which resolves the window's target activity /
/// workspace and stores them in `InitialConfigureState::Configured`. The
/// returned `WlSurface` is the client-side handle the caller passes back
/// into [`attach_and_map`] once any race-window action has run.
///
/// Split out from the canonical `map_window` helper used by the sibling
/// `*_cross_activity.rs` tests so the test body can wedge a
/// `RemoveActivity` action between configure and map — specifically between
/// the configure-time capture and the chain at `src/handlers/compositor.rs:139-155`.
fn commit_initial(f: &mut Fixture, id: ClientId) -> WlSurface {
    let window = f.client(id).create_window();
    let surface = window.surface.clone();
    window.commit();
    f.roundtrip(id);
    surface
}

/// Second half of the map flow: attach a buffer, set the size, ack the
/// last configure, commit, and roundtrip. The server-side
/// `CompositorHandler::commit` then runs the chain at
/// `src/handlers/compositor.rs:139-155` that this test exists to pin
/// (`target_workspace_id.filter(contains_key)` → name-based re-resolution
/// → `AddWindowTarget::Output` fallthrough).
fn attach_and_map(f: &mut Fixture, id: ClientId, surface: &WlSurface, w: u16, h: u16) {
    let window = f.client(id).window(surface);
    window.attach_new_buffer();
    window.set_size(w, h);
    window.ack_last_and_commit();
    f.roundtrip(id);
}

#[test]
fn open_on_activity_falls_through_when_target_workspace_destroyed_between_configure_and_map() {
    // Config: alpha (seed/active) + beta (declared so the activity pool is
    // not single-element, but otherwise unused). The window-rule names
    // `gamma`, which is NOT declared in the config — it is created at
    // runtime via `Action::CreateActivity` so that `send_initial_configure`
    // resolves the target after gamma's view has been materialized on the
    // active output.
    let kdl = "\
        activity \"alpha\"\n\
        activity \"beta\"\n\
        window-rule {\n\
            open-on-activity \"gamma\"\n\
        }\n\
    ";
    let config = Config::parse_mem(kdl).expect("test KDL must parse");
    let mut f = Fixture::with_config(config);

    f.add_output(1, (1920, 1080));
    f.niri_state().refresh_and_flush_clients();

    // Create gamma at runtime so the rule has a real activity to resolve
    // against at configure time. Capture `gamma_id` and the active
    // output's `output_id` before the map flow so the mid-flow oracle
    // can name the materialized view directly.
    f.niri_state()
        .do_action_inner(Action::CreateActivity("gamma".into()), false)
        .expect("CreateActivity must succeed on a unique name");
    f.niri_state().refresh_and_flush_clients();

    let gamma_id = f
        .niri()
        .layout
        .activities()
        .iter()
        .find(|a| a.name() == "gamma")
        .expect("gamma must exist after CreateActivity")
        .id();
    let output_id = f
        .niri()
        .layout
        .monitors()
        .next()
        .expect("output1 must be present")
        .output_id();
    let alpha_id = f
        .niri()
        .layout
        .activities()
        .iter()
        .find(|a| a.name() == "alpha")
        .expect("alpha must be config-seeded")
        .id();

    // Configure half: surface + commit + roundtrip. After this returns
    // the server has run `send_initial_configure`, which (per the
    // `open-on-activity "gamma"` rule) materialized gamma's view on
    // output1 and pinned `target_workspace_id` in the
    // `InitialConfigureState::Configured` state.
    let client_id = f.add_client();
    let surface = commit_initial(&mut f, client_id);

    // Mid-flow oracle (race-occurred pin, part 1): the target workspace
    // exists in the pool right now. If `view_in_activity_or_materialize`
    // has not yet run, this assert would fail before we even reach the
    // destruction step — surfacing a configure-side regression cleanly.
    let target_ws_id = f
        .niri()
        .layout
        .view_for(gamma_id, &output_id)
        .expect("gamma's view on output1 must be materialized by send_initial_configure")
        .active();
    // Defense in depth: verify_invariants enforces this in debug builds
    // (every WorkspaceView.active() is a pool key), but the explicit assert
    // documents the precondition for the oracle below.
    assert!(
        f.niri().layout.workspace_pool().contains_key(&target_ws_id),
        "precondition: gamma's active workspace must be live in the pool \
         at the configure→map boundary",
    );

    // Wedge: destroy gamma between configure and map. Gamma is runtime
    // (not config-pinned), not the last activity, has no windows and no
    // exclusive named workspaces — `RemoveActivity` succeeds and removes
    // gamma's auto-materialized empty workspace via the cross-activity
    // destroy path.
    f.niri_state()
        .do_action_inner(
            Action::RemoveActivity(ActivityReference::Name("gamma".into())),
            false,
        )
        .expect(
            "RemoveActivity(gamma) must succeed: runtime, not last, \
             no windows, unnamed exclusive workspace",
        );
    f.niri_state().refresh_and_flush_clients();

    // Mid-flow oracle (destruction-occurred pin, part 2): the target
    // workspace is gone from the pool. Without this assert the test could
    // pass without the destruction step actually running — e.g. if a
    // future refactor short-circuited the destruction such that the
    // workspace was never removed from the pool, the test would land on
    // alpha for the wrong reason (no race occurred). This pins
    // "destruction occurred between configure and map," NOT "the
    // race-resolution chain was actually exercised."
    assert!(
        !f.niri().layout.workspace_pool().contains_key(&target_ws_id),
        "race-occurred pin: gamma's workspace must be destroyed before map",
    );

    // Map half: attach buffer + ack + commit + roundtrip. The
    // server-side commit handler runs the chain — tier (1) returns
    // `None` (`!contains_key`); tier (2) returns `None`
    // (`workspace_name` is `None` for the auto-materialized empty);
    // net `workspace_id = None`. `output1` is alive, so the target is
    // `AddWindowTarget::Output(output1)` and the window lands on
    // alpha's active workspace there.
    attach_and_map(&mut f, client_id, &surface, 100, 100);

    // Capture the connected output handle first; `niri_output` borrows
    // `&self` and would conflict with the `&mut self` borrow taken by
    // `niri()` below.
    let mapped_output = f.niri_output(1);

    // Map success: a regression that silently dropped the window
    // (e.g. a future `?` short-circuit) is caught here, BEFORE the
    // workspace-shape asserts that would otherwise panic on iterator
    // emptiness.
    //
    // The window's owning workspace must be alpha-tagged and bound to
    // output1. Mirrors the alpha-fall-through assertion shape from
    // `window_opening_cross_activity::open_on_activity_unknown_name_falls_back_to_active_activity`.
    let layout = &f.niri().layout;
    let mapped = layout
        .windows_all()
        .next()
        .expect("window must be mapped into the layout despite the race");
    let win_id = LayoutElement::id(mapped.1);
    let (out, ws) = layout
        .workspaces_all()
        .find(|(_, ws)| ws.has_window(win_id))
        .expect("mapped window must live on a workspace in the pool");
    assert!(
        ws.activities().contains(&alpha_id),
        "fallback target must be an alpha-tagged workspace \
         (active activity at map time)",
    );
    let out = out.expect("workspace must be bound to a connected output");
    // Defense in depth: with one output, this is trivially satisfied —
    // the assert exists to guard against an unexpected disconnection
    // mid-flow that would route the window to a phantom output handle.
    assert!(
        out.matches(&mapped_output),
        "fallback target must be bound to output1 (the only connected output)",
    );

    // The cascade must NOT have flipped the active activity. Gamma was
    // inactive at destruction time (alpha is the seed), so
    // `RemoveActivity` does not exercise the active-cursor cascade.
    assert_eq!(
        f.niri().layout.active_activity_id(),
        alpha_id,
        "active activity must remain alpha; the configure→destroy→map race \
         must not bleed into the active cursor",
    );
}
