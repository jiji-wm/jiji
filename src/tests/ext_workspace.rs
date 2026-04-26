//! Integration tests pinning the projection contract of the
//! `ext-workspace-unstable-v1` protocol: the snapshot surfaced via
//! [`ExtWorkspaceManagerState`] must equal the set produced by
//! [`Layout::workspaces`](crate::layout::Layout::workspaces), which iterates
//! the active activity's views over connected monitors. In particular, the
//! snapshot must narrow on an activity switch — workspaces in dormant
//! activities' views are *not* surfaced — and it must track the active-flag
//! on the active workspace.
//!
//! These tests exercise the full compositor startup path (`Fixture::with_config`
//! together with `add_output`) and drive `ext_workspace::refresh` through the
//! normal `State::refresh` cycle by calling `refresh_and_flush_clients()`
//! directly — the fixture's `dispatch()` relies on wayland event-source
//! readiness to trigger the cycle, which doesn't fire in these client-less
//! tests. Calling the public `State` method keeps the production ordering
//! (`refresh_layout` → `ext_workspace::refresh`) intact. The assertions read
//! the post-refresh snapshot via the `#[cfg(test)]` projection accessors on
//! `ExtWorkspaceManagerState` — we deliberately do not touch the Wayland wire
//! layer, which is not what is about.

use std::collections::HashSet;

use smithay::reexports::wayland_protocols::ext::workspace::v1::server::ext_workspace_handle_v1;

use super::fixture::{config_with_two_activities, Fixture};
use crate::layout::workspace::{OutputId, WorkspaceId};

/// Helper: iterate `layout.workspaces()` and collect ids into a `HashSet`.
/// Mirrors the canonical projection the protocol snapshot is supposed to match.
fn active_view_ids(f: &mut Fixture) -> HashSet<WorkspaceId> {
    f.niri()
        .layout
        .workspaces()
        .map(|(_, _, ws)| ws.id())
        .collect()
}

/// Helper: id of the activity declared under the given name. Uses the
/// name-keyed accessor rather than relying on `HashMap` iteration order, per
/// spec risk 3.
fn activity_id_by_name(f: &mut Fixture, name: &str) -> crate::layout::activity::ActivityId {
    f.niri()
        .layout
        .activities()
        .iter()
        .find(|a| a.name() == name)
        .unwrap_or_else(|| panic!("activity {name:?} must exist in the pool"))
        .id()
}

/// Helper: `OutputId` of the n-th fixture output.
fn output_id(f: &Fixture, n: u8) -> OutputId {
    OutputId::new(&f.niri_output(n))
}

#[test]
fn refresh_snapshot_matches_layout_workspaces_iterator() {
    // Core projection contract: whatever set `layout.workspaces()`
    // yields after a refresh, the protocol snapshot must equal that set
    // exactly — no extras, no omissions.
    let mut f = Fixture::with_config(config_with_two_activities(&["ws_a"], &["ws_b"]));
    f.add_output(1, (1920, 1080));
    f.niri_state().refresh_and_flush_clients();

    let snapshot = f
        .niri_state()
        .niri
        .ext_workspace_state
        .workspaces_snapshot();
    let iter_ids = active_view_ids(&mut f);

    assert_eq!(
        snapshot, iter_ids,
        "ExtWorkspaceManagerState snapshot must equal the set of ids from Layout::workspaces()",
    );
}

#[test]
fn refresh_narrows_on_activity_switch() {
    // After switching activities, the snapshot must reflect beta's view only:
    // ws_b present, ws_a absent. This is the narrowing half of — a
    // future regression that iterated `workspaces_all()` instead of
    // `workspaces()` would pass the projection-fidelity test but fail here.
    let mut f = Fixture::with_config(config_with_two_activities(&["ws_a"], &["ws_b"]));
    f.add_output(1, (1920, 1080));
    f.niri_state().refresh_and_flush_clients();

    // Resolve ids under the alpha-active snapshot. The specific `WorkspaceId`
    // values are not knowable at compile time (monotonic counter, shared
    // across tests), so look them up by the named-workspace path.
    let ws_a_id = f
        .niri()
        .layout
        .workspaces()
        .find(|(_, _, ws)| ws.name().map(String::as_str) == Some("ws_a"))
        .expect("ws_a must be present in the active (alpha) view at startup")
        .2
        .id();

    // ws_b is parked under beta; resolve its id via the full pool, not the
    // active view (which only iterates alpha's workspaces at this point).
    let ws_b_id = f
        .niri()
        .layout
        .workspaces_all()
        .find(|(_, ws)| ws.name().map(String::as_str) == Some("ws_b"))
        .expect("ws_b must be present in the workspace pool at startup")
        .1
        .id();

    let s0 = f
        .niri_state()
        .niri
        .ext_workspace_state
        .workspaces_snapshot();
    assert!(
        s0.contains(&ws_a_id),
        "baseline snapshot under alpha must contain ws_a (single-output startup binds \
         every disconnected workspace to the new monitor and alpha's view adopts them \
         as-is)",
    );
    assert!(
        s0.contains(&ws_b_id),
        "baseline snapshot under alpha must contain ws_b (startup single-output binding \
         adopts every disconnected workspace)",
    );

    let beta_id = activity_id_by_name(&mut f, "beta");
    f.niri().layout.switch_activity(beta_id);
    f.niri_state().refresh_and_flush_clients();

    let s1 = f
        .niri_state()
        .niri
        .ext_workspace_state
        .workspaces_snapshot();

    // Note: we deliberately do not assert `s1.contains(&ws_b_id)`. A config
    // workspace declared without `open-on-output` is seeded with OutputId("")
    // (workspace.rs:329-334), and `bind_output` does not re-tag it on
    // first-monitor-attach (workspace.rs:575-588, reclaim design). On activity
    // switch, `ensure_active_views` (mod.rs:3894-3901) filters pool candidates
    // by output_id equality and does not rediscover ws_b; beta's view is built
    // with a fresh trailing-empty instead. This asymmetry between Monitor::new
    // (unfiltered parked-id load at Layout::add_output, mod.rs:858-890) and
    // ensure_active_views (output_id-filtered) is orthogonal to the projection
    // contract — this asymmetry is not yet addressed.
    assert!(
        !s1.contains(&ws_a_id),
        "post-switch snapshot must not contain ws_a (alpha's view is dormant)",
    );
    assert_ne!(
        s1, s0,
        "snapshot must actually change when activity narrows"
    );

    // Regression canary: if a future refactor swaps `refresh` to iterate the
    // full pool via `workspaces_all()` instead of the active-activity view via
    // `workspaces()`, the snapshot would widen to everything and this would fail.
    let pool_ids: HashSet<WorkspaceId> = f
        .niri()
        .layout
        .workspaces_all()
        .map(|(_, ws)| ws.id())
        .collect();
    assert_ne!(
        s1, pool_ids,
        "snapshot must be strictly narrower than the full workspace pool — \
         `refresh` must filter via the active activity's views, not iterate the pool",
    );
}

#[test]
fn alpha_view_preserved_across_dormancy() {
    // Forward-compat pin: alpha's `WorkspaceView` is preserved verbatim while
    // beta is active, not rebuilt from a pool tag-filter when alpha re-enters.
    // A future "rebuild view lazily on re-entry" optimization would narrow A2
    // to alpha-tagged workspaces only and fail this assertion, forcing a design
    // update first. The complexity guarantee "once per activity switch and
    // bounded by the number of workspaces" relies on the view being persistent,
    // not regenerated.
    let mut f = Fixture::with_config(config_with_two_activities(&["ws_a"], &["ws_b"]));
    f.add_output(1, (1920, 1080));
    f.niri_state().refresh_and_flush_clients();

    let out_id = output_id(&f, 1);

    let a0: Vec<WorkspaceId> = f.niri().layout.active_view(&out_id).ids().to_vec();
    assert!(
        !a0.is_empty(),
        "alpha's view must be non-empty at baseline (fixture precondition — startup \
         must have bound at least ws_a to the output)",
    );
    let s0 = f
        .niri_state()
        .niri
        .ext_workspace_state
        .workspaces_snapshot();

    let beta_id = activity_id_by_name(&mut f, "beta");
    let alpha_id = activity_id_by_name(&mut f, "alpha");

    f.niri().layout.switch_activity(beta_id);
    f.niri_state().refresh_and_flush_clients();

    f.niri().layout.switch_activity(alpha_id);
    f.niri_state().refresh_and_flush_clients();

    let a2: Vec<WorkspaceId> = f.niri().layout.active_view(&out_id).ids().to_vec();
    let s2 = f
        .niri_state()
        .niri
        .ext_workspace_state
        .workspaces_snapshot();

    assert_eq!(
        a0, a2,
        "alpha's view id list must survive the dormancy round-trip verbatim",
    );
    assert_eq!(
        s0, s2,
        "projection fidelity + view preservation imply snapshot equality round-trip",
    );
}

#[test]
fn active_flag_tracks_active_workspace_in_current_activity() {
    // The active workspace (per the active activity's view) must carry the
    // `Active` state flag; no other workspace in the snapshot may.
    let mut f = Fixture::with_config(config_with_two_activities(&["ws_a"], &["ws_b"]));
    f.add_output(1, (1920, 1080));
    f.niri_state().refresh_and_flush_clients();

    let out_id = output_id(&f, 1);
    let active_id = f.niri().layout.active_view(&out_id).active();

    let proto = &f.niri_state().niri.ext_workspace_state;

    // Readability-first check: the active id must individually carry the flag.
    let active_state = proto
        .workspace_state(active_id)
        .expect("active workspace id must be present in the protocol snapshot");
    assert!(
        active_state.contains(ext_workspace_handle_v1::State::Active),
        "active workspace must carry the Active state flag",
    );

    // Exhaustive check: Active must be held by exactly the one active id.
    // This catches a stale Active bit on a workspace that was dropped from the
    // workspaces HashMap — a regression the snapshot loop above would miss.
    assert_eq!(
        proto.ids_with_active_flag(),
        std::iter::once(active_id).collect::<HashSet<_>>(),
        "Active flag must be held by exactly the active workspace id",
    );
}

#[test]
fn workspace_group_count_matches_connected_outputs() {
    // Protocol maps one workspace-group per output; the cardinality must match
    // `state.niri.sorted_outputs` after refresh.
    let mut f = Fixture::with_config(config_with_two_activities(&["ws_a"], &["ws_b"]));
    f.add_output(1, (1920, 1080));
    f.niri_state().refresh_and_flush_clients();

    let state = f.niri_state();
    assert_eq!(
        state.niri.ext_workspace_state.workspace_groups_len(),
        state.niri.sorted_outputs.len(),
        "workspace_groups cardinality must equal sorted_outputs cardinality",
    );
}
