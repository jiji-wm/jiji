//! Pins `State::reload_config` end-to-end across an activity-cascade reload:
//! a config drops the active activity (alpha) and one of its named workspaces
//! (`ws_a`), keeps another activity (beta) and its workspace (`ws_b`).
//!
//! Two load-bearing properties this test pins:
//!
//! 1. **The orphan-rebind production path** (`Layout::reconcile_activities_on_reload_remove`
//!    orphan-rebind leg). Without it, `Workspace::new_with_config_no_outputs` seeds named config
//!    workspaces with the empty-string `OutputId` sentinel, `Monitor::new`'s lift loop pulls every
//!    disconnected workspace into the seed-active activity's view at first-monitor-attach
//!    regardless of `activities` tagging, and the cascade target's `ensure_all_activity_views`
//!    cannot reclaim such an orphan (sentinel ≠ real output id). On reload-drop-active the orphan's
//!    only anchoring view evaporates with `self.activities.remove`, and
//!    `Layout::verify_invariants`' pool-keys-equal-union check would panic. The named `ws_a`
//!    workspace declared under the dropped alpha is what drives the prewalk → `unname_workspace` →
//!    reconcile-add precondition assert; replacing it with an unnamed runtime workspace silently
//!    drops that leg. Review-stop on any change that swaps `ws_a` out for an unnamed substitute.
//!
//! 2. **The cascade-target arm exercised**: `previous_id == None` → first-declaration-order
//!    non-remove-set survivor (= beta), per the cascade-target resolution in
//!    `Layout::reconcile_activities_on_reload_remove`. The reload comes from the seed-active state
//!    with no prior `switch_activity`, so `previous_id` is `None`. This pins the `or_else(||
//!    ...find...)` branch, distinct from the `previous.filter(...)` branch covered by the
//!    layout-suite cluster.

use jiji_config::Config;
use jiji_ipc::state::{EventStreamState, EventStreamStatePart as _};
use jiji_ipc::Event;

use super::fixture::Fixture;
use crate::layout::workspace::OutputId;

#[test]
fn reload_config_removing_active_activity_cascades_and_recreates_names() {
    // Initial config: alpha (seed-active) + beta, one named workspace each.
    let initial_kdl = "\
        activity \"alpha\"\n\
        activity \"beta\"\n\
        workspace \"ws_a\" { activity \"alpha\"; }\n\
        workspace \"ws_b\" { activity \"beta\"; }\n\
    ";
    let initial_config = Config::parse_mem(initial_kdl).expect("initial KDL must parse");
    let mut f = Fixture::with_config(initial_config);

    f.add_output(1, (1920, 1080));
    f.niri_state().refresh_and_flush_clients();

    let out1 = f.niri_output(1);
    let out1_id = OutputId::new(&out1);

    // Capture pre-reload ids — `ws_a_id` and `alpha_id` will disappear after
    // the reload, so they must be captured here to assert on. Mirrors the
    // capture-before-removal precedent in
    // `remove_activity_workspace_membership_cross_activity.rs`.
    let (alpha_id, beta_id, ws_a_id, ws_b_id) = {
        let layout = &f.niri().layout;
        let alpha_id = layout
            .activities()
            .iter()
            .find(|a| a.name() == "alpha")
            .expect("alpha must be config-declared from the initial KDL")
            .id();
        let beta_id = layout
            .activities()
            .iter()
            .find(|a| a.name() == "beta")
            .expect("beta must be config-declared from the initial KDL")
            .id();
        let ws_a_id = layout
            .workspaces_all()
            .find(|(_, ws)| ws.name() == Some(&"ws_a".to_owned()))
            .expect("ws_a must be in the workspace pool")
            .1
            .id();
        let ws_b_id = layout
            .workspaces_all()
            .find(|(_, ws)| ws.name() == Some(&"ws_b".to_owned()))
            .expect("ws_b must be in the workspace pool")
            .1
            .id();
        (alpha_id, beta_id, ws_a_id, ws_b_id)
    };

    // Pre-reload precondition: alpha is the seed-active activity, previous
    // is None (no prior switch_activity).
    assert_eq!(
        f.niri().layout.active_activity_id(),
        alpha_id,
        "precondition: alpha is the seed-active activity",
    );
    assert!(
        f.niri().layout.activities().previous_id().is_none(),
        "precondition: previous_id is None (no prior switch_activity)",
    );

    // Install the event tap AFTER the seed-burst so we capture only the
    // reload's deltas. The Layer-3 ordering pin reads from this buffer.
    f.install_event_tap();

    // Reload: drop alpha (and its workspace `ws_a`), keep beta + ws_b.
    let reload_kdl = "\
        activity \"beta\"\n\
        workspace \"ws_b\" { activity \"beta\"; }\n\
    ";
    let reload_config = Config::parse_mem(reload_kdl).expect("reload KDL must parse");
    f.reload_config(reload_config);
    f.niri_state().refresh_and_flush_clients();

    let layout = &f.niri().layout;

    // (a) Cascade target arm: previous_id == None → first-declaration-order
    // non-remove-set survivor (beta). This is the cascade-target branch
    // distinct from `previous.filter(...)`.
    assert_eq!(
        layout.active_activity_id(),
        beta_id,
        "cascade target arm: previous_id == None → first-declaration-order survivor (beta)",
    );

    // (b) Pool: ws_a gone from workspaces pool; ws_b survives;
    // alpha gone from activities pool; beta present + still config-declared;
    // no surviving activity's view contains ws_a_id.
    assert!(
        !layout.workspaces_all().any(|(_, ws)| ws.id() == ws_a_id),
        "ws_a (alpha-exclusive) must be destroyed by the reload",
    );
    assert!(
        layout.workspaces_all().any(|(_, ws)| ws.id() == ws_b_id),
        "ws_b (beta-exclusive) must survive the reload",
    );
    assert!(
        !layout.activities().contains(alpha_id),
        "alpha must be dropped from the activity pool",
    );
    let beta = layout
        .activities()
        .get(beta_id)
        .expect("beta must remain in the activity pool");
    assert!(
        beta.is_config_declared(),
        "beta must remain config-declared after reload",
    );
    for activity in layout.activities().iter() {
        for view in activity.views().values() {
            assert!(
                !view.ids().contains(&ws_a_id),
                "no surviving activity's view may contain ws_a_id={ws_a_id:?}",
            );
        }
    }

    // (c) Per-monitor view + IPC replicate snapshot.
    assert_eq!(
        layout.active_view(&out1_id).active(),
        ws_b_id,
        "post-reload: monitor's active workspace must be ws_b (beta's only workspace)",
    );

    let post_replicate = f.replicate_event_stream_state();
    let mut state = EventStreamState::default();
    for ev in post_replicate.iter().cloned() {
        let _ = state.apply(ev);
    }
    let active_count = state
        .activities
        .activities
        .values()
        .filter(|a| a.is_active)
        .count();
    assert_eq!(
        active_count, 1,
        "exactly one activity must carry is_active=true in the replay snapshot",
    );
    let beta_replay = state
        .activities
        .activities
        .get(&beta_id.get())
        .expect("beta must be in the replay activities map");
    assert!(
        beta_replay.is_active,
        "beta must be is_active=true in the replay snapshot",
    );

    // (d) Optional event-ordering layer: within the cascade tick,
    // `ActivityRemoved { id: alpha_id }` precedes
    // `ActivitySwitched { previous_id: Some(alpha_id) }`. Encodes the
    // structure-before-state contract on `Event::ActivityRemoved`.
    let captured = f.drain_events();
    let activity_removed_alpha_idx = captured.iter().position(|e| match e {
        Event::ActivityRemoved { id } => *id == alpha_id.get(),
        _ => false,
    });
    let activity_switched_idx = captured
        .iter()
        .position(|e| matches!(e, Event::ActivitySwitched { .. }));
    if let (Some(removed_idx), Some(switched_idx)) =
        (activity_removed_alpha_idx, activity_switched_idx)
    {
        assert!(
            removed_idx < switched_idx,
            "ActivityRemoved {{ alpha }} (idx {removed_idx}) must precede ActivitySwitched \
             (idx {switched_idx}) within the cascade tick",
        );
    }
}

#[test]
fn reload_reassigning_workspace_activity_syncs_views_end_to_end() {
    // A reload that neither adds nor removes an activity but *reassigns* a named
    // workspace from the active activity (alpha) to a dormant one (beta) exercises
    // the membership↔view sweep wired into the reload-add reconcile end-to-end:
    // the config reset narrows `ws_m`'s membership to {beta}, leaving its view
    // entry stranded in alpha; the sweep must install it into beta's view and
    // drop it from alpha's. A regression that skipped the sweep would leave
    // `ws_m` with beta membership but no beta view (and a stale alpha entry),
    // tripping `verify_invariants`' pool==union pass.
    let initial_kdl = "\
        activity \"alpha\"\n\
        activity \"beta\"\n\
        workspace \"ws_m\" { activity \"alpha\"; }\n\
    ";
    let initial_config = Config::parse_mem(initial_kdl).expect("initial KDL must parse");
    let mut f = Fixture::with_config(initial_config);

    f.add_output(1, (1920, 1080));
    f.niri_state().refresh_and_flush_clients();

    let out1 = f.niri_output(1);
    let out1_id = OutputId::new(&out1);

    let (alpha_id, beta_id, ws_m_id) = {
        let layout = &f.niri().layout;
        let alpha_id = layout
            .activities()
            .iter()
            .find(|a| a.name() == "alpha")
            .expect("alpha config-declared")
            .id();
        let beta_id = layout
            .activities()
            .iter()
            .find(|a| a.name() == "beta")
            .expect("beta config-declared")
            .id();
        let ws_m_id = layout
            .workspaces_all()
            .find(|(_, ws)| ws.name() == Some(&"ws_m".to_owned()))
            .expect("ws_m in the pool")
            .1
            .id();
        (alpha_id, beta_id, ws_m_id)
    };

    // Precondition: alpha active, ws_m in alpha's view and NOT beta's.
    {
        let layout = &f.niri().layout;
        assert_eq!(layout.active_activity_id(), alpha_id, "alpha seed-active");
        assert!(
            layout
                .activities()
                .get(alpha_id)
                .expect("alpha live")
                .views()
                .get(&out1_id)
                .expect("alpha view")
                .ids()
                .contains(&ws_m_id),
            "precondition: ws_m sits in alpha's view",
        );
    }

    // Reload: reassign ws_m to beta; both activities stay declared.
    let reload_kdl = "\
        activity \"alpha\"\n\
        activity \"beta\"\n\
        workspace \"ws_m\" { activity \"beta\"; }\n\
    ";
    let reload_config = Config::parse_mem(reload_kdl).expect("reload KDL must parse");
    f.reload_config(reload_config);
    f.niri_state().refresh_and_flush_clients();

    let layout = &f.niri().layout;

    // Membership reset to {beta}.
    assert_eq!(
        layout
            .workspaces_all()
            .find(|(_, ws)| ws.id() == ws_m_id)
            .expect("ws_m survives")
            .1
            .activities(),
        &std::collections::HashSet::from([beta_id]),
        "config reset must narrow ws_m's membership to {{beta}}",
    );

    // Sweep installed ws_m into beta's view and dropped it from alpha's.
    assert!(
        layout
            .activities()
            .get(beta_id)
            .expect("beta live")
            .views()
            .get(&out1_id)
            .expect("beta view")
            .ids()
            .contains(&ws_m_id),
        "sweep must install ws_m into beta's view",
    );
    assert!(
        !layout
            .activities()
            .get(alpha_id)
            .expect("alpha live")
            .views()
            .get(&out1_id)
            .expect("alpha view")
            .ids()
            .contains(&ws_m_id),
        "sweep must drop the stale ws_m entry from alpha's view",
    );
}
