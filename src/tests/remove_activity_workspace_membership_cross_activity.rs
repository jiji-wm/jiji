//! Pins the two distinct workspace-membership branches inside
//! [`Layout::remove_activity`](crate::layout::Layout::remove_activity):
//! the **shrink** branch (a workspace shared with at least one other activity
//! has its `activities` set pruned) and the **destroy** branch (a workspace
//! exclusively bound to the removed activity, unnamed and empty, is dropped
//! from the pool and from every surviving activity's views).
//!
//! ## Why two #[test] fns
//!
//! The two branches live at adjacent code sites (`mod.rs:5019-5036` for
//! destroy, `mod.rs:5042-5050` for shrink) but a regression that conflates them
//! — for example, "destroy whenever the removed activity is one of the
//! workspace's activities", ignoring the multi-activity guard — would orphan
//! the shared workspace in Test 1 and leak its window pool entry. Splitting the
//! coverage keeps each branch's discriminating assertion attributable to its
//! own failure mode.

use niri_config::{Action, ActivityReference, Config, WorkspaceReference};

use super::client::ClientId;
use super::fixture::Fixture;
use crate::layout::workspace::{OutputId, WorkspaceId};

/// Drive a full initial-commit → ack-buffer roundtrip so the window is
/// mapped and present in the layout. Mirrors the helpers in
/// `event_stream_replay_cross_activity.rs` and other cross-activity tests.
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
fn remove_activity_shrinks_shared_workspace_membership() {
    // Config: alpha is the only config-declared activity (so it is the seed
    // active activity). beta is created at runtime so RemoveActivity is
    // permitted to operate on it (config-declared activities reject removal).
    let kdl = "\
        activity \"alpha\"\n\
    ";
    let config = Config::parse_mem(kdl).expect("test KDL must parse");
    let mut f = Fixture::with_config(config);

    f.add_output(1, (1920, 1080));
    f.niri_state().refresh_and_flush_clients();

    let out1 = f.niri_output(1);
    let out1_id = OutputId::new(&out1);

    // Resolve alpha's id and snapshot alpha's active workspace id (W) before
    // any membership mutation. `active_view` returns alpha's view of out1
    // because alpha is active and out1 is the bound output; `ids()[0]` is
    // alpha's seed workspace at this point (no trailing-empty has spawned
    // yet because no window has mapped).
    let alpha_id = f
        .niri()
        .layout
        .activities()
        .iter()
        .find(|a| a.name() == "alpha")
        .expect("alpha must be present in the config-seeded activity pool")
        .id();
    let w_id = f.niri().layout.active_view(&out1_id).ids()[0];

    // Create beta at runtime — does not flip active.
    f.niri_state()
        .do_action_inner(Action::CreateActivity("beta".to_string()), false)
        .expect("CreateActivity(beta) must succeed on a unique name");
    f.niri_state().refresh_and_flush_clients();

    let beta_id = f
        .niri()
        .layout
        .activities()
        .iter()
        .find(|a| a.name() == "beta")
        .expect("beta must be present after CreateActivity")
        .id();

    // Multi-activity assignment: W now belongs to {alpha, beta}. The order
    // [alpha, beta] keeps alpha first — matching the seed-active activity.
    f.niri_state()
        .do_action_inner(
            Action::SetWorkspaceActivities(
                Some(WorkspaceReference::Id(w_id.get())),
                vec![
                    ActivityReference::Name("alpha".into()),
                    ActivityReference::Name("beta".into()),
                ],
            ),
            false,
        )
        .expect(
            "SetWorkspaceActivities with valid workspace ref and existing activity \
             names must succeed",
        );
    f.niri_state().refresh_and_flush_clients();

    // Map a window onto alpha's active workspace W. After this, alpha's view
    // of out1 will contain W plus an auto-spawned trailing empty workspace,
    // so `pre_alpha_view_position` must be captured AFTER the map.
    let client_id = f.add_client();
    map_window(&mut f, client_id, 100, 100);
    f.niri_state().refresh_and_flush_clients();

    // Pre-removal sanity: W has activities = {alpha, beta}, the window is on
    // W, and W appears in alpha's view of out1.
    let pre_window_count = f.niri().layout.windows_all().count();
    assert_eq!(
        pre_window_count, 1,
        "pre-removal sanity: exactly one window should be in the pool",
    );
    {
        let layout = &f.niri().layout;
        let w = layout
            .workspaces_all()
            .find(|(_, ws)| ws.id() == w_id)
            .expect("W must exist in the pool")
            .1;
        assert_eq!(
            w.activities().len(),
            2,
            "pre-removal sanity: W must have exactly two activities (alpha, beta)",
        );
        assert!(
            w.activities().contains(&alpha_id) && w.activities().contains(&beta_id),
            "pre-removal sanity: W's activities must be exactly {{alpha, beta}}",
        );
    }
    let mut window_ws_id: Option<WorkspaceId> = None;
    f.niri()
        .layout
        .with_windows_all(|_win, _out, ws_id, _layout| {
            window_ws_id = ws_id;
        });
    assert_eq!(
        window_ws_id,
        Some(w_id),
        "pre-removal sanity: the mapped window must be on W",
    );

    // Snapshot alpha's view position of W after mapping (and after a
    // possible trailing-empty spawn). RemoveActivity must not perturb this.
    let pre_alpha_view_position = f
        .niri()
        .layout
        .active_view(&out1_id)
        .position_of(w_id)
        .expect("alpha's view of out1 must contain W after mapping");

    // Drop beta. W has activities = {alpha, beta}, so the SHRINK branch
    // fires: W remains in the pool, the window stays put, but W's
    // activities set drops beta.
    f.niri_state()
        .do_action_inner(
            Action::RemoveActivity(ActivityReference::Name("beta".into())),
            false,
        )
        .expect(
            "RemoveActivity(beta) must succeed: runtime, not last, no windows on \
             exclusive workspaces, no hard-block in effect",
        );
    f.niri_state().refresh_and_flush_clients();

    // (a) W remains in the pool — catches a regression that orphaned a
    //     shared workspace by routing through destroy.
    {
        let layout = &f.niri().layout;
        assert!(
            layout.workspaces_all().any(|(_, ws)| ws.id() == w_id),
            "shrink: W must still be present in the pool after RemoveActivity(beta)",
        );

        // (b) Window pool count unchanged — same regression's side effect.
        assert_eq!(
            layout.windows_all().count(),
            pre_window_count,
            "shrink: window pool count must be unchanged after RemoveActivity(beta)",
        );

        // (c) W now has activities = {alpha} — beta pruned.
        let w = layout
            .workspaces_all()
            .find(|(_, ws)| ws.id() == w_id)
            .expect("W must remain in the pool after shrink")
            .1;
        assert_eq!(
            w.activities().len(),
            1,
            "shrink: W's activities set must shrink to exactly one entry",
        );
        assert!(
            w.activities().contains(&alpha_id),
            "shrink: W's surviving activity membership must be {{alpha}}",
        );
        assert!(
            !w.activities().contains(&beta_id),
            "shrink: beta must be removed from W's activities set",
        );

        // (d) beta is gone from the activity pool.
        assert!(
            layout.activities().get(beta_id).is_none(),
            "shrink: beta must be removed from the activity pool",
        );

        // (e) alpha's view of out1 still contains W at its pre-removal
        //     position. Snapshot-based — the literal index would be brittle
        //     if a trailing-empty existed before W (it does not in this
        //     config, but be explicit).
    }
    let mut window_ws_id_post: Option<WorkspaceId> = None;
    f.niri()
        .layout
        .with_windows_all(|_win, _out, ws_id, _layout| {
            window_ws_id_post = ws_id;
        });
    assert_eq!(
        window_ws_id_post,
        Some(w_id),
        "shrink: the window must still resolve to W after RemoveActivity(beta)",
    );

    let post_alpha_view_position = f
        .niri()
        .layout
        .active_view(&out1_id)
        .position_of(w_id)
        .expect("shrink: alpha's view of out1 must still contain W after RemoveActivity(beta)");
    assert_eq!(
        post_alpha_view_position, pre_alpha_view_position,
        "shrink: W's position in alpha's view of out1 must be preserved across the shrink",
    );
}

#[test]
fn remove_activity_destroys_exclusive_unnamed_empty_workspace() {
    // Config: alpha is the only config-declared activity. gamma is created
    // and switched into at runtime; `ensure_view_for` materializes a fresh
    // unnamed-empty workspace bound exclusively to gamma. RemoveActivity
    // then cascades the active cursor back to alpha (via previous_id) and
    // destroys gamma's exclusive workspace.
    let kdl = "\
        activity \"alpha\"\n\
    ";
    let config = Config::parse_mem(kdl).expect("test KDL must parse");
    let mut f = Fixture::with_config(config);

    f.add_output(1, (1920, 1080));
    f.niri_state().refresh_and_flush_clients();

    let out1 = f.niri_output(1);
    let out1_id = OutputId::new(&out1);

    let alpha_id = f
        .niri()
        .layout
        .activities()
        .iter()
        .find(|a| a.name() == "alpha")
        .expect("alpha must be present in the config-seeded activity pool")
        .id();

    // Create gamma at runtime.
    f.niri_state()
        .do_action_inner(Action::CreateActivity("gamma".to_string()), false)
        .expect("CreateActivity(gamma) must succeed on a unique name");
    f.niri_state().refresh_and_flush_clients();

    let gamma_id = f
        .niri()
        .layout
        .activities()
        .iter()
        .find(|a| a.name() == "gamma")
        .expect("gamma must be present after CreateActivity")
        .id();

    // Switch into gamma — `ensure_view_for(gamma, out1_id)` materializes a
    // fresh unnamed-empty workspace bound to gamma.
    f.niri_state()
        .do_action_inner(
            Action::SwitchActivity(ActivityReference::Name("gamma".into())),
            false,
        )
        .expect("SwitchActivity(gamma) must succeed: target exists, no hard-block in effect");
    f.niri_state().refresh_and_flush_clients();

    // Capture gamma's freshly materialized workspace id. At this point the
    // active view is gamma's view of out1, which has exactly one entry (the
    // fresh-branch path of `ensure_view_for`). Captured *before*
    // `RemoveActivity` because the destroy branch removes it from the pool —
    // post-removal we'd have nothing to assert against.
    let gamma_view_ids = f.niri().layout.active_view(&out1_id).ids().to_vec();
    assert_eq!(
        gamma_view_ids.len(),
        1,
        "post-switch sanity: gamma's view of out1 must have exactly one workspace \
         (ensure_view_for fresh branch)",
    );
    let gamma_ws_id = gamma_view_ids[0];

    // Pre-removal sanity: the destroy branch's preconditions hold.
    {
        let layout = &f.niri().layout;
        let ws = layout
            .workspaces_all()
            .find(|(_, ws)| ws.id() == gamma_ws_id)
            .expect("gamma's fresh workspace must exist in the pool")
            .1;
        assert_eq!(
            ws.activities().len(),
            1,
            "pre-removal sanity: gamma's fresh workspace must have exactly one activity",
        );
        assert!(
            ws.activities().contains(&gamma_id),
            "pre-removal sanity: gamma's fresh workspace must be exclusively bound to gamma",
        );
        assert!(
            !ws.has_windows(),
            "pre-removal sanity: gamma's fresh workspace must be empty",
        );
        assert!(
            ws.name().is_none(),
            "pre-removal sanity: gamma's fresh workspace must be unnamed",
        );
        assert_eq!(
            layout.activities().active_id(),
            gamma_id,
            "pre-removal sanity: gamma must be the active activity",
        );
        assert_eq!(
            layout.activities().previous_id(),
            Some(alpha_id),
            "pre-removal sanity: previous_id must be alpha after switching from alpha to gamma",
        );
    }

    // Remove gamma. Cascade fires (gamma is active → alpha via previous_id),
    // then the destroy branch removes gamma's exclusive unnamed-empty
    // workspace from both the pool and every activity's views.
    f.niri_state()
        .do_action_inner(
            Action::RemoveActivity(ActivityReference::Name("gamma".into())),
            false,
        )
        .expect(
            "RemoveActivity(gamma) must succeed: runtime, not last, no windows on \
             exclusive workspaces, no hard-block in effect",
        );
    f.niri_state().refresh_and_flush_clients();

    // (a) gamma's workspace is gone from the pool — the destroy contract.
    {
        let layout = &f.niri().layout;
        assert!(
            !layout
                .workspaces_all()
                .any(|(_, ws)| ws.id() == gamma_ws_id),
            "destroy: gamma's exclusive workspace must be removed from the pool",
        );

        // (b) gamma is gone from the activity pool.
        assert!(
            layout.activities().get(gamma_id).is_none(),
            "destroy: gamma must be removed from the activity pool",
        );

        // (c) Cascade landed on alpha (the previous_id at remove time).
        assert_eq!(
            layout.activities().active_id(),
            alpha_id,
            "destroy: cascade must repoint the active cursor to alpha",
        );

        // (d) previous_id must not point at the now-freed ActivityId. A
        //     regression that flipped active_id correctly but left
        //     previous_id dangling at gamma_id would silently survive here
        //     and later panic on SwitchActivityPrevious.
        assert_ne!(
            layout.activities().previous_id(),
            Some(gamma_id),
            "destroy: cascade must scrub the dead activity from previous_id \
             (else SwitchActivityPrevious would later panic on a freed ActivityId)",
        );

        // (e) No surviving activity's view contains the dead id. Catches a
        //     regression that destroyed the workspace from the pool but
        //     forgot to scrub it from per-activity views.
        for activity in layout.activities().iter() {
            for (_oid, view) in activity.views().iter() {
                assert!(
                    !view.ids().contains(&gamma_ws_id),
                    "destroy: no surviving activity's view may contain the dead workspace id; \
                     activity {:?} has a view containing gamma_ws_id={:?}",
                    activity.id(),
                    gamma_ws_id,
                );
            }
        }
    }
}
