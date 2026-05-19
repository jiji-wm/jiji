//! Pins the multi-activity fan-out contract on
//! [`Layout::move_workspace_to_output_by_id`](crate::layout::Layout::move_workspace_to_output_by_id):
//! when a workspace belongs to multiple activities, moving it across outputs must update
//! *every* activity's view of the source and target outputs, not just the active activity's.
//!
//! ## Why this is a discriminating regression test
//!
//! The prior `move_workspace_to_output_by_id` body migrated only the active activity's
//! views (via `remove_workspace_from_monitor` / `insert_workspace_onto_monitor`), so a
//! dormant activity that previously materialized a view of the source output kept the
//! workspace's stale id under the wrong `OutputId` key. The next switch into that activity
//! surfaced the workspace under the source output even though its `Workspace.output_id`
//! had been rebound to the target. The fan-out fix added in `move_workspace_to_output_by_id`
//! mirrors the source-side `set_workspace_activities` precedent (single-entry drop vs.
//! `remove_at`) plus a target-side insert one slot before the trailing-empty bookend.
//!
//! The test is **discriminating** rather than merely existential: assertions (c) and (e)
//! would fail under a regression that re-introduced the active-only migration. A
//! pre-move sanity check (assertion 3) ensures beta's view of out1 actually contains
//! `ws_shared` before the move — without that canary, "beta.views[out1] does not contain
//! ws_shared" would be vacuously true if warm-up never materialized beta's view of out1
//! and the test would discriminate nothing.

use jiji_config::Config;

use super::fixture::Fixture;
use crate::layout::workspace::OutputId;

#[test]
fn move_workspace_to_monitor_fans_out_to_dormant_activity_views() {
    // Two activities (alpha first → seed/active by default) sharing one workspace
    // `ws_shared`. Inline KDL so the two-activity-shared-workspace shape is
    // self-contained — `config_with_two_activities` cannot express a workspace
    // that lists multiple `activity` lines. The explicit `open-on-output
    // "headless-1"` is load-bearing: without it, `Workspace::new_with_config_no_outputs`
    // seeds `output_id` to `Some(OutputId(""))` (the unbound-preference sentinel),
    // and `Workspace::bind_output` only refreshes `output_id` when it already
    // matches — the empty-string sentinel matches no output, so it survives binding.
    // `ensure_view_for(beta, out1_id)` filters the pool on
    // `ws.output_id() == Some(&out1_id)`; with the sentinel intact, ws_shared is
    // not lifted into beta's view of out1 and the warm-up below cannot populate
    // the dormant view that the discriminating assertions need. The sentinel
    // sentinel semantics is not yet fully addressed;
    // pinning the connector here gives ws_shared a `bind_output`-matching
    // `OutputId("headless-1")` that gets refreshed to canonical
    // `OutputId("niri headless 1")` on first bind.
    let kdl = "\
        activity \"alpha\"\n\
        activity \"beta\"\n\
        workspace \"ws_shared\" {\n\
            activity \"alpha\";\n\
            activity \"beta\";\n\
            open-on-output \"headless-1\";\n\
        }\n\
    ";
    let config = Config::parse_mem(kdl).expect("test KDL must parse");
    let mut f = Fixture::with_config(config);

    f.add_output(1, (1920, 1080));
    f.add_output(2, (1920, 1080));
    f.niri_state().refresh_and_flush_clients();

    let out1 = f.niri_output(1);
    let out2 = f.niri_output(2);
    let out1_id = OutputId::new(&out1);
    let out2_id = OutputId::new(&out2);

    // Resolve activity ids and ws_shared's id.
    let alpha_id = f
        .niri()
        .layout
        .activities()
        .iter()
        .find(|a| a.name() == "alpha")
        .expect("alpha must be present in the config-seeded activity pool")
        .id();
    let beta_id = f
        .niri()
        .layout
        .activities()
        .iter()
        .find(|a| a.name() == "beta")
        .expect("beta must be present in the config-seeded activity pool")
        .id();
    let ws_shared_id = f
        .niri()
        .layout
        .workspaces_all()
        .find(|(_, ws)| ws.name() == Some(&"ws_shared".to_owned()))
        .expect("ws_shared must exist in the workspace pool from the config")
        .1
        .id();

    // Warm up beta's view at out1: switch into beta and back to alpha.
    // ensure_view_for materializes beta's view of out1 the first time beta is
    // active, so post-warm-up beta carries a dormant view of out1 that
    // contains ws_shared. Without this warm-up, beta has no out1 view and the
    // dormant-target-no-view leave-absent branch is the one exercised — also
    // valid coverage, but the discriminating case requires a populated dormant
    // view at the source.
    f.niri().layout.switch_activity(beta_id);
    f.niri_state().refresh_and_flush_clients();
    f.niri().layout.switch_activity(alpha_id);
    f.niri_state().refresh_and_flush_clients();

    // Pre-move sanity check: beta's dormant view of out1 must actually contain
    // ws_shared. If this fails, the warm-up sequence is wrong and every
    // "beta.views[out1] does not contain ws_shared" assertion below would be
    // vacuously true. Investigate warm-up before suspecting fan-out.
    {
        let layout = &f.niri().layout;
        let beta = layout
            .activities()
            .get(beta_id)
            .expect("beta_id must be live in the pool");
        let beta_out1 = beta.views().get(&out1_id).unwrap_or_else(|| {
            panic!(
                "warm-up must have materialized beta's view of out1; views = {:?}",
                beta.views().keys().collect::<Vec<_>>()
            )
        });
        assert!(
            beta_out1.ids().contains(&ws_shared_id),
            "pre-move sanity: beta's view of out1 must contain ws_shared after warm-up — \
             without this, the discriminating assertions on dormant beta views are vacuous; \
             beta's view of out1 ids: {:?}, ws_shared_id: {:?}",
            beta_out1.ids(),
            ws_shared_id,
        );
        assert!(
            beta.views().contains_key(&out2_id),
            "pre-move sanity: warm-up must materialize beta's view of out2 (else fan-out's \
             target-side error message blames fan-out for a warm-up bug)",
        );
    }

    // Locate ws_shared's position in alpha's (active) view of out1 so we
    // pass the right `old_idx` to the by-id mover. The fixture default-actives
    // alpha and out1 (output index 1 → primary), so the active view is
    // alpha's view of out1.
    let old_idx = f
        .niri()
        .layout
        .active_view(&out1_id)
        .position_of(ws_shared_id)
        .expect("alpha's view of out1 must contain ws_shared");

    // Execute the cross-output move. `old_output: None` selects the active
    // monitor's view as the source; `&out2` is the target.
    f.niri()
        .layout
        .move_workspace_to_output_by_id(old_idx, None, &out2);
    f.niri_state().refresh_and_flush_clients();

    // (a) ws_shared's bound output is now out2.
    {
        let layout = &f.niri().layout;
        let ws = layout
            .workspaces_all()
            .find(|(_, ws)| ws.id() == ws_shared_id)
            .expect("ws_shared must remain in the pool")
            .1;
        assert_eq!(
            ws.output_id(),
            Some(&out2_id),
            "ws_shared.output_id must be retargeted to out2 by the move",
        );
    }

    // (b) – (e): per-activity view fan-out. Snapshot reads are taken under a
    // single layout borrow per activity to keep the assertion ordering tight
    // and avoid stale views across the multi-step inspection.
    {
        let layout = &f.niri().layout;
        let alpha = layout
            .activities()
            .get(alpha_id)
            .expect("alpha_id must be live in the pool");
        let beta = layout
            .activities()
            .get(beta_id)
            .expect("beta_id must be live in the pool");

        // (b) alpha's view of out1 must no longer contain ws_shared (active
        // activity's source-side migration — covered by the pre-fix code path
        // already, asserted here as a non-regression baseline).
        if let Some(alpha_out1) = alpha.views().get(&out1_id) {
            assert!(
                !alpha_out1.ids().contains(&ws_shared_id),
                "alpha's view of out1 must not contain ws_shared after the move",
            );
        }

        // (c) DISCRIMINATING — beta's dormant view of out1 must no longer
        // contain ws_shared. Pre-fix, this assertion fails: the dormant view
        // retains the stale id. The pre-move sanity check above guarantees
        // beta's out1 view existed and contained ws_shared, so this is not
        // vacuous.
        if let Some(beta_out1) = beta.views().get(&out1_id) {
            assert!(
                !beta_out1.ids().contains(&ws_shared_id),
                "beta's dormant view of out1 must not contain ws_shared after fan-out — \
                 discriminating regression assertion",
            );
        }

        // (d) alpha's view of out2 must contain ws_shared (active-side target
        // migration; non-regression baseline like (b)).
        let alpha_out2 = alpha
            .views()
            .get(&out2_id)
            .expect("alpha's view of out2 must exist after the active-side move");
        assert!(
            alpha_out2.ids().contains(&ws_shared_id),
            "alpha's view of out2 must contain ws_shared after the move",
        );

        // (e) DISCRIMINATING — beta's view of out2 must contain ws_shared.
        // Pre-fix, beta either has no view of out2 (no fan-out at all) or its
        // view of out2 lacks ws_shared. Either failure mode is caught here.
        let beta_out2 = beta
            .views()
            .get(&out2_id)
            .expect("beta's view of out2 must be materialized by the fan-out");
        assert!(
            beta_out2.ids().contains(&ws_shared_id),
            "beta's view of out2 must contain ws_shared after fan-out — \
             discriminating regression assertion",
        );
    }

    // (f) End-to-end: switching into beta surfaces ws_shared under out2.
    // This is the user-visible symptom the fan-out fix ultimately repairs.
    f.niri().layout.switch_activity(beta_id);
    f.niri_state().refresh_and_flush_clients();
    {
        let layout = &f.niri().layout;
        assert_eq!(
            layout.activities().active_id(),
            beta_id,
            "post-switch sanity: switch_activity(beta_id) must actually flip the active activity",
        );
        let active_out2 = layout.active_view(&out2_id);
        assert!(
            active_out2.ids().contains(&ws_shared_id),
            "after switching into beta, the active view of out2 must contain ws_shared",
        );
    }
}
