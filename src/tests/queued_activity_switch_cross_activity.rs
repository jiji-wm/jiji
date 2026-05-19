//! Pins the hard-block / activity-switch queue drain end-to-end.
//!
//! The unit tests at `src/ipc/server.rs` cover the registry primitives in
//! isolation (FIFO, depth-1 admission, closed-receiver prune, re-block FIFO
//! pin). What they cannot cover — because they sit below the real
//! `Layout::is_activity_switch_hard_blocked` gate, the real
//! `do_action_inner` dispatch, and the real `State::refresh` drain site — is
//! the integration plane: an action arrives while a hard block is armed, the
//! gate rejects the synchronous dispatch, the IPC site parks a waiter, and
//! the drain woken by `refresh` performs the action and signals the waiter
//! after the block clears.
//!
//! Two `#[test]` fns:
//!
//! 1. [`hard_blocked_switch_activity_queues_drains_on_unblock`] — primary end-to-end pin: armed
//!    `interactive_move` rejects synchronous `Action::SwitchActivity`, the queued waiter stays
//!    parked across one refresh tick (load-bearing: registry occupied, `active_activity_id`
//!    unchanged, `try_recv` returns `Empty` not `Closed`), the move ends, the next refresh tick
//!    drains the waiter, the activity flips, and the waiter receives `Ok(Ok(()))`.
//! 2. [`depth_one_admission_rejects_second_enqueue_per_connection`] — depth-1 admission: a second
//!    enqueue under the same `IpcConnId` is rejected with the literal `"request already queued"`
//!    error string, while a different connection enqueueing under its own id succeeds. Both pending
//!    waiters drain on unblock.
//!
//! These tests dispatch through `do_action_inner` and the new
//! `IpcServer::test_simulate_blocked_request` helper rather than driving
//! real Unix-socket clients: that machinery is irrelevant to the registry /
//! drain contract, and routing through it would require setting up an async
//! `process` task in the test fixture.

use std::time::{Duration, Instant};

use jiji_config::{Action, ActivityReference};
use jiji_ipc::ActivityReferenceArg;
use smithay::utils::Point;

use super::fixture::{config_with_two_activities, Fixture};
use crate::ipc::server::{IpcConnId, IpcServer};
use crate::layout::{ActivitySwitchBlock, DoActionError, DoActionOutcome};

/// Borrow the [`IpcServer`] from the fixture, panicking with a clear message
/// if the fixture was constructed without one.
fn ipc(f: &mut Fixture) -> &IpcServer {
    f.niri()
        .ipc_server
        .as_ref()
        .expect("test fixture must construct IpcServer; check Fixture::with_config")
}

/// Wait up to `timeout` for a single value to arrive on `rx`.
///
/// Panics with `panic_msg` if the channel closes before delivering a value
/// (sender dropped — itself a regression), or if the deadline expires without
/// a value arriving (drain did not signal — signals the drain contract
/// regressed).
fn recv_or_timeout<T>(rx: &async_channel::Receiver<T>, timeout: Duration, panic_msg: &str) -> T {
    let deadline = Instant::now() + timeout;
    loop {
        match rx.try_recv() {
            Ok(val) => return val,
            Err(async_channel::TryRecvError::Closed) => {
                panic!("{panic_msg} — channel closed (sender dropped before signalling)");
            }
            Err(async_channel::TryRecvError::Empty) => {
                if Instant::now() >= deadline {
                    panic!("{panic_msg} — timed out after {timeout:?} (drain did not signal; hard-block clear path regressed?)");
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }
}

/// Add an output and create a single mapped window on the seed activity
/// (alpha) so [`crate::layout::Layout::interactive_move_begin`] has a real
/// target to arm against. Returns the smithay `Window` id used as
/// [`crate::window::Mapped`]'s `LayoutElement::Id`.
fn set_up_with_window(f: &mut Fixture) -> smithay::desktop::Window {
    f.add_output(1, (1920, 1080));
    let id = f.add_client();
    let window = f.client(id).create_window();
    let surface = window.surface.clone();
    window.commit();
    f.roundtrip(id);

    let window = f.client(id).window(&surface);
    window.attach_new_buffer();
    window.set_size(100, 100);
    window.ack_last_and_commit();
    f.double_roundtrip(id);

    let niri = f.niri();
    let mut iter = niri.layout.windows();
    let mapped = iter
        .next()
        .expect("the test fixture must contain exactly one mapped window after set_up_with_window")
        .1;
    assert!(
        iter.next().is_none(),
        "expected exactly one mapped window after set_up_with_window",
    );
    mapped.window.clone()
}

#[test]
fn hard_blocked_switch_activity_queues_drains_on_unblock() {
    let mut f = Fixture::with_config(config_with_two_activities(&["alpha-1"], &["beta-1"]));
    let window_id = set_up_with_window(&mut f);

    // Arm an interactive move so the hard-block gate fires.
    let output = f.niri_output(1);
    let armed =
        f.niri()
            .layout
            .interactive_move_begin(window_id.clone(), &output, Point::default());
    assert!(armed, "interactive_move_begin must arm the move");
    assert_eq!(
        f.niri().layout.is_activity_switch_hard_blocked(),
        Some(ActivitySwitchBlock::InteractiveMove),
        "precondition: hard-block must be armed before attempting the switch",
    );

    let pre_active = f.niri().layout.active_activity_id();
    let beta_id = f
        .niri()
        .layout
        .resolve_activity_ref(&ActivityReferenceArg::Name("beta".into()))
        .expect("beta must resolve in the seeded pool");
    assert_ne!(
        pre_active, beta_id,
        "precondition: switch target must differ from active so a successful drain is observable",
    );

    // Sanity-check the synchronous gate via `do_action_inner`. Behaviour
    // pinned here:
    //   - `Action::SwitchActivity` arm at input/mod.rs returns
    //     `Err(DoActionError::ActivitySwitchBlocked(_))` while hard-blocked.
    //   - active activity is unchanged (the gate aborts before mutation).
    let err = f.niri_state().do_action_inner(
        Action::SwitchActivity(ActivityReference::Name("beta".into())),
        false,
    );
    assert!(
        matches!(err, Err(DoActionError::ActivitySwitchBlocked(_))),
        "blocked switch must surface ActivitySwitchBlocked; got {err:?}",
    );
    assert_eq!(
        f.niri().layout.active_activity_id(),
        pre_active,
        "active activity must not change while hard-blocked",
    );

    // Enqueue a waiter via the registry-insert helper. This mirrors the
    // [`jiji_ipc::Request::Action`] arm's blocked-path bookkeeping (depth-1
    // admission, bounded(1) channel, BlockedWaiter insert) without needing a
    // real process() task.
    let conn_id = IpcConnId::specific(101);
    let rx = ipc(&mut f)
        .test_simulate_blocked_request(
            conn_id,
            jiji_config::Action::SwitchActivity(ActivityReference::Name("beta".into())),
        )
        .expect("first enqueue under a fresh conn_id must be admitted");

    // === DISCRIMINATING LEG ===
    // While still hard-blocked, drive a refresh tick. The drain at the end
    // of `State::refresh` must observe the still-armed block and *return
    // early without walking* (server.rs fast-path before the drain loop):
    //   - the registry entry stays put (key still present),
    //   - the active activity is unchanged (no dispatch happened),
    //   - the receiver is `Empty` (no signal sent), critically NOT `Closed` (which would mean the
    //     waiter's tx was dropped — a regression that would silently consume the test if asserted
    //     with `is_err()`).
    f.niri_state().refresh_and_flush_clients();
    assert!(
        ipc(&mut f).test_blocked_waiters_contains_key(conn_id),
        "registry entry must persist across a refresh while still hard-blocked",
    );
    assert_eq!(
        f.niri().layout.active_activity_id(),
        pre_active,
        "active activity must not change while drain is short-circuited by the hard block",
    );
    assert!(
        matches!(rx.try_recv(), Err(async_channel::TryRecvError::Empty),),
        "receiver must be Empty (sender alive, no signal) while still hard-blocked — \
         a Closed result would mean the waiter's tx was dropped, which is a regression",
    );

    // End the move; gate clears.
    f.niri().layout.interactive_move_end(&window_id);
    assert!(
        f.niri().layout.is_activity_switch_hard_blocked().is_none(),
        "interactive_move_end must clear the hard block",
    );

    // Drive the drain via a refresh tick. The drain walks the registry,
    // pops the waiter, calls `do_action_inner`, and signals the waiter
    // with `Ok(())`.
    f.niri_state().refresh_and_flush_clients();

    assert!(
        !ipc(&mut f).test_blocked_waiters_contains_key(conn_id),
        "registry must be empty after drain on unblock",
    );
    assert_eq!(
        f.niri().layout.active_activity_id(),
        beta_id,
        "drain must perform the queued SwitchActivity → active activity flips to beta",
    );
    let observed = recv_or_timeout(
        &rx,
        Duration::from_secs(5),
        "conn drain did not signal within 5s",
    );
    assert!(
        matches!(observed, Ok(DoActionOutcome::Handled)),
        "drained waiter must observe Ok(Handled) (Handled ≡ performed); got {observed:?}",
    );
}

#[test]
fn depth_one_admission_rejects_second_enqueue_per_connection() {
    let mut f = Fixture::with_config(config_with_two_activities(&["alpha-1"], &["beta-1"]));
    let window_id = set_up_with_window(&mut f);

    let output = f.niri_output(1);
    assert!(
        f.niri()
            .layout
            .interactive_move_begin(window_id.clone(), &output, Point::default()),
        "interactive_move_begin must arm the move",
    );
    assert_eq!(
        f.niri().layout.is_activity_switch_hard_blocked(),
        Some(ActivitySwitchBlock::InteractiveMove),
    );

    let beta_id = f
        .niri()
        .layout
        .resolve_activity_ref(&ActivityReferenceArg::Name("beta".into()))
        .expect("beta must resolve in the seeded pool");

    let conn_a = IpcConnId::specific(201);
    let conn_b = IpcConnId::specific(202);

    // First enqueue on conn_a — admission accepts.
    let rx_a = ipc(&mut f)
        .test_simulate_blocked_request(
            conn_a,
            jiji_config::Action::SwitchActivity(ActivityReference::Name("beta".into())),
        )
        .expect("first enqueue under conn_a must be admitted");

    // Second enqueue on conn_a — admission rejects with the live error
    // string. The literal must match server.rs's "request already queued"
    // verbatim; the test would silently pass on a paraphrase like "queue
    // full" otherwise.
    let dup_err = ipc(&mut f)
        .test_simulate_blocked_request(
            conn_a,
            jiji_config::Action::SwitchActivity(ActivityReference::Name("beta".into())),
        )
        .expect_err("second enqueue under the same conn_id must be rejected (depth-1 admission)");
    assert_eq!(
        dup_err, "request already queued",
        "depth-1 rejection must surface the live error string verbatim",
    );

    // A different connection (conn_b) enqueueing in parallel must be
    // admitted — the gate is per-connection, not global.
    let rx_b = ipc(&mut f)
        .test_simulate_blocked_request(
            conn_b,
            jiji_config::Action::SwitchActivity(ActivityReference::Name("beta".into())),
        )
        .expect("enqueue under a distinct conn_id must be admitted");

    // End move + refresh; both waiters drain in FIFO order. Both target
    // beta — the second `SwitchActivity(beta)` after the first lands on the
    // already-active activity and is a no-op fast-path inside
    // `Layout::switch_activity` (returns Ok at the IPC dispatch layer).
    f.niri().layout.interactive_move_end(&window_id);
    f.niri_state().refresh_and_flush_clients();

    assert!(
        !ipc(&mut f).test_blocked_waiters_contains_key(conn_a),
        "conn_a registry entry must drain on unblock",
    );
    assert!(
        !ipc(&mut f).test_blocked_waiters_contains_key(conn_b),
        "conn_b registry entry must drain on unblock",
    );
    assert_eq!(
        f.niri().layout.active_activity_id(),
        beta_id,
        "drain must land active activity on beta",
    );
    assert!(
        matches!(
            recv_or_timeout(
                &rx_a,
                Duration::from_secs(5),
                "conn_a drain did not signal within 5s",
            ),
            Ok(DoActionOutcome::Handled)
        ),
        "conn_a waiter must observe Ok(Handled)",
    );
    assert!(
        matches!(
            recv_or_timeout(
                &rx_b,
                Duration::from_secs(5),
                "conn_b drain did not signal within 5s",
            ),
            Ok(DoActionOutcome::Handled)
        ),
        "conn_b waiter must observe Ok(Handled) (no-op same-target SwitchActivity)",
    );
}
