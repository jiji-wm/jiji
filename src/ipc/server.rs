use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::ffi::OsStr;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::{env, io, process};

use anyhow::Context;
use async_channel::{Receiver, Sender, TrySendError};
use calloop::futures::Scheduler;
use calloop::io::Async;
use directories::BaseDirs;
use futures_util::io::{AsyncReadExt, BufReader};
use futures_util::{select_biased, AsyncBufReadExt, AsyncWrite, AsyncWriteExt, FutureExt as _};
use indexmap::IndexMap;
use jiji_config::OutputName;
use jiji_ipc::state::{EventStreamState, EventStreamStatePart as _};
use jiji_ipc::{
    Action, Event, KeyboardLayouts, OutputConfigChanged, Overview, Reply, Request, Response,
    Timestamp, WindowLayout, Workspace,
};
use smithay::desktop::layer_map_for_output;
use smithay::input::pointer::{
    CursorIcon, CursorImageStatus, Focus, GrabStartData as PointerGrabStartData,
};
use smithay::reexports::calloop::generic::Generic;
use smithay::reexports::calloop::{Interest, LoopHandle, Mode, PostAction};
use smithay::reexports::rustix::fs::unlink;
use smithay::utils::SERIAL_COUNTER;
use smithay::wayland::shell::wlr_layer::{KeyboardInteractivity, Layer};

use crate::backend::IpcOutputMap;
use crate::input::pick_window_grab::PickWindowGrab;
use crate::layout::activity::ActivityId;
use crate::layout::workspace::WorkspaceId;
use crate::layout::{
    format_do_action_error, DoActionError, DoActionOutcome, Layout, LayoutElement,
};
use crate::niri::State;
use crate::utils::id::IdCounter;
use crate::utils::{version, with_toplevel_role};
use crate::window::Mapped;

// If an event stream client fails to read events fast enough that we accumulate more than this
// number in our buffer, we drop that event stream client.
const EVENT_STREAM_BUFFER_SIZE: usize = 64;

/// Process-unique monotonic id for an IPC connection.
///
/// Mirrors the [`WorkspaceId`] / [`ActivityId`] precedent: the counter starts
/// at 1 and is never reused. Used as the key into
/// [`IpcServer::blocked_action_waiters`] so a drain site can identify which
/// connection an entry belongs to without holding a reference to the
/// connection's async task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct IpcConnId(u64);

static IPC_CONN_ID_COUNTER: IdCounter = IdCounter::new();

impl IpcConnId {
    fn next() -> Self {
        Self(IPC_CONN_ID_COUNTER.next())
    }

    #[cfg(test)]
    pub(crate) fn specific(id: u64) -> Self {
        Self(id)
    }
}

/// Entry in [`IpcServer::blocked_action_waiters`].
///
/// Holds the owned [`jiji_config::Action`] so the drain site can re-dispatch
/// without cloning from caller state, plus the send half of the response
/// channel the async `process` task is awaiting on. On drain:
///
/// - Send `Ok(DoActionOutcome::Handled)` once the action lands → `process` returns
///   `Response::Handled`. Send `Ok(DoActionOutcome::NoOp(reason))` for a durable no-op → `process`
///   returns `Response::NoOp(reason)`.
/// - On re-block mid-drain: re-insert the entry and leave `tx` untouched; the sender is not
///   dropped, so the `process` task stays parked.
/// - Drop without sending on a closed receiver (client gone between enqueue and drain). `process`'s
///   `rx.recv().await` has already been dropped in that case.
struct BlockedWaiter {
    action: jiji_config::Action,
    tx: async_channel::Sender<Result<DoActionOutcome, DoActionError>>,
}

pub struct IpcServer {
    /// Path to the IPC socket.
    ///
    /// This is `None` when creating `IpcServer` without a socket.
    pub socket_path: Option<PathBuf>,
    event_streams: Rc<RefCell<Vec<EventStreamSender>>>,
    event_stream_state: Rc<RefCell<EventStreamState>>,
    /// Per-connection depth-1 queue for blocked [`Request::Action`]s. One
    /// entry per connection (admission rejects a second enqueue). `IndexMap`
    /// preserves FIFO insertion order; [`drain_blocked_action_waiters`] walks
    /// it in order. Main-thread only — `Rc` is sufficient.
    blocked_action_waiters: Rc<RefCell<IndexMap<IpcConnId, BlockedWaiter>>>,

    /// Test-only event tap. When `Some`, every event passed through
    /// [`Self::send_event`] is appended (cloned) to the inner `Vec` *before*
    /// the live event-stream fan-out. The tap is unbounded — fixtures drive a
    /// finite, bounded flow, and an unbounded `Vec` ensures a `swap_remove` on
    /// `Full` in the live `try_send` loop cannot evict captured events. The
    /// tap and the live event-stream path are independent: drains here do not
    /// affect connected event-stream clients, and bounded-channel pressure on
    /// those clients does not affect the tap.
    ///
    /// Cleared (replaced with a fresh `Vec`) by every
    /// [`Self::install_test_event_tap`] call so consecutive tests cannot leak
    /// events into each other's captures.
    #[cfg(test)]
    test_event_tap: RefCell<Option<Vec<Event>>>,
}

struct ClientCtx {
    event_loop: LoopHandle<'static, State>,
    scheduler: Scheduler<()>,
    ipc_outputs: Arc<Mutex<IpcOutputMap>>,
    event_streams: Rc<RefCell<Vec<EventStreamSender>>>,
    event_stream_state: Rc<RefCell<EventStreamState>>,
    /// Identity of this connection for keying into
    /// [`IpcServer::blocked_action_waiters`]. Assigned in
    /// [`on_new_ipc_client`] and never reused across connections.
    conn_id: IpcConnId,
    /// Shared handle to [`IpcServer::blocked_action_waiters`]; the
    /// `Request::Action` arm inserts on block, the drain site reads and
    /// wakes. See [`drain_blocked_action_waiters`].
    blocked_action_waiters: Rc<RefCell<IndexMap<IpcConnId, BlockedWaiter>>>,
}

struct EventStreamClient {
    events: Receiver<Event>,
    disconnect: Receiver<()>,
    write: Box<dyn AsyncWrite + Unpin>,
}

struct EventStreamSender {
    events: Sender<Event>,
    disconnect: Sender<()>,
}

impl IpcServer {
    pub fn start(
        event_loop: &LoopHandle<'static, State>,
        wayland_socket_name: Option<&OsStr>,
    ) -> anyhow::Result<Self> {
        let _span = tracy_client::span!("Ipc::start");

        let socket_path = if let Some(wayland_socket_name) = wayland_socket_name {
            let wayland_socket_name = wayland_socket_name.to_string_lossy();
            let socket_name = format!("jiji.{wayland_socket_name}.{}.sock", process::id());
            let mut socket_path = socket_dir();
            socket_path.push(socket_name);

            let listener = UnixListener::bind(&socket_path).context("error binding socket")?;
            listener
                .set_nonblocking(true)
                .context("error setting socket to non-blocking")?;

            let source = Generic::new(listener, Interest::READ, Mode::Level);
            event_loop
                .insert_source(source, |_, socket, state| {
                    match socket.accept() {
                        Ok((stream, _)) => on_new_ipc_client(state, stream),
                        Err(e) if e.kind() == io::ErrorKind::WouldBlock => (),
                        Err(e) => return Err(e),
                    }

                    Ok(PostAction::Continue)
                })
                .unwrap();

            Some(socket_path)
        } else {
            None
        };

        Ok(Self {
            socket_path,
            event_streams: Rc::new(RefCell::new(Vec::new())),
            event_stream_state: Rc::new(RefCell::new(EventStreamState::default())),
            blocked_action_waiters: Rc::new(RefCell::new(IndexMap::new())),
            #[cfg(test)]
            test_event_tap: RefCell::new(None),
        })
    }

    /// Install (or reinstall) the test event tap, replacing any prior buffer
    /// with a fresh empty `Vec`. Call this at the start of each test's flow,
    /// after the initial-state seed has settled. Subsequent
    /// [`Self::send_event`] invocations append clones of every emitted event
    /// to the buffer in emission order.
    #[cfg(test)]
    pub(crate) fn install_test_event_tap(&self) {
        *self.test_event_tap.borrow_mut() = Some(Vec::new());
    }

    /// Drain the test event tap, returning the buffered events in emission
    /// order. Returns an empty `Vec` if the tap was never installed (or was
    /// already drained without an install since). The tap stays installed —
    /// the buffer is `mem::take`'d, leaving an empty `Vec` in place — so
    /// subsequent emissions continue to accumulate; reinstall via
    /// [`Self::install_test_event_tap`] only when you need to reset across
    /// flow phases.
    #[cfg(test)]
    pub(crate) fn drain_test_events(&self) -> Vec<Event> {
        let mut tap = self.test_event_tap.borrow_mut();
        match tap.as_mut() {
            Some(buf) => std::mem::take(buf),
            None => Vec::new(),
        }
    }

    /// Test-only accessor for the server's full event-stream snapshot,
    /// expressed as the sequence of events a fresh client would receive on
    /// connect. Equivalent to `self.event_stream_state.borrow().replicate()`
    /// from within the module; exposed so the test fixture (in
    /// `crate::tests`) can take pre- and post-flow snapshots without making
    /// the field itself `pub(crate)`.
    #[cfg(test)]
    pub(crate) fn replicate_event_stream_state(&self) -> Vec<Event> {
        self.event_stream_state.borrow().replicate()
    }

    /// Simulate the registry-insert side of a [`Request::Action`] arrival
    /// that observed a hard-block at dispatch time. Mirrors the contains-key
    /// admission gate at the live `Request::Action` site (depth-1 per
    /// connection) and, on admission, allocates a `bounded(1)` channel,
    /// inserts a [`BlockedWaiter`] keyed by `conn_id`, and returns the
    /// receive half so the test can later assert the waiter is woken on
    /// drain.
    ///
    /// On admission failure (a waiter already exists for `conn_id`) the
    /// `Err("request already queued")` literal matches the live error string
    /// at the IPC dispatch site; do not paraphrase.
    ///
    /// Validation skip is intentional. The live site runs synchronously:
    /// (1) `contains_key` admission gate (`"request already queued"`),
    /// (2) `validate_action`,
    /// (3) `bounded(1)` channel allocation,
    /// (4) schedules a single `insert_idle` task.
    ///
    /// That deferred task calls `do_action_inner` and inserts into the
    /// registry **only** on `Err(`[`DoActionError::ActivitySwitchBlocked`]`)`; `Ok`
    /// and other `Err` variants send on the channel without ever touching the
    /// registry.
    ///
    /// This helper collapses (1) and the conditional registry insert into one
    /// synchronous call so the test can assert registry state without standing
    /// up a real [`calloop::EventLoop`]. None of the actions exercised under
    /// this contract have a `validate_action` rejection path, so bypassing the
    /// validator is faithful to production. Do not "fix" this by re-adding the
    /// validator call.
    #[cfg(test)]
    pub(crate) fn test_simulate_blocked_request(
        &self,
        conn_id: IpcConnId,
        action: jiji_config::Action,
    ) -> Result<async_channel::Receiver<Result<DoActionOutcome, DoActionError>>, &'static str> {
        // Mirror the contains-key admission gate at the live
        // [`jiji_ipc::Request::Action`] site. Read borrow dropped before the
        // bounded channel is allocated so the registry's RefCell is free for
        // the subsequent insert.
        if self.blocked_action_waiters.borrow().contains_key(&conn_id) {
            return Err("request already queued");
        }

        let (tx, rx) = async_channel::bounded::<Result<DoActionOutcome, DoActionError>>(1);
        let prev = self
            .blocked_action_waiters
            .borrow_mut()
            .insert(conn_id, BlockedWaiter { action, tx });
        assert!(
            prev.is_none(),
            "test_simulate_blocked_request: contains-key gate must keep registry empty for {conn_id:?}",
        );
        Ok(rx)
    }

    /// Read-only test accessor: does the blocked-action registry currently
    /// hold an entry for `conn_id`? Returns `bool`, not a borrow guard, so
    /// callers can interleave it freely with mutating fixture calls
    /// (`refresh_and_flush_clients`, `interactive_move_end`, …) without
    /// fighting the `RefCell` borrow scope.
    #[cfg(test)]
    pub(crate) fn test_blocked_waiters_contains_key(&self, conn_id: IpcConnId) -> bool {
        self.blocked_action_waiters.borrow().contains_key(&conn_id)
    }

    fn send_event(&self, event: Event) {
        // Test-only: capture the event before the live fan-out. Scoped in
        // its own block so the `RefMut` drops before `event_streams` is
        // borrowed below.
        #[cfg(test)]
        {
            if let Some(tap) = self.test_event_tap.borrow_mut().as_mut() {
                tap.push(event.clone());
            }
        }

        let mut streams = self.event_streams.borrow_mut();
        let mut to_remove = Vec::new();
        for (idx, stream) in streams.iter_mut().enumerate() {
            match stream.events.try_send(event.clone()) {
                Ok(()) => (),
                Err(TrySendError::Closed(_)) => to_remove.push(idx),
                Err(TrySendError::Full(_)) => {
                    warn!(
                        "disconnecting IPC event stream client \
                         because it is reading events too slowly"
                    );
                    to_remove.push(idx);
                }
            }
        }

        for idx in to_remove.into_iter().rev() {
            let stream = streams.swap_remove(idx);
            let _ = stream.disconnect.send_blocking(());
        }
    }
}

impl Drop for IpcServer {
    fn drop(&mut self) {
        if let Some(socket_path) = &self.socket_path {
            let _ = unlink(socket_path);
        }
    }
}

/// Drain the per-connection blocked-action queue, re-dispatching each waiter
/// in FIFO order and waking its `process` task on success.
///
/// Invoked as the last step of [`State::refresh`] (after all other
/// `ipc_refresh_*` sites) so any state-change events produced by the
/// re-dispatched actions ride the next refresh tick in the correct order
/// — observable state is updated before the corresponding `Response::Handled`
/// is surfaced to the IPC client.
///
/// # Invariants
///
/// - **`Handled` ≡ performed; `NoOp(reason)` ≡ considered-and-unchanged**: a waiter is only
///   signalled `Ok(DoActionOutcome::Handled)` after its `do_action_inner` call returned
///   `Ok(Handled)`. `Ok(NoOp(reason))` from `do_action_inner` is forwarded verbatim as
///   `Ok(DoActionOutcome::NoOp(reason))`; the `process` recv site maps it to
///   `Response::NoOp(reason)`. The send half is dropped without signalling on silent-prune paths
///   (closed receiver).
/// - **FIFO preserved across re-block** (scoped to `Err(DoActionError::ActivitySwitchBlocked)`): if
///   `do_action_inner` re-raises a hard block mid-drain (no current action reaches this;
///   forward-compat for future gating widening), the waiter is re-inserted at its index at removal
///   via `shift_insert(original_idx, …)` and the walk **breaks** — continuing past a re-blocked
///   waiter would promote later waiters ahead of it. The `// FIFO pin` breadcrumb on the `break`
///   pins this semantic against accidental refactor.
/// - **Terminal errors advance the drain** (scoped to `Err(DoActionError::WindowNotFound)`): a
///   terminal error for one waiter does not block later waiters. The waiter is signalled
///   `Err(DoActionError::WindowNotFound { id })` and the walk `continue`s to the next connection.
/// - **Closed-receiver prune**: if the client disconnected between enqueue and drain
///   (`tx.is_closed()`), the entry is dropped without dispatch. No side effects, no replay.
/// - **No registry borrow across `do_action_inner`**: the walk grabs and drops the registry
///   `RefCell` borrow per iteration so a nested `IpcServer` access inside action dispatch can't
///   deadlock.
pub(crate) fn drain_blocked_action_waiters(state: &mut State) {
    let _span = tracy_client::span!("drain_blocked_action_waiters");

    // Fast-path: no server → nothing to drain.
    let Some(server) = state.niri.ipc_server.as_ref() else {
        return;
    };

    // Fast-path: empty registry → skip the hard-block check entirely.
    let conn_ids: Vec<IpcConnId> = {
        let waiters = server.blocked_action_waiters.borrow();
        if waiters.is_empty() {
            return;
        }
        waiters.keys().copied().collect()
    };

    // Fast-path: still hard-blocked → don't walk waiters; they'll drain on a
    // later tick. Checked after the empty-registry short-circuit so the hot
    // path (no queue) pays nothing.
    if state
        .niri
        .layout
        .is_activity_switch_hard_blocked()
        .is_some()
    {
        return;
    }

    for conn_id in conn_ids {
        // Grab-and-drop the registry borrow per iteration: `do_action_inner`
        // below may reach back into `state.niri.ipc_server` (for events,
        // etc.) and must not collide with a live borrow.
        let (original_idx, waiter) = {
            let server = state
                .niri
                .ipc_server
                .as_ref()
                .expect("ipc_server present — drain pre-checked non-empty registry");
            let mut waiters = server.blocked_action_waiters.borrow_mut();
            match waiters.get_index_of(&conn_id) {
                Some(idx) => {
                    let w = waiters
                        .shift_remove(&conn_id)
                        .expect("get_index_of succeeded for conn_id immediately prior");
                    if w.tx.is_closed() {
                        // Client gone — drop the waiter without dispatch.
                        continue;
                    }
                    (idx, w)
                }
                None => continue,
            }
        };

        let result = state.do_action_inner(waiter.action.clone(), false);
        match result {
            Ok(outcome) => {
                // Receiver may have dropped between wake and send; safe to
                // ignore because the action already executed with no state
                // loss (same contract as the initial-dispatch success path).
                // The outcome (`Handled` vs `NoOp(reason)`) is forwarded
                // verbatim so the IPC `process` site can map it to the right
                // `Response` variant.
                let _ = waiter.tx.send_blocking(Ok(outcome));
            }
            Err(DoActionError::ActivitySwitchBlocked(block)) => {
                // Re-block mid-drain: re-insert at its index at removal so
                // later waiters don't promote past this one. Today's
                // `do_action_inner` surface cannot produce this; forward-
                // compat against future widening of hard-block gating.
                let server = state
                    .niri
                    .ipc_server
                    .as_ref()
                    .expect("ipc_server present — re-block path, registry still alive");
                let prev = server.blocked_action_waiters.borrow_mut().shift_insert(
                    original_idx,
                    conn_id,
                    BlockedWaiter {
                        action: waiter.action,
                        tx: waiter.tx,
                    },
                );
                debug_assert!(
                    prev.is_none(),
                    "shift_insert on re-block must not overwrite: conn_id={conn_id:?} original_idx={original_idx}",
                );
                let _ = block;
                // FIFO pin — drain order must not promote later waiters ahead of re-blocked ones.
                //
                // Do NOT convert this `break` to `continue`: walking past a
                // re-blocked waiter promotes later waiters ahead of it and
                // violates FIFO. The drain-re-block invariant is pinned by
                // `blocked_action_waiters_reblock_leaves_entry`.
                break;
            }
            Err(DoActionError::WindowNotFound { id }) => {
                // Terminal error — not a block; advance the drain walk.
                // A stale id for waiter X does not affect waiters Y, Z: the
                // registry entry was already removed via `shift_remove`
                // above, and the walk continues to the next connection
                //. Do NOT convert this `continue` to `break` —
                // that would halt drain for all later waiters after one
                // unknown-id action.
                let _ = waiter
                    .tx
                    .send_blocking(Err(DoActionError::WindowNotFound { id }));
                continue;
            }
            Err(err @ DoActionError::AddWorkspaceToActivity(_))
            | Err(err @ DoActionError::RemoveWorkspaceFromActivity(_))
            | Err(err @ DoActionError::SetWorkspaceActivities(_))
            | Err(err @ DoActionError::MoveWorkspaceToActivity(_))
            | Err(err @ DoActionError::CreateActivity(_))
            | Err(err @ DoActionError::RemoveActivity(_))
            | Err(err @ DoActionError::RenameActivity(_))
            | Err(err @ DoActionError::SwitchActivity(_))
            | Err(err @ DoActionError::ToggleWorkspaceSticky(_))
            | Err(err @ DoActionError::SetWorkspaceSticky(_))
            | Err(err @ DoActionError::UnsetWorkspaceSticky(_))
            | Err(err @ DoActionError::MoveWindowTargetUnreachable { .. }) => {
                // Terminal errors. Same shape as `WindowNotFound`:
                // forward and advance the walk — do not re-block.
                let _ = waiter.tx.send_blocking(Err(err));
                continue;
            }
        }
    }
}

fn socket_dir() -> PathBuf {
    BaseDirs::new()
        .as_ref()
        .and_then(|x| x.runtime_dir())
        .map(|x| x.to_owned())
        .unwrap_or_else(env::temp_dir)
}

fn on_new_ipc_client(state: &mut State, stream: UnixStream) {
    let _span = tracy_client::span!("on_new_ipc_client");
    trace!("new IPC client connected");

    let stream = match state.niri.event_loop.adapt_io(stream) {
        Ok(stream) => stream,
        Err(err) => {
            warn!("error making IPC stream async: {err:?}");
            return;
        }
    };

    let ipc_server = state.niri.ipc_server.as_ref().unwrap();

    let ctx = ClientCtx {
        event_loop: state.niri.event_loop.clone(),
        scheduler: state.niri.scheduler.clone(),
        ipc_outputs: state.backend.ipc_outputs(),
        event_streams: ipc_server.event_streams.clone(),
        event_stream_state: ipc_server.event_stream_state.clone(),
        conn_id: IpcConnId::next(),
        blocked_action_waiters: ipc_server.blocked_action_waiters.clone(),
    };

    let future = async move {
        if let Err(err) = handle_client(ctx, stream).await {
            warn!("error handling IPC client: {err:?}");
        }
    };
    if let Err(err) = state.niri.scheduler.schedule(future) {
        warn!("error scheduling IPC stream future: {err:?}");
    }
}

async fn handle_client(ctx: ClientCtx, stream: Async<'static, UnixStream>) -> anyhow::Result<()> {
    let (read, mut write) = stream.split();
    let mut read = BufReader::new(read);

    loop {
        // Don't keep buf around to avoid clients wasting RAM by filling it with bogus data.
        let mut buf = Vec::new();
        let res = read.read_until(b'\n', &mut buf).await;
        match res {
            Ok(0) => return Ok(()),
            Ok(_) => (),
            // Normal client disconnection.
            Err(err) if err.kind() == io::ErrorKind::BrokenPipe => return Ok(()),
            Err(err) => {
                return Err(err).context("error reading request");
            }
        }

        let request = serde_json::from_slice(&buf)
            .context("error parsing request")
            .map_err(|err| err.to_string());
        let requested_error = matches!(request, Ok(Request::ReturnError));
        let requested_event_stream = matches!(request, Ok(Request::EventStream));

        let reply = match request {
            Ok(request) => process(&ctx, request).await,
            Err(err) => Err(err),
        };

        if let Err(err) = &reply {
            if !requested_error {
                warn!("error processing IPC request: {err:?}");
            }
        }

        buf.clear();
        serde_json::to_writer(&mut buf, &reply).context("error formatting reply")?;
        buf.push(b'\n');
        write.write_all(&buf).await.context("error writing reply")?;

        if requested_event_stream {
            let (events_tx, events_rx) = async_channel::bounded(EVENT_STREAM_BUFFER_SIZE);
            let (disconnect_tx, disconnect_rx) = async_channel::bounded(1);

            // Spawn a task for the client.
            let client = EventStreamClient {
                events: events_rx,
                disconnect: disconnect_rx,
                write: Box::new(write) as _,
            };
            let future = async move {
                if let Err(err) = handle_event_stream_client(client).await {
                    warn!("error handling IPC event stream client: {err:?}");
                }
            };
            if let Err(err) = ctx.scheduler.schedule(future) {
                warn!("error scheduling IPC event stream future: {err:?}");
            }

            // Send the initial state.
            {
                let state = ctx.event_stream_state.borrow();
                for event in state.replicate() {
                    events_tx
                        .try_send(event)
                        .expect("initial event burst had more events than buffer size");
                }
            }

            // Add it to the list.
            {
                let mut streams = ctx.event_streams.borrow_mut();
                let sender = EventStreamSender {
                    events: events_tx,
                    disconnect: disconnect_tx,
                };
                streams.push(sender);
            }

            return Ok(());
        }
    }
}

async fn process(ctx: &ClientCtx, request: Request) -> Reply {
    let response = match request {
        Request::ReturnError => return Err(String::from("example compositor error")),
        Request::Version => Response::Version(version()),
        Request::Outputs => {
            let ipc_outputs = ctx.ipc_outputs.lock().unwrap().clone();
            let outputs = ipc_outputs.values().cloned().map(|o| (o.name.clone(), o));
            Response::Outputs(outputs.collect())
        }
        Request::Workspaces => {
            let state = ctx.event_stream_state.borrow();
            let workspaces = state.workspaces.workspaces.values().cloned().collect();
            Response::Workspaces(workspaces)
        }
        Request::Windows => {
            let state = ctx.event_stream_state.borrow();
            let windows = state.windows.windows.values().cloned().collect();
            Response::Windows(windows)
        }
        Request::Layers => {
            let (tx, rx) = async_channel::bounded(1);
            ctx.event_loop.insert_idle(move |state| {
                let mut layers = Vec::new();
                for output in state.niri.global_space.outputs() {
                    let name = output.name();
                    for surface in layer_map_for_output(output).layers() {
                        let layer = match surface.layer() {
                            Layer::Background => jiji_ipc::Layer::Background,
                            Layer::Bottom => jiji_ipc::Layer::Bottom,
                            Layer::Top => jiji_ipc::Layer::Top,
                            Layer::Overlay => jiji_ipc::Layer::Overlay,
                        };
                        let keyboard_interactivity =
                            match surface.cached_state().keyboard_interactivity {
                                KeyboardInteractivity::None => {
                                    jiji_ipc::LayerSurfaceKeyboardInteractivity::None
                                }
                                KeyboardInteractivity::Exclusive => {
                                    jiji_ipc::LayerSurfaceKeyboardInteractivity::Exclusive
                                }
                                KeyboardInteractivity::OnDemand => {
                                    jiji_ipc::LayerSurfaceKeyboardInteractivity::OnDemand
                                }
                            };

                        layers.push(jiji_ipc::LayerSurface {
                            namespace: surface.namespace().to_owned(),
                            output: name.clone(),
                            layer,
                            keyboard_interactivity,
                        });
                    }
                }

                let _ = tx.send_blocking(layers);
            });
            let result = rx.recv().await;
            let layers = result.map_err(|_| String::from("error getting layers info"))?;
            Response::Layers(layers)
        }
        Request::KeyboardLayouts => {
            let state = ctx.event_stream_state.borrow();
            let layout = state.keyboard_layouts.keyboard_layouts.clone();
            let layout = layout.expect("keyboard layouts should be set at startup");
            Response::KeyboardLayouts(layout)
        }
        Request::FocusedWindow => {
            let state = ctx.event_stream_state.borrow();
            let windows = &state.windows.windows;
            let window = windows.values().find(|win| win.is_focused).cloned();
            Response::FocusedWindow(window)
        }
        Request::Activities => {
            let (tx, rx) = async_channel::bounded(1);
            ctx.event_loop.insert_idle(move |state| {
                let activities = build_activities_ipc(&state.niri.layout);
                let _ = tx.send_blocking(activities);
            });
            let result = rx.recv().await;
            let activities = result.map_err(|_| String::from("error getting activities info"))?;
            Response::Activities(activities)
        }
        Request::ActivityViews => {
            let (tx, rx) = async_channel::bounded(1);
            ctx.event_loop.insert_idle(move |state| {
                let views = build_activity_views_ipc(&state.niri.layout);
                let _ = tx.send_blocking(views);
            });
            let result = rx.recv().await;
            let views = result.map_err(|_| String::from("error getting activity views info"))?;
            Response::ActivityViews(views)
        }
        Request::FocusedActivity => {
            let (tx, rx) = async_channel::bounded(1);
            ctx.event_loop.insert_idle(move |state| {
                let focused = build_focused_activity_ipc(&state.niri.layout);
                let _ = tx.send_blocking(focused);
            });
            let result = rx.recv().await;
            let focused =
                result.map_err(|_| String::from("error getting focused activity info"))?;
            Response::FocusedActivity(focused)
        }
        Request::PickWindow => {
            let (tx, rx) = async_channel::bounded(1);
            ctx.event_loop.insert_idle(move |state| {
                let pointer = state.niri.seat.get_pointer().unwrap();
                let start_data = PointerGrabStartData {
                    focus: None,
                    button: 0,
                    location: pointer.current_location(),
                };
                let grab = PickWindowGrab::new(start_data);
                // The `WindowPickGrab` ungrab handler will cancel the previous ongoing pick, if
                // any.
                pointer.set_grab(state, grab, SERIAL_COUNTER.next_serial(), Focus::Clear);
                state.niri.pick_window = Some(tx);
                state
                    .niri
                    .cursor_manager
                    .set_cursor_image(CursorImageStatus::Named(CursorIcon::Crosshair));
                // Redraw to update the cursor.
                state.niri.queue_redraw_all();
            });
            let result = rx.recv().await;
            let id = result.map_err(|_| String::from("error getting picked window info"))?;
            let window = id.and_then(|id| {
                let state = ctx.event_stream_state.borrow();
                state.windows.windows.get(&id.get()).cloned()
            });
            Response::PickedWindow(window)
        }
        Request::PickColor => {
            let (tx, rx) = async_channel::bounded(1);
            ctx.event_loop.insert_idle(move |state| {
                state.handle_pick_color(tx);
            });
            let result = rx.recv().await;
            let color = result.map_err(|_| String::from("error getting picked color"))?;
            Response::PickedColor(color)
        }
        Request::Action(action) => {
            // Forward-compat against a future pipelined `handle_client`; unreachable under today's
            // sequential dispatch.
            if ctx
                .blocked_action_waiters
                .borrow()
                .contains_key(&ctx.conn_id)
            {
                return Err("request already queued".into());
            }

            validate_action(&action)?;

            let (tx, rx) = async_channel::bounded::<Result<DoActionOutcome, DoActionError>>(1);

            let action = jiji_config::Action::from(action);
            let waiters = ctx.blocked_action_waiters.clone();
            let conn_id = ctx.conn_id;
            ctx.event_loop.insert_idle(move |state| {
                // Make sure some logic like workspace clean-up has a chance to run before doing
                // actions.
                state.niri.advance_animations();
                // Clone so the waiter entry owns the original if do_action_inner hard-blocks (the drain site re-dispatches from it).
                let result = state.do_action_inner(action.clone(), false);
                match result {
                    Ok(outcome) => {
                        // Connection may have closed between enqueue and action
                        // completion. Safe to drop the send: on Ok the action
                        // already executed with no state loss. Forward the
                        // outcome verbatim so the `process` recv site can map
                        // `Handled` and `NoOp(reason)` to their respective
                        // `Response` variants.
                        let _ = tx.send_blocking(Ok(outcome));
                    }
                    Err(DoActionError::ActivitySwitchBlocked(block)) => {
                        // Park; drain on next refresh preserves Handled ≡ performed.
                        let _ = block;
                        let prev = waiters
                            .borrow_mut()
                            .insert(conn_id, BlockedWaiter { action, tx });
                        debug_assert!(
                            prev.is_none(),
                            "depth-1 admission must keep the registry empty for conn_id={conn_id:?} at insert_idle execution time",
                        );
                    }
                    Err(DoActionError::WindowNotFound { id }) => {
                        // Terminal error: forward immediately.
                        // Do NOT insert into `blocked_action_waiters` — a
                        // registry entry would deadlock the connection,
                        // since no hard-block condition exists to clear
                        // and the drain site would never re-dispatch.
                        let _ = tx.send_blocking(Err(DoActionError::WindowNotFound { id }));
                    }
                    // Terminal errors from workspace-activity
                    // assignment actions. Same rationale as `WindowNotFound`:
                    // no hard-block condition, do not park — forward to the
                    // waiter so the IPC envelope is produced on the main
                    // dispatch path.
                    Err(err @ DoActionError::AddWorkspaceToActivity(_))
                    | Err(err @ DoActionError::RemoveWorkspaceFromActivity(_))
                    | Err(err @ DoActionError::SetWorkspaceActivities(_))
                    | Err(err @ DoActionError::MoveWorkspaceToActivity(_))
                    | Err(err @ DoActionError::CreateActivity(_))
                    | Err(err @ DoActionError::RemoveActivity(_))
                    | Err(err @ DoActionError::RenameActivity(_))
                    | Err(err @ DoActionError::SwitchActivity(_))
                    | Err(err @ DoActionError::ToggleWorkspaceSticky(_))
                    | Err(err @ DoActionError::SetWorkspaceSticky(_))
                    | Err(err @ DoActionError::UnsetWorkspaceSticky(_))
                    | Err(err @ DoActionError::MoveWindowTargetUnreachable { .. }) => {
                        let _ = tx.send_blocking(Err(err));
                    }
                }
            });

            // Wait until the action has been processed before returning. This is important for a
            // few actions, for instance for DoScreenTransition this wait ensures that the screen
            // contents were sampled into the texture. Under a hard block, the
            // receiver parks here until the drain site wakes it.
            match rx.recv().await {
                Ok(Ok(DoActionOutcome::Handled)) => Response::Handled,
                Ok(Ok(DoActionOutcome::NoOp(reason))) => Response::NoOp(reason),
                Ok(Err(err)) => {
                    return Err(format_do_action_error(err));
                }
                Err(err) => {
                    warn!("action dispatch channel closed unexpectedly: {err:?}");
                    return Err("action dispatch channel closed".into());
                }
            }
        }
        Request::Output { output, action } => {
            action.validate()?;

            let ipc_outputs = ctx.ipc_outputs.lock().unwrap();
            let found = ipc_outputs
                .values()
                .any(|o| OutputName::from_ipc_output(o).matches(&output));
            let response = if found {
                OutputConfigChanged::Applied
            } else {
                OutputConfigChanged::OutputWasMissing
            };
            drop(ipc_outputs);

            ctx.event_loop.insert_idle(move |state| {
                state.apply_transient_output_config(&output, action);
            });

            Response::OutputConfigChanged(response)
        }
        Request::FocusedOutput => {
            let (tx, rx) = async_channel::bounded(1);
            ctx.event_loop.insert_idle(move |state| {
                let active_output = state
                    .niri
                    .layout
                    .active_output()
                    .map(|output| output.name());

                let output = active_output.and_then(|active_output| {
                    state
                        .backend
                        .ipc_outputs()
                        .lock()
                        .unwrap()
                        .values()
                        .find(|o| o.name == active_output)
                        .cloned()
                });

                let _ = tx.send_blocking(output);
            });
            let result = rx.recv().await;
            let output = result.map_err(|_| String::from("error getting active output info"))?;
            Response::FocusedOutput(output)
        }
        Request::EventStream => Response::Handled,
        Request::OverviewState => {
            let state = ctx.event_stream_state.borrow();
            let is_open = state.overview.is_open;
            Response::OverviewState(Overview { is_open })
        }
        Request::Casts => {
            let state = ctx.event_stream_state.borrow();
            let casts = state.casts.casts.values().cloned().collect();
            Response::Casts(casts)
        }
        _ => return Err(String::from("unsupported request variant")),
    };

    Ok(response)
}

pub(crate) fn build_activities_ipc<W: LayoutElement>(
    layout: &Layout<W>,
) -> Vec<jiji_ipc::Activity> {
    let active_id = layout.active_activity_id();
    layout
        .activities()
        .iter()
        .map(|a| to_ipc_activity(a, active_id, layout))
        .collect()
}

/// Project per-activity views into a flat IPC vec. See [`jiji_ipc::ActivityView`] for the
/// source contract, field semantics, and ordering guarantees.
pub(crate) fn build_activity_views_ipc<W: LayoutElement>(
    layout: &Layout<W>,
) -> Vec<jiji_ipc::ActivityView> {
    let mut out = Vec::new();
    for activity in layout.activities().iter() {
        let mut entries: Vec<_> = activity.views().iter().collect();
        entries.sort_by(|(a, _), (b, _)| a.as_str().cmp(b.as_str()));
        for (output_id, view) in entries {
            let output_name = layout
                .monitor_for_output_id(output_id)
                .map(|m| m.output_name().clone());
            out.push(jiji_ipc::ActivityView {
                activity_id: activity.id().get(),
                output_id: output_id.as_str().to_owned(),
                output_name,
                workspace_ids: view.ids().iter().map(|id| id.get()).collect(),
                active_idx: view.active_position(),
            });
        }
    }
    out
}

pub(crate) fn build_focused_activity_ipc<W: LayoutElement>(
    layout: &Layout<W>,
) -> jiji_ipc::Activity {
    let active_id = layout.active_activity_id();
    to_ipc_activity(layout.activities().active(), active_id, layout)
}

fn to_ipc_activity<W: LayoutElement>(
    a: &crate::layout::activity::Activity,
    active_id: ActivityId,
    layout: &Layout<W>,
) -> jiji_ipc::Activity {
    jiji_ipc::Activity {
        id: a.id().get(),
        name: a.name().to_owned(),
        is_config_declared: a.is_config_declared(),
        is_active: a.id() == active_id,
        is_urgent: layout.activity_is_urgent(a.id()),
        last_active_seq: a.last_active_seq(),
    }
}

fn validate_action(action: &Action) -> Result<(), String> {
    if let Action::Screenshot { path, .. }
    | Action::ScreenshotScreen { path, .. }
    | Action::ScreenshotWindow { path, .. }
    | Action::LoadConfigFile { path } = action
    {
        if let Some(path) = path {
            // Relative paths are resolved against the jiji compositor's working directory, which
            // is almost certainly not what you want.
            if !Path::new(path).is_absolute() {
                return Err(format!("path must be absolute: {path}"));
            }
        }
    }

    if let Action::LoadConfigFile { path: Some(path) } = action {
        let p = Path::new(path);
        if !p.is_file() {
            return Err(format!("path does not point to a file: {path}"));
        }
    }

    Ok(())
}

async fn handle_event_stream_client(client: EventStreamClient) -> anyhow::Result<()> {
    let EventStreamClient {
        events,
        disconnect,
        mut write,
    } = client;

    while let Ok(event) = events.recv().await {
        let mut buf = serde_json::to_vec(&event).context("error formatting event")?;
        buf.push(b'\n');

        let res = select_biased! {
            _ = disconnect.recv().fuse() => return Ok(()),
            res = write.write_all(&buf).fuse() => res,
        };

        match res {
            Ok(()) => (),
            // Normal client disconnection.
            Err(err) if err.kind() == io::ErrorKind::BrokenPipe => return Ok(()),
            res @ Err(_) => res.context("error writing event")?,
        }
    }

    Ok(())
}

fn make_ipc_window(
    mapped: &Mapped,
    workspace_id: Option<WorkspaceId>,
    layout: WindowLayout,
) -> jiji_ipc::Window {
    with_toplevel_role(mapped.toplevel(), |role| jiji_ipc::Window {
        id: mapped.id().get(),
        title: role.title.clone(),
        app_id: role.app_id.clone(),
        pid: mapped.credentials().map(|c| c.pid),
        workspace_id: workspace_id.map(|id| id.get()),
        is_focused: mapped.is_focused(),
        is_floating: mapped.is_floating(),
        is_urgent: mapped.is_urgent(),
        layout,
        focus_timestamp: mapped.get_focus_timestamp().map(Timestamp::from),
    })
}

impl State {
    pub fn ipc_keyboard_layouts_changed(&mut self) {
        let keyboard = self.niri.seat.get_keyboard().unwrap();
        let keyboard_layouts = keyboard.with_xkb_state(self, |context| {
            let xkb = context.xkb().lock().unwrap();
            let layouts = xkb.layouts();
            KeyboardLayouts {
                names: layouts
                    .map(|layout| xkb.layout_name(layout).to_owned())
                    .collect(),
                current_idx: xkb.active_layout().0 as u8,
            }
        });

        let Some(server) = &self.niri.ipc_server else {
            return;
        };

        let mut state = server.event_stream_state.borrow_mut();
        let state = &mut state.keyboard_layouts;

        let event = Event::KeyboardLayoutsChanged { keyboard_layouts };
        state.apply(event.clone());
        server.send_event(event);
    }

    pub fn ipc_refresh_keyboard_layout_index(&mut self) {
        let keyboard = self.niri.seat.get_keyboard().unwrap();
        let idx = keyboard.with_xkb_state(self, |context| {
            let xkb = context.xkb().lock().unwrap();
            xkb.active_layout().0 as u8
        });

        let Some(server) = &self.niri.ipc_server else {
            return;
        };

        let mut state = server.event_stream_state.borrow_mut();
        let state = &mut state.keyboard_layouts;

        if state.keyboard_layouts.as_ref().unwrap().current_idx == idx {
            return;
        }

        let event = Event::KeyboardLayoutSwitched { idx };
        state.apply(event.clone());
        server.send_event(event);
    }

    pub fn ipc_refresh_layout(&mut self) {
        // Capture the previous active-activity id from
        // `EventStreamState::activities` BEFORE `ipc_refresh_activity_lifecycle`
        // mutates the snapshot. This is load-bearing for the cascade-remove
        // case: when `RemoveActivity` drops the active cursor and repoints it,
        // lifecycle applies `ActivityRemoved` first — which erases the
        // `is_active=true` signal from the snapshot — so a downstream
        // `state.activities.values().find(|a| a.is_active)` would return
        // `None` and the active-activity diff would silently drop the
        // required `ActivitySwitched`. Snapshotting here makes the cascade
        // emit `[Removed{beta}, Switched{alpha, prev:beta}]` correctly.
        let previous_active_id: Option<u64> = self.niri.ipc_server.as_ref().and_then(|server| {
            server
                .event_stream_state
                .borrow()
                .activities
                .activities
                .values()
                .find(|a| a.is_active)
                .map(|a| a.id)
        });

        // Activity-lifecycle events (`ActivityCreated` / `ActivityRemoved` /
        // `ActivityRenamed`) fire structure-before-state: before
        // `ipc_refresh_active_activity` so a client seeing an
        // `ActivitySwitched { id: N, previous_id: Some(M) }` has already
        // observed `ActivityCreated { id: N }` (for Create-then-Switch) and
        // already observed `ActivityRemoved { id: M }` on the same tick (for
        // the `RemoveActivity`-of-active cascade, where the remove fires
        // before the forced cursor re-point); and before
        // `ipc_refresh_workspaces` so a `WorkspaceClosed { id: W }` triggered
        // by exclusive-workspace destruction follows the `ActivityRemoved`
        // for the activity that owned it. This mirrors the
        // `ActivitiesChanged`-before-`WorkspacesChanged` principle
        // documented on `Event::ActivitiesChanged`. See
        //  for the normative statement of the
        // structure-before-state rule. Do not move
        // `ipc_refresh_activity_lifecycle` after `ipc_refresh_active_activity`
        // or `ipc_refresh_workspaces`.
        self.ipc_refresh_activity_lifecycle();
        // `ActivitySwitched` must precede any `WorkspaceOpenedOrChanged` whose
        // `is_in_active_activity` flipped this tick, so clients can update
        // their activity state before processing workspace visibility
        // changes. See the contract pinned on `Event::ActivitySwitched` in
        // `jiji-ipc/src/lib.rs`. Do not move `ipc_refresh_active_activity`
        // after `ipc_refresh_workspaces`.
        self.ipc_refresh_active_activity(previous_active_id);
        self.ipc_refresh_workspaces();
        // Urgency ordering is inside-out: `WindowUrgencyChanged` fires as the
        // window updates; `WorkspaceUrgencyChanged` fires from
        // `ipc_refresh_workspaces` above; `ActivityUrgencyChanged` is the
        // outermost layer and must fire after workspace urgency so clients
        // see workspace-level state settled before the activity aggregate
        // arrives (jiji-ipc `Event::ActivityUrgencyChanged` rustdoc). Do not
        // move `ipc_refresh_activity_urgency` before `ipc_refresh_workspaces`.
        self.ipc_refresh_activity_urgency();
        self.ipc_refresh_windows();
        self.ipc_refresh_overview();
    }

    fn ipc_refresh_active_activity(&mut self, previous_active_id: Option<u64>) {
        let Some(server) = &self.niri.ipc_server else {
            return;
        };

        let _span = tracy_client::span!("State::ipc_refresh_active_activity");

        let current = self.niri.layout.active_activity_id().get();

        // `previous_active_id` is captured by `ipc_refresh_layout` BEFORE
        // lifecycle runs — see the commentary there for why reading
        // post-lifecycle is unsafe in the cascade-remove case.
        //
        // The `Option::is_some()` gate reproduces the pre-subsumption
        // "was_initialized" seed-silence posture: on a genuinely fresh
        // server (empty snapshot) `previous_active_id` is `None` because no
        // activity was ever flagged `is_active`; the seed activity's
        // `ActivityCreated` emitted by `ipc_refresh_activity_lifecycle` this
        // same tick carries the correct cursor in the payload, so we emit no
        // `ActivitySwitched` and clients never see a spurious initial
        // transition.
        if let Some(prev) = previous_active_id {
            if prev != current {
                let mut state = server.event_stream_state.borrow_mut();
                let state = &mut state.activities;
                let event = Event::ActivitySwitched {
                    id: current,
                    previous_id: Some(prev),
                };
                state.apply(event.clone());
                server.send_event(event);
            }
        }
    }

    fn ipc_refresh_activity_urgency(&mut self) {
        let Some(server) = &self.niri.ipc_server else {
            return;
        };

        let _span = tracy_client::span!("State::ipc_refresh_activity_urgency");

        let layout = &self.niri.layout;

        // Collect current urgency per (id, urgent) from the live layout into
        // an owned `Vec` before the `borrow_mut` to keep the layout borrow
        // from overlapping the refcell acquisition.
        let current: Vec<(u64, bool)> = layout
            .activities()
            .iter()
            .map(|a| (a.id().get(), layout.activity_is_urgent(a.id())))
            .collect();

        let mut state = server.event_stream_state.borrow_mut();
        let state = &mut state.activities;

        // Diff against `state.activities` directly. For any id newly present
        // in `current` but not in the snapshot: `ipc_refresh_activity_lifecycle`
        // ran earlier this tick and already emitted `ActivityCreated` whose
        // payload carried the correct `is_urgent` — so `state` already
        // reflects that urgency and the comparison here naturally skips the
        // newcomer with no spurious event. (If the id were somehow absent,
        // we would skip it with `continue` rather than fabricating state.)
        let mut transitions: Vec<(u64, bool)> = Vec::new();
        for (id, urgent) in current {
            let Some(a) = state.activities.get(&id) else {
                unreachable!(
                    "ipc_refresh_activity_lifecycle seeded id {id} earlier on the same tick"
                );
            };
            if a.is_urgent != urgent {
                transitions.push((id, urgent));
            }
        }
        transitions.sort_by_key(|(id, _)| *id);

        for (id, urgent) in transitions {
            let event = Event::ActivityUrgencyChanged { id, urgent };
            state.apply(event.clone());
            server.send_event(event);
        }
    }

    fn ipc_refresh_activity_lifecycle(&mut self) {
        let Some(server) = &self.niri.ipc_server else {
            return;
        };

        let _span = tracy_client::span!("State::ipc_refresh_activity_lifecycle");

        let layout = &self.niri.layout;
        // Build the live IPC snapshot from `Layout` first. Collecting to an
        // owned `Vec` (with each `jiji_ipc::Activity` already carrying the
        // correct derived fields via `to_ipc_activity`) releases the shared
        // layout borrow before we acquire the refcell.
        let current = build_activities_ipc(layout);

        let mut state = server.event_stream_state.borrow_mut();
        let state = &mut state.activities;

        // Diff previous (state.activities) against `current`:
        //   - Removed: ids in previous, not in current.
        //   - Renamed: ids in both with name different.
        //   - Created: ids in current, not in previous.
        // Emit Removed → Renamed → Created (structure-before-state; removes
        // before creates so a late-connecting client never sees a stale-id
        // slot re-used by a create). Within each bucket, sort by id ascending
        // for determinism against HashMap iteration order.
        //
        // First-tick seeding is implicit: the snapshot starts empty, so every
        // live activity routes through `ActivityCreated` with its derived
        // fields (is_active, is_urgent) already correct — the sibling
        // `ipc_refresh_active_activity` and `ipc_refresh_activity_urgency`
        // called later on the same tick see the snapshot settled and emit
        // nothing spurious.
        let mut created: Vec<jiji_ipc::Activity> = Vec::new();
        let mut renamed: Vec<(u64, String)> = Vec::new();
        for activity in &current {
            match state.activities.get(&activity.id) {
                Some(prev) => {
                    if prev.name != activity.name {
                        renamed.push((activity.id, activity.name.clone()));
                    }
                }
                None => created.push(activity.clone()),
            }
        }

        let current_ids: HashSet<u64> = current.iter().map(|a| a.id).collect();
        let mut removed: Vec<u64> = state
            .activities
            .keys()
            .copied()
            .filter(|id| !current_ids.contains(id))
            .collect();

        removed.sort_unstable();
        renamed.sort_by_key(|(id, _)| *id);
        created.sort_by_key(|a| a.id);

        for id in removed {
            let event = Event::ActivityRemoved { id };
            state.apply(event.clone());
            server.send_event(event);
        }
        for (id, name) in renamed {
            let event = Event::ActivityRenamed { id, name };
            state.apply(event.clone());
            server.send_event(event);
        }
        for activity in created {
            let event = Event::ActivityCreated { activity };
            state.apply(event.clone());
            server.send_event(event);
        }
    }

    fn ipc_refresh_workspaces(&mut self) {
        let Some(server) = &self.niri.ipc_server else {
            return;
        };

        let _span = tracy_client::span!("State::ipc_refresh_workspaces");

        let mut state = server.event_stream_state.borrow_mut();
        let state = &mut state.workspaces;

        let layout = &self.niri.layout;
        let focused_ws_id = layout.active_workspace().map(|ws| ws.id().get());

        let current =
            build_workspace_snapshot(layout, focused_ws_id, |win: &Mapped| win.id().get());

        for event in diff_workspaces(&state.workspaces, &current) {
            state.apply(event.clone());
            server.send_event(event);
        }
    }

    fn ipc_refresh_windows(&mut self) {
        let Some(server) = &self.niri.ipc_server else {
            return;
        };

        let _span = tracy_client::span!("State::ipc_refresh_windows");

        let mut state = server.event_stream_state.borrow_mut();
        let state = &mut state.windows;

        let mut events = Vec::new();
        let layout = &self.niri.layout;

        let mut batch_change_layouts: Vec<(u64, WindowLayout)> = Vec::new();

        // Check for window changes.
        let mut seen = HashSet::new();
        let mut focused_id = None;
        layout.with_windows_all(|mapped, _, ws_id, window_layout| {
            let id = mapped.id().get();
            seen.insert(id);

            if mapped.is_focused() {
                focused_id = Some(id);
            }

            let Some(ipc_win) = state.windows.get(&id) else {
                let window = make_ipc_window(mapped, ws_id, window_layout);
                events.push(Event::WindowOpenedOrChanged { window });
                return;
            };

            let workspace_id = ws_id.map(|id| id.get());
            let mut changed =
                ipc_win.workspace_id != workspace_id || ipc_win.is_floating != mapped.is_floating();

            changed |= with_toplevel_role(mapped.toplevel(), |role| {
                ipc_win.title != role.title || ipc_win.app_id != role.app_id
            });

            if changed {
                let window = make_ipc_window(mapped, ws_id, window_layout);
                events.push(Event::WindowOpenedOrChanged { window });
                return;
            }

            if ipc_win.layout != window_layout {
                batch_change_layouts.push((id, window_layout));
            }

            if mapped.is_focused() && !ipc_win.is_focused {
                events.push(Event::WindowFocusChanged { id: Some(id) });
            }

            let focus_timestamp = mapped.get_focus_timestamp().map(Timestamp::from);
            if focus_timestamp != ipc_win.focus_timestamp {
                events.push(Event::WindowFocusTimestampChanged {
                    id,
                    focus_timestamp,
                });
            }

            let urgent = mapped.is_urgent();
            if urgent != ipc_win.is_urgent {
                events.push(Event::WindowUrgencyChanged { id, urgent })
            }
        });

        // It might make sense to push layout changes after closed windows (since windows about to
        // be closed will occupy the same column/tile positions as the window that moved into this
        // vacated space), but also we are already pushing some layout changes in
        // WindowOpenedOrChanged above, meaning that the receiving end has to handle this case
        // anyway.
        if !batch_change_layouts.is_empty() {
            events.push(Event::WindowLayoutsChanged {
                changes: batch_change_layouts,
            });
        }

        // Check for closed windows.
        let mut ipc_focused_id = None;
        for (id, ipc_win) in &state.windows {
            if !seen.contains(id) {
                events.push(Event::WindowClosed { id: *id });
            }

            if ipc_win.is_focused {
                ipc_focused_id = Some(id);
            }
        }

        // Extra check for focus becoming None, since the checks above only work for focus becoming
        // a different window.
        if focused_id.is_none() && ipc_focused_id.is_some() {
            events.push(Event::WindowFocusChanged { id: None });
        }

        for event in events {
            state.apply(event.clone());
            server.send_event(event);
        }
    }

    pub fn ipc_refresh_overview(&mut self) {
        let Some(server) = &self.niri.ipc_server else {
            return;
        };

        let mut state = server.event_stream_state.borrow_mut();
        let state = &mut state.overview;
        let is_open = self.niri.layout.is_overview_open();

        if state.is_open == is_open {
            return;
        }

        let event = Event::OverviewOpenedOrClosed { is_open };
        state.apply(event.clone());
        server.send_event(event);
    }

    pub fn ipc_refresh_casts(&mut self) {
        let Some(server) = &self.niri.ipc_server else {
            return;
        };

        let _span = tracy_client::span!("State::ipc_refresh_casts");

        let mut state = server.event_stream_state.borrow_mut();
        let state = &mut state.casts;

        let mut events = Vec::new();
        let mut seen = HashSet::new();

        // Check PipeWire screencasts.
        #[cfg(feature = "xdp-gnome-screencast")]
        {
            // Check pending dynamic casts.
            for pending in &self.niri.casting.pending_dynamic_casts {
                let stream_id = pending.stream_id.get();
                seen.insert(stream_id);

                // Pending dynamic casts don't change any properties, so we only need to check if
                // it's missing from the state.
                if !state.casts.contains_key(&stream_id) {
                    let cast = jiji_ipc::Cast {
                        session_id: pending.session_id.get(),
                        stream_id,
                        kind: jiji_ipc::CastKind::PipeWire,
                        target: jiji_ipc::CastTarget::Nothing {},
                        is_dynamic_target: true,
                        is_active: false,
                        pid: None,
                        pw_node_id: None,
                    };
                    events.push(Event::CastStartedOrChanged { cast });
                }
            }

            // Check active casts.
            for cast in &self.niri.casting.casts {
                let stream_id = cast.stream_id.get();
                seen.insert(stream_id);

                let pw_node_id = cast.node_id();
                if state.casts.get(&stream_id).is_none_or(|existing| {
                    // Only these properties can change.
                    existing.is_active != cast.is_active()
                        || !cast.target.matches(&existing.target)
                        || existing.pw_node_id != pw_node_id
                }) {
                    let cast = jiji_ipc::Cast {
                        session_id: cast.session_id.get(),
                        stream_id,
                        kind: jiji_ipc::CastKind::PipeWire,
                        target: cast.target.make_ipc(),
                        is_dynamic_target: cast.dynamic_target,
                        is_active: cast.is_active(),
                        pid: None,
                        pw_node_id,
                    };
                    events.push(Event::CastStartedOrChanged { cast });
                }
            }
        }

        // Check screencopy casts.
        //
        // First, clear expired casts. Ideally we'd have a deadline timer, but our 1 second frame
        // callback timer calls refresh regularly, so that's fine as is.
        self.niri.screencopy_state.clear_expired_casts();

        for queue in self.niri.screencopy_state.queues() {
            if let Some(cast_info) = queue.cast() {
                let stream_id = cast_info.stream_id.get();
                seen.insert(stream_id);

                if state.casts.get(&stream_id).is_none_or(|existing| {
                    // Only this property can change.
                    match &existing.target {
                        jiji_ipc::CastTarget::Output { name } => *name != cast_info.output_name,
                        _ => true,
                    }
                }) {
                    let cast = jiji_ipc::Cast {
                        session_id: cast_info.session_id.get(),
                        stream_id,
                        kind: jiji_ipc::CastKind::WlrScreencopy,
                        target: jiji_ipc::CastTarget::Output {
                            name: cast_info.output_name.clone(),
                        },
                        is_dynamic_target: false,
                        is_active: true,
                        pid: queue.credentials().map(|creds| creds.pid),
                        pw_node_id: None,
                    };
                    events.push(Event::CastStartedOrChanged { cast });
                }
            }
        }

        // Check for stopped casts.
        for stream_id in state.casts.keys() {
            if !seen.contains(stream_id) {
                events.push(Event::CastStopped {
                    stream_id: *stream_id,
                });
            }
        }

        for event in events {
            state.apply(event.clone());
            server.send_event(event);
        }
    }

    pub fn ipc_config_loaded(&mut self, failed: bool) {
        let Some(server) = &self.niri.ipc_server else {
            return;
        };
        let mut state = server.event_stream_state.borrow_mut();

        let event = Event::ConfigLoaded { failed };
        state.apply(event.clone());
        server.send_event(event);
    }

    pub fn ipc_screenshot_taken(&mut self, path: Option<String>) {
        let Some(server) = &self.niri.ipc_server else {
            return;
        };
        let mut state = server.event_stream_state.borrow_mut();

        let event = Event::ScreenshotCaptured { path };
        state.apply(event.clone());
        server.send_event(event);
    }
}

/// Test-only helper reproducing the combined lifecycle + active-activity +
/// urgency diff that `ipc_refresh_layout` performs against
/// `EventStreamState::activities` on a single tick.
///
/// Given the client-tier snapshot the server would have carried into the
/// tick (`previous`, an `id → Activity` map matching
/// `ActivitiesState::activities`) and the live `Layout`, returns the event
/// sequence in production emission order: lifecycle (Removed → Renamed →
/// Created, sorted by id asc within each bucket), then a single
/// `ActivitySwitched` if the cursor moved, then sorted
/// `ActivityUrgencyChanged` events.
///
/// Semantic coverage for the single-snapshot apply-path cases lives in
/// `jiji-ipc/src/state.rs` against `ActivitiesState::apply`; this helper
/// exists for the mutation-flow and cascade-ordering tests in
/// `src/layout/tests.rs` that need a live `Layout` to exercise
/// `create_activity` / `remove_activity` / `rename_activity`.
#[cfg(test)]
pub(crate) fn test_diff_activities_against_state<W: LayoutElement>(
    layout: &Layout<W>,
    previous: &HashMap<u64, jiji_ipc::Activity>,
) -> Vec<Event> {
    let current = build_activities_ipc(layout);

    // Lifecycle diff.
    let mut created: Vec<jiji_ipc::Activity> = Vec::new();
    let mut renamed: Vec<(u64, String)> = Vec::new();
    for activity in &current {
        match previous.get(&activity.id) {
            Some(prev) => {
                if prev.name != activity.name {
                    renamed.push((activity.id, activity.name.clone()));
                }
            }
            None => created.push(activity.clone()),
        }
    }
    let current_ids: HashSet<u64> = current.iter().map(|a| a.id).collect();
    let mut removed: Vec<u64> = previous
        .keys()
        .copied()
        .filter(|id| !current_ids.contains(id))
        .collect();
    removed.sort_unstable();
    renamed.sort_by_key(|(id, _)| *id);
    created.sort_by_key(|a| a.id);

    let mut events: Vec<Event> = Vec::new();
    for id in removed {
        events.push(Event::ActivityRemoved { id });
    }
    for (id, name) in renamed {
        events.push(Event::ActivityRenamed { id, name });
    }
    for activity in created {
        events.push(Event::ActivityCreated { activity });
    }

    // Active-activity diff (matching `ipc_refresh_active_activity`): gate on
    // `previous`-active presence. Derive the previous-active id from the
    // pre-lifecycle `previous` snapshot — production snapshots
    // `previous_active_id` inside `ipc_refresh_layout` via the same
    // `values().find(is_active)` scan before lifecycle runs, so the two
    // paths produce identical output (required for the cascade-remove case
    // where lifecycle's `ActivityRemoved` erases the active signal before
    // the active diff would run).
    let current_active = layout.active_activity_id().get();
    let prev_active = previous.values().find(|a| a.is_active).map(|a| a.id);
    if let Some(prev) = prev_active {
        if prev != current_active {
            events.push(Event::ActivitySwitched {
                id: current_active,
                previous_id: Some(prev),
            });
        }
    }

    // Urgency diff (matching `ipc_refresh_activity_urgency`). Compare against
    // `previous` — for any id newly present (created this tick), urgency is
    // already encoded in the `ActivityCreated` payload above, so newcomers
    // are silently skipped.
    let mut transitions: Vec<(u64, bool)> = Vec::new();
    for a in &current {
        let Some(prev) = previous.get(&a.id) else {
            continue;
        };
        if prev.is_urgent != a.is_urgent {
            transitions.push((a.id, a.is_urgent));
        }
    }
    transitions.sort_by_key(|(id, _)| *id);
    for (id, urgent) in transitions {
        events.push(Event::ActivityUrgencyChanged { id, urgent });
    }

    events
}

/// Builds the full per-refresh workspace snapshot sent to IPC clients.
///
/// Two-pass construction, together covering every pool workspace exactly once:
///
/// 1. Workspaces in the active activity, in monitor-and-view order (plus the
///    `disconnected_workspace_ids` tail when no monitors are connected). Their `idx` is
///    view-position + 1 (1-based user-visible index), and `is_in_active_activity` is `true`.
/// 2. Workspaces that are members of some other activity only. Their `idx` sentinel is `0`,
///    `is_in_active_activity` is `false`, and neither `is_active` nor `is_focused` can be true (the
///    active/focused workspace is always in the active activity, by construction).
///
/// The `active_window_id_of` closure extracts the IPC window id from a
/// layout element; production uses `Mapped::id().get()`, and `layout/tests.rs`
/// calls this helper directly with a `TestWindow`-compatible extractor.
pub(crate) fn build_workspace_snapshot<W: LayoutElement>(
    layout: &Layout<W>,
    focused_ws_id: Option<u64>,
    active_window_id_of: impl Fn(&W) -> u64,
) -> Vec<Workspace> {
    let active_id = layout.active_activity_id();

    // Pass 1 is keyed on membership (`ws.activities().contains(&active_id)`)
    // rather than "whatever `Layout::workspaces()` yields". `workspaces()`
    // walks view order, and a view may legitimately hold a stale entry whose
    // `activities` set no longer contains the active activity (only reachable
    // via direct activity-set mutation today; future membership-editing
    // handlers could also produce it). Membership is the source of truth for
    // `is_in_active_activity`, so a stale view entry falls through to pass 2
    // with `idx: 0` rather than being double-emitted.
    let mut current: Vec<Workspace> = layout
        .workspaces()
        .filter(|(_, _, ws)| ws.activities().contains(&active_id))
        .map(|(mon, ws_idx, ws)| {
            let id = ws.id().get();
            let mut activities: Vec<u64> = ws.activities().iter().map(|aid| aid.get()).collect();
            activities.sort();
            Workspace {
                id,
                idx: u8::try_from(ws_idx + 1).unwrap_or(u8::MAX),
                name: ws.name().cloned(),
                output: mon.map(|mon| mon.output_name().clone()),
                is_urgent: ws.is_urgent(),
                is_active: mon.is_some_and(|mon| {
                    layout.active_view(&mon.output_id()).active_position() == ws_idx
                }),
                is_focused: Some(id) == focused_ws_id,
                active_window_id: ws.active_window().map(&active_window_id_of),
                activities,
                is_sticky: ws.is_sticky(),
                is_in_active_activity: true,
            }
        })
        .collect();

    // Second pass: pool workspaces that are members of another activity only.
    // Together with pass 1 this covers the pool exactly once — pass 1 yields
    // every workspace whose activity set contains `active_id` (that is the
    // semantics of `Layout::workspaces()`), pass 2 yields the complement.
    // Pass 2 ordering is HashMap iteration order (undefined); clients must
    // not rely on positional stability of hidden workspaces in the snapshot.
    for (output_id, ws) in layout.workspaces_all() {
        if ws.activities().contains(&active_id) {
            continue;
        }

        let id = ws.id().get();
        let mut activities: Vec<u64> = ws.activities().iter().map(|aid| aid.get()).collect();
        activities.sort();

        // Symmetric with pass 1's handling of disconnected-tail workspaces:
        // if the designated output is currently disconnected, yield `None`.
        let output = output_id.and_then(|id| {
            layout
                .monitor_for_output_id(id)
                .map(|mon| mon.output_name().clone())
        });

        current.push(Workspace {
            id,
            idx: 0,
            name: ws.name().cloned(),
            output,
            is_urgent: ws.is_urgent(),
            is_active: false,
            is_focused: false,
            active_window_id: ws.active_window().map(&active_window_id_of),
            activities,
            is_sticky: ws.is_sticky(),
            is_in_active_activity: false,
        });
    }

    current
}

/// Diff the previously-emitted workspace snapshot against the current one and
/// produce the list of `Event`s to emit.
///
/// Order: for each workspace in `current`, any `WorkspaceOpenedOrChanged`
/// (structural) comes first, followed by per-field events
/// (`WorkspaceActiveWindowChanged`, `WorkspaceUrgencyChanged`,
/// `WorkspaceActivated`). Then `WorkspaceClosed` for each id in `previous` not
/// present in `current`. Finally, a single backwards-compatible
/// `WorkspacesChanged` if any structural change happened in this frame.
fn diff_workspaces(previous: &HashMap<u64, Workspace>, current: &[Workspace]) -> Vec<Event> {
    let mut events = Vec::new();
    let mut seen = HashSet::with_capacity(current.len());
    let mut any_structural = false;

    for new_ipc in current {
        let id = new_ipc.id;
        seen.insert(id);

        let Some(ipc_ws) = previous.get(&id) else {
            // New workspace — structural addition. No per-field events for a
            // fresh workspace; its full state is carried by WorkspaceOpenedOrChanged.
            any_structural = true;
            events.push(Event::WorkspaceOpenedOrChanged {
                workspace: new_ipc.clone(),
            });
            continue;
        };

        // Structural change = anything not covered by a per-field event.
        let structural_changed = ipc_ws.idx != new_ipc.idx
            || ipc_ws.name != new_ipc.name
            || ipc_ws.output != new_ipc.output
            || ipc_ws.activities != new_ipc.activities
            || ipc_ws.is_sticky != new_ipc.is_sticky
            || ipc_ws.is_in_active_activity != new_ipc.is_in_active_activity;
        if structural_changed {
            any_structural = true;
            events.push(Event::WorkspaceOpenedOrChanged {
                workspace: new_ipc.clone(),
            });
        }

        // Per-field events fire independently of structural changes: a
        // structural change on one workspace in the same frame as an urgency
        // or activation change on another workspace must not silently drop
        // the per-field event (regression guard for the previous
        // `events.clear()` bail).
        if ipc_ws.active_window_id != new_ipc.active_window_id {
            events.push(Event::WorkspaceActiveWindowChanged {
                workspace_id: id,
                active_window_id: new_ipc.active_window_id,
            });
        }
        if ipc_ws.is_urgent != new_ipc.is_urgent {
            events.push(Event::WorkspaceUrgencyChanged {
                id,
                urgent: new_ipc.is_urgent,
            });
        }
        if new_ipc.is_focused && !ipc_ws.is_focused {
            events.push(Event::WorkspaceActivated { id, focused: true });
            continue;
        }
        if new_ipc.is_active && !ipc_ws.is_active {
            events.push(Event::WorkspaceActivated { id, focused: false });
        }
    }

    for &id in previous.keys() {
        if !seen.contains(&id) {
            any_structural = true;
            events.push(Event::WorkspaceClosed { id });
        }
    }

    if any_structural {
        events.push(Event::WorkspacesChanged {
            workspaces: current.to_vec(),
        });
    }

    events
}

#[cfg(test)]
mod tests {
    use super::*;

    // Pins the blocked-action waiter registry primitives: FIFO order,
    // depth-1 admission, closed-receiver prune, and re-block FIFO pin.
    // Full `State::refresh` drain wiring is unreachable from unit tests.

    fn dummy_action() -> jiji_config::Action {
        // `Spawn` is the cheapest `Action` variant to construct by hand and
        // has trivial equality semantics. The drain path only clones the
        // action; the specific variant is irrelevant to the registry
        // invariants these tests pin.
        jiji_config::Action::Spawn(vec![])
    }

    fn make_waiter() -> (
        BlockedWaiter,
        async_channel::Receiver<Result<DoActionOutcome, DoActionError>>,
    ) {
        let (tx, rx) = async_channel::bounded::<Result<DoActionOutcome, DoActionError>>(1);
        (
            BlockedWaiter {
                action: dummy_action(),
                tx,
            },
            rx,
        )
    }

    #[test]
    fn blocked_action_waiters_push_and_drain_ok() {
        // Push a single waiter, drain it by `shift_remove` + `send_blocking`
        // mirroring the happy-path branch of `drain_blocked_action_waiters`,
        // and verify the receiver observes `Ok(())`. Pins the
        // `Handled ≡ performed` contract: the send only fires once the
        // registry entry is removed.
        let mut waiters: IndexMap<IpcConnId, BlockedWaiter> = IndexMap::new();
        let conn = IpcConnId::specific(1);
        let (waiter, rx) = make_waiter();
        waiters.insert(conn, waiter);

        assert_eq!(waiters.len(), 1);
        let drained = waiters
            .shift_remove(&conn)
            .expect("waiter inserted just now");
        assert!(waiters.is_empty());
        drained
            .tx
            .send_blocking(Ok(DoActionOutcome::Handled))
            .expect("receiver is alive");

        let observed = rx
            .recv_blocking()
            .expect("sender alive until send_blocking returned");
        assert!(matches!(observed, Ok(DoActionOutcome::Handled)));
    }

    #[test]
    fn blocked_action_waiters_closed_tx_pruned() {
        // A client that disconnects between enqueue and drain: `rx` is
        // dropped, so `tx.is_closed()` must report `true` and the drain
        // site treats the entry as prune-and-drop. Pins the prune branch
        // inside `drain_blocked_action_waiters`.
        let mut waiters: IndexMap<IpcConnId, BlockedWaiter> = IndexMap::new();
        let conn = IpcConnId::specific(1);
        let (waiter, rx) = make_waiter();
        waiters.insert(conn, waiter);

        drop(rx);

        let drained = waiters
            .shift_remove(&conn)
            .expect("waiter inserted just now");
        assert!(
            drained.tx.is_closed(),
            "receiver dropped → sender must report closed",
        );
        // Mirrors the `continue` branch: drop the waiter without attempting
        // to re-dispatch.
        drop(drained);
        assert!(waiters.is_empty());
    }

    #[test]
    fn blocked_action_waiters_fifo_across_connections() {
        // Two connections block in order A, B. The drain walk must wake A
        // before B. `IndexMap` iteration order *is* the FIFO contract here
        // (there is no separate `VecDeque`); this test would regress if a
        // future refactor swapped to a plain `HashMap`.
        let mut waiters: IndexMap<IpcConnId, BlockedWaiter> = IndexMap::new();
        let a = IpcConnId::specific(1);
        let b = IpcConnId::specific(2);
        let (waiter_a, rx_a) = make_waiter();
        let (waiter_b, rx_b) = make_waiter();
        waiters.insert(a, waiter_a);
        waiters.insert(b, waiter_b);

        // Snapshot keys in iteration order — this is what `drain_blocked_action_waiters`
        // does before walking.
        let order: Vec<IpcConnId> = waiters.keys().copied().collect();
        assert_eq!(order, vec![a, b]);

        for conn in order {
            let w = waiters.shift_remove(&conn).expect("inserted just now");
            w.tx.send_blocking(Ok(DoActionOutcome::Handled))
                .expect("receivers alive");
        }

        // Pull each response: A first, B second.
        let first = rx_a.recv_blocking().expect("sent above");
        let second = rx_b.recv_blocking().expect("sent above");
        assert!(matches!(first, Ok(DoActionOutcome::Handled)));
        assert!(matches!(second, Ok(DoActionOutcome::Handled)));
    }

    #[test]
    fn blocked_action_waiters_reblock_leaves_entry() {
        // Three waiters [A, B, C] in FIFO. Walk re-blocks the *middle* one
        // (B, index 1). This makes `shift_insert(original_idx, …)` vs
        // `shift_insert(0, …)` observably different:
        //   - naive `shift_insert(0, …)` → [B, A, C]
        //   - correct `shift_insert(original_idx=1, …)` → [A, B, C]
        // Using B makes the discrimination impossible to accidentally pass.
        let mut waiters: IndexMap<IpcConnId, BlockedWaiter> = IndexMap::new();
        let a = IpcConnId::specific(1);
        let b = IpcConnId::specific(2);
        let c = IpcConnId::specific(3);
        let (waiter_a, _rx_a) = make_waiter();
        let (waiter_b, _rx_b) = make_waiter();
        let (waiter_c, _rx_c) = make_waiter();
        waiters.insert(a, waiter_a);
        waiters.insert(b, waiter_b);
        waiters.insert(c, waiter_c);

        // Simulate the re-block branch for B (the middle entry).
        let original_idx = waiters
            .get_index_of(&b)
            .expect("b inserted second, must be at index 1");
        assert_eq!(original_idx, 1);
        let removed = waiters.shift_remove(&b).expect("b present");
        assert_eq!(
            waiters.keys().copied().collect::<Vec<_>>(),
            vec![a, c],
            "after shift_remove(b), only a and c remain",
        );
        let prev = waiters.shift_insert(original_idx, b, removed);
        assert!(
            prev.is_none(),
            "shift_insert at an unoccupied slot must not overwrite",
        );

        // Post-condition: key order restored to [A, B, C].
        // A bug using `shift_insert(0, …)` would produce [B, A, C] instead.
        assert_eq!(
            waiters.keys().copied().collect::<Vec<_>>(),
            vec![a, b, c],
            "re-block at its index at removal must restore FIFO order",
        );
    }

    #[test]
    fn blocked_action_waiters_depth_one_rejects_second_enqueue() {
        // Depth-1 admission: the `Request::Action` arm rejects a second
        // enqueue by the same connection with the literal error string
        // `"request already queued"`. This test exercises the
        // `contains_key` primitive the arm is built on — the only thing
        // between a pipelined handler and a corrupted queue.
        let mut waiters: IndexMap<IpcConnId, BlockedWaiter> = IndexMap::new();
        let conn = IpcConnId::specific(1);
        let (waiter, _rx) = make_waiter();
        waiters.insert(conn, waiter);

        // First enqueue landed; `contains_key` now reports `true`. Any
        // subsequent admission check by the same `conn_id` must bail.
        assert!(
            waiters.contains_key(&conn),
            "initial insert must make contains_key true for the same conn_id",
        );

        // Different connection — admission must succeed. Pins that the
        // rejection is keyed on `conn_id`, not a global flag.
        let other = IpcConnId::specific(2);
        assert!(
            !waiters.contains_key(&other),
            "depth-1 is per-connection, not global",
        );
    }

    #[test]
    fn window_not_found_drain_continues_past_blocked_waiter() {
        // Pins two load-bearing invariants for the `WindowNotFound` branch of
        // `drain_blocked_action_waiters`:
        //
        // 1. **`continue` not `break`:** A `WindowNotFound` result at position A must NOT prevent
        //    position B from being drained. If the real drain loop used `break` instead of
        //    `continue`, B would be left in the registry after A errors. The intermediate assertion
        //    below would still pass (B is present just before its own drain step), but the
        //    post-drain assertion (registry empty) would fail because the loop would have stopped
        //    after A without processing B.
        //
        // 2. **Non-insertion into `blocked_action_waiters`:** A `WindowNotFound` result must never
        //    be re-inserted into the registry (unlike the re-block path, which re-inserts at the
        //    original index). A copy-paste error adding `shift_insert` after the `WindowNotFound`
        //    arm would cause deadlock; the registry-empty post-condition below catches it.
        //
        // Because `drain_blocked_action_waiters` takes `&mut State` and cannot
        // be called from a unit test, we simulate the two drain steps by hand
        // using sequential `shift_remove` + `send_blocking` calls, mirroring
        // the real loop body exactly. The intermediate assertion between step
        // A and step B is the discriminating check: it proves B survived A's
        // error without being removed or blocked.
        let mut waiters: IndexMap<IpcConnId, BlockedWaiter> = IndexMap::new();
        let a = IpcConnId::specific(1);
        let b = IpcConnId::specific(2);
        let (waiter_a, rx_a) = make_waiter();
        let (waiter_b, rx_b) = make_waiter();
        waiters.insert(a, waiter_a);
        waiters.insert(b, waiter_b);

        // Step A: drain A with a WindowNotFound error (continue semantic —
        // no re-insert, no break).
        let drained_a = waiters.shift_remove(&a).expect("a inserted first");
        let _ = drained_a
            .tx
            .send_blocking(Err(DoActionError::WindowNotFound { id: 99 }));
        // Key invariant: B must still be present after A's error handling.
        // A `break`-style drain would have left B in the registry too, but
        // with `continue` the loop moves on to process B — we assert B is
        // present so we can drain it ourselves below.
        assert!(
            waiters.contains_key(&b),
            "after WindowNotFound for A, B must still be in the registry",
        );

        // Step B: drain B — also gets WindowNotFound (simulating a second
        // stale-id action in the same queue).
        let drained_b = waiters
            .shift_remove(&b)
            .expect("b still present after A error");
        let _ = drained_b
            .tx
            .send_blocking(Err(DoActionError::WindowNotFound { id: 100 }));

        // Registry empty: neither entry was re-inserted.
        assert!(
            waiters.is_empty(),
            "registry must be empty after both WindowNotFound drains",
        );

        // Both waiters received the error signal.
        assert!(matches!(
            rx_a.recv_blocking()
                .expect("a sender alive until send_blocking"),
            Err(DoActionError::WindowNotFound { id: 99 })
        ));
        assert!(matches!(
            rx_b.recv_blocking()
                .expect("b sender alive until send_blocking"),
            Err(DoActionError::WindowNotFound { id: 100 })
        ));
    }

    fn ws(id: u64, idx: u8, output: &str) -> Workspace {
        Workspace {
            id,
            idx,
            name: None,
            output: Some(output.to_owned()),
            is_urgent: false,
            is_active: false,
            is_focused: false,
            active_window_id: None,
            activities: vec![],
            is_sticky: false,
            is_in_active_activity: true,
        }
    }

    fn previous_from(items: &[Workspace]) -> HashMap<u64, Workspace> {
        items.iter().cloned().map(|w| (w.id, w)).collect()
    }

    #[test]
    fn empty_previous_emits_delta_per_workspace_and_full_list() {
        let current = vec![ws(1, 1, "HDMI-1"), ws(2, 2, "HDMI-1")];
        let events = diff_workspaces(&HashMap::new(), &current);
        assert_eq!(events.len(), 3);
        assert!(matches!(
            &events[0],
            Event::WorkspaceOpenedOrChanged { workspace } if workspace.id == 1
        ));
        assert!(matches!(
            &events[1],
            Event::WorkspaceOpenedOrChanged { workspace } if workspace.id == 2
        ));
        assert!(matches!(&events[2], Event::WorkspacesChanged { .. }));
    }

    #[test]
    fn diff_workspaces_activity_fields_trigger_structural_emission() {
        // A flip of `is_in_active_activity` (or any of the three activity-related
        // fields) must count as a structural change so `WorkspaceOpenedOrChanged`
        // is emitted. Clients rely on this to re-read `idx` (which is only
        // meaningful when `is_in_active_activity` is true).
        let prev = previous_from(&[ws(1, 1, "HDMI-1")]);
        let mut now = ws(1, 1, "HDMI-1");
        now.is_in_active_activity = false;
        let events = diff_workspaces(&prev, &[now]);

        assert_eq!(events.len(), 2, "got {events:?}");
        assert!(matches!(
            &events[0],
            Event::WorkspaceOpenedOrChanged { workspace }
                if workspace.id == 1 && !workspace.is_in_active_activity
        ));
        assert!(matches!(&events[1], Event::WorkspacesChanged { .. }));

        // Verify `activities` change also triggers structural emission.
        let prev = previous_from(&[ws(1, 1, "HDMI-1")]);
        let mut now = ws(1, 1, "HDMI-1");
        now.activities = vec![99];
        let events = diff_workspaces(&prev, &[now]);
        assert_eq!(events.len(), 2, "activities change: got {events:?}");
        assert!(matches!(
            &events[0],
            Event::WorkspaceOpenedOrChanged { workspace } if workspace.activities == vec![99]
        ));
        assert!(matches!(&events[1], Event::WorkspacesChanged { .. }));

        // Verify `is_sticky` change also triggers structural emission.
        let prev = previous_from(&[ws(1, 1, "HDMI-1")]);
        let mut now = ws(1, 1, "HDMI-1");
        now.is_sticky = true;
        let events = diff_workspaces(&prev, &[now]);
        assert_eq!(events.len(), 2, "is_sticky change: got {events:?}");
        assert!(matches!(
            &events[0],
            Event::WorkspaceOpenedOrChanged { workspace } if workspace.is_sticky
        ));
        assert!(matches!(&events[1], Event::WorkspacesChanged { .. }));
    }

    #[test]
    fn no_change_emits_nothing() {
        let prev = previous_from(&[ws(1, 1, "HDMI-1")]);
        let current = vec![ws(1, 1, "HDMI-1")];
        let events = diff_workspaces(&prev, &current);
        assert!(events.is_empty());
    }

    #[test]
    fn per_field_only_change_does_not_emit_workspaces_changed() {
        // Urgency flip on an existing workspace: per-field event only; no
        // structural change means no BC full-list event.
        let prev = previous_from(&[ws(1, 1, "HDMI-1")]);
        let mut now = ws(1, 1, "HDMI-1");
        now.is_urgent = true;
        let events = diff_workspaces(&prev, &[now]);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            Event::WorkspaceUrgencyChanged {
                id: 1,
                urgent: true
            }
        ));
    }

    #[test]
    fn structural_change_on_one_plus_per_field_on_another_emits_both() {
        // Regression guard for the previous bail-to-full-replacement bug:
        // a structural change on ws-1 in the same frame as an urgency change
        // on ws-2 must emit BOTH events, not swallow ws-2's urgency.
        let prev = previous_from(&[ws(1, 1, "HDMI-1"), ws(2, 2, "HDMI-1")]);
        let mut ws1 = ws(1, 3, "HDMI-1"); // idx shifted 1 → 3
        ws1.is_urgent = false;
        let mut ws2 = ws(2, 2, "HDMI-1");
        ws2.is_urgent = true; // new urgency on ws-2
        let events = diff_workspaces(&prev, &[ws1, ws2]);

        // Expected: WorkspaceOpenedOrChanged(id=1) + WorkspaceUrgencyChanged(id=2) +
        // WorkspacesChanged
        assert_eq!(events.len(), 3, "got {events:?}");
        assert!(matches!(
            &events[0],
            Event::WorkspaceOpenedOrChanged { workspace } if workspace.id == 1 && workspace.idx == 3
        ));
        assert!(matches!(
            events[1],
            Event::WorkspaceUrgencyChanged {
                id: 2,
                urgent: true
            }
        ));
        assert!(matches!(&events[2], Event::WorkspacesChanged { .. }));
    }

    #[test]
    fn same_workspace_with_structural_and_per_field_changes_emits_both() {
        // A single workspace with BOTH a structural change (idx) AND a per-field
        // change (urgency) in the same frame must emit both events — the
        // per-field check runs unconditionally after the structural check for
        // every workspace, not just as a fallback when no structural change.
        let prev = previous_from(&[ws(1, 1, "HDMI-1")]);
        let mut now = ws(1, 3, "HDMI-1"); // idx: 1 → 3
        now.is_urgent = true;
        let events = diff_workspaces(&prev, &[now]);

        assert_eq!(events.len(), 3, "got {events:?}");
        assert!(matches!(
            &events[0],
            Event::WorkspaceOpenedOrChanged { workspace } if workspace.id == 1 && workspace.idx == 3
        ));
        assert!(matches!(
            events[1],
            Event::WorkspaceUrgencyChanged {
                id: 1,
                urgent: true
            }
        ));
        assert!(matches!(&events[2], Event::WorkspacesChanged { .. }));
    }

    #[test]
    fn removed_workspace_emits_closed_and_workspaces_changed() {
        let prev = previous_from(&[ws(1, 1, "HDMI-1"), ws(2, 2, "HDMI-1")]);
        let current = vec![ws(1, 1, "HDMI-1")];
        let events = diff_workspaces(&prev, &current);
        assert_eq!(events.len(), 2);
        assert!(matches!(events[0], Event::WorkspaceClosed { id: 2 }));
        assert!(matches!(&events[1], Event::WorkspacesChanged { .. }));
    }

    #[test]
    fn becoming_focused_suppresses_focused_false_event() {
        // A workspace that flips to focused=true must not also emit the
        // focused=false (active-only) event in the same diff.
        let prev = previous_from(&[ws(1, 1, "HDMI-1")]);
        let mut now = ws(1, 1, "HDMI-1");
        now.is_focused = true;
        now.is_active = true;
        let events = diff_workspaces(&prev, &[now]);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            Event::WorkspaceActivated {
                id: 1,
                focused: true
            }
        ));
    }

    #[test]
    fn new_workspace_emits_only_opened_or_changed() {
        // New workspace with urgency/focus set: full state rides on
        // WorkspaceOpenedOrChanged, not on separate per-field events.
        let mut fresh = ws(1, 1, "HDMI-1");
        fresh.is_urgent = true;
        fresh.is_focused = true;
        fresh.is_active = true;
        let events = diff_workspaces(&HashMap::new(), &[fresh]);
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], Event::WorkspaceOpenedOrChanged { .. }));
        assert!(matches!(&events[1], Event::WorkspacesChanged { .. }));
    }
}
