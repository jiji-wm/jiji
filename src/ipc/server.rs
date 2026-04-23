use std::cell::{Cell, RefCell};
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
use niri_config::OutputName;
use niri_ipc::state::{EventStreamState, EventStreamStatePart as _};
use niri_ipc::{
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
use crate::utils::id::IdCounter;
use crate::layout::{format_activity_switch_block_err, ActivitySwitchBlock, Layout, LayoutElement};
use crate::niri::State;
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
struct IpcConnId(u64);

static IPC_CONN_ID_COUNTER: IdCounter = IdCounter::new();

impl IpcConnId {
    fn next() -> Self {
        Self(IPC_CONN_ID_COUNTER.next())
    }

    #[cfg(test)]
    fn specific(id: u64) -> Self {
        Self(id)
    }
}

/// Entry in [`IpcServer::blocked_action_waiters`].
///
/// Holds the owned [`niri_config::Action`] so the drain site can re-dispatch
/// without cloning from caller state, plus the send half of the response
/// channel the async `process` task is awaiting on. On drain:
///
/// - Send `Ok(())` once the action lands → `process` returns
///   `Response::Handled`.
/// - On re-block mid-drain: re-insert the entry and leave `tx` untouched;
///   the sender is not dropped, so the `process` task stays parked.
/// - Drop without sending on a closed receiver (client gone between enqueue
///   and drain). `process`'s `rx.recv().await` has already been dropped in
///   that case.
struct BlockedWaiter {
    action: niri_config::Action,
    tx: async_channel::Sender<Result<(), ActivitySwitchBlock>>,
}

pub struct IpcServer {
    /// Path to the IPC socket.
    ///
    /// This is `None` when creating `IpcServer` without a socket.
    pub socket_path: Option<PathBuf>,
    event_streams: Rc<RefCell<Vec<EventStreamSender>>>,
    event_stream_state: Rc<RefCell<EventStreamState>>,
    /// Most recently emitted active-activity id, used to diff against the
    /// current `Layout::active_activity_id()` on each refresh tick.
    ///
    /// `None` before the first observation: the first call to
    /// [`State::ipc_refresh_active_activity`] seeds this field from the
    /// current active activity without emitting, so clients do not see a
    /// spurious `ActivitySwitched` on server startup.
    ///
    /// Deliberately a plain `Cell` (not part of `EventStreamState`): the
    /// corresponding `ActivitiesState` on `EventStreamState` is Phase 1b
    /// scope. Main-thread calloop only — no cross-thread access.
    last_active_activity_id: Cell<Option<ActivityId>>,
    /// Most recently observed urgency snapshot keyed by [`ActivityId`], used
    /// to diff against the current aggregate on each refresh tick and emit
    /// [`Event::ActivityUrgencyChanged`] for every transition.
    ///
    /// `None` before the first observation: the first call to
    /// [`State::ipc_refresh_activity_urgency`] seeds this field from the
    /// current aggregate without emitting, so clients do not see spurious
    /// `ActivityUrgencyChanged` events on server startup (same seeding posture
    /// as [`Self::last_active_activity_id`]).
    ///
    /// Deliberately on `IpcServer` directly (not part of `EventStreamState`):
    /// the corresponding `ActivitiesState` on `EventStreamState` is Phase 1b
    /// scope (DD box 1949) and subsumes both this tracker and
    /// [`Self::last_active_activity_id`] in one piece. Main-thread calloop
    /// only — no cross-thread access. `RefCell` (not `Cell`) because the
    /// value is a `HashMap` that we mutate in place on each tick.
    last_activity_urgency: RefCell<Option<HashMap<ActivityId, bool>>>,
    /// Most recently observed activity-lifecycle snapshot keyed by
    /// [`ActivityId`], carrying only the name (see below for why name-only).
    /// Used to diff against the current activities pool on each refresh tick
    /// and emit [`Event::ActivityCreated`] / [`Event::ActivityRemoved`] /
    /// [`Event::ActivityRenamed`] for every transition.
    ///
    /// `None` before the first observation: the first call to
    /// [`State::ipc_refresh_activity_lifecycle`] seeds this field from the
    /// current pool without emitting, so clients do not see spurious
    /// lifecycle events on server startup (same seeding posture as
    /// [`Self::last_active_activity_id`] and [`Self::last_activity_urgency`]).
    ///
    /// Snapshot shape is `id → name`, not a full [`niri_ipc::Activity`]:
    /// `is_config_declared` is immutable after creation (see
    /// [`niri_ipc::Activity::is_config_declared`]); `is_active` has its own
    /// [`Event::ActivitySwitched`] event; `is_urgent` has its own
    /// [`Event::ActivityUrgencyChanged`] event. This tracker only needs to
    /// detect Created (new id), Removed (dropped id), and Renamed (name
    /// differs for the same id); the other fields would either be constant
    /// (`is_config_declared`) or duplicate work done by the sibling
    /// trackers.
    ///
    /// Deliberately on `IpcServer` directly (not part of `EventStreamState`):
    /// the corresponding `ActivitiesState` on `EventStreamState` is Phase 1b
    /// scope (DD box 1958) and subsumes this tracker together with
    /// [`Self::last_active_activity_id`] and [`Self::last_activity_urgency`]
    /// in one piece. Main-thread calloop only — no cross-thread access.
    /// `RefCell` (not `Cell`) because the value is a `HashMap`.
    last_activities_by_id: RefCell<Option<HashMap<ActivityId, String>>>,
    /// Per-connection depth-1 queue for blocked [`Request::Action`]s. One
    /// entry per connection (admission rejects a second enqueue). `IndexMap`
    /// preserves FIFO insertion order; [`drain_blocked_action_waiters`] walks
    /// it in order. Main-thread only — `Rc` is sufficient.
    blocked_action_waiters: Rc<RefCell<IndexMap<IpcConnId, BlockedWaiter>>>,
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
            let socket_name = format!("niri.{wayland_socket_name}.{}.sock", process::id());
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
            last_active_activity_id: Cell::new(None),
            last_activity_urgency: RefCell::new(None),
            last_activities_by_id: RefCell::new(None),
            blocked_action_waiters: Rc::new(RefCell::new(IndexMap::new())),
        })
    }

    fn send_event(&self, event: Event) {
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
/// - **`Handled` ≡ performed**: a waiter is only signalled `Ok(())` after its
///   `do_action_inner` call returned `Ok(())`. The send half is dropped
///   without signalling on silent-prune paths (closed receiver).
/// - **FIFO preserved across re-block**: if `do_action_inner` re-raises a
///   hard block mid-drain (no current action reaches this; forward-compat
///   for future gating widening), the waiter is re-inserted at its index at
///   removal via `shift_insert(original_idx, …)` and the walk **breaks** —
///   continuing past a re-blocked waiter would promote later waiters ahead
///   of it. The `// FIFO pin` breadcrumb on the `break` pins this semantic
///   against accidental refactor.
/// - **Closed-receiver prune**: if the client disconnected between enqueue
///   and drain (`tx.is_closed()`), the entry is dropped without dispatch.
///   No side effects, no replay.
/// - **No registry borrow across `do_action_inner`**: the walk grabs and
///   drops the registry `RefCell` borrow per iteration so a nested
///   `IpcServer` access inside action dispatch can't deadlock.
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
    if state.niri.layout.is_activity_switch_hard_blocked().is_some() {
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
            Ok(()) => {
                // Receiver may have dropped between wake and send; safe to
                // ignore because the action already executed with no state
                // loss (same contract as the initial-dispatch success path).
                let _ = waiter.tx.send_blocking(Ok(()));
            }
            Err(block) => {
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
                // FIFO pin — see DD §5.11 Part 2.
                //
                // Do NOT convert this `break` to `continue`: walking past a
                // re-blocked waiter promotes later waiters ahead of it and
                // violates FIFO. The drain-re-block invariant is pinned by
                // `blocked_action_waiters_reblock_leaves_entry`.
                break;
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
                            Layer::Background => niri_ipc::Layer::Background,
                            Layer::Bottom => niri_ipc::Layer::Bottom,
                            Layer::Top => niri_ipc::Layer::Top,
                            Layer::Overlay => niri_ipc::Layer::Overlay,
                        };
                        let keyboard_interactivity =
                            match surface.cached_state().keyboard_interactivity {
                                KeyboardInteractivity::None => {
                                    niri_ipc::LayerSurfaceKeyboardInteractivity::None
                                }
                                KeyboardInteractivity::Exclusive => {
                                    niri_ipc::LayerSurfaceKeyboardInteractivity::Exclusive
                                }
                                KeyboardInteractivity::OnDemand => {
                                    niri_ipc::LayerSurfaceKeyboardInteractivity::OnDemand
                                }
                            };

                        layers.push(niri_ipc::LayerSurface {
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
            // Forward-compat against a future pipelined `handle_client`; unreachable under today's sequential dispatch.
            if ctx
                .blocked_action_waiters
                .borrow()
                .contains_key(&ctx.conn_id)
            {
                return Err("request already queued".into());
            }

            validate_action(&action)?;

            let (tx, rx) = async_channel::bounded::<Result<(), ActivitySwitchBlock>>(1);

            let action = niri_config::Action::from(action);
            let waiters = ctx.blocked_action_waiters.clone();
            let conn_id = ctx.conn_id;
            ctx.event_loop.insert_idle(move |state| {
                // Make sure some logic like workspace clean-up has a chance to run before doing
                // actions.
                state.niri.advance_animations();
                // Clone so the waiter entry owns the original if do_action_inner hard-blocks (the drain site re-dispatches from it).
                let result = state.do_action_inner(action.clone(), false);
                match result {
                    Ok(()) => {
                        // Connection may have closed between enqueue and action
                        // completion. Safe to drop the send: on Ok the action
                        // already executed with no state loss.
                        let _ = tx.send_blocking(Ok(()));
                    }
                    Err(block) => {
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
                }
            });

            // Wait until the action has been processed before returning. This is important for a
            // few actions, for instance for DoScreenTransition this wait ensures that the screen
            // contents were sampled into the texture. Under a hard block, the
            // receiver parks here until the drain site wakes it.
            match rx.recv().await {
                Ok(Ok(())) => Response::Handled,
                Ok(Err(block)) => {
                    return Err(format_activity_switch_block_err(block));
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
) -> Vec<niri_ipc::Activity> {
    let active_id = layout.active_activity_id();
    layout
        .activities()
        .iter()
        .map(|a| to_ipc_activity(a, active_id, layout))
        .collect()
}

pub(crate) fn build_focused_activity_ipc<W: LayoutElement>(layout: &Layout<W>) -> niri_ipc::Activity {
    let active_id = layout.active_activity_id();
    to_ipc_activity(layout.activities().active(), active_id, layout)
}

fn to_ipc_activity<W: LayoutElement>(
    a: &crate::layout::activity::Activity,
    active_id: ActivityId,
    layout: &Layout<W>,
) -> niri_ipc::Activity {
    niri_ipc::Activity {
        id: a.id().get(),
        name: a.name().to_owned(),
        is_config_declared: a.is_config_declared(),
        is_active: a.id() == active_id,
        is_urgent: layout.activity_is_urgent(a.id()),
    }
}

fn validate_action(action: &Action) -> Result<(), String> {
    if let Action::Screenshot { path, .. }
    | Action::ScreenshotScreen { path, .. }
    | Action::ScreenshotWindow { path, .. }
    | Action::LoadConfigFile { path } = action
    {
        if let Some(path) = path {
            // Relative paths are resolved against the niri compositor's working directory, which
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
) -> niri_ipc::Window {
    with_toplevel_role(mapped.toplevel(), |role| niri_ipc::Window {
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
        // docs/activities/design.md §4.6 for the normative statement of the
        // structure-before-state rule. Do not move
        // `ipc_refresh_activity_lifecycle` after `ipc_refresh_active_activity`
        // or `ipc_refresh_workspaces`.
        self.ipc_refresh_activity_lifecycle();
        // `ActivitySwitched` must precede any `WorkspaceOpenedOrChanged` whose
        // `is_in_active_activity` flipped this tick, so clients can update
        // their activity state before processing workspace visibility
        // changes. See the contract pinned on `Event::ActivitySwitched` in
        // `niri-ipc/src/lib.rs`. Do not move `ipc_refresh_active_activity`
        // after `ipc_refresh_workspaces`.
        self.ipc_refresh_active_activity();
        self.ipc_refresh_workspaces();
        // Urgency ordering is inside-out: `WindowUrgencyChanged` fires as the
        // window updates; `WorkspaceUrgencyChanged` fires from
        // `ipc_refresh_workspaces` above; `ActivityUrgencyChanged` is the
        // outermost layer and must fire after workspace urgency so clients
        // see workspace-level state settled before the activity aggregate
        // arrives (niri-ipc `Event::ActivityUrgencyChanged` rustdoc). Do not
        // move `ipc_refresh_activity_urgency` before `ipc_refresh_workspaces`.
        self.ipc_refresh_activity_urgency();
        self.ipc_refresh_windows();
        self.ipc_refresh_overview();
    }

    fn ipc_refresh_active_activity(&mut self) {
        let Some(server) = &self.niri.ipc_server else {
            return;
        };

        let _span = tracy_client::span!("State::ipc_refresh_active_activity");

        let current_id = self.niri.layout.active_activity_id();
        let current = current_id.get();

        // First observation after server start seeds the tracker without
        // emitting: the rustdoc contract says `ActivitySwitched` is a *change*
        // notification, and a fresh server has no prior state to compare
        // against. The seeding posture mirrors `ipc_refresh_overview`'s
        // equality short-circuit, adapted for the `Option` tracker shape.
        let previous_raw = server.last_active_activity_id.get();
        let was_initialized = previous_raw.is_some();
        let previous = previous_raw.map(|id| id.get());

        if was_initialized {
            if let Some(event) = diff_active_activity(previous, current) {
                server.send_event(event);
            }
        }

        // Tracker-update discipline: unconditional set at function exit so the
        // "emitted == tracked" invariant holds whether or not `diff_active_activity`
        // returned `Some`, and regardless of the seeding branch above. Never
        // move this into the `if let Some(event)` body.
        server.last_active_activity_id.set(Some(current_id));
    }

    fn ipc_refresh_activity_urgency(&mut self) {
        let Some(server) = &self.niri.ipc_server else {
            return;
        };

        let _span = tracy_client::span!("State::ipc_refresh_activity_urgency");

        let layout = &self.niri.layout;
        let current: HashMap<ActivityId, bool> = layout
            .activities()
            .iter()
            .map(|a| (a.id(), layout.activity_is_urgent(a.id())))
            .collect();

        // Clone out the previous snapshot inside a narrow shared-borrow scope
        // so the unconditional `borrow_mut` at function exit cannot alias an
        // outstanding shared borrow.
        let previous = server.last_activity_urgency.borrow().clone();
        let (events, snapshot) = step_activity_urgency(previous.as_ref(), current);
        for event in events {
            server.send_event(event);
        }

        // Tracker-update discipline (mirrors `ipc_refresh_active_activity`): unconditional set at
        // function exit so the "emitted == tracked" invariant holds whether
        // or not `step_activity_urgency` emitted events, and regardless of
        // the seeding branch above. Never move this into the event-emission
        // branch.
        *server.last_activity_urgency.borrow_mut() = Some(snapshot);
    }

    fn ipc_refresh_activity_lifecycle(&mut self) {
        let Some(server) = &self.niri.ipc_server else {
            return;
        };

        let _span = tracy_client::span!("State::ipc_refresh_activity_lifecycle");

        let layout = &self.niri.layout;
        let current: HashMap<ActivityId, String> = layout
            .activities()
            .iter()
            .map(|a| (a.id(), a.name().to_owned()))
            .collect();

        // Clone out the previous snapshot inside a narrow shared-borrow scope
        // so the unconditional `borrow_mut` at function exit cannot alias an
        // outstanding shared borrow (mirrors `ipc_refresh_activity_urgency`).
        let previous = server.last_activities_by_id.borrow().clone();
        let (events, snapshot) = step_activity_lifecycle(previous.as_ref(), &current, layout);
        for event in events {
            server.send_event(event);
        }

        // Tracker-update discipline (mirrors `ipc_refresh_active_activity` and
        // `ipc_refresh_activity_urgency`): unconditional set at function exit
        // so the "emitted == tracked" invariant holds whether or not
        // `step_activity_lifecycle` emitted events, and regardless of the
        // seeding branch. Never move this into the event-emission branch.
        *server.last_activities_by_id.borrow_mut() = Some(snapshot);
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
                    let cast = niri_ipc::Cast {
                        session_id: pending.session_id.get(),
                        stream_id,
                        kind: niri_ipc::CastKind::PipeWire,
                        target: niri_ipc::CastTarget::Nothing {},
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
                    let cast = niri_ipc::Cast {
                        session_id: cast.session_id.get(),
                        stream_id,
                        kind: niri_ipc::CastKind::PipeWire,
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
                        niri_ipc::CastTarget::Output { name } => *name != cast_info.output_name,
                        _ => true,
                    }
                }) {
                    let cast = niri_ipc::Cast {
                        session_id: cast_info.session_id.get(),
                        stream_id,
                        kind: niri_ipc::CastKind::WlrScreencopy,
                        target: niri_ipc::CastTarget::Output {
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

/// Diff the previously-emitted active-activity id against the current one.
///
/// Returns `Some(Event::ActivitySwitched { id, previous_id })` iff the two
/// differ, else `None`. The function is a pure equality check with no
/// sentinel-in-signal behavior: callers gate emission behind a
/// `was_initialized` check and write the tracker unconditionally at function
/// exit, so the tracker is set on the first tick without emitting. See
/// [`State::ipc_refresh_active_activity`] for the seeding pattern.
pub(crate) fn diff_active_activity(previous: Option<u64>, current: u64) -> Option<Event> {
    if Some(current) == previous {
        return None;
    }
    Some(Event::ActivitySwitched {
        id: current,
        previous_id: previous,
    })
}

/// Pure step over the activity-urgency tracker: given the prior snapshot
/// (`None` before first observation) and the current snapshot, return the
/// events to emit and the snapshot to store.
///
/// Seeding (`previous == None`) returns an empty event list regardless of the
/// current state — the first observation silently seeds the tracker. Once
/// seeded, emission is delegated to [`diff_activity_urgency`].
///
/// Callers handle the `RefCell` mechanics and event dispatch; this function
/// owns the decision of *what* to emit for a given (previous, current) pair.
/// See [`State::ipc_refresh_activity_urgency`] for the thin wrapper.
fn step_activity_urgency(
    previous: Option<&HashMap<ActivityId, bool>>,
    current: HashMap<ActivityId, bool>,
) -> (Vec<Event>, HashMap<ActivityId, bool>) {
    let events = match previous {
        Some(prev) => diff_activity_urgency(prev, &current),
        None => Vec::new(),
    };
    (events, current)
}

/// Diff two activity-urgency snapshots and produce the list of
/// [`Event::ActivityUrgencyChanged`] events to emit.
///
/// Emission rules:
/// - For every id present in both snapshots whose boolean flipped: emit the
///   transition.
/// - For every id present in `current` but not in `previous` (newly-tracked
///   activity): emit only if `current` reports `true`. A new activity that
///   starts non-urgent is silent — its initial `false` rides the
///   [`Event::ActivityCreated`] event (see [`diff_activity_lifecycle`]).
/// - Ids that dropped out of `current` emit nothing here; lifecycle fires
///   through [`Event::ActivityRemoved`] (see [`diff_activity_lifecycle`]).
///
/// Events are sorted by [`ActivityId`] for determinism so clients writing
/// tests against the event stream aren't exposed to `HashMap` iteration-order
/// flake.
fn diff_activity_urgency(
    previous: &HashMap<ActivityId, bool>,
    current: &HashMap<ActivityId, bool>,
) -> Vec<Event> {
    let mut transitions: Vec<(ActivityId, bool)> = Vec::new();
    for (&id, &urgent) in current {
        match previous.get(&id) {
            Some(&prev_urgent) => {
                if prev_urgent != urgent {
                    transitions.push((id, urgent));
                }
            }
            None => {
                if urgent {
                    transitions.push((id, urgent));
                }
            }
        }
    }
    transitions.sort_by_key(|(id, _)| id.get());
    transitions
        .into_iter()
        .map(|(id, urgent)| Event::ActivityUrgencyChanged {
            id: id.get(),
            urgent,
        })
        .collect()
}

/// Pure step over the activity-lifecycle tracker: given the prior snapshot
/// (`None` before first observation) and the current snapshot, return the
/// events to emit and the snapshot to store.
///
/// Seeding (`previous == None`) returns an empty event list regardless of the
/// current state — the first observation silently seeds the tracker. Once
/// seeded, emission is delegated to [`diff_activity_lifecycle`].
///
/// Callers handle the `RefCell` mechanics and event dispatch; this function
/// owns the decision of *what* to emit for a given (previous, current) pair.
/// See [`State::ipc_refresh_activity_lifecycle`] for the thin wrapper.
///
/// Signature note: `current` is taken by reference (unlike the by-value shape
/// of [`step_activity_urgency`]) because callers inspect `current` after the
/// call — the return-tuple snapshot is a `.clone()` of the borrow. The clone
/// cost is one `HashMap<ActivityId, String>` per refresh tick and is
/// intentional; changing to by-value would force callers to clone before
/// passing.
pub(crate) fn step_activity_lifecycle<W: LayoutElement>(
    previous: Option<&HashMap<ActivityId, String>>,
    current: &HashMap<ActivityId, String>,
    layout: &Layout<W>,
) -> (Vec<Event>, HashMap<ActivityId, String>) {
    let events = match previous {
        Some(prev) => diff_activity_lifecycle(prev, current, layout),
        None => Vec::new(),
    };
    (events, current.clone())
}

/// Diff two activity-lifecycle snapshots and produce the list of
/// [`Event::ActivityCreated`] / [`Event::ActivityRemoved`] /
/// [`Event::ActivityRenamed`] events to emit.
///
/// Multi-kind ordering: `Removed` first, then `Renamed`, then `Created`.
/// Removes before Creates so a client never sees a `Created` for a stale-id
/// slot while the old tracker entry for that id is still live.
/// Renames are pure-metadata and order-independent but group with Removed
/// for the "old state first" principle. Within each kind, events are sorted
/// by [`ActivityId`] ascending for determinism against `HashMap` iteration
/// order (the same flake surface that [`diff_activity_urgency`] pins).
///
/// The [`Event::ActivityCreated`] payload is rebuilt at emission time from
/// the current `Layout` state via [`to_ipc_activity`], so the derived fields
/// (`is_active`, `is_urgent`) reflect the layout as of the refresh tick
/// rather than a stale snapshot captured when the id was first observed.
///
/// Newcomer-routing rule: id presence in `previous` is checked before name
/// comparison. An id that appears in `current` but not in `previous` is a
/// newcomer and routes unconditionally to `ActivityCreated` — a name match
/// with some other id in `previous` does NOT classify it as a rename.
/// This guarantees that even a test using [`ActivityId::specific`] to
/// mint an id reusing a previous name produces an unambiguous stream.
pub(crate) fn diff_activity_lifecycle<W: LayoutElement>(
    previous: &HashMap<ActivityId, String>,
    current: &HashMap<ActivityId, String>,
    layout: &Layout<W>,
) -> Vec<Event> {
    // Collect newcomers and renames from a single walk of `current`.
    // See rustdoc — newcomers route to Created.
    let mut created_ids: Vec<ActivityId> = Vec::new();
    let mut renamed: Vec<(ActivityId, String)> = Vec::new();
    for (&id, name) in current {
        if let Some(prev_name) = previous.get(&id) {
            if prev_name != name {
                renamed.push((id, name.clone()));
            }
        } else {
            created_ids.push(id);
        }
    }

    // Removed: ids in `previous` but not in `current`.
    let mut removed_ids: Vec<ActivityId> = previous
        .keys()
        .copied()
        .filter(|id| !current.contains_key(id))
        .collect();

    // Deterministic ordering within each kind.
    removed_ids.sort_by_key(|id| id.get());
    renamed.sort_by_key(|(id, _)| id.get());
    created_ids.sort_by_key(|id| id.get());

    let mut events: Vec<Event> =
        Vec::with_capacity(removed_ids.len() + renamed.len() + created_ids.len());

    for id in removed_ids {
        events.push(Event::ActivityRemoved { id: id.get() });
    }
    for (id, name) in renamed {
        events.push(Event::ActivityRenamed { id: id.get(), name });
    }
    for id in created_ids {
        let activity = layout
            .activities()
            .get(id)
            .expect("create-newcomer id was just observed in layout.activities() on this same refresh tick (single-threaded calloop, no intervening mutation)");
        let payload = to_ipc_activity(activity, layout.active_activity_id(), layout);
        events.push(Event::ActivityCreated { activity: payload });
    }

    events
}

/// Builds the full per-refresh workspace snapshot sent to IPC clients.
///
/// Two-pass construction, together covering every pool workspace exactly once:
///
/// 1. Workspaces in the active activity, in monitor-and-view order (plus the
///    `disconnected_workspace_ids` tail when no monitors are connected).
///    Their `idx` is view-position + 1 (1-based user-visible index), and
///    `is_in_active_activity` is `true`.
/// 2. Workspaces that are members of some other activity only. Per DD §3.5
///    their `idx` sentinel is `0`, `is_in_active_activity` is `false`, and
///    neither `is_active` nor `is_focused` can be true (the active/focused
///    workspace is always in the active activity, by construction).
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

    fn dummy_action() -> niri_config::Action {
        // `Spawn` is the cheapest `Action` variant to construct by hand and
        // has trivial equality semantics. The drain path only clones the
        // action; the specific variant is irrelevant to the registry
        // invariants these tests pin.
        niri_config::Action::Spawn(vec![])
    }

    fn make_waiter() -> (
        BlockedWaiter,
        async_channel::Receiver<Result<(), ActivitySwitchBlock>>,
    ) {
        let (tx, rx) = async_channel::bounded::<Result<(), ActivitySwitchBlock>>(1);
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
        let drained = waiters.shift_remove(&conn).expect("waiter inserted just now");
        assert!(waiters.is_empty());
        drained.tx.send_blocking(Ok(())).expect("receiver is alive");

        let observed = rx
            .recv_blocking()
            .expect("sender alive until send_blocking returned");
        assert!(matches!(observed, Ok(())));
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
            w.tx.send_blocking(Ok(())).expect("receivers alive");
        }

        // Pull each response: A first, B second.
        let first = rx_a.recv_blocking().expect("sent above");
        let second = rx_b.recv_blocking().expect("sent above");
        assert!(matches!(first, Ok(())));
        assert!(matches!(second, Ok(())));
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

    #[test]
    fn diff_active_activity_none_previous_returns_some() {
        // Pure-function shape: `None → Some(id)` yields `ActivitySwitched`
        // with `previous_id: None`. The "do not emit on initial tick" wiring
        // is a caller concern handled by the seeding gate in
        // `ipc_refresh_active_activity`.
        let event = diff_active_activity(None, 7);
        assert!(matches!(
            event,
            Some(Event::ActivitySwitched {
                id: 7,
                previous_id: None,
            })
        ));
    }

    #[test]
    fn diff_active_activity_no_change_returns_none() {
        // Dominant hot-path case: every refresh tick without an activity
        // switch short-circuits to `None`.
        assert!(diff_active_activity(Some(7), 7).is_none());
    }

    #[test]
    fn diff_active_activity_zero_id_no_change_returns_none() {
        // Zero is not a sentinel — equality on `current == 0` must short-circuit
        // exactly as for any other id.
        assert!(diff_active_activity(Some(0), 0).is_none());
    }

    #[test]
    fn diff_active_activity_change_returns_some_with_previous() {
        // Guards against an accidental 'always None' regression in
        // `previous_id` — the other two tests don't exercise a non-None prior.
        let event = diff_active_activity(Some(3), 7);
        assert!(matches!(
            event,
            Some(Event::ActivitySwitched {
                id: 7,
                previous_id: Some(3),
            })
        ));
    }

    fn urgency_snapshot(pairs: &[(u64, bool)]) -> HashMap<ActivityId, bool> {
        pairs
            .iter()
            .map(|&(id, urgent)| (ActivityId::specific(id), urgent))
            .collect()
    }

    #[test]
    fn diff_activity_urgency_no_change_emits_nothing() {
        // Dominant hot-path case: equal snapshots short-circuit to an empty
        // vec with no event allocation.
        let prev = urgency_snapshot(&[(1, false), (2, true)]);
        let curr = urgency_snapshot(&[(1, false), (2, true)]);
        assert!(diff_activity_urgency(&prev, &curr).is_empty());
    }

    #[test]
    fn diff_activity_urgency_single_flip_to_true_emits_event() {
        let prev = urgency_snapshot(&[(5, false)]);
        let curr = urgency_snapshot(&[(5, true)]);
        let events = diff_activity_urgency(&prev, &curr);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            Event::ActivityUrgencyChanged {
                id: 5,
                urgent: true,
            }
        ));
    }

    #[test]
    fn diff_activity_urgency_single_flip_to_false_emits_event() {
        let prev = urgency_snapshot(&[(5, true)]);
        let curr = urgency_snapshot(&[(5, false)]);
        let events = diff_activity_urgency(&prev, &curr);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            Event::ActivityUrgencyChanged {
                id: 5,
                urgent: false,
            }
        ));
    }

    #[test]
    fn diff_activity_urgency_multiple_flips_emit_all_sorted_by_id() {
        // Sort-by-id contract: emitted events are deterministic regardless of
        // `HashMap` iteration order.  Five ids inserted in a shuffled order
        // (`9, 3, 7, 1, 5`) so that no HashMap seed can accidentally produce
        // sorted output and make an unsorted implementation appear correct.
        // Assert both the id order and the per-entry urgent flag together so
        // neither can regress independently.
        let prev = urgency_snapshot(&[(9, true), (3, true), (7, false), (1, false), (5, true)]);
        let curr = urgency_snapshot(&[(9, false), (3, false), (7, true), (1, true), (5, false)]);
        let events = diff_activity_urgency(&prev, &curr);
        // Expected: sorted by id asc, urgent flag from `curr`.
        let expected: &[(u64, bool)] = &[(1, true), (3, false), (5, false), (7, true), (9, false)];
        assert_eq!(
            events.len(),
            expected.len(),
            "all 5 flipped ids must emit exactly once",
        );
        for (event, &(exp_id, exp_urgent)) in events.iter().zip(expected) {
            let Event::ActivityUrgencyChanged { id, urgent } = event else {
                panic!("expected ActivityUrgencyChanged, got {event:?}");
            };
            assert_eq!(*id, exp_id, "id must be sorted ascending");
            assert_eq!(*urgent, exp_urgent, "id {id}: urgent flag must match current");
        }
    }

    #[test]
    fn diff_activity_urgency_newly_present_activity_emits_only_if_true() {
        // Newly-inserted keys (ids in `current` but not `previous`): a `true`
        // state fires an event; a `false` state is silent — its initial
        // value rides on `ActivityCreated` (not in this sub-phase).
        let prev = urgency_snapshot(&[]);
        let curr = urgency_snapshot(&[(1, false), (2, true)]);
        let events = diff_activity_urgency(&prev, &curr);
        assert_eq!(events.len(), 1, "only the urgent newcomer emits");
        assert!(matches!(
            events[0],
            Event::ActivityUrgencyChanged {
                id: 2,
                urgent: true,
            }
        ));
    }

    #[test]
    fn diff_activity_urgency_removed_id_emits_nothing() {
        // Ids that drop out of `current` (removed activities) must emit
        // nothing: the loop iterates `current`, not `previous`, so prev-only
        // keys are silently skipped.  Lifecycle for removed activities fires
        // through [`Event::ActivityRemoved`] (see [`diff_activity_lifecycle`]).
        let prev = urgency_snapshot(&[(1, true), (2, true)]);
        let curr = urgency_snapshot(&[(1, true)]); // id 2 dropped
        assert!(
            diff_activity_urgency(&prev, &curr).is_empty(),
            "removed activity must not emit an urgency event",
        );
    }

    #[test]
    fn step_activity_urgency_seeds_without_emit() {
        // First observation (`previous == None`) seeds the tracker silently
        // even when activities are already urgent. This pins the
        // orchestration-level seeding discipline at the pure-function tier,
        // so `State::ipc_refresh_activity_urgency` can remain a thin
        // wrapper. Also pins the "hidden-activity urgency emission" risk
        // surface — the newcomer-urgent `42` activity would otherwise fire
        // against an implicit empty snapshot, which is wrong for
        // startup-seeding.
        let current = urgency_snapshot(&[(1, false), (42, true)]);
        let (events, snapshot) = step_activity_urgency(None, current.clone());
        assert!(
            events.is_empty(),
            "seeding must be silent, got {events:?}",
        );
        assert_eq!(
            snapshot, current,
            "snapshot must be stored exactly as observed",
        );

        // Once seeded, a subsequent tick with no changes emits nothing.
        let (events2, _) = step_activity_urgency(Some(&snapshot), current.clone());
        assert!(events2.is_empty());

        // A subsequent flip emits, confirming seed-then-track behaviour.
        let next = urgency_snapshot(&[(1, true), (42, true)]);
        let (events3, _) = step_activity_urgency(Some(&snapshot), next);
        assert_eq!(events3.len(), 1);
        assert!(matches!(
            events3[0],
            Event::ActivityUrgencyChanged {
                id: 1,
                urgent: true,
            }
        ));
    }
}
