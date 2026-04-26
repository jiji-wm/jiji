use std::os::fd::AsFd as _;
use std::os::unix::net::UnixStream;
use std::sync::atomic::Ordering;
use std::time::Duration;

use calloop::generic::Generic;
use calloop::{EventLoop, Interest, LoopHandle, Mode, PostAction};
use niri_config::Config;
use smithay::output::Output;

use super::client::{Client, ClientId};
use super::server::Server;
use crate::niri::{NewClient, Niri};

pub struct Fixture {
    pub event_loop: EventLoop<'static, State>,
    pub handle: LoopHandle<'static, State>,
    pub state: State,
}

pub struct State {
    pub server: Server,
    pub clients: Vec<Client>,
}

impl Fixture {
    pub fn new() -> Self {
        Self::with_config(Config::default())
    }

    pub fn with_config(config: Config) -> Self {
        let event_loop = EventLoop::try_new().unwrap();
        let handle = event_loop.handle();

        let server = Server::new(config);
        let fd = server.event_loop.as_fd().try_clone_to_owned().unwrap();
        let source = Generic::new(fd, Interest::READ, Mode::Level);
        handle
            .insert_source(source, |_, _, state: &mut State| {
                state.server.dispatch();
                Ok(PostAction::Continue)
            })
            .unwrap();

        let state = State {
            server,
            clients: Vec::new(),
        };

        Self {
            event_loop,
            handle,
            state,
        }
    }

    pub fn dispatch(&mut self) {
        self.event_loop
            .dispatch(Duration::ZERO, &mut self.state)
            .unwrap();
    }

    pub fn niri_state(&mut self) -> &mut crate::niri::State {
        &mut self.state.server.state
    }

    pub fn niri(&mut self) -> &mut Niri {
        &mut self.niri_state().niri
    }

    pub fn niri_output(&self, n: u8) -> Output {
        let niri = &self.state.server.state.niri;
        let idx = usize::from(n - 1);
        let output = niri.global_space.outputs().nth(idx).unwrap();
        output.clone()
    }

    pub fn niri_focus_output(&mut self, n: u8) {
        let niri = &mut self.state.server.state.niri;
        let idx = usize::from(n - 1);
        let output = niri.global_space.outputs().nth(idx).unwrap();
        niri.layout.focus_output(output);
    }

    pub fn niri_complete_animations(&mut self) {
        let niri = self.niri();
        niri.clock.set_complete_instantly(true);
        niri.advance_animations();
        niri.clock.set_complete_instantly(false);
    }

    pub fn add_output(&mut self, n: u8, size: (u16, u16)) {
        let state = self.niri_state();
        let niri = &mut state.niri;
        state.backend.headless().add_output(niri, n, size);
    }

    pub fn add_client(&mut self) -> ClientId {
        let (sock1, sock2) = UnixStream::pair().unwrap();
        self.niri().insert_client(NewClient {
            client: sock1,
            restricted: false,
            credentials_unknown: false,
        });

        let client = Client::new(sock2);
        let id = client.id;

        let fd = client.event_loop.as_fd().try_clone_to_owned().unwrap();
        let source = Generic::new(fd, Interest::READ, Mode::Level);
        self.handle
            .insert_source(source, move |_, _, state: &mut State| {
                state.client(id).dispatch();
                Ok(PostAction::Continue)
            })
            .unwrap();

        self.state.clients.push(client);
        self.roundtrip(id);
        id
    }

    pub fn client(&mut self, id: ClientId) -> &mut Client {
        self.state.client(id)
    }

    pub fn roundtrip(&mut self, id: ClientId) {
        let client = self.state.client(id);
        let data = client.send_sync();
        while !data.done.load(Ordering::Relaxed) {
            self.dispatch();
        }
    }

    /// Roundtrip twice in a row.
    ///
    /// For some reason, when running tests on many threads at once, a single roundtrip is
    /// sometimes not sufficient to get the configure events to the client.
    ///
    /// I suspect that this is because these configure events are sent from the niri loop callback,
    /// so they arrive after the sync done event and don't get processed in that client dispatch
    /// cycle. I'm not sure why this would be dependent on multithreading. But if this is indeed
    /// the issue, then a double roundtrip fixes it.
    pub fn double_roundtrip(&mut self, id: ClientId) {
        self.roundtrip(id);
        self.roundtrip(id);
    }

    /// Install the in-memory event-stream tap on the fixture's `IpcServer`.
    /// Each call replaces the prior buffer with a fresh empty `Vec`, so
    /// successive flows in a single test cannot leak captured events into
    /// each other.
    ///
    /// Backed by an unbounded `Vec` (not the bounded `async_channel` used by
    /// real event-stream clients), so the tap is immune to the `try_send`
    /// `Full`-eviction path that would drop events on a real client. Use
    /// after any one-time seed `refresh_and_flush_clients()` so the captured
    /// stream is exactly the deltas produced by the test's flow.
    pub fn install_event_tap(&mut self) {
        self.niri()
            .ipc_server
            .as_ref()
            .expect(
                "IpcServer present in test fixture — \
                 Server::new constructs IpcServer for headless State",
            )
            .install_test_event_tap();
    }

    /// Drain the events captured since the most recent `install_event_tap`
    /// (or since the last `drain_events`). Order is emission order.
    pub fn drain_events(&mut self) -> Vec<niri_ipc::Event> {
        self.niri()
            .ipc_server
            .as_ref()
            .expect(
                "IpcServer present in test fixture — \
                 Server::new constructs IpcServer for headless State",
            )
            .drain_test_events()
    }

    /// Drive `State::reload_config(Ok(config))` end-to-end: validate phase of
    /// `reconcile_activities_on_reload_remove`, the unname-prewalk, and
    /// `reconcile_activities_on_reload_add`. Mirrors how the niri loop
    /// dispatches a reload after a config-file change.
    pub fn reload_config(&mut self, config: Config) {
        self.niri_state().reload_config(Ok(config));
    }

    /// Snapshot the server's current event-stream state as a replay burst
    /// (the same sequence a freshly-connected client would receive before
    /// any flow deltas). Forwards to `IpcServer::replicate_event_stream_state`.
    pub fn replicate_event_stream_state(&mut self) -> Vec<niri_ipc::Event> {
        self.niri()
            .ipc_server
            .as_ref()
            .expect(
                "IpcServer present in test fixture — \
                 Server::new constructs IpcServer for headless State",
            )
            .replicate_event_stream_state()
    }
}

impl State {
    pub fn client(&mut self, id: ClientId) -> &mut Client {
        self.clients.iter_mut().find(|c| c.id == id).unwrap()
    }
}

/// Build a [`Config`] with two top-level activities (`"alpha"`, `"beta"`) and
/// one `workspace` per entry in the two slices, each tagged with its matching
/// activity. `"alpha"` is first-declared, so it is the default / seed activity
/// the `Layout` starts on.
///
/// Used by the `ext_workspace` integration tests to exercise
/// activity-filtered projection through `ext_workspace::refresh`. The helper
/// goes through `Config::parse_mem` (rather than field-by-field construction)
/// to match every other `Config`-building test site and to stay robust against
/// future additions to `Config`.
#[cfg(test)]
pub(super) fn config_with_two_activities(
    alpha_workspaces: &[&str],
    beta_workspaces: &[&str],
) -> Config {
    use std::fmt::Write as _;

    let mut src = String::new();
    src.push_str("activity \"alpha\"\n");
    src.push_str("activity \"beta\"\n");
    for ws in alpha_workspaces {
        writeln!(src, "workspace \"{ws}\" {{ activity \"alpha\"; }}").unwrap();
    }
    for ws in beta_workspaces {
        writeln!(src, "workspace \"{ws}\" {{ activity \"beta\"; }}").unwrap();
    }
    Config::parse_mem(&src).expect("parse_mem must succeed on the generated KDL template")
}
