use std::collections::{BTreeMap, HashMap};
use std::io::ErrorKind;
use std::iter::Peekable;
use std::path::Path;
use std::{env, slice};

use anyhow::{anyhow, bail, Context};
use jiji_config::OutputName;
use jiji_ipc::socket::Socket;
use jiji_ipc::{
    Action, Activity, Bookmark, Cast, CastKind, CastTarget, Event, KeyboardLayouts, LogicalOutput,
    Mode, Output, OutputConfigChanged, Overview, Request, Response, Transform, Window,
    WindowLayout, Workspace,
};
use serde_json::json;

use crate::cli::Msg;
use crate::utils::version;

/// Render the activity annotation for a workspace row.
///
/// Returns ` [activity "X"]` (single) or ` [activities "X", "Y"]` (multi),
/// with the leading space included so the caller can append it directly.
///
/// Activity labels are looked up in `names` (id → name); ids missing from
/// the map fall back to bare numerics (`[activity 99]`). The label list is
/// sorted ascending for determinism — the wire-side `activities` Vec is
/// id-sorted but the displayed list is name-sorted so the human-readable
/// output is stable across rename events.
fn format_annotation(ws: &Workspace, names: &HashMap<u64, String>) -> String {
    debug_assert!(
        !ws.activities.is_empty(),
        "workspace activities must be non-empty per jiji-ipc Workspace contract"
    );

    let mut labels: Vec<String> = ws
        .activities
        .iter()
        .map(|aid| match names.get(aid) {
            Some(name) => format!("\"{name}\""),
            None => format!("{aid}"),
        })
        .collect();
    labels.sort();

    let keyword = if ws.activities.len() == 1 {
        "activity"
    } else {
        "activities"
    };
    format!(" [{keyword} {}]", labels.join(", "))
}

/// Render a single workspace row in the activity-aware human-readable shape.
///
/// Column layout:
/// `<is_active 3ch><idx-or-dash 2ch right>  <id=N 6ch left>  <name 10ch left><annotation>`
///
/// Hidden workspaces (`!is_in_active_activity`) render `-` in the idx column
/// rather than the sentinel `0` carried in `ws.idx` — the `idx` field is only
/// meaningful when `is_in_active_activity` is true (see the `Workspace.idx`
/// contract in `jiji-ipc`).
///
/// Names wider than 10 characters push the annotation rightward without truncation —
/// the column width is a left-pad minimum, not a cap.
fn format_row(ws: &Workspace, names: &HashMap<u64, String>) -> String {
    let is_active = if ws.is_active { " * " } else { "   " };
    let idx_or_dash = if ws.is_in_active_activity {
        format!("{}", ws.idx)
    } else {
        "-".to_owned()
    };
    let id_col = format!("id={}", ws.id);
    let name_col = match ws.name.as_deref() {
        Some(n) => format!("\"{n}\""),
        None => String::new(),
    };
    let annotation = format_annotation(ws, names);

    format!("{is_active}{idx_or_dash:>2}  {id_col:<6}  {name_col:<10}{annotation}")
}

pub fn handle_msg(mut msg: Msg, json: bool) -> anyhow::Result<()> {
    // For actions taking paths, prepend the jiji CLI's working directory.
    if let Msg::Action {
        action:
            Action::Screenshot { path, .. }
            | Action::ScreenshotScreen { path, .. }
            | Action::ScreenshotWindow { path, .. },
    } = &mut msg
    {
        if let Some(path) = path {
            ensure_absolute_path(path).context("error making the path absolute")?;
        }
    }

    let request = match &msg {
        Msg::Version => Request::Version,
        Msg::Outputs => Request::Outputs,
        Msg::FocusedWindow => Request::FocusedWindow,
        Msg::FocusedOutput => Request::FocusedOutput,
        Msg::Activities => Request::Activities,
        Msg::ActivityViews => Request::ActivityViews,
        Msg::FocusedActivity => Request::FocusedActivity,
        Msg::Bookmarks => Request::Bookmarks,
        Msg::PickWindow => Request::PickWindow,
        Msg::PickColor => Request::PickColor,
        Msg::Action { action } => Request::Action(action.clone()),
        Msg::Output { output, action } => Request::Output {
            output: output.clone(),
            action: action.clone(),
        },
        Msg::Workspaces => Request::Workspaces,
        Msg::Windows => Request::Windows,
        Msg::Layers => Request::Layers,
        Msg::KeyboardLayouts => Request::KeyboardLayouts,
        Msg::EventStream => Request::EventStream,
        Msg::RequestError => Request::ReturnError,
        Msg::OverviewState => Request::OverviewState,
        Msg::Casts => Request::Casts,
    };

    let mut socket = Socket::connect().context("error connecting to the jiji socket")?;

    let result = socket.send(request);

    // For errors that can be caused by a version mismatch between the running jiji instance and
    // the jiji msg CLI, we will try to fetch and compare the versions.
    let check_compositor_version = match &result {
        Err(err) => {
            // Response JSON parsing errors.
            matches!(
                err.kind(),
                ErrorKind::InvalidData | ErrorKind::UnexpectedEof
            )
        }
        // Error returned from jiji.
        Ok(Err(_)) => true,
        _ => false,
    };

    let compositor_version = if check_compositor_version && !matches!(msg, Msg::Version) {
        // Reconnect to support older jiji versions with one request per connection.
        Socket::connect()
            .and_then(|mut socket| socket.send(Request::Version))
            .ok()
    } else {
        None
    };

    // Default SIGPIPE so that our prints don't panic on stdout closing.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }

    // Check for CLI-server version mismatch to add helpful context.
    match compositor_version {
        Some(Ok(Response::Version(compositor_version))) => {
            let cli_version = version();
            if cli_version != compositor_version {
                eprintln!("Running jiji compositor has a different version from the jiji CLI:");
                eprintln!("Compositor version: {compositor_version}");
                eprintln!("CLI version:        {cli_version}");
                eprintln!("Did you forget to restart jiji after an update?");
                eprintln!();
            }
        }
        Some(_) => {
            eprintln!("Unable to get the running jiji compositor version.");
            eprintln!("Did you forget to restart jiji after an update?");
            eprintln!();
        }
        None => {
            // Communication error, or the original request was already a version request, or the
            // original request had succeeded. Don't add irrelevant context.
        }
    }

    let reply = result.context("error communicating with jiji")?;
    let response = reply.map_err(|err_msg| anyhow!(err_msg).context("jiji returned an error"))?;

    match msg {
        Msg::RequestError => {
            bail!("unexpected response: expected an error, got {response:?}");
        }
        Msg::Version => {
            let Response::Version(compositor_version) = response else {
                bail!("unexpected response: expected Version, got {response:?}");
            };

            let cli_version = version();

            if json {
                println!(
                    "{}",
                    json!({
                        "compositor": compositor_version,
                        "cli": cli_version,
                    })
                );
                return Ok(());
            }

            if cli_version != compositor_version {
                eprintln!("Running jiji compositor has a different version from the jiji CLI.");
                eprintln!("Did you forget to restart jiji after an update?");
                eprintln!();
            }

            println!("Compositor version: {compositor_version}");
            println!("CLI version:        {cli_version}");
        }
        Msg::Outputs => {
            let Response::Outputs(outputs) = response else {
                bail!("unexpected response: expected Outputs, got {response:?}");
            };

            if json {
                let output =
                    serde_json::to_string(&outputs).context("error formatting response")?;
                println!("{output}");
                return Ok(());
            }

            let mut outputs = outputs
                .into_values()
                .map(|out| (OutputName::from_ipc_output(&out), out))
                .collect::<Vec<_>>();
            outputs.sort_unstable_by(|a, b| a.0.compare(&b.0));

            for (_name, output) in outputs.into_iter() {
                print_output(output)?;
                println!();
            }
        }
        Msg::FocusedWindow => {
            let Response::FocusedWindow(window) = response else {
                bail!("unexpected response: expected FocusedWindow, got {response:?}");
            };

            if json {
                let window = serde_json::to_string(&window).context("error formatting response")?;
                println!("{window}");
                return Ok(());
            }

            if let Some(window) = window {
                print_window(&window);
            } else {
                println!("No window is focused.");
            }
        }
        Msg::Windows => {
            let Response::Windows(mut windows) = response else {
                bail!("unexpected response: expected Windows, got {response:?}");
            };

            if json {
                let windows =
                    serde_json::to_string(&windows).context("error formatting response")?;
                println!("{windows}");
                return Ok(());
            }

            windows.sort_unstable_by_key(|a| a.id);

            for window in windows {
                print_window(&window);
                println!();
            }
        }
        Msg::Layers => {
            let Response::Layers(mut layers) = response else {
                bail!("unexpected response: expected Layers, got {response:?}");
            };

            if json {
                let layers = serde_json::to_string(&layers).context("error formatting response")?;
                println!("{layers}");
                return Ok(());
            }

            layers.sort_by(|a, b| {
                Ord::cmp(&a.output, &b.output)
                    .then_with(|| Ord::cmp(&a.layer, &b.layer))
                    .then_with(|| Ord::cmp(&a.namespace, &b.namespace))
            });
            let mut iter = layers.iter().peekable();

            let print = |surface: &jiji_ipc::LayerSurface| {
                println!("    Surface:");
                println!("      Namespace: \"{}\"", &surface.namespace);

                let interactivity = match surface.keyboard_interactivity {
                    jiji_ipc::LayerSurfaceKeyboardInteractivity::None => "none",
                    jiji_ipc::LayerSurfaceKeyboardInteractivity::Exclusive => "exclusive",
                    jiji_ipc::LayerSurfaceKeyboardInteractivity::OnDemand => "on-demand",
                };
                println!("      Keyboard interactivity: {interactivity}");
            };

            let print_layer = |iter: &mut Peekable<slice::Iter<jiji_ipc::LayerSurface>>,
                               output: &str,
                               layer| {
                let mut empty = true;
                while let Some(surface) = iter.next_if(|s| s.output == output && s.layer == layer) {
                    empty = false;
                    println!();
                    print(surface);
                }
                if empty {
                    println!(" (empty)\n");
                } else {
                    println!();
                }
            };

            while let Some(surface) = iter.peek() {
                let output = &surface.output;
                println!("Output \"{output}\":");

                print!("  Background layer:");
                print_layer(&mut iter, output, jiji_ipc::Layer::Background);

                print!("  Bottom layer:");
                print_layer(&mut iter, output, jiji_ipc::Layer::Bottom);

                print!("  Top layer:");
                print_layer(&mut iter, output, jiji_ipc::Layer::Top);

                print!("  Overlay layer:");
                print_layer(&mut iter, output, jiji_ipc::Layer::Overlay);
            }
        }
        Msg::FocusedOutput => {
            let Response::FocusedOutput(output) = response else {
                bail!("unexpected response: expected FocusedOutput, got {response:?}");
            };

            if json {
                let output = serde_json::to_string(&output).context("error formatting response")?;
                println!("{output}");
                return Ok(());
            }

            if let Some(output) = output {
                print_output(output)?;
            } else {
                println!("No output is focused.");
            }
        }
        Msg::Activities => {
            let Response::Activities(response) = response else {
                bail!("unexpected response: expected Activities, got {response:?}");
            };

            if json {
                let s = serde_json::to_string(&response).context("error formatting response")?;
                println!("{s}");
                return Ok(());
            }

            if response.is_empty() {
                println!("No activities.");
                return Ok(());
            }

            print_activities(&response);
        }
        Msg::ActivityViews => {
            let Response::ActivityViews(response) = response else {
                bail!("unexpected response: expected ActivityViews, got {response:?}");
            };

            if json {
                let s = serde_json::to_string(&response).context("error formatting response")?;
                println!("{s}");
                return Ok(());
            }

            if response.is_empty() {
                println!("No activity views.");
                return Ok(());
            }

            for view in &response {
                let output_name = view.output_name.as_deref().unwrap_or("(disconnected)");
                println!(
                    "activity {} on {} [output_id={}]: ws={:?} active_idx={}",
                    view.activity_id,
                    output_name,
                    view.output_id,
                    view.workspace_ids,
                    view.active_idx,
                );
            }
        }
        Msg::FocusedActivity => {
            let Response::FocusedActivity(activity) = response else {
                bail!("unexpected response: expected FocusedActivity, got {response:?}");
            };

            if json {
                let s = serde_json::to_string(&activity).context("error formatting response")?;
                println!("{s}");
                return Ok(());
            }

            print_activity(&activity);
        }
        Msg::Bookmarks => {
            let Response::Bookmarks(bookmarks) = response else {
                bail!("unexpected response: expected Bookmarks, got {response:?}");
            };

            if json {
                let s = serde_json::to_string(&bookmarks).context("error formatting response")?;
                println!("{s}");
                return Ok(());
            }

            print_bookmarks(&bookmarks);
        }
        Msg::PickWindow => {
            let Response::PickedWindow(window) = response else {
                bail!("unexpected response: expected PickedWindow, got {response:?}");
            };

            if json {
                let window = serde_json::to_string(&window).context("error formatting response")?;
                println!("{window}");
                return Ok(());
            }

            if let Some(window) = window {
                print_window(&window);
            } else {
                println!("No window selected.");
            }
        }
        Msg::PickColor => {
            let Response::PickedColor(color) = response else {
                bail!("unexpected response: expected PickedColor, got {response:?}");
            };

            if json {
                let color = serde_json::to_string(&color).context("error formatting response")?;
                println!("{color}");
                return Ok(());
            }

            if let Some(color) = color {
                let [r, g, b] = color.rgb.map(|v| (v.clamp(0., 1.) * 255.).round() as u8);

                println!("Picked color: rgb({r}, {g}, {b})",);
                println!("Hex: #{r:02x}{g:02x}{b:02x}");
            } else {
                println!("No color was picked.");
            }
        }
        Msg::Action { .. } => {
            let Response::Handled = response else {
                bail!("unexpected response: expected Handled, got {response:?}");
            };
        }
        Msg::Output { output, .. } => {
            let Response::OutputConfigChanged(response) = response else {
                bail!("unexpected response: expected OutputConfigChanged, got {response:?}");
            };

            if json {
                let response =
                    serde_json::to_string(&response).context("error formatting response")?;
                println!("{response}");
                return Ok(());
            }

            if response == OutputConfigChanged::OutputWasMissing {
                println!("Output \"{output}\" is not connected.");
                println!("The change will apply when it is connected.");
            }
        }
        Msg::Workspaces => {
            let Response::Workspaces(response) = response else {
                bail!("unexpected response: expected Workspaces, got {response:?}");
            };

            if json {
                let response =
                    serde_json::to_string(&response).context("error formatting response")?;
                println!("{response}");
                return Ok(());
            }

            if response.is_empty() {
                println!("No workspaces.");
                return Ok(());
            }

            // Chain a second request to resolve activity ids → names for the
            // row annotations. The chained request opens a fresh connection
            // rather than reusing `socket`, so older jiji versions that close
            // the socket after a single request degrade cleanly to the
            // empty-map fallback instead of surfacing an IO error from a
            // closed-after-Workspaces socket. The version-fetch path earlier
            // in this function follows the same one-request-per-connection
            // discipline for the same reason. Each failure arm emits a
            // distinct diagnostic and falls through with an empty map (three
            // separate arms rather than a single `_ =>` catch-all, so the
            // failure mode is debuggable from stderr alone): annotations
            // degrade to the bare numeric form `[activity 7]` and the
            // active-activity header is omitted. The snapshot may also
            // straddle an activity-switch tick — the printer renders
            // whichever activity was current at the chained request, which
            // is best-effort by construction.
            let (names, active_name): (HashMap<u64, String>, Option<String>) =
                match Socket::connect().and_then(|mut s| s.send(Request::Activities)) {
                    Ok(Ok(Response::Activities(acts))) => {
                        let actives: Vec<_> = acts.iter().filter(|a| a.is_active).collect();
                        debug_assert!(
                            actives.len() <= 1,
                            "jiji-ipc Activities reported >1 active activity: {actives:?}"
                        );
                        let active_name = actives.first().map(|a| a.name.clone());
                        let names = acts.into_iter().map(|a| (a.id, a.name)).collect();
                        (names, active_name)
                    }
                    Ok(Ok(other)) => {
                        eprintln!(
                            "jiji msg workspaces: unexpected response to Activities request \
                             ({other:?}); rendering bare numeric annotations"
                        );
                        (HashMap::new(), None)
                    }
                    Ok(Err(reply_err)) => {
                        eprintln!(
                            "jiji msg workspaces: compositor rejected Activities request \
                             ({reply_err}); this jiji version may not support activities"
                        );
                        (HashMap::new(), None)
                    }
                    Err(io_err) => {
                        eprintln!(
                            "jiji msg workspaces: IO error on chained Activities request \
                             (failed to connect or send: {io_err}); rendering bare numeric \
                             annotations"
                        );
                        (HashMap::new(), None)
                    }
                };

            if let Some(name) = &active_name {
                println!("Active activity: \"{name}\"");
                println!();
            }

            // Partition by output; rows with no output go into a separate
            // Disconnected bucket printed last. BTreeMap gives stable
            // ascending output-name ordering for the printed sections.
            let mut by_output: BTreeMap<&str, Vec<&Workspace>> = BTreeMap::new();
            let mut disconnected: Vec<&Workspace> = Vec::new();
            for ws in &response {
                match ws.output.as_deref() {
                    Some(name) => by_output.entry(name).or_default().push(ws),
                    None => disconnected.push(ws),
                }
            }

            // Within an output: visible rows first (is_in_active_activity desc),
            // then by idx asc, then id asc. The idx sort is only meaningful
            // for the visible block — hidden rows all have `idx == 0` so they
            // fall through to the id tiebreaker.
            for bucket in by_output.values_mut() {
                bucket.sort_by(|a, b| {
                    b.is_in_active_activity
                        .cmp(&a.is_in_active_activity)
                        .then(a.idx.cmp(&b.idx))
                        .then(a.id.cmp(&b.id))
                });
            }
            disconnected.sort_by_key(|w| w.id);

            let mut first_section = true;
            for (output_name, workspaces) in &by_output {
                if !first_section {
                    println!();
                }
                first_section = false;
                println!("Output \"{output_name}\":");
                let mut prev_in_active: Option<bool> = None;
                for ws in workspaces {
                    if let Some(prev) = prev_in_active {
                        if prev && !ws.is_in_active_activity {
                            println!();
                        }
                    }
                    println!("{}", format_row(ws, &names));
                    prev_in_active = Some(ws.is_in_active_activity);
                }
            }

            // Skip the Disconnected header entirely when the bucket is empty —
            // a bare header with nothing under it is misleading noise.
            if !disconnected.is_empty() {
                if !first_section {
                    println!();
                }
                println!("Disconnected:");
                for ws in &disconnected {
                    println!("{}", format_row(ws, &names));
                }
            }
        }
        Msg::KeyboardLayouts => {
            let Response::KeyboardLayouts(response) = response else {
                bail!("unexpected response: expected KeyboardLayouts, got {response:?}");
            };

            if json {
                let response =
                    serde_json::to_string(&response).context("error formatting response")?;
                println!("{response}");
                return Ok(());
            }

            let KeyboardLayouts { names, current_idx } = response;
            let current_idx = usize::from(current_idx);

            println!("Keyboard layouts:");
            for (idx, name) in names.iter().enumerate() {
                let is_active = if idx == current_idx { " * " } else { "   " };
                println!("{is_active}{idx} {name}");
            }
        }
        Msg::EventStream => {
            let Response::Handled = response else {
                bail!("unexpected response: expected Handled, got {response:?}");
            };

            if !json {
                println!("Started reading events.");
            }

            let mut read_event = socket.read_events();
            loop {
                let event = read_event().context("error reading event from jiji")?;

                if json {
                    let event = serde_json::to_string(&event).context("error formatting event")?;
                    println!("{event}");
                    continue;
                }

                match event {
                    Event::WorkspacesChanged { workspaces } => {
                        println!("Workspaces changed: {workspaces:?}");
                    }
                    Event::WorkspaceUrgencyChanged { id, urgent } => {
                        println!("Workspace {id}: urgency changed to {urgent}");
                    }
                    Event::WorkspaceActivated { id, focused } => {
                        let word = if focused { "focused" } else { "activated" };
                        println!("Workspace {word}: {id}");
                    }
                    Event::WorkspaceActiveWindowChanged {
                        workspace_id,
                        active_window_id,
                    } => {
                        println!(
                            "Workspace {workspace_id}: \
                             active window changed to {active_window_id:?}"
                        );
                    }
                    Event::WindowsChanged { windows } => {
                        println!("Windows changed: {windows:?}");
                    }
                    Event::WindowOpenedOrChanged { window } => {
                        println!("Window opened or changed: {window:?}");
                    }
                    Event::WindowClosed { id } => {
                        println!("Window closed: {id}");
                    }
                    Event::WindowFocusChanged { id } => {
                        println!("Window focus changed: {id:?}");
                    }
                    Event::WindowFocusTimestampChanged {
                        id,
                        focus_timestamp,
                    } => {
                        println!("Window {id}: focus timestamp changed to {focus_timestamp:?}");
                    }
                    Event::WindowUrgencyChanged { id, urgent } => {
                        println!("Window {id}: urgency changed to {urgent}");
                    }
                    Event::WindowLayoutsChanged { changes } => {
                        println!("Window layouts changed: {changes:?}");
                    }
                    Event::KeyboardLayoutsChanged { keyboard_layouts } => {
                        println!("Keyboard layouts changed: {keyboard_layouts:?}");
                    }
                    Event::KeyboardLayoutSwitched { idx } => {
                        println!("Keyboard layout switched: {idx}");
                    }
                    Event::OverviewOpenedOrClosed { is_open: opened } => {
                        println!("Overview toggled: {opened}");
                    }
                    Event::ConfigLoaded { failed } => {
                        let status = if failed {
                            "with an error"
                        } else {
                            "successfully"
                        };
                        println!("Config loaded {status}");
                    }
                    Event::ScreenshotCaptured { path } => {
                        let mut parts = vec![];
                        parts.push("copied to clipboard".to_string());
                        if let Some(path) = &path {
                            parts.push(format!("saved to {path}"));
                        }
                        let description = parts.join(" and ");
                        println!("Screenshot captured: {description}");
                    }
                    Event::CastsChanged { casts } => {
                        println!("Casts changed: {casts:?}");
                    }
                    Event::CastStartedOrChanged { cast } => {
                        println!("Cast started or changed: {cast:?}");
                    }
                    Event::CastStopped { stream_id } => {
                        println!("Cast stopped: stream id {stream_id}");
                    }
                    _ => println!("Unknown event: {event:?}"),
                }
            }
        }
        Msg::OverviewState => {
            let Response::OverviewState(response) = response else {
                bail!("unexpected response: expected Overview, got {response:?}");
            };

            if json {
                let response =
                    serde_json::to_string(&response).context("error formatting response")?;
                println!("{response}");
                return Ok(());
            }

            let Overview { is_open } = response;
            if is_open {
                println!("Overview is open.");
            } else {
                println!("Overview is closed.");
            }
        }
        Msg::Casts => {
            let Response::Casts(mut casts) = response else {
                bail!("unexpected response: expected Casts, got {response:?}");
            };

            if json {
                let casts = serde_json::to_string(&casts).context("error formatting response")?;
                println!("{casts}");
                return Ok(());
            }

            if casts.is_empty() {
                println!("No screencasts.");
                return Ok(());
            }

            casts.sort_by_key(|c| (c.session_id, c.stream_id));
            for cast in casts {
                print_cast(&cast);
                println!();
            }
        }
    }

    Ok(())
}

fn print_output(output: Output) -> anyhow::Result<()> {
    let Output {
        name,
        make,
        model,
        serial,
        physical_size,
        modes,
        current_mode,
        is_custom_mode,
        vrr_supported,
        vrr_enabled,
        logical,
    } = output;

    let serial = serial.as_deref().unwrap_or("Unknown");
    println!(r#"Output "{make} {model} {serial}" ({name})"#);

    let print_qualifier = |is_preferred: bool, is_current: bool, is_custom_mode: bool| {
        let mut qualifier = Vec::new();
        if is_current {
            qualifier.push("current");
            if is_custom_mode {
                qualifier.push("custom");
            };
        };

        if is_preferred {
            qualifier.push("preferred");
        };

        if qualifier.is_empty() {
            String::new()
        } else {
            format!(" ({})", qualifier.join(", "))
        }
    };

    if let Some(current) = current_mode {
        let mode = *modes
            .get(current)
            .context("invalid response: current mode does not exist")?;
        let Mode {
            width,
            height,
            refresh_rate,
            is_preferred,
        } = mode;
        let refresh = refresh_rate as f64 / 1000.;

        // This is technically the current mode, but the println below already specifies that.
        let qualifier = print_qualifier(is_preferred, false, is_custom_mode);
        println!("  Current mode: {width}x{height} @ {refresh:.3} Hz{qualifier}");
    } else {
        println!("  Disabled");
    }

    if vrr_supported {
        let enabled = if vrr_enabled { "enabled" } else { "disabled" };
        println!("  Variable refresh rate: supported, {enabled}");
    } else {
        println!("  Variable refresh rate: not supported");
    }

    if let Some((width, height)) = physical_size {
        println!("  Physical size: {width}x{height} mm");
    } else {
        println!("  Physical size: unknown");
    }

    if let Some(logical) = logical {
        let LogicalOutput {
            x,
            y,
            width,
            height,
            scale,
            transform,
        } = logical;
        println!("  Logical position: {x}, {y}");
        println!("  Logical size: {width}x{height}");
        println!("  Scale: {scale}");

        let transform = match transform {
            Transform::Normal => "normal",
            Transform::_90 => "90° counter-clockwise",
            Transform::_180 => "180°",
            Transform::_270 => "270° counter-clockwise",
            Transform::Flipped => "flipped horizontally",
            Transform::Flipped90 => "90° counter-clockwise, flipped horizontally",
            Transform::Flipped180 => "flipped vertically",
            Transform::Flipped270 => "270° counter-clockwise, flipped horizontally",
        };
        println!("  Transform: {transform}");
    }

    println!("  Available modes:");
    for (idx, mode) in modes.into_iter().enumerate() {
        let Mode {
            width,
            height,
            refresh_rate,
            is_preferred,
        } = mode;
        let refresh = refresh_rate as f64 / 1000.;

        let is_current = Some(idx) == current_mode;
        let qualifier = print_qualifier(is_preferred, is_current, is_custom_mode);

        println!("    {width}x{height}@{refresh:.3}{qualifier}");
    }
    Ok(())
}

fn print_window(window: &Window) {
    let focused = if window.is_focused { " (focused)" } else { "" };
    let urgent = if window.is_urgent { " (urgent)" } else { "" };
    println!("Window ID {}:{focused}{urgent}", window.id);

    if let Some(title) = &window.title {
        println!("  Title: \"{title}\"");
    } else {
        println!("  Title: (unset)");
    }

    if let Some(app_id) = &window.app_id {
        println!("  App ID: \"{app_id}\"");
    } else {
        println!("  App ID: (unset)");
    }

    if let Some(app_tag) = &window.app_tag {
        println!("  App Tag: \"{app_tag}\"");
    }

    println!(
        "  Is floating: {}",
        if window.is_floating { "yes" } else { "no" }
    );

    if let Some(pid) = window.pid {
        println!("  PID: {pid}");
    } else {
        println!("  PID: (unknown)");
    }

    if let Some(workspace_id) = window.workspace_id {
        println!("  Workspace ID: {workspace_id}");
    } else {
        println!("  Workspace ID: (none)");
    }

    let WindowLayout {
        pos_in_scrolling_layout,
        tile_size,
        window_size,
        tile_pos_in_workspace_view,
        window_offset_in_tile,
    } = window.layout;

    println!("  Layout:");
    println!(
        "    Tile size: {} x {}",
        fmt_rounded(tile_size.0),
        fmt_rounded(tile_size.1)
    );

    if let Some(pos) = pos_in_scrolling_layout {
        println!("    Scrolling position: column {}, tile {}", pos.0, pos.1);
    }

    if let Some(pos) = tile_pos_in_workspace_view {
        println!(
            "    Workspace-view position: {}, {}",
            fmt_rounded(pos.0),
            fmt_rounded(pos.1)
        );
    }

    println!("    Window size: {} x {}", window_size.0, window_size.1);
    println!(
        "    Window offset in tile: {} x {}",
        fmt_rounded(window_offset_in_tile.0),
        fmt_rounded(window_offset_in_tile.1)
    );
}

fn print_activities(activities: &[Activity]) {
    for activity in activities {
        print_activity(activity);
    }
}

fn print_activity(activity: &Activity) {
    let active = if activity.is_active { " * " } else { "   " };
    let config = if activity.is_config_declared {
        " (config)"
    } else {
        ""
    };
    let urgent = if activity.is_urgent { " (urgent)" } else { "" };
    println!(
        "{active}{id} \"{name}\"{config}{urgent}",
        id = activity.id,
        name = activity.name,
    );
}

fn print_bookmarks(bookmarks: &[Bookmark]) {
    if bookmarks.is_empty() {
        println!("No bookmarks.");
        return;
    }

    println!("Bookmarks:");
    for bm in bookmarks {
        let (ws, out) = match &bm.workspace {
            None => ("ws (mid-move)".to_owned(), String::new()),
            Some(placement) => {
                let ws = match (placement.name.as_deref(), placement.idx) {
                    (Some(name), _) => format!("ws \"{name}\""),
                    (None, Some(idx)) => format!("ws {idx}"),
                    (None, None) => "ws (unresolved)".to_owned(),
                };
                let out = placement
                    .output
                    .as_deref()
                    .map(|o| format!(" ({o})"))
                    .unwrap_or_default();
                (ws, out)
            }
        };
        let title = bm.title.as_deref().unwrap_or("(untitled)");
        let act = match bm.activity_name.as_deref() {
            Some(name) => format!("\"{name}\""),
            None => format!("{}", bm.activity_id),
        };
        let key = bm
            .key
            .as_deref()
            .map(|k| format!(" [key {k}]"))
            .unwrap_or_default();
        println!(
            "  [{id}] window {window} \"{title}\" on {ws}{out} [activity {act}]{key}",
            id = bm.id,
            window = bm.window_id,
        );
    }
}

fn print_cast(cast: &Cast) {
    let active = if cast.is_active { "" } else { " (inactive)" };
    println!("Cast stream ID {}:{active}", cast.stream_id);
    println!("  Session ID: {}", cast.session_id);

    let kind = match cast.kind {
        CastKind::PipeWire => "PipeWire",
        CastKind::WlrScreencopy => "wlr-screencopy",
    };
    println!("  Kind: {kind}");

    match &cast.target {
        CastTarget::Nothing {} => {
            println!("  Target: nothing (cleared)");
        }
        CastTarget::Output { name } => {
            println!("  Target: output \"{name}\"");
        }
        CastTarget::Window { id } => {
            println!("  Target: window {id}");
        }
    }

    if cast.is_dynamic_target {
        println!("  Dynamic cast target");
    }

    if let Some(pid) = cast.pid {
        println!("  PID: {pid}");
    }

    if let Some(node_id) = cast.pw_node_id {
        println!("  PipeWire node ID: {node_id}");
    }
}

fn fmt_rounded(x: f64) -> String {
    let r = x.round();
    if (r - x).abs() <= 0.005 {
        format!("{r}")
    } else {
        format!("{x:.2}")
    }
}

fn ensure_absolute_path(path: &mut String) -> anyhow::Result<()> {
    let p = Path::new(path);
    if p.is_relative() {
        let mut cwd = env::current_dir().context("error getting current working directory")?;
        cwd.push(p);
        match cwd.into_os_string().into_string() {
            Ok(absolute) => *path = absolute,
            Err(cwd) => bail!("couldn't convert absolute path to string: {cwd:?}"),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use insta::assert_snapshot;

    use super::*;

    #[test]
    fn test_fmt_rounded() {
        assert_snapshot!(fmt_rounded(1.9), @"1.90");
        assert_snapshot!(fmt_rounded(1.994), @"1.99");
        assert_snapshot!(fmt_rounded(1.996), @"2");
        assert_snapshot!(fmt_rounded(2.0), @"2");
        assert_snapshot!(fmt_rounded(2.004), @"2");
        assert_snapshot!(fmt_rounded(2.006), @"2.01");
        assert_snapshot!(fmt_rounded(2.1), @"2.10");
    }

    fn ws(
        id: u64,
        idx: u8,
        name: Option<&str>,
        output: Option<&str>,
        activities: Vec<u64>,
        is_in_active_activity: bool,
        is_active: bool,
    ) -> Workspace {
        Workspace {
            id,
            idx,
            name: name.map(str::to_owned),
            output: output.map(str::to_owned),
            is_urgent: false,
            is_active,
            is_focused: false,
            active_window_id: None,
            activities,
            is_sticky: false,
            is_in_active_activity,
        }
    }

    #[test]
    fn annotation_single_activity_visible() {
        let w = ws(6, 1, None, Some("DP-3"), vec![1], true, true);
        let names = HashMap::from([(1u64, "Default".to_owned())]);
        assert_eq!(format_annotation(&w, &names), " [activity \"Default\"]");
    }

    #[test]
    fn annotation_multi_activity_visible() {
        let w = ws(15, 3, None, Some("DP-3"), vec![1, 4], true, false);
        let names = HashMap::from([(1u64, "Default".to_owned()), (4u64, "work".to_owned())]);
        assert_eq!(
            format_annotation(&w, &names),
            " [activities \"Default\", \"work\"]"
        );
    }

    #[test]
    fn annotation_single_activity_hidden() {
        let w = ws(4, 0, None, Some("DP-3"), vec![2], false, false);
        let names = HashMap::from([(2u64, "niri".to_owned())]);
        assert_eq!(format_annotation(&w, &names), " [activity \"niri\"]");
    }

    #[test]
    fn annotation_multi_activity_hidden() {
        let w = ws(11, 0, Some("notes"), Some("DP-3"), vec![2, 3], false, false);
        let names = HashMap::from([(2u64, "niri".to_owned()), (3u64, "research".to_owned())]);
        assert_eq!(
            format_annotation(&w, &names),
            " [activities \"niri\", \"research\"]"
        );
    }

    #[test]
    fn annotation_unknown_activity_falls_back_to_id() {
        let w = ws(4, 0, None, Some("DP-3"), vec![99], false, false);
        let names = HashMap::new();
        assert_eq!(format_annotation(&w, &names), " [activity 99]");
    }

    #[test]
    fn annotation_sorts_activities_by_name_asc() {
        // Wire order `[4, 1]`; expected annotation list `"Default", "work"`
        // (ascending by quoted-name string). Pins independence from the
        // wire-side activities Vec ordering.
        let w = ws(15, 3, None, Some("DP-3"), vec![4, 1], true, false);
        let names = HashMap::from([(1u64, "Default".to_owned()), (4u64, "work".to_owned())]);
        assert_eq!(
            format_annotation(&w, &names),
            " [activities \"Default\", \"work\"]"
        );
    }

    #[test]
    fn format_row_visible_named() {
        let w = ws(6, 1, Some("main"), Some("DP-3"), vec![1], true, true);
        let names = HashMap::from([(1u64, "Default".to_owned())]);
        assert_eq!(
            format_row(&w, &names),
            " *  1  id=6    \"main\"     [activity \"Default\"]"
        );
    }

    #[test]
    fn format_row_visible_unnamed() {
        let w = ws(8, 2, None, Some("DP-3"), vec![1], true, false);
        let names = HashMap::from([(1u64, "Default".to_owned())]);
        assert_eq!(
            format_row(&w, &names),
            "    2  id=8               [activity \"Default\"]"
        );
    }

    #[test]
    fn format_row_hidden_unnamed_renders_dash() {
        // Load-bearing: pins the hidden-row `-` rendering rule. The
        // `(idx=0, !is_in_active_activity)` sentinel must never surface
        // as a literal `0` in human-readable output.
        let w = ws(4, 0, None, Some("DP-3"), vec![2], false, false);
        let names = HashMap::from([(2u64, "niri".to_owned())]);
        assert_eq!(
            format_row(&w, &names),
            "    -  id=4               [activity \"niri\"]"
        );
    }

    #[test]
    fn format_row_hidden_named_multi_activity() {
        let w = ws(11, 0, Some("notes"), Some("DP-3"), vec![2, 3], false, false);
        let names = HashMap::from([(2u64, "niri".to_owned()), (3u64, "research".to_owned())]);
        assert_eq!(
            format_row(&w, &names),
            "    -  id=11   \"notes\"    [activities \"niri\", \"research\"]"
        );
    }
}
