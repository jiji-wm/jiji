use niri_config::PresetSize;
use smithay::desktop::Window;
use smithay::output::Output;
use smithay::wayland::shell::xdg::ToplevelSurface;
use smithay::wayland::xdg_activation::XdgActivationTokenData;

use super::ResolvedWindowRules;
use crate::layout::activity::ActivityId;
use crate::layout::workspace::WorkspaceId;

#[derive(Debug)]
pub struct Unmapped {
    pub window: Window,
    pub state: InitialConfigureState,
    /// Activation token, if one was used on this unmapped window.
    pub activation_token_data: Option<XdgActivationTokenData>,
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum InitialConfigureState {
    /// The window has not been initially configured yet.
    NotConfigured {
        /// Whether the window requested to be fullscreened, and the requested output, if any.
        wants_fullscreen: Option<Option<Output>>,

        /// Whether the window requested to be maximized.
        wants_maximized: bool,
    },
    /// The window has been configured.
    Configured {
        /// Up-to-date rules.
        ///
        /// We start tracking window rules when sending the initial configure, since they don't
        /// affect anything before that.
        rules: ResolvedWindowRules,

        /// Resolved scrolling default width for this window.
        ///
        /// `None` means that the window will pick its own width.
        width: Option<PresetSize>,

        /// Resolved scrolling default height for this window.
        ///
        /// `None` means that the window will pick its own height.
        height: Option<PresetSize>,

        /// Resolved floating default width for this window.
        ///
        /// `None` means that the window will pick its own width.
        floating_width: Option<PresetSize>,

        /// Resolved floating default height for this window.
        ///
        /// `None` means that the window will pick its own height.
        floating_height: Option<PresetSize>,

        /// Whether the window should open full-width.
        is_full_width: bool,

        /// Output to open this window on.
        ///
        /// This can be `None` in cases like:
        ///
        /// - There are no outputs connected.
        /// - This is a dialog with a parent, and there was no explicit output set, so this dialog
        ///   should fetch the parent's current output again upon mapping.
        output: Option<Output>,

        /// Workspace to open this window on.
        workspace_name: Option<String>,

        /// Activity that scoped the configure-time monitor / workspace
        /// resolution, when an `open-on-activity` window rule was in
        /// effect. `Some(activity_id)` means the
        /// `workspace_name` lookup at map-time must go through
        /// [`Layout::find_workspace_in_activity_by_name`] rather than
        /// `find_workspace_by_name`, so a hidden-activity workspace is
        /// found. `None` preserves the pre-`open-on-activity` behavior
        /// (active-activity-only resolution).
        target_activity: Option<ActivityId>,

        /// Workspace id directly resolved at configure time, used only
        /// when `target_activity.is_some()` AND the configure-time `ws`
        /// resolution settled on a workspace in the target activity.
        ///
        /// Necessary for the `open-on-activity` "alone" case: the resolved
        /// workspace is often a freshly-materialized
        /// unnamed empty in the hidden activity, so `workspace_name` is
        /// `None` and re-resolving by name at map-time would fall through
        /// to the active activity. Carrying the id directly preserves the
        /// hidden-activity routing.
        ///
        /// Validated at map-time against `workspace_pool().contains_key`
        /// — if the workspace was destroyed between configure and map,
        /// the chain falls through to `output` / `Auto` (matches the
        /// `workspace_name` fallthrough discipline; no error log).
        target_workspace_id: Option<WorkspaceId>,

        /// Whether the window should be maximized.
        ///
        /// This corresponds to the window having the Maximized toplevel state. However, if the
        /// window is also pending fullscreen, then it has the Fullscreen toplevel state, so we
        /// need to store pending maximized elsewhere, hence this field.
        is_pending_maximized: bool,
    },
}

impl Unmapped {
    /// Wraps a newly created window that hasn't been initially configured yet.
    pub fn new(window: Window) -> Self {
        Self {
            window,
            state: InitialConfigureState::NotConfigured {
                wants_fullscreen: None,
                wants_maximized: false,
            },
            activation_token_data: None,
        }
    }

    pub fn needs_initial_configure(&self) -> bool {
        matches!(self.state, InitialConfigureState::NotConfigured { .. })
    }

    pub fn toplevel(&self) -> &ToplevelSurface {
        self.window.toplevel().expect("no X11 support")
    }
}
