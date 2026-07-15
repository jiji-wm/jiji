//! Window layout logic.
//!
//! Niri implements scrollable tiling with dynamic workspaces. The scrollable tiling is mostly
//! orthogonal to any particular workspace system, though outputs living in separate coordinate
//! spaces suggest per-output workspaces.
//!
//! I chose a dynamic workspace system because I think it works very well. In particular, it works
//! naturally across outputs getting added and removed, since workspaces can move between outputs
//! as necessary.
//!
//! In the layout, one output (the first one to be added) is designated as *primary*. This is where
//! workspaces from disconnected outputs will move. Currently, the primary output has no other
//! distinction from other outputs.
//!
//! Where possible, jiji tries to follow these principles with regards to outputs:
//!
//! 1. Disconnecting and reconnecting the same output must not change the layout.
//!    * This includes both secondary outputs and the primary output.
//! 2. Connecting an output must not change the layout for any workspaces that were never on that
//!    output.
//!
//! Therefore, we implement the following logic: every workspace keeps track of which output it
//! originated on—its *original output*. When an output disconnects, its workspaces are appended to
//! the (potentially new) primary output, but remember their original output. Then, if the original
//! output connects again, all workspaces originally from there move back to that output.
//!
//! In order to avoid surprising behavior, if the user creates or moves any new windows onto a
//! workspace, it forgets its original output, and its current output becomes its original output.
//! Imagine a scenario: the user works with a laptop and a monitor at home, then takes their laptop
//! with them, disconnecting the monitor, and keeps working as normal, using the second monitor's
//! workspace just like any other. Then they come back, reconnect the second monitor, and now we
//! don't want an unassuming workspace to end up on it.

use std::cmp::min;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::time::Duration;
use std::{fmt, mem};

use jiji_config::utils::MergeWith as _;
use jiji_config::{
    Config, CornerRadius, FlattenedAppearance, LayoutPart, PresetSize,
    Workspace as WorkspaceConfig, WorkspaceReference,
};
use jiji_ipc::{ActivityReferenceArg, ColumnDisplay, PositionChange, SizeChange, WindowLayout};
use monitor::{InsertHint, InsertPosition, InsertWorkspace, MonitorAddWindowTarget};
use scrolling::{Column, ColumnWidth};
use smithay::backend::renderer::element::surface::WaylandSurfaceRenderElement;
use smithay::backend::renderer::element::utils::RescaleRenderElement;
use smithay::backend::renderer::gles::{GlesRenderer, GlesTexture};
use smithay::output::{self, Output};
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::utils::{Logical, Point, Rectangle, Scale, Serial, Size, Transform};
use tile::{Tile, TileRenderElement};
use workspace::{WorkspaceAddWindowTarget, WorkspaceId};

pub(crate) use self::activity::ReloadActivityRemovalError;
use self::activity::{Activities, Activity, ActivityId, WorkspaceView};
pub use self::activity::{
    AddWorkspaceToActivityError, CreateActivityError, MoveWorkspaceToActivityError,
    RemoveActivityError, RemoveWorkspaceFromActivityError, RenameActivityError,
    SetWorkspaceActivitiesError, SwitchActivityError, WorkspaceStickyError,
};
use self::bookmarks::Bookmarks;
pub use self::monitor::MonitorRenderElement;
use self::monitor::{ActivitySwitch, Monitor, SlideDirection, StripCtx, WorkspaceSwitch};
use self::workspace::{OutputId, Workspace};
use crate::animation::{Animation, Clock};
use crate::input::swipe_tracker::SwipeTracker;
use crate::layout::scrolling::ScrollDirection;
use crate::niri_render_elements;
use crate::render_helpers::background_effect::BackgroundEffectElement;
use crate::render_helpers::offscreen::OffscreenData;
use crate::render_helpers::renderer::NiriRenderer;
use crate::render_helpers::snapshot::RenderSnapshot;
use crate::render_helpers::solid_color::{SolidColorBuffer, SolidColorRenderElement};
use crate::render_helpers::texture::TextureBuffer;
use crate::render_helpers::xray::{Xray, XrayPos};
use crate::render_helpers::{BakedBuffer, RenderCtx};
use crate::rubber_band::RubberBand;
use crate::utils::transaction::{Transaction, TransactionBlocker};
use crate::utils::{
    ensure_min_max_size_maybe_zero, output_matches_name, output_size,
    round_logical_in_physical_max1, ResizeEdge,
};
use crate::window::ResolvedWindowRules;

pub mod activity;
pub mod bookmarks;
pub mod closing_window;
pub mod floating;
pub mod focus_ring;
pub mod insert_hint_element;
pub mod monitor;
pub mod opening_window;
pub mod scrolling;
pub mod shadow;
pub mod tab_indicator;
pub mod tile;
pub mod workspace;

#[cfg(test)]
mod tests;

/// Size changes up to this many pixels don't animate.
pub const RESIZE_ANIMATION_THRESHOLD: f64 = 10.;

/// Pointer needs to move this far to pull a window from the layout.
const INTERACTIVE_MOVE_START_THRESHOLD: f64 = 256. * 256.;

/// Opacity of interactively moved tiles targeting the scrolling layout.
const INTERACTIVE_MOVE_ALPHA: f64 = 0.75;

/// Amount of touchpad movement to toggle the overview.
const OVERVIEW_GESTURE_MOVEMENT: f64 = 300.;

const OVERVIEW_GESTURE_RUBBER_BAND: RubberBand = RubberBand {
    stiffness: 0.5,
    limit: 0.05,
};

/// Size-relative units.
pub struct SizeFrac;

niri_render_elements! {
    LayoutElementRenderElement<R> => {
        Wayland = WaylandSurfaceRenderElement<R>,
        SolidColor = SolidColorRenderElement,
        BackgroundEffect = BackgroundEffectElement,
    }
}

pub type LayoutElementRenderSnapshot =
    RenderSnapshot<BakedBuffer<TextureBuffer<GlesTexture>>, BakedBuffer<SolidColorBuffer>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SizingMode {
    Normal,
    Maximized,
    Fullscreen,
}

/// Read-only context for render and hit-test paths on a monitor.
///
/// Bundles `(&pool, &view)` so render-path method signatures don't have to
/// juggle both as separate args, and so the view's storage location can
/// change without re-touching every call site.
///
/// Prefer `layout.ctx_for(mon)` to construct this at call sites. Both borrows
/// are shared, so a caller holding `&mon` can pass `ctx` into `&self` methods
/// on the same `mon` without conflict. The raw constructor `LayoutCtx::new`
/// is `pub(crate)`: it remains available for in-crate sites where `&pool` is
/// already bound separately from the layout and `Layout::ctx_for` can't be
/// called, but crate-external callers must go through `Layout::ctx_for`.
#[derive(Debug)]
pub struct LayoutCtx<'a, W: LayoutElement> {
    pool: &'a HashMap<WorkspaceId, Workspace<W>>,
    view: &'a WorkspaceView,
}

// Manual `Copy`/`Clone` — the derives would otherwise inherit `W: Copy` /
// `W: Clone`, but `LayoutCtx` only holds shared references so `W` doesn't need
// either bound.
impl<W: LayoutElement> Copy for LayoutCtx<'_, W> {}
impl<W: LayoutElement> Clone for LayoutCtx<'_, W> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, W: LayoutElement> LayoutCtx<'a, W> {
    pub(crate) fn new(
        pool: &'a HashMap<WorkspaceId, Workspace<W>>,
        view: &'a WorkspaceView,
    ) -> Self {
        Self { pool, view }
    }

    pub fn pool(&self) -> &'a HashMap<WorkspaceId, Workspace<W>> {
        self.pool
    }

    pub fn view(&self) -> &'a WorkspaceView {
        self.view
    }

    /// Borrow a workspace by id. Panics if the id is not in the pool — a broken
    /// pool/view invariant, not user error.
    pub fn workspace(&self, id: WorkspaceId) -> &'a Workspace<W> {
        self.pool
            .get(&id)
            .expect("workspace id must be a key in the pool")
    }

    /// Borrow the workspace at visual position `pos` within the view. Panics
    /// if `pos` is out of range for the view, or the id is absent from the
    /// pool — both indicate a broken invariant.
    pub fn workspace_at(&self, pos: usize) -> &'a Workspace<W> {
        self.workspace(self.view.ids()[pos])
    }
}

pub trait LayoutElement {
    /// Type that can be used as a unique ID of this element.
    type Id: PartialEq + std::fmt::Debug + Clone;

    /// Unique ID of this element.
    fn id(&self) -> &Self::Id;

    /// Updates the config for the element.
    fn update_config(&mut self, blur_config: jiji_config::Blur) {
        let _ = blur_config;
    }

    /// Visual size of the element.
    ///
    /// This is what the user would consider the size, i.e. excluding CSD shadows and whatnot.
    /// Corresponds to the Wayland window geometry size.
    fn size(&self) -> Size<i32, Logical>;

    /// Returns the location of the element's buffer relative to the element's visual geometry.
    ///
    /// I.e. if the element has CSD shadows, its buffer location will have negative coordinates.
    fn buf_loc(&self) -> Point<i32, Logical>;

    /// Checks whether a point is in the element's input region.
    ///
    /// The point is relative to the element's visual geometry.
    fn is_in_input_region(&self, point: Point<f64, Logical>) -> bool;

    /// Renders the element at the given visual location.
    ///
    /// The element should be rendered in such a way that its visual geometry ends up at the given
    /// location.
    fn render<R: NiriRenderer>(
        &self,
        mut ctx: RenderCtx<R>,
        location: Point<f64, Logical>,
        scale: Scale<f64>,
        alpha: f32,
        xray_pos: XrayPos,
        push: &mut dyn FnMut(LayoutElementRenderElement<R>),
    ) {
        self.render_popups(ctx.r(), location, scale, alpha, xray_pos, push);
        self.render_normal(ctx.r(), location, scale, alpha, push);
    }

    /// Renders the non-popup parts of the element.
    fn render_normal<R: NiriRenderer>(
        &self,
        ctx: RenderCtx<R>,
        location: Point<f64, Logical>,
        scale: Scale<f64>,
        alpha: f32,
        push: &mut dyn FnMut(LayoutElementRenderElement<R>),
    ) {
        let _ = (ctx, location, scale, alpha, push);
    }

    /// Renders the popups of the element.
    fn render_popups<R: NiriRenderer>(
        &self,
        ctx: RenderCtx<R>,
        location: Point<f64, Logical>,
        scale: Scale<f64>,
        alpha: f32,
        xray_pos: XrayPos,
        push: &mut dyn FnMut(LayoutElementRenderElement<R>),
    ) {
        let _ = (ctx, location, scale, alpha, xray_pos, push);
    }

    /// Renders the background effect behind the main surface of the element.
    #[allow(clippy::too_many_arguments)]
    fn render_background_effect(
        &self,
        _ctx: RenderCtx<GlesRenderer>,
        _geometry: Rectangle<f64, Logical>,
        _scale: f64,
        _clip_to_geometry: bool,
        _surface_anim_scale: Scale<f64>,
        _radius: CornerRadius,
        _xray_pos: XrayPos,
        _push: &mut dyn FnMut(BackgroundEffectElement),
    ) {
    }

    /// Requests the element to change its size.
    ///
    /// The size request is stored and will be continuously sent to the element on any further
    /// state changes.
    fn request_size(
        &mut self,
        size: Size<i32, Logical>,
        mode: SizingMode,
        animate: bool,
        transaction: Option<Transaction>,
    );

    /// Requests the element to change size once, clearing the request afterwards.
    fn request_size_once(&mut self, size: Size<i32, Logical>, animate: bool) {
        self.request_size(size, SizingMode::Normal, animate, None);
    }

    fn min_size(&self) -> Size<i32, Logical>;
    fn max_size(&self) -> Size<i32, Logical>;
    fn is_wl_surface(&self, wl_surface: &WlSurface) -> bool;
    fn has_ssd(&self) -> bool;
    fn set_preferred_scale_transform(&self, scale: output::Scale, transform: Transform);
    fn output_enter(&self, output: &Output);
    fn output_leave(&self, output: &Output);
    fn set_offscreen_data(&self, data: Option<OffscreenData>);
    fn set_activated(&mut self, active: bool);
    fn set_active_in_column(&mut self, active: bool);
    fn set_floating(&mut self, floating: bool);
    fn set_bounds(&self, bounds: Size<i32, Logical>);
    fn is_ignoring_opacity_window_rule(&self) -> bool;

    fn is_urgent(&self) -> bool;

    fn configure_intent(&self) -> ConfigureIntent;
    fn send_pending_configure(&mut self);

    /// The element's current sizing mode.
    ///
    /// This will *not* switch immediately after a [`LayoutElement::request_size()`] call.
    fn sizing_mode(&self) -> SizingMode;

    /// The sizing mode that we're requesting the element to assume.
    ///
    /// This *will* switch immediately after a [`LayoutElement::request_size()`] call.
    fn pending_sizing_mode(&self) -> SizingMode;

    /// Size previously requested through [`LayoutElement::request_size()`].
    fn requested_size(&self) -> Option<Size<i32, Logical>>;

    /// Non-fullscreen size that we expect this window has or will shortly have.
    ///
    /// This can be different from [`requested_size()`](LayoutElement::requested_size()). For
    /// example, for floating windows this will generally return the current window size, rather
    /// than the last size that we requested, since we want floating windows to be able to change
    /// size freely. But not always: if we just requested a floating window to resize and it hasn't
    /// responded to it yet, this will return the newly requested size.
    ///
    /// This function should never return a 0 size component. `None` means there's no known
    /// expected size (for example, the window is fullscreen).
    ///
    /// The default impl is for testing only, it will not preserve the window's own size changes.
    fn expected_size(&self) -> Option<Size<i32, Logical>> {
        if self.sizing_mode().is_fullscreen() {
            return None;
        }

        let mut requested = self.requested_size().unwrap_or_default();
        let current = self.size();
        if requested.w == 0 {
            requested.w = current.w;
        }
        if requested.h == 0 {
            requested.h = current.h;
        }
        Some(requested)
    }

    fn is_windowed_fullscreen(&self) -> bool {
        false
    }
    fn is_pending_windowed_fullscreen(&self) -> bool {
        false
    }
    fn request_windowed_fullscreen(&mut self, value: bool) {
        let _ = value;
    }

    /// The effective geometry corner radius for this element.
    ///
    /// Returns zero when the element is in windowed fullscreen, since fullscreen windows have
    /// square corners.
    ///
    /// This method only handles windowed fullscreen and not maximized/real fullscreen. This is
    /// because windowed fullscreen is handled by the element itself, whereas other sizing modes
    /// are handled externally by the Tile, so the corner radius changes for those modes is also
    /// handled externally.
    fn geometry_corner_radius(&self) -> CornerRadius {
        if self.is_windowed_fullscreen() {
            return CornerRadius::default();
        }
        self.rules().geometry_corner_radius.unwrap_or_default()
    }

    fn is_child_of(&self, parent: &Self) -> bool;

    fn rules(&self) -> &ResolvedWindowRules;

    /// Runs periodic clean-up tasks.
    fn refresh(&self);

    fn take_animation_snapshot(&mut self) -> Option<LayoutElementRenderSnapshot>;

    fn set_interactive_resize(&mut self, data: Option<InteractiveResizeData>);
    fn cancel_interactive_resize(&mut self);
    fn interactive_resize_data(&self) -> Option<InteractiveResizeData>;

    fn on_commit(&mut self, serial: Serial);
}

#[derive(Debug)]
pub struct Layout<W: LayoutElement> {
    /// Connected monitors. Empty when no outputs are attached.
    monitors: Vec<Monitor<W>>,
    /// Index of the primary monitor within `monitors`. When `monitors.is_empty()`, this holds
    /// a sentinel `0` (the field has no meaning in that state — guard with `monitors.is_empty()`
    /// before indexing). Otherwise `primary_idx < monitors.len()`.
    primary_idx: usize,
    /// Index of the active monitor within `monitors`. Sentinel/invariant identical to
    /// `primary_idx`.
    active_monitor_idx: usize,
    /// Ids of workspaces kept alive while no monitor is connected, in display order as preserved
    /// from the last-disconnected monitor (or from config at startup). Empty when any monitor is
    /// connected; when the first monitor reconnects, these are moved onto it in this order.
    disconnected_workspace_ids: Vec<WorkspaceId>,
    /// Owning pool of `Workspace<W>` values keyed by id.
    ///
    /// Every id appearing in any view in the active activity's `views` map, or in
    /// `disconnected_workspace_ids`, is a key here; the pool keys equal the disjoint union of
    /// those two sources — no orphans, no duplicates. Pool values are never drained out during
    /// output reconnect; monitors bind/unbind the `Smithay` output on their workspaces in place.
    pub(super) workspaces: HashMap<WorkspaceId, Workspace<W>>,
    /// Ordered pool of activities plus active / previous cursors.
    ///
    /// Cross-field invariant: every `Workspace.activities` entry (in every value of
    /// `self.workspaces`) is a key in this `Activities` map, and each workspace's set is
    /// non-empty. Checked in `Layout::verify_invariants`. New workspaces created at runtime
    /// are stamped with `self.activities.active_id()` at construction; this field owns the
    /// seed that makes the ctor contract realizable.
    activities: Activities,
    /// Curated window bookmarks the user walks with forward/backward.
    ///
    /// Cross-field guarantee, asserted in [`Self::verify_invariants`]: every
    /// *attached* bookmark (window anchor, or a rule anchor with a live
    /// attachment) anchors a window present in `windows_all()` (prune-on-close
    /// drops dead entries); a dangling rule anchor is exempt (no window to
    /// check). Ids are unique, and the walk cursor is in bounds when `Some`.
    bookmarks: Bookmarks<W::Id>,
    /// Whether the layout should draw as active.
    ///
    /// This normally indicates that the layout has keyboard focus, but not always. E.g. when the
    /// screenshot UI is open, it keeps the layout drawing as active.
    is_active: bool,
    /// Map from monitor name to id of its last active workspace.
    ///
    /// This data is stored upon monitor removal and is used to restore the active workspace when
    /// the monitor is reconnected.
    ///
    /// The workspace id does not necessarily point to a valid workspace. If it doesn't, then it is
    /// simply ignored.
    last_active_workspace_id: HashMap<String, WorkspaceId>,
    /// Ongoing interactive move.
    interactive_move: Option<InteractiveMoveState<W>>,
    /// Ongoing drag-and-drop operation.
    dnd: Option<DndData<W>>,
    /// Clock for driving animations.
    clock: Clock,
    /// Time that we last updated render elements for.
    update_render_elements_time: Duration,
    /// Whether the overview is open.
    ///
    /// This is a boolean flag that controls things like where input goes to. The actual animation
    /// is controlled by overview_progress.
    overview_open: bool,
    /// The overview zoom progress.
    overview_progress: Option<OverviewProgress>,
    /// Configurable properties of the layout.
    options: Rc<Options>,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Options {
    pub layout: jiji_config::Layout,
    pub animations: jiji_config::Animations,
    pub gestures: jiji_config::Gestures,
    pub overview: jiji_config::Overview,
    pub blur: jiji_config::Blur,
    pub bookmarks: jiji_config::BookmarksConfig,
    // Debug flags.
    pub disable_resize_throttling: bool,
    pub disable_transactions: bool,
    pub deactivate_unfocused_windows: bool,
}

/// Why an activity switch was rejected without proceeding. Returned by
/// [`Layout::is_activity_switch_hard_blocked`] for callers that need to gate
/// before reaching [`Layout::switch_activity`].
///
/// Only the live-input states block: workspace-switch
/// animations are *not* covered, since the switch itself snaps them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActivitySwitchBlock {
    InteractiveMove,
    Dnd,
    WorkspaceSwitchGesture,
}

impl fmt::Display for ActivitySwitchBlock {
    /// Stable human-readable token for each block reason.
    ///
    /// These three strings are part of the observable IPC wire contract: the
    /// `Request::Action` dispatch formats them into a `Reply::Err("activity
    /// switch blocked: {token}")` string that external clients may
    /// pattern-match. Changing any token is a breaking change. The tokens are
    /// pinned by `activity_switch_block_display_matches_wire_contract` in
    /// `src/layout/tests.rs`; the full envelope string is assembled by
    /// [`format_activity_switch_block_err`] and pinned by
    /// `activity_switch_block_err_envelope_matches_wire_contract` there and by
    /// the serde roundtrip `reply_err_format_for_activity_switch_block` in
    /// `jiji-ipc`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::InteractiveMove => "interactive window move",
            Self::Dnd => "drag and drop",
            Self::WorkspaceSwitchGesture => "workspace switch gesture",
        };
        f.write_str(s)
    }
}

/// Assemble the full IPC wire error string for a hard-block reason.
///
/// The format `"activity switch blocked: {token}"` is the stable
/// observable contract that IPC clients may pattern-match against. Both the
/// `ipc/server.rs` call site and the envelope pin test call this function, so
/// a regression to the format string here will fail
/// `activity_switch_block_err_envelope_matches_wire_contract`.
pub(crate) fn format_activity_switch_block_err(block: ActivitySwitchBlock) -> String {
    format!("activity switch blocked: {block}")
}

/// Error type returned by [`State::do_action_inner`] for actions that can fail
/// with either an activity-switch hard-block, a missing window id, or a
/// layout-side validation rejection from one of the activity-action handlers.
///
/// Wraps [`ActivitySwitchBlock`] (the hard-block reasons), surfaces
/// [`DoActionError::WindowNotFound`] for the `Action::FocusWindow` "window no
/// longer exists" wire contract, and carries the layout-side validation enums
/// (e.g. [`AddWorkspaceToActivityError`]) verbatim as payloads on outer
/// variants. Each outer variant's `Display` impl delegates to the wrapped
/// inner enum's `Display` (one-line `e.fmt(f)`), so the inner enum's tokens
/// are the source of truth for the wire string.
///
/// Kept `pub(crate)` because it is an internal dispatch error — the IPC
/// surface flattens it to a `Reply::Err(String)` envelope via
/// [`format_do_action_error`].
///
/// The `Display` tokens and envelope format are part of the stable observable
/// IPC wire contract and are pinned by three layered tests:
///
/// - `do_action_error_display_matches_wire_contract` in `src/layout/tests.rs` — pins each variant's
///   `Display` token.
/// - `do_action_error_envelope_matches_wire_contract` in `src/layout/tests.rs` — pins the full
///   envelope string assembled by [`format_do_action_error`], including byte-identity with the
///   existing `ActivitySwitchBlocked` envelopes.
/// - `reply_err_format_for_window_not_found` in `jiji-ipc` — roundtrips the wire envelope through
///   `Reply::Err` JSON.
///
/// Any change to the token strings or the envelope wording must update all
/// three pin sites together. Token drift on a wrapped inner enum's `Display`
/// fails the outer pin tests in addition to the inner enum's own pin tests.
///
/// Do **not** add a wildcard arm to [`DoActionError::disposition`] — the
/// single exhaustive match in `disposition()` is now the load-bearing
/// classification guard. The two `ipc/server.rs` dispatch sites (drain-walk
/// and insert-idle) both match on `disposition()` rather than re-listing
/// variants themselves, so parity between the two sites is structural rather
/// than maintained by parallel variant lists.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum DoActionError {
    /// A hard-block (interactive move, DnD, or workspace-switch
    /// gesture) prevented an activity switch. The IPC dispatch path parks
    /// this error on the per-connection queue and re-dispatches on drain.
    ActivitySwitchBlocked(ActivitySwitchBlock),
    /// `Action::FocusWindow { id }` was dispatched with an id that
    /// does not resolve to any pool-owned window. Terminal error — the
    /// waiter is signalled immediately, never parked.
    WindowNotFound { id: u64 },
    /// `Action::MoveWindowToWorkspace { reference: Id(_), .. }` resolved to a
    /// workspace id that names nothing movable: either the id is absent from
    /// the workspace pool, or the workspace is bound to an output that is
    /// not currently connected. Terminal error — replaces the pre-fix silent
    /// `Response::Handled` (which surfaced as a CLI false-success line).
    MoveWindowTargetUnreachable { ws_id: u64 },
    /// `Action::MoveWindowToWorkspace` or `Action::MoveWindowToWorkspaceById`
    /// was dispatched with a `Name` reference that resolves to no workspace
    /// reachable from the active-view scope (unknown name). Terminal error —
    /// the `Name`-fall-through sibling of `MoveWindowTargetUnreachable` (which
    /// covers the `Id`-only path). The `name` field carries the offending token
    /// for the wire message.
    ///
    /// Note: the `Index` arm is out of scope — `find_output_and_workspace_index`
    /// always returns `Some` for `Index` (saturating clamp), so the `Index`
    /// form keeps its clamp behaviour unchanged.
    ///
    /// Note: the `name` field carries a bare workspace name string, unlike the
    /// `reference` field of [`DoActionError::FocusWorkspaceTargetUnknown`] which
    /// carries a pre-formatted token (`id:N`, index, or name) — the two payloads
    /// are not the same kind of value.
    MoveWindowTargetUnknownName { name: String },
    /// `Action::CreateActivity` validation failed. Wraps the layout-side
    /// [`CreateActivityError`]. Terminal error.
    CreateActivity(CreateActivityError),
    /// `Action::RemoveActivity` validation failed. Wraps the layout-side
    /// [`RemoveActivityError`]. Terminal error.
    RemoveActivity(RemoveActivityError),
    /// `Action::RenameActivity` validation failed. Wraps the layout-side
    /// [`RenameActivityError`]. Terminal error.
    RenameActivity(RenameActivityError),
    /// `Action::SwitchActivity` was dispatched with a reference that does
    /// not resolve. Wraps the dispatch-side [`SwitchActivityError`].
    /// Terminal error.
    SwitchActivity(SwitchActivityError),
    /// `Action::AddWorkspaceToActivity` validation failed. Wraps
    /// the layout-side [`AddWorkspaceToActivityError`]. Terminal error.
    AddWorkspaceToActivity(AddWorkspaceToActivityError),
    /// `Action::RemoveWorkspaceFromActivity` validation failed. Wraps
    /// the layout-side [`RemoveWorkspaceFromActivityError`]. Terminal error.
    RemoveWorkspaceFromActivity(RemoveWorkspaceFromActivityError),
    /// `Action::SetWorkspaceActivities` validation failed. Wraps
    /// the layout-side [`SetWorkspaceActivitiesError`]. Terminal error.
    SetWorkspaceActivities(SetWorkspaceActivitiesError),
    /// `Action::MoveWorkspaceToActivity` validation failed. Wraps
    /// the layout-side [`MoveWorkspaceToActivityError`]. Terminal error.
    MoveWorkspaceToActivity(MoveWorkspaceToActivityError),
    /// `Action::ToggleWorkspaceSticky` failed. Wraps the layout-side
    /// [`WorkspaceStickyError`], shared with `SetWorkspaceSticky` and
    /// `UnsetWorkspaceSticky` — this outer variant, not the payload,
    /// identifies which verb failed. Terminal error.
    ToggleWorkspaceSticky(WorkspaceStickyError),
    /// `Action::SetWorkspaceSticky` failed. Wraps the layout-side
    /// [`WorkspaceStickyError`], shared with `ToggleWorkspaceSticky` and
    /// `UnsetWorkspaceSticky` — this outer variant, not the payload,
    /// identifies which verb failed. Terminal error.
    SetWorkspaceSticky(WorkspaceStickyError),
    /// `Action::UnsetWorkspaceSticky` failed. Wraps the layout-side
    /// [`WorkspaceStickyError`], shared with `ToggleWorkspaceSticky` and
    /// `SetWorkspaceSticky` — this outer variant, not the payload,
    /// identifies which verb failed. Terminal error.
    UnsetWorkspaceSticky(WorkspaceStickyError),
    /// `Action::FocusWorkspace { activity: Some(_), .. }` resolved to a
    /// workspace that is not in the requested activity, or the caller used a
    /// positional index (unsupported in activity-scoped lookup). Terminal
    /// error — never parked on the blocked-waiter queue.
    FocusWorkspaceInActivity(FocusWorkspaceInActivityError),
    /// `Action::FocusWorkspace { activity: None, .. }` was dispatched with a
    /// `Name` or `Id` reference that resolves to no workspace reachable from
    /// the active-view + disconnected-pool scope (unknown name/id, or an id
    /// known only to a dormant activity's exclusive workspaces). Terminal
    /// error — replaces the prior silent no-op on the IPC path. The
    /// `reference` field carries the offending token as a human-readable
    /// string for the wire message.
    ///
    /// Note: the `Index` arm is out of scope — `find_output_and_workspace_index`
    /// always returns `Some` for `Index` and upstream clamps out-of-range
    /// indices, so the `Index` form keeps its clamp behaviour unchanged.
    FocusWorkspaceTargetUnknown { reference: String },
    /// A bookmark action was dispatched with a bookmark id absent from the
    /// list (remove, jump, move, assign/unassign key, or a keybind-driven
    /// jump). Terminal error — the waiter is signalled immediately, never
    /// parked. One variant serves every id-taking bookmark action: the
    /// failure semantics and wire message are identical, and the caller knows
    /// which verb it invoked.
    BookmarkNotFound { id: u64 },
    /// `Action::AssignBookmarkKey` was dispatched with a string that fails to
    /// parse as a [`jiji_config::Key`], names a non-keysym trigger, or names a
    /// keysym with no modifiers. Terminal error — no state mutation. `key` is
    /// the offending raw string as given by the caller.
    BookmarkKeyInvalid { key: String, reason: String },
    /// `Action::AssignBookmarkKey` was dispatched with a key that already
    /// matches a static config bind, a recent-windows bind, or another
    /// bookmark's key. Terminal error — no state mutation. `key` is the
    /// canonical formatted key string.
    BookmarkKeyCollision { key: String },
    /// `Action::RenameBookmark` was dispatched with a name that fails
    /// [`bookmarks::BookmarkName::new`] validation (empty after trim, or
    /// containing a control character). Terminal error — no state mutation.
    /// `name` is the offending raw string as given by the caller.
    BookmarkNameInvalid { name: String, reason: String },
    /// A jump (`Action::JumpToBookmark` or a keybind-driven jump) targeted a
    /// rule bookmark that currently has no attached window. Terminal error — no
    /// state mutation.
    BookmarkDangling { id: u64 },
    /// `Action::AddBookmarkRule` was dispatched with a rule that fails to
    /// compile (a bad regex) or names no fields at all. Terminal error — no
    /// state mutation. `reason` describes the failure.
    BookmarkRuleInvalid { reason: String },
    /// `Action::SetAppearanceOverride` was dispatched with a payload that
    /// fails to resolve: a color or regex field failed to compile. Terminal
    /// error — no state mutation. `reason` names the offending field and
    /// value.
    AppearanceOverrideInvalid { reason: String },
}

/// How the IPC dispatch path must react to a [`DoActionError`].
///
/// `Park`: the error is a transient hard-block. The dispatch path parks the
/// waiter on the per-connection queue ([`crate::ipc::server::IpcServer::blocked_action_waiters`])
/// and re-dispatches it on the next drain once the hard-block clears.
///
/// `Terminal`: the waiter is signalled immediately. Parking a terminal error
/// would deadlock the connection — there is no hard-block condition for a
/// later drain to clear, so a parked terminal waiter would never be
/// re-dispatched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Disposition {
    Park,
    Terminal,
}

impl DoActionError {
    /// Classifies this error as [`Disposition::Park`] or
    /// [`Disposition::Terminal`] for the IPC dispatch path.
    ///
    /// This is a single exhaustive match with no wildcard arm by design: a
    /// new variant must fail to compile here until its disposition is an
    /// explicit author decision, rather than silently inheriting whatever a
    /// wildcard arm picks.
    pub(crate) fn disposition(&self) -> Disposition {
        match self {
            Self::ActivitySwitchBlocked(_) => Disposition::Park,
            Self::WindowNotFound { .. }
            | Self::MoveWindowTargetUnreachable { .. }
            | Self::MoveWindowTargetUnknownName { .. }
            | Self::CreateActivity(_)
            | Self::RemoveActivity(_)
            | Self::RenameActivity(_)
            | Self::SwitchActivity(_)
            | Self::AddWorkspaceToActivity(_)
            | Self::RemoveWorkspaceFromActivity(_)
            | Self::SetWorkspaceActivities(_)
            | Self::MoveWorkspaceToActivity(_)
            | Self::ToggleWorkspaceSticky(_)
            | Self::SetWorkspaceSticky(_)
            | Self::UnsetWorkspaceSticky(_)
            | Self::FocusWorkspaceInActivity(_)
            | Self::FocusWorkspaceTargetUnknown { .. }
            | Self::BookmarkNotFound { .. }
            | Self::BookmarkKeyInvalid { .. }
            | Self::BookmarkKeyCollision { .. }
            | Self::BookmarkNameInvalid { .. }
            | Self::BookmarkDangling { .. }
            | Self::BookmarkRuleInvalid { .. }
            | Self::AppearanceOverrideInvalid { .. } => Disposition::Terminal,
        }
    }
}

impl From<ActivitySwitchBlock> for DoActionError {
    fn from(block: ActivitySwitchBlock) -> Self {
        Self::ActivitySwitchBlocked(block)
    }
}

/// Successful outcome of [`crate::niri::State::do_action_inner`]. Mirrors the
/// jiji-ipc `Reply::Ok(...)` envelope: `Handled` is the default for actions
/// that change state; `NoOp(reason)` is the typed signal that the action was
/// considered and determined to leave compositor state unchanged.
///
/// The IPC dispatch sites at `src/ipc/server.rs` (`process` recv-site and
/// `drain_blocked_action_waiters`) map each variant to a `Response`:
/// `Handled` ⇒ `Response::Handled`, `NoOp(reason)` ⇒
/// `Response::NoOp(reason)`. Only `Action::MoveWindowToWorkspace` /
/// `Action::MoveWindowToWorkspaceById` currently produce `NoOp`; every other
/// dispatch arm continues to return `Handled`.
///
/// Unlike [`DoActionError`], this enum is matched at a single recv-site in
/// `process` (no parity-pair between dispatch sites to guard). A `_` wildcard
/// would be safe wrt site-parity but would silently subsume future variants
/// — the exhaustive match is therefore for classification surface, not
/// cross-site consistency.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DoActionOutcome {
    Handled,
    NoOp(jiji_ipc::NoOpReason),
}

/// Outcome of [`Layout::toggle_workspace_sticky`]. Carries enough information
/// for the dispatch layer to log the toggle direction and decide whether to
/// fire the cursor-warp / redraw pair.
///
/// The sum type makes the `(StickyOff, active_affected: true)` combination
/// unrepresentable: `active_affected` is only meaningful when toggling on
/// (Unset never touches `activities`, so toggle-off always has
/// `active_affected = false`). `StickyOff` simply does not carry the field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToggleWorkspaceStickyOutcome {
    /// Toggled from off to on. `active_affected` bubbles up from the inner
    /// `set_workspace_activities` call and indicates whether the active
    /// activity's workspace set changed (cursor-warp / redraw trigger).
    StickyOn {
        ws_id: WorkspaceId,
        active_affected: bool,
    },
    /// Toggled from on to off. Activities set is unchanged; no redraw needed.
    StickyOff { ws_id: WorkspaceId },
}

impl fmt::Display for DoActionError {
    /// Stable human-readable token for each error variant.
    ///
    /// These strings are part of the observable IPC wire contract. The full
    /// envelope is assembled by [`format_do_action_error`]; the tokens are
    /// pinned by `do_action_error_display_matches_wire_contract` in
    /// `src/layout/tests.rs` and the envelope is pinned by
    /// `do_action_error_envelope_matches_wire_contract` there and by the
    /// serde roundtrip `reply_err_format_for_window_not_found` in `jiji-ipc`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ActivitySwitchBlocked(block) => write!(f, "{block}"),
            Self::WindowNotFound { id } => write!(f, "window not found: id={id}"),
            Self::MoveWindowTargetUnreachable { ws_id } => {
                write!(f, "workspace not reachable for move: id={ws_id}")
            }
            Self::MoveWindowTargetUnknownName { name } => {
                write!(f, "workspace not found for move: {name}")
            }
            // Outer variants wrapping a layout-side `*Error` delegate to the
            // inner enum's `Display`. The inner enum's tokens are the source
            // of truth for the wire string; `format_do_action_error` returns
            // the bare token without further wrapping.
            Self::CreateActivity(e) => e.fmt(f),
            Self::RemoveActivity(e) => e.fmt(f),
            Self::RenameActivity(e) => e.fmt(f),
            Self::SwitchActivity(e) => e.fmt(f),
            Self::AddWorkspaceToActivity(e) => e.fmt(f),
            Self::RemoveWorkspaceFromActivity(e) => e.fmt(f),
            Self::SetWorkspaceActivities(e) => e.fmt(f),
            Self::MoveWorkspaceToActivity(e) => e.fmt(f),
            Self::ToggleWorkspaceSticky(e) => e.fmt(f),
            Self::SetWorkspaceSticky(e) => e.fmt(f),
            Self::UnsetWorkspaceSticky(e) => e.fmt(f),
            Self::FocusWorkspaceInActivity(e) => e.fmt(f),
            Self::FocusWorkspaceTargetUnknown { reference } => {
                write!(f, "workspace not found: {reference}")
            }
            Self::BookmarkNotFound { id } => write!(f, "bookmark not found: id={id}"),
            Self::BookmarkKeyInvalid { key, reason } => {
                write!(f, "invalid bookmark key: {key}: {reason}")
            }
            Self::BookmarkKeyCollision { key } => write!(f, "bookmark key already bound: {key}"),
            Self::BookmarkNameInvalid { name, reason } => {
                write!(f, "invalid bookmark name: {name}: {reason}")
            }
            Self::BookmarkDangling { id } => write!(f, "bookmark has no attached window: id={id}"),
            Self::BookmarkRuleInvalid { reason } => write!(f, "invalid bookmark rule: {reason}"),
            Self::AppearanceOverrideInvalid { reason } => {
                write!(f, "appearance override invalid: {reason}")
            }
        }
    }
}

/// Assemble the full IPC wire error string for a [`DoActionError`].
///
/// Parallels [`format_activity_switch_block_err`]. For the
/// `ActivitySwitchBlocked` variant this function delegates to
/// [`format_activity_switch_block_err`] so the envelope is produced
/// byte-identical to the pre-1b wire contract — IPC clients that
/// pattern-match `"activity switch blocked: ..."` continue to work. Every
/// other variant routes through `Display`, which itself delegates to the
/// wrapped inner enum's `Display` for outer variants that carry a layout-side
/// `*Error` payload.
///
/// Both `ipc/server.rs` and the `do_action_error_envelope_matches_wire_contract`
/// pin test call this function, so any regression to the format strings fails
/// the test before reaching the wire. Adding a new outer variant with a bare
/// inner-enum `Display` requires no edit here — the catch-all already covers
/// it.
pub(crate) fn format_do_action_error(err: DoActionError) -> String {
    match err {
        DoActionError::ActivitySwitchBlocked(block) => format_activity_switch_block_err(block),
        err => format!("{err}"),
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
enum InteractiveMoveState<W: LayoutElement> {
    /// Initial rubberbanding; the window remains in the layout.
    Starting {
        /// The window we're moving.
        window_id: W::Id,
        /// Current pointer delta from the starting location.
        pointer_delta: Point<f64, Logical>,
        /// Pointer location within the visual window geometry as ratio from geometry size.
        ///
        /// This helps the pointer remain inside the window as it resizes.
        pointer_ratio_within_window: (f64, f64),
    },
    /// Moving; the window is no longer in the layout.
    Moving(InteractiveMoveData<W>),
}

#[derive(Debug)]
struct InteractiveMoveData<W: LayoutElement> {
    /// The window being moved.
    pub(self) tile: Tile<W>,
    /// Output where the window is currently located/rendered.
    pub(self) output: Output,
    /// Current pointer position within output.
    pub(self) pointer_pos_within_output: Point<f64, Logical>,
    /// Window column width.
    pub(self) width: ColumnWidth,
    /// Whether the window column was full-width.
    pub(self) is_full_width: bool,
    /// Whether the window targets the floating layout.
    pub(self) is_floating: bool,
    /// Pointer location within the visual window geometry as ratio from geometry size.
    ///
    /// This helps the pointer remain inside the window as it resizes.
    pub(self) pointer_ratio_within_window: (f64, f64),
    /// Config overrides for the output where the window is currently located.
    ///
    /// Cached here to be accessible while an output is removed.
    pub(self) output_config: Option<jiji_config::LayoutPart>,
    /// Config overrides for the workspace where the window is currently located.
    ///
    /// To avoid sudden window changes when starting an interactive move, it will remember the
    /// config overrides for the workspace where the move originated from. As soon as the window
    /// moves over some different workspace though, this override will reset.
    pub(self) workspace_config: Option<(WorkspaceId, jiji_config::LayoutPart)>,
}

#[derive(Debug)]
pub struct DndData<W: LayoutElement> {
    /// Output where the pointer is currently located.
    output: Output,
    /// Current pointer position within output.
    pointer_pos_within_output: Point<f64, Logical>,
    /// Ongoing DnD hold to activate something.
    hold: Option<DndHold<W>>,
}

#[derive(Debug)]
struct DndHold<W: LayoutElement> {
    /// Time when we started holding on the target.
    start_time: Duration,
    target: DndHoldTarget<W::Id>,
}

#[derive(Debug, PartialEq, Eq)]
enum DndHoldTarget<WindowId> {
    Window(WindowId),
    Workspace(WorkspaceId),
}

#[derive(Debug, Clone, Copy)]
pub struct InteractiveResizeData {
    pub(self) edges: ResizeEdge,
}

#[derive(Debug, Clone, Copy)]
pub enum ConfigureIntent {
    /// A configure is not needed (no changes to server pending state).
    NotNeeded,
    /// A configure is throttled (due to resizing too fast for example).
    Throttled,
    /// Can send the configure if it isn't throttled externally (only size changed).
    CanSend,
    /// Should send the configure regardless of external throttling (something other than size
    /// changed).
    ShouldSend,
}

/// Tile that was just removed from the layout.
pub struct RemovedTile<W: LayoutElement> {
    tile: Tile<W>,
    /// Width of the column the tile was in.
    width: ColumnWidth,
    /// Whether the column the tile was in was full-width.
    is_full_width: bool,
    /// Whether the tile was floating.
    is_floating: bool,
}

/// Outcome of [`Layout::move_window_to_pool_workspace`] for the in-pool
/// resolution path. See that method's rustdoc for the four-arm dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MoveWindowToPoolOutcome {
    /// Target resolves into some monitor's active-activity view. The caller
    /// must fall through to the existing index-based path
    /// ([`Layout::move_to_workspace`] / [`Layout::move_to_output`]) — the
    /// pool entry point intentionally declines to handle the in-view case so
    /// it cannot double-activate.
    DelegateToActiveView,
    /// Cross-activity move completed. The resolved target pool id is
    /// returned so the caller can drive the `focus:true` activation flow
    /// (pick a target activity that contains this workspace; switch into it;
    /// focus the moved window).
    MovedDormant { ws_id: WorkspaceId },
    /// The source workspace had no focused tile to move (e.g. the active slot
    /// is an empty bookend). Caller should treat this as a no-op: no activity
    /// switch, no focus change.
    NothingToMove,
}

/// Failure mode for [`Layout::move_window_to_pool_workspace`]. Surfaces to IPC
/// as [`DoActionError::MoveWindowTargetUnreachable`], replacing the pre-fix
/// silent `Response::Handled` that the CLI had been printing as a false-success
/// line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MoveWindowToPoolError {
    /// Target is not movable-to: either `target_raw_id` does not resolve to
    /// any pool key, or the target workspace's `output_id` does not name a
    /// currently connected monitor (parked in `disconnected_workspace_ids` or
    /// bound to an output that never reconnected this session).
    TargetUnreachable,
}

/// Error returned by [`Layout::resolve_workspace_in_activity`] when the
/// workspace cannot be found within the given activity's scope.
///
/// This is a terminal error: the IPC dispatch sites must forward it
/// immediately rather than parking it on the blocked-waiter queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FocusWorkspaceInActivityError {
    /// No workspace in the pool both matches the reference and belongs to the
    /// given activity. This includes the case where the reference matches no
    /// workspace anywhere in the pool (total miss), as well as the case where
    /// a matching workspace exists but is assigned to a different activity.
    WorkspaceNotInActivity,
    /// Positional index references are not supported for activity-scoped
    /// lookup; use a name or the `id:N` form instead.
    IndexUnsupported,
}

impl fmt::Display for FocusWorkspaceInActivityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WorkspaceNotInActivity => {
                write!(f, "workspace does not belong to the given activity")
            }
            Self::IndexUnsupported => write!(
                f,
                "index references are not supported with --activity; use a name or id:N"
            ),
        }
    }
}

impl std::error::Error for FocusWorkspaceInActivityError {}

/// Whether to activate a newly added window.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum ActivateWindow {
    /// Activate unconditionally.
    Yes,
    /// Activate based on heuristics.
    #[default]
    Smart,
    /// Do not activate.
    No,
}

/// Where to put a newly added window.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum AddWindowTarget<'a, W: LayoutElement> {
    /// No particular preference.
    #[default]
    Auto,
    /// On this output.
    Output(&'a Output),
    /// On this workspace.
    Workspace(WorkspaceId),
    /// Next to this existing window.
    NextTo(&'a W::Id),
}

/// Type of the window hit from `window_under()`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HitType {
    /// The hit is within a window's input region and can be used for sending events to it.
    Input {
        /// Position of the window's buffer.
        win_pos: Point<f64, Logical>,
    },
    /// The hit can activate a window, but it is not in the input region so cannot send events.
    ///
    /// For example, this could be clicking on a tile border outside the window.
    Activate {
        /// Whether the hit was on the tab indicator.
        is_tab_indicator: bool,
    },
}

/// Outcome of [`Layout::resolve_insert_target`]: where a workspace-insertion
/// call should land.
///
/// Not to be confused with [`InsertWorkspace`] (a pointer-position
/// drop-target request) or [`InsertPosition`] (where in a column a tile
/// lands). This enum is purely the *bookend-reuse resolution outcome* for the
/// four `*_workspace_*_on` insertion helpers and the `interactive_move_end`
/// `NewAt` arm — it carries no rendering or pointer information.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BookendResolution {
    /// Reuse the top empty workspace (index 0); the caller performs no insert.
    ReuseTop,
    /// Reuse the trailing bookend workspace (`view_len - 1`); the caller
    /// performs no insert.
    ReuseTrailing,
    /// No bookend applies; insert a fresh workspace at this index.
    InsertAt(usize),
}

#[derive(Debug)]
enum OverviewProgress {
    Animation(Animation),
    Gesture(OverviewGesture),
    Open,
}

#[derive(Debug)]
struct OverviewGesture {
    tracker: SwipeTracker,
    /// Start point.
    start: f64,
    /// Current progress.
    value: f64,
}

impl SizingMode {
    #[must_use]
    pub fn is_normal(&self) -> bool {
        matches!(self, Self::Normal)
    }

    #[must_use]
    pub fn is_fullscreen(&self) -> bool {
        matches!(self, Self::Fullscreen)
    }

    #[must_use]
    pub fn is_maximized(&self) -> bool {
        matches!(self, Self::Maximized)
    }
}

impl<W: LayoutElement> InteractiveMoveState<W> {
    fn moving(&self) -> Option<&InteractiveMoveData<W>> {
        match self {
            InteractiveMoveState::Moving(move_) => Some(move_),
            _ => None,
        }
    }

    fn moving_mut(&mut self) -> Option<&mut InteractiveMoveData<W>> {
        match self {
            InteractiveMoveState::Moving(move_) => Some(move_),
            _ => None,
        }
    }
}

impl<W: LayoutElement> InteractiveMoveData<W> {
    fn tile_render_location(&self, zoom: f64) -> Point<f64, Logical> {
        let scale = Scale::from(self.output.current_scale().fractional_scale());
        let window_size = self.tile.window_size();
        let pointer_offset_within_window = Point::from((
            window_size.w * self.pointer_ratio_within_window.0,
            window_size.h * self.pointer_ratio_within_window.1,
        ));
        let pos = self.pointer_pos_within_output
            - (pointer_offset_within_window + self.tile.window_loc() - self.tile.render_offset())
                .upscale(zoom);
        // Round to physical pixels.
        pos.to_physical_precise_round(scale).to_logical(scale)
    }
}

impl ActivateWindow {
    pub fn map_smart(self, f: impl FnOnce() -> bool) -> bool {
        match self {
            ActivateWindow::Yes => true,
            ActivateWindow::Smart => f(),
            ActivateWindow::No => false,
        }
    }
}

impl HitType {
    pub fn offset_win_pos(mut self, offset: Point<f64, Logical>) -> Self {
        match &mut self {
            HitType::Input { win_pos } => *win_pos += offset,
            HitType::Activate { .. } => (),
        }
        self
    }

    pub fn hit_tile<W: LayoutElement>(
        tile: &Tile<W>,
        tile_pos: Point<f64, Logical>,
        point: Point<f64, Logical>,
    ) -> Option<(&W, Self)> {
        let pos_within_tile = point - tile_pos;
        tile.hit(pos_within_tile)
            .map(|hit| (tile.window(), hit.offset_win_pos(tile_pos)))
    }

    pub fn to_activate(self) -> Self {
        match self {
            HitType::Input { .. } => HitType::Activate {
                is_tab_indicator: false,
            },
            HitType::Activate { .. } => self,
        }
    }
}

impl Options {
    fn from_config(config: &Config) -> Self {
        Self {
            layout: config.layout.clone(),
            animations: config.animations.clone(),
            gestures: config.gestures,
            overview: config.overview,
            blur: config.blur,
            bookmarks: config.bookmarks.clone(),
            disable_resize_throttling: config.debug.disable_resize_throttling,
            disable_transactions: config.debug.disable_transactions,
            deactivate_unfocused_windows: config.debug.deactivate_unfocused_windows,
        }
    }

    fn with_merged_layout(mut self, part: Option<&jiji_config::LayoutPart>) -> Self {
        if let Some(part) = part {
            self.layout.merge_with(part);
        }
        self
    }

    fn adjusted_for_scale(mut self, scale: f64) -> Self {
        self.layout.gaps = round_logical_in_physical_max1(scale, self.layout.gaps);
        self
    }
}

impl OverviewProgress {
    fn value(&self) -> f64 {
        match self {
            OverviewProgress::Animation(anim) => anim.value(),
            OverviewProgress::Gesture(gesture) => gesture.value,
            OverviewProgress::Open => 1.,
        }
    }

    fn is_animation(&self) -> bool {
        matches!(self, OverviewProgress::Animation(_))
    }
}

/// Name of the runtime "Default" activity seeded at startup. Referenced in
/// `with_options` and `with_options_and_workspaces`; extracted as a constant
/// so a single grep finds all uses.
const DEFAULT_ACTIVITY_NAME: &str = "Default";

/// Resolve the activity-membership set for a config-declared workspace against
/// the given `Activities` pool.
///
/// Precedence ( auto-expansion):
/// 1. If `sticky` is `Some(true)`, the workspace is auto-tagged with every activity id in the pool.
///    Sticky beats any explicit `activity "..."` list.
/// 2. Else, if the config has no `activity "..."` entries, the workspace is stamped with exactly
///    the currently-active activity id.
/// 3. Else, each entry in `ws_config.activities` is resolved case-insensitively via
///    [`Activities::resolve_config_names`]. Unknown names produce a `warn!`; if every entry is
///    unknown, a second `warn!` notes that the workspace falls back to `{active_id}` so the
///    non-empty invariant required by `Workspace::new*` is preserved.
///
/// This is a free function (not an `&mut self` method) so it can be called
/// during `Layout::with_options_and_workspaces`, where `self` does not yet
/// exist. The associated method [`Layout::resolve_workspace_activities`]
/// delegates here for callers that do have `&self`.
fn resolve_workspace_activities_for(
    activities: &Activities,
    ws_config: &WorkspaceConfig,
) -> HashSet<ActivityId> {
    let is_sticky = ws_config.sticky.unwrap_or(false);
    if is_sticky {
        return activities.iter().map(|a| a.id()).collect();
    }

    if ws_config.activities.is_empty() {
        return HashSet::from([activities.active_id()]);
    }

    let (resolved, unknown) = activities.resolve_config_names(&ws_config.activities);
    for name in &unknown {
        warn!(
            "workspace {:?}: unknown activity {:?} (no matching top-level `activity` block)",
            ws_config.name.0, name,
        );
    }
    if resolved.is_empty() {
        warn!(
            "workspace {:?}: every declared activity name was unknown; falling back to \
             the currently-active activity",
            ws_config.name.0,
        );
        return HashSet::from([activities.active_id()]);
    }
    resolved
}

impl<W: LayoutElement> Layout<W> {
    pub fn new(clock: Clock, config: &Config) -> Self {
        Self::with_options_and_workspaces(clock, config, Options::from_config(config))
    }

    pub fn with_options(clock: Clock, options: Options) -> Self {
        Self {
            monitors: Vec::new(),
            primary_idx: 0,
            active_monitor_idx: 0,
            disconnected_workspace_ids: Vec::new(),
            workspaces: HashMap::new(),
            activities: Activities::new(Activity::new_runtime(DEFAULT_ACTIVITY_NAME.to_owned())),
            bookmarks: Bookmarks::default(),
            is_active: true,
            last_active_workspace_id: HashMap::new(),
            interactive_move: None,
            dnd: None,
            clock,
            update_render_elements_time: Duration::ZERO,
            overview_open: false,
            overview_progress: None,
            options: Rc::new(options),
        }
    }

    fn with_options_and_workspaces(clock: Clock, config: &Config, options: Options) -> Self {
        let opts = Rc::new(options);

        let activities = Activities::from_config_or_default(&config.activities);

        let mut workspaces: HashMap<WorkspaceId, Workspace<W>> = HashMap::new();
        let workspace_ids = config
            .workspaces
            .iter()
            .map(|ws| {
                let ws_activities = resolve_workspace_activities_for(&activities, ws);
                let workspace = Workspace::new_with_config_no_outputs(
                    Some(ws.clone()),
                    ws_activities,
                    clock.clone(),
                    opts.clone(),
                );
                let id = workspace.id();
                assert!(
                    workspaces.insert(id, workspace).is_none(),
                    "fresh id must be unique"
                );
                id
            })
            .collect();

        Self {
            monitors: Vec::new(),
            primary_idx: 0,
            active_monitor_idx: 0,
            disconnected_workspace_ids: workspace_ids,
            workspaces,
            activities,
            bookmarks: Bookmarks::default(),
            is_active: true,
            last_active_workspace_id: HashMap::new(),
            interactive_move: None,
            dnd: None,
            clock,
            update_render_elements_time: Duration::ZERO,
            overview_open: false,
            overview_progress: None,
            options: opts,
        }
    }

    pub fn add_output(&mut self, output: Output, layout_config: Option<LayoutPart>) {
        let seed_activity = self.activities.active_id();
        if self.monitors.is_empty() {
            // Reconnecting from a fully-disconnected state: partition the parked workspaces by
            // membership in the seed-active activity. Members populate the seed monitor's view
            // (in their saved order); non-members are bound to the output and routed into their
            // own activities' views by the materializer's real-tag lift and the membership
            // residue pass below — a dormant-declared workspace boots into its activity's view
            // rather than being adopted wholesale by the active view.
            let drained = mem::take(&mut self.disconnected_workspace_ids);
            let active_id = seed_activity;
            let (members, non_members): (Vec<WorkspaceId>, Vec<WorkspaceId>) =
                drained.iter().copied().partition(|id| {
                    self.workspaces
                        .get(id)
                        .expect("parked id must be a live pool key")
                        .activities()
                        .contains(&active_id)
                });

            // Pass the remembered active workspace unfiltered: Monitor::new's list-match makes
            // member-filtering implicit — a non-member remembered id matches no member, so the
            // seed view falls back to its default activation (first member, or the bookend when
            // there are no members).
            let remembered = self.last_active_workspace_id.remove(&output.name());
            let output_id = OutputId::new(&output);

            let (mut monitor, view) = Monitor::new(
                output,
                members,
                remembered,
                &mut self.workspaces,
                self.clock.clone(),
                self.options.clone(),
                layout_config,
                seed_activity,
            );
            monitor.overview_open = self.overview_open;
            monitor.set_overview_progress(&view, self.overview_progress.as_ref());

            // Insert view before pushing the monitor to keep the domain-parity invariant tight.
            assert!(
                self.activities
                    .active_mut()
                    .views_mut()
                    .insert(output_id, view)
                    .is_none(),
                "output must not already have a view in the active activity",
            );
            self.monitors.push(monitor);
            self.primary_idx = 0;
            self.active_monitor_idx = 0;

            // Bind the non-member parked workspaces to the new output before the materializer
            // runs, mirroring Monitor::new's member loop. This preserves today's window
            // output_enter semantics (a dormant-view workspace on a connected output is bound —
            // switch_activity never binds/unbinds) and refreshes real connector-form tags to the
            // make/model/serial form so the materializer's real-tag lift filter can match by `==`.
            let mon_output = self.monitors[0].output.clone();
            let mon_options = self.monitors[0].options.clone();
            for id in &non_members {
                let ws = self
                    .workspaces
                    .get_mut(id)
                    .expect("parked id must be a live pool key");
                ws.bind_output(&mon_output);
                ws.update_config(mon_options.clone());
            }

            // Materialize a view for every activity on the new output. The lift branch pulls
            // real-tagged workspaces (members and non-members alike) into their member activities'
            // views; the source-side dedup is per-activity, so cross-activity sharing survives.
            self.ensure_all_activity_views();

            // Residue install: any drained id the materializer could not lift (sentinel-tagged, or
            // tagged for a different output, or shared into a member whose view the lift skipped)
            // is installed into every member activity's boot-output view by membership.
            self.install_drained_by_membership(&drained, 0);

            // Reseat: a residue install lands above a fresh trailing bookend without moving the
            // view's id-keyed active cursor, leaving the bookend active. Reseat every non-active
            // boot-output view whose active is an empty bookend onto its first real workspace so
            // the two install mechanisms (lift vs. residue) do not diverge by tag shape.
            self.reseat_bookend_active_boot(0, active_id);

            // Remembered-nonmember fallback: when the remembered active workspace is a live
            // non-member, Monitor::new activated the seed default instead. Reseat each of its
            // member activities' boot-output views onto it so reconnect restores the workspace the
            // user last had focused, even though it belongs to a dormant activity now.
            match remembered {
                Some(r) if !self.workspaces.contains_key(&r) => {
                    trace!(
                        "add_output: remembered active workspace {r:?} is no longer a live pool \
                         key; no remembered-nonmember fallback",
                    );
                }
                Some(r) => {
                    let is_nonmember = self
                        .workspaces
                        .get(&r)
                        .is_some_and(|ws| !ws.activities().contains(&active_id));
                    if is_nonmember {
                        let boot_out_id = self.monitors[0].output_id();
                        let member_acts: Vec<ActivityId> = {
                            let r_acts = self
                                .workspaces
                                .get(&r)
                                .expect("checked live above")
                                .activities()
                                .clone();
                            self.activities
                                .iter()
                                .filter(|a| r_acts.contains(&a.id()))
                                .map(|a| a.id())
                                .collect()
                        };
                        for act_id in member_acts {
                            let view = self
                                .activities
                                .get_mut(act_id)
                                .expect("act_id sourced from activities.iter()")
                                .views_mut()
                                .get_mut(&boot_out_id)
                                .expect("member activity holds a boot-output view");
                            let pos = view.position_of(r);
                            debug_assert!(
                                pos.is_some(),
                                "remembered non-member fallback: {r:?} must be present in member \
                                 activity {act_id:?}'s boot-output view — the residue pass \
                                 guarantees presence",
                            );
                            if let Some(pos) = pos {
                                view.set_active_at(pos);
                            }
                        }
                    }
                    // A remembered live member is already activated by the seed view — no action.
                }
                None => {}
            }

            return;
        }

        // Empty bookends peeled off the primary view are flushed through
        // `destroy_workspaces_cross_activity` after the split-borrow scope closes
        // and the new monitor's view is installed in the active activity.
        let mut doomed_ids: Vec<WorkspaceId> = Vec::new();

        let primary_idx = self.primary_idx;
        let primary_output_id = self.monitors[primary_idx].output_id();
        let (monitors, pool, primary_view) = self.monitors_pool_view_mut(&primary_output_id);
        let primary = &mut monitors[primary_idx];

        let mut stopped_primary_ws_switch = false;

        let mut workspace_ids: Vec<WorkspaceId> = vec![];
        for i in (0..primary_view.len()).rev() {
            let id = primary_view.ids()[i];
            let matches = pool
                .get(&id)
                .expect("view id must be a key in the pool")
                .output_id
                .as_ref()
                .is_some_and(|oid| oid.matches(&output));
            if matches {
                let ws = pool
                    .get_mut(&id)
                    .expect("workspace id must be a key in the pool");
                // Workspace was bound to primary's output while its designated output
                // was disconnected; unbind before handing it to the new monitor so that
                // subsequent bind_output doesn't leave windows marked as present on
                // both outputs.
                ws.unbind_output(&primary.output);

                // FIXME: this can be coded in a way that the workspace switch won't be
                // affected if the removed workspace is invisible. But this is good enough
                // for now.
                if primary.workspace_switch.is_some() {
                    primary.workspace_switch = None;
                    stopped_primary_ws_switch = true;
                }

                // The user could've closed a window while remaining on this workspace, on
                // another monitor. However, we will add an empty workspace in the end
                // instead.
                if ws.has_windows_or_name() {
                    workspace_ids.push(id);
                } else {
                    // Empty unnamed workspaces don't come along — accumulate for a
                    // cross-activity destroy flush after the scope closes.
                    doomed_ids.push(id);
                }

                // Without this exception, the first monitor to connect can end up
                // with the first empty workspace focused instead of the first named
                // workspace (under `empty_workspace_above_first`, `remove_at`'s
                // default shift would land focus on the forced-empty first
                // workspace).
                let active_pos_before = primary_view.active_position();
                let keep_active_pinned = primary.options.layout.empty_workspace_above_first
                    && active_pos_before == 1
                    && i <= active_pos_before;

                primary_view.remove_at(i);

                if keep_active_pinned {
                    let new_pos = 1.min(primary_view.len() - 1);
                    primary_view.set_active_at(new_pos);
                }
            }
        }

        // If we stopped a workspace switch, then we might need to clean up workspaces.
        // Also if empty_workspace_above_first is set and there are only 2 workspaces left,
        // both will be empty and one of them needs to be removed. clean_up_workspaces
        // takes care of this.

        let needs_cleanup = stopped_primary_ws_switch
            || (primary.options.layout.empty_workspace_above_first && primary_view.len() == 2);
        if needs_cleanup {
            let ids_to_destroy = {
                let (monitors, pool, view) = self.monitors_pool_view_mut(&primary_output_id);
                Self::clean_up_workspaces_on(monitors, pool, view, primary_idx)
            };
            Self::destroy_workspaces_cross_activity(
                &mut self.activities,
                &mut self.workspaces,
                ids_to_destroy,
            );
        }

        workspace_ids.reverse();

        let ws_id_to_activate = self.last_active_workspace_id.remove(&output.name());
        let output_id = OutputId::new(&output);

        let (mut monitor, view) = Monitor::new(
            output,
            workspace_ids,
            ws_id_to_activate,
            &mut self.workspaces,
            self.clock.clone(),
            self.options.clone(),
            layout_config,
            seed_activity,
        );
        monitor.overview_open = self.overview_open;
        monitor.set_overview_progress(&view, self.overview_progress.as_ref());
        // Insert view before pushing the monitor to keep the domain-parity invariant tight.
        assert!(
            self.activities
                .active_mut()
                .views_mut()
                .insert(output_id, view)
                .is_none(),
            "output must not already have a view in the active activity",
        );
        self.monitors.push(monitor);

        // Flush doomed empty bookends now that the new monitor's view is installed in
        // the active activity: views ↔ pool ↔ monitors parity is intact so
        // `verify_invariants` would pass at this suspend point, and the helper can
        // patch any other activities' views that still reference these ids.
        Self::destroy_workspaces_cross_activity(
            &mut self.activities,
            &mut self.workspaces,
            doomed_ids,
        );

        // Materialize bookend views for every dormant activity on the newly-connected output —
        // the per-activity bookend invariant requires every activity to hold a view for every
        // connected monitor.
        //
        // On partial reconnect, dormant activities' views for `output_id` are materialized
        // here via `ensure_all_activity_views` → `ensure_view_for`. The lift branch lifts
        // every workspace in the pool whose `output_id()` matches and whose `activities()`
        // contains the materializing activity — this reclaims workspaces that were
        // system-migrated to primary on the partial disconnect (see `remove_output`'s
        // dormant walk for the migration site). No new code path is required; the existing
        // materializer call is now load-bearing for both first-monitor bootstrap and
        // dormant-reclaim.
        self.ensure_all_activity_views();
    }

    pub fn remove_output(&mut self, output: &Output) {
        assert!(
            !self.monitors.is_empty(),
            "tried to remove output when there were already none",
        );

        let idx = self
            .monitors
            .iter()
            .position(|mon| &mon.output == output)
            .expect("trying to remove non-existing output");
        let output_id = OutputId::new(output);

        // Evict the view from the active activity BEFORE removing the monitor: both
        // take_workspace_ids and last_active_workspace_id need both values, and the
        // domain-parity invariant (views.len() == monitors.len()) must hold at every suspend point.
        let view = self
            .activities
            .active_mut()
            .views_mut()
            .remove(&output_id)
            .expect("connected output must have a view in the active activity");

        let monitor = self.monitors.remove(idx);

        self.last_active_workspace_id
            .insert(monitor.output_name().clone(), view.active());

        let (mut workspace_ids, doomed_ids) = self.take_workspace_ids(&monitor, &view);

        if self.monitors.is_empty() {
            // The active activity's `views` map must be empty when no monitors are connected;
            // dormant activities may retain views keyed by previously-disconnected outputs
            // (validated by the disconnected-output bookend pass). The active activity's view for
            // `output_id` was already evicted above and its workspaces flow into
            // `workspace_ids` via `take_workspace_ids`. Under the new posture every other
            // activity also holds a view for the now-removed output: walk those, classify each
            // dormant view id into either "kept" (named or has windows — joins
            // `workspace_ids` to be parked in the disconnected pool) or "doomed" (empty
            // unnamed — joins `doomed_ids` to be destroyed by
            // `destroy_workspaces_cross_activity`), then clear the activity view. The
            // disconnected pool may not hold empty unnamed workspaces (asserted at
            // `verify_invariants`), so the classification mirrors `take_workspace_ids`.
            let mut doomed_ids = doomed_ids;
            let doomed_set: HashSet<WorkspaceId> = doomed_ids.iter().copied().collect();
            let mut already_kept: HashSet<WorkspaceId> = workspace_ids.iter().copied().collect();
            let mut already_doomed: HashSet<WorkspaceId> = doomed_set;
            let to_visit: Vec<(ActivityId, Vec<WorkspaceId>)> = self
                .activities
                .iter_mut()
                .filter_map(|activity| {
                    activity
                        .views_mut()
                        .remove(&output_id)
                        .map(|v| (activity.id(), v.ids().to_vec()))
                })
                .collect();
            // Full-disconnect destination is a flat disconnected pool, not per-activity views,
            // so activity attribution is not needed — diverges from the partial-disconnect walk
            // below which carries act_id as part of its (act_id, ws_id) migration pairs.
            for (_act_id, ids) in to_visit {
                for ws_id in ids {
                    if already_kept.contains(&ws_id) || already_doomed.contains(&ws_id) {
                        continue;
                    }
                    let ws = self
                        .workspaces
                        .get_mut(&ws_id)
                        .expect("dormant view id must be a live pool key");
                    // A named or windowed workspace whose output_id is the empty-string sentinel
                    // (produced by `new_with_config_no_outputs` without an explicit
                    // `open_on_output`) is parked like any other kept dormant id — it is legal in
                    // the disconnected pool. `bind_output` never latches the sentinel
                    // (`OutputName::matches("")` is false for every real connector), so a parked
                    // sentinel's `output_id` stays a pure reconnect-routing hint that the
                    // first-monitor drain re-installs by activity membership, not by tag. Empty
                    // unnamed sentinels still doom via the generic `!has_windows_or_name()` arm
                    // below.
                    if ws.has_windows_or_name() {
                        // Fire output_leave for all windows on this workspace before it joins
                        // the disconnected pool; mirrors the unbind done for active-view kept ids
                        // in take_workspace_ids.
                        ws.unbind_output(&monitor.output);
                        workspace_ids.push(ws_id);
                        already_kept.insert(ws_id);
                    } else {
                        doomed_ids.push(ws_id);
                        already_doomed.insert(ws_id);
                    }
                }
            }

            // Reset options on every parked id (active-view ids plus the just-merged dormant
            // ones) — layout-root options replace per-monitor merged ones for the disconnected
            // lifetime.
            for id in &workspace_ids {
                self.workspaces
                    .get_mut(id)
                    .expect("workspace id must be a key in the pool")
                    .update_config(self.options.clone());
            }

            self.disconnected_workspace_ids = workspace_ids;
            self.primary_idx = 0;
            self.active_monitor_idx = 0;
            Self::destroy_workspaces_cross_activity(
                &mut self.activities,
                &mut self.workspaces,
                doomed_ids,
            );
            return;
        }

        if self.primary_idx >= idx {
            // Update primary_idx to either still point at the same monitor, or at some other
            // monitor if the primary has been removed.
            self.primary_idx = self.primary_idx.saturating_sub(1);
        }
        if self.active_monitor_idx >= idx {
            // Update active_monitor_idx to either still point at the same monitor, or at some
            // other monitor if the active monitor has been removed.
            self.active_monitor_idx = self.active_monitor_idx.saturating_sub(1);
        }

        let primary_idx = self.primary_idx;
        self.append_workspaces_to_monitor(primary_idx, workspace_ids);

        // Partial-disconnect dormant walk: drain every dormant activity's view for the
        // disconnecting output and migrate its named / window-bearing workspaces into the
        // dormant activity's existing view for the primary monitor. Mirrors the
        // full-disconnect walk in the `monitors.is_empty()` branch above, with two
        // differences: the named / windowed survivors land in a dormant view on primary
        // (not in the disconnected pool), and the unbind/bind step retargets their
        // Smithay `output_enter` markers from the disconnecting output to primary. The
        // workspace's own `output_id` field is left pointing at the disconnecting output
        // (`Workspace::bind_output` only refreshes `output_id` when it already matches
        // the bound output — see workspace.rs:602-607), so a future `add_output` for
        // the same monitor reclaims them via `ensure_view_for`'s pool-tag lift branch.
        let primary_output = self.monitors[primary_idx].output.clone();
        let primary_output_id = OutputId::new(&primary_output);
        let primary_options = self.monitors[primary_idx].options.clone();

        let mut doomed_ids = doomed_ids;
        // `already_doomed` deduplicates doom pushes across activities: a workspace shared by
        // multiple dormant activities (e.g. `ws.activities = {A, B}`) appears in every activity's
        // drained view; doom must be pushed exactly once because
        // `destroy_workspaces_cross_activity` asserts the pool entry is present on each id it
        // processes — a duplicate would cause a double-remove panic.
        let mut already_doomed: HashSet<WorkspaceId> = doomed_ids.iter().copied().collect();
        let mut migrate_to_primary: Vec<(ActivityId, WorkspaceId)> = Vec::new();

        // Collect under `&mut self.activities` only; the `&mut self.workspaces` and
        // `&mut self.activities` reborrows in the migration phase below need this borrow
        // released first. The active activity's view for `output_id` was already removed
        // via the `views_mut().remove(&output_id)` eviction on `activities.active_mut()`
        // near the top of `remove_output` (before the `monitors.is_empty()` branch), so
        // the `filter_map` here naturally yields no entry for the active activity — no
        // defensive `active_id`-skip required.
        let to_visit: Vec<(ActivityId, Vec<WorkspaceId>)> = self
            .activities
            .iter_mut()
            .filter_map(|activity| {
                activity
                    .views_mut()
                    .remove(&output_id)
                    .map(|v| (activity.id(), v.ids().to_vec()))
            })
            .collect();

        for (act_id, ids) in to_visit {
            for ws_id in ids {
                if already_doomed.contains(&ws_id) {
                    continue;
                }
                let ws = self
                    .workspaces
                    .get(&ws_id)
                    .expect("dormant view id must be a live pool key");
                // A named / windowed workspace whose output_id is the empty-string sentinel
                // migrates to primary like any other kept dormant id (mirrors the full-disconnect
                // walk). `bind_output(&primary)` below never re-tags a non-matching id, so the
                // sentinel survives as a pure reconnect-routing hint; the migration assert accepts
                // it. Empty unnamed workspaces still doom.
                if !ws.has_windows_or_name() {
                    doomed_ids.push(ws_id);
                    already_doomed.insert(ws_id);
                    continue;
                }
                // Each (act_id, ws_id) pair gets its own migrate entry — dormant activities
                // have independent destination views, so a workspace shared across activities A
                // and B must migrate into *both* A.views[primary] and B.views[primary].
                migrate_to_primary.push((act_id, ws_id));
            }
        }

        // Mutation phase — classification is complete, so we can reborrow workspaces and
        // activities separately on each iteration without conflicting with the
        // already-released `iter_mut` from the collect step.
        for (act_id, ws_id) in migrate_to_primary {
            let ws = self
                .workspaces
                .get_mut(&ws_id)
                .expect("workspace id must be a key in the pool");
            // Binding, not the tag, is what migration relies on: residency in a view keyed by the
            // disconnecting output implies the workspace is bound to it (every view-install path
            // binds), so the unbind/bind pair below is correct regardless of `output_id`. The tag
            // is a pure reconnect-routing hint and legitimately takes any `Some` shape here: the
            // disconnecting output's id (homed here), the empty-string sentinel (never pinned), or
            // a foreign connector (pinned to a disconnected/other output, displayed here by the
            // membership-routed drain). `None` alone is incoherent — every pool workspace is
            // constructed with a hint or has it patched at mint time.
            debug_assert!(
                ws.output_id().is_some(),
                "dormant view migration: view-resident workspace {ws_id:?} carries no output_id \
                 routing hint at all — hint must be a real output id or the sentinel",
            );
            ws.unbind_output(&monitor.output);
            ws.bind_output(&primary_output);
            ws.update_config(primary_options.clone());

            let primary_view = self
                .activities
                .get_mut(act_id)
                .expect("act_id sourced from filter_map; activity must still be live")
                .views_mut()
                .get_mut(&primary_output_id)
                .expect(
                    "dormant activity must hold a view for primary — \
                     ensure_all_activity_views materialized it on primary's add_output",
                );
            let insert_pos = primary_view.len() - 1;
            debug_assert!(
                !primary_view.ids().contains(&ws_id),
                "dormant view migration must not insert a duplicate id into this activity's \
                 primary view — each (activity, workspace) pair reaches this insert at most \
                 once: the source view was drained by the filter_map collect above, so the \
                 same ws_id cannot recur within act_id's iter, and different act_ids write \
                 to different views[primary_output_id] entries",
            );
            primary_view.insert(insert_pos, ws_id);
        }

        Self::destroy_workspaces_cross_activity(
            &mut self.activities,
            &mut self.workspaces,
            doomed_ids,
        );

        // After cross-activity destroy may have dropped single-entry dormant views, the
        // per-activity bookend invariant could be missing a view for one of the still-connected
        // monitors. Re-run the materializer so every remaining activity holds a bookend view
        // for every connected monitor.
        self.ensure_all_activity_views();
    }

    pub fn add_column_by_idx(
        &mut self,
        monitor_idx: usize,
        workspace_idx: usize,
        column: Column<W>,
        activate: bool,
    ) {
        assert!(
            !self.monitors.is_empty(),
            "add_column_by_idx requires at least one connected monitor",
        );
        let mon_out = self.monitors[monitor_idx].output_id();
        let (monitors, pool, view) = self.monitors_pool_view_mut(&mon_out);
        Self::add_column_on(
            monitors,
            pool,
            view,
            monitor_idx,
            workspace_idx,
            column,
            activate,
        );

        if activate {
            self.active_monitor_idx = monitor_idx;
        }

        // End of split borrow. `add_column_on` no longer mints inline; the sweep maintains
        // the active view's bookends along with every dormant activity's view at the same
        // output.
        self.normalize_view_bookends();
    }

    /// Repairs a violated trailing (or, under EWAF, leading) bookend in every activity's view of
    /// every connected monitor's output, minting and inserting a fresh empty workspace at each
    /// violated slot.
    ///
    /// The state-based counterpart of `assert_view_bookends`: where that assertion panics on a
    /// bookend slot occupied by a windowed-or-named entry, this sweep repairs it — the repair
    /// predicate mirrors the negation of that assertion's two trailing/leading presence clauses,
    /// so a view is touched here if and only if one of those two clauses would trip. It does not
    /// enforce the EWAF length-2 rule (a len-2 EWAF view is illegal unless its second entry is a
    /// shared workspace pinning the trailing slot) — that shape needs a workspace *removed*, not
    /// minted, and stays with the subtractive collapse/reclaim helpers. Idempotent: each repair
    /// leaves the slot it just touched trivially satisfying the presence clause that flagged it,
    /// so an immediate second call structurally returns 0 regardless of caller; `verify_invariants`
    /// running after every public entry point additionally confirms no violation survives a
    /// completed call. Additive-only — never removes, renames, or reassigns a workspace — and
    /// scoped to views keyed by currently-connected monitor outputs; by the connected-keyspace
    /// invariant (every activity's `views` map is purged of a disconnecting output's entry —
    /// see `remove_output`, enforced by `Layout::verify_invariants`), there is no
    /// disconnected-output keyspace left to visit.
    ///
    /// Scope: examines every activity's view — active included — independent of which
    /// workspace a caller just touched, rather than patching a single caller-designated
    /// `ws_id` within the dormant activities sharing it.
    ///
    /// Returns the number of workspaces minted.
    fn normalize_view_bookends(&mut self) -> usize {
        let mut minted = 0;
        let active_id = self.activities.active_id();

        let known_monitors: Vec<(usize, Output, Rc<Options>, OutputId, bool)> = self
            .monitors
            .iter()
            .enumerate()
            .map(|(mon_idx, mon)| {
                let ewaf = mon.options.layout.empty_workspace_above_first;
                (
                    mon_idx,
                    mon.output.clone(),
                    mon.options.clone(),
                    mon.output_id(),
                    ewaf,
                )
            })
            .collect();

        for (mon_idx, mon_output, mon_options, mon_out_id, ewaf) in known_monitors {
            let clock = self.clock.clone();

            // Scan every activity's view of this output for a violated bookend — the negation of
            // the two presence clauses `assert_view_bookends` checks — before mutating anything,
            // mirroring the classify-then-mutate split in `reseat_bookend_active_boot`.
            let mut needs_repair: Vec<(ActivityId, bool, bool)> = Vec::new();
            for a in self.activities.iter() {
                let Some(view) = a.views().get(&mon_out_id) else {
                    debug_assert!(
                        false,
                        "normalize_view_bookends: activity {:?} has no view for connected \
                         output {mon_out_id:?} — per-activity bookend invariant violated \
                         (membership↔view coherence bug)",
                        a.id(),
                    );
                    continue;
                };
                let last_id = *view.ids().last().expect("view non-empty by construction");
                let first_id = *view.ids().first().expect("view non-empty by construction");
                let needs_trailing = self
                    .workspaces
                    .get(&last_id)
                    .expect("view id must be a key in the pool")
                    .has_windows_or_name();
                let needs_leading = ewaf
                    && self
                        .workspaces
                        .get(&first_id)
                        .expect("view id must be a key in the pool")
                        .has_windows_or_name();
                if needs_trailing || needs_leading {
                    needs_repair.push((a.id(), needs_trailing, needs_leading));
                }
            }

            for (act_id, needs_trailing, needs_leading) in needs_repair {
                if needs_trailing {
                    let ws = Workspace::new(
                        &mon_output,
                        HashSet::from([act_id]),
                        clock.clone(),
                        mon_options.clone(),
                    );
                    let id = ws.id();
                    assert!(
                        self.workspaces.insert(id, ws).is_none(),
                        "fresh id must be unique",
                    );
                    let view = self
                        .activities
                        .get_mut(act_id)
                        .expect("act_id sourced from activities.iter()")
                        .views_mut()
                        .get_mut(&mon_out_id)
                        .expect("view existed at scan time");
                    let idx = view.len();
                    view.insert(idx, id);
                    minted += 1;

                    if act_id == active_id {
                        if let Some(switch) = &mut self.monitors[mon_idx].workspace_switch {
                            // idx is the pre-mutation view's length, always beyond target_idx's
                            // valid range, so this guard is structurally a no-op on the trailing
                            // arm — kept verbatim for symmetry with the leading arm below and
                            // fidelity to `add_workspace_at_on`'s mint site.
                            if idx as f64 <= switch.target_idx() {
                                switch.offset(1);
                            }
                        }
                    }
                }

                if needs_leading {
                    let ws = Workspace::new(
                        &mon_output,
                        HashSet::from([act_id]),
                        clock.clone(),
                        mon_options.clone(),
                    );
                    let id = ws.id();
                    assert!(
                        self.workspaces.insert(id, ws).is_none(),
                        "fresh id must be unique",
                    );
                    let view = self
                        .activities
                        .get_mut(act_id)
                        .expect("act_id sourced from activities.iter()")
                        .views_mut()
                        .get_mut(&mon_out_id)
                        .expect("view existed at scan time");
                    let idx = 0;
                    view.insert(idx, id);
                    minted += 1;

                    if act_id == active_id {
                        if let Some(switch) = &mut self.monitors[mon_idx].workspace_switch {
                            if idx as f64 <= switch.target_idx() {
                                switch.offset(1);
                            }
                        }
                    }
                }
            }
        }

        minted
    }

    /// Install every drained workspace into each member activity's boot-output view that does not
    /// already hold it, then mint the bookends those inserts require.
    ///
    /// Runs after the first-monitor materializer to cover the drained ids the materializer's
    /// real-tag lift cannot reach: sentinel- or wrong-output-tagged non-members, and the
    /// shared-member gap where a workspace was lifted into one member's view but not another's.
    /// Iterates activities in declaration order and tests membership, so a dead membership id on a
    /// workspace is naturally ignored. A member activity lacking a boot-output view after the
    /// materializer is a structurally-impossible miss (the per-activity bookend invariant
    /// guarantees the view) — `debug_assert!` + skip, mirroring the discipline in
    /// [`Self::reconcile_views_with_membership`], never a silent `continue`.
    ///
    /// Called outside any `(monitors, pool, view)` split-borrow scope so `&mut self` is available
    /// for the trailing/EWAF-leading bookend sweep, like the post-split flushes in
    /// [`Self::add_output`] / [`Self::remove_output`].
    fn install_drained_by_membership(&mut self, drained: &[WorkspaceId], mon_idx: usize) {
        let boot_out_id = self.monitors[mon_idx].output_id();
        for &ws_id in drained {
            let ws_acts = match self.workspaces.get(&ws_id) {
                Some(ws) => ws.activities().clone(),
                None => {
                    debug_assert!(
                        false,
                        "install_drained_by_membership: drained id {ws_id:?} is not a live pool \
                         key",
                    );
                    continue;
                }
            };
            let member_acts: Vec<ActivityId> = self
                .activities
                .iter()
                .filter(|a| ws_acts.contains(&a.id()))
                .map(|a| a.id())
                .collect();
            let mut installed_any = false;
            for act_id in member_acts {
                let pool = &mut self.workspaces;
                let activities = &mut self.activities;
                let activity = activities
                    .get_mut(act_id)
                    .expect("act_id sourced from activities.iter()");
                let Some(view) = activity.views_mut().get_mut(&boot_out_id) else {
                    debug_assert!(
                        false,
                        "install_drained_by_membership: member activity {act_id:?} has no view \
                         for the boot output {boot_out_id:?} after the materializer — \
                         per-activity bookend invariant violated (membership↔view coherence bug)",
                    );
                    continue;
                };
                if view.position_of(ws_id).is_some() {
                    continue;
                }
                Self::view_insert_above_trailing_bookend(pool, view, ws_id);
                installed_any = true;
            }
            if installed_any {
                self.normalize_view_bookends();
            }
        }
    }

    /// Reseat every non-active boot-output view whose active id resolves to an empty unnamed
    /// bookend onto the first windowed-or-named entry it holds.
    ///
    /// A [`WorkspaceView`]'s active is an id, so a residue install above a fresh trailing bookend
    /// leaves the bookend active. The lift branch of [`Self::ensure_view_for`] instead activates
    /// the first lifted body, so without this reseat two dormant views holding the same workspaces
    /// would open differently depending on whether the workspaces were lifted (real tag) or
    /// residue-installed (sentinel tag). Scoped to the fresh first-monitor views — all boot-output
    /// views are freshly built here — so no in-flight animation can be snapped; do not generalize
    /// into a sweep.
    fn reseat_bookend_active_boot(&mut self, mon_idx: usize, active_id: ActivityId) {
        let boot_out_id = self.monitors[mon_idx].output_id();
        let mut reseats: Vec<(ActivityId, usize)> = Vec::new();
        for a in self.activities.iter() {
            if a.id() == active_id {
                continue;
            }
            let Some(view) = a.views().get(&boot_out_id) else {
                debug_assert!(
                    false,
                    "reseat_bookend_active_boot: activity {:?} has no view for the boot output \
                     {boot_out_id:?} after ensure_all_activity_views — per-activity bookend \
                     invariant violated (view-materialization coherence bug)",
                    a.id(),
                );
                continue;
            };
            let active_is_bookend = self
                .workspaces
                .get(&view.active())
                .is_some_and(|ws| !ws.has_windows_or_name());
            if !active_is_bookend {
                continue;
            }
            let first_body = view.ids().iter().position(|id| {
                self.workspaces
                    .get(id)
                    .is_some_and(|ws| ws.has_windows_or_name())
            });
            if let Some(pos) = first_body {
                reseats.push((a.id(), pos));
            }
        }
        for (act_id, pos) in reseats {
            self.activities
                .get_mut(act_id)
                .expect("act_id sourced from activities.iter()")
                .views_mut()
                .get_mut(&boot_out_id)
                .expect("view existed at scan time")
                .set_active_at(pos);
        }
    }

    /// Adds a new window to the layout.
    ///
    /// Returns an output that the window was added to, if there were any outputs.
    #[allow(clippy::too_many_arguments)]
    pub fn add_window(
        &mut self,
        window: W,
        target: AddWindowTarget<W>,
        width: Option<PresetSize>,
        height: Option<PresetSize>,
        is_full_width: bool,
        is_floating: bool,
        activate: ActivateWindow,
    ) -> Option<&Output> {
        let scrolling_height = height.map(SizeChange::from);
        let id = window.id().clone();

        if self.monitors.is_empty() {
            let (ws_idx, target) = match target {
                AddWindowTarget::Auto => {
                    if self.disconnected_workspace_ids.is_empty() {
                        let ws = Workspace::new_no_outputs(
                            HashSet::from([self.activities.active_id()]),
                            self.clock.clone(),
                            self.options.clone(),
                        );
                        let ws_id = ws.id();
                        assert!(
                            self.workspaces.insert(ws_id, ws).is_none(),
                            "fresh id must be unique",
                        );
                        self.disconnected_workspace_ids.push(ws_id);
                    }

                    (0, WorkspaceAddWindowTarget::Auto)
                }
                AddWindowTarget::Output(_) => panic!(),
                AddWindowTarget::Workspace(ws_id) => {
                    let ws_idx = self
                        .disconnected_workspace_ids
                        .iter()
                        .position(|id| *id == ws_id)
                        .unwrap();
                    (ws_idx, WorkspaceAddWindowTarget::Auto)
                }
                AddWindowTarget::NextTo(next_to) => {
                    if self
                        .interactive_move
                        .as_ref()
                        .and_then(|move_| {
                            if let InteractiveMoveState::Moving(move_) = move_ {
                                Some(move_)
                            } else {
                                None
                            }
                        })
                        .filter(|move_| next_to == move_.tile.window().id())
                        .is_some()
                    {
                        // The next_to window is being interactively moved. If there are no
                        // other windows, we may have no workspaces at all.
                        if self.disconnected_workspace_ids.is_empty() {
                            let ws = Workspace::new_no_outputs(
                                HashSet::from([self.activities.active_id()]),
                                self.clock.clone(),
                                self.options.clone(),
                            );
                            let ws_id = ws.id();
                            assert!(
                                self.workspaces.insert(ws_id, ws).is_none(),
                                "fresh id must be unique",
                            );
                            self.disconnected_workspace_ids.push(ws_id);
                        }

                        (0, WorkspaceAddWindowTarget::Auto)
                    } else {
                        let ws_idx = self
                            .disconnected_workspace_ids
                            .iter()
                            .position(|id| {
                                self.workspaces
                                    .get(id)
                                    .expect("id must be a key in the workspace pool")
                                    .has_window(next_to)
                            })
                            .unwrap();
                        (ws_idx, WorkspaceAddWindowTarget::NextTo(next_to))
                    }
                }
            };
            let ws_id = self.disconnected_workspace_ids[ws_idx];
            let ws = self
                .workspaces
                .get_mut(&ws_id)
                .expect("id must be a key in the workspace pool");

            let scrolling_width = ws.resolve_scrolling_width(&window, width);

            let tile = ws.make_tile(window);
            ws.add_tile(
                None,
                tile,
                target,
                activate,
                scrolling_width,
                is_full_width,
                is_floating,
            );

            // Set the default height for scrolling windows.
            if !is_floating {
                if let Some(change) = scrolling_height {
                    ws.set_window_height(Some(&id), change);
                }
            }

            return None;
        }

        let seed_activity = self.activities.active_id();

        // Hidden-target shortcut: when `AddWindowTarget::Workspace(ws_id)` names a
        // workspace that is **not** present in the active activity's view for any
        // monitor (e.g. an `open-on-activity` window rule routed the window into a
        // hidden activity per), the active-view-keyed lookups below would
        // panic on the `expect("...active activity")` assertions. Fall through to
        // a pool-only path that resolves the monitor via the workspace's bound
        // `output_id()` and adds the tile straight to the pool entry, leaving
        // every active-activity `WorkspaceView` untouched.
        if let AddWindowTarget::Workspace(ws_id) = &target {
            let active_views = self.activities.active().views();
            let in_active_view = active_views.values().any(|view| view.ids().contains(ws_id));
            if !in_active_view {
                return self.add_window_to_hidden_workspace(
                    *ws_id,
                    window,
                    width,
                    scrolling_height,
                    is_full_width,
                    is_floating,
                    activate,
                );
            }
        }

        let views = self.activities.active().views();
        let pool = &mut self.workspaces;
        let monitors = &mut self.monitors[..];
        let active_monitor_idx = &mut self.active_monitor_idx;
        let (mon_idx, target) = match target {
            AddWindowTarget::Auto => (*active_monitor_idx, MonitorAddWindowTarget::Auto),
            AddWindowTarget::Output(output) => {
                let mon_idx = monitors
                    .iter()
                    .position(|mon| mon.output == *output)
                    .unwrap();

                (mon_idx, MonitorAddWindowTarget::Auto)
            }
            AddWindowTarget::Workspace(ws_id) => {
                // Visible-target path: the early shortcut above ruled out
                // hidden-activity ids, so the active view *must* contain `ws_id`.
                debug_assert!(
                    views.values().any(|view| view.ids().contains(&ws_id)),
                    "hidden-target shortcut must have intercepted ws_id not in any active view"
                );
                let mon_idx = monitors
                    .iter()
                    .position(|mon| {
                        views
                            .get(&OutputId::new(&mon.output))
                            .expect("connected output must have a view in the active activity")
                            .ids()
                            .contains(&ws_id)
                    })
                    .expect("hidden-target shortcut above must catch ws_id not in any active view");

                (
                    mon_idx,
                    MonitorAddWindowTarget::Workspace {
                        id: ws_id,
                        column_idx: None,
                    },
                )
            }
            AddWindowTarget::NextTo(next_to) => {
                if let Some(output) = self
                    .interactive_move
                    .as_ref()
                    .and_then(|move_| {
                        if let InteractiveMoveState::Moving(move_) = move_ {
                            Some(move_)
                        } else {
                            None
                        }
                    })
                    .filter(|move_| next_to == move_.tile.window().id())
                    .map(|move_| move_.output.clone())
                {
                    // The next_to window is being interactively moved.
                    let mon_idx = monitors
                        .iter()
                        .position(|mon| mon.output == output)
                        .unwrap_or(*active_monitor_idx);

                    (mon_idx, MonitorAddWindowTarget::Auto)
                } else {
                    let mon_idx = monitors
                        .iter()
                        .position(|mon| {
                            views
                                .get(&OutputId::new(&mon.output))
                                .expect("connected output must have a view in the active activity")
                                .ids()
                                .iter()
                                .any(|id| {
                                    pool.get(id)
                                        .expect("view id must be a key in the pool")
                                        .has_window(next_to)
                                })
                        })
                        .unwrap();
                    (mon_idx, MonitorAddWindowTarget::NextTo(next_to))
                }
            }
        };
        // End the earlier `views`/`pool`/`monitors`/`active_monitor_idx` borrows so we can
        // re-borrow the active activity mutably for the chosen monitor's view.
        let mon_output_id = OutputId::new(&monitors[mon_idx].output);
        let view = self
            .activities
            .active_mut()
            .views_mut()
            .get_mut(&mon_output_id)
            .expect("connected output must have a view in the active activity");
        let monitors = &mut self.monitors[..];
        let pool = &mut self.workspaces;
        let active_monitor_idx = &mut self.active_monitor_idx;
        let scrolling_width = {
            let mon = &monitors[mon_idx];
            let (ws_idx, _) = mon.resolve_add_window_target(pool, view, target);
            mon.workspace_at(pool, view, ws_idx)
                .resolve_scrolling_width(&window, width)
        };

        Self::add_window_on(
            monitors,
            pool,
            view,
            mon_idx,
            window,
            target,
            activate,
            scrolling_width,
            is_full_width,
            is_floating,
            seed_activity,
        );

        if activate.map_smart(|| false) {
            *active_monitor_idx = mon_idx;
        }

        // Resolve `ws_id` (the workspace that received the window) while the split-borrow is
        // still in scope. Used for the scrolling-height patch below and to assert a receiving
        // workspace exists before the post-split-borrow sweep.
        let receiving_ws_id = view.ids().iter().copied().find(|ws_id| {
            pool.get(ws_id)
                .expect("view id must be a key in the pool")
                .has_window(&id)
        });

        // Set the default height for scrolling windows.
        if !is_floating {
            if let Some(change) = scrolling_height {
                let ws_id = receiving_ws_id
                    .expect("a scrolling window must land on some workspace in the active view");
                let ws = pool
                    .get_mut(&ws_id)
                    .expect("view id must be a key in the pool");
                ws.set_window_height(Some(&id), change);
            }
        }

        // End of split borrow. The sweep repairs any view — dormant, or redundantly the active
        // one `add_column_on` already handled — left with a violated bookend by this add. The
        // receiving workspace must exist for any successful add, including floating windows.
        receiving_ws_id
            .expect("add_window must have placed the window on some active-view workspace");
        self.normalize_view_bookends();

        Some(&self.monitors[mon_idx].output)
    }

    /// Pool-only [`Self::add_window`] path used when the requested
    /// `AddWindowTarget::Workspace(ws_id)` is not present in any active
    /// activity's `WorkspaceView` — i.e. the target workspace lives in a
    /// hidden activity ( `open-on-activity` window-rule into a
    /// non-active activity).
    ///
    /// Resolves the owning monitor through the workspace's bound
    /// `output_id()` (Some(real OutputId) for connected workspaces) and
    /// adds the tile straight to the pool entry via `Workspace::add_tile`.
    /// Bypasses every `WorkspaceView` lookup and every `Monitor`-keyed
    /// add path so the active-activity views remain untouched. The user
    /// will see the new window the next time they switch to that activity.
    ///
    /// Returns the monitor's `Output` to mirror [`Self::add_window`]'s
    /// return shape, so the caller (`compositor.rs`) can `queue_redraw`
    /// against it.
    #[allow(clippy::too_many_arguments)]
    fn add_window_to_hidden_workspace(
        &mut self,
        ws_id: WorkspaceId,
        window: W,
        width: Option<PresetSize>,
        scrolling_height: Option<SizeChange>,
        is_full_width: bool,
        is_floating: bool,
        activate: ActivateWindow,
    ) -> Option<&Output> {
        let id = window.id().clone();
        let ws_output_id = self
            .workspaces
            .get(&ws_id)
            .expect("AddWindowTarget::Workspace must name a live pool key")
            .output_id()
            .cloned();
        // Resolve the monitor displaying this workspace. A bound workspace resolves by its real
        // `output_id`. A workspace routed into a dormant view by activity membership carries the
        // empty-string sentinel `output_id` (a config workspace with no `open-on-output` keeps it
        // until it is reclaim-bound), so it matches no monitor by tag; fall back to the view that
        // currently holds it. If neither resolves — the workspace is on no connected monitor —
        // return None so the caller drops the window normally rather than orphaning a tile in the
        // pool.
        let mon_idx = ws_output_id
            .as_ref()
            .and_then(|oid| self.monitors.iter().position(|mon| mon.output_id() == *oid))
            .or_else(|| {
                let holding = self.workspace_holding_output(ws_id)?;
                self.monitors
                    .iter()
                    .position(|mon| mon.output_id() == holding)
            })?;

        let mon_output = self.monitors[mon_idx].output.clone();
        let ws = self
            .workspaces
            .get_mut(&ws_id)
            .expect("AddWindowTarget::Workspace must name a live pool key");
        let scrolling_width = ws.resolve_scrolling_width(&window, width);
        let tile = ws.make_tile(window);
        ws.add_tile(
            Some(&mon_output),
            tile,
            WorkspaceAddWindowTarget::Auto,
            activate,
            scrolling_width,
            is_full_width,
            is_floating,
        );

        if !is_floating {
            if let Some(change) = scrolling_height {
                ws.set_window_height(Some(&id), change);
            }
        }

        // Hidden-target path only fires when ws_id is NOT in any active view, so the sweep's
        // active-view arm finds nothing here — only the dormant view may need a bookend appended.
        self.normalize_view_bookends();

        Some(&self.monitors[mon_idx].output)
    }

    /// Remove a window from the layout because it closed.
    ///
    /// Prunes the bookmark anchored to this window before delegating to
    /// [`Self::remove_window_inner`]. The in-layout detach-for-relocation in
    /// `interactive_move_update` calls the inner directly instead: pruning a
    /// merely-relocating window (re-added immediately) would silently destroy
    /// its bookmark every time it is dragged.
    pub fn remove_window(
        &mut self,
        window: &W::Id,
        transaction: Transaction,
    ) -> Option<RemovedTile<W>> {
        self.bookmarks.prune_window(window);
        self.remove_window_inner(window, transaction)
    }

    fn remove_window_inner(
        &mut self,
        window: &W::Id,
        transaction: Transaction,
    ) -> Option<RemovedTile<W>> {
        if let Some(state) = &self.interactive_move {
            match state {
                InteractiveMoveState::Starting { window_id, .. } => {
                    if window_id == window {
                        self.interactive_move_end(window);
                    }
                }
                InteractiveMoveState::Moving(move_) => {
                    if move_.tile.window().id() == window {
                        let Some(InteractiveMoveState::Moving(move_)) =
                            self.interactive_move.take()
                        else {
                            unreachable!()
                        };

                        let views_map = self.activities.active_mut().views_mut();
                        for mon in self.monitors.iter_mut() {
                            let view = views_map
                                .get_mut(&OutputId::new(&mon.output))
                                .expect("connected output must have a view in the active activity");
                            mon.dnd_scroll_gesture_end(view);
                        }

                        // Unlock the view on the workspaces.
                        for ws in self.workspaces_mut() {
                            ws.dnd_scroll_gesture_end();
                        }

                        // Pair the `output_enter` fired on the tile during drag start /
                        // cross-output update (in `interactive_move_update`). The tile is
                        // returned to the caller here; firing `output_leave` first ensures
                        // the drag-tracked marker is not carried outside the compositor's
                        // view of this window's output bindings.
                        move_.tile.window().output_leave(&move_.output);

                        return Some(RemovedTile {
                            tile: move_.tile,
                            width: move_.width,
                            is_full_width: move_.is_full_width,
                            is_floating: false,
                        });
                    }
                }
            }
        }

        for mon_idx in 0..self.monitors.len() {
            let mon_out = self.monitors[mon_idx].output_id();
            let (active_pos, view_len, ids) = {
                let view = self.active_view(&mon_out);
                (view.active_position(), view.len(), view.ids().to_vec())
            };
            for (idx, id) in ids.iter().copied().enumerate() {
                let has_window = self
                    .workspaces
                    .get(&id)
                    .expect("workspace id must be a key in the pool")
                    .has_window(window);
                if !has_window {
                    continue;
                }

                let removed = {
                    let mon_output = self.monitors[mon_idx].output.clone();
                    let ws = self
                        .workspaces
                        .get_mut(&id)
                        .expect("workspace id must be a key in the pool");
                    ws.remove_tile(Some(&mon_output), window, transaction)
                };

                let ws_empty = !self
                    .workspaces
                    .get(&id)
                    .expect("workspace id must be a key in the pool")
                    .has_windows_or_name();

                // Clean up empty workspaces that are not active and not last. A workspace
                // shared across activities is pinned: reclaiming it here would leave every
                // other member activity's view referencing a dead pool key, panicking on
                // the next switch into one of them. It stays as an empty middle instead.
                if ws_empty
                    && idx != active_pos
                    && idx != view_len - 1
                    && self.monitors[mon_idx].workspace_switch.is_none()
                {
                    if Self::workspace_is_safe_to_reclaim(&self.workspaces, id) {
                        self.active_view_mut(&mon_out).remove_at(idx);
                        assert!(
                            self.workspaces.remove(&id).is_some(),
                            "view id must be a key in the pool",
                        );
                    } else {
                        let n = self
                            .workspaces
                            .get(&id)
                            .map_or(0, |ws| ws.activities().len());
                        trace!(
                            "remove_window: workspace {id:?} emptied but pinned \
                             (shared across {n} activities)",
                        );
                    }
                }

                // Special case handling when empty_workspace_above_first is set and all
                // workspaces are empty.
                let special = self.monitors[mon_idx]
                    .options
                    .layout
                    .empty_workspace_above_first
                    && self.active_view(&mon_out).len() == 2
                    && self.monitors[mon_idx].workspace_switch.is_none();
                if special {
                    let mon = &self.monitors[mon_idx];
                    let pool = &self.workspaces;
                    let view = self.active_view(&mon_out);
                    assert!(!mon.workspace_at(pool, view, 0).has_windows_or_name());
                    assert!(!mon.workspace_at(pool, view, 1).has_windows_or_name());
                    let drop_id = self.active_view(&mon_out).ids()[1];
                    // A shared second entry is pinned by another activity's view; the
                    // length-2 view is then the honest minimal shape (the shared empty
                    // doubles as the trailing bookend).
                    if Self::workspace_is_safe_to_reclaim(&self.workspaces, drop_id) {
                        self.active_view_mut(&mon_out).remove_at(1);
                        assert!(
                            self.workspaces.remove(&drop_id).is_some(),
                            "view id must be a key in the pool",
                        );
                    } else {
                        let n = self
                            .workspaces
                            .get(&drop_id)
                            .map_or(0, |ws| ws.activities().len());
                        trace!(
                            "remove_window: workspace {drop_id:?} emptied but pinned \
                             (shared across {n} activities, EWAF trailing slot)",
                        );
                    }
                }
                return Some(removed);
            }
        }

        for idx in 0..self.disconnected_workspace_ids.len() {
            let id = self.disconnected_workspace_ids[idx];
            let ws = self
                .workspaces
                .get_mut(&id)
                .expect("workspace id must be a key in the pool");
            if ws.has_window(window) {
                let removed = ws.remove_tile(None, window, transaction);

                // Clean up empty workspaces.
                if !ws.has_windows_or_name() {
                    self.disconnected_workspace_ids.remove(idx);
                    assert!(
                        self.workspaces.remove(&id).is_some(),
                        "id must be a key in the workspace pool",
                    );
                }

                return Some(removed);
            }
        }

        // Reap a dead-client tile from a dormant-activity view. A window living
        // exclusively in a non-active activity's workspace on a connected output is
        // present in the pool but in neither the active-view loop above (which only
        // walks each monitor's active view) nor the disconnected loop (which only
        // fires when no monitors are connected). Without this arm its tile leaks.
        //
        // Resolve ids/output under a shared borrow first, then mutate, per the
        // pool-then-monitors borrow-split recipe.
        let dormant_hit = self
            .workspaces
            .iter()
            .find(|(_, ws)| ws.has_window(window))
            .map(|(id, ws)| (*id, ws.activities().len()));

        if let Some((id, activity_count)) = dormant_hit {
            // Resolve the monitor and owning view from the *view-keyed* output, NOT from
            // `ws.output_id()`. A partial disconnect (`remove_output` with monitors still
            // non-empty) migrates a window-bearing dormant workspace into the dormant
            // activity's view on the primary monitor but deliberately leaves the
            // workspace's `output_id` field pointing at the now-disconnected output (the
            // reclaim-on-reconnect tag; `bind_output` refreshes `output_id` only when it
            // already matches the bound output). So `ws.output_id()` can name a gone
            // output here even though the view itself lives on a connected monitor.
            // Scanning the activity views for the one holding `id` yields the keyed output,
            // which the migration guarantees is connected — that is the assumption the
            // `.expect`s below rely on, making them genuinely unreachable.
            let output_id = self
                .activities
                .iter()
                .flat_map(|activity| activity.views().iter())
                .find(|(_, view)| view.position_of(id).is_some())
                .map(|(out, _)| out.clone())
                .expect("a dormant-view window's workspace must live in some activity view");
            let mon_idx = self
                .monitors
                .iter()
                .position(|mon| mon.output_id() == output_id)
                .expect("a dormant view's keyed output must be a connected monitor");

            let removed = {
                let mon_output = self.monitors[mon_idx].output.clone();
                let ws = self
                    .workspaces
                    .get_mut(&id)
                    .expect("pool scan just located this id");
                ws.remove_tile(Some(&mon_output), window, transaction)
            };

            // Optionally reclaim the now-empty dormant workspace, mirroring the cleanup
            // the other two arms perform. An empty unnamed *middle* workspace in a dormant
            // view violates no invariant (the "no empty middle" rule is active-view-only),
            // so reclaiming is best-effort: only drop the workspace when it is safe to
            // remove from its owning view without breaking that view's bookend invariant,
            // which `assert_view_bookends` checks on every view including dormant ones.
            let ws_empty = !self
                .workspaces
                .get(&id)
                .expect("pool scan just located this id")
                .has_windows_or_name();

            if ws_empty && activity_count == 1 {
                let ewaf = self.monitors[mon_idx]
                    .options
                    .layout
                    .empty_workspace_above_first;

                // The workspace is exclusive to one activity, so it appears in exactly one
                // view (per-view uniqueness, one output binding). Locate that view and
                // reclaim only when `id` is a non-bookend, non-degenerate entry. Reaching
                // this arm means the workspace is dormant — the active-view loop above
                // already handled every window on an active view.
                let owning_view = self
                    .activities
                    .iter()
                    .filter_map(|activity| activity.views().get(&output_id))
                    .find(|view| view.position_of(id).is_some());

                let safe_to_reclaim = owning_view.is_some_and(|view| {
                    let pos = view
                        .position_of(id)
                        .expect("filtered to views containing id");
                    let len = view.len();
                    let is_trailing = pos + 1 == len;
                    let is_leading = pos == 0;
                    if is_trailing {
                        return false;
                    }
                    if ewaf {
                        // Leading is a bookend under EWAF, and a middle drop must not leave
                        // a length-2 view (EWAF requires 1 or 3+).
                        !is_leading && len > 3
                    } else {
                        true
                    }
                });

                if safe_to_reclaim {
                    let pool = &mut self.workspaces;
                    Self::destroy_workspaces_cross_activity(&mut self.activities, pool, [id]);
                }
            }

            return Some(removed);
        }

        None
    }

    pub fn descendants_added(&mut self, id: &W::Id) -> bool {
        for ws in self.workspaces_mut() {
            if ws.descendants_added(id) {
                return true;
            }
        }

        false
    }

    pub fn update_window(&mut self, window: &W::Id, serial: Option<Serial>) {
        if let Some(InteractiveMoveState::Moving(move_)) = &mut self.interactive_move {
            if move_.tile.window().id() == window {
                // Do this before calling update_window() so it can get up-to-date info.
                if let Some(serial) = serial {
                    move_.tile.window_mut().on_commit(serial);
                }

                move_.tile.update_window();
                return;
            }
        }

        for mon_idx in 0..self.monitors.len() {
            let mon_out = self.monitors[mon_idx].output_id();
            let ids: Vec<WorkspaceId> = self.active_view(&mon_out).ids().to_vec();
            for id in ids {
                let ws = self
                    .workspaces
                    .get_mut(&id)
                    .expect("workspace id must be a key in the pool");
                if ws.has_window(window) {
                    ws.update_window(window, serial);
                    return;
                }
            }
        }
        for id in &self.disconnected_workspace_ids {
            let ws = self
                .workspaces
                .get_mut(id)
                .expect("workspace id must be a key in the pool");
            if ws.has_window(window) {
                ws.update_window(window, serial);
                return;
            }
        }
    }

    /// Returns the (index-in-active-view, workspace) for `id` iff the id is in
    /// some monitor's active view or in `disconnected_workspace_ids`.
    /// **Active-view + disconnected-pool scoped** — workspaces exclusive to a
    /// dormant activity yield `None` here even though they exist in the pool.
    /// Compose with `Layout::resolve_workspace_id` when you need
    /// pool-membership semantics.
    pub fn find_workspace_by_id(&self, id: WorkspaceId) -> Option<(usize, &Workspace<W>)> {
        for mon in &self.monitors {
            if let Some(index) = self.active_view(&mon.output_id()).position_of(id) {
                let workspace = self
                    .workspaces
                    .get(&id)
                    .expect("view id must be a key in the pool");
                return Some((index, workspace));
            }
        }
        if let Some(index) = self
            .disconnected_workspace_ids
            .iter()
            .position(|ws_id| *ws_id == id)
        {
            let workspace = self
                .workspaces
                .get(&id)
                .expect("id must be a key in the workspace pool");
            return Some((index, workspace));
        }

        None
    }

    /// Returns the Smithay `Output` of the `Monitor` currently owning the workspace with the
    /// given id, or `None` if no monitor currently owns this workspace or the id is unknown.
    pub fn output_for_workspace(&self, id: WorkspaceId) -> Option<&Output> {
        self.monitors
            .iter()
            .find(|mon| self.active_view(&mon.output_id()).ids().contains(&id))
            .map(|mon| &mon.output)
    }

    pub fn find_workspace_by_name(&self, workspace_name: &str) -> Option<(usize, &Workspace<W>)> {
        let pool = &self.workspaces;
        for mon in &self.monitors {
            let view = self.active_view(&mon.output_id());
            if let Some(index) = view.ids().iter().position(|id| {
                pool.get(id)
                    .expect("view id must be a key in the pool")
                    .name
                    .as_ref()
                    .is_some_and(|name| name.eq_ignore_ascii_case(workspace_name))
            }) {
                let id = view.ids()[index];
                return Some((
                    index,
                    pool.get(&id).expect("view id must be a key in the pool"),
                ));
            }
        }
        if let Some((index, id)) =
            self.disconnected_workspace_ids
                .iter()
                .enumerate()
                .find(|(_, id)| {
                    let workspace = pool
                        .get(id)
                        .expect("id must be a key in the workspace pool");
                    workspace
                        .name
                        .as_ref()
                        .is_some_and(|name| name.eq_ignore_ascii_case(workspace_name))
                })
        {
            return Some((
                index,
                pool.get(id)
                    .expect("id must be a key in the workspace pool"),
            ));
        }

        None
    }

    pub fn find_workspace_by_ref(
        &mut self,
        reference: WorkspaceReference,
    ) -> Option<&mut Workspace<W>> {
        if let WorkspaceReference::Index(index) = reference {
            let index = index.saturating_sub(1) as usize;
            if self.monitors.is_empty() {
                return None;
            }
            let mon_out = self.monitors[self.active_monitor_idx].output_id();
            let id = self.active_view(&mon_out).ids().get(index).copied()?;
            Some(
                self.workspaces
                    .get_mut(&id)
                    .expect("view id must be a key in the pool"),
            )
        } else {
            self.workspaces_mut().find(|ws| match &reference {
                WorkspaceReference::Name(ref_name) => ws
                    .name
                    .as_ref()
                    .is_some_and(|name| name.eq_ignore_ascii_case(ref_name)),
                WorkspaceReference::Id(id) => ws.id().get() == *id,
                WorkspaceReference::Index(_) => unreachable!(),
            })
        }
    }

    pub fn unname_workspace(&mut self, workspace_name: &str) {
        self.unname_workspace_by_ref(WorkspaceReference::Name(workspace_name.into()));
    }

    pub fn unname_workspace_by_ref(&mut self, reference: WorkspaceReference) {
        let id = self.find_workspace_by_ref(reference).map(|ws| ws.id());
        if let Some(id) = id {
            self.unname_workspace_by_id(id);
        }
    }

    pub fn unname_workspace_by_id(&mut self, id: WorkspaceId) {
        if let Some(mon_idx) = self
            .monitors
            .iter()
            .position(|mon| self.active_view(&mon.output_id()).ids().contains(&id))
        {
            self.workspaces
                .get_mut(&id)
                .expect("view id must be a key in the pool")
                .unname();
            if self.monitors[mon_idx].workspace_switch.is_none() {
                let mon_out = self.monitors[mon_idx].output_id();
                let ids_to_destroy = {
                    let (monitors, pool, view) = self.monitors_pool_view_mut(&mon_out);
                    Self::clean_up_workspaces_on(monitors, pool, view, mon_idx)
                };
                Self::destroy_workspaces_cross_activity(
                    &mut self.activities,
                    &mut self.workspaces,
                    ids_to_destroy,
                );
            }
            return;
        }

        if let Some(idx) = self
            .disconnected_workspace_ids
            .iter()
            .position(|ws_id| *ws_id == id)
        {
            let ws = self
                .workspaces
                .get_mut(&id)
                .expect("workspace id must be a key in the pool");
            ws.unname();

            // Clean up empty workspaces.
            if !ws.has_windows() {
                self.disconnected_workspace_ids.remove(idx);
                assert!(
                    self.workspaces.remove(&id).is_some(),
                    "id must be a key in the workspace pool",
                );
            }
        }
    }

    /// Locate a window by its `wl_surface` across the entire workspace pool
    /// (not only the active activity). The interactive-move tile is checked
    /// first; otherwise every workspace's scrolling and floating children
    /// are scanned. The returned `Option<&Output>` is resolved from the
    /// workspace's bound output id via `monitor_for_output_id`. Returns
    /// `Some((window, None))` when the winning workspace's bound `OutputId`
    /// is absent (workspace never attached to any output), OR when the bound
    /// id is present but not currently in `self.monitors` (disconnected
    /// output).
    ///
    /// This widen (pool-spanning instead of active-activity-only) is
    /// required for correct surface-event routing to windows on dormant
    /// activities — their surface commits, ack_configures and popups must
    /// still reach the layout after an activity switch.
    pub fn find_window_and_output(&self, wl_surface: &WlSurface) -> Option<(&W, Option<&Output>)> {
        if let Some(InteractiveMoveState::Moving(move_)) = &self.interactive_move {
            if move_.tile.window().is_wl_surface(wl_surface) {
                return Some((move_.tile.window(), Some(&move_.output)));
            }
        }

        for ws in self.workspaces.values() {
            if let Some(window) = ws.find_wl_surface(wl_surface) {
                let output = ws
                    .output_id()
                    .and_then(|oid| self.monitor_for_output_id(oid))
                    .map(|m| &m.output);
                return Some((window, output));
            }
        }

        None
    }

    /// Mutable twin of [`Self::find_window_and_output`]. Uses the same
    /// pool-spanning scan. The two-phase borrow pattern (shared-scan to
    /// locate `(id, output_id)`, then `workspaces.get_mut(&id)`) is
    /// preserved deliberately — a single-phase `values_mut()` scan would
    /// conflict with the `monitor_for_output_id` lookup needed to resolve
    /// `&Output` from the winning workspace's bound output id.
    ///
    /// Returns `Some((&mut W, None))` when the winning workspace is not
    /// currently bound to any connected monitor — either no `OutputId` (the
    /// workspace was never attached to any output), or the bound `OutputId`
    /// is present but not found in `self.monitors` (disconnected output).
    pub fn find_window_and_output_mut(
        &mut self,
        wl_surface: &WlSurface,
    ) -> Option<(&mut W, Option<&Output>)> {
        if let Some(InteractiveMoveState::Moving(move_)) = &mut self.interactive_move {
            if move_.tile.window().is_wl_surface(wl_surface) {
                return Some((move_.tile.window_mut(), Some(&move_.output)));
            }
        }

        // Find the matching workspace id and remember its bound output id, if
        // any. `OutputId` is owned/cheap; copying it here lets the shared
        // borrow of the pool drop before the next section.
        let matching: Option<(WorkspaceId, Option<OutputId>)> = self
            .workspaces
            .iter()
            .find(|(_, ws)| ws.find_wl_surface(wl_surface).is_some())
            .map(|(id, ws)| (*id, ws.output_id().cloned()));

        let (id, output_id) = matching?;
        // `self.monitors` and `self.workspaces` are disjoint fields, so the
        // borrow checker permits a shared borrow of one concurrent with a
        // mutable borrow of the other — provided we reach each through the
        // field directly (not via `self.monitor_for_output_id(...)` which
        // takes `&self`).
        let output: Option<&Output> = output_id.as_ref().and_then(|oid| {
            self.monitors
                .iter()
                .find(|m| m.output_id() == *oid)
                .map(|m| &m.output)
        });
        let ws = self
            .workspaces
            .get_mut(&id)
            .expect("workspace id must be a key in the pool (located in step 1)");
        ws.find_wl_surface_mut(wl_surface).map(|w| (w, output))
    }

    /// Computes the window-geometry-relative target rect for popup unconstraining.
    ///
    /// We will try to fit popups inside this rect.
    pub fn popup_target_rect(&self, window: &W::Id) -> Rectangle<f64, Logical> {
        if let Some(InteractiveMoveState::Moving(move_)) = &self.interactive_move {
            if move_.tile.window().id() == window {
                // Follow the scrolling layout logic and fit the popup horizontally within the
                // window geometry.
                let width = move_.tile.window_size().w;
                let height = output_size(&move_.output).h;
                let mut target = Rectangle::from_size(Size::from((width, height)));
                // FIXME: ideally this shouldn't include the tile render offset, but the code
                // duplication would be a bit annoying for this edge case.
                target.loc.y -= move_.tile_render_location(1.).y;
                target.loc.y -= move_.tile.window_loc().y;
                return target;
            }
        }

        self.workspaces_all()
            .find_map(|(_, ws)| ws.popup_target_rect(window))
            .expect("popup_target_rect called for window not in any pool workspace")
    }

    pub fn update_output_size(&mut self, output: &Output) {
        let _span = tracy_client::span!("Layout::update_output_size");

        if !self.monitors.iter().any(|m| &m.output == output) {
            error!("monitor missing in update_output_size()");
            return;
        }
        let output_id = OutputId::new(output);
        let view = self
            .activities
            .active()
            .views()
            .get(&output_id)
            .expect("connected output must have a view in the active activity");
        let pool = &mut self.workspaces;
        let mon = self
            .monitors
            .iter_mut()
            .find(|m| &m.output == output)
            .expect("monitor for connected output must exist");

        mon.update_output_size(pool, view);
    }

    pub fn scroll_amount_to_activate(&self, window: &W::Id) -> f64 {
        if let Some(InteractiveMoveState::Moving(move_)) = &self.interactive_move {
            if move_.tile.window().id() == window {
                return 0.;
            }
        }

        let pool = &self.workspaces;
        for mon in self.monitors() {
            for id in self.active_view(&mon.output_id()).ids() {
                let ws = pool
                    .get(id)
                    .expect("workspace id must be a key in the pool");
                if ws.has_window(window) {
                    return ws.scroll_amount_to_activate(window);
                }
            }
        }

        0.
    }

    pub fn should_trigger_focus_follows_mouse_on(&self, window: &W::Id) -> bool {
        // During an animation, it's easy to trigger focus-follows-mouse on the previous workspace,
        // especially when clicking to switch workspace on a bar of some kind. This cancels the
        // workspace switch, which is annoying and not intended.
        //
        // This function allows focus-follows-mouse to trigger only on the animation target
        // workspace.
        if let Some(InteractiveMoveState::Moving(move_)) = &self.interactive_move {
            if move_.tile.window().id() == window {
                return true;
            }
        }

        if self.monitors.is_empty() {
            return true;
        }

        let pool = &self.workspaces;
        let (mon, ws_idx) = self
            .monitors
            .iter()
            .find_map(|mon| {
                self.active_view(&mon.output_id())
                    .ids()
                    .iter()
                    .position(|id| {
                        pool.get(id)
                            .expect("view id must be a key in the pool")
                            .has_window(window)
                    })
                    .map(|ws_idx| (mon, ws_idx))
            })
            .unwrap();

        // During a gesture, focus-follows-mouse does not cause any unintended workspace switches.
        if let Some(WorkspaceSwitch::Gesture(_)) = mon.workspace_switch {
            return true;
        }

        ws_idx == self.active_view(&mon.output_id()).active_position()
    }

    pub fn activate_window(&mut self, window: &W::Id) {
        if let Some(InteractiveMoveState::Moving(move_)) = &self.interactive_move {
            if move_.tile.window().id() == window {
                return;
            }
        }

        let views_map = self.activities.active_mut().views_mut();
        let pool = &mut self.workspaces;
        let monitors = &mut self.monitors[..];
        let active_monitor_idx = &mut self.active_monitor_idx;

        for (monitor_idx, mon) in monitors.iter_mut().enumerate() {
            let view = views_map
                .get_mut(&OutputId::new(&mon.output))
                .expect("connected output must have a view in the active activity");
            for workspace_idx in 0..view.len() {
                let id = view.ids()[workspace_idx];
                let ws = pool
                    .get_mut(&id)
                    .expect("workspace id must be a key in the pool");
                if ws.activate_window(window) {
                    *active_monitor_idx = monitor_idx;

                    // If currently in the middle of a vertical swipe between the target workspace
                    // and some other, don't switch the workspace.
                    match &mon.workspace_switch {
                        Some(WorkspaceSwitch::Gesture(gesture))
                            if gesture.current_idx.floor() == workspace_idx as f64
                                || gesture.current_idx.ceil() == workspace_idx as f64 => {}
                        _ => mon.switch_workspace(view, workspace_idx),
                    }

                    return;
                }
            }
        }
    }

    pub fn activate_window_without_raising(&mut self, window: &W::Id) {
        if let Some(InteractiveMoveState::Moving(move_)) = &self.interactive_move {
            if move_.tile.window().id() == window {
                return;
            }
        }

        let views_map = self.activities.active_mut().views_mut();
        let pool = &mut self.workspaces;
        let monitors = &mut self.monitors[..];
        let active_monitor_idx = &mut self.active_monitor_idx;

        for (monitor_idx, mon) in monitors.iter_mut().enumerate() {
            let view = views_map
                .get_mut(&OutputId::new(&mon.output))
                .expect("connected output must have a view in the active activity");
            for workspace_idx in 0..view.len() {
                let id = view.ids()[workspace_idx];
                let ws = pool
                    .get_mut(&id)
                    .expect("workspace id must be a key in the pool");
                if ws.activate_window_without_raising(window) {
                    *active_monitor_idx = monitor_idx;

                    // If currently in the middle of a vertical swipe between the target workspace
                    // and some other, don't switch the workspace.
                    match &mon.workspace_switch {
                        Some(WorkspaceSwitch::Gesture(gesture))
                            if gesture.current_idx.floor() == workspace_idx as f64
                                || gesture.current_idx.ceil() == workspace_idx as f64 => {}
                        _ => mon.switch_workspace(view, workspace_idx),
                    }

                    return;
                }
            }
        }
    }

    pub fn active_output(&self) -> Option<&Output> {
        if self.monitors.is_empty() {
            return None;
        }
        Some(&self.monitors[self.active_monitor_idx].output)
    }

    pub fn active_workspace(&self) -> Option<&Workspace<W>> {
        if self.monitors.is_empty() {
            return None;
        }
        let mon = &self.monitors[self.active_monitor_idx];
        let view = self.active_view(&mon.output_id());
        Some(mon.active_workspace_ref(&self.workspaces, view))
    }

    pub fn active_workspace_mut(&mut self) -> Option<&mut Workspace<W>> {
        if self.monitors.is_empty() {
            return None;
        }
        let mon_idx = self.active_monitor_idx;
        let mon_out = self.monitors[mon_idx].output_id();
        let view = self
            .activities
            .active()
            .views()
            .get(&mon_out)
            .expect("connected output must have a view in the active activity");
        let ws_id = view.active();
        Some(
            self.workspaces
                .get_mut(&ws_id)
                .expect("view id must be a key in the pool"),
        )
    }

    pub fn windows_for_output(&self, output: &Output) -> impl Iterator<Item = &W> + '_ {
        assert!(
            !self.monitors.is_empty(),
            "windows_for_output requires at least one connected monitor",
        );

        let moving_window = self
            .interactive_move
            .as_ref()
            .and_then(|x| x.moving())
            .filter(|move_| move_.output == *output)
            .map(|move_| move_.tile.window())
            .into_iter();

        let mon = self
            .monitors
            .iter()
            .find(|mon| &mon.output == output)
            .unwrap();
        let pool = &self.workspaces;
        let mon_windows = self
            .active_view(&mon.output_id())
            .ids()
            .iter()
            .flat_map(move |id| {
                pool.get(id)
                    .expect("workspace id must be a key in the pool")
                    .windows()
            });

        moving_window.chain(mon_windows)
    }

    pub fn windows_for_output_mut(&mut self, output: &Output) -> impl Iterator<Item = &mut W> + '_ {
        // Split the struct fields: find the monitor index first, then borrow pool + output
        // disjointly. `interactive_move`'s matching window is yielded first.
        let (mi, is_interactive_match) = {
            assert!(
                !self.monitors.is_empty(),
                "windows_for_output_mut requires at least one connected monitor",
            );
            let mi = self
                .monitors
                .iter()
                .position(|mon| &mon.output == output)
                .unwrap();
            let is_match = self
                .interactive_move
                .as_ref()
                .and_then(|x| x.moving())
                .is_some_and(|move_| move_.output == *output);
            (mi, is_match)
        };

        // Collect ids before taking the mutable borrow of `interactive_move`; otherwise the
        // mutable borrow held by `moving_window` would overlap with the shared borrow of
        // `self.activities` via `active_view`.
        let ids: Vec<WorkspaceId> = {
            let mon_out = self.monitors[mi].output_id();
            // Iterate ids in order, hand out non-overlapping `&mut Workspace<W>` via raw ptr. Safe
            // because `view.ids()` has no duplicates.
            self.active_view(&mon_out).ids().to_vec()
        };

        let moving_window = if is_interactive_match {
            self.interactive_move
                .as_mut()
                .and_then(|x| x.moving_mut())
                .map(|move_| move_.tile.window_mut())
        } else {
            None
        }
        .into_iter();

        let pool = &mut self.workspaces;
        let pool_ptr: *mut HashMap<WorkspaceId, Workspace<W>> = pool;
        let mon_windows = ids.into_iter().flat_map(move |id| {
            // SAFETY: each `id` is unique in `ids`; borrow checker can't prove it.
            let ws = unsafe { &mut *pool_ptr }.get_mut(&id).unwrap();
            ws.windows_mut()
        });

        moving_window.chain(mon_windows)
    }

    pub fn with_windows(
        &self,
        mut f: impl FnMut(&W, Option<&Output>, Option<WorkspaceId>, WindowLayout),
    ) {
        if let Some(InteractiveMoveState::Moving(move_)) = &self.interactive_move {
            // We don't fill any positions for interactively moved windows.
            let layout = move_.tile.ipc_layout_template();
            f(move_.tile.window(), Some(&move_.output), None, layout);
        }

        let pool = &self.workspaces;
        for mon in &self.monitors {
            for id in self.active_view(&mon.output_id()).ids() {
                let ws = pool
                    .get(id)
                    .expect("workspace id must be a key in the pool");
                for (tile, layout) in ws.tiles_with_ipc_layouts() {
                    f(tile.window(), Some(&mon.output), Some(*id), layout);
                }
            }
        }
        for id in &self.disconnected_workspace_ids {
            let ws = pool
                .get(id)
                .expect("workspace id must be a key in the pool");
            for (tile, layout) in ws.tiles_with_ipc_layouts() {
                f(tile.window(), None, Some(*id), layout);
            }
        }
    }

    pub fn with_windows_mut(&mut self, mut f: impl FnMut(&mut W, Option<&Output>)) {
        if let Some(InteractiveMoveState::Moving(move_)) = &mut self.interactive_move {
            f(move_.tile.window_mut(), Some(&move_.output));
        }

        for mon_idx in 0..self.monitors.len() {
            let mon_out = self.monitors[mon_idx].output_id();
            let ids: Vec<WorkspaceId> = self.active_view(&mon_out).ids().to_vec();
            let output = self.monitors[mon_idx].output.clone();
            for id in ids {
                let ws = self
                    .workspaces
                    .get_mut(&id)
                    .expect("workspace id must be a key in the pool");
                for win in ws.windows_mut() {
                    f(win, Some(&output));
                }
            }
        }
        for id in &self.disconnected_workspace_ids {
            let ws = self
                .workspaces
                .get_mut(id)
                .expect("workspace id must be a key in the pool");
            for win in ws.windows_mut() {
                f(win, None);
            }
        }
    }

    /// Cross-activity twin of [`Self::with_windows`]. Iterates every
    /// workspace in the pool directly (not via the active activity's views),
    /// yielding the interactive-move window first. For each window the
    /// closure receives the owning workspace id (`None` only for the
    /// interactive-move tile, which is not attached to a workspace) and
    /// the bound `&Output` (`None` for pool workspaces whose bound output
    /// is disconnected or who have no bound output).
    ///
    /// Iteration order among pool workspaces is pool order (undefined —
    /// the pool is a `HashMap<WorkspaceId, Workspace<W>>`). Callers that
    /// need stable order must sort.
    pub fn with_windows_all(
        &self,
        mut f: impl FnMut(&W, Option<&Output>, Option<WorkspaceId>, WindowLayout),
    ) {
        if let Some(InteractiveMoveState::Moving(move_)) = &self.interactive_move {
            let layout = move_.tile.ipc_layout_template();
            f(move_.tile.window(), Some(&move_.output), None, layout);
        }

        for ws in self.workspaces.values() {
            let output = ws
                .output_id()
                .and_then(|oid| self.monitor_for_output_id(oid))
                .map(|m| &m.output);
            for (tile, layout) in ws.tiles_with_ipc_layouts() {
                f(tile.window(), output, Some(ws.id()), layout);
            }
        }
    }

    /// Cross-activity twin of [`Self::with_windows_mut`]. Same iteration
    /// shape as [`Self::with_windows_all`] but passes each window by
    /// `&mut W`. The interactive-move window is yielded first.
    ///
    /// Borrow recipe: the monitor lookup (`&Output` for each workspace)
    /// is pre-hoisted into an `OutputId → Output` map so the closure can
    /// hold `&mut self.workspaces` without conflicting with an immutable
    /// borrow of `self.monitors`. `Output` is `Clone` and cheap (wraps an
    /// `Arc`); this fires at the existing `refresh_mapped_cast_*` cadence.
    pub fn with_windows_all_mut(&mut self, mut f: impl FnMut(&mut W, Option<&Output>)) {
        if let Some(InteractiveMoveState::Moving(move_)) = &mut self.interactive_move {
            f(move_.tile.window_mut(), Some(&move_.output));
        }

        let mon_by_id: HashMap<OutputId, Output> = self
            .monitors
            .iter()
            .map(|m| (m.output_id(), m.output.clone()))
            .collect();

        for ws in self.workspaces.values_mut() {
            let output = ws.output_id().and_then(|oid| mon_by_id.get(oid));
            for win in ws.windows_mut() {
                f(win, output);
            }
        }
    }

    pub fn active_monitor_ref(&self) -> Option<&Monitor<W>> {
        if self.monitors.is_empty() {
            return None;
        }
        Some(&self.monitors[self.active_monitor_idx])
    }

    /// Borrow the workspace pool.
    pub fn workspace_pool(&self) -> &HashMap<WorkspaceId, Workspace<W>> {
        &self.workspaces
    }

    /// Diagnostic counts for the latency canary: `(pool size, total
    /// workspaces summed across every connected monitor's active-activity
    /// view, disconnected-pool size)`. These are the three quantities that
    /// the per-input-event `advance_animations` walk iterates, so a
    /// pathological count here localizes an input-latency regression. Cheap
    /// (`HashMap::len` plus one `WorkspaceView::ids().len()` per monitor);
    /// safe to call on the slow path only.
    pub fn latency_debug_counts(&self) -> (usize, usize, usize) {
        let pool = self.workspaces.len();
        let active_view = self
            .monitors
            .iter()
            .map(|m| self.active_view(&m.output_id()).ids().len())
            .sum();
        let disconnected = self.disconnected_workspace_ids.len();
        (pool, active_view, disconnected)
    }

    /// Id of the currently-active [`Activity`]. Callers that need to stamp a freshly-built
    /// workspace with the active activity (e.g. reloading output config via
    /// [`Monitor::update_layout_config`]) capture this before taking a split-borrow of the
    /// workspace pool.
    pub(crate) fn active_activity_id(&self) -> ActivityId {
        self.activities.active_id()
    }

    /// Borrow the activities pool (read-only).
    ///
    /// Use this when you need the full pool (e.g. IPC projections that iterate
    /// all activities). For single-activity lookups prefer the narrow helpers
    /// like [`Self::active_activity_id`] or `resolve_activity_ref`.
    pub(crate) fn activities(&self) -> &Activities {
        &self.activities
    }

    /// Read-only access to the curated bookmark state, for the IPC read surface.
    pub(crate) fn bookmarks(&self) -> &Bookmarks<W::Id> {
        &self.bookmarks
    }

    /// Aggregate urgency for an activity: `true` iff some workspace in the pool
    /// whose `activities` set contains `id` is itself urgent ( bubble:
    /// window → workspace → activity). Returns `false` for an unknown
    /// `ActivityId` (silent no-match — no workspaces will satisfy the filter).
    pub(crate) fn activity_is_urgent(&self, id: ActivityId) -> bool {
        self.workspaces
            .values()
            .filter(|ws| ws.activities().contains(&id))
            .any(|ws| ws.is_urgent())
    }

    /// Returns `Some(true)` if `wl_surface` belongs to a window whose workspace
    /// is in the currently-active activity; `Some(false)` if it belongs to a
    /// window on a workspace in a different (hidden) activity; `None` if no
    /// pool workspace owns the surface (destroyed, in-flight interactive move,
    /// never mapped).
    ///
    /// Caller is responsible for resolving the toplevel root surface first
    /// (typically via `Niri::find_root_shell_surface`) —
    /// [`LayoutElement::is_wl_surface`] only matches the toplevel's root
    /// `WlSurface`, so passing a subsurface or popup surface yields `None`.
    ///
    /// Does not consult `interactive_move`: per, a window being
    /// interactively moved hard-blocks activity switch, so its inhibitor
    /// state doesn't need migration during a switch. If that assumption is
    /// ever relaxed, mirror the `interactive_move` arm from
    /// [`Self::find_window_and_output`] here.
    pub fn is_wl_surface_on_active_activity(&self, wl_surface: &WlSurface) -> Option<bool> {
        let active = self.activities.active_id();
        for ws in self.workspaces.values() {
            if ws.find_wl_surface(wl_surface).is_some() {
                return Some(ws.activities().contains(&active));
            }
        }
        None
    }

    /// Split-borrow helper: return `(&mut monitors, &mut pool)` for external callers that iterate
    /// monitors and call mutating `Monitor` methods threading `&mut pool`. Returns `(&mut [], ...)`
    /// if no outputs are connected.
    pub fn monitors_and_pool_mut(
        &mut self,
    ) -> (&mut [Monitor<W>], &mut HashMap<WorkspaceId, Workspace<W>>) {
        (&mut self.monitors[..], &mut self.workspaces)
    }

    /// Triple-split helper: return `(&mut monitors, &mut pool, &mut view)` for the view
    /// keyed by `output_id` in the active activity. Mirrors [`Self::monitors_and_pool_mut`]
    /// but also hands out the active `WorkspaceView` for `output_id`.
    ///
    /// Panics if `output_id` has no entry in the active activity's views — that is a broken
    /// cross-field invariant, not user input.
    pub fn monitors_pool_view_mut(
        &mut self,
        output_id: &OutputId,
    ) -> (
        &mut [Monitor<W>],
        &mut HashMap<WorkspaceId, Workspace<W>>,
        &mut WorkspaceView,
    ) {
        let view = self
            .activities
            .active_mut()
            .views_mut()
            .get_mut(output_id)
            .expect("connected output must have a view in the active activity");
        (&mut self.monitors[..], &mut self.workspaces, view)
    }

    /// Remove the workspace at `view_idx` from the monitor at `mon_idx`, unbind it from the
    /// output, and return its id. The workspace value remains in `self.workspaces` under that id —
    /// caller decides whether to re-attach it to another monitor or drop it.
    fn remove_workspace_from_monitor(
        &mut self,
        mon_idx: usize,
        mut view_idx: usize,
    ) -> WorkspaceId {
        let seed_activity = self.activities.active_id();
        let mon_out = self.monitors[mon_idx].output_id();

        if view_idx == self.active_view(&mon_out).len() - 1 {
            let (monitors, pool, view) = self.monitors_pool_view_mut(&mon_out);
            Self::add_workspace_bottom_on(monitors, pool, view, mon_idx, seed_activity);
        }
        if self.monitors[mon_idx]
            .options
            .layout
            .empty_workspace_above_first
            && view_idx == 0
        {
            let (monitors, pool, view) = self.monitors_pool_view_mut(&mon_out);
            Self::add_workspace_top_on(monitors, pool, view, mon_idx, seed_activity);
            view_idx += 1;
        }

        // For monitor current workspace removal, we focus previous rather than next (<= rather
        // than <). This is different from columns and tiles, but it lets move-workspace-to-monitor
        // back and forth to preserve position. `WorkspaceView::remove_at` enforces this rule.
        let id = self.active_view(&mon_out).ids()[view_idx];
        self.active_view_mut(&mon_out).remove_at(view_idx);

        let ids_to_destroy = {
            let (monitors, pool, view) = self.monitors_pool_view_mut(&mon_out);
            let mon = &mut monitors[mon_idx];
            pool.get_mut(&id)
                .expect("view id must be a key in the pool")
                .unbind_output(&mon.output);

            mon.workspace_switch = None;
            Self::clean_up_workspaces_on(monitors, pool, view, mon_idx)
        };
        // `remove_at(view_idx)` already removed `id` from the view before
        // `clean_up_workspaces_on` ran, so the pruner never sees it.
        debug_assert!(
            !ids_to_destroy.contains(&id),
            "clean_up must not prune the extracted workspace id",
        );
        Self::destroy_workspaces_cross_activity(
            &mut self.activities,
            &mut self.workspaces,
            ids_to_destroy,
        );

        id
    }

    /// Attach an existing pool-held workspace to the monitor at `mon_idx` at `view_idx`.
    ///
    /// `id` must already be a key in `self.workspaces`. Binds the workspace to this monitor's
    /// output, refreshes its config, inserts it into the view (adding a top empty bookend first if
    /// `empty_workspace_above_first` is on), optionally activates it, clears any in-flight
    /// `workspace_switch`, and runs `clean_up_workspaces`.
    fn insert_workspace_onto_monitor(
        &mut self,
        mon_idx: usize,
        id: WorkspaceId,
        mut view_idx: usize,
        activate: bool,
    ) {
        let seed_activity = self.activities.active_id();
        let mon_out = self.monitors[mon_idx].output_id();
        let ids_to_destroy = {
            let (monitors, pool, view) = self.monitors_pool_view_mut(&mon_out);

            {
                let mon = &monitors[mon_idx];
                let ws = pool
                    .get_mut(&id)
                    .expect("workspace id must be a key in the pool");
                ws.bind_output(&mon.output);
                ws.update_config(mon.options.clone());
            }

            // Don't insert past the last empty workspace.
            if view_idx == view.len() {
                view_idx -= 1;
            }
            if view_idx == 0 && monitors[mon_idx].options.layout.empty_workspace_above_first {
                // Insert a new empty workspace on top to prepare for insertion of new workspace.
                Self::add_workspace_top_on(monitors, pool, view, mon_idx, seed_activity);
                view_idx += 1;
            }

            let mon = &mut monitors[mon_idx];
            view.insert(view_idx, id);

            if activate {
                mon.workspace_switch = None;
                mon.activate_workspace(view, view_idx);
            }

            mon.workspace_switch = None;
            Self::clean_up_workspaces_on(monitors, pool, view, mon_idx)
        };
        Self::destroy_workspaces_cross_activity(
            &mut self.activities,
            &mut self.workspaces,
            ids_to_destroy,
        );
    }

    /// Attach a list of existing pool-held workspaces to the monitor at `mon_idx`, in order, just
    /// above the bottom empty workspace.
    ///
    /// All `workspace_ids` must already be keys in `self.workspaces`.
    fn append_workspaces_to_monitor(&mut self, mon_idx: usize, workspace_ids: Vec<WorkspaceId>) {
        if workspace_ids.is_empty() {
            return;
        }

        let seed_activity = self.activities.active_id();
        let mon_out = self.monitors[mon_idx].output_id();
        let ids_to_destroy = {
            let (monitors, pool, view) = self.monitors_pool_view_mut(&mon_out);

            {
                let mon = &monitors[mon_idx];
                for id in &workspace_ids {
                    let ws = pool
                        .get_mut(id)
                        .expect("workspace id must be a key in the pool");
                    ws.bind_output(&mon.output);
                    ws.update_config(mon.options.clone());
                }
            }

            let mon = &mut monitors[mon_idx];
            let empty_was_focused = view.active_position() == view.len() - 1;

            // Insert in place so the view stays non-empty at every step
            // (`WorkspaceView` requires at least one id).
            let start = view.len() - 1;
            for (offset, id) in workspace_ids.into_iter().enumerate() {
                let insert_pos = start + offset;
                view.insert(insert_pos, id);
            }

            // If empty_workspace_above_first is set and the first workspace is now no longer empty,
            // add a new empty workspace on top.
            if mon.options.layout.empty_workspace_above_first
                && mon.workspace_at(pool, view, 0).has_windows_or_name()
            {
                Self::add_workspace_top_on(monitors, pool, view, mon_idx, seed_activity);
            }

            let mon = &mut monitors[mon_idx];
            // If the empty workspace was focused on the primary monitor, keep it focused.
            // Use `set_active_at` (not `activate`) so `previous` isn't clobbered — this is
            // an output reshuffle, not a user-visible workspace switch.
            if empty_was_focused {
                let last = view.len() - 1;
                view.set_active_at(last);
            }

            // FIXME: if we're adding workspaces to currently invisible positions
            // (outside the workspace switch), we don't need to cancel it.
            mon.workspace_switch = None;
            Self::clean_up_workspaces_on(monitors, pool, view, mon_idx)
        };
        Self::destroy_workspaces_cross_activity(
            &mut self.activities,
            &mut self.workspaces,
            ids_to_destroy,
        );
    }

    /// Detach the workspaces owned by `monitor` from its output.
    ///
    /// Non-empty workspaces stay in the pool with `output` unbound; empty unnamed
    /// workspaces (typically the bookends added by `Monitor::new`) are accumulated into
    /// the second tuple element rather than removed inline, so the caller can flush
    /// them through [`Layout::destroy_workspaces_cross_activity`] once any remaining
    /// `&mut self` work has finished.
    ///
    /// Returns `(kept, doomed)`. The caller MUST flush `doomed` through
    /// [`Layout::destroy_workspaces_cross_activity`] before the next `refresh`; until
    /// then both the pool and every activity view referencing those ids are out of sync.
    ///
    /// Used when the output is disconnecting and `monitor` has already been removed
    /// from `self.monitors`.
    fn take_workspace_ids(
        &mut self,
        monitor: &Monitor<W>,
        view: &WorkspaceView,
    ) -> (Vec<WorkspaceId>, Vec<WorkspaceId>) {
        let pool = &mut self.workspaces;
        let mut kept = Vec::with_capacity(view.ids().len());
        let mut doomed: Vec<WorkspaceId> = Vec::new();
        for id in view.ids() {
            let ws = pool
                .get_mut(id)
                .expect("monitor ids must be keys in the pool");
            if ws.has_windows_or_name() {
                ws.unbind_output(&monitor.output);
                kept.push(*id);
            } else {
                doomed.push(*id);
            }
        }
        (kept, doomed)
    }

    // --- Row-2 leaf structural methods (workspace insert/remove/cleanup).
    //
    // These are associated functions (not `&mut self`) because the row-1/row-3 outer methods
    // below call them from within a `(monitors, pool)` split-borrow. Making them `&mut self`
    // would force borrow-checker conflicts at every call site. External callers grab the split
    // via `monitors_and_pool_mut()`.

    fn add_workspace_at_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        view: &mut WorkspaceView,
        mon_idx: usize,
        idx: usize,
        seed_activity: ActivityId,
    ) {
        let mon = &mut monitors[mon_idx];
        let ws = Workspace::new(
            &mon.output,
            HashSet::from([seed_activity]),
            mon.clock.clone(),
            mon.options.clone(),
        );

        let id = ws.id();
        assert!(pool.insert(id, ws).is_none(), "fresh id must be unique");
        view.insert(idx, id);

        if let Some(switch) = &mut mon.workspace_switch {
            if idx as f64 <= switch.target_idx() {
                switch.offset(1);
            }
        }
    }

    fn add_workspace_top_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        view: &mut WorkspaceView,
        mon_idx: usize,
        seed_activity: ActivityId,
    ) {
        Self::add_workspace_at_on(monitors, pool, view, mon_idx, 0, seed_activity);
    }

    fn add_workspace_bottom_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        view: &mut WorkspaceView,
        mon_idx: usize,
        seed_activity: ActivityId,
    ) {
        let len = view.len();
        Self::add_workspace_at_on(monitors, pool, view, mon_idx, len, seed_activity);
    }

    /// Insert an existing pool workspace into `view`, keeping a trailing-empty
    /// bookend at the tail.
    ///
    /// A windowed-or-named workspace lands just above a trailing empty unnamed
    /// workspace, reusing it as the bookend — appending past it would strand
    /// that empty mid-view (a stray empty workspace above the inserted one
    /// until the focus-driven cleanup reaps it) and force a redundant fresh
    /// mint. An empty unnamed workspace appends at the tail: it is itself a
    /// valid bookend. A view whose tail is windowed-or-named (a disconnected
    /// output's view, where the bookend rule is not maintained) also appends
    /// at the tail.
    ///
    /// Returns the insertion position so callers patching the view the monitor
    /// is rendering can shift an in-flight workspace switch.
    fn view_insert_above_trailing_bookend(
        pool: &HashMap<WorkspaceId, Workspace<W>>,
        view: &mut WorkspaceView,
        ws_id: WorkspaceId,
    ) -> usize {
        let windowed_or_named = |id: &WorkspaceId| {
            pool.get(id)
                .expect("view id must be a key in the pool")
                .has_windows_or_name()
        };
        let pos = if windowed_or_named(&ws_id)
            && view
                .ids()
                .last()
                .is_some_and(|last| !windowed_or_named(last))
        {
            view.len() - 1
        } else {
            view.len()
        };
        view.insert(pos, ws_id);
        pos
    }

    /// Prunes empty unnamed workspaces from this monitor's view, returning the
    /// set of `WorkspaceId`s that were dropped from the view but NOT yet
    /// removed from the pool.
    ///
    /// Callers MUST pass the returned `Vec<WorkspaceId>` through
    /// [`Layout::destroy_workspaces_cross_activity`] before the next
    /// `refresh` — otherwise the pool will keep workspaces that no activity's
    /// view references, and `verify_invariants` will trip at the next refresh.
    #[must_use]
    fn clean_up_workspaces_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        view: &mut WorkspaceView,
        mon_idx: usize,
    ) -> Vec<WorkspaceId> {
        let mon = &mut monitors[mon_idx];
        assert!(mon.workspace_switch.is_none());

        let mut pruned = Vec::new();

        let range_start = if mon.options.layout.empty_workspace_above_first {
            1
        } else {
            0
        };
        for idx in (range_start..view.len() - 1).rev() {
            if view.active_position() == idx {
                continue;
            }

            if !mon.workspace_at(pool, view, idx).has_windows_or_name() {
                let id = view.ids()[idx];
                // A shared workspace is pinned in every member activity's view; pruning it
                // here while destroy_workspaces_cross_activity skips the pool removal for
                // shared ids would orphan it (in the pool but eventually in no view). Skip
                // it — the same policy as remove_window.
                if !Self::workspace_is_safe_to_reclaim(pool, id) {
                    let n = pool.get(&id).map_or(0, |ws| ws.activities().len());
                    trace!(
                        "clean_up_workspaces_on: workspace {id:?} emptied but pinned \
                         (shared across {n} activities)",
                    );
                    continue;
                }
                view.remove_at(idx);
                pruned.push(id);
            }
        }

        // Special case handling when empty_workspace_above_first is set and all workspaces
        // are empty.
        if mon.options.layout.empty_workspace_above_first && view.len() == 2 {
            assert!(!mon.workspace_at(pool, view, 0).has_windows_or_name());
            assert!(!mon.workspace_at(pool, view, 1).has_windows_or_name());
            let id = view.ids()[1];
            // A shared second entry is pinned by another activity's view; the length-2 view
            // is the honest minimal EWAF shape when the trailing slot is shared.
            if !Self::workspace_is_safe_to_reclaim(pool, id) {
                let n = pool.get(&id).map_or(0, |ws| ws.activities().len());
                trace!(
                    "clean_up_workspaces_on: workspace {id:?} emptied but pinned \
                     (shared across {n} activities, EWAF trailing slot)",
                );
            } else {
                view.remove_at(1);
                pruned.push(id);
            }
        }

        pruned
    }

    /// Per-id guard matching the skip policy of
    /// [`Layout::destroy_workspaces_cross_activity`] ( shared-workspace
    /// cleanup rule). Returns `true` only when `ws_id` is a live pool key
    /// whose workspace is exclusive to a single activity.
    ///
    /// Returns `false` in two cases:
    /// - `ws_id` is absent from the pool. Caller bug; returning `false` keeps the predicate total
    ///   so callers that gate the whole reclaim block on this predicate (rather than funneling
    ///   through `destroy_workspaces_cross_activity`) don't lose the authoritative panic. A
    ///   `debug_assert!` fires in debug builds to surface caller bugs early.
    /// - `workspace.activities().len() > 1`. Shared membership — another activity still references
    ///   the workspace, so reclaim must be skipped per the contract: "a workspace with `activities
    ///   = {A, B}` that becomes empty is not removed".
    ///
    /// The `len() == 1` form is a deliberate narrowing. The full
    /// rule is "empty AND every activity in its `activities` set has
    /// another, non-empty workspace to fall back to on the same output";
    /// `len() == 1` checks only the membership-exclusivity portion of that
    /// predicate; callers separately enforce the emptiness precondition.
    /// Tightening to the full per-(activity, output) fallback check is
    /// deferred.
    ///
    /// `pub(crate)` so `clean_up_workspaces_on` and other reclaim sites can
    /// share the policy rather than re-encoding the rule.
    pub(crate) fn workspace_is_safe_to_reclaim(
        pool: &HashMap<WorkspaceId, Workspace<W>>,
        ws_id: WorkspaceId,
    ) -> bool {
        let Some(ws) = pool.get(&ws_id) else {
            debug_assert!(
                pool.contains_key(&ws_id),
                "workspace_is_safe_to_reclaim: ws_id {ws_id:?} is absent from the pool (caller bug)",
            );
            return false;
        };
        ws.activities().len() == 1
    }

    /// Locate the output whose view currently holds `ws_id`, scanning every activity's views
    /// (first match wins).
    ///
    /// `Workspace::output_id()` is a reclaim tag, not a location: after a partial disconnect it
    /// can point at a monitor `ws_id` no longer lives on (the migration walk in
    /// [`Self::remove_output`] preserves the tag but relocates the id — see
    /// `Workspace::bind_output`'s rustdoc for why the tag isn't refreshed on transfer). Callers
    /// that need to find the view actually holding `ws_id` must use this resolver instead.
    ///
    /// With any monitor connected, the pool==union invariant checked in
    /// [`Self::verify_invariants`] guarantees every live pool id appears in some activity's view
    /// — a `None` result while `self.monitors` is non-empty is therefore a membership↔view
    /// coherence bug, not a legitimate outcome, and is surfaced via `debug_assert!` (debug-loud,
    /// release-healable, matching [`Self::workspace_is_safe_to_reclaim`]'s philosophy). `None` is
    /// the expected result only in the fully-disconnected window, where every workspace is parked
    /// in `disconnected_workspace_ids` and no activity holds any view.
    ///
    /// First-match semantics only: when `ws_id` is shared across activities, this resolver does
    /// not assert that every holding view agrees on the same output. A shared workspace held
    /// under *different* outputs in different activities is a legal, production-reachable steady
    /// state after a partial disconnect — pinned by
    /// `remove_workspace_from_activity_scopes_to_own_activity_view_after_divergent_partial_disconnect`
    /// and its `set_workspace_activities` sibling in `tests.rs` — so same-output cross-view
    /// agreement is deliberately not asserted anywhere. Separately, `ws_id` held in more than one
    /// view of the *same* activity (across different outputs) is a distinct, not-yet-fixed
    /// incoherence: `verify_invariants`'s per-view uniqueness check only dedupes within a single
    /// `WorkspaceView`, not across an activity's per-output views, so this case remains
    /// unenforced.
    fn workspace_holding_output(&self, ws_id: WorkspaceId) -> Option<OutputId> {
        let holding = self.activities.iter().find_map(|activity| {
            activity
                .views()
                .iter()
                .find_map(|(out_id, view)| view.position_of(ws_id).map(|_| out_id.clone()))
        });
        debug_assert!(
            holding.is_some() || self.monitors.is_empty(),
            "workspace_holding_output: {ws_id:?} not found in any activity's view while a \
             monitor is connected — pool==union invariant violated (membership↔view coherence \
             bug)",
        );
        holding
    }

    /// Reclaim empty unnamed workspaces whose activity-membership set just
    /// shrank to a single activity, restoring the no-empty-middle invariant
    /// for their new exclusive owner.
    ///
    /// Callers pass every workspace id whose `activities` set just shrank.
    /// The helper processes each id independently — classify then mutate —
    /// so an earlier cull cannot stale a later position check.
    ///
    /// Per id:
    ///
    /// 1. Skip unless [`Self::workspace_is_safe_to_reclaim`] returns `true` (still shared → keep
    ///    pinned; absent id → `debug_assert!` fires, skip).
    /// 2. Skip if `has_windows_or_name()` — only empty unnamed candidates need reclaiming.
    /// 3. Locate the holding view via view-keyed resolution: take the sole remaining activity id,
    ///    scan that activity's views for one that contains `ws_id`, capture `(OutputId, pos)`.
    ///    Using `Workspace::output_id()` here is forbidden — that field is a deliberately-stale
    ///    reclaim tag after partial-disconnect, so it may not match any live view key.
    /// 4. Resolve the EWAF flag from the connected monitor when the view key matches a monitor's
    ///    `output_id()`; fall back to the layout-root option otherwise (dormant / disconnected
    ///    views), matching `assert_view_bookends`' documented source-of-truth split.
    /// 5. Legality decision (check order is load-bearing):
    ///    - **Cull** if `ewaf && view.len() == 2 && pos == 1` — the all-empty EWAF len-2 collapse;
    ///      this carve-out overrides the trailing-slot and active-position keeps that follow.
    ///    - **Keep** if the workspace is in a protected position: sole entry (`view.len() == 1`),
    ///      trailing bookend (`pos == view.len() - 1`), EWAF leading bookend (`ewaf && pos == 0`),
    ///      or the active / remembered- active position (`pos == view.active_position()`).
    ///    - **Cull** otherwise — an illegal middle entry, whether the view is active today or
    ///      dormant (a delayed-bomb on the next switch).
    /// 6. Before culling into an active view that is mid-animation: snap that monitor's
    ///    `workspace_switch` to `None` first. The mutation is scoped to the one (activity, output)
    ///    view that holds the workspace, so only the holding monitor is snapped — a narrower
    ///    footprint than the two methods that snap all monitors unconditionally.
    /// 7. Cull via [`Self::destroy_workspaces_cross_activity`], which patches every view before
    ///    removing the pool entry (ordering load-bearing per its rustdoc).
    fn reclaim_unpinned_empty_workspaces(
        &mut self,
        narrowed: impl IntoIterator<Item = WorkspaceId>,
    ) {
        for ws_id in narrowed {
            // Step 1: skip still-shared ids; debug_assert! fires on absent ids.
            if !Self::workspace_is_safe_to_reclaim(&self.workspaces, ws_id) {
                let n = self
                    .workspaces
                    .get(&ws_id)
                    .map_or(0, |ws| ws.activities().len());
                trace!(
                    "reclaim_unpinned_empty_workspaces: workspace {ws_id:?} still shared \
                     across {n} activities, skipping",
                );
                continue;
            }

            // Step 2: skip workspaces that have content.
            {
                let ws = self
                    .workspaces
                    .get(&ws_id)
                    .expect("workspace_is_safe_to_reclaim returned true → id is a live pool key");
                if ws.has_windows_or_name() {
                    continue;
                }
            }

            // Step 3: view-keyed resolution. Take the sole remaining activity,
            // scan its views for the one holding ws_id. ws.output_id() is
            // intentionally not used — it may be a stale reclaim tag after
            // partial-disconnect (the field is not updated when a workspace
            // migrates between monitors during a reconnect).
            let sole_activity_id = {
                let ws = self
                    .workspaces
                    .get(&ws_id)
                    .expect("id is a live pool key after step 2");
                *ws.activities()
                    .iter()
                    .next()
                    .expect("workspace_is_safe_to_reclaim guarantees activities().len() == 1")
            };

            let holding = self
                .activities
                .get(sole_activity_id)
                .expect("sole_activity_id came from ws.activities() — must be a live activity")
                .views()
                .iter()
                .find_map(|(out_id, view)| {
                    view.position_of(ws_id).map(|pos| (out_id.clone(), pos))
                });

            let (view_key, pos) = match holding {
                Some(pair) => pair,
                None => {
                    // Defensive: ws_id is in the pool and in no view of its sole
                    // remaining activity. This is a membership↔view coherence gap
                    // that a separate cleanup path is responsible for. Skip
                    // without panicking so this helper stays narrowly scoped.
                    trace!(
                        "reclaim_unpinned_empty_workspaces: workspace {ws_id:?} not found \
                         in any view of activity {sole_activity_id:?}, skipping",
                    );
                    continue;
                }
            };

            // Step 4: resolve EWAF. Connected monitors use their per-monitor
            // merged options; views on disconnected / dormant outputs fall back
            // to the layout-root option. This mirrors assert_view_bookends'
            // documented source-of-truth split (monitor.rs:1994-1996).
            let ewaf = self
                .monitors
                .iter()
                .find(|m| m.output_id() == view_key)
                .map(|m| m.options.layout.empty_workspace_above_first)
                .unwrap_or(self.options.layout.empty_workspace_above_first);

            // Re-borrow view for the legality check. The sole_activity_id borrow
            // above ended; re-enter with a shared borrow.
            let (view_len, active_pos) = {
                let view = self
                    .activities
                    .get(sole_activity_id)
                    .expect("sole_activity_id is still live")
                    .views()
                    .get(&view_key)
                    .expect("view_key was found in the scan above");
                (view.len(), view.active_position())
            };

            // Step 5: legality decision. Check order is load-bearing: the EWAF
            // len-2 cull must override the trailing-slot keep, since pos == 1 ==
            // view.len() - 1 when len == 2.
            let should_cull = if ewaf && view_len == 2 && pos == 1 {
                // All-empty EWAF len-2 collapse: the single non-leading entry is
                // no longer a valid bookend anchor for a shared workspace.
                true
            } else if view_len == 1
                || pos == view_len - 1
                || (ewaf && pos == 0)
                || pos == active_pos
            {
                // Protected positions: sole entry, trailing bookend, EWAF leading
                // bookend, or the active / remembered-active position.
                false
            } else {
                // Illegal middle: either an active-view violation (no-empty-middle
                // assert at monitor.rs:1931) or a dormant-view delayed-bomb (the
                // assert_view_bookends check is not animation-gated, so it fires
                // on the next switch into this activity).
                true
            };

            if !should_cull {
                continue;
            }

            // Step 6: snap the holding monitor's animation before mutating its
            // active view. assert_view_bookends (called from verify_invariants
            // for every view including dormant ones) is not animation-gated —
            // only the no-empty-middle check at monitor.rs:1921 gates on
            // workspace_switch.is_none(). Snapping here prevents
            // clean_up_workspaces_on's assert!(mon.workspace_switch.is_none())
            // from firing if the caller-level animate path runs next.
            // Only the holding monitor is snapped because the mutation is
            // provably scoped to this one (activity, output) view; snapping all
            // monitors (the precedent at mod.rs:7014-7020 and :6860-6866) is
            // unnecessarily broad for a per-id mutation.
            if sole_activity_id == self.activities.active_id() {
                for mon in &mut self.monitors {
                    if mon.output_id() == view_key
                        && matches!(mon.workspace_switch, Some(WorkspaceSwitch::Animation(_)))
                    {
                        mon.workspace_switch = None;
                    }
                }
            }

            // Step 7: cull. destroy_workspaces_cross_activity patches every
            // activity's views before the pool removal — ordering is load-bearing
            // per its rustdoc.
            Self::destroy_workspaces_cross_activity(
                &mut self.activities,
                &mut self.workspaces,
                [ws_id],
            );
        }
    }

    /// Drops workspaces from the pool after patching every activity's
    /// `WorkspaceView` that still references them. The retain closure mirrors
    /// `RemoveActivity`'s exclusive-workspace destruction pass: single-entry
    /// views are dropped entirely (a `WorkspaceView` cannot be zero-sized),
    /// multi-entry views get `remove_at`-patched so `active`/`previous` stay
    /// coherent.
    ///
    /// **Shared-id skip policy.** Each id is checked at loop entry:
    /// if the pool contains a workspace with `activities().len() > 1` (shared
    /// across activities), that id is `warn!`-logged and skipped — both the
    /// per-activity retain sweep and the `pool.remove` assert are bypassed.
    /// Skipping rather than panicking keeps a batched destroy recoverable when
    /// a caller passes a shared id; the `warn!` ensures the skip is not silent.
    /// Shared workspaces are constructible (`set_workspace_activities`, sticky
    /// expansion), so the skip is load-bearing, not defensive.
    ///
    /// Absent ids (not present in the pool at all) are a caller bug; they are
    /// not skipped here and fall through to the `pool.remove` assert, which is
    /// the authoritative panic for genuinely dead ids.
    ///
    /// **Ordering is load-bearing:** every activity's view is patched before
    /// `pool.remove` for each id, so that debug assertions and
    /// `verify_invariants` always see pool and views in agreement. Do not
    /// hoist the `pool.remove` calls out of the per-id loop.
    #[track_caller]
    fn destroy_workspaces_cross_activity(
        activities: &mut Activities,
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        ids: impl IntoIterator<Item = WorkspaceId>,
    ) {
        for ws_id in ids {
            if let Some(ws) = pool.get(&ws_id) {
                if ws.activities().len() > 1 {
                    warn!(
                        "destroy_workspaces_cross_activity: skipping shared id {ws_id:?} \
                         (activities.len() = {})",
                        ws.activities().len(),
                    );
                    continue;
                }
            }
            for activity in activities.iter_mut() {
                activity.views_mut().retain(|_output_id, view| {
                    let Some(pos) = view.position_of(ws_id) else {
                        return true;
                    };
                    if view.len() == 1 {
                        return false;
                    }
                    view.remove_at(pos);
                    true
                });
            }
            assert!(
                pool.remove(&ws_id).is_some(),
                "destroy id {ws_id:?} must be a live pool key",
            );
        }
    }

    // --- Row-1 / Row-3 outer methods: add-tile family and movement between workspaces.
    //
    // Associated functions for the same reason as the row-2 leaves: they call each other (and
    // the row-2 leaves) mutually, so `&mut self` would re-borrow `self` inside a split-borrow
    // scope.

    #[allow(clippy::too_many_arguments)]
    fn add_window_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        view: &mut WorkspaceView,
        mon_idx: usize,
        window: W,
        target: MonitorAddWindowTarget<W>,
        activate: ActivateWindow,
        width: ColumnWidth,
        is_full_width: bool,
        is_floating: bool,
        seed_activity: ActivityId,
    ) {
        let mon = &monitors[mon_idx];
        let (workspace_idx, target) = mon.resolve_add_window_target(pool, view, target);
        let tile = mon
            .workspace_at(pool, view, workspace_idx)
            .make_tile(window);

        Self::add_resolved_tile_on(
            monitors,
            pool,
            view,
            mon_idx,
            workspace_idx,
            tile,
            target,
            activate,
            true,
            width,
            is_full_width,
            is_floating,
            seed_activity,
        );
    }

    fn add_column_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        view: &mut WorkspaceView,
        mon_idx: usize,
        workspace_idx: usize,
        column: Column<W>,
        activate: bool,
    ) {
        let mon = &mut monitors[mon_idx];
        let workspace = mon.workspace_at_mut(pool, view, workspace_idx);

        workspace.add_column(Some(&mon.output), column, activate);

        // After adding a new window, workspace becomes this output's own.
        if workspace.name().is_none() {
            workspace.output_id = Some(OutputId::new(&mon.output));
        }

        if activate {
            monitors[mon_idx].activate_workspace(view, workspace_idx);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn add_tile_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        view: &mut WorkspaceView,
        mon_idx: usize,
        tile: Tile<W>,
        target: MonitorAddWindowTarget<W>,
        activate: ActivateWindow,
        // FIXME: Refactor ActivateWindow enum to make this better.
        allow_to_activate_workspace: bool,
        width: ColumnWidth,
        is_full_width: bool,
        is_floating: bool,
        seed_activity: ActivityId,
    ) {
        let (workspace_idx, target) =
            monitors[mon_idx].resolve_add_window_target(pool, view, target);
        Self::add_resolved_tile_on(
            monitors,
            pool,
            view,
            mon_idx,
            workspace_idx,
            tile,
            target,
            activate,
            allow_to_activate_workspace,
            width,
            is_full_width,
            is_floating,
            seed_activity,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn add_resolved_tile_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        view: &mut WorkspaceView,
        mon_idx: usize,
        mut workspace_idx: usize,
        tile: Tile<W>,
        target: WorkspaceAddWindowTarget<W>,
        activate: ActivateWindow,
        // FIXME: Refactor ActivateWindow enum to make this better.
        allow_to_activate_workspace: bool,
        width: ColumnWidth,
        is_full_width: bool,
        is_floating: bool,
        seed_activity: ActivityId,
    ) {
        let mon = &mut monitors[mon_idx];
        let workspace = mon.workspace_at_mut(pool, view, workspace_idx);

        workspace.add_tile(
            Some(&mon.output),
            tile,
            target,
            activate,
            width,
            is_full_width,
            is_floating,
        );

        // After adding a new window, workspace becomes this output's own.
        if workspace.name().is_none() {
            workspace.output_id = Some(OutputId::new(&mon.output));
        }

        if workspace_idx == view.len() - 1 {
            // Insert a new empty workspace.
            Self::add_workspace_bottom_on(monitors, pool, view, mon_idx, seed_activity);
        }

        if monitors[mon_idx].options.layout.empty_workspace_above_first && workspace_idx == 0 {
            Self::add_workspace_top_on(monitors, pool, view, mon_idx, seed_activity);
            workspace_idx += 1;
        }

        if allow_to_activate_workspace && activate.map_smart(|| false) {
            monitors[mon_idx].activate_workspace(view, workspace_idx);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn add_tile_to_column_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        view: &mut WorkspaceView,
        mon_idx: usize,
        workspace_idx: usize,
        column_idx: usize,
        tile_idx: Option<usize>,
        tile: Tile<W>,
        activate: bool,
        // FIXME: Refactor ActivateWindow enum to make this better.
        allow_to_activate_workspace: bool,
    ) {
        let mon = &mut monitors[mon_idx];
        let workspace = mon.workspace_at_mut(pool, view, workspace_idx);

        workspace.add_tile_to_column(Some(&mon.output), column_idx, tile_idx, tile, activate);

        // After adding a new window, workspace becomes this output's own.
        if workspace.name().is_none() {
            workspace.output_id = Some(OutputId::new(&mon.output));
        }

        // Since we're adding window to an existing column, the workspace isn't empty, and
        // therefore cannot be the last one, so we never need to insert a new empty workspace.

        if allow_to_activate_workspace && activate {
            mon.activate_workspace(view, workspace_idx);
        }
    }

    fn move_down_or_to_workspace_down_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        view: &mut WorkspaceView,
        mon_idx: usize,
        seed_activity: ActivityId,
    ) {
        if !monitors[mon_idx].active_workspace(pool, view).move_down() {
            Self::move_to_workspace_down_on(monitors, pool, view, mon_idx, true, seed_activity);
        }
    }

    fn move_up_or_to_workspace_up_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        view: &mut WorkspaceView,
        mon_idx: usize,
        seed_activity: ActivityId,
    ) {
        if !monitors[mon_idx].active_workspace(pool, view).move_up() {
            Self::move_to_workspace_up_on(monitors, pool, view, mon_idx, true, seed_activity);
        }
    }

    fn focus_window_or_workspace_down_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        view: &mut WorkspaceView,
        mon_idx: usize,
    ) {
        if !monitors[mon_idx].active_workspace(pool, view).focus_down() {
            monitors[mon_idx].switch_workspace_down(view);
        }
    }

    fn focus_window_or_workspace_up_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        view: &mut WorkspaceView,
        mon_idx: usize,
    ) {
        if !monitors[mon_idx].active_workspace(pool, view).focus_up() {
            monitors[mon_idx].switch_workspace_up(view);
        }
    }

    fn move_to_workspace_up_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        view: &mut WorkspaceView,
        mon_idx: usize,
        focus: bool,
        seed_activity: ActivityId,
    ) {
        let mon = &mut monitors[mon_idx];
        let source_workspace_idx = view.active_position();

        let new_idx = source_workspace_idx.saturating_sub(1);
        if new_idx == source_workspace_idx {
            return;
        }
        let new_id = view.ids()[new_idx];

        let workspace = mon.workspace_at_mut(pool, view, source_workspace_idx);
        let Some(removed) = workspace.remove_active_tile(Some(&mon.output), Transaction::new())
        else {
            return;
        };

        let activate = if focus {
            ActivateWindow::Yes
        } else {
            ActivateWindow::Smart
        };

        Self::add_tile_on(
            monitors,
            pool,
            view,
            mon_idx,
            removed.tile,
            MonitorAddWindowTarget::Workspace {
                id: new_id,
                column_idx: None,
            },
            activate,
            true,
            removed.width,
            removed.is_full_width,
            removed.is_floating,
            seed_activity,
        );
    }

    fn move_to_workspace_down_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        view: &mut WorkspaceView,
        mon_idx: usize,
        focus: bool,
        seed_activity: ActivityId,
    ) {
        let mon = &mut monitors[mon_idx];
        let source_workspace_idx = view.active_position();

        let new_idx = min(source_workspace_idx + 1, view.len() - 1);
        if new_idx == source_workspace_idx {
            return;
        }
        let new_id = view.ids()[new_idx];

        let workspace = mon.workspace_at_mut(pool, view, source_workspace_idx);
        let Some(removed) = workspace.remove_active_tile(Some(&mon.output), Transaction::new())
        else {
            return;
        };

        let activate = if focus {
            ActivateWindow::Yes
        } else {
            ActivateWindow::Smart
        };

        Self::add_tile_on(
            monitors,
            pool,
            view,
            mon_idx,
            removed.tile,
            MonitorAddWindowTarget::Workspace {
                id: new_id,
                column_idx: None,
            },
            activate,
            true,
            removed.width,
            removed.is_full_width,
            removed.is_floating,
            seed_activity,
        );
    }

    /// Resolve which of the two bookend slots (if any) a workspace-insertion
    /// call should reuse instead of inserting a fresh workspace at
    /// `insert_idx`.
    ///
    /// This is the canonical statement of the bookend-reuse rules shared by
    /// `move_to_new_workspace_up_on`, `move_to_new_workspace_down_on`,
    /// `add_workspace_up_on`, `add_workspace_down_on`, and the `NewAt` arm of
    /// `interactive_move_end`'s insertion-target resolution.
    ///
    /// Check order is a contract, not an accident: the EWAF-top check fires
    /// **before** the trailing-reuse check. The two conditions only overlap
    /// when `view_len == 1` (both resolve to slot 0 today), so the order is
    /// observationally moot at present, but a future edit must not swap it.
    ///
    /// # Panics
    ///
    /// Debug-underflow-panics if `view_len == 0`. Callers must guarantee
    /// `view_len >= 1`, which the trailing-bookend invariant provides for any
    /// connected monitor's view; `view_len - 1` deliberately keeps this loud
    /// failure rather than silently clamping on invariant breach.
    fn resolve_insert_target(ewaf: bool, insert_idx: usize, view_len: usize) -> BookendResolution {
        if ewaf && insert_idx == 0 {
            BookendResolution::ReuseTop
        } else if view_len - 1 <= insert_idx {
            BookendResolution::ReuseTrailing
        } else {
            BookendResolution::InsertAt(insert_idx)
        }
    }

    /// Remove the active tile from its workspace and place it on a freshly
    /// inserted workspace above the current position.
    ///
    /// The insertion index is `view.active_position()`, so the new workspace
    /// lands above the source; the source shifts down by one after insertion.
    /// At the edges, [`Self::resolve_insert_target`] may reuse a bookend slot
    /// instead of inserting:
    ///
    /// - Top-bookend reuse is structurally unreachable here: `insert_idx == 0` requires the source
    ///   at slot 0, which under ewaf is the forced-empty slot, so `remove_active_tile` bails first.
    ///   Kept for structural symmetry with the other insertion sites.
    /// - Trailing-bookend reuse is unreachable for the up direction because the trailing empty is
    ///   never the focused source.
    ///
    /// If either reuse arm fires on invariant breach with target == source,
    /// the tile re-lands on the source workspace as a new column.
    ///
    /// `focus` controls whether the view follows the window:
    /// `true` → `ActivateWindow::Yes`; `false` → `ActivateWindow::Smart`.
    fn move_to_new_workspace_up_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        view: &mut WorkspaceView,
        mon_idx: usize,
        focus: bool,
        seed_activity: ActivityId,
    ) {
        let mon = &mut monitors[mon_idx];
        let source_idx = view.active_position();
        let insert_idx = source_idx;

        // (1) Remove the active tile first; bail if the workspace is empty.
        let workspace = mon.workspace_at_mut(pool, view, source_idx);
        let Some(removed) = workspace.remove_active_tile(Some(&mon.output), Transaction::new())
        else {
            return;
        };

        // (2) Resolve the target workspace id. Both reuse arms are unreachable
        //     under intact bookend invariants for the up direction (see rustdoc).
        let ewaf = monitors[mon_idx].options.layout.empty_workspace_above_first;
        let target_id = match Self::resolve_insert_target(ewaf, insert_idx, view.len()) {
            BookendResolution::ReuseTop => view.ids()[0],
            BookendResolution::ReuseTrailing => view.ids()[view.len() - 1],
            BookendResolution::InsertAt(idx) => {
                Self::add_workspace_at_on(monitors, pool, view, mon_idx, idx, seed_activity);
                view.ids()[idx]
            }
        };

        // (3) Place the tile on the target workspace using an id-based target so
        //     the up-direction index shift (source is now at insert_idx + 1)
        //     cannot produce a stale-index lookup.
        let activate = if focus {
            ActivateWindow::Yes
        } else {
            ActivateWindow::Smart
        };

        Self::add_tile_on(
            monitors,
            pool,
            view,
            mon_idx,
            removed.tile,
            MonitorAddWindowTarget::Workspace {
                id: target_id,
                column_idx: None,
            },
            activate,
            true,
            removed.width,
            removed.is_full_width,
            removed.is_floating,
            seed_activity,
        );
    }

    /// Remove the active tile from its workspace and place it on a freshly
    /// inserted workspace below the current position.
    ///
    /// The insertion index is `view.active_position() + 1`, so the new
    /// workspace lands directly below the source. At the edges,
    /// [`Self::resolve_insert_target`] may reuse a bookend slot instead of
    /// inserting:
    ///
    /// - Top-bookend reuse is unreachable for the down direction because `insert_idx = source_idx +
    ///   1 >= 1` always. Kept for structural symmetry with the other insertion sites.
    /// - Trailing-bookend reuse is live: the window lands on the existing trailing empty workspace
    ///   at the bottom edge.
    ///
    /// If either reuse arm fires on invariant breach with target == source,
    /// the tile re-lands on the source workspace as a new column.
    ///
    /// `focus` controls whether the view follows the window:
    /// `true` → `ActivateWindow::Yes`; `false` → `ActivateWindow::Smart`.
    fn move_to_new_workspace_down_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        view: &mut WorkspaceView,
        mon_idx: usize,
        focus: bool,
        seed_activity: ActivityId,
    ) {
        let mon = &mut monitors[mon_idx];
        let source_idx = view.active_position();
        let insert_idx = source_idx + 1;

        // (1) Remove the active tile first; bail if the workspace is empty.
        let workspace = mon.workspace_at_mut(pool, view, source_idx);
        let Some(removed) = workspace.remove_active_tile(Some(&mon.output), Transaction::new())
        else {
            return;
        };

        // (2) Resolve the target workspace id. Top-bookend reuse is unreachable
        //     for the down direction; trailing-bookend reuse is live (see rustdoc).
        let ewaf = monitors[mon_idx].options.layout.empty_workspace_above_first;
        let target_id = match Self::resolve_insert_target(ewaf, insert_idx, view.len()) {
            BookendResolution::ReuseTop => view.ids()[0],
            BookendResolution::ReuseTrailing => view.ids()[view.len() - 1],
            BookendResolution::InsertAt(idx) => {
                Self::add_workspace_at_on(monitors, pool, view, mon_idx, idx, seed_activity);
                view.ids()[idx]
            }
        };

        // (3) Place the tile on the target workspace.
        let activate = if focus {
            ActivateWindow::Yes
        } else {
            ActivateWindow::Smart
        };

        Self::add_tile_on(
            monitors,
            pool,
            view,
            mon_idx,
            removed.tile,
            MonitorAddWindowTarget::Workspace {
                id: target_id,
                column_idx: None,
            },
            activate,
            true,
            removed.width,
            removed.is_full_width,
            removed.is_floating,
            seed_activity,
        );
    }

    /// Insert a fresh empty workspace directly above the active workspace and
    /// focus it.
    ///
    /// The insertion index is `view.active_position()`. At the edges,
    /// [`Self::resolve_insert_target`] may reuse a bookend slot instead of
    /// inserting:
    ///
    /// - Top-bookend reuse is live: from the forced-empty slot 0 there is nothing to bail on, so
    ///   the call reaches here with `insert_idx == 0`, reuses the EWAF slot, and focuses it (a
    ///   no-op because it is already active).
    /// - Trailing-bookend reuse is live: focusing the trailing bookend itself produces `insert_idx
    ///   == view.len() - 1`, which collapses to a reuse with an immediate focus of the trailing
    ///   slot.
    /// - Otherwise a fresh workspace is inserted at `insert_idx` and focused.
    fn add_workspace_up_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        view: &mut WorkspaceView,
        mon_idx: usize,
        seed_activity: ActivityId,
    ) {
        let insert_idx = view.active_position();

        // Both reuse arms are live for the up direction (see rustdoc).
        let ewaf = monitors[mon_idx].options.layout.empty_workspace_above_first;
        let target_idx = match Self::resolve_insert_target(ewaf, insert_idx, view.len()) {
            BookendResolution::ReuseTop => 0,
            BookendResolution::ReuseTrailing => view.len() - 1,
            BookendResolution::InsertAt(idx) => {
                Self::add_workspace_at_on(monitors, pool, view, mon_idx, idx, seed_activity);
                idx
            }
        };

        monitors[mon_idx].activate_workspace(view, target_idx);
    }

    /// Insert a fresh empty workspace directly below the active workspace and
    /// focus it.
    ///
    /// The insertion index is `view.active_position() + 1`. At the edges,
    /// [`Self::resolve_insert_target`] may reuse a bookend slot instead of
    /// inserting:
    ///
    /// - Top-bookend reuse is unreachable for the down direction because `insert_idx =
    ///   active_position() + 1 >= 1` always. Kept for structural symmetry with the up variant.
    /// - Trailing-bookend reuse is live in two scenarios: (a) from the last content workspace the
    ///   insert would land on the trailing slot, which collapses to a reuse (same as
    ///   `FocusWorkspaceDown` at the bottom edge); (b) from the trailing bookend itself `insert_idx
    ///   == view.len()-1` and focus stays in place.
    /// - Otherwise a fresh workspace is inserted at `insert_idx` and focused.
    fn add_workspace_down_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        view: &mut WorkspaceView,
        mon_idx: usize,
        seed_activity: ActivityId,
    ) {
        let insert_idx = view.active_position() + 1;

        // Top-bookend reuse is unreachable for the down direction; trailing-bookend
        // reuse is live (see rustdoc).
        let ewaf = monitors[mon_idx].options.layout.empty_workspace_above_first;
        let target_idx = match Self::resolve_insert_target(ewaf, insert_idx, view.len()) {
            BookendResolution::ReuseTop => 0,
            BookendResolution::ReuseTrailing => view.len() - 1,
            BookendResolution::InsertAt(idx) => {
                Self::add_workspace_at_on(monitors, pool, view, mon_idx, idx, seed_activity);
                idx
            }
        };

        monitors[mon_idx].activate_workspace(view, target_idx);
    }

    #[allow(clippy::too_many_arguments)]
    #[must_use]
    fn move_to_workspace_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        view: &mut WorkspaceView,
        mon_idx: usize,
        window: Option<&W::Id>,
        idx: usize,
        activate: ActivateWindow,
        seed_activity: ActivityId,
    ) -> Vec<WorkspaceId> {
        let mon = &mut monitors[mon_idx];
        let source_workspace_idx = if let Some(window) = window {
            view.ids()
                .iter()
                .position(|id| {
                    pool.get(id)
                        .expect("view id must be a key in the pool")
                        .has_window(window)
                })
                .unwrap()
        } else {
            view.active_position()
        };

        let new_idx = min(idx, view.len() - 1);
        if new_idx == source_workspace_idx {
            return Vec::new();
        }
        let new_id = view.ids()[new_idx];

        let active_window_id = mon.active_window(pool, view).map(|win| win.id().clone());
        let activate =
            activate.map_smart(|| window.is_none_or(|win| active_window_id.as_ref() == Some(win)));

        let workspace = mon.workspace_at_mut(pool, view, source_workspace_idx);
        let transaction = Transaction::new();
        let removed = if let Some(window) = window {
            workspace.remove_tile(Some(&mon.output), window, transaction)
        } else if let Some(removed) = workspace.remove_active_tile(Some(&mon.output), transaction) {
            removed
        } else {
            return Vec::new();
        };

        Self::add_tile_on(
            monitors,
            pool,
            view,
            mon_idx,
            removed.tile,
            MonitorAddWindowTarget::Workspace {
                id: new_id,
                column_idx: None,
            },
            if activate {
                ActivateWindow::Yes
            } else {
                ActivateWindow::No
            },
            true,
            removed.width,
            removed.is_full_width,
            removed.is_floating,
            seed_activity,
        );

        if monitors[mon_idx].workspace_switch.is_none() {
            Self::clean_up_workspaces_on(monitors, pool, view, mon_idx)
        } else {
            Vec::new()
        }
    }

    fn move_column_to_workspace_up_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        view: &mut WorkspaceView,
        mon_idx: usize,
        activate: bool,
        seed_activity: ActivityId,
    ) {
        let mon = &mut monitors[mon_idx];
        let source_workspace_idx = view.active_position();

        let new_idx = source_workspace_idx.saturating_sub(1);
        if new_idx == source_workspace_idx {
            return;
        }

        // Check floating status on a shared borrow first so we can recurse into the sibling method
        // without a `&mut pool` conflict.
        if mon
            .workspace_at(pool, view, source_workspace_idx)
            .floating_is_active()
        {
            Self::move_to_workspace_up_on(monitors, pool, view, mon_idx, activate, seed_activity);
            return;
        }

        let workspace = mon.workspace_at_mut(pool, view, source_workspace_idx);
        let Some(column) = workspace.remove_active_column(Some(&mon.output)) else {
            return;
        };

        Self::add_column_on(monitors, pool, view, mon_idx, new_idx, column, activate);
    }

    fn move_column_to_workspace_down_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        view: &mut WorkspaceView,
        mon_idx: usize,
        activate: bool,
        seed_activity: ActivityId,
    ) {
        let mon = &mut monitors[mon_idx];
        let source_workspace_idx = view.active_position();

        let new_idx = min(source_workspace_idx + 1, view.len() - 1);
        if new_idx == source_workspace_idx {
            return;
        }

        if mon
            .workspace_at(pool, view, source_workspace_idx)
            .floating_is_active()
        {
            Self::move_to_workspace_down_on(monitors, pool, view, mon_idx, activate, seed_activity);
            return;
        }

        let workspace = mon.workspace_at_mut(pool, view, source_workspace_idx);
        let Some(column) = workspace.remove_active_column(Some(&mon.output)) else {
            return;
        };

        Self::add_column_on(monitors, pool, view, mon_idx, new_idx, column, activate);
    }

    #[must_use]
    fn move_column_to_workspace_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        view: &mut WorkspaceView,
        mon_idx: usize,
        idx: usize,
        activate: bool,
        seed_activity: ActivityId,
    ) -> Vec<WorkspaceId> {
        let mon = &mut monitors[mon_idx];
        let source_workspace_idx = view.active_position();

        let new_idx = min(idx, view.len() - 1);
        if new_idx == source_workspace_idx {
            return Vec::new();
        }

        if mon
            .workspace_at(pool, view, source_workspace_idx)
            .floating_is_active()
        {
            let activate = if activate {
                ActivateWindow::Smart
            } else {
                ActivateWindow::No
            };
            return Self::move_to_workspace_on(
                monitors,
                pool,
                view,
                mon_idx,
                None,
                idx,
                activate,
                seed_activity,
            );
        }

        let workspace = mon.workspace_at_mut(pool, view, source_workspace_idx);
        let Some(column) = workspace.remove_active_column(Some(&mon.output)) else {
            return Vec::new();
        };

        Self::add_column_on(monitors, pool, view, mon_idx, new_idx, column, activate);

        // Column moved to an existing target workspace; no empty trailing workspaces to prune.
        Vec::new()
    }

    #[must_use]
    fn move_workspace_down_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        view: &mut WorkspaceView,
        mon_idx: usize,
        seed_activity: ActivityId,
    ) -> Vec<WorkspaceId> {
        let active_idx = view.active_position();
        let mut new_idx = min(active_idx + 1, view.len() - 1);
        if new_idx == active_idx {
            return Vec::new();
        }

        view.swap(active_idx, new_idx);

        if new_idx == view.len() - 1 {
            // Insert a new empty workspace.
            Self::add_workspace_bottom_on(monitors, pool, view, mon_idx, seed_activity);
        }

        if monitors[mon_idx].options.layout.empty_workspace_above_first && active_idx == 0 {
            Self::add_workspace_top_on(monitors, pool, view, mon_idx, seed_activity);
            new_idx += 1;
        }

        let mon = &mut monitors[mon_idx];
        let previous_workspace_id = view.previous();
        mon.activate_workspace(view, new_idx);
        mon.workspace_switch = None;
        view.set_previous(previous_workspace_id);

        Self::clean_up_workspaces_on(monitors, pool, view, mon_idx)
    }

    #[must_use]
    fn move_workspace_up_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        view: &mut WorkspaceView,
        mon_idx: usize,
        seed_activity: ActivityId,
    ) -> Vec<WorkspaceId> {
        let active_idx = view.active_position();
        let mut new_idx = active_idx.saturating_sub(1);
        if new_idx == active_idx {
            return Vec::new();
        }

        view.swap(active_idx, new_idx);

        if active_idx == view.len() - 1 {
            // Insert a new empty workspace.
            Self::add_workspace_bottom_on(monitors, pool, view, mon_idx, seed_activity);
        }

        if monitors[mon_idx].options.layout.empty_workspace_above_first && new_idx == 0 {
            Self::add_workspace_top_on(monitors, pool, view, mon_idx, seed_activity);
            new_idx += 1;
        }

        let mon = &mut monitors[mon_idx];
        let previous_workspace_id = view.previous();
        mon.activate_workspace(view, new_idx);
        mon.workspace_switch = None;
        view.set_previous(previous_workspace_id);

        Self::clean_up_workspaces_on(monitors, pool, view, mon_idx)
    }

    #[must_use]
    fn move_workspace_to_idx_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        view: &mut WorkspaceView,
        mon_idx: usize,
        old_idx: usize,
        new_idx: usize,
        seed_activity: ActivityId,
    ) -> Vec<WorkspaceId> {
        if view.len() <= old_idx {
            return Vec::new();
        }

        let new_idx = new_idx.clamp(0, view.len() - 1);
        if old_idx == new_idx {
            return Vec::new();
        }

        view.move_within(old_idx, new_idx);

        if new_idx > old_idx {
            if new_idx == view.len() - 1 {
                // Insert a new empty workspace.
                Self::add_workspace_bottom_on(monitors, pool, view, mon_idx, seed_activity);
            }

            if monitors[mon_idx].options.layout.empty_workspace_above_first && old_idx == 0 {
                Self::add_workspace_top_on(monitors, pool, view, mon_idx, seed_activity);
            }
        } else {
            if old_idx == view.len() - 1 {
                // Insert a new empty workspace.
                Self::add_workspace_bottom_on(monitors, pool, view, mon_idx, seed_activity);
            }

            if monitors[mon_idx].options.layout.empty_workspace_above_first && new_idx == 0 {
                Self::add_workspace_top_on(monitors, pool, view, mon_idx, seed_activity);
            }
        }

        monitors[mon_idx].workspace_switch = None;

        Self::clean_up_workspaces_on(monitors, pool, view, mon_idx)
    }

    pub fn monitors(&self) -> impl Iterator<Item = &Monitor<W>> + '_ {
        self.monitors.iter()
    }

    pub fn monitors_mut(&mut self) -> impl Iterator<Item = &mut Monitor<W>> + '_ {
        self.monitors.iter_mut()
    }

    pub fn monitor_for_output(&self, output: &Output) -> Option<&Monitor<W>> {
        self.monitors().find(|mon| &mon.output == output)
    }

    pub fn monitor_for_output_mut(&mut self, output: &Output) -> Option<&mut Monitor<W>> {
        self.monitors_mut().find(|mon| &mon.output == output)
    }

    /// Resolves an `OutputId` to its connected monitor, if any.
    pub(crate) fn monitor_for_output_id(&self, output_id: &OutputId) -> Option<&Monitor<W>> {
        self.monitors
            .iter()
            .find(|mon| mon.output_id() == *output_id)
    }

    /// Reader seam for an arbitrary activity's `WorkspaceView` keyed by
    /// connected output.
    ///
    /// Returns `None` if `activity_id` does not exist or has no view entry
    /// for `output_id`. Unlike [`Self::active_view`] this is permissive: a
    /// hidden activity may legitimately lack a view for a given output until
    /// [`Self::view_in_activity_or_materialize`] populates it. Used by
    /// `xdg_shell::send_initial_configure` to read a freshly-materialized
    /// hidden-activity view for `open-on-activity` window-rule wiring.
    pub(crate) fn view_for(
        &self,
        activity_id: ActivityId,
        output_id: &OutputId,
    ) -> Option<&WorkspaceView> {
        self.activities.get(activity_id)?.views().get(output_id)
    }

    /// Reader seam for the active `WorkspaceView` keyed by connected output.
    ///
    /// Reads the view for `output_id` from the currently active activity's
    /// `views` map; panics when `output_id` has no entry — that is a broken
    /// cross-field invariant, not user input.
    pub(crate) fn active_view(&self, output_id: &OutputId) -> &WorkspaceView {
        self.activities
            .active()
            .views()
            .get(output_id)
            .expect("connected output must have a view in the active activity")
    }

    /// Writer counterpart to [`Layout::active_view`]. Panics on an unknown
    /// `output_id` for the same reason.
    pub(crate) fn active_view_mut(&mut self, output_id: &OutputId) -> &mut WorkspaceView {
        self.activities
            .active_mut()
            .views_mut()
            .get_mut(output_id)
            .expect("connected output must have a view in the active activity")
    }

    /// Build the shared-borrow render/hit-test context for `mon`.
    ///
    /// Convenience wrapper that bundles `&self.workspaces` with the monitor's
    /// active view via [`Layout::active_view`]. Use at call sites where
    /// `&self` is freely available; in-crate sites that have already bound
    /// the pool separately can call `LayoutCtx::new` directly.
    pub fn ctx_for<'a>(&'a self, mon: &Monitor<W>) -> LayoutCtx<'a, W> {
        LayoutCtx::new(&self.workspaces, self.active_view(&mon.output_id()))
    }

    /// Build the Incoming-tagged strip context for `mon`'s active view.
    ///
    /// Unlike [`Layout::outgoing_ctx_for`], this always returns a value: only the Outgoing
    /// strip is gated on an activity switch being in flight, so the Incoming strip — the
    /// monitor's currently active view — always exists.
    ///
    /// This and [`Layout::outgoing_ctx_for`] are the only paths that mint a [`StripCtx`]
    /// outside `crate::layout`.
    pub fn incoming_ctx_for<'a>(&'a self, mon: &Monitor<W>) -> StripCtx<'a, W> {
        StripCtx::incoming(self.ctx_for(mon))
    }

    /// Build the Outgoing-tagged strip context for `mon`'s activity switch, if one is in flight.
    ///
    /// Returns `None` when no activity-switch transition is in flight on `mon`. While one is,
    /// the outgoing activity (`activity_switch.from`) is resolved from the live pool — that the
    /// `from` id is a live activity key is a verified invariant, so the lookup `.expect()`s.
    /// The per-output *view* lookup is tolerated (`None`): a cross-activity destroy can drop a
    /// dormant single-entry view mid-flight without immediate re-materialization. The view is
    /// keyed by `mon.output_id()` (never a workspace's own `output_id`, which is deliberately
    /// stale post-partial-disconnect).
    ///
    /// This is the only place that mints an Outgoing [`StripCtx`] — the tag is unforgeable
    /// outside `crate::layout`.
    pub fn outgoing_ctx_for<'a>(&'a self, mon: &Monitor<W>) -> Option<StripCtx<'a, W>> {
        let switch = mon.activity_switch.as_ref()?;
        let view = self
            .activities
            .get(switch.from)
            .expect("in-flight activity switch outgoing id must be a live activity")
            .views()
            .get(&mon.output_id())?;
        Some(StripCtx::outgoing(LayoutCtx::new(&self.workspaces, view)))
    }

    pub fn monitor_for_workspace(&self, workspace_name: &str) -> Option<&Monitor<W>> {
        let pool = &self.workspaces;
        self.monitors().find(|monitor| {
            self.active_view(&monitor.output_id())
                .ids()
                .iter()
                .any(|id| {
                    pool.get(id)
                        .expect("view id must be a key in the pool")
                        .name
                        .as_ref()
                        .is_some_and(|name| name.eq_ignore_ascii_case(workspace_name))
                })
        })
    }

    /// Return the connected monitor that currently owns a workspace named
    /// `workspace_name` (case-insensitive, matching the
    /// [`Self::find_workspace_by_name`] / [`Self::monitor_for_workspace`]
    /// precedent) AND whose activities set contains `activity_id`.
    ///
    /// Differs from [`Self::monitor_for_workspace`] in two ways:
    /// 1. The lookup walks the workspace pool keyed by `output_id` ↔ monitor, not the active
    ///    activity's `WorkspaceView`. Hidden-activity workspaces are visible to this scan.
    /// 2. The match is filtered by activity membership.
    ///
    /// Returns `None` if the workspace exists but is unbound (`output_id` is
    /// `None` or `Some(OutputId(""))` — the disconnected / unbound-config
    /// sentinels) or if no workspace matches. The empty-`OutputId` sentinel
    /// is never the id of a real connected monitor, so the
    /// `monitor_for_output_id` filter naturally rejects it.
    ///
    /// Used by `xdg_shell::send_initial_configure` to scope the
    /// `open-on-workspace`-derived monitor lookup against an
    /// `open-on-activity` target activity: if the named
    /// workspace is not tagged with the target activity, the chain falls
    /// through to subsequent targets.
    ///
    /// **Asymmetry note vs [`Self::find_workspace_in_activity_by_name`]:**
    /// this helper requires the workspace to be bound to a connected
    /// output (returns `None` for the empty-`OutputId` sentinel and for
    /// workspaces whose bound output is currently disconnected). The
    /// `find_workspace_in_activity_by_name` sibling has no such
    /// requirement — it returns the pool entry regardless of binding.
    /// `send_initial_configure` exploits this difference: the `mon`
    /// resolution chain wants a real monitor (cannot route a window to
    /// nothing), so unbound matches fall through to active-monitor
    /// fallback; the `ws` resolution wants the workspace itself (the
    /// monitor was already chosen above), so unbound matches still
    /// select the right workspace via the sibling helper.
    pub fn monitor_for_workspace_in_activity(
        &self,
        workspace_name: &str,
        activity_id: ActivityId,
    ) -> Option<&Monitor<W>> {
        let ws = self.workspaces.values().find(|ws| {
            ws.activities().contains(&activity_id)
                && ws
                    .name
                    .as_ref()
                    .is_some_and(|name| name.eq_ignore_ascii_case(workspace_name))
        })?;
        let out_id = ws.output_id()?;
        self.monitor_for_output_id(out_id)
    }

    /// Pool-walk lookup of a workspace by name, scoped to a specific
    /// activity.
    ///
    /// Returns the first workspace whose activities set contains
    /// `activity_id` and whose name matches `workspace_name`
    /// case-insensitively (matching [`Self::find_workspace_by_name`]'s
    /// `eq_ignore_ascii_case` precedent). The walk does
    /// **not** filter by `output_id`, so it succeeds for hidden-activity
    /// workspaces that aren't in the active view.
    ///
    /// Distinct from [`Self::find_workspace_by_name`], which walks
    /// `monitors × active_view.ids` plus the disconnected-workspace list
    /// and returns active-activity matches only (cross-activity widening of
    /// that helper is intentionally out of scope).
    ///
    /// Used by `compositor.rs` map-time re-resolution of
    /// `InitialConfigureState::Configured.workspace_name` when the
    /// configure-time target was a hidden activity.
    pub fn find_workspace_in_activity_by_name(
        &self,
        workspace_name: &str,
        activity_id: ActivityId,
    ) -> Option<&Workspace<W>> {
        self.workspaces.values().find(|ws| {
            ws.activities().contains(&activity_id)
                && ws
                    .name
                    .as_ref()
                    .is_some_and(|name| name.eq_ignore_ascii_case(workspace_name))
        })
    }

    pub fn outputs(&self) -> impl Iterator<Item = &Output> + '_ {
        self.monitors().map(|mon| &mon.output)
    }

    pub fn move_left(&mut self) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.move_left();
    }

    pub fn move_right(&mut self) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.move_right();
    }

    pub fn move_column_to_first(&mut self) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.move_column_to_first();
    }

    pub fn move_column_to_last(&mut self) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.move_column_to_last();
    }

    pub fn move_column_left_or_to_output(&mut self, output: &Output) -> bool {
        if let Some(workspace) = self.active_workspace_mut() {
            if workspace.move_left() {
                return false;
            }
        }

        self.move_column_to_output(output, None, true);
        true
    }

    pub fn move_column_right_or_to_output(&mut self, output: &Output) -> bool {
        if let Some(workspace) = self.active_workspace_mut() {
            if workspace.move_right() {
                return false;
            }
        }

        self.move_column_to_output(output, None, true);
        true
    }

    pub fn move_column_to_index(&mut self, index: usize) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.move_column_to_index(index);
    }

    pub fn move_down(&mut self) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.move_down();
    }

    pub fn move_up(&mut self) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.move_up();
    }

    pub fn move_down_or_to_workspace_down(&mut self) {
        if self.monitors.is_empty() {
            return;
        }
        let active_monitor_idx = self.active_monitor_idx;
        let mon_out = self.monitors[active_monitor_idx].output_id();
        let seed_activity = self.activities.active_id();
        let (monitors, pool, view) = self.monitors_pool_view_mut(&mon_out);
        Self::move_down_or_to_workspace_down_on(
            monitors,
            pool,
            view,
            active_monitor_idx,
            seed_activity,
        );
    }

    pub fn move_up_or_to_workspace_up(&mut self) {
        if self.monitors.is_empty() {
            return;
        }
        let active_monitor_idx = self.active_monitor_idx;
        let mon_out = self.monitors[active_monitor_idx].output_id();
        let seed_activity = self.activities.active_id();
        let (monitors, pool, view) = self.monitors_pool_view_mut(&mon_out);
        Self::move_up_or_to_workspace_up_on(
            monitors,
            pool,
            view,
            active_monitor_idx,
            seed_activity,
        );
    }

    pub fn consume_or_expel_window_left(&mut self, window: Option<&W::Id>) {
        if let Some(InteractiveMoveState::Moving(move_)) = &mut self.interactive_move {
            if window.is_none() || window == Some(move_.tile.window().id()) {
                return;
            }
        }

        let workspace = if let Some(window) = window {
            Some(
                self.workspaces_mut()
                    .find(|ws| ws.has_window(window))
                    .unwrap(),
            )
        } else {
            self.active_workspace_mut()
        };

        let Some(workspace) = workspace else {
            return;
        };
        workspace.consume_or_expel_window_left(window);
    }

    pub fn consume_or_expel_window_right(&mut self, window: Option<&W::Id>) {
        if let Some(InteractiveMoveState::Moving(move_)) = &mut self.interactive_move {
            if window.is_none() || window == Some(move_.tile.window().id()) {
                return;
            }
        }

        let workspace = if let Some(window) = window {
            Some(
                self.workspaces_mut()
                    .find(|ws| ws.has_window(window))
                    .unwrap(),
            )
        } else {
            self.active_workspace_mut()
        };

        let Some(workspace) = workspace else {
            return;
        };
        workspace.consume_or_expel_window_right(window);
    }

    pub fn focus_left(&mut self) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.focus_left();
    }

    pub fn focus_right(&mut self) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.focus_right();
    }

    pub fn move_view_left(&mut self) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.move_view_left();
    }

    pub fn move_view_right(&mut self) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.move_view_right();
    }

    pub fn focus_column_first(&mut self) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.focus_column_first();
    }

    pub fn focus_column_last(&mut self) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.focus_column_last();
    }

    pub fn focus_column_right_or_first(&mut self) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.focus_column_right_or_first();
    }

    pub fn focus_column_left_or_last(&mut self) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.focus_column_left_or_last();
    }

    pub fn focus_column(&mut self, index: usize) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.focus_column(index);
    }

    pub fn focus_window_up_or_output(&mut self, output: &Output) -> bool {
        if let Some(workspace) = self.active_workspace_mut() {
            if workspace.focus_up() {
                return false;
            }
        }

        self.focus_output(output);
        true
    }

    pub fn focus_window_down_or_output(&mut self, output: &Output) -> bool {
        if let Some(workspace) = self.active_workspace_mut() {
            if workspace.focus_down() {
                return false;
            }
        }

        self.focus_output(output);
        true
    }

    pub fn focus_column_left_or_output(&mut self, output: &Output) -> bool {
        if let Some(workspace) = self.active_workspace_mut() {
            if workspace.focus_left() {
                return false;
            }
        }

        self.focus_output(output);
        true
    }

    pub fn focus_column_right_or_output(&mut self, output: &Output) -> bool {
        if let Some(workspace) = self.active_workspace_mut() {
            if workspace.focus_right() {
                return false;
            }
        }

        self.focus_output(output);
        true
    }

    pub fn focus_window_in_column(&mut self, index: u8) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.focus_window_in_column(index);
    }

    pub fn focus_down(&mut self) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.focus_down();
    }

    pub fn focus_up(&mut self) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.focus_up();
    }

    pub fn focus_down_or_left(&mut self) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.focus_down_or_left();
    }

    pub fn focus_down_or_right(&mut self) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.focus_down_or_right();
    }

    pub fn focus_up_or_left(&mut self) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.focus_up_or_left();
    }

    pub fn focus_up_or_right(&mut self) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.focus_up_or_right();
    }

    pub fn focus_window_or_workspace_down(&mut self) {
        if self.monitors.is_empty() {
            return;
        }
        let active_monitor_idx = self.active_monitor_idx;
        let mon_out = self.monitors[active_monitor_idx].output_id();
        let (monitors, pool, view) = self.monitors_pool_view_mut(&mon_out);
        Self::focus_window_or_workspace_down_on(monitors, pool, view, active_monitor_idx);
    }

    pub fn focus_window_or_workspace_up(&mut self) {
        if self.monitors.is_empty() {
            return;
        }
        let active_monitor_idx = self.active_monitor_idx;
        let mon_out = self.monitors[active_monitor_idx].output_id();
        let (monitors, pool, view) = self.monitors_pool_view_mut(&mon_out);
        Self::focus_window_or_workspace_up_on(monitors, pool, view, active_monitor_idx);
    }

    pub fn focus_window_top(&mut self) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.focus_window_top();
    }

    pub fn focus_window_bottom(&mut self) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.focus_window_bottom();
    }

    pub fn focus_window_down_or_top(&mut self) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.focus_window_down_or_top();
    }

    pub fn focus_window_up_or_bottom(&mut self) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.focus_window_up_or_bottom();
    }

    pub fn move_to_workspace_up(&mut self, focus: bool) {
        if self.monitors.is_empty() {
            return;
        }
        let active_monitor_idx = self.active_monitor_idx;
        let mon_out = self.monitors[active_monitor_idx].output_id();
        let seed_activity = self.activities.active_id();
        let (monitors, pool, view) = self.monitors_pool_view_mut(&mon_out);
        Self::move_to_workspace_up_on(
            monitors,
            pool,
            view,
            active_monitor_idx,
            focus,
            seed_activity,
        );
    }

    pub fn move_to_workspace_down(&mut self, focus: bool) {
        if self.monitors.is_empty() {
            return;
        }
        let active_monitor_idx = self.active_monitor_idx;
        let mon_out = self.monitors[active_monitor_idx].output_id();
        let seed_activity = self.activities.active_id();
        let (monitors, pool, view) = self.monitors_pool_view_mut(&mon_out);
        Self::move_to_workspace_down_on(
            monitors,
            pool,
            view,
            active_monitor_idx,
            focus,
            seed_activity,
        );
    }

    /// Move the focused window to a new workspace inserted above the current
    /// one on the active monitor.
    ///
    /// No-ops when there are no monitors or the active workspace is empty.
    /// Under `empty_workspace_above_first` the forced-empty slot 0 is always
    /// empty, so invoking this action from slot 0 no-ops (remove_active_tile
    /// bails on an empty source).  From the first content slot (index 1) a
    /// genuine insert occurs at index 1 and the forced-empty slot 0 is
    /// undisturbed.
    pub fn move_to_new_workspace_up(&mut self, focus: bool) {
        if self.monitors.is_empty() {
            return;
        }
        let active_monitor_idx = self.active_monitor_idx;
        let mon_out = self.monitors[active_monitor_idx].output_id();
        let seed_activity = self.activities.active_id();
        let (monitors, pool, view) = self.monitors_pool_view_mut(&mon_out);
        Self::move_to_new_workspace_up_on(
            monitors,
            pool,
            view,
            active_monitor_idx,
            focus,
            seed_activity,
        );
    }

    /// Move the focused window to a new workspace inserted below the current
    /// one on the active monitor.
    ///
    /// No-ops when there are no monitors or the active workspace is empty.
    /// At the bottom edge the window lands on the existing trailing empty
    /// workspace instead of inserting a new one (same destination as
    /// `move_to_workspace_down` at the bottom edge).
    pub fn move_to_new_workspace_down(&mut self, focus: bool) {
        if self.monitors.is_empty() {
            return;
        }
        let active_monitor_idx = self.active_monitor_idx;
        let mon_out = self.monitors[active_monitor_idx].output_id();
        let seed_activity = self.activities.active_id();
        let (monitors, pool, view) = self.monitors_pool_view_mut(&mon_out);
        Self::move_to_new_workspace_down_on(
            monitors,
            pool,
            view,
            active_monitor_idx,
            focus,
            seed_activity,
        );
    }

    /// Insert a fresh empty workspace directly above the current one on the
    /// active monitor and focus it.
    ///
    /// No-ops when there are no monitors.  At the edges the bookend-reuse rules
    /// apply: when `empty_workspace_above_first` is set and the active position
    /// is 0, the existing forced-empty top slot is reused; from the trailing
    /// bookend the trailing bookend is reused.  The fresh workspace is ephemeral:
    /// it is pruned if focus leaves it while it is still empty and unnamed.
    /// Populate it or set a name to keep it.
    pub fn add_workspace_up(&mut self) {
        if self.monitors.is_empty() {
            return;
        }
        let active_monitor_idx = self.active_monitor_idx;
        let mon_out = self.monitors[active_monitor_idx].output_id();
        let seed_activity = self.activities.active_id();
        let (monitors, pool, view) = self.monitors_pool_view_mut(&mon_out);
        Self::add_workspace_up_on(monitors, pool, view, active_monitor_idx, seed_activity);
    }

    /// Insert a fresh empty workspace directly below the current one on the
    /// active monitor and focus it.
    ///
    /// No-ops when there are no monitors.  At the bottom edge the existing
    /// trailing empty workspace is reused instead of inserting a new one
    /// (same destination as `switch_workspace_down` at the bottom edge).  The
    /// fresh workspace is ephemeral: it is pruned if focus leaves it while it
    /// is still empty and unnamed.  Populate it or set a name to keep it.
    pub fn add_workspace_down(&mut self) {
        if self.monitors.is_empty() {
            return;
        }
        let active_monitor_idx = self.active_monitor_idx;
        let mon_out = self.monitors[active_monitor_idx].output_id();
        let seed_activity = self.activities.active_id();
        let (monitors, pool, view) = self.monitors_pool_view_mut(&mon_out);
        Self::add_workspace_down_on(monitors, pool, view, active_monitor_idx, seed_activity);
    }

    pub fn move_to_workspace(
        &mut self,
        window: Option<&W::Id>,
        idx: usize,
        activate: ActivateWindow,
    ) {
        if let Some(InteractiveMoveState::Moving(move_)) = &mut self.interactive_move {
            if window.is_none() || window == Some(move_.tile.window().id()) {
                return;
            }
        }

        if self.monitors.is_empty() {
            return;
        }

        let mon_idx = if let Some(window) = window {
            // The id-based action lookup in `input/mod.rs` uses the
            // pool-spanning `windows_all()`, so `window` may name a
            // window on a workspace bound to a dormant activity. This
            // active-view walk can't reach it; silently drop with a
            // `warn!` — real cross-activity move semantics are
            // designed in.
            let views = self.activities.active().views();
            let pool = &self.workspaces;
            let Some(mon_idx) = self.monitors.iter().position(|mon| {
                let view = views
                    .get(&OutputId::new(&mon.output))
                    .expect("connected output must have a view in the active activity");
                mon.has_window(pool, view, window)
            }) else {
                warn!(
                    "move_to_workspace: window {:?} is not on the active activity; \
                     cross-activity move-by-id semantics are deferred. \
                     Dropping action.",
                    window,
                );
                return;
            };
            mon_idx
        } else {
            self.active_monitor_idx
        };

        let seed_activity = self.activities.active_id();
        let mon_out = self.monitors[mon_idx].output_id();
        let ids_to_destroy = {
            let (monitors, pool, view) = self.monitors_pool_view_mut(&mon_out);
            Self::move_to_workspace_on(
                monitors,
                pool,
                view,
                mon_idx,
                window,
                idx,
                activate,
                seed_activity,
            )
        };
        Self::destroy_workspaces_cross_activity(
            &mut self.activities,
            &mut self.workspaces,
            ids_to_destroy,
        );
    }

    /// Move a window to a workspace addressed by raw pool id, including
    /// workspaces that belong only to a dormant activity.
    ///
    /// Dispatch shape (the four arms — by *target* locality only; the caller
    /// is responsible for filtering out the dormant-*source* case before
    /// invocation, since this entry point assumes the source workspace is the
    /// active one for `window: None` or live under the active view for
    /// `window: Some(_)`). Arms are numbered by execution order:
    ///
    /// 1. (arm 1 — unknown id) `target_raw_id` does not resolve to any pool key →
    ///    `Err(MoveWindowToPoolError::TargetUnreachable)`.
    /// 2. (arm 2 — active view) Target is in some monitor's active-activity view →
    ///    `Ok(MoveWindowToPoolOutcome::DelegateToActiveView)`. The caller falls through to the
    ///    existing [`Self::move_to_workspace`] / [`Self::move_to_output`] index path so the
    ///    in-activity move keeps its established semantics (smart activation, cursor warp, redraw).
    ///    Adding cross-activity activation here would double-activate on `focus:true`.
    /// 3. (arm 3 — disconnected output) Target resolves to a workspace whose `output_id` is not
    ///    currently connected (no live monitor) → `Err(MoveWindowToPoolError::TargetUnreachable)`.
    /// 4. (arm 4 — dormant cross-activity) Target is dormant and bound to a connected output →
    ///    perform the cross-activity move: detach the tile from its source workspace (always the
    ///    active workspace for `window: None`, the workspace holding the named window for `window:
    ///    Some(_)`), insert it into the target pool workspace's `ScrollingSpace` via
    ///    [`Workspace::add_tile`], and run [`Self::normalize_view_bookends`] so a landing on the
    ///    target activity's trailing-empty bookend mints a fresh one. Returns
    ///    `Ok(MoveWindowToPoolOutcome::MovedDormant { ws_id })` carrying the resolved pool id so
    ///    the caller can drive the `focus:true` activation flow (which mirrors the
    ///    `Action::FocusWindow` arm against the moved window).
    ///
    /// `activate` controls whether the moved tile is the active tile in its
    /// new column. The bookend sweep runs unconditionally because the
    /// bookend invariant is a structural property, not a focus property.
    ///
    /// In debug builds a `verify_invariants` assertion runs immediately
    /// after the source-side detach (before the destination attach) to catch
    /// a vacated-bookend regression class for free.
    pub(crate) fn move_window_to_pool_workspace(
        &mut self,
        window: Option<&W::Id>,
        target_raw_id: u64,
        activate: ActivateWindow,
    ) -> Result<MoveWindowToPoolOutcome, MoveWindowToPoolError> {
        let target_ws_id = self.resolve_workspace_id(target_raw_id).ok_or_else(|| {
            debug!(
                "move_window_to_pool_workspace: TargetUnreachable: unknown pool id \
                 ({target_raw_id})"
            );
            MoveWindowToPoolError::TargetUnreachable
        })?;

        // Arm 2: target is in some monitor's active view — defer to the
        // index-based path (caller falls through).
        for mon in &self.monitors {
            if self
                .active_view(&mon.output_id())
                .position_of(target_ws_id)
                .is_some()
            {
                return Ok(MoveWindowToPoolOutcome::DelegateToActiveView);
            }
        }

        // Arm 3: target bound to no connected output.
        let target_output_id = self
            .workspaces
            .get(&target_ws_id)
            .expect("resolve_workspace_id succeeded, so the id must be a pool key")
            .output_id()
            .ok_or_else(|| {
                debug!(
                    "move_window_to_pool_workspace: TargetUnreachable: workspace has no \
                     bound output (ws_id={target_raw_id})"
                );
                MoveWindowToPoolError::TargetUnreachable
            })?
            .clone();
        let target_mon_idx = self
            .monitors
            .iter()
            .position(|mon| mon.output_id() == target_output_id)
            .ok_or_else(|| {
                debug!(
                    "move_window_to_pool_workspace: TargetUnreachable: output not in \
                     connected monitors (ws_id={target_raw_id})"
                );
                MoveWindowToPoolError::TargetUnreachable
            })?;

        // Source-side detach. For `window: None` the source is the active
        // workspace's focused tile (matches the existing `move_to_workspace`
        // active-position contract). For `window: Some(_)` the caller has
        // already guaranteed the window lives in the active view (via the
        // dormant-source filter); locate it on whichever monitor's active
        // view contains it.
        let source_mon_idx = if let Some(window) = window {
            let views = self.activities.active().views();
            let pool = &self.workspaces;
            self.monitors
                .iter()
                .position(|mon| {
                    let view = views
                        .get(&OutputId::new(&mon.output))
                        .expect("connected output must have a view in the active activity");
                    mon.has_window(pool, view, window)
                })
                .expect(
                    "caller filters dormant-source: the named window must be on the active view",
                )
        } else {
            self.active_monitor_idx
        };

        let source_output = self.monitors[source_mon_idx].output.clone();
        let source_mon_out_id = self.monitors[source_mon_idx].output_id();

        let removed = {
            let pool = &mut self.workspaces;
            let view = self
                .activities
                .active()
                .views()
                .get(&source_mon_out_id)
                .expect("connected output must have a view in the active activity");
            let source_pos = if let Some(window) = window {
                view.ids()
                    .iter()
                    .position(|id| {
                        pool.get(id)
                            .expect("view id must be a key in the pool")
                            .has_window(window)
                    })
                    .expect("source-window monitor lookup above guarantees presence")
            } else {
                view.active_position()
            };
            let source_ws_id = view.ids()[source_pos];
            let workspace = pool
                .get_mut(&source_ws_id)
                .expect("view id must be a key in the pool");
            if let Some(window) = window {
                workspace.remove_tile(Some(&source_output), window, Transaction::new())
            } else {
                let Some(removed) =
                    workspace.remove_active_tile(Some(&source_output), Transaction::new())
                else {
                    // Source workspace had no tile to move (e.g. focused slot is
                    // an empty bookend). Nothing to do — not a target error.
                    return Ok(MoveWindowToPoolOutcome::NothingToMove);
                };
                removed
            }
        };

        // Mid-detach invariant assert: source side is now vacated, destination
        // not yet populated. Catches a regression in the source-side bookend
        // maintenance.
        #[cfg(debug_assertions)]
        self.verify_invariants();

        // Destination attach into the dormant pool workspace. The handler
        // passes `Smart` only when `focus:true` was requested, so collapse
        // `Smart` to `Yes` — the moved tile becomes the active tile in its
        // new column under the destination activity, matching the in-activity
        // `move_to_workspace` post-move smart-activation outcome.
        let target_output = self.monitors[target_mon_idx].output.clone();
        let activate = match activate {
            ActivateWindow::Smart | ActivateWindow::Yes => ActivateWindow::Yes,
            ActivateWindow::No => ActivateWindow::No,
        };

        {
            let target_ws = self
                .workspaces
                .get_mut(&target_ws_id)
                .expect("target_ws_id must remain a pool key across the detach scope");
            target_ws.add_tile(
                Some(&target_output),
                removed.tile,
                WorkspaceAddWindowTarget::Auto,
                activate,
                removed.width,
                removed.is_full_width,
                removed.is_floating,
            );
        }

        // Maintain the per-activity bookend invariant on the destination
        // dormant view. Safe to run here, before the source-side cleanup below: the
        // mid-detach `verify_invariants` above already certified every other view
        // conformant.
        self.normalize_view_bookends();

        // Source-side trailing-empty cleanup, mirroring the in-activity
        // `move_to_workspace_on` post-step. The hoist-then-destructure shape
        // matches the borrow-order discipline at every other split-borrow site.
        let ids_to_destroy = {
            let mon_out = self.monitors[source_mon_idx].output_id();
            let (monitors, pool, view) = self.monitors_pool_view_mut(&mon_out);
            if monitors[source_mon_idx].workspace_switch.is_none() {
                Self::clean_up_workspaces_on(monitors, pool, view, source_mon_idx)
            } else {
                Vec::new()
            }
        };
        Self::destroy_workspaces_cross_activity(
            &mut self.activities,
            &mut self.workspaces,
            ids_to_destroy,
        );

        Ok(MoveWindowToPoolOutcome::MovedDormant {
            ws_id: target_ws_id,
        })
    }

    pub fn move_column_to_workspace_up(&mut self, activate: bool) {
        if self.monitors.is_empty() {
            return;
        }
        let active_monitor_idx = self.active_monitor_idx;
        let mon_out = self.monitors[active_monitor_idx].output_id();
        let seed_activity = self.activities.active_id();
        let (monitors, pool, view) = self.monitors_pool_view_mut(&mon_out);
        Self::move_column_to_workspace_up_on(
            monitors,
            pool,
            view,
            active_monitor_idx,
            activate,
            seed_activity,
        );

        // End of split borrow. The associated fn routes through `add_column_on` for the
        // scrolling path, which no longer mints inline, or through `add_tile_on` for the
        // floating-recursion path, which still mints the active-view bookend inline; neither
        // path touches dormant views. The sweep maintains the active view's bookend on the
        // scrolling path and repairs any dormant view left with a violated bookend at the
        // receiving slot.
        self.normalize_view_bookends();
    }

    pub fn move_column_to_workspace_down(&mut self, activate: bool) {
        if self.monitors.is_empty() {
            return;
        }
        let active_monitor_idx = self.active_monitor_idx;
        let mon_out = self.monitors[active_monitor_idx].output_id();
        let seed_activity = self.activities.active_id();
        let (monitors, pool, view) = self.monitors_pool_view_mut(&mon_out);
        Self::move_column_to_workspace_down_on(
            monitors,
            pool,
            view,
            active_monitor_idx,
            activate,
            seed_activity,
        );

        // End of split borrow. See `move_column_to_workspace_up` for the rationale — the
        // scrolling `add_column_on` path relies on the sweep for its active-view bookend,
        // the floating-recursion `add_tile_on` path still mints it inline, and neither
        // path touches dormant views.
        self.normalize_view_bookends();
    }

    pub fn move_column_to_workspace(&mut self, idx: usize, activate: bool) {
        if self.monitors.is_empty() {
            return;
        }
        let active_monitor_idx = self.active_monitor_idx;
        let mon_out = self.monitors[active_monitor_idx].output_id();
        let seed_activity = self.activities.active_id();
        let ids_to_destroy = {
            let (monitors, pool, view) = self.monitors_pool_view_mut(&mon_out);
            Self::move_column_to_workspace_on(
                monitors,
                pool,
                view,
                active_monitor_idx,
                idx,
                activate,
                seed_activity,
            )
        };
        Self::destroy_workspaces_cross_activity(
            &mut self.activities,
            &mut self.workspaces,
            ids_to_destroy,
        );

        // End of split borrow. Fires after `destroy_workspaces_cross_activity` so the sweep
        // observes every view in its post-destroy shape. See `move_column_to_workspace_up`
        // for the broader rationale.
        self.normalize_view_bookends();
    }

    pub fn switch_workspace_up(&mut self) {
        if self.monitors.is_empty() {
            return;
        }
        let mon_idx = self.active_monitor_idx;
        let mon_out = self.monitors[mon_idx].output_id();
        let (monitors, _, view) = self.monitors_pool_view_mut(&mon_out);
        monitors[mon_idx].switch_workspace_up(view);
    }

    pub fn switch_workspace_down(&mut self) {
        if self.monitors.is_empty() {
            return;
        }
        let mon_idx = self.active_monitor_idx;
        let mon_out = self.monitors[mon_idx].output_id();
        let (monitors, _, view) = self.monitors_pool_view_mut(&mon_out);
        monitors[mon_idx].switch_workspace_down(view);
    }

    pub fn switch_workspace(&mut self, idx: usize) {
        if self.monitors.is_empty() {
            return;
        }
        let mon_idx = self.active_monitor_idx;
        let mon_out = self.monitors[mon_idx].output_id();
        let (monitors, _, view) = self.monitors_pool_view_mut(&mon_out);
        monitors[mon_idx].switch_workspace(view, idx);
    }

    pub fn switch_workspace_auto_back_and_forth(&mut self, idx: usize) {
        if self.monitors.is_empty() {
            return;
        }
        let mon_idx = self.active_monitor_idx;
        let mon_out = self.monitors[mon_idx].output_id();
        let (monitors, _, view) = self.monitors_pool_view_mut(&mon_out);
        monitors[mon_idx].switch_workspace_auto_back_and_forth(view, idx);
    }

    pub fn switch_workspace_previous(&mut self) {
        if self.monitors.is_empty() {
            return;
        }
        let mon_idx = self.active_monitor_idx;
        let mon_out = self.monitors[mon_idx].output_id();
        let (monitors, _, view) = self.monitors_pool_view_mut(&mon_out);
        monitors[mon_idx].switch_workspace_previous(view);
    }

    /// Locate the pool workspace owning `window` and return its id plus a
    /// clone of its `activities` set. Returns `None` if the window id is not
    /// present anywhere in the pool. Used by the `Action::FocusWindow`
    /// dispatcher to compute a target activity for a window that lives on a
    /// dormant workspace.
    ///
    /// Trait-agnostic by design: the `Mapped`-specific `last_focused_activity`
    /// hint is read at the call site and passed into
    /// [`Self::pick_activity_for_hidden_window`] separately, so that
    /// `Layout<TestWindow>` tests of the picker do not need a
    /// `last_focused_activity` plumb on `TestWindow`.
    pub(crate) fn window_ws_and_activity_hint(&self, window: &W::Id) -> Option<WorkspaceId> {
        self.workspaces.values().find_map(|ws| {
            if ws.windows().any(|w| w.id() == window) {
                Some(ws.id())
            } else {
                None
            }
        })
    }

    /// Pick a non-active target activity for an `Action::FocusWindow { id }`
    /// dispatch that resolved a window on a hidden workspace.
    ///
    /// Three-tier tiebreaker, in order — each tier excludes the currently-active
    /// activity so the picker never returns `active_id()` from a productive path
    /// (the caller already resolved the "visible" fast-path before reaching here):
    /// 1. `hint` if `ws.activities.contains(hint) && hint != active_id()`. `hint` is the callee's
    ///    `Mapped.last_focused_activity` — the activity that was active the last time the user
    ///    focused *this specific* window.
    /// 2. `Activities::previous_id()` if that id is in `ws.activities` and is not the current
    ///    active.
    /// 3. First activity in `self.activities.iter()` (IndexMap display order) whose id is in
    ///    `ws.activities` and is not the current active.
    ///
    /// Invariant relied on: every workspace's `activities` set is non-empty
    /// (seeded at construction, enforced by `Layout::verify_invariants`). If
    /// the set is empty we hit an `unreachable!` with that contract named —
    /// a silent `.first()?` or `_ => active_id()` fallback is a review-stop
    /// bug (fork CLAUDE.md).
    ///
    /// Degenerate tail: if the workspace is tagged *only* with the currently
    /// active activity (and the caller invoked the picker anyway instead of
    /// short-circuiting on the visibility fast-path), every tier filters the
    /// lone candidate out and the function returns `active_id()`. This arm is
    /// not reached from the production dispatch path — the caller is expected
    /// to detect `target == active_id()` and no-op.
    pub(crate) fn pick_activity_for_hidden_window(
        &self,
        ws_id: WorkspaceId,
        hint: Option<ActivityId>,
    ) -> ActivityId {
        let active = self.activities.active_id();
        let ws = self
            .workspaces
            .get(&ws_id)
            .expect("ws_id must be a key in the pool (caller resolved it via windows_all)");
        let activities = ws.activities();

        if activities.is_empty() {
            unreachable!(
                "workspace.activities must be non-empty (Layout invariant, verify_invariants)"
            );
        }

        // Tier 1: MRU hint.
        if let Some(h) = hint {
            if h != active && activities.contains(&h) {
                return h;
            }
        }

        // Tier 2: Activities::previous_id.
        if let Some(prev) = self.activities.previous_id() {
            if prev != active && activities.contains(&prev) {
                return prev;
            }
        }

        // Tier 3: first activity in declaration order whose id is in ws.activities.
        for act in self.activities.iter() {
            let id = act.id();
            if id != active && activities.contains(&id) {
                return id;
            }
        }

        // Degenerate tail — see rustdoc above. Not reached from the production
        // dispatch path; the caller short-circuits on `target == active_id()`.
        active
    }

    /// Returns `Some(reason)` when the three live-input conditions in the
    /// activities design forbid switching activities — interactive window move, DnD, or a
    /// workspace-switch gesture in flight on *any* monitor. Returns `None`
    /// otherwise; in particular, an in-flight `WorkspaceSwitch::Animation` is
    /// not a hard block (it is snapped by `switch_activity` itself).
    ///
    /// The keybinding and IPC dispatch sites consult this reader and drop /
    /// queue the action accordingly. The `debug_assert!` at the top of
    /// [`Self::switch_activity`] pins that no caller bypasses the gate.
    pub(crate) fn is_activity_switch_hard_blocked(&self) -> Option<ActivitySwitchBlock> {
        if self.interactive_move.is_some() {
            return Some(ActivitySwitchBlock::InteractiveMove);
        }
        if self.dnd.is_some() {
            return Some(ActivitySwitchBlock::Dnd);
        }
        // Gesture can be in flight on any monitor, not just the active one.
        for mon in &self.monitors {
            if matches!(mon.workspace_switch, Some(WorkspaceSwitch::Gesture(_))) {
                return Some(ActivitySwitchBlock::WorkspaceSwitchGesture);
            }
        }
        None
    }

    /// Flip the active activity cursor to `target`.
    ///
    /// Hard-block conditions from (interactive move, DnD, workspace-
    /// switch gesture) MUST be filtered by the caller via
    /// [`Self::is_activity_switch_hard_blocked`]; the entry `debug_assert!`
    /// below pins this contract.
    ///
    /// Ordering:
    /// 1. No-op fast-path on `ActivityId` equality — avoids a `HashMap` lookup on the dominant
    ///    real-input path (`target == active_activity_id()`). Load-bearing for the hot path.
    /// 2. Reject unknown `target` with `error!` + early return, follows the
    ///    early-return-with-`error!` pattern from `update_output_size` and
    ///    `update_render_elements`, with the offending id in the message for log correlation. Step
    ///    2→3 ordering is also pinned by the `debug_assert!` inside [`Activities::set_active`].
    /// 3. Flip cursors via [`Activities::set_active`].
    /// 4. Snap any in-flight `WorkspaceSwitch` on every monitor by setting it to `None`. Per this
    ///    is the load-bearing snap step of the snap+proceed contract for
    ///    `WorkspaceSwitch::Animation`. Gestures are filtered out by the caller-side hard-block
    ///    guard, so the only `Some(_)` reachable here is `Animation(_)`; the unconditional
    ///    `mon.workspace_switch = None` matches every other clear site in the codebase. Fractional
    ///    positions inside `WorkspaceSwitch` refer to positions in the previously-active activity's
    ///    view, so leaving them around after the flip would be incoherent.
    /// 5. Lazily populate the target activity's per-output views via
    ///    [`Self::ensure_all_activity_views`] — see step 3.
    ///
    /// # Focus restoration
    ///
    /// No focus is poked directly from here; the read path picks it up on the
    /// next `State::refresh`. [`Self::focus`] reads the active activity's view
    /// for the active monitor, and each workspace's own persisted
    /// `active_column_idx` / `active_tile_idx` selects the window. Global
    /// keyboard focus is reassigned by `State::refresh` →
    /// `update_keyboard_focus` before rendering; `active_monitor_idx` is not
    /// mutated by the switch, so focus returns to the output that held it.
    ///
    /// At the end of the switch, two post-conditions are pinned as
    /// `debug_assert!`s (release builds pay nothing):
    ///
    /// - `view.active ∈ view.ids()` for every entry in the active activity's `views()`. Today
    ///   [`WorkspaceView`] upholds this at construction and in every mutator; the assertion
    ///   documents the contract that the only public entry point flipping activities is required to
    ///   preserve.
    /// - `active_monitor_idx < monitors.len()` when monitors are non-empty. Monitors don't mutate
    ///   during a switch, so this is also a pin, not a fix — a drift would be a caller-side bug.
    pub fn switch_activity(&mut self, target: ActivityId) {
        debug_assert!(
            self.is_activity_switch_hard_blocked().is_none(),
            "switch_activity called while hard-blocked: caller must filter via \
             is_activity_switch_hard_blocked",
        );
        // Capture before set_active so we can record the departing strip.
        let outgoing = self.activities.active_id();
        if target == outgoing {
            return;
        }
        if !self.activities.contains(target) {
            error!("switch_activity: target {target:?} is not a live activity id");
            return;
        }
        self.activities.set_active(target);
        // Snap any in-flight WorkspaceSwitch::Animation (snap+proceed contract).
        // Gestures are filtered by the caller-side hard-block guard, so the only Some(_)
        // reachable here is Animation(_); the unconditional clear matches every other clear
        // site in the codebase. Not reached on the no-op or unknown-id paths above, which
        // both early-return before touching monitor state.
        for mon in &mut self.monitors {
            mon.workspace_switch = None;
        }
        self.ensure_all_activity_views();

        // Arm the activity-switch transition on each connected monitor.
        // Config gate: if the activity-switch animation is marked off, skip (zero new state).
        // Overview gate: skip when the overview is open — the overview already gives a
        // full spatial map; a simultaneous strip slide would compound confusingly.
        // Assignment over an existing Some(_) is the snap+restart contract — no continuity math.
        if !self.options.animations.activity_switch.0.off && !self.overview_open {
            // Direction: higher pool index → Left (incoming strip from the right).
            // Both positions are `.expect`-safe: `target` passed `contains` above;
            // `outgoing` was the active id, which is always a live key.
            let target_pos = self
                .activities
                .position_of(target)
                .expect("target passed contains check and must be in the pool");
            let outgoing_pos = self
                .activities
                .position_of(outgoing)
                .expect("outgoing was the active id and must be in the pool");
            let dir = if target_pos > outgoing_pos {
                SlideDirection::Left
            } else {
                SlideDirection::Right
            };
            for mon in &mut self.monitors {
                mon.activity_switch = Some(ActivitySwitch {
                    from: outgoing,
                    anim: Animation::new(
                        mon.clock.clone(),
                        0.,
                        1.,
                        0.,
                        self.options.animations.activity_switch.0,
                    ),
                    dir,
                });
            }
        }

        // Post-condition pins. See the "Focus restoration" rustdoc paragraph for rationale.
        #[cfg(debug_assertions)]
        {
            for (output_id, view) in self.activities.active().views() {
                debug_assert!(
                    view.ids().contains(&view.active()),
                    "switch_activity post-condition: view.active ({:?}) must be in view.ids \
                     for output {:?}",
                    view.active(),
                    output_id,
                );
            }
            if !self.monitors.is_empty() {
                debug_assert!(
                    self.active_monitor_idx < self.monitors.len(),
                    "switch_activity post-condition: active_monitor_idx {} out of range \
                     (monitors.len() = {})",
                    self.active_monitor_idx,
                    self.monitors.len(),
                );
            }
        }
    }

    /// Repair every activity's `views` so they agree with pool membership, in both directions.
    ///
    /// Total and repair-oriented (debug-loud, release-healable): it installs a view entry for
    /// every (activity, workspace) membership that lacks one, and removes every view entry whose
    /// workspace no longer carries that activity in its `activities` set. Callers run it after a
    /// membership-mutating pass (config-reload reset, sticky (re-)expansion, activity creation)
    /// has left views and membership out of agreement.
    ///
    /// Ordering is load-bearing:
    ///
    /// - **Installs run before removals.** A workspace that moved wholesale from activity `L` to
    ///   activity `A` must resolve its holding output via [`Self::workspace_holding_output`] while
    ///   `L`'s view still holds it. Removing from `L` first would strand the id in no view, the
    ///   resolver would return `None`, and the install would skip — minting the very
    ///   membership-without-view incoherence this method exists to close.
    /// - **Bookend repair for windowed/named installs runs after removals** so
    ///   [`Self::normalize_view_bookends`] sees final view positions.
    ///
    /// The install worklist is deterministic — activities walked in declaration order, workspaces
    /// sorted by `WorkspaceId.get()` (creation order) — so view order does not flap across runs.
    ///
    /// Every skip is `trace!` / `debug_assert!`-loud. Does not call `Self::verify_invariants` —
    /// callers do.
    fn reconcile_views_with_membership(&mut self) {
        if self.monitors.is_empty() {
            // In the fully-disconnected window every activity's `views` map is empty and parked
            // workspaces live directly in `disconnected_workspace_ids` — nothing to reconcile.
            return;
        }

        let disconnected: HashSet<WorkspaceId> =
            self.disconnected_workspace_ids.iter().copied().collect();

        // Compute both worklists read-only before mutating anything.
        let mut installs: Vec<(WorkspaceId, ActivityId)> = Vec::new();
        let mut removals: Vec<(ActivityId, OutputId, WorkspaceId)> = Vec::new();
        for activity in self.activities.iter() {
            let act_id = activity.id();

            // Installs (membership → view): pool workspaces that carry this activity but are
            // absent from every one of its views. Sorted by creation order (mirrors the
            // `ensure_view_for` lift sort) so the resulting view order is deterministic.
            let union: HashSet<WorkspaceId> = activity
                .views()
                .values()
                .flat_map(|view| view.ids().iter().copied())
                .collect();
            let mut per_activity: Vec<WorkspaceId> = self
                .workspaces
                .iter()
                .filter(|(ws_id, ws)| {
                    ws.activities().contains(&act_id)
                        && !disconnected.contains(ws_id)
                        && !union.contains(ws_id)
                })
                .map(|(ws_id, _)| *ws_id)
                .collect();
            per_activity.sort_by_key(|id| id.get());
            for ws_id in per_activity {
                installs.push((ws_id, act_id));
            }

            // Removals (view → membership): view entries whose workspace no longer carries this
            // activity. Positions are re-resolved at removal time, so only the ids are recorded.
            for (out_id, view) in activity.views() {
                for &ws_id in view.ids() {
                    debug_assert!(
                        self.workspaces.contains_key(&ws_id),
                        "reconcile_views_with_membership: view id {ws_id:?} for activity \
                         {act_id:?} has no pool entry — zombie view id in reconcile removals",
                    );
                    let lacks_membership = self
                        .workspaces
                        .get(&ws_id)
                        .is_some_and(|ws| !ws.activities().contains(&act_id));
                    if lacks_membership {
                        removals.push((act_id, out_id.clone(), ws_id));
                    }
                }
            }
        }

        // Snap any in-flight animation on every monitor when the active activity's view length or
        // positions are about to change — the mutation can shrink or reshuffle the active view,
        // and a stale fractional switch target would trip the animation-bounds check in
        // `Monitor::verify_invariants` (`before_idx`/`after_idx < active_view.len()`) after it.
        let active_id = self.activities.active_id();
        let active_affected = installs.iter().any(|(_, act_id)| *act_id == active_id)
            || removals.iter().any(|(act_id, _, _)| *act_id == active_id);
        if active_affected {
            for mon in &mut self.monitors {
                if matches!(mon.workspace_switch, Some(WorkspaceSwitch::Animation(_))) {
                    mon.workspace_switch = None;
                }
            }
        }

        // Installs first: each membership without a view gets one, keyed by the output whose view
        // currently holds the workspace (in the losing activity, before any removal runs).
        let mut installed_windowed: Vec<OutputId> = Vec::new();
        for (ws_id, act_id) in installs {
            let Some(out_id) = self.workspace_holding_output(ws_id) else {
                trace!(
                    "reconcile_views_with_membership: {ws_id:?} has no holding view; \
                     membership→view install into {act_id:?} skipped",
                );
                continue;
            };
            let inserted = {
                let pool = &mut self.workspaces;
                let activities = &mut self.activities;
                let activity = activities
                    .get_mut(act_id)
                    .expect("act_id came from self.activities.iter()");
                if let Some(view) = activity.views_mut().get_mut(&out_id) {
                    Self::view_insert_above_trailing_bookend(pool, view, ws_id);
                    true
                } else {
                    // `out_id` names a currently-connected output (`workspace_holding_output`
                    // only returns view keys, and every view key is a connected output's id per
                    // the connected-keyspace invariant), and the per-activity bookend invariant
                    // guarantees every activity holds a view for every connected output — so this
                    // activity must have a view keyed by `out_id`. Reaching here means that
                    // invariant is violated (membership↔view coherence bug); skip rather than
                    // mint a membership-without-view incoherence on top of it.
                    debug_assert!(
                        false,
                        "reconcile_views_with_membership: activity {act_id:?} has no view for \
                         {out_id:?}, a connected output — per-activity bookend invariant \
                         violated (membership↔view coherence bug)",
                    );
                    false
                }
            };
            if inserted
                && self
                    .workspaces
                    .get(&ws_id)
                    .is_some_and(|ws| ws.has_windows_or_name())
            {
                installed_windowed.push(out_id);
            }
        }

        // Removals: drop view entries whose workspace lost the membership. Positions are
        // re-resolved per removal because an earlier removal in the same view shifts them.
        let mut dropped_any_view_entry = false;
        let mut removed_ids: Vec<WorkspaceId> = Vec::new();
        let mut affected_pairs: Vec<(ActivityId, OutputId)> = Vec::new();
        for (act_id, out_id, ws_id) in removals {
            // Every view key is a currently-connected output per the connected-keyspace invariant;
            // the check mirrors the runtime Remove patches' is-connected guard for the
            // drop-to-zero flag.
            let is_connected = self.monitors.iter().any(|m| m.output_id() == out_id);
            let activity = self
                .activities
                .get_mut(act_id)
                .expect("act_id came from self.activities.iter()");
            let Some(view) = activity.views_mut().get_mut(&out_id) else {
                trace!(
                    "reconcile_views_with_membership: view for {out_id:?} in {act_id:?} already \
                     gone; removal of {ws_id:?} skipped",
                );
                continue;
            };
            let Some(pos) = view.position_of(ws_id) else {
                trace!(
                    "reconcile_views_with_membership: {ws_id:?} already absent from {act_id:?}'s \
                     {out_id:?} view; removal skipped",
                );
                continue;
            };
            if view.len() == 1 {
                // Drop the single-entry view outright — mirrors the
                // `destroy_workspaces_cross_activity` single-entry retain-drop path.
                activity.views_mut().remove(&out_id);
                if is_connected {
                    dropped_any_view_entry = true;
                }
            } else {
                view.remove_at(pos);
            }
            removed_ids.push(ws_id);
            affected_pairs.push((act_id, out_id));
        }

        // Normalize any all-empty exclusive EWAF len-2 view a removal left behind. The collapse is
        // idempotent (a non-len-2 view early-returns), so duplicate pairs are harmless.
        for (act_id, out_id) in &affected_pairs {
            self.collapse_empty_exclusive_ewaf_len2_view(*act_id, out_id);
        }

        // A whole-view drop on a connected output breaks `active.views.len() == monitors.len()`;
        // the materializer reinstates the per-activity bookend view.
        if dropped_any_view_entry {
            self.ensure_all_activity_views();
        }

        // Bookend repair for windowed/named installs, after removals so the sweep sees final
        // positions. An empty install is itself a valid bookend and needs no repair.
        for out_id in &installed_windowed {
            if self.monitors.iter().all(|m| m.output_id() != *out_id) {
                debug_assert!(
                    false,
                    "reconcile_views_with_membership: installed view output {out_id:?} must be \
                     connected (connected-keyspace invariant)",
                );
            }
        }
        if !installed_windowed.is_empty() {
            self.normalize_view_bookends();
        }

        // A removal may have narrowed a workspace to a single activity while it sits in an illegal
        // middle position; reclaim any such orphaned pinned empties. Dedup so an id removed from
        // two activities isn't reclaimed twice (the second pass would `debug_assert!` on the
        // already-culled id).
        removed_ids.sort_by_key(|id| id.get());
        removed_ids.dedup();
        self.reclaim_unpinned_empty_workspaces(removed_ids);
    }

    /// Cull the second entry of an all-empty exclusive EWAF length-2 view, restoring the
    /// "1 or 3+ unless second shared" bookend rule.
    ///
    /// Shared by [`Self::reconcile_views_with_membership`] and the runtime membership-removal
    /// patches so all three sites apply one policy. Mirrors `clean_up_workspaces_on`'s EWAF
    /// special case and [`Self::reclaim_unpinned_empty_workspaces`]' EWAF resolution: resolve the
    /// EWAF flag from the connected monitor's merged options (layout-root fallback for
    /// dormant/disconnected views); if the view exists, is length 2, both entries are empty
    /// unnamed, and the second entry is safe to reclaim (exclusive to a single activity), snap the
    /// holding monitor's animation when `act_id` is active and destroy the second entry.
    ///
    /// Guards, not asserts: this is a repair path and must stay total. A windowed first entry
    /// under EWAF is already-illegal input, left for `Self::verify_invariants` to name.
    fn collapse_empty_exclusive_ewaf_len2_view(&mut self, act_id: ActivityId, out_id: &OutputId) {
        let ewaf = self
            .monitors
            .iter()
            .find(|m| &m.output_id() == out_id)
            .map(|m| m.options.layout.empty_workspace_above_first)
            .unwrap_or(self.options.layout.empty_workspace_above_first);
        if !ewaf {
            return;
        }

        let ids = match self
            .activities
            .get(act_id)
            .and_then(|activity| activity.views().get(out_id))
        {
            Some(view) if view.len() == 2 => [view.ids()[0], view.ids()[1]],
            _ => return,
        };

        let both_empty = ids.iter().all(|id| {
            self.workspaces
                .get(id)
                .is_some_and(|ws| !ws.has_windows_or_name())
        });
        if !both_empty {
            return;
        }
        // A shared second entry is the legal minimal EWAF shape; only an exclusive one collapses.
        if !Self::workspace_is_safe_to_reclaim(&self.workspaces, ids[1]) {
            return;
        }

        if act_id == self.activities.active_id() {
            for mon in &mut self.monitors {
                if &mon.output_id() == out_id
                    && matches!(mon.workspace_switch, Some(WorkspaceSwitch::Animation(_)))
                {
                    mon.workspace_switch = None;
                }
            }
        }

        Self::destroy_workspaces_cross_activity(
            &mut self.activities,
            &mut self.workspaces,
            [ids[1]],
        );
    }

    // Make every activity's `views` map cover every connected monitor. Required by the
    // per-activity bookend invariant: every (activity, connected-monitor) pair must hold a
    // `WorkspaceView` whose first and last workspaces satisfy the trailing-empty (and EWAF
    // leading-empty) rule.
    //
    // Connected-keyspace invariant: in the connected steady state every activity's view keys are
    // currently-connected outputs. `remove_output`'s dormant-view drain (both its partial- and
    // full-disconnect branches) removes every activity's view entry for the disconnecting output,
    // so no dormant view keyed by a disconnected output survives to be re-materialized here — the
    // skip-existing guard below therefore only ever short-circuits on a view for a still-connected
    // output, never resurrects a stale one. Any residual disconnected-output view (a coherence
    // bug, not a steady state) is still caught by `Layout::verify_invariants`' disconnected-output
    // bookend pass.
    //
    // No-op when no monitors are connected: the `monitors.is_empty()` branch of
    // `verify_invariants` requires every activity's `views` map to be empty in that state, and
    // `disconnected_workspace_ids` then holds workspaces directly without per-activity views.
    //
    // Borrow discipline: we read `&self.monitors` once into a `Vec` of triples
    // `(OutputId, Option<Output>, Rc<Options>)` (`Option<Output>` is always `Some` today; the
    // `None` shape on `ensure_view_for` is reserved for a future caller that mints a view
    // without a live `Output` handle). After the snapshot we drop the borrow and only then
    // dispatch to [`Self::ensure_view_for`], which owns the pool/activities split-borrow
    // internally.
    pub(super) fn ensure_all_activity_views(&mut self) {
        if self.monitors.is_empty() {
            return;
        }

        let known_outputs: Vec<(OutputId, Option<Output>, Rc<Options>)> = self
            .monitors
            .iter()
            .map(|m| (m.output_id(), Some(m.output.clone()), m.options.clone()))
            .collect();

        let activity_ids: Vec<ActivityId> = self.activities.iter().map(|a| a.id()).collect();

        for activity_id in activity_ids {
            for (output_id, output_opt, options) in &known_outputs {
                let already = self
                    .activities
                    .get(activity_id)
                    .is_some_and(|a| a.views().contains_key(output_id));
                if already {
                    continue;
                }
                self.ensure_view_for(activity_id, output_id.clone(), output_opt.as_ref(), options);
            }
        }
    }

    /// Materialize a `WorkspaceView` entry on `activity_id`'s `views` map for
    /// `output_id` if and only if no entry exists yet.
    ///
    /// The body mirrors the per-output allocation discipline of
    /// [`Self::ensure_all_activity_views`] — pre-tagged candidates from the
    /// pool are lifted into the new view (sorted by `WorkspaceId.get()` as a
    /// placeholder for `cmp_by_config_then_creation`), padded with a fresh
    /// trailing empty so the bookend "last must be empty / unnamed" invariant
    /// holds; under `empty_workspace_above_first` for this output, a fresh
    /// leading empty is also prepended so the "first must be empty / unnamed"
    /// and "1 or 3+" length rules hold. The active index becomes 1 in the
    /// EWAF lift branch so the first lifted workspace stays selected.
    ///
    /// `output` is `Some(&Output)` for outputs that are currently connected
    /// (`mon_options` is then the per-monitor merged options, mirroring
    /// `Monitor::new`'s ctor — using `self.options` would silently violate
    /// the EWAF first/last-empty invariants for outputs whose
    /// `layout_config` flips the flag) and `None` for outputs that are known
    /// only because some activity's view is keyed by them (e.g. a stale
    /// dormant view for an output that disconnected while the activity was
    /// inactive). On the `None` branch we use `Workspace::new_no_outputs`
    /// and then patch `ws.output_id` so the materialized bookend can be
    /// reclaimed by a future `bind_output` (mirrors the established pattern
    /// in `reconcile_activities_on_reload_remove` at the orphan-rebind site).
    ///
    /// Used by [`Self::ensure_all_activity_views`] and by
    /// [`Self::view_in_activity_or_materialize`] (target = arbitrary
    /// activity, e.g. an inactive activity addressed by an
    /// `open-on-activity` window rule).
    ///
    /// Caller must ensure the entry does not yet exist (else the final
    /// insert asserts).
    pub(crate) fn ensure_view_for(
        &mut self,
        activity_id: ActivityId,
        output_id: OutputId,
        output: Option<&Output>,
        mon_options: &Rc<Options>,
    ) {
        assert!(
            !output_id.as_str().is_empty(),
            "ensure_view_for requires a non-sentinel OutputId",
        );
        let clock = self.clock.clone();

        // Helper: mint a fresh empty workspace for this (activity, output) pair. The `Some`
        // branch is the only one fired today — `ensure_all_activity_views` narrows
        // `known_outputs` to connected monitors. The `None` branch (using
        // `Workspace::new_no_outputs` and patching `output_id` to the known OutputId — same
        // pattern as the orphan-rebind in `reconcile_activities_on_reload_remove`) is reserved
        // for a future caller that needs to materialize a view without a live `Output` handle.
        let mint_empty = |workspaces: &mut HashMap<WorkspaceId, Workspace<W>>| -> WorkspaceId {
            let mut ws = match output {
                Some(o) => Workspace::new(
                    o,
                    HashSet::from([activity_id]),
                    clock.clone(),
                    mon_options.clone(),
                ),
                None => Workspace::new_no_outputs(
                    HashSet::from([activity_id]),
                    clock.clone(),
                    mon_options.clone(),
                ),
            };
            if output.is_none() {
                ws.output_id = Some(output_id.clone());
            }
            let id = ws.id();
            assert!(
                workspaces.insert(id, ws).is_none(),
                "fresh id must be unique",
            );
            id
        };

        // Collect pre-tagged candidates under a shared borrow of the pool, so we can
        // later take `&mut self.workspaces` without a conflicting active borrow.
        let mut tagged: Vec<WorkspaceId> = self
            .workspaces
            .values()
            .filter(|ws| {
                ws.output_id() == Some(&output_id) && ws.activities().contains(&activity_id)
            })
            .map(|ws| ws.id())
            .collect();
        // sort by WorkspaceId (monotonic creation counter) — placeholder for
        // cmp_by_config_then_creation which requires is_config_declared on Workspace.
        tagged.sort_by_key(|id| id.get());

        // Source-side dedup: every workspace we are about to lift into this fresh
        // (activity_id, output_id) view may already appear in another view of the same
        // activity. Typical case: the partial-disconnect walk in `remove_output` migrated
        // these ids into another `(activity_id, other_output)` view with `output_id`
        // preserved on the pool entry; this call is now materializing the
        // reconnecting output's view via `ensure_all_activity_views`. Without this drop the
        // workspace appears in both views and the primary-monitor "own monitor exists"
        // invariant fires at `verify_invariants` on the next `switch_activity`.
        //
        // Diverges intentionally from `move_workspace_to_output_by_id`'s source-side drop
        // pattern: that path collapses a single-entry view by removing the map entry
        // (`activity.views_mut().remove(&source_out_id)`); here we never do that. The
        // sibling view's trailing-empty bookend (and, under EWAF, the leading-empty
        // bookend) must survive in place, since that view is the canonical per-activity
        // bookend for `(activity_id, other_output)` and removing the map entry would
        // violate the per-activity bookend invariant. A single-entry view consisting only
        // of the trailing-empty bookend is structurally valid — `assert_view_bookends`
        // accepts last==empty-unnamed at len=1 and EWAF's "1 or 3+" rule permits len=1.
        // Empty case (fresh-mint path) has no lifted ids and therefore no dedup work.
        if !tagged.is_empty() {
            let activity = self
                .activities
                .get_mut(activity_id)
                .expect("activity id must be a live key");
            // Clone the keyset to release the inner shared borrow before taking the
            // mutable borrow on each entry inside the loop — iterating `views().keys()`
            // while calling `views_mut().get_mut(...)` would mix shared and mutable
            // borrows of the same `HashMap`.
            let other_output_ids: Vec<OutputId> = activity
                .views()
                .keys()
                .filter(|oid| **oid != output_id)
                .cloned()
                .collect();
            for other_out in &other_output_ids {
                let view = activity
                    .views_mut()
                    .get_mut(other_out)
                    .expect("collected key must still be present");
                for ws_id in &tagged {
                    if let Some(pos) = view.position_of(*ws_id) {
                        debug_assert!(
                            view.len() > 1,
                            "sibling view must retain at least one bookend after dedup — \
                             `tagged` must never contain a bookend of this view",
                        );
                        view.remove_at(pos);
                    }
                }
            }
        }

        let ewaf = mon_options.layout.empty_workspace_above_first;

        let view = if tagged.is_empty() {
            // Fresh branch: a single trailing empty satisfies all EWAF invariants at len=1.
            let id = mint_empty(&mut self.workspaces);
            WorkspaceView::new(vec![id], 0)
        } else {
            // Lift branch: pre-tagged candidates form the body. Append a fresh trailing empty
            // so the trailing-empty / trailing-unnamed bookend holds. Under EWAF for this
            // output, also prepend a fresh leading empty so the leading-empty / leading-unnamed
            // / "1 or 3+" rules hold; the active index becomes 1 so the first lifted workspace
            // stays selected.
            let bottom_id = mint_empty(&mut self.workspaces);

            let mut ids = tagged;
            let mut active_idx = 0usize;

            if ewaf {
                let top_id = mint_empty(&mut self.workspaces);
                ids.insert(0, top_id);
                active_idx = 1;
            }

            ids.push(bottom_id);

            WorkspaceView::new(ids, active_idx)
        };

        // Explicit contains_key → insert rather than `.entry().or_insert_with()`: the
        // closure arg would need to capture `&mut self.workspaces` via `||`, conflicting
        // with this outer `&mut self.activities` reborrow.
        let activity = self
            .activities
            .get_mut(activity_id)
            .expect("activity id must be a live key");
        assert!(
            activity.views_mut().insert(output_id, view).is_none(),
            "caller must ensure the entry does not yet exist",
        );
    }

    /// Ensure `activity_id`'s `views` map has a `WorkspaceView` entry for the
    /// connected monitor identified by `output_id`, materializing one through
    /// [`Self::ensure_view_for`] if needed.
    ///
    /// Used by `xdg_shell::send_initial_configure` to address a hidden
    /// activity targeted by an `open-on-activity` window rule.
    /// Side-effecting only; the materialized view can be read back via
    /// `self.activities.get(activity_id).views()[output_id]`. Returns `()`
    /// rather than `&WorkspaceView` so callers can keep their subsequent
    /// reads on `&self.layout` instead of holding a `&mut self.layout`
    /// borrow across send_configure (keeps the call site free of an
    /// outstanding `&mut self.layout` borrow).
    ///
    /// No-op when `output_id` is not connected: the `monitor_for_output_id`
    /// lookup returns `None` and we skip the materialize. Caller is
    /// responsible for handling the missing-entry case after the call (in
    /// practice: `send_initial_configure` will then fall through its
    /// precedence chain to a different target).
    pub(crate) fn view_in_activity_or_materialize(
        &mut self,
        activity_id: ActivityId,
        output_id: &OutputId,
    ) {
        if self
            .activities
            .get(activity_id)
            .is_some_and(|a| a.views().contains_key(output_id))
        {
            return;
        }
        let Some((output, options)) = self
            .monitor_for_output_id(output_id)
            .map(|mon| (mon.output.clone(), mon.options.clone()))
        else {
            // Output not connected — nothing to materialize against.
            return;
        };
        self.ensure_view_for(activity_id, output_id.clone(), Some(&output), &options);
    }

    /// Switch to a recently-active activity at recency position `depth`.
    ///
    /// `depth = 0` names the currently-active activity — explicit no-op, returns immediately.
    /// `depth = 1` (the legacy default) switches to the previously-active activity.
    /// `depth = N` switches to the activity at recency position N in the MRU list computed by
    /// [`Activities::recency_ordered`]. If `depth` exceeds the number of activated activities
    /// minus one, it is clamped to the oldest-activated activity (no error is returned).
    ///
    /// Activities with `last_active_seq == 0` (never activated this session) are excluded from
    /// the reachable target set; the clamp always lands on an activity that has been activated
    /// at least once.
    pub(crate) fn switch_activity_previous(&mut self, depth: u32) {
        // depth = 0 means "current active" by definition — nothing to do.
        if depth == 0 {
            return;
        }
        let ordered = self.activities.recency_ordered();
        let activated_count = ordered
            .iter()
            .filter(|&&id| {
                self.activities
                    .get(id)
                    .expect("recency_ordered ids must be live keys in the pool")
                    .last_active_seq()
                    > 0
            })
            .count();
        // Fewer than 2 activated activities means there is no other activated activity to
        // switch to — toggling on a pristine pool (only the seed ever stamped) must not
        // error, mirroring the pre-depth no-previous early-return.
        if activated_count <= 1 {
            return;
        }
        // Clamp depth to the index of the oldest-activated entry in the ordered list.
        let idx = depth.min(activated_count.saturating_sub(1) as u32) as usize;
        let target = ordered[idx];
        // depth >= 1 (checked above) + activated_count >= 2 (checked above) guarantees
        // idx >= 1, so target is never ordered[0] (the active activity, which holds the
        // maximum last_active_seq by the recency invariant).
        if target == self.activities.active_id() {
            unreachable!(
                "activated_count >= 2 and depth >= 1 guarantee idx >= 1; \
                 ordered[0] is always the active activity by the max-seq invariant, \
                 so ordered[idx] cannot equal active_id()"
            );
        }
        self.switch_activity(target);
    }

    /// Create a fresh runtime activity with `name`, returning its id.
    ///
    /// Validation is delegated to [`Activities::create_runtime`] — empty /
    /// whitespace-only names and case-insensitive duplicates are rejected
    /// without mutating any pool state (see [`CreateActivityError`]).
    ///
    /// On success, the new activity id is unioned into every sticky workspace's
    /// `activities` set. Sticky workspaces are "present on every activity" by
    /// definition; whenever the activity pool grows, those sets must grow with
    /// it, or the pool-level union invariant (each workspace's `activities` set
    /// is a subset of live activity ids) would diverge from the intended
    /// semantics. The new activity's `views` map is populated eagerly, not
    /// lazily on first switch: [`Self::ensure_all_activity_views`] materializes
    /// a bookend view for every connected monitor, then
    /// [`Self::reconcile_views_with_membership`] sweeps in any stale-tagged
    /// sticky workspaces the materializer's tag-filtered lift misses.
    ///
    /// The `active` / `previous` cursors on [`Activities`] are not touched; the
    /// caller remains on whatever activity was active before the call.
    pub(crate) fn create_activity(
        &mut self,
        name: String,
    ) -> Result<ActivityId, CreateActivityError> {
        let id = self.activities.create_runtime(name)?;
        for ws in self.workspaces.values_mut() {
            if ws.is_sticky() {
                ws.activities.insert(id);
            }
        }
        // Materialize a bookend view for the new activity on every connected monitor; the
        // per-activity bookend invariant requires it eagerly, not lazily on first switch.
        self.ensure_all_activity_views();

        // Sticky expansion above widened membership; the materializer's tag-filtered lift misses
        // stale-tagged sticky workspaces, so sweep them into the new activity's views. Only the
        // install direction can fire here — creation only widens membership.
        self.reconcile_views_with_membership();

        #[cfg(debug_assertions)]
        self.verify_invariants();

        Ok(id)
    }

    /// Remove the runtime activity identified by `reference`, returning its id
    /// on success.
    ///
    /// Rejection rules (all evaluated before any mutation — see):
    ///
    /// - Unknown reference → [`RemoveActivityError::NotFound`].
    /// - Target is `is_config_declared` → [`RemoveActivityError::ConfigDeclared`].
    /// - Target is the only activity in the pool → [`RemoveActivityError::LastRemaining`].
    /// - Any exclusive workspace (one whose `activities` set is `{target}`) has windows →
    ///   [`RemoveActivityError::ExclusiveWorkspaceHasWindows`].
    /// - Any exclusive workspace is named → [`RemoveActivityError::ExclusiveNamedWorkspace`].
    ///
    /// Precedence on simultaneous violations:
    /// `ConfigDeclared` > `LastRemaining` > `ExclusiveWorkspaceHasWindows` >
    /// `ExclusiveNamedWorkspace`. The exclusive-workspace walk classifies
    /// every candidate before choosing an error, so the result does not depend
    /// on `HashMap` iteration order (non-deterministic across runs).
    ///
    /// On success, the mutation sequence is:
    ///
    /// 1. **Cascade the active cursor** if the target was active, via [`Self::switch_activity`] to
    ///    `previous_id()` (or, failing that, to the first other live activity in declaration
    ///    order). This is the only path that snaps in-flight workspace-switch animations and
    ///    lazy-populates views on the new active activity.
    /// 2. **Destroy exclusive unnamed-empty workspaces** — drop each from the pool, and from every
    ///    activity's per-output `views`. A view with a single entry pointing at the removed id
    ///    drops entirely (mirrors the "view removal when empty" rule); otherwise
    ///    [`WorkspaceView::remove_at`] shifts the active / previous cursors.
    /// 3. **Prune shared workspaces** — drop the target id from every workspace's `activities` set
    ///    where it still appears. At least one other activity id remains in each set (those are
    ///    shared by definition), preserving the non-empty-activities invariant.
    /// 4. **Remove the activity from the pool** via [`Activities::remove`]. That call also clears
    ///    `previous` if it pointed at the removed id — covering both the post-cascade case (where
    ///    step 1 just set `previous = Some(target)`) and the non-active case where `previous`
    ///    already equaled the target.
    pub(crate) fn remove_activity(
        &mut self,
        reference: &ActivityReferenceArg,
    ) -> Result<ActivityId, RemoveActivityError> {
        let target = self
            .resolve_activity_ref(reference)
            .ok_or(RemoveActivityError::NotFound)?;

        // Validation pass — read-only. No field on `self` is mutated until
        // every error class has been ruled out; this mirrors the atomicity
        // contract that `create_activity` inherits from `create_runtime`.
        {
            let target_act = self
                .activities
                .get(target)
                .expect("resolve_activity_ref returned a live id");
            if target_act.is_config_declared() {
                return Err(RemoveActivityError::ConfigDeclared);
            }
        }
        if self.activities.len() == 1 {
            return Err(RemoveActivityError::LastRemaining);
        }

        // Walk all workspaces once, classifying exclusive candidates. Both
        // error conditions are accumulated before we choose which to return,
        // because `HashMap::values()` iteration is non-deterministic — an
        // early-return on the first named-empty would mask a has-windows
        // violation encountered later and flip the user-facing error based on
        // hash order.
        let mut has_windows_violation = false;
        let mut named_violation = false;
        let mut destroy_ids: Vec<WorkspaceId> = Vec::new();
        for ws in self.workspaces.values() {
            if ws.activities().len() == 1 && ws.activities().contains(&target) {
                if ws.has_windows() {
                    has_windows_violation = true;
                } else if ws.name().is_some() {
                    named_violation = true;
                } else {
                    destroy_ids.push(ws.id());
                }
            }
        }
        if has_windows_violation {
            return Err(RemoveActivityError::ExclusiveWorkspaceHasWindows);
        }
        if named_violation {
            return Err(RemoveActivityError::ExclusiveNamedWorkspace);
        }

        // Mutation phase. Order matters: cascade first (so `switch_activity`'s
        // hard-block `debug_assert!` is satisfied by the same outer gate as
        // the keybinding dispatcher), then workspace destruction, then pool
        // prune, then the activity-pool removal.
        if target == self.activities.active_id() {
            let cascade_target = self
                .activities
                .previous_id()
                .or_else(|| {
                    self.activities
                        .iter()
                        .map(|a| a.id())
                        .find(|id| *id != target)
                })
                .expect("LastRemaining was rejected above, so at least one other id exists");
            self.switch_activity(cascade_target);
        }

        // Destroy exclusive unnamed-empty workspaces. The `retain` guard drops
        // the view entry entirely when removing would empty it — `WorkspaceView`
        // cannot be zero-sized, and "View removal when empty" specifies
        // the entry is dropped in that case. Otherwise `remove_at` patches
        // `active` / `previous` and shifts the id list.
        for ws_id in destroy_ids {
            for activity in self.activities.iter_mut() {
                activity.views_mut().retain(|_output_id, view| {
                    let Some(pos) = view.position_of(ws_id) else {
                        return true;
                    };
                    if view.len() == 1 {
                        return false;
                    }
                    view.remove_at(pos);
                    true
                });
            }
            assert!(
                self.workspaces.remove(&ws_id).is_some(),
                "destroy id {ws_id:?} must be a live pool key",
            );
        }

        // Prune shared workspaces — ones where the target coexists with at
        // least one other activity. Exclusives were destroyed or errored out
        // above, so every remaining membership is shared and the set stays
        // non-empty after the remove. Collect pruned ids for the post-remove
        // reclaim pass below.
        let mut pruned_ids: Vec<WorkspaceId> = Vec::new();
        for ws in self.workspaces.values_mut() {
            if ws.activities().contains(&target) {
                debug_assert!(
                    ws.activities().len() > 1,
                    "exclusive workspaces were destroyed or errored in the validation pass",
                );
                ws.activities.remove(&target);
                pruned_ids.push(ws.id());
            }
        }

        // Remove from the pool. `Activities::remove` clears `previous` when it
        // pointed at `target`, covering both the post-cascade case (step 1 set
        // previous = Some(target)) and any non-active case where previous
        // happened to already equal target.
        let _ = self.activities.remove(target);

        // Any monitor whose activity-switch transition was departing from `target` now holds a
        // dead id in `from`. Snap those before the invariant check fires.
        self.snap_stale_activity_switches();

        // Destroying exclusive workspaces above may have dropped single-entry views in other
        // activities (cross-activity `retain` at `destroy_workspaces_cross_activity`). Re-run
        // the materializer so every remaining activity holds a bookend view for every
        // connected monitor.
        self.ensure_all_activity_views();

        // The membership prune above shrank each pruned workspace's activities set
        // by one. Any that became exclusive to a single activity may now sit in an
        // illegal middle position (previously legal as a pinned shared workspace).
        self.reclaim_unpinned_empty_workspaces(pruned_ids);

        #[cfg(debug_assertions)]
        self.verify_invariants();

        Ok(target)
    }

    /// Clear any in-flight activity-switch transition whose outgoing activity is no longer a live
    /// pool key.
    ///
    /// Defensive: does not assume the caller redirected any in-flight transition before deleting
    /// an activity. Any `from` id that is no longer a live pool key is cleared. Without this
    /// sweep the monitor state would hold a permanently stale dead id; in debug builds,
    /// `verify_invariants` (which checks `from` liveness) would panic on the next refresh.
    fn snap_stale_activity_switches(&mut self) {
        let activities = &self.activities;
        for mon in &mut self.monitors {
            if mon
                .activity_switch
                .as_ref()
                .is_some_and(|s| !activities.contains(s.from))
            {
                mon.activity_switch = None;
            }
        }
    }

    /// Clear every monitor's in-flight activity-switch transition, snapping it to its end.
    ///
    /// Used when entering or leaving the overview: a strip slide and the overview spatial map
    /// must not run simultaneously. The close direction is a structural no-op (nothing can be
    /// armed while the overview is open, enforced by the overview/slide exclusion invariant), so
    /// the unconditional clear is simply simpler than gating on direction.
    fn snap_all_activity_switches(&mut self) {
        for mon in &mut self.monitors {
            mon.activity_switch = None;
        }
    }

    /// Rename the runtime activity identified by `reference`, returning its id
    /// on success.
    ///
    /// Rejection rules (all evaluated before any mutation):
    ///
    /// - Unknown reference → [`RenameActivityError::NotFound`].
    /// - Target is `is_config_declared` → [`RenameActivityError::ConfigDeclared`].
    /// - `name.trim().is_empty()` → [`RenameActivityError::EmptyName`].
    /// - `name` collides case-insensitively with a *different* activity's name →
    ///   [`RenameActivityError::DuplicateName`]. Renaming to a case variant of the target's own
    ///   current name (or its exact current name) succeeds — the target is excluded from the
    ///   duplicate scan.
    ///
    /// Precedence on overlapping violations:
    /// `NotFound` > `ConfigDeclared` > `EmptyName` / `DuplicateName`. The
    /// outer check rejects config-declared before delegating to
    /// [`Activities::rename_runtime`], so an empty or duplicate rename of a
    /// config-declared activity surfaces as `ConfigDeclared`, never as the
    /// inner validation error.
    ///
    /// Rename is pure metadata: no view patching, no cascade, no workspace-set
    /// changes. The [`Activities`] pool's `active` / `previous` cursors are
    /// unaffected.
    pub(crate) fn rename_activity(
        &mut self,
        reference: &ActivityReferenceArg,
        name: String,
    ) -> Result<ActivityId, RenameActivityError> {
        let target = self
            .resolve_activity_ref(reference)
            .ok_or(RenameActivityError::NotFound)?;

        if self
            .activities
            .get(target)
            .expect("resolve_activity_ref returned a live id")
            .is_config_declared()
        {
            return Err(RenameActivityError::ConfigDeclared);
        }

        self.activities.rename_runtime(target, name)?;

        #[cfg(debug_assertions)]
        self.verify_invariants();

        Ok(target)
    }

    /// Returns `Some(ActivitySwitchBlock::WorkspaceSwitchGesture)` if any
    /// monitor carries an in-flight `WorkspaceSwitch::Gesture(_)`, else `None`.
    ///
    /// `RemoveWorkspaceFromActivity` and `SetWorkspaceActivities` are
    /// hard-blocked by an in-flight workspace-switch *gesture* — removing an id
    /// from the current activity's `view.ids` would invalidate the gesture's
    /// fractional position targets.
    /// `AddWorkspaceToActivity` is not gated: the insert adjusts any in-flight
    /// workspace switch on the target monitor when the active view is patched,
    /// and is safe during animations and gestures alike.
    ///
    /// This predicate is weaker than [`Self::is_activity_switch_hard_blocked`]:
    /// it does **not** consider `self.interactive_move` / `self.dnd` (those are
    /// orthogonal to workspace-activity membership) and it does **not** block
    /// on `WorkspaceSwitch::Animation(_)` — animations are snapped by the
    /// mutator before proceeding, matching `switch_activity`'s snap+proceed
    /// contract.
    pub(crate) fn is_workspace_activity_assignment_blocked_by_gesture(
        &self,
    ) -> Option<ActivitySwitchBlock> {
        for mon in &self.monitors {
            if matches!(mon.workspace_switch, Some(WorkspaceSwitch::Gesture(_))) {
                return Some(ActivitySwitchBlock::WorkspaceSwitchGesture);
            }
        }
        None
    }

    /// Add `workspace` to `activity_ref`'s membership set and, if the activity
    /// has a view for the output currently holding the workspace, insert the id above the
    /// view's trailing-empty bookend (or append when the workspace is itself
    /// empty-unnamed, since it is a valid bookend position).
    ///
    /// Resolution order:
    /// 1. `activity_ref` → `AddWorkspaceToActivityError::ActivityNotFound`.
    /// 2. `workspace` (via [`Self::find_workspace_by_ref`] or, when `None`, the active workspace) →
    ///    `AddWorkspaceToActivityError::WorkspaceNotFound`.
    ///
    /// Semantics:
    /// - No-op when the workspace's `activities` set already contains the target id (returns `Ok`,
    ///   no state touched).
    /// - Otherwise `activity_id` is inserted into `ws.activities`. The workspace's holding view is
    ///   located via [`Self::workspace_holding_output`] — **not** `ws.output_id()`, which is a
    ///   reclaim tag that can go stale after a partial disconnect. If the target activity already
    ///   has a view for the holding output, the id is inserted above the trailing-empty bookend (or
    ///   appended when the workspace is itself empty-unnamed). Every activity holds a view for
    ///   every connected output eagerly (the per-activity bookend invariant), so there is no
    ///   "dormant activity without a view" case to fabricate around. The pre-insert absence guard
    ///   checks every view of the target activity, not just the holding-output one, so a
    ///   pre-existing membership↔view incoherence elsewhere in the same activity can't be
    ///   compounded by a double-install.
    ///
    /// Not gated. The insert adjusts any in-flight workspace switch on the
    /// target monitor when the active view is patched, so this is safe during
    /// both workspace-switch animations and gestures.
    pub(crate) fn add_workspace_to_activity(
        &mut self,
        workspace: Option<WorkspaceReference>,
        activity_ref: &ActivityReferenceArg,
    ) -> Result<(WorkspaceId, ActivityId), AddWorkspaceToActivityError> {
        let activity_id = self
            .resolve_activity_ref(activity_ref)
            .ok_or(AddWorkspaceToActivityError::ActivityNotFound)?;

        let ws_id = self
            .resolve_workspace_ref_or_active(workspace)
            .ok_or(AddWorkspaceToActivityError::WorkspaceNotFound)?;

        // No-op early-exit: the workspace already belongs to the activity.
        // Do not touch views or invariants — the state is unchanged.
        {
            let ws = self
                .workspaces
                .get(&ws_id)
                .expect("resolved ws_id must be a live pool key");
            if ws.activities().contains(&activity_id) {
                return Ok((ws_id, activity_id));
            }
        }

        // Resolve the holding view's output before entering the split-borrow scope below —
        // `workspace_holding_output` takes `&self`, which cannot coexist with the
        // `&mut self.workspaces` / `&mut self.activities` split that follows.
        let holding_out_id = self.workspace_holding_output(ws_id);

        // Patch ws.activities + the target activity's view in a tight scope so
        // the split mutable borrows on `self.workspaces` / `self.activities`
        // die before the bookend sweep below needs `&mut self`.
        {
            let pool = &mut self.workspaces;
            let activities = &mut self.activities;
            let active_id = activities.active_id();

            let ws = pool
                .get_mut(&ws_id)
                .expect("resolved ws_id must be a live pool key");
            ws.activities.insert(activity_id);

            if let Some(out_id_ref) = holding_out_id.as_ref() {
                if let Some(activity) = activities.get_mut(activity_id) {
                    // Widened guard: absent from every view of the target activity, not just
                    // the holding-output view. Under the no-op early-exit above the workspace
                    // cannot legitimately already be in one of this activity's views (per-view
                    // uniqueness is derived from pool membership) — this check is belt-and-
                    // braces against a pre-existing membership↔view incoherence elsewhere in
                    // the same activity (a separate, not-yet-fixed producer) being compounded
                    // by a double-install.
                    let already_present = activity
                        .views()
                        .values()
                        .any(|view| view.ids().contains(&ws_id));
                    if !already_present {
                        if let Some(view) = activity.views_mut().get_mut(out_id_ref) {
                            let pos = Self::view_insert_above_trailing_bookend(pool, view, ws_id);
                            // A mid-view insert into the view the monitor is
                            // rendering must shift an in-flight workspace
                            // switch, mirroring `add_workspace_at_on`.
                            if activity_id == active_id {
                                if let Some(mon) = self
                                    .monitors
                                    .iter_mut()
                                    .find(|m| &m.output_id() == out_id_ref)
                                {
                                    if let Some(switch) = &mut mon.workspace_switch {
                                        if pos as f64 <= switch.target_idx() {
                                            switch.offset(1);
                                        }
                                    }
                                }
                            }
                        } else {
                            // `holding_out_id` names a currently-connected output, and the
                            // per-activity bookend invariant guarantees every activity holds a
                            // view for every connected output — so this activity must have a
                            // view keyed by `out_id_ref`. Reaching here means that invariant is
                            // violated; `ws.activities.insert` above already ran, so silently
                            // skipping the view patch would mint a membership-without-view
                            // incoherence.
                            debug_assert!(
                                false,
                                "add_workspace_to_activity: activity {activity_id:?} has no \
                                 view for {out_id_ref:?}, a connected output — per-activity \
                                 bookend invariant violated (membership↔view coherence bug)",
                            );
                        }
                    }
                }
            }
        }

        // Per-activity bookend invariant: every (activity, connected output)
        // view must end in an empty unnamed workspace. The insert above keeps
        // an existing trailing-empty bookend at the tail, so the only rule a
        // windowed-or-named `ws_id` can still break is EWAF's leading empty —
        // when the target view held a single empty entry (doubling as both
        // bookends) and `ws_id` landed at position 0. Both active and dormant
        // views are repaired by the sweep's leading-empty arm. An empty unnamed
        // `ws_id` is itself a valid bookend and the entire repair block
        // short-circuits.
        //
        // Silent-skip case: when `ws_id` has no holding view in any activity — the
        // fully-disconnected window, where every workspace is parked in
        // `disconnected_workspace_ids` and no activity holds any view — no
        // sweep runs here. The bookend rule is re-asserted on reconnect via
        // `Monitor::new`'s materializer. Matches the `set_workspace_name` precedent — no
        // warn!, no eager repair, no rendering happens against a disconnected output.
        let ws_needs_repair = self
            .workspaces
            .get(&ws_id)
            .expect("resolved ws_id must be a live pool key")
            .has_windows_or_name();
        if ws_needs_repair {
            if let Some(out_id) = holding_out_id {
                if self.monitors.iter().any(|m| m.output_id() == out_id) {
                    self.normalize_view_bookends();
                }
            }
        }

        #[cfg(debug_assertions)]
        self.verify_invariants();

        Ok((ws_id, activity_id))
    }

    /// Remove `workspace` from `activity_ref`'s membership set and patch the
    /// view actually holding the workspace within that activity (if any).
    ///
    /// Resolution order:
    /// 1. `activity_ref` → `RemoveWorkspaceFromActivityError::ActivityNotFound`.
    /// 2. `workspace` → `RemoveWorkspaceFromActivityError::WorkspaceNotFound`.
    ///
    /// Semantics:
    /// - No-op when the workspace's `activities` set does not contain the target id (returns `Ok`).
    /// - If removing would empty `ws.activities`, returns
    ///   `RemoveWorkspaceFromActivityError::LastActivity` without mutating any state
    ///   (guard-before-mutate discipline; invariant is enforced by `Layout::verify_invariants`).
    /// - Otherwise: if the active activity has an in-flight `WorkspaceSwitch::Animation` on any
    ///   monitor AND the target activity is the active one, the animation is snapped on every
    ///   monitor first (matching `switch_activity`'s snap+proceed contract). Then `activity_id` is
    ///   removed from `ws.activities`. Finally, `activity_id`'s own views are scanned for the entry
    ///   holding `ws_id` — **not** `ws.output_id()`, which is a reclaim tag that can go stale after
    ///   a partial disconnect (see [`Self::workspace_holding_output`]'s rustdoc) — and the holding
    ///   view (if any) is patched: if its single entry was `ws_id` it is dropped entirely
    ///   (mirroring `destroy_workspaces_cross_activity` behavior); otherwise
    ///   `WorkspaceView::remove_at` shifts the active / previous cursors.
    /// - Active-activity special case: when dropping the last view entry on a connected monitor's
    ///   output for the active activity, the cross-field invariant `active.views.len() ==
    ///   monitors.len()` would be violated — [`Self::ensure_all_activity_views`] is called
    ///   immediately after the drop to reinstate the view (fresh trailing empty + EWAF leading
    ///   empty if applicable).
    ///
    /// Callers that dispatch via IPC must first consult
    /// [`Self::is_workspace_activity_assignment_blocked_by_gesture`] — a
    /// gesture in flight on any monitor blocks the call.
    /// This method does not re-check that predicate; the caller owns the
    /// decision to queue vs. proceed.
    pub(crate) fn remove_workspace_from_activity(
        &mut self,
        workspace: Option<WorkspaceReference>,
        activity_ref: &ActivityReferenceArg,
    ) -> Result<(WorkspaceId, ActivityId), RemoveWorkspaceFromActivityError> {
        let activity_id = self
            .resolve_activity_ref(activity_ref)
            .ok_or(RemoveWorkspaceFromActivityError::ActivityNotFound)?;

        let ws_id = self
            .resolve_workspace_ref_or_active(workspace)
            .ok_or(RemoveWorkspaceFromActivityError::WorkspaceNotFound)?;

        // Read-only inspection phase — no mutation until every error class
        // has been ruled out ( guard-before-mutate).
        let (is_member, len_before) = {
            let ws = self
                .workspaces
                .get(&ws_id)
                .expect("resolved ws_id must be a live pool key");
            (
                ws.activities().contains(&activity_id),
                ws.activities().len(),
            )
        };

        if !is_member {
            // No-op: the workspace is already not a member. "No-op
            // if already not in activity".
            return Ok((ws_id, activity_id));
        }

        if len_before == 1 {
            // `activities.len() - 1 == 0` — removing would violate.
            return Err(RemoveWorkspaceFromActivityError::LastActivity);
        }

        // Active-activity branch: snap any in-flight workspace-switch
        // animation on every monitor before patching views. Mirrors
        // `switch_activity`'s step-4 clear. Only runs when the mutation
        // would affect the active activity's view that the animation's
        // fractional positions refer to.
        if activity_id == self.activities.active_id() {
            for mon in &mut self.monitors {
                if matches!(mon.workspace_switch, Some(WorkspaceSwitch::Animation(_))) {
                    mon.workspace_switch = None;
                }
            }
        }

        {
            let ws = self
                .workspaces
                .get_mut(&ws_id)
                .expect("resolved ws_id must be a live pool key");
            ws.activities.remove(&activity_id);
        }

        // Patch the view actually holding `ws_id` within `activity_id` (if any). Track any
        // view-entry drop on a connected monitor so the materializer can reinstate the dropped
        // view via ensure_all_activity_views below.
        //
        // Scans `activity_id`'s own views rather than calling the layout-wide
        // `Self::workspace_holding_output` — narrower and the correct scope here: `is_member`
        // above guarantees `ws_id` was a member of `activity_id`, and the per-activity bookend
        // invariant guarantees every activity holds a view for every connected output, so the
        // workspace's holding view (if it has one at all) must be one of `activity_id`'s own
        // views, not some other activity's.
        let mut dropped_any_view_entry = false;
        let mut collapse_out_id: Option<OutputId> = None;
        let holding = {
            let activity = self
                .activities
                .get(activity_id)
                .expect("resolve_activity_ref returned a live id");
            activity
                .views()
                .iter()
                .find_map(|(out_id, view)| view.position_of(ws_id).map(|pos| (out_id.clone(), pos)))
        };
        // `is_member` above guarantees `ws_id` was a member of `activity_id`; the per-activity
        // bookend invariant guarantees every activity holds a view for every connected output.
        // With any monitor connected, `holding == None` is therefore a membership↔view
        // coherence bug, not a legitimate outcome — mirrors `workspace_holding_output`'s own
        // debug-loud discipline.
        debug_assert!(
            holding.is_some() || self.monitors.is_empty(),
            "remove_workspace_from_activity: {ws_id:?} was a member of {activity_id:?} but no \
             view in that activity holds it while a monitor is connected — membership↔view \
             coherence bug",
        );
        if let Some((out_id, pos)) = holding {
            // `is_connected` is always true here: every view key is a currently-connected
            // output's OutputId — dormant activities' views for a disconnecting output are
            // migrated away by `remove_output`'s partial-disconnect walk, so a `holding` result
            // always names a connected output.
            let is_connected = self.monitors.iter().any(|m| m.output_id() == out_id);
            let activity = self
                .activities
                .get_mut(activity_id)
                .expect("resolve_activity_ref returned a live id");
            let view = activity
                .views_mut()
                .get_mut(&out_id)
                .expect("holding view was found in the shared-borrow scan above");
            if view.len() == 1 {
                // Drop the single-entry view outright — mirrors the
                // `destroy_workspaces_cross_activity` single-entry retain-drop path.
                activity.views_mut().remove(&out_id);
                if is_connected {
                    dropped_any_view_entry = true;
                }
            } else {
                view.remove_at(pos);
                // A `remove_at` on a length-3 view leaves length 2, which the collapse helper
                // normalizes below when both survivors are empty and the second is exclusive.
                collapse_out_id = Some(out_id);
            }
        }

        // Reinstate dropped views via the materializer. `ensure_all_activity_views` takes
        // `&mut self`, so every nested borrow above must already be released (they are — the
        // scope above ended).
        if dropped_any_view_entry {
            self.ensure_all_activity_views();
        }

        // Normalize any all-empty exclusive EWAF len-2 view the removal left behind, before the
        // narrowed-workspace reclaim and the tail verify (the len-2 shape trips
        // `assert_view_bookends`). Guarded and idempotent — a non-len-2 view early-returns.
        if let Some(out_id) = collapse_out_id {
            self.collapse_empty_exclusive_ewaf_len2_view(activity_id, &out_id);
        }

        // The narrowing may have left ws_id exclusively in one activity's view
        // while it sits in an illegal middle position (previously legal as a
        // pinned shared workspace). Reclaim if so.
        self.reclaim_unpinned_empty_workspaces([ws_id]);

        #[cfg(debug_assertions)]
        self.verify_invariants();

        Ok((ws_id, activity_id))
    }

    /// Replace the `activities` set of `workspace` with `activity_refs`,
    /// patching every affected activity's view holding the workspace (if any).
    ///
    /// Resolution order (guard-before-mutate — no mutation happens until
    /// every error class is ruled out):
    /// 1. Every ref in `activity_refs` is resolved. The first unresolvable ref short-circuits to
    ///    `SetWorkspaceActivitiesError::ActivityNotFound` — precedence over every subsequent check,
    ///    matching the `Add` / `Remove` precedent.
    /// 2. If the resolved set is empty → `EmptyActivityList`.
    /// 3. Resolve the workspace reference. On miss →
    ///    `SetWorkspaceActivitiesError::WorkspaceNotFound`.
    ///
    /// Wire-surfaces via
    /// `DoActionError::SetWorkspaceActivities(SetWorkspaceActivitiesError::WorkspaceNotFound)`.
    ///
    /// Semantics:
    /// - Symmetric diff: `to_remove = old ∖ new`, `to_add = new ∖ old`. Removes scan each removed
    ///   activity's own views for the entry holding the workspace, then apply the same single-entry
    ///   drop / `remove_at` patch as [`Self::remove_workspace_from_activity`]. Adds insert above
    ///   the trailing-empty bookend in the target activity's view for the workspace's layout-wide
    ///   holding output (via [`Self::workspace_holding_output`] — **not** `ws.output_id()`, which
    ///   can go stale after a partial disconnect), or append when the workspace is itself
    ///   empty-unnamed.
    /// - No-op when `new == old` — returns without mutating any state.
    /// - If the active activity id is in the symmetric diff AND any monitor has a
    ///   `WorkspaceSwitch::Animation`, the animation is snapped on every monitor before patching (
    ///   snap+proceed, mirroring `remove_workspace_from_activity`).
    /// - If the active activity's view for a connected monitor is emptied by a single-entry drop on
    ///   the Remove side, [`Self::ensure_all_activity_views`] is called to reinstate the
    ///   cross-field invariant `active.views.len() == monitors.len()`.
    ///
    /// Does NOT destroy the workspace when `to_remove` prunes its last
    /// activity — the `EmptyActivityList` gate guarantees `new` is
    /// non-empty, so `ws.activities` stays non-empty post-call.
    /// distinguishes "prune the activity tag" from "destroy the workspace";
    /// only the latter goes through `destroy_workspaces_cross_activity`.
    ///
    /// The `focus: bool` parameter of the IPC action is NOT surfaced here —
    /// this method does not flip the active activity. Callers that chain
    /// into `switch_activity` handle the cursor move and the
    /// inhibitor refresh at the dispatch layer (`MoveWorkspaceToActivity`
    /// is the only such caller today).
    ///
    /// Returns `(WorkspaceId, new_activities_set, active_activity_affected)`.
    /// The `active_activity_affected` flag is `true` iff the active
    /// activity's id is in the symmetric diff — the dispatch layer uses it
    /// to decide whether a cursor warp / redraw is needed.
    ///
    /// Callers that dispatch via IPC must first consult
    /// [`Self::is_workspace_activity_assignment_blocked_by_gesture`] — a
    /// gesture in flight on any monitor blocks the call.
    pub(crate) fn set_workspace_activities(
        &mut self,
        workspace: Option<WorkspaceReference>,
        activity_refs: &[ActivityReferenceArg],
    ) -> Result<(WorkspaceId, HashSet<ActivityId>, bool), SetWorkspaceActivitiesError> {
        // Resolve every activity ref. First unresolvable ref → ActivityNotFound
        // (precedence over EmptyActivityList — matches the `resolve_activity_ref`
        // precedence of `Add` / `Remove` so a `[unresolvable_id]` of length 1
        // yields `ActivityNotFound`, not `EmptyActivityList`).
        let mut new_set: HashSet<ActivityId> = HashSet::with_capacity(activity_refs.len());
        for r in activity_refs {
            let id = self
                .resolve_activity_ref(r)
                .ok_or(SetWorkspaceActivitiesError::ActivityNotFound)?;
            new_set.insert(id);
        }

        if new_set.is_empty() {
            return Err(SetWorkspaceActivitiesError::EmptyActivityList);
        }

        // Resolve the workspace. On miss → WorkspaceNotFound (wire-surfaced).
        let ws_id = self
            .resolve_workspace_ref_or_active(workspace)
            .ok_or(SetWorkspaceActivitiesError::WorkspaceNotFound)?;

        // Snapshot the old membership set before any mutation. Resolve the layout-wide holding
        // output before the split-borrow scope below — `workspace_holding_output` takes `&self`,
        // which cannot coexist with the `&mut self.workspaces` write that follows.
        let old_set = self
            .workspaces
            .get(&ws_id)
            .expect("resolved ws_id must be a live pool key")
            .activities()
            .clone();
        let holding_out_id = self.workspace_holding_output(ws_id);

        let to_remove: Vec<ActivityId> = old_set.difference(&new_set).copied().collect();
        let to_add: Vec<ActivityId> = new_set.difference(&old_set).copied().collect();
        let active_id = self.activities.active_id();
        let active_activity_affected =
            to_remove.contains(&active_id) || to_add.contains(&active_id);

        // Identity case: new == old, no mutation.
        if to_remove.is_empty() && to_add.is_empty() {
            return Ok((ws_id, new_set, active_activity_affected));
        }

        // Snap any in-flight animation on every monitor if the active activity
        // is in the diff — the animation's fractional targets refer to the
        // active activity's view, which may change length (snap+proceed).
        if active_activity_affected {
            for mon in &mut self.monitors {
                if matches!(mon.workspace_switch, Some(WorkspaceSwitch::Animation(_))) {
                    mon.workspace_switch = None;
                }
            }
        }

        {
            let ws = self
                .workspaces
                .get_mut(&ws_id)
                .expect("resolved ws_id must be a live pool key");
            ws.activities = new_set.clone();
        }

        // Patch each affected activity's view. Track any view-entry drop on a connected monitor
        // so the materializer can reinstate the dropped view via ensure_all_activity_views below.
        let mut dropped_any_view_entry = false;
        // (activity, output) pairs a Remove narrowed to length 2 — normalized after the loop.
        let mut collapse_pairs: Vec<(ActivityId, OutputId)> = Vec::new();

        // Removes first — mirrors `remove_workspace_from_activity`'s single-entry drop vs
        // `remove_at` branch. Each removed activity is scanned independently for its own holding
        // view (not the layout-wide `holding_out_id`): under a stale-tag / partial-disconnect
        // history, different activities in `to_remove` can hold `ws_id` in views keyed by
        // different outputs. The drop-to-zero flag fires for any activity (active or dormant)
        // hitting the single-entry path on a connected output.
        for act_id in &to_remove {
            let holding = {
                let activity = self
                    .activities
                    .get(*act_id)
                    .expect("resolve_activity_ref returned a live id");
                activity.views().iter().find_map(|(out_id, view)| {
                    view.position_of(ws_id).map(|pos| (out_id.clone(), pos))
                })
            };
            // `act_id` is drawn from `to_remove = old_set ∖ new_set`, so `ws_id` was a member
            // of `*act_id` before this call; the per-activity bookend invariant guarantees every
            // activity holds a view for every connected output. With any monitor connected,
            // `holding == None` is therefore a membership↔view coherence bug — mirrors
            // `workspace_holding_output`'s own debug-loud discipline.
            debug_assert!(
                holding.is_some() || self.monitors.is_empty(),
                "set_workspace_activities: {ws_id:?} was a member of {act_id:?} but no view in \
                 that activity holds it while a monitor is connected — membership↔view \
                 coherence bug",
            );
            let Some((out_id, pos)) = holding else {
                continue;
            };
            // is_connected is always true here: every view key is a currently-connected
            // output's OutputId, per the connected-keyspace invariant (dormant activities'
            // views for a disconnecting output are migrated away by `remove_output`'s
            // partial-disconnect walk).
            let is_connected = self.monitors.iter().any(|m| m.output_id() == out_id);
            let activity = self
                .activities
                .get_mut(*act_id)
                .expect("resolve_activity_ref returned a live id");
            let view = activity
                .views_mut()
                .get_mut(&out_id)
                .expect("holding view was found in the shared-borrow scan above");
            if view.len() == 1 {
                activity.views_mut().remove(&out_id);
                if is_connected {
                    dropped_any_view_entry = true;
                }
            } else {
                view.remove_at(pos);
                // Length-3 → length-2: a candidate for the all-empty exclusive EWAF collapse
                // applied after the loop.
                collapse_pairs.push((*act_id, out_id));
            }
        }

        // Adds: insert into the target activity's view for the layout-wide holding output,
        // keeping a trailing-empty bookend at the tail. Every activity holds a view for every
        // connected output eagerly (the per-activity bookend invariant), so there is no "dormant
        // activity without a view" case to fabricate around. Widened absence guard mirrors
        // `add_workspace_to_activity`: absent from every view of the target activity, not just
        // the holding-output view. No in-flight switch shift is needed even when the active view
        // is patched: any switch was snapped above (the active activity is in the diff).
        if let Some(out_id_ref) = holding_out_id.as_ref() {
            for act_id in &to_add {
                let activity = self
                    .activities
                    .get_mut(*act_id)
                    .expect("resolve_activity_ref returned a live id");
                let already_present = activity
                    .views()
                    .values()
                    .any(|view| view.ids().contains(&ws_id));
                if !already_present {
                    if let Some(view) = activity.views_mut().get_mut(out_id_ref) {
                        Self::view_insert_above_trailing_bookend(&self.workspaces, view, ws_id);
                    } else {
                        // `holding_out_id` names a currently-connected output, and the
                        // per-activity bookend invariant guarantees every activity holds a view
                        // for every connected output — so this activity must have a view keyed
                        // by `out_id_ref`. Reaching here means that invariant is violated;
                        // `ws.activities = new_set` above already ran, so silently skipping the
                        // view patch would mint a membership-without-view incoherence.
                        debug_assert!(
                            false,
                            "set_workspace_activities: activity {act_id:?} has no view for \
                             {out_id_ref:?}, a connected output — per-activity bookend \
                             invariant violated (membership↔view coherence bug)",
                        );
                    }
                }
            }
        }

        // Whether the post-loop bookend repair can run at all, keyed by the holding output
        // being currently connected.
        let holding_output_connected = holding_out_id
            .as_ref()
            .is_some_and(|out_id| self.monitors.iter().any(|m| &m.output_id() == out_id));

        // Per-activity bookend invariant: every (activity, connected output)
        // view must end in an empty unnamed workspace. The Add loop above
        // keeps an existing trailing-empty bookend at the tail, so the only
        // rule a windowed-or-named `ws_id` can still break is EWAF's leading
        // empty — when a target view held a single empty entry (doubling as
        // both bookends) and `ws_id` landed at position 0. Both active and
        // dormant views are repaired by the sweep's leading-empty arm. An
        // empty unnamed `ws_id` is itself a valid bookend and the entire
        // repair block short-circuits.
        //
        // Silent-skip case: when `ws_id` has no holding view in any activity — the
        // fully-disconnected window, where every workspace is parked in
        // `disconnected_workspace_ids` and no activity holds any view — no
        // sweep runs here. The bookend rule is re-asserted on reconnect via
        // `Monitor::new`'s materializer. Matches the `set_workspace_name` precedent — no
        // warn!, no eager repair, no rendering happens against a disconnected output.
        let ws_needs_repair = self
            .workspaces
            .get(&ws_id)
            .expect("resolved ws_id must be a live pool key")
            .has_windows_or_name();
        if ws_needs_repair && holding_output_connected {
            // Safe against the sweep's stricter missing-view assert: ws_needs_repair means
            // ws_id is windowed-or-named, and the bookend invariant forbids that from being
            // a view's sole (len-1) entry — the only shape the to_remove loop above drops —
            // so no activity here can be missing a view when this runs.
            self.normalize_view_bookends();
        }

        // Reinstate any dropped views via the materializer — mirrors the
        // `remove_workspace_from_activity` recipe.
        if dropped_any_view_entry {
            self.ensure_all_activity_views();
        }

        // Normalize any all-empty exclusive EWAF len-2 view a Remove left behind, before the
        // narrowed-workspace reclaim and the tail verify (the len-2 shape trips
        // `assert_view_bookends`). Guarded and idempotent — duplicate pairs and non-len-2 views
        // early-return.
        for (act_id, out_id) in &collapse_pairs {
            self.collapse_empty_exclusive_ewaf_len2_view(*act_id, out_id);
        }

        // When to_remove is non-empty, ws_id may have been narrowed to a single
        // activity. If it now sits in an illegal middle position under its sole
        // remaining owner, reclaim it.
        if !to_remove.is_empty() {
            self.reclaim_unpinned_empty_workspaces([ws_id]);
        }

        #[cfg(debug_assertions)]
        self.verify_invariants();

        Ok((ws_id, new_set, active_activity_affected))
    }

    /// Move `workspace` from the currently-active activity to
    /// `activity_ref`.
    ///
    /// Sugar for `AddWorkspaceToActivity(target) +
    /// RemoveWorkspaceFromActivity(active)`, applied atomically in that
    /// order. The Add-then-Remove ordering guarantees that the
    /// workspace is never transiently outside every activity, so the
    /// non-empty invariant is never violated — even when the workspace was
    /// previously exclusive to the active activity.
    ///
    /// Multi-activity semantics: if the workspace belongs to
    /// `{active, X, Y}`, after the move it belongs to `{X, Y, target}` —
    /// the workspace leaves the active activity but stays in the others.
    ///
    /// Resolution order:
    /// 1. `activity_ref` → `ActivityNotFound`.
    /// 2. `workspace` → `WorkspaceNotFound`.
    /// 3. If workspace's `activities` set does not contain the active activity id →
    ///    `WorkspaceNotInActiveActivity`.
    ///
    /// No-op cases (return `Ok` without mutating):
    /// - `target == active_id`.
    ///
    /// Implementation strategy: this method delegates to
    /// [`Self::add_workspace_to_activity`] + [`Self::remove_workspace_from_activity`]
    /// rather than reimplementing view-patching inline. Each delegate runs
    /// its own `verify_invariants`; the intermediate state (workspace
    /// tagged with both `active` and `target`) satisfies every invariant
    /// individually, so chaining is correctness-preserving.
    /// "Sugar for Add + Remove" is the load-bearing contract —
    /// inlining view-patching here would make the two paths drift on the
    /// next bug fix.
    ///
    /// If the workspace already has `target` in its activities set (but
    /// `source != target`), the Add step is a no-op at
    /// [`Self::add_workspace_to_activity`]; the Remove step still prunes
    /// source.
    ///
    /// The `focus: bool` discriminator is NOT consumed here — the dispatch
    /// layer chains into [`Self::switch_activity`] after a successful
    /// move, and fires the keyboard-shortcut-inhibitor refresh on
    /// that path. Keeping focus-flipping out of the Layout method keeps
    /// this API orthogonal to cursor movement.
    ///
    /// Returns `(workspace_id, target_activity_id, source_activity_id)`
    /// where `source_activity_id` is always the currently-active activity
    /// id at call time — logged by the dispatch layer.
    ///
    /// Callers that dispatch via IPC must first consult the appropriate
    /// hard-block predicate:
    /// - `focus: false` → [`Self::is_workspace_activity_assignment_blocked_by_gesture`]
    /// - `focus: true` → [`Self::is_activity_switch_hard_blocked`]
    pub(crate) fn move_workspace_to_activity(
        &mut self,
        workspace: Option<WorkspaceReference>,
        activity_ref: &ActivityReferenceArg,
    ) -> Result<(WorkspaceId, ActivityId, ActivityId), MoveWorkspaceToActivityError> {
        let target_id = self
            .resolve_activity_ref(activity_ref)
            .ok_or(MoveWorkspaceToActivityError::ActivityNotFound)?;

        let ws_id = self
            .resolve_workspace_ref_or_active(workspace)
            .ok_or(MoveWorkspaceToActivityError::WorkspaceNotFound)?;

        // Move requires the workspace to be a member of the currently-active
        // activity.
        let source_id = self.activities.active_id();
        {
            let ws = self
                .workspaces
                .get(&ws_id)
                .expect("resolved ws_id must be a live pool key");
            if !ws.activities().contains(&source_id) {
                return Err(MoveWorkspaceToActivityError::WorkspaceNotInActiveActivity);
            }
        }

        // No-op: target == source implies the workspace stays put.
        // Subsumes the "No-op if workspace already exclusively
        // in target" row — when source == target, the workspace's
        // activities cannot change. Leaves verify_invariants untouched.
        if target_id == source_id {
            return Ok((ws_id, target_id, source_id));
        }

        // Delegate to Add-then-Remove. Both operations are `&mut self`
        // and run their own `verify_invariants`. The intermediate state
        // (workspace tagged with both source and target) is invariant-
        // preserving — this is the correctness argument for "sugar for
        // Add + Remove".
        //
        // If the workspace already has target (but source != target), the
        // Add is a no-op at the delegate; the Remove still prunes source.
        // After Add the workspace has both {source, target, ...} — len >= 2,
        // so Remove's LastActivity guard will not fire.
        self.add_workspace_to_activity(
            Some(WorkspaceReference::Id(ws_id.get())),
            &ActivityReferenceArg::Id(target_id.get()),
        )
        .expect(
            "Add cannot fail: target was just resolved and ws_id is live; the delegate's only \
             error classes (ActivityNotFound, WorkspaceNotFound) are already ruled out",
        );
        self.remove_workspace_from_activity(
            Some(WorkspaceReference::Id(ws_id.get())),
            &ActivityReferenceArg::Id(source_id.get()),
        )
        .expect(
            "Remove cannot fail: source is the active activity (must resolve), ws_id is live, \
             and Add guaranteed ws.activities.len() >= 2 so LastActivity is not reachable",
        );

        // Delegates already ran `verify_invariants` in debug builds — no
        // extra chain needed here. Return the triple the dispatch layer
        // logs.
        Ok((ws_id, target_id, source_id))
    }

    /// Set `workspace.is_sticky = true` and expand its `activities` set to all
    /// live activity ids.
    ///
    /// Resolution order:
    /// 1. `workspace` → `WorkspaceNotFound`. Wire-surfaced via
    ///    `DoActionError::SetWorkspaceSticky(WorkspaceStickyError::WorkspaceNotFound)`.
    ///
    /// No-op cases (return `Ok((ws_id, false))` without mutating):
    /// - `ws.is_sticky() == true` AND `ws.activities() == all_live_ids` — the typical state for an
    ///   already-sticky workspace.
    ///
    /// Implementation strategy: delegate to [`Self::set_workspace_activities`]
    /// with the full live id set as the target activity list, then flip
    /// `is_sticky = true`. The flag flip is deliberately ordered AFTER the
    /// delegate's symmetric-diff machinery so any intermediate
    /// `verify_invariants` chain sees a coherent state. The flag flip alone,
    /// with `activities` already at all_ids, is invariant-preserving by
    /// construction.
    ///
    /// Returns `(workspace_id, active_affected)` where `active_affected`
    /// bubbles up from the delegate — `true` when the symmetric diff added or
    /// removed the active activity from the workspace's set, signalling that
    /// the dispatch layer must redraw.
    pub(crate) fn set_workspace_sticky(
        &mut self,
        workspace: Option<WorkspaceReference>,
    ) -> Result<(WorkspaceId, bool), WorkspaceStickyError> {
        let ws_id = self
            .resolve_workspace_ref_or_active(workspace)
            .ok_or(WorkspaceStickyError::WorkspaceNotFound)?;

        // Two views of the full live id set: a `HashSet<ActivityId>` for the
        // no-op equality check, and a `Vec<ActivityReferenceArg>` for the
        // delegate call.
        let all_live_ids: HashSet<ActivityId> = self.activities.iter().map(|a| a.id()).collect();

        // No-op early-exit on the typical "already sticky + already expanded"
        // state. Skip the entire delegate / flag-flip path so we don't churn
        // animation state or rerun verify_invariants chains.
        {
            let ws = self
                .workspaces
                .get(&ws_id)
                .expect("resolved ws_id must be a live pool key");
            if ws.is_sticky() && ws.activities() == &all_live_ids {
                return Ok((ws_id, false));
            }
        }

        // Delegate to set_workspace_activities. The delegate handles animation
        // snap, view patching, and ensure_all_activity_views. All three delegate
        // error classes are statically impossible here:
        //   - ActivityNotFound: ids came from the live pool.
        //   - EmptyActivityList: Activities is non-empty by type.
        //   - WorkspaceNotFound: ws_id was resolved above.
        let all_ids_arg: Vec<ActivityReferenceArg> = all_live_ids
            .iter()
            .map(|id| ActivityReferenceArg::Id(id.get()))
            .collect();
        let (_, _, active_affected) = self
            .set_workspace_activities(Some(WorkspaceReference::Id(ws_id.get())), &all_ids_arg)
            .expect(
                "set_workspace_activities cannot fail: ids come from the live activity pool \
                 (ActivityNotFound impossible), Activities is non-empty \
                 (EmptyActivityList impossible), and ws_id was just resolved \
                 (WorkspaceNotFound impossible)",
            );

        // Flip is_sticky AFTER the delegate so any intermediate
        // verify_invariants run sees a coherent intermediate state. The flip
        // alone (with activities already at all_ids) is invariant-preserving.
        self.workspaces
            .get_mut(&ws_id)
            .expect("resolved ws_id must be a live pool key")
            .is_sticky = true;

        #[cfg(debug_assertions)]
        self.verify_invariants();

        Ok((ws_id, active_affected))
    }

    /// Clear `workspace.is_sticky` without touching its `activities` set
    /// (`is_sticky` is the auto-expansion trigger; toggling off
    /// does not narrow `activities`).
    ///
    /// Resolution order:
    /// 1. `workspace` → `WorkspaceNotFound`. Wire-surfaced via
    ///    `DoActionError::UnsetWorkspaceSticky(WorkspaceStickyError::WorkspaceNotFound)`.
    ///
    /// No-op cases (return `Ok(ws_id)` without mutating):
    /// - `ws.is_sticky() == false` — already not sticky.
    ///
    /// The workspace's `activities` set is intentionally left intact: the
    /// user can narrow it via `RemoveWorkspaceFromActivity` /
    /// `SetWorkspaceActivities` afterwards. Config-declared `sticky true`
    /// workspaces re-expand on the next config reload —
    /// session-only-effect is the documented contract.
    pub(crate) fn unset_workspace_sticky(
        &mut self,
        workspace: Option<WorkspaceReference>,
    ) -> Result<WorkspaceId, WorkspaceStickyError> {
        let ws_id = self
            .resolve_workspace_ref_or_active(workspace)
            .ok_or(WorkspaceStickyError::WorkspaceNotFound)?;

        let ws = self
            .workspaces
            .get_mut(&ws_id)
            .expect("resolved ws_id must be a live pool key");
        if !ws.is_sticky() {
            return Ok(ws_id);
        }
        ws.is_sticky = false;

        #[cfg(debug_assertions)]
        self.verify_invariants();

        Ok(ws_id)
    }

    /// Toggle `workspace.is_sticky`, dispatching to
    /// [`Self::set_workspace_sticky`] / [`Self::unset_workspace_sticky`].
    ///
    /// Resolution order:
    /// 1. `workspace` → `WorkspaceNotFound`. Wire-surfaced via
    ///    `DoActionError::ToggleWorkspaceSticky(WorkspaceStickyError::WorkspaceNotFound)`.
    ///
    /// Dispatches on `is_sticky` alone. The non-empty `activities`
    /// invariant makes the `sticky == true ∧ activities == ∅` state
    /// unreachable, so the flag is a faithful signal of the current sticky
    /// state.
    pub(crate) fn toggle_workspace_sticky(
        &mut self,
        workspace: Option<WorkspaceReference>,
    ) -> Result<ToggleWorkspaceStickyOutcome, WorkspaceStickyError> {
        let ws_id = self
            .resolve_workspace_ref_or_active(workspace)
            .ok_or(WorkspaceStickyError::WorkspaceNotFound)?;

        // Read current flag in a narrow scope so the shared borrow releases
        // before the delegate call.
        let currently_sticky = {
            self.workspaces
                .get(&ws_id)
                .expect("resolved ws_id must be a live pool key")
                .is_sticky()
        };

        if currently_sticky {
            self.unset_workspace_sticky(Some(WorkspaceReference::Id(ws_id.get())))
                .expect(
                    "unset cannot fail: ws_id was just resolved \
                     (WorkspaceNotFound impossible)",
                );
            Ok(ToggleWorkspaceStickyOutcome::StickyOff { ws_id })
        } else {
            let (_, active_affected) = self
                .set_workspace_sticky(Some(WorkspaceReference::Id(ws_id.get())))
                .expect(
                    "set cannot fail: ws_id was just resolved \
                     (WorkspaceNotFound impossible)",
                );
            Ok(ToggleWorkspaceStickyOutcome::StickyOn {
                ws_id,
                active_affected,
            })
        }
    }

    /// Resolve `workspace` to an id: an explicit reference resolves via
    /// [`Self::find_workspace_by_ref`], `None` falls back to the currently
    /// active workspace.
    ///
    /// Returns the id only, not a `&mut Workspace` — the `None` arm goes
    /// through [`Self::active_workspace_mut`], so returning a reference would
    /// keep the `&mut self` borrow alive past the call. Callers that need to
    /// split into `&mut self.workspaces` / `&mut self.activities` afterwards
    /// depend on the borrow having already dropped.
    fn resolve_workspace_ref_or_active(
        &mut self,
        workspace: Option<WorkspaceReference>,
    ) -> Option<WorkspaceId> {
        match workspace {
            Some(r) => self.find_workspace_by_ref(r).map(|ws| ws.id()),
            None => self.active_workspace_mut().map(|ws| ws.id()),
        }
    }

    /// Returns `Some(id)` iff `raw` matches a workspace id in the canonical
    /// pool — including workspaces that belong only to dormant activities
    /// (mirror of [`Layout::resolve_activity_ref`]).
    ///
    /// Callers that need active-view membership must compose with
    /// [`Layout::find_workspace_by_id`], which is active-view +
    /// disconnected-pool scoped.
    ///
    /// Do not collapse the chain at the call site (the `Id` arm of
    /// `Niri::find_output_and_workspace_index`) into a single filter. The two
    /// scopes (pool-membership vs. active-view + disconnected-pool) are not
    /// equivalent — a workspace exclusive to a dormant activity is visible to
    /// the resolver but invisible to the finder, and the chain's
    /// `?`-propagation depends on this asymmetry. Pinned by
    /// `resolve_workspace_id_finds_dormant_activity_workspace` in
    /// `src/layout/tests.rs`.
    pub(crate) fn resolve_workspace_id(&self, raw: u64) -> Option<WorkspaceId> {
        self.workspaces
            .values()
            .find(|ws| ws.id().get() == raw)
            .map(|ws| ws.id())
    }

    /// Resolve a workspace reference scoped to a specific activity.
    ///
    /// Scans the pool for workspaces that both match `reference` and belong to
    /// `activity` (via [`Workspace::activities`]). The scan is pool-wide, so
    /// workspaces belonging to dormant activities also resolve — the caller is
    /// responsible for any further active-view filtering.
    ///
    /// # Errors
    ///
    /// - [`FocusWorkspaceInActivityError::IndexUnsupported`] — positional indices are not
    ///   meaningful for activity-scoped lookup; callers must use `WorkspaceReference::Name` or
    ///   `WorkspaceReference::Id`.
    /// - [`FocusWorkspaceInActivityError::WorkspaceNotInActivity`] — no pool workspace matches the
    ///   reference within the given activity's membership set.
    pub(crate) fn resolve_workspace_in_activity(
        &self,
        activity: ActivityId,
        reference: &WorkspaceReference,
    ) -> Result<WorkspaceId, FocusWorkspaceInActivityError> {
        if matches!(reference, WorkspaceReference::Index(_)) {
            return Err(FocusWorkspaceInActivityError::IndexUnsupported);
        }
        self.workspaces
            .values()
            .find(|ws| {
                ws.activities().contains(&activity)
                    && match reference {
                        WorkspaceReference::Id(raw) => ws.id().get() == *raw,
                        WorkspaceReference::Name(name) => ws
                            .name
                            .as_ref()
                            .is_some_and(|n| n.eq_ignore_ascii_case(name)),
                        WorkspaceReference::Index(_) => {
                            unreachable!("index arm guarded above")
                        }
                    }
            })
            .map(|ws| ws.id())
            .ok_or(FocusWorkspaceInActivityError::WorkspaceNotInActivity)
    }

    /// Resolve an [`ActivityReferenceArg`] to an [`ActivityId`] if the pool
    /// contains a matching activity, `None` otherwise.
    ///
    /// Names are unique across the activity pool, so the
    /// first-match walk is deterministic. `ActivityId` is opaque: the `Id`
    /// variant scans `iter()` comparing raw `u64` values rather than hashing
    /// into the underlying `IndexMap`.
    pub(crate) fn resolve_activity_ref(&self, r: &ActivityReferenceArg) -> Option<ActivityId> {
        match r {
            ActivityReferenceArg::Id(raw) => self
                .activities
                .iter()
                .find(|a| a.id().get() == *raw)
                .map(|a| a.id()),
            ActivityReferenceArg::Name(s) => self
                .activities
                .iter()
                .find(|a| a.name() == s)
                .map(|a| a.id()),
        }
    }

    pub fn consume_into_column(&mut self) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.consume_into_column();
    }

    pub fn expel_from_column(&mut self) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.expel_from_column();
    }

    pub fn swap_window_in_direction(&mut self, direction: ScrollDirection) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.swap_window_in_direction(direction);
    }

    pub fn toggle_column_tabbed_display(&mut self) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.toggle_column_tabbed_display();
    }

    pub fn set_column_display(&mut self, display: ColumnDisplay) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.set_column_display(display);
    }

    pub fn center_column(&mut self) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.center_column();
    }

    pub fn center_window(&mut self, id: Option<&W::Id>) {
        if let Some(InteractiveMoveState::Moving(move_)) = &mut self.interactive_move {
            if id.is_none() || id == Some(move_.tile.window().id()) {
                return;
            }
        }

        let workspace = if let Some(id) = id {
            Some(self.workspaces_mut().find(|ws| ws.has_window(id)).unwrap())
        } else {
            self.active_workspace_mut()
        };

        let Some(workspace) = workspace else {
            return;
        };
        workspace.center_window(id);
    }

    pub fn center_visible_columns(&mut self) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.center_visible_columns();
    }

    pub fn focus(&self) -> Option<&W> {
        self.focus_with_output().map(|(win, _out)| win)
    }

    pub fn focus_with_output(&self) -> Option<(&W, &Output)> {
        if let Some(InteractiveMoveState::Moving(move_)) = &self.interactive_move {
            return Some((move_.tile.window(), &move_.output));
        }

        if self.monitors.is_empty() {
            return None;
        }

        let mon = &self.monitors[self.active_monitor_idx];
        let view = self.active_view(&mon.output_id());
        mon.active_window(&self.workspaces, view)
            .map(|win| (win, &mon.output))
    }

    pub fn interactive_moved_window_under(
        &self,
        output: &Output,
        pos_within_output: Point<f64, Logical>,
    ) -> Option<(&W, HitType)> {
        if let Some(InteractiveMoveState::Moving(move_)) = &self.interactive_move {
            if move_.output == *output {
                if self.overview_progress.is_some() {
                    let zoom = self.overview_zoom();
                    let tile_pos = move_.tile_render_location(zoom);
                    let pos_within_tile = (pos_within_output - tile_pos).downscale(zoom);
                    // During the overview animation, we cannot do input hits because we cannot
                    // really represent scaled windows properly.
                    let (win, hit) =
                        HitType::hit_tile(&move_.tile, Point::from((0., 0.)), pos_within_tile)?;
                    Some((win, hit.to_activate()))
                } else {
                    let tile_pos = move_.tile_render_location(1.);
                    HitType::hit_tile(&move_.tile, tile_pos, pos_within_output)
                }
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Returns the window under the cursor and the hit type.
    pub fn window_under(
        &self,
        output: &Output,
        pos_within_output: Point<f64, Logical>,
    ) -> Option<(&W, HitType)> {
        let mon = self.monitor_for_output(output)?;
        let ctx = self.ctx_for(mon);
        mon.window_under(ctx, pos_within_output)
    }

    pub fn resize_edges_under(
        &self,
        output: &Output,
        pos_within_output: Point<f64, Logical>,
    ) -> Option<ResizeEdge> {
        let mon = self.monitor_for_output(output)?;
        let ctx = self.ctx_for(mon);
        mon.resize_edges_under(ctx, pos_within_output)
    }

    pub fn workspace_under(
        &self,
        extended_bounds: bool,
        output: &Output,
        pos_within_output: Point<f64, Logical>,
    ) -> Option<&Workspace<W>> {
        if self
            .interactive_moved_window_under(output, pos_within_output)
            .is_some()
        {
            return None;
        }

        let mon = self.monitor_for_output(output)?;
        let ctx = self.ctx_for(mon);
        if extended_bounds {
            mon.workspace_under(ctx, pos_within_output)
                .map(|(ws, _)| ws)
        } else {
            mon.workspace_under_narrow(ctx, pos_within_output)
        }
    }

    pub fn overview_zoom(&self) -> f64 {
        let progress = self.overview_progress.as_ref().map(|p| p.value());
        compute_overview_zoom(&self.options, progress)
    }

    #[cfg(debug_assertions)]
    fn verify_invariants(&self) {
        use approx::assert_abs_diff_eq;

        let zoom = self.overview_zoom();

        let mut move_win_id = None;
        if let Some(state) = &self.interactive_move {
            match state {
                InteractiveMoveState::Starting {
                    window_id,
                    pointer_delta: _,
                    pointer_ratio_within_window: _,
                } => {
                    assert!(
                        self.has_window(window_id),
                        "interactive move must be on an existing window"
                    );
                    move_win_id = Some(window_id.clone());
                }
                InteractiveMoveState::Moving(move_) => {
                    assert_eq!(self.clock, move_.tile.clock);
                    assert!(move_.tile.window().pending_sizing_mode().is_normal());

                    move_.tile.verify_invariants();

                    let scale = move_.output.current_scale().fractional_scale();
                    let options = Options::clone(&self.options)
                        .with_merged_layout(move_.output_config.as_ref())
                        .with_merged_layout(move_.workspace_config.as_ref().map(|(_, c)| c))
                        .adjusted_for_scale(scale);
                    assert_eq!(
                        &*move_.tile.options, &options,
                        "interactive moved tile options must be \
                         base options adjusted for output scale"
                    );

                    let tile_pos = move_.tile_render_location(zoom);
                    let rounded_pos = tile_pos.to_physical_precise_round(scale).to_logical(scale);

                    // Tile position must be rounded to physical pixels.
                    assert_abs_diff_eq!(tile_pos.x, rounded_pos.x, epsilon = 1e-5);
                    assert_abs_diff_eq!(tile_pos.y, rounded_pos.y, epsilon = 1e-5);

                    if let Some(alpha) = &move_.tile.alpha_animation {
                        if move_.is_floating {
                            assert_eq!(
                                alpha.anim.to(),
                                1.,
                                "interactively moved floating tile can animate alpha only to 1"
                            );

                            assert!(
                                !alpha.hold_after_done,
                                "interactively moved floating tile \
                                 cannot have held alpha animation"
                            );
                        } else {
                            assert_ne!(
                                alpha.anim.to(),
                                1.,
                                "interactively moved scrolling tile must animate alpha to not 1"
                            );

                            assert!(
                                alpha.hold_after_done,
                                "interactively moved scrolling tile \
                                 must have held alpha animation"
                            );
                        }
                    }
                }
            }
        }

        // Activities-internal invariants: active/previous cursor validity + distinctness.
        self.activities.verify_invariants();

        // Bookmark cross-field invariants. Ids are unique, `next_id` stays ahead
        // of every listed id, no two bookmarks hold an equal key, and the walk
        // cursor is in bounds when `Some` — these apply to every entry. Window
        // uniqueness and windows_all-presence (the prune-on-close guarantee)
        // apply to `Window` anchors AND *attached* `Rule` anchors: an attached
        // rule implies a live window, asserted here. Dangling rule anchors carry
        // no window and are exempt from the window checks. Saved-activity
        // liveness and list order are deliberately NOT asserted: dead activities
        // are legal (the restore fallback covers them), and no order property is
        // a standing invariant. Key uniqueness here is raw `Key` equality;
        // normalized (`ModKey`-aware) collisions are excluded both at assign time
        // (input dispatch, against config binds and sibling bookmarks) and on
        // every config reload (`revalidate_bookmark_keys`, same two checks) —
        // this generic-`Layout` code cannot itself re-check `ModKey`
        // normalization. `last_visited`'s only standing invariant is
        // monotonicity against `next_id`: it is resolved lazily at walk time,
        // so a remembered id that has since been removed or whose entry has
        // dangled is legal (an implication, not a biconditional) and
        // deliberately not asserted here.
        {
            let list = self.bookmarks.list();
            if let Some(cursor) = self.bookmarks.walk_cursor() {
                assert!(
                    cursor < list.len(),
                    "bookmark walk cursor {cursor} out of bounds (len {})",
                    list.len(),
                );
            }
            if let Some(last_visited) = self.bookmarks.last_visited() {
                assert!(
                    last_visited.get() < self.bookmarks.next_id(),
                    "bookmark last_visited id {} not below next_id {}",
                    last_visited.get(),
                    self.bookmarks.next_id(),
                );
            }
            let mut seen_ids = HashSet::new();
            // `W::Id` is only `PartialEq`, so window uniqueness uses a linear
            // scan rather than a hash set — fine for a hand-curated list.
            let mut seen_windows: Vec<&W::Id> = Vec::with_capacity(list.len());
            let mut seen_keys: Vec<jiji_config::Key> = Vec::with_capacity(list.len());
            for bookmark in list {
                assert!(
                    seen_ids.insert(bookmark.id().get()),
                    "duplicate bookmark id {} in list",
                    bookmark.id().get(),
                );
                assert!(
                    bookmark.id().get() < self.bookmarks.next_id(),
                    "bookmark id {} not below next_id {}",
                    bookmark.id().get(),
                    self.bookmarks.next_id(),
                );
                // A live window (a `Window` anchor, or an attached `Rule`) is
                // subject to uniqueness and prune-on-close; a dangling rule
                // anchor is exempt.
                if let Some(window) = bookmark.anchor().window() {
                    assert!(
                        !seen_windows.contains(&window),
                        "more than one bookmark anchors window {window:?}",
                    );
                    seen_windows.push(window);
                    assert!(
                        self.windows_all().any(|(_, w)| w.id() == window),
                        "bookmark references window {window:?} absent from windows_all \
                         (prune-on-close guarantee broken)",
                    );
                }
                if let Some(key) = bookmark.key() {
                    let key = key.key();
                    assert!(
                        !seen_keys.contains(&key),
                        "more than one bookmark holds key {key:?}",
                    );
                    seen_keys.push(key);
                }
            }
            if let Some(target) = self.bookmarks.return_target() {
                assert!(
                    self.windows_all().any(|(_, w)| w.id() == target),
                    "bookmark return target {target:?} absent from windows_all \
                     (prune-on-close guarantee broken)",
                );
            }
        }

        // Every Workspace.activities is a non-empty subset of the Activities keyset.
        // Walks the full pool unconditionally (before the zero-monitor / any-monitor split below);
        // the pool walk is cheap (HashSet::contains per workspace activity entry; typical set
        // size is small).
        for (ws_id, workspace) in &self.workspaces {
            assert!(
                !workspace.activities.is_empty(),
                "workspace {ws_id:?} activities set must be non-empty",
            );
            for activity_id in workspace.activities.iter() {
                assert!(
                    self.activities.contains(*activity_id),
                    "workspace {ws_id:?} references activity {activity_id:?}, not a live key in \
                     Layout.activities (live: {:?})",
                    self.activities.iter().map(|a| a.id()).collect::<Vec<_>>(),
                );
            }
        }

        // Cross-field: active activity's `views` map has exactly one entry per connected monitor.
        // If add_output/remove_output lifecycle drifts (e.g. view inserted after monitor push or
        // removed after monitor pop), subsequent active_view lookups panic far from root cause;
        // catching domain parity here surfaces it at verify time.
        let active_views = self.activities.active().views();
        assert_eq!(
            active_views.len(),
            self.monitors.len(),
            "active activity's views map size ({}) must equal monitor count ({})",
            active_views.len(),
            self.monitors.len(),
        );
        for mon in &self.monitors {
            assert!(
                active_views.contains_key(&OutputId::new(&mon.output)),
                "active activity's views map is missing an entry for connected monitor {:?}",
                OutputId::new(&mon.output),
            );
        }

        // Per-activity bookend invariant: every activity must hold a view entry for every
        // connected monitor. Asserts the materializer keeps pace with every call site that
        // creates an activity or mutates view membership.
        let active_activity_id_for_check = self.activities.active_id();
        for activity in self.activities.iter() {
            if activity.id() == active_activity_id_for_check {
                continue;
            }
            let act_id = activity.id();
            let act_name = activity.name();
            for mon in &self.monitors {
                let out_id = OutputId::new(&mon.output);
                assert!(
                    activity.views().contains_key(&out_id),
                    "activity {act_id:?} ({act_name:?}) is missing a view for connected \
                     monitor {out_id:?}",
                );
            }
        }

        let mut seen_workspace_id = HashSet::new();
        let mut seen_workspace_name = Vec::<String>::new();

        let pool = &self.workspaces;

        // Pool keys equal the union of every live activity's views over every
        // `WorkspaceView.ids()`, plus `disconnected_workspace_ids`. The union is across
        // activities (not just the active one) so dormant views on inactive activities also
        // anchor their ids into the pool — without that, switching back to an inactive
        // activity could find its view ids have been GCed. A single `WorkspaceId` can
        // legitimately appear in multiple views (e.g. sticky workspaces, future
        // `activities = {A, B}` membership), so cross-view duplicates fold into the HashSet
        // without assertion; per-view uniqueness is still enforced by a local set below.
        //
        // The same walk also pins membership↔view coherence in both directions: every id held
        // by a view must carry that activity in its own `activities` set (view→membership), and
        // — via the `per_activity_ids` accumulator checked further below — every id that carries
        // an activity in `activities` must be held by some view of that activity
        // (membership→view). `reconcile_views_with_membership` is the runtime repair dual that
        // restores this coherence after a membership-mutating pass; these asserts are what pin
        // that it actually does.
        let mut expected_keys: HashSet<WorkspaceId> = HashSet::new();
        let mut per_activity_ids: HashMap<ActivityId, HashSet<WorkspaceId>> = HashMap::new();
        for activity in self.activities.iter() {
            let act_id = activity.id();
            let ids_for_activity = per_activity_ids.entry(act_id).or_default();
            for view in activity.views().values() {
                let mut per_view: HashSet<WorkspaceId> = HashSet::new();
                for id in view.ids() {
                    assert!(
                        per_view.insert(*id),
                        "workspace id must appear at most once within a single WorkspaceView",
                    );
                    // Sticky workspaces and future `activities = {A, B}` configurations
                    // legitimately produce the same `WorkspaceId` in multiple views; per-view
                    // uniqueness is enforced separately above.
                    expected_keys.insert(*id);
                    ids_for_activity.insert(*id);
                    let ws = pool
                        .get(id)
                        .expect("every view id must be in the pool — no zombies");
                    assert!(
                        ws.activities().contains(&act_id),
                        "workspace {id:?} is held by activity {act_id:?}'s view, but the \
                         workspace's activities set lacks that activity",
                    );
                }
            }
        }
        for id in &self.disconnected_workspace_ids {
            assert!(
                expected_keys.insert(*id),
                "disconnected_workspace_ids entry must not already appear in any activity's view",
            );
            assert!(
                pool.contains_key(id),
                "disconnected_workspace_ids entry must be a key in the workspace pool",
            );
        }
        let pool_keys: HashSet<WorkspaceId> = pool.keys().copied().collect();
        assert_eq!(
            expected_keys, pool_keys,
            "pool keys must equal the union of every activity's views over all outputs plus disconnected_workspace_ids",
        );

        // Membership→view: every pool workspace's `activities` set must be backed by a view of
        // each of those activities holding it. Unscoped — checked for the whole pool regardless
        // of monitor connectivity, except for ids parked in `disconnected_workspace_ids`, which
        // by definition sit outside every activity's views while disconnected (mirrors the
        // `disconnected` guard in `reconcile_views_with_membership`). In the fully-disconnected
        // window every pool id is in `disconnected_workspace_ids`, so this walk is vacuous there
        // by construction; with any monitor connected the list is asserted empty below, so
        // nothing is exempted.
        let disconnected: HashSet<WorkspaceId> =
            self.disconnected_workspace_ids.iter().copied().collect();
        for (ws_id, ws) in pool.iter() {
            if disconnected.contains(ws_id) {
                continue;
            }
            for act_id in ws.activities() {
                let ids_for_activity = per_activity_ids
                    .get(act_id)
                    .expect("every live activity got an entry above, even with zero views");
                assert!(
                    ids_for_activity.contains(ws_id),
                    "workspace {ws_id:?} carries activity {act_id:?} in its activities set, but \
                     no view of that activity holds it",
                );
            }
        }

        if self.monitors.is_empty() {
            assert!(
                self.activities.active().views().is_empty(),
                "when no monitors are connected the active activity's views map must be empty",
            );

            // Sentinel invariants when no monitors are connected: idx fields hold placeholder 0.
            assert_eq!(
                self.primary_idx, 0,
                "primary_idx must be 0 when no monitors are connected",
            );
            assert_eq!(
                self.active_monitor_idx, 0,
                "active_monitor_idx must be 0 when no monitors are connected",
            );

            for id in &self.disconnected_workspace_ids {
                let workspace = pool
                    .get(id)
                    .expect("id must be a key in the workspace pool");

                assert!(
                    workspace.has_windows_or_name(),
                    "with no outputs there cannot be empty unnamed workspaces"
                );

                assert_eq!(self.clock, workspace.clock);

                assert_eq!(
                    workspace.base_options, self.options,
                    "workspace base options must be synchronized with layout"
                );

                assert!(
                    seen_workspace_id.insert(workspace.id()),
                    "workspace id must be unique"
                );

                if let Some(name) = &workspace.name {
                    assert!(
                        !seen_workspace_name
                            .iter()
                            .any(|n| n.eq_ignore_ascii_case(name)),
                        "workspace name must be unique"
                    );
                    seen_workspace_name.push(name.clone());
                }

                workspace.verify_invariants(move_win_id.as_ref());
            }

            return;
        }

        assert!(
            self.disconnected_workspace_ids.is_empty(),
            "disconnected_workspace_ids must be empty when any monitor is connected",
        );
        let monitors = &self.monitors;
        let primary_idx = self.primary_idx;
        let active_monitor_idx = self.active_monitor_idx;
        assert!(primary_idx < monitors.len());
        assert!(active_monitor_idx < monitors.len());

        let mut saw_view_offset_gesture = false;

        for (idx, monitor) in monitors.iter().enumerate() {
            assert_eq!(self.clock, monitor.clock);
            assert_eq!(
                monitor.base_options, self.options,
                "monitor base options must be synchronized with layout"
            );

            assert_eq!(self.overview_open, monitor.overview_open);
            assert_eq!(
                self.overview_progress.as_ref().map(|p| p.value()),
                monitor.overview_progress_value()
            );

            let mon_output_id = monitor.output_id();
            let active_view = self.active_view(&mon_output_id);
            let mut views_for_monitor: Vec<&WorkspaceView> = vec![active_view];
            let active_activity_id = self.activities.active_id();
            for activity in self.activities.iter() {
                if activity.id() == active_activity_id {
                    continue;
                }
                if let Some(view) = activity.views().get(&mon_output_id) {
                    views_for_monitor.push(view);
                }
            }
            monitor.verify_invariants(pool, &views_for_monitor);

            // Activity-switch invariants: when a transition is in flight —
            //   1. `from` must be a live activity id.
            //   2. `from` must differ from the active activity id (the active id is the incoming
            //      strip; `from` is the outgoing one).
            // Cross-monitor consistency (all Some entries share one `from` and `dir`)
            // is verified in the pass below, outside this per-monitor loop.
            if let Some(switch) = &monitor.activity_switch {
                assert!(
                    self.activities.contains(switch.from),
                    "Monitor activity_switch.from {:?} must be a live activity id",
                    switch.from,
                );
                assert_ne!(
                    switch.from,
                    self.activities.active_id(),
                    "Monitor activity_switch.from must not equal the active activity id \
                     (from is the outgoing strip, active is the incoming one)",
                );
            }

            if idx == primary_idx {
                for id in self.active_view(&monitor.output_id()).ids() {
                    let ws = pool
                        .get(id)
                        .expect("workspace id must be a key in the pool");
                    if ws
                        .output_id
                        .as_ref()
                        .is_some_and(|id| id.matches(&monitor.output))
                    {
                        // This is the primary monitor's own workspace.
                        continue;
                    }

                    let own_monitor_exists = monitors.iter().any(|m| {
                        ws.output_id
                            .as_ref()
                            .is_some_and(|id| id.matches(&m.output))
                    });
                    assert!(
                        !own_monitor_exists,
                        "primary monitor cannot have workspaces for which their own monitor exists"
                    );
                }
            } else {
                assert!(
                    self.active_view(&monitor.output_id())
                        .ids()
                        .iter()
                        .any(|id| {
                            pool.get(id)
                                .and_then(|ws| ws.output_id.as_ref())
                                .is_some_and(|oid| oid.matches(&monitor.output))
                        }),
                    "secondary monitor must not have any non-own workspaces"
                );
            }

            // FIXME: verify that primary doesn't have any workspaces for which their own monitor
            // exists.

            for id in self.active_view(&monitor.output_id()).ids() {
                let workspace = pool
                    .get(id)
                    .expect("workspace id must be a key in the pool");
                assert!(
                    seen_workspace_id.insert(workspace.id()),
                    "workspace id must be unique"
                );

                if let Some(name) = &workspace.name {
                    assert!(
                        !seen_workspace_name
                            .iter()
                            .any(|n| n.eq_ignore_ascii_case(name)),
                        "workspace name must be unique"
                    );
                    seen_workspace_name.push(name.clone());
                }

                workspace.verify_invariants(move_win_id.as_ref());

                let has_view_offset_gesture = workspace.scrolling().view_offset().is_gesture();
                if self.dnd.is_some() || self.interactive_move.is_some() {
                    // We'd like to check that all workspaces have the gesture here, furthermore we
                    // want to check that they have the gesture only if the interactive move
                    // targets the scrolling layout. However, we cannot do that because we start
                    // and stop the gesture lazily. Otherwise the gesture code would pollute a lot
                    // of places like adding new workspaces, implicitly moving windows between
                    // floating and tiling on fullscreen, etc.
                    //
                    // assert!(
                    //     has_view_offset_gesture,
                    //     "during an interactive move in the scrolling layout, \
                    //      all workspaces should be in a view offset gesture"
                    // );
                } else if saw_view_offset_gesture {
                    assert!(
                        !has_view_offset_gesture,
                        "only one workspace can have an ongoing view offset gesture"
                    );
                }
                saw_view_offset_gesture = has_view_offset_gesture;
            }
        }

        // Cross-monitor activity-switch consistency: all Some entries must share one `from` id
        // and one `dir` value.  Mixed Some/None is legal (a monitor that connected mid-flight was
        // never armed), but two Some entries with different `from` or different `dir` values
        // indicate a logic bug.
        {
            let mut common_from: Option<ActivityId> = None;
            let mut common_dir: Option<SlideDirection> = None;
            for monitor in monitors.iter() {
                if let Some(switch) = &monitor.activity_switch {
                    match common_from {
                        None => common_from = Some(switch.from),
                        Some(f) => assert_eq!(
                            f, switch.from,
                            "all monitors with an in-flight activity-switch must share \
                             the same outgoing activity id (expected {:?}, found {:?})",
                            f, switch.from,
                        ),
                    }
                    match common_dir {
                        None => common_dir = Some(switch.dir),
                        Some(d) => assert_eq!(
                            d, switch.dir,
                            "all monitors with an in-flight activity-switch must share \
                             the same direction (expected {:?}, found {:?})",
                            d, switch.dir,
                        ),
                    }
                }
            }
        }

        // Overview/slide exclusion: the overview spatial map and an activity-strip slide must
        // never run simultaneously. Arming is suppressed while the overview is open, and both
        // overview entry paths snap every in-flight switch, so no monitor may have an armed
        // activity_switch while the overview is open.
        if self.overview_open {
            for monitor in monitors.iter() {
                assert!(
                    monitor.activity_switch.is_none(),
                    "no monitor may have an in-flight activity-switch while the overview is open",
                );
            }
        }

        // Partial-disconnect migration invariant: when any monitor is connected, every
        // activity's `views` map is keyed exclusively by connected monitors' OutputIds.
        // The partial-disconnect walk in `remove_output` drains dormant views for the
        // disconnecting output and migrates their workspaces into the dormant view for
        // primary, so a stale (disconnected-output) view key must never reach
        // `verify_invariants` while at least one monitor remains.
        //
        // The carve-out for `monitors.is_empty()` covers the fully-disconnected window:
        // `remove_output`'s full-disconnect branch clears every activity's views map
        // before returning, so every map is empty in that state. This is the invariant
        // that lets the `Request::ActivityViews` IPC contract claim `output_name: None`
        // is unreachable in steady state.
        let connected_output_ids: HashSet<OutputId> = self
            .monitors
            .iter()
            .map(|m| OutputId::new(&m.output))
            .collect();
        if !self.monitors.is_empty() {
            for activity in self.activities.iter() {
                let act_id = activity.id();
                let act_name = activity.name();
                for out_id in activity.views().keys() {
                    assert!(
                        connected_output_ids.contains(out_id),
                        "activity {act_id:?} ({act_name:?}) view keyed by {out_id:?} \
                         has no matching connected monitor — partial-disconnect migration \
                         must drain views for the disconnecting output",
                    );
                }
            }
        } else {
            for activity in self.activities.iter() {
                assert!(
                    activity.views().is_empty(),
                    "fully-disconnected state requires every activity's views map to be \
                     empty; activity {:?} ({:?}) retains keys {:?}",
                    activity.id(),
                    activity.name(),
                    activity.views().keys().collect::<Vec<_>>(),
                );
            }
        }
    }

    pub fn advance_animations(&mut self) {
        let _span = tracy_client::span!("Layout::advance_animations");

        let mut dnd_scroll = None;
        let mut is_dnd = false;
        if let Some(dnd) = &self.dnd {
            dnd_scroll = Some((dnd.output.clone(), dnd.pointer_pos_within_output, true));
            is_dnd = true;
        }

        if let Some(InteractiveMoveState::Moving(move_)) = &mut self.interactive_move {
            move_.tile.advance_animations();

            if dnd_scroll.is_none() {
                dnd_scroll = Some((
                    move_.output.clone(),
                    move_.pointer_pos_within_output,
                    !move_.is_floating,
                ));
            }
        }

        let is_overview_open = self.overview_open;

        // Scroll the view if needed.
        if let Some((output, pos_within_output, is_scrolling)) = dnd_scroll {
            let output_id = OutputId::new(&output);
            if self.monitors.iter().any(|m| m.output == output) {
                let view = self
                    .activities
                    .active_mut()
                    .views_mut()
                    .get_mut(&output_id)
                    .expect("connected output must have a view in the active activity");
                let pool = &mut self.workspaces;
                let mon = self
                    .monitors
                    .iter_mut()
                    .find(|m| m.output == output)
                    .expect("monitor for connected output must exist");
                let mut scrolled = false;

                let zoom = mon.overview_zoom();
                scrolled |= mon.dnd_scroll_gesture_scroll(view, pos_within_output, 1. / zoom);

                if is_scrolling {
                    let ctx = LayoutCtx::new(&*pool, view);
                    if let Some((ws_id, geo)) = mon
                        .workspace_under(ctx, pos_within_output)
                        .map(|(ws, geo)| (ws.id(), geo))
                    {
                        let ws = pool
                            .get_mut(&ws_id)
                            .expect("workspace id must be a key in the pool");
                        // As far as the DnD scroll gesture is concerned, the workspace spans
                        // across the whole monitor horizontally.
                        let ws_pos = Point::from((0., geo.loc.y));
                        scrolled |=
                            ws.dnd_scroll_gesture_scroll(pos_within_output - ws_pos, 1. / zoom);
                    }
                }

                if scrolled {
                    // Don't trigger DnD hold while scrolling.
                    if let Some(dnd) = &mut self.dnd {
                        dnd.hold = None;
                    }
                } else if is_dnd {
                    let ctx = LayoutCtx::new(&*pool, view);
                    let target = mon
                        .window_under(ctx, pos_within_output)
                        .map(|(win, _)| DndHoldTarget::Window(win.id().clone()))
                        .or_else(|| {
                            mon.workspace_under_narrow(ctx, pos_within_output)
                                .map(|ws| DndHoldTarget::Workspace(ws.id()))
                        });

                    let dnd = self.dnd.as_mut().unwrap();
                    if let Some(target) = target {
                        let now = self.clock.now_unadjusted();
                        let start_time = if let Some(hold) = &mut dnd.hold {
                            if hold.target != target {
                                hold.start_time = now;
                            }
                            hold.target = target;
                            hold.start_time
                        } else {
                            let hold = dnd.hold.insert(DndHold {
                                start_time: now,
                                target,
                            });
                            hold.start_time
                        };

                        // Delay copied from gnome-shell.
                        let delay = Duration::from_millis(750);
                        if delay <= now.saturating_sub(start_time) {
                            let hold = dnd.hold.take().unwrap();

                            // Synchronize workspace switch to overview close to get a
                            // monotonic animation.
                            let config = is_overview_open
                                .then_some(self.options.animations.overview_open_close.0);

                            let ws_idx = match hold.target {
                                DndHoldTarget::Window(id) => {
                                    let mut found = None;
                                    for (i, view_id) in view.ids().iter().enumerate() {
                                        let ws = pool
                                            .get_mut(view_id)
                                            .expect("view id must be a key in the pool");
                                        if ws.activate_window(&id) {
                                            found = Some(i);
                                            break;
                                        }
                                    }
                                    found.unwrap()
                                }
                                DndHoldTarget::Workspace(id) => view.position_of(id).unwrap(),
                            };

                            mon.dnd_scroll_gesture_end(view);
                            mon.activate_workspace_with_anim_config(view, ws_idx, config);

                            self.focus_output(&output);

                            if is_overview_open {
                                self.close_overview();
                            }
                        }
                    } else {
                        // No target, reset the hold timer.
                        dnd.hold = None;
                    }
                }
            }
        }

        if let Some(OverviewProgress::Animation(anim)) = &mut self.overview_progress {
            if anim.is_done() {
                if self.overview_open {
                    self.overview_progress = Some(OverviewProgress::Open);
                } else {
                    self.overview_progress = None;
                }
            }
        }

        // Accumulate pruned ids across every monitor under one triple-borrow
        // scope; flush them through `destroy_workspaces_cross_activity` once
        // the scope closes. Calling the helper inside the loop would fight the
        // borrow checker because it re-borrows `&mut self.activities` and
        // `&mut self.workspaces`.
        let ids_to_destroy: Vec<WorkspaceId> = {
            let views_map = self.activities.active_mut().views_mut();
            let pool = &mut self.workspaces;
            let overview = self.overview_progress.as_ref();
            let monitors = &mut self.monitors[..];
            let mut accum = Vec::new();
            for mon_idx in 0..monitors.len() {
                let view = views_map
                    .get_mut(&OutputId::new(&monitors[mon_idx].output))
                    .expect("connected output must have a view in the active activity");
                monitors[mon_idx].set_overview_progress(view, overview);
                let workspace_switch_finished = monitors[mon_idx].advance_animations(pool, view);
                if workspace_switch_finished {
                    accum.extend(Self::clean_up_workspaces_on(monitors, pool, view, mon_idx));
                }
            }
            accum
        };
        Self::destroy_workspaces_cross_activity(
            &mut self.activities,
            &mut self.workspaces,
            ids_to_destroy,
        );

        // Advance workspaces in outgoing-activity views. This pass is separate from the
        // main loop above because the main loop holds &mut self.activities through its
        // views_map borrow; resolving the outgoing activity inside it would not compile.
        // Disjoint field borrows: &mut self.workspaces + &self.monitors + &self.activities.
        {
            let pool = &mut self.workspaces;
            for mon in &self.monitors {
                let Some(switch) = &mon.activity_switch else {
                    continue;
                };
                let activity = self
                    .activities
                    .get(switch.from)
                    .expect("in-flight activity switch outgoing id must be a live activity");
                // View presence is tolerated rather than expected: a cross-activity destroy can
                // legitimately drop a dormant single-entry view mid-flight without immediate
                // re-materialization (the view.len()==1 → drop entry retain and
                // destroy_workspaces_cross_activity).
                //
                // Sticky/shared workspaces present in both the active view (advanced in the main
                // loop above) and the outgoing view receive advance_animations twice per tick.
                // This is currently benign because advance_animations is clock-sampled (reads
                // Clock::now() and is idempotent within a tick). A future delta-based animation
                // model would need to deduplicate the workspace set across both passes.
                let Some(view) = activity.views().get(&mon.output_id()) else {
                    continue;
                };
                for id in view.ids() {
                    pool.get_mut(id)
                        .expect("view id must be a key in the pool")
                        .advance_animations();
                }
            }
        }

        for id in &self.disconnected_workspace_ids {
            self.workspaces
                .get_mut(id)
                .expect("workspace id must be a key in the pool")
                .advance_animations();
        }
    }

    pub fn are_animations_ongoing(&self, output: Option<&Output>) -> bool {
        // Keep advancing animations if we might need to scroll the view.
        if let Some(dnd) = &self.dnd {
            if output.is_none_or(|output| *output == dnd.output) {
                return true;
            }
        }

        if let Some(InteractiveMoveState::Moving(move_)) = &self.interactive_move {
            if output.is_none_or(|output| *output == move_.output) {
                if move_.tile.are_animations_ongoing() {
                    return true;
                }

                // Keep advancing animations if we might need to scroll the view.
                if !move_.is_floating || self.overview_open {
                    return true;
                }
            }
        }

        if self
            .overview_progress
            .as_ref()
            .is_some_and(|p| p.is_animation())
        {
            return true;
        }

        for mon in self.monitors() {
            if output.is_some_and(|output| mon.output != *output) {
                continue;
            }

            let mon_output_id = mon.output_id();
            let view = self.active_view(&mon_output_id);
            // Resolve the outgoing view when an activity-switch transition is in flight.
            // View presence is tolerated (else None) — same expect/tolerate split as the
            // advance pass; a mid-flight dormant-view evaporation is non-fatal here.
            let outgoing_view = mon.activity_switch.as_ref().and_then(|s| {
                self.activities
                    .get(s.from)
                    .expect("in-flight activity switch outgoing id must be a live activity")
                    .views()
                    .get(&mon_output_id)
            });
            if mon.are_animations_ongoing(&self.workspaces, view, outgoing_view) {
                return true;
            }
        }

        false
    }

    pub fn update_render_elements(&mut self, output: Option<&Output>) {
        let _span = tracy_client::span!("Layout::update_render_elements");

        self.update_render_elements_time = self.clock.now();

        let zoom = self.overview_zoom();
        if let Some(InteractiveMoveState::Moving(move_)) = &mut self.interactive_move {
            if output.is_none_or(|output| move_.output == *output) {
                let pos_within_output = move_.tile_render_location(zoom);

                // We're not on any specific workspace so we can't compute a "workspace view" rect.
                // Let's instead compute a rect relative to the output.
                //
                // FIXME: we could make the colors match up better in the overview by figuring out
                // where a centered workspace would currently be, and computing the view rect
                // against that. Since most of the time the dragged window will be on a centered
                // workspace.
                let view_rect =
                    Rectangle::new(pos_within_output.upscale(-1.), output_size(&move_.output))
                        .downscale(zoom);

                move_.tile.update_render_elements(true, view_rect);
            }
        }

        self.update_insert_hint(output);

        if self.monitors.is_empty() {
            if output.is_some() {
                error!("update_render_elements called with no monitors but Some output");
            }
            return;
        }

        let views_map = self.activities.active_mut().views_mut();
        let pool = &mut self.workspaces;
        for (idx, mon) in self.monitors.iter_mut().enumerate() {
            if output.is_none_or(|output| mon.output == *output) {
                let is_active = self.is_active
                    && idx == self.active_monitor_idx
                    && !matches!(self.interactive_move, Some(InteractiveMoveState::Moving(_)));
                let view = views_map
                    .get_mut(&OutputId::new(&mon.output))
                    .expect("connected output must have a view in the active activity");
                mon.set_overview_progress(view, self.overview_progress.as_ref());
                mon.update_render_elements(pool, view, is_active);
            }
        }

        // Update render elements in outgoing-activity views. This pass is separate from the
        // main loop above because the main loop holds &mut self.activities through its
        // views_map borrow; resolving the outgoing activity inside it would not compile.
        // Disjoint field borrows: &mut self.workspaces + &self.monitors + &self.activities.
        {
            let pool = &mut self.workspaces;
            for mon in &self.monitors {
                if output.is_some_and(|output| mon.output != *output) {
                    continue;
                }
                let Some(switch) = &mon.activity_switch else {
                    continue;
                };
                let activity = self
                    .activities
                    .get(switch.from)
                    .expect("in-flight activity switch outgoing id must be a live activity");
                // View presence is tolerated rather than expected: a cross-activity destroy can
                // legitimately drop a dormant single-entry view mid-flight without immediate
                // re-materialization.
                //
                // Sticky/shared workspaces present in both the active view (updated in the main
                // loop above) and the outgoing view receive update_render_elements twice per
                // frame. This is benign — update_render_elements is idempotent within a frame.
                let Some(view) = activity.views().get(&mon.output_id()) else {
                    continue;
                };
                // Update every id in the outgoing view, not just the rendered (culled) subset:
                // updating more than rendered is safe and renderer-free, and it avoids growing
                // the geo-ids family for the small outgoing view.
                for id in view.ids() {
                    pool.get_mut(id)
                        .expect("view id must be a key in the pool")
                        .update_render_elements(false);
                }
            }
        }
    }

    pub fn update_shaders(&mut self) {
        if let Some(InteractiveMoveState::Moving(move_)) = &mut self.interactive_move {
            move_.tile.update_shaders();
        }

        let views_map = self.activities.active().views();
        let pool = &mut self.workspaces;
        for mon in &mut self.monitors {
            let view = views_map
                .get(&OutputId::new(&mon.output))
                .expect("connected output must have a view in the active activity");
            mon.update_shaders(pool, view);
        }
        for id in &self.disconnected_workspace_ids {
            self.workspaces
                .get_mut(id)
                .expect("workspace id must be a key in the pool")
                .update_shaders();
        }
    }

    fn update_insert_hint(&mut self, output: Option<&Output>) {
        let _span = tracy_client::span!("Layout::update_insert_hint");

        for mon in self.monitors_mut() {
            mon.insert_hint = None;
        }

        if !matches!(self.interactive_move, Some(InteractiveMoveState::Moving(_))) {
            return;
        }
        let Some(InteractiveMoveState::Moving(move_)) = self.interactive_move.take() else {
            unreachable!()
        };
        if output.is_some_and(|out| &move_.output != out) {
            self.interactive_move = Some(InteractiveMoveState::Moving(move_));
            return;
        }

        let _span = tracy_client::span!("Layout::update_insert_hint::update");

        let output_id = OutputId::new(&move_.output);
        if self.monitors.iter().any(|m| m.output == move_.output) {
            let view = self
                .activities
                .active()
                .views()
                .get(&output_id)
                .expect("connected output must have a view in the active activity");
            let pool = &mut self.workspaces;
            let mon = self
                .monitors
                .iter_mut()
                .find(|m| m.output == move_.output)
                .expect("monitor for connected output must exist");
            let zoom = mon.overview_zoom();
            let ctx = LayoutCtx::new(&*pool, view);
            let (insert_ws, geo) = mon.insert_position(ctx, move_.pointer_pos_within_output);
            match insert_ws {
                InsertWorkspace::Existing(ws_id) => {
                    let ws = pool
                        .get_mut(&ws_id)
                        .expect("workspace id must be a key in the pool");
                    let pos_within_workspace =
                        (move_.pointer_pos_within_output - geo.loc).downscale(zoom);
                    let position = if move_.is_floating {
                        InsertPosition::Floating
                    } else {
                        ws.scrolling_insert_position(pos_within_workspace)
                    };

                    let border_width = move_.tile.effective_border_width().unwrap_or(0.);
                    let corner_radius = move_
                        .tile
                        .window()
                        .geometry_corner_radius()
                        .expanded_by(border_width as f32);
                    mon.insert_hint = Some(InsertHint {
                        workspace: insert_ws,
                        position,
                        corner_radius,
                    });
                }
                InsertWorkspace::NewAt(_) => {
                    let position = if move_.is_floating {
                        InsertPosition::Floating
                    } else {
                        InsertPosition::NewColumn(0)
                    };
                    mon.insert_hint = Some(InsertHint {
                        workspace: insert_ws,
                        position,
                        corner_radius: CornerRadius::default(),
                    });
                }
            }
        }

        self.interactive_move = Some(InteractiveMoveState::Moving(move_));
    }

    /// Thin `&self` wrapper over [`resolve_workspace_activities_for`]. Used by
    /// `ensure_named_workspace` (config-reload path); the startup loop in
    /// [`Self::with_options_and_workspaces`] calls the free function directly
    /// because `self` does not yet exist there.
    pub(crate) fn resolve_workspace_activities(
        &self,
        ws_config: &WorkspaceConfig,
    ) -> HashSet<ActivityId> {
        resolve_workspace_activities_for(&self.activities, ws_config)
    }

    /// Removal half of the config-reload activity reconciliation.
    ///
    /// Walks `self.activities` once, accumulating every live config-declared
    /// activity whose name is not present (case-insensitively, via the same
    /// [`Activities::resolve_config_names`] matcher the additive path uses) in
    /// `config_activities`. Runtime activities survive reload unconditionally
    /// per "Runtime activities on reload" — they are not candidates.
    ///
    /// Validation-then-mutation: every rejection class is computed over the
    /// unmutated state, and `Err` leaves `self` byte-for-byte unchanged. Only
    /// when validation completes do we touch `self.activities`,
    /// `self.workspaces`, or the active cursor.
    ///
    /// Rejection rules (all evaluated before any mutation):
    ///
    /// - Any workspace exclusive to an activity in the remove-set has windows →
    ///   [`ReloadActivityRemovalError::ExclusiveWorkspaceHasWindows`]. Both named and unnamed
    ///   exclusives are candidates here; reload is a user-initiated config change that accepts
    ///   workspace churn, unlike the IPC `RemoveActivity` path's
    ///   [`RemoveActivityError::ExclusiveNamedWorkspace`] guard.
    /// - The remove-set covers every activity in the pool (no runtime activities survive to absorb
    ///   the cascade) → [`ReloadActivityRemovalError::WouldEmptyPool`].
    /// - The active activity is in the remove-set AND [`Self::is_activity_switch_hard_blocked`]
    ///   returns `Some(_)` → [`ReloadActivityRemovalError::HardBlockedCascade`]. The cascade step
    ///   below calls [`Self::switch_activity`], whose entry `debug_assert!` requires the caller to
    ///   have filtered on the hard-block gate.
    ///
    /// Determinism on `Err`: the walks collect candidates before picking a
    /// reported offender so `HashMap::values()`'s non-deterministic iteration
    /// order doesn't flip the user-visible error message between runs. Chosen
    /// offenders are the min by `WorkspaceId::get()` / `ActivityId::get()`
    /// ascending — same precedent as
    /// `remove_activity_error_precedence_windows_beats_named` in.
    ///
    /// On success, the mutation sequence is:
    ///
    /// 1. **Cascade the active cursor** if the active activity is in the remove-set. The cascade
    ///    target is `previous_id()` when it is not itself in the remove-set; otherwise the first
    ///    declaration-order id not in the remove-set. Routed through [`Self::switch_activity`] so
    ///    in-flight workspace-switch animations snap and `ensure_all_activity_views` runs.
    /// 2. **Per target in the remove-set (declaration order), destroy every exclusive workspace** —
    ///    both named and unnamed — by dropping the id from every activity's per-output `views` (the
    ///    same retain-drop-or- shift recipe [`Self::remove_activity`] uses) and from
    ///    `self.workspaces`. The `ensure_named_workspace` pass downstream in `State::reload_config`
    ///    will recreate named config workspaces against the post-remove activity pool.
    /// 3. **Prune the target from every remaining workspace's `activities` set** where it still
    ///    appears. The set shrinks by one but remains non-empty: exclusives were destroyed in step
    ///    2, so every remaining membership is shared with at least one other activity.
    /// 4. **Remove the target from the activity pool** via [`Activities::remove`]. The
    ///    switch_activity call satisfies the `id != self.active` precondition; the would-empty-pool
    ///    validation guard satisfies the `len() > 1` precondition (evaluated once over the whole
    ///    remove-set, not per-iteration).
    ///
    /// Call ordering: this method must run BEFORE
    /// [`Self::reconcile_activities_on_reload_add`] so that the additive path's
    /// `resolve_workspace_activities_for` on config workspaces sees a post-remove
    /// pool. Any config workspace whose `activity` reference was dropped from the
    /// new config falls through to the additive path's `unknown`-name fallback
    /// (`{active_id}`).
    ///
    /// In debug builds, runs [`Self::verify_invariants`] at the tail so
    /// callers need not re-check.
    pub(crate) fn reconcile_activities_on_reload_remove(
        &mut self,
        config_activities: &[jiji_config::ActivityDecl],
    ) -> Result<(), ReloadActivityRemovalError> {
        // Validation phase: read-only, no `self` mutation until every error
        // class has been ruled out. Mirrors the atomicity contract of
        // `Layout::remove_activity`.

        // Walk `self.activities.iter()` (declaration order via IndexMap) and
        // keep config-declared activities whose name is not found in the new
        // config (case-insensitive match, same matcher `resolve_config_names`
        // uses).
        let new_names: Vec<&str> = config_activities
            .iter()
            .map(|a| a.name.0.as_str())
            .collect();
        let remove_set: HashSet<ActivityId> = self
            .activities
            .iter()
            .filter(|a| {
                a.is_config_declared()
                    && !new_names.iter().any(|n| n.eq_ignore_ascii_case(a.name()))
            })
            .map(|a| a.id())
            .collect();

        if remove_set.is_empty() {
            return Ok(());
        }

        // Classify every exclusive workspace of a remove-set activity.
        // Non-deterministic HashMap::values() order: collect all offenders then
        // pick the min by WorkspaceId::get() so the user-visible error doesn't
        // flip between runs. Same precedent as
        // `remove_activity_error_precedence_windows_beats_named`.
        let mut has_windows_offenders: Vec<(WorkspaceId, ActivityId)> = Vec::new();
        for ws in self.workspaces.values() {
            // Exclusive ⇔ `activities().len() == 1 && sole ∈ remove_set`. For
            // len == 1, `all(|id| remove_set.contains(id))` is equivalent to
            // "sole activity is in remove_set", which is the required semantics.
            if ws.activities().len() == 1
                && ws.activities().iter().all(|id| remove_set.contains(id))
                && ws.has_windows()
            {
                let sole = *ws
                    .activities()
                    .iter()
                    .next()
                    .expect("len == 1 check above guarantees one entry");
                has_windows_offenders.push((ws.id(), sole));
            }
        }
        if let Some((ws_id, act_id)) = has_windows_offenders
            .into_iter()
            .min_by_key(|(ws_id, _)| ws_id.get())
        {
            let name = self
                .activities
                .get(act_id)
                .expect("offender activity id came from remove_set; still live in pool")
                .name()
                .to_owned();
            return Err(ReloadActivityRemovalError::ExclusiveWorkspaceHasWindows {
                activity_name: name,
                workspace_id: ws_id,
            });
        }

        // Would-empty-pool check. Evaluated once over the whole remove-set
        // (not per-iteration) — the cleanest guarantee is that the
        // post-remove-set pool has ≥ 1 entry.
        debug_assert!(
            remove_set.len() <= self.activities.len(),
            "remove_set ⊆ live activities, so subtraction cannot overflow",
        );
        if remove_set.len() >= self.activities.len() {
            // Deterministic error-name pick: min by ActivityId::get() across
            // the remove_set.
            let act_id = *remove_set
                .iter()
                .min_by_key(|id| id.get())
                .expect("remove_set non-empty (empty case returned Ok above)");
            let name = self
                .activities
                .get(act_id)
                .expect("remove_set ids came from self.activities.iter() above")
                .name()
                .to_owned();
            return Err(ReloadActivityRemovalError::WouldEmptyPool {
                activity_name: name,
            });
        }

        // Hard-block guard. Only relevant when the active activity is in the
        // remove-set (the cascade below will call switch_activity); otherwise
        // no switch_activity call happens, so no hard-block gate is needed.
        if remove_set.contains(&self.activities.active_id()) {
            if let Some(block) = self.is_activity_switch_hard_blocked() {
                let active_id = self.activities.active_id();
                let name = self
                    .activities
                    .get(active_id)
                    .expect("active_id is always a live key in the pool")
                    .name()
                    .to_owned();
                return Err(ReloadActivityRemovalError::HardBlockedCascade {
                    activity_name: name,
                    block,
                });
            }
        }

        // Mutation phase: atomic from here. Validation ruled out every error
        // class, so every `.expect()` below names a condition the validator
        // established.

        // Cascade the active cursor if the active activity is in the remove-set.
        // Prefer `previous_id()` when it is not itself in the remove-set;
        // otherwise the first declaration-order id not in the remove-set.
        // Resolve the cascade target inside a narrow shared-borrow scope so the
        // subsequent `&mut self` call via switch_activity compiles.
        if remove_set.contains(&self.activities.active_id()) {
            let cascade_target: ActivityId = {
                let previous = self
                    .activities
                    .previous_id()
                    .filter(|id| !remove_set.contains(id));
                previous
                    .or_else(|| {
                        self.activities
                            .iter()
                            .map(|a| a.id())
                            .find(|id| !remove_set.contains(id))
                    })
                    .expect("WouldEmptyPool was rejected above, so at least one id survives")
            };
            self.switch_activity(cascade_target);
        }

        // Rebind orphan workspaces from to-be-removed activities into the
        // cascade target's view, before the remove pass drops the activities.
        //
        // An "orphan" is a workspace that appears in some to-be-removed
        // activity's view of a given output AND is not present in any
        // surviving activity's view of that same output. The membership
        // pruning below leaves it in the pool, but its only anchoring view
        // evaporates with `self.activities.remove`.
        //
        // Note: a workspace tagged with *both* a removed and a surviving
        // activity (e.g. `activities = [alpha, gamma]`) can still be an
        // orphan if the surviving activity's view on that output does not
        // contain it. This arises because `ensure_view_for` filters by real-
        // output match and never lifts a sentinel-`OutputId("")` workspace.
        // The predicate must be "not anchored by any surviving view on that
        // output", not "workspace.activities disjoint from remove_set" — the
        // latter misses the mixed-tag case and would leave the workspace
        // unanchored after the remove pass prunes the activity memberships.
        //
        // Background on how an orphan can reach us: named config workspaces
        // seeded via `Workspace::new_with_config_no_outputs` carry the
        // empty-string sentinel from `unwrap_or_default()` on `open_on_output`,
        // and `Workspace::bind_output` only refreshes `output_id` when
        // `matches(output)` is already true (reclaim semantic of
        // `Workspace::bind_output`'s guard). The sentinel matches no real
        // output. The first-monitor drain routes parked workspaces into their
        // member activities' views by membership, so a well-formed boot no
        // longer strands a sentinel-tagged workspace in a non-member activity's
        // view. This rebind remains as defense for orphan shapes produced by
        // paths that are not yet coherence-guaranteed — a workspace present in a
        // removed activity's view but anchored by no surviving activity's view
        // on that output. On reload-drop-active, the cascade target's
        // `ensure_all_activity_views` cannot reclaim such an orphan (its
        // `output_id` is the sentinel, not the real output), so without this
        // rebind the orphan would lose its only anchoring view.
        //
        // We rebind here rather than fixing the sentinel at its source: that
        // upstream fix touches `bind_output`'s reclaim semantic — deferred.
        // Cascade-time rebind is the lowest-ripple choice.
        let cascade_target_id = self.activities.active_id();

        // Pass 1: collect (ws_id, out_id) pairs that are anchored by at
        // least one surviving activity's view. Snapshot before the orphan
        // walk to avoid concurrent-borrow issues.
        let surviving_anchored: HashSet<(WorkspaceId, OutputId)> = self
            .activities
            .iter()
            .filter(|a| !remove_set.contains(&a.id()))
            .flat_map(|a| {
                a.views().iter().flat_map(|(out_id, view)| {
                    view.ids().iter().map(move |ws_id| (*ws_id, out_id.clone()))
                })
            })
            .collect();

        // Pass 2: enumerate workspaces in to-be-removed views that are not
        // in the surviving-anchored snapshot.
        let mut orphans: Vec<(WorkspaceId, OutputId)> = Vec::new();
        for activity in self.activities.iter() {
            if !remove_set.contains(&activity.id()) {
                continue;
            }
            for (out_id, view) in activity.views() {
                for ws_id in view.ids() {
                    if !surviving_anchored.contains(&(*ws_id, out_id.clone())) {
                        orphans.push((*ws_id, out_id.clone()));
                    }
                }
            }
        }

        // A single workspace can appear in multiple to-be-removed activities'
        // views; insert once per (ws_id, out_id) pair.
        let mut seen: HashSet<(WorkspaceId, OutputId)> = HashSet::new();
        for (ws_id, out_id) in orphans {
            if !seen.insert((ws_id, out_id.clone())) {
                continue;
            }

            // Refresh the sentinel `OutputId("")` if present: bind the orphan
            // to the real output it now lives on so future
            // `ensure_all_activity_views` filters can recognise it. Gated strictly
            // on the empty-string sentinel — preserves the documented
            // `bind_output` reclaim semantic for any other shape.
            {
                let ws = self
                    .workspaces
                    .get_mut(&ws_id)
                    .expect("orphan id was sourced from a live view above");
                if ws
                    .output_id
                    .as_ref()
                    .is_none_or(|oid| oid.as_str().is_empty())
                {
                    debug_assert!(
                        !out_id.as_str().is_empty(),
                        "rebind must produce a real OutputId, not the sentinel",
                    );
                    ws.output_id = Some(out_id.clone());
                }
            }

            let cascade_target_act = self
                .activities
                .get_mut(cascade_target_id)
                .expect("cascade target was just switched to via switch_activity above");
            let view = cascade_target_act
                .views_mut()
                .get_mut(&out_id)
                .unwrap_or_else(|| {
                    unreachable!(
                        "cascade target's view on {out_id:?} must exist post-ensure_all_activity_views",
                    )
                });
            debug_assert!(
                view.position_of(ws_id).is_none(),
                "orphan must not already be in cascade target's view \
                 (surviving_anchored would have caught it)",
            );
            debug_assert!(
                view.len() > 0,
                "ensure_all_activity_views guarantees view non-empty",
            );
            // When the cascade target's view was a fresh-branch singleton (len == 1
            // means only the trailing-empty bookend, per Monitor::verify_invariants
            // "1 or 3+ length" rule), the user was about to land on a blank
            // workspace because `ensure_view_for` couldn't lift the orphan
            // (sentinel output_id). Now that the orphan is rebound into this view,
            // shift the active cursor onto it so the user lands on the orphan that
            // was active under the removed activity.
            let was_fresh_singleton = view.len() == 1;
            // Insert before the trailing-empty bookend so Monitor invariant
            // "last must be empty/unnamed" is preserved.
            let insert_pos = view.len() - 1;
            view.insert(insert_pos, ws_id);
            if was_fresh_singleton {
                view.set_active_at(insert_pos);
            }
        }

        // Per target in the remove-set, destroy exclusive workspaces (named and
        // unnamed) and prune shared memberships.
        //
        // A Vec<ActivityId> snapshot is required because `Activities::remove`
        // shrinks the map each pass, so iterating the live Activities would
        // skip entries.
        let targets: Vec<ActivityId> = self
            .activities
            .iter()
            .map(|a| a.id())
            .filter(|id| remove_set.contains(id))
            .collect();
        // Collect every workspace id whose membership set is narrowed by the
        // prune passes below so the reclaim helper can check them afterwards.
        let mut all_pruned_ids: Vec<WorkspaceId> = Vec::new();
        for target in targets {
            // Pre-collect destroy ids to release the shared borrow on
            // self.workspaces before the `iter_mut()` + `remove()` mutations.
            let destroy_ids: Vec<WorkspaceId> = self
                .workspaces
                .values()
                .filter(|ws| ws.activities().len() == 1 && ws.activities().contains(&target))
                .map(|ws| ws.id())
                .collect();

            // Patch every activity's `views`: drop view entries that would
            // become empty, shift `active` / `previous` otherwise. Same recipe
            // `Layout::remove_activity` uses at the exclusive-destroy site.
            for ws_id in &destroy_ids {
                for activity in self.activities.iter_mut() {
                    activity.views_mut().retain(|_output_id, view| {
                        let Some(pos) = view.position_of(*ws_id) else {
                            return true;
                        };
                        if view.len() == 1 {
                            return false;
                        }
                        view.remove_at(pos);
                        true
                    });
                }
                assert!(
                    self.workspaces.remove(ws_id).is_some(),
                    "destroy id {ws_id:?} came from values() above — must be a live pool key",
                );
            }

            // Prune `target` from every remaining workspace's `activities` set
            // where it still appears. Exclusives were destroyed just above, so
            // every remaining membership is shared — the set shrinks but stays
            // non-empty.
            for ws in self.workspaces.values_mut() {
                if ws.activities().contains(&target) {
                    debug_assert!(
                        ws.activities().len() > 1,
                        "exclusives of {target:?} were destroyed in the pass above",
                    );
                    ws.activities.remove(&target);
                    all_pruned_ids.push(ws.id());
                }
            }

            // Remove the activity from the pool. `Activities::remove`'s
            // preconditions are satisfied: target ≠ active (step 6 cascaded if
            // it was), and len > 1 (the WouldEmptyPool check ruled out a
            // would-empty final state).
            let _ = self.activities.remove(target);
        }

        // Any monitors whose activity-switch transition was departing from a now-removed activity
        // hold a dead id in `from`. Snap those before the invariant check fires.
        self.snap_stale_activity_switches();

        // Removing activities may have caused `destroy_workspaces_cross_activity` to drop
        // single-entry views in other activities (mirroring the `remove_activity` path); the
        // materializer re-installs bookend views for every remaining activity on every
        // connected monitor.
        self.ensure_all_activity_views();

        // The membership prune passes above shrank each collected workspace's
        // activities set by one per removed target. Any that became exclusive to a
        // single activity may now sit in an illegal middle position.
        self.reclaim_unpinned_empty_workspaces(all_pruned_ids);

        #[cfg(debug_assertions)]
        self.verify_invariants();

        Ok(())
    }

    /// Additive half of the config-reload activity reconciliation.
    ///
    /// For each entry in `config_activities`, in declaration order:
    /// - If an existing activity resolves case-insensitively against `name`, keep its id (and every
    ///   workspace's `activities` set referencing it), and flip its `is_config_declared` flag to
    ///   `true` if it was runtime. The stored name is NOT updated to the config casing — runtime
    ///   spelling is preserved per bullet 1.
    /// - Otherwise, mint a fresh config-declared activity with the given name and append it to the
    ///   pool.
    ///
    /// After the per-entry walk, the pool is reordered so config-declared
    /// activities occupy positions `[0, N)` in config-declaration order;
    /// runtime-only activities (those not named in `config_activities`) fall
    /// to the trailer in their current relative order (AD6 display-order
    /// semantics).
    ///
    /// Sticky workspaces (whose `activities` set must equal every live
    /// activity id) are re-expanded to the full post-reconcile id set, in
    /// case newly-added config activities grew the universe.
    ///
    /// Workspaces whose `name` matches a `config_workspaces` entry have their
    /// `activities` set overwritten with
    /// [`resolve_workspace_activities_for`]'s result — a config-declared
    /// workspace's activity assignment is always authoritative on reload
    /// (workspace-activity assignments on reload). Sticky workspaces are
    /// skipped (already handled above). Dynamic (unnamed / non-config)
    /// workspaces keep their runtime assignments.
    ///
    /// This covers the additive / same-name-preserving half of the reload
    /// reconciliation. Removal-side rules (exclusive-workspace rejection of activities that
    /// disappear from config, active-cursor cascade when the active activity
    /// is removed, and destruction of empty exclusive workspaces) are handled
    /// by a separate entry point on `Layout` for the removal-side rules.
    ///
    /// The active / previous activity cursors are not touched here.
    ///
    /// **Preconditions:** callers must have already cleared the name off any
    /// workspace whose name is not present in `config_workspaces`; see
    /// `State::reload_config` for the prewalk that does this. In debug builds
    /// this precondition is enforced via debug-assert. In release builds,
    /// named workspaces absent from `config_workspaces` are silently skipped —
    /// the caller contract is the only safeguard.
    ///
    /// This must be called *before* [`Self::update_config`] so that
    /// [`Self::ensure_named_workspace`] (which consults the post-reload
    /// pool via [`Self::resolve_workspace_activities`]) sees the reconciled
    /// activities.
    ///
    /// In debug builds, runs [`Self::verify_invariants`] at the tail so
    /// callers need not re-check.
    pub(crate) fn reconcile_activities_on_reload_add(
        &mut self,
        config_activities: &[jiji_config::ActivityDecl],
        config_workspaces: &[jiji_config::Workspace],
    ) {
        // Precondition: every named non-sticky workspace's name must appear in
        // config_workspaces. The caller (State::reload_config) must have run
        // the unname_workspace prewalk before calling this function. Enforce in
        // debug builds so a future caller omitting the prewalk is caught early.
        #[cfg(debug_assertions)]
        {
            let config_ws_names: HashSet<&str> = config_workspaces
                .iter()
                .map(|w| w.name.0.as_str())
                .collect();
            for ws in self.workspaces.values() {
                if ws.is_sticky() {
                    continue;
                }
                if let Some(name) = ws.name() {
                    debug_assert!(
                        config_ws_names.contains(name.as_str()),
                        "reconcile_activities_on_reload_add: workspace {:?} still named {:?} \
                         but {:?} is absent from config_workspaces — caller must run the \
                         unname_workspace prewalk in State::reload_config first",
                        ws.id(),
                        name,
                        name,
                    );
                }
            }
        }

        // Walk config activities in declaration order, resolve each against the
        // live pool, promoting on-match or appending on-miss.
        // `resolve_config_names` returns at most one match per name because
        // `ActivityNameSet` at parse time guarantees config names are unique.
        let mut config_declared_ids: Vec<ActivityId> = Vec::with_capacity(config_activities.len());
        for entry in config_activities {
            let name = entry.name.0.clone();
            // _unknown is intentionally discarded here: names not matched by
            // any live activity are minted below via `add_config_declared`.
            let (resolved, _unknown) = self
                .activities
                .resolve_config_names(std::slice::from_ref(&name));
            if let Some(id) = resolved.into_iter().next() {
                let needs_promotion = !self
                    .activities
                    .get(id)
                    .expect("resolved id must be a live key")
                    .is_config_declared();
                if needs_promotion {
                    self.activities.promote_to_config_declared(id);
                }
                config_declared_ids.push(id);
            } else {
                let new_id = self.activities.add_config_declared(name);
                config_declared_ids.push(new_id);
            }
        }

        // Reorder to match config declaration order; runtime-only activities
        // follow after the prefix in their current relative order.
        self.activities
            .reorder_to_match_config(&config_declared_ids);

        // Sticky re-expansion — snapshot the live id universe before the
        // mutating walk so the borrow on `self.activities` is released while
        // we mutate `self.workspaces`.
        let all_ids: HashSet<ActivityId> = self.activities.iter().map(|a| a.id()).collect();
        for ws in self.workspaces.values_mut() {
            if ws.is_sticky() {
                ws.activities = all_ids.clone();
            }
        }

        // Reset config-declared workspaces' `activities` sets. Pre-collect
        // (ws_id, new_set) pairs so the shared borrow of `self.activities` is
        // released before the mutating second walk.
        let mut resets: Vec<(WorkspaceId, HashSet<ActivityId>)> = Vec::new();
        for ws in self.workspaces.values() {
            if ws.is_sticky() {
                continue;
            }
            let Some(name) = ws.name() else { continue };
            if let Some(ws_config) = config_workspaces.iter().find(|w| w.name.0 == *name) {
                let new_set = resolve_workspace_activities_for(&self.activities, ws_config);
                resets.push((ws.id(), new_set));
            }
        }
        for (id, set) in resets {
            self.workspaces
                .get_mut(&id)
                .expect("id came from values() above — still live")
                .activities = set;
        }

        // Newly-added config-declared activities (or freshly-promoted ones) must hold a bookend
        // view on every connected monitor — the per-activity bookend invariant.
        self.ensure_all_activity_views();

        // The materializer runs first so fresh activities get their tag-filtered lift views; the
        // sweep then heals what the tag filter missed (stale-tagged sticky members) and what the
        // wholesale membership resets and sticky re-expansion above changed on *existing* views,
        // in both directions.
        self.reconcile_views_with_membership();

        #[cfg(debug_assertions)]
        self.verify_invariants();
    }

    pub fn ensure_named_workspace(&mut self, ws_config: &WorkspaceConfig) {
        if self.find_workspace_by_name(&ws_config.name.0).is_some() {
            return;
        }

        let clock = self.clock.clone();
        let options = self.options.clone();

        if self.monitors.is_empty() {
            let ws_activities = self.resolve_workspace_activities(ws_config);
            let ws = Workspace::new_with_config_no_outputs(
                Some(ws_config.clone()),
                ws_activities,
                clock,
                options,
            );
            let id = ws.id();
            assert!(
                self.workspaces.insert(id, ws).is_none(),
                "fresh id must be unique",
            );
            self.disconnected_workspace_ids.insert(0, id);
            return;
        }

        let primary_idx = self.primary_idx;
        let active_monitor_idx = self.active_monitor_idx;
        let mon_idx = ws_config
            .open_on_output
            .as_deref()
            .map(|name| {
                self.monitors
                    .iter()
                    .position(|monitor| output_matches_name(&monitor.output, name))
                    .unwrap_or(primary_idx)
            })
            .unwrap_or(active_monitor_idx);
        let ws_activities = self.resolve_workspace_activities(ws_config);
        let active_id = self.activities.active_id();
        let active_is_member = ws_activities.contains(&active_id);
        let mon = &self.monitors[mon_idx];

        let ws = Workspace::new_with_config(
            &mon.output,
            Some(ws_config.clone()),
            ws_activities.clone(),
            clock,
            options,
        );
        let id = ws.id();
        assert!(
            self.workspaces.insert(id, ws).is_none(),
            "fresh id must be unique",
        );

        let out_id = self.monitors[mon_idx].output_id();

        if active_is_member {
            // The active activity declares this workspace (always the case for sticky and for
            // configs that name no activities): keep the existing top-of-strip behavior —
            // `insert_workspace_onto_monitor` binds the output, updates config, and inserts the
            // workspace into the active view at position 0.
            self.insert_workspace_onto_monitor(mon_idx, id, 0, false);
        } else {
            // Declared exclusively for dormant activities: never touch the active view. Replicate
            // `insert_workspace_onto_monitor`'s per-workspace setup on the new (empty) workspace
            // only — `bind_output` on a windowless workspace is a tag refresh + size sync (no
            // `output_enter`), and `update_config` merges the monitor's options.
            let mon_output = self.monitors[mon_idx].output.clone();
            let mon_options = self.monitors[mon_idx].options.clone();
            let ws = self
                .workspaces
                .get_mut(&id)
                .expect("just-inserted id must be a live pool key");
            ws.bind_output(&mon_output);
            ws.update_config(mon_options);
        }

        // Install into every declared member activity's view except the active one, keyed by the
        // resolved monitor's output. Walk `self.activities` in declaration order — iterating the
        // membership `HashSet` directly would leak its non-deterministic order into view order.
        let member_ids: Vec<ActivityId> = self
            .activities
            .iter()
            .map(|a| a.id())
            .filter(|a_id| *a_id != active_id && ws_activities.contains(a_id))
            .collect();
        for act_id in member_ids {
            let pool = &mut self.workspaces;
            let activities = &mut self.activities;
            let activity = activities
                .get_mut(act_id)
                .expect("member id came from self.activities.iter()");
            // Widened absence guard, mirroring `add_workspace_to_activity`: absent from every view
            // of the target activity, not just the holding-output one.
            let already_present = activity
                .views()
                .values()
                .any(|view| view.ids().contains(&id));
            if already_present {
                continue;
            }
            if let Some(view) = activity.views_mut().get_mut(&out_id) {
                Self::view_insert_above_trailing_bookend(pool, view, id);
            } else {
                debug_assert!(
                    false,
                    "ensure_named_workspace: activity {act_id:?} has no view for {out_id:?}, a \
                     connected output — per-activity bookend invariant violated (membership↔view \
                     coherence bug)",
                );
            }
        }

        // Repair the EWAF leading-empty rule across every dormant member view. The named
        // workspace inserts above the trailing bookend, so it never lands at a trailing slot; only
        // the EWAF leading rule can need repair, which the sweep handles (redundantly for the
        // active view, already maintained by `insert_workspace_onto_monitor` in the
        // active-member branch).
        self.normalize_view_bookends();

        #[cfg(debug_assertions)]
        self.verify_invariants();
    }

    pub fn update_config(&mut self, config: &Config, overrides: &FlattenedAppearance) {
        // Update workspace-specific config for all named workspaces.
        for ws in self.workspaces_mut() {
            let Some(name) = ws.name() else { continue };
            if let Some(config) = config.workspaces.iter().find(|w| &w.name.0 == name) {
                ws.update_layout_config(config.layout.clone().map(|x| x.0));
            }
        }

        let mut options = Options::from_config(config);
        options.layout.focus_ring.merge_with(&overrides.focus_ring);
        if let Some(color) = overrides.background_color {
            options.layout.background_color = color;
        }
        self.update_options(options);
    }

    /// Read-only access to the composed [`Options`] the layout is currently
    /// rendering with, for headless render-model assertions.
    #[cfg(test)]
    pub(crate) fn options(&self) -> &Options {
        &self.options
    }

    fn update_options(&mut self, options: Options) {
        let options = Rc::new(options);

        if let Some(InteractiveMoveState::Moving(move_)) = &mut self.interactive_move {
            let view_size = output_size(&move_.output);
            let scale = move_.output.current_scale().fractional_scale();
            let options = Options::clone(&options)
                .with_merged_layout(move_.output_config.as_ref())
                .with_merged_layout(move_.workspace_config.as_ref().map(|(_, c)| c))
                .adjusted_for_scale(scale);
            move_.tile.update_config(view_size, scale, Rc::new(options));
        }

        let seed_activity = self.activities.active_id();
        let views_map = self.activities.active_mut().views_mut();
        let pool = &mut self.workspaces;
        for mon in &mut self.monitors {
            let view = views_map
                .get_mut(&OutputId::new(&mon.output))
                .expect("connected output must have a view in the active activity");
            mon.update_config(pool, view, options.clone(), seed_activity);
        }
        for id in &self.disconnected_workspace_ids {
            self.workspaces
                .get_mut(id)
                .expect("workspace id must be a key in the pool")
                .update_config(options.clone());
        }

        self.options = options;
    }

    pub fn toggle_width(&mut self, forwards: bool) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.toggle_width(forwards);
    }

    pub fn toggle_window_width(&mut self, window: Option<&W::Id>, forwards: bool) {
        if let Some(InteractiveMoveState::Moving(move_)) = &mut self.interactive_move {
            if window.is_none() || window == Some(move_.tile.window().id()) {
                return;
            }
        }

        let workspace = if let Some(window) = window {
            Some(
                self.workspaces_mut()
                    .find(|ws| ws.has_window(window))
                    .unwrap(),
            )
        } else {
            self.active_workspace_mut()
        };

        let Some(workspace) = workspace else {
            return;
        };
        workspace.toggle_window_width(window, forwards);
    }

    pub fn toggle_window_height(&mut self, window: Option<&W::Id>, forwards: bool) {
        if let Some(InteractiveMoveState::Moving(move_)) = &mut self.interactive_move {
            if window.is_none() || window == Some(move_.tile.window().id()) {
                return;
            }
        }

        let workspace = if let Some(window) = window {
            Some(
                self.workspaces_mut()
                    .find(|ws| ws.has_window(window))
                    .unwrap(),
            )
        } else {
            self.active_workspace_mut()
        };

        let Some(workspace) = workspace else {
            return;
        };
        workspace.toggle_window_height(window, forwards);
    }

    pub fn toggle_full_width(&mut self) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.toggle_full_width();
    }

    pub fn set_column_width(&mut self, change: SizeChange) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.set_column_width(change);
    }

    pub fn set_window_width(&mut self, window: Option<&W::Id>, change: SizeChange) {
        if let Some(InteractiveMoveState::Moving(move_)) = &mut self.interactive_move {
            if window.is_none() || window == Some(move_.tile.window().id()) {
                return;
            }
        }

        let workspace = if let Some(window) = window {
            Some(
                self.workspaces_mut()
                    .find(|ws| ws.has_window(window))
                    .unwrap(),
            )
        } else {
            self.active_workspace_mut()
        };

        let Some(workspace) = workspace else {
            return;
        };
        workspace.set_window_width(window, change);
    }

    pub fn set_window_height(&mut self, window: Option<&W::Id>, change: SizeChange) {
        if let Some(InteractiveMoveState::Moving(move_)) = &mut self.interactive_move {
            if window.is_none() || window == Some(move_.tile.window().id()) {
                return;
            }
        }

        let workspace = if let Some(window) = window {
            Some(
                self.workspaces_mut()
                    .find(|ws| ws.has_window(window))
                    .unwrap(),
            )
        } else {
            self.active_workspace_mut()
        };

        let Some(workspace) = workspace else {
            return;
        };
        workspace.set_window_height(window, change);
    }

    pub fn reset_window_height(&mut self, window: Option<&W::Id>) {
        if let Some(InteractiveMoveState::Moving(move_)) = &mut self.interactive_move {
            if window.is_none() || window == Some(move_.tile.window().id()) {
                return;
            }
        }

        let workspace = if let Some(window) = window {
            Some(
                self.workspaces_mut()
                    .find(|ws| ws.has_window(window))
                    .unwrap(),
            )
        } else {
            self.active_workspace_mut()
        };

        let Some(workspace) = workspace else {
            return;
        };
        workspace.reset_window_height(window);
    }

    pub fn expand_column_to_available_width(&mut self) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.expand_column_to_available_width();
    }

    pub fn toggle_window_floating(&mut self, window: Option<&W::Id>) {
        if let Some(InteractiveMoveState::Moving(move_)) = &mut self.interactive_move {
            if window.is_none() || window == Some(move_.tile.window().id()) {
                move_.is_floating = !move_.is_floating;

                // When going to floating, restore the floating window size.
                if move_.is_floating {
                    let floating_size = move_.tile.floating_window_size;
                    let win = move_.tile.window_mut();
                    let mut size =
                        floating_size.unwrap_or_else(|| win.expected_size().unwrap_or_default());

                    // Apply min/max size window rules. If requesting a concrete size, apply
                    // completely; if requesting (0, 0), apply only when min/max results in a fixed
                    // size.
                    let min_size = win.min_size();
                    let max_size = win.max_size();
                    size.w = ensure_min_max_size_maybe_zero(size.w, min_size.w, max_size.w);
                    size.h = ensure_min_max_size_maybe_zero(size.h, min_size.h, max_size.h);

                    win.request_size_once(size, true);

                    // Animate the tile back to opaque.
                    move_.tile.animate_alpha(
                        INTERACTIVE_MOVE_ALPHA,
                        1.,
                        self.options.animations.window_movement.0,
                    );

                    // Unlock the view on the workspaces.
                    for ws in self.workspaces_mut() {
                        ws.dnd_scroll_gesture_end();
                    }
                } else {
                    // Animate the tile back to semitransparent.
                    move_.tile.animate_alpha(
                        1.,
                        INTERACTIVE_MOVE_ALPHA,
                        self.options.animations.window_movement.0,
                    );
                    move_.tile.hold_alpha_animation_after_done();
                }

                return;
            }
        }

        let workspace = if let Some(window) = window {
            Some(
                self.workspaces_mut()
                    .find(|ws| ws.has_window(window))
                    .unwrap(),
            )
        } else {
            self.active_workspace_mut()
        };

        let Some(workspace) = workspace else {
            return;
        };
        workspace.toggle_window_floating(window);
    }

    pub fn set_window_floating(&mut self, window: Option<&W::Id>, floating: bool) {
        if let Some(InteractiveMoveState::Moving(move_)) = &mut self.interactive_move {
            if window.is_none() || window == Some(move_.tile.window().id()) {
                if move_.is_floating != floating {
                    self.toggle_window_floating(window);
                }
                return;
            }
        }

        let workspace = if let Some(window) = window {
            Some(
                self.workspaces_mut()
                    .find(|ws| ws.has_window(window))
                    .unwrap(),
            )
        } else {
            self.active_workspace_mut()
        };

        let Some(workspace) = workspace else {
            return;
        };
        workspace.set_window_floating(window, floating);
    }

    pub fn focus_floating(&mut self) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.focus_floating();
    }

    pub fn focus_tiling(&mut self) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.focus_tiling();
    }

    pub fn switch_focus_floating_tiling(&mut self) {
        let Some(workspace) = self.active_workspace_mut() else {
            return;
        };
        workspace.switch_focus_floating_tiling();
    }

    pub fn move_floating_window(
        &mut self,
        id: Option<&W::Id>,
        x: PositionChange,
        y: PositionChange,
        animate: bool,
    ) {
        if let Some(InteractiveMoveState::Moving(move_)) = &mut self.interactive_move {
            if id.is_none() || id == Some(move_.tile.window().id()) {
                return;
            }
        }

        let workspace = if let Some(id) = id {
            Some(self.workspaces_mut().find(|ws| ws.has_window(id)).unwrap())
        } else {
            self.active_workspace_mut()
        };

        let Some(workspace) = workspace else {
            return;
        };
        workspace.move_floating_window(id, x, y, animate);
    }

    pub fn focus_output(&mut self, output: &Output) {
        for (idx, mon) in self.monitors.iter().enumerate() {
            if &mon.output == output {
                self.active_monitor_idx = idx;
                return;
            }
        }
    }

    pub fn move_to_output(
        &mut self,
        window: Option<&W::Id>,
        output: &Output,
        target_ws_idx: Option<usize>,
        activate: ActivateWindow,
    ) {
        if let Some(InteractiveMoveState::Moving(move_)) = &mut self.interactive_move {
            if window.is_none() || window == Some(move_.tile.window().id()) {
                return;
            }
        }

        if self.monitors.is_empty() {
            return;
        }

        let new_idx = self
            .monitors
            .iter()
            .position(|mon| &mon.output == output)
            .unwrap();

        let (mon_idx, ws_idx) = if let Some(window) = window {
            // Mirrors the gate in `move_to_workspace`: the id-based
            // action lookup may resolve a window on a workspace bound
            // to a dormant activity, which this active-view-only
            // walk cannot reach. Silently drop with a `warn!` —
            // cross-activity move-by-id semantics are deferred
            let views = self.activities.active().views();
            let Some((mon_idx, ws_idx)) =
                self.monitors.iter().enumerate().find_map(|(mon_idx, mon)| {
                    views
                        .get(&OutputId::new(&mon.output))
                        .expect("connected output must have a view in the active activity")
                        .ids()
                        .iter()
                        .position(|id| {
                            self.workspaces
                                .get(id)
                                .is_some_and(|ws| ws.has_window(window))
                        })
                        .map(|ws_idx| (mon_idx, ws_idx))
                })
            else {
                warn!(
                    "move_to_output: window {:?} is not on the active activity; \
                     cross-activity move-by-id semantics are deferred. \
                     Dropping action.",
                    window,
                );
                return;
            };
            (mon_idx, ws_idx)
        } else {
            let mon_idx = self.active_monitor_idx;
            let mon = &self.monitors[mon_idx];
            (
                mon_idx,
                self.active_view(&mon.output_id()).active_position(),
            )
        };

        let workspace_idx = target_ws_idx.unwrap_or_else(|| {
            self.active_view(&self.monitors[new_idx].output_id())
                .active_position()
        });
        if mon_idx == new_idx && ws_idx == workspace_idx {
            return;
        }

        {
            let mon = &self.monitors[new_idx];
            let view = self.active_view(&mon.output_id());
            if view.len() <= workspace_idx {
                return;
            }
        }

        let ws_id = {
            let mon = &self.monitors[new_idx];
            let view = self.active_view(&mon.output_id());
            view.ids()[workspace_idx]
        };

        let seed_activity = self.activities.active_id();
        let active_monitor_idx_val = self.active_monitor_idx;
        let source_out = self.monitors[mon_idx].output_id();
        let removed = {
            let (monitors, pool, view) = self.monitors_pool_view_mut(&source_out);
            let mon = &mut monitors[mon_idx];
            let active_window_id = mon.active_window(pool, view).map(|w| w.id().clone());
            let activate_eager = activate.map_smart(|| {
                window.is_none_or(|win| {
                    mon_idx == active_monitor_idx_val && active_window_id.as_ref() == Some(win)
                })
            });
            let activate = if activate_eager {
                ActivateWindow::Yes
            } else {
                ActivateWindow::No
            };

            let ws = mon.workspace_at_mut(pool, view, ws_idx);
            let transaction = Transaction::new();
            let mut removed = if let Some(window) = window {
                ws.remove_tile(Some(&mon.output), window, transaction)
            } else if let Some(removed) = ws.remove_active_tile(Some(&mon.output), transaction) {
                removed
            } else {
                return;
            };

            removed.tile.stop_move_animations();
            (removed, activate)
        };
        let (removed, activate) = removed;

        // new_idx may differ from mon_idx; re-borrow its view.
        let target_out = self.monitors[new_idx].output_id();
        let (monitors, pool, view) = self.monitors_pool_view_mut(&target_out);
        Self::add_tile_on(
            monitors,
            pool,
            view,
            new_idx,
            removed.tile,
            MonitorAddWindowTarget::Workspace {
                id: ws_id,
                column_idx: None,
            },
            activate,
            true,
            removed.width,
            removed.is_full_width,
            removed.is_floating,
            seed_activity,
        );
        if activate.map_smart(|| false) {
            self.active_monitor_idx = new_idx;
        }

        let mon_out = self.monitors[mon_idx].output_id();
        if self.monitors[mon_idx].workspace_switch.is_none() {
            let ids_to_destroy = {
                let (monitors, pool, view) = self.monitors_pool_view_mut(&mon_out);
                Self::clean_up_workspaces_on(monitors, pool, view, mon_idx)
            };
            Self::destroy_workspaces_cross_activity(
                &mut self.activities,
                &mut self.workspaces,
                ids_to_destroy,
            );
        }
    }

    pub fn move_column_to_output(
        &mut self,
        output: &Output,
        target_ws_idx: Option<usize>,
        activate: bool,
    ) {
        if self.monitors.is_empty() {
            return;
        }

        let new_idx = self
            .monitors
            .iter()
            .position(|mon| &mon.output == output)
            .unwrap();

        let active_monitor_idx = self.active_monitor_idx;
        let current_out = self.monitors[active_monitor_idx].output_id();
        let active_pos = self.active_view(&current_out).active_position();

        // Check floating status on a shared borrow first; move_to_output needs `&mut self`,
        // so we can't take a mutable borrow yet.
        let is_floating = {
            let pool = &self.workspaces;
            let view = self.active_view(&current_out);
            self.monitors[active_monitor_idx]
                .workspace_at(pool, view, active_pos)
                .floating_is_active()
        };
        if is_floating {
            self.move_to_output(None, output, None, ActivateWindow::Smart);
            return;
        }

        // Scrolling path.
        let (monitors, pool, view) = self.monitors_pool_view_mut(&current_out);
        let current = &mut monitors[active_monitor_idx];
        let current_output_ref = &current.output;
        let ws = current.workspace_at_mut(pool, view, active_pos);

        let Some(column) = ws.remove_active_column(Some(current_output_ref)) else {
            return;
        };

        let new_out = self.monitors[new_idx].output_id();
        let new_view = self.active_view(&new_out);
        let workspace_idx = target_ws_idx
            .unwrap_or(new_view.active_position())
            .min(new_view.len() - 1);
        self.add_column_by_idx(new_idx, workspace_idx, column, activate);
    }

    pub fn move_workspace_to_output(&mut self, output: &Output) -> bool {
        if self.monitors.is_empty() {
            return false;
        }
        let mon_out = self.monitors[self.active_monitor_idx].output_id();
        let idx = self.active_view(&mon_out).active_position();
        self.move_workspace_to_output_by_id(idx, None, output)
    }

    // FIXME: accept workspace by id
    pub fn move_workspace_to_output_by_id(
        &mut self,
        old_idx: usize,
        old_output: Option<Output>,
        new_output: &Output,
    ) -> bool {
        // Resolve monitor indices, bail-out conditions, and the `activate` flag under a shared
        // borrow so the subsequent self-methods (`remove_workspace_from_monitor`,
        // `insert_workspace_onto_monitor`) can take `&mut self`.
        let (current_idx, target_idx, target_pos, activate) = {
            if self.monitors.is_empty() {
                return false;
            }

            let current_idx = if let Some(old_output) = old_output {
                self.monitors
                    .iter()
                    .position(|mon| mon.output == old_output)
                    .unwrap()
            } else {
                self.active_monitor_idx
            };
            let target_idx = self
                .monitors
                .iter()
                .position(|mon| mon.output == *new_output)
                .unwrap();

            let current = &self.monitors[current_idx];
            let current_view = self.active_view(&current.output_id());
            if current_view.len() <= old_idx {
                return false;
            }

            // Only switch active monitor if the workspace to be moved is the currently focused
            // one on the current monitor. Computed eagerly on both the cross-output and same-
            // output paths; on the same-output short-circuit these are pure reads of view state,
            // so the wasted work is harmless and keeps the shared-borrow scope rectangular.
            let activate =
                current_idx == self.active_monitor_idx && old_idx == current_view.active_position();
            let target_view = self.active_view(&self.monitors[target_idx].output_id());
            let target_pos = target_view.active_position() + 1;

            (current_idx, target_idx, target_pos, activate)
        };

        // Do not do anything if the output is already correct.
        if current_idx == target_idx {
            // Just update the designated output id since this is an explicit movement action.
            let mon_out = self.monitors[current_idx].output_id();
            let (monitors, pool, view) = self.monitors_pool_view_mut(&mon_out);
            let mon = &mut monitors[current_idx];
            let new_output_id = Some(OutputId::new(mon.output()));
            mon.workspace_at_mut(pool, view, old_idx).output_id = new_output_id;
            return false;
        }

        let ws_id = self.remove_workspace_from_monitor(current_idx, old_idx);
        self.workspaces
            .get_mut(&ws_id)
            .expect("workspace id must be a key in the pool")
            .output_id = Some(OutputId::new(new_output));
        self.insert_workspace_onto_monitor(target_idx, ws_id, target_pos, activate);

        // Cross-activity fan-out: a workspace that belongs to multiple activities must move in
        // *every* activity's view of the source output to the corresponding view of the target
        // output — otherwise dormant activities keep a stale entry pointing at the prior output
        // and the next switch surfaces the workspace under the wrong monitor. The active
        // activity's view is already migrated by remove/insert_workspace_onto_monitor above, so
        // skip it here to avoid the `WorkspaceView::insert` duplicate-id panic.
        //
        // Source side: single-entry drop vs. `remove_at` to preserve
        // `WorkspaceView::active`/`previous` — mirrors `set_workspace_activities`.
        // Target side: insert at `len-1` to preserve the trailing-empty bookend
        // that `ensure_view_for` materialized — the windowed-workspace path of
        // `view_insert_above_trailing_bookend`, with the gating implicit here via
        // invariants (the moved workspace is windowed/named; `ensure_view_for`
        // guarantees an empty-unnamed tail). Dormant activities lacking a
        // target-side view are left absent — `ensure_view_for` materializes
        // lazily on the next switch.
        let source_out_id = self.monitors[current_idx].output_id();
        let target_out_id = self.monitors[target_idx].output_id();
        let active_id = self.activities.active_id();

        let act_ids: Vec<ActivityId> = self
            .workspaces
            .get(&ws_id)
            .expect(
                "ws_id must be a live pool key — pool ownership is invariant across \
                 remove/insert_workspace_onto_monitor (only the view binding moves)",
            )
            .activities()
            .iter()
            .copied()
            .collect();

        for act_id in act_ids {
            if act_id == active_id {
                continue;
            }
            let activity = self.activities.get_mut(act_id).unwrap_or_else(|| {
                unreachable!(
                    "Layout invariant: act_id {act_id:?} from workspace {ws_id:?}.activities \
                     must be a live key in Layout.activities \
                     (verify_invariants:6138-6145)"
                )
            });

            // Source side: drop the workspace from this activity's source-output view if present.
            // Single-entry view → remove the map entry entirely (mirrors set_workspace_activities);
            // multi-entry view → `remove_at` to preserve the rest.
            if let Some(view) = activity.views_mut().get_mut(&source_out_id) {
                if let Some(pos) = view.position_of(ws_id) {
                    if view.len() == 1 {
                        activity.views_mut().remove(&source_out_id);
                    } else {
                        view.remove_at(pos);
                    }
                }
            }

            // Target side: insert just before the trailing-empty bookend. `view.len() - 1` is
            // safe because every existing view has at least one id by `WorkspaceView`'s
            // non-empty invariant. If no view exists for the target output, leave absent —
            // `ensure_view_for` materializes a fresh view on the next switch into this activity.
            if let Some(view) = activity.views_mut().get_mut(&target_out_id) {
                if !view.ids().contains(&ws_id) {
                    let insert_pos = view.len() - 1;
                    view.insert(insert_pos, ws_id);
                }
            }
        }

        if activate {
            // The insert_workspace_onto_monitor call above took &mut self, so the earlier
            // shared-borrow scope is gone. `monitors` is still non-empty (early-return above guards
            // it, and nothing in the call chain drops monitors).
            assert!(
                !self.monitors.is_empty(),
                "monitors must be non-empty after insert_workspace_onto_monitor",
            );
            self.active_monitor_idx = target_idx;
        }

        activate
    }

    pub fn set_fullscreen(&mut self, id: &W::Id, is_fullscreen: bool) {
        // Check if this is a request to unset the windowed fullscreen state.
        if !is_fullscreen {
            let mut handled = false;
            self.with_windows_mut(|window, _| {
                if window.id() == id && window.is_pending_windowed_fullscreen() {
                    window.request_windowed_fullscreen(false);
                    handled = true;
                }
            });
            if handled {
                return;
            }
        }

        if let Some(InteractiveMoveState::Moving(move_)) = &self.interactive_move {
            if move_.tile.window().id() == id {
                return;
            }
        }

        for ws in self.workspaces_mut() {
            if ws.has_window(id) {
                ws.set_fullscreen(id, is_fullscreen);
                return;
            }
        }
    }

    pub fn toggle_fullscreen(&mut self, id: &W::Id) {
        if let Some(InteractiveMoveState::Moving(move_)) = &self.interactive_move {
            if move_.tile.window().id() == id {
                return;
            }
        }

        for ws in self.workspaces_mut() {
            if ws.has_window(id) {
                ws.toggle_fullscreen(id);
                return;
            }
        }
    }

    pub fn toggle_windowed_fullscreen(&mut self, id: &W::Id) {
        let (_, window) = self.windows_all().find(|(_, win)| win.id() == id).expect(
            "toggle_windowed_fullscreen called with window id not present in the workspace pool",
        );
        if window.pending_sizing_mode().is_fullscreen() {
            // Remove the real fullscreen.
            for ws in self.workspaces_mut() {
                if ws.has_window(id) {
                    ws.set_fullscreen(id, false);
                    break;
                }
            }
        }

        // Walk the full pool (not just the active activity) so the windowed-fullscreen
        // flip reaches a window left behind on a dormant activity's workspace.
        // This will switch is_pending_fullscreen() to false right away.
        self.with_windows_all_mut(|window, _| {
            if window.id() == id {
                window.request_windowed_fullscreen(!window.is_pending_windowed_fullscreen());
            }
        });
    }

    pub fn set_maximized(&mut self, id: &W::Id, maximize: bool) {
        if let Some(InteractiveMoveState::Moving(move_)) = &self.interactive_move {
            if move_.tile.window().id() == id {
                return;
            }
        }

        for ws in self.workspaces_mut() {
            if ws.has_window(id) {
                ws.set_maximized(id, maximize);
                return;
            }
        }
    }

    pub fn toggle_maximized(&mut self, id: &W::Id) {
        if let Some(InteractiveMoveState::Moving(move_)) = &self.interactive_move {
            if move_.tile.window().id() == id {
                return;
            }
        }

        for ws in self.workspaces_mut() {
            if ws.has_window(id) {
                ws.toggle_maximized(id);
                return;
            }
        }
    }

    pub fn workspace_switch_gesture_begin(&mut self, output: &Output, is_touchpad: bool) {
        assert!(
            !self.monitors.is_empty(),
            "workspace_switch_gesture_begin requires at least one connected monitor",
        );
        let views_map = self.activities.active_mut().views_mut();
        let monitors = &mut self.monitors;

        for monitor in monitors {
            let view = views_map
                .get_mut(&OutputId::new(&monitor.output))
                .expect("connected output must have a view in the active activity");
            // Cancel the gesture on other outputs.
            if &monitor.output != output {
                monitor.workspace_switch_gesture_end(view, None);
                continue;
            }

            monitor.workspace_switch_gesture_begin(view, is_touchpad);
        }
    }

    pub fn workspace_switch_gesture_update(
        &mut self,
        delta_y: f64,
        timestamp: Duration,
        is_touchpad: bool,
    ) -> Option<Option<Output>> {
        if self.monitors.is_empty() {
            return None;
        }
        let views_map = self.activities.active().views();
        let monitors = &mut self.monitors;

        for monitor in monitors {
            let view = views_map
                .get(&OutputId::new(&monitor.output))
                .expect("connected output must have a view in the active activity");
            if let Some(refresh) =
                monitor.workspace_switch_gesture_update(view, delta_y, timestamp, is_touchpad)
            {
                if refresh {
                    return Some(Some(monitor.output.clone()));
                } else {
                    return Some(None);
                }
            }
        }

        None
    }

    pub fn workspace_switch_gesture_end(&mut self, is_touchpad: Option<bool>) -> Option<Output> {
        if self.monitors.is_empty() {
            return None;
        }
        let views_map = self.activities.active_mut().views_mut();
        let monitors = &mut self.monitors;

        for monitor in monitors {
            let view = views_map
                .get_mut(&OutputId::new(&monitor.output))
                .expect("connected output must have a view in the active activity");
            if monitor.workspace_switch_gesture_end(view, is_touchpad) {
                return Some(monitor.output.clone());
            }
        }

        None
    }

    pub fn view_offset_gesture_begin(
        &mut self,
        output: &Output,
        workspace_idx: Option<usize>,
        is_touchpad: bool,
    ) {
        assert!(
            !self.monitors.is_empty(),
            "view_offset_gesture_begin requires at least one connected monitor",
        );
        let views_map = self.activities.active().views();
        let monitors = &mut self.monitors;

        let pool = &mut self.workspaces;
        for monitor in monitors {
            let view = views_map
                .get(&OutputId::new(&monitor.output))
                .expect("connected output must have a view in the active activity");
            for (idx, id) in view.ids().to_vec().into_iter().enumerate() {
                let ws = pool
                    .get_mut(&id)
                    .expect("workspace id must be a key in the pool");
                // Cancel the gesture on other workspaces.
                if &monitor.output != output
                    || idx != workspace_idx.unwrap_or(view.active_position())
                {
                    ws.view_offset_gesture_end(None);
                    continue;
                }

                ws.view_offset_gesture_begin(is_touchpad);
            }
        }
    }

    pub fn view_offset_gesture_update(
        &mut self,
        delta_x: f64,
        timestamp: Duration,
        is_touchpad: bool,
    ) -> Option<Option<Output>> {
        let zoom = self.overview_zoom();
        let delta_x = delta_x / zoom;

        let views_map = self.activities.active().views();
        let pool = &mut self.workspaces;
        if self.monitors.is_empty() {
            return None;
        }
        let monitors = &mut self.monitors;

        for monitor in monitors {
            let view = views_map
                .get(&OutputId::new(&monitor.output))
                .expect("connected output must have a view in the active activity");
            for id in view.ids().to_vec() {
                let ws = pool
                    .get_mut(&id)
                    .expect("workspace id must be a key in the pool");
                if let Some(refresh) =
                    ws.view_offset_gesture_update(delta_x, timestamp, is_touchpad)
                {
                    if refresh {
                        return Some(Some(monitor.output.clone()));
                    } else {
                        return Some(None);
                    }
                }
            }
        }

        None
    }

    pub fn view_offset_gesture_end(&mut self, is_touchpad: Option<bool>) -> Option<Output> {
        let views_map = self.activities.active().views();
        let pool = &mut self.workspaces;
        if self.monitors.is_empty() {
            return None;
        }
        let monitors = &mut self.monitors;

        for monitor in monitors {
            let view = views_map
                .get(&OutputId::new(&monitor.output))
                .expect("connected output must have a view in the active activity");
            for id in view.ids().to_vec() {
                let ws = pool
                    .get_mut(&id)
                    .expect("workspace id must be a key in the pool");
                if ws.view_offset_gesture_end(is_touchpad) {
                    return Some(monitor.output.clone());
                }
            }
        }

        None
    }

    pub fn overview_gesture_begin(&mut self) {
        self.overview_open = true;

        self.snap_all_activity_switches();

        let value = self.overview_progress.take().map_or(0., |p| p.value());
        let gesture = OverviewGesture {
            tracker: SwipeTracker::new(),
            start: value,
            value,
        };
        self.overview_progress = Some(OverviewProgress::Gesture(gesture));

        self.set_monitors_overview_state();
    }

    pub fn overview_gesture_update(&mut self, delta_y: f64, timestamp: Duration) -> Option<bool> {
        let Some(OverviewProgress::Gesture(gesture)) = &mut self.overview_progress else {
            return None;
        };

        gesture.tracker.push(delta_y, timestamp);

        let total_height = OVERVIEW_GESTURE_MOVEMENT;
        let pos = gesture.tracker.pos() / total_height;
        let new_value = gesture.start + pos;
        let new_value = OVERVIEW_GESTURE_RUBBER_BAND.clamp(0., 1., new_value);

        if gesture.value == new_value {
            return Some(false);
        }

        gesture.value = new_value;
        self.set_monitors_overview_state();

        Some(true)
    }

    pub fn overview_gesture_end(&mut self) -> bool {
        let Some(OverviewProgress::Gesture(gesture)) = &mut self.overview_progress else {
            return false;
        };

        // Take into account any idle time between the last event and now.
        let now = self.clock.now_unadjusted();
        gesture.tracker.push(0., now);

        let total_height = OVERVIEW_GESTURE_MOVEMENT;

        let mut velocity = gesture.tracker.velocity() / total_height;
        let current_pos = gesture.tracker.pos() / total_height;
        let pos = gesture.tracker.projected_end_pos() / total_height;

        let new_value = gesture.start + pos;
        let new_value = new_value.clamp(0., 1.).round();

        velocity *=
            OVERVIEW_GESTURE_RUBBER_BAND.clamp_derivative(0., 1., gesture.start + current_pos);

        self.overview_open = new_value == 1.;
        self.overview_progress = Some(OverviewProgress::Animation(Animation::new(
            self.clock.clone(),
            gesture.value,
            new_value,
            velocity,
            self.options.animations.overview_open_close.0,
        )));

        self.set_monitors_overview_state();

        true
    }

    pub fn interactive_move_begin(
        &mut self,
        window_id: W::Id,
        output: &Output,
        start_pos_within_output: Point<f64, Logical>,
    ) -> bool {
        if self.interactive_move.is_some() {
            return false;
        }

        let pool = &self.workspaces;
        let views = self.activities.active().views();
        let Some((mon, (ws, ws_geo))) = self.monitors().find_map(|mon| {
            let view = views
                .get(&OutputId::new(&mon.output))
                .expect("connected output must have a view in the active activity");
            let ctx = LayoutCtx::new(pool, view);
            mon.workspaces_with_render_geo(ctx)
                .find(|(ws, _)| ws.has_window(&window_id))
                .map(|rv| (mon, rv))
        }) else {
            return false;
        };

        if mon.output() != output {
            return false;
        }

        let zoom = mon.overview_zoom();

        let is_floating = ws.is_floating(&window_id);
        let (tile, tile_offset, _visible) = ws
            .tiles_with_render_positions()
            .find(|(tile, _, _)| tile.window().id() == &window_id)
            .unwrap();
        let window_offset = tile.window_loc();

        let tile_pos = ws_geo.loc + tile_offset.upscale(zoom);

        let pointer_offset_within_window =
            start_pos_within_output - tile_pos - window_offset.upscale(zoom);
        let window_size = tile.window_size().upscale(zoom);
        let pointer_ratio_within_window = (
            f64::clamp(pointer_offset_within_window.x / window_size.w, 0., 1.),
            f64::clamp(pointer_offset_within_window.y / window_size.h, 0., 1.),
        );

        self.interactive_move = Some(InteractiveMoveState::Starting {
            window_id,
            pointer_delta: Point::from((0., 0.)),
            pointer_ratio_within_window,
        });

        let views_map = self.activities.active().views();
        for mon in self.monitors.iter_mut() {
            let view = views_map
                .get(&OutputId::new(&mon.output))
                .expect("connected output must have a view in the active activity");
            mon.dnd_scroll_gesture_begin(view);
        }

        // Lock the view for scrolling interactive move.
        if !is_floating {
            for ws in self.workspaces_mut() {
                ws.dnd_scroll_gesture_begin();
            }
        }

        true
    }

    pub fn interactive_move_update(
        &mut self,
        window: &W::Id,
        delta: Point<f64, Logical>,
        output: Output,
        pointer_pos_within_output: Point<f64, Logical>,
    ) -> bool {
        let Some(state) = self.interactive_move.take() else {
            return false;
        };

        match state {
            InteractiveMoveState::Starting {
                window_id,
                mut pointer_delta,
                pointer_ratio_within_window,
            } => {
                if window_id != *window {
                    self.interactive_move = Some(InteractiveMoveState::Starting {
                        window_id,
                        pointer_delta,
                        pointer_ratio_within_window,
                    });
                    return false;
                }

                let zoom = self.overview_zoom();
                let delta = delta.downscale(zoom);

                pointer_delta += delta;

                let (cx, cy) = (pointer_delta.x, pointer_delta.y);
                let sq_dist = cx * cx + cy * cy;

                let factor = RubberBand {
                    stiffness: 1.0,
                    limit: 0.5,
                }
                .band(sq_dist / INTERACTIVE_MOVE_START_THRESHOLD);

                let (is_floating, tile, workspace_config) = self
                    .workspaces_mut()
                    .find(|ws| ws.has_window(&window_id))
                    .map(|ws| {
                        let workspace_config = ws.layout_config().cloned().map(|c| (ws.id(), c));
                        (
                            ws.is_floating(&window_id),
                            ws.tiles_mut()
                                .find(|tile| *tile.window().id() == window_id)
                                .unwrap(),
                            workspace_config,
                        )
                    })
                    .unwrap();
                tile.interactive_move_offset = pointer_delta.upscale(factor);

                // Put it back to be able to easily return.
                self.interactive_move = Some(InteractiveMoveState::Starting {
                    window_id: window_id.clone(),
                    pointer_delta,
                    pointer_ratio_within_window,
                });

                if !is_floating && sq_dist < INTERACTIVE_MOVE_START_THRESHOLD {
                    return true;
                }

                let output_config = self
                    .monitors()
                    .find(|mon| mon.output() == &output)
                    .and_then(|mon| mon.layout_config().cloned());

                // If the pointer is currently on the window's own output, then we can animate the
                // window movement from its current (rubberbanded and possibly moved away) position
                // to the pointer. Otherwise, we just teleport it as the layout code is not aware
                // of monitor positions.
                //
                // FIXME: when and if the layout code knows about monitor positions, this will be
                // potentially animatable.
                let pool = &self.workspaces;
                let views = self.activities.active().views();
                let mut tile_pos = None;
                if let Some((mon, (ws, ws_geo))) = self.monitors().find_map(|mon| {
                    let view = views
                        .get(&OutputId::new(&mon.output))
                        .expect("connected output must have a view in the active activity");
                    let ctx = LayoutCtx::new(pool, view);
                    mon.workspaces_with_render_geo(ctx)
                        .find(|(ws, _)| ws.has_window(window))
                        .map(|rv| (mon, rv))
                }) {
                    if mon.output() == &output {
                        let (_, tile_offset, _) = ws
                            .tiles_with_render_positions()
                            .find(|(tile, _, _)| tile.window().id() == window)
                            .unwrap();

                        let zoom = mon.overview_zoom();
                        tile_pos = Some((ws_geo.loc + tile_offset.upscale(zoom), zoom));
                    }
                }

                // Clear it before calling remove_window() to avoid running interactive_move_end()
                // in the middle of interactive_move_update() and the confusion that causes.
                self.interactive_move = None;

                // Unset fullscreen before removing the tile. This will restore its size properly,
                // and move it to floating if needed, so we don't have to deal with that here.
                let ws = self
                    .workspaces_mut()
                    .find(|ws| ws.has_window(&window_id))
                    .unwrap();
                ws.set_fullscreen(window, false);
                ws.set_maximized(window, false);

                let RemovedTile {
                    mut tile,
                    width,
                    is_full_width,
                    is_floating,
                } = self
                    .remove_window_inner(window, Transaction::new())
                    .unwrap();

                tile.stop_move_animations();
                tile.interactive_move_offset = Point::from((0., 0.));
                tile.window().output_enter(&output);
                tile.window().set_preferred_scale_transform(
                    output.current_scale(),
                    output.current_transform(),
                );

                let view_size = output_size(&output);
                let scale = output.current_scale().fractional_scale();
                let options = Options::clone(&self.options)
                    .with_merged_layout(output_config.as_ref())
                    .with_merged_layout(workspace_config.as_ref().map(|(_, c)| c))
                    .adjusted_for_scale(scale);
                tile.update_config(view_size, scale, Rc::new(options));

                if is_floating {
                    // Unlock the view in case we locked it moving a fullscreen window that is
                    // going to unfullscreen to floating.
                    for ws in self.workspaces_mut() {
                        ws.dnd_scroll_gesture_end();
                    }
                } else {
                    // Animate to semitransparent.
                    tile.animate_alpha(
                        1.,
                        INTERACTIVE_MOVE_ALPHA,
                        self.options.animations.window_movement.0,
                    );
                    tile.hold_alpha_animation_after_done();
                }

                let mut data = InteractiveMoveData {
                    tile,
                    output,
                    pointer_pos_within_output,
                    width,
                    is_full_width,
                    is_floating,
                    pointer_ratio_within_window,
                    output_config,
                    workspace_config,
                };

                if let Some((tile_pos, zoom)) = tile_pos {
                    let new_tile_pos = data.tile_render_location(zoom);
                    data.tile
                        .animate_move_from((tile_pos - new_tile_pos).downscale(zoom));
                }

                self.interactive_move = Some(InteractiveMoveState::Moving(data));
            }
            InteractiveMoveState::Moving(mut move_) => {
                if window != move_.tile.window().id() {
                    self.interactive_move = Some(InteractiveMoveState::Moving(move_));
                    return false;
                }

                let mut ws_id = None;
                if let Some(mon) = self.monitor_for_output(&output) {
                    let ctx = self.ctx_for(mon);
                    let (insert_ws, _) = mon.insert_position(ctx, move_.pointer_pos_within_output);
                    if let InsertWorkspace::Existing(id) = insert_ws {
                        ws_id = Some(id);
                    }
                }

                // If moved over a different workspace, reset the config override.
                let mut update_config = false;
                if let Some((id, _)) = &move_.workspace_config {
                    if Some(*id) != ws_id {
                        move_.workspace_config = None;
                        update_config = true;
                    }
                }

                if output != move_.output {
                    move_.tile.window().output_leave(&move_.output);
                    move_.tile.window().output_enter(&output);
                    move_.tile.window().set_preferred_scale_transform(
                        output.current_scale(),
                        output.current_transform(),
                    );
                    move_.output = output.clone();
                    self.focus_output(&output);

                    move_.output_config = self
                        .monitor_for_output(&output)
                        .and_then(|mon| mon.layout_config().cloned());

                    update_config = true;
                }

                if update_config {
                    let view_size = output_size(&output);
                    let scale = output.current_scale().fractional_scale();
                    let options = Options::clone(&self.options)
                        .with_merged_layout(move_.output_config.as_ref())
                        .with_merged_layout(move_.workspace_config.as_ref().map(|(_, c)| c))
                        .adjusted_for_scale(scale);
                    move_.tile.update_config(view_size, scale, Rc::new(options));
                }

                move_.pointer_pos_within_output = pointer_pos_within_output;

                self.interactive_move = Some(InteractiveMoveState::Moving(move_));
            }
        }

        true
    }

    pub fn interactive_move_end(&mut self, window: &W::Id) {
        let Some(move_) = &self.interactive_move else {
            return;
        };

        let move_ = match move_ {
            InteractiveMoveState::Starting { window_id, .. } => {
                if window_id != window {
                    return;
                }

                let Some(InteractiveMoveState::Starting { window_id, .. }) =
                    self.interactive_move.take()
                else {
                    unreachable!()
                };

                let views_map = self.activities.active_mut().views_mut();
                for mon in self.monitors.iter_mut() {
                    let view = views_map
                        .get_mut(&OutputId::new(&mon.output))
                        .expect("connected output must have a view in the active activity");
                    mon.dnd_scroll_gesture_end(view);
                }

                for ws in self.workspaces_mut() {
                    if let Some(tile) = ws.tiles_mut().find(|tile| *tile.window().id() == window_id)
                    {
                        let offset = tile.interactive_move_offset;
                        tile.interactive_move_offset = Point::from((0., 0.));
                        tile.animate_move_from(offset);
                    }

                    // Unlock the view on the workspaces, but if the moved window was active,
                    // preserve that.
                    let moved_tile_was_active =
                        ws.active_window().is_some_and(|win| *win.id() == window_id);

                    ws.dnd_scroll_gesture_end();

                    if moved_tile_was_active {
                        ws.activate_window(&window_id);
                    }
                }

                return;
            }
            InteractiveMoveState::Moving(move_) => move_,
        };

        if window != move_.tile.window().id() {
            return;
        }

        let Some(InteractiveMoveState::Moving(mut move_)) = self.interactive_move.take() else {
            unreachable!()
        };

        let views_map = self.activities.active_mut().views_mut();
        for mon in self.monitors.iter_mut() {
            let view = views_map
                .get_mut(&OutputId::new(&mon.output))
                .expect("connected output must have a view in the active activity");
            mon.dnd_scroll_gesture_end(view);
        }

        // Unlock the view on the workspaces.
        if !move_.is_floating {
            for ws in self.workspaces_mut() {
                ws.dnd_scroll_gesture_end();
            }

            // Also animate the tile back to opaque.
            move_.tile.animate_alpha(
                INTERACTIVE_MOVE_ALPHA,
                1.,
                self.options.animations.window_movement.0,
            );
        }

        // Dragging in the overview shouldn't switch the workspace and so on.
        let allow_to_activate_workspace = !self.overview_open;

        // Pair the `output_enter` fired on the tile during drag start / cross-output update
        // (in `interactive_move_update`). Whichever branch below handles the drop will re-enter
        // the window against the destination output via `add_tile(Some(&mon.output), ...)`,
        // or leave it unbound when no monitors are connected — both cases require the
        // drag-tracked marker to be cleared first so we don't leak it when the destination
        // output differs (including the case where `move_.output` has just been disconnected).
        move_.tile.window().output_leave(&move_.output);

        let seed_activity = self.activities.active_id();
        let pool = &mut self.workspaces;
        if self.monitors.is_empty() {
            let workspaces = &mut self.disconnected_workspace_ids;
            if workspaces.is_empty() {
                let ws = Workspace::new_no_outputs(
                    HashSet::from([seed_activity]),
                    self.clock.clone(),
                    self.options.clone(),
                );
                let id = ws.id();
                assert!(pool.insert(id, ws).is_none(), "fresh id must be unique");
                workspaces.push(id);
            }
            let ws = pool
                .get_mut(&workspaces[0])
                .expect("id must be a key in the workspace pool");

            // No point in trying to use the pointer position without outputs.
            ws.add_tile(
                None,
                move_.tile,
                WorkspaceAddWindowTarget::Auto,
                ActivateWindow::Yes,
                move_.width,
                move_.is_full_width,
                move_.is_floating,
            );
            return;
        }

        // Pre-capture everything read from self.monitors: the monitors_pool_view_mut call below
        // takes &mut monitors and would NLL-conflict with a later re-read.
        let active_monitor_idx_val = self.active_monitor_idx;
        let target_output_matches = self.monitors.iter().any(|mon| mon.output == move_.output);
        let target_mon_idx = self
            .monitors
            .iter()
            .position(|mon| mon.output == move_.output)
            .unwrap_or(active_monitor_idx_val);
        let target_out = self.monitors[target_mon_idx].output_id();
        let (monitors, pool, view) = self.monitors_pool_view_mut(&target_out);

        let (mon_idx, insert_ws, position, offset, zoom) = if target_output_matches {
            let mon = &mut monitors[target_mon_idx];
            let zoom = mon.overview_zoom();

            let ctx = LayoutCtx::new(&*pool, view);
            let (insert_ws, geo) = mon.insert_position(ctx, move_.pointer_pos_within_output);
            let (position, offset) = match insert_ws {
                InsertWorkspace::Existing(ws_id) => {
                    let ws_idx = view.position_of(ws_id).unwrap();

                    let position = if move_.is_floating {
                        InsertPosition::Floating
                    } else {
                        let pos_within_workspace =
                            (move_.pointer_pos_within_output - geo.loc).downscale(zoom);
                        let ws = mon.workspace_at_mut(pool, view, ws_idx);
                        ws.scrolling_insert_position(pos_within_workspace)
                    };

                    (position, Some(geo.loc))
                }
                InsertWorkspace::NewAt(_) => {
                    let position = if move_.is_floating {
                        InsertPosition::Floating
                    } else {
                        InsertPosition::NewColumn(0)
                    };

                    (position, None)
                }
            };

            (target_mon_idx, insert_ws, position, offset, zoom)
        } else {
            let mon_idx = active_monitor_idx_val;
            let mon = &monitors[mon_idx];
            let zoom = mon.overview_zoom();
            // No point in trying to use the pointer position on the wrong output.
            let ws = mon.workspace_at(pool, view, 0);
            let ws_id = ws.id();
            let ws_geo = mon.workspaces_render_geo(view).next().unwrap();

            let position = if move_.is_floating {
                InsertPosition::Floating
            } else {
                ws.scrolling_insert_position(Point::from((0., 0.)))
            };

            let insert_ws = InsertWorkspace::Existing(ws_id);
            (mon_idx, insert_ws, position, Some(ws_geo.loc), zoom)
        };

        let win_id = move_.tile.window().id().clone();
        let tile_render_loc = move_.tile_render_location(zoom);

        let ws_idx = match insert_ws {
            InsertWorkspace::Existing(ws_id) => view.position_of(ws_id).unwrap(),
            InsertWorkspace::NewAt(ws_idx) => {
                let ewaf = monitors[mon_idx].options.layout.empty_workspace_above_first;
                match Self::resolve_insert_target(ewaf, ws_idx, view.len()) {
                    BookendResolution::ReuseTop => 0,
                    BookendResolution::ReuseTrailing => view.len() - 1,
                    BookendResolution::InsertAt(idx) => {
                        Self::add_workspace_at_on(
                            monitors,
                            pool,
                            view,
                            mon_idx,
                            idx,
                            seed_activity,
                        );
                        idx
                    }
                }
            }
        };

        match position {
            InsertPosition::NewColumn(column_idx) => {
                let ws_id = view.ids()[ws_idx];
                Self::add_tile_on(
                    monitors,
                    pool,
                    view,
                    mon_idx,
                    move_.tile,
                    MonitorAddWindowTarget::Workspace {
                        id: ws_id,
                        column_idx: Some(column_idx),
                    },
                    ActivateWindow::Yes,
                    allow_to_activate_workspace,
                    move_.width,
                    move_.is_full_width,
                    false,
                    seed_activity,
                );
            }
            InsertPosition::InColumn(column_idx, tile_idx) => {
                Self::add_tile_to_column_on(
                    monitors,
                    pool,
                    view,
                    mon_idx,
                    ws_idx,
                    column_idx,
                    Some(tile_idx),
                    move_.tile,
                    true,
                    allow_to_activate_workspace,
                );
            }
            InsertPosition::Floating => {
                let tile_render_loc = move_.tile_render_location(zoom);

                let mut tile = move_.tile;
                tile.floating_pos = None;

                match insert_ws {
                    InsertWorkspace::Existing(_) => {
                        if let Some(offset) = offset {
                            let pos = (tile_render_loc - offset).downscale(zoom);
                            let pos = monitors[mon_idx]
                                .workspace_at(pool, view, ws_idx)
                                .floating_logical_to_size_frac(pos);
                            tile.floating_pos = Some(pos);
                        } else {
                            error!(
                                "offset unset for inserting a floating tile \
                                 to existing workspace"
                            );
                        }
                    }
                    InsertWorkspace::NewAt(_) => {
                        // When putting a floating tile on a new workspace, we don't really
                        // have a good pre-existing position.
                    }
                }

                // Set the floating size so it takes into account any window resizing that
                // took place during the move.
                if let Some(size) = tile.window().expected_size() {
                    tile.floating_window_size = Some(size);
                }

                let ws_id = view.ids()[ws_idx];
                Self::add_tile_on(
                    monitors,
                    pool,
                    view,
                    mon_idx,
                    tile,
                    MonitorAddWindowTarget::Workspace {
                        id: ws_id,
                        column_idx: None,
                    },
                    ActivateWindow::Yes,
                    allow_to_activate_workspace,
                    move_.width,
                    move_.is_full_width,
                    true,
                    seed_activity,
                );
            }
        }

        // needed because empty_workspace_above_first could have modified the idx.
        // Find the (id, geo) pair first, then borrow `&mut Workspace<W>` once via the
        // pool — the iterator can't escape a mutable borrow through the closure.
        let geo_pairs: Vec<_> = monitors[mon_idx]
            .workspaces_with_render_geo_ids(view, false)
            .collect();
        let mut found_ws_geo = None;
        for (id, geo) in geo_pairs {
            if pool
                .get(&id)
                .expect("workspace id must be a key in the pool")
                .has_window(&win_id)
            {
                found_ws_geo = Some((id, geo));
                break;
            }
        }
        let (found_id, ws_geo) = found_ws_geo.unwrap();
        let ws = pool
            .get_mut(&found_id)
            .expect("workspace id must be a key in the pool");
        let (tile, tile_offset) = ws
            .tiles_with_render_positions_mut(false)
            .find(|(tile, _)| tile.window().id() == &win_id)
            .unwrap();
        let new_tile_render_loc = ws_geo.loc + tile_offset.upscale(zoom);

        tile.animate_move_from((tile_render_loc - new_tile_render_loc).downscale(zoom));
    }

    pub fn interactive_move_is_moving_above_output(&self, output: &Output) -> bool {
        let Some(InteractiveMoveState::Moving(move_)) = &self.interactive_move else {
            return false;
        };

        move_.output == *output
    }

    pub fn dnd_update(&mut self, output: Output, pointer_pos_within_output: Point<f64, Logical>) {
        let begin_gesture = self.dnd.is_none();

        self.dnd = Some(DndData {
            output,
            pointer_pos_within_output,
            hold: None,
        });

        if begin_gesture {
            let views_map = self.activities.active().views();
            for mon in self.monitors.iter_mut() {
                let view = views_map
                    .get(&OutputId::new(&mon.output))
                    .expect("connected output must have a view in the active activity");
                mon.dnd_scroll_gesture_begin(view);
            }

            for ws in self.workspaces_mut() {
                ws.dnd_scroll_gesture_begin();
            }
        }
    }

    pub fn dnd_end(&mut self) {
        if self.dnd.is_none() {
            return;
        }

        self.dnd = None;

        let views_map = self.activities.active_mut().views_mut();
        for mon in self.monitors.iter_mut() {
            let view = views_map
                .get_mut(&OutputId::new(&mon.output))
                .expect("connected output must have a view in the active activity");
            mon.dnd_scroll_gesture_end(view);
        }

        for ws in self.workspaces_mut() {
            ws.dnd_scroll_gesture_end();
        }
    }

    pub fn interactive_resize_begin(&mut self, window: W::Id, edges: ResizeEdge) -> bool {
        let views_map = self.activities.active().views();
        let pool = &mut self.workspaces;
        for mon in &mut self.monitors {
            let view = views_map
                .get(&OutputId::new(&mon.output))
                .expect("connected output must have a view in the active activity");
            for id in view.ids() {
                let ws = pool
                    .get_mut(id)
                    .expect("workspace id must be a key in the pool");
                if ws.has_window(&window) {
                    return ws.interactive_resize_begin(window, edges);
                }
            }
        }
        for id in &self.disconnected_workspace_ids {
            let ws = self
                .workspaces
                .get_mut(id)
                .expect("workspace id must be a key in the pool");
            if ws.has_window(&window) {
                return ws.interactive_resize_begin(window, edges);
            }
        }

        false
    }

    pub fn interactive_resize_update(
        &mut self,
        window: &W::Id,
        delta: Point<f64, Logical>,
    ) -> bool {
        if let Some(InteractiveMoveState::Moving(move_)) = &self.interactive_move {
            if move_.tile.window().id() == window {
                return false;
            }
        }

        let views_map = self.activities.active().views();
        let pool = &mut self.workspaces;
        for mon in &mut self.monitors {
            let view = views_map
                .get(&OutputId::new(&mon.output))
                .expect("connected output must have a view in the active activity");
            for id in view.ids() {
                let ws = pool
                    .get_mut(id)
                    .expect("workspace id must be a key in the pool");
                if ws.has_window(window) {
                    return ws.interactive_resize_update(window, delta);
                }
            }
        }
        for id in &self.disconnected_workspace_ids {
            let ws = self
                .workspaces
                .get_mut(id)
                .expect("workspace id must be a key in the pool");
            if ws.has_window(window) {
                return ws.interactive_resize_update(window, delta);
            }
        }

        false
    }

    pub fn interactive_resize_end(&mut self, window: &W::Id) {
        if let Some(InteractiveMoveState::Moving(move_)) = &self.interactive_move {
            if move_.tile.window().id() == window {
                return;
            }
        }

        let views_map = self.activities.active().views();
        let pool = &mut self.workspaces;
        for mon in &mut self.monitors {
            let view = views_map
                .get(&OutputId::new(&mon.output))
                .expect("connected output must have a view in the active activity");
            for id in view.ids() {
                let ws = pool
                    .get_mut(id)
                    .expect("workspace id must be a key in the pool");
                if ws.has_window(window) {
                    ws.interactive_resize_end(Some(window));
                    return;
                }
            }
        }
        for id in &self.disconnected_workspace_ids {
            let ws = self
                .workspaces
                .get_mut(id)
                .expect("workspace id must be a key in the pool");
            if ws.has_window(window) {
                ws.interactive_resize_end(Some(window));
                return;
            }
        }
    }

    /// Reorder the active workspace down within the active activity's view, minting a
    /// fresh trailing-empty bookend in the active view (and a leading-empty under EWAF)
    /// when the swap pushes the active workspace into a bookend slot.
    ///
    /// No sweep call is required: this entry point and its `_up` / `_to_idx`
    /// siblings mutate **only the active view's order** via `WorkspaceView::swap` /
    /// `move_within` and mint bookends in the active view via
    /// `Self::add_workspace_{bottom,top}_on`. Workspace identities are preserved and no
    /// shared workspace's content (tiles, name, activity-set) changes — so a dormant view
    /// that shared one of the reordered workspaces at a bookend slot cannot break its
    /// per-view bookend invariant via these paths. Contrast with the column-add public
    /// entry points (`add_column_by_idx`, `move_column_to_workspace*`) which widen the
    /// receiving workspace's content and thus require the sweep.
    ///
    /// Workspaces may still be destroyed downstream via `clean_up_workspaces_on` →
    /// `destroy_workspaces_cross_activity`, but that pass's shared-id skip (only workspaces
    /// with `activities().len() == 1` are destroyed) ensures no dormant view's content is
    /// mutated — weakening that skip would also need this entry point to grow a sweep call.
    pub fn move_workspace_down(&mut self) {
        if self.monitors.is_empty() {
            return;
        }
        let active_monitor_idx = self.active_monitor_idx;
        let mon_out = self.monitors[active_monitor_idx].output_id();
        let seed_activity = self.activities.active_id();
        let ids_to_destroy = {
            let (monitors, pool, view) = self.monitors_pool_view_mut(&mon_out);
            Self::move_workspace_down_on(monitors, pool, view, active_monitor_idx, seed_activity)
        };
        Self::destroy_workspaces_cross_activity(
            &mut self.activities,
            &mut self.workspaces,
            ids_to_destroy,
        );
    }

    /// See [`Self::move_workspace_down`] for the no-shared-content-mutation invariant
    /// that lets this entry point skip the sweep.
    pub fn move_workspace_up(&mut self) {
        if self.monitors.is_empty() {
            return;
        }
        let active_monitor_idx = self.active_monitor_idx;
        let mon_out = self.monitors[active_monitor_idx].output_id();
        let seed_activity = self.activities.active_id();
        let ids_to_destroy = {
            let (monitors, pool, view) = self.monitors_pool_view_mut(&mon_out);
            Self::move_workspace_up_on(monitors, pool, view, active_monitor_idx, seed_activity)
        };
        Self::destroy_workspaces_cross_activity(
            &mut self.activities,
            &mut self.workspaces,
            ids_to_destroy,
        );
    }

    /// See [`Self::move_workspace_down`] for the no-shared-content-mutation invariant
    /// that lets this entry point skip the sweep.
    pub fn move_workspace_to_idx(
        &mut self,
        reference: Option<(Option<Output>, usize)>,
        new_idx: usize,
    ) {
        if self.monitors.is_empty() {
            return;
        }

        let (mon_idx, old_idx) = if let Some((output, old_idx)) = reference {
            if let Some(output) = output {
                let Some(mi) = self.monitors.iter().position(|m| m.output == output) else {
                    return;
                };
                (mi, old_idx)
            } else {
                (self.active_monitor_idx, old_idx)
            }
        } else {
            let mon_idx = self.active_monitor_idx;
            let mon_out = self.monitors[mon_idx].output_id();
            (mon_idx, self.active_view(&mon_out).active_position())
        };

        let mon_out = self.monitors[mon_idx].output_id();
        let seed_activity = self.activities.active_id();
        let ids_to_destroy = {
            let (monitors, pool, view) = self.monitors_pool_view_mut(&mon_out);
            Self::move_workspace_to_idx_on(
                monitors,
                pool,
                view,
                mon_idx,
                old_idx,
                new_idx,
                seed_activity,
            )
        };
        Self::destroy_workspaces_cross_activity(
            &mut self.activities,
            &mut self.workspaces,
            ids_to_destroy,
        );
    }

    /// Assign `name` to a workspace selected by `reference` (or the active
    /// workspace when `reference` is `None`).
    ///
    /// ## Bookend invariant
    ///
    /// Per-view bookend rule: every connected output's per-activity view ends
    /// in a workspace whose `name.is_none() && !has_windows()`, and — under
    /// `empty_workspace_above_first` — also begins with one. Naming a
    /// workspace that currently sits at a bookend slot of any view violates
    /// that rule, so a fresh empty must be minted in its place — but only
    /// when the freshly-named workspace was actually at a bookend slot before
    /// the rename. The active activity's view of the workspace's hosting
    /// monitor is patched inline via `add_workspace_top_on` /
    /// `add_workspace_bottom_on` (`monitors_pool_view_mut(&out)` returns the
    /// active activity's view for that output); every dormant activity's view
    /// of that same output is patched by the sweep.
    ///
    /// The monitor lookup itself fans out across every activity's views (not
    /// just the active activity's), so a workspace held only in a dormant
    /// activity's view of a connected monitor — reachable today only via
    /// `WorkspaceReference::Id` — locates its hosting monitor and patches the
    /// dormant view. In that dormant-only case the active view does not
    /// contain `wsid`, so the active-view bookend mint conditional
    /// (`add_top` / `add_bottom`) is unconditionally false and the inline
    /// `add_workspace_{top,bottom}_on` calls no-op; the sweep then carries the
    /// full bookend repair.
    ///
    /// Workspaces reachable only via `Layout.disconnected_workspace_ids` or
    /// held only at a bookend slot of a dormant activity's view keyed by an
    /// `OutputId` no connected monitor matches still no-op here; their bookend
    /// repair has no live `mon_idx` for the sweep to run against.
    ///
    /// ## Asymmetry with `unset_workspace_name`
    ///
    /// `set_workspace_name` *adds* `name.is_some()` at a slot whose contract
    /// requires `name.is_none()` — so it can break the bookend rule and
    /// requires the mint described above. `unset_workspace_name` clears
    /// `name` to `None`, moving the workspace toward — never away from — the
    /// bookend predicate's `name.is_none()` requirement. The bookend slot
    /// identity is also preserved: `clean_up_workspaces_on`'s prune range
    /// excludes the trailing slot (and the leading slot under EWAF), so the
    /// unset path needs no bookend wiring.
    pub fn set_workspace_name(&mut self, name: String, reference: Option<WorkspaceReference>) {
        // ignore the request if the name is already used by another workspace
        if self.find_workspace_by_name(&name).is_some() {
            return;
        }

        let ws = if let Some(reference) = reference {
            self.find_workspace_by_ref(reference)
        } else {
            self.active_workspace_mut()
        };
        let Some(ws) = ws else {
            return;
        };

        ws.name.replace(name);

        let wsid = ws.id();

        // Locate the workspace's actual monitor — some activity's view on
        // some connected output must contain `wsid`. Using `active_monitor_idx`
        // here would silently no-op for a `WorkspaceReference::Id` that
        // targets a workspace currently shown on a different output. The
        // fan-out across every activity's views (not just the active
        // activity's) also catches the dormant-only case: a workspace held
        // only in a dormant activity's view of a connected monitor still
        // locates its hosting monitor and reaches the per-view bookend repair
        // below.
        //
        // Returns `None` when `wsid` belongs to neither the connected-monitor
        // set nor any connected monitor's activity views. Two boundary cases
        // share this no-live-mon_idx structural blocker:
        //
        // (a) Disconnected-pool entry: `wsid` reachable only via
        //     `Layout.disconnected_workspace_ids`. Remediates on reconnect
        //     via `Monitor::new`'s `with_merged_layout` materializer.
        //
        // (b) Stale-keyed dormant view: `wsid` held only at a bookend slot of
        //     a dormant activity's view whose `OutputId` key matches no
        //     connected monitor (output removed after the view was materialized).
        //     Does not remediate on reconnect of the *original* output unless
        //     that specific output returns; remains the deferred boundary,
        //     requiring reconciliation of per-monitor EWAF override divergence
        //     between layout-root validation and the reconnect materializer.
        let activities = &self.activities;
        let mon_idx_opt = self.monitors.iter().position(|mon| {
            let out_id = mon.output_id();
            activities.iter().any(|a| {
                a.views()
                    .get(&out_id)
                    .is_some_and(|v| v.ids().contains(&wsid))
            })
        });

        if let Some(mon_idx) = mon_idx_opt {
            let monitor = &self.monitors[mon_idx];
            let mon_out = monitor.output_id();
            let monitor_view = self.active_view(&mon_out);
            let add_top = monitor.options.layout.empty_workspace_above_first
                && monitor_view.ids().first().is_some_and(|id| *id == wsid);
            let add_bottom = monitor_view.ids().last().is_some_and(|id| *id == wsid);
            let seed_activity = self.activities.active_id();
            {
                let (monitors, pool, view) = self.monitors_pool_view_mut(&mon_out);
                if add_top {
                    Self::add_workspace_top_on(monitors, pool, view, mon_idx, seed_activity);
                }
                if add_bottom {
                    Self::add_workspace_bottom_on(monitors, pool, view, mon_idx, seed_activity);
                }
            }

            // End of split borrow. `add_workspace_{top,bottom}_on` patches the
            // active view's bookend slot inline; the sweep additionally repairs
            // every dormant activity that holds `wsid` at the same bookend slot
            // of its own view (redundantly re-examining the active view, which
            // is already conformant from the inline patch above).
            //
            // The `wsid` lookup anchors to the workspace's *actual* monitor
            // rather than `active_monitor_idx` and fans out across every
            // activity's views, so a `WorkspaceReference::Id` targeting a
            // workspace on another connected monitor, or held only in a
            // dormant activity's view, still reaches the correct `mon_idx` and
            // the sweep carries the repair.
            //
            // Two remaining silent-skip cases share the no-live-mon_idx
            // structural blocker and are handled by the `None` branch above:
            // (a) disconnected-pool entry — remediates on reconnect via
            //     `Monitor::new`; (b) stale-keyed dormant view (output
            //     removed after materialization, `OutputId` no longer matched
            //     by any connected monitor) — does not auto-remediate on
            //     reconnect of that output unless the same output returns.
            self.normalize_view_bookends();
        }
    }

    pub fn unset_workspace_name(&mut self, reference: Option<WorkspaceReference>) {
        let ws = if let Some(reference) = reference {
            self.find_workspace_by_ref(reference)
        } else {
            self.active_workspace_mut()
        };
        let Some(ws) = ws else {
            return;
        };
        let id = ws.id();

        self.unname_workspace_by_id(id);
    }

    pub fn set_monitors_overview_state(&mut self) {
        let views_map = self.activities.active().views();
        for mon in &mut self.monitors {
            let view = views_map
                .get(&OutputId::new(&mon.output))
                .expect("connected output must have a view in the active activity");
            mon.overview_open = self.overview_open;
            mon.set_overview_progress(view, self.overview_progress.as_ref());
        }
    }

    pub fn toggle_overview(&mut self) {
        self.overview_open = !self.overview_open;

        self.snap_all_activity_switches();

        let from = self.overview_progress.take().map_or(0., |p| p.value());
        let to = if self.overview_open { 1. } else { 0. };

        self.overview_progress = Some(OverviewProgress::Animation(Animation::new(
            self.clock.clone(),
            from,
            to,
            0.,
            self.options.animations.overview_open_close.0,
        )));

        self.set_monitors_overview_state();
    }

    pub fn open_overview(&mut self) -> bool {
        if self.overview_open {
            return false;
        }

        self.toggle_overview();
        true
    }

    pub fn close_overview(&mut self) -> bool {
        if !self.overview_open {
            return false;
        }

        self.toggle_overview();
        true
    }

    pub fn toggle_overview_to_workspace(&mut self, ws_idx: usize) {
        let config = self.options.animations.overview_open_close.0;
        if !self.monitors.is_empty() {
            let mon_idx = self.active_monitor_idx;
            let mon_out = self.monitors[mon_idx].output_id();
            let (monitors, _, view) = self.monitors_pool_view_mut(&mon_out);
            monitors[mon_idx].activate_workspace_with_anim_config(view, ws_idx, Some(config));
        }
        self.toggle_overview();
    }

    pub fn start_open_animation_for_window(&mut self, window: &W::Id) {
        if let Some(InteractiveMoveState::Moving(move_)) = &self.interactive_move {
            if move_.tile.window().id() == window {
                return;
            }
        }

        for ws in self.workspaces_mut() {
            if ws.start_open_animation(window) {
                return;
            }
        }
    }

    pub fn store_unmap_snapshot(
        &mut self,
        renderer: &mut GlesRenderer,
        xray: Option<&mut Xray>,
        xray_has_blocked_out_layers: bool,
        window: &W::Id,
    ) {
        let _span = tracy_client::span!("Layout::store_unmap_snapshot");

        let zoom = self.overview_zoom();

        if let Some(InteractiveMoveState::Moving(move_)) = &mut self.interactive_move {
            if move_.tile.window().id() == window {
                let pos_within_output = move_.tile_render_location(zoom);

                // Computation matches update_render_elements().
                let view_rect =
                    Rectangle::new(pos_within_output.upscale(-1.), output_size(&move_.output))
                        .downscale(zoom);
                move_.tile.update_render_elements(false, view_rect);

                move_.tile.store_unmap_snapshot_if_empty(
                    renderer,
                    xray,
                    xray_has_blocked_out_layers,
                    XrayPos::new(pos_within_output, zoom),
                );
                return;
            }
        }

        let views_map = self.activities.active().views();
        let pool = &mut self.workspaces;
        for mon in &mut self.monitors {
            let view = views_map
                .get(&OutputId::new(&mon.output))
                .expect("connected output must have a view in the active activity");
            for (id, geo) in mon.workspaces_with_render_geo_ids(view, false) {
                let ws = pool
                    .get_mut(&id)
                    .expect("workspace id must be a key in the pool");
                if ws.has_window(window) {
                    ws.store_unmap_snapshot_if_empty(
                        renderer,
                        xray,
                        xray_has_blocked_out_layers,
                        XrayPos::new(geo.loc, zoom),
                        window,
                    );
                    return;
                }
            }
        }
        for id in &self.disconnected_workspace_ids {
            let ws = pool
                .get_mut(id)
                .expect("workspace id must be a key in the pool");
            if ws.has_window(window) {
                ws.store_unmap_snapshot_if_empty(
                    renderer,
                    xray,
                    xray_has_blocked_out_layers,
                    XrayPos::default(),
                    window,
                );
                return;
            }
        }
    }

    pub fn clear_unmap_snapshot(&mut self, window: &W::Id) {
        if let Some(InteractiveMoveState::Moving(move_)) = &mut self.interactive_move {
            if move_.tile.window().id() == window {
                let _ = move_.tile.take_unmap_snapshot();
                return;
            }
        }

        let views_map = self.activities.active().views();
        let pool = &mut self.workspaces;
        for mon in &mut self.monitors {
            let view = views_map
                .get(&OutputId::new(&mon.output))
                .expect("connected output must have a view in the active activity");
            for id in view.ids() {
                let ws = pool
                    .get_mut(id)
                    .expect("workspace id must be a key in the pool");
                if ws.has_window(window) {
                    ws.clear_unmap_snapshot(window);
                    return;
                }
            }
        }
        for id in &self.disconnected_workspace_ids {
            let ws = self
                .workspaces
                .get_mut(id)
                .expect("workspace id must be a key in the pool");
            if ws.has_window(window) {
                ws.clear_unmap_snapshot(window);
                return;
            }
        }
    }

    pub fn start_close_animation_for_window(
        &mut self,
        renderer: &mut GlesRenderer,
        window: &W::Id,
        blocker: TransactionBlocker,
    ) {
        let _span = tracy_client::span!("Layout::start_close_animation_for_window");

        let zoom = self.overview_zoom();

        if let Some(InteractiveMoveState::Moving(move_)) = &mut self.interactive_move {
            if move_.tile.window().id() == window {
                let Some(snapshot) = move_.tile.take_unmap_snapshot() else {
                    return;
                };
                let tile_pos = move_.tile_render_location(zoom);
                let tile_size = move_.tile.tile_size();

                let output = move_.output.clone();
                let pointer_pos_within_output = move_.pointer_pos_within_output;
                let output_id = OutputId::new(&output);
                if !self.monitors.iter().any(|m| m.output == output) {
                    return;
                }
                let view = self
                    .activities
                    .active()
                    .views()
                    .get(&output_id)
                    .expect("connected output must have a view in the active activity");
                let pool = &mut self.workspaces;
                let mon = self
                    .monitors
                    .iter_mut()
                    .find(|m| m.output == output)
                    .expect("monitor for connected output must exist");
                let ctx = LayoutCtx::new(&*pool, view);
                let Some((ws_id, ws_geo)) = mon
                    .workspace_under(ctx, pointer_pos_within_output)
                    .map(|(ws, geo)| (ws.id(), geo))
                else {
                    return;
                };
                let ws = pool
                    .get_mut(&ws_id)
                    .expect("workspace id must be a key in the pool");

                let tile_pos = tile_pos - ws_geo.loc;
                ws.start_close_animation_for_tile(renderer, snapshot, tile_size, tile_pos, blocker);
                return;
            }
        }

        let views_map = self.activities.active().views();
        let pool = &mut self.workspaces;
        for mon in &mut self.monitors {
            let view = views_map
                .get(&OutputId::new(&mon.output))
                .expect("connected output must have a view in the active activity");
            for id in view.ids() {
                let ws = pool
                    .get_mut(id)
                    .expect("workspace id must be a key in the pool");
                if ws.has_window(window) {
                    ws.start_close_animation_for_window(renderer, window, blocker);
                    return;
                }
            }
        }
        for id in &self.disconnected_workspace_ids {
            let ws = self
                .workspaces
                .get_mut(id)
                .expect("workspace id must be a key in the pool");
            if ws.has_window(window) {
                ws.start_close_animation_for_window(renderer, window, blocker);
                return;
            }
        }
    }

    pub fn render_interactive_move_for_output<R: NiriRenderer>(
        &self,
        ctx: RenderCtx<R>,
        output: &Output,
        push: &mut dyn FnMut(RescaleRenderElement<TileRenderElement<R>>),
    ) {
        if self.update_render_elements_time != self.clock.now() {
            error!("clock moved between updating render elements and rendering");
        }

        let Some(InteractiveMoveState::Moving(move_)) = &self.interactive_move else {
            return;
        };

        if &move_.output != output {
            return;
        }

        let scale = Scale::from(move_.output.current_scale().fractional_scale());
        let zoom = self.overview_zoom();
        let pos_in_backdrop = move_.tile_render_location(zoom);
        let xray_pos = XrayPos::new(pos_in_backdrop, zoom);

        move_
            .tile
            .render(ctx, pos_in_backdrop, xray_pos, true, &mut |elem| {
                push(RescaleRenderElement::from_element(
                    elem,
                    pos_in_backdrop.to_physical_precise_round(scale),
                    zoom,
                ));
            });
    }

    /// Drop any reference, from an activity view or from
    /// `disconnected_workspace_ids`, to a [`WorkspaceId`] that is no longer a key
    /// in the pool — logging once per occurrence — and return early when the
    /// layout is already coherent (the overwhelmingly common case, allocation-free).
    ///
    /// `verify_invariants` asserts that the pool keys equal the union of every
    /// view plus the disconnected pool ("every view id must be in the pool — no
    /// zombies"), but that chain is `#[cfg(debug_assertions)]` and compiled out of
    /// release builds. So a desync introduced by a cull-path corner case — observed
    /// under a mass window teardown during an out-of-memory event — sits latent in a
    /// release session until the next view walk reaches its
    /// `.expect("workspace id must be a key in the pool")` and aborts the whole
    /// compositor. Pruning the dangling reference here downgrades that
    /// session-fatal abort to a self-correcting, logged glitch and keeps the
    /// session alive for live introspection. This contains the blast radius; it
    /// does not close the originating desync.
    ///
    /// Called at the top of the refresh cycle, before any view walk. Healing the
    /// reference means the warning fires once per occurrence rather than every
    /// frame, with no dedup bookkeeping.
    pub fn repair_view_pool_coherence(&mut self) {
        let _span = tracy_client::span!("Layout::repair_view_pool_coherence");

        let pool = &self.workspaces;
        let mut stale: HashSet<WorkspaceId> = HashSet::new();
        for activity in self.activities.iter() {
            for view in activity.views().values() {
                for id in view.ids() {
                    if !pool.contains_key(id) {
                        stale.insert(*id);
                    }
                }
            }
        }
        for id in &self.disconnected_workspace_ids {
            if !pool.contains_key(id) {
                stale.insert(*id);
            }
        }

        if stale.is_empty() {
            return;
        }

        for ws_id in &stale {
            warn!(
                "view↔pool divergence: workspace {ws_id:?} is referenced by an activity \
                 view or disconnected_workspace_ids but is absent from the workspace pool; \
                 pruning the dangling reference. A cull path desynced the layout — without \
                 this repair the next view walk would abort the session.",
            );
        }

        // Mirror `destroy_workspaces_cross_activity`'s view patching: a
        // `WorkspaceView` cannot be zero-length, so drop the whole entry when the
        // stale id is its sole occupant; otherwise shift it out and let `remove_at`
        // patch the view's `active` / `previous`.
        for ws_id in &stale {
            for activity in self.activities.iter_mut() {
                activity.views_mut().retain(|_output_id, view| {
                    let Some(pos) = view.position_of(*ws_id) else {
                        return true;
                    };
                    if view.len() == 1 {
                        return false;
                    }
                    view.remove_at(pos);
                    true
                });
            }
        }
        self.disconnected_workspace_ids
            .retain(|id| !stale.contains(id));
    }

    pub fn refresh(&mut self, is_active: bool) {
        let _span = tracy_client::span!("Layout::refresh");

        self.is_active = is_active;

        // Observe the focus for the bookmark state machine before touching any
        // monitor. Capture the focused id first (shared borrow) so the mutable
        // borrow of `self.bookmarks` does not overlap. This heals a stale walk
        // cursor and, under MRU order, promotes the focused bookmark to front;
        // a walk's own focus change is filtered out synchronously at commit.
        let focused = self.focus().map(|w| w.id().clone());
        let bookmark_order = self.options.bookmarks.order;
        self.bookmarks
            .observe_focus(focused.as_ref(), bookmark_order);

        let mut ongoing_scrolling_dnd = self.dnd.is_some().then_some(true);

        if let Some(InteractiveMoveState::Moving(move_)) = &mut self.interactive_move {
            let win = move_.tile.window_mut();

            win.set_active_in_column(true);
            win.set_floating(move_.is_floating);
            win.set_activated(true);

            win.set_interactive_resize(None);

            win.set_bounds(output_size(&move_.output).to_i32_round());

            win.send_pending_configure();
            win.refresh();

            ongoing_scrolling_dnd.get_or_insert(!move_.is_floating);
        } else if let Some(InteractiveMoveState::Starting { window_id, .. }) =
            &self.interactive_move
        {
            ongoing_scrolling_dnd.get_or_insert_with(|| {
                let (_, _, ws) = self
                    .workspaces()
                    .find(|(_, _, ws)| ws.has_window(window_id))
                    .unwrap();
                !ws.is_floating(window_id)
            });
        }

        let views_map = self.activities.active_mut().views_mut();
        let pool = &mut self.workspaces;
        let active_monitor_idx = self.active_monitor_idx;
        for (idx, mon) in self.monitors.iter_mut().enumerate() {
            let view = views_map
                .get_mut(&OutputId::new(&mon.output))
                .expect("connected output must have a view in the active activity");
            let is_active = self.is_active
                && idx == active_monitor_idx
                && !matches!(self.interactive_move, Some(InteractiveMoveState::Moving(_)));

            if ongoing_scrolling_dnd.is_some() && self.overview_open {
                // Begin the scroll on new monitors and when opening the overview.
                mon.dnd_scroll_gesture_begin(view);
            } else if !self.overview_open {
                mon.dnd_scroll_gesture_end(view);
            }

            let active_pos = view.active_position();
            for (ws_idx, id) in view.ids().to_vec().into_iter().enumerate() {
                let ws = pool
                    .get_mut(&id)
                    .expect("workspace id must be a key in the pool");
                let is_focused = is_active && ws_idx == active_pos;
                ws.refresh(is_active, is_focused);

                if let Some(is_scrolling) = ongoing_scrolling_dnd {
                    // Lock or unlock the view for scrolling interactive move.
                    if is_scrolling {
                        ws.dnd_scroll_gesture_begin();
                    } else {
                        ws.dnd_scroll_gesture_end();
                    }
                } else {
                    // Cancel the view offset gesture after workspace switches, moves, etc.
                    if !self.overview_open && ws_idx != active_pos {
                        ws.view_offset_gesture_end(None);
                    }
                }
            }
        }
        for id in &self.disconnected_workspace_ids {
            let ws = self
                .workspaces
                .get_mut(id)
                .expect("workspace id must be a key in the pool");
            ws.refresh(false, false);
            ws.view_offset_gesture_end(None);
        }

        // Cross-field invariants (pool keys ↔ per-view ordered ids, plus
        // monitor-level index bounds and every per-workspace/column/tile
        // invariant) are only encoded as runtime convention, not in the type
        // system. Run the full chain once per refresh tick so drift from any
        // mutation path trips an assert in debug builds immediately, rather
        // than surfacing later as a corrupt render or a mysterious panic deep
        // inside a read path. This fires on every input event, surface commit,
        // and frame callback in debug builds; the Tracy span below lets a
        // future session decide whether per-Nth gating is worth it. Release
        // builds (which disable `debug_assertions`) skip this entirely.
        #[cfg(debug_assertions)]
        {
            let _span = tracy_client::span!("Layout::verify_invariants");
            self.verify_invariants();
        }
    }

    /// Workspaces visible in the currently-active activity: for each connected
    /// monitor, the ordered view-id list is flattened into `(monitor, idx,
    /// workspace)` tuples, followed by disconnected-pool entries.
    ///
    /// This is the activity-filtered default — callers that iterate "all
    /// workspaces the user can see right now" use this. Consumers that must
    /// cross activity boundaries (IPC projections that advertise hidden
    /// workspaces, sticky-expansion, `FocusWindow { id }` against a window on
    /// a dormant activity) use [`Self::workspaces_all`] instead.
    ///
    /// `WindowMru::new` at `src/ui/mru.rs` is a key dependent on the
    /// activity-scope invariant: it iterates `layout.workspaces()` (and
    /// equivalently `layout.windows()`) to build the per-activity MRU list,
    /// ensuring windows on dormant activities are excluded. Cross-activity MRU
    /// is a deliberate future opt-in ( `MruScope::AllActivities`), not
    /// an accidental widen.
    pub fn workspaces(
        &self,
    ) -> impl Iterator<Item = (Option<&Monitor<W>>, usize, &Workspace<W>)> + '_ {
        let pool = &self.workspaces;
        let views = self.activities.active().views();
        let iter_monitors = self.monitors.iter().flat_map(move |mon| {
            let view = views
                .get(&OutputId::new(&mon.output))
                .expect("connected output must have a view in the active activity");
            view.ids().iter().enumerate().map(move |(idx, id)| {
                (
                    Some(mon),
                    idx,
                    pool.get(id)
                        .expect("workspace id must be a key in the pool"),
                )
            })
        });
        let iter_disconnected =
            self.disconnected_workspace_ids
                .iter()
                .enumerate()
                .map(move |(idx, id)| {
                    (
                        None,
                        idx,
                        pool.get(id)
                            .expect("id must be a key in the workspace pool"),
                    )
                });

        iter_monitors.chain(iter_disconnected)
    }

    pub fn workspaces_mut(&mut self) -> impl Iterator<Item = &mut Workspace<W>> + '_ {
        // Pool owns every `Workspace<W>`; ordering among pool values is not defined.
        self.workspaces.values_mut()
    }

    /// Every workspace in the pool, crossing activity boundaries.
    /// Order is not guaranteed (iterates the HashMap). Intended for callers
    /// that intentionally need the full cross-activity view, e.g. IPC
    /// responses that expose `idx = 0` rows for hidden workspaces, `FocusWindow { id }` lookup
    /// against hidden workspaces, and config-reload / urgency paths.
    ///
    /// Each item pairs the workspace with its bound output id (via
    /// [`Workspace::output_id`]); `None` indicates a pool workspace that is
    /// not currently bound to any monitor. No interactive-move prepend —
    /// callers that need the interactive-move tile should inspect
    /// `self.interactive_move` directly.
    ///
    /// For monitor-ordered iteration of the active activity's workspaces
    /// (connected outputs) followed by disconnected entries, use
    /// [`Self::workspaces`].
    pub fn workspaces_all(&self) -> impl Iterator<Item = (Option<&OutputId>, &Workspace<W>)> + '_ {
        self.workspaces.values().map(|ws| (ws.output_id(), ws))
    }

    /// Workspaces that (a) are members of `activity_id` and (b) are bound
    /// to `output_id`. Order is pool order (unsorted). Callers that need a
    /// stable user-facing order ("config-declared first, then creation
    /// order") must sort the result themselves.
    ///
    /// Sticky workspaces are included naturally: the filter is pure
    /// activity-set membership, so any sticky workspace whose `activities`
    /// set contains `activity_id` is returned.
    ///
    /// Precondition: `activity_id` should be a live key in
    /// `self.activities`. An unknown id yields an empty iterator silently;
    /// callers that want a hard gate should check [`Activities::contains`]
    /// before calling.
    pub fn workspaces_with_activity<'a>(
        &'a self,
        activity_id: ActivityId,
        output_id: &'a OutputId,
    ) -> impl Iterator<Item = &'a Workspace<W>> + 'a {
        self.workspaces.values().filter(move |ws| {
            ws.output_id() == Some(output_id) && ws.activities().contains(&activity_id)
        })
    }

    /// Windows visible in the currently-active activity: the interactive-move
    /// window (if any) first, then every window on every workspace returned
    /// by [`Self::workspaces`] paired with the owning monitor.
    ///
    /// This is the activity-filtered default — callers that iterate "all
    /// windows the user can see right now" use this. Consumers that must
    /// cross activity boundaries (IPC event-stream projections, foreign-
    /// toplevel advertising, screencasting bookkeeping, surface-event
    /// routing for windows on dormant activities) use [`Self::windows_all`]
    /// or [`Self::with_windows_all`] instead.
    pub fn windows(&self) -> impl Iterator<Item = (Option<&Monitor<W>>, &W)> {
        let moving_window = self
            .interactive_move
            .as_ref()
            .and_then(|x| x.moving())
            .map(|move_| (self.monitor_for_output(&move_.output), move_.tile.window()))
            .into_iter();

        let rest = self
            .workspaces()
            .flat_map(|(mon, _, ws)| ws.windows().map(move |win| (mon, win)));

        moving_window.chain(rest)
    }

    /// Every window in the pool, crossing activity boundaries. Pairs each
    /// window with the bound output id of its owning workspace. `None` means
    /// the workspace has no bound `OutputId` at all (pool workspaces that
    /// were never attached to any output); a workspace whose bound output is
    /// currently disconnected still yields `Some(&oid)`. The interactive-move
    /// window is yielded first (mirrors [`Self::windows`]); its paired output
    /// id is borrowed from whichever bound workspace currently lives on the
    /// move's output, or `None` if no live monitor matches the move's output
    /// (an edge case during output reconfiguration — a `warn!` is emitted).
    ///
    /// Iteration order is pool order (undefined — the pool is a
    /// `HashMap<WorkspaceId, Workspace<W>>`). Callers that need a stable
    /// order must sort the result.
    ///
    /// Use this for cross-activity surface-routing, IPC event-stream
    /// projections, foreign-toplevel advertising, and screencasting
    /// bookkeeping — anywhere a window on a dormant activity must remain
    /// observable.
    pub fn windows_all(&self) -> impl Iterator<Item = (Option<&OutputId>, &W)> + '_ {
        let moving = self.interactive_move.as_ref().and_then(|x| x.moving());
        // Borrow a `&OutputId` from a pool workspace bound to the move's output.
        // `Monitor` does not store an `OutputId` field — `OutputId::new(&output)`
        // constructs fresh — so there is no reference to borrow from the monitor
        // directly. Instead, first verify the live-monitor exists (emit `warn!` if
        // not, since that's a genuine invariant break during output reconfiguration),
        // then borrow the id from any pool workspace bound to that monitor.
        let moving_oid = moving.and_then(|move_| {
            let needle = OutputId::new(&move_.output);
            if self.monitor_for_output_id(&needle).is_none() {
                warn!(
                    "windows_all: interactive-move output {:?} has no live monitor",
                    needle,
                );
                return None;
            }
            self.workspaces.values().find_map(|ws| {
                let ws_oid = ws.output_id()?;
                (*ws_oid == needle).then_some(ws_oid)
            })
        });
        let moving_window = moving
            .map(|move_| (moving_oid, move_.tile.window()))
            .into_iter();

        let rest = self
            .workspaces
            .values()
            .flat_map(|ws| ws.windows().map(move |win| (ws.output_id(), win)));

        moving_window.chain(rest)
    }

    /// Returns `true` if the window id exists anywhere in the workspace pool,
    /// including on workspaces bound to dormant activities. For the narrower
    /// active-activity check (monitor-ordered view plus disconnected entries),
    /// use [`Self::windows`].
    pub fn has_window(&self, window: &W::Id) -> bool {
        self.windows_all().any(|(_, win)| win.id() == window)
    }

    pub fn is_overview_open(&self) -> bool {
        self.overview_open
    }
}

fn compute_overview_zoom(options: &Options, overview_progress: Option<f64>) -> f64 {
    // Clamp to some sane values.
    let zoom = options.overview.zoom.clamp(0.0001, 0.75);

    if let Some(p) = overview_progress {
        (1. - p * (1. - zoom)).max(0.0001)
    } else {
        1.
    }
}
