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
//! Where possible, niri tries to follow these principles with regards to outputs:
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
use std::collections::HashMap;
use std::mem;
use std::rc::Rc;
use std::time::Duration;

use monitor::{InsertHint, InsertPosition, InsertWorkspace, MonitorAddWindowTarget};
use niri_config::utils::MergeWith as _;
use niri_config::{
    Config, CornerRadius, LayoutPart, PresetSize, Workspace as WorkspaceConfig, WorkspaceReference,
};
use niri_ipc::{ColumnDisplay, PositionChange, SizeChange, WindowLayout};
use scrolling::{Column, ColumnWidth};
use smithay::backend::renderer::element::surface::WaylandSurfaceRenderElement;
use smithay::backend::renderer::element::utils::RescaleRenderElement;
use smithay::backend::renderer::gles::{GlesRenderer, GlesTexture};
use smithay::output::{self, Output};
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::utils::{Logical, Point, Rectangle, Scale, Serial, Size, Transform};
use tile::{Tile, TileRenderElement};
use workspace::{WorkspaceAddWindowTarget, WorkspaceId};

use self::activity::WorkspaceView;
pub use self::monitor::MonitorRenderElement;
use self::monitor::{Monitor, WorkspaceSwitch};
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
/// Construct at call sites as `LayoutCtx::new(pool, mon.view())`. Both borrows
/// are shared, so a caller holding `&mon` can pass `ctx` into `&self` methods
/// on the same `mon` without conflict.
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
    pub fn new(pool: &'a HashMap<WorkspaceId, Workspace<W>>, view: &'a WorkspaceView) -> Self {
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
    fn update_config(&mut self, blur_config: niri_config::Blur) {
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
    /// Every id appearing in any `Monitor.view.ids()` or in `disconnected_workspace_ids` is a key
    /// here; the pool keys equal the disjoint union of those two sources — no orphans, no
    /// duplicates. Pool values are never drained out during output reconnect; monitors bind/unbind
    /// the `Smithay` output on their workspaces in place.
    pub(super) workspaces: HashMap<WorkspaceId, Workspace<W>>,
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
    pub layout: niri_config::Layout,
    pub animations: niri_config::Animations,
    pub gestures: niri_config::Gestures,
    pub overview: niri_config::Overview,
    pub blur: niri_config::Blur,
    // Debug flags.
    pub disable_resize_throttling: bool,
    pub disable_transactions: bool,
    pub deactivate_unfocused_windows: bool,
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
    pub(self) output_config: Option<niri_config::LayoutPart>,
    /// Config overrides for the workspace where the window is currently located.
    ///
    /// To avoid sudden window changes when starting an interactive move, it will remember the
    /// config overrides for the workspace where the move originated from. As soon as the window
    /// moves over some different workspace though, this override will reset.
    pub(self) workspace_config: Option<(WorkspaceId, niri_config::LayoutPart)>,
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
            disable_resize_throttling: config.debug.disable_resize_throttling,
            disable_transactions: config.debug.disable_transactions,
            deactivate_unfocused_windows: config.debug.deactivate_unfocused_windows,
        }
    }

    fn with_merged_layout(mut self, part: Option<&niri_config::LayoutPart>) -> Self {
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

        let mut workspaces: HashMap<WorkspaceId, Workspace<W>> = HashMap::new();
        let workspace_ids = config
            .workspaces
            .iter()
            .map(|ws| {
                let workspace = Workspace::new_with_config_no_outputs(
                    Some(ws.clone()),
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
        if self.monitors.is_empty() {
            // Reconnecting from a fully-disconnected state: the first monitor takes over all
            // workspaces that were parked in `disconnected_workspace_ids` in their saved order.
            let workspace_ids = mem::take(&mut self.disconnected_workspace_ids);
            let ws_id_to_activate = self.last_active_workspace_id.remove(&output.name());

            let mut monitor = Monitor::new(
                output,
                workspace_ids,
                ws_id_to_activate,
                &mut self.workspaces,
                self.clock.clone(),
                self.options.clone(),
                layout_config,
            );
            monitor.overview_open = self.overview_open;
            monitor.set_overview_progress(self.overview_progress.as_ref());

            self.monitors.push(monitor);
            self.primary_idx = 0;
            self.active_monitor_idx = 0;
            return;
        }

        let primary_idx = self.primary_idx;
        let pool = &mut self.workspaces;
        let primary = &mut self.monitors[primary_idx];

        let mut stopped_primary_ws_switch = false;

        let mut workspace_ids: Vec<WorkspaceId> = vec![];
        for i in (0..primary.view.len()).rev() {
            let id = primary.view.ids()[i];
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
                    // Empty unnamed workspaces don't come along — drop from the pool.
                    assert!(
                        pool.remove(&id).is_some(),
                        "view id must be a key in the pool",
                    );
                }

                // Without this exception, the first monitor to connect can end up
                // with the first empty workspace focused instead of the first named
                // workspace (under `empty_workspace_above_first`, `remove_at`'s
                // default shift would land focus on the forced-empty first
                // workspace).
                let active_pos_before = primary.view.active_position();
                let keep_active_pinned = primary.options.layout.empty_workspace_above_first
                    && active_pos_before == 1
                    && i <= active_pos_before;

                primary.view.remove_at(i);

                if keep_active_pinned {
                    let new_pos = 1.min(primary.view.len() - 1);
                    primary.view.set_active_at(new_pos);
                }
            }
        }

        // If we stopped a workspace switch, then we might need to clean up workspaces.
        // Also if empty_workspace_above_first is set and there are only 2 workspaces left,
        // both will be empty and one of them needs to be removed. clean_up_workspaces
        // takes care of this.

        let needs_cleanup = stopped_primary_ws_switch
            || (primary.options.layout.empty_workspace_above_first && primary.view.len() == 2);
        if needs_cleanup {
            Self::clean_up_workspaces_on(&mut self.monitors[..], &mut self.workspaces, primary_idx);
        }

        workspace_ids.reverse();

        let ws_id_to_activate = self.last_active_workspace_id.remove(&output.name());

        let mut monitor = Monitor::new(
            output,
            workspace_ids,
            ws_id_to_activate,
            &mut self.workspaces,
            self.clock.clone(),
            self.options.clone(),
            layout_config,
        );
        monitor.overview_open = self.overview_open;
        monitor.set_overview_progress(self.overview_progress.as_ref());
        self.monitors.push(monitor);
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
        let monitor = self.monitors.remove(idx);

        self.last_active_workspace_id
            .insert(monitor.output_name().clone(), monitor.view.active());

        let workspace_ids = self.take_workspace_ids(&monitor);

        if self.monitors.is_empty() {
            // Removed the last monitor. Values already live in the pool; just reset their options
            // to layout-root ones and park the id order for the next reconnect.
            for id in &workspace_ids {
                self.workspaces
                    .get_mut(id)
                    .expect("workspace id must be a key in the pool")
                    .update_config(self.options.clone());
            }

            self.disconnected_workspace_ids = workspace_ids;
            self.primary_idx = 0;
            self.active_monitor_idx = 0;
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
        Self::add_column_on(
            &mut self.monitors,
            &mut self.workspaces,
            monitor_idx,
            workspace_idx,
            column,
            activate,
        );

        if activate {
            self.active_monitor_idx = monitor_idx;
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
                        let ws =
                            Workspace::new_no_outputs(self.clock.clone(), self.options.clone());
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
                            let ws =
                                Workspace::new_no_outputs(self.clock.clone(), self.options.clone());
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
                let mon_idx = monitors
                    .iter()
                    .position(|mon| mon.view.ids().contains(&ws_id))
                    .unwrap();

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
                            mon.view.ids().iter().any(|id| {
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
        let scrolling_width = {
            let mon = &monitors[mon_idx];
            let (ws_idx, _) = mon.resolve_add_window_target(pool, target);
            mon.workspace_at(pool, ws_idx)
                .resolve_scrolling_width(&window, width)
        };

        Self::add_window_on(
            monitors,
            pool,
            mon_idx,
            window,
            target,
            activate,
            scrolling_width,
            is_full_width,
            is_floating,
        );

        if activate.map_smart(|| false) {
            *active_monitor_idx = mon_idx;
        }

        // Set the default height for scrolling windows.
        if !is_floating {
            if let Some(change) = scrolling_height {
                let ws_id = monitors[mon_idx]
                    .view
                    .ids()
                    .iter()
                    .copied()
                    .find(|ws_id| {
                        pool.get(ws_id)
                            .expect("view id must be a key in the pool")
                            .has_window(&id)
                    })
                    .unwrap();
                let ws = pool
                    .get_mut(&ws_id)
                    .expect("view id must be a key in the pool");
                ws.set_window_height(Some(&id), change);
            }
        }

        Some(&monitors[mon_idx].output)
    }

    pub fn remove_window(
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

                        for mon in self.monitors_mut() {
                            mon.dnd_scroll_gesture_end();
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

        let pool = &mut self.workspaces;
        for mon in &mut self.monitors {
            let active_pos = mon.view.active_position();
            for idx in 0..mon.view.len() {
                let id = mon.view.ids()[idx];
                let ws = pool
                    .get_mut(&id)
                    .expect("workspace id must be a key in the pool");
                if ws.has_window(window) {
                    let removed = ws.remove_tile(Some(&mon.output), window, transaction);

                    let ws_empty = !ws.has_windows_or_name();

                    // Clean up empty workspaces that are not active and not last.
                    if ws_empty
                        && idx != active_pos
                        && idx != mon.view.len() - 1
                        && mon.workspace_switch.is_none()
                    {
                        mon.view.remove_at(idx);
                        assert!(
                            pool.remove(&id).is_some(),
                            "view id must be a key in the pool",
                        );
                    }

                    // Special case handling when empty_workspace_above_first is set and all
                    // workspaces are empty.
                    if mon.options.layout.empty_workspace_above_first
                        && mon.view.len() == 2
                        && mon.workspace_switch.is_none()
                    {
                        assert!(!mon.workspace_at(pool, 0).has_windows_or_name());
                        assert!(!mon.workspace_at(pool, 1).has_windows_or_name());
                        let drop_id = mon.view.ids()[1];
                        mon.view.remove_at(1);
                        assert!(
                            pool.remove(&drop_id).is_some(),
                            "view id must be a key in the pool",
                        );
                    }
                    return Some(removed);
                }
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

        let pool = &mut self.workspaces;
        for mon in &mut self.monitors {
            for id in mon.view.ids() {
                let ws = pool
                    .get_mut(id)
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

    pub fn find_workspace_by_id(&self, id: WorkspaceId) -> Option<(usize, &Workspace<W>)> {
        for mon in &self.monitors {
            if let Some(index) = mon.view.position_of(id) {
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
            .find(|mon| mon.view.ids().contains(&id))
            .map(|mon| &mon.output)
    }

    pub fn find_workspace_by_name(&self, workspace_name: &str) -> Option<(usize, &Workspace<W>)> {
        let pool = &self.workspaces;
        for mon in &self.monitors {
            if let Some(index) = mon.view.ids().iter().position(|id| {
                pool.get(id)
                    .expect("view id must be a key in the pool")
                    .name
                    .as_ref()
                    .is_some_and(|name| name.eq_ignore_ascii_case(workspace_name))
            }) {
                let id = mon.view.ids()[index];
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
            let id = self
                .active_monitor()
                .and_then(|m| m.view.ids().get(index).copied())?;
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
            .position(|mon| mon.view.ids().contains(&id))
        {
            self.workspaces
                .get_mut(&id)
                .expect("view id must be a key in the pool")
                .unname();
            if self.monitors[mon_idx].workspace_switch.is_none() {
                Self::clean_up_workspaces_on(&mut self.monitors, &mut self.workspaces, mon_idx);
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

    pub fn find_window_and_output(&self, wl_surface: &WlSurface) -> Option<(&W, Option<&Output>)> {
        if let Some(InteractiveMoveState::Moving(move_)) = &self.interactive_move {
            if move_.tile.window().is_wl_surface(wl_surface) {
                return Some((move_.tile.window(), Some(&move_.output)));
            }
        }

        let pool = &self.workspaces;
        for mon in &self.monitors {
            for id in mon.view.ids() {
                let ws = pool
                    .get(id)
                    .expect("workspace id must be a key in the pool");
                if let Some(window) = ws.find_wl_surface(wl_surface) {
                    return Some((window, Some(&mon.output)));
                }
            }
        }
        for id in &self.disconnected_workspace_ids {
            let ws = pool
                .get(id)
                .expect("workspace id must be a key in the pool");
            if let Some(window) = ws.find_wl_surface(wl_surface) {
                return Some((window, None));
            }
        }

        None
    }

    pub fn find_window_and_output_mut(
        &mut self,
        wl_surface: &WlSurface,
    ) -> Option<(&mut W, Option<&Output>)> {
        if let Some(InteractiveMoveState::Moving(move_)) = &mut self.interactive_move {
            if move_.tile.window().is_wl_surface(wl_surface) {
                return Some((move_.tile.window_mut(), Some(&move_.output)));
            }
        }

        // First find the (monitor_idx, id) that matches; then borrow mut from pool.
        let pool = &self.workspaces;
        let mut matching: Option<(usize, WorkspaceId)> = None;
        for (mi, mon) in self.monitors.iter().enumerate() {
            for id in mon.view.ids() {
                if pool
                    .get(id)
                    .expect("workspace id must be a key in the pool")
                    .find_wl_surface(wl_surface)
                    .is_some()
                {
                    matching = Some((mi, *id));
                    break;
                }
            }
            if matching.is_some() {
                break;
            }
        }
        if let Some((mi, id)) = matching {
            let output = &self.monitors[mi].output;
            let ws = self
                .workspaces
                .get_mut(&id)
                .expect("workspace id must be a key in the pool");
            return ws
                .find_wl_surface_mut(wl_surface)
                .map(|w| (w, Some(output)));
        }

        let matching_id = self.disconnected_workspace_ids.iter().copied().find(|id| {
            self.workspaces
                .get(id)
                .expect("workspace id must be a key in the pool")
                .find_wl_surface(wl_surface)
                .is_some()
        });
        if let Some(id) = matching_id {
            return self
                .workspaces
                .get_mut(&id)
                .unwrap()
                .find_wl_surface_mut(wl_surface)
                .map(|w| (w, None));
        }

        None
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

        self.workspaces()
            .find_map(|(_, _, ws)| ws.popup_target_rect(window))
            .unwrap()
    }

    pub fn update_output_size(&mut self, output: &Output) {
        let _span = tracy_client::span!("Layout::update_output_size");

        let Some(mon) = self.monitors.iter_mut().find(|m| &m.output == output) else {
            error!("monitor missing in update_output_size()");
            return;
        };

        mon.update_output_size(&mut self.workspaces);
    }

    pub fn scroll_amount_to_activate(&self, window: &W::Id) -> f64 {
        if let Some(InteractiveMoveState::Moving(move_)) = &self.interactive_move {
            if move_.tile.window().id() == window {
                return 0.;
            }
        }

        let pool = &self.workspaces;
        for mon in self.monitors() {
            for id in mon.view.ids() {
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
                mon.view
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

        ws_idx == mon.view.active_position()
    }

    pub fn activate_window(&mut self, window: &W::Id) {
        if let Some(InteractiveMoveState::Moving(move_)) = &self.interactive_move {
            if move_.tile.window().id() == window {
                return;
            }
        }

        let pool = &mut self.workspaces;
        let monitors = &mut self.monitors[..];
        let active_monitor_idx = &mut self.active_monitor_idx;

        for (monitor_idx, mon) in monitors.iter_mut().enumerate() {
            for workspace_idx in 0..mon.view.len() {
                let id = mon.view.ids()[workspace_idx];
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
                        _ => mon.switch_workspace(workspace_idx),
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

        let pool = &mut self.workspaces;
        let monitors = &mut self.monitors[..];
        let active_monitor_idx = &mut self.active_monitor_idx;

        for (monitor_idx, mon) in monitors.iter_mut().enumerate() {
            for workspace_idx in 0..mon.view.len() {
                let id = mon.view.ids()[workspace_idx];
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
                        _ => mon.switch_workspace(workspace_idx),
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
        Some(mon.active_workspace_ref(&self.workspaces))
    }

    pub fn active_workspace_mut(&mut self) -> Option<&mut Workspace<W>> {
        if self.monitors.is_empty() {
            return None;
        }
        let mon = &self.monitors[self.active_monitor_idx];
        Some(mon.active_workspace(&mut self.workspaces))
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
        let mon_windows = mon.view.ids().iter().flat_map(move |id| {
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

        let moving_window = if is_interactive_match {
            self.interactive_move
                .as_mut()
                .and_then(|x| x.moving_mut())
                .map(|move_| move_.tile.window_mut())
        } else {
            None
        }
        .into_iter();

        let mon = &self.monitors[mi];
        let pool = &mut self.workspaces;

        // Iterate ids in order, hand out non-overlapping `&mut Workspace<W>` via raw ptr. Safe
        // because `view.ids()` has no duplicates.
        let ids: Vec<WorkspaceId> = mon.view.ids().to_vec();
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
            for id in mon.view.ids() {
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

        let pool = &mut self.workspaces;
        for mon in &mut self.monitors {
            for id in mon.view.ids() {
                let ws = pool
                    .get_mut(id)
                    .expect("workspace id must be a key in the pool");
                for win in ws.windows_mut() {
                    f(win, Some(&mon.output));
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

    fn active_monitor(&mut self) -> Option<&mut Monitor<W>> {
        if self.monitors.is_empty() {
            return None;
        }
        Some(&mut self.monitors[self.active_monitor_idx])
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

    /// Split-borrow helper: return `(&mut monitors, &mut pool)` for external callers that iterate
    /// monitors and call mutating `Monitor` methods threading `&mut pool`. Returns `(&mut [], ...)`
    /// if no outputs are connected.
    pub fn monitors_and_pool_mut(
        &mut self,
    ) -> (&mut [Monitor<W>], &mut HashMap<WorkspaceId, Workspace<W>>) {
        (&mut self.monitors[..], &mut self.workspaces)
    }

    /// Remove the workspace at `view_idx` from the monitor at `mon_idx`, unbind it from the
    /// output, and return its id. The workspace value remains in `self.workspaces` under that id —
    /// caller decides whether to re-attach it to another monitor or drop it.
    fn remove_workspace_from_monitor(
        &mut self,
        mon_idx: usize,
        mut view_idx: usize,
    ) -> WorkspaceId {
        let (monitors, pool) = self.monitors_and_pool_mut();

        if view_idx == monitors[mon_idx].view.len() - 1 {
            Self::add_workspace_bottom_on(monitors, pool, mon_idx);
        }
        if monitors[mon_idx].options.layout.empty_workspace_above_first && view_idx == 0 {
            Self::add_workspace_top_on(monitors, pool, mon_idx);
            view_idx += 1;
        }

        let mon = &mut monitors[mon_idx];
        // For monitor current workspace removal, we focus previous rather than next (<= rather
        // than <). This is different from columns and tiles, but it lets move-workspace-to-monitor
        // back and forth to preserve position. `WorkspaceView::remove_at` enforces this rule.
        let id = mon.view.ids()[view_idx];
        mon.view.remove_at(view_idx);

        pool.get_mut(&id)
            .expect("view id must be a key in the pool")
            .unbind_output(&mon.output);

        mon.workspace_switch = None;
        Self::clean_up_workspaces_on(monitors, pool, mon_idx);

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
        let (monitors, pool) = self.monitors_and_pool_mut();

        {
            let mon = &monitors[mon_idx];
            let ws = pool
                .get_mut(&id)
                .expect("workspace id must be a key in the pool");
            ws.bind_output(&mon.output);
            ws.update_config(mon.options.clone());
        }

        // Don't insert past the last empty workspace.
        if view_idx == monitors[mon_idx].view.len() {
            view_idx -= 1;
        }
        if view_idx == 0 && monitors[mon_idx].options.layout.empty_workspace_above_first {
            // Insert a new empty workspace on top to prepare for insertion of new workspace.
            Self::add_workspace_top_on(monitors, pool, mon_idx);
            view_idx += 1;
        }

        let mon = &mut monitors[mon_idx];
        mon.view.insert(view_idx, id);

        if activate {
            mon.workspace_switch = None;
            mon.activate_workspace(view_idx);
        }

        mon.workspace_switch = None;
        Self::clean_up_workspaces_on(monitors, pool, mon_idx);
    }

    /// Attach a list of existing pool-held workspaces to the monitor at `mon_idx`, in order, just
    /// above the bottom empty workspace.
    ///
    /// All `workspace_ids` must already be keys in `self.workspaces`.
    fn append_workspaces_to_monitor(&mut self, mon_idx: usize, workspace_ids: Vec<WorkspaceId>) {
        if workspace_ids.is_empty() {
            return;
        }

        let (monitors, pool) = self.monitors_and_pool_mut();

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
        let empty_was_focused = mon.view.active_position() == mon.view.len() - 1;

        // Insert in place so the view stays non-empty at every step
        // (`WorkspaceView` requires at least one id).
        let start = mon.view.len() - 1;
        for (offset, id) in workspace_ids.into_iter().enumerate() {
            let insert_pos = start + offset;
            mon.view.insert(insert_pos, id);
        }

        // If empty_workspace_above_first is set and the first workspace is now no longer empty,
        // add a new empty workspace on top.
        if mon.options.layout.empty_workspace_above_first
            && mon.workspace_at(pool, 0).has_windows_or_name()
        {
            Self::add_workspace_top_on(monitors, pool, mon_idx);
        }

        let mon = &mut monitors[mon_idx];
        // If the empty workspace was focused on the primary monitor, keep it focused.
        // Use `set_active_at` (not `activate`) so `previous` isn't clobbered — this is
        // an output reshuffle, not a user-visible workspace switch.
        if empty_was_focused {
            mon.view.set_active_at(mon.view.len() - 1);
        }

        // FIXME: if we're adding workspaces to currently invisible positions
        // (outside the workspace switch), we don't need to cancel it.
        mon.workspace_switch = None;
        Self::clean_up_workspaces_on(monitors, pool, mon_idx);
    }

    /// Detach the workspaces owned by `monitor` from its output.
    ///
    /// Non-empty workspaces stay in the pool with `output` unbound; empty unnamed workspaces
    /// (typically the bookends added by `Monitor::new`) are removed from the pool since no caller
    /// needs them back. Returns the retained ids in view order. Used when the output is
    /// disconnecting and `monitor` has already been removed from `self.monitors`.
    fn take_workspace_ids(&mut self, monitor: &Monitor<W>) -> Vec<WorkspaceId> {
        let pool = &mut self.workspaces;
        let mut kept = Vec::with_capacity(monitor.view.ids().len());
        for id in monitor.view.ids() {
            let ws = pool
                .get_mut(id)
                .expect("monitor ids must be keys in the pool");
            if ws.has_windows_or_name() {
                ws.unbind_output(&monitor.output);
                kept.push(*id);
            } else {
                // Empty bookends: drop from the pool.
                assert!(
                    pool.remove(id).is_some(),
                    "monitor id must be a key in the pool",
                );
            }
        }
        kept
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
        mon_idx: usize,
        idx: usize,
    ) {
        let mon = &mut monitors[mon_idx];
        let ws = Workspace::new(&mon.output, mon.clock.clone(), mon.options.clone());

        let id = ws.id();
        assert!(pool.insert(id, ws).is_none(), "fresh id must be unique");
        mon.view.insert(idx, id);

        if let Some(switch) = &mut mon.workspace_switch {
            if idx as f64 <= switch.target_idx() {
                switch.offset(1);
            }
        }
    }

    fn add_workspace_top_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        mon_idx: usize,
    ) {
        Self::add_workspace_at_on(monitors, pool, mon_idx, 0);
    }

    fn add_workspace_bottom_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        mon_idx: usize,
    ) {
        let len = monitors[mon_idx].view.len();
        Self::add_workspace_at_on(monitors, pool, mon_idx, len);
    }

    fn clean_up_workspaces_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        mon_idx: usize,
    ) {
        let mon = &mut monitors[mon_idx];
        assert!(mon.workspace_switch.is_none());

        let range_start = if mon.options.layout.empty_workspace_above_first {
            1
        } else {
            0
        };
        for idx in (range_start..mon.view.len() - 1).rev() {
            if mon.view.active_position() == idx {
                continue;
            }

            if !mon.workspace_at(pool, idx).has_windows_or_name() {
                let id = mon.view.ids()[idx];
                mon.view.remove_at(idx);
                assert!(
                    pool.remove(&id).is_some(),
                    "view id must be a key in the pool",
                );
            }
        }

        // Special case handling when empty_workspace_above_first is set and all workspaces
        // are empty.
        if mon.options.layout.empty_workspace_above_first && mon.view.len() == 2 {
            assert!(!mon.workspace_at(pool, 0).has_windows_or_name());
            assert!(!mon.workspace_at(pool, 1).has_windows_or_name());
            let id = mon.view.ids()[1];
            mon.view.remove_at(1);
            assert!(
                pool.remove(&id).is_some(),
                "view id must be a key in the pool",
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
        mon_idx: usize,
        window: W,
        target: MonitorAddWindowTarget<W>,
        activate: ActivateWindow,
        width: ColumnWidth,
        is_full_width: bool,
        is_floating: bool,
    ) {
        let mon = &monitors[mon_idx];
        let (workspace_idx, target) = mon.resolve_add_window_target(pool, target);
        let tile = mon.workspace_at(pool, workspace_idx).make_tile(window);

        Self::add_resolved_tile_on(
            monitors,
            pool,
            mon_idx,
            workspace_idx,
            tile,
            target,
            activate,
            true,
            width,
            is_full_width,
            is_floating,
        );
    }

    fn add_column_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        mon_idx: usize,
        mut workspace_idx: usize,
        column: Column<W>,
        activate: bool,
    ) {
        let mon = &mut monitors[mon_idx];
        let workspace = mon.workspace_at_mut(pool, workspace_idx);

        workspace.add_column(Some(&mon.output), column, activate);

        // After adding a new window, workspace becomes this output's own.
        if workspace.name().is_none() {
            workspace.output_id = Some(OutputId::new(&mon.output));
        }

        if workspace_idx == mon.view.len() - 1 {
            Self::add_workspace_bottom_on(monitors, pool, mon_idx);
        }
        if monitors[mon_idx].options.layout.empty_workspace_above_first && workspace_idx == 0 {
            Self::add_workspace_top_on(monitors, pool, mon_idx);
            workspace_idx += 1;
        }

        if activate {
            monitors[mon_idx].activate_workspace(workspace_idx);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn add_tile_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        mon_idx: usize,
        tile: Tile<W>,
        target: MonitorAddWindowTarget<W>,
        activate: ActivateWindow,
        // FIXME: Refactor ActivateWindow enum to make this better.
        allow_to_activate_workspace: bool,
        width: ColumnWidth,
        is_full_width: bool,
        is_floating: bool,
    ) {
        let (workspace_idx, target) = monitors[mon_idx].resolve_add_window_target(pool, target);
        Self::add_resolved_tile_on(
            monitors,
            pool,
            mon_idx,
            workspace_idx,
            tile,
            target,
            activate,
            allow_to_activate_workspace,
            width,
            is_full_width,
            is_floating,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn add_resolved_tile_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
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
    ) {
        let mon = &mut monitors[mon_idx];
        let workspace = mon.workspace_at_mut(pool, workspace_idx);

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

        if workspace_idx == mon.view.len() - 1 {
            // Insert a new empty workspace.
            Self::add_workspace_bottom_on(monitors, pool, mon_idx);
        }

        if monitors[mon_idx].options.layout.empty_workspace_above_first && workspace_idx == 0 {
            Self::add_workspace_top_on(monitors, pool, mon_idx);
            workspace_idx += 1;
        }

        if allow_to_activate_workspace && activate.map_smart(|| false) {
            monitors[mon_idx].activate_workspace(workspace_idx);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn add_tile_to_column_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
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
        let workspace = mon.workspace_at_mut(pool, workspace_idx);

        workspace.add_tile_to_column(Some(&mon.output), column_idx, tile_idx, tile, activate);

        // After adding a new window, workspace becomes this output's own.
        if workspace.name().is_none() {
            workspace.output_id = Some(OutputId::new(&mon.output));
        }

        // Since we're adding window to an existing column, the workspace isn't empty, and
        // therefore cannot be the last one, so we never need to insert a new empty workspace.

        if allow_to_activate_workspace && activate {
            mon.activate_workspace(workspace_idx);
        }
    }

    fn move_down_or_to_workspace_down_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        mon_idx: usize,
    ) {
        if !monitors[mon_idx].active_workspace(pool).move_down() {
            Self::move_to_workspace_down_on(monitors, pool, mon_idx, true);
        }
    }

    fn move_up_or_to_workspace_up_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        mon_idx: usize,
    ) {
        if !monitors[mon_idx].active_workspace(pool).move_up() {
            Self::move_to_workspace_up_on(monitors, pool, mon_idx, true);
        }
    }

    fn focus_window_or_workspace_down_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        mon_idx: usize,
    ) {
        if !monitors[mon_idx].active_workspace(pool).focus_down() {
            monitors[mon_idx].switch_workspace_down();
        }
    }

    fn focus_window_or_workspace_up_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        mon_idx: usize,
    ) {
        if !monitors[mon_idx].active_workspace(pool).focus_up() {
            monitors[mon_idx].switch_workspace_up();
        }
    }

    fn move_to_workspace_up_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        mon_idx: usize,
        focus: bool,
    ) {
        let mon = &mut monitors[mon_idx];
        let source_workspace_idx = mon.view.active_position();

        let new_idx = source_workspace_idx.saturating_sub(1);
        if new_idx == source_workspace_idx {
            return;
        }
        let new_id = mon.view.ids()[new_idx];

        let workspace = mon.workspace_at_mut(pool, source_workspace_idx);
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
        );
    }

    fn move_to_workspace_down_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        mon_idx: usize,
        focus: bool,
    ) {
        let mon = &mut monitors[mon_idx];
        let source_workspace_idx = mon.view.active_position();

        let new_idx = min(source_workspace_idx + 1, mon.view.len() - 1);
        if new_idx == source_workspace_idx {
            return;
        }
        let new_id = mon.view.ids()[new_idx];

        let workspace = mon.workspace_at_mut(pool, source_workspace_idx);
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
        );
    }

    fn move_to_workspace_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        mon_idx: usize,
        window: Option<&W::Id>,
        idx: usize,
        activate: ActivateWindow,
    ) {
        let mon = &mut monitors[mon_idx];
        let source_workspace_idx = if let Some(window) = window {
            mon.view
                .ids()
                .iter()
                .position(|id| {
                    pool.get(id)
                        .expect("view id must be a key in the pool")
                        .has_window(window)
                })
                .unwrap()
        } else {
            mon.view.active_position()
        };

        let new_idx = min(idx, mon.view.len() - 1);
        if new_idx == source_workspace_idx {
            return;
        }
        let new_id = mon.view.ids()[new_idx];

        let active_window_id = mon.active_window(pool).map(|win| win.id().clone());
        let activate =
            activate.map_smart(|| window.is_none_or(|win| active_window_id.as_ref() == Some(win)));

        let workspace = mon.workspace_at_mut(pool, source_workspace_idx);
        let transaction = Transaction::new();
        let removed = if let Some(window) = window {
            workspace.remove_tile(Some(&mon.output), window, transaction)
        } else if let Some(removed) = workspace.remove_active_tile(Some(&mon.output), transaction) {
            removed
        } else {
            return;
        };

        Self::add_tile_on(
            monitors,
            pool,
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
        );

        if monitors[mon_idx].workspace_switch.is_none() {
            Self::clean_up_workspaces_on(monitors, pool, mon_idx);
        }
    }

    fn move_column_to_workspace_up_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        mon_idx: usize,
        activate: bool,
    ) {
        let mon = &mut monitors[mon_idx];
        let source_workspace_idx = mon.view.active_position();

        let new_idx = source_workspace_idx.saturating_sub(1);
        if new_idx == source_workspace_idx {
            return;
        }

        // Check floating status on a shared borrow first so we can recurse into the sibling method
        // without a `&mut pool` conflict.
        if mon
            .workspace_at(pool, source_workspace_idx)
            .floating_is_active()
        {
            Self::move_to_workspace_up_on(monitors, pool, mon_idx, activate);
            return;
        }

        let workspace = mon.workspace_at_mut(pool, source_workspace_idx);
        let Some(column) = workspace.remove_active_column(Some(&mon.output)) else {
            return;
        };

        Self::add_column_on(monitors, pool, mon_idx, new_idx, column, activate);
    }

    fn move_column_to_workspace_down_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        mon_idx: usize,
        activate: bool,
    ) {
        let mon = &mut monitors[mon_idx];
        let source_workspace_idx = mon.view.active_position();

        let new_idx = min(source_workspace_idx + 1, mon.view.len() - 1);
        if new_idx == source_workspace_idx {
            return;
        }

        if mon
            .workspace_at(pool, source_workspace_idx)
            .floating_is_active()
        {
            Self::move_to_workspace_down_on(monitors, pool, mon_idx, activate);
            return;
        }

        let workspace = mon.workspace_at_mut(pool, source_workspace_idx);
        let Some(column) = workspace.remove_active_column(Some(&mon.output)) else {
            return;
        };

        Self::add_column_on(monitors, pool, mon_idx, new_idx, column, activate);
    }

    fn move_column_to_workspace_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        mon_idx: usize,
        idx: usize,
        activate: bool,
    ) {
        let mon = &mut monitors[mon_idx];
        let source_workspace_idx = mon.view.active_position();

        let new_idx = min(idx, mon.view.len() - 1);
        if new_idx == source_workspace_idx {
            return;
        }

        if mon
            .workspace_at(pool, source_workspace_idx)
            .floating_is_active()
        {
            let activate = if activate {
                ActivateWindow::Smart
            } else {
                ActivateWindow::No
            };
            Self::move_to_workspace_on(monitors, pool, mon_idx, None, idx, activate);
            return;
        }

        let workspace = mon.workspace_at_mut(pool, source_workspace_idx);
        let Some(column) = workspace.remove_active_column(Some(&mon.output)) else {
            return;
        };

        Self::add_column_on(monitors, pool, mon_idx, new_idx, column, activate);
    }

    fn move_workspace_down_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        mon_idx: usize,
    ) {
        let mon = &mut monitors[mon_idx];
        let active_idx = mon.view.active_position();
        let mut new_idx = min(active_idx + 1, mon.view.len() - 1);
        if new_idx == active_idx {
            return;
        }

        mon.view.swap(active_idx, new_idx);

        if new_idx == mon.view.len() - 1 {
            // Insert a new empty workspace.
            Self::add_workspace_bottom_on(monitors, pool, mon_idx);
        }

        if monitors[mon_idx].options.layout.empty_workspace_above_first && active_idx == 0 {
            Self::add_workspace_top_on(monitors, pool, mon_idx);
            new_idx += 1;
        }

        let mon = &mut monitors[mon_idx];
        let previous_workspace_id = mon.view.previous();
        mon.activate_workspace(new_idx);
        mon.workspace_switch = None;
        mon.view.set_previous(previous_workspace_id);

        Self::clean_up_workspaces_on(monitors, pool, mon_idx);
    }

    fn move_workspace_up_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        mon_idx: usize,
    ) {
        let mon = &mut monitors[mon_idx];
        let active_idx = mon.view.active_position();
        let mut new_idx = active_idx.saturating_sub(1);
        if new_idx == active_idx {
            return;
        }

        mon.view.swap(active_idx, new_idx);

        if active_idx == mon.view.len() - 1 {
            // Insert a new empty workspace.
            Self::add_workspace_bottom_on(monitors, pool, mon_idx);
        }

        if monitors[mon_idx].options.layout.empty_workspace_above_first && new_idx == 0 {
            Self::add_workspace_top_on(monitors, pool, mon_idx);
            new_idx += 1;
        }

        let mon = &mut monitors[mon_idx];
        let previous_workspace_id = mon.view.previous();
        mon.activate_workspace(new_idx);
        mon.workspace_switch = None;
        mon.view.set_previous(previous_workspace_id);

        Self::clean_up_workspaces_on(monitors, pool, mon_idx);
    }

    fn move_workspace_to_idx_on(
        monitors: &mut [Monitor<W>],
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        mon_idx: usize,
        old_idx: usize,
        new_idx: usize,
    ) {
        let mon = &mut monitors[mon_idx];
        if mon.view.len() <= old_idx {
            return;
        }

        let new_idx = new_idx.clamp(0, mon.view.len() - 1);
        if old_idx == new_idx {
            return;
        }

        mon.view.move_within(old_idx, new_idx);

        if new_idx > old_idx {
            if new_idx == mon.view.len() - 1 {
                // Insert a new empty workspace.
                Self::add_workspace_bottom_on(monitors, pool, mon_idx);
            }

            if monitors[mon_idx].options.layout.empty_workspace_above_first && old_idx == 0 {
                Self::add_workspace_top_on(monitors, pool, mon_idx);
            }
        } else {
            if old_idx == monitors[mon_idx].view.len() - 1 {
                // Insert a new empty workspace.
                Self::add_workspace_bottom_on(monitors, pool, mon_idx);
            }

            if monitors[mon_idx].options.layout.empty_workspace_above_first && new_idx == 0 {
                Self::add_workspace_top_on(monitors, pool, mon_idx);
            }
        }

        monitors[mon_idx].workspace_switch = None;

        Self::clean_up_workspaces_on(monitors, pool, mon_idx);
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

    pub fn monitor_for_workspace(&self, workspace_name: &str) -> Option<&Monitor<W>> {
        let pool = &self.workspaces;
        self.monitors().find(|monitor| {
            monitor.view.ids().iter().any(|id| {
                pool.get(id)
                    .expect("view id must be a key in the pool")
                    .name
                    .as_ref()
                    .is_some_and(|name| name.eq_ignore_ascii_case(workspace_name))
            })
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
        Self::move_down_or_to_workspace_down_on(
            &mut self.monitors,
            &mut self.workspaces,
            active_monitor_idx,
        );
    }

    pub fn move_up_or_to_workspace_up(&mut self) {
        if self.monitors.is_empty() {
            return;
        }
        let active_monitor_idx = self.active_monitor_idx;
        Self::move_up_or_to_workspace_up_on(
            &mut self.monitors,
            &mut self.workspaces,
            active_monitor_idx,
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
        Self::focus_window_or_workspace_down_on(
            &mut self.monitors,
            &mut self.workspaces,
            active_monitor_idx,
        );
    }

    pub fn focus_window_or_workspace_up(&mut self) {
        if self.monitors.is_empty() {
            return;
        }
        let active_monitor_idx = self.active_monitor_idx;
        Self::focus_window_or_workspace_up_on(
            &mut self.monitors,
            &mut self.workspaces,
            active_monitor_idx,
        );
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
        Self::move_to_workspace_up_on(
            &mut self.monitors,
            &mut self.workspaces,
            active_monitor_idx,
            focus,
        );
    }

    pub fn move_to_workspace_down(&mut self, focus: bool) {
        if self.monitors.is_empty() {
            return;
        }
        let active_monitor_idx = self.active_monitor_idx;
        Self::move_to_workspace_down_on(
            &mut self.monitors,
            &mut self.workspaces,
            active_monitor_idx,
            focus,
        );
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
            let pool = &self.workspaces;
            self.monitors
                .iter()
                .position(|mon| mon.has_window(pool, window))
                .unwrap()
        } else {
            self.active_monitor_idx
        };

        Self::move_to_workspace_on(
            &mut self.monitors,
            &mut self.workspaces,
            mon_idx,
            window,
            idx,
            activate,
        );
    }

    pub fn move_column_to_workspace_up(&mut self, activate: bool) {
        if self.monitors.is_empty() {
            return;
        }
        let active_monitor_idx = self.active_monitor_idx;
        Self::move_column_to_workspace_up_on(
            &mut self.monitors,
            &mut self.workspaces,
            active_monitor_idx,
            activate,
        );
    }

    pub fn move_column_to_workspace_down(&mut self, activate: bool) {
        if self.monitors.is_empty() {
            return;
        }
        let active_monitor_idx = self.active_monitor_idx;
        Self::move_column_to_workspace_down_on(
            &mut self.monitors,
            &mut self.workspaces,
            active_monitor_idx,
            activate,
        );
    }

    pub fn move_column_to_workspace(&mut self, idx: usize, activate: bool) {
        if self.monitors.is_empty() {
            return;
        }
        let active_monitor_idx = self.active_monitor_idx;
        Self::move_column_to_workspace_on(
            &mut self.monitors,
            &mut self.workspaces,
            active_monitor_idx,
            idx,
            activate,
        );
    }

    pub fn switch_workspace_up(&mut self) {
        let Some(monitor) = self.active_monitor() else {
            return;
        };
        monitor.switch_workspace_up();
    }

    pub fn switch_workspace_down(&mut self) {
        let Some(monitor) = self.active_monitor() else {
            return;
        };
        monitor.switch_workspace_down();
    }

    pub fn switch_workspace(&mut self, idx: usize) {
        let Some(monitor) = self.active_monitor() else {
            return;
        };
        monitor.switch_workspace(idx);
    }

    pub fn switch_workspace_auto_back_and_forth(&mut self, idx: usize) {
        let Some(monitor) = self.active_monitor() else {
            return;
        };
        monitor.switch_workspace_auto_back_and_forth(idx);
    }

    pub fn switch_workspace_previous(&mut self) {
        let Some(monitor) = self.active_monitor() else {
            return;
        };
        monitor.switch_workspace_previous();
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
        mon.active_window(&self.workspaces)
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
        let ctx = LayoutCtx::new(&self.workspaces, &mon.view);
        mon.window_under(ctx, pos_within_output)
    }

    pub fn resize_edges_under(
        &self,
        output: &Output,
        pos_within_output: Point<f64, Logical>,
    ) -> Option<ResizeEdge> {
        let mon = self.monitor_for_output(output)?;
        let ctx = LayoutCtx::new(&self.workspaces, &mon.view);
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
        let ctx = LayoutCtx::new(&self.workspaces, &mon.view);
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
        use std::collections::HashSet;

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

        let mut seen_workspace_id = HashSet::new();
        let mut seen_workspace_name = Vec::<String>::new();

        let pool = &self.workspaces;

        // Pool keys equal the disjoint union of every Monitor.view.ids() and
        // disconnected_workspace_ids. Build the expected key set and compare.
        let mut expected_keys: HashSet<WorkspaceId> = HashSet::new();
        for mon in &self.monitors {
            for id in mon.view.ids() {
                assert!(
                    expected_keys.insert(*id),
                    "workspace id must appear in at most one monitor view",
                );
                assert!(
                    pool.contains_key(id),
                    "Monitor.view.ids entry must be a key in the workspace pool",
                );
            }
        }
        for id in &self.disconnected_workspace_ids {
            assert!(
                expected_keys.insert(*id),
                "workspace id must appear at most once in disconnected_workspace_ids",
            );
            assert!(
                pool.contains_key(id),
                "disconnected_workspace_ids entry must be a key in the workspace pool",
            );
        }
        let pool_keys: HashSet<WorkspaceId> = pool.keys().copied().collect();
        assert_eq!(
            expected_keys, pool_keys,
            "pool keys must equal the union of Monitor.view.ids and disconnected_workspace_ids",
        );

        if self.monitors.is_empty() {
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

            monitor.verify_invariants(pool);

            if idx == primary_idx {
                for id in monitor.view.ids() {
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
                    monitor.view.ids().iter().any(|id| {
                        pool.get(id)
                            .and_then(|ws| ws.output_id.as_ref())
                            .is_some_and(|oid| oid.matches(&monitor.output))
                    }),
                    "secondary monitor must not have any non-own workspaces"
                );
            }

            // FIXME: verify that primary doesn't have any workspaces for which their own monitor
            // exists.

            for id in monitor.view.ids() {
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
            let pool = &mut self.workspaces;
            if let Some(mon) = self.monitors.iter_mut().find(|m| m.output == output) {
                let mut scrolled = false;

                let zoom = mon.overview_zoom();
                scrolled |= mon.dnd_scroll_gesture_scroll(pos_within_output, 1. / zoom);

                if is_scrolling {
                    let ctx = LayoutCtx::new(&*pool, &mon.view);
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
                    let ctx = LayoutCtx::new(&*pool, &mon.view);
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
                                    for (i, view_id) in mon.view.ids().iter().enumerate() {
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
                                DndHoldTarget::Workspace(id) => mon.view.position_of(id).unwrap(),
                            };

                            mon.dnd_scroll_gesture_end();
                            mon.activate_workspace_with_anim_config(ws_idx, config);

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

        let pool = &mut self.workspaces;
        let overview = self.overview_progress.as_ref();
        let monitors = &mut self.monitors[..];
        for mon_idx in 0..monitors.len() {
            monitors[mon_idx].set_overview_progress(overview);
            let workspace_switch_finished = monitors[mon_idx].advance_animations(pool);
            if workspace_switch_finished {
                Self::clean_up_workspaces_on(monitors, pool, mon_idx);
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

            if mon.are_animations_ongoing(&self.workspaces) {
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

        let pool = &mut self.workspaces;
        for (idx, mon) in self.monitors.iter_mut().enumerate() {
            if output.is_none_or(|output| mon.output == *output) {
                let is_active = self.is_active
                    && idx == self.active_monitor_idx
                    && !matches!(self.interactive_move, Some(InteractiveMoveState::Moving(_)));
                mon.set_overview_progress(self.overview_progress.as_ref());
                mon.update_render_elements(pool, is_active);
            }
        }
    }

    pub fn update_shaders(&mut self) {
        if let Some(InteractiveMoveState::Moving(move_)) = &mut self.interactive_move {
            move_.tile.update_shaders();
        }

        let pool = &mut self.workspaces;
        for mon in &mut self.monitors {
            mon.update_shaders(pool);
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

        let pool = &mut self.workspaces;
        if let Some(mon) = self.monitors.iter_mut().find(|m| m.output == move_.output) {
            let zoom = mon.overview_zoom();
            let ctx = LayoutCtx::new(&*pool, &mon.view);
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

    pub fn ensure_named_workspace(&mut self, ws_config: &WorkspaceConfig) {
        if self.find_workspace_by_name(&ws_config.name.0).is_some() {
            return;
        }

        let clock = self.clock.clone();
        let options = self.options.clone();

        if self.monitors.is_empty() {
            let ws = Workspace::new_with_config_no_outputs(Some(ws_config.clone()), clock, options);
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
        let mon = &self.monitors[mon_idx];

        let ws = Workspace::new_with_config(&mon.output, Some(ws_config.clone()), clock, options);
        let id = ws.id();
        assert!(
            self.workspaces.insert(id, ws).is_none(),
            "fresh id must be unique",
        );
        self.insert_workspace_onto_monitor(mon_idx, id, 0, false);
    }

    pub fn update_config(&mut self, config: &Config) {
        // Update workspace-specific config for all named workspaces.
        for ws in self.workspaces_mut() {
            let Some(name) = ws.name() else { continue };
            if let Some(config) = config.workspaces.iter().find(|w| &w.name.0 == name) {
                ws.update_layout_config(config.layout.clone().map(|x| x.0));
            }
        }

        self.update_options(Options::from_config(config));
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

        let pool = &mut self.workspaces;
        for mon in &mut self.monitors {
            mon.update_config(pool, options.clone());
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
            self.monitors
                .iter()
                .enumerate()
                .find_map(|(mon_idx, mon)| {
                    mon.view
                        .ids()
                        .iter()
                        .position(|id| {
                            self.workspaces
                                .get(id)
                                .is_some_and(|ws| ws.has_window(window))
                        })
                        .map(|ws_idx| (mon_idx, ws_idx))
                })
                .unwrap()
        } else {
            let mon_idx = self.active_monitor_idx;
            let mon = &self.monitors[mon_idx];
            (mon_idx, mon.view.active_position())
        };

        let workspace_idx = target_ws_idx.unwrap_or(self.monitors[new_idx].view.active_position());
        if mon_idx == new_idx && ws_idx == workspace_idx {
            return;
        }

        let mon = &self.monitors[new_idx];
        if mon.view.len() <= workspace_idx {
            return;
        }

        let ws_id = mon.view.ids()[workspace_idx];

        let pool = &mut self.workspaces;
        let monitors = &mut self.monitors[..];
        let active_monitor_idx = &mut self.active_monitor_idx;
        let mon = &mut monitors[mon_idx];
        let active_window_id = mon.active_window(pool).map(|w| w.id().clone());
        let activate = activate.map_smart(|| {
            window.is_none_or(|win| {
                mon_idx == *active_monitor_idx && active_window_id.as_ref() == Some(win)
            })
        });
        let activate = if activate {
            ActivateWindow::Yes
        } else {
            ActivateWindow::No
        };

        let ws = mon.workspace_at_mut(pool, ws_idx);
        let transaction = Transaction::new();
        let mut removed = if let Some(window) = window {
            ws.remove_tile(Some(&mon.output), window, transaction)
        } else if let Some(removed) = ws.remove_active_tile(Some(&mon.output), transaction) {
            removed
        } else {
            return;
        };

        removed.tile.stop_move_animations();

        Self::add_tile_on(
            monitors,
            pool,
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
        );
        if activate.map_smart(|| false) {
            *active_monitor_idx = new_idx;
        }

        if monitors[mon_idx].workspace_switch.is_none() {
            Self::clean_up_workspaces_on(monitors, pool, mon_idx);
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
        let pool = &mut self.workspaces;
        let current = &self.monitors[active_monitor_idx];
        let active_pos = current.view.active_position();

        // Check floating status on a shared borrow first; move_to_output needs `&mut self`,
        // so we can't take a mutable borrow yet.
        if current.workspace_at(pool, active_pos).floating_is_active() {
            self.move_to_output(None, output, None, ActivateWindow::Smart);
            return;
        }

        // Scrolling path.
        let current = &mut self.monitors[active_monitor_idx];
        let current_output_ref = &current.output;
        let ws = current.workspace_at_mut(pool, active_pos);

        let Some(column) = ws.remove_active_column(Some(current_output_ref)) else {
            return;
        };

        let workspace_idx = target_ws_idx
            .unwrap_or(self.monitors[new_idx].view.active_position())
            .min(self.monitors[new_idx].view.len() - 1);
        self.add_column_by_idx(new_idx, workspace_idx, column, activate);
    }

    pub fn move_workspace_to_output(&mut self, output: &Output) -> bool {
        if self.monitors.is_empty() {
            return false;
        }
        let idx = self.monitors[self.active_monitor_idx]
            .view
            .active_position();
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
            if current.view.len() <= old_idx {
                return false;
            }

            // Only switch active monitor if the workspace to be moved is the currently focused
            // one on the current monitor. Computed eagerly on both the cross-output and same-
            // output paths; on the same-output short-circuit these are pure reads of view state,
            // so the wasted work is harmless and keeps the shared-borrow scope rectangular.
            let activate =
                current_idx == self.active_monitor_idx && old_idx == current.view.active_position();
            let target_pos = self.monitors[target_idx].view.active_position() + 1;

            (current_idx, target_idx, target_pos, activate)
        };

        // Do not do anything if the output is already correct.
        if current_idx == target_idx {
            // Just update the designated output id since this is an explicit movement action.
            let (monitors, pool) = self.monitors_and_pool_mut();
            let mon = &mut monitors[current_idx];
            let new_output_id = Some(OutputId::new(mon.output()));
            mon.workspace_at_mut(pool, old_idx).output_id = new_output_id;
            return false;
        }

        let ws_id = self.remove_workspace_from_monitor(current_idx, old_idx);
        self.workspaces
            .get_mut(&ws_id)
            .expect("workspace id must be a key in the pool")
            .output_id = Some(OutputId::new(new_output));
        self.insert_workspace_onto_monitor(target_idx, ws_id, target_pos, activate);

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
        let (_, window) = self.windows().find(|(_, win)| win.id() == id).unwrap();
        if window.pending_sizing_mode().is_fullscreen() {
            // Remove the real fullscreen.
            for ws in self.workspaces_mut() {
                if ws.has_window(id) {
                    ws.set_fullscreen(id, false);
                    break;
                }
            }
        }

        // This will switch is_pending_fullscreen() to false right away.
        self.with_windows_mut(|window, _| {
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
        let monitors = &mut self.monitors;

        for monitor in monitors {
            // Cancel the gesture on other outputs.
            if &monitor.output != output {
                monitor.workspace_switch_gesture_end(None);
                continue;
            }

            monitor.workspace_switch_gesture_begin(is_touchpad);
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
        let monitors = &mut self.monitors;

        for monitor in monitors {
            if let Some(refresh) =
                monitor.workspace_switch_gesture_update(delta_y, timestamp, is_touchpad)
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
        let monitors = &mut self.monitors;

        for monitor in monitors {
            if monitor.workspace_switch_gesture_end(is_touchpad) {
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
        let monitors = &mut self.monitors;

        let pool = &mut self.workspaces;
        for monitor in monitors {
            for (idx, id) in monitor.view.ids().to_vec().into_iter().enumerate() {
                let ws = pool
                    .get_mut(&id)
                    .expect("workspace id must be a key in the pool");
                // Cancel the gesture on other workspaces.
                if &monitor.output != output
                    || idx != workspace_idx.unwrap_or(monitor.view.active_position())
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

        let pool = &mut self.workspaces;
        if self.monitors.is_empty() {
            return None;
        }
        let monitors = &mut self.monitors;

        for monitor in monitors {
            for id in monitor.view.ids().to_vec() {
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
        let pool = &mut self.workspaces;
        if self.monitors.is_empty() {
            return None;
        }
        let monitors = &mut self.monitors;

        for monitor in monitors {
            for id in monitor.view.ids().to_vec() {
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
        let Some((mon, (ws, ws_geo))) = self.monitors().find_map(|mon| {
            let ctx = LayoutCtx::new(pool, &mon.view);
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

        for mon in self.monitors_mut() {
            mon.dnd_scroll_gesture_begin();
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
                let mut tile_pos = None;
                if let Some((mon, (ws, ws_geo))) = self.monitors().find_map(|mon| {
                    let ctx = LayoutCtx::new(pool, &mon.view);
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
                } = self.remove_window(window, Transaction::new()).unwrap();

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
                    let ctx = LayoutCtx::new(&self.workspaces, &mon.view);
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

                for mon in self.monitors_mut() {
                    mon.dnd_scroll_gesture_end();
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

        for mon in self.monitors_mut() {
            mon.dnd_scroll_gesture_end();
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

        let pool = &mut self.workspaces;
        if self.monitors.is_empty() {
            let workspaces = &mut self.disconnected_workspace_ids;
            if workspaces.is_empty() {
                let ws = Workspace::new_no_outputs(self.clock.clone(), self.options.clone());
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

        let monitors = &mut self.monitors[..];
        let active_monitor_idx = &mut self.active_monitor_idx;

        let (mon_idx, insert_ws, position, offset, zoom) =
            if let Some(mon_idx) = monitors.iter().position(|mon| mon.output == move_.output) {
                let mon = &mut monitors[mon_idx];
                let zoom = mon.overview_zoom();

                let ctx = LayoutCtx::new(&*pool, &mon.view);
                let (insert_ws, geo) = mon.insert_position(ctx, move_.pointer_pos_within_output);
                let (position, offset) = match insert_ws {
                    InsertWorkspace::Existing(ws_id) => {
                        let ws_idx = mon.view.position_of(ws_id).unwrap();

                        let position = if move_.is_floating {
                            InsertPosition::Floating
                        } else {
                            let pos_within_workspace =
                                (move_.pointer_pos_within_output - geo.loc).downscale(zoom);
                            let ws = mon.workspace_at_mut(pool, ws_idx);
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

                (mon_idx, insert_ws, position, offset, zoom)
            } else {
                let mon_idx = *active_monitor_idx;
                let mon = &monitors[mon_idx];
                let zoom = mon.overview_zoom();
                // No point in trying to use the pointer position on the wrong output.
                let ws = mon.workspace_at(pool, 0);
                let ws_id = ws.id();
                let ws_geo = mon.workspaces_render_geo().next().unwrap();

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
            InsertWorkspace::Existing(ws_id) => monitors[mon_idx].view.position_of(ws_id).unwrap(),
            InsertWorkspace::NewAt(ws_idx) => {
                let mon = &monitors[mon_idx];
                if mon.options.layout.empty_workspace_above_first && ws_idx == 0 {
                    // Reuse the top empty workspace.
                    0
                } else if mon.view.len() - 1 <= ws_idx {
                    // Reuse the bottom empty workspace.
                    mon.view.len() - 1
                } else {
                    Self::add_workspace_at_on(monitors, pool, mon_idx, ws_idx);
                    ws_idx
                }
            }
        };

        match position {
            InsertPosition::NewColumn(column_idx) => {
                let ws_id = monitors[mon_idx].view.ids()[ws_idx];
                Self::add_tile_on(
                    monitors,
                    pool,
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
                );
            }
            InsertPosition::InColumn(column_idx, tile_idx) => {
                Self::add_tile_to_column_on(
                    monitors,
                    pool,
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
                                .workspace_at(pool, ws_idx)
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

                let ws_id = monitors[mon_idx].view.ids()[ws_idx];
                Self::add_tile_on(
                    monitors,
                    pool,
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
                );
            }
        }

        // needed because empty_workspace_above_first could have modified the idx.
        // Find the (id, geo) pair first, then borrow `&mut Workspace<W>` once via the
        // pool — the iterator can't escape a mutable borrow through the closure.
        let geo_pairs: Vec<_> = monitors[mon_idx]
            .workspaces_with_render_geo_ids(false)
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
            for mon in self.monitors_mut() {
                mon.dnd_scroll_gesture_begin();
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

        for mon in self.monitors_mut() {
            mon.dnd_scroll_gesture_end();
        }

        for ws in self.workspaces_mut() {
            ws.dnd_scroll_gesture_end();
        }
    }

    pub fn interactive_resize_begin(&mut self, window: W::Id, edges: ResizeEdge) -> bool {
        let pool = &mut self.workspaces;
        for mon in &mut self.monitors {
            for id in mon.view.ids() {
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

        let pool = &mut self.workspaces;
        for mon in &mut self.monitors {
            for id in mon.view.ids() {
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

        let pool = &mut self.workspaces;
        for mon in &mut self.monitors {
            for id in mon.view.ids() {
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

    pub fn move_workspace_down(&mut self) {
        if self.monitors.is_empty() {
            return;
        }
        let active_monitor_idx = self.active_monitor_idx;
        Self::move_workspace_down_on(&mut self.monitors, &mut self.workspaces, active_monitor_idx);
    }

    pub fn move_workspace_up(&mut self) {
        if self.monitors.is_empty() {
            return;
        }
        let active_monitor_idx = self.active_monitor_idx;
        Self::move_workspace_up_on(&mut self.monitors, &mut self.workspaces, active_monitor_idx);
    }

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
            (mon_idx, self.monitors[mon_idx].view.active_position())
        };

        Self::move_workspace_to_idx_on(
            &mut self.monitors,
            &mut self.workspaces,
            mon_idx,
            old_idx,
            new_idx,
        );
    }

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

        // if `empty_workspace_above_first` is set and `ws` is the first
        // workspace on a monitor, another empty workspace needs to
        // be added before.
        // Conversely, if `ws` was the last workspace on a monitor, an
        // empty workspace needs to be added after.

        if !self.monitors.is_empty() {
            let mon_idx = self.active_monitor_idx;
            let monitor = &self.monitors[mon_idx];
            let add_top = monitor.options.layout.empty_workspace_above_first
                && monitor.view.ids().first().is_some_and(|id| *id == wsid);
            let add_bottom = monitor.view.ids().last().is_some_and(|id| *id == wsid);
            let (monitors, pool) = (&mut self.monitors[..], &mut self.workspaces);
            if add_top {
                Self::add_workspace_top_on(monitors, pool, mon_idx);
            }
            if add_bottom {
                Self::add_workspace_bottom_on(monitors, pool, mon_idx);
            }
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
        for mon in &mut self.monitors {
            mon.overview_open = self.overview_open;
            mon.set_overview_progress(self.overview_progress.as_ref());
        }
    }

    pub fn toggle_overview(&mut self) {
        self.overview_open = !self.overview_open;

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
            self.monitors[self.active_monitor_idx]
                .activate_workspace_with_anim_config(ws_idx, Some(config));
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

        let pool = &mut self.workspaces;
        for mon in &mut self.monitors {
            for (id, geo) in mon.workspaces_with_render_geo_ids(false) {
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

        let pool = &mut self.workspaces;
        for mon in &mut self.monitors {
            for id in mon.view.ids() {
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
                let pool = &mut self.workspaces;
                let Some(mon) = self.monitors.iter_mut().find(|m| m.output == output) else {
                    return;
                };
                let ctx = LayoutCtx::new(&*pool, &mon.view);
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

        let pool = &mut self.workspaces;
        for mon in &mut self.monitors {
            for id in mon.view.ids() {
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

    pub fn refresh(&mut self, is_active: bool) {
        let _span = tracy_client::span!("Layout::refresh");

        self.is_active = is_active;

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

        let pool = &mut self.workspaces;
        let active_monitor_idx = self.active_monitor_idx;
        for (idx, mon) in self.monitors.iter_mut().enumerate() {
            let is_active = self.is_active
                && idx == active_monitor_idx
                && !matches!(self.interactive_move, Some(InteractiveMoveState::Moving(_)));

            if ongoing_scrolling_dnd.is_some() && self.overview_open {
                // Begin the scroll on new monitors and when opening the overview.
                mon.dnd_scroll_gesture_begin();
            } else if !self.overview_open {
                mon.dnd_scroll_gesture_end();
            }

            let active_pos = mon.view.active_position();
            for (ws_idx, id) in mon.view.ids().to_vec().into_iter().enumerate() {
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

        // Cross-field invariants (pool ↔ view ids ↔ monitor indices, plus every
        // per-workspace/column/tile invariant) are only encoded as runtime
        // convention, not in the type system. Run the full chain once per
        // refresh tick so drift from any mutation path — activity switches and
        // other Phase 1a additions included — trips an assert in debug builds
        // immediately, rather than surfacing later as a corrupt render or a
        // mysterious panic deep inside a read path. Release builds (which
        // disable `debug_assertions`) skip this entirely.
        #[cfg(debug_assertions)]
        self.verify_invariants();
    }

    pub fn workspaces(
        &self,
    ) -> impl Iterator<Item = (Option<&Monitor<W>>, usize, &Workspace<W>)> + '_ {
        let pool = &self.workspaces;
        let iter_monitors = self.monitors.iter().flat_map(move |mon| {
            mon.view.ids().iter().enumerate().map(move |(idx, id)| {
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

    pub fn has_window(&self, window: &W::Id) -> bool {
        self.windows().any(|(_, win)| win.id() == window)
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
