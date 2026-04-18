use std::cmp::min;
use std::collections::HashMap;
use std::iter::zip;
use std::marker::PhantomData;
use std::rc::Rc;
use std::time::Duration;

use niri_config::{CornerRadius, LayoutPart};
use smithay::backend::renderer::element::utils::{
    CropRenderElement, Relocate, RelocateRenderElement, RescaleRenderElement,
};
use smithay::output::Output;
use smithay::utils::{Logical, Point, Rectangle, Size};

use super::activity::WorkspaceView;
use super::insert_hint_element::{InsertHintElement, InsertHintRenderElement};
use super::scrolling::{Column, ColumnWidth};
use super::tile::Tile;
use super::workspace::{
    compute_working_area, OutputId, Workspace, WorkspaceAddWindowTarget, WorkspaceId,
    WorkspaceRenderElement,
};
use super::{compute_overview_zoom, ActivateWindow, HitType, LayoutElement, Options};
use crate::animation::{Animation, Clock};
use crate::input::swipe_tracker::SwipeTracker;
use crate::niri_render_elements;
use crate::render_helpers::renderer::NiriRenderer;
use crate::render_helpers::shadow::ShadowRenderElement;
use crate::render_helpers::solid_color::SolidColorRenderElement;
use crate::render_helpers::xray::XrayPos;
use crate::render_helpers::RenderCtx;
use crate::rubber_band::RubberBand;
use crate::utils::transaction::Transaction;
use crate::utils::{
    output_size, round_logical_in_physical, round_logical_in_physical_max1, ResizeEdge,
};

/// Amount of touchpad movement to scroll the height of one workspace.
const WORKSPACE_GESTURE_MOVEMENT: f64 = 300.;

const WORKSPACE_GESTURE_RUBBER_BAND: RubberBand = RubberBand {
    stiffness: 0.5,
    limit: 0.05,
};

/// Amount of DnD edge scrolling to scroll the height of one workspace.
///
/// This constant is tied to the default dnd-edge-workspace-switch max-speed setting.
const WORKSPACE_DND_EDGE_SCROLL_MOVEMENT: f64 = 1500.;

#[derive(Debug)]
pub struct Monitor<W: LayoutElement> {
    /// Output for this monitor.
    pub(super) output: Output,
    /// Cached name of the output.
    output_name: String,
    /// Latest known scale for this output.
    scale: smithay::output::Scale,
    /// Latest known size for this output.
    view_size: Size<f64, Logical>,
    /// Latest known working area for this output.
    ///
    /// Not rounded to physical pixels.
    // FIXME: since this is used for things like DnD scrolling edges in the overview, ideally this
    // should only consider overlay and top layer-shell surfaces. However, Smithay doesn't easily
    // let you do this at the moment.
    working_area: Rectangle<f64, Logical>,
    /// Active / previous cursor and visual ordering of workspaces for this monitor.
    ///
    /// Workspace values live in `Layout.workspaces`; `view.ids()` is the
    /// authoritative ordering for this monitor. `view` is never empty.
    pub(super) view: WorkspaceView,
    /// Witness for `W`; workspace values live in the pool.
    _phantom: PhantomData<W>,
    /// In-progress switch between workspaces.
    pub(super) workspace_switch: Option<WorkspaceSwitch>,
    /// Indication where an interactively-moved window is about to be placed.
    pub(super) insert_hint: Option<InsertHint>,
    /// Insert hint element for rendering.
    insert_hint_element: InsertHintElement,
    /// Location to render the insert hint element.
    insert_hint_render_loc: Option<InsertHintRenderLoc>,
    /// Whether the overview is open.
    pub(super) overview_open: bool,
    /// Progress of the overview zoom animation, 1 is fully in overview.
    overview_progress: Option<OverviewProgress>,
    /// Clock for driving animations.
    pub(super) clock: Clock,
    /// Configurable properties of the layout as received from the parent layout.
    pub(super) base_options: Rc<Options>,
    /// Configurable properties of the layout.
    pub(super) options: Rc<Options>,
    /// Layout config overrides for this monitor.
    layout_config: Option<niri_config::LayoutPart>,
}

#[derive(Debug)]
pub enum WorkspaceSwitch {
    Animation(Animation),
    Gesture(WorkspaceSwitchGesture),
}

#[derive(Debug)]
pub struct WorkspaceSwitchGesture {
    /// Index of the workspace where the gesture was started.
    center_idx: usize,
    /// Fractional workspace index where the gesture was started.
    ///
    /// Can differ from center_idx when starting a gesture in the middle between workspaces, for
    /// example by "catching" an animation.
    start_idx: f64,
    /// Current, fractional workspace index.
    pub(super) current_idx: f64,
    /// Animation for the extra offset to the current position.
    ///
    /// For example, if there's a workspace switch during a DnD scroll.
    animation: Option<Animation>,
    tracker: SwipeTracker,
    /// Whether the gesture is controlled by the touchpad.
    is_touchpad: bool,
    /// Whether the gesture is clamped to +-1 workspace around the center.
    is_clamped: bool,

    // If this gesture is for drag-and-drop scrolling, this is the last event's unadjusted
    // timestamp.
    dnd_last_event_time: Option<Duration>,
    // Time when the drag-and-drop scroll delta became non-zero, used for debouncing.
    //
    // If `None` then the scroll delta is currently zero.
    dnd_nonzero_start_time: Option<Duration>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum InsertPosition {
    NewColumn(usize),
    InColumn(usize, usize),
    Floating,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum InsertWorkspace {
    Existing(WorkspaceId),
    NewAt(usize),
}

#[derive(Debug)]
pub(super) struct InsertHint {
    pub workspace: InsertWorkspace,
    pub position: InsertPosition,
    pub corner_radius: CornerRadius,
}

#[derive(Debug, Clone, Copy)]
struct InsertHintRenderLoc {
    workspace: InsertWorkspace,
    location: Point<f64, Logical>,
}

#[derive(Debug)]
pub(super) enum OverviewProgress {
    Animation(Animation),
    Value(f64),
}

/// Where to put a newly added window.
#[derive(Debug, Default, PartialEq, Eq)]
pub enum MonitorAddWindowTarget<'a, W: LayoutElement> {
    /// No particular preference.
    #[default]
    Auto,
    /// On this workspace.
    Workspace {
        /// Id of the target workspace.
        id: WorkspaceId,
        /// Override where the window will open as a new column.
        column_idx: Option<usize>,
    },
    /// Next to this existing window.
    NextTo(&'a W::Id),
}

impl<'a, W: LayoutElement> Copy for MonitorAddWindowTarget<'a, W> {}

impl<'a, W: LayoutElement> Clone for MonitorAddWindowTarget<'a, W> {
    fn clone(&self) -> Self {
        *self
    }
}

niri_render_elements! {
    MonitorInnerRenderElement<R> => {
        Workspace = CropRenderElement<WorkspaceRenderElement<R>>,
        InsertHint = CropRenderElement<InsertHintRenderElement>,
        UncroppedInsertHint = InsertHintRenderElement,
        Shadow = ShadowRenderElement,
        SolidColor = SolidColorRenderElement,
    }
}

pub type MonitorRenderElement<R> =
    RelocateRenderElement<RescaleRenderElement<MonitorInnerRenderElement<R>>>;

impl WorkspaceSwitch {
    pub fn current_idx(&self) -> f64 {
        match self {
            WorkspaceSwitch::Animation(anim) => anim.value(),
            WorkspaceSwitch::Gesture(gesture) => {
                gesture.current_idx + gesture.animation.as_ref().map_or(0., |anim| anim.value())
            }
        }
    }

    pub fn target_idx(&self) -> f64 {
        match self {
            WorkspaceSwitch::Animation(anim) => anim.to(),
            WorkspaceSwitch::Gesture(gesture) => gesture.current_idx,
        }
    }

    pub fn offset(&mut self, delta: isize) {
        match self {
            WorkspaceSwitch::Animation(anim) => anim.offset(delta as f64),
            WorkspaceSwitch::Gesture(gesture) => {
                if delta >= 0 {
                    gesture.center_idx += delta as usize;
                } else {
                    gesture.center_idx -= (-delta) as usize;
                }
                gesture.start_idx += delta as f64;
                gesture.current_idx += delta as f64;
            }
        }
    }

    fn is_animation_ongoing(&self) -> bool {
        match self {
            WorkspaceSwitch::Animation(_) => true,
            WorkspaceSwitch::Gesture(gesture) => gesture.animation.is_some(),
        }
    }
}

impl WorkspaceSwitchGesture {
    fn min_max(&self, workspace_count: usize) -> (f64, f64) {
        if self.is_clamped {
            let min = self.center_idx.saturating_sub(1) as f64;
            let max = (self.center_idx + 1).min(workspace_count - 1) as f64;
            (min, max)
        } else {
            (0., (workspace_count - 1) as f64)
        }
    }

    fn animate_from(&mut self, from: f64, clock: Clock, config: niri_config::Animation) {
        let current = self.animation.as_ref().map_or(0., Animation::value);
        self.animation = Some(Animation::new(clock, from + current, 0., 0., config));
    }
}

impl InsertWorkspace {
    fn existing_id(self) -> Option<WorkspaceId> {
        match self {
            InsertWorkspace::Existing(id) => Some(id),
            InsertWorkspace::NewAt(_) => None,
        }
    }
}

impl OverviewProgress {
    pub fn value(&self) -> f64 {
        match self {
            OverviewProgress::Animation(anim) => anim.value(),
            OverviewProgress::Value(v) => *v,
        }
    }

    pub fn clamped_value(&self) -> f64 {
        match self {
            OverviewProgress::Animation(anim) => anim.clamped_value(),
            OverviewProgress::Value(v) => *v,
        }
    }
}

impl From<&super::OverviewProgress> for OverviewProgress {
    fn from(value: &super::OverviewProgress) -> Self {
        match value {
            super::OverviewProgress::Animation(anim) => Self::Animation(anim.clone()),
            super::OverviewProgress::Gesture(gesture) => Self::Value(gesture.value),
            super::OverviewProgress::Open => Self::Value(1.),
        }
    }
}

impl<W: LayoutElement> Monitor<W> {
    /// Build a monitor owning `workspace_ids`.
    ///
    /// All `workspace_ids` must already be keys in `pool`. `Monitor::new` binds each of them to
    /// `output`, syncs config, then inserts empty bookend workspace(s) into the pool (top and/or
    /// bottom per `options.layout.empty_workspace_above_first`). The resulting `view.ids()` is
    /// `[optional_top_empty, ...workspace_ids_in_order..., bottom_empty]`.
    pub fn new(
        output: Output,
        workspace_ids: Vec<WorkspaceId>,
        ws_id_to_activate: Option<WorkspaceId>,
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        clock: Clock,
        base_options: Rc<Options>,
        layout_config: Option<LayoutPart>,
    ) -> Self {
        let options =
            Rc::new(Options::clone(&base_options).with_merged_layout(layout_config.as_ref()));

        let scale = output.current_scale();
        let view_size = output_size(&output);
        let working_area = compute_working_area(&output);

        let mut active_workspace_idx = 0;

        for (idx, id) in workspace_ids.iter().enumerate() {
            let ws = pool
                .get_mut(id)
                .expect("workspace_ids must be keys in the pool");
            assert!(ws.has_windows_or_name());

            ws.bind_output(&output);
            ws.update_config(options.clone());

            if ws_id_to_activate.is_some_and(|want| *id == want) {
                active_workspace_idx = idx;
            }
        }

        let mut ids = workspace_ids;

        if options.layout.empty_workspace_above_first && !ids.is_empty() {
            let ws = Workspace::new(&output, clock.clone(), options.clone());
            let id = ws.id();
            assert!(pool.insert(id, ws).is_none(), "fresh id must be unique");
            ids.insert(0, id);
            active_workspace_idx += 1;
        }

        let bottom = Workspace::new(&output, clock.clone(), options.clone());
        let bottom_id = bottom.id();
        assert!(
            pool.insert(bottom_id, bottom).is_none(),
            "fresh id must be unique",
        );
        ids.push(bottom_id);

        let view = WorkspaceView::new(ids, active_workspace_idx);

        Self {
            output_name: output.name(),
            output,
            scale,
            view_size,
            working_area,
            view,
            _phantom: PhantomData,
            insert_hint: None,
            insert_hint_element: InsertHintElement::new(options.layout.insert_hint),
            insert_hint_render_loc: None,
            overview_open: false,
            overview_progress: None,
            workspace_switch: None,
            clock,
            base_options,
            options,
            layout_config,
        }
    }

    /// Drop this monitor. Non-empty workspaces stay in the pool (unbound from this monitor's
    /// output); empty unnamed workspaces (typically the bookends added by `Monitor::new`) are
    /// removed from the pool since no caller needs them back. Returns the retained ids in view
    /// order.
    pub fn into_workspace_ids(
        self,
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
    ) -> Vec<WorkspaceId> {
        let mut kept = Vec::with_capacity(self.view.ids().len());
        for id in self.view.ids() {
            let ws = pool
                .get_mut(id)
                .expect("monitor ids must be keys in the pool");
            if ws.has_windows_or_name() {
                ws.unbind_output(&self.output);
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

    /// Borrow the workspace this monitor displays at position `pos`.
    ///
    /// Panics if `pos` is out of bounds for `self.view.ids()` or if the id is absent from `pool`;
    /// both indicate a broken pool/view invariant, not user error.
    pub fn workspace_at<'a>(
        &self,
        pool: &'a HashMap<WorkspaceId, Workspace<W>>,
        pos: usize,
    ) -> &'a Workspace<W> {
        let id = self.view.ids()[pos];
        pool.get(&id).expect("view id must be a key in the pool")
    }

    /// Mutably borrow the workspace this monitor displays at position `pos`.
    ///
    /// Panics on the same conditions as [`workspace_at`](Self::workspace_at).
    pub fn workspace_at_mut<'a>(
        &self,
        pool: &'a mut HashMap<WorkspaceId, Workspace<W>>,
        pos: usize,
    ) -> &'a mut Workspace<W> {
        let id = self.view.ids()[pos];
        pool.get_mut(&id)
            .expect("view id must be a key in the pool")
    }

    /// Number of workspaces this monitor displays, including empty bookends.
    pub fn workspaces_len(&self) -> usize {
        self.view.len()
    }

    pub fn output(&self) -> &Output {
        &self.output
    }

    pub fn output_name(&self) -> &String {
        &self.output_name
    }

    pub fn active_workspace_idx(&self) -> usize {
        self.view.active_position()
    }

    pub fn active_workspace_ref<'a>(
        &self,
        pool: &'a HashMap<WorkspaceId, Workspace<W>>,
    ) -> &'a Workspace<W> {
        self.workspace_at(pool, self.view.active_position())
    }

    pub fn find_named_workspace<'a>(
        &self,
        pool: &'a HashMap<WorkspaceId, Workspace<W>>,
        workspace_name: &str,
    ) -> Option<&'a Workspace<W>> {
        self.view.ids().iter().find_map(|id| {
            let ws = pool.get(id).expect("view id must be a key in the pool");
            ws.name
                .as_ref()
                .is_some_and(|name| name.eq_ignore_ascii_case(workspace_name))
                .then_some(ws)
        })
    }

    pub fn find_named_workspace_index(
        &self,
        pool: &HashMap<WorkspaceId, Workspace<W>>,
        workspace_name: &str,
    ) -> Option<usize> {
        self.view.ids().iter().position(|id| {
            pool.get(id)
                .expect("view id must be a key in the pool")
                .name
                .as_ref()
                .is_some_and(|name| name.eq_ignore_ascii_case(workspace_name))
        })
    }

    pub fn active_workspace<'a>(
        &self,
        pool: &'a mut HashMap<WorkspaceId, Workspace<W>>,
    ) -> &'a mut Workspace<W> {
        self.workspace_at_mut(pool, self.view.active_position())
    }

    pub fn windows<'a>(
        &'a self,
        pool: &'a HashMap<WorkspaceId, Workspace<W>>,
    ) -> impl Iterator<Item = &'a W> + 'a {
        self.view
            .ids()
            .iter()
            .map(move |id| pool.get(id).expect("view id must be a key in the pool"))
            .flat_map(|ws| ws.windows())
    }

    pub fn has_window(&self, pool: &HashMap<WorkspaceId, Workspace<W>>, window: &W::Id) -> bool {
        self.windows(pool).any(|win| win.id() == window)
    }

    pub fn add_workspace_at(&mut self, pool: &mut HashMap<WorkspaceId, Workspace<W>>, idx: usize) {
        let ws = Workspace::new(&self.output, self.clock.clone(), self.options.clone());

        let id = ws.id();
        assert!(pool.insert(id, ws).is_none(), "fresh id must be unique");
        self.view.insert(idx, id);

        if let Some(switch) = &mut self.workspace_switch {
            if idx as f64 <= switch.target_idx() {
                switch.offset(1);
            }
        }
    }

    pub fn add_workspace_top(&mut self, pool: &mut HashMap<WorkspaceId, Workspace<W>>) {
        self.add_workspace_at(pool, 0);
    }

    pub fn add_workspace_bottom(&mut self, pool: &mut HashMap<WorkspaceId, Workspace<W>>) {
        self.add_workspace_at(pool, self.view.len());
    }

    pub fn activate_workspace(&mut self, idx: usize) {
        self.activate_workspace_with_anim_config(idx, None);
    }

    pub fn activate_workspace_with_anim_config(
        &mut self,
        idx: usize,
        config: Option<niri_config::Animation>,
    ) {
        // FIXME: also compute and use current velocity.
        let current_idx = self.workspace_render_idx();

        let changed = self.view.activate(idx);

        let config = config.unwrap_or(self.options.animations.workspace_switch.0);

        match &mut self.workspace_switch {
            // During a DnD scroll, we want to visually animate even if idx matches the active idx.
            Some(WorkspaceSwitch::Gesture(gesture)) if gesture.dnd_last_event_time.is_some() => {
                gesture.center_idx = idx;

                // Adjust start_idx to make current_idx point at idx.
                let current_pos = gesture.current_idx - gesture.start_idx;
                gesture.start_idx = idx as f64 - current_pos;
                let prev_current_idx = gesture.current_idx;
                gesture.current_idx = idx as f64;

                let current_idx_delta = gesture.current_idx - prev_current_idx;
                gesture.animate_from(-current_idx_delta, self.clock.clone(), config);
            }
            _ => {
                // Don't animate if nothing changed.
                if !changed {
                    return;
                }

                self.workspace_switch = Some(WorkspaceSwitch::Animation(Animation::new(
                    self.clock.clone(),
                    current_idx,
                    idx as f64,
                    0.,
                    config,
                )));
            }
        }
    }

    pub(super) fn resolve_add_window_target<'a>(
        &self,
        pool: &HashMap<WorkspaceId, Workspace<W>>,
        target: MonitorAddWindowTarget<'a, W>,
    ) -> (usize, WorkspaceAddWindowTarget<'a, W>) {
        match target {
            MonitorAddWindowTarget::Auto => {
                (self.view.active_position(), WorkspaceAddWindowTarget::Auto)
            }
            MonitorAddWindowTarget::Workspace { id, column_idx } => {
                let idx = self
                    .view
                    .position_of(id)
                    .expect("workspace id must be on this monitor");
                let target = if let Some(column_idx) = column_idx {
                    WorkspaceAddWindowTarget::NewColumnAt(column_idx)
                } else {
                    WorkspaceAddWindowTarget::Auto
                };
                (idx, target)
            }
            MonitorAddWindowTarget::NextTo(win_id) => {
                let idx = self
                    .view
                    .ids()
                    .iter()
                    .position(|id| {
                        pool.get(id)
                            .expect("view id must be a key in the pool")
                            .has_window(win_id)
                    })
                    .unwrap();
                (idx, WorkspaceAddWindowTarget::NextTo(win_id))
            }
        }
    }

    pub fn add_window(
        &mut self,
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        window: W,
        target: MonitorAddWindowTarget<W>,
        activate: ActivateWindow,
        width: ColumnWidth,
        is_full_width: bool,
        is_floating: bool,
    ) {
        // Currently, everything a workspace sets on a Tile is the same across all workspaces of a
        // monitor. So we can use any workspace, not necessarily the exact target workspace.
        let tile = self.workspace_at_mut(pool, 0).make_tile(window);

        self.add_tile(
            pool,
            tile,
            target,
            activate,
            true,
            width,
            is_full_width,
            is_floating,
        );
    }

    pub fn add_column(
        &mut self,
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        mut workspace_idx: usize,
        column: Column<W>,
        activate: bool,
    ) {
        let workspace = self.workspace_at_mut(pool, workspace_idx);

        workspace.add_column(Some(&self.output), column, activate);

        // After adding a new window, workspace becomes this output's own.
        if workspace.name().is_none() {
            workspace.output_id = Some(OutputId::new(&self.output));
        }

        if workspace_idx == self.view.len() - 1 {
            self.add_workspace_bottom(pool);
        }
        if self.options.layout.empty_workspace_above_first && workspace_idx == 0 {
            self.add_workspace_top(pool);
            workspace_idx += 1;
        }

        if activate {
            self.activate_workspace(workspace_idx);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_tile(
        &mut self,
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        tile: Tile<W>,
        target: MonitorAddWindowTarget<W>,
        activate: ActivateWindow,
        // FIXME: Refactor ActivateWindow enum to make this better.
        allow_to_activate_workspace: bool,
        width: ColumnWidth,
        is_full_width: bool,
        is_floating: bool,
    ) {
        let (mut workspace_idx, target) = self.resolve_add_window_target(pool, target);

        let workspace = self.workspace_at_mut(pool, workspace_idx);

        workspace.add_tile(
            Some(&self.output),
            tile,
            target,
            activate,
            width,
            is_full_width,
            is_floating,
        );

        // After adding a new window, workspace becomes this output's own.
        if workspace.name().is_none() {
            workspace.output_id = Some(OutputId::new(&self.output));
        }

        if workspace_idx == self.view.len() - 1 {
            // Insert a new empty workspace.
            self.add_workspace_bottom(pool);
        }

        if self.options.layout.empty_workspace_above_first && workspace_idx == 0 {
            self.add_workspace_top(pool);
            workspace_idx += 1;
        }

        if allow_to_activate_workspace && activate.map_smart(|| false) {
            self.activate_workspace(workspace_idx);
        }
    }

    pub fn add_tile_to_column(
        &mut self,
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        workspace_idx: usize,
        column_idx: usize,
        tile_idx: Option<usize>,
        tile: Tile<W>,
        activate: bool,
        // FIXME: Refactor ActivateWindow enum to make this better.
        allow_to_activate_workspace: bool,
    ) {
        let workspace = self.workspace_at_mut(pool, workspace_idx);

        workspace.add_tile_to_column(Some(&self.output), column_idx, tile_idx, tile, activate);

        // After adding a new window, workspace becomes this output's own.
        if workspace.name().is_none() {
            workspace.output_id = Some(OutputId::new(&self.output));
        }

        // Since we're adding window to an existing column, the workspace isn't empty, and
        // therefore cannot be the last one, so we never need to insert a new empty workspace.

        if allow_to_activate_workspace && activate {
            self.activate_workspace(workspace_idx);
        }
    }

    pub fn clean_up_workspaces(&mut self, pool: &mut HashMap<WorkspaceId, Workspace<W>>) {
        assert!(self.workspace_switch.is_none());

        let range_start = if self.options.layout.empty_workspace_above_first {
            1
        } else {
            0
        };
        for idx in (range_start..self.view.len() - 1).rev() {
            if self.view.active_position() == idx {
                continue;
            }

            if !self.workspace_at(pool, idx).has_windows_or_name() {
                let id = self.view.ids()[idx];
                self.view.remove_at(idx);
                assert!(
                    pool.remove(&id).is_some(),
                    "view id must be a key in the pool",
                );
            }
        }

        // Special case handling when empty_workspace_above_first is set and all workspaces
        // are empty.
        if self.options.layout.empty_workspace_above_first && self.view.len() == 2 {
            assert!(!self.workspace_at(pool, 0).has_windows_or_name());
            assert!(!self.workspace_at(pool, 1).has_windows_or_name());
            let id = self.view.ids()[1];
            self.view.remove_at(1);
            assert!(
                pool.remove(&id).is_some(),
                "view id must be a key in the pool",
            );
        }
    }

    pub fn unname_workspace(
        &mut self,
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        id: WorkspaceId,
    ) -> bool {
        if !self.view.ids().contains(&id) {
            return false;
        }

        pool.get_mut(&id)
            .expect("view id must be a key in the pool")
            .unname();

        if self.workspace_switch.is_none() {
            self.clean_up_workspaces(pool);
        }

        true
    }

    /// Remove the workspace at `idx` from this monitor's view, unbind it from the output, and
    /// return its id. The workspace value remains in `pool` under that id — caller decides whether
    /// to re-attach it to another monitor or drop it.
    pub fn remove_workspace_by_idx(
        &mut self,
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        mut idx: usize,
    ) -> WorkspaceId {
        if idx == self.view.len() - 1 {
            self.add_workspace_bottom(pool);
        }
        if self.options.layout.empty_workspace_above_first && idx == 0 {
            self.add_workspace_top(pool);
            idx += 1;
        }

        // For monitor current workspace removal, we focus previous rather than next (<= rather
        // than <). This is different from columns and tiles, but it lets move-workspace-to-monitor
        // back and forth to preserve position. `WorkspaceView::remove_at` enforces this rule.
        let id = self.view.ids()[idx];
        self.view.remove_at(idx);

        pool.get_mut(&id)
            .expect("view id must be a key in the pool")
            .unbind_output(&self.output);

        self.workspace_switch = None;
        self.clean_up_workspaces(pool);

        id
    }

    /// Attach an existing workspace to this monitor at position `idx`.
    ///
    /// The workspace value must already be a key in `pool`. Binds the workspace to this output,
    /// refreshes its config, inserts `id` into the view (adding a top empty bookend first if
    /// `empty_workspace_above_first` is on), optionally activates it, clears any in-flight
    /// `workspace_switch`, and runs `clean_up_workspaces`.
    pub fn insert_workspace(
        &mut self,
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        id: WorkspaceId,
        mut idx: usize,
        activate: bool,
    ) {
        let ws = pool
            .get_mut(&id)
            .expect("workspace id must be a key in the pool");
        ws.bind_output(&self.output);
        ws.update_config(self.options.clone());

        // Don't insert past the last empty workspace.
        if idx == self.view.len() {
            idx -= 1;
        }
        if idx == 0 && self.options.layout.empty_workspace_above_first {
            // Insert a new empty workspace on top to prepare for insertion of new workspace.
            self.add_workspace_top(pool);
            idx += 1;
        }

        self.view.insert(idx, id);

        if activate {
            self.workspace_switch = None;
            self.activate_workspace(idx);
        }

        self.workspace_switch = None;
        self.clean_up_workspaces(pool);
    }

    /// Attach a list of existing pool-held workspaces to this monitor, in order, just above the
    /// bottom empty workspace.
    ///
    /// All `workspace_ids` must already be keys in `pool`.
    pub fn append_workspaces(
        &mut self,
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        workspace_ids: Vec<WorkspaceId>,
    ) {
        if workspace_ids.is_empty() {
            return;
        }

        for id in &workspace_ids {
            let ws = pool
                .get_mut(id)
                .expect("workspace id must be a key in the pool");
            ws.bind_output(&self.output);
            ws.update_config(self.options.clone());
        }

        let empty_was_focused = self.view.active_position() == self.view.len() - 1;

        // Insert in place so the view stays non-empty at every step
        // (`WorkspaceView` requires at least one id).
        let start = self.view.len() - 1;
        for (offset, id) in workspace_ids.into_iter().enumerate() {
            let insert_pos = start + offset;
            self.view.insert(insert_pos, id);
        }

        // If empty_workspace_above_first is set and the first workspace is now no longer empty,
        // add a new empty workspace on top.
        if self.options.layout.empty_workspace_above_first
            && self.workspace_at(pool, 0).has_windows_or_name()
        {
            self.add_workspace_top(pool);
        }

        // If the empty workspace was focused on the primary monitor, keep it focused.
        // Use `set_active_at` (not `activate`) so `previous` isn't clobbered — this is
        // an output reshuffle, not a user-visible workspace switch.
        if empty_was_focused {
            self.view.set_active_at(self.view.len() - 1);
        }

        // FIXME: if we're adding workspaces to currently invisible positions
        // (outside the workspace switch), we don't need to cancel it.
        self.workspace_switch = None;
        self.clean_up_workspaces(pool);
    }

    pub fn move_down_or_to_workspace_down(
        &mut self,
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
    ) {
        if !self.active_workspace(pool).move_down() {
            self.move_to_workspace_down(pool, true);
        }
    }

    pub fn move_up_or_to_workspace_up(&mut self, pool: &mut HashMap<WorkspaceId, Workspace<W>>) {
        if !self.active_workspace(pool).move_up() {
            self.move_to_workspace_up(pool, true);
        }
    }

    pub fn focus_window_or_workspace_down(
        &mut self,
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
    ) {
        if !self.active_workspace(pool).focus_down() {
            self.switch_workspace_down();
        }
    }

    pub fn focus_window_or_workspace_up(&mut self, pool: &mut HashMap<WorkspaceId, Workspace<W>>) {
        if !self.active_workspace(pool).focus_up() {
            self.switch_workspace_up();
        }
    }

    pub fn move_to_workspace_up(
        &mut self,
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        focus: bool,
    ) {
        let source_workspace_idx = self.view.active_position();

        let new_idx = source_workspace_idx.saturating_sub(1);
        if new_idx == source_workspace_idx {
            return;
        }
        let new_id = self.view.ids()[new_idx];

        let workspace = self.workspace_at_mut(pool, source_workspace_idx);
        let Some(removed) = workspace.remove_active_tile(Some(&self.output), Transaction::new())
        else {
            return;
        };

        let activate = if focus {
            ActivateWindow::Yes
        } else {
            ActivateWindow::Smart
        };

        self.add_tile(
            pool,
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

    pub fn move_to_workspace_down(
        &mut self,
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        focus: bool,
    ) {
        let source_workspace_idx = self.view.active_position();

        let new_idx = min(source_workspace_idx + 1, self.view.len() - 1);
        if new_idx == source_workspace_idx {
            return;
        }
        let new_id = self.view.ids()[new_idx];

        let workspace = self.workspace_at_mut(pool, source_workspace_idx);
        let Some(removed) = workspace.remove_active_tile(Some(&self.output), Transaction::new())
        else {
            return;
        };

        let activate = if focus {
            ActivateWindow::Yes
        } else {
            ActivateWindow::Smart
        };

        self.add_tile(
            pool,
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

    pub fn move_to_workspace(
        &mut self,
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        window: Option<&W::Id>,
        idx: usize,
        activate: ActivateWindow,
    ) {
        let source_workspace_idx = if let Some(window) = window {
            self.view
                .ids()
                .iter()
                .position(|id| {
                    pool.get(id)
                        .expect("view id must be a key in the pool")
                        .has_window(window)
                })
                .unwrap()
        } else {
            self.view.active_position()
        };

        let new_idx = min(idx, self.view.len() - 1);
        if new_idx == source_workspace_idx {
            return;
        }
        let new_id = self.view.ids()[new_idx];

        let active_window_id = self.active_window(pool).map(|win| win.id().clone());
        let activate =
            activate.map_smart(|| window.is_none_or(|win| active_window_id.as_ref() == Some(win)));

        let workspace = self.workspace_at_mut(pool, source_workspace_idx);
        let transaction = Transaction::new();
        let removed = if let Some(window) = window {
            workspace.remove_tile(Some(&self.output), window, transaction)
        } else if let Some(removed) = workspace.remove_active_tile(Some(&self.output), transaction)
        {
            removed
        } else {
            return;
        };

        self.add_tile(
            pool,
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

        if self.workspace_switch.is_none() {
            self.clean_up_workspaces(pool);
        }
    }

    pub fn move_column_to_workspace_up(
        &mut self,
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        activate: bool,
    ) {
        let source_workspace_idx = self.view.active_position();

        let new_idx = source_workspace_idx.saturating_sub(1);
        if new_idx == source_workspace_idx {
            return;
        }

        let workspace = self.workspace_at_mut(pool, source_workspace_idx);
        if workspace.floating_is_active() {
            self.move_to_workspace_up(pool, activate);
            return;
        }

        let Some(column) = workspace.remove_active_column(Some(&self.output)) else {
            return;
        };

        self.add_column(pool, new_idx, column, activate);
    }

    pub fn move_column_to_workspace_down(
        &mut self,
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        activate: bool,
    ) {
        let source_workspace_idx = self.view.active_position();

        let new_idx = min(source_workspace_idx + 1, self.view.len() - 1);
        if new_idx == source_workspace_idx {
            return;
        }

        let workspace = self.workspace_at_mut(pool, source_workspace_idx);
        if workspace.floating_is_active() {
            self.move_to_workspace_down(pool, activate);
            return;
        }

        let Some(column) = workspace.remove_active_column(Some(&self.output)) else {
            return;
        };

        self.add_column(pool, new_idx, column, activate);
    }

    pub fn move_column_to_workspace(
        &mut self,
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        idx: usize,
        activate: bool,
    ) {
        let source_workspace_idx = self.view.active_position();

        let new_idx = min(idx, self.view.len() - 1);
        if new_idx == source_workspace_idx {
            return;
        }

        let workspace = self.workspace_at_mut(pool, source_workspace_idx);
        if workspace.floating_is_active() {
            let activate = if activate {
                ActivateWindow::Smart
            } else {
                ActivateWindow::No
            };
            self.move_to_workspace(pool, None, idx, activate);
            return;
        }

        let Some(column) = workspace.remove_active_column(Some(&self.output)) else {
            return;
        };

        self.add_column(pool, new_idx, column, activate);
    }

    pub fn switch_workspace_up(&mut self) {
        let new_idx = match &self.workspace_switch {
            // During a DnD scroll, select the prev apparent workspace.
            Some(WorkspaceSwitch::Gesture(gesture)) if gesture.dnd_last_event_time.is_some() => {
                let current = gesture.current_idx;
                let new = current.ceil() - 1.;
                new.clamp(0., (self.view.len() - 1) as f64) as usize
            }
            _ => self.view.active_position().saturating_sub(1),
        };

        self.activate_workspace(new_idx);
    }

    pub fn switch_workspace_down(&mut self) {
        let new_idx = match &self.workspace_switch {
            // During a DnD scroll, select the next apparent workspace.
            Some(WorkspaceSwitch::Gesture(gesture)) if gesture.dnd_last_event_time.is_some() => {
                let current = gesture.current_idx;
                let new = current.floor() + 1.;
                new.clamp(0., (self.view.len() - 1) as f64) as usize
            }
            _ => min(self.view.active_position() + 1, self.view.len() - 1),
        };

        self.activate_workspace(new_idx);
    }

    pub fn switch_workspace(&mut self, idx: usize) {
        self.activate_workspace(min(idx, self.view.len() - 1));
    }

    pub fn switch_workspace_auto_back_and_forth(&mut self, idx: usize) {
        let idx = min(idx, self.view.len() - 1);

        if idx == self.view.active_position() {
            if let Some(prev_idx) = self.view.previous_position() {
                self.switch_workspace(prev_idx);
            }
        } else {
            self.switch_workspace(idx);
        }
    }

    pub fn switch_workspace_previous(&mut self) {
        if let Some(idx) = self.view.previous_position() {
            self.switch_workspace(idx);
        }
    }

    pub fn active_window<'a>(&self, pool: &'a HashMap<WorkspaceId, Workspace<W>>) -> Option<&'a W> {
        self.active_workspace_ref(pool).active_window()
    }

    pub fn advance_animations(&mut self, pool: &mut HashMap<WorkspaceId, Workspace<W>>) {
        match &mut self.workspace_switch {
            Some(WorkspaceSwitch::Animation(anim)) => {
                if anim.is_done() {
                    self.workspace_switch = None;
                    self.clean_up_workspaces(pool);
                }
            }
            Some(WorkspaceSwitch::Gesture(gesture)) => {
                // Make sure the last event time doesn't go too much out of date (for
                // monitors not under cursor), causing sudden jumps.
                //
                // This happens after any dnd_scroll_gesture_scroll() calls (in
                // Layout::advance_animations()), so it doesn't mess up the time delta there.
                if let Some(last_time) = &mut gesture.dnd_last_event_time {
                    let now = self.clock.now_unadjusted();
                    if *last_time != now {
                        *last_time = now;

                        // If last_time was already == now, then dnd_scroll_gesture_scroll() must've
                        // updated the gesture already. Therefore, when this code runs, the pointer
                        // must be outside the DnD scrolling zone.
                        gesture.dnd_nonzero_start_time = None;
                    }
                }

                if let Some(anim) = &mut gesture.animation {
                    if anim.is_done() {
                        gesture.animation = None;
                    }
                }
            }
            None => (),
        }

        for id in self.view.ids() {
            pool.get_mut(id)
                .expect("view id must be a key in the pool")
                .advance_animations();
        }
    }

    pub(super) fn are_animations_ongoing(&self, pool: &HashMap<WorkspaceId, Workspace<W>>) -> bool {
        self.workspace_switch
            .as_ref()
            .is_some_and(|s| s.is_animation_ongoing())
            || self
                .view
                .ids()
                .iter()
                .map(|id| pool.get(id).expect("view id must be a key in the pool"))
                .any(|ws| ws.are_animations_ongoing())
    }

    pub fn are_transitions_ongoing(&self, pool: &HashMap<WorkspaceId, Workspace<W>>) -> bool {
        self.workspace_switch.is_some()
            || self
                .view
                .ids()
                .iter()
                .map(|id| pool.get(id).expect("view id must be a key in the pool"))
                .any(|ws| ws.are_transitions_ongoing())
    }

    pub fn update_render_elements(
        &mut self,
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        is_active: bool,
    ) {
        let mut insert_hint_ws_geo = None;
        let insert_hint_ws_id = self
            .insert_hint
            .as_ref()
            .and_then(|hint| hint.workspace.existing_id());

        for (id, geo) in self
            .workspaces_with_render_geo_ids(true)
            .collect::<Vec<_>>()
        {
            let ws = pool
                .get_mut(&id)
                .expect("view id must be a key in the pool");
            ws.update_render_elements(is_active);

            if Some(id) == insert_hint_ws_id {
                insert_hint_ws_geo = Some(geo);
            }
        }

        self.insert_hint_render_loc = None;
        if let Some(hint) = &self.insert_hint {
            match hint.workspace {
                InsertWorkspace::Existing(ws_id) => {
                    if let Some(ws) = pool
                        .get(&ws_id)
                        .filter(|_| self.view.ids().contains(&ws_id))
                    {
                        if let Some(mut area) = ws.insert_hint_area(hint.position) {
                            let scale = ws.scale().fractional_scale();
                            let view_size = ws.view_size();

                            // Make sure the hint is at least partially visible.
                            if matches!(hint.position, InsertPosition::NewColumn(_)) {
                                let zoom = self.overview_zoom();
                                let geo = insert_hint_ws_geo.unwrap();
                                let geo = geo.downscale(zoom);

                                area.loc.x = area.loc.x.max(-geo.loc.x - area.size.w / 2.);
                                area.loc.x =
                                    area.loc.x.min(geo.loc.x + geo.size.w - area.size.w / 2.);
                            }

                            // Round to physical pixels.
                            area = area.to_physical_precise_round(scale).to_logical(scale);

                            let view_rect = Rectangle::new(area.loc.upscale(-1.), view_size);
                            self.insert_hint_element.update_render_elements(
                                area.size,
                                view_rect,
                                hint.corner_radius,
                                scale,
                            );
                            self.insert_hint_render_loc = Some(InsertHintRenderLoc {
                                workspace: hint.workspace,
                                location: area.loc,
                            });
                        }
                    } else {
                        error!("insert hint workspace missing from monitor");
                    }
                }
                InsertWorkspace::NewAt(ws_idx) => {
                    let scale = self.scale.fractional_scale();
                    let zoom = self.overview_zoom();
                    let gap = self.workspace_gap(zoom);

                    let hint_gap = round_logical_in_physical(scale, gap * 0.1);
                    let hint_height = gap - hint_gap * 2.;

                    let next_ws_geo = self.workspaces_render_geo().nth(ws_idx).unwrap();
                    let hint_width = round_logical_in_physical(scale, next_ws_geo.size.w * 0.75);
                    let hint_x =
                        round_logical_in_physical(scale, (next_ws_geo.size.w - hint_width) / 2.);

                    let hint_loc_diff = Point::from((-hint_x, hint_height + hint_gap));
                    let hint_loc = next_ws_geo.loc - hint_loc_diff;
                    let hint_size = Size::from((hint_width, hint_height));

                    // Sometimes the hint ends up 1 px wider than necessary and/or 1 px
                    // narrower than necessary. The values here seem correct. Might have to do with
                    // how zooming out currently doesn't round to output scale properly.

                    // Compute view rect as if we're above the next workspace (rather than below
                    // the previous one).
                    let view_rect = Rectangle::new(hint_loc_diff, next_ws_geo.size);

                    self.insert_hint_element.update_render_elements(
                        hint_size,
                        view_rect,
                        CornerRadius::default(),
                        scale,
                    );
                    self.insert_hint_render_loc = Some(InsertHintRenderLoc {
                        workspace: hint.workspace,
                        location: hint_loc,
                    });
                }
            }
        }
    }

    pub fn update_config(
        &mut self,
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        base_options: Rc<Options>,
    ) {
        let options =
            Rc::new(Options::clone(&base_options).with_merged_layout(self.layout_config.as_ref()));

        if self.options.layout.empty_workspace_above_first
            != options.layout.empty_workspace_above_first
            && self.view.len() > 1
        {
            if options.layout.empty_workspace_above_first {
                self.add_workspace_top(pool);
            } else if self.workspace_switch.is_none() && self.view.active_position() != 0 {
                let id = self.view.ids()[0];
                self.view.remove_at(0);
                assert!(
                    pool.remove(&id).is_some(),
                    "view id must be a key in the pool",
                );
            }
        }

        for id in self.view.ids() {
            pool.get_mut(id)
                .expect("view id must be a key in the pool")
                .update_config(options.clone());
        }

        self.insert_hint_element
            .update_config(options.layout.insert_hint);

        self.base_options = base_options;
        self.options = options;
    }

    pub fn update_layout_config(
        &mut self,
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        layout_config: Option<niri_config::LayoutPart>,
    ) -> bool {
        if self.layout_config == layout_config {
            return false;
        }

        self.layout_config = layout_config;
        self.update_config(pool, self.base_options.clone());

        true
    }

    pub fn update_shaders(&mut self, pool: &mut HashMap<WorkspaceId, Workspace<W>>) {
        for id in self.view.ids() {
            pool.get_mut(id)
                .expect("view id must be a key in the pool")
                .update_shaders();
        }

        self.insert_hint_element.update_shaders();
    }

    pub fn update_output_size(&mut self, pool: &mut HashMap<WorkspaceId, Workspace<W>>) {
        self.scale = self.output.current_scale();
        self.view_size = output_size(&self.output);
        self.working_area = compute_working_area(&self.output);

        for id in self.view.ids() {
            pool.get_mut(id)
                .expect("view id must be a key in the pool")
                .update_output_size(&self.output);
        }
    }

    pub fn move_workspace_down(&mut self, pool: &mut HashMap<WorkspaceId, Workspace<W>>) {
        let active_idx = self.view.active_position();
        let mut new_idx = min(active_idx + 1, self.view.len() - 1);
        if new_idx == active_idx {
            return;
        }

        self.view.swap(active_idx, new_idx);

        if new_idx == self.view.len() - 1 {
            // Insert a new empty workspace.
            self.add_workspace_bottom(pool);
        }

        if self.options.layout.empty_workspace_above_first && active_idx == 0 {
            self.add_workspace_top(pool);
            new_idx += 1;
        }

        let previous_workspace_id = self.view.previous();
        self.activate_workspace(new_idx);
        self.workspace_switch = None;
        self.view.set_previous(previous_workspace_id);

        self.clean_up_workspaces(pool);
    }

    pub fn move_workspace_up(&mut self, pool: &mut HashMap<WorkspaceId, Workspace<W>>) {
        let active_idx = self.view.active_position();
        let mut new_idx = active_idx.saturating_sub(1);
        if new_idx == active_idx {
            return;
        }

        self.view.swap(active_idx, new_idx);

        if active_idx == self.view.len() - 1 {
            // Insert a new empty workspace.
            self.add_workspace_bottom(pool);
        }

        if self.options.layout.empty_workspace_above_first && new_idx == 0 {
            self.add_workspace_top(pool);
            new_idx += 1;
        }

        let previous_workspace_id = self.view.previous();
        self.activate_workspace(new_idx);
        self.workspace_switch = None;
        self.view.set_previous(previous_workspace_id);

        self.clean_up_workspaces(pool);
    }

    pub fn move_workspace_to_idx(
        &mut self,
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        old_idx: usize,
        new_idx: usize,
    ) {
        if self.view.len() <= old_idx {
            return;
        }

        let new_idx = new_idx.clamp(0, self.view.len() - 1);
        if old_idx == new_idx {
            return;
        }

        self.view.move_within(old_idx, new_idx);

        if new_idx > old_idx {
            if new_idx == self.view.len() - 1 {
                // Insert a new empty workspace.
                self.add_workspace_bottom(pool);
            }

            if self.options.layout.empty_workspace_above_first && old_idx == 0 {
                self.add_workspace_top(pool);
            }
        } else {
            if old_idx == self.view.len() - 1 {
                // Insert a new empty workspace.
                self.add_workspace_bottom(pool);
            }

            if self.options.layout.empty_workspace_above_first && new_idx == 0 {
                self.add_workspace_top(pool);
            }
        }

        self.workspace_switch = None;

        self.clean_up_workspaces(pool);
    }

    /// Returns the geometry of the active window relative to and clamped to the output.
    ///
    /// During animations, assumes the final view position.
    pub fn active_window_visual_rectangle(
        &self,
        pool: &HashMap<WorkspaceId, Workspace<W>>,
    ) -> Option<Rectangle<f64, Logical>> {
        if self.overview_open {
            return None;
        }

        self.active_workspace_ref(pool)
            .active_window_visual_rectangle()
    }

    fn workspace_size(&self, zoom: f64) -> Size<f64, Logical> {
        let ws_size = self.view_size.upscale(zoom);
        let scale = self.scale.fractional_scale();
        ws_size.to_physical_precise_ceil(scale).to_logical(scale)
    }

    fn workspace_gap(&self, zoom: f64) -> f64 {
        let scale = self.scale.fractional_scale();
        let gap = self.view_size.h * 0.1 * zoom;
        round_logical_in_physical_max1(scale, gap)
    }

    fn workspace_size_with_gap(&self, zoom: f64) -> Size<f64, Logical> {
        let gap = self.workspace_gap(zoom);
        self.workspace_size(zoom) + Size::from((0., gap))
    }

    pub fn overview_zoom(&self) -> f64 {
        let progress = self.overview_progress.as_ref().map(|p| p.value());
        compute_overview_zoom(&self.options, progress)
    }

    pub(super) fn set_overview_progress(&mut self, progress: Option<&super::OverviewProgress>) {
        let prev_render_idx = self.workspace_render_idx();
        self.overview_progress = progress.map(OverviewProgress::from);
        let new_render_idx = self.workspace_render_idx();

        // If the view jumped (can happen when going from corrected to uncorrected render_idx, for
        // example when toggling the overview in the middle of an overview animation), then restart
        // the workspace switch to avoid jumps.
        if prev_render_idx != new_render_idx {
            if let Some(WorkspaceSwitch::Animation(anim)) = &mut self.workspace_switch {
                // FIXME: maintain velocity.
                *anim = anim.restarted(prev_render_idx, anim.to(), 0.);
            }
        }
    }

    #[cfg(test)]
    pub(super) fn overview_progress_value(&self) -> Option<f64> {
        self.overview_progress.as_ref().map(|p| p.value())
    }

    pub fn workspace_render_idx(&self) -> f64 {
        // If workspace switch and overview progress are matching animations, then compute a
        // correction term to make the movement appear monotonic.
        if let (
            Some(WorkspaceSwitch::Animation(switch_anim)),
            Some(OverviewProgress::Animation(progress_anim)),
        ) = (&self.workspace_switch, &self.overview_progress)
        {
            if switch_anim.start_time() == progress_anim.start_time()
                && (switch_anim.duration().as_secs_f64() - progress_anim.duration().as_secs_f64())
                    .abs()
                    <= 0.001
            {
                #[rustfmt::skip]
                // How this was derived:
                //
                // - Assume we're animating a zoom + switch. Consider switch "from" and "to".
                //   These are render_idx values, so first workspace to second would have switch
                //   from = 0. and to = 1. regardless of the zoom level.
                //
                // - At the start, the point at "from" is at Y = 0. We're moving the point at "to"
                //   to Y = 0. We want this to be a monotonic motion in apparent coordinates (after
                //   zoom).
                //
                // - Height at the start:
                //   from_height = (size.h + gap) * from_zoom.
                //
                // - Current height:
                //   current_height = (size.h + gap) * zoom.
                //
                // - We're moving the "to" point to Y = 0:
                //   to_y = 0.
                //
                // - The initial position of the point we're moving:
                //   from_y = (to - from) * from_height.
                //
                // - We want this point to travel monotonically in apparent coordinates:
                //   current_y = from_y + (to_y - from_y) * progress,
                //   where progress is from 0 to 1, equals to the animation progress (switch and
                //   zoom are the same since they are synchronized).
                //
                // - Derive the Y of the first workspace from this:
                //   first_y = current_y - to * current_height.
                //
                // Now, let's substitute and rearrange the terms.
                //
                // - current_y = from_y + (0 - (to - from) * from_height) * progress
                // - progress = (switch_anim.value() - from) / (to - from)
                // - current_y = from_y - (to - from) * from_height * (switch_anim.value() - from) / (to - from)
                // - current_y = from_y - from_height * (switch_anim.value() - from)
                // - first_y = from_y - from_height * (switch_anim.value() - from) - to * current_height
                // - first_y = (to - from) * from_height - from_height * (switch_anim.value() - from) - to * current_height
                // - first_y = to * from_height - switch_anim.value() * from_height - to * current_height
                // - first_y = -switch_anim.value() * from_height + to * (from_height - current_height)
                let from = progress_anim.from();
                let from_zoom = compute_overview_zoom(&self.options, Some(from));
                let from_ws_height_with_gap = self.workspace_size_with_gap(from_zoom).h;

                let zoom = self.overview_zoom();
                let ws_height_with_gap = self.workspace_size_with_gap(zoom).h;

                let first_ws_y = -switch_anim.value() * from_ws_height_with_gap
                    + switch_anim.to() * (from_ws_height_with_gap - ws_height_with_gap);

                return -first_ws_y / ws_height_with_gap;
            }
        };

        if let Some(switch) = &self.workspace_switch {
            switch.current_idx()
        } else {
            self.view.active_position() as f64
        }
    }

    pub fn workspaces_render_geo(&self) -> impl Iterator<Item = Rectangle<f64, Logical>> {
        let scale = self.scale.fractional_scale();
        let zoom = self.overview_zoom();

        let ws_size = self.workspace_size(zoom);
        let gap = self.workspace_gap(zoom);
        let ws_height_with_gap = ws_size.h + gap;

        let static_offset = (self.view_size.to_point() - ws_size.to_point()).downscale(2.);
        let static_offset = static_offset
            .to_physical_precise_round(scale)
            .to_logical(scale);

        let first_ws_y = -self.workspace_render_idx() * ws_height_with_gap;
        let first_ws_y = round_logical_in_physical(scale, first_ws_y);

        // Return position for one-past-last workspace too.
        (0..=self.view.len()).map(move |idx| {
            let y = first_ws_y + idx as f64 * ws_height_with_gap;
            let loc = Point::from((0., y)) + static_offset;

            // Even though all components that go into loc are rounded to physical pixels, the
            // floating point addition may lose precision. This can result for example in the
            // current workspace having y = 0.0000000000002 and thus missing pointer hits at the
            // monitor edge with y = 0. So, post-round the location too.
            let loc = loc.to_physical_precise_round(scale).to_logical(scale);

            Rectangle::new(loc, ws_size)
        })
    }

    pub fn workspaces_with_render_geo<'a>(
        &'a self,
        pool: &'a HashMap<WorkspaceId, Workspace<W>>,
    ) -> impl Iterator<Item = (&'a Workspace<W>, Rectangle<f64, Logical>)> + 'a {
        let output_geo = Rectangle::from_size(self.view_size);

        let geo = self.workspaces_render_geo();
        let ids = self.view.ids();
        zip(ids.iter(), geo)
            .map(move |(id, geo)| {
                (
                    pool.get(id).expect("view id must be a key in the pool"),
                    geo,
                )
            })
            // Cull out workspaces outside the output.
            .filter(move |(_ws, geo)| geo.intersection(output_geo).is_some())
    }

    pub fn workspaces_with_render_geo_idx<'a>(
        &'a self,
        pool: &'a HashMap<WorkspaceId, Workspace<W>>,
    ) -> impl Iterator<Item = ((usize, &'a Workspace<W>), Rectangle<f64, Logical>)> + 'a {
        let output_geo = Rectangle::from_size(self.view_size);

        let geo = self.workspaces_render_geo();
        let ids = self.view.ids();
        zip(ids.iter().enumerate(), geo)
            .map(move |((idx, id), geo)| {
                (
                    (
                        idx,
                        pool.get(id).expect("view id must be a key in the pool"),
                    ),
                    geo,
                )
            })
            // Cull out workspaces outside the output.
            .filter(move |(_ws, geo)| geo.intersection(output_geo).is_some())
    }

    /// Same shape as [`workspaces_with_render_geo`](Self::workspaces_with_render_geo) but yields
    /// ids instead of workspace references. Does not borrow the pool; callers thread their own
    /// `&mut pool` borrow through each yielded id.
    pub fn workspaces_with_render_geo_ids<'a>(
        &'a self,
        cull: bool,
    ) -> impl Iterator<Item = (WorkspaceId, Rectangle<f64, Logical>)> + 'a {
        let output_geo = Rectangle::from_size(self.view_size);

        let geo = self.workspaces_render_geo();
        let ids = self.view.ids();
        zip(ids.iter().copied(), geo)
            // Cull out workspaces outside the output.
            .filter(move |(_id, geo)| !cull || geo.intersection(output_geo).is_some())
    }

    pub fn workspace_under<'a>(
        &'a self,
        pool: &'a HashMap<WorkspaceId, Workspace<W>>,
        pos_within_output: Point<f64, Logical>,
    ) -> Option<(&'a Workspace<W>, Rectangle<f64, Logical>)> {
        let (ws, geo) = self
            .workspaces_with_render_geo(pool)
            .find_map(|(ws, geo)| {
                // Extend width to entire output.
                let loc = Point::from((0., geo.loc.y));
                let size = Size::from((self.view_size.w, geo.size.h));
                let bounds = Rectangle::new(loc, size);

                bounds.contains(pos_within_output).then_some((ws, geo))
            })?;
        Some((ws, geo))
    }

    pub fn workspace_under_narrow<'a>(
        &'a self,
        pool: &'a HashMap<WorkspaceId, Workspace<W>>,
        pos_within_output: Point<f64, Logical>,
    ) -> Option<&'a Workspace<W>> {
        self.workspaces_with_render_geo(pool)
            .find_map(|(ws, geo)| geo.contains(pos_within_output).then_some(ws))
    }

    pub fn window_under<'a>(
        &'a self,
        pool: &'a HashMap<WorkspaceId, Workspace<W>>,
        pos_within_output: Point<f64, Logical>,
    ) -> Option<(&'a W, HitType)> {
        let (ws, geo) = self.workspace_under(pool, pos_within_output)?;

        if self.overview_progress.is_some() {
            let zoom = self.overview_zoom();
            let pos_within_workspace = (pos_within_output - geo.loc).downscale(zoom);
            let (win, hit) = ws.window_under(pos_within_workspace)?;
            // During the overview animation, we cannot do input hits because we cannot really
            // represent scaled windows properly.
            Some((win, hit.to_activate()))
        } else {
            let (win, hit) = ws.window_under(pos_within_output - geo.loc)?;
            Some((win, hit.offset_win_pos(geo.loc)))
        }
    }

    pub fn resize_edges_under(
        &self,
        pool: &HashMap<WorkspaceId, Workspace<W>>,
        pos_within_output: Point<f64, Logical>,
    ) -> Option<ResizeEdge> {
        if self.overview_progress.is_some() {
            return None;
        }

        let (ws, geo) = self.workspace_under(pool, pos_within_output)?;
        ws.resize_edges_under(pos_within_output - geo.loc)
    }

    pub(super) fn insert_position(
        &self,
        pool: &HashMap<WorkspaceId, Workspace<W>>,
        pos_within_output: Point<f64, Logical>,
    ) -> (InsertWorkspace, Rectangle<f64, Logical>) {
        let mut iter = self.workspaces_with_render_geo_idx(pool);

        let dummy = Rectangle::default();

        // Monitors always have at least one workspace.
        let ((idx, ws), geo) = iter.next().unwrap();

        // Check if above first.
        if pos_within_output.y < geo.loc.y {
            return (InsertWorkspace::NewAt(idx), dummy);
        }

        let contains = move |geo: Rectangle<f64, Logical>| {
            geo.loc.y <= pos_within_output.y && pos_within_output.y < geo.loc.y + geo.size.h
        };

        // Check first.
        if contains(geo) {
            return (InsertWorkspace::Existing(ws.id()), geo);
        }

        let mut last_geo = geo;
        let mut last_idx = idx;
        for ((idx, ws), geo) in iter {
            // Check gap above.
            let gap_loc = Point::from((last_geo.loc.x, last_geo.loc.y + last_geo.size.h));
            let gap_size = Size::from((geo.size.w, geo.loc.y - gap_loc.y));
            let gap_geo = Rectangle::new(gap_loc, gap_size);
            if contains(gap_geo) {
                return (InsertWorkspace::NewAt(idx), dummy);
            }

            // Check workspace itself.
            if contains(geo) {
                return (InsertWorkspace::Existing(ws.id()), geo);
            }

            last_geo = geo;
            last_idx = idx;
        }

        // Anything below.
        (InsertWorkspace::NewAt(last_idx + 1), dummy)
    }

    pub fn render_above_top_layer(&self, pool: &HashMap<WorkspaceId, Workspace<W>>) -> bool {
        // Render above the top layer only if the view is stationary.
        if self.workspace_switch.is_some() || self.overview_progress.is_some() {
            return false;
        }

        self.active_workspace_ref(pool).render_above_top_layer()
    }

    pub fn render_insert_hint_between_workspaces<R: NiriRenderer>(
        &self,
        renderer: &mut R,
        push: &mut dyn FnMut(MonitorRenderElement<R>),
    ) {
        if self.options.layout.insert_hint.off {
            return;
        }
        let Some(render_loc) = self.insert_hint_render_loc else {
            return;
        };
        let InsertWorkspace::NewAt(_) = render_loc.workspace else {
            return;
        };

        self.insert_hint_element
            .render(renderer, render_loc.location, &mut |elem| {
                let elem = MonitorInnerRenderElement::UncroppedInsertHint(elem);
                let elem = RescaleRenderElement::from_element(elem, Point::default(), 1.);
                let elem =
                    RelocateRenderElement::from_element(elem, Point::default(), Relocate::Relative);
                push(elem);
            });
    }

    pub fn render_workspaces<R: NiriRenderer>(
        &self,
        pool: &HashMap<WorkspaceId, Workspace<W>>,
        mut ctx: RenderCtx<R>,
        focus_ring: bool,
        push: &mut dyn FnMut(MonitorRenderElement<R>),
    ) {
        let _span = tracy_client::span!("Monitor::render_workspaces");

        let scale = self.scale.fractional_scale();
        // Ceil the height in physical pixels.
        let height = (self.view_size.h * scale).ceil() as i32;

        // Crop the elements to prevent them overflowing, currently visible during a workspace
        // switch.
        //
        // HACK: crop to infinite bounds at least horizontally where we
        // know there's no workspace joining or monitor bounds, otherwise
        // it will cut pixel shaders and mess up the coordinate space.
        // There's also a damage tracking bug which causes glitched
        // rendering for maximized GTK windows.
        //
        // FIXME: use proper bounds after fixing the Crop element.
        let crop_bounds = if self.workspace_switch.is_some() || self.overview_progress.is_some() {
            Rectangle::new(
                Point::from((-i32::MAX / 2, 0)),
                Size::from((i32::MAX, height)),
            )
        } else {
            Rectangle::new(
                Point::from((-i32::MAX / 2, -i32::MAX / 2)),
                Size::from((i32::MAX, i32::MAX)),
            )
        };

        let zoom = self.overview_zoom();

        let insert_hint_render_loc = self
            .insert_hint_render_loc
            .filter(|_| !self.options.layout.insert_hint.off);

        let scale_relocate = move |geo: Rectangle<f64, Logical>, elem| {
            let elem = RescaleRenderElement::from_element(elem, Point::from((0, 0)), zoom);
            RelocateRenderElement::from_element(
                elem,
                // The offset we get from workspaces_with_render_geo() is already
                // rounded to physical pixels, but it's in the logical coordinate
                // space, so we need to convert it to physical.
                geo.loc.to_physical_precise_round(scale),
                Relocate::Relative,
            )
        };

        for (ws, geo) in self.workspaces_with_render_geo(pool) {
            // Macro instead of closure because ws and insert hint have different elem types.
            macro_rules! push {
                () => {{
                    &mut |elem| {
                        let elem = CropRenderElement::from_element(elem, scale, crop_bounds);
                        if let Some(elem) = elem {
                            let elem = MonitorInnerRenderElement::from(elem);
                            push(scale_relocate(geo, elem));
                        }
                    }
                }};
            }

            let xray_pos = XrayPos::new(geo.loc, zoom);

            ws.render_floating(ctx.r(), xray_pos, focus_ring, push!());

            if let Some(loc) = insert_hint_render_loc {
                if loc.workspace == InsertWorkspace::Existing(ws.id()) {
                    self.insert_hint_element
                        .render(ctx.renderer, loc.location, push!());
                }
            }

            ws.render_scrolling(ctx.r(), xray_pos, focus_ring, push!());
        }
    }

    pub fn render_workspace_shadows<R: NiriRenderer>(
        &self,
        pool: &HashMap<WorkspaceId, Workspace<W>>,
        renderer: &mut R,
        push: &mut dyn FnMut(MonitorRenderElement<R>),
    ) {
        let Some(progress) = self.overview_progress.as_ref().map(|p| p.clamped_value()) else {
            return;
        };
        let alpha = progress.clamp(0., 1.) as f32;

        let _span = tracy_client::span!("Monitor::render_workspace_shadows");

        let scale = self.scale.fractional_scale();
        let zoom = self.overview_zoom();

        for (ws, geo) in self.workspaces_with_render_geo(pool) {
            ws.render_shadow(renderer, &mut |elem| {
                let elem = elem.with_alpha(alpha);
                let elem = MonitorInnerRenderElement::Shadow(elem);
                let elem = RescaleRenderElement::from_element(elem, Point::from((0, 0)), zoom);
                let elem = RelocateRenderElement::from_element(
                    elem,
                    geo.loc.to_physical_precise_round(scale),
                    Relocate::Relative,
                );
                push(elem);
            });
        }
    }

    pub fn workspace_switch_gesture_begin(&mut self, is_touchpad: bool) {
        let center_idx = self.view.active_position();
        let current_idx = self.workspace_render_idx();

        let gesture = WorkspaceSwitchGesture {
            center_idx,
            start_idx: current_idx,
            current_idx,
            animation: None,
            tracker: SwipeTracker::new(),
            is_touchpad,
            is_clamped: !self.overview_open,
            dnd_last_event_time: None,
            dnd_nonzero_start_time: None,
        };
        self.workspace_switch = Some(WorkspaceSwitch::Gesture(gesture));
    }

    pub fn dnd_scroll_gesture_begin(&mut self) {
        if let Some(WorkspaceSwitch::Gesture(WorkspaceSwitchGesture {
            dnd_last_event_time: Some(_),
            ..
        })) = &self.workspace_switch
        {
            // Already active.
            return;
        }

        if !self.overview_open {
            // This gesture is only for the overview.
            return;
        }

        let center_idx = self.view.active_position();
        let current_idx = self.workspace_render_idx();

        let gesture = WorkspaceSwitchGesture {
            center_idx,
            start_idx: current_idx,
            current_idx,
            animation: None,
            tracker: SwipeTracker::new(),
            is_touchpad: false,
            is_clamped: false,
            dnd_last_event_time: Some(self.clock.now_unadjusted()),
            dnd_nonzero_start_time: None,
        };
        self.workspace_switch = Some(WorkspaceSwitch::Gesture(gesture));
    }

    pub fn workspace_switch_gesture_update(
        &mut self,
        delta_y: f64,
        timestamp: Duration,
        is_touchpad: bool,
    ) -> Option<bool> {
        let Some(WorkspaceSwitch::Gesture(gesture)) = &self.workspace_switch else {
            return None;
        };

        if gesture.is_touchpad != is_touchpad || gesture.dnd_last_event_time.is_some() {
            return None;
        }

        let zoom = self.overview_zoom();
        let total_height = if gesture.is_touchpad {
            WORKSPACE_GESTURE_MOVEMENT
        } else {
            self.workspace_size_with_gap(1.).h
        };

        let Some(WorkspaceSwitch::Gesture(gesture)) = &mut self.workspace_switch else {
            return None;
        };

        // Reduce the effect of zoom on the touchpad somewhat.
        let delta_scale = if gesture.is_touchpad {
            (zoom - 1.) / 2.5 + 1.
        } else {
            zoom
        };

        let delta_y = delta_y / delta_scale;
        let mut rubber_band = WORKSPACE_GESTURE_RUBBER_BAND;
        rubber_band.limit /= zoom;

        gesture.tracker.push(delta_y, timestamp);

        let pos = gesture.tracker.pos() / total_height;

        let (min, max) = gesture.min_max(self.view.len());
        let new_idx = gesture.start_idx + pos;
        let new_idx = rubber_band.clamp(min, max, new_idx);

        if gesture.current_idx == new_idx {
            return Some(false);
        }

        gesture.current_idx = new_idx;
        Some(true)
    }

    pub fn dnd_scroll_gesture_scroll(&mut self, pos: Point<f64, Logical>, speed: f64) -> bool {
        let zoom = self.overview_zoom();

        let Some(WorkspaceSwitch::Gesture(gesture)) = &mut self.workspace_switch else {
            return false;
        };

        let Some(last_time) = gesture.dnd_last_event_time else {
            // Not a DnD scroll.
            return false;
        };

        let config = &self.options.gestures.dnd_edge_workspace_switch;
        let trigger_height = config.trigger_height;

        // Restrict the scrolling horizontally to the strip of workspaces to avoid unwanted trigger
        // after using the hot corner or during horizontal scroll.
        let width = self.view_size.w * zoom;
        let x = pos.x - (self.view_size.w - width) / 2.;

        // Consider the working area so layer-shell docks and such don't prevent scrolling.
        let y = pos.y - self.working_area.loc.y;
        let height = self.working_area.size.h;

        let y = y.clamp(0., height);
        let trigger_height = trigger_height.clamp(0., height / 2.);

        let delta = if x < 0. || width <= x {
            // Outside the bounds horizontally.
            0.
        } else if y < trigger_height {
            -(trigger_height - y)
        } else if height - y < trigger_height {
            trigger_height - (height - y)
        } else {
            0.
        };

        let delta = if trigger_height < 0.01 {
            // Sanity check for trigger-height 0 or small window sizes.
            0.
        } else {
            // Normalize to [0, 1].
            delta / trigger_height
        };
        let delta = delta * speed;

        let now = self.clock.now_unadjusted();
        gesture.dnd_last_event_time = Some(now);

        if delta == 0. {
            // We're outside the scrolling zone.
            gesture.dnd_nonzero_start_time = None;
            return false;
        }

        let nonzero_start = *gesture.dnd_nonzero_start_time.get_or_insert(now);

        // Delay starting the gesture a bit to avoid unwanted movement when dragging across
        // monitors.
        let delay = Duration::from_millis(u64::from(config.delay_ms));
        if now.saturating_sub(nonzero_start) < delay {
            return true;
        }

        let time_delta = now.saturating_sub(last_time).as_secs_f64();

        let delta = delta * time_delta * config.max_speed;

        gesture.tracker.push(delta, now);

        let total_height = WORKSPACE_DND_EDGE_SCROLL_MOVEMENT;
        let pos = gesture.tracker.pos() / total_height;
        let unclamped = gesture.start_idx + pos;

        let (min, max) = gesture.min_max(self.view.len());
        let clamped = unclamped.clamp(min, max);

        // Make sure that DnD scrolling too much outside the min/max does not "build up".
        gesture.start_idx += clamped - unclamped;
        gesture.current_idx = clamped;

        true
    }

    pub fn workspace_switch_gesture_end(&mut self, is_touchpad: Option<bool>) -> bool {
        let Some(WorkspaceSwitch::Gesture(gesture)) = &self.workspace_switch else {
            return false;
        };

        if is_touchpad.is_some_and(|x| gesture.is_touchpad != x) {
            return false;
        }

        let zoom = self.overview_zoom();
        let total_height = if gesture.dnd_last_event_time.is_some() {
            WORKSPACE_DND_EDGE_SCROLL_MOVEMENT
        } else if gesture.is_touchpad {
            WORKSPACE_GESTURE_MOVEMENT
        } else {
            self.workspace_size_with_gap(1.).h
        };

        let Some(WorkspaceSwitch::Gesture(gesture)) = &mut self.workspace_switch else {
            return false;
        };

        // Take into account any idle time between the last event and now.
        let now = self.clock.now_unadjusted();
        gesture.tracker.push(0., now);

        let mut rubber_band = WORKSPACE_GESTURE_RUBBER_BAND;
        rubber_band.limit /= zoom;

        let mut velocity = gesture.tracker.velocity() / total_height;
        let current_pos = gesture.tracker.pos() / total_height;
        let pos = gesture.tracker.projected_end_pos() / total_height;

        let (min, max) = gesture.min_max(self.view.len());
        let new_idx = gesture.start_idx + pos;

        let new_idx = new_idx.clamp(min, max);
        let new_idx = new_idx.round() as usize;

        velocity *= rubber_band.clamp_derivative(min, max, gesture.start_idx + current_pos);

        self.view.activate(new_idx);
        self.workspace_switch = Some(WorkspaceSwitch::Animation(Animation::new(
            self.clock.clone(),
            gesture.current_idx,
            new_idx as f64,
            velocity,
            self.options.animations.workspace_switch.0,
        )));

        true
    }

    pub fn dnd_scroll_gesture_end(&mut self) {
        if !matches!(
            self.workspace_switch,
            Some(WorkspaceSwitch::Gesture(WorkspaceSwitchGesture {
                dnd_last_event_time: Some(_),
                ..
            }))
        ) {
            // Not a DnD scroll.
            return;
        };

        self.workspace_switch_gesture_end(None);
    }

    pub fn scale(&self) -> smithay::output::Scale {
        self.scale
    }

    pub fn view_size(&self) -> Size<f64, Logical> {
        self.view_size
    }

    pub fn working_area(&self) -> Rectangle<f64, Logical> {
        self.working_area
    }

    pub fn layout_config(&self) -> Option<&niri_config::LayoutPart> {
        self.layout_config.as_ref()
    }

    #[cfg(test)]
    pub(super) fn verify_invariants(&self, pool: &HashMap<WorkspaceId, Workspace<W>>) {
        use approx::assert_abs_diff_eq;

        let options =
            Options::clone(&self.base_options).with_merged_layout(self.layout_config.as_ref());
        assert_eq!(&*self.options, &options);

        // `WorkspaceView::new` already enforces `!ids.is_empty()` and no mutator empties the
        // view, so `view.len() >= 1` is a `WorkspaceView` invariant — not re-asserted here.
        for (i, id) in self.view.ids().iter().enumerate() {
            assert!(
                pool.contains_key(id),
                "view.ids[{i}] must be a key in the workspace pool",
            );
        }
        assert!(self.view.active_position() < self.view.len());

        if let Some(WorkspaceSwitch::Animation(anim)) = &self.workspace_switch {
            let before_idx = anim.from() as usize;
            let after_idx = anim.to() as usize;

            assert!(before_idx < self.view.len());
            assert!(after_idx < self.view.len());
        }

        let ws = |id: WorkspaceId| -> &Workspace<W> {
            pool.get(&id).expect("view id must be a key in the pool")
        };

        let last_id = *self.view.ids().last().unwrap();
        let first_id = *self.view.ids().first().unwrap();

        assert!(
            !ws(last_id).has_windows(),
            "monitor must have an empty workspace in the end"
        );
        if self.options.layout.empty_workspace_above_first {
            assert!(
                !ws(first_id).has_windows(),
                "first workspace must be empty when empty_workspace_above_first is set"
            )
        }

        assert!(
            ws(last_id).name.is_none(),
            "monitor must have an unnamed workspace in the end"
        );
        if self.options.layout.empty_workspace_above_first {
            assert!(
                ws(first_id).name.is_none(),
                "first workspace must be unnamed when empty_workspace_above_first is set"
            )
        }

        if self.options.layout.empty_workspace_above_first {
            assert!(
                self.view.len() != 2,
                "if empty_workspace_above_first is set there must be just 1 or 3+ workspaces"
            )
        }

        // If there's no workspace switch in progress, there can't be any non-last non-active
        // empty workspaces. If empty_workspace_above_first is set then the first workspace
        // will be empty too.
        let pre_skip = if self.options.layout.empty_workspace_above_first {
            1
        } else {
            0
        };
        if self.workspace_switch.is_none() {
            for (idx, id) in self
                .view
                .ids()
                .iter()
                .enumerate()
                .skip(pre_skip)
                .rev()
                // skip last
                .skip(1)
            {
                if idx != self.view.active_position() {
                    assert!(
                        ws(*id).has_windows_or_name(),
                        "non-active workspace can't be empty and unnamed except the last one"
                    );
                }
            }
        }

        for id in self.view.ids() {
            let workspace = ws(*id);
            assert_eq!(self.clock, workspace.clock);

            assert_eq!(
                self.scale().integer_scale(),
                workspace.scale().integer_scale()
            );
            assert_eq!(
                self.scale().fractional_scale(),
                workspace.scale().fractional_scale()
            );
            assert_eq!(self.view_size, workspace.view_size());
            assert_eq!(self.working_area, workspace.working_area());

            assert_eq!(
                workspace.base_options, self.options,
                "workspace options must be synchronized with monitor"
            );
        }

        let scale = self.scale().fractional_scale();
        let iter = self.workspaces_with_render_geo(pool);
        for (_ws, ws_geo) in iter {
            let pos = ws_geo.loc;
            let rounded_pos = pos.to_physical_precise_round(scale).to_logical(scale);

            // Workspace positions must be rounded to physical pixels.
            assert_abs_diff_eq!(pos.x, rounded_pos.x, epsilon = 1e-5);
            assert_abs_diff_eq!(pos.y, rounded_pos.y, epsilon = 1e-5);
        }
    }
}
