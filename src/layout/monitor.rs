use std::cmp::min;
use std::collections::{HashMap, HashSet};
use std::iter::zip;
use std::marker::PhantomData;
use std::rc::Rc;
use std::time::Duration;

use jiji_config::{CornerRadius, LayoutPart};
use smithay::backend::renderer::element::utils::{
    CropRenderElement, Relocate, RelocateRenderElement, RescaleRenderElement,
};
use smithay::output::Output;
use smithay::utils::{Logical, Point, Rectangle, Size};

use super::activity::{ActivityId, WorkspaceView};
use super::insert_hint_element::{InsertHintElement, InsertHintRenderElement};
use super::workspace::{
    compute_working_area, OutputId, Workspace, WorkspaceAddWindowTarget, WorkspaceId,
    WorkspaceRenderElement,
};
use super::{compute_overview_zoom, HitType, LayoutCtx, LayoutElement, Options};
use crate::animation::{Animation, Clock};
use crate::input::swipe_tracker::SwipeTracker;
use crate::niri_render_elements;
use crate::render_helpers::renderer::NiriRenderer;
use crate::render_helpers::shadow::ShadowRenderElement;
use crate::render_helpers::solid_color::SolidColorRenderElement;
use crate::render_helpers::xray::XrayPos;
use crate::render_helpers::RenderCtx;
use crate::rubber_band::RubberBand;
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
    /// Witness for `W`; workspace values live in the pool.
    _phantom: PhantomData<W>,
    /// In-progress switch between workspaces.
    pub(super) workspace_switch: Option<WorkspaceSwitch>,
    /// In-progress switch between activities.
    pub(super) activity_switch: Option<ActivitySwitch>,
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
    layout_config: Option<jiji_config::LayoutPart>,
}

#[derive(Debug)]
pub enum WorkspaceSwitch {
    Animation(Animation),
    Gesture(WorkspaceSwitchGesture),
}

/// Which direction the incoming activity strip slides in from.
///
/// `Left` means the new strip enters from the right side of the screen (the user navigated
/// forward / to a higher-index activity); `Right` means it enters from the left (backward
/// navigation). Fixed at arm time and never changes mid-flight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlideDirection {
    Left,
    Right,
}

/// Which of the two activity strips a geometry/render call is computing for.
///
/// During an activity switch two strips coexist: the `Incoming` one (the newly active
/// activity's view, sliding into place) and the `Outgoing` one (the departing activity's
/// view, sliding away). They are horizontally offset by opposite amounts so they never
/// overlap. When no switch is in flight only `Incoming` is meaningful and its offset is 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivityStrip {
    Incoming,
    Outgoing,
}

/// In-progress switch between activities on one monitor.
///
/// Pure data carrier — holds the outgoing activity id, a 0→1 animation, and the
/// direction resolved at arm time. No renderer types are referenced here so the
/// state is coverable by the headless test harness.
///
/// The `anim` runs from 0.0 to 1.0 on the configured activity-switch curve.
/// `from` names the outgoing activity (the departing strip). `dir` encodes
/// which side the incoming strip enters from.
#[derive(Debug)]
pub struct ActivitySwitch {
    /// The outgoing activity id — the strip that is sliding away.
    pub(super) from: ActivityId,
    /// 0.0 → 1.0 animation on the configured activity-switch curve.
    pub(super) anim: Animation,
    /// Direction fixed at arm time: Left means the incoming strip enters from the right,
    /// Right means it enters from the left.
    pub(super) dir: SlideDirection,
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

    fn animate_from(&mut self, from: f64, clock: Clock, config: jiji_config::Animation) {
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
    /// Build a monitor that displays `workspace_ids` plus bookend workspaces.
    ///
    /// All `workspace_ids` must already be keys in `pool`. `Monitor::new` binds each of them to
    /// `output`, syncs config, then inserts empty bookend workspace(s) into the pool (top and/or
    /// bottom per `options.layout.empty_workspace_above_first`). Returns the new `Monitor` and
    /// a `WorkspaceView` whose `ids()` is `[optional_top_empty, ...workspace_ids_in_order...,
    /// bottom_empty]`; the caller stores the view in the active activity's `views` map.
    // Hits clippy::too_many_arguments (8/7). Every argument is load-bearing — splitting into
    // helper structs would obscure the call-site contract with `Layout::add_output`; the only
    // call site passes them all in from `Layout` fields. `seed_activity` is required by the
    // Ctor contract on `Workspace::new*`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        output: Output,
        workspace_ids: Vec<WorkspaceId>,
        ws_id_to_activate: Option<WorkspaceId>,
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        clock: Clock,
        base_options: Rc<Options>,
        layout_config: Option<LayoutPart>,
        seed_activity: ActivityId,
    ) -> (Self, WorkspaceView) {
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
            let ws = Workspace::new(
                &output,
                HashSet::from([seed_activity]),
                clock.clone(),
                options.clone(),
            );
            let id = ws.id();
            assert!(pool.insert(id, ws).is_none(), "fresh id must be unique");
            ids.insert(0, id);
            active_workspace_idx += 1;
        }

        let bottom = Workspace::new(
            &output,
            HashSet::from([seed_activity]),
            clock.clone(),
            options.clone(),
        );
        let bottom_id = bottom.id();
        assert!(
            pool.insert(bottom_id, bottom).is_none(),
            "fresh id must be unique",
        );
        ids.push(bottom_id);

        let view = WorkspaceView::new(ids, active_workspace_idx);

        let monitor = Self {
            output_name: output.name(),
            output,
            scale,
            view_size,
            working_area,
            _phantom: PhantomData,
            insert_hint: None,
            insert_hint_element: InsertHintElement::new(options.layout.insert_hint),
            insert_hint_render_loc: None,
            overview_open: false,
            overview_progress: None,
            workspace_switch: None,
            activity_switch: None,
            clock,
            base_options,
            options,
            layout_config,
        };
        (monitor, view)
    }

    /// Borrow the workspace this monitor displays at position `pos`.
    ///
    /// Panics if `pos` is out of bounds for `view.ids()` or if the id is absent from `pool`;
    /// both indicate a broken pool/view invariant, not user error.
    pub fn workspace_at<'a>(
        &self,
        pool: &'a HashMap<WorkspaceId, Workspace<W>>,
        view: &WorkspaceView,
        pos: usize,
    ) -> &'a Workspace<W> {
        let id = view.ids()[pos];
        pool.get(&id).expect("view id must be a key in the pool")
    }

    /// Mutably borrow the workspace this monitor displays at position `pos`.
    ///
    /// Panics on the same conditions as [`workspace_at`](Self::workspace_at).
    pub fn workspace_at_mut<'a>(
        &self,
        pool: &'a mut HashMap<WorkspaceId, Workspace<W>>,
        view: &WorkspaceView,
        pos: usize,
    ) -> &'a mut Workspace<W> {
        let id = view.ids()[pos];
        pool.get_mut(&id)
            .expect("view id must be a key in the pool")
    }

    /// Number of workspaces this monitor displays, including empty bookends.
    pub fn workspaces_len(view: &WorkspaceView) -> usize {
        view.len()
    }

    pub fn output(&self) -> &Output {
        &self.output
    }

    pub fn output_name(&self) -> &String {
        &self.output_name
    }

    /// Stable identifier of this monitor's output.
    ///
    /// Equivalent to `OutputId::new(self.output())`, exposed as a shortcut for
    /// callers that need to key lookups (e.g. `Layout::active_view`) by
    /// monitor identity without reaching through `output()` themselves.
    pub fn output_id(&self) -> OutputId {
        OutputId::new(&self.output)
    }

    pub fn active_workspace_idx(view: &WorkspaceView) -> usize {
        view.active_position()
    }

    pub fn active_workspace_ref<'a>(
        &self,
        pool: &'a HashMap<WorkspaceId, Workspace<W>>,
        view: &WorkspaceView,
    ) -> &'a Workspace<W> {
        self.workspace_at(pool, view, view.active_position())
    }

    pub fn find_named_workspace<'a>(
        &self,
        pool: &'a HashMap<WorkspaceId, Workspace<W>>,
        view: &WorkspaceView,
        workspace_name: &str,
    ) -> Option<&'a Workspace<W>> {
        view.ids().iter().find_map(|id| {
            let ws = pool.get(id).expect("view id must be a key in the pool");
            ws.name
                .as_ref()
                .is_some_and(|name| name.eq_ignore_ascii_case(workspace_name))
                .then_some(ws)
        })
    }

    pub fn find_named_workspace_index(
        pool: &HashMap<WorkspaceId, Workspace<W>>,
        view: &WorkspaceView,
        workspace_name: &str,
    ) -> Option<usize> {
        view.ids().iter().position(|id| {
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
        view: &WorkspaceView,
    ) -> &'a mut Workspace<W> {
        self.workspace_at_mut(pool, view, view.active_position())
    }

    pub fn windows<'a>(
        &'a self,
        pool: &'a HashMap<WorkspaceId, Workspace<W>>,
        view: &'a WorkspaceView,
    ) -> impl Iterator<Item = &'a W> + 'a {
        view.ids()
            .iter()
            .map(move |id| pool.get(id).expect("view id must be a key in the pool"))
            .flat_map(|ws| ws.windows())
    }

    pub fn has_window(
        &self,
        pool: &HashMap<WorkspaceId, Workspace<W>>,
        view: &WorkspaceView,
        window: &W::Id,
    ) -> bool {
        self.windows(pool, view).any(|win| win.id() == window)
    }

    pub fn activate_workspace(&mut self, view: &mut WorkspaceView, idx: usize) {
        self.activate_workspace_with_anim_config(view, idx, None);
    }

    pub fn activate_workspace_with_anim_config(
        &mut self,
        view: &mut WorkspaceView,
        idx: usize,
        config: Option<jiji_config::Animation>,
    ) {
        // FIXME: also compute and use current velocity.
        let current_idx = self.workspace_render_idx(view);

        let changed = view.activate(idx);

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
        view: &WorkspaceView,
        target: MonitorAddWindowTarget<'a, W>,
    ) -> (usize, WorkspaceAddWindowTarget<'a, W>) {
        match target {
            MonitorAddWindowTarget::Auto => {
                (view.active_position(), WorkspaceAddWindowTarget::Auto)
            }
            MonitorAddWindowTarget::Workspace { id, column_idx } => {
                let idx = view
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
                let idx = view
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

    pub fn switch_workspace_up(&mut self, view: &mut WorkspaceView) {
        let new_idx = match &self.workspace_switch {
            // During a DnD scroll, select the prev apparent workspace.
            Some(WorkspaceSwitch::Gesture(gesture)) if gesture.dnd_last_event_time.is_some() => {
                let current = gesture.current_idx;
                let new = current.ceil() - 1.;
                new.clamp(0., (view.len() - 1) as f64) as usize
            }
            _ => view.active_position().saturating_sub(1),
        };

        self.activate_workspace(view, new_idx);
    }

    pub fn switch_workspace_down(&mut self, view: &mut WorkspaceView) {
        let new_idx = match &self.workspace_switch {
            // During a DnD scroll, select the next apparent workspace.
            Some(WorkspaceSwitch::Gesture(gesture)) if gesture.dnd_last_event_time.is_some() => {
                let current = gesture.current_idx;
                let new = current.floor() + 1.;
                new.clamp(0., (view.len() - 1) as f64) as usize
            }
            _ => min(view.active_position() + 1, view.len() - 1),
        };

        self.activate_workspace(view, new_idx);
    }

    pub fn switch_workspace(&mut self, view: &mut WorkspaceView, idx: usize) {
        self.activate_workspace(view, min(idx, view.len() - 1));
    }

    pub fn switch_workspace_auto_back_and_forth(&mut self, view: &mut WorkspaceView, idx: usize) {
        let idx = min(idx, view.len() - 1);

        if idx == view.active_position() {
            if let Some(prev_idx) = view.previous_position() {
                self.switch_workspace(view, prev_idx);
            }
        } else {
            self.switch_workspace(view, idx);
        }
    }

    pub fn switch_workspace_previous(&mut self, view: &mut WorkspaceView) {
        if let Some(idx) = view.previous_position() {
            self.switch_workspace(view, idx);
        }
    }

    pub fn active_window<'a>(
        &self,
        pool: &'a HashMap<WorkspaceId, Workspace<W>>,
        view: &WorkspaceView,
    ) -> Option<&'a W> {
        self.active_workspace_ref(pool, view).active_window()
    }

    /// Advances per-monitor animations. Returns `true` when a workspace-switch animation
    /// completed this tick; the caller must then run `Layout::clean_up_workspaces_on` for this
    /// monitor — `clean_up_workspaces` lives on `Layout` (which owns the pool), so
    /// `Monitor` cannot call it inline.
    #[must_use]
    pub fn advance_animations(
        &mut self,
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        view: &WorkspaceView,
    ) -> bool {
        let mut workspace_switch_finished = false;
        match &mut self.workspace_switch {
            Some(WorkspaceSwitch::Animation(anim)) if anim.is_done() => {
                self.workspace_switch = None;
                workspace_switch_finished = true;
            }
            Some(WorkspaceSwitch::Animation(_)) => {}
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

        if self
            .activity_switch
            .as_ref()
            .is_some_and(|s| s.anim.is_done())
        {
            self.activity_switch = None;
        }

        for id in view.ids() {
            pool.get_mut(id)
                .expect("view id must be a key in the pool")
                .advance_animations();
        }

        workspace_switch_finished
    }

    pub(super) fn are_animations_ongoing(
        &self,
        pool: &HashMap<WorkspaceId, Workspace<W>>,
        view: &WorkspaceView,
        outgoing_view: Option<&WorkspaceView>,
    ) -> bool {
        self.workspace_switch
            .as_ref()
            .is_some_and(|s| s.is_animation_ongoing())
            || self
                .activity_switch
                .as_ref()
                .is_some_and(|s| !s.anim.is_done())
            || view
                .ids()
                .iter()
                .map(|id| pool.get(id).expect("view id must be a key in the pool"))
                .any(|ws| ws.are_animations_ongoing())
            || outgoing_view
                .into_iter()
                .flat_map(|v| v.ids())
                .map(|id| pool.get(id).expect("view id must be a key in the pool"))
                .any(|ws| ws.are_animations_ongoing())
    }

    pub fn are_transitions_ongoing(
        &self,
        pool: &HashMap<WorkspaceId, Workspace<W>>,
        view: &WorkspaceView,
    ) -> bool {
        self.workspace_switch.is_some()
            || self.activity_switch.is_some()
            || view
                .ids()
                .iter()
                .map(|id| pool.get(id).expect("view id must be a key in the pool"))
                .any(|ws| ws.are_transitions_ongoing())
    }

    pub fn update_render_elements(
        &mut self,
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        view: &WorkspaceView,
        is_active: bool,
    ) {
        let mut insert_hint_ws_geo = None;
        let insert_hint_ws_id = self
            .insert_hint
            .as_ref()
            .and_then(|hint| hint.workspace.existing_id());

        for (id, geo) in self
            .workspaces_with_render_geo_ids(view, true)
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
                    if let Some(ws) = pool.get(&ws_id).filter(|_| view.ids().contains(&ws_id)) {
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

                    let next_ws_geo = self.workspaces_render_geo(view).nth(ws_idx).unwrap();
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
        view: &mut WorkspaceView,
        base_options: Rc<Options>,
        seed_activity: ActivityId,
    ) {
        let options =
            Rc::new(Options::clone(&base_options).with_merged_layout(self.layout_config.as_ref()));

        if self.options.layout.empty_workspace_above_first
            != options.layout.empty_workspace_above_first
            && view.len() > 1
        {
            if options.layout.empty_workspace_above_first {
                // Inlined former `self.add_workspace_top(pool)` — the pool-taking
                // structural methods live on `Layout` (which owns the pool), but
                // `update_config` runs on `&mut Monitor` and only needs a fresh top
                // bookend here, so we build it directly.
                let ws = Workspace::new(
                    &self.output,
                    HashSet::from([seed_activity]),
                    self.clock.clone(),
                    self.options.clone(),
                );
                let id = ws.id();
                assert!(pool.insert(id, ws).is_none(), "fresh id must be unique");
                view.insert(0, id);
                if let Some(switch) = &mut self.workspace_switch {
                    if 0. <= switch.target_idx() {
                        switch.offset(1);
                    }
                }
            } else if self.workspace_switch.is_none() && view.active_position() != 0 {
                let id = view.ids()[0];
                view.remove_at(0);
                assert!(
                    pool.remove(&id).is_some(),
                    "view id must be a key in the pool",
                );
            }
        }

        for id in view.ids() {
            pool.get_mut(id)
                .expect("view id must be a key in the pool")
                .update_config(options.clone());
        }

        self.insert_hint_element
            .update_config(options.layout.insert_hint);

        // Config reload snaps any in-flight activity-switch transition (snap+proceed contract).
        self.activity_switch = None;

        self.base_options = base_options;
        self.options = options;
    }

    pub fn update_layout_config(
        &mut self,
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        view: &mut WorkspaceView,
        layout_config: Option<jiji_config::LayoutPart>,
        seed_activity: ActivityId,
    ) -> bool {
        if self.layout_config == layout_config {
            return false;
        }

        self.layout_config = layout_config;
        self.update_config(pool, view, self.base_options.clone(), seed_activity);

        true
    }

    pub fn update_shaders(
        &mut self,
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        view: &WorkspaceView,
    ) {
        for id in view.ids() {
            pool.get_mut(id)
                .expect("view id must be a key in the pool")
                .update_shaders();
        }

        self.insert_hint_element.update_shaders();
    }

    pub fn update_output_size(
        &mut self,
        pool: &mut HashMap<WorkspaceId, Workspace<W>>,
        view: &WorkspaceView,
    ) {
        self.scale = self.output.current_scale();
        self.view_size = output_size(&self.output);
        self.working_area = compute_working_area(&self.output);

        for id in view.ids() {
            pool.get_mut(id)
                .expect("view id must be a key in the pool")
                .update_output_size(&self.output);
        }
    }

    /// Returns the geometry of the active window relative to and clamped to the output.
    ///
    /// During animations, assumes the final view position.
    pub fn active_window_visual_rectangle(
        &self,
        pool: &HashMap<WorkspaceId, Workspace<W>>,
        view: &WorkspaceView,
    ) -> Option<Rectangle<f64, Logical>> {
        if self.overview_open {
            return None;
        }

        self.active_workspace_ref(pool, view)
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

    pub(super) fn set_overview_progress(
        &mut self,
        view: &WorkspaceView,
        progress: Option<&super::OverviewProgress>,
    ) {
        let prev_render_idx = self.workspace_render_idx(view);
        self.overview_progress = progress.map(OverviewProgress::from);
        let new_render_idx = self.workspace_render_idx(view);

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

    #[cfg(debug_assertions)]
    pub(super) fn overview_progress_value(&self) -> Option<f64> {
        self.overview_progress.as_ref().map(|p| p.value())
    }

    pub fn workspace_render_idx(&self, view: &WorkspaceView) -> f64 {
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
            view.active_position() as f64
        }
    }

    /// Horizontal offset applied to one strip's workspace rects during an activity switch.
    ///
    /// Returns `0.` when no transition is in flight — a single `Option` check, the entire idle
    /// cost. While a switch is armed, for a `Left` direction the incoming strip arrives from
    /// `+stride` (right side) and settles to 0, while the outgoing strip departs toward `-stride`
    /// (left side); signs are negated for a `Right` direction. `stride` is the per-workspace
    /// horizontal pitch (workspace width plus gap), mirroring the vertical pitch used by the y
    /// layout.
    fn activity_switch_x_offset(&self, strip: ActivityStrip) -> f64 {
        let Some(switch) = &self.activity_switch else {
            return 0.;
        };

        let scale = self.scale.fractional_scale();
        let zoom = self.overview_zoom();
        let stride = self.workspace_size(zoom).w + self.workspace_gap(zoom);

        let p = switch.anim.value();
        let sign = match switch.dir {
            SlideDirection::Left => 1.,
            SlideDirection::Right => -1.,
        };

        let x = match strip {
            ActivityStrip::Incoming => sign * (1. - p) * stride,
            ActivityStrip::Outgoing => -sign * p * stride,
        };
        round_logical_in_physical(scale, x)
    }

    pub fn workspaces_render_geo(
        &self,
        view: &WorkspaceView,
    ) -> impl Iterator<Item = Rectangle<f64, Logical>> {
        self.workspaces_render_geo_for_strip(view, ActivityStrip::Incoming)
    }

    /// Strip-aware core of [`workspaces_render_geo`](Self::workspaces_render_geo).
    ///
    /// Yields the same vertical layout as the public method but adds the activity-switch
    /// horizontal offset for `strip` to every rect's `loc`. The offset is applied here, before
    /// any culling downstream, so the cull filters see the post-offset positions.
    pub(super) fn workspaces_render_geo_for_strip(
        &self,
        view: &WorkspaceView,
        strip: ActivityStrip,
    ) -> impl Iterator<Item = Rectangle<f64, Logical>> {
        // Partial I3 guard: the Outgoing strip is only meaningful while a switch is in flight.
        // The full invariant (that `view` is the correct strip's view) is not checked here because
        // this layer does not have access to both the active and `switch.from` views
        // simultaneously. A structural fix (fold `strip` into the view-resolution so the
        // illegal pairing is unrepresentable) is the correct long-term resolution; see the
        // architectural escalation.
        debug_assert!(
            strip == ActivityStrip::Incoming || self.activity_switch.is_some(),
            "Outgoing strip requested with no activity switch in flight",
        );

        let scale = self.scale.fractional_scale();
        let zoom = self.overview_zoom();

        let ws_size = self.workspace_size(zoom);
        let gap = self.workspace_gap(zoom);
        let ws_height_with_gap = ws_size.h + gap;

        let x_offset = self.activity_switch_x_offset(strip);

        let static_offset = (self.view_size.to_point() - ws_size.to_point()).downscale(2.);
        let static_offset = static_offset
            .to_physical_precise_round(scale)
            .to_logical(scale);

        let first_ws_y = -self.workspace_render_idx(view) * ws_height_with_gap;
        let first_ws_y = round_logical_in_physical(scale, first_ws_y);

        // Return position for one-past-last workspace too.
        (0..=view.len()).map(move |idx| {
            let y = first_ws_y + idx as f64 * ws_height_with_gap;
            let loc = Point::from((x_offset, y)) + static_offset;

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
        ctx: LayoutCtx<'a, W>,
    ) -> impl Iterator<Item = (&'a Workspace<W>, Rectangle<f64, Logical>)> + 'a {
        let output_geo = Rectangle::from_size(self.view_size);

        let geo = self.workspaces_render_geo(ctx.view());
        zip(ctx.view().ids().iter().copied(), geo)
            .map(move |(id, geo)| (ctx.workspace(id), geo))
            // Cull out workspaces outside the output.
            .filter(move |(_ws, geo)| geo.intersection(output_geo).is_some())
    }

    /// Same as [`workspaces_with_render_geo`](Self::workspaces_with_render_geo) but for the
    /// outgoing activity strip during a switch. `ctx` must bundle the outgoing activity's view.
    pub fn workspaces_with_render_geo_outgoing<'a>(
        &'a self,
        ctx: LayoutCtx<'a, W>,
    ) -> impl Iterator<Item = (&'a Workspace<W>, Rectangle<f64, Logical>)> + 'a {
        let output_geo = Rectangle::from_size(self.view_size);

        let geo = self.workspaces_render_geo_for_strip(ctx.view(), ActivityStrip::Outgoing);
        zip(ctx.view().ids().iter().copied(), geo)
            .map(move |(id, geo)| (ctx.workspace(id), geo))
            // Cull out workspaces outside the output.
            .filter(move |(_ws, geo)| geo.intersection(output_geo).is_some())
    }

    pub fn workspaces_with_render_geo_idx<'a>(
        &'a self,
        ctx: LayoutCtx<'a, W>,
    ) -> impl Iterator<Item = ((usize, &'a Workspace<W>), Rectangle<f64, Logical>)> + 'a {
        let output_geo = Rectangle::from_size(self.view_size);

        let geo = self.workspaces_render_geo(ctx.view());
        zip(ctx.view().ids().iter().copied().enumerate(), geo)
            .map(move |((idx, id), geo)| ((idx, ctx.workspace(id)), geo))
            // Cull out workspaces outside the output.
            .filter(move |(_ws, geo)| geo.intersection(output_geo).is_some())
    }

    /// Same shape as [`workspaces_with_render_geo`](Self::workspaces_with_render_geo) but yields
    /// ids instead of workspace references. Does not borrow the pool; callers thread their own
    /// `&mut pool` borrow through each yielded id.
    pub fn workspaces_with_render_geo_ids<'a>(
        &'a self,
        view: &'a WorkspaceView,
        cull: bool,
    ) -> impl Iterator<Item = (WorkspaceId, Rectangle<f64, Logical>)> + 'a {
        let output_geo = Rectangle::from_size(self.view_size);

        let geo = self.workspaces_render_geo(view);
        let ids = view.ids();
        zip(ids.iter().copied(), geo)
            // Cull out workspaces outside the output.
            .filter(move |(_id, geo)| !cull || geo.intersection(output_geo).is_some())
    }

    pub fn workspace_under<'a>(
        &'a self,
        ctx: LayoutCtx<'a, W>,
        pos_within_output: Point<f64, Logical>,
    ) -> Option<(&'a Workspace<W>, Rectangle<f64, Logical>)> {
        let (ws, geo) = self.workspaces_with_render_geo(ctx).find_map(|(ws, geo)| {
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
        ctx: LayoutCtx<'a, W>,
        pos_within_output: Point<f64, Logical>,
    ) -> Option<&'a Workspace<W>> {
        self.workspaces_with_render_geo(ctx)
            .find_map(|(ws, geo)| geo.contains(pos_within_output).then_some(ws))
    }

    pub fn window_under<'a>(
        &'a self,
        ctx: LayoutCtx<'a, W>,
        pos_within_output: Point<f64, Logical>,
    ) -> Option<(&'a W, HitType)> {
        let (ws, geo) = self.workspace_under(ctx, pos_within_output)?;

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
        ctx: LayoutCtx<'_, W>,
        pos_within_output: Point<f64, Logical>,
    ) -> Option<ResizeEdge> {
        if self.overview_progress.is_some() {
            return None;
        }

        let (ws, geo) = self.workspace_under(ctx, pos_within_output)?;
        ws.resize_edges_under(pos_within_output - geo.loc)
    }

    pub(super) fn insert_position(
        &self,
        ctx: LayoutCtx<'_, W>,
        pos_within_output: Point<f64, Logical>,
    ) -> (InsertWorkspace, Rectangle<f64, Logical>) {
        let mut iter = self.workspaces_with_render_geo_idx(ctx);

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

    pub fn render_above_top_layer(&self, ctx: LayoutCtx<'_, W>) -> bool {
        // Render above the top layer only if the view is stationary.
        if self.workspace_switch.is_some()
            || self.overview_progress.is_some()
            || self.activity_switch.is_some()
        {
            return false;
        }

        ctx.workspace_at(ctx.view().active_position())
            .render_above_top_layer()
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
        lctx: LayoutCtx<'_, W>,
        mut ctx: RenderCtx<R>,
        focus_ring: bool,
        strip: ActivityStrip,
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
        let crop_bounds = if self.workspace_switch.is_some()
            || self.overview_progress.is_some()
            || self.activity_switch.is_some()
        {
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

        // The two strip iterators have distinct opaque types; dispatching via a macro avoids
        // a Vec collect on the idle/Incoming path (zero allocation when no switch is in flight).
        // Macro instead of closure because ws and insert hint have different elem types.
        macro_rules! push_for_geo {
            ($geo:expr) => {{
                &mut |elem| {
                    let elem = CropRenderElement::from_element(elem, scale, crop_bounds);
                    if let Some(elem) = elem {
                        let elem = MonitorInnerRenderElement::from(elem);
                        push(scale_relocate($geo, elem));
                    }
                }
            }};
        }

        // The active layer renders on top, so that raising the tiling layer
        // (e.g. switching focus to it) brings tiled windows in front of
        // floating ones — not only the other way around. The insert hint
        // belongs to the tiling layer and stays immediately above it.
        macro_rules! render_insert_hint_for {
            ($ws:expr, $geo:expr) => {
                if let Some(loc) = insert_hint_render_loc {
                    if loc.workspace == InsertWorkspace::Existing($ws.id()) {
                        self.insert_hint_element.render(
                            ctx.renderer,
                            loc.location,
                            push_for_geo!($geo),
                        );
                    }
                }
            };
        }

        macro_rules! render_ws {
            ($ws:expr, $geo:expr) => {{
                let ws = $ws;
                let geo = $geo;
                let xray_pos = XrayPos::new(geo.loc, zoom);
                if ws.floating_is_active() {
                    ws.render_floating(ctx.r(), xray_pos, focus_ring, push_for_geo!(geo));
                    render_insert_hint_for!(ws, geo);
                    ws.render_scrolling(ctx.r(), xray_pos, focus_ring, push_for_geo!(geo));
                } else {
                    render_insert_hint_for!(ws, geo);
                    ws.render_scrolling(ctx.r(), xray_pos, focus_ring, push_for_geo!(geo));
                    ws.render_floating(ctx.r(), xray_pos, focus_ring, push_for_geo!(geo));
                }
            }};
        }

        match strip {
            ActivityStrip::Incoming => {
                for (ws, geo) in self.workspaces_with_render_geo(lctx) {
                    render_ws!(ws, geo);
                }
            }
            ActivityStrip::Outgoing => {
                for (ws, geo) in self.workspaces_with_render_geo_outgoing(lctx) {
                    render_ws!(ws, geo);
                }
            }
        }
    }

    pub fn render_workspace_shadows<R: NiriRenderer>(
        &self,
        ctx: LayoutCtx<'_, W>,
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

        for (ws, geo) in self.workspaces_with_render_geo(ctx) {
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

    pub fn workspace_switch_gesture_begin(&mut self, view: &WorkspaceView, is_touchpad: bool) {
        let center_idx = view.active_position();
        let current_idx = self.workspace_render_idx(view);

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

    pub fn dnd_scroll_gesture_begin(&mut self, view: &WorkspaceView) {
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

        let center_idx = view.active_position();
        let current_idx = self.workspace_render_idx(view);

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
        view: &WorkspaceView,
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

        let (min, max) = gesture.min_max(view.len());
        let new_idx = gesture.start_idx + pos;
        let new_idx = rubber_band.clamp(min, max, new_idx);

        if gesture.current_idx == new_idx {
            return Some(false);
        }

        gesture.current_idx = new_idx;
        Some(true)
    }

    pub fn dnd_scroll_gesture_scroll(
        &mut self,
        view: &WorkspaceView,
        pos: Point<f64, Logical>,
        speed: f64,
    ) -> bool {
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

        let (min, max) = gesture.min_max(view.len());
        let clamped = unclamped.clamp(min, max);

        // Make sure that DnD scrolling too much outside the min/max does not "build up".
        gesture.start_idx += clamped - unclamped;
        gesture.current_idx = clamped;

        true
    }

    pub fn workspace_switch_gesture_end(
        &mut self,
        view: &mut WorkspaceView,
        is_touchpad: Option<bool>,
    ) -> bool {
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

        let (min, max) = gesture.min_max(view.len());
        let new_idx = gesture.start_idx + pos;

        let new_idx = new_idx.clamp(min, max);
        let new_idx = new_idx.round() as usize;

        velocity *= rubber_band.clamp_derivative(min, max, gesture.start_idx + current_pos);

        view.activate(new_idx);
        self.workspace_switch = Some(WorkspaceSwitch::Animation(Animation::new(
            self.clock.clone(),
            gesture.current_idx,
            new_idx as f64,
            velocity,
            self.options.animations.workspace_switch.0,
        )));

        true
    }

    pub fn dnd_scroll_gesture_end(&mut self, view: &mut WorkspaceView) {
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

        self.workspace_switch_gesture_end(view, None);
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

    pub fn layout_config(&self) -> Option<&jiji_config::LayoutPart> {
        self.layout_config.as_ref()
    }

    #[cfg(debug_assertions)]
    pub(super) fn verify_invariants(
        &self,
        pool: &HashMap<WorkspaceId, Workspace<W>>,
        views: &[&WorkspaceView],
    ) {
        use approx::assert_abs_diff_eq;

        assert!(
            !views.is_empty(),
            "Monitor::verify_invariants requires at least one view — the active activity's view \
             for this monitor's output",
        );

        let options =
            Options::clone(&self.base_options).with_merged_layout(self.layout_config.as_ref());
        assert_eq!(&*self.options, &options);

        let ewaf = self.options.layout.empty_workspace_above_first;
        let active_view = views[0];

        // Per-view assertions: every activity's view for this monitor's output must hold the
        // structural bookend / pool-membership / length rules (1, or 3+, or 2 when the second
        // entry is a shared workspace pinned by another activity under EWAF). The in-flight switch
        // animation bounds-check looks at the active view only — `WorkspaceSwitch` lives on the
        // monitor and points at positions inside the *active* activity's view.
        for (vi, view) in views.iter().enumerate() {
            for (i, id) in view.ids().iter().enumerate() {
                assert!(
                    pool.contains_key(id),
                    "views[{vi}].ids[{i}] must be a key in the workspace pool",
                );
            }
            assert!(view.active_position() < view.len());

            assert_view_bookends(pool, view, ewaf, None);
        }

        if let Some(WorkspaceSwitch::Animation(anim)) = &self.workspace_switch {
            let before_idx = anim.from() as usize;
            let after_idx = anim.to() as usize;

            assert!(before_idx < active_view.len());
            assert!(after_idx < active_view.len());
        }

        let ws = |id: WorkspaceId| -> &Workspace<W> {
            pool.get(&id).expect("view id must be a key in the pool")
        };

        // The "no non-active empty workspaces in the middle" rule is scoped to the active view:
        // it relies on `view.active_position()` matching the monitor's current focus state, which
        // is meaningful only for the active activity's view.
        let pre_skip = if ewaf { 1 } else { 0 };
        if self.workspace_switch.is_none() {
            for (idx, id) in active_view
                .ids()
                .iter()
                .enumerate()
                .skip(pre_skip)
                .rev()
                // skip last
                .skip(1)
            {
                if idx != active_view.active_position() {
                    let workspace = ws(*id);
                    // A shared workspace is pinned in every member activity's view, so an
                    // empty unnamed middle entry is its legal steady state.
                    assert!(
                        workspace.has_windows_or_name() || workspace.activities().len() > 1,
                        "non-active workspace can't be empty and unnamed except the last one \
                         or a shared (pinned) workspace"
                    );
                }
            }
        }

        // Scale / view-size / working-area / option synchronization: each workspace whose id is
        // anywhere in views[0..] for this monitor is currently bound to this monitor (or will be
        // on the next bind cycle), so its render-side fields must match. A workspace appearing in
        // multiple views is re-checked once per view; the redundant work is bounded by the
        // activity count and stays cheap.
        for view in views {
            for id in view.ids() {
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
        }

        // Render-geo walk: active view only — monitor render state (workspace_switch offsets,
        // overview zoom, layout placement) is driven by the active activity's view.
        let scale = self.scale().fractional_scale();
        let ctx = LayoutCtx::new(pool, active_view);
        let iter = self.workspaces_with_render_geo(ctx);
        for (_ws, ws_geo) in iter {
            let pos = ws_geo.loc;
            let rounded_pos = pos.to_physical_precise_round(scale).to_logical(scale);

            // Workspace positions must be rounded to physical pixels.
            assert_abs_diff_eq!(pos.x, rounded_pos.x, epsilon = 1e-5);
            assert_abs_diff_eq!(pos.y, rounded_pos.y, epsilon = 1e-5);
        }
    }
}

/// Per-view bookend invariant: trailing workspace empty + unnamed, and (under EWAF) leading
/// workspace empty + unnamed plus the length rule (1, or 3+, or 2 when the second entry is a
/// shared workspace pinned by another activity). Shared between
/// `Monitor::verify_invariants` (called per activity-view of a connected monitor) and
/// `Layout::verify_invariants` (called for views on outputs that are no longer connected).
///
/// Callers pass the EWAF flag explicitly because the source-of-truth differs by call site:
/// connected monitors use their per-monitor merged options; disconnected outputs use the
/// layout-root options (matching what `ensure_all_activity_views` uses when materializing).
///
/// The optional `ctx` parameter is appended to each assertion message to identify which
/// `(activity_id, output_id)` pair tripped the invariant, aiding proptest failure diagnosis.
#[cfg(debug_assertions)]
pub(super) fn assert_view_bookends<W: LayoutElement>(
    pool: &HashMap<WorkspaceId, Workspace<W>>,
    view: &WorkspaceView,
    ewaf: bool,
    ctx: Option<(&super::activity::ActivityId, &super::OutputId)>,
) {
    let ctx_str = ctx
        .map(|(a, o)| format!(" [activity={a:?}, output={o:?}]"))
        .unwrap_or_default();

    let ws = |id: WorkspaceId| -> &Workspace<W> {
        pool.get(&id).expect("view id must be a key in the pool")
    };

    let last_id = *view.ids().last().unwrap();
    let first_id = *view.ids().first().unwrap();

    assert!(
        !ws(last_id).has_windows(),
        "view must have an empty workspace at the end{ctx_str}",
    );
    assert!(
        ws(last_id).name.is_none(),
        "view must have an unnamed workspace at the end{ctx_str}",
    );

    if ewaf {
        assert!(
            !ws(first_id).has_windows(),
            "first workspace must be empty when empty_workspace_above_first is set{ctx_str}",
        );
        assert!(
            ws(first_id).name.is_none(),
            "first workspace must be unnamed when empty_workspace_above_first is set{ctx_str}",
        );
        // A shared second entry is pinned by another activity's view and doubles as
        // the trailing bookend, making length 2 the honest minimal shape.
        assert!(
            view.len() != 2 || ws(view.ids()[1]).activities().len() > 1,
            "if empty_workspace_above_first is set there must be just 1 or 3+ workspaces, \
             unless a shared workspace pins the second entry{ctx_str}",
        );
    }
}
