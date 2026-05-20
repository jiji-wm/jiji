use std::cell::{Cell, OnceCell, RefCell};
use std::collections::{HashMap, HashSet};

use jiji_config::utils::{Flag, MergeWith as _};
use jiji_config::workspace::WorkspaceName;
use jiji_config::{
    CenterFocusedColumn, FloatOrInt, OutputName, Struts, TabIndicatorLength, TabIndicatorPosition,
    WorkspaceReference,
};
use proptest::prelude::*;
use proptest_derive::Arbitrary;
use smithay::output::{Mode, PhysicalProperties, Subpixel};
use smithay::utils::Rectangle;

use super::activity::ActivityId;
use super::*;

mod animations;
mod fullscreen;

impl<W: LayoutElement> Default for Layout<W> {
    fn default() -> Self {
        Self::with_options(Clock::with_time(Duration::ZERO), Default::default())
    }
}

#[derive(Debug)]
struct TestWindowInner {
    id: usize,
    parent_id: Cell<Option<usize>>,
    bbox: Cell<Rectangle<i32, Logical>>,
    initial_bbox: Rectangle<i32, Logical>,
    requested_size: Cell<Option<Size<i32, Logical>>>,
    // Emulates the window ignoring the compositor-provided size.
    forced_size: Cell<Option<Size<i32, Logical>>>,
    min_size: Size<i32, Logical>,
    max_size: Size<i32, Logical>,
    pending_sizing_mode: Cell<SizingMode>,
    pending_activated: Cell<bool>,
    sizing_mode: Cell<SizingMode>,
    is_windowed_fullscreen: Cell<bool>,
    is_pending_windowed_fullscreen: Cell<bool>,
    animate_next_configure: Cell<bool>,
    animation_snapshot: RefCell<Option<LayoutElementRenderSnapshot>>,
    rules: ResolvedWindowRules,
    /// Test-only urgency flag. Production `Mapped` derives urgency from the
    /// xdg-activation protocol; tests toggle it directly via
    /// [`TestWindow::set_urgent`] to exercise urgency-propagation paths
    /// (`Workspace::is_urgent` → `Layout::activity_is_urgent`).
    is_urgent: Cell<bool>,
    // Per-output `output_enter` count, matching Smithay's `Window::output_enter` semantics.
    // Incremented on `output_enter`, decremented on `output_leave`; entries are dropped at zero.
    // Used by `verify_output_bindings` to catch bind/unbind symmetry violations in the layout.
    bound_outputs: RefCell<HashMap<Output, u32>>,
}

#[derive(Debug, Clone)]
struct TestWindow(Rc<TestWindowInner>);

#[derive(Debug, Clone, Arbitrary)]
struct TestWindowParams {
    #[proptest(strategy = "1..=5usize")]
    id: usize,
    #[proptest(strategy = "arbitrary_parent_id()")]
    parent_id: Option<usize>,
    is_floating: bool,
    #[proptest(strategy = "arbitrary_bbox()")]
    bbox: Rectangle<i32, Logical>,
    #[proptest(strategy = "arbitrary_min_max_size()")]
    min_max_size: (Size<i32, Logical>, Size<i32, Logical>),
    #[proptest(strategy = "prop::option::of(arbitrary_rules())")]
    rules: Option<ResolvedWindowRules>,
}

impl TestWindowParams {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            parent_id: None,
            is_floating: false,
            bbox: Rectangle::from_size(Size::from((100, 200))),
            min_max_size: Default::default(),
            rules: None,
        }
    }
}

impl TestWindow {
    fn new(params: TestWindowParams) -> Self {
        Self(Rc::new(TestWindowInner {
            id: params.id,
            parent_id: Cell::new(params.parent_id),
            bbox: Cell::new(params.bbox),
            initial_bbox: params.bbox,
            requested_size: Cell::new(None),
            forced_size: Cell::new(None),
            min_size: params.min_max_size.0,
            max_size: params.min_max_size.1,
            pending_sizing_mode: Cell::new(SizingMode::Normal),
            pending_activated: Cell::new(false),
            sizing_mode: Cell::new(SizingMode::Normal),
            is_windowed_fullscreen: Cell::new(false),
            is_pending_windowed_fullscreen: Cell::new(false),
            animate_next_configure: Cell::new(false),
            animation_snapshot: RefCell::new(None),
            rules: params.rules.unwrap_or_default(),
            is_urgent: Cell::new(false),
            bound_outputs: RefCell::new(HashMap::new()),
        }))
    }

    fn set_urgent(&self, urgent: bool) {
        self.0.is_urgent.set(urgent);
    }

    fn bound_outputs(&self) -> Vec<Output> {
        self.0.bound_outputs.borrow().keys().cloned().collect()
    }

    fn communicate(&self) -> bool {
        let mut changed = false;

        let size = self.0.forced_size.get().or(self.0.requested_size.get());
        if let Some(size) = size {
            assert!(size.w >= 0);
            assert!(size.h >= 0);

            let mut new_bbox = self.0.initial_bbox;
            if size.w != 0 {
                new_bbox.size.w = size.w;
            }
            if size.h != 0 {
                new_bbox.size.h = size.h;
            }

            if self.0.bbox.get() != new_bbox {
                if self.0.animate_next_configure.get() {
                    self.0.animation_snapshot.replace(Some(RenderSnapshot {
                        contents: Vec::new(),
                        contents_with_blocked_out_bg: None,
                        blocked_out_contents: Vec::new(),
                        block_out_from: None,
                        size: self.0.bbox.get().size.to_f64(),
                        texture: OnceCell::new(),
                        texture_with_blocked_out_bg: Default::default(),
                        blocked_out_texture: OnceCell::new(),
                    }));
                }

                self.0.bbox.set(new_bbox);
                changed = true;
            }
        }

        self.0.animate_next_configure.set(false);

        if self.0.sizing_mode.get() != self.0.pending_sizing_mode.get() {
            self.0.sizing_mode.set(self.0.pending_sizing_mode.get());
            changed = true;
        }

        if self.0.is_windowed_fullscreen.get() != self.0.is_pending_windowed_fullscreen.get() {
            self.0
                .is_windowed_fullscreen
                .set(self.0.is_pending_windowed_fullscreen.get());
            changed = true;
        }

        changed
    }
}

impl LayoutElement for TestWindow {
    type Id = usize;

    fn id(&self) -> &Self::Id {
        &self.0.id
    }

    fn size(&self) -> Size<i32, Logical> {
        self.0.bbox.get().size
    }

    fn buf_loc(&self) -> Point<i32, Logical> {
        (0, 0).into()
    }

    fn is_in_input_region(&self, _point: Point<f64, Logical>) -> bool {
        false
    }

    fn request_size(
        &mut self,
        size: Size<i32, Logical>,
        mode: SizingMode,
        _animate: bool,
        _transaction: Option<Transaction>,
    ) {
        if self.0.requested_size.get() != Some(size) {
            self.0.requested_size.set(Some(size));
            self.0.animate_next_configure.set(true);
        }

        self.0.pending_sizing_mode.set(mode);

        if mode.is_fullscreen() {
            self.0.is_pending_windowed_fullscreen.set(false);
        }
    }

    fn min_size(&self) -> Size<i32, Logical> {
        self.0.min_size
    }

    fn max_size(&self) -> Size<i32, Logical> {
        self.0.max_size
    }

    fn is_wl_surface(&self, _wl_surface: &WlSurface) -> bool {
        false
    }

    fn set_preferred_scale_transform(&self, _scale: output::Scale, _transform: Transform) {}

    fn has_ssd(&self) -> bool {
        false
    }

    fn output_enter(&self, output: &Output) {
        *self
            .0
            .bound_outputs
            .borrow_mut()
            .entry(output.clone())
            .or_insert(0) += 1;
    }

    fn output_leave(&self, output: &Output) {
        let mut bindings = self.0.bound_outputs.borrow_mut();
        if let Some(count) = bindings.get_mut(output) {
            *count -= 1;
            if *count == 0 {
                bindings.remove(output);
            }
        }
    }

    fn set_offscreen_data(&self, _data: Option<OffscreenData>) {}

    fn set_activated(&mut self, active: bool) {
        self.0.pending_activated.set(active);
    }

    fn set_bounds(&self, _bounds: Size<i32, Logical>) {}

    fn is_ignoring_opacity_window_rule(&self) -> bool {
        false
    }

    fn configure_intent(&self) -> ConfigureIntent {
        ConfigureIntent::CanSend
    }

    fn send_pending_configure(&mut self) {}

    fn set_active_in_column(&mut self, _active: bool) {}

    fn set_floating(&mut self, _floating: bool) {}

    fn sizing_mode(&self) -> SizingMode {
        self.0.sizing_mode.get()
    }

    fn pending_sizing_mode(&self) -> SizingMode {
        self.0.pending_sizing_mode.get()
    }

    fn requested_size(&self) -> Option<Size<i32, Logical>> {
        self.0.requested_size.get()
    }

    fn is_windowed_fullscreen(&self) -> bool {
        self.0.is_windowed_fullscreen.get()
    }

    fn is_pending_windowed_fullscreen(&self) -> bool {
        self.0.is_pending_windowed_fullscreen.get()
    }

    fn request_windowed_fullscreen(&mut self, value: bool) {
        self.0.is_pending_windowed_fullscreen.set(value);
    }

    fn is_child_of(&self, parent: &Self) -> bool {
        self.0.parent_id.get() == Some(parent.0.id)
    }

    fn refresh(&self) {}

    fn rules(&self) -> &ResolvedWindowRules {
        &self.0.rules
    }

    fn take_animation_snapshot(&mut self) -> Option<LayoutElementRenderSnapshot> {
        self.0.animation_snapshot.take()
    }

    fn set_interactive_resize(&mut self, _data: Option<InteractiveResizeData>) {}

    fn cancel_interactive_resize(&mut self) {}

    fn on_commit(&mut self, _serial: Serial) {}

    fn interactive_resize_data(&self) -> Option<InteractiveResizeData> {
        None
    }

    fn is_urgent(&self) -> bool {
        self.0.is_urgent.get()
    }
}

fn arbitrary_size() -> impl Strategy<Value = Size<i32, Logical>> {
    any::<(u16, u16)>().prop_map(|(w, h)| Size::from((w.max(1).into(), h.max(1).into())))
}

fn arbitrary_bbox() -> impl Strategy<Value = Rectangle<i32, Logical>> {
    any::<(i16, i16, u16, u16)>().prop_map(|(x, y, w, h)| {
        let loc: Point<i32, _> = Point::from((x.into(), y.into()));
        let size: Size<i32, _> = Size::from((w.max(1).into(), h.max(1).into()));
        Rectangle::new(loc, size)
    })
}

fn arbitrary_size_change() -> impl Strategy<Value = SizeChange> {
    prop_oneof![
        (0..).prop_map(SizeChange::SetFixed),
        (0f64..).prop_map(SizeChange::SetProportion),
        any::<i32>().prop_map(SizeChange::AdjustFixed),
        any::<f64>().prop_map(SizeChange::AdjustProportion),
        // Interactive resize can have negative values here.
        Just(SizeChange::SetFixed(-100)),
    ]
}

fn arbitrary_position_change() -> impl Strategy<Value = PositionChange> {
    prop_oneof![
        (-1000f64..1000f64).prop_map(PositionChange::SetFixed),
        any::<f64>().prop_map(PositionChange::SetProportion),
        (-1000f64..1000f64).prop_map(PositionChange::AdjustFixed),
        any::<f64>().prop_map(PositionChange::AdjustProportion),
        any::<f64>().prop_map(PositionChange::SetFixed),
        any::<f64>().prop_map(PositionChange::AdjustFixed),
    ]
}

fn arbitrary_min_max() -> impl Strategy<Value = (i32, i32)> {
    prop_oneof![
        Just((0, 0)),
        (1..65536).prop_map(|n| (n, n)),
        (1..65536).prop_map(|min| (min, 0)),
        (1..).prop_map(|max| (0, max)),
        (1..65536, 1..).prop_map(|(min, max): (i32, i32)| (min, max.max(min))),
    ]
}

fn arbitrary_min_max_size() -> impl Strategy<Value = (Size<i32, Logical>, Size<i32, Logical>)> {
    prop_oneof![
        5 => (arbitrary_min_max(), arbitrary_min_max()).prop_map(
            |((min_w, max_w), (min_h, max_h))| {
                let min_size = Size::from((min_w, min_h));
                let max_size = Size::from((max_w, max_h));
                (min_size, max_size)
            },
        ),
        1 => arbitrary_min_max().prop_map(|(w, h)| {
            let size = Size::from((w, h));
            (size, size)
        }),
    ]
}

prop_compose! {
    fn arbitrary_rules()(
        focus_ring in arbitrary_focus_ring(),
        border in arbitrary_border(),
    ) -> ResolvedWindowRules {
        ResolvedWindowRules {
            focus_ring,
            border,
            ..ResolvedWindowRules::default()
        }
    }
}

fn arbitrary_view_offset_gesture_delta() -> impl Strategy<Value = f64> {
    prop_oneof![(-10f64..10f64), (-50000f64..50000f64),]
}

fn arbitrary_resize_edge() -> impl Strategy<Value = ResizeEdge> {
    prop_oneof![
        Just(ResizeEdge::RIGHT),
        Just(ResizeEdge::BOTTOM),
        Just(ResizeEdge::LEFT),
        Just(ResizeEdge::TOP),
        Just(ResizeEdge::BOTTOM_RIGHT),
        Just(ResizeEdge::BOTTOM_LEFT),
        Just(ResizeEdge::TOP_RIGHT),
        Just(ResizeEdge::TOP_LEFT),
        Just(ResizeEdge::empty()),
    ]
}

fn arbitrary_scale() -> impl Strategy<Value = f64> {
    prop_oneof![Just(1.), Just(1.5), Just(2.),]
}

fn arbitrary_msec_delta() -> impl Strategy<Value = i32> {
    prop_oneof![
        1 => Just(-1000),
        2 => Just(-10),
        1 => Just(0),
        2 => Just(10),
        6 => Just(1000),
    ]
}

fn arbitrary_parent_id() -> impl Strategy<Value = Option<usize>> {
    prop_oneof![
        5 => Just(None),
        1 => prop::option::of(1..=5usize),
    ]
}

fn arbitrary_scroll_direction() -> impl Strategy<Value = ScrollDirection> {
    prop_oneof![Just(ScrollDirection::Left), Just(ScrollDirection::Right)]
}

fn arbitrary_column_display() -> impl Strategy<Value = ColumnDisplay> {
    prop_oneof![Just(ColumnDisplay::Normal), Just(ColumnDisplay::Tabbed)]
}

#[derive(Debug, Clone, Arbitrary)]
enum Op {
    AddOutput(#[proptest(strategy = "1..=5usize")] usize),
    AddScaledOutput {
        #[proptest(strategy = "1..=5usize")]
        id: usize,
        #[proptest(strategy = "arbitrary_scale()")]
        scale: f64,
        #[proptest(strategy = "prop::option::of(arbitrary_layout_part().prop_map(Box::new))")]
        layout_config: Option<Box<jiji_config::LayoutPart>>,
    },
    RemoveOutput(#[proptest(strategy = "1..=5usize")] usize),
    FocusOutput(#[proptest(strategy = "1..=5usize")] usize),
    UpdateOutputLayoutConfig {
        #[proptest(strategy = "1..=5usize")]
        id: usize,
        #[proptest(strategy = "prop::option::of(arbitrary_layout_part().prop_map(Box::new))")]
        layout_config: Option<Box<jiji_config::LayoutPart>>,
    },
    AddNamedWorkspace {
        #[proptest(strategy = "1..=5usize")]
        ws_name: usize,
        #[proptest(strategy = "prop::option::of(1..=5usize)")]
        output_name: Option<usize>,
        #[proptest(strategy = "prop::option::of(arbitrary_layout_part().prop_map(Box::new))")]
        layout_config: Option<Box<jiji_config::LayoutPart>>,
    },
    UnnameWorkspace {
        #[proptest(strategy = "1..=5usize")]
        ws_name: usize,
    },
    UpdateWorkspaceLayoutConfig {
        #[proptest(strategy = "1..=5usize")]
        ws_name: usize,
        #[proptest(strategy = "prop::option::of(arbitrary_layout_part().prop_map(Box::new))")]
        layout_config: Option<Box<jiji_config::LayoutPart>>,
    },
    AddWindow {
        params: TestWindowParams,
    },
    AddWindowNextTo {
        params: TestWindowParams,
        #[proptest(strategy = "1..=5usize")]
        next_to_id: usize,
    },
    AddWindowToNamedWorkspace {
        params: TestWindowParams,
        #[proptest(strategy = "1..=5usize")]
        ws_name: usize,
    },
    CloseWindow(#[proptest(strategy = "1..=5usize")] usize),
    FullscreenWindow(#[proptest(strategy = "1..=5usize")] usize),
    SetFullscreenWindow {
        #[proptest(strategy = "1..=5usize")]
        window: usize,
        is_fullscreen: bool,
    },
    ToggleWindowedFullscreen(#[proptest(strategy = "1..=5usize")] usize),
    FocusColumnLeft,
    FocusColumnRight,
    FocusColumnFirst,
    FocusColumnLast,
    FocusColumnRightOrFirst,
    FocusColumnLeftOrLast,
    FocusColumn(#[proptest(strategy = "1..=5usize")] usize),
    FocusWindowOrMonitorUp(#[proptest(strategy = "1..=2u8")] u8),
    FocusWindowOrMonitorDown(#[proptest(strategy = "1..=2u8")] u8),
    FocusColumnOrMonitorLeft(#[proptest(strategy = "1..=2u8")] u8),
    FocusColumnOrMonitorRight(#[proptest(strategy = "1..=2u8")] u8),
    FocusWindowDown,
    FocusWindowUp,
    FocusWindowDownOrColumnLeft,
    FocusWindowDownOrColumnRight,
    FocusWindowUpOrColumnLeft,
    FocusWindowUpOrColumnRight,
    FocusWindowOrWorkspaceDown,
    FocusWindowOrWorkspaceUp,
    FocusWindow(#[proptest(strategy = "1..=5usize")] usize),
    FocusWindowInColumn(#[proptest(strategy = "1..=5u8")] u8),
    FocusWindowTop,
    FocusWindowBottom,
    FocusWindowDownOrTop,
    FocusWindowUpOrBottom,
    MoveColumnLeft,
    MoveColumnRight,
    MoveColumnToFirst,
    MoveColumnToLast,
    MoveColumnLeftOrToMonitorLeft(#[proptest(strategy = "1..=2u8")] u8),
    MoveColumnRightOrToMonitorRight(#[proptest(strategy = "1..=2u8")] u8),
    MoveColumnToIndex(#[proptest(strategy = "1..=5usize")] usize),
    MoveWindowDown,
    MoveWindowUp,
    MoveWindowDownOrToWorkspaceDown,
    MoveWindowUpOrToWorkspaceUp,
    ConsumeOrExpelWindowLeft {
        #[proptest(strategy = "proptest::option::of(1..=5usize)")]
        id: Option<usize>,
    },
    ConsumeOrExpelWindowRight {
        #[proptest(strategy = "proptest::option::of(1..=5usize)")]
        id: Option<usize>,
    },
    ConsumeWindowIntoColumn,
    ExpelWindowFromColumn,
    SwapWindowInDirection(#[proptest(strategy = "arbitrary_scroll_direction()")] ScrollDirection),
    ToggleColumnTabbedDisplay,
    SetColumnDisplay(#[proptest(strategy = "arbitrary_column_display()")] ColumnDisplay),
    CenterColumn,
    CenterWindow {
        #[proptest(strategy = "proptest::option::of(1..=5usize)")]
        id: Option<usize>,
    },
    CenterVisibleColumns,
    FocusWorkspaceDown,
    FocusWorkspaceUp,
    FocusWorkspace(#[proptest(strategy = "0..=4usize")] usize),
    FocusWorkspaceAutoBackAndForth(#[proptest(strategy = "0..=4usize")] usize),
    FocusWorkspacePrevious,
    MoveWindowToWorkspaceDown(bool),
    MoveWindowToWorkspaceUp(bool),
    MoveWindowToWorkspace {
        #[proptest(strategy = "proptest::option::of(1..=5usize)")]
        window_id: Option<usize>,
        #[proptest(strategy = "0..=4usize")]
        workspace_idx: usize,
    },
    MoveColumnToWorkspaceDown(bool),
    MoveColumnToWorkspaceUp(bool),
    MoveColumnToWorkspace(#[proptest(strategy = "0..=4usize")] usize, bool),
    MoveWorkspaceDown,
    MoveWorkspaceUp,
    MoveWorkspaceToIndex {
        #[proptest(strategy = "proptest::option::of(1..=5usize)")]
        ws_name: Option<usize>,
        #[proptest(strategy = "0..=4usize")]
        target_idx: usize,
    },
    MoveWorkspaceToMonitor {
        #[proptest(strategy = "proptest::option::of(1..=5usize)")]
        ws_name: Option<usize>,
        #[proptest(strategy = "0..=5usize")]
        output_id: usize,
    },
    SetWorkspaceName {
        #[proptest(strategy = "1..=5usize")]
        new_ws_name: usize,
        #[proptest(strategy = "proptest::option::of(1..=5usize)")]
        ws_name: Option<usize>,
    },
    UnsetWorkspaceName {
        #[proptest(strategy = "proptest::option::of(1..=5usize)")]
        ws_name: Option<usize>,
    },
    MoveWindowToOutput {
        #[proptest(strategy = "proptest::option::of(1..=5usize)")]
        window_id: Option<usize>,
        #[proptest(strategy = "1..=5usize")]
        output_id: usize,
        #[proptest(strategy = "proptest::option::of(0..=4usize)")]
        target_ws_idx: Option<usize>,
    },
    MoveColumnToOutput {
        #[proptest(strategy = "1..=5usize")]
        output_id: usize,
        #[proptest(strategy = "proptest::option::of(0..=4usize)")]
        target_ws_idx: Option<usize>,
        activate: bool,
    },
    SwitchPresetColumnWidth,
    SwitchPresetColumnWidthBack,
    SwitchPresetWindowWidth {
        #[proptest(strategy = "proptest::option::of(1..=5usize)")]
        id: Option<usize>,
    },
    SwitchPresetWindowWidthBack {
        #[proptest(strategy = "proptest::option::of(1..=5usize)")]
        id: Option<usize>,
    },
    SwitchPresetWindowHeight {
        #[proptest(strategy = "proptest::option::of(1..=5usize)")]
        id: Option<usize>,
    },
    SwitchPresetWindowHeightBack {
        #[proptest(strategy = "proptest::option::of(1..=5usize)")]
        id: Option<usize>,
    },
    MaximizeColumn,
    MaximizeWindowToEdges {
        #[proptest(strategy = "proptest::option::of(1..=5usize)")]
        id: Option<usize>,
    },
    SetColumnWidth(#[proptest(strategy = "arbitrary_size_change()")] SizeChange),
    SetWindowWidth {
        #[proptest(strategy = "proptest::option::of(1..=5usize)")]
        id: Option<usize>,
        #[proptest(strategy = "arbitrary_size_change()")]
        change: SizeChange,
    },
    SetWindowHeight {
        #[proptest(strategy = "proptest::option::of(1..=5usize)")]
        id: Option<usize>,
        #[proptest(strategy = "arbitrary_size_change()")]
        change: SizeChange,
    },
    ResetWindowHeight {
        #[proptest(strategy = "proptest::option::of(1..=5usize)")]
        id: Option<usize>,
    },
    ExpandColumnToAvailableWidth,
    ToggleWindowFloating {
        #[proptest(strategy = "proptest::option::of(1..=5usize)")]
        id: Option<usize>,
    },
    SetWindowFloating {
        #[proptest(strategy = "proptest::option::of(1..=5usize)")]
        id: Option<usize>,
        floating: bool,
    },
    FocusFloating,
    FocusTiling,
    SwitchFocusFloatingTiling,
    MoveFloatingWindow {
        #[proptest(strategy = "proptest::option::of(1..=5usize)")]
        id: Option<usize>,
        #[proptest(strategy = "arbitrary_position_change()")]
        x: PositionChange,
        #[proptest(strategy = "arbitrary_position_change()")]
        y: PositionChange,
        animate: bool,
    },
    SetParent {
        #[proptest(strategy = "1..=5usize")]
        id: usize,
        #[proptest(strategy = "prop::option::of(1..=5usize)")]
        new_parent_id: Option<usize>,
    },
    SetForcedSize {
        #[proptest(strategy = "1..=5usize")]
        id: usize,
        #[proptest(strategy = "proptest::option::of(arbitrary_size())")]
        size: Option<Size<i32, Logical>>,
    },
    Communicate(#[proptest(strategy = "1..=5usize")] usize),
    Refresh {
        is_active: bool,
    },
    AdvanceAnimations {
        #[proptest(strategy = "arbitrary_msec_delta()")]
        msec_delta: i32,
    },
    CompleteAnimations,
    MoveWorkspaceToOutput(#[proptest(strategy = "1..=5usize")] usize),
    ViewOffsetGestureBegin {
        #[proptest(strategy = "1..=5usize")]
        output_idx: usize,
        #[proptest(strategy = "proptest::option::of(0..=4usize)")]
        workspace_idx: Option<usize>,
        is_touchpad: bool,
    },
    ViewOffsetGestureUpdate {
        #[proptest(strategy = "arbitrary_view_offset_gesture_delta()")]
        delta: f64,
        timestamp: Duration,
        is_touchpad: bool,
    },
    ViewOffsetGestureEnd {
        is_touchpad: Option<bool>,
    },
    WorkspaceSwitchGestureBegin {
        #[proptest(strategy = "1..=5usize")]
        output_idx: usize,
        is_touchpad: bool,
    },
    WorkspaceSwitchGestureUpdate {
        #[proptest(strategy = "-400f64..400f64")]
        delta: f64,
        timestamp: Duration,
        is_touchpad: bool,
    },
    WorkspaceSwitchGestureEnd {
        is_touchpad: Option<bool>,
    },
    OverviewGestureBegin,
    OverviewGestureUpdate {
        #[proptest(strategy = "-400f64..400f64")]
        delta: f64,
        timestamp: Duration,
    },
    OverviewGestureEnd,
    InteractiveMoveBegin {
        #[proptest(strategy = "1..=5usize")]
        window: usize,
        #[proptest(strategy = "1..=5usize")]
        output_idx: usize,
        #[proptest(strategy = "-20000f64..20000f64")]
        px: f64,
        #[proptest(strategy = "-20000f64..20000f64")]
        py: f64,
    },
    InteractiveMoveUpdate {
        #[proptest(strategy = "1..=5usize")]
        window: usize,
        #[proptest(strategy = "-20000f64..20000f64")]
        dx: f64,
        #[proptest(strategy = "-20000f64..20000f64")]
        dy: f64,
        #[proptest(strategy = "1..=5usize")]
        output_idx: usize,
        #[proptest(strategy = "-20000f64..20000f64")]
        px: f64,
        #[proptest(strategy = "-20000f64..20000f64")]
        py: f64,
    },
    InteractiveMoveEnd {
        #[proptest(strategy = "1..=5usize")]
        window: usize,
    },
    DndUpdate {
        #[proptest(strategy = "1..=5usize")]
        output_idx: usize,
        #[proptest(strategy = "-20000f64..20000f64")]
        px: f64,
        #[proptest(strategy = "-20000f64..20000f64")]
        py: f64,
    },
    DndEnd,
    InteractiveResizeBegin {
        #[proptest(strategy = "1..=5usize")]
        window: usize,
        #[proptest(strategy = "arbitrary_resize_edge()")]
        edges: ResizeEdge,
    },
    InteractiveResizeUpdate {
        #[proptest(strategy = "1..=5usize")]
        window: usize,
        #[proptest(strategy = "-20000f64..20000f64")]
        dx: f64,
        #[proptest(strategy = "-20000f64..20000f64")]
        dy: f64,
    },
    InteractiveResizeEnd {
        #[proptest(strategy = "1..=5usize")]
        window: usize,
    },
    ToggleOverview,
    UpdateConfig {
        #[proptest(strategy = "arbitrary_layout_part().prop_map(Box::new)")]
        layout_config: Box<jiji_config::LayoutPart>,
    },
}

impl Op {
    fn apply(self, layout: &mut Layout<TestWindow>) {
        match self {
            Op::AddOutput(id) => {
                let name = format!("output{id}");
                if layout.outputs().any(|o| o.name() == name) {
                    return;
                }

                let output = Output::new(
                    name.clone(),
                    PhysicalProperties {
                        size: Size::from((1280, 720)),
                        subpixel: Subpixel::Unknown,
                        make: String::new(),
                        model: String::new(),
                        serial_number: String::new(),
                    },
                );
                output.change_current_state(
                    Some(Mode {
                        size: Size::from((1280, 720)),
                        refresh: 60000,
                    }),
                    None,
                    None,
                    None,
                );
                output.user_data().insert_if_missing(|| OutputName {
                    connector: name,
                    make: None,
                    model: None,
                    serial: None,
                });
                layout.add_output(output.clone(), None);
            }
            Op::AddScaledOutput {
                id,
                scale,
                layout_config,
            } => {
                let name = format!("output{id}");
                if layout.outputs().any(|o| o.name() == name) {
                    return;
                }

                let output = Output::new(
                    name.clone(),
                    PhysicalProperties {
                        size: Size::from((1280, 720)),
                        subpixel: Subpixel::Unknown,
                        make: String::new(),
                        model: String::new(),
                        serial_number: String::new(),
                    },
                );
                output.change_current_state(
                    Some(Mode {
                        size: Size::from((1280, 720)),
                        refresh: 60000,
                    }),
                    None,
                    Some(smithay::output::Scale::Fractional(scale)),
                    None,
                );
                output.user_data().insert_if_missing(|| OutputName {
                    connector: name,
                    make: None,
                    model: None,
                    serial: None,
                });
                layout.add_output(output.clone(), layout_config.map(|x| *x));
            }
            Op::RemoveOutput(id) => {
                let name = format!("output{id}");
                let Some(output) = layout.outputs().find(|o| o.name() == name).cloned() else {
                    return;
                };

                layout.remove_output(&output);
            }
            Op::FocusOutput(id) => {
                let name = format!("output{id}");
                let Some(output) = layout.outputs().find(|o| o.name() == name).cloned() else {
                    return;
                };

                layout.focus_output(&output);
            }
            Op::UpdateOutputLayoutConfig { id, layout_config } => {
                let name = format!("output{id}");
                let seed_activity = layout.active_activity_id();
                let Some(mon_out) = layout
                    .monitors()
                    .find(|m| m.output_name() == &name)
                    .map(|m| m.output_id())
                else {
                    return;
                };
                let (monitors, pool, view) = layout.monitors_pool_view_mut(&mon_out);
                let mon = monitors
                    .iter_mut()
                    .find(|m| m.output_name() == &name)
                    .expect("mon_out matched above");

                mon.update_layout_config(pool, view, layout_config.map(|x| *x), seed_activity);
            }
            Op::AddNamedWorkspace {
                ws_name,
                output_name,
                layout_config,
            } => {
                layout.ensure_named_workspace(&WorkspaceConfig {
                    name: WorkspaceName(format!("ws{ws_name}")),
                    open_on_output: output_name.map(|name| format!("output{name}")),
                    layout: layout_config.map(|x| jiji_config::WorkspaceLayoutPart(*x)),
                    activities: Vec::new(),
                    sticky: None,
                });
            }
            Op::UnnameWorkspace { ws_name } => {
                layout.unname_workspace(&format!("ws{ws_name}"));
            }
            Op::UpdateWorkspaceLayoutConfig {
                ws_name,
                layout_config,
            } => {
                let ws_name = format!("ws{ws_name}");
                let Some(ws) = layout
                    .workspaces_mut()
                    .find(|ws| ws.name() == Some(&ws_name))
                else {
                    return;
                };

                ws.update_layout_config(layout_config.map(|x| *x));
            }
            Op::SetWorkspaceName {
                new_ws_name,
                ws_name,
            } => {
                let ws_ref =
                    ws_name.map(|ws_name| WorkspaceReference::Name(format!("ws{ws_name}")));
                layout.set_workspace_name(format!("ws{new_ws_name}"), ws_ref);
            }
            Op::UnsetWorkspaceName { ws_name } => {
                let ws_ref =
                    ws_name.map(|ws_name| WorkspaceReference::Name(format!("ws{ws_name}")));
                layout.unset_workspace_name(ws_ref);
            }
            Op::AddWindow { mut params } => {
                if layout.has_window(&params.id) {
                    return;
                }
                if let Some(parent_id) = params.parent_id {
                    if parent_id_causes_loop(layout, params.id, parent_id) {
                        params.parent_id = None;
                    }
                }

                let is_floating = params.is_floating;
                let win = TestWindow::new(params);
                layout.add_window(
                    win,
                    AddWindowTarget::Auto,
                    None,
                    None,
                    false,
                    is_floating,
                    ActivateWindow::default(),
                );
            }
            Op::AddWindowNextTo {
                mut params,
                next_to_id,
            } => {
                let mut found_next_to = false;

                if let Some(InteractiveMoveState::Moving(move_)) = &layout.interactive_move {
                    let win_id = move_.tile.window().0.id;
                    if win_id == params.id {
                        return;
                    }
                    if win_id == next_to_id {
                        found_next_to = true;
                    }
                }

                let views_snapshot: Vec<_> = layout
                    .monitors
                    .iter()
                    .map(|mon| layout.active_view(&mon.output_id()).ids().to_vec())
                    .collect();
                let pool = &layout.workspaces;
                for (_mon, view_ids) in layout.monitors.iter().zip(&views_snapshot) {
                    for id in view_ids {
                        let ws = pool.get(id).unwrap();
                        for win in ws.windows() {
                            if win.0.id == params.id {
                                return;
                            }

                            if win.0.id == next_to_id {
                                found_next_to = true;
                            }
                        }
                    }
                }
                for id in &layout.disconnected_workspace_ids {
                    let ws = pool.get(id).unwrap();
                    for win in ws.windows() {
                        if win.0.id == params.id {
                            return;
                        }

                        if win.0.id == next_to_id {
                            found_next_to = true;
                        }
                    }
                }

                if !found_next_to {
                    return;
                }

                if let Some(parent_id) = params.parent_id {
                    if parent_id_causes_loop(layout, params.id, parent_id) {
                        params.parent_id = None;
                    }
                }

                let is_floating = params.is_floating;
                let win = TestWindow::new(params);
                layout.add_window(
                    win,
                    AddWindowTarget::NextTo(&next_to_id),
                    None,
                    None,
                    false,
                    is_floating,
                    ActivateWindow::default(),
                );
            }
            Op::AddWindowToNamedWorkspace {
                mut params,
                ws_name,
            } => {
                let ws_name = format!("ws{ws_name}");
                let mut ws_id = None;

                if let Some(InteractiveMoveState::Moving(move_)) = &layout.interactive_move {
                    if move_.tile.window().0.id == params.id {
                        return;
                    }
                }

                let views_snapshot: Vec<_> = layout
                    .monitors
                    .iter()
                    .map(|mon| layout.active_view(&mon.output_id()).ids().to_vec())
                    .collect();
                let pool = &layout.workspaces;
                for (_mon, view_ids) in layout.monitors.iter().zip(&views_snapshot) {
                    for id in view_ids {
                        let ws = pool.get(id).unwrap();
                        for win in ws.windows() {
                            if win.0.id == params.id {
                                return;
                            }
                        }

                        if ws
                            .name
                            .as_ref()
                            .is_some_and(|name| name.eq_ignore_ascii_case(&ws_name))
                        {
                            ws_id = Some(ws.id());
                        }
                    }
                }
                for id in &layout.disconnected_workspace_ids {
                    let ws = pool.get(id).unwrap();
                    for win in ws.windows() {
                        if win.0.id == params.id {
                            return;
                        }
                    }

                    if ws
                        .name
                        .as_ref()
                        .is_some_and(|name| name.eq_ignore_ascii_case(&ws_name))
                    {
                        ws_id = Some(ws.id());
                    }
                }

                let Some(ws_id) = ws_id else {
                    return;
                };

                if let Some(parent_id) = params.parent_id {
                    if parent_id_causes_loop(layout, params.id, parent_id) {
                        params.parent_id = None;
                    }
                }

                let is_floating = params.is_floating;
                let win = TestWindow::new(params);
                layout.add_window(
                    win,
                    AddWindowTarget::Workspace(ws_id),
                    None,
                    None,
                    false,
                    is_floating,
                    ActivateWindow::default(),
                );
            }
            Op::CloseWindow(id) => {
                layout.remove_window(&id, Transaction::new());
            }
            Op::FullscreenWindow(id) => {
                if !layout.has_window(&id) {
                    return;
                }
                layout.toggle_fullscreen(&id);
            }
            Op::SetFullscreenWindow {
                window,
                is_fullscreen,
            } => {
                if !layout.has_window(&window) {
                    return;
                }
                layout.set_fullscreen(&window, is_fullscreen);
            }
            Op::ToggleWindowedFullscreen(id) => {
                if !layout.has_window(&id) {
                    return;
                }
                layout.toggle_windowed_fullscreen(&id);
            }
            Op::FocusColumnLeft => layout.focus_left(),
            Op::FocusColumnRight => layout.focus_right(),
            Op::FocusColumnFirst => layout.focus_column_first(),
            Op::FocusColumnLast => layout.focus_column_last(),
            Op::FocusColumnRightOrFirst => layout.focus_column_right_or_first(),
            Op::FocusColumnLeftOrLast => layout.focus_column_left_or_last(),
            Op::FocusColumn(index) => layout.focus_column(index),
            Op::FocusWindowOrMonitorUp(id) => {
                let name = format!("output{id}");
                let Some(output) = layout.outputs().find(|o| o.name() == name).cloned() else {
                    return;
                };

                layout.focus_window_up_or_output(&output);
            }
            Op::FocusWindowOrMonitorDown(id) => {
                let name = format!("output{id}");
                let Some(output) = layout.outputs().find(|o| o.name() == name).cloned() else {
                    return;
                };

                layout.focus_window_down_or_output(&output);
            }
            Op::FocusColumnOrMonitorLeft(id) => {
                let name = format!("output{id}");
                let Some(output) = layout.outputs().find(|o| o.name() == name).cloned() else {
                    return;
                };

                layout.focus_column_left_or_output(&output);
            }
            Op::FocusColumnOrMonitorRight(id) => {
                let name = format!("output{id}");
                let Some(output) = layout.outputs().find(|o| o.name() == name).cloned() else {
                    return;
                };

                layout.focus_column_right_or_output(&output);
            }
            Op::FocusWindowDown => layout.focus_down(),
            Op::FocusWindowUp => layout.focus_up(),
            Op::FocusWindowDownOrColumnLeft => layout.focus_down_or_left(),
            Op::FocusWindowDownOrColumnRight => layout.focus_down_or_right(),
            Op::FocusWindowUpOrColumnLeft => layout.focus_up_or_left(),
            Op::FocusWindowUpOrColumnRight => layout.focus_up_or_right(),
            Op::FocusWindowOrWorkspaceDown => layout.focus_window_or_workspace_down(),
            Op::FocusWindowOrWorkspaceUp => layout.focus_window_or_workspace_up(),
            Op::FocusWindow(id) => layout.activate_window(&id),
            Op::FocusWindowInColumn(index) => layout.focus_window_in_column(index),
            Op::FocusWindowTop => layout.focus_window_top(),
            Op::FocusWindowBottom => layout.focus_window_bottom(),
            Op::FocusWindowDownOrTop => layout.focus_window_down_or_top(),
            Op::FocusWindowUpOrBottom => layout.focus_window_up_or_bottom(),
            Op::MoveColumnLeft => layout.move_left(),
            Op::MoveColumnRight => layout.move_right(),
            Op::MoveColumnToFirst => layout.move_column_to_first(),
            Op::MoveColumnToLast => layout.move_column_to_last(),
            Op::MoveColumnLeftOrToMonitorLeft(id) => {
                let name = format!("output{id}");
                let Some(output) = layout.outputs().find(|o| o.name() == name).cloned() else {
                    return;
                };

                layout.move_column_left_or_to_output(&output);
            }
            Op::MoveColumnRightOrToMonitorRight(id) => {
                let name = format!("output{id}");
                let Some(output) = layout.outputs().find(|o| o.name() == name).cloned() else {
                    return;
                };

                layout.move_column_right_or_to_output(&output);
            }
            Op::MoveColumnToIndex(index) => layout.move_column_to_index(index),
            Op::MoveWindowDown => layout.move_down(),
            Op::MoveWindowUp => layout.move_up(),
            Op::MoveWindowDownOrToWorkspaceDown => layout.move_down_or_to_workspace_down(),
            Op::MoveWindowUpOrToWorkspaceUp => layout.move_up_or_to_workspace_up(),
            Op::ConsumeOrExpelWindowLeft { id } => {
                let id = id.filter(|id| layout.has_window(id));
                layout.consume_or_expel_window_left(id.as_ref());
            }
            Op::ConsumeOrExpelWindowRight { id } => {
                let id = id.filter(|id| layout.has_window(id));
                layout.consume_or_expel_window_right(id.as_ref());
            }
            Op::ConsumeWindowIntoColumn => layout.consume_into_column(),
            Op::ExpelWindowFromColumn => layout.expel_from_column(),
            Op::SwapWindowInDirection(direction) => layout.swap_window_in_direction(direction),
            Op::ToggleColumnTabbedDisplay => layout.toggle_column_tabbed_display(),
            Op::SetColumnDisplay(display) => layout.set_column_display(display),
            Op::CenterColumn => layout.center_column(),
            Op::CenterWindow { id } => {
                let id = id.filter(|id| layout.has_window(id));
                layout.center_window(id.as_ref());
            }
            Op::CenterVisibleColumns => layout.center_visible_columns(),
            Op::FocusWorkspaceDown => layout.switch_workspace_down(),
            Op::FocusWorkspaceUp => layout.switch_workspace_up(),
            Op::FocusWorkspace(idx) => layout.switch_workspace(idx),
            Op::FocusWorkspaceAutoBackAndForth(idx) => {
                layout.switch_workspace_auto_back_and_forth(idx)
            }
            Op::FocusWorkspacePrevious => layout.switch_workspace_previous(),
            Op::MoveWindowToWorkspaceDown(focus) => layout.move_to_workspace_down(focus),
            Op::MoveWindowToWorkspaceUp(focus) => layout.move_to_workspace_up(focus),
            Op::MoveWindowToWorkspace {
                window_id,
                workspace_idx,
            } => {
                let window_id = window_id.filter(|id| layout.has_window(id));
                layout.move_to_workspace(window_id.as_ref(), workspace_idx, ActivateWindow::Smart);
            }
            Op::MoveColumnToWorkspaceDown(focus) => layout.move_column_to_workspace_down(focus),
            Op::MoveColumnToWorkspaceUp(focus) => layout.move_column_to_workspace_up(focus),
            Op::MoveColumnToWorkspace(idx, focus) => layout.move_column_to_workspace(idx, focus),
            Op::MoveWindowToOutput {
                window_id,
                output_id: id,
                target_ws_idx,
            } => {
                let name = format!("output{id}");
                let Some(output) = layout.outputs().find(|o| o.name() == name).cloned() else {
                    return;
                };
                let mon_out = layout.monitor_for_output(&output).unwrap().output_id();
                let view_len = layout.active_view(&mon_out).len();

                let window_id = window_id.filter(|id| layout.has_window(id));
                let target_ws_idx = target_ws_idx.filter(|idx| view_len > *idx);
                layout.move_to_output(
                    window_id.as_ref(),
                    &output,
                    target_ws_idx,
                    ActivateWindow::Smart,
                );
            }
            Op::MoveColumnToOutput {
                output_id: id,
                target_ws_idx,
                activate,
            } => {
                let name = format!("output{id}");
                let Some(output) = layout.outputs().find(|o| o.name() == name).cloned() else {
                    return;
                };

                layout.move_column_to_output(&output, target_ws_idx, activate);
            }
            Op::MoveWorkspaceDown => layout.move_workspace_down(),
            Op::MoveWorkspaceUp => layout.move_workspace_up(),
            Op::MoveWorkspaceToIndex {
                ws_name: Some(ws_name),
                target_idx,
            } => {
                if layout.monitors.is_empty() {
                    return;
                }
                let views_snapshot: Vec<_> = layout
                    .monitors
                    .iter()
                    .map(|m| layout.active_view(&m.output_id()).ids().to_vec())
                    .collect();
                let pool = &layout.workspaces;
                let Some((old_idx, old_output)) = layout
                    .monitors
                    .iter()
                    .zip(views_snapshot.iter())
                    .find_map(|(monitor, ids)| {
                        ids.iter()
                            .enumerate()
                            .find_map(|(i, id)| {
                                let ws = pool.get(id).unwrap();
                                if ws.name == Some(format!("ws{ws_name}")) {
                                    Some(i)
                                } else {
                                    None
                                }
                            })
                            .map(|i| (i, monitor.output.clone()))
                    })
                else {
                    return;
                };

                layout.move_workspace_to_idx(Some((Some(old_output), old_idx)), target_idx)
            }
            Op::MoveWorkspaceToIndex {
                ws_name: None,
                target_idx,
            } => layout.move_workspace_to_idx(None, target_idx),
            Op::MoveWorkspaceToMonitor {
                ws_name: None,
                output_id: id,
            } => {
                let name = format!("output{id}");
                let Some(output) = layout.outputs().find(|o| o.name() == name).cloned() else {
                    return;
                };
                layout.move_workspace_to_output(&output);
            }
            Op::MoveWorkspaceToMonitor {
                ws_name: Some(ws_name),
                output_id: id,
            } => {
                let name = format!("output{id}");
                let Some(output) = layout.outputs().find(|o| o.name() == name).cloned() else {
                    return;
                };
                if layout.monitors.is_empty() {
                    return;
                }
                let views_snapshot: Vec<_> = layout
                    .monitors
                    .iter()
                    .map(|m| layout.active_view(&m.output_id()).ids().to_vec())
                    .collect();
                let pool = &layout.workspaces;
                let Some((old_idx, old_output)) = layout
                    .monitors
                    .iter()
                    .zip(views_snapshot.iter())
                    .find_map(|(monitor, ids)| {
                        ids.iter()
                            .enumerate()
                            .find_map(|(i, id)| {
                                let ws = pool.get(id).unwrap();
                                if ws.name == Some(format!("ws{ws_name}")) {
                                    Some(i)
                                } else {
                                    None
                                }
                            })
                            .map(|i| (i, monitor.output.clone()))
                    })
                else {
                    return;
                };

                layout.move_workspace_to_output_by_id(old_idx, Some(old_output), &output);
            }
            Op::SwitchPresetColumnWidth => layout.toggle_width(true),
            Op::SwitchPresetColumnWidthBack => layout.toggle_width(false),
            Op::SwitchPresetWindowWidth { id } => {
                let id = id.filter(|id| layout.has_window(id));
                layout.toggle_window_width(id.as_ref(), true);
            }
            Op::SwitchPresetWindowWidthBack { id } => {
                let id = id.filter(|id| layout.has_window(id));
                layout.toggle_window_width(id.as_ref(), false);
            }
            Op::SwitchPresetWindowHeight { id } => {
                let id = id.filter(|id| layout.has_window(id));
                layout.toggle_window_height(id.as_ref(), true);
            }
            Op::SwitchPresetWindowHeightBack { id } => {
                let id = id.filter(|id| layout.has_window(id));
                layout.toggle_window_height(id.as_ref(), false);
            }
            Op::MaximizeColumn => layout.toggle_full_width(),
            Op::MaximizeWindowToEdges { id } => {
                let id = id.or_else(|| layout.focus().map(|win| *win.id()));
                let Some(id) = id else {
                    return;
                };
                if !layout.has_window(&id) {
                    return;
                }
                layout.toggle_maximized(&id);
            }
            Op::SetColumnWidth(change) => layout.set_column_width(change),
            Op::SetWindowWidth { id, change } => {
                let id = id.filter(|id| layout.has_window(id));
                layout.set_window_width(id.as_ref(), change);
            }
            Op::SetWindowHeight { id, change } => {
                let id = id.filter(|id| layout.has_window(id));
                layout.set_window_height(id.as_ref(), change);
            }
            Op::ResetWindowHeight { id } => {
                let id = id.filter(|id| layout.has_window(id));
                layout.reset_window_height(id.as_ref());
            }
            Op::ExpandColumnToAvailableWidth => layout.expand_column_to_available_width(),
            Op::ToggleWindowFloating { id } => {
                let id = id.filter(|id| layout.has_window(id));
                layout.toggle_window_floating(id.as_ref());
            }
            Op::SetWindowFloating { id, floating } => {
                let id = id.filter(|id| layout.has_window(id));
                layout.set_window_floating(id.as_ref(), floating);
            }
            Op::FocusFloating => {
                layout.focus_floating();
            }
            Op::FocusTiling => {
                layout.focus_tiling();
            }
            Op::SwitchFocusFloatingTiling => {
                layout.switch_focus_floating_tiling();
            }
            Op::MoveFloatingWindow { id, x, y, animate } => {
                let id = id.filter(|id| layout.has_window(id));
                layout.move_floating_window(id.as_ref(), x, y, animate);
            }
            Op::SetParent {
                id,
                mut new_parent_id,
            } => {
                if !layout.has_window(&id) {
                    return;
                }

                if let Some(parent_id) = new_parent_id {
                    if parent_id_causes_loop(layout, id, parent_id) {
                        new_parent_id = None;
                    }
                }

                let mut update = false;

                if let Some(InteractiveMoveState::Moving(move_)) = &layout.interactive_move {
                    if move_.tile.window().0.id == id {
                        move_.tile.window().0.parent_id.set(new_parent_id);
                        update = true;
                    }
                }

                let views_snapshot: Vec<_> = layout
                    .monitors
                    .iter()
                    .map(|m| layout.active_view(&m.output_id()).ids().to_vec())
                    .collect();
                let pool = &layout.workspaces;
                'outer: {
                    for view_ids in &views_snapshot {
                        for id_ in view_ids {
                            let ws = pool.get(id_).unwrap();
                            for win in ws.windows() {
                                if win.0.id == id {
                                    win.0.parent_id.set(new_parent_id);
                                    update = true;
                                    break 'outer;
                                }
                            }
                        }
                    }
                    for id_ in &layout.disconnected_workspace_ids {
                        let ws = pool.get(id_).unwrap();
                        for win in ws.windows() {
                            if win.0.id == id {
                                win.0.parent_id.set(new_parent_id);
                                update = true;
                                break 'outer;
                            }
                        }
                    }
                }

                if update {
                    if let Some(new_parent_id) = new_parent_id {
                        layout.descendants_added(&new_parent_id);
                    }
                }
            }
            Op::SetForcedSize { id, size } => {
                for (_mon, win) in layout.windows() {
                    if win.0.id == id {
                        win.0.forced_size.set(size);
                        return;
                    }
                }
            }
            Op::Communicate(id) => {
                let mut update = false;

                if let Some(InteractiveMoveState::Moving(move_)) = &layout.interactive_move {
                    if move_.tile.window().0.id == id {
                        if move_.tile.window().communicate() {
                            update = true;
                        }

                        if update {
                            // FIXME: serial.
                            layout.update_window(&id, None);
                        }
                        return;
                    }
                }

                let views_snapshot: Vec<_> = layout
                    .monitors
                    .iter()
                    .map(|m| layout.active_view(&m.output_id()).ids().to_vec())
                    .collect();
                let pool = &layout.workspaces;
                'outer: {
                    for view_ids in &views_snapshot {
                        for id_ in view_ids {
                            let ws = pool.get(id_).unwrap();
                            for win in ws.windows() {
                                if win.0.id == id {
                                    if win.communicate() {
                                        update = true;
                                    }
                                    break 'outer;
                                }
                            }
                        }
                    }
                    for id_ in &layout.disconnected_workspace_ids {
                        let ws = pool.get(id_).unwrap();
                        for win in ws.windows() {
                            if win.0.id == id {
                                if win.communicate() {
                                    update = true;
                                }
                                break 'outer;
                            }
                        }
                    }
                }

                if update {
                    // FIXME: serial.
                    layout.update_window(&id, None);
                }
            }
            Op::Refresh { is_active } => {
                layout.refresh(is_active);
            }
            Op::AdvanceAnimations { msec_delta } => {
                let mut now = layout.clock.now_unadjusted();
                if msec_delta >= 0 {
                    now = now.saturating_add(Duration::from_millis(msec_delta as u64));
                } else {
                    now = now.saturating_sub(Duration::from_millis(-msec_delta as u64));
                }
                layout.clock.set_unadjusted(now);
                layout.advance_animations();
            }
            Op::CompleteAnimations => {
                layout.clock.set_complete_instantly(true);
                layout.advance_animations();
                layout.clock.set_complete_instantly(false);
            }
            Op::MoveWorkspaceToOutput(id) => {
                let name = format!("output{id}");
                let Some(output) = layout.outputs().find(|o| o.name() == name).cloned() else {
                    return;
                };

                layout.move_workspace_to_output(&output);
            }
            Op::ViewOffsetGestureBegin {
                output_idx: id,
                workspace_idx,
                is_touchpad: normalize,
            } => {
                let name = format!("output{id}");
                let Some(output) = layout.outputs().find(|o| o.name() == name).cloned() else {
                    return;
                };

                layout.view_offset_gesture_begin(&output, workspace_idx, normalize);
            }
            Op::ViewOffsetGestureUpdate {
                delta,
                timestamp,
                is_touchpad,
            } => {
                layout.view_offset_gesture_update(delta, timestamp, is_touchpad);
            }
            Op::ViewOffsetGestureEnd { is_touchpad } => {
                layout.view_offset_gesture_end(is_touchpad);
            }
            Op::WorkspaceSwitchGestureBegin {
                output_idx: id,
                is_touchpad,
            } => {
                let name = format!("output{id}");
                let Some(output) = layout.outputs().find(|o| o.name() == name).cloned() else {
                    return;
                };

                layout.workspace_switch_gesture_begin(&output, is_touchpad);
            }
            Op::WorkspaceSwitchGestureUpdate {
                delta,
                timestamp,
                is_touchpad,
            } => {
                layout.workspace_switch_gesture_update(delta, timestamp, is_touchpad);
            }
            Op::WorkspaceSwitchGestureEnd { is_touchpad } => {
                layout.workspace_switch_gesture_end(is_touchpad);
            }
            Op::OverviewGestureBegin => {
                layout.overview_gesture_begin();
            }
            Op::OverviewGestureUpdate { delta, timestamp } => {
                layout.overview_gesture_update(delta, timestamp);
            }
            Op::OverviewGestureEnd => {
                layout.overview_gesture_end();
            }
            Op::InteractiveMoveBegin {
                window,
                output_idx,
                px,
                py,
            } => {
                let name = format!("output{output_idx}");
                let Some(output) = layout.outputs().find(|o| o.name() == name).cloned() else {
                    return;
                };
                layout.interactive_move_begin(window, &output, Point::from((px, py)));
            }
            Op::InteractiveMoveUpdate {
                window,
                dx,
                dy,
                output_idx,
                px,
                py,
            } => {
                let name = format!("output{output_idx}");
                let Some(output) = layout.outputs().find(|o| o.name() == name).cloned() else {
                    return;
                };
                layout.interactive_move_update(
                    &window,
                    Point::from((dx, dy)),
                    output,
                    Point::from((px, py)),
                );
            }
            Op::InteractiveMoveEnd { window } => {
                layout.interactive_move_end(&window);
            }
            Op::DndUpdate { output_idx, px, py } => {
                let name = format!("output{output_idx}");
                let Some(output) = layout.outputs().find(|o| o.name() == name).cloned() else {
                    return;
                };
                layout.dnd_update(output, Point::from((px, py)));
            }
            Op::DndEnd => {
                layout.dnd_end();
            }
            Op::InteractiveResizeBegin { window, edges } => {
                layout.interactive_resize_begin(window, edges);
            }
            Op::InteractiveResizeUpdate { window, dx, dy } => {
                layout.interactive_resize_update(&window, Point::from((dx, dy)));
            }
            Op::InteractiveResizeEnd { window } => {
                layout.interactive_resize_end(&window);
            }
            Op::ToggleOverview => {
                layout.toggle_overview();
            }
            Op::UpdateConfig { layout_config } => {
                let options = Options {
                    layout: jiji_config::Layout::from_part(&layout_config),
                    ..Default::default()
                };

                layout.update_options(options);
            }
        }
    }
}

#[track_caller]
fn check_ops_on_layout(layout: &mut Layout<TestWindow>, ops: impl IntoIterator<Item = Op>) {
    for op in ops {
        op.apply(layout);
        layout.verify_invariants();
        verify_output_bindings(layout);
    }
}

/// Asserts the bind/unbind symmetry contract from `Workspace::bind_output`: every window is
/// marked (via `output_enter`) against exactly the `Output` of the `Monitor` that owns its
/// workspace, and windows on disconnected workspaces carry no bindings. Runs after every `Op`
/// in `check_ops_on_layout` so `check_ops` proptest sequences catch any site that rebinds
/// without unbinding, forgets to unbind on transfer, or drops the Smithay markers.
#[track_caller]
fn verify_output_bindings(layout: &Layout<TestWindow>) {
    let pool = layout.workspace_pool();
    for mon in &layout.monitors {
        let view = layout.active_view(&mon.output_id());
        for id in view.ids() {
            let ws = pool.get(id).unwrap();
            for win in ws.windows() {
                let bound = win.bound_outputs();
                assert_eq!(
                    bound,
                    vec![mon.output.clone()],
                    "window {:?} on monitor {} must be bound to exactly that monitor's \
                     output; got {:?}",
                    win.id(),
                    mon.output.name(),
                    bound.iter().map(|o| o.name()).collect::<Vec<_>>(),
                );
            }
        }
    }
    for id in &layout.disconnected_workspace_ids {
        let ws = pool.get(id).unwrap();
        for win in ws.windows() {
            let bound = win.bound_outputs();
            assert!(
                bound.is_empty(),
                "window {:?} on a disconnected workspace must have no bound outputs; got {:?}",
                win.id(),
                bound.iter().map(|o| o.name()).collect::<Vec<_>>(),
            );
        }
    }
}

#[track_caller]
fn check_ops(ops: impl IntoIterator<Item = Op>) -> Layout<TestWindow> {
    let mut layout = Layout::default();
    check_ops_on_layout(&mut layout, ops);
    layout
}

#[track_caller]
fn check_ops_with_options(
    options: Options,
    ops: impl IntoIterator<Item = Op>,
) -> Layout<TestWindow> {
    let mut layout = Layout::with_options(Clock::with_time(Duration::ZERO), options);
    check_ops_on_layout(&mut layout, ops);
    layout
}

/// Test-side helper: install a freshly-minted [`Activity`] into the pool and immediately
/// materialize bookend views on every connected monitor.
///
/// Production code creates activities via `Layout::create_activity`, which atomically inserts
/// and materializes. The lower-level `Activities::insert` bypass is `pub(super)` and reachable
/// only from this test module; it leaves the new activity without views, violating the
/// per-activity bookend invariant the next time `verify_invariants` runs. This helper restores
/// that invariant in a single call.
#[track_caller]
fn test_insert_activity(layout: &mut Layout<TestWindow>, activity: super::activity::Activity) {
    layout.activities.insert(activity);
    layout.ensure_all_activity_views();
}

/// Mint a fresh empty workspace tagged exclusively to `activity_id`, bound to `mon_idx`'s
/// output, and inserted into the pool. Returns the new id. Used by tests that need to
/// hand-roll dormant views with a proper trailing-empty bookend.
#[track_caller]
fn test_mint_empty_for(
    layout: &mut Layout<TestWindow>,
    mon_idx: usize,
    activity_id: ActivityId,
) -> WorkspaceId {
    let mon = &layout.monitors[mon_idx];
    let ws = Workspace::new(
        &mon.output,
        HashSet::from([activity_id]),
        layout.clock.clone(),
        mon.options.clone(),
    );
    let id = ws.id();
    assert!(
        layout.workspaces.insert(id, ws).is_none(),
        "fresh id must be unique",
    );
    id
}

/// Test-side helper: override `activity_id`'s view for `output_id` with a hand-rolled
/// `new_view`, dropping any materializer-installed bookend workspaces that the override no
/// longer references so the pool-keys union invariant stays satisfied.
///
/// Direct `activity.views_mut().insert(...)` calls from tests that bypass the materializer
/// leak the materializer's freshly-minted bookend workspaces into the pool — they remain pool
/// keys but no view references them, tripping `Layout::verify_invariants`' "pool keys must
/// equal the union of every activity's views over all outputs plus disconnected_workspace_ids"
/// assertion. This helper performs the override and the cleanup atomically.
#[track_caller]
fn test_override_activity_view(
    layout: &mut Layout<TestWindow>,
    activity_id: ActivityId,
    output_id: OutputId,
    new_view: WorkspaceView,
) {
    let materialized: Vec<WorkspaceId> = layout
        .activities
        .get(activity_id)
        .expect("activity must be a live key")
        .views()
        .get(&output_id)
        .map(|v| v.ids().to_vec())
        .unwrap_or_default();
    let keep: HashSet<WorkspaceId> = new_view.ids().iter().copied().collect();
    layout
        .activities
        .get_mut(activity_id)
        .expect("activity must be a live key")
        .views_mut()
        .insert(output_id, new_view);
    for id in materialized {
        if !keep.contains(&id) {
            layout.workspaces.remove(&id);
        }
    }
}

#[test]
fn operations_dont_panic() {
    if std::env::var_os("RUN_SLOW_TESTS").is_none() {
        eprintln!("ignoring slow test");
        return;
    }

    let every_op = [
        Op::AddOutput(0),
        Op::AddOutput(1),
        Op::AddOutput(2),
        Op::RemoveOutput(0),
        Op::RemoveOutput(1),
        Op::RemoveOutput(2),
        Op::FocusOutput(0),
        Op::FocusOutput(1),
        Op::FocusOutput(2),
        Op::AddNamedWorkspace {
            ws_name: 1,
            output_name: Some(1),
            layout_config: None,
        },
        Op::UnnameWorkspace { ws_name: 1 },
        Op::AddWindow {
            params: TestWindowParams::new(0),
        },
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::AddWindowNextTo {
            params: TestWindowParams::new(2),
            next_to_id: 1,
        },
        Op::AddWindowToNamedWorkspace {
            params: TestWindowParams::new(3),
            ws_name: 1,
        },
        Op::CloseWindow(0),
        Op::CloseWindow(1),
        Op::CloseWindow(2),
        Op::FullscreenWindow(1),
        Op::FullscreenWindow(2),
        Op::FullscreenWindow(3),
        Op::MaximizeWindowToEdges { id: Some(1) },
        Op::MaximizeWindowToEdges { id: Some(2) },
        Op::MaximizeWindowToEdges { id: Some(3) },
        Op::FocusColumnLeft,
        Op::FocusColumnRight,
        Op::FocusColumnRightOrFirst,
        Op::FocusColumnLeftOrLast,
        Op::FocusWindowOrMonitorUp(0),
        Op::FocusWindowOrMonitorDown(1),
        Op::FocusColumnOrMonitorLeft(0),
        Op::FocusColumnOrMonitorRight(1),
        Op::FocusWindowUp,
        Op::FocusWindowUpOrColumnLeft,
        Op::FocusWindowUpOrColumnRight,
        Op::FocusWindowOrWorkspaceUp,
        Op::FocusWindowDown,
        Op::FocusWindowDownOrColumnLeft,
        Op::FocusWindowDownOrColumnRight,
        Op::FocusWindowOrWorkspaceDown,
        Op::MoveColumnLeft,
        Op::MoveColumnRight,
        Op::MoveColumnLeftOrToMonitorLeft(0),
        Op::MoveColumnRightOrToMonitorRight(1),
        Op::ConsumeWindowIntoColumn,
        Op::ExpelWindowFromColumn,
        Op::CenterColumn,
        Op::FocusWorkspaceDown,
        Op::FocusWorkspaceUp,
        Op::FocusWorkspace(1),
        Op::FocusWorkspace(2),
        Op::MoveWindowToWorkspaceDown(true),
        Op::MoveWindowToWorkspaceUp(true),
        Op::MoveWindowToWorkspace {
            window_id: None,
            workspace_idx: 1,
        },
        Op::MoveWindowToWorkspace {
            window_id: None,
            workspace_idx: 2,
        },
        Op::MoveColumnToWorkspaceDown(true),
        Op::MoveColumnToWorkspaceUp(true),
        Op::MoveColumnToWorkspace(1, true),
        Op::MoveColumnToWorkspace(2, true),
        Op::MoveWindowDown,
        Op::MoveWindowDownOrToWorkspaceDown,
        Op::MoveWindowUp,
        Op::MoveWindowUpOrToWorkspaceUp,
        Op::ConsumeOrExpelWindowLeft { id: None },
        Op::ConsumeOrExpelWindowRight { id: None },
        Op::MoveWorkspaceToOutput(1),
        Op::ToggleColumnTabbedDisplay,
    ];

    for third in &every_op {
        for second in &every_op {
            for first in &every_op {
                // eprintln!("{first:?}, {second:?}, {third:?}");

                let mut layout = Layout::default();
                first.clone().apply(&mut layout);
                layout.verify_invariants();
                verify_output_bindings(&layout);
                second.clone().apply(&mut layout);
                layout.verify_invariants();
                verify_output_bindings(&layout);
                third.clone().apply(&mut layout);
                layout.verify_invariants();
                verify_output_bindings(&layout);
            }
        }
    }
}

#[test]
fn operations_from_starting_state_dont_panic() {
    if std::env::var_os("RUN_SLOW_TESTS").is_none() {
        eprintln!("ignoring slow test");
        return;
    }

    // Running every op from an empty state doesn't get us to all the interesting states. So,
    // also run it from a manually-created starting state with more things going on to exercise
    // more code paths.
    let setup_ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::MoveWindowToWorkspaceDown(true),
        Op::AddWindow {
            params: TestWindowParams::new(2),
        },
        Op::AddWindow {
            params: TestWindowParams::new(3),
        },
        Op::FocusColumnLeft,
        Op::ConsumeWindowIntoColumn,
        Op::AddWindow {
            params: TestWindowParams::new(4),
        },
        Op::AddOutput(2),
        Op::AddWindow {
            params: TestWindowParams::new(5),
        },
        Op::MoveWindowToOutput {
            window_id: None,
            output_id: 2,
            target_ws_idx: None,
        },
        Op::FocusOutput(1),
        Op::Communicate(1),
        Op::Communicate(2),
        Op::Communicate(3),
        Op::Communicate(4),
        Op::Communicate(5),
    ];

    let every_op = [
        Op::AddOutput(0),
        Op::AddOutput(1),
        Op::AddOutput(2),
        Op::RemoveOutput(0),
        Op::RemoveOutput(1),
        Op::RemoveOutput(2),
        Op::FocusOutput(0),
        Op::FocusOutput(1),
        Op::FocusOutput(2),
        Op::AddNamedWorkspace {
            ws_name: 1,
            output_name: Some(1),
            layout_config: None,
        },
        Op::UnnameWorkspace { ws_name: 1 },
        Op::AddWindow {
            params: TestWindowParams::new(0),
        },
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::AddWindow {
            params: TestWindowParams::new(2),
        },
        Op::AddWindowNextTo {
            params: TestWindowParams::new(6),
            next_to_id: 0,
        },
        Op::AddWindowNextTo {
            params: TestWindowParams::new(7),
            next_to_id: 1,
        },
        Op::AddWindowToNamedWorkspace {
            params: TestWindowParams::new(5),
            ws_name: 1,
        },
        Op::CloseWindow(0),
        Op::CloseWindow(1),
        Op::CloseWindow(2),
        Op::FullscreenWindow(1),
        Op::FullscreenWindow(2),
        Op::FullscreenWindow(3),
        Op::MaximizeWindowToEdges { id: Some(1) },
        Op::MaximizeWindowToEdges { id: Some(2) },
        Op::MaximizeWindowToEdges { id: Some(3) },
        Op::SetFullscreenWindow {
            window: 1,
            is_fullscreen: false,
        },
        Op::SetFullscreenWindow {
            window: 1,
            is_fullscreen: true,
        },
        Op::SetFullscreenWindow {
            window: 2,
            is_fullscreen: false,
        },
        Op::SetFullscreenWindow {
            window: 2,
            is_fullscreen: true,
        },
        Op::FocusColumnLeft,
        Op::FocusColumnRight,
        Op::FocusColumnRightOrFirst,
        Op::FocusColumnLeftOrLast,
        Op::FocusWindowOrMonitorUp(0),
        Op::FocusWindowOrMonitorDown(1),
        Op::FocusColumnOrMonitorLeft(0),
        Op::FocusColumnOrMonitorRight(1),
        Op::FocusWindowUp,
        Op::FocusWindowUpOrColumnLeft,
        Op::FocusWindowUpOrColumnRight,
        Op::FocusWindowOrWorkspaceUp,
        Op::FocusWindowDown,
        Op::FocusWindowDownOrColumnLeft,
        Op::FocusWindowDownOrColumnRight,
        Op::FocusWindowOrWorkspaceDown,
        Op::MoveColumnLeft,
        Op::MoveColumnRight,
        Op::MoveColumnLeftOrToMonitorLeft(0),
        Op::MoveColumnRightOrToMonitorRight(1),
        Op::ConsumeWindowIntoColumn,
        Op::ExpelWindowFromColumn,
        Op::CenterColumn,
        Op::FocusWorkspaceDown,
        Op::FocusWorkspaceUp,
        Op::FocusWorkspace(1),
        Op::FocusWorkspace(2),
        Op::FocusWorkspace(3),
        Op::MoveWindowToWorkspaceDown(true),
        Op::MoveWindowToWorkspaceUp(true),
        Op::MoveWindowToWorkspace {
            window_id: None,
            workspace_idx: 1,
        },
        Op::MoveWindowToWorkspace {
            window_id: None,
            workspace_idx: 2,
        },
        Op::MoveWindowToWorkspace {
            window_id: None,
            workspace_idx: 3,
        },
        Op::MoveColumnToWorkspaceDown(true),
        Op::MoveColumnToWorkspaceUp(true),
        Op::MoveColumnToWorkspace(1, true),
        Op::MoveColumnToWorkspace(2, true),
        Op::MoveColumnToWorkspace(3, true),
        Op::MoveWindowDown,
        Op::MoveWindowDownOrToWorkspaceDown,
        Op::MoveWindowUp,
        Op::MoveWindowUpOrToWorkspaceUp,
        Op::ConsumeOrExpelWindowLeft { id: None },
        Op::ConsumeOrExpelWindowRight { id: None },
        Op::ToggleColumnTabbedDisplay,
    ];

    for third in &every_op {
        for second in &every_op {
            for first in &every_op {
                // eprintln!("{first:?}, {second:?}, {third:?}");

                let mut layout = Layout::default();
                for op in &setup_ops {
                    op.clone().apply(&mut layout);
                }

                let mut layout = Layout::default();
                first.clone().apply(&mut layout);
                layout.verify_invariants();
                verify_output_bindings(&layout);
                second.clone().apply(&mut layout);
                layout.verify_invariants();
                verify_output_bindings(&layout);
                third.clone().apply(&mut layout);
                layout.verify_invariants();
                verify_output_bindings(&layout);
            }
        }
    }
}

#[test]
fn primary_active_workspace_idx_not_updated_on_output_add() {
    let ops = [
        Op::AddOutput(1),
        Op::AddOutput(2),
        Op::FocusOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(0),
        },
        Op::FocusOutput(2),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::RemoveOutput(2),
        Op::FocusWorkspace(3),
        Op::AddOutput(2),
    ];

    check_ops(ops);
}

#[test]
fn named_workspace_reattaches_to_same_output_after_reconnect() {
    // A named workspace originally owned by output1 carries a window through a disconnect and
    // reconnect. On reconnect, it must reattach to output1 (not to a sibling output that stayed
    // connected), and the window must still be present on it.
    let ops = [
        Op::AddOutput(1),
        Op::AddOutput(2),
        Op::AddNamedWorkspace {
            ws_name: 1,
            output_name: Some(1),
            layout_config: None,
        },
        Op::AddWindowToNamedWorkspace {
            params: TestWindowParams::new(0),
            ws_name: 1,
        },
        Op::RemoveOutput(1),
        Op::AddOutput(1),
    ];

    let layout = check_ops(ops);

    let (_idx, ws) = layout
        .find_workspace_by_name("ws1")
        .expect("named workspace must still exist after reconnect");
    assert!(
        ws.has_window(&0),
        "window added to named workspace must persist through disconnect/reconnect",
    );
    let output = layout
        .output_for_workspace(ws.id())
        .expect("named workspace must be bound to a connected output after reconnect");
    assert_eq!(
        output.name(),
        "output1",
        "named workspace must reattach to its original output after reconnect",
    );
}

#[test]
fn window_closed_on_previous_workspace() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(0),
        },
        Op::FocusWorkspaceDown,
        Op::CloseWindow(0),
    ];

    check_ops(ops);
}

#[test]
fn removing_output_must_keep_empty_focus_on_primary() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(0),
        },
        Op::AddOutput(2),
        Op::RemoveOutput(1),
    ];

    let layout = check_ops(ops);

    let mon_out = layout.monitors[0].output_id();
    // The workspace from the removed output was inserted at position 0, so the active workspace
    // must change to 1 to keep the focus on the empty workspace.
    assert_eq!(layout.active_view(&mon_out).active_position(), 1);
}

#[test]
fn move_to_workspace_by_idx_does_not_leave_empty_workspaces() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(0),
        },
        Op::AddOutput(2),
        Op::FocusOutput(2),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::RemoveOutput(1),
        Op::MoveWindowToWorkspace {
            window_id: Some(0),
            workspace_idx: 2,
        },
    ];

    let layout = check_ops(ops);

    let mon_out = layout.monitors[0].output_id();
    let view = layout.active_view(&mon_out);
    assert!(layout.monitors[0]
        .workspace_at(layout.workspace_pool(), view, 1)
        .has_windows());
}

#[test]
fn empty_workspaces_dont_move_back_to_original_output() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::FocusWorkspaceDown,
        Op::AddWindow {
            params: TestWindowParams::new(2),
        },
        Op::AddOutput(2),
        Op::RemoveOutput(1),
        Op::FocusWorkspace(1),
        Op::CloseWindow(1),
        Op::AddOutput(1),
    ];

    check_ops(ops);
}

#[test]
fn named_workspaces_dont_update_original_output_on_adding_window() {
    let ops = [
        Op::AddOutput(1),
        Op::SetWorkspaceName {
            new_ws_name: 1,
            ws_name: None,
        },
        Op::AddOutput(2),
        Op::RemoveOutput(1),
        Op::FocusWorkspaceUp,
        // Adding a window updates the original output for unnamed workspaces.
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        // Connecting the previous output should move the named workspace back since its
        // original output wasn't updated.
        Op::AddOutput(1),
    ];

    let layout = check_ops(ops);
    let (mon, _, ws) = layout
        .workspaces()
        .find(|(_, _, ws)| ws.name().is_some())
        .unwrap();
    assert!(ws.name().is_some()); // Sanity check.
    let mon = mon.unwrap();
    assert_eq!(mon.output_name(), "output1");
}

#[test]
fn workspaces_update_original_output_on_moving_to_same_output() {
    let ops = [
        Op::AddOutput(1),
        Op::SetWorkspaceName {
            new_ws_name: 1,
            ws_name: None,
        },
        Op::AddOutput(2),
        Op::RemoveOutput(1),
        Op::FocusWorkspaceUp,
        Op::MoveWorkspaceToOutput(2),
        Op::AddOutput(1),
    ];

    let layout = check_ops(ops);
    let (mon, _, ws) = layout
        .workspaces()
        .find(|(_, _, ws)| ws.name().is_some())
        .unwrap();
    assert!(ws.name().is_some()); // Sanity check.
    let mon = mon.unwrap();
    assert_eq!(mon.output_name(), "output2");
}

#[test]
fn workspaces_update_original_output_on_moving_to_same_monitor() {
    let ops = [
        Op::AddOutput(1),
        Op::SetWorkspaceName {
            new_ws_name: 1,
            ws_name: None,
        },
        Op::AddOutput(2),
        Op::RemoveOutput(1),
        Op::FocusWorkspaceUp,
        Op::MoveWorkspaceToMonitor {
            ws_name: Some(1),
            output_id: 2,
        },
        Op::AddOutput(1),
    ];

    let layout = check_ops(ops);
    let (mon, _, ws) = layout
        .workspaces()
        .find(|(_, _, ws)| ws.name().is_some())
        .unwrap();
    assert!(ws.name().is_some()); // Sanity check.
    let mon = mon.unwrap();
    assert_eq!(mon.output_name(), "output2");
}

#[test]
fn large_negative_height_change() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::SetWindowHeight {
            id: None,
            change: SizeChange::AdjustProportion(-1e129),
        },
    ];

    let mut options = Options::default();
    options.layout.border.off = false;
    options.layout.border.width = 1.;

    check_ops_with_options(options, ops);
}

#[test]
fn large_max_size() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams {
                min_max_size: (Size::from((0, 0)), Size::from((i32::MAX, i32::MAX))),
                ..TestWindowParams::new(1)
            },
        },
    ];

    let mut options = Options::default();
    options.layout.border.off = false;
    options.layout.border.width = 1.;

    check_ops_with_options(options, ops);
}

#[test]
fn workspace_cleanup_during_switch() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::FocusWorkspaceDown,
        Op::CloseWindow(1),
    ];

    check_ops(ops);
}

#[test]
fn workspace_transfer_during_switch() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::AddOutput(2),
        Op::FocusOutput(2),
        Op::AddWindow {
            params: TestWindowParams::new(2),
        },
        Op::RemoveOutput(1),
        Op::FocusWorkspaceDown,
        Op::FocusWorkspaceDown,
        Op::AddOutput(1),
    ];

    check_ops(ops);
}

#[test]
fn workspace_transfer_during_switch_from_last() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::AddOutput(2),
        Op::RemoveOutput(1),
        Op::FocusWorkspaceUp,
        Op::AddOutput(1),
    ];

    check_ops(ops);
}

#[test]
fn workspace_transfer_during_switch_gets_cleaned_up() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::RemoveOutput(1),
        Op::AddOutput(2),
        Op::MoveColumnToWorkspaceDown(true),
        Op::MoveColumnToWorkspaceDown(true),
        Op::AddOutput(1),
    ];

    check_ops(ops);
}

#[test]
fn move_workspace_to_output() {
    let ops = [
        Op::AddOutput(1),
        Op::AddOutput(2),
        Op::FocusOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(0),
        },
        Op::MoveWorkspaceToOutput(2),
    ];

    let layout = check_ops(ops);

    let active_monitor_idx = layout.active_monitor_idx;
    let mon0_out = layout.monitors[0].output_id();
    let mon1_out = layout.monitors[1].output_id();
    let view0 = layout.active_view(&mon0_out).clone();
    let view1 = layout.active_view(&mon1_out).clone();

    let pool = layout.workspace_pool();
    let monitors = &layout.monitors;
    assert_eq!(active_monitor_idx, 1);
    assert_eq!(view0.len(), 1);
    assert!(!monitors[0].workspace_at(pool, &view0, 0).has_windows());
    assert_eq!(view1.active_position(), 0);
    assert_eq!(view1.len(), 2);
    assert!(monitors[1].workspace_at(pool, &view1, 0).has_windows());
}

#[test]
fn open_right_of_on_different_workspace() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::FocusWorkspaceDown,
        Op::AddWindow {
            params: TestWindowParams::new(2),
        },
        Op::AddWindowNextTo {
            params: TestWindowParams::new(3),
            next_to_id: 1,
        },
    ];

    let layout = check_ops(ops);

    let mon_out = layout.monitors[0].output_id();
    let view = layout.active_view(&mon_out).clone();
    let pool = layout.workspace_pool();
    let monitors = &layout.monitors;

    let mon = &monitors[0];
    assert_eq!(
        view.active_position(),
        1,
        "the second workspace must remain active"
    );
    assert_eq!(
        mon.workspace_at(pool, &view, 0)
            .scrolling()
            .active_column_idx(),
        1,
        "the new window must become active"
    );
}

#[test]
// empty_workspace_above_first = true
fn open_right_of_on_different_workspace_ewaf() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::FocusWorkspaceDown,
        Op::AddWindow {
            params: TestWindowParams::new(2),
        },
        Op::AddWindowNextTo {
            params: TestWindowParams::new(3),
            next_to_id: 1,
        },
    ];

    let options = Options {
        layout: jiji_config::Layout {
            empty_workspace_above_first: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let layout = check_ops_with_options(options, ops);

    let mon_out = layout.monitors[0].output_id();
    let view = layout.active_view(&mon_out).clone();
    let pool = layout.workspace_pool();
    let monitors = &layout.monitors;

    let mon = &monitors[0];
    assert_eq!(
        view.active_position(),
        2,
        "the second workspace must remain active"
    );
    assert_eq!(
        mon.workspace_at(pool, &view, 1)
            .scrolling()
            .active_column_idx(),
        1,
        "the new window must become active"
    );
}

#[test]
fn removing_all_outputs_preserves_empty_named_workspaces() {
    let ops = [
        Op::AddOutput(1),
        Op::AddNamedWorkspace {
            ws_name: 1,
            output_name: None,
            layout_config: None,
        },
        Op::AddNamedWorkspace {
            ws_name: 2,
            output_name: None,
            layout_config: None,
        },
        Op::RemoveOutput(1),
    ];

    let layout = check_ops(ops);

    assert!(
        layout.monitors.is_empty(),
        "removing the only output should leave no monitors",
    );
    assert_eq!(layout.disconnected_workspace_ids.len(), 2);
}

#[test]
fn config_change_updates_cached_sizes() {
    let mut config = Config::default();
    let border = &mut config.layout.border;
    border.off = false;
    border.width = 2.;

    let mut layout = Layout::new(Clock::default(), &config);

    Op::AddWindow {
        params: TestWindowParams {
            bbox: Rectangle::from_size(Size::from((1280, 200))),
            ..TestWindowParams::new(1)
        },
    }
    .apply(&mut layout);

    config.layout.border.width = 4.;
    layout.update_config(&config);

    layout.verify_invariants();
    verify_output_bindings(&layout);
}

#[test]
fn preset_height_change_removes_preset() {
    let mut config = Config::default();
    config.layout.preset_window_heights = vec![PresetSize::Fixed(1), PresetSize::Fixed(2)];

    let mut layout = Layout::new(Clock::default(), &config);

    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::AddWindow {
            params: TestWindowParams::new(2),
        },
        Op::ConsumeOrExpelWindowLeft { id: None },
        Op::SwitchPresetWindowHeight { id: None },
        Op::SwitchPresetWindowHeight { id: None },
    ];
    for op in ops {
        op.apply(&mut layout);
    }

    // Leave only one.
    config.layout.preset_window_heights = vec![PresetSize::Fixed(1)];

    layout.update_config(&config);

    layout.verify_invariants();
    verify_output_bindings(&layout);
}

#[test]
fn set_window_height_recomputes_to_auto() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(0),
        },
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::ConsumeOrExpelWindowLeft { id: None },
        Op::AddWindow {
            params: TestWindowParams::new(2),
        },
        Op::ConsumeOrExpelWindowLeft { id: None },
        Op::SetWindowHeight {
            id: None,
            change: SizeChange::SetFixed(100),
        },
        Op::FocusWindowUp,
        Op::SetWindowHeight {
            id: None,
            change: SizeChange::SetFixed(200),
        },
    ];

    check_ops(ops);
}

#[test]
fn one_window_in_column_becomes_weight_1() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(0),
        },
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::ConsumeOrExpelWindowLeft { id: None },
        Op::AddWindow {
            params: TestWindowParams::new(2),
        },
        Op::ConsumeOrExpelWindowLeft { id: None },
        Op::SetWindowHeight {
            id: None,
            change: SizeChange::SetFixed(100),
        },
        Op::Communicate(2),
        Op::FocusWindowUp,
        Op::SetWindowHeight {
            id: None,
            change: SizeChange::SetFixed(200),
        },
        Op::Communicate(1),
        Op::CloseWindow(0),
        Op::CloseWindow(1),
    ];

    check_ops(ops);
}

#[test]
fn fixed_height_takes_max_non_auto_into_account() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(0),
        },
        Op::SetWindowHeight {
            id: Some(0),
            change: SizeChange::SetFixed(704),
        },
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::ConsumeOrExpelWindowLeft { id: None },
    ];

    let options = Options {
        layout: jiji_config::Layout {
            border: jiji_config::Border {
                off: false,
                width: 4.,
                ..Default::default()
            },
            gaps: 0.,
            ..Default::default()
        },
        ..Default::default()
    };
    check_ops_with_options(options, ops);
}

#[test]
fn start_interactive_move_then_remove_window() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(0),
        },
        Op::InteractiveMoveBegin {
            window: 0,
            output_idx: 1,
            px: 0.,
            py: 0.,
        },
        Op::CloseWindow(0),
    ];

    check_ops(ops);
}

#[test]
fn interactive_move_onto_empty_output() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(0),
        },
        Op::InteractiveMoveBegin {
            window: 0,
            output_idx: 1,
            px: 0.,
            py: 0.,
        },
        Op::AddOutput(2),
        Op::InteractiveMoveUpdate {
            window: 0,
            dx: 1000.,
            dy: 0.,
            output_idx: 2,
            px: 0.,
            py: 0.,
        },
        Op::InteractiveMoveEnd { window: 0 },
    ];

    check_ops(ops);
}

#[test]
fn interactive_move_onto_empty_output_ewaf() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(0),
        },
        Op::InteractiveMoveBegin {
            window: 0,
            output_idx: 1,
            px: 0.,
            py: 0.,
        },
        Op::AddOutput(2),
        Op::InteractiveMoveUpdate {
            window: 0,
            dx: 1000.,
            dy: 0.,
            output_idx: 2,
            px: 0.,
            py: 0.,
        },
        Op::InteractiveMoveEnd { window: 0 },
    ];

    let options = Options {
        layout: jiji_config::Layout {
            empty_workspace_above_first: true,
            ..Default::default()
        },
        ..Default::default()
    };
    check_ops_with_options(options, ops);
}

#[test]
fn interactive_move_onto_last_workspace() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(0),
        },
        Op::InteractiveMoveBegin {
            window: 0,
            output_idx: 1,
            px: 0.,
            py: 0.,
        },
        Op::InteractiveMoveUpdate {
            window: 0,
            dx: 1000.,
            dy: 0.,
            output_idx: 1,
            px: 0.,
            py: 0.,
        },
        Op::FocusWorkspaceDown,
        Op::AdvanceAnimations { msec_delta: 1000 },
        Op::InteractiveMoveEnd { window: 0 },
    ];

    check_ops(ops);
}

#[test]
fn interactive_move_onto_first_empty_workspace() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::InteractiveMoveBegin {
            window: 1,
            output_idx: 1,
            px: 0.,
            py: 0.,
        },
        Op::InteractiveMoveUpdate {
            window: 1,
            dx: 1000.,
            dy: 0.,
            output_idx: 1,
            px: 0.,
            py: 0.,
        },
        Op::FocusWorkspaceUp,
        Op::AdvanceAnimations { msec_delta: 1000 },
        Op::InteractiveMoveEnd { window: 1 },
    ];
    let options = Options {
        layout: jiji_config::Layout {
            empty_workspace_above_first: true,
            ..Default::default()
        },
        ..Default::default()
    };
    check_ops_with_options(options, ops);
}

#[test]
fn output_active_workspace_is_preserved() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::FocusWorkspaceDown,
        Op::AddWindow {
            params: TestWindowParams::new(2),
        },
        Op::RemoveOutput(1),
        Op::AddOutput(1),
    ];

    let layout = check_ops(ops);

    let mon_out = layout.monitors[0].output_id();
    assert_eq!(layout.active_view(&mon_out).active_position(), 1);
}

#[test]
fn output_active_workspace_is_preserved_with_other_outputs() {
    let ops = [
        Op::AddOutput(1),
        Op::AddOutput(2),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::FocusWorkspaceDown,
        Op::AddWindow {
            params: TestWindowParams::new(2),
        },
        Op::RemoveOutput(1),
        Op::AddOutput(1),
    ];

    let layout = check_ops(ops);

    let mon_out = layout.monitors[1].output_id();
    assert_eq!(layout.active_view(&mon_out).active_position(), 1);
}

#[test]
fn named_workspace_to_output() {
    let ops = [
        Op::AddNamedWorkspace {
            ws_name: 1,
            output_name: None,
            layout_config: None,
        },
        Op::AddOutput(1),
        Op::MoveWorkspaceToOutput(1),
        Op::FocusWorkspaceUp,
    ];
    check_ops(ops);
}

#[test]
// empty_workspace_above_first = true
fn named_workspace_to_output_ewaf() {
    let ops = [
        Op::AddNamedWorkspace {
            ws_name: 1,
            output_name: Some(2),
            layout_config: None,
        },
        Op::AddOutput(1),
        Op::AddOutput(2),
    ];
    let options = Options {
        layout: jiji_config::Layout {
            empty_workspace_above_first: true,
            ..Default::default()
        },
        ..Default::default()
    };
    check_ops_with_options(options, ops);
}

#[test]
fn move_window_to_empty_workspace_above_first() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::MoveWorkspaceUp,
        Op::MoveWorkspaceDown,
        Op::FocusWorkspaceUp,
        Op::MoveWorkspaceDown,
    ];
    let options = Options {
        layout: jiji_config::Layout {
            empty_workspace_above_first: true,
            ..Default::default()
        },
        ..Default::default()
    };
    check_ops_with_options(options, ops);
}

#[test]
fn move_window_to_different_output() {
    let ops = [
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::AddOutput(1),
        Op::AddOutput(2),
        Op::MoveWorkspaceToOutput(2),
    ];
    let options = Options {
        layout: jiji_config::Layout {
            empty_workspace_above_first: true,
            ..Default::default()
        },
        ..Default::default()
    };
    check_ops_with_options(options, ops);
}

#[test]
fn close_window_empty_ws_above_first() {
    let ops = [
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::AddOutput(1),
        Op::CloseWindow(1),
    ];
    let options = Options {
        layout: jiji_config::Layout {
            empty_workspace_above_first: true,
            ..Default::default()
        },
        ..Default::default()
    };
    check_ops_with_options(options, ops);
}

#[test]
fn add_and_remove_output() {
    let ops = [
        Op::AddOutput(2),
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::RemoveOutput(2),
    ];
    let options = Options {
        layout: jiji_config::Layout {
            empty_workspace_above_first: true,
            ..Default::default()
        },
        ..Default::default()
    };
    check_ops_with_options(options, ops);
}

// Move the currently-focused forced-top-empty workspace under
// `empty_workspace_above_first`. Focus follows the moved workspace — the
// same way the trailing bottom-empty case always behaved — so
// `clean_up_workspaces` skips the active cursor's position and the moved
// empty stays at position 3.
#[test]
fn move_workspace_to_idx_active_at_top_empty_under_ewaf() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::FocusWorkspaceDown,
        Op::AddWindow {
            params: TestWindowParams::new(2),
        },
        Op::FocusWorkspaceDown,
        Op::AddWindow {
            params: TestWindowParams::new(3),
        },
        Op::FocusWorkspace(0),
        Op::MoveWorkspaceToIndex {
            ws_name: None,
            target_idx: 2,
        },
    ];
    let options = Options {
        layout: jiji_config::Layout {
            empty_workspace_above_first: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let layout = check_ops_with_options(options, ops);

    let mon_out = layout.monitors[0].output_id();
    let view = layout.active_view(&mon_out).clone();
    let pool = layout.workspace_pool();
    let named: Vec<bool> = view
        .ids()
        .iter()
        .map(|id| pool.get(id).unwrap().has_windows_or_name())
        .collect();

    assert_eq!(view.len(), 6, "expected 6 workspaces, got {named:?}");
    assert_eq!(
        named,
        vec![false, true, true, false, true, false],
        "expected [E, W, W, E, W, E]"
    );
    assert_eq!(
        view.active_position(),
        3,
        "focus should follow the moved empty"
    );
}

#[test]
fn add_output_consolidation_preserves_ewaf_pin() {
    // Exercises `Layout::add_output`'s `keep_active_pinned` branch: when a
    // reconnecting output's original workspaces are moved off the primary under
    // `empty_workspace_above_first`, focus must stay on the first named
    // workspace (position 1), not slip onto the forced-empty position 0.
    let ops = [
        Op::AddOutput(1),
        Op::AddOutput(2),
        Op::FocusOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(0),
        },
        Op::FocusOutput(2),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::RemoveOutput(2),
        Op::AddOutput(2),
    ];
    let options = Options {
        layout: jiji_config::Layout {
            empty_workspace_above_first: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let layout = check_ops_with_options(options, ops);

    let mon_out = layout.monitors[0].output_id();
    assert_eq!(layout.active_view(&mon_out).active_position(), 1);
}

#[test]
fn switch_ewaf_on() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
    ];

    let mut layout = check_ops(ops);
    layout.update_options(Options {
        layout: jiji_config::Layout {
            empty_workspace_above_first: true,
            ..Default::default()
        },
        ..Default::default()
    });
    layout.verify_invariants();
    verify_output_bindings(&layout);
}

#[test]
fn switch_ewaf_off() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
    ];

    let options = Options {
        layout: jiji_config::Layout {
            empty_workspace_above_first: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut layout = check_ops_with_options(options, ops);
    layout.update_options(Options::default());
    layout.verify_invariants();
    verify_output_bindings(&layout);
}

#[test]
fn interactive_move_drop_on_other_output_during_animation() {
    let ops = [
        Op::AddOutput(3),
        Op::AddWindow {
            params: TestWindowParams::new(3),
        },
        Op::InteractiveMoveBegin {
            window: 3,
            output_idx: 3,
            px: 0.0,
            py: 0.0,
        },
        Op::FocusWorkspaceDown,
        Op::AddOutput(4),
        Op::InteractiveMoveUpdate {
            window: 3,
            dx: 0.0,
            dy: 8300.68619826683,
            output_idx: 4,
            px: 0.0,
            py: 0.0,
        },
        Op::RemoveOutput(4),
        Op::InteractiveMoveEnd { window: 3 },
    ];
    check_ops(ops);
}

#[test]
fn add_window_next_to_only_interactively_moved_without_outputs() {
    let ops = [
        Op::AddWindow {
            params: TestWindowParams::new(2),
        },
        Op::AddOutput(1),
        Op::InteractiveMoveBegin {
            window: 2,
            output_idx: 1,
            px: 0.0,
            py: 0.0,
        },
        Op::InteractiveMoveUpdate {
            window: 2,
            dx: 0.0,
            dy: 3586.692842955048,
            output_idx: 1,
            px: 0.0,
            py: 0.0,
        },
        Op::RemoveOutput(1),
        // We have no outputs, and the only existing window is interactively moved, meaning there
        // are no workspaces either.
        Op::AddWindowNextTo {
            params: TestWindowParams::new(3),
            next_to_id: 2,
        },
    ];

    check_ops(ops);
}

#[test]
fn interactive_move_toggle_floating_ends_dnd_gesture() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(2),
        },
        Op::InteractiveMoveBegin {
            window: 2,
            output_idx: 1,
            px: 0.0,
            py: 0.0,
        },
        Op::InteractiveMoveUpdate {
            window: 2,
            dx: 0.0,
            dy: 3586.692842955048,
            output_idx: 1,
            px: 0.0,
            py: 0.0,
        },
        Op::Refresh { is_active: false },
        Op::ToggleWindowFloating { id: None },
        Op::InteractiveMoveEnd { window: 2 },
    ];

    check_ops(ops);
}

#[test]
fn interactive_move_from_workspace_with_layout_config() {
    let ops = [
        Op::AddNamedWorkspace {
            ws_name: 1,
            output_name: Some(2),
            layout_config: Some(Box::new(jiji_config::LayoutPart {
                border: Some(jiji_config::BorderRule {
                    on: true,
                    ..Default::default()
                }),
                ..Default::default()
            })),
        },
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(2),
        },
        Op::InteractiveMoveBegin {
            window: 2,
            output_idx: 1,
            px: 0.0,
            py: 0.0,
        },
        Op::InteractiveMoveUpdate {
            window: 2,
            dx: 0.0,
            dy: 3586.692842955048,
            output_idx: 1,
            px: 0.0,
            py: 0.0,
        },
        // Now remove and add the output. It will have the same workspace.
        Op::RemoveOutput(1),
        Op::AddOutput(1),
        Op::InteractiveMoveUpdate {
            window: 2,
            dx: 0.0,
            dy: 0.0,
            output_idx: 1,
            px: 0.0,
            py: 0.0,
        },
        // Now move onto a different workspace.
        Op::FocusWorkspaceDown,
        Op::CompleteAnimations,
        Op::InteractiveMoveUpdate {
            window: 2,
            dx: 0.0,
            dy: 0.0,
            output_idx: 1,
            px: 0.0,
            py: 0.0,
        },
    ];

    check_ops(ops);
}

#[test]
fn add_window_to_named_workspace_with_distinct_layout_config() {
    // Regression guard for the pre-fix `monitors[mon_idx].workspace_at_mut(pool, 0).make_tile(_)`
    // proxy in `Layout::add_window_on`. The proxy built the new `Tile` against workspace[0]'s
    // `Rc<Options>`, so `Column::new_with_tile` baked `options.layout.default_column_display`
    // from workspace[0] into `column.display_mode`. `ScrollingSpace::add_column` then
    // re-stamps `column.options` and every tile's options with the target's `Rc<Options>`
    // via `Column::update_config` — which makes `verify_invariants`'s
    // `Rc::ptr_eq(&self.options, &column.options)` guard at `scrolling.rs` trivially hold —
    // but `display_mode` is one of the fields `update_config` never recomputes, so it
    // survives as the only observable footprint of the bug.
    //
    // Setup: the target named workspace has `default_column_display: Tabbed`; the pre-existing
    // trailing empty (workspace[0] from the proxy's perspective) keeps the default `Normal`.
    // A second named workspace is prepended so the target lands at view position 1 — the
    // targeted-`AddWindow` path that the proxy would have short-circuited through
    // workspace[0]. After the add, the column on the target must report `Tabbed`; with the
    // proxy in place it would report `Normal`.
    let ops = [
        Op::AddOutput(1),
        Op::AddNamedWorkspace {
            ws_name: 1,
            output_name: Some(1),
            layout_config: Some(Box::new(jiji_config::LayoutPart {
                default_column_display: Some(ColumnDisplay::Tabbed),
                ..Default::default()
            })),
        },
        Op::AddNamedWorkspace {
            ws_name: 2,
            output_name: Some(1),
            layout_config: None,
        },
        Op::AddWindowToNamedWorkspace {
            params: TestWindowParams::new(1),
            ws_name: 1,
        },
    ];

    let layout = check_ops(ops);

    let mon_out = layout.monitors[0].output_id();
    let view = layout.active_view(&mon_out).clone();
    let pool = layout.workspace_pool();
    let mon = &layout.monitors[0];
    // View after the two `AddNamedWorkspace` ops: [ws2 @ 0, ws1 @ 1, default_empty @ 2].
    // `ws1` is the `Tabbed` target, at a non-zero position — the proxy would have read
    // `default_column_display` from `ws2` (position 0) instead.
    let col = mon
        .workspace_at(pool, &view, 1)
        .scrolling()
        .columns()
        .next()
        .expect("a column was just added to ws1");
    assert_eq!(
        col.display_mode(),
        ColumnDisplay::Tabbed,
        "column must be stamped from the target workspace's options, not workspace[0]'s",
    );
}

#[test]
fn set_width_fixed_negative() {
    let ops = [
        Op::AddOutput(3),
        Op::AddWindow {
            params: TestWindowParams::new(3),
        },
        Op::ToggleWindowFloating { id: Some(3) },
        Op::SetColumnWidth(SizeChange::SetFixed(-100)),
    ];
    check_ops(ops);
}

#[test]
fn set_height_fixed_negative() {
    let ops = [
        Op::AddOutput(3),
        Op::AddWindow {
            params: TestWindowParams::new(3),
        },
        Op::ToggleWindowFloating { id: Some(3) },
        Op::SetWindowHeight {
            id: None,
            change: SizeChange::SetFixed(-100),
        },
    ];
    check_ops(ops);
}

#[test]
fn interactive_resize_to_negative() {
    let ops = [
        Op::AddOutput(3),
        Op::AddWindow {
            params: TestWindowParams::new(3),
        },
        Op::ToggleWindowFloating { id: Some(3) },
        Op::InteractiveResizeBegin {
            window: 3,
            edges: ResizeEdge::BOTTOM_RIGHT,
        },
        Op::InteractiveResizeUpdate {
            window: 3,
            dx: -10000.,
            dy: -10000.,
        },
    ];
    check_ops(ops);
}

#[test]
fn windows_on_other_workspaces_remain_activated() {
    let ops = [
        Op::AddOutput(3),
        Op::AddWindow {
            params: TestWindowParams::new(3),
        },
        Op::FocusWorkspaceDown,
        Op::Refresh { is_active: true },
    ];

    let layout = check_ops(ops);
    let (_, win) = layout.windows().next().unwrap();
    assert!(win.0.pending_activated.get());
}

#[test]
fn stacking_add_parent_brings_up_child() {
    let ops = [
        Op::AddOutput(0),
        Op::AddWindow {
            params: TestWindowParams {
                is_floating: true,
                parent_id: Some(1),
                ..TestWindowParams::new(0)
            },
        },
        Op::AddWindow {
            params: TestWindowParams {
                is_floating: true,
                ..TestWindowParams::new(1)
            },
        },
    ];

    check_ops(ops);
}

#[test]
fn stacking_add_parent_brings_up_descendants() {
    let ops = [
        Op::AddOutput(0),
        Op::AddWindow {
            params: TestWindowParams {
                is_floating: true,
                parent_id: Some(2),
                ..TestWindowParams::new(0)
            },
        },
        Op::AddWindow {
            params: TestWindowParams {
                is_floating: true,
                parent_id: Some(0),
                ..TestWindowParams::new(1)
            },
        },
        Op::AddWindow {
            params: TestWindowParams {
                is_floating: true,
                ..TestWindowParams::new(2)
            },
        },
    ];

    check_ops(ops);
}

#[test]
fn stacking_activate_brings_up_descendants() {
    let ops = [
        Op::AddOutput(0),
        Op::AddWindow {
            params: TestWindowParams {
                is_floating: true,
                ..TestWindowParams::new(0)
            },
        },
        Op::AddWindow {
            params: TestWindowParams {
                is_floating: true,
                parent_id: Some(0),
                ..TestWindowParams::new(1)
            },
        },
        Op::AddWindow {
            params: TestWindowParams {
                is_floating: true,
                parent_id: Some(1),
                ..TestWindowParams::new(2)
            },
        },
        Op::AddWindow {
            params: TestWindowParams {
                is_floating: true,
                ..TestWindowParams::new(3)
            },
        },
        Op::FocusWindow(0),
    ];

    check_ops(ops);
}

#[test]
fn stacking_set_parent_brings_up_child() {
    let ops = [
        Op::AddOutput(0),
        Op::AddWindow {
            params: TestWindowParams {
                is_floating: true,
                ..TestWindowParams::new(0)
            },
        },
        Op::AddWindow {
            params: TestWindowParams {
                is_floating: true,
                ..TestWindowParams::new(1)
            },
        },
        Op::SetParent {
            id: 0,
            new_parent_id: Some(1),
        },
    ];

    check_ops(ops);
}

#[test]
fn move_window_to_workspace_with_different_active_output() {
    let ops = [
        Op::AddOutput(0),
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(0),
        },
        Op::FocusOutput(1),
        Op::MoveWindowToWorkspace {
            window_id: Some(0),
            workspace_idx: 2,
        },
    ];

    check_ops(ops);
}

#[test]
fn set_first_workspace_name() {
    let ops = [
        Op::AddOutput(0),
        Op::SetWorkspaceName {
            new_ws_name: 0,
            ws_name: None,
        },
    ];

    check_ops(ops);
}

#[test]
fn set_first_workspace_name_ewaf() {
    let ops = [
        Op::AddOutput(0),
        Op::SetWorkspaceName {
            new_ws_name: 0,
            ws_name: None,
        },
    ];

    let options = Options {
        layout: jiji_config::Layout {
            empty_workspace_above_first: true,
            ..Default::default()
        },
        ..Default::default()
    };
    check_ops_with_options(options, ops);
}

#[test]
fn set_last_workspace_name() {
    let ops = [
        Op::AddOutput(0),
        Op::AddWindow {
            params: TestWindowParams::new(0),
        },
        Op::FocusWorkspaceDown,
        Op::SetWorkspaceName {
            new_ws_name: 0,
            ws_name: None,
        },
    ];

    check_ops(ops);
}

#[test]
fn move_workspace_to_same_monitor_doesnt_reorder() {
    let ops = [
        Op::AddOutput(0),
        Op::SetWorkspaceName {
            new_ws_name: 0,
            ws_name: None,
        },
        Op::AddWindow {
            params: TestWindowParams::new(0),
        },
        Op::FocusWorkspaceDown,
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::AddWindow {
            params: TestWindowParams::new(2),
        },
        Op::MoveWorkspaceToMonitor {
            ws_name: Some(0),
            output_id: 0,
        },
    ];

    let layout = check_ops(ops);
    let counts: Vec<_> = layout
        .workspaces()
        .map(|(_, _, ws)| ws.windows().count())
        .collect();
    assert_eq!(counts, &[1, 2, 0]);
}

#[test]
fn removing_window_above_preserves_focused_window() {
    let ops = [
        Op::AddOutput(0),
        Op::AddWindow {
            params: TestWindowParams::new(0),
        },
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::AddWindow {
            params: TestWindowParams::new(2),
        },
        Op::FocusColumnFirst,
        Op::ConsumeWindowIntoColumn,
        Op::ConsumeWindowIntoColumn,
        Op::FocusWindowDown,
        Op::CloseWindow(0),
    ];

    let layout = check_ops(ops);
    let win = layout.focus().unwrap();
    assert_eq!(win.0.id, 1);
}

#[test]
fn preset_column_width_fixed_correct_with_border() {
    let ops = [
        Op::AddOutput(0),
        Op::AddWindow {
            params: TestWindowParams::new(0),
        },
        Op::SwitchPresetColumnWidth,
    ];

    let options = Options {
        layout: jiji_config::Layout {
            preset_column_widths: vec![PresetSize::Fixed(500)],
            ..Default::default()
        },
        ..Default::default()
    };
    let mut layout = check_ops_with_options(options, ops);

    let win = layout.windows().next().unwrap().1;
    assert_eq!(win.requested_size().unwrap().w, 500);

    // Add border.
    let options = Options {
        layout: jiji_config::Layout {
            preset_column_widths: vec![PresetSize::Fixed(500)],
            border: jiji_config::Border {
                off: false,
                width: 5.,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    layout.update_options(options);

    // With border, the window gets less size.
    let win = layout.windows().next().unwrap().1;
    assert_eq!(win.requested_size().unwrap().w, 490);

    // However, preset fixed width will still work correctly.
    layout.toggle_width(true);
    let win = layout.windows().next().unwrap().1;
    assert_eq!(win.requested_size().unwrap().w, 500);
}

#[test]
fn preset_column_width_reset_after_set_width() {
    let ops = [
        Op::AddOutput(0),
        Op::AddWindow {
            params: TestWindowParams::new(0),
        },
        Op::SwitchPresetColumnWidth,
        Op::SetWindowWidth {
            id: None,
            change: SizeChange::AdjustFixed(-10),
        },
        Op::SwitchPresetColumnWidth,
    ];

    let options = Options {
        layout: jiji_config::Layout {
            preset_column_widths: vec![PresetSize::Fixed(500), PresetSize::Fixed(1000)],
            ..Default::default()
        },
        ..Default::default()
    };
    let layout = check_ops_with_options(options, ops);
    let win = layout.windows().next().unwrap().1;
    assert_eq!(win.requested_size().unwrap().w, 500);
}

#[test]
fn move_column_to_workspace_unfocused_with_multiple_monitors() {
    let ops = [
        Op::AddOutput(1),
        Op::SetWorkspaceName {
            new_ws_name: 101,
            ws_name: None,
        },
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::FocusWorkspaceDown,
        Op::SetWorkspaceName {
            new_ws_name: 102,
            ws_name: None,
        },
        Op::AddWindow {
            params: TestWindowParams::new(2),
        },
        Op::AddOutput(2),
        Op::FocusOutput(2),
        Op::SetWorkspaceName {
            new_ws_name: 201,
            ws_name: None,
        },
        Op::AddWindow {
            params: TestWindowParams::new(3),
        },
        Op::AddWindow {
            params: TestWindowParams::new(4),
        },
        Op::MoveColumnToOutput {
            output_id: 1,
            target_ws_idx: Some(0),
            activate: false,
        },
        Op::FocusOutput(1),
    ];

    let layout = check_ops(ops);

    assert_eq!(layout.active_workspace().unwrap().name().unwrap(), "ws102");

    let pool = layout.workspace_pool();
    for (mon, win) in layout.windows() {
        let mon = mon.unwrap();
        let view = layout.active_view(&mon.output_id());
        let ws = view
            .ids()
            .iter()
            .filter_map(|id| pool.get(id))
            .find(|w| w.has_window(win.id()))
            .unwrap();

        assert_eq!(
            ws.name().unwrap(),
            match win.id() {
                1 | 4 => "ws101",
                2 => "ws102",
                3 => "ws201",
                _ => unreachable!(),
            }
        );
    }
}

#[test]
fn move_column_to_workspace_down_focus_false_on_floating_window() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::AddWindow {
            params: TestWindowParams::new(2),
        },
        Op::ToggleWindowFloating { id: None },
        Op::MoveColumnToWorkspaceDown(false),
    ];

    let layout = check_ops(ops);

    let mon_out = layout.monitors[0].output_id();
    assert_eq!(layout.active_view(&mon_out).active_position(), 0);
}

#[test]
fn move_column_to_workspace_focus_false_on_floating_window() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::AddWindow {
            params: TestWindowParams::new(2),
        },
        Op::ToggleWindowFloating { id: None },
        Op::MoveColumnToWorkspace(1, false),
    ];

    let layout = check_ops(ops);

    let mon_out = layout.monitors[0].output_id();
    assert_eq!(layout.active_view(&mon_out).active_position(), 0);
}

#[test]
fn restore_to_floating_persists_across_fullscreen_maximize() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::ToggleWindowFloating { id: None },
        // Maximize then fullscreen.
        Op::MaximizeWindowToEdges { id: None },
        Op::FullscreenWindow(1),
        // Unfullscreen.
        Op::FullscreenWindow(1),
    ];

    let mut layout = check_ops(ops);

    // Unfullscreening should return the window to the maximized state.
    let scrolling = layout.active_workspace().unwrap().scrolling();
    assert!(scrolling.tiles().next().is_some());

    let ops = [
        // Unmaximize.
        Op::MaximizeWindowToEdges { id: None },
    ];
    check_ops_on_layout(&mut layout, ops);

    // Unmaximize should return the window back to floating.
    let scrolling = layout.active_workspace().unwrap().scrolling();
    assert!(scrolling.tiles().next().is_none());
}

#[test]
fn unmaximize_during_fullscreen_does_not_float() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::ToggleWindowFloating { id: None },
        // Maximize then fullscreen.
        Op::MaximizeWindowToEdges { id: None },
        Op::FullscreenWindow(1),
        // Unmaximize.
        Op::MaximizeWindowToEdges { id: None },
    ];

    let mut layout = check_ops(ops);

    // Unmaximize shouldn't have changed the window state since it's fullscreen.
    let scrolling = layout.active_workspace().unwrap().scrolling();
    assert!(scrolling.tiles().next().is_some());

    let ops = [
        // Unfullscreen.
        Op::FullscreenWindow(1),
    ];
    check_ops_on_layout(&mut layout, ops);

    // Unfullscreen should return the window back to floating.
    let scrolling = layout.active_workspace().unwrap().scrolling();
    assert!(scrolling.tiles().next().is_none());
}

#[test]
fn move_column_to_workspace_maximize_and_fullscreen() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::MaximizeWindowToEdges { id: None },
        Op::FullscreenWindow(1),
        Op::MoveColumnToWorkspaceDown(true),
        Op::FullscreenWindow(1),
    ];

    let layout = check_ops(ops);
    let (_, win) = layout.windows().next().unwrap();

    // Unfullscreening should return to maximized because the window was maximized before.
    assert_eq!(win.pending_sizing_mode(), SizingMode::Maximized);
}

#[test]
fn move_window_to_workspace_maximize_and_fullscreen() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::MaximizeWindowToEdges { id: None },
        Op::FullscreenWindow(1),
        Op::MoveWindowToWorkspaceDown(true),
        Op::FullscreenWindow(1),
    ];

    let layout = check_ops(ops);
    let (_, win) = layout.windows().next().unwrap();

    // Unfullscreening should return to maximized because the window was maximized before.
    //
    // FIXME: it currently doesn't because windows themselves can only be either fullscreen or
    // maximized. So when a window is fullscreen, whether it is also maximized or not is stored in
    // the column. MoveWindowToWorkspace removes the window from the column and this information is
    // forgotten.
    assert_eq!(win.pending_sizing_mode(), SizingMode::Normal);
}

#[test]
fn tabs_with_different_border() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams {
                rules: Some(ResolvedWindowRules {
                    border: jiji_config::BorderRule {
                        on: true,
                        ..Default::default()
                    },
                    ..ResolvedWindowRules::default()
                }),
                ..TestWindowParams::new(2)
            },
        },
        Op::SwitchPresetWindowHeight { id: None },
        Op::ToggleColumnTabbedDisplay,
        Op::AddWindow {
            params: TestWindowParams::new(3),
        },
        Op::ConsumeOrExpelWindowLeft { id: None },
    ];

    let options = Options {
        layout: jiji_config::Layout {
            struts: Struts {
                left: FloatOrInt(0.),
                right: FloatOrInt(0.),
                top: FloatOrInt(20000.),
                bottom: FloatOrInt(0.),
            },
            ..Default::default()
        },
        ..Default::default()
    };
    check_ops_with_options(options, ops);
}

#[test]
fn expel_pending_left_from_fullscreen_tabbed_column() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::FullscreenWindow(1),
        Op::Communicate(1),
        // 1 is now fullscreen, view_offset_to_restore is set.
        Op::ToggleColumnTabbedDisplay,
        Op::AddWindow {
            params: TestWindowParams::new(2),
        },
        Op::ConsumeOrExpelWindowLeft { id: Some(2) },
        // 2 is consumed into a fullscreen column, fullscreen is requested but not applied.
        //
        // Now, get it back out while keeping it focused.
        //
        // Importantly, we expel it *left*, which results in adding a new column with the exact
        // same active_column_idx.
        Op::FocusWindow(2),
        Op::ConsumeOrExpelWindowLeft { id: None },
    ];

    check_ops(ops);
}

#[test]
fn workspace_render_geo_at_fractional_scale() {
    let ops = [
        Op::AddScaledOutput {
            id: 1,
            scale: 1.1,
            layout_config: None,
        },
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::FocusWorkspaceDown,
        Op::CompleteAnimations,
    ];

    let layout = check_ops(ops);

    let mon_out = layout.monitors[0].output_id();
    let view = layout.active_view(&mon_out).clone();
    let pool = layout.workspace_pool();
    let monitors = &layout.monitors;

    let mon = &monitors[0];
    let ctx = LayoutCtx::new(pool, &view);
    let mut iter = mon.workspaces_with_render_geo(ctx);
    let (_ws, geo) = iter.next().unwrap();
    assert!(
        iter.next().is_none(),
        "animations are completed, only one workspace should be visible"
    );
    assert_eq!(
        geo.loc.y, 0.,
        "active workspace must be at y = 0 exactly, \
         otherwise a pointer against the screen edge at y = 0 won't hit it"
    );
}

fn parent_id_causes_loop(layout: &Layout<TestWindow>, id: usize, mut parent_id: usize) -> bool {
    if parent_id == id {
        return true;
    }

    'outer: loop {
        for (_, win) in layout.windows() {
            if win.0.id == parent_id {
                match win.0.parent_id.get() {
                    Some(new_parent_id) => {
                        if new_parent_id == id {
                            // Found a loop.
                            return true;
                        }

                        parent_id = new_parent_id;
                        continue 'outer;
                    }
                    // Reached window with no parent.
                    None => return false,
                }
            }
        }

        // Parent is not in the layout.
        return false;
    }
}

fn arbitrary_spacing() -> impl Strategy<Value = f64> {
    // Give equal weight to:
    // - 0: the element is disabled
    // - 4: some reasonable value
    // - random value, likely unreasonably big
    prop_oneof![Just(0.), Just(4.), ((1.)..=65535.)]
}

fn arbitrary_spacing_neg() -> impl Strategy<Value = f64> {
    // Give equal weight to:
    // - 0: the element is disabled
    // - 4: some reasonable value
    // - -4: some reasonable negative value
    // - random value, likely unreasonably big
    prop_oneof![Just(0.), Just(4.), Just(-4.), ((1.)..=65535.)]
}

fn arbitrary_struts() -> impl Strategy<Value = Struts> {
    (
        arbitrary_spacing_neg(),
        arbitrary_spacing_neg(),
        arbitrary_spacing_neg(),
        arbitrary_spacing_neg(),
    )
        .prop_map(|(left, right, top, bottom)| Struts {
            left: FloatOrInt(left),
            right: FloatOrInt(right),
            top: FloatOrInt(top),
            bottom: FloatOrInt(bottom),
        })
}

fn arbitrary_center_focused_column() -> impl Strategy<Value = CenterFocusedColumn> {
    prop_oneof![
        Just(CenterFocusedColumn::Never),
        Just(CenterFocusedColumn::OnOverflow),
        Just(CenterFocusedColumn::Always),
    ]
}

fn arbitrary_tab_indicator_position() -> impl Strategy<Value = TabIndicatorPosition> {
    prop_oneof![
        Just(TabIndicatorPosition::Left),
        Just(TabIndicatorPosition::Right),
        Just(TabIndicatorPosition::Top),
        Just(TabIndicatorPosition::Bottom),
    ]
}

prop_compose! {
    fn arbitrary_focus_ring()(
        off in any::<bool>(),
        width in prop::option::of(arbitrary_spacing().prop_map(FloatOrInt)),
    ) -> jiji_config::BorderRule {
        jiji_config::BorderRule {
            off,
            on: !off,
            width,
            ..Default::default()
        }
    }
}

prop_compose! {
    fn arbitrary_border()(
        off in any::<bool>(),
        width in prop::option::of(arbitrary_spacing().prop_map(FloatOrInt)),
    ) -> jiji_config::BorderRule {
        jiji_config::BorderRule {
            off,
            on: !off,
            width,
            ..Default::default()
        }
    }
}

prop_compose! {
    fn arbitrary_shadow()(
        off in any::<bool>(),
        softness in prop::option::of(arbitrary_spacing().prop_map(FloatOrInt)),
    ) -> jiji_config::ShadowRule {
        jiji_config::ShadowRule {
            off,
            on: !off,
            softness,
            ..Default::default()
        }
    }
}

prop_compose! {
    fn arbitrary_tab_indicator()(
        off in any::<bool>(),
        hide_when_single_tab in prop::option::of(any::<bool>().prop_map(Flag)),
        place_within_column in prop::option::of(any::<bool>().prop_map(Flag)),
        width in prop::option::of(arbitrary_spacing().prop_map(FloatOrInt)),
        gap in prop::option::of(arbitrary_spacing_neg().prop_map(FloatOrInt)),
        length in prop::option::of((0f64..2f64)
            .prop_map(|x| TabIndicatorLength { total_proportion: Some(x) })),
        position in prop::option::of(arbitrary_tab_indicator_position()),
    ) -> jiji_config::TabIndicatorPart {
        jiji_config::TabIndicatorPart {
            off,
            on: !off,
            hide_when_single_tab,
            place_within_column,
            width,
            gap,
            length,
            position,
            ..Default::default()
        }
    }
}

prop_compose! {
    fn arbitrary_layout_part()(
        gaps in prop::option::of(arbitrary_spacing().prop_map(FloatOrInt)),
        struts in prop::option::of(arbitrary_struts()),
        focus_ring in prop::option::of(arbitrary_focus_ring()),
        border in prop::option::of(arbitrary_border()),
        shadow in prop::option::of(arbitrary_shadow()),
        tab_indicator in prop::option::of(arbitrary_tab_indicator()),
        center_focused_column in prop::option::of(arbitrary_center_focused_column()),
        always_center_single_column in prop::option::of(any::<bool>().prop_map(Flag)),
        empty_workspace_above_first in prop::option::of(any::<bool>().prop_map(Flag)),
    ) -> jiji_config::LayoutPart {
        jiji_config::LayoutPart {
            gaps,
            struts,
            center_focused_column,
            always_center_single_column,
            empty_workspace_above_first,
            focus_ring,
            border,
            shadow,
            tab_indicator,
            ..Default::default()
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: if std::env::var_os("RUN_SLOW_TESTS").is_none() {
            eprintln!("ignoring slow test");
            0
        } else {
            ProptestConfig::default().cases
        },
        ..ProptestConfig::default()
    })]

    #[test]
    fn random_operations_dont_panic(
        ops: Vec<Op>,
        layout_config in arbitrary_layout_part(),
    ) {
        // eprintln!("{ops:?}");
        let options = Options {
            layout: jiji_config::Layout::from_part(&layout_config),
            ..Default::default()
        };

        check_ops_with_options(options, ops);
    }
}

#[test]
fn workspace_is_sticky_defaults_false() {
    let ws = Workspace::<TestWindow>::new_no_outputs(
        HashSet::from([ActivityId::specific(1)]),
        Clock::with_time(Duration::ZERO),
        Default::default(),
    );
    assert!(
        !ws.is_sticky(),
        "is_sticky must default to false on a freshly-constructed workspace",
    );
}

fn make_test_output(name: &str) -> Output {
    let output = Output::new(
        name.to_owned(),
        PhysicalProperties {
            size: Size::from((1280, 720)),
            subpixel: Subpixel::Unknown,
            make: String::new(),
            model: String::new(),
            serial_number: String::new(),
        },
    );
    output.change_current_state(
        Some(Mode {
            size: Size::from((1280, 720)),
            refresh: 60000,
        }),
        None,
        None,
        None,
    );
    output.user_data().insert_if_missing(|| OutputName {
        connector: name.to_owned(),
        make: None,
        model: None,
        serial: None,
    });
    output
}

#[test]
fn workspace_activities_initializes_from_ctor_param() {
    // Pin the ctor contract: `activities` seeds the workspace's activity
    // membership verbatim. Both the with-output and no-outputs flavors must honor it.
    let output = make_test_output("output1");
    let seed: HashSet<ActivityId> =
        HashSet::from([ActivityId::specific(7), ActivityId::specific(42)]);

    let via_new_with_config = Workspace::<TestWindow>::new_with_config(
        &output,
        None,
        seed.clone(),
        Clock::with_time(Duration::ZERO),
        Default::default(),
    );
    assert_eq!(
        via_new_with_config.activities(),
        &seed,
        "activities must equal the ctor-seed on Workspace::new_with_config",
    );
    assert!(
        !via_new_with_config.is_sticky(),
        "is_sticky must default to false on Workspace::new_with_config",
    );

    let via_no_outputs = Workspace::<TestWindow>::new_with_config_no_outputs(
        None,
        seed.clone(),
        Clock::with_time(Duration::ZERO),
        Default::default(),
    );
    assert_eq!(
        via_no_outputs.activities(),
        &seed,
        "activities must equal the ctor-seed on Workspace::new_with_config_no_outputs",
    );
    assert!(
        !via_no_outputs.is_sticky(),
        "is_sticky must default to false on Workspace::new_with_config_no_outputs",
    );
}

#[test]
#[should_panic(expected = "activities must be non-empty")]
fn workspace_new_panics_on_empty_activities() {
    // Ctors reject an empty activity set.
    let _ = Workspace::<TestWindow>::new_no_outputs(
        HashSet::new(),
        Clock::with_time(Duration::ZERO),
        Default::default(),
    );
}

#[test]
#[should_panic(expected = "activities must be non-empty")]
fn workspace_new_with_config_panics_on_empty_activities() {
    // The caller-facing `new_with_config` ctor also rejects an empty
    // activity set. Distinct from the `new_no_outputs` path so a deletion of
    // the assert in either caller-facing ctor is caught independently.
    let output = make_test_output("output1");
    let _ = Workspace::<TestWindow>::new_with_config(
        &output,
        None,
        HashSet::new(),
        Clock::with_time(Duration::ZERO),
        Default::default(),
    );
}

#[test]
fn layout_new_with_workspaces_stamps_active_activity() {
    // Pin the empty-activities fallback in `resolve_workspace_activities_for`
    // via the `with_options_and_workspaces` startup loop: when no `activity`
    // entries are declared on a workspace config and no activities are in the
    // config, the resolver falls back to the currently-active (seed) activity
    // id.
    let config = Config {
        workspaces: vec![
            WorkspaceConfig {
                name: WorkspaceName("main".to_owned()),
                open_on_output: None,
                layout: None,
                activities: Vec::new(),
                sticky: None,
            },
            WorkspaceConfig {
                name: WorkspaceName("side".to_owned()),
                open_on_output: None,
                layout: None,
                activities: Vec::new(),
                sticky: None,
            },
        ],
        ..Config::default()
    };

    let layout = Layout::<TestWindow>::new(Clock::with_time(Duration::ZERO), &config);
    let seed_id = layout.active_activity_id();

    for ws in layout.workspaces.values() {
        assert_eq!(
            ws.activities(),
            &HashSet::from([seed_id]),
            "workspace {:?} must be stamped with exactly the seed activity",
            ws.id(),
        );
    }

    layout.verify_invariants();
}

#[test]
fn layout_seed_stamps_workspace_with_declared_activities() {
    // A workspace that names one or more declared activities must be stamped
    // with exactly those ids; a workspace that names none must fall back to
    // the currently-active (first-declared) activity. `is_sticky` stays false
    // on both because neither config block sets it.
    let config = Config {
        activities: vec![
            jiji_config::ActivityDecl {
                name: jiji_config::ActivityName("Work".to_owned()),
            },
            jiji_config::ActivityDecl {
                name: jiji_config::ActivityName("Personal".to_owned()),
            },
        ],
        workspaces: vec![
            WorkspaceConfig {
                name: WorkspaceName("chat".to_owned()),
                open_on_output: None,
                layout: None,
                activities: vec!["Work".to_owned(), "Personal".to_owned()],
                sticky: None,
            },
            WorkspaceConfig {
                name: WorkspaceName("music".to_owned()),
                open_on_output: None,
                layout: None,
                activities: Vec::new(),
                sticky: None,
            },
        ],
        ..Config::default()
    };

    let layout = Layout::<TestWindow>::new(Clock::with_time(Duration::ZERO), &config);
    let work_id = layout
        .activities
        .iter()
        .find(|a| a.name() == "Work")
        .expect("Work activity must be seeded from config")
        .id();
    let personal_id = layout
        .activities
        .iter()
        .find(|a| a.name() == "Personal")
        .expect("Personal activity must be seeded from config")
        .id();

    let chat = layout
        .workspaces
        .values()
        .find(|w| w.name() == Some(&"chat".to_owned()))
        .expect("chat workspace must be present");
    assert_eq!(chat.activities(), &HashSet::from([work_id, personal_id]));
    assert!(!chat.is_sticky());

    let music = layout
        .workspaces
        .values()
        .find(|w| w.name() == Some(&"music".to_owned()))
        .expect("music workspace must be present");
    assert_eq!(
        music.activities(),
        &HashSet::from([work_id]),
        "empty config.activities must fall back to the currently-active (first-declared) id",
    );
    assert!(!music.is_sticky());

    layout.verify_invariants();
}

#[test]
fn layout_seed_sticky_stamps_all_activity_ids() {
    // Sticky beats an explicit `activity "..."` list: the workspace
    // must be auto-tagged with every activity id in the pool.
    let config = Config {
        activities: vec![
            jiji_config::ActivityDecl {
                name: jiji_config::ActivityName("Work".to_owned()),
            },
            jiji_config::ActivityDecl {
                name: jiji_config::ActivityName("Personal".to_owned()),
            },
        ],
        workspaces: vec![WorkspaceConfig {
            name: WorkspaceName("utils".to_owned()),
            open_on_output: None,
            layout: None,
            activities: vec!["Work".to_owned()],
            sticky: Some(true),
        }],
        ..Config::default()
    };

    let layout = Layout::<TestWindow>::new(Clock::with_time(Duration::ZERO), &config);
    let work_id = layout
        .activities
        .iter()
        .find(|a| a.name() == "Work")
        .expect("Work activity must be seeded from config")
        .id();
    let personal_id = layout
        .activities
        .iter()
        .find(|a| a.name() == "Personal")
        .expect("Personal activity must be seeded from config")
        .id();

    let utils = layout
        .workspaces
        .values()
        .find(|w| w.name() == Some(&"utils".to_owned()))
        .expect("utils workspace must be present");
    assert_eq!(utils.activities(), &HashSet::from([work_id, personal_id]));
    assert!(utils.is_sticky());

    layout.verify_invariants();
}

#[test]
fn layout_seed_unknown_activity_name_falls_back_to_default() {
    // Pin the all-unknown fallback arm in `resolve_workspace_activities_for`:
    // when every name in a workspace's `activity` list is unrecognised, the
    // resolver must fall back to the currently-active (first-declared) activity
    // id so the non-empty-activities invariant on `Workspace::new*` is
    // preserved.
    let config = Config {
        activities: vec![
            jiji_config::ActivityDecl {
                name: jiji_config::ActivityName("Work".to_owned()),
            },
            jiji_config::ActivityDecl {
                name: jiji_config::ActivityName("Personal".to_owned()),
            },
        ],
        workspaces: vec![WorkspaceConfig {
            name: WorkspaceName("chat".to_owned()),
            open_on_output: None,
            layout: None,
            activities: vec!["Bogus1".to_owned(), "Bogus2".to_owned()],
            sticky: None,
        }],
        ..Config::default()
    };

    let layout = Layout::<TestWindow>::new(Clock::with_time(Duration::ZERO), &config);
    let work_id = layout
        .activities
        .iter()
        .find(|a| a.name() == "Work")
        .expect("Work activity must be seeded from config")
        .id();

    let chat = layout
        .workspaces
        .values()
        .find(|w| w.name() == Some(&"chat".to_owned()))
        .expect("chat workspace must be present");

    assert_eq!(
        chat.activities(),
        &HashSet::from([work_id]),
        "all-unknown activity names must fall back to the currently-active id",
    );
    assert!(!chat.is_sticky());

    layout.verify_invariants();
}

#[test]
fn build_activities_ipc_mirrors_seed_state() {
    // A fresh layout has exactly one seed activity. Pin the IPC-projection
    // helpers against accidental drift in field wiring (active-id comparison,
    // `is_urgent` aggregation, name / config-declared flags).
    let mut layout = Layout::<TestWindow>::default();
    let seed_id = layout.active_activity_id();

    let ipc = crate::ipc::server::build_activities_ipc(&layout);
    assert_eq!(ipc.len(), 1, "default seed activity must be the only one");

    let only = &ipc[0];
    // `Activities::new(Activity::new_runtime(DEFAULT_ACTIVITY_NAME))` via
    // `new_runtime` → `is_config_declared == false`.
    assert_eq!(only.name, "Default");
    assert!(only.is_active);
    assert!(
        !only.is_urgent,
        "no urgent windows → aggregate is_urgent is false",
    );
    assert!(!only.is_config_declared);

    let focused = crate::ipc::server::build_focused_activity_ipc(&layout);
    assert_eq!(&focused, only);

    // --- Two-activity extension ---
    // A single-activity pool trivially satisfies `is_active == true` and
    // `focused == iter().next()`. Insert a second activity and switch to it so
    // those trivial degeneracies can no longer mask bugs in `is_active` wiring
    // or `build_focused_activity_ipc` returning the wrong entry.
    let beta_activity = super::activity::Activity::new_runtime("beta".to_owned());
    let beta_id = beta_activity.id();
    test_insert_activity(&mut layout, beta_activity);
    layout.switch_activity(beta_id);

    let ipc = crate::ipc::server::build_activities_ipc(&layout);
    assert_eq!(ipc.len(), 2);

    let active_entries: Vec<_> = ipc.iter().filter(|a| a.is_active).collect();
    assert_eq!(
        active_entries.len(),
        1,
        "exactly one activity must be active"
    );
    assert_eq!(active_entries[0].id, beta_id.get(), "beta must be active");
    assert_eq!(active_entries[0].name, "beta");

    let seed_entry = ipc.iter().find(|a| a.id == seed_id.get()).unwrap();
    assert!(!seed_entry.is_active, "seed must no longer be active");

    let focused = crate::ipc::server::build_focused_activity_ipc(&layout);
    assert_eq!(focused.id, beta_id.get(), "focused must be beta");

    layout.verify_invariants();
}

#[test]
fn build_activity_views_ipc_projects_view_map_across_activities_and_outputs() {
    // Pin `build_activity_views_ipc` against the source contract: one entry per
    // extant `(activity, output_id)` view pair, with outer order = activity
    // declaration order, inner order = `OutputId.as_str()` sort,
    // `output_name` resolved through `monitor_for_output_id`, and the
    // per-view `workspace_ids` / `active_idx` projection.
    //
    // Two activities × two connected monitors → asymmetric extant-view count
    // (alpha: 2 entries; beta: 1 — beta's out1 entry was migrated to out2 by the
    // partial-disconnect dormant walk). Outputs are added in reverse lex order
    // (output2 before output1) so the inner-sort assertion proves the helper
    // actively sorts rather than passing through insertion order.
    let ops = [Op::AddOutput(2), Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let beta = layout.create_activity("Beta".to_owned()).expect("create");

    // Switch to beta so the projection covers both an active and a dormant view.
    layout.switch_activity(beta);

    // Capture OutputIds before removal so they survive as lookup keys.
    // After `AddOutput(2), AddOutput(1)`: monitors[0]=output2, monitors[1]=output1.
    let out_connected = layout.monitors[0].output_id(); // output2, stays connected
    let out_disconnected = layout.monitors[1].output_id(); // output1, about to be removed
    let out_connected_name = layout.monitors[0].output_name().clone();

    // Remove output1: beta (active) loses its output1 view via the active-view
    // eviction; alpha (dormant) has its output1 view drained by the
    // partial-disconnect walk. Both activities end up with views keyed only by the
    // remaining monitor's output_id.
    let remove_output = layout.monitors[1].output().clone();
    layout.remove_output(&remove_output);

    let ipc = crate::ipc::server::build_activity_views_ipc(&layout);

    // `expected_pair_count` guards against bugs in the views() iterator; the literal
    // `2` guards against the Cartesian "fix" (2 activities × 2 original outputs would
    // give 4 — the post-drain 2 proves the partial-disconnect walk ran).
    let expected_pair_count: usize = layout.activities().iter().map(|a| a.views().len()).sum();
    assert_eq!(
        ipc.len(),
        expected_pair_count,
        "entry count must equal sum of per-activity view counts (extant pairs, not Cartesian)",
    );
    assert_eq!(
        ipc.len(),
        2,
        "alpha has 1 entry (output2), beta has 1 entry (output2) — partial-disconnect walk \
         drained both activities' views for output1",
    );

    // Outer ordering: activity declaration order (alpha first, beta second).
    let activity_ids_in_order: Vec<u64> = ipc.iter().map(|v| v.activity_id).collect();
    assert_eq!(
        activity_ids_in_order,
        vec![alpha.get(), beta.get()],
        "outer ordering must be activity declaration order",
    );

    // Inner ordering: `OutputId.as_str()` sort within each activity block. With a
    // single entry per activity post-drain, the inner-sort assertion is trivially
    // satisfied, but we still pin the per-activity entry shape and that no entry
    // references the disconnected output.
    let alpha_block: Vec<&str> = ipc
        .iter()
        .filter(|v| v.activity_id == alpha.get())
        .map(|v| v.output_id.as_str())
        .collect();
    let beta_block: Vec<&str> = ipc
        .iter()
        .filter(|v| v.activity_id == beta.get())
        .map(|v| v.output_id.as_str())
        .collect();
    assert_eq!(
        alpha_block.len(),
        1,
        "alpha has only the connected-output entry post-drain",
    );
    assert_eq!(
        beta_block.len(),
        1,
        "beta has only the connected-output entry post-drain",
    );
    assert!(
        !ipc.iter().any(|v| v.output_id == out_disconnected.as_str()),
        "no entry must reference the disconnected output",
    );

    // Every entry's `workspace_ids` / `active_idx` / `output_name` matches the
    // underlying view + monitor connector lookup. Every entry must have
    // `output_name: Some(_)` — `None` is reserved for forward compatibility.
    for entry in &ipc {
        let activity_id = super::activity::ActivityId::specific(entry.activity_id);
        let activity = layout
            .activities()
            .get(activity_id)
            .expect("entry's activity_id must be live");
        let key = if entry.output_id == out_connected.as_str() {
            &out_connected
        } else {
            panic!("unexpected output_id in IPC entry: {}", entry.output_id);
        };
        let view = activity
            .views()
            .get(key)
            .expect("IPC entry must back to an extant view");

        let expected_ids: Vec<u64> = view.ids().iter().map(|id| id.get()).collect();
        assert_eq!(
            entry.workspace_ids, expected_ids,
            "workspace_ids must equal view.ids()",
        );
        assert_eq!(
            entry.active_idx,
            view.active_position(),
            "active_idx must equal view.active_position()",
        );
        assert_eq!(
            entry.output_name,
            Some(out_connected_name.clone()),
            "output_name must be Some(connector) for connected outputs",
        );
    }

    layout.verify_invariants();
}

#[test]
fn backwards_compat_default_config_seeds_single_implicit_default_activity() {
    // Backwards-compat pin for: when no `activity` blocks appear in
    // the config, exactly one implicit runtime "Default" activity is seeded,
    // it is active, it is not config-declared, and no previous activity exists.
    //
    // Uses `Layout::new` (not `Layout::default`) to exercise the
    // `with_options_and_workspaces` path that calls
    // `Activities::from_config_or_default`.
    let layout = Layout::<TestWindow>::new(Clock::with_time(Duration::ZERO), &Config::default());

    assert_eq!(
        layout.activities.len(),
        1,
        "empty config must seed exactly one implicit Default activity",
    );
    assert_eq!(
        layout.activities.active().name(),
        super::DEFAULT_ACTIVITY_NAME,
        "seeded activity must carry the DEFAULT_ACTIVITY_NAME string",
    );
    assert!(
        !layout.activities.active().is_config_declared(),
        "seeded default must be a runtime activity, not a config-declared one",
    );
    assert_eq!(
        layout.activities.previous_id(),
        None,
        "no previous activity on a fresh single-activity layout",
    );

    layout.verify_invariants();
}

#[test]
fn backwards_compat_workspaces_stamped_with_default_activity_on_empty_config() {
    // Backwards-compat pin for: when there are workspace config blocks
    // but no `activity` blocks, every workspace must be stamped with the
    // implicit Default activity id, must be visible (idx >= 1) on a connected
    // monitor, and must not be sticky.
    let config = Config {
        workspaces: vec![
            WorkspaceConfig {
                name: WorkspaceName("main".to_owned()),
                open_on_output: None,
                layout: None,
                activities: Vec::new(),
                sticky: None,
            },
            WorkspaceConfig {
                name: WorkspaceName("side".to_owned()),
                open_on_output: None,
                layout: None,
                activities: Vec::new(),
                sticky: None,
            },
        ],
        ..Config::default()
    };

    let mut layout = Layout::<TestWindow>::new(Clock::with_time(Duration::ZERO), &config);
    let default_id = layout.active_activity_id();
    check_ops_on_layout(&mut layout, [Op::AddOutput(1)]);

    // Pool-level: every workspace carries exactly {default_id} and is not sticky.
    for ws in layout.workspaces.values() {
        assert_eq!(
            ws.activities(),
            &HashSet::from([default_id]),
            "workspace {:?} must be stamped with exactly the default activity id",
            ws.id(),
        );
        assert!(
            !ws.is_sticky(),
            "workspace {:?} must not be sticky when config has no sticky block",
            ws.id(),
        );
    }

    // IPC wire: every workspace in the snapshot must have is_in_active_activity
    // = true and idx matching its view position + 1.
    let mon_out = layout.monitors[0].output_id();
    let view = layout.active_view(&mon_out);
    let view_ids: Vec<WorkspaceId> = view.ids().to_vec();

    let snapshot = crate::ipc::server::build_workspace_snapshot(&layout, None, test_window_id_of);

    assert_eq!(
        snapshot.len(),
        view_ids.len(),
        "snapshot must contain exactly one entry per view position (no duplicate emission)",
    );

    for (pos, id) in view_ids.iter().enumerate() {
        let expected_idx = u8::try_from(pos + 1).unwrap_or(u8::MAX);
        let ws = snapshot
            .iter()
            .find(|ws| ws.id == id.get())
            .unwrap_or_else(|| panic!("workspace {:?} must appear in the snapshot", id));
        assert_eq!(
            ws.idx, expected_idx,
            "workspace at view position {pos} must have idx = {expected_idx}",
        );
        assert!(
            ws.is_in_active_activity,
            "workspace at view position {pos} must have is_in_active_activity = true",
        );
        assert_eq!(
            ws.activities,
            vec![default_id.get()],
            "workspace at view position {pos} must list exactly the default activity id",
        );
        assert!(
            !ws.is_sticky,
            "workspace at view position {pos} must have is_sticky = false",
        );
    }

    layout.verify_invariants();
}

#[test]
fn backwards_compat_switch_activity_self_and_previous_are_noops_with_only_default() {
    // Backwards-compat pin for: with only the seeded Default activity,
    // both switch_activity(default_id) and switch_activity_previous() are
    // pure no-ops — active and previous remain unchanged.
    //
    // This consolidates the single-activity no-op contract in one visible
    // place for a future reader of the backwards-compat story, even though
    // the individual paths are also covered by
    // `layout_switch_activity_no_op_on_same_target` and
    // `layout_switch_activity_previous_no_op_when_no_previous`.
    let mut layout =
        Layout::<TestWindow>::new(Clock::with_time(Duration::ZERO), &Config::default());
    let default_id = layout.active_activity_id();
    assert_eq!(layout.activities.len(), 1);

    // switch_activity to the already-active id: no-op.
    layout.switch_activity(default_id);
    assert_eq!(layout.active_activity_id(), default_id);
    assert_eq!(
        layout.activities.previous_id(),
        None,
        "switch_activity no-op must not record a previous activity",
    );
    layout.verify_invariants();

    // switch_activity_previous with no previous: no-op.
    layout.switch_activity_previous();
    assert_eq!(layout.active_activity_id(), default_id);
    assert_eq!(
        layout.activities.previous_id(),
        None,
        "switch_activity_previous with no previous must not change state",
    );
    layout.verify_invariants();
}

#[test]
fn active_activity_views_populated_on_add_output() {
    let ops = [Op::AddOutput(1)];
    let layout = check_ops(ops);

    assert_eq!(layout.monitors.len(), 1);
    assert_eq!(layout.activities.active().views().len(), 1);
    let mon_out = layout.monitors[0].output_id();
    assert!(!layout.active_view(&mon_out).ids().is_empty());
}

#[test]
fn active_activity_views_evicted_on_remove_output() {
    let ops = [Op::AddOutput(1), Op::AddOutput(2), Op::RemoveOutput(1)];
    let layout = check_ops(ops);

    assert_eq!(layout.monitors.len(), 1);
    assert_eq!(layout.activities.active().views().len(), 1);
    let remaining_out = layout.monitors[0].output_id();
    assert!(layout
        .activities
        .active()
        .views()
        .contains_key(&remaining_out));
}

#[test]
fn active_activity_views_empty_when_no_monitors() {
    let ops = [Op::AddOutput(1), Op::RemoveOutput(1)];
    let layout = check_ops(ops);

    assert!(layout.monitors.is_empty());
    assert!(layout.activities.active().views().is_empty());
    layout.verify_invariants();
}

#[test]
fn layout_switch_activity_no_op_on_same_target() {
    // No-monitor Layout to isolate the pure cursor-flip path from any
    // view-population or refresh effects.
    let layout = Layout::<TestWindow>::default();
    let seed_id = layout.active_activity_id();
    assert_eq!(layout.activities.previous_id(), None);

    let mut layout = layout;
    layout.switch_activity(seed_id);

    assert_eq!(layout.active_activity_id(), seed_id);
    // No-op must leave previous untouched.
    assert_eq!(layout.activities.previous_id(), None);
    layout.verify_invariants();
}

#[test]
fn layout_switch_activity_unknown_target_leaves_state_unchanged() {
    let mut layout = Layout::<TestWindow>::default();
    let seed_id = layout.active_activity_id();

    // `u64::MAX` cannot collide with a runtime-minted id (counter starts at 0).
    layout.switch_activity(ActivityId::specific(u64::MAX));

    // State must be unchanged: still the seed, still no previous.
    assert_eq!(layout.active_activity_id(), seed_id);
    assert_eq!(layout.activities.previous_id(), None);
    layout.verify_invariants();
}

#[test]
fn layout_switch_activity_no_op_preserves_verify_invariants() {
    // With a connected monitor, `verify_invariants` walks the
    // `active_views.len() == self.monitors.len()` assertion. Pin that the
    // no-op `switch_activity` path does not trip it.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let seed_id = layout.active_activity_id();

    layout.switch_activity(seed_id);

    assert_eq!(layout.active_activity_id(), seed_id);
    layout.verify_invariants();
}

#[test]
fn layout_switch_activity_previous_no_op_when_no_previous() {
    // Fresh pool: no previous activity has ever been recorded, so the call
    // must be a pure no-op — neither the active cursor nor `previous` moves.
    let mut layout = Layout::<TestWindow>::default();
    let seed_id = layout.active_activity_id();
    assert_eq!(layout.activities.previous_id(), None);

    layout.switch_activity_previous();

    assert_eq!(layout.active_activity_id(), seed_id);
    assert_eq!(layout.activities.previous_id(), None);
    layout.verify_invariants();
}

#[test]
fn layout_switch_activity_previous_toggles_active() {
    // Two activities: seed (alpha) and beta. Switch to beta, which records
    // previous = seed_id. Then switch_activity_previous must flip back to
    // seed_id and record previous = beta_id.
    let mut layout = Layout::<TestWindow>::default();
    let seed_id = layout.active_activity_id();

    let beta_activity = super::activity::Activity::new_runtime("beta".to_owned());
    let beta_id = beta_activity.id();
    test_insert_activity(&mut layout, beta_activity);

    // Switch to beta — this populates previous = seed_id.
    layout.switch_activity(beta_id);
    assert_eq!(layout.active_activity_id(), beta_id);
    assert_eq!(layout.activities.previous_id(), Some(seed_id));

    // Toggle back — must land on seed_id with previous = Some(beta_id).
    layout.switch_activity_previous();

    assert_eq!(layout.active_activity_id(), seed_id);
    assert_eq!(layout.activities.previous_id(), Some(beta_id));
    layout.verify_invariants();
}

#[test]
fn switch_activity_bootstraps_view_on_first_visit() {
    // First switch to a brand-new activity on a monitor that's been connected only for
    // the seed activity: ensure_all_activity_views must allocate a fresh empty workspace for
    // beta (no pre-tagged candidates), wrap it in a view, and install it into beta's
    // per-output map.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let seed_id = layout.active_activity_id();
    let out_id = layout.monitors[0].output_id();
    let pool_size_before = layout.workspaces.len();

    let beta = super::activity::Activity::new_runtime("beta".to_owned());
    let beta_id = beta.id();
    test_insert_activity(&mut layout, beta);

    layout.switch_activity(beta_id);

    assert_eq!(layout.active_activity_id(), beta_id);
    let beta_view = &layout.activities.active().views()[&out_id];
    assert_eq!(beta_view.len(), 1, "fresh bootstrap view holds one id");
    let fresh_id = beta_view.ids()[0];
    let fresh_ws = layout
        .workspaces
        .get(&fresh_id)
        .expect("bootstrap view id must be in the pool");
    assert!(
        fresh_ws.activities().contains(&beta_id) && !fresh_ws.activities().contains(&seed_id),
        "fresh workspace is tagged with beta only",
    );
    assert_eq!(
        layout.workspaces.len(),
        pool_size_before + 1,
        "exactly one new pool entry",
    );
    layout.verify_invariants();
}

#[test]
fn switch_activity_preserves_seed_dormant_view() {
    // After the first switch to beta, seed's pre-existing view stays intact as a dormant
    // snapshot. The widened pool-keys union must see workspace ids from both seed's
    // dormant view and beta's newly-populated view.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let seed_id = layout.active_activity_id();
    let out_id = layout.monitors[0].output_id();

    // Snapshot seed's view ids before switching away.
    let seed_view_ids: Vec<_> = layout.active_view(&out_id).ids().to_vec();

    let beta = super::activity::Activity::new_runtime("beta".to_owned());
    let beta_id = beta.id();
    test_insert_activity(&mut layout, beta);

    layout.switch_activity(beta_id);

    let seed_dormant = layout
        .activities
        .get(seed_id)
        .expect("seed activity still present")
        .views()
        .get(&out_id)
        .expect("seed's view must persist across the switch");
    assert_eq!(seed_dormant.ids(), seed_view_ids.as_slice());

    // Both seed's dormant view and beta's active view contribute keys to the widened
    // pool-keys union — verify_invariants would panic otherwise.
    layout.verify_invariants();
}

#[test]
fn switch_activity_reuses_existing_view_entry() {
    // Going back to an activity whose view was already populated must not allocate
    // another workspace: contains_key hits in ensure_all_activity_views and the loop skips.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let seed_id = layout.active_activity_id();

    let beta = super::activity::Activity::new_runtime("beta".to_owned());
    let beta_id = beta.id();
    test_insert_activity(&mut layout, beta);

    layout.switch_activity(beta_id);
    layout.verify_invariants();
    let pool_size_after_first_beta = layout.workspaces.len();

    layout.switch_activity(seed_id);
    layout.verify_invariants();
    assert_eq!(
        layout.workspaces.len(),
        pool_size_after_first_beta,
        "returning to seed must not allocate",
    );

    layout.switch_activity(beta_id);
    layout.verify_invariants();
    assert_eq!(
        layout.workspaces.len(),
        pool_size_after_first_beta,
        "second visit to beta reuses the prior view with no fresh allocation",
    );
}

#[test]
fn switch_activity_with_tagged_workspace_builds_view_from_pool() {
    // Pre-tag an existing seed-owned workspace with beta BEFORE inserting beta. The
    // per-activity bookend invariant fires the materializer at insert time; with the
    // pre-tag already in place, the materializer lifts the tagged workspace into
    // beta's fresh view instead of allocating a stand-alone fresh empty.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let out_id = layout.monitors[0].output_id();

    // Mint beta out-of-pool so we have a known `beta_id` to pre-tag with.
    let beta = super::activity::Activity::new_runtime("beta".to_owned());
    let beta_id = beta.id();

    // Pick a workspace currently on this monitor (the bottom bookend is always present)
    // and add beta to its activity set. `activities` is pub(super) on Workspace, so this
    // direct mutation is available from the test module in the same layout parent.
    let pick = layout.active_view(&out_id).ids()[0];
    layout
        .workspaces
        .get_mut(&pick)
        .expect("picked id must be a pool key")
        .activities
        .insert(beta_id);

    let pool_size_before = layout.workspaces.len();

    // Insert beta and materialize — the materializer's lift branch picks up `pick`.
    test_insert_activity(&mut layout, beta);

    let beta_view = layout
        .activities
        .get(beta_id)
        .expect("beta is live")
        .views()
        .get(&out_id)
        .expect("beta has a view for out_id post-materialize");
    assert!(
        beta_view.ids().contains(&pick),
        "beta's view must include the pre-tagged workspace",
    );
    assert_eq!(
        beta_view.len(),
        2,
        "lift branch shape: [pick, fresh bottom_empty]",
    );
    assert_eq!(
        beta_view.ids()[0],
        pick,
        "lift branch puts pick at position 0",
    );
    assert_eq!(
        beta_view.active_position(),
        0,
        "active stays on pick, not bottom_empty",
    );
    assert_eq!(
        layout.workspaces.len(),
        pool_size_before + 1,
        "lift branch appends one fresh trailing empty so monitor invariants hold",
    );

    // Switching to beta is now a pure activate-cursor flip; the view is already there.
    layout.switch_activity(beta_id);
    assert_eq!(layout.active_activity_id(), beta_id);
    layout.verify_invariants();
}

#[test]
fn switch_activity_creates_views_for_all_connected_monitors() {
    // ensure_all_activity_views loops over every connected monitor. With two outputs,
    // switching to a brand-new activity must populate a view entry for *each*
    // output — not just the first one the loop visits.
    let ops = [Op::AddOutput(1), Op::AddOutput(2)];
    let mut layout = check_ops(ops);
    let out1 = layout.monitors[0].output_id();
    let out2 = layout.monitors[1].output_id();
    assert_ne!(out1, out2);

    let beta = super::activity::Activity::new_runtime("beta".to_owned());
    let beta_id = beta.id();
    test_insert_activity(&mut layout, beta);

    layout.switch_activity(beta_id);

    assert_eq!(layout.active_activity_id(), beta_id);
    let beta_views = layout.activities.active().views();
    assert_eq!(
        beta_views.len(),
        2,
        "ensure_all_activity_views must cover both connected outputs",
    );
    assert!(
        beta_views.contains_key(&out1),
        "beta must have a view for out1",
    );
    assert!(
        beta_views.contains_key(&out2),
        "beta must have a view for out2",
    );

    layout.verify_invariants();
}

#[test]
fn switch_activity_tagged_workspace_output_affinity() {
    // Two outputs: output1 has no pre-tagged candidate for beta (forces the
    // fresh-allocation branch in `ensure_all_activity_views`), while output2 has one
    // pre-tagged workspace (forces the lift-from-pool branch).
    // Assertions:
    //   - output2's view contains the pre-tagged id (no fresh allocation there)
    //   - output1's view holds one freshly-minted id (not in pool_ids_before)
    //   - pool grows by exactly 2 (out1's fresh empty + out2's lift-branch trailing empty)
    let ops = [Op::AddOutput(1), Op::AddOutput(2)];
    let mut layout = check_ops(ops);
    let out1 = layout.monitors[0].output_id();
    let out2 = layout.monitors[1].output_id();
    assert_ne!(out1, out2);

    // Mint beta out-of-pool so we have a known `beta_id` to pre-tag with.
    let beta = super::activity::Activity::new_runtime("beta".to_owned());
    let beta_id = beta.id();

    // Tag a workspace bound to out2 (only) with beta BEFORE inserting beta — the
    // per-activity bookend materializer fires at insert and consumes the pre-tag.
    let pick = layout
        .workspaces
        .values()
        .find(|ws| ws.output_id() == Some(&out2))
        .map(|ws| ws.id())
        .expect("at least one workspace must be bound to out2");
    layout
        .workspaces
        .get_mut(&pick)
        .expect("pick must be a pool key")
        .activities
        .insert(beta_id);

    let pool_ids_before: HashSet<WorkspaceId> = layout.workspaces.keys().copied().collect();
    let pool_size_before = pool_ids_before.len();

    test_insert_activity(&mut layout, beta);
    layout.switch_activity(beta_id);

    let beta_views = layout.activities.active().views();

    // output2: must contain the pre-tagged id with a fresh trailing empty appended (lift branch).
    let out2_view = beta_views
        .get(&out2)
        .expect("beta must have a view for out2");
    assert!(
        out2_view.ids().contains(&pick),
        "beta's out2 view must include the pre-tagged workspace",
    );
    assert_eq!(
        out2_view.len(),
        2,
        "lift branch shape on out2: [pick, fresh bottom_empty]",
    );
    assert_eq!(
        out2_view.ids()[0],
        pick,
        "lift branch puts pick at position 0 on out2",
    );
    assert_eq!(
        out2_view.active_position(),
        0,
        "active stays on pick, not bottom_empty, on out2",
    );

    // output1: must contain exactly one freshly-minted id — allocation branch.
    let out1_view = beta_views
        .get(&out1)
        .expect("beta must have a view for out1");
    assert_eq!(out1_view.len(), 1, "fresh out1 view holds exactly one id");
    let fresh_id = out1_view.ids()[0];
    assert!(
        !pool_ids_before.contains(&fresh_id),
        "out1's bootstrap id must be freshly allocated",
    );

    // Pool grows by exactly two: out1's fresh empty + out2's lift-branch trailing empty.
    assert_eq!(
        layout.workspaces.len(),
        pool_size_before + 2,
        "fresh empty for out1 + lift-branch trailing empty for out2",
    );

    layout.verify_invariants();
}

#[test]
fn switch_activity_lift_branch_appends_bottom_empty_for_named_tagged_workspace() {
    // The lift branch must allocate a fresh trailing empty even when the only
    // pre-tagged candidate is named — otherwise `Monitor::verify_invariants` trips
    // "monitor must have an empty workspace in the end" (monitor.rs:1724) the moment
    // `Layout::verify_invariants` walks the new view. Pre-fix `ensure_all_activity_views`
    // produced `[named_pick]` (len=1) and panicked on the very first invariant chain.
    let ops = [
        Op::AddOutput(1),
        Op::AddNamedWorkspace {
            ws_name: 1,
            output_name: Some(1),
            layout_config: None,
        },
    ];
    let mut layout = check_ops(ops);
    let out_id = layout.monitors[0].output_id();

    let beta = super::activity::Activity::new_runtime("beta".to_owned());
    let beta_id = beta.id();

    // Pre-tag the named workspace with beta BEFORE the activity is inserted, so the
    // materializer's lift branch picks it up at insert time.
    let pick = layout
        .workspaces
        .values()
        .find(|ws| ws.name() == Some(&"ws1".to_owned()))
        .map(|ws| ws.id())
        .expect("ws1 must exist after AddNamedWorkspace");
    layout
        .workspaces
        .get_mut(&pick)
        .expect("pick must be a pool key")
        .activities
        .insert(beta_id);

    test_insert_activity(&mut layout, beta);
    layout.switch_activity(beta_id);

    let beta_view = &layout.activities.active().views()[&out_id];
    assert_eq!(
        beta_view.len(),
        2,
        "lift branch must append a bottom empty: [named_pick, bottom_empty]",
    );
    assert_eq!(
        beta_view.ids()[0],
        pick,
        "the named pre-tagged workspace stays at position 0",
    );
    assert_eq!(
        beta_view.active_position(),
        0,
        "active index points at the named pick (no top-empty without EWAF)",
    );
    let bottom_id = beta_view.ids()[1];
    let bottom = layout
        .workspaces
        .get(&bottom_id)
        .expect("bottom id must be a pool key");
    assert!(!bottom.has_windows(), "trailing bottom must be empty");
    assert!(bottom.name().is_none(), "trailing bottom must be unnamed");

    layout.verify_invariants();
}

#[test]
fn switch_activity_lift_branch_named_tagged_with_ewaf_adds_top_and_bottom_empty() {
    // When `empty_workspace_above_first` is enabled globally, the lift branch must
    // bookend the pre-tagged candidate with both a fresh top-empty (to satisfy
    // monitor.rs:1739-1751 "first must be empty + unnamed" and the "1 or 3+" rule)
    // and a fresh bottom-empty. The active index must shift to 1 so the named pick
    // remains selected after switching — landing on the top-empty would be a silent
    // focus regression.
    let options = Options {
        layout: jiji_config::Layout {
            empty_workspace_above_first: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let ops = [
        Op::AddOutput(1),
        Op::AddNamedWorkspace {
            ws_name: 1,
            output_name: Some(1),
            layout_config: None,
        },
    ];
    let mut layout = check_ops_with_options(options, ops);
    let out_id = layout.monitors[0].output_id();

    let beta = super::activity::Activity::new_runtime("beta".to_owned());
    let beta_id = beta.id();

    let pick = layout
        .workspaces
        .values()
        .find(|ws| ws.name() == Some(&"ws1".to_owned()))
        .map(|ws| ws.id())
        .expect("ws1 must exist after AddNamedWorkspace");
    layout
        .workspaces
        .get_mut(&pick)
        .expect("pick must be a pool key")
        .activities
        .insert(beta_id);

    test_insert_activity(&mut layout, beta);
    layout.switch_activity(beta_id);

    let beta_view = &layout.activities.active().views()[&out_id];
    assert_eq!(
        beta_view.len(),
        3,
        "EWAF lift shape: [top_empty, named_pick, bottom_empty]",
    );
    assert_eq!(
        beta_view.ids()[1],
        pick,
        "named pick sits at position 1 between the two fresh empties",
    );
    assert_eq!(
        beta_view.active_position(),
        1,
        "active index must shift to 1 so the named pick stays focused under EWAF",
    );
    let top = layout
        .workspaces
        .get(&beta_view.ids()[0])
        .expect("top id must be a pool key");
    assert!(!top.has_windows(), "EWAF top must be empty");
    assert!(top.name().is_none(), "EWAF top must be unnamed");
    let bottom = layout
        .workspaces
        .get(&beta_view.ids()[2])
        .expect("bottom id must be a pool key");
    assert!(!bottom.has_windows(), "trailing bottom must be empty");
    assert!(bottom.name().is_none(), "trailing bottom must be unnamed");

    layout.verify_invariants();
}

#[test]
fn switch_activity_lift_branch_reads_ewaf_from_monitor_not_layout() {
    // Per-monitor `Rc<Options>` discriminator: the global `self.options` keeps EWAF
    // disabled, but the monitor's `layout_config` flips it on. `ensure_all_activity_views`
    // must read `monitors[i].options.layout.empty_workspace_above_first`, not
    // `self.options.layout.empty_workspace_above_first`. If it reads the global, the
    // view comes back as `[named_pick, bottom_empty]` (len=2) and trips
    // monitor.rs:1746-1751 "1 or 3+" the moment `verify_invariants` runs.
    let layout_part = jiji_config::LayoutPart {
        empty_workspace_above_first: Some(Flag(true)),
        ..Default::default()
    };
    let ops = [
        Op::AddScaledOutput {
            id: 1,
            scale: 1.0,
            layout_config: Some(Box::new(layout_part)),
        },
        Op::AddNamedWorkspace {
            ws_name: 1,
            output_name: Some(1),
            layout_config: None,
        },
    ];
    let mut layout = check_ops(ops);
    let out_id = layout.monitors[0].output_id();

    // Sanity: global options must NOT have EWAF enabled. Otherwise this test is
    // vacuous — it would pass whether the implementation reads global or per-monitor.
    assert!(
        !layout.options.layout.empty_workspace_above_first,
        "global EWAF must stay false so the per-monitor capture is what's exercised",
    );
    assert!(
        layout.monitors[0]
            .options
            .layout
            .empty_workspace_above_first,
        "per-monitor EWAF must be true via layout_config merge",
    );

    let beta = super::activity::Activity::new_runtime("beta".to_owned());
    let beta_id = beta.id();

    let pick = layout
        .workspaces
        .values()
        .find(|ws| ws.name() == Some(&"ws1".to_owned()))
        .map(|ws| ws.id())
        .expect("ws1 must exist after AddNamedWorkspace");
    layout
        .workspaces
        .get_mut(&pick)
        .expect("pick must be a pool key")
        .activities
        .insert(beta_id);

    test_insert_activity(&mut layout, beta);
    layout.switch_activity(beta_id);

    let beta_view = &layout.activities.active().views()[&out_id];
    assert_eq!(
        beta_view.len(),
        3,
        "per-monitor EWAF must yield [top_empty, named_pick, bottom_empty]",
    );
    assert_eq!(
        beta_view.ids()[1],
        pick,
        "named pick sits at position 1 under per-monitor EWAF",
    );
    assert_eq!(
        beta_view.active_position(),
        1,
        "active index reflects per-monitor EWAF top-empty prepend",
    );

    layout.verify_invariants();
}

#[test]
fn switch_activity_lift_branch_per_monitor_ewaf_two_monitors_mixed() {
    // Two monitors: out1 has EWAF on (via per-monitor layout_config), out2 has EWAF off
    // (uses default options). Both have a named workspace pre-tagged with beta. Structural
    // risk: if `ensure_all_activity_views` hoists `mon_options` outside the loop and reads the
    // first monitor's value for all monitors, the second monitor's view is built with the
    // wrong bookend discipline — either over-bookending (3 for out2 when 2 is correct) or
    // under-bookending (2 for out1 when 3 is correct, tripping monitor.rs:1746 "1 or 3+").
    let layout_part = jiji_config::LayoutPart {
        empty_workspace_above_first: Some(Flag(true)),
        ..Default::default()
    };
    let ops = [
        Op::AddScaledOutput {
            id: 1,
            scale: 1.0,
            layout_config: Some(Box::new(layout_part)),
        },
        Op::AddScaledOutput {
            id: 2,
            scale: 1.0,
            layout_config: None,
        },
        Op::AddNamedWorkspace {
            ws_name: 1,
            output_name: Some(1),
            layout_config: None,
        },
        Op::AddNamedWorkspace {
            ws_name: 2,
            output_name: Some(2),
            layout_config: None,
        },
    ];
    let mut layout = check_ops(ops);
    let out1 = layout.monitors[0].output_id();
    let out2 = layout.monitors[1].output_id();

    // Discriminator sanity: out1 has EWAF on, out2 has EWAF off.
    assert!(
        layout.monitors[0]
            .options
            .layout
            .empty_workspace_above_first,
        "out1 must have per-monitor EWAF enabled",
    );
    assert!(
        !layout.monitors[1]
            .options
            .layout
            .empty_workspace_above_first,
        "out2 must have EWAF disabled so the two monitors differ",
    );

    let beta = super::activity::Activity::new_runtime("beta".to_owned());
    let beta_id = beta.id();

    // Tag ws1 (bound to out1) and ws2 (bound to out2) with beta BEFORE inserting beta.
    let pick1 = layout
        .workspaces
        .values()
        .find(|ws| ws.name() == Some(&"ws1".to_owned()))
        .map(|ws| ws.id())
        .expect("ws1 must exist");
    let pick2 = layout
        .workspaces
        .values()
        .find(|ws| ws.name() == Some(&"ws2".to_owned()))
        .map(|ws| ws.id())
        .expect("ws2 must exist");
    layout
        .workspaces
        .get_mut(&pick1)
        .expect("pick1 must be a pool key")
        .activities
        .insert(beta_id);
    layout
        .workspaces
        .get_mut(&pick2)
        .expect("pick2 must be a pool key")
        .activities
        .insert(beta_id);

    test_insert_activity(&mut layout, beta);
    layout.switch_activity(beta_id);

    let beta_views = layout.activities.active().views();
    let view1 = beta_views
        .get(&out1)
        .expect("beta must have a view for out1");
    let view2 = beta_views
        .get(&out2)
        .expect("beta must have a view for out2");

    // out1 (EWAF on): must be [top_empty, pick1, bottom_empty] — len 3, active at 1.
    assert_eq!(
        view1.len(),
        3,
        "out1 with EWAF: lift shape must be [top_empty, pick1, bottom_empty]",
    );
    assert_eq!(
        view1.ids()[1],
        pick1,
        "pick1 sits at position 1 under per-monitor EWAF",
    );
    assert_eq!(
        view1.active_position(),
        1,
        "active index must be 1 (top-empty prepended) for out1",
    );

    // out2 (EWAF off): must be [pick2, bottom_empty] — len 2, active at 0.
    assert_eq!(
        view2.len(),
        2,
        "out2 without EWAF: lift shape must be [pick2, bottom_empty]",
    );
    assert_eq!(
        view2.ids()[0],
        pick2,
        "pick2 sits at position 0 (no top-empty on out2)",
    );
    assert_eq!(
        view2.active_position(),
        0,
        "active index must be 0 for out2 (no top-empty prepend)",
    );

    layout.verify_invariants();
}

#[test]
fn switch_activity_lift_branch_sticky_workspace_is_lifted() {
    // `create_activity` (mod.rs:4194-4205) auto-tags every sticky workspace with the
    // new activity's id, making sticky workspaces permanent lift candidates. Switching
    // to that activity must lift the sticky workspace into its view with correct bookends,
    // not skip it. Structural risk: a filter bug that excludes sticky workspaces from the
    // `tagged` collection would make the lift branch silently fall through to the fresh
    // branch, producing a single empty view instead of [sticky_ws, bottom_empty].
    //
    // Note: we use direct pool mutation (`is_sticky` is pub(super)) instead of
    // `create_activity` to isolate the `ensure_all_activity_views` lift-branch path from the
    // `create_activity` auto-tagging path.
    let ops = [
        Op::AddOutput(1),
        Op::AddNamedWorkspace {
            ws_name: 1,
            output_name: Some(1),
            layout_config: None,
        },
    ];
    let mut layout = check_ops(ops);
    let out_id = layout.monitors[0].output_id();

    let beta = super::activity::Activity::new_runtime("beta".to_owned());
    let beta_id = beta.id();

    // Mark ws1 as sticky and tag it with beta BEFORE inserting beta — simulating what
    // `create_activity` does for sticky workspaces, but exercising the materializer's lift
    // branch in isolation by pre-staging the tag.
    let pick = layout
        .workspaces
        .values()
        .find(|ws| ws.name() == Some(&"ws1".to_owned()))
        .map(|ws| ws.id())
        .expect("ws1 must exist after AddNamedWorkspace");
    {
        let ws = layout
            .workspaces
            .get_mut(&pick)
            .expect("pick must be a pool key");
        ws.is_sticky = true;
        ws.activities.insert(beta_id);
    }

    let pool_size_before = layout.workspaces.len();

    test_insert_activity(&mut layout, beta);
    layout.switch_activity(beta_id);

    let beta_view = &layout.activities.active().views()[&out_id];

    // Lift branch must include the sticky workspace and append a fresh bottom empty.
    assert_eq!(
        beta_view.len(),
        2,
        "sticky lift shape: [sticky_ws1, bottom_empty]",
    );
    assert_eq!(
        beta_view.ids()[0],
        pick,
        "sticky workspace sits at position 0",
    );
    assert_eq!(
        beta_view.active_position(),
        0,
        "active index points at the sticky workspace (no EWAF)",
    );
    assert_eq!(
        layout.workspaces.len(),
        pool_size_before + 1,
        "lift branch appends exactly one fresh trailing empty",
    );

    layout.verify_invariants();
}

// --- Per-activity bookend invariant: every (activity, connected-monitor) pair holds a
// bookended `WorkspaceView`. The materializer runs at create_activity / add_output /
// switch / view-mutator sites; the verifier enforces it on every refresh.

#[test]
fn create_activity_materializes_bookend_views_on_all_monitors() {
    // `Layout::create_activity` runs the per-activity materializer, so the newly-minted
    // activity must hold a fresh-empty trailing-empty view on every connected monitor
    // immediately — no `switch_activity` needed.
    let ops = [Op::AddOutput(1), Op::AddOutput(2)];
    let mut layout = check_ops(ops);
    let out1 = layout.monitors[0].output_id();
    let out2 = layout.monitors[1].output_id();

    let beta = layout.create_activity("Beta".to_owned()).expect("create");

    let beta_views = layout.activities.get(beta).expect("beta live").views();
    assert_eq!(
        beta_views.len(),
        2,
        "materializer must install a view per connected monitor at create-time",
    );
    let v1 = beta_views.get(&out1).expect("view for out1");
    let v2 = beta_views.get(&out2).expect("view for out2");
    assert_eq!(
        v1.len(),
        1,
        "fresh branch yields a single trailing-empty bookend on out1"
    );
    assert_eq!(
        v2.len(),
        1,
        "fresh branch yields a single trailing-empty bookend on out2"
    );

    layout.verify_invariants();
}

#[test]
fn create_activity_with_ewaf_materializes_trailing_empty_at_len_one() {
    // Under EWAF the materializer's fresh branch still produces a single trailing-empty
    // view (len == 1 is the EWAF "1 or 3+ length rule" allowed singleton). It does NOT
    // prepend a leading empty for the fresh branch — the leading-empty rule only fires
    // when there's at least one lifted body workspace to bookend.
    let options = Options {
        layout: jiji_config::Layout {
            empty_workspace_above_first: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops_with_options(options, ops);
    let out_id = layout.monitors[0].output_id();

    let beta = layout.create_activity("Beta".to_owned()).expect("create");

    let beta_view = layout
        .activities
        .get(beta)
        .expect("beta live")
        .views()
        .get(&out_id)
        .expect("view for out_id");
    assert_eq!(
        beta_view.len(),
        1,
        "fresh branch under EWAF still yields a single trailing-empty bookend (len==1 \
         singleton allowed by the EWAF 1-or-3+ rule)",
    );

    layout.verify_invariants();
}

#[test]
fn remove_activity_remaining_activities_keep_bookends() {
    // Removing one runtime activity that holds an exclusive-empty workspace destroys that
    // workspace and the activity. The materializer then re-runs over the remaining
    // activities; every surviving activity still has a bookend view per connected monitor.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();

    let beta = layout.create_activity("Beta".to_owned()).expect("create");
    let gamma = layout.create_activity("Gamma".to_owned()).expect("create");

    layout
        .remove_activity(&ActivityReferenceArg::Id(beta.get()))
        .expect("remove beta");

    assert!(!layout.activities.contains(beta));
    let out_id = layout.monitors[0].output_id();
    for (act, name) in [(alpha, "alpha"), (gamma, "gamma")] {
        let view = layout
            .activities
            .get(act)
            .expect("{name} must remain live")
            .views()
            .get(&out_id)
            .unwrap_or_else(|| panic!("{name} must hold a bookend view on out_id"));
        assert!(
            !view.ids().is_empty(),
            "{name}'s view is non-empty under the bookend invariant",
        );
    }

    layout.verify_invariants();
}

#[test]
fn add_output_materializes_views_for_every_existing_activity() {
    // Connecting a new monitor must materialize bookend views on the new output for EVERY
    // activity, not just the active one.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);

    let beta = layout.create_activity("Beta".to_owned()).expect("create");
    let gamma = layout.create_activity("Gamma".to_owned()).expect("create");

    // Connect a second monitor.
    check_ops_on_layout(&mut layout, [Op::AddOutput(2)]);

    let out2 = layout.monitors[1].output_id();
    for act in [beta, gamma] {
        assert!(
            layout
                .activities
                .get(act)
                .expect("activity live")
                .views()
                .contains_key(&out2),
            "connecting a new monitor must materialize a view for every existing activity",
        );
    }

    layout.verify_invariants();
}

#[test]
fn disconnect_reconnect_cycle_preserves_dormant_bookend_on_other_activities() {
    // Two monitors, three activities. Disconnect one — every activity's view for that
    // output is drained (the partial-disconnect dormant walk migrates surviving
    // workspaces into each activity's view for the remaining monitor and clears the
    // (activity, out1) entry). Reconnecting fires `ensure_all_activity_views` →
    // `ensure_view_for` for every activity, materializing a fresh view for out1.
    let ops = [Op::AddOutput(1), Op::AddOutput(2)];
    let mut layout = check_ops(ops);
    let seed = layout.active_activity_id();
    let out1 = layout.monitors[0].output_id();

    let beta = layout.create_activity("Beta".to_owned()).expect("create");
    let gamma = layout.create_activity("Gamma".to_owned()).expect("create");

    // Disconnect output1.
    check_ops_on_layout(&mut layout, [Op::RemoveOutput(1)]);

    // No activity holds a view keyed by out1 after the partial-disconnect drain.
    for activity in layout.activities.iter() {
        assert!(
            !activity.views().contains_key(&out1),
            "activity {:?} ({:?}) must not hold a view keyed by the disconnected output",
            activity.id(),
            activity.name(),
        );
    }
    layout.verify_invariants();

    // Reconnect — every activity must regain a view for out1 via the materializer.
    check_ops_on_layout(&mut layout, [Op::AddOutput(1)]);
    let out1_post = layout
        .monitors
        .iter()
        .find(|m| m.output_name() == "output1");
    assert!(out1_post.is_some(), "output1 must be connected again");

    for act_id in [seed, beta, gamma] {
        let activity = layout
            .activities
            .get(act_id)
            .expect("activity must remain live");
        assert!(
            activity.views().contains_key(&out1),
            "activity {:?} ({:?}) must hold a view for the reconnected output",
            act_id,
            activity.name(),
        );
    }
    layout.verify_invariants();
}

#[test]
fn set_workspace_activities_reinstates_view_after_single_entry_drop_in_dormant_activity() {
    // Set(W, [alpha]) where W previously had {alpha, beta} and beta's dormant view's single
    // non-bookend entry was W. The single-entry drop triggers the materializer to reinstall
    // a bookend on beta's view — `dropped_any_view_entry` covers dormant drops too.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let mon_out = layout.monitors[0].output_id();

    let beta = layout.create_activity("Beta".to_owned()).expect("create");
    let target_ws_id = layout
        .active_view(&mon_out)
        .ids()
        .first()
        .copied()
        .expect("alpha view has ws");

    layout
        .workspaces
        .get_mut(&target_ws_id)
        .expect("live")
        .activities = [alpha, beta].into_iter().collect();
    // Beta's view = [target] alone — explicitly drop the materialized bookend. The override
    // helper cleans up the orphaned bookend from the pool.
    test_override_activity_view(
        &mut layout,
        beta,
        mon_out.clone(),
        WorkspaceView::new(vec![target_ws_id], 0),
    );

    layout
        .set_workspace_activities(
            Some(WorkspaceReference::Id(target_ws_id.get())),
            &[ActivityReferenceArg::Id(alpha.get())],
        )
        .expect("set must succeed");

    // Beta's view: the materializer re-installed a fresh bookend after the single-entry drop.
    let beta_view = layout
        .activities
        .get(beta)
        .expect("beta live")
        .views()
        .get(&mon_out)
        .expect("beta's view re-installed by the materializer after single-entry drop");
    assert_eq!(beta_view.len(), 1, "fresh bookend");
    assert!(
        !beta_view.ids().contains(&target_ws_id),
        "target dropped from beta"
    );

    layout.verify_invariants();
}

#[test]
fn ensure_all_activity_views_no_op_when_monitors_empty() {
    // `ensure_all_activity_views` early-returns when monitors is empty — no view fabrication,
    // no pool growth. The no-monitors-empty branch of `verify_invariants` requires every
    // activity's views map to be empty in that state.
    let mut layout = Layout::<TestWindow>::default();
    let alpha = layout.active_activity_id();
    let pool_size_before = layout.workspaces.len();

    // No monitors connected.
    assert!(layout.monitors.is_empty());

    let beta_activity = super::activity::Activity::new_runtime("beta".to_owned());
    test_insert_activity(&mut layout, beta_activity);

    // Both activities' views remain empty (materializer is a no-op).
    assert!(layout.activities.get(alpha).unwrap().views().is_empty());
    assert_eq!(
        layout.workspaces.len(),
        pool_size_before,
        "monitors-empty materializer does not allocate",
    );

    layout.verify_invariants();
}

#[test]
fn add_window_to_active_workspace_maintains_dormant_view_bookend() {
    // Cross-activity bookend maintenance: a workspace shared between alpha (active) and
    // beta (dormant) where beta's view has it as the trailing entry. When a window opens
    // into the workspace via the active path, the dormant_view_bookend_fixup must append
    // a fresh trailing empty to beta's view too.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let mon_out = layout.monitors[0].output_id();

    let beta = layout.create_activity("Beta".to_owned()).expect("create");
    let w_id = layout
        .active_view(&mon_out)
        .ids()
        .first()
        .copied()
        .expect("alpha has a workspace");
    // `SetWorkspaceActivities` from {alpha} → {alpha, beta} adds beta. The Add branch
    // appends w_id to beta's view (which currently holds only its materialized bookend),
    // yielding beta's view = [beta_bookend, w_id]. w_id is the trailing entry.
    layout
        .set_workspace_activities(
            Some(WorkspaceReference::Id(w_id.get())),
            &[
                ActivityReferenceArg::Id(alpha.get()),
                ActivityReferenceArg::Id(beta.get()),
            ],
        )
        .expect("set must succeed");

    let beta_view_before = layout
        .activities
        .get(beta)
        .expect("beta live")
        .views()
        .get(&mon_out)
        .expect("beta has view")
        .clone();
    assert_eq!(beta_view_before.ids().last(), Some(&w_id));

    // Add a window via the active path.
    let win = TestWindow::new(TestWindowParams::new(7));
    layout.add_window(
        win,
        AddWindowTarget::Auto,
        None,
        None,
        false,
        false,
        ActivateWindow::Smart,
    );

    // Beta's view must have grown: the trailing entry is no longer w_id (now non-empty), a
    // fresh empty was appended after it.
    let beta_view_after = layout
        .activities
        .get(beta)
        .expect("beta live")
        .views()
        .get(&mon_out)
        .expect("beta has view");
    assert!(
        beta_view_after.len() > beta_view_before.len(),
        "dormant_view_bookend_fixup must have extended beta's view past w_id",
    );
    assert_ne!(
        beta_view_after.ids().last(),
        Some(&w_id),
        "w_id is no longer the trailing entry of beta's view",
    );

    layout.verify_invariants();
}

/// Sets up: alpha view = `[W_src(has window, active), W_target]` on output1; beta is a dormant
/// activity whose view ends in `W_target`. The active workspace's column will be moved into
/// `W_target` by the action under test, after which `W_target` is no longer empty — so beta's
/// dormant view bookend invariant requires a fresh empty appended.
///
/// Returns `(layout, alpha, beta, mon_out, w_target_id)` for the assertion phase.
#[track_caller]
fn setup_shared_trailing_bookend_fixture() -> (
    Layout<TestWindow>,
    ActivityId,
    ActivityId,
    OutputId,
    WorkspaceId,
) {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
    ];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let mon_out = layout.monitors[0].output_id();

    // After AddWindow the active view is `[W_src(has window, active), W_target(empty)]`.
    let w_target_id = layout
        .active_view(&mon_out)
        .ids()
        .last()
        .copied()
        .expect("alpha view has trailing bookend");

    let beta = layout.create_activity("Beta".to_owned()).expect("create");
    // Set on W_target: {alpha} → {alpha, beta}. The Add branch appends W_target to beta's view
    // (currently `[W_beta_seed]`), yielding beta's view = `[W_beta_seed, W_target]`.
    layout
        .set_workspace_activities(
            Some(WorkspaceReference::Id(w_target_id.get())),
            &[
                ActivityReferenceArg::Id(alpha.get()),
                ActivityReferenceArg::Id(beta.get()),
            ],
        )
        .expect("set must succeed");

    let beta_trailing = layout
        .activities
        .get(beta)
        .expect("beta live")
        .views()
        .get(&mon_out)
        .expect("beta view materialized")
        .ids()
        .last()
        .copied();
    assert_eq!(
        beta_trailing,
        Some(w_target_id),
        "fixture precondition: w_target is beta's trailing entry",
    );

    (layout, alpha, beta, mon_out, w_target_id)
}

/// Common post-condition: beta's dormant view grew by exactly one fresh empty appended after
/// `w_target_id` (i.e. the fixup ran), and `layout.verify_invariants` passes (which itself
/// re-checks `Monitor::verify_invariants`'s per-view bookend assertion).
#[track_caller]
fn assert_dormant_trailing_fixup_landed(
    layout: &Layout<TestWindow>,
    beta: ActivityId,
    mon_out: &OutputId,
    w_target_id: WorkspaceId,
    beta_len_before: usize,
) {
    let beta_view_after = layout
        .activities
        .get(beta)
        .expect("beta live")
        .views()
        .get(mon_out)
        .expect("beta view live");
    assert_eq!(
        beta_view_after.len(),
        beta_len_before + 1,
        "dormant_view_bookend_fixup must have appended exactly one fresh empty",
    );
    assert_ne!(
        beta_view_after.ids().last(),
        Some(&w_target_id),
        "w_target_id is no longer beta's trailing entry — a fresh empty came after it",
    );

    layout.verify_invariants();
}

#[test]
fn move_column_to_workspace_down_into_shared_trailing_bookend_appends_dormant_bookend() {
    // alpha view = `[W_src(has window, active at 0), W_target]`; beta view trails W_target.
    // `move_column_to_workspace_down` moves the column from W_src into W_target. W_target
    // gains content; beta's dormant view still ends in W_target. Without the public-entry-
    // point fixup wiring, the per-view bookend invariant trips when verify_invariants runs.
    let (mut layout, _alpha, beta, mon_out, w_target_id) = setup_shared_trailing_bookend_fixture();
    let beta_len_before = layout
        .activities
        .get(beta)
        .unwrap()
        .views()
        .get(&mon_out)
        .unwrap()
        .len();

    layout.move_column_to_workspace_down(false);

    assert_dormant_trailing_fixup_landed(&layout, beta, &mon_out, w_target_id, beta_len_before);
}

#[test]
fn move_column_to_workspace_into_shared_trailing_bookend_appends_dormant_bookend() {
    // Same fixture as the `_down` test; the explicit-idx entry point fires on `idx = 1`.
    let (mut layout, _alpha, beta, mon_out, w_target_id) = setup_shared_trailing_bookend_fixture();
    let beta_len_before = layout
        .activities
        .get(beta)
        .unwrap()
        .views()
        .get(&mon_out)
        .unwrap()
        .len();

    layout.move_column_to_workspace(1, false);

    assert_dormant_trailing_fixup_landed(&layout, beta, &mon_out, w_target_id, beta_len_before);
}

#[test]
fn move_column_to_workspace_up_into_shared_trailing_bookend_appends_dormant_bookend() {
    // Source must be at position > 0 for `_up` to do anything. Move-column-down with
    // `activate = true` first to grow the view to `[W_target(empty), W_src(has window, active at
    // 1), W_trail(empty)]`, then set up beta to share W_target at trailing.
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::MoveColumnToWorkspaceDown(true),
    ];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let mon_out = layout.monitors[0].output_id();

    // After the ops: alpha view = `[W_target(empty), W_src(has window, active at 1),
    // W_trail(empty)]`.
    let alpha_view_ids: Vec<WorkspaceId> = layout.active_view(&mon_out).ids().to_vec();
    assert_eq!(alpha_view_ids.len(), 3, "fixture: alpha view has 3 entries");
    assert_eq!(
        layout.active_view(&mon_out).active_position(),
        1,
        "fixture: alpha active at position 1",
    );
    // Position 0 is the formerly-empty trailing bookend that `move_column_to_workspace_down`
    // minted on the previous step; the column moved down into it, so it is now the source
    // workspace for the upcoming `_up` move. It will also receive the column when `_up`
    // reverses the move — making it the "target" from the fixup's perspective.
    let w_target_id = alpha_view_ids[0];

    let beta = layout.create_activity("Beta".to_owned()).expect("create");
    layout
        .set_workspace_activities(
            Some(WorkspaceReference::Id(w_target_id.get())),
            &[
                ActivityReferenceArg::Id(alpha.get()),
                ActivityReferenceArg::Id(beta.get()),
            ],
        )
        .expect("set must succeed");

    let beta_len_before = layout
        .activities
        .get(beta)
        .unwrap()
        .views()
        .get(&mon_out)
        .unwrap()
        .len();
    assert_eq!(
        layout
            .activities
            .get(beta)
            .unwrap()
            .views()
            .get(&mon_out)
            .unwrap()
            .ids()
            .last()
            .copied(),
        Some(w_target_id),
        "fixture precondition: beta's trailing entry is w_target",
    );

    layout.move_column_to_workspace_up(false);

    assert_dormant_trailing_fixup_landed(&layout, beta, &mon_out, w_target_id, beta_len_before);
}

#[test]
fn add_column_by_idx_into_shared_trailing_bookend_appends_dormant_bookend() {
    // Exercise `add_column_by_idx` via its only public caller: `move_column_to_output`. Two
    // outputs; alpha has a window on output1 and an empty workspace on output2. beta shares
    // output2's workspace at its trailing position. Moving the column from output1 to output2
    // lands it on output2's active workspace via `add_column_by_idx`.
    let ops = [
        Op::AddOutput(1),
        Op::AddOutput(2),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
    ];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let out2_id = layout.monitors[1].output_id();
    let out2_output = layout.monitors[1].output.clone();

    // Output2's view in alpha has a single bookend workspace. That's the target.
    let w_target_id = layout
        .active_view(&out2_id)
        .ids()
        .first()
        .copied()
        .expect("alpha's output2 view has a bookend");

    let beta = layout.create_activity("Beta".to_owned()).expect("create");
    layout
        .set_workspace_activities(
            Some(WorkspaceReference::Id(w_target_id.get())),
            &[
                ActivityReferenceArg::Id(alpha.get()),
                ActivityReferenceArg::Id(beta.get()),
            ],
        )
        .expect("set must succeed");

    let beta_len_before = layout
        .activities
        .get(beta)
        .unwrap()
        .views()
        .get(&out2_id)
        .unwrap()
        .len();

    layout.move_column_to_output(&out2_output, None, true);

    assert_dormant_trailing_fixup_landed(&layout, beta, &out2_id, w_target_id, beta_len_before);
}

#[test]
fn add_column_by_idx_into_shared_leading_bookend_under_ewaf_prepends_dormant_bookend() {
    // EWAF (empty_workspace_above_first) variant: position 0 is also a bookend slot. The
    // helper at `dormant_view_bookend_fixup` mints a leading empty in dormant views when
    // `ewaf && is_first` — i.e. the receiving workspace sits at position 0 of a dormant
    // view. Construct beta's dormant view via `test_override_activity_view` so the shared
    // workspace lands at position 0 (`set_workspace_activities`'s Add branch only appends,
    // so we can't get it there via Set alone).
    let options = Options {
        layout: jiji_config::Layout {
            empty_workspace_above_first: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let ops = [
        Op::AddOutput(1),
        Op::AddOutput(2),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
    ];
    let mut layout = check_ops_with_options(options, ops);
    let alpha = layout.active_activity_id();
    let out2_id = layout.monitors[1].output_id();
    let out2_output = layout.monitors[1].output.clone();

    // Output2's view in alpha has a single bookend workspace (no content yet on output2, so
    // EWAF hasn't minted a leading empty there). That single workspace is the target.
    let w_target_id = layout
        .active_view(&out2_id)
        .ids()
        .first()
        .copied()
        .expect("alpha's output2 view has a bookend");

    let beta = layout.create_activity("Beta".to_owned()).expect("create");
    // Widen ownership directly on the pool entry rather than going through
    // `set_workspace_activities` — the latter would leave beta's dormant view at len 2
    // momentarily and trip the EWAF "1 or 3+" rule before the override below restores it.
    layout
        .workspaces
        .get_mut(&w_target_id)
        .expect("w_target live")
        .activities = [alpha, beta].into_iter().collect();

    // Construct beta's output2 view with W_target at position 0 (leading slot). Under
    // EWAF, `Monitor::verify_invariants` requires either 1 or 3+ workspaces in a view —
    // mint two fresh empties tagged to beta to fill out a valid 3-wide layout
    // `[W_target, W_filler_mid, W_filler_trail]` where W_target is the leading slot.
    let w_filler_mid = test_mint_empty_for(&mut layout, 1, beta);
    let w_filler_trail = test_mint_empty_for(&mut layout, 1, beta);
    test_override_activity_view(
        &mut layout,
        beta,
        out2_id.clone(),
        WorkspaceView::new(vec![w_target_id, w_filler_mid, w_filler_trail], 1),
    );
    // Pin that the manually-assembled fixture state is already invariant-clean before
    // the action under test fires.
    layout.verify_invariants();

    let beta_len_before = layout
        .activities
        .get(beta)
        .unwrap()
        .views()
        .get(&out2_id)
        .unwrap()
        .len();

    // Move the column from output1's active workspace to output2's W_target (idx 0). Under
    // EWAF, `add_column_on` mints a top bookend in alpha (since workspace_idx == 0). The
    // dormant-view fixup mirrors for beta: W_target is at beta's position 0 → mint a
    // leading empty there.
    layout.move_column_to_output(&out2_output, Some(0), true);

    let beta_view_after = layout
        .activities
        .get(beta)
        .unwrap()
        .views()
        .get(&out2_id)
        .unwrap();
    assert_eq!(
        beta_view_after.len(),
        beta_len_before + 1,
        "EWAF leading share: helper prepends exactly one leading-fixup empty",
    );
    assert_ne!(
        beta_view_after.ids().first(),
        Some(&w_target_id),
        "w_target is no longer beta's leading entry — a fresh empty came before it",
    );

    layout.verify_invariants();
}

#[test]
fn column_add_into_shared_trailing_bookend_keeps_verify_invariants_passing() {
    // Regression pin against the existing per-view bookend assertion in
    // `Monitor::verify_invariants`. With the public-entry-point fixup wiring in place,
    // adding a column to a workspace at a dormant view's trailing position leaves every
    // monitor's per-view assertion satisfied. Without the wiring, the same operation would
    // trip the assertion (proven by removing the fixup call and observing the panic during
    // local discrimination).
    let (mut layout, _alpha, _beta, _mon_out, _w_target_id) =
        setup_shared_trailing_bookend_fixture();

    // Fire any of the four newly-wired entry points — the regression pin is that the
    // post-action verify_invariants passes. `move_column_to_workspace_down` is the simplest.
    layout.move_column_to_workspace_down(false);

    // `verify_invariants` runs all of `Monitor::verify_invariants` which in turn runs
    // `assert_view_bookends` per view in every connected monitor — the assertion that would
    // trip without the new wiring.
    layout.verify_invariants();
}

#[test]
fn move_column_to_workspace_down_floating_path_appends_dormant_bookend() {
    // Exercises the floating-recursion branch of `move_column_to_workspace_down_on`:
    // when the active workspace has `floating_is_active() == true`, the inner associated fn
    // routes through `move_to_workspace_down_on` → `add_tile_on` rather than the scrolling
    // `add_column_on` path. The public-entry-point fixup at the call site fires
    // unconditionally, so the dormant bookend must be repaired regardless of which inner
    // branch executed. Without the unconditional placement at the public entry point, a
    // refactor that moved the fixup inside `add_column_on` would silently leave the
    // floating path unfixed.
    let (mut layout, _alpha, beta, mon_out, w_target_id) = setup_shared_trailing_bookend_fixture();

    // Add a floating window to the active (source) workspace so we can put floating focus
    // on it. The fixture workspace currently holds one tiling window (id=1); use id=2.
    let float_win = TestWindow::new(TestWindowParams {
        id: 2,
        is_floating: true,
        ..TestWindowParams::new(2)
    });
    layout.add_window(
        float_win,
        AddWindowTarget::Auto,
        None,
        None,
        false,
        true,
        ActivateWindow::default(),
    );
    layout.focus_floating();
    // Verify the floating branch will actually be taken.
    assert!(
        layout
            .active_workspace()
            .expect("active workspace exists")
            .floating_is_active(),
        "fixture precondition: floating must be active so the floating-recursion branch fires",
    );

    let beta_len_before = layout
        .activities
        .get(beta)
        .unwrap()
        .views()
        .get(&mon_out)
        .unwrap()
        .len();

    layout.move_column_to_workspace_down(false);

    assert_dormant_trailing_fixup_landed(&layout, beta, &mon_out, w_target_id, beta_len_before);
}

#[test]
fn dormant_view_bookend_fixup_len1_ewaf_mints_both_bookends() {
    // A dormant view of length 1 under EWAF contains a single workspace that is
    // simultaneously `is_last` (position == len-1 == 0) and `ewaf && is_first`
    // (position == 0). Both branches in `dormant_view_bookend_fixup`'s loop body fire
    // on the same needs_fixup entry, minting a new trailing empty AND a new leading empty
    // in one call — growing the view from length 1 to length 3 with the original
    // workspace sandwiched in the middle.
    let options = Options {
        layout: jiji_config::Layout {
            empty_workspace_above_first: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let ops = [
        Op::AddOutput(1),
        Op::AddOutput(2),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
    ];
    let mut layout = check_ops_with_options(options, ops);
    let alpha = layout.active_activity_id();
    let out2_id = layout.monitors[1].output_id();
    let out2_output = layout.monitors[1].output.clone();

    // Alpha's out2 view has a single trailing bookend (no content on out2 yet, so EWAF
    // has not minted a leading empty). That single workspace is the shared target.
    let w_target_id = layout
        .active_view(&out2_id)
        .ids()
        .first()
        .copied()
        .expect("alpha's out2 view has a bookend");

    let beta = layout.create_activity("Beta".to_owned()).expect("create");
    // Directly widen the workspace's activity set so beta shares it. The
    // `set_workspace_activities` Add branch would give beta a len-2 view (its seed
    // workspace + w_target); we want len-1 to hit the dual-mint case. Bypass via the pool.
    layout
        .workspaces
        .get_mut(&w_target_id)
        .expect("w_target live")
        .activities = [alpha, beta].into_iter().collect();

    // Construct beta's out2 view as a length-1 singleton [W_target, active=0]. Under
    // EWAF, len==1 satisfies the "1 or 3+" rule (the singleton-allowed branch).
    test_override_activity_view(
        &mut layout,
        beta,
        out2_id.clone(),
        WorkspaceView::new(vec![w_target_id], 0),
    );
    // Verify the fixture state is invariant-clean before the action under test fires.
    layout.verify_invariants();

    // Move the column from out1 to out2's w_target (idx 0). Under EWAF, `add_column_on`
    // mints both a leading and a trailing bookend in alpha's view. The fixup observes that
    // w_target is at position 0 of beta's len-1 view, so both `is_last` and
    // `ewaf && is_first` are true — the helper mints one trailing empty AND one leading
    // empty, growing beta's out2 view from [W_target] to [fresh_top, W_target, fresh_bottom].
    layout.move_column_to_output(&out2_output, Some(0), true);

    let beta_view_after = layout
        .activities
        .get(beta)
        .unwrap()
        .views()
        .get(&out2_id)
        .unwrap();
    assert_eq!(
        beta_view_after.len(),
        3,
        "len-1 EWAF dormant view: fixup must mint both leading and trailing empty, growing view to 3",
    );
    assert_ne!(
        beta_view_after.ids().first(),
        Some(&w_target_id),
        "w_target is no longer at position 0 — a fresh leading empty was prepended",
    );
    assert_ne!(
        beta_view_after.ids().last(),
        Some(&w_target_id),
        "w_target is no longer at the trailing position — a fresh trailing empty was appended",
    );

    layout.verify_invariants();
}

#[test]
fn set_workspace_name_trailing_dormant_view_appends_bookend() {
    // Naming a workspace at the trailing bookend slot of a dormant view violates the
    // per-view bookend invariant on that view (name.is_some() at a slot whose contract
    // requires name.is_none()). The active view's bookend is patched inline via
    // `add_workspace_bottom_on`; the dormant view is patched via
    // `dormant_view_bookend_fixup`.
    let (mut layout, alpha, beta, mon_out, w_target_id) = setup_shared_trailing_bookend_fixture();
    // Pin the fixture: w_target is at the trailing slot of both alpha (active) and beta
    // (dormant). Both views need a fresh trailing empty after the rename.
    let alpha_view = layout.active_view(&mon_out).ids().to_vec();
    assert_eq!(
        alpha_view.last().copied(),
        Some(w_target_id),
        "fixture precondition: w_target is alpha's trailing slot",
    );
    let beta_len_before = layout
        .activities
        .get(beta)
        .unwrap()
        .views()
        .get(&mon_out)
        .unwrap()
        .len();
    let alpha_len_before = alpha_view.len();

    layout.set_workspace_name(
        "named".to_owned(),
        Some(WorkspaceReference::Id(w_target_id.get())),
    );

    // Alpha's active view: w_target was named, so a fresh trailing empty was appended.
    let alpha_view_after = layout.active_view(&mon_out).ids().to_vec();
    assert_eq!(
        alpha_view_after.len(),
        alpha_len_before + 1,
        "active view grew by exactly one fresh trailing empty",
    );
    assert_eq!(
        alpha_view_after[alpha_len_before - 1],
        w_target_id,
        "w_target is no longer the trailing slot but still occupies its old position",
    );
    let new_trailing_id = *alpha_view_after.last().expect("trailing slot exists");
    let new_trailing_ws = layout
        .workspaces
        .get(&new_trailing_id)
        .expect("trailing id is a pool key");
    assert!(
        new_trailing_ws.name.is_none(),
        "fresh trailing empty must be unnamed",
    );
    assert!(
        !new_trailing_ws.has_windows(),
        "fresh trailing empty must hold no windows",
    );

    // w_target itself now carries the new name.
    assert_eq!(
        layout
            .workspaces
            .get(&w_target_id)
            .expect("w_target live")
            .name
            .as_deref(),
        Some("named"),
        "w_target now carries the new name",
    );

    // Dormant view (beta): w_target was at beta's trailing slot too, so the dormant fixup
    // appended a fresh trailing empty.
    assert_dormant_trailing_fixup_landed(&layout, beta, &mon_out, w_target_id, beta_len_before);
    let _ = alpha;
}

#[test]
fn set_workspace_name_under_ewaf_leading_dormant_view_prepends_bookend() {
    // EWAF variant: position 0 is also a bookend slot. Naming a workspace at position 0 of
    // both active and dormant views must mint a fresh leading empty in each.
    let options = Options {
        layout: jiji_config::Layout {
            empty_workspace_above_first: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
    ];
    let mut layout = check_ops_with_options(options, ops);
    let alpha = layout.active_activity_id();
    let mon_out = layout.monitors[0].output_id();

    // Under EWAF, alpha's view is `[W_top_empty, W_src(has window), W_trail_empty]`. The
    // leading workspace W_top_empty is the rename target.
    let w_target_id = layout
        .active_view(&mon_out)
        .ids()
        .first()
        .copied()
        .expect("alpha view has a leading bookend under EWAF");

    let beta = layout.create_activity("Beta".to_owned()).expect("create");
    // Widen w_target's activity set so beta shares it. Bypass `set_workspace_activities`:
    // we want a hand-rolled beta view with w_target at position 0, and the Add branch would
    // append it instead.
    layout
        .workspaces
        .get_mut(&w_target_id)
        .expect("w_target live")
        .activities = [alpha, beta].into_iter().collect();

    // Construct beta's view with w_target at position 0. EWAF requires 1 or 3+; use 3
    // with w_target leading and two fresh empties filling the middle and trailing slots.
    let w_filler_mid = test_mint_empty_for(&mut layout, 0, beta);
    let w_filler_trail = test_mint_empty_for(&mut layout, 0, beta);
    test_override_activity_view(
        &mut layout,
        beta,
        mon_out.clone(),
        WorkspaceView::new(vec![w_target_id, w_filler_mid, w_filler_trail], 1),
    );
    layout.verify_invariants();

    let beta_len_before = layout
        .activities
        .get(beta)
        .unwrap()
        .views()
        .get(&mon_out)
        .unwrap()
        .len();
    let alpha_len_before = layout.active_view(&mon_out).ids().len();

    layout.set_workspace_name(
        "named".to_owned(),
        Some(WorkspaceReference::Id(w_target_id.get())),
    );

    // Alpha's active view: w_target was at position 0 → fresh leading empty prepended.
    let alpha_view_after = layout.active_view(&mon_out).ids().to_vec();
    assert_eq!(
        alpha_view_after.len(),
        alpha_len_before + 1,
        "active view grew by exactly one fresh leading empty",
    );
    assert_ne!(
        alpha_view_after.first(),
        Some(&w_target_id),
        "w_target is no longer alpha's leading slot — a fresh empty was prepended",
    );

    // Dormant view (beta): w_target was at position 0 → fresh leading empty prepended.
    let beta_view_after = layout
        .activities
        .get(beta)
        .unwrap()
        .views()
        .get(&mon_out)
        .unwrap();
    assert_eq!(
        beta_view_after.len(),
        beta_len_before + 1,
        "EWAF leading share: dormant fixup prepends exactly one leading empty",
    );
    assert_ne!(
        beta_view_after.ids().first(),
        Some(&w_target_id),
        "w_target is no longer beta's leading slot — a fresh empty was prepended",
    );

    layout.verify_invariants();
}

#[test]
fn set_workspace_name_via_id_targets_non_active_monitor() {
    // Cross-monitor `Id` rename regression pin: `WorkspaceReference::Id` can target a workspace on
    // a monitor that is *not* `active_monitor_idx`. The bookend mint must anchor to the
    // workspace's actual monitor, not the active one. Without the fix, this test trips the
    // per-view bookend assertion on monitor 1 at the next `verify_invariants` call.
    let ops = [
        Op::AddOutput(1),
        Op::AddOutput(2),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
    ];
    let mut layout = check_ops(ops);
    // Focus stays on monitor 0 after AddWindow.
    assert_eq!(layout.active_monitor_idx, 0, "fixture: focus on monitor 0");
    let mon0_out = layout.monitors[0].output_id();
    let mon1_out = layout.monitors[1].output_id();

    // Monitor 1's active view holds a single bookend workspace — the rename target.
    let w_target_id = layout
        .active_view(&mon1_out)
        .ids()
        .first()
        .copied()
        .expect("monitor 1 has a bookend workspace");

    let mon0_view_before = layout.active_view(&mon0_out).ids().to_vec();
    let mon1_len_before = layout.active_view(&mon1_out).ids().len();

    layout.set_workspace_name(
        "named".to_owned(),
        Some(WorkspaceReference::Id(w_target_id.get())),
    );

    // Monitor 1: w_target was at the trailing slot of a len-1 view → fresh trailing empty
    // appended.
    let mon1_view_after = layout.active_view(&mon1_out).ids().to_vec();
    assert_eq!(
        mon1_view_after.len(),
        mon1_len_before + 1,
        "monitor 1's view grew by exactly one fresh trailing empty",
    );
    assert_ne!(
        mon1_view_after.last(),
        Some(&w_target_id),
        "w_target is no longer monitor 1's trailing slot",
    );

    // Monitor 0: unchanged.
    let mon0_view_after = layout.active_view(&mon0_out).ids().to_vec();
    assert_eq!(
        mon0_view_after, mon0_view_before,
        "monitor 0's view is untouched by a rename targeted at monitor 1",
    );

    // The pinned regression: `verify_invariants` runs `assert_view_bookends` per view in
    // every connected monitor — would trip on monitor 1 without the fix.
    layout.verify_invariants();
}

#[test]
fn set_workspace_name_dormant_only_view_via_id_appends_bookend() {
    // Regression pin: when `wsid` is held only in a dormant activity's view of a
    // connected monitor (no active-activity view contains it), the monitor lookup
    // must still locate the hosting monitor and `dormant_view_bookend_fixup` must
    // re-mint the trailing bookend on the dormant view. Without the fan-out, the
    // active-view-only predicate returns `None` and the silent skip leaves the
    // dormant view's trailing slot named — `verify_invariants` then trips at the
    // next mutating refresh.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let seed_id = layout.active_activity_id();
    let mon_out = layout.monitors[0].output_id();

    let beta_id = layout.create_activity("Beta".to_owned()).expect("create");
    // Bootstrap beta's view by switching to it, then switch back so beta's view is dormant
    // and seed is active. Beta's view at this point holds exactly its own bookend empty,
    // tagged with `{beta}` only — not shared with seed.
    layout.switch_activity(beta_id);
    layout.switch_activity_previous();
    assert_eq!(layout.active_activity_id(), seed_id);

    let beta_view_before = layout
        .activities
        .get(beta_id)
        .expect("beta live")
        .views()
        .get(&mon_out)
        .expect("beta view materialized")
        .ids()
        .to_vec();
    let w_target_id = *beta_view_before
        .last()
        .expect("beta view holds at least the bookend");
    // Pin the fixture: seed's active view does NOT contain w_target_id (dormant-only).
    let seed_view_before = layout.active_view(&mon_out).ids().to_vec();
    assert!(
        !seed_view_before.contains(&w_target_id),
        "fixture: w_target lives only in beta's dormant view",
    );
    // Structural pin: w_target's workspace is tagged to beta only, confirming the
    // dormant-only framing — seed does not own it.
    assert_eq!(
        layout
            .workspaces
            .get(&w_target_id)
            .expect("w_target is a pool key")
            .activities,
        HashSet::from([beta_id]),
        "fixture: w_target workspace belongs to beta only",
    );
    let seed_len_before = seed_view_before.len();
    let beta_len_before = beta_view_before.len();

    layout.set_workspace_name(
        "named".to_owned(),
        Some(WorkspaceReference::Id(w_target_id.get())),
    );

    // Seed's active view is unchanged — w_target was never in it.
    let seed_view_after = layout.active_view(&mon_out).ids().to_vec();
    assert_eq!(
        seed_view_after.len(),
        seed_len_before,
        "seed's active view length unchanged — w_target is dormant-only",
    );
    assert_eq!(
        seed_view_after, seed_view_before,
        "seed's active view ids unchanged",
    );

    // Beta's dormant view grew by exactly one fresh trailing empty after the now-named
    // w_target.
    let beta_view_after = layout
        .activities
        .get(beta_id)
        .expect("beta live")
        .views()
        .get(&mon_out)
        .expect("beta view live")
        .ids()
        .to_vec();
    assert_eq!(
        beta_view_after.len(),
        beta_len_before + 1,
        "dormant view grew by exactly one fresh trailing empty",
    );
    assert_eq!(
        beta_view_after[beta_len_before - 1],
        w_target_id,
        "w_target stays at its old position; a fresh empty follows it",
    );
    let new_trailing_id = *beta_view_after.last().expect("trailing slot exists");
    let new_trailing_ws = layout
        .workspaces
        .get(&new_trailing_id)
        .expect("trailing id is a pool key");
    assert!(
        new_trailing_ws.name.is_none(),
        "fresh trailing empty must be unnamed",
    );
    assert!(
        !new_trailing_ws.has_windows(),
        "fresh trailing empty must hold no windows",
    );

    // w_target itself now carries the new name.
    assert_eq!(
        layout
            .workspaces
            .get(&w_target_id)
            .expect("w_target live")
            .name
            .as_deref(),
        Some("named"),
        "w_target now carries the new name",
    );

    layout.verify_invariants();
}

#[test]
fn set_workspace_name_under_ewaf_dormant_only_view_via_id_prepends_bookend() {
    // EWAF mirror of the trailing-bookend dormant-only regression. With
    // `empty_workspace_above_first` set, position 0 of every view is also a bookend slot.
    // A workspace held only at position 0 of a dormant activity's view of a connected
    // monitor — addressed via `WorkspaceReference::Id` — must still be located and
    // patched by `dormant_view_bookend_fixup`, which prepends a fresh leading empty.
    let options = Options {
        layout: jiji_config::Layout {
            empty_workspace_above_first: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops_with_options(options, ops);
    let seed_id = layout.active_activity_id();
    let mon_out = layout.monitors[0].output_id();

    let beta_id = layout.create_activity("Beta".to_owned()).expect("create");
    layout.switch_activity(beta_id);
    layout.switch_activity_previous();
    assert_eq!(layout.active_activity_id(), seed_id);

    // Beta's dormant view under EWAF has at least a leading bookend at position 0.
    let beta_view_before = layout
        .activities
        .get(beta_id)
        .expect("beta live")
        .views()
        .get(&mon_out)
        .expect("beta view materialized")
        .ids()
        .to_vec();
    // Rename target is the leading slot. Bootstrap minted len = 1 under EWAF (the
    // single-entry shape is bookend-legal); to expose the leading-slot rename specifically
    // we need len >= 3 with the target at position 0. Hand-roll beta's view via
    // `test_override_activity_view` to fill the middle and trailing slots, leaving the
    // existing bootstrap entry at position 0 as the rename target.
    let w_target_id = beta_view_before[0];
    let w_filler_mid = test_mint_empty_for(&mut layout, 0, beta_id);
    let w_filler_trail = test_mint_empty_for(&mut layout, 0, beta_id);
    test_override_activity_view(
        &mut layout,
        beta_id,
        mon_out.clone(),
        WorkspaceView::new(vec![w_target_id, w_filler_mid, w_filler_trail], 1),
    );
    layout.verify_invariants();

    // Pin: seed's active view does NOT contain w_target_id.
    let seed_view_before = layout.active_view(&mon_out).ids().to_vec();
    assert!(
        !seed_view_before.contains(&w_target_id),
        "fixture: w_target lives only in beta's dormant view",
    );
    let beta_len_before = layout
        .activities
        .get(beta_id)
        .unwrap()
        .views()
        .get(&mon_out)
        .unwrap()
        .len();
    let seed_len_before = seed_view_before.len();

    layout.set_workspace_name(
        "named".to_owned(),
        Some(WorkspaceReference::Id(w_target_id.get())),
    );

    // Seed's active view unchanged.
    let seed_view_after = layout.active_view(&mon_out).ids().to_vec();
    assert_eq!(
        seed_view_after.len(),
        seed_len_before,
        "seed's active view length unchanged",
    );

    // Beta's dormant view prepended a fresh leading empty.
    let beta_view_after = layout
        .activities
        .get(beta_id)
        .unwrap()
        .views()
        .get(&mon_out)
        .unwrap()
        .ids()
        .to_vec();
    assert_eq!(
        beta_view_after.len(),
        beta_len_before + 1,
        "dormant view grew by exactly one fresh leading empty under EWAF",
    );
    let new_leading_id = *beta_view_after.first().expect("leading slot exists");
    assert_ne!(
        new_leading_id, w_target_id,
        "w_target is no longer beta's leading slot — a fresh empty was prepended",
    );
    let new_leading_ws = layout
        .workspaces
        .get(&new_leading_id)
        .expect("leading id is a pool key");
    assert!(
        new_leading_ws.name.is_none(),
        "fresh leading empty must be unnamed",
    );
    assert!(
        !new_leading_ws.has_windows(),
        "fresh leading empty must hold no windows",
    );
    assert_eq!(
        beta_view_after[1], w_target_id,
        "w_target is now at position 1 — shifted by the prepended leading empty",
    );
    // w_target itself now carries the new name.
    assert_eq!(
        layout
            .workspaces
            .get(&w_target_id)
            .expect("w_target live")
            .name
            .as_deref(),
        Some("named"),
        "w_target now carries the new name",
    );

    layout.verify_invariants();
}

#[test]
fn set_workspace_name_dormant_view_on_non_active_monitor_via_id_appends_bookend() {
    // Multi-monitor regression pin: fan-out must locate the correct mon_idx even when the
    // workspace lives only in a dormant activity's view on a monitor that is NOT the active
    // monitor. Without the fan-out, `active_monitor_idx` (pointing at M0) would be used, the
    // lookup would fail, and the dormant view on M1 would be left with a named trailing slot.
    let ops = [Op::AddOutput(1), Op::AddOutput(2)];
    let mut layout = check_ops(ops);
    let seed_id = layout.active_activity_id();
    // M0 is active (first output added). Confirm we have two monitors.
    assert_eq!(layout.monitors.len(), 2, "fixture: two monitors required");
    let mon0_out = layout.monitors[0].output_id();
    let mon1_out = layout.monitors[1].output_id();

    let beta_id = layout.create_activity("Beta".to_owned()).expect("create");
    // Bootstrap beta's view on M1 by focusing M1, switching to beta (materializes beta's
    // view on M1), then switching back to seed so beta's view on M1 is dormant.
    layout.focus_output(&layout.monitors[1].output.clone());
    layout.switch_activity(beta_id);
    layout.switch_activity_previous();
    assert_eq!(layout.active_activity_id(), seed_id);

    // Restore active monitor to M0 so active_monitor_idx != 1 during the rename.
    layout.focus_output(&layout.monitors[0].output.clone());
    assert_eq!(
        layout.active_monitor_idx, 0,
        "fixture: active monitor is M0 (idx 0)",
    );

    // w_target lives only in beta's dormant view on M1 — NOT in seed's view on M0 or M1.
    let beta_view_m1_before = layout
        .activities
        .get(beta_id)
        .expect("beta live")
        .views()
        .get(&mon1_out)
        .expect("beta view on M1 materialized")
        .ids()
        .to_vec();
    let w_target_id = *beta_view_m1_before
        .last()
        .expect("beta's M1 view holds at least the bookend");
    assert!(
        !layout.active_view(&mon0_out).ids().contains(&w_target_id),
        "fixture: w_target not in seed's M0 active view",
    );
    assert!(
        !layout.active_view(&mon1_out).ids().contains(&w_target_id),
        "fixture: w_target not in seed's M1 active view",
    );
    let beta_len_m1_before = beta_view_m1_before.len();

    layout.set_workspace_name(
        "named".to_owned(),
        Some(WorkspaceReference::Id(w_target_id.get())),
    );

    // Beta's dormant view on M1 grew by exactly one fresh trailing empty after w_target.
    let beta_view_m1_after = layout
        .activities
        .get(beta_id)
        .expect("beta live")
        .views()
        .get(&mon1_out)
        .expect("beta view on M1 live")
        .ids()
        .to_vec();
    assert_eq!(
        beta_view_m1_after.len(),
        beta_len_m1_before + 1,
        "beta's M1 dormant view grew by exactly one fresh trailing empty",
    );
    assert_eq!(
        beta_view_m1_after[beta_len_m1_before - 1],
        w_target_id,
        "w_target stays at its old position; a fresh empty follows it",
    );
    let new_trailing_id = *beta_view_m1_after.last().expect("trailing slot exists");
    let new_trailing_ws = layout
        .workspaces
        .get(&new_trailing_id)
        .expect("trailing id is a pool key");
    assert!(
        new_trailing_ws.name.is_none(),
        "fresh trailing empty must be unnamed",
    );
    assert!(
        !new_trailing_ws.has_windows(),
        "fresh trailing empty must hold no windows",
    );
    // w_target itself now carries the new name.
    assert_eq!(
        layout
            .workspaces
            .get(&w_target_id)
            .expect("w_target live")
            .name
            .as_deref(),
        Some("named"),
        "w_target now carries the new name",
    );
    // Seed's M0 active view does not contain w_target (dormant-only on M1).
    assert!(
        !layout.active_view(&mon0_out).ids().contains(&w_target_id),
        "seed's M0 active view still does not contain w_target",
    );

    layout.verify_invariants();
}

#[test]
fn switch_activity_focus_follows_active_activity_view() {
    // `Layout::focus()` reads the active activity's view for the active monitor and
    // selects the window via that workspace's own active_column_idx / active_tile_idx.
    // Pin that the read path naturally tracks the active activity after a switch — no
    // explicit activate_workspace / focus-poke from switch_activity is required.
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
    ];
    let mut layout = check_ops(ops);
    let seed_id = layout.active_activity_id();

    // Seed starts focused on window 1.
    assert_eq!(layout.focus().map(|w| *w.id()), Some(1));

    let beta = super::activity::Activity::new_runtime("beta".to_owned());
    let beta_id = beta.id();
    test_insert_activity(&mut layout, beta);

    layout.switch_activity(beta_id);
    assert_eq!(layout.active_activity_id(), beta_id);
    // Beta bootstrapped with a fresh empty workspace on this output, so no window.
    assert!(
        layout.focus().is_none(),
        "beta has no windows; focus read must be None without any activate_workspace call",
    );
    layout.verify_invariants();

    layout.switch_activity_previous();
    assert_eq!(layout.active_activity_id(), seed_id);
    assert_eq!(
        layout.focus().map(|w| *w.id()),
        Some(1),
        "focus returns to the seed's window via the read path — no explicit refocus needed",
    );
    layout.verify_invariants();
}

#[test]
fn switch_activity_preserves_active_monitor_idx_in_range() {
    // Multi-output: make output2 the active monitor, switch activities, and confirm
    // active_monitor_idx is preserved and still in range. Pins the post-condition end-to-end.
    let ops = [Op::AddOutput(1), Op::AddOutput(2), Op::FocusOutput(2)];
    let mut layout = check_ops(ops);
    let mon_idx_before = layout.active_monitor_idx;
    assert!(mon_idx_before < layout.monitors.len());

    let beta = super::activity::Activity::new_runtime("beta".to_owned());
    let beta_id = beta.id();
    test_insert_activity(&mut layout, beta);

    layout.switch_activity(beta_id);

    assert_eq!(
        layout.active_monitor_idx, mon_idx_before,
        "switch_activity must not mutate active_monitor_idx",
    );
    assert!(
        layout.active_monitor_idx < layout.monitors.len(),
        "active_monitor_idx must remain in range after the switch",
    );
    layout.verify_invariants();
}

#[test]
fn switch_activity_view_active_remains_in_ids_after_switch() {
    // Seed with ≥2 workspaces on one output, activate the second workspace so
    // view.active is off position 0, round-trip through beta, and confirm seed's
    // view.active is still in view.ids (the post-condition pin that verify_invariants
    // alone doesn't check, since WorkspaceView enforces it structurally).
    let ops = [
        Op::AddOutput(1),
        Op::AddNamedWorkspace {
            ws_name: 1,
            output_name: None,
            layout_config: None,
        },
    ];
    let mut layout = check_ops(ops);
    let seed_id = layout.active_activity_id();
    let out_id = layout.monitors[0].output_id();

    // Expect ≥2 ids in the seed's view (the named ws and the default trailing empty).
    assert!(
        layout.active_view(&out_id).ids().len() >= 2,
        "setup expects a seed view with multiple workspaces",
    );

    // Move off position 0 so view.active is a non-first id — the drift-exposing shape
    // for the post-condition walk.
    layout.switch_workspace(1);
    let seed_active_before = layout.active_view(&out_id).active();
    assert_ne!(
        seed_active_before,
        layout.active_view(&out_id).ids()[0],
        "switch_workspace(1) must have moved view.active off position 0",
    );

    let beta = super::activity::Activity::new_runtime("beta".to_owned());
    let beta_id = beta.id();
    test_insert_activity(&mut layout, beta);

    layout.switch_activity(beta_id);
    layout.verify_invariants();

    layout.switch_activity(seed_id);
    layout.verify_invariants();

    let seed_view_after = &layout.activities.active().views()[&out_id];
    assert!(
        seed_view_after.ids().contains(&seed_view_after.active()),
        "view.active must remain a member of view.ids after the round-trip switch",
    );
    assert_eq!(
        seed_view_after.active(),
        seed_active_before,
        "round-trip switch preserves seed's previously-active workspace id",
    );
    // active_position() .expects the membership relation — resolving without panic
    // is the direct pin of the post-condition.
    let _ = seed_view_after.active_position();
}

#[test]
fn is_activity_switch_hard_blocked_returns_some_during_interactive_move() {
    // Arrange a real interactive move so the field is `Some(_)` via the public API
    // path rather than constructing the private InteractiveMoveState directly.
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(0),
        },
    ];
    let mut layout = check_ops(ops);

    assert!(layout.is_activity_switch_hard_blocked().is_none());

    let output = layout
        .outputs()
        .find(|o| o.name() == "output1")
        .cloned()
        .expect("output1 must exist after AddOutput(1)");
    let armed = layout.interactive_move_begin(0, &output, Point::from((0., 0.)));
    assert!(armed, "interactive_move_begin must arm the move");

    assert_eq!(
        layout.is_activity_switch_hard_blocked(),
        Some(super::ActivitySwitchBlock::InteractiveMove),
    );

    layout.interactive_move_end(&0);
    assert!(layout.is_activity_switch_hard_blocked().is_none());
}

#[test]
fn is_activity_switch_hard_blocked_returns_some_during_dnd() {
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);

    assert!(layout.is_activity_switch_hard_blocked().is_none());

    let output = layout
        .outputs()
        .find(|o| o.name() == "output1")
        .cloned()
        .expect("output1 must exist after AddOutput(1)");
    layout.dnd_update(output, Point::from((0., 0.)));

    assert_eq!(
        layout.is_activity_switch_hard_blocked(),
        Some(super::ActivitySwitchBlock::Dnd),
    );

    layout.dnd_end();
    assert!(layout.is_activity_switch_hard_blocked().is_none());
}

#[test]
fn is_activity_switch_hard_blocked_returns_some_during_workspace_switch_gesture_on_any_monitor() {
    // Two monitors: arm the gesture on the SECOND monitor (output2) so a future
    // regression that hard-codes monitors[0] in the reader fails this test.
    let ops = [Op::AddOutput(1), Op::AddOutput(2)];
    let mut layout = check_ops(ops);

    assert!(layout.is_activity_switch_hard_blocked().is_none());

    let output2 = layout
        .outputs()
        .find(|o| o.name() == "output2")
        .cloned()
        .expect("output2 must exist after AddOutput(2)");
    layout.workspace_switch_gesture_begin(&output2, true);

    // Canary load-bearing: the gesture must land on a non-zero index so a
    // monitors[0]-hardcoded reader fails the assertion below.
    assert!(layout.monitors[0].workspace_switch.is_none());
    assert!(matches!(
        layout.monitors[1].workspace_switch,
        Some(super::monitor::WorkspaceSwitch::Gesture(_)),
    ));

    assert_eq!(
        layout.is_activity_switch_hard_blocked(),
        Some(super::ActivitySwitchBlock::WorkspaceSwitchGesture),
    );

    layout.workspace_switch_gesture_end(Some(true));
    assert!(layout.is_activity_switch_hard_blocked().is_none());
}

#[test]
fn switch_activity_snaps_in_flight_animation_and_proceeds() {
    // Single monitor + a named workspace gives the seed view ≥2 ids, so
    // `switch_workspace(1)` arms a `WorkspaceSwitch::Animation`. Then assert
    // the activity switch (a) is not hard-blocked, (b) does not panic, (c)
    // snaps the animation to None on every monitor, (d) flips to beta.
    let ops = [
        Op::AddOutput(1),
        Op::AddNamedWorkspace {
            ws_name: 1,
            output_name: None,
            layout_config: None,
        },
    ];
    let mut layout = check_ops(ops);
    let seed_id = layout.active_activity_id();
    let out_id = layout.monitors[0].output_id();
    // The named-workspace setup leaves active at position 1 (the named ws), with the
    // empty trailing ws at position 0 — switching to a different position arms the
    // animation. Pick whichever non-active position exists.
    let target_pos = if layout.active_view(&out_id).active_position() == 0 {
        1
    } else {
        0
    };
    layout.switch_workspace(target_pos);
    assert!(
        matches!(
            layout.monitors[0].workspace_switch,
            Some(super::monitor::WorkspaceSwitch::Animation(_)),
        ),
        "switch_workspace(target_pos) must arm a WorkspaceSwitch::Animation",
    );
    // Animation is NOT a hard block — pin the reader contract.
    assert!(layout.is_activity_switch_hard_blocked().is_none());

    let beta = super::activity::Activity::new_runtime("beta".to_owned());
    let beta_id = beta.id();
    test_insert_activity(&mut layout, beta);

    layout.switch_activity(beta_id);

    for mon in &layout.monitors {
        assert!(
            mon.workspace_switch.is_none(),
            "switch_activity must clear in-flight WorkspaceSwitch on every monitor",
        );
    }
    assert_eq!(layout.active_activity_id(), beta_id);
    assert_eq!(layout.activities.previous_id(), Some(seed_id));
    layout.verify_invariants();
}

#[test]
fn activity_switch_block_display_matches_wire_contract() {
    // Pins the `Display` tokens that form the token half of the IPC wire
    // contract. The full envelope string is assembled by
    // `format_activity_switch_block_err` and pinned by
    // `activity_switch_block_err_envelope_matches_wire_contract` below and the
    // serde roundtrip in `jiji-ipc`. A change to either layer's strings without
    // updating the other will be caught by one of these tests.
    use super::ActivitySwitchBlock;
    assert_eq!(
        format!("{}", ActivitySwitchBlock::InteractiveMove),
        "interactive window move",
    );
    assert_eq!(format!("{}", ActivitySwitchBlock::Dnd), "drag and drop");
    assert_eq!(
        format!("{}", ActivitySwitchBlock::WorkspaceSwitchGesture),
        "workspace switch gesture",
    );
}

#[test]
fn activity_switch_block_err_envelope_matches_wire_contract() {
    // Pins the full IPC wire error string produced by
    // `format_activity_switch_block_err`. Both `ipc/server.rs` and this test
    // call that helper, so a regression to its format string will fail here.
    // A change to the prefix or suffix wording will be caught by this test;
    // a change to a token string will be caught by the token-level test above.
    use super::{format_activity_switch_block_err, ActivitySwitchBlock};
    for (block, expected) in [
        (
            ActivitySwitchBlock::InteractiveMove,
            "activity switch blocked: interactive window move",
        ),
        (
            ActivitySwitchBlock::Dnd,
            "activity switch blocked: drag and drop",
        ),
        (
            ActivitySwitchBlock::WorkspaceSwitchGesture,
            "activity switch blocked: workspace switch gesture",
        ),
    ] {
        assert_eq!(format_activity_switch_block_err(block), expected);
    }
}

#[test]
fn do_action_error_display_matches_wire_contract() {
    // Pins the `Display` tokens for `DoActionError`. The full envelope is
    // assembled by `format_do_action_error` and pinned by
    // `do_action_error_envelope_matches_wire_contract` below; the serde
    // roundtrip `reply_err_format_for_window_not_found` in `jiji-ipc` pins
    // the wire JSON. A change to any token without updating all three
    // layers will be caught here.
    use super::{ActivitySwitchBlock, DoActionError};
    // Delegated tokens — must match the `ActivitySwitchBlock::Display` strings
    // exactly (byte-identity is load-bearing for the envelope).
    assert_eq!(
        format!(
            "{}",
            DoActionError::ActivitySwitchBlocked(ActivitySwitchBlock::InteractiveMove)
        ),
        "interactive window move",
    );
    assert_eq!(
        format!(
            "{}",
            DoActionError::ActivitySwitchBlocked(ActivitySwitchBlock::Dnd)
        ),
        "drag and drop",
    );
    assert_eq!(
        format!(
            "{}",
            DoActionError::ActivitySwitchBlocked(ActivitySwitchBlock::WorkspaceSwitchGesture)
        ),
        "workspace switch gesture",
    );
    // New token for the wire contract.
    assert_eq!(
        format!("{}", DoActionError::WindowNotFound { id: 42 }),
        "window not found: id=42",
    );
    // Workspace-activity assignment tokens — plain lowercase, no
    // payload interpolation. Outer variants delegate to the wrapped inner
    // enum's `Display`; the envelope test confirms byte-identity and
    // disambiguates which clause is load-bearing.
    use super::{
        AddWorkspaceToActivityError, CreateActivityError, MoveWorkspaceToActivityError,
        RemoveActivityError, RemoveWorkspaceFromActivityError, RenameActivityError,
        SetWorkspaceActivitiesError, SetWorkspaceStickyError, SwitchActivityError,
        ToggleWorkspaceStickyError, UnsetWorkspaceStickyError,
    };
    assert_eq!(
        format!(
            "{}",
            DoActionError::AddWorkspaceToActivity(AddWorkspaceToActivityError::ActivityNotFound)
        ),
        "activity not found",
    );
    assert_eq!(
        format!(
            "{}",
            DoActionError::AddWorkspaceToActivity(AddWorkspaceToActivityError::WorkspaceNotFound)
        ),
        "workspace not found",
    );
    assert_eq!(
        format!(
            "{}",
            DoActionError::RemoveWorkspaceFromActivity(
                RemoveWorkspaceFromActivityError::ActivityNotFound
            )
        ),
        "activity not found",
    );
    assert_eq!(
        format!(
            "{}",
            DoActionError::RemoveWorkspaceFromActivity(
                RemoveWorkspaceFromActivityError::WorkspaceNotFound
            )
        ),
        "workspace not found",
    );
    assert_eq!(
        format!(
            "{}",
            DoActionError::RemoveWorkspaceFromActivity(
                RemoveWorkspaceFromActivityError::LastActivity
            )
        ),
        "workspace would be left with no activities",
    );
    // SetWorkspaceActivities tokens. ActivityNotFound shares
    // text with Add / Remove — byte-identity pins the shared row.
    assert_eq!(
        format!(
            "{}",
            DoActionError::SetWorkspaceActivities(SetWorkspaceActivitiesError::ActivityNotFound)
        ),
        "activity not found",
    );
    assert_eq!(
        format!(
            "{}",
            DoActionError::SetWorkspaceActivities(SetWorkspaceActivitiesError::EmptyActivityList)
        ),
        "activities list is empty",
    );
    // MoveWorkspaceToActivity tokens. ActivityNotFound shares
    // text with the other rows; the workspace-not-in-active text is
    // the wire-contract wording, minus the "use Add…" suggestion
    // (docstring concern, not wire token).
    assert_eq!(
        format!(
            "{}",
            DoActionError::MoveWorkspaceToActivity(MoveWorkspaceToActivityError::ActivityNotFound)
        ),
        "activity not found",
    );
    assert_eq!(
        format!(
            "{}",
            DoActionError::MoveWorkspaceToActivity(
                MoveWorkspaceToActivityError::WorkspaceNotInActiveActivity
            )
        ),
        "workspace not in active activity",
    );
    // Newly wire-surfaced workspace-miss rows for Set / Move (the silent
    // intercepts on these actions were dropped to harmonize the
    // workspace-miss contract with Add / Remove).
    assert_eq!(
        format!(
            "{}",
            DoActionError::SetWorkspaceActivities(SetWorkspaceActivitiesError::WorkspaceNotFound)
        ),
        "workspace not found",
    );
    assert_eq!(
        format!(
            "{}",
            DoActionError::MoveWorkspaceToActivity(MoveWorkspaceToActivityError::WorkspaceNotFound)
        ),
        "workspace not found",
    );
    // Activity-pool action tokens (CreateActivity / RemoveActivity /
    // RenameActivity / SwitchActivity).
    assert_eq!(
        format!(
            "{}",
            DoActionError::CreateActivity(CreateActivityError::EmptyName)
        ),
        "activity name must not be empty",
    );
    assert_eq!(
        format!(
            "{}",
            DoActionError::CreateActivity(CreateActivityError::DuplicateName)
        ),
        "activity name already exists",
    );
    assert_eq!(
        format!(
            "{}",
            DoActionError::RemoveActivity(RemoveActivityError::NotFound)
        ),
        "activity not found",
    );
    assert_eq!(
        format!(
            "{}",
            DoActionError::RemoveActivity(RemoveActivityError::ConfigDeclared)
        ),
        "activity is config-declared; edit config and reload to remove",
    );
    assert_eq!(
        format!(
            "{}",
            DoActionError::RemoveActivity(RemoveActivityError::LastRemaining)
        ),
        "cannot remove the last remaining activity",
    );
    assert_eq!(
        format!(
            "{}",
            DoActionError::RemoveActivity(RemoveActivityError::ExclusiveWorkspaceHasWindows)
        ),
        "activity owns an exclusive workspace with windows; close or move them first",
    );
    assert_eq!(
        format!(
            "{}",
            DoActionError::RemoveActivity(RemoveActivityError::ExclusiveNamedWorkspace)
        ),
        "activity owns a named exclusive workspace (even if empty); unname it first",
    );
    assert_eq!(
        format!(
            "{}",
            DoActionError::RenameActivity(RenameActivityError::NotFound)
        ),
        "activity not found",
    );
    assert_eq!(
        format!(
            "{}",
            DoActionError::RenameActivity(RenameActivityError::ConfigDeclared)
        ),
        "activity is config-declared; edit config and reload to rename",
    );
    assert_eq!(
        format!(
            "{}",
            DoActionError::RenameActivity(RenameActivityError::EmptyName)
        ),
        "activity name must not be empty",
    );
    assert_eq!(
        format!(
            "{}",
            DoActionError::RenameActivity(RenameActivityError::DuplicateName)
        ),
        "activity name already exists",
    );
    assert_eq!(
        format!(
            "{}",
            DoActionError::SwitchActivity(SwitchActivityError::NotFound)
        ),
        "activity not found",
    );
    // Sticky cohort tokens. Each enum has a single `WorkspaceNotFound`
    // variant; the dispatch arm now surfaces it instead of silently
    // returning `Ok(())`.
    assert_eq!(
        format!(
            "{}",
            DoActionError::ToggleWorkspaceSticky(ToggleWorkspaceStickyError::WorkspaceNotFound)
        ),
        "workspace not found",
    );
    assert_eq!(
        format!(
            "{}",
            DoActionError::SetWorkspaceSticky(SetWorkspaceStickyError::WorkspaceNotFound)
        ),
        "workspace not found",
    );
    assert_eq!(
        format!(
            "{}",
            DoActionError::UnsetWorkspaceSticky(UnsetWorkspaceStickyError::WorkspaceNotFound)
        ),
        "workspace not found",
    );
}

#[test]
fn do_action_error_envelope_matches_wire_contract() {
    // Pins the full IPC wire envelopes produced by `format_do_action_error`.
    //
    // The `ActivitySwitchBlocked` cases re-assert the three envelopes
    // already pinned by `activity_switch_block_err_envelope_matches_wire_contract`
    // — this is a deliberate regression guard against byte-identity drift if
    // `format_do_action_error` ever stops delegating to
    // `format_activity_switch_block_err`.
    //
    // The `WindowNotFound` case pins the new envelope.
    use super::{
        format_do_action_error, ActivitySwitchBlock, AddWorkspaceToActivityError,
        CreateActivityError, DoActionError, MoveWorkspaceToActivityError, RemoveActivityError,
        RemoveWorkspaceFromActivityError, RenameActivityError, SetWorkspaceActivitiesError,
        SetWorkspaceStickyError, SwitchActivityError, ToggleWorkspaceStickyError,
        UnsetWorkspaceStickyError,
    };
    for (err, expected) in [
        (
            DoActionError::ActivitySwitchBlocked(ActivitySwitchBlock::InteractiveMove),
            "activity switch blocked: interactive window move",
        ),
        (
            DoActionError::ActivitySwitchBlocked(ActivitySwitchBlock::Dnd),
            "activity switch blocked: drag and drop",
        ),
        (
            DoActionError::ActivitySwitchBlocked(ActivitySwitchBlock::WorkspaceSwitchGesture),
            "activity switch blocked: workspace switch gesture",
        ),
        (
            DoActionError::WindowNotFound { id: 42 },
            "window not found: id=42",
        ),
        (
            DoActionError::WindowNotFound { id: 0 },
            "window not found: id=0",
        ),
        (
            DoActionError::AddWorkspaceToActivity(AddWorkspaceToActivityError::ActivityNotFound),
            "activity not found",
        ),
        (
            DoActionError::AddWorkspaceToActivity(AddWorkspaceToActivityError::WorkspaceNotFound),
            "workspace not found",
        ),
        (
            DoActionError::RemoveWorkspaceFromActivity(
                RemoveWorkspaceFromActivityError::ActivityNotFound,
            ),
            "activity not found",
        ),
        (
            DoActionError::RemoveWorkspaceFromActivity(
                RemoveWorkspaceFromActivityError::WorkspaceNotFound,
            ),
            "workspace not found",
        ),
        (
            DoActionError::RemoveWorkspaceFromActivity(
                RemoveWorkspaceFromActivityError::LastActivity,
            ),
            "workspace would be left with no activities",
        ),
        (
            DoActionError::SetWorkspaceActivities(SetWorkspaceActivitiesError::ActivityNotFound),
            "activity not found",
        ),
        (
            DoActionError::SetWorkspaceActivities(SetWorkspaceActivitiesError::EmptyActivityList),
            "activities list is empty",
        ),
        (
            DoActionError::MoveWorkspaceToActivity(MoveWorkspaceToActivityError::ActivityNotFound),
            "activity not found",
        ),
        (
            DoActionError::MoveWorkspaceToActivity(
                MoveWorkspaceToActivityError::WorkspaceNotInActiveActivity,
            ),
            "workspace not in active activity",
        ),
        // Newly wire-surfaced workspace-miss rows for Set / Move.
        (
            DoActionError::SetWorkspaceActivities(SetWorkspaceActivitiesError::WorkspaceNotFound),
            "workspace not found",
        ),
        (
            DoActionError::MoveWorkspaceToActivity(MoveWorkspaceToActivityError::WorkspaceNotFound),
            "workspace not found",
        ),
        // CreateActivity rows.
        (
            DoActionError::CreateActivity(CreateActivityError::EmptyName),
            "activity name must not be empty",
        ),
        (
            DoActionError::CreateActivity(CreateActivityError::DuplicateName),
            "activity name already exists",
        ),
        // RemoveActivity rows.
        (
            DoActionError::RemoveActivity(RemoveActivityError::NotFound),
            "activity not found",
        ),
        (
            DoActionError::RemoveActivity(RemoveActivityError::ConfigDeclared),
            "activity is config-declared; edit config and reload to remove",
        ),
        (
            DoActionError::RemoveActivity(RemoveActivityError::LastRemaining),
            "cannot remove the last remaining activity",
        ),
        (
            DoActionError::RemoveActivity(RemoveActivityError::ExclusiveWorkspaceHasWindows),
            "activity owns an exclusive workspace with windows; close or move them first",
        ),
        (
            DoActionError::RemoveActivity(RemoveActivityError::ExclusiveNamedWorkspace),
            "activity owns a named exclusive workspace (even if empty); unname it first",
        ),
        // RenameActivity rows.
        (
            DoActionError::RenameActivity(RenameActivityError::NotFound),
            "activity not found",
        ),
        (
            DoActionError::RenameActivity(RenameActivityError::ConfigDeclared),
            "activity is config-declared; edit config and reload to rename",
        ),
        (
            DoActionError::RenameActivity(RenameActivityError::EmptyName),
            "activity name must not be empty",
        ),
        (
            DoActionError::RenameActivity(RenameActivityError::DuplicateName),
            "activity name already exists",
        ),
        // SwitchActivity row.
        (
            DoActionError::SwitchActivity(SwitchActivityError::NotFound),
            "activity not found",
        ),
        // Sticky cohort rows.
        (
            DoActionError::ToggleWorkspaceSticky(ToggleWorkspaceStickyError::WorkspaceNotFound),
            "workspace not found",
        ),
        (
            DoActionError::SetWorkspaceSticky(SetWorkspaceStickyError::WorkspaceNotFound),
            "workspace not found",
        ),
        (
            DoActionError::UnsetWorkspaceSticky(UnsetWorkspaceStickyError::WorkspaceNotFound),
            "workspace not found",
        ),
    ] {
        assert_eq!(format_do_action_error(err), expected);
    }
}

#[test]
fn resolve_activity_ref_by_id_and_name() {
    let layout = Layout::<TestWindow>::default();
    let seed_id = layout.active_activity_id();
    let seed_name = layout.activities.active().name().to_owned();

    assert_eq!(
        layout.resolve_activity_ref(&ActivityReferenceArg::Id(seed_id.get())),
        Some(seed_id),
    );
    assert_eq!(
        layout.resolve_activity_ref(&ActivityReferenceArg::Name(seed_name)),
        Some(seed_id),
    );

    assert_eq!(
        layout.resolve_activity_ref(&ActivityReferenceArg::Id(u64::MAX)),
        None,
    );
    assert_eq!(
        layout.resolve_activity_ref(&ActivityReferenceArg::Name("does-not-exist".into())),
        None,
    );
}

#[test]
fn resolve_workspace_id_by_raw_returns_pool_match_only() {
    // A live pool entry resolves; an unminted raw u64 does not.
    let layout = check_ops([Op::AddOutput(1)]);
    let seed_id = layout
        .workspaces
        .keys()
        .next()
        .copied()
        .expect("AddOutput must have seeded at least one workspace");

    assert_eq!(
        layout.resolve_workspace_id(seed_id.get()),
        Some(seed_id),
        "live pool match must round-trip through resolve_workspace_id",
    );
    assert_eq!(
        layout.resolve_workspace_id(u64::MAX),
        None,
        "u64::MAX (never minted) must not match any pool entry",
    );
}

#[test]
fn resolve_workspace_id_finds_dormant_activity_workspace() {
    // This state (workspace in pool, no view binding, no disconnected entry)
    // is not reachable through normal `Op`s; the manual `workspaces.insert`
    // deliberately bypasses `verify_invariants` to pin the resolver/finder
    // scope asymmetry. Do not promote this fixture to a shared baseline.
    //
    // Load-bearing pin for the architectural choice between full-pool and
    // active-view-scoped resolver. A workspace that belongs exclusively to a
    // dormant activity (Beta) must be visible to `resolve_workspace_id`
    // (canonical pool), but invisible to `find_workspace_by_id` (active-view
    // + disconnected-pool scoped). The two-filter chain at
    // `find_output_and_workspace_index` therefore returns `None` end-to-end —
    // the resolver alone is not sufficient to act on a dormant workspace.
    let mut layout = check_ops([Op::AddOutput(1)]);
    let beta_id = layout
        .create_activity("Beta".to_owned())
        .expect("create beta must succeed");

    // Mint a workspace whose `activities` set is exactly `{Beta}` — never
    // tagged with Alpha, so it cannot appear in Alpha's view. The
    // workspace stays unbound (no output), which is what makes it
    // dormant from the active-activity's perspective even though the
    // pool entry exists.
    let dormant_id = {
        let ws = Workspace::<TestWindow>::new_no_outputs(
            HashSet::from([beta_id]),
            layout.clock.clone(),
            layout.options.clone(),
        );
        let id = ws.id();
        layout.workspaces.insert(id, ws);
        id
    };

    // Pool-walk resolver sees the dormant entry.
    assert_eq!(
        layout.resolve_workspace_id(dormant_id.get()),
        Some(dormant_id),
        "resolve_workspace_id must see workspaces on dormant activities",
    );

    // Active-view-scoped finder does NOT see it: Alpha (current) has no
    // view entry for `dormant_id`, and the workspace is not in
    // `disconnected_workspace_ids` (the active monitor is connected).
    assert!(
        layout.find_workspace_by_id(dormant_id).is_none(),
        "find_workspace_by_id must be blind to workspaces exclusive to a dormant activity",
    );

    // Compose the two filters as `find_output_and_workspace_index` does
    // (the consumer at `Niri::find_output_and_workspace_index` is not
    // reachable from this test module — pin the unit-level composition
    // here): `resolve_workspace_id` succeeds, `find_workspace_by_id`
    // fails, the chain yields `None`.
    let chain = layout
        .resolve_workspace_id(dormant_id.get())
        .and_then(|id| layout.find_workspace_by_id(id));
    assert!(
        chain.is_none(),
        "two-filter chain must propagate `None` once active-view scope is enforced",
    );
}

#[test]
fn workspaces_all_covers_pool_including_disconnected() {
    // Remove the only output so both named workspaces land in
    // `disconnected_workspace_ids`. `workspaces_all` must still yield them,
    // since it is pool-driven and ignores monitor/view structure.
    let ops = [
        Op::AddOutput(1),
        Op::AddNamedWorkspace {
            ws_name: 1,
            output_name: None,
            layout_config: None,
        },
        Op::AddNamedWorkspace {
            ws_name: 2,
            output_name: None,
            layout_config: None,
        },
        Op::RemoveOutput(1),
    ];
    let layout = check_ops(ops);

    assert!(layout.monitors.is_empty());
    assert_eq!(layout.disconnected_workspace_ids.len(), 2);

    let pool_count = layout.workspaces.len();
    let walked: Vec<_> = layout.workspaces_all().collect();
    assert_eq!(
        walked.len(),
        pool_count,
        "workspaces_all must yield every pool entry exactly once",
    );

    // The disconnected_workspace_ids set must be fully covered, regardless
    // of what affinity `output_id` each disconnected workspace still
    // carries (unbind_output does not clear the id — it is preserved as a
    // reconnect hint).
    let seen_ids: HashSet<WorkspaceId> = walked.iter().map(|(_, ws)| ws.id()).collect();
    for id in &layout.disconnected_workspace_ids {
        assert!(
            seen_ids.contains(id),
            "workspaces_all missed disconnected workspace {id:?}",
        );
    }

    // Every tuple's output_id must equal the workspace's own output_id —
    // `workspaces_all` is a faithful mirror of `Workspace::output_id`.
    for (output_id, ws) in &walked {
        assert_eq!(*output_id, ws.output_id());
    }
}

#[test]
fn workspaces_all_output_id_matches_workspace() {
    // Two connected outputs + no manual removal: every workspace is bound.
    // `workspaces_all`'s first tuple element must exactly mirror
    // `Workspace::output_id` for every yielded entry.
    let ops = [Op::AddOutput(1), Op::AddOutput(2)];
    let layout = check_ops(ops);

    assert!(!layout.monitors.is_empty());
    let pool_count = layout.workspaces.len();
    let mut yielded = 0;
    for (output_id, ws) in layout.workspaces_all() {
        assert_eq!(
            output_id,
            ws.output_id(),
            "workspaces_all output_id must mirror Workspace::output_id for id={:?}",
            ws.id(),
        );
        yielded += 1;
    }
    assert_eq!(yielded, pool_count);
}

#[test]
fn workspaces_with_activity_filters_by_membership() {
    // Two workspaces on a single output, both initially stamped with alpha.
    // Re-stamp exactly one with beta-only so alpha_ids is non-empty after
    // the mutation; the vacuous-assertion trap from having a single workspace
    // is avoided because alpha_ws_id stays alpha-only.
    //
    // Assertions:
    //   alpha filter: exact set {alpha_ws_id} — no more, no less.
    //   beta filter: exact set {beta_ws_id}.
    let ops = [
        Op::AddOutput(1),
        Op::AddNamedWorkspace {
            ws_name: 1,
            output_name: None,
            layout_config: None,
        },
    ];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let mon_out = layout.monitors[0].output_id();

    // Capture every alpha-tagged workspace id on this output before any mutation. We then
    // re-stamp exactly one to be beta-only so the alpha filter has a known size.
    let mut on_output: Vec<WorkspaceId> = layout
        .workspaces
        .values()
        .filter(|ws| ws.output_id() == Some(&mon_out) && ws.activities().contains(&alpha))
        .map(|ws| ws.id())
        .collect();
    on_output.sort_by_key(|id| id.get());
    assert!(
        on_output.len() >= 2,
        "need at least two alpha-tagged workspaces on the output for a non-vacuous test"
    );

    // Mint a distinct activity and install it in the pool. The materializer mints one
    // fresh empty-bookend workspace for beta on this monitor as a side effect — we account
    // for it in the beta filter assertion below.
    let beta_activity = super::activity::Activity::new_runtime("beta".to_owned());
    let beta = beta_activity.id();
    test_insert_activity(&mut layout, beta_activity);
    let beta_materialized_bookend_id = layout
        .activities()
        .get(beta)
        .expect("beta live")
        .views()
        .get(&mon_out)
        .expect("materializer installed beta's view")
        .ids()[0];

    // Stamp the first alpha-tagged workspace (lowest id) with beta-only; leave the rest as
    // alpha-only. Direct field mutation is legal here: tests live in the `super` module and
    // `Workspace::activities` is `pub(super)`.
    let beta_ws_id = on_output[0];
    let expected_alpha_ids: HashSet<WorkspaceId> = on_output[1..].iter().copied().collect();
    layout
        .workspaces
        .get_mut(&beta_ws_id)
        .expect("beta_ws_id must be a pool key")
        .activities = std::iter::once(beta).collect();

    let alpha_ids: HashSet<WorkspaceId> = layout
        .workspaces_with_activity(alpha, &mon_out)
        .map(|ws| ws.id())
        .collect();
    assert_eq!(
        alpha_ids, expected_alpha_ids,
        "alpha filter must yield exactly the alpha-only workspaces",
    );

    let beta_ids: HashSet<WorkspaceId> = layout
        .workspaces_with_activity(beta, &mon_out)
        .map(|ws| ws.id())
        .collect();
    assert_eq!(
        beta_ids,
        HashSet::from([beta_ws_id, beta_materialized_bookend_id]),
        "beta filter on this output must include the beta-stamped workspace AND the \
         materializer's fresh bookend",
    );

    layout.verify_invariants();
}

#[test]
fn workspaces_with_activity_respects_output_filter() {
    // Two outputs; every pool workspace carries the seed activity. The
    // filter must partition strictly by `output_id` — a workspace bound to
    // output1 must not surface under the output2 query and vice versa.
    let ops = [Op::AddOutput(1), Op::AddOutput(2)];
    let layout = check_ops(ops);
    let seed = layout.active_activity_id();
    let out1 = layout.monitors[0].output_id();
    let out2 = layout.monitors[1].output_id();
    assert_ne!(out1, out2);

    let on_out1: HashSet<WorkspaceId> = layout
        .workspaces_with_activity(seed, &out1)
        .map(|ws| ws.id())
        .collect();
    let on_out2: HashSet<WorkspaceId> = layout
        .workspaces_with_activity(seed, &out2)
        .map(|ws| ws.id())
        .collect();

    assert!(!on_out1.is_empty(), "out1 filter must yield at least one");
    assert!(!on_out2.is_empty(), "out2 filter must yield at least one");
    assert!(
        on_out1.is_disjoint(&on_out2),
        "a workspace cannot be bound to two outputs simultaneously",
    );

    for (_, ws) in layout.workspaces_all() {
        if let Some(bound) = ws.output_id() {
            if *bound == out1 {
                assert!(on_out1.contains(&ws.id()));
                assert!(!on_out2.contains(&ws.id()));
            } else if *bound == out2 {
                assert!(on_out2.contains(&ws.id()));
                assert!(!on_out1.contains(&ws.id()));
            }
        }
    }
}

#[test]
fn workspaces_with_activity_includes_sticky() {
    // A sticky workspace whose `activities` set contains the query id must
    // surface through the filter — `is_sticky` is not a separate code path
    // in the helper, it is pure membership. The auto-expansion that keeps
    // sticky workspaces in every activity's set is tested elsewhere; here
    // we only pin the filter's behavior given a sticky+member workspace.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let seed = layout.active_activity_id();
    let mon_out = layout.monitors[0].output_id();

    let sticky_id = {
        let ws = layout
            .workspaces
            .values_mut()
            .find(|ws| ws.output_id() == Some(&mon_out))
            .expect("at least one workspace must be bound to the connected output");
        ws.is_sticky = true;
        ws.id()
    };

    let ids: HashSet<WorkspaceId> = layout
        .workspaces_with_activity(seed, &mon_out)
        .map(|ws| ws.id())
        .collect();
    assert!(
        ids.contains(&sticky_id),
        "sticky workspace with seed in its activity set must appear in the filter",
    );

    layout.verify_invariants();
}

#[test]
fn layout_activity_is_urgent_aggregates_workspace_urgency() {
    // Aggregation rule pin ( bubble: window → workspace → activity).
    // Build two activities that share a workspace and one activity that does
    // not; flip window urgency on the shared workspace and assert the two
    // sharing activities report urgent, while the unrelated one does not.
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
    ];
    let mut layout = check_ops(ops);
    let seed_id = layout.active_activity_id();

    // Mint and insert a second activity `beta` and tag the (single) workspace
    // with both seed and beta — a shared workspace.
    let beta_activity = super::activity::Activity::new_runtime("beta".to_owned());
    let beta_id = beta_activity.id();
    test_insert_activity(&mut layout, beta_activity);

    // Mint and insert a third activity `gamma` that does NOT get tagged on
    // any workspace — its aggregate urgency must remain `false` no matter
    // what happens on the shared workspace.
    let gamma_activity = super::activity::Activity::new_runtime("gamma".to_owned());
    let gamma_id = gamma_activity.id();
    test_insert_activity(&mut layout, gamma_activity);

    // Find the workspace holding the window and tag it with seed + beta.
    let ws_id = layout
        .workspaces
        .values()
        .find(|ws| ws.windows().any(|w| *w.id() == 1usize))
        .expect("workspace holding the test window must exist")
        .id();
    layout
        .workspaces
        .get_mut(&ws_id)
        .expect("ws_id must be a pool key")
        .activities = [seed_id, beta_id].into_iter().collect();

    // Initially no urgent windows → every activity reports `false`.
    assert!(!layout.activity_is_urgent(seed_id));
    assert!(!layout.activity_is_urgent(beta_id));
    assert!(!layout.activity_is_urgent(gamma_id));

    // Flip the window urgent → seed and beta aggregate to `true`; gamma
    // stays `false` (its membership set is empty).
    let win = layout
        .workspaces
        .get(&ws_id)
        .expect("ws_id must be a pool key")
        .windows()
        .next()
        .expect("workspace must have the window")
        .clone();
    win.set_urgent(true);

    assert!(
        layout.activity_is_urgent(seed_id),
        "seed activity shares the workspace with the urgent window",
    );
    assert!(
        layout.activity_is_urgent(beta_id),
        "beta activity shares the workspace with the urgent window",
    );
    assert!(
        !layout.activity_is_urgent(gamma_id),
        "gamma has no workspaces → aggregate stays false",
    );

    // Clear urgency → aggregate flips back symmetrically.
    win.set_urgent(false);
    assert!(!layout.activity_is_urgent(seed_id));
    assert!(!layout.activity_is_urgent(beta_id));
    assert!(!layout.activity_is_urgent(gamma_id));

    layout.verify_invariants();
}

#[test]
fn layout_activity_is_urgent_unknown_id_returns_false() {
    // Pin silent-no-match behavior: an ActivityId that was never inserted
    // must yield `false` rather than panicking. Mirrors the
    // `workspaces_with_activity_unknown_activity_yields_empty` precedent.
    let ops = [Op::AddOutput(1)];
    let layout = check_ops(ops);

    assert!(
        !layout.activity_is_urgent(super::activity::ActivityId::specific(99999)),
        "unknown activity id must yield false, not panic",
    );
}

#[test]
fn workspaces_with_activity_unknown_activity_yields_empty() {
    // An ActivityId that was never inserted into `layout.activities` must
    // yield an empty iterator — the docstring guarantees silent empty
    // rather than a panic for unknown ids.
    let ops = [Op::AddOutput(1)];
    let layout = check_ops(ops);

    // Mint a fresh activity but deliberately do NOT insert it; its id is
    // unknown to the layout's Activities map.
    let dead_activity = super::activity::Activity::new_runtime("dead".to_owned());
    let dead_id = dead_activity.id();
    // `dead_activity` is intentionally dropped without inserting.

    let out = layout.monitors[0].output_id();
    let results: Vec<_> = layout.workspaces_with_activity(dead_id, &out).collect();
    assert!(
        results.is_empty(),
        "unknown activity id must yield an empty iterator, got {} workspaces",
        results.len(),
    );
}

/// Shared active-window-id extractor for `TestWindow`. The IPC helper is
/// generic so production can pass `|win| win.id().get()` against `Mapped`
/// while tests pass this widened cast.
fn test_window_id_of(win: &TestWindow) -> u64 {
    *win.id() as u64
}

#[test]
fn ipc_workspace_snapshot_hidden_workspace_has_idx_zero_and_flag_false() {
    // A workspace whose activity set is disjoint from the active activity's
    // id must surface through pass 2 of the snapshot builder with the
    // Sentinel `idx: 0`, `is_in_active_activity: false`, and neither
    // is_active nor is_focused (pass 2 hardcodes both to false).
    let ops = [
        Op::AddOutput(1),
        Op::AddNamedWorkspace {
            ws_name: 1,
            output_name: None,
            layout_config: None,
        },
    ];
    let mut layout = check_ops(ops);
    let mon_out = layout.monitors[0].output_id();

    let mut on_output: Vec<WorkspaceId> = layout
        .workspaces
        .values()
        .filter(|ws| ws.output_id() == Some(&mon_out))
        .map(|ws| ws.id())
        .collect();
    on_output.sort_by_key(|id| id.get());
    assert!(
        on_output.len() >= 2,
        "need at least two workspaces on the output for a non-vacuous test",
    );

    let beta_activity = super::activity::Activity::new_runtime("beta".to_owned());
    let beta = beta_activity.id();
    test_insert_activity(&mut layout, beta_activity);

    // Stamp the lowest-id workspace with beta-only so it leaves the active
    // (seed) activity's membership.
    let hidden_ws_id = on_output[0];
    layout
        .workspaces
        .get_mut(&hidden_ws_id)
        .expect("hidden_ws_id must be a pool key")
        .activities = std::iter::once(beta).collect();

    let snapshot = crate::ipc::server::build_workspace_snapshot(&layout, None, test_window_id_of);

    let expected_output_name = layout.monitors[0].output_name().clone();

    let hidden = snapshot
        .iter()
        .find(|ws| ws.id == hidden_ws_id.get())
        .expect("hidden workspace must appear in the snapshot via pass 2");

    assert_eq!(hidden.idx, 0, "hidden workspace must have idx sentinel 0");
    assert!(
        !hidden.is_in_active_activity,
        "hidden workspace must have is_in_active_activity = false",
    );
    assert!(!hidden.is_active, "pass 2 must never emit is_active = true");
    assert!(
        !hidden.is_focused,
        "pass 2 must never emit is_focused = true",
    );
    assert_eq!(
        hidden.output,
        Some(expected_output_name),
        "hidden workspace on a connected output must have its output name resolved",
    );
    assert_eq!(
        hidden.activities,
        vec![beta.get()],
        "hidden workspace activities must list the owning activity id",
    );

    layout.verify_invariants();
}

#[test]
fn ipc_workspace_snapshot_active_activity_workspace_has_view_position_idx() {
    // Regression guard for pass 1: every workspace in the active activity
    // must appear with `idx = view_position + 1` and `is_in_active_activity
    // = true`, matching the pre-refactor behavior.
    let ops = [
        Op::AddOutput(1),
        Op::AddNamedWorkspace {
            ws_name: 1,
            output_name: None,
            layout_config: None,
        },
    ];
    let layout = check_ops(ops);
    let mon_out = layout.monitors[0].output_id();
    let view = layout.active_view(&mon_out);
    let view_ids: Vec<WorkspaceId> = view.ids().to_vec();
    let active_ws_id = view.active();
    assert!(
        view_ids.len() >= 2,
        "need at least two workspaces in the active view for a non-vacuous test",
    );

    // Pass a real focused workspace id so the is_focused branch is exercised.
    let focused_ws_id = view_ids[0].get();
    let snapshot = crate::ipc::server::build_workspace_snapshot(
        &layout,
        Some(focused_ws_id),
        test_window_id_of,
    );

    for (pos, id) in view_ids.iter().enumerate() {
        let expected_idx = u8::try_from(pos + 1).unwrap_or(u8::MAX);
        let ws = snapshot
            .iter()
            .find(|ws| ws.id == id.get())
            .unwrap_or_else(|| panic!("workspace {:?} must appear in the snapshot", id));
        assert_eq!(
            ws.idx, expected_idx,
            "pass 1 workspace at view position {pos} must have idx = {expected_idx}",
        );
        assert!(
            ws.is_in_active_activity,
            "pass 1 workspace must have is_in_active_activity = true",
        );
    }

    // Exactly one workspace must be active (the monitor's active workspace).
    let active_count = snapshot.iter().filter(|ws| ws.is_active).count();
    assert_eq!(
        active_count, 1,
        "exactly one workspace must be marked is_active in the snapshot",
    );
    let active_entry = snapshot
        .iter()
        .find(|ws| ws.is_active)
        .expect("exactly one active workspace was asserted above");
    assert_eq!(
        active_entry.id,
        active_ws_id.get(),
        "is_active must be set on the view's active workspace id",
    );

    // Exactly one workspace must be focused (the one we passed as focused_ws_id).
    let focused_count = snapshot.iter().filter(|ws| ws.is_focused).count();
    assert_eq!(
        focused_count, 1,
        "exactly one workspace must be marked is_focused in the snapshot",
    );
    let focused_entry = snapshot
        .iter()
        .find(|ws| ws.is_focused)
        .expect("exactly one focused workspace was asserted above");
    assert_eq!(
        focused_entry.id, focused_ws_id,
        "is_focused must be set on the workspace matching the supplied focused_ws_id",
    );
}

#[test]
fn ipc_workspace_snapshot_mixed_visibility_preserves_pass_disjointness() {
    // Seed a mix: one workspace alpha-only (pass 1, view-position idx), one
    // beta-only (pass 2, idx 0), one in both (pass 1 wins because it's in
    // the active activity). The two passes must together emit every pool
    // workspace exactly once — no duplicate ids, no omissions.
    let ops = [
        Op::AddOutput(1),
        Op::AddNamedWorkspace {
            ws_name: 1,
            output_name: None,
            layout_config: None,
        },
        Op::AddNamedWorkspace {
            ws_name: 2,
            output_name: None,
            layout_config: None,
        },
    ];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let mon_out = layout.monitors[0].output_id();

    let mut on_output: Vec<WorkspaceId> = layout
        .workspaces
        .values()
        .filter(|ws| ws.output_id() == Some(&mon_out))
        .map(|ws| ws.id())
        .collect();
    on_output.sort_by_key(|id| id.get());
    assert!(
        on_output.len() >= 3,
        "need at least three workspaces on the output for a disjointness test",
    );

    let beta_activity = super::activity::Activity::new_runtime("beta".to_owned());
    let beta = beta_activity.id();
    test_insert_activity(&mut layout, beta_activity);

    // ws_b: beta-only (hidden, pass 2).
    // ws_c: alpha + beta (visible via alpha, pass 1).
    // Remaining workspaces: alpha-only (pass 1).
    let ws_b = on_output[0];
    let ws_c = on_output[1];

    layout
        .workspaces
        .get_mut(&ws_b)
        .expect("ws_b must be a pool key")
        .activities = std::iter::once(beta).collect();
    layout
        .workspaces
        .get_mut(&ws_c)
        .expect("ws_c must be a pool key")
        .activities = [alpha, beta].into_iter().collect();

    let view = layout.active_view(&mon_out);
    let ws_c_view_pos = view
        .position_of(ws_c)
        .expect("ws_c must be present in the active view");
    let ws_c_expected_idx = u8::try_from(ws_c_view_pos + 1).unwrap_or(u8::MAX);

    let snapshot = crate::ipc::server::build_workspace_snapshot(&layout, None, test_window_id_of);

    let pool_ids: HashSet<u64> = layout
        .workspaces_all()
        .map(|(_, ws)| ws.id().get())
        .collect();
    let snapshot_ids: HashSet<u64> = snapshot.iter().map(|ws| ws.id).collect();
    assert_eq!(
        snapshot_ids, pool_ids,
        "snapshot must cover the pool exactly once (no omissions, no extras)",
    );
    assert_eq!(
        snapshot.len(),
        pool_ids.len(),
        "snapshot must not contain duplicate workspace ids",
    );

    let b = snapshot
        .iter()
        .find(|ws| ws.id == ws_b.get())
        .expect("ws_b must appear");
    assert_eq!(b.idx, 0, "beta-only workspace goes through pass 2 (idx 0)");
    assert!(!b.is_in_active_activity);
    assert_eq!(
        b.activities,
        vec![beta.get()],
        "beta-only workspace must list only beta in activities",
    );

    let c = snapshot
        .iter()
        .find(|ws| ws.id == ws_c.get())
        .expect("ws_c must appear");
    assert_eq!(
        c.idx, ws_c_expected_idx,
        "alpha+beta workspace is in the active activity and must use \
         its view-position idx ({}), not the sentinel 0",
        ws_c_expected_idx,
    );
    assert!(c.is_in_active_activity);
    let mut expected_c_activities = vec![alpha.get(), beta.get()];
    expected_c_activities.sort();
    assert_eq!(
        c.activities, expected_c_activities,
        "alpha+beta workspace must list both activity ids (sorted)",
    );

    layout.verify_invariants();
}

#[test]
fn create_activity_valid_name_succeeds() {
    // Happy path: create_activity("Beta") on a fresh layout must mint a new
    // runtime activity, leave the seed active, and satisfy the pool-level
    // invariants (verify_invariants) immediately.
    let mut layout = Layout::<TestWindow>::default();
    let seed_id = layout.active_activity_id();

    let id = layout
        .create_activity("Beta".to_owned())
        .expect("valid name must succeed");

    assert_ne!(id, seed_id, "new activity must have a distinct id");
    assert_eq!(layout.activities.len(), 2);
    assert_eq!(
        layout.active_activity_id(),
        seed_id,
        "create_activity must not flip the active cursor",
    );
    assert_eq!(
        layout.activities.previous_id(),
        None,
        "create_activity must not touch the previous cursor",
    );

    let created = layout
        .activities
        .get(id)
        .expect("new activity must be present in the pool");
    assert_eq!(created.name(), "Beta");
    assert!(
        !created.is_config_declared(),
        "runtime-created activity must not be flagged as config-declared",
    );
    assert!(
        created.views().is_empty(),
        "new runtime activity starts with no views (lazy population on switch)",
    );

    layout.verify_invariants();
}

#[test]
fn create_activity_empty_name_errs() {
    // Empty name → EmptyName, pool size unchanged, invariants preserved.
    let mut layout = Layout::<TestWindow>::default();
    let size_before = layout.activities.len();

    let err = layout
        .create_activity(String::new())
        .expect_err("empty name must be rejected");
    assert_eq!(err, CreateActivityError::EmptyName);
    assert_eq!(
        layout.activities.len(),
        size_before,
        "rejected create must not grow the pool",
    );

    layout.verify_invariants();
}

#[test]
fn create_activity_whitespace_only_name_errs() {
    // Whitespace-only name trims to empty and must be rejected as EmptyName
    // (not DuplicateName). Documents the trim-then-check rule.
    let mut layout = Layout::<TestWindow>::default();

    let err = layout
        .create_activity("   ".to_owned())
        .expect_err("whitespace-only name must be rejected");
    assert_eq!(err, CreateActivityError::EmptyName);
}

#[test]
fn create_activity_duplicate_name_errs_case_insensitive() {
    // Seed activity is "Default"; requesting "default" (lowercase) must collide
    // case-insensitively and be rejected without mutating the pool.
    let mut layout = Layout::<TestWindow>::default();
    let size_before = layout.activities.len();

    let err = layout
        .create_activity("default".to_owned())
        .expect_err("case-insensitive collision must be rejected");
    assert_eq!(err, CreateActivityError::DuplicateName);
    assert_eq!(
        layout.activities.len(),
        size_before,
        "rejected duplicate must not grow the pool",
    );
}

#[test]
fn create_activity_duplicate_name_errs_exact() {
    // After a successful create of "Beta", a second create_activity("Beta")
    // must collide on exact name and be rejected.
    let mut layout = Layout::<TestWindow>::default();
    layout
        .create_activity("Beta".to_owned())
        .expect("first create must succeed");
    let size_before = layout.activities.len();

    let err = layout
        .create_activity("Beta".to_owned())
        .expect_err("exact-name duplicate must be rejected");
    assert_eq!(err, CreateActivityError::DuplicateName);
    assert_eq!(layout.activities.len(), size_before);
}

#[test]
fn create_activity_duplicate_name_errs_case_insensitive_runtime() {
    // Case-insensitive collision with a runtime-created activity (not just the
    // config-seeded default) must also be rejected without growing the pool.
    let mut layout = Layout::<TestWindow>::default();
    layout
        .create_activity("Beta".to_owned())
        .expect("first create must succeed");

    let err = layout
        .create_activity("BETA".to_owned())
        .expect_err("case-insensitive collision with runtime activity must be rejected");
    assert_eq!(err, CreateActivityError::DuplicateName);
    assert_eq!(
        layout.activities.len(),
        2,
        "pool must not grow on rejection"
    );
}

#[test]
fn create_activity_expands_sticky_workspaces() {
    // A sticky workspace is "present on every activity" by the activities contract. When
    // a new runtime activity is created, its id must be unioned into every
    // sticky workspace's `activities` set; non-sticky workspaces' sets must
    // be untouched.
    let ops = [
        Op::AddOutput(1),
        Op::AddNamedWorkspace {
            ws_name: 1,
            output_name: None,
            layout_config: None,
        },
    ];
    let mut layout = check_ops(ops);
    let seed_id = layout.active_activity_id();
    let mon_out = layout.monitors[0].output_id();

    // Partition the two workspaces currently bound to the monitor: flip the
    // first sticky, leave the second as a non-sticky baseline. Direct field
    // access is permitted because this test module is a submodule of `layout`.
    let (sticky_id, nonsticky_id) = {
        let mut bound: Vec<WorkspaceId> = layout
            .workspaces
            .values()
            .filter(|ws| ws.output_id() == Some(&mon_out))
            .map(|ws| ws.id())
            .collect();
        bound.sort_by_key(|id| id.get());
        assert!(
            bound.len() >= 2,
            "test precondition: at least two workspaces bound to the output",
        );
        let sticky_id = bound[0];
        let nonsticky_id = bound[1];

        let sticky = layout
            .workspaces
            .get_mut(&sticky_id)
            .expect("sticky candidate must be in the pool");
        sticky.is_sticky = true;
        sticky.activities = HashSet::from([seed_id]);

        (sticky_id, nonsticky_id)
    };

    let beta_id = layout
        .create_activity("Beta".to_owned())
        .expect("valid name must succeed");

    // Sticky workspace's set must now contain both ids.
    let sticky_ws = layout
        .workspaces
        .get(&sticky_id)
        .expect("sticky workspace must still be in the pool");
    assert_eq!(
        sticky_ws.activities(),
        &HashSet::from([seed_id, beta_id]),
        "sticky workspace must gain the new activity id",
    );

    // Non-sticky workspace's set must NOT contain beta.
    let nonsticky_ws = layout
        .workspaces
        .get(&nonsticky_id)
        .expect("non-sticky workspace must still be in the pool");
    assert_eq!(
        nonsticky_ws.activities(),
        &HashSet::from([seed_id]),
        "non-sticky workspace's activities set must be untouched by create",
    );

    layout.verify_invariants();
}

#[test]
fn create_activity_no_view_population() {
    // Lazy view creation: create_activity does not populate per-output
    // views — those materialize on the first switch_activity to the new id.
    let mut layout = Layout::<TestWindow>::default();
    let id = layout
        .create_activity("Beta".to_owned())
        .expect("valid name must succeed");

    let created = layout
        .activities
        .get(id)
        .expect("new activity must be present");
    assert!(
        created.views().is_empty(),
        "new activity has no views until it becomes active",
    );
}

#[test]
fn create_activity_verify_invariants_empty_pool() {
    // Zero-workspace, zero-monitor layout (the default shape): verify_invariants
    // must accept the layout after create_activity, confirming the
    // sticky-expansion loop is a no-op when the workspace pool is empty.
    let mut layout = Layout::<TestWindow>::default();
    assert!(
        layout.monitors.is_empty(),
        "default layout starts with no monitors",
    );

    layout
        .create_activity("Beta".to_owned())
        .expect("valid name must succeed");

    layout.verify_invariants();
}

#[test]
fn rename_activity_valid_name_succeeds() {
    // Happy path: rename_activity must update the target's name, leave the
    // active cursor untouched, and satisfy verify_invariants.
    let mut layout = Layout::<TestWindow>::default();
    let beta_id = layout
        .create_activity("Beta".to_owned())
        .expect("create must succeed");
    let seed_id = layout.active_activity_id();
    let prev_before = layout.activities.previous_id();

    let returned = layout
        .rename_activity(&ActivityReferenceArg::Id(beta_id.get()), "Gamma".to_owned())
        .expect("valid rename must succeed");
    assert_eq!(returned, beta_id, "rename must return the resolved id");

    let renamed = layout
        .activities
        .get(beta_id)
        .expect("renamed activity must still be in the pool");
    assert_eq!(renamed.name(), "Gamma");
    assert_eq!(
        layout.active_activity_id(),
        seed_id,
        "rename must not flip the active cursor",
    );
    assert_eq!(
        layout.activities.previous_id(),
        prev_before,
        "rename must not touch the previous cursor",
    );

    layout.verify_invariants();
}

#[test]
fn rename_activity_empty_name_errs() {
    // Empty name → EmptyName; pool unchanged, name unchanged.
    let mut layout = Layout::<TestWindow>::default();
    let beta_id = layout
        .create_activity("Beta".to_owned())
        .expect("create must succeed");

    let err = layout
        .rename_activity(&ActivityReferenceArg::Id(beta_id.get()), String::new())
        .expect_err("empty name must be rejected");
    assert_eq!(err, RenameActivityError::EmptyName);
    assert_eq!(
        layout
            .activities
            .get(beta_id)
            .expect("beta still present")
            .name(),
        "Beta",
        "rejected rename must not mutate the name",
    );

    layout.verify_invariants();
}

#[test]
fn rename_activity_whitespace_only_name_errs() {
    // Whitespace-only name trims to empty and must be rejected as EmptyName
    // (not DuplicateName). Pins the trim-then-check rule.
    let mut layout = Layout::<TestWindow>::default();
    let beta_id = layout
        .create_activity("Beta".to_owned())
        .expect("create must succeed");

    let err = layout
        .rename_activity(&ActivityReferenceArg::Id(beta_id.get()), "   ".to_owned())
        .expect_err("whitespace-only name must be rejected");
    assert_eq!(err, RenameActivityError::EmptyName);
}

#[test]
fn rename_activity_duplicate_name_errs_case_insensitive() {
    // Two activities Beta + Gamma. Attempting to rename Beta to "GAMMA"
    // must collide case-insensitively with Gamma's existing name.
    let mut layout = Layout::<TestWindow>::default();
    let beta_id = layout
        .create_activity("Beta".to_owned())
        .expect("create must succeed");
    layout
        .create_activity("Gamma".to_owned())
        .expect("create must succeed");

    let err = layout
        .rename_activity(&ActivityReferenceArg::Id(beta_id.get()), "GAMMA".to_owned())
        .expect_err("case-insensitive collision must be rejected");
    assert_eq!(err, RenameActivityError::DuplicateName);
    assert_eq!(
        layout
            .activities
            .get(beta_id)
            .expect("beta still present")
            .name(),
        "Beta",
        "rejected rename must not mutate the name",
    );
}

#[test]
fn rename_activity_config_declared_errs() {
    // Config-declared activities cannot be renamed at runtime (mirrors the
    // remove policy — edit config + reload instead).
    let cfg = vec![
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("Work".to_owned()),
        },
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("Personal".to_owned()),
        },
    ];
    #[allow(clippy::field_reassign_with_default)]
    let mut layout = {
        let mut layout = Layout::<TestWindow>::default();
        layout.activities = super::activity::Activities::from_config_or_default(&cfg);
        layout
    };
    let work_id = layout.active_activity_id();

    let err = layout
        .rename_activity(
            &ActivityReferenceArg::Id(work_id.get()),
            "Office".to_owned(),
        )
        .expect_err("config-declared rename must err");
    assert_eq!(err, RenameActivityError::ConfigDeclared);
    assert_eq!(
        layout
            .activities
            .get(work_id)
            .expect("work still present")
            .name(),
        "Work",
        "rejected rename must not mutate the name",
    );

    layout.verify_invariants();
}

#[test]
fn rename_activity_not_found_errs() {
    // Unknown id → NotFound. Uses u64::MAX (never minted by ActivityId::next).
    let mut layout = Layout::<TestWindow>::default();

    let err = layout
        .rename_activity(&ActivityReferenceArg::Id(u64::MAX), "Whatever".to_owned())
        .expect_err("unknown id must err");
    assert_eq!(err, RenameActivityError::NotFound);

    layout.verify_invariants();
}

#[test]
fn rename_activity_self_same_name_is_noop() {
    // Renaming an activity to its exact current name must succeed (Ok(id))
    // rather than self-colliding in the duplicate scan. Load-bearing for the
    // "exclude self from dup scan" invariant.
    let mut layout = Layout::<TestWindow>::default();
    let beta_id = layout
        .create_activity("Beta".to_owned())
        .expect("create must succeed");

    let returned = layout
        .rename_activity(&ActivityReferenceArg::Id(beta_id.get()), "Beta".to_owned())
        .expect("self-same-name rename must succeed");
    assert_eq!(returned, beta_id);
    assert_eq!(
        layout
            .activities
            .get(beta_id)
            .expect("beta still present")
            .name(),
        "Beta",
    );

    layout.verify_invariants();
}

#[test]
fn rename_activity_self_case_change_succeeds() {
    // Renaming "Beta" → "beta" on the same activity must succeed — the
    // duplicate scan must exclude the target id, otherwise every case-change
    // rename would self-collide. Regression pin against a future "simplify"
    // that accidentally reuses create_runtime's uniform scan.
    let mut layout = Layout::<TestWindow>::default();
    let beta_id = layout
        .create_activity("Beta".to_owned())
        .expect("create must succeed");

    let returned = layout
        .rename_activity(&ActivityReferenceArg::Id(beta_id.get()), "beta".to_owned())
        .expect("self case-change rename must succeed");
    assert_eq!(returned, beta_id);
    assert_eq!(
        layout
            .activities
            .get(beta_id)
            .expect("beta still present")
            .name(),
        "beta",
        "rename must actually persist the new casing",
    );

    layout.verify_invariants();
}

#[test]
fn rename_activity_does_not_touch_views_or_workspace_sets() {
    // Rename is pure metadata: every workspace's `activities` set and every
    // activity's `views` map must be bitwise unchanged after a rename.
    let ops = [
        Op::AddOutput(1),
        Op::AddNamedWorkspace {
            ws_name: 1,
            output_name: None,
            layout_config: None,
        },
    ];
    let mut layout = check_ops(ops);
    let seed_id = layout.active_activity_id();
    let mon_out = layout.monitors[0].output_id();

    // Flip one workspace sticky so create_activity unions beta into it — this
    // ensures the post-rename workspace sets contain multiple activity ids, so
    // a regression that accidentally mutates the sets is observable.
    let sticky_id = {
        let sticky_id = layout
            .workspaces
            .values()
            .find(|ws| ws.output_id() == Some(&mon_out))
            .expect("test precondition: at least one workspace bound to the output")
            .id();
        let sticky = layout
            .workspaces
            .get_mut(&sticky_id)
            .expect("sticky candidate must be in the pool");
        sticky.is_sticky = true;
        sticky.activities = HashSet::from([seed_id]);
        sticky_id
    };

    let beta_id = layout
        .create_activity("Beta".to_owned())
        .expect("create must succeed");

    // Snapshot the state we expect to be preserved.
    let workspace_sets_before: HashMap<WorkspaceId, HashSet<ActivityId>> = layout
        .workspaces
        .iter()
        .map(|(id, ws)| (*id, ws.activities().clone()))
        .collect();
    // `WorkspaceView` doesn't derive `PartialEq` — project to a structural
    // tuple (ids, active, previous) that captures every observable field and
    // is `Eq`-comparable.
    type ViewProjection = (Vec<WorkspaceId>, WorkspaceId, Option<WorkspaceId>);
    fn project_views(a: &super::activity::Activity) -> HashMap<OutputId, ViewProjection> {
        a.views()
            .iter()
            .map(|(out, v)| (out.clone(), (v.ids().to_vec(), v.active(), v.previous())))
            .collect()
    }
    let views_before: HashMap<ActivityId, HashMap<OutputId, ViewProjection>> = layout
        .activities
        .iter()
        .map(|a| (a.id(), project_views(a)))
        .collect();

    layout
        .rename_activity(&ActivityReferenceArg::Id(beta_id.get()), "Gamma".to_owned())
        .expect("valid rename must succeed");

    // Every workspace's activities set must be bitwise unchanged.
    for (ws_id, set_before) in &workspace_sets_before {
        let set_after = layout
            .workspaces
            .get(ws_id)
            .expect("workspace must still be in the pool")
            .activities();
        assert_eq!(
            set_after, set_before,
            "rename must not touch workspace {ws_id:?}'s activities set",
        );
    }

    // Every activity's views map must be bitwise unchanged.
    for (act_id, views_before_one) in &views_before {
        let after = project_views(
            layout
                .activities
                .get(*act_id)
                .expect("activity must still be in the pool"),
        );
        assert_eq!(
            &after, views_before_one,
            "rename must not touch activity {act_id:?}'s views",
        );
    }

    // Sanity: the sticky workspace still lists both ids (seed + beta).
    assert_eq!(
        layout
            .workspaces
            .get(&sticky_id)
            .expect("sticky workspace still in pool")
            .activities(),
        &HashSet::from([seed_id, beta_id]),
    );

    layout.verify_invariants();
}

#[test]
fn remove_activity_unknown_reference_errs_not_found() {
    // Unknown name → NotFound; pool unchanged, invariants preserved.
    let mut layout = Layout::<TestWindow>::default();
    let size_before = layout.activities.len();

    let err = layout
        .remove_activity(&ActivityReferenceArg::Name("Ghost".to_owned()))
        .expect_err("unknown reference must err");
    assert_eq!(err, RemoveActivityError::NotFound);
    assert_eq!(
        layout.activities.len(),
        size_before,
        "pool must not shrink on NotFound",
    );

    layout.verify_invariants();
}

#[test]
fn remove_activity_config_declared_errs() {
    // A config-declared activity cannot be removed at runtime (
    // bullet 1). Seed a config-declared "Alpha" as the first activity, create
    // a runtime "Beta", then attempt to remove Alpha → ConfigDeclared.
    let cfg = vec![jiji_config::ActivityDecl {
        name: jiji_config::ActivityName("Alpha".to_owned()),
    }];
    // Swap in a config-declared seed pool after the default construction.
    // `Layout` has no public builder that takes a pool; this is a test-only
    // shortcut. clippy's `field_reassign_with_default` flags the combined
    // construct-and-assign pattern even when it's the intent, hence the
    // scoped `#[allow]`.
    #[allow(clippy::field_reassign_with_default)]
    let mut layout = {
        let mut layout = Layout::<TestWindow>::default();
        layout.activities = super::activity::Activities::from_config_or_default(&cfg);
        layout
    };
    let alpha_id = layout.active_activity_id();

    layout
        .create_activity("Beta".to_owned())
        .expect("runtime create must succeed");
    let size_before = layout.activities.len();

    let err = layout
        .remove_activity(&ActivityReferenceArg::Id(alpha_id.get()))
        .expect_err("config-declared removal must err");
    assert_eq!(err, RemoveActivityError::ConfigDeclared);
    assert_eq!(
        layout.activities.len(),
        size_before,
        "pool must not shrink on ConfigDeclared",
    );
    assert!(
        layout.activities.contains(alpha_id),
        "alpha must still be in the pool after a rejected remove",
    );

    layout.verify_invariants();
}

#[test]
fn remove_activity_last_remaining_errs() {
    // Fresh default layout has exactly one activity; removing it must fail
    // with LastRemaining ( bullet 3).
    let mut layout = Layout::<TestWindow>::default();
    let seed_id = layout.active_activity_id();
    assert_eq!(layout.activities.len(), 1);

    let err = layout
        .remove_activity(&ActivityReferenceArg::Id(seed_id.get()))
        .expect_err("last-remaining removal must err");
    assert_eq!(err, RemoveActivityError::LastRemaining);
    assert_eq!(
        layout.activities.len(),
        1,
        "pool size must stay at 1 after LastRemaining",
    );

    layout.verify_invariants();
}

#[test]
fn remove_activity_exclusive_workspace_with_windows_errs() {
    // Exclusive (activities == {beta}) workspace carrying windows must block
    // removal ( bullet 2). Test that the pool is untouched after the
    // rejection — no partial mutation.
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(42),
        },
    ];
    let mut layout = check_ops(ops);

    let beta_activity = super::activity::Activity::new_runtime("beta".to_owned());
    let beta = beta_activity.id();
    test_insert_activity(&mut layout, beta_activity);

    // Pick the window-carrying workspace and flip its activities set to
    // {beta} exclusively.
    let window_ws_id = layout
        .workspaces
        .values()
        .find(|ws| ws.has_windows())
        .expect("window was added")
        .id();
    layout
        .workspaces
        .get_mut(&window_ws_id)
        .expect("window_ws_id must be a live pool key")
        .activities = std::iter::once(beta).collect();

    let size_before = layout.activities.len();
    let pool_before = layout.workspaces.len();

    let err = layout
        .remove_activity(&ActivityReferenceArg::Id(beta.get()))
        .expect_err("exclusive-with-windows must err");
    assert_eq!(err, RemoveActivityError::ExclusiveWorkspaceHasWindows);
    assert_eq!(layout.activities.len(), size_before, "pool unchanged");
    assert_eq!(layout.workspaces.len(), pool_before, "workspaces unchanged");
    assert!(layout.activities.contains(beta), "beta still present");

    layout.verify_invariants();
}

#[test]
fn remove_activity_exclusive_named_workspace_errs() {
    // Exclusive named workspace blocks removal even when empty (
    // "Exclusive workspace destruction semantics" — named-empty is preserved
    // specifically because the user gave it a name).
    let ops = [
        Op::AddOutput(1),
        Op::AddNamedWorkspace {
            ws_name: 7,
            output_name: None,
            layout_config: None,
        },
    ];
    let mut layout = check_ops(ops);

    let beta_activity = super::activity::Activity::new_runtime("beta".to_owned());
    let beta = beta_activity.id();
    test_insert_activity(&mut layout, beta_activity);

    // Find the named workspace and flip it to exclusively beta.
    let named_ws_id = layout
        .workspaces
        .values()
        .find(|ws| ws.name() == Some(&"ws7".to_owned()))
        .expect("named workspace was added")
        .id();
    layout
        .workspaces
        .get_mut(&named_ws_id)
        .expect("named_ws_id must be a live pool key")
        .activities = std::iter::once(beta).collect();

    let size_before = layout.activities.len();

    let err = layout
        .remove_activity(&ActivityReferenceArg::Id(beta.get()))
        .expect_err("exclusive-named must err");
    assert_eq!(err, RemoveActivityError::ExclusiveNamedWorkspace);
    assert_eq!(layout.activities.len(), size_before, "pool unchanged");
    assert!(
        layout.workspaces.contains_key(&named_ws_id),
        "named workspace must survive a rejected remove",
    );

    layout.verify_invariants();
}

#[test]
fn remove_activity_runtime_non_active_with_shared_only_prunes() {
    // Non-active runtime target; only shared workspaces reference it. Remove
    // must succeed, and every shared workspace must drop the target id from
    // its `activities` set while keeping every other id.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();

    let beta_activity = super::activity::Activity::new_runtime("beta".to_owned());
    let beta = beta_activity.id();
    test_insert_activity(&mut layout, beta_activity);

    // Union beta into the alpha workspace (making it shared {alpha, beta}).
    let shared_ws_id = layout
        .workspaces
        .values()
        .find(|ws| ws.activities() == &HashSet::from([alpha]))
        .map(|ws| ws.id())
        .expect("AddOutput allocated a seed workspace");
    layout
        .workspaces
        .get_mut(&shared_ws_id)
        .expect("shared_ws_id must be a live pool key")
        .activities = [alpha, beta].into_iter().collect();

    let pool_size_before = layout.workspaces.len();

    layout
        .remove_activity(&ActivityReferenceArg::Id(beta.get()))
        .expect("non-active shared-only remove must succeed");

    assert_eq!(layout.activities.len(), 1, "beta dropped from pool");
    assert!(!layout.activities.contains(beta), "beta no longer live");
    // The shared workspace is retained (only its beta tag is pruned). The materializer's
    // exclusive-to-beta bookend, though, IS destroyed by `remove_activity` (it's an empty
    // unnamed exclusive of the removed activity) — pool drops by exactly one.
    assert_eq!(
        layout.workspaces.len(),
        pool_size_before - 1,
        "shared workspace is retained; only beta's materialized bookend is destroyed",
    );
    let shared_ws = layout
        .workspaces
        .get(&shared_ws_id)
        .expect("shared workspace must still exist");
    assert_eq!(
        shared_ws.activities(),
        &HashSet::from([alpha]),
        "shared workspace must have beta pruned, alpha retained",
    );

    layout.verify_invariants();
}

#[test]
fn remove_activity_runtime_non_active_destroys_empty_unnamed_exclusives() {
    // Non-active target with an empty unnamed exclusive workspace: the
    // workspace must be destroyed (removed from the pool and from every
    // activity's views). Baseline pool size returns to pre-allocation.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    // Capture the baseline BEFORE inserting beta, so it doesn't include the materializer's
    // freshly-minted exclusive bookend (which is also destroyed by `remove_activity(beta)`).
    let pool_size_baseline = layout.workspaces.len();

    let beta_activity = super::activity::Activity::new_runtime("beta".to_owned());
    let beta = beta_activity.id();
    test_insert_activity(&mut layout, beta_activity);

    // Allocate a fresh exclusive unnamed workspace tagged to beta.
    let mon_out = layout.monitors[0].output_id();
    let output = layout.monitors[0].output.clone();
    let beta_ws = Workspace::new(
        &output,
        HashSet::from([beta]),
        layout.clock.clone(),
        layout.options.clone(),
    );
    let beta_ws_id = beta_ws.id();
    assert!(
        layout.workspaces.insert(beta_ws_id, beta_ws).is_none(),
        "fresh id is unique",
    );
    // Sanity: our fresh workspace is on the output and exclusively beta's.
    assert_eq!(
        layout
            .workspaces
            .get(&beta_ws_id)
            .expect("just inserted")
            .output_id(),
        Some(&mon_out),
    );

    layout
        .remove_activity(&ActivityReferenceArg::Id(beta.get()))
        .expect("non-active empty-unnamed-exclusive remove must succeed");

    assert!(!layout.activities.contains(beta), "beta dropped");
    assert!(
        !layout.workspaces.contains_key(&beta_ws_id),
        "exclusive empty unnamed workspace must be destroyed",
    );
    assert_eq!(
        layout.workspaces.len(),
        pool_size_baseline,
        "workspace count must return to pre-allocation baseline (materializer's exclusive \
         bookend was also exclusive-empty-unnamed and is also destroyed)",
    );

    layout.verify_invariants();
}

#[test]
fn remove_activity_active_cascades_to_previous() {
    // Active target with a non-None previous: cascade to previous, then
    // remove. After the cascade, previous points at the now-removed target;
    // `Activities::remove` must clear previous to None.
    let mut layout = Layout::<TestWindow>::default();
    let alpha = layout.active_activity_id();

    let beta = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");

    layout.switch_activity(beta);
    assert_eq!(layout.active_activity_id(), beta);
    assert_eq!(layout.activities.previous_id(), Some(alpha));

    layout
        .remove_activity(&ActivityReferenceArg::Id(beta.get()))
        .expect("removing the active, with previous set, must cascade");

    assert_eq!(
        layout.active_activity_id(),
        alpha,
        "cascade target was previous (alpha)",
    );
    assert_eq!(
        layout.activities.previous_id(),
        None,
        "previous pointed at the removed target and must be cleared",
    );
    assert_eq!(layout.activities.len(), 1);

    layout.verify_invariants();
}

#[test]
fn remove_activity_active_cascades_to_first_remaining_when_no_previous() {
    // Active target with previous == None: cascade to the first other
    // activity in declaration order. Here alpha is active with no previous,
    // and beta is the only other entry.
    let mut layout = Layout::<TestWindow>::default();
    let alpha = layout.active_activity_id();

    let beta = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");

    // No switch — previous is None.
    assert_eq!(layout.active_activity_id(), alpha);
    assert_eq!(layout.activities.previous_id(), None);

    layout
        .remove_activity(&ActivityReferenceArg::Id(alpha.get()))
        .expect("removing active with no previous must cascade to first remaining");

    assert_eq!(
        layout.active_activity_id(),
        beta,
        "cascade target was the first other activity in declaration order",
    );
    assert_eq!(
        layout.activities.previous_id(),
        None,
        "post-cascade previous pointed at the removed alpha and must be cleared",
    );
    assert_eq!(layout.activities.len(), 1);

    layout.verify_invariants();
}

#[test]
fn remove_activity_clears_previous_pointer_at_target() {
    // Non-active target, but `previous` already points at it. The pool
    // mutator must clear `previous` when it points at the id being removed,
    // not just on the cascade branch.
    let mut layout = Layout::<TestWindow>::default();
    let alpha = layout.active_activity_id();

    let beta = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");

    // switch to beta (active=beta, previous=alpha), then back (active=alpha,
    // previous=beta). Now beta is NOT active, but previous points at it.
    layout.switch_activity(beta);
    layout.switch_activity(alpha);
    assert_eq!(layout.active_activity_id(), alpha);
    assert_eq!(layout.activities.previous_id(), Some(beta));

    layout
        .remove_activity(&ActivityReferenceArg::Id(beta.get()))
        .expect("non-active remove of previous-pointed id must succeed");

    assert_eq!(layout.active_activity_id(), alpha);
    assert_eq!(
        layout.activities.previous_id(),
        None,
        "previous pointed at the removed beta and must be cleared",
    );
    assert_eq!(layout.activities.len(), 1);

    layout.verify_invariants();
}

#[test]
fn remove_activity_sticky_workspace_pruned_not_destroyed() {
    // A sticky workspace carries every activity id by definition. Creating
    // beta via `create_activity` expands its set to {alpha, beta}. Removing
    // beta must prune the set to {alpha} and leave the workspace intact —
    // including its is_sticky flag.
    let ops = [
        Op::AddOutput(1),
        Op::AddNamedWorkspace {
            ws_name: 1,
            output_name: None,
            layout_config: None,
        },
    ];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let mon_out = layout.monitors[0].output_id();

    let sticky_id = {
        let mut bound: Vec<WorkspaceId> = layout
            .workspaces
            .values()
            .filter(|ws| ws.output_id() == Some(&mon_out))
            .map(|ws| ws.id())
            .collect();
        bound.sort_by_key(|id| id.get());
        let sticky_id = *bound.first().expect("at least one workspace is bound");
        let sticky = layout
            .workspaces
            .get_mut(&sticky_id)
            .expect("sticky_id must be a live pool key");
        sticky.is_sticky = true;
        sticky.activities = HashSet::from([alpha]);
        sticky_id
    };

    let beta = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");
    // Post-expansion sanity: sticky set carries both ids.
    assert_eq!(
        layout
            .workspaces
            .get(&sticky_id)
            .expect("sticky must survive create")
            .activities(),
        &HashSet::from([alpha, beta]),
    );

    layout
        .remove_activity(&ActivityReferenceArg::Id(beta.get()))
        .expect("non-active remove with sticky-shared workspace must succeed");

    let sticky_ws = layout
        .workspaces
        .get(&sticky_id)
        .expect("sticky workspace must survive remove");
    assert!(
        sticky_ws.is_sticky(),
        "is_sticky flag must be preserved across remove",
    );
    assert_eq!(
        sticky_ws.activities(),
        &HashSet::from([alpha]),
        "sticky workspace must have beta pruned",
    );
    assert_eq!(layout.activities.len(), 1);

    layout.verify_invariants();
}

#[test]
fn remove_activity_error_precedence_windows_beats_named() {
    // When a target activity has BOTH an exclusive workspace with windows AND
    // an exclusive named workspace, `ExclusiveWorkspaceHasWindows` must win
    // over `ExclusiveNamedWorkspace` regardless of HashMap iteration order.
    // This pins the accumulate-all-violations design at mod.rs:4047-4071.
    let ops = [
        Op::AddOutput(1),
        Op::AddNamedWorkspace {
            ws_name: 7,
            output_name: None,
            layout_config: None,
        },
        Op::AddWindow {
            params: TestWindowParams::new(42),
        },
    ];
    let mut layout = check_ops(ops);

    let beta_activity = super::activity::Activity::new_runtime("beta".to_owned());
    let beta = beta_activity.id();
    test_insert_activity(&mut layout, beta_activity);

    // Flip the named workspace to exclusively beta.
    let named_ws_id = layout
        .workspaces
        .values()
        .find(|ws| ws.name() == Some(&"ws7".to_owned()))
        .expect("named workspace was added")
        .id();
    layout
        .workspaces
        .get_mut(&named_ws_id)
        .expect("named_ws_id must be a live pool key")
        .activities = std::iter::once(beta).collect();

    // Flip the window-carrying workspace to exclusively beta.
    let windowed_ws_id = layout
        .workspaces
        .values()
        .find(|ws| ws.has_windows())
        .expect("window was added")
        .id();
    layout
        .workspaces
        .get_mut(&windowed_ws_id)
        .expect("windowed_ws_id must be a live pool key")
        .activities = std::iter::once(beta).collect();

    assert_ne!(
        named_ws_id, windowed_ws_id,
        "test requires two distinct exclusive workspaces to pin precedence between simultaneous violations",
    );
    assert!(
        !layout.workspaces[&named_ws_id].has_windows(),
        "named workspace must be empty so ExclusiveNamedWorkspace is the only violation it contributes",
    );

    // Both violations are present; the has-windows check must take precedence.
    let err = layout
        .remove_activity(&ActivityReferenceArg::Id(beta.get()))
        .expect_err("compound violation must err");
    assert_eq!(
        err,
        RemoveActivityError::ExclusiveWorkspaceHasWindows,
        "ExclusiveWorkspaceHasWindows must outrank ExclusiveNamedWorkspace",
    );

    // Pool must be completely untouched.
    assert!(layout.activities.contains(beta), "beta still present");

    layout.verify_invariants();
}

#[test]
fn remove_activity_view_patching_both_branches() {
    // Exercises BOTH branches of the retain closure in the exclusive-workspace
    // destruction loop (mod.rs:4097-4113):
    //
    //   - Drop branch (view.len() == 1 → return false): gamma's dormant view for the output
    //     contains only beta_ws_id; after removal the whole entry is dropped.
    //   - Patch branch (view.len() > 1 → remove_at): alpha's active view for the output contains
    //     [alpha_default_ws, beta_ws_id]; after removal only [alpha_default_ws] remains.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);

    let beta_activity = super::activity::Activity::new_runtime("beta".to_owned());
    let beta = beta_activity.id();
    test_insert_activity(&mut layout, beta_activity);

    let gamma_activity = super::activity::Activity::new_runtime("gamma".to_owned());
    let gamma = gamma_activity.id();
    test_insert_activity(&mut layout, gamma_activity);

    // Allocate an exclusive unnamed empty workspace for beta.
    let mon_out = layout.monitors[0].output_id();
    let output = layout.monitors[0].output.clone();
    let beta_ws = Workspace::new(
        &output,
        HashSet::from([beta]),
        layout.clock.clone(),
        layout.options.clone(),
    );
    let beta_ws_id = beta_ws.id();
    assert!(
        layout.workspaces.insert(beta_ws_id, beta_ws).is_none(),
        "fresh id must be unique",
    );

    // Retrieve alpha's existing workspace id from its active view.
    let alpha_default_ws_id = layout
        .active_view(&mon_out)
        .ids()
        .first()
        .copied()
        .expect("active view must have at least one workspace");

    // Patch branch: inject beta_ws_id into alpha's active view alongside
    // the existing workspace so the view has two entries.
    layout
        .activities
        .active_mut()
        .views_mut()
        .get_mut(&mon_out)
        .expect("active activity must have a view for the connected output")
        .insert(1, beta_ws_id);

    // Drop branch: give gamma a dormant view whose sole entry is beta_ws_id.
    test_override_activity_view(
        &mut layout,
        gamma,
        mon_out.clone(),
        WorkspaceView::new(vec![beta_ws_id], 0),
    );

    // Sanity: both branches are populated before removal.
    assert_eq!(
        layout.active_view(&mon_out).ids(),
        &[alpha_default_ws_id, beta_ws_id],
        "alpha view must contain both workspaces before removal",
    );
    assert_eq!(
        layout
            .activities
            .get(gamma)
            .expect("gamma live")
            .views()
            .get(&mon_out)
            .expect("gamma has view")
            .ids(),
        &[beta_ws_id],
        "gamma view must contain only beta_ws_id before removal",
    );

    layout
        .remove_activity(&ActivityReferenceArg::Id(beta.get()))
        .expect("remove beta with exclusive ws must succeed");

    // Patch branch result: alpha's view now contains only the original workspace.
    assert_eq!(
        layout.active_view(&mon_out).ids(),
        &[alpha_default_ws_id],
        "alpha view must have beta_ws_id removed (patch branch)",
    );

    // Drop branch result: gamma's old view entry for the output (the single beta_ws_id) was
    // dropped by the destroy_workspaces_cross_activity retain closure. The per-activity
    // bookend materializer then runs (because `remove_activity` calls it on success), so
    // gamma ends up with a freshly-minted single-bookend view — NOT the original
    // pointing at beta_ws_id.
    let gamma_view = layout
        .activities
        .get(gamma)
        .expect("gamma still live")
        .views()
        .get(&mon_out)
        .expect("gamma's view re-materialized by the per-activity bookend invariant");
    assert!(
        !gamma_view.ids().contains(&beta_ws_id),
        "gamma's view must no longer reference the destroyed beta_ws_id (drop branch)",
    );
    assert_eq!(
        gamma_view.len(),
        1,
        "re-materialized view holds exactly one fresh bookend id",
    );

    assert!(!layout.activities.contains(beta), "beta removed");
    assert!(
        !layout.workspaces.contains_key(&beta_ws_id),
        "beta_ws destroyed"
    );

    layout.verify_invariants();
}

#[test]
fn remove_activity_active_with_exclusive_workspace_cascades_and_destroys() {
    // Compound path: active activity with an exclusive unnamed-empty workspace.
    // switch_activity → ensure_all_activity_views allocates an exclusive workspace for
    // beta on the connected output. Both the cascade (active → previous) and
    // exclusive-ws destruction must complete in a single `remove_activity` call.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();

    let alpha_ws_id = {
        let mon_out = layout.monitors[0].output_id();
        layout
            .active_view(&mon_out)
            .ids()
            .first()
            .copied()
            .expect("alpha's initial view must have a workspace")
    };

    let beta = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");

    // Switch so beta is active; ensure_all_activity_views allocates an unnamed-empty
    // exclusive workspace for beta on the output.
    layout.switch_activity(beta);
    assert_eq!(layout.active_activity_id(), beta);
    assert_eq!(layout.activities.previous_id(), Some(alpha));

    // Identify the exclusive workspace that ensure_all_activity_views created for beta.
    let beta_ws_id = layout
        .workspaces
        .values()
        .find(|ws| {
            ws.activities().len() == 1
                && ws.activities().contains(&beta)
                && !ws.has_windows()
                && ws.name().is_none()
        })
        .expect("ensure_all_activity_views must have created an exclusive empty workspace for beta")
        .id();

    let pool_size_before = layout.workspaces.len();

    layout
        .remove_activity(&ActivityReferenceArg::Id(beta.get()))
        .expect("removing active with exclusive ws must succeed");

    // Cascade: alpha is now active.
    assert_eq!(
        layout.active_activity_id(),
        alpha,
        "cascade must land on alpha (previous)",
    );
    assert_eq!(
        layout.activities.previous_id(),
        None,
        "previous pointed at the removed beta and must be cleared",
    );

    // Destruction: beta's exclusive workspace must be gone; alpha's ws survives.
    assert!(
        !layout.workspaces.contains_key(&beta_ws_id),
        "exclusive unnamed-empty ws must be destroyed",
    );
    assert!(
        layout.workspaces.contains_key(&alpha_ws_id),
        "alpha's workspace must survive",
    );
    assert_eq!(
        layout.workspaces.len(),
        pool_size_before - 1,
        "workspace count must decrease by one (the destroyed beta ws)",
    );

    assert!(!layout.activities.contains(beta), "beta removed from pool");
    assert_eq!(layout.activities.len(), 1);

    layout.verify_invariants();
}

#[test]
fn remove_activity_multiple_exclusive_workspaces_all_destroyed() {
    // Target activity owns TWO unnamed-empty exclusive workspaces (N > 1
    // path through the destroy_ids loop). Both must be removed from the pool.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    // Capture the baseline BEFORE inserting beta, so it doesn't include the materializer's
    // exclusive bookend (also destroyed by `remove_activity(beta)`).
    let pool_size_baseline = layout.workspaces.len();

    let beta_activity = super::activity::Activity::new_runtime("beta".to_owned());
    let beta = beta_activity.id();
    test_insert_activity(&mut layout, beta_activity);

    let output = layout.monitors[0].output.clone();

    // Allocate two exclusive unnamed empty workspaces for beta.
    let beta_ws1 = Workspace::new(
        &output,
        HashSet::from([beta]),
        layout.clock.clone(),
        layout.options.clone(),
    );
    let beta_ws1_id = beta_ws1.id();

    let beta_ws2 = Workspace::new(
        &output,
        HashSet::from([beta]),
        layout.clock.clone(),
        layout.options.clone(),
    );
    let beta_ws2_id = beta_ws2.id();

    assert!(
        layout.workspaces.insert(beta_ws1_id, beta_ws1).is_none(),
        "beta_ws1 id must be unique",
    );
    assert!(
        layout.workspaces.insert(beta_ws2_id, beta_ws2).is_none(),
        "beta_ws2 id must be unique",
    );
    // pool has: baseline (alpha's view) + 1 (materializer's beta bookend) + 2 (beta_ws1, beta_ws2)
    assert_eq!(
        layout.workspaces.len(),
        pool_size_baseline + 3,
        "two exclusive workspaces plus the materializer's beta bookend",
    );

    layout
        .remove_activity(&ActivityReferenceArg::Id(beta.get()))
        .expect("remove beta with two exclusive workspaces must succeed");

    assert!(!layout.activities.contains(beta), "beta removed");
    assert!(
        !layout.workspaces.contains_key(&beta_ws1_id),
        "beta_ws1 must be destroyed",
    );
    assert!(
        !layout.workspaces.contains_key(&beta_ws2_id),
        "beta_ws2 must be destroyed",
    );
    assert_eq!(
        layout.workspaces.len(),
        pool_size_baseline,
        "workspace count must return to pre-allocation baseline",
    );

    layout.verify_invariants();
}

#[test]
fn remove_activity_success_via_name_reference() {
    // Successful removal resolved by name rather than id. Exercises the Name
    // arm of `resolve_activity_ref` on the happy path.
    let mut layout = Layout::<TestWindow>::default();
    let alpha = layout.active_activity_id();

    let beta = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");

    assert_eq!(layout.activities.len(), 2);

    layout
        .remove_activity(&ActivityReferenceArg::Name("Beta".to_owned()))
        .expect("remove by name must succeed");

    assert!(!layout.activities.contains(beta), "beta removed");
    assert!(layout.activities.contains(alpha), "alpha intact");
    assert_eq!(layout.activities.len(), 1);

    layout.verify_invariants();
}

// Build a layout where the active view has an empty unnamed workspace
// strictly between populated/active positions, call
// `clean_up_workspaces_on` directly, and confirm the returned `Vec`
// contains the pruned id. The pool MUST still hold the id — caller is
// responsible for the follow-up `destroy_workspaces_cross_activity`.
#[test]
fn clean_up_workspaces_on_returns_pruned_ids_main_loop() {
    // Fresh output with one window yields [W1, E_bottom]. Insert a spare
    // empty unnamed workspace at position 1 directly — that puts an empty
    // strictly between the named W1 and the trailing empty, which is the
    // exact main-loop prunable shape.
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
    ];
    let mut layout = check_ops(ops);
    let mon_out = layout.monitors[0].output_id();
    let seed_activity = layout.active_activity_id();

    // Sanity: expected [W1, E_bottom] shape with active on W1 (pos 0).
    {
        let view = layout.active_view(&mon_out);
        let pool = layout.workspace_pool();
        let named: Vec<bool> = view
            .ids()
            .iter()
            .map(|id| pool.get(id).unwrap().has_windows_or_name())
            .collect();
        assert_eq!(named, vec![true, false], "baseline");
        assert_eq!(view.active_position(), 0);
    }

    // Add an empty unnamed workspace between W1 and E_bottom. The resulting
    // shape is [W1 (active), E_mid, E_bottom] — E_mid is prunable.
    let spare = Workspace::<TestWindow>::new_no_outputs(
        HashSet::from([seed_activity]),
        layout.clock.clone(),
        layout.options.clone(),
    );
    let prunable_id = spare.id();
    layout.workspaces.insert(prunable_id, spare);
    // Bind to this monitor's output so invariants stay happy.
    let mon_output = layout.monitors[0].output.clone();
    layout
        .workspaces
        .get_mut(&prunable_id)
        .unwrap()
        .bind_output(&mon_output);
    layout.active_view_mut(&mon_out).insert(1, prunable_id);

    // Call `clean_up_workspaces_on` directly.
    let pruned = {
        let (monitors, pool, view) = layout.monitors_pool_view_mut(&mon_out);
        Layout::<TestWindow>::clean_up_workspaces_on(monitors, pool, view, 0)
    };

    assert_eq!(
        pruned,
        vec![prunable_id],
        "main-loop branch must return pruned id",
    );
    assert!(
        layout.workspace_pool().contains_key(&prunable_id),
        "clean_up must leave the pool untouched — caller owns destroy",
    );
    assert!(
        !layout.active_view(&mon_out).ids().contains(&prunable_id),
        "view must no longer contain pruned id",
    );

    // Finish the contract so the layout stays consistent for `verify_invariants`.
    Layout::<TestWindow>::destroy_workspaces_cross_activity(
        &mut layout.activities,
        &mut layout.workspaces,
        pruned,
    );
    layout.verify_invariants();
}

// The `empty_workspace_above_first && view.len() == 2` branch past the main
// loop also contributes the second id to the returned `Vec`.
#[test]
fn clean_up_workspaces_on_returns_pruned_ids_empty_above_first() {
    // `add_output` already ran cleanup, so under EWAF a fresh output may
    // land with a single entry. Force the `view.len() == 2` branch: insert
    // a second empty (bottom) and move active onto it. Then the special-case
    // past the main loop prunes position 1.
    let options = Options {
        layout: jiji_config::Layout {
            empty_workspace_above_first: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut layout = check_ops_with_options(options, [Op::AddOutput(1)]);
    let mon_out = layout.monitors[0].output_id();
    let seed_activity = layout.active_activity_id();

    // Coerce the view into exactly two entries [E, E] if needed.
    let extra = Workspace::<TestWindow>::new_no_outputs(
        HashSet::from([seed_activity]),
        layout.clock.clone(),
        layout.options.clone(),
    );
    let extra_id = extra.id();
    layout.workspaces.insert(extra_id, extra);
    let mon_output = layout.monitors[0].output.clone();
    layout
        .workspaces
        .get_mut(&extra_id)
        .unwrap()
        .bind_output(&mon_output);
    {
        let view = layout.active_view_mut(&mon_out);
        // Append so active stays at pos 0 and the new id sits at pos 1.
        let end = view.len();
        view.insert(end, extra_id);
    }
    // `clean_up_workspaces_on`'s special branch asserts active is not
    // on position 1 (since `view.len() == 2`). Ensure active is at pos 0.
    assert_eq!(layout.active_view(&mon_out).active_position(), 0);
    assert_eq!(layout.active_view(&mon_out).len(), 2);

    let pruned = {
        let (monitors, pool, view) = layout.monitors_pool_view_mut(&mon_out);
        Layout::<TestWindow>::clean_up_workspaces_on(monitors, pool, view, 0)
    };

    assert!(
        pruned.contains(&extra_id),
        "EWAF special-case branch must contribute pruned id; got {pruned:?}",
    );
    assert!(
        layout.workspace_pool().contains_key(&extra_id),
        "clean_up must leave the pool untouched",
    );

    Layout::<TestWindow>::destroy_workspaces_cross_activity(
        &mut layout.activities,
        &mut layout.workspaces,
        pruned,
    );
    layout.verify_invariants();
}

// Two activities share a common output, each with its own view containing
// the destroyed workspace id. After `destroy_workspaces_cross_activity`,
// both views drop (single-entry) or patch (multi-entry) the id, and the
// workspace is gone from the pool.
#[test]
fn destroy_workspaces_cross_activity_patches_other_activity_views() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
    ];
    let mut layout = check_ops(ops);
    let mon_out = layout.monitors[0].output_id();
    let seed_activity = layout.active_activity_id();
    let mon_output = layout.monitors[0].output.clone();

    // Shape: `[W1 (active, pos 0), doomed_empty, E_bottom]` — doomed at a non-terminal
    // middle position so the retain closure takes the remove_at branch.
    let doomed_id = {
        let spare = Workspace::<TestWindow>::new_no_outputs(
            HashSet::from([seed_activity]),
            layout.clock.clone(),
            layout.options.clone(),
        );
        let id = spare.id();
        layout.workspaces.insert(id, spare);
        layout
            .workspaces
            .get_mut(&id)
            .expect("doomed spare must be in pool")
            .bind_output(&mon_output);
        id
    };
    layout.active_view_mut(&mon_out).insert(1, doomed_id);

    // Sanity-check the seeded shape.
    assert_eq!(layout.active_view(&mon_out).len(), 3);
    assert_eq!(layout.active_view(&mon_out).active_position(), 0);
    assert_eq!(layout.active_view(&mon_out).ids()[1], doomed_id);

    // Create a second activity (Beta) and seed its view so Beta's view also
    // contains `doomed_id` alongside a spare id. Beta's view must have at
    // least two entries so the retain closure takes the `remove_at` branch
    // (not the single-entry drop branch).
    let beta_id = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");
    let spare_id = {
        // Add a spare workspace to the pool under Beta's membership.
        let spare = Workspace::<TestWindow>::new_no_outputs(
            HashSet::from([beta_id]),
            layout.clock.clone(),
            layout.options.clone(),
        );
        let id = spare.id();
        layout.workspaces.insert(id, spare);
        layout
            .workspaces
            .get_mut(&id)
            .expect("spare must be in pool")
            .bind_output(&mon_output);
        id
    };
    // Seed Beta's view with a stale reference to doomed_id. doomed_id.activities
    // remains {Alpha} (exclusive), so the guard clears and destroy proceeds.
    // The retain closure patches Beta's view because it iterates every activity's
    // views regardless of workspace membership.
    test_override_activity_view(
        &mut layout,
        beta_id,
        mon_out.clone(),
        WorkspaceView::new(vec![doomed_id, spare_id], 0),
    );

    // Call destroy with `doomed_id`. Active activity's view contains it at
    // a non-terminal position (pos 1) and has multiple entries — it must be
    // removed_at. Beta's view has two entries — the retain closure's
    // remove_at branch fires there too.
    Layout::<TestWindow>::destroy_workspaces_cross_activity(
        &mut layout.activities,
        &mut layout.workspaces,
        vec![doomed_id],
    );

    assert!(
        !layout.workspace_pool().contains_key(&doomed_id),
        "doomed id must be gone from the pool",
    );
    assert!(
        !layout.active_view(&mon_out).ids().contains(&doomed_id),
        "active view must no longer contain doomed id",
    );
    let beta_view = layout
        .activities
        .get(beta_id)
        .expect("beta")
        .views()
        .get(&mon_out)
        .expect("beta view retained (multi-entry)");
    assert_eq!(beta_view.ids(), &[spare_id]);
    layout.verify_invariants();
}

// Single-entry Beta view containing the doomed id must be dropped entirely
// by the retain closure.
#[test]
fn destroy_workspaces_cross_activity_drops_single_entry_view() {
    let mut layout = check_ops([Op::AddOutput(1)]);
    let mon_out = layout.monitors[0].output_id();

    // On a fresh single-output layout `check_ops([AddOutput(1)])` produces
    // view `[E_bottom]` of length 1, so `view.ids()[0]` is both the last
    // element and the active position. When `destroy_workspaces_cross_activity`
    // runs, the retain closure sees `view.len() == 1` on Alpha's view and
    // drops it entirely — this collapses active_views to length 0 vs the
    // single connected monitor, which trips the domain-parity check inside
    // `verify_invariants`. That is the intended branch under test (the
    // single-entry drop), not a bug, so `verify_invariants` is intentionally
    // not called here.
    let doomed_id = {
        let view = layout.active_view(&mon_out);
        view.ids()[view.len() - 1]
    };

    let beta_id = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");
    // Seed Beta's view with a stale reference to doomed_id. doomed_id.activities
    // remains {Alpha} (exclusive), so the guard clears and destroy proceeds.
    // The retain closure patches Beta's view because it iterates every activity's
    // views regardless of workspace membership.
    test_override_activity_view(
        &mut layout,
        beta_id,
        mon_out.clone(),
        WorkspaceView::new(vec![doomed_id], 0),
    );

    Layout::<TestWindow>::destroy_workspaces_cross_activity(
        &mut layout.activities,
        &mut layout.workspaces,
        vec![doomed_id],
    );

    assert!(
        !layout.workspace_pool().contains_key(&doomed_id),
        "doomed id must be gone from the pool",
    );
    assert!(
        layout
            .activities
            .get(beta_id)
            .expect("beta")
            .views()
            .get(&mon_out)
            .is_none(),
        "single-entry beta view must be dropped",
    );
}

// Baseline for the shared-workspace cleanup rule: a workspace owned
// by exactly one activity is safe to reclaim.
#[test]
fn workspace_is_safe_to_reclaim_true_for_single_activity_membership() {
    let layout = check_ops([Op::AddOutput(1)]);
    let mon_out = layout.monitors[0].output_id();
    let view = layout.active_view(&mon_out);
    let id = view.ids()[view.len() - 1];

    // A fresh `check_ops([AddOutput(1)])` layout's bookend is exclusive to
    // the active (Alpha) activity; `activities().len() == 1`.
    assert_eq!(
        layout
            .workspaces
            .get(&id)
            .expect("bookend in pool")
            .activities()
            .len(),
        1,
        "precondition: bookend starts out exclusive to Alpha",
    );
    assert!(
        Layout::<TestWindow>::workspace_is_safe_to_reclaim(&layout.workspaces, id),
        "exclusive-membership workspace must be safe to reclaim",
    );
}

// Guard's primary correctness: a workspace shared between two activities
// must NOT be safe to reclaim — per the contract, "a workspace with
// `activities = {A, B}` that becomes empty is not removed".
#[test]
fn workspace_is_safe_to_reclaim_false_for_shared_membership() {
    let mut layout = check_ops([Op::AddOutput(1)]);
    let mon_out = layout.monitors[0].output_id();
    let id = {
        let view = layout.active_view(&mon_out);
        view.ids()[view.len() - 1]
    };
    let beta_id = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");

    // Mark the bookend as shared between Alpha and Beta.
    layout
        .workspaces
        .get_mut(&id)
        .expect("bookend in pool")
        .activities
        .insert(beta_id);
    assert_eq!(
        layout
            .workspaces
            .get(&id)
            .expect("bookend in pool")
            .activities()
            .len(),
        2,
        "precondition: bookend is now shared across two activities",
    );

    assert!(
        !Layout::<TestWindow>::workspace_is_safe_to_reclaim(&layout.workspaces, id),
        "shared-membership workspace must NOT be safe to reclaim",
    );
    // verify_invariants intentionally skipped: bookend.activities is widened
    // without a matching Beta view entry.
}

// Defensive branch: an id not present in the pool returns `false`. The
// authoritative panic for genuinely dead ids lives in
// `destroy_workspaces_cross_activity`'s `pool.remove.is_some()` assert;
// returning `false` here simply keeps the predicate total.
#[test]
fn workspace_is_safe_to_reclaim_false_for_absent_id() {
    let layout = check_ops([Op::AddOutput(1)]);

    // Manufacture an id that cannot possibly be in the pool by allocating a
    // throwaway workspace and dropping it — its `WorkspaceId` is unique and
    // never inserted.
    let absent_id = Workspace::<TestWindow>::new_no_outputs(
        HashSet::from([layout.active_activity_id()]),
        layout.clock.clone(),
        layout.options.clone(),
    )
    .id();
    assert!(
        !layout.workspaces.contains_key(&absent_id),
        "precondition: absent_id must not be in the pool",
    );

    assert!(
        !Layout::<TestWindow>::workspace_is_safe_to_reclaim(&layout.workspaces, absent_id),
        "id absent from pool must return false (defensive branch)",
    );
}

// End-to-end absent-id panic: an id not present in the pool must not be
// silently skipped by `destroy_workspaces_cross_activity` — it must reach the
// `pool.remove` assert and panic. Absent ids are a caller bug; the assert is
// the authoritative diagnostic.
#[test]
#[should_panic(expected = "must be a live pool key")]
fn destroy_workspaces_cross_activity_panics_for_absent_id() {
    let mut layout = check_ops([Op::AddOutput(1)]);

    // Manufacture an id that cannot possibly be in the pool.
    let absent_id = Workspace::<TestWindow>::new_no_outputs(
        HashSet::from([layout.active_activity_id()]),
        layout.clock.clone(),
        layout.options.clone(),
    )
    .id();
    assert!(
        !layout.workspaces.contains_key(&absent_id),
        "precondition: absent_id must not be in the pool",
    );

    // Must panic — absent ids are not skipped; the pool.remove assert fires.
    Layout::<TestWindow>::destroy_workspaces_cross_activity(
        &mut layout.activities,
        &mut layout.workspaces,
        vec![absent_id],
    );
}

// End-to-end skip: when `destroy_workspaces_cross_activity` is handed an id
// that fails the guard (shared membership), pool and every activity's views
// must be unchanged, and `verify_invariants` must still hold.
#[test]
fn destroy_workspaces_cross_activity_skips_shared_id_keeps_pool_and_views_intact() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
    ];
    let mut layout = check_ops(ops);
    let mon_out = layout.monitors[0].output_id();
    let mon_output = layout.monitors[0].output.clone();
    let seed_activity = layout.active_activity_id();

    // Seed a shared workspace that both Alpha (active) and Beta reference.
    // Shape Alpha's view as `[W1 (active), shared, E_bottom]` so `shared`
    // sits at a non-terminal middle position. `shared` is named so that
    // `has_windows_or_name()` returns `true`, clearing the non-terminal-empty
    // bookend check inside `Monitor::verify_invariants` (monitor.rs:1772).
    let shared_id = {
        let spare = Workspace::<TestWindow>::new_with_config_no_outputs(
            Some(WorkspaceConfig {
                name: WorkspaceName("shared".to_owned()),
                open_on_output: None,
                layout: None,
                activities: Vec::new(),
                sticky: None,
            }),
            HashSet::from([seed_activity]),
            layout.clock.clone(),
            layout.options.clone(),
        );
        let id = spare.id();
        layout.workspaces.insert(id, spare);
        layout
            .workspaces
            .get_mut(&id)
            .expect("shared spare must be in pool")
            .bind_output(&mon_output);
        id
    };
    layout.active_view_mut(&mon_out).insert(1, shared_id);

    let beta_id = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");
    layout
        .workspaces
        .get_mut(&shared_id)
        .expect("shared in pool")
        .activities
        .insert(beta_id);
    // Beta view length 2 so it would hit `remove_at` if the skip misfired.
    let beta_spare_id = {
        let spare = Workspace::<TestWindow>::new_no_outputs(
            HashSet::from([beta_id]),
            layout.clock.clone(),
            layout.options.clone(),
        );
        let id = spare.id();
        layout.workspaces.insert(id, spare);
        layout
            .workspaces
            .get_mut(&id)
            .expect("beta spare must be in pool")
            .bind_output(&mon_output);
        id
    };
    test_override_activity_view(
        &mut layout,
        beta_id,
        mon_out.clone(),
        WorkspaceView::new(vec![shared_id, beta_spare_id], 0),
    );

    // Snapshot the shapes that must not change after the (no-op) destroy.
    let alpha_view_before: Vec<_> = layout.active_view(&mon_out).ids().to_vec();
    let beta_view_before: Vec<_> = layout
        .activities
        .get(beta_id)
        .expect("beta")
        .views()
        .get(&mon_out)
        .expect("beta view")
        .ids()
        .to_vec();
    let pool_size_before = layout.workspaces.len();

    // Precondition: the guard must say "skip".
    assert!(
        !Layout::<TestWindow>::workspace_is_safe_to_reclaim(&layout.workspaces, shared_id),
        "precondition: shared id must fail the guard",
    );

    Layout::<TestWindow>::destroy_workspaces_cross_activity(
        &mut layout.activities,
        &mut layout.workspaces,
        vec![shared_id],
    );

    assert_eq!(
        layout.workspaces.len(),
        pool_size_before,
        "pool size must be unchanged after a skipped destroy",
    );
    assert!(
        layout.workspaces.contains_key(&shared_id),
        "shared id must still be in the pool after a skipped destroy",
    );
    assert_eq!(
        layout.active_view(&mon_out).ids(),
        alpha_view_before.as_slice(),
        "Alpha's view must be unchanged after a skipped destroy",
    );
    assert_eq!(
        layout
            .activities
            .get(beta_id)
            .expect("beta")
            .views()
            .get(&mon_out)
            .expect("beta view")
            .ids(),
        beta_view_before.as_slice(),
        "Beta's view must be unchanged after a skipped destroy",
    );
    layout.verify_invariants();
}

// Per-id guard: a batch containing one shared (skipped) id alongside one
// exclusive (doomed) id must reclaim only the exclusive sibling. Pins that
// the skip is per-iteration, not a batch-level short-circuit.
#[test]
fn destroy_workspaces_cross_activity_mixed_batch_drops_only_safe_id() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
    ];
    let mut layout = check_ops(ops);
    let mon_out = layout.monitors[0].output_id();
    let mon_output = layout.monitors[0].output.clone();
    let seed_activity = layout.active_activity_id();

    // Two spares, both seeded into Alpha's view between W1 and E_bottom:
    // `[W1 (active, pos 0), shared, doomed, E_bottom]`. `shared` is named so
    // that `has_windows_or_name()` returns `true`, clearing the
    // non-terminal-empty bookend check inside `Monitor::verify_invariants`
    // (monitor.rs:1772).
    let shared_id = {
        let spare = Workspace::<TestWindow>::new_with_config_no_outputs(
            Some(WorkspaceConfig {
                name: WorkspaceName("shared".to_owned()),
                open_on_output: None,
                layout: None,
                activities: Vec::new(),
                sticky: None,
            }),
            HashSet::from([seed_activity]),
            layout.clock.clone(),
            layout.options.clone(),
        );
        let id = spare.id();
        layout.workspaces.insert(id, spare);
        layout
            .workspaces
            .get_mut(&id)
            .expect("shared spare must be in pool")
            .bind_output(&mon_output);
        id
    };
    let doomed_id = {
        let spare = Workspace::<TestWindow>::new_no_outputs(
            HashSet::from([seed_activity]),
            layout.clock.clone(),
            layout.options.clone(),
        );
        let id = spare.id();
        layout.workspaces.insert(id, spare);
        layout
            .workspaces
            .get_mut(&id)
            .expect("doomed spare must be in pool")
            .bind_output(&mon_output);
        id
    };
    layout.active_view_mut(&mon_out).insert(1, shared_id);
    layout.active_view_mut(&mon_out).insert(2, doomed_id);

    // Widen `shared_id` to Beta's membership so it fails the guard. Leave
    // `doomed_id` exclusive to Alpha so the guard clears it. Beta does not
    // need a view entry on this output — cross-activity view patching is
    // covered by `destroy_workspaces_cross_activity_patches_other_activity_views`.
    let beta_id = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");
    layout
        .workspaces
        .get_mut(&shared_id)
        .expect("shared in pool")
        .activities
        .insert(beta_id);

    Layout::<TestWindow>::destroy_workspaces_cross_activity(
        &mut layout.activities,
        &mut layout.workspaces,
        vec![shared_id, doomed_id],
    );

    assert!(
        layout.workspaces.contains_key(&shared_id),
        "shared id must survive (skipped by guard)",
    );
    assert!(
        !layout.workspaces.contains_key(&doomed_id),
        "doomed id must be removed (passed guard)",
    );
    let alpha_ids = layout.active_view(&mon_out).ids().to_vec();
    assert!(
        alpha_ids.contains(&shared_id),
        "Alpha's view must still reference shared_id",
    );
    assert!(
        !alpha_ids.contains(&doomed_id),
        "Alpha's view must no longer reference doomed_id",
    );
    layout.verify_invariants();
}

// Regression: `remove_output` must flush doomed empty bookends through
// `destroy_workspaces_cross_activity` so that other activities' views that
// reference those bookends are also patched. Without the flush the pool and
// view fall out of sync and `verify_invariants` trips.
#[test]
fn take_workspace_ids_doomed_flushed_from_other_activity_view_on_remove_output() {
    // One output, one window. Alpha's view: `[W1 (active, pos 0), E_bottom]`.
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
    ];
    let mut layout = check_ops(ops);
    let mon_out = layout.monitors[0].output_id();
    let mon_output = layout.monitors[0].output.clone();

    // Grab the trailing empty bookend id — this is what `take_workspace_ids`
    // will peel off as a doomed id when the output disconnects.
    let alpha_view = layout.active_view(&mon_out);
    let e_bottom_id = *alpha_view
        .ids()
        .last()
        .expect("view must have a trailing bookend");
    assert!(
        !layout
            .workspaces
            .get(&e_bottom_id)
            .expect("bookend in pool")
            .has_windows_or_name(),
        "bookend must be empty and unnamed",
    );

    // Create Beta and seed its view with [extra_spare, e_bottom_id] so the
    // bookend appears in a second activity's view. The extra spare keeps Beta's
    // view length >= 2 so the doomed path hits remove_at, not the single-entry
    // drop (that branch is covered by the `drops_single_entry_view` test).
    let beta_id = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");
    let beta_spare_id = {
        let spare = Workspace::<TestWindow>::new_no_outputs(
            HashSet::from([beta_id]),
            layout.clock.clone(),
            layout.options.clone(),
        );
        let id = spare.id();
        layout.workspaces.insert(id, spare);
        layout
            .workspaces
            .get_mut(&id)
            .expect("spare must be in pool")
            .bind_output(&mon_output);
        id
    };
    // Seed Beta's view with a stale reference to doomed_id. doomed_id.activities
    // remains {Alpha} (exclusive), so the guard clears and destroy proceeds.
    // The retain closure patches Beta's view because it iterates every activity's
    // views regardless of workspace membership.
    test_override_activity_view(
        &mut layout,
        beta_id,
        mon_out.clone(),
        WorkspaceView::new(vec![beta_spare_id, e_bottom_id], 0),
    );

    // Disconnect the output. `remove_output` → `take_workspace_ids` returns
    // `e_bottom_id` as a doomed bookend and must flush it through
    // `destroy_workspaces_cross_activity`, which patches Beta's view.
    layout.remove_output(&mon_output);

    // Pool must not contain the doomed bookend.
    assert!(
        !layout.workspace_pool().contains_key(&e_bottom_id),
        "doomed bookend must be gone from the pool after remove_output",
    );
    // beta_spare_id was also empty and unnamed — it must be doomed and gone.
    assert!(
        !layout.workspace_pool().contains_key(&beta_spare_id),
        "beta's empty unnamed spare must be doomed and removed from the pool after remove_output",
    );
    // Beta's view must have e_bottom_id patched out.
    let beta_view_ids: Vec<_> = layout
        .activities
        .get(beta_id)
        .expect("beta must exist")
        .views()
        .values()
        .flat_map(|v| v.ids().iter().copied())
        .collect();
    assert!(
        !beta_view_ids.contains(&e_bottom_id),
        "beta's view must not contain the doomed bookend after remove_output",
    );
    layout.verify_invariants();
}

// `remove_output` no-monitors-left: a dormant activity's view that contains a
// window-bearing workspace must land that workspace in `disconnected_workspace_ids`
// (the "kept" branch), not in `doomed_ids`. Pin that (a) the workspace survives in
// the pool, (b) its options are reset to layout-root, and (c) `unbind_output` was
// called (C1 fix — dormant workspaces were previously missing this).
#[test]
fn remove_last_output_classifies_dormant_view_kept_leg() {
    // One output. Alpha's view: `[W1 (active, window-bearing), E_bottom]`.
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
    ];
    let mut layout = check_ops(ops);
    let mon_out = layout.monitors[0].output_id();
    let mon_output = layout.monitors[0].output.clone();

    // Create Beta. The materializer gives it a fresh bookend view for output1.
    let beta_id = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");

    // Find the window-bearing workspace in alpha's active view.
    let alpha_view = layout.active_view(&mon_out).ids().to_vec();
    let w1_id = alpha_view[0];
    assert!(
        layout
            .workspaces
            .get(&w1_id)
            .expect("w1 in pool")
            .has_windows(),
        "precondition: w1 must be window-bearing",
    );

    // Widen w1's activities to include Beta so the pool entry carries {Alpha, Beta}.
    layout
        .workspaces
        .get_mut(&w1_id)
        .expect("w1 in pool")
        .activities
        .insert(beta_id);

    // Hand-roll Beta's view: [w1_id, beta_bookend] — w1_id is window-bearing and at
    // position 0 (non-trailing), bookend is trailing. This satisfies the bookend
    // invariant while placing a window-bearing workspace in Beta's dormant view.
    let beta_bookend_id = test_mint_empty_for(&mut layout, 0, beta_id);
    test_override_activity_view(
        &mut layout,
        beta_id,
        mon_out.clone(),
        WorkspaceView::new(vec![w1_id, beta_bookend_id], 1),
    );

    // Disconnect the only output.
    layout.remove_output(&mon_output);

    // w1 is window-bearing — it must have survived into disconnected_workspace_ids.
    assert!(
        layout.disconnected_workspace_ids.contains(&w1_id),
        "window-bearing dormant workspace must join disconnected_workspace_ids on remove_output",
    );
    assert!(
        layout.workspace_pool().contains_key(&w1_id),
        "window-bearing dormant workspace must remain in the pool",
    );

    // Options must have been reset to layout-root (same as active-view kept ids).
    assert_eq!(
        layout
            .workspaces
            .get(&w1_id)
            .expect("w1 in pool")
            .base_options,
        layout.options,
        "kept dormant workspace options must be reset to layout-root after remove_output",
    );

    // beta_bookend was empty and unnamed — it must be doomed and removed.
    assert!(
        !layout.workspace_pool().contains_key(&beta_bookend_id),
        "empty unnamed dormant bookend must be doomed and gone after remove_output",
    );

    layout.verify_invariants();
}

// `advance_animations` runs `clean_up_workspaces_on` inside a triple-borrow
// scope and must flush all pruned ids through `destroy_workspaces_cross_activity`
// after the scope closes — not inline per iteration, which would re-borrow
// `&mut self.activities` and fight the borrow checker.
//
// This test pins that post-loop flush by checking that a completed
// workspace-switch animation triggers the cleanup and leaves `verify_invariants`
// satisfied. A multi-monitor accumulate shape (ids from two monitors folded
// before one flush) is exercised indirectly by the animation-heavy proptests;
// the borrow-checker enforces the accumulate structure anyway.
#[test]
fn advance_animations_destroy_flushes_after_loop_scope() {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::FocusWorkspaceDown,
        Op::AddWindow {
            params: TestWindowParams::new(2),
        },
        Op::FocusWorkspaceUp,
    ];
    let mut layout = check_ops(ops);
    // Advance until any workspace_switch animation finishes; destroy flush
    // must run once and leave invariants intact.
    for _ in 0..200 {
        layout.advance_animations();
        if !layout.are_animations_ongoing(None) {
            break;
        }
    }
    layout.verify_invariants();
}

// ---------------------------------------------------------------------------
// cross-activity window iteration helpers.
// Pin the activity-scoped default of `Layout::windows()` / `Layout::workspaces()`
// against the pool-spanning `windows_all()` / `with_windows_all` /
// `with_windows_all_mut`. The `WindowMru::new` call site at
// `src/ui/mru.rs:587` iterates `layout.workspaces()` and is activity-scoped
// by the same contract these tests pin — see the rustdoc on
// `Layout::workspaces` / `Layout::windows` for the consumer-facing statement.
// ---------------------------------------------------------------------------

/// Seeds a two-activity layout with one window on each activity:
/// - seed activity gets window id 1 on a single connected output;
/// - beta activity gets window id 2 on the same output (after switching to beta so `Op::AddWindow`
///   lands on beta's active workspace).
///
/// Returns the layout with the active activity switched back to `seed`.
fn seed_two_activities_with_one_window_each() -> (Layout<TestWindow>, ActivityId, ActivityId) {
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
    ];
    let mut layout = check_ops(ops);
    let seed_id = layout.active_activity_id();

    let beta = super::activity::Activity::new_runtime("beta".to_owned());
    let beta_id = beta.id();
    test_insert_activity(&mut layout, beta);

    layout.switch_activity(beta_id);
    check_ops_on_layout(
        &mut layout,
        [Op::AddWindow {
            params: TestWindowParams::new(2),
        }],
    );

    layout.switch_activity(seed_id);
    layout.verify_invariants();
    (layout, seed_id, beta_id)
}

#[test]
fn windows_activity_scoped_excludes_hidden_activity_windows() {
    // Pins the "activity-filtered default" contract of `Layout::windows()` —
    // the call site `input/mod.rs:882` (`Action::FocusWindowPrevious`) and
    // `ui/mru.rs:587` (`WindowMru::new` iterating `layout.workspaces()`)
    // both rely on this scoping to restrict focus-cycling and MRU display
    // to the active activity.
    let (layout, _seed_id, _beta_id) = seed_two_activities_with_one_window_each();

    let visible_ids: Vec<usize> = layout.windows().map(|(_, w)| *w.id()).collect();
    assert_eq!(
        visible_ids,
        vec![1],
        "windows() must be activity-scoped: beta's window (id=2) is hidden",
    );

    // Corollary at the workspaces-iterator level (pins the MRU scope
    // contract from `ui/mru.rs:587`): `layout.workspaces()` walks only the
    // active activity's views.
    let visible_ws_windows: Vec<usize> = layout
        .workspaces()
        .flat_map(|(_, _, ws)| ws.windows().map(|w| *w.id()))
        .collect();
    assert_eq!(
        visible_ws_windows,
        vec![1],
        "workspaces() must be activity-scoped: the MRU builder only sees the \
         active activity's windows",
    );
}

#[test]
fn windows_all_spans_pool_across_activities() {
    // Pool-spanning iteration: every window in the pool must be reachable
    // via `windows_all()`, regardless of which activity owns its workspace.
    let (layout, _seed_id, _beta_id) = seed_two_activities_with_one_window_each();

    let mut all_ids: Vec<usize> = layout.windows_all().map(|(_, w)| *w.id()).collect();
    all_ids.sort_unstable();
    assert_eq!(
        all_ids,
        vec![1, 2],
        "windows_all() must yield every window in the pool, including \
         windows on dormant activities",
    );

    // Count matches the pool total — no hidden windows dropped silently.
    let pool_total: usize = layout
        .workspace_pool()
        .values()
        .map(|ws| ws.windows().count())
        .sum();
    assert_eq!(layout.windows_all().count(), pool_total);
}

#[test]
fn windows_all_output_id_follows_workspace_binding() {
    // The `Option<&OutputId>` paired with each window must be the
    // owning workspace's bound output id; a hidden-activity window on a
    // connected output is therefore `Some(oid)`, not `None`.
    let (layout, _seed_id, _beta_id) = seed_two_activities_with_one_window_each();
    let expected_oid = layout.monitors[0].output_id();

    for (oid, win) in layout.windows_all() {
        let oid = oid.expect("all pool workspaces here are bound to the single connected output");
        assert_eq!(
            *oid,
            expected_oid,
            "windows_all must pair each window with its owning workspace's \
             bound output id (window id {:?})",
            win.id(),
        );
    }
}

#[test]
fn with_windows_all_yields_hidden_activity_windows() {
    // The closure API must visit every pool window, including those on a
    // dormant activity's workspaces. Pins the IPC event-stream /
    // foreign-toplevel / screencasting consumer contract.
    let (layout, _seed_id, _beta_id) = seed_two_activities_with_one_window_each();
    let expected_output = layout.monitors[0].output.clone();

    let mut seen: Vec<(usize, bool, bool)> = Vec::new();
    layout.with_windows_all(|win, output, ws_id, _| {
        let has_output = output.is_some_and(|o| *o == expected_output);
        let has_ws_id = ws_id.is_some();
        seen.push((*win.id(), has_output, has_ws_id));
    });
    seen.sort_by_key(|(id, _, _)| *id);

    assert_eq!(
        seen,
        vec![(1, true, true), (2, true, true)],
        "with_windows_all must yield both seed's and beta's windows with a \
         resolved &Output and a populated workspace id",
    );
}

#[test]
fn with_windows_all_mut_yields_hidden_activity_windows() {
    // Mutable twin: the two-phase borrow recipe (pre-hoisted monitor map,
    // then `values_mut()`) must still reach every pool window. Asserts the
    // closure observes both windows with `&mut W` and pairs each with the
    // bound `&Output` resolved through the hoisted map.
    let (mut layout, _seed_id, _beta_id) = seed_two_activities_with_one_window_each();
    let expected_output = layout.monitors[0].output.clone();

    let mut seen: Vec<(usize, bool)> = Vec::new();
    layout.with_windows_all_mut(|win, output| {
        let has_output = output.is_some_and(|o| *o == expected_output);
        // `win` is taken as `&mut`; we only record the id, which is enough
        // to confirm the exclusive-borrow signature and that every pool
        // window is reachable.
        seen.push((*win.id(), has_output));
    });
    seen.sort_by_key(|(id, _)| *id);

    assert_eq!(
        seen,
        vec![(1, true), (2, true)],
        "with_windows_all_mut must yield both activities' windows, each \
         paired with the bound &Output resolved through the pre-hoisted \
         monitor map",
    );

    layout.verify_invariants();
}

#[test]
fn with_windows_all_mut_mutation_persists() {
    // Pins that `&mut W` received in the closure actually reaches the stored
    // window — i.e. mutations land, not just that the window is visited.
    // Uses `set_activated(true)` as the observable mutation, then re-iterates
    // via `windows_all()` to confirm every window reports the flipped state.
    let (mut layout, _seed_id, _beta_id) = seed_two_activities_with_one_window_each();

    // Flip every window to activated = true via the mutable iterator.
    layout.with_windows_all_mut(|win, _output| {
        win.set_activated(true);
    });

    // Re-iterate via the immutable path and assert the mutation persisted.
    let not_activated: Vec<usize> = layout
        .windows_all()
        .filter_map(|(_, w)| {
            if w.0.pending_activated.get() {
                None
            } else {
                Some(*w.id())
            }
        })
        .collect();

    assert!(
        not_activated.is_empty(),
        "all windows must have pending_activated = true after with_windows_all_mut flip; \
         these ids were not activated: {not_activated:?}",
    );

    layout.verify_invariants();
}

#[test]
fn windows_all_interactive_move_first() {
    // The interactive-move window must be the first item yielded by
    // `windows_all()`, `with_windows_all`, and `with_windows_all_mut` once
    // the move is in `Moving` state (Begin + Update transitions the state
    // machine from `Starting` to `Moving`, which is when iterators prepend).
    // Seeds two activities with one window each (ids 1 and 2), arms an
    // interactive move on window 1, and verifies first-yield ordering.
    let (mut layout, _seed_id, _beta_id) = seed_two_activities_with_one_window_each();

    // Arm a move: Begin + Update to reach `InteractiveMoveState::Moving`.
    // dx must exceed INTERACTIVE_MOVE_START_THRESHOLD^0.5 (= 256) for a
    // non-floating window so the state machine actually transitions from
    // `Starting` to `Moving`; a sub-threshold delta returns early and
    // `windows_all()` would not prepend the window.
    check_ops_on_layout(
        &mut layout,
        [
            Op::InteractiveMoveBegin {
                window: 1,
                output_idx: 1,
                px: 0.,
                py: 0.,
            },
            Op::InteractiveMoveUpdate {
                window: 1,
                dx: 300.,
                dy: 0.,
                output_idx: 1,
                px: 300.,
                py: 0.,
            },
        ],
    );

    // windows_all() — first element must be the moving window.
    let first_all = layout
        .windows_all()
        .next()
        .map(|(_, w)| *w.id())
        .expect("windows_all must yield at least one element during interactive move");
    assert_eq!(
        first_all, 1,
        "windows_all must yield the interactive-move window first",
    );

    // with_windows_all — first callback arg must be the moving window.
    let mut first_with_all: Option<usize> = None;
    layout.with_windows_all(|win, _output, _ws_id, _layout| {
        if first_with_all.is_none() {
            first_with_all = Some(*win.id());
        }
    });
    assert_eq!(
        first_with_all,
        Some(1),
        "with_windows_all must yield the interactive-move window first",
    );

    // with_windows_all_mut — first callback arg must be the moving window.
    let mut first_with_all_mut: Option<usize> = None;
    layout.with_windows_all_mut(|win, _output| {
        if first_with_all_mut.is_none() {
            first_with_all_mut = Some(*win.id());
        }
    });
    assert_eq!(
        first_with_all_mut,
        Some(1),
        "with_windows_all_mut must yield the interactive-move window first",
    );

    layout.interactive_move_end(&1);
    layout.verify_invariants();
}

#[test]
fn windows_all_disconnected_pool_yields_stale_output_id() {
    // After `RemoveOutput`, the workspace moves to `disconnected_workspace_ids`
    // but retains its bound `output_id` (needed for reconnect routing). The
    // corrected rustdoc for `windows_all` says: `None` is only for workspaces
    // never attached to any output; a workspace whose bound output is
    // disconnected still yields `Some(&oid)` with the stale OutputId.
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
    ];
    let mut layout = check_ops(ops);

    // Capture the output id before removal.
    let stale_oid = layout.monitors[0].output_id();

    // Remove the only output — workspace goes to disconnected_workspace_ids.
    check_ops_on_layout(&mut layout, [Op::RemoveOutput(1)]);

    assert!(
        layout.monitors.is_empty(),
        "no monitors must remain after RemoveOutput",
    );

    // The window should still appear in windows_all(), paired with Some(stale_oid).
    let mut found = false;
    for (oid, win) in layout.windows_all() {
        if *win.id() == 1 {
            found = true;
            let oid = oid.expect(
                "disconnected workspace retains its bound output_id; \
                 windows_all must yield Some(stale_oid), not None",
            );
            assert_eq!(
                *oid, stale_oid,
                "disconnected workspace's output id must match the stale OutputId \
                 of the removed output (window id 1)",
            );
        }
    }
    assert!(
        found,
        "window 1 must be reachable via windows_all after RemoveOutput",
    );

    layout.verify_invariants();
}

// -----------------------------------------------------------------------------
// Refresh-shape tests for activity mutation flows (`create_activity`,
// `remove_activity`, `rename_activity`, cascade-remove-active).
//
// These exercise the refresh-method shape against a live `Layout` via the
// test-only helper `crate::ipc::server::test_diff_activities_against_state`,
// which reproduces the combined lifecycle + active + urgency diff pass that
// `ipc_refresh_layout` performs against `EventStreamState::activities` on a
// single tick. Semantic coverage for single-snapshot apply-path cases lives
// in `jiji-ipc/src/state.rs`; this tier only pins the interaction between
// `Layout` mutations and the refresh emission order.
// -----------------------------------------------------------------------------

/// Build the client-tier snapshot that `EventStreamState::activities` would
/// hold after a full `ipc_refresh_layout` tick against `layout`.
fn activities_state_snapshot_from_layout<W: LayoutElement>(
    layout: &Layout<W>,
) -> HashMap<u64, jiji_ipc::Activity> {
    crate::ipc::server::build_activities_ipc(layout)
        .into_iter()
        .map(|a| (a.id, a))
        .collect()
}

#[test]
fn create_activity_refresh_emits_activity_created() {
    // Seed the refresh snapshot from the initial pool, perform
    // `create_activity`, diff again — exactly one `ActivityCreated` for the
    // new id with the expected derived-field shape.
    //
    // Also exercises the `is_urgent=true` branch: after establishing the
    // "Work" snapshot, a second activity is created, switched to, and given
    // an urgent window before the next diff — the payload must carry
    // `is_urgent=true`, guarding the derived-field rebuild in the refresh
    // method against staling from a prior snapshot.
    let mut layout = Layout::<TestWindow>::default();

    let seed = activities_state_snapshot_from_layout(&layout);

    let new_id = layout
        .create_activity("Work".to_owned())
        .expect("valid name must succeed");

    let events = crate::ipc::server::test_diff_activities_against_state(&layout, &seed);
    assert_eq!(events.len(), 1, "got {events:?}");
    let jiji_ipc::Event::ActivityCreated { activity } = &events[0] else {
        panic!("expected ActivityCreated, got {:?}", events[0]);
    };
    assert_eq!(activity.id, new_id.get());
    assert_eq!(activity.name, "Work");
    assert!(!activity.is_config_declared);
    assert!(
        !activity.is_active,
        "create does not flip the active cursor",
    );
    assert!(
        !activity.is_urgent,
        "fresh runtime activity has no urgent windows"
    );

    // Extended coverage: is_urgent=true branch. Re-seed from the current
    // pool (seeded at {default, Work}), create "Urgent", switch to it, add an
    // urgent window, then diff — payload must report is_urgent=true.
    let reseed = activities_state_snapshot_from_layout(&layout);
    let urgent_id = layout
        .create_activity("Urgent".to_owned())
        .expect("valid name must succeed");
    layout.switch_activity(urgent_id);
    let win = TestWindow::new(TestWindowParams::new(42));
    layout.add_window(
        win.clone(),
        AddWindowTarget::Auto,
        None,
        None,
        false,
        false,
        ActivateWindow::default(),
    );
    win.set_urgent(true);
    let events2 = crate::ipc::server::test_diff_activities_against_state(&layout, &reseed);
    // Expect: ActivityCreated(Urgent) + ActivitySwitched(Urgent).
    // The Urgent activity starts with is_urgent=true in the Created payload
    // because the urgent window is already attached when we snapshot.
    let created: Vec<_> = events2
        .iter()
        .filter_map(|e| match e {
            jiji_ipc::Event::ActivityCreated { activity } => Some(activity),
            _ => None,
        })
        .collect();
    assert_eq!(
        created.len(),
        1,
        "exactly one ActivityCreated, got {events2:?}"
    );
    let urgent_activity = created[0];
    assert_eq!(urgent_activity.id, urgent_id.get());
    assert!(
        urgent_activity.is_urgent,
        "ActivityCreated payload must carry is_urgent=true when the activity already has urgent windows",
    );
}

#[test]
fn remove_activity_refresh_emits_activity_removed() {
    // Seed, create "Work", re-seed refresh snapshot, remove "Work", diff —
    // exactly one `ActivityRemoved { id }` where id matches the removed
    // activity.
    let mut layout = Layout::<TestWindow>::default();
    let work_id = layout
        .create_activity("Work".to_owned())
        .expect("valid name must succeed");

    let seed = activities_state_snapshot_from_layout(&layout);

    layout
        .remove_activity(&ActivityReferenceArg::Id(work_id.get()))
        .expect("runtime activity with no exclusive content must remove cleanly");

    let events = crate::ipc::server::test_diff_activities_against_state(&layout, &seed);
    assert_eq!(events.len(), 1, "got {events:?}");
    assert!(matches!(
        events[0],
        jiji_ipc::Event::ActivityRemoved { id } if id == work_id.get(),
    ));
}

#[test]
fn rename_activity_refresh_emits_activity_renamed() {
    // Seed, create "Work", re-seed refresh snapshot, rename to "Office",
    // diff — exactly one `ActivityRenamed { id, name: "Office" }`.
    let mut layout = Layout::<TestWindow>::default();
    let work_id = layout
        .create_activity("Work".to_owned())
        .expect("valid name must succeed");

    let seed = activities_state_snapshot_from_layout(&layout);

    layout
        .rename_activity(
            &ActivityReferenceArg::Id(work_id.get()),
            "Office".to_owned(),
        )
        .expect("valid rename");

    let events = crate::ipc::server::test_diff_activities_against_state(&layout, &seed);
    assert_eq!(events.len(), 1);
    let jiji_ipc::Event::ActivityRenamed { id, name } = &events[0] else {
        panic!("expected ActivityRenamed, got {:?}", events[0]);
    };
    assert_eq!(*id, work_id.get());
    assert_eq!(name, "Office");
}

#[test]
fn refresh_first_tick_seeds_silently_then_live_activities_emit_created_with_correct_derived_fields()
{
    // Regression pin for the first-tick seeding invariant: on a fresh
    // `EventStreamState::activities` (empty map), the combined refresh pass
    // emits `ActivityCreated` for every live activity — NOT `ActivitySwitched`
    // — and each `ActivityCreated` payload carries the correct derived
    // fields (`is_active`, `is_urgent`) sourced from the live `Layout`.
    //
    // Pin targets (see the spec Risks section):
    // - No spurious `ActivitySwitched { previous_id: None }` on first tick.
    // - `ActivityCreated` payload freshness: `to_ipc_activity` must read live `Layout` state, not
    //   reuse a stale snapshot.
    let mut layout = Layout::<TestWindow>::default();
    let seed_id = layout.active_activity_id();

    // Make the seed activity urgent so the payload-freshness pin bites:
    // add a window, flip it urgent.
    let win = TestWindow::new(TestWindowParams::new(7));
    layout.add_window(
        win.clone(),
        AddWindowTarget::Auto,
        None,
        None,
        false,
        false,
        ActivateWindow::default(),
    );
    win.set_urgent(true);
    assert!(
        layout.activity_is_urgent(seed_id),
        "precondition: seed activity is urgent",
    );

    // First-tick previous snapshot is empty.
    let previous: HashMap<u64, jiji_ipc::Activity> = HashMap::new();
    let events = crate::ipc::server::test_diff_activities_against_state(&layout, &previous);

    // No ActivitySwitched on the first tick.
    for event in &events {
        assert!(
            !matches!(event, jiji_ipc::Event::ActivitySwitched { .. }),
            "no ActivitySwitched must be emitted on the first (seeding) tick, got {event:?}",
        );
    }

    // Exactly one ActivityCreated for the seed activity.
    let created: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            jiji_ipc::Event::ActivityCreated { activity } => Some(activity),
            _ => None,
        })
        .collect();
    assert_eq!(
        created.len(),
        1,
        "exactly one ActivityCreated on first tick, got {events:?}",
    );
    let activity = created[0];
    assert_eq!(activity.id, seed_id.get());
    assert!(
        activity.is_active,
        "seed activity is the live active cursor — is_active must be true in the Created payload",
    );
    assert!(
        activity.is_urgent,
        "is_urgent must reflect live Layout state (urgent seed window), not a stale snapshot",
    );
}

#[test]
fn remove_active_activity_cascade_refresh_emits_removed_before_switch() {
    // Pins the event sequence produced by the cascade case — RemoveActivity
    // of the active activity re-points the cursor — at the refresh-method
    // tier:
    //
    //   [ActivityRemoved { id: B }, ActivitySwitched { id: A, previous_id: Some(B) }]
    //
    // The production call order (`lifecycle → active → urgency`) is guarded
    // by the block comment on `ipc_refresh_layout` (naming), not by
    // this test — the helper `test_diff_activities_against_state` bakes that
    // order in. Swapping the two `ipc_refresh_*` calls in production would
    // leave this test passing. Accepted trade-off: no `State`-level test
    // harness exists and the comment-only ordering guard mirrors sibling
    // refresh contracts.
    let mut layout = Layout::<TestWindow>::default();
    let alpha = layout.active_activity_id();

    let beta = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");
    layout.switch_activity(beta);
    assert_eq!(layout.active_activity_id(), beta);

    // Seed the refresh snapshot at this settled-pre-cascade state.
    let seed = activities_state_snapshot_from_layout(&layout);

    // Perform the cascade: remove the active beta. previous was alpha, so
    // the cursor re-points at alpha.
    layout
        .remove_activity(&ActivityReferenceArg::Id(beta.get()))
        .expect("active removal with previous cascade");
    assert_eq!(
        layout.active_activity_id(),
        alpha,
        "cascade target is alpha"
    );

    let all_events = crate::ipc::server::test_diff_activities_against_state(&layout, &seed);
    assert_eq!(all_events.len(), 2, "got {all_events:?}");
    assert!(
        matches!(&all_events[0], jiji_ipc::Event::ActivityRemoved { id } if *id == beta.get()),
        "first event must be ActivityRemoved for beta, got {:?}",
        all_events[0],
    );
    let jiji_ipc::Event::ActivitySwitched { id, previous_id } = &all_events[1] else {
        panic!(
            "second event must be ActivitySwitched, got {:?}",
            all_events[1]
        );
    };
    assert_eq!(*id, alpha.get());
    assert_eq!(*previous_id, Some(beta.get()));
}

#[test]
fn diff_activity_lifecycle_multi_kind_ordering() {
    // Pins the full Removed → Renamed → Created emission order when all three
    // buckets fire on the same tick. Seeded with four activities (alpha, beta,
    // gamma, delta). Previous snapshot represents: beta and delta absent (route
    // to Created), gamma with stale name (route to Renamed), two fake ids not
    // present in the layout (route to Removed). Diff yields:
    //   Removed {fake_epsilon_id=500, fake_zeta_id=999} — ascending sort
    //   Renamed {gamma_id, name: "Gamma2"} ← snapshot name differs
    //   Created {beta_id, delta_id} — ascending sort
    //
    // Two entries per Removed and Created bucket exercise within-bucket
    // ascending-id sort: without sort_unstable/sort_by_key the HashMap
    // iteration order is non-deterministic and the assertions fail.
    let mut layout = Layout::<TestWindow>::default();
    let alpha = layout.active_activity_id();

    let beta = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");
    let gamma = layout
        .create_activity("Gamma".to_owned())
        .expect("create gamma");
    let delta = layout
        .create_activity("Delta".to_owned())
        .expect("create delta");

    // Rename gamma in the layout so snapshot will see the old name.
    layout
        .rename_activity(&ActivityReferenceArg::Id(gamma.get()), "Gamma2".to_owned())
        .expect("rename gamma");

    // Build a synthetic previous snapshot:
    //   - alpha: present, name correct (no change)
    //   - beta: absent (will route to Created)
    //   - gamma: present with OLD name "Gamma" (will route to Renamed)
    //   - delta: absent (will route to Created; id > beta, so Created bucket is [beta, delta])
    //   - fake_epsilon (id=500): present in snapshot but not in layout (will route to Removed)
    //   - fake_zeta (id=999): present in snapshot but not in layout (will route to Removed)
    // Two Removed entries (500 < 999) and two Created entries (beta < delta) exercise
    // within-bucket ascending-id sort, catching any regression that drops the sort calls.
    let fake_epsilon_id: u64 = 500;
    let fake_zeta_id: u64 = 999;
    let mut previous: HashMap<u64, jiji_ipc::Activity> = HashMap::new();
    previous.insert(
        alpha.get(),
        jiji_ipc::Activity {
            id: alpha.get(),
            name: layout
                .activities()
                .iter()
                .find(|a| a.id() == alpha)
                .expect("alpha in pool")
                .name()
                .to_owned(),
            is_active: true,
            is_urgent: false,
            is_config_declared: false,
        },
    );
    previous.insert(
        gamma.get(),
        jiji_ipc::Activity {
            id: gamma.get(),
            name: "Gamma".to_owned(), // stale name — triggers Renamed
            is_active: false,
            is_urgent: false,
            is_config_declared: false,
        },
    );
    previous.insert(
        fake_epsilon_id,
        jiji_ipc::Activity {
            id: fake_epsilon_id,
            name: "Epsilon".to_owned(), // not in layout — triggers Removed
            is_active: false,
            is_urgent: false,
            is_config_declared: false,
        },
    );
    previous.insert(
        fake_zeta_id,
        jiji_ipc::Activity {
            id: fake_zeta_id,
            name: "Zeta".to_owned(), // not in layout — triggers Removed
            is_active: false,
            is_urgent: false,
            is_config_declared: false,
        },
    );

    let events = crate::ipc::server::test_diff_activities_against_state(&layout, &previous);

    // Extract events by kind in emission order.
    let removed: Vec<u64> = events
        .iter()
        .filter_map(|e| match e {
            jiji_ipc::Event::ActivityRemoved { id } => Some(*id),
            _ => None,
        })
        .collect();
    let renamed: Vec<(u64, &str)> = events
        .iter()
        .filter_map(|e| match e {
            jiji_ipc::Event::ActivityRenamed { id, name } => Some((*id, name.as_str())),
            _ => None,
        })
        .collect();
    let created: Vec<u64> = events
        .iter()
        .filter_map(|e| match e {
            jiji_ipc::Event::ActivityCreated { activity } => Some(activity.id),
            _ => None,
        })
        .collect();

    // All Removed events precede all Renamed events, which precede all Created.
    let removed_positions: Vec<usize> = events
        .iter()
        .enumerate()
        .filter_map(|(i, e)| matches!(e, jiji_ipc::Event::ActivityRemoved { .. }).then_some(i))
        .collect();
    let renamed_positions: Vec<usize> = events
        .iter()
        .enumerate()
        .filter_map(|(i, e)| matches!(e, jiji_ipc::Event::ActivityRenamed { .. }).then_some(i))
        .collect();
    let created_positions: Vec<usize> = events
        .iter()
        .enumerate()
        .filter_map(|(i, e)| matches!(e, jiji_ipc::Event::ActivityCreated { .. }).then_some(i))
        .collect();

    assert!(
        !removed_positions.is_empty()
            && !renamed_positions.is_empty()
            && !created_positions.is_empty(),
        "must have at least one event per kind, got {events:?}",
    );
    assert!(
        removed_positions
            .iter()
            .all(|r| renamed_positions.iter().all(|n| r < n)),
        "all Removed must precede all Renamed, got {events:?}",
    );
    assert!(
        renamed_positions
            .iter()
            .all(|n| created_positions.iter().all(|c| n < c)),
        "all Renamed must precede all Created, got {events:?}",
    );

    // Correct ids in each bucket, in ascending-id order (pins the sort).
    assert_eq!(
        removed,
        vec![fake_epsilon_id, fake_zeta_id],
        "Removed bucket must be sorted ascending: {removed:?}",
    );
    assert_eq!(
        renamed,
        vec![(gamma.get(), "Gamma2")],
        "Renamed bucket: {renamed:?}"
    );
    // beta was created before delta, so beta.get() < delta.get() by construction.
    // The sort in production must produce this order; if the sort is dropped the
    // HashMap iteration order may flip them, breaking this assertion non-deterministically.
    assert_eq!(
        created,
        vec![beta.get(), delta.get()],
        "Created bucket must be sorted ascending: {created:?}",
    );
}

#[test]
fn diff_activity_lifecycle_newcomer_routes_to_created_not_renamed() {
    // Pins that the lifecycle diff keys on id-presence, not name-match:
    // a new activity with the same *name* as a removed activity must route
    // to Created (for the new id) + Removed (for the old id), not Renamed.
    let mut layout = Layout::<TestWindow>::default();
    let alpha = layout.active_activity_id();

    // Create "Work" in the layout under a fresh id.
    let new_work_id = layout
        .create_activity("Work".to_owned())
        .expect("create Work");

    // Snapshot contains a *different* id (old_work_id) also named "Work" —
    // simulates a prior activity with the same name that was replaced.
    let old_work_id: u64 = 999; // not present in the live layout
    let mut previous: HashMap<u64, jiji_ipc::Activity> = HashMap::new();
    previous.insert(
        alpha.get(),
        jiji_ipc::Activity {
            id: alpha.get(),
            name: layout
                .activities()
                .iter()
                .find(|a| a.id() == alpha)
                .expect("alpha in pool")
                .name()
                .to_owned(),
            is_active: true,
            is_urgent: false,
            is_config_declared: false,
        },
    );
    previous.insert(
        old_work_id,
        jiji_ipc::Activity {
            id: old_work_id,
            name: "Work".to_owned(), // same name as the new activity, different id
            is_active: false,
            is_urgent: false,
            is_config_declared: false,
        },
    );

    let events = crate::ipc::server::test_diff_activities_against_state(&layout, &previous);

    // Must contain ActivityRemoved for the old id and ActivityCreated for
    // the new id — NOT ActivityRenamed.
    assert!(
        events
            .iter()
            .any(|e| matches!(e, jiji_ipc::Event::ActivityRemoved { id } if *id == old_work_id)),
        "expected ActivityRemoved for old id {old_work_id}, got {events:?}",
    );
    assert!(
        events.iter().any(|e| matches!(
            e,
            jiji_ipc::Event::ActivityCreated { activity } if activity.id == new_work_id.get()
        )),
        "expected ActivityCreated for new id {}, got {events:?}",
        new_work_id.get(),
    );
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, jiji_ipc::Event::ActivityRenamed { .. })),
        "must NOT emit ActivityRenamed when id-presence drives routing, got {events:?}",
    );
}

// ── Partial-disconnect dormant migration ──────────────────────────────────────

#[test]
fn switch_activity_dormant_view_survives_output_disconnect_and_reconnect() {
    // Beta's dormant workspaces migrate into beta's view for the still-connected primary
    // monitor on partial disconnect, then are reclaimed back into a fresh
    // `Activity.views[out1]` on reconnect by `ensure_view_for`'s lift branch.
    let ops = [Op::AddOutput(1), Op::AddOutput(2)];
    let mut layout = check_ops(ops);
    let seed_id = layout.active_activity_id();
    let out1 = layout.monitors[0].output_id();
    let out2 = layout.monitors[1].output_id();

    let beta = super::activity::Activity::new_runtime("beta".to_owned());
    let beta_id = beta.id();
    test_insert_activity(&mut layout, beta);

    // Switch to beta and populate its view on output1 with a non-bookend active workspace.
    layout.switch_activity(beta_id);
    layout.verify_invariants();

    // Move active to a middle position so the restore assertion has teeth.
    layout.add_window(
        TestWindow::new(TestWindowParams::new(10)),
        AddWindowTarget::Auto,
        None,
        None,
        false,
        false,
        ActivateWindow::default(),
    );
    layout.verify_invariants();
    layout.switch_workspace_down();
    layout.verify_invariants();
    layout.add_window(
        TestWindow::new(TestWindowParams::new(11)),
        AddWindowTarget::Auto,
        None,
        None,
        false,
        false,
        ActivateWindow::default(),
    );
    layout.verify_invariants();
    layout.switch_workspace_down();
    layout.verify_invariants();
    layout.switch_workspace_up();
    layout.verify_invariants();

    // Snapshot beta's active view for output1 before switching away.
    let beta_view_ids: Vec<WorkspaceId> = layout.active_view(&out1).ids().to_vec();
    // Identify the surviving (named or window-bearing) ids — these are the ones that
    // partial disconnect migrates to beta's view for the remaining monitor; the
    // trailing empty unnamed is doomed.
    let surviving_ids: Vec<WorkspaceId> = beta_view_ids
        .iter()
        .copied()
        .filter(|id| {
            layout
                .workspaces
                .get(id)
                .expect("snapshot ids must be live pool keys")
                .has_windows_or_name()
        })
        .collect();
    assert!(
        !surviving_ids.is_empty(),
        "test setup: pre-disconnect beta view must contain at least one named / windowed workspace",
    );

    // Switch back to seed. Beta's view becomes dormant.
    layout.switch_activity(seed_id);
    layout.verify_invariants();

    // Remove output1. The partial-disconnect walk drains beta's view for output1; named /
    // windowed ids migrate to beta's view for output2 (the remaining monitor).
    check_ops_on_layout(&mut layout, [Op::RemoveOutput(1)]);

    let beta_after = layout
        .activities
        .get(beta_id)
        .expect("beta must still be present after output disconnect");
    assert!(
        !beta_after.views().contains_key(&out1),
        "beta's view for the disconnected output must be drained",
    );
    let beta_out2 = beta_after
        .views()
        .get(&out2)
        .expect("beta must hold a view for the still-connected output");
    for id in &surviving_ids {
        assert!(
            beta_out2.ids().contains(id),
            "surviving workspace {id:?} must appear in beta's view for the remaining monitor",
        );
        // Pool entry must survive with `output_id` still pointing at the disconnecting
        // output, so reconnect-time reclaim via `ensure_view_for`'s lift branch can find
        // it.
        let ws = layout
            .workspaces
            .get(id)
            .expect("migrated workspace must remain a pool key");
        assert_eq!(
            ws.output_id().cloned(),
            Some(out1.clone()),
            "migrated workspace must retain output_id pointing at the disconnecting output",
        );
    }
    layout.verify_invariants();

    // Reconnect output1, then switch to beta. `ensure_view_for`'s lift branch reclaims
    // the migrated workspaces into a fresh beta.views[out1] entry.
    check_ops_on_layout(&mut layout, [Op::AddOutput(1)]);
    layout.switch_activity(beta_id);
    layout.verify_invariants();

    let beta_restored = layout.active_view(&out1);
    for id in &surviving_ids {
        assert!(
            beta_restored.ids().contains(id),
            "after reconnect, surviving workspace {id:?} must be reclaimed into beta's \
             fresh view for the reconnected output",
        );
    }

    // Dedup contract: the source-side drop in `ensure_view_for`'s lift branch must have
    // removed the migrated ids from beta.views[out2] when the reclaim materializer lifted
    // them into the fresh beta.views[out1]. Without the drop, these workspaces would
    // appear in both views and the primary-monitor "own monitor exists" invariant fires
    // at `verify_invariants` on the next switch.
    let beta_after_reconnect = layout.activities.get(beta_id).expect("beta must be live");
    let out1_ids: std::collections::HashSet<WorkspaceId> = beta_after_reconnect
        .views()
        .get(&out1)
        .expect("beta has out1 view")
        .ids()
        .iter()
        .copied()
        .collect();
    let out2_ids: std::collections::HashSet<WorkspaceId> = beta_after_reconnect
        .views()
        .get(&out2)
        .expect("beta has out2 view")
        .ids()
        .iter()
        .copied()
        .collect();
    let intersection: Vec<WorkspaceId> = out1_ids.intersection(&out2_ids).copied().collect();
    assert!(
        intersection.is_empty(),
        "ensure_view_for's lift branch must drop lifted ids from sibling views; \
         found {intersection:?} present in both beta.views[out1] and beta.views[out2]",
    );
}

#[test]
fn partial_disconnect_drains_every_activity_view_for_disconnecting_output() {
    // The partial-disconnect dormant walk drains views keyed by the disconnecting
    // output from *every* activity, not just the active one. Surviving named /
    // windowed workspaces migrate to that activity's view for the remaining monitor;
    // pool entries are retained with their original `output_id` preserved.
    let ops = [Op::AddOutput(1), Op::AddOutput(2)];
    let mut layout = check_ops(ops);
    let seed_id = layout.active_activity_id();
    let out1 = layout.monitors[0].output_id();
    let out2 = layout.monitors[1].output_id();

    let beta = super::activity::Activity::new_runtime("beta".to_owned());
    let beta_id = beta.id();
    test_insert_activity(&mut layout, beta);

    // Switch to beta so it gets views for both outputs, then switch back to seed —
    // beta is now dormant with views for both.
    layout.switch_activity(beta_id);
    layout.verify_invariants();
    let beta_out1_ids: Vec<WorkspaceId> = layout.active_view(&out1).ids().to_vec();

    layout.switch_activity(seed_id);
    layout.verify_invariants();

    // Identify surviving ids (named / window-bearing). Without explicit windows, beta's
    // pre-disconnect view for out1 is just the single trailing empty bookend, which is
    // unnamed and empty — so the surviving set is empty by design here; the test then
    // pins the drain-only path.
    let surviving_ids: Vec<WorkspaceId> = beta_out1_ids
        .iter()
        .copied()
        .filter(|id| {
            layout
                .workspaces
                .get(id)
                .expect("snapshot ids must be live pool keys")
                .has_windows_or_name()
        })
        .collect();

    check_ops_on_layout(&mut layout, [Op::RemoveOutput(1)]);

    // Neither active (seed) nor dormant (beta) holds a view keyed by out1.
    assert!(
        !layout.activities.active().views().contains_key(&out1),
        "active activity (seed) must not hold a view keyed by the disconnected output",
    );
    let beta_after = layout
        .activities
        .get(beta_id)
        .expect("beta must remain present after disconnect");
    assert!(
        !beta_after.views().contains_key(&out1),
        "dormant activity (beta) must not hold a view keyed by the disconnected output",
    );

    // This test is drain-only: beta's pre-disconnect out1 view holds only the trailing
    // empty bookend, which is unnamed and empty — so surviving_ids is empty by design.
    // The drain assertion above pins that the view key is removed regardless. Migration
    // coverage (named / windowed workspaces) is exercised by
    // `partial_disconnect_migrates_dormant_activity_views_to_primary` and
    // `partial_disconnect_dormant_walk_handles_multi_activity_membership`.
    assert!(
        surviving_ids.is_empty(),
        "test invariant: this fixture produces an empty surviving_ids set by design — \
         drain-only path",
    );
    let _beta_out2 = beta_after
        .views()
        .get(&out2)
        .expect("beta must hold a view for the still-connected output");

    layout.verify_invariants();
}

#[test]
fn reconnect_restores_saved_view_active_via_last_active_workspace_id() {
    // The active activity's reconnect path (Monitor::new → ws_id_to_activate) must restore
    // view.active to the workspace that was active before the disconnect. This pins the
    // last_active_workspace_id map as the mechanism for the active activity's reconnect hint.
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(0),
        },
        Op::FocusWorkspaceDown,
        Op::AddWindow {
            params: TestWindowParams::new(1),
        },
        Op::FocusWorkspaceDown,
        Op::FocusWorkspaceUp,
    ];
    let mut layout = check_ops(ops);
    let out1 = layout.monitors[0].output_id();

    // Verify view.active is in a non-bookend position so the reconnect pin has teeth.
    let pre_disconnect_active = layout.active_view(&out1).active();
    let pre_disconnect_pos = layout.active_view(&out1).active_position();
    let pre_disconnect_len = layout.active_view(&out1).len();
    assert!(
        pre_disconnect_pos > 0 && pre_disconnect_pos < pre_disconnect_len - 1,
        "test setup: active must be in a middle position (pos={pre_disconnect_pos}, \
         len={pre_disconnect_len})",
    );

    // Disconnect then reconnect output1.
    check_ops_on_layout(&mut layout, [Op::RemoveOutput(1), Op::AddOutput(1)]);

    let post_reconnect_active = layout.active_view(&out1).active();
    assert_eq!(
        post_reconnect_active, pre_disconnect_active,
        "after reconnect, view.active must equal the workspace that was active before disconnect",
    );
    layout.verify_invariants();
}

// ── Box 1968: MoveWorkspace* active-view scoping ──────────────────────────────

/// Set up a layout with two activities.
///
/// Beta is left **active** with ≥3 workspaces on output1 and view.active at a middle position
/// (position 1 of len=3), so MoveWorkspace* operations on beta's active view produce real
/// mutations. Seed is dormant; its view for output1 (the initial single-empty-workspace view)
/// is the reference that must not be touched.
///
/// Returns `(layout, seed_id, beta_id, out1, seed_dormant_ids, seed_dormant_active)`.
fn setup_two_activities_with_move_workspaces_test() -> (
    Layout<TestWindow>,
    ActivityId,
    ActivityId,
    OutputId,
    Vec<WorkspaceId>,
    WorkspaceId,
) {
    let ops = [Op::AddOutput(1), Op::AddOutput(2)];
    let mut layout = check_ops(ops);
    let seed_id = layout.active_activity_id();
    let out1 = layout.monitors[0].output_id();

    // Snapshot seed's view for out1 while seed is still active (before switch).
    // This is the dormant reference that must survive every MoveWorkspace* call on beta.
    let seed_dormant_ids: Vec<WorkspaceId> = layout.active_view(&out1).ids().to_vec();
    let seed_dormant_active = layout.active_view(&out1).active();

    let beta = super::activity::Activity::new_runtime("beta".to_owned());
    let beta_id = beta.id();
    test_insert_activity(&mut layout, beta);

    // Switch to beta (seed becomes dormant; its view snapshot above is now stable).
    layout.switch_activity(beta_id);
    layout.verify_invariants();

    // Build beta's view on out1: [ws_a(win), ws_b(win), ws_c(empty)], active at ws_c (pos 2).
    // Then focus up to ws_b (pos 1) — a non-bookend middle position — so MoveWorkspace* ops
    // find something to move and actually mutate the ids vec.
    layout.add_window(
        TestWindow::new(TestWindowParams::new(20)),
        AddWindowTarget::Auto,
        None,
        None,
        false,
        false,
        ActivateWindow::default(),
    );
    layout.verify_invariants();
    layout.switch_workspace_down();
    layout.verify_invariants();
    layout.add_window(
        TestWindow::new(TestWindowParams::new(21)),
        AddWindowTarget::Auto,
        None,
        None,
        false,
        false,
        ActivateWindow::default(),
    );
    layout.verify_invariants();
    layout.switch_workspace_down();
    layout.verify_invariants();
    // Focus up to land at middle position (pos 1 = ws_b).
    layout.switch_workspace_up();
    layout.verify_invariants();

    let beta_view_len = layout.active_view(&out1).len();
    let beta_active_pos = layout.active_view(&out1).active_position();
    assert!(
        beta_view_len >= 3,
        "test setup: beta's view on out1 must have ≥3 entries; got {beta_view_len}",
    );
    assert!(
        beta_active_pos > 0 && beta_active_pos < beta_view_len - 1,
        "test setup: beta's active must be at a middle position (pos={beta_active_pos}, \
         len={beta_view_len})",
    );

    (
        layout,
        seed_id,
        beta_id,
        out1,
        seed_dormant_ids,
        seed_dormant_active,
    )
}

#[test]
fn move_workspace_down_operates_only_on_active_activity_view() {
    // move_workspace_down operates exclusively on the active activity's (beta's) view via
    // monitors_pool_view_mut → activities.active_mut().views_mut().
    // Seed's dormant view for output1 must be unchanged after the call.
    //
    // Anti-triviality: we assert beta's active view.active workspace id changed (the move
    // happened) before checking seed's dormant view is untouched.
    let (mut layout, seed_id, _beta_id, out1, seed_ids_before, seed_active_before) =
        setup_two_activities_with_move_workspaces_test();

    // Snapshot beta's full ids order before the call.
    let beta_ids_order_before: Vec<WorkspaceId> = layout.active_view(&out1).ids().to_vec();

    layout.move_workspace_down();
    layout.verify_invariants();

    // Teeth check: move_workspace_down must have mutated beta's active view. The trailing empty
    // workspace is replaced by a fresh one (different WorkspaceId), so the ids vec must differ
    // even if clean_up_workspaces_on restores the active workspace to its original position.
    let beta_ids_order_after: Vec<WorkspaceId> = layout.active_view(&out1).ids().to_vec();
    assert_ne!(
        beta_ids_order_after, beta_ids_order_before,
        "move_workspace_down must mutate beta's active view ids vec \
         (trailing empty is recycled, so the vec must differ from the pre-call snapshot); \
         if equal the operation was a no-op and the test is vacuous",
    );

    // Scoping check: seed's dormant view for out1 must be untouched.
    let seed_dormant = layout
        .activities
        .get(seed_id)
        .expect("seed must remain in pool after move_workspace_down")
        .views()
        .get(&out1)
        .expect("seed must retain its dormant view for out1");
    assert_eq!(
        seed_dormant.ids(),
        seed_ids_before.as_slice(),
        "move_workspace_down must not mutate seed's dormant ids for output1",
    );
    assert_eq!(
        seed_dormant.active(),
        seed_active_before,
        "move_workspace_down must not mutate seed's dormant active workspace id for output1",
    );
}

#[test]
fn move_workspace_up_operates_only_on_active_activity_view() {
    // Symmetric to move_workspace_down_operates_only_on_active_activity_view.
    // move_workspace_up also routes through monitors_pool_view_mut, scoping to the active
    // (beta's) view exclusively. Seed's dormant view must be untouched.
    let (mut layout, seed_id, _beta_id, out1, seed_ids_before, seed_active_before) =
        setup_two_activities_with_move_workspaces_test();

    // Snapshot beta's full ids order before the call.
    let beta_ids_order_before: Vec<WorkspaceId> = layout.active_view(&out1).ids().to_vec();
    let beta_active_pos_before = layout.active_view(&out1).active_position();

    // Active is at middle position (> 0), so move_workspace_up can move upward.
    layout.move_workspace_up();
    layout.verify_invariants();

    // Teeth check: move_workspace_up must have moved the active workspace to a lower position.
    // view.active (the id) stays the same; the ids order changes.
    let beta_view_after = layout.active_view(&out1);
    let beta_active_pos_after = beta_view_after.active_position();
    assert!(
        beta_active_pos_after < beta_active_pos_before,
        "move_workspace_up must move the active workspace to a lower position \
         (was {beta_active_pos_before}, now {beta_active_pos_after}); \
         if equal the operation was a no-op and the test is vacuous",
    );
    let beta_ids_order_after: Vec<WorkspaceId> = beta_view_after.ids().to_vec();
    assert_ne!(
        beta_ids_order_after, beta_ids_order_before,
        "move_workspace_up must reorder beta's active view ids vec",
    );

    // Scoping check: seed's dormant view for out1 must be untouched.
    let seed_dormant = layout
        .activities
        .get(seed_id)
        .expect("seed must remain in pool after move_workspace_up")
        .views()
        .get(&out1)
        .expect("seed must retain its dormant view for out1");
    assert_eq!(
        seed_dormant.ids(),
        seed_ids_before.as_slice(),
        "move_workspace_up must not mutate seed's dormant ids for output1",
    );
    assert_eq!(
        seed_dormant.active(),
        seed_active_before,
        "move_workspace_up must not mutate seed's dormant active workspace id for output1",
    );
}

#[test]
fn move_workspace_to_idx_operates_only_on_active_activity_view() {
    // move_workspace_to_idx (the third _on leaf) also routes through monitors_pool_view_mut
    // → active activity's view only. Seed's dormant view for output1 must be unchanged.
    let (mut layout, seed_id, _beta_id, out1, seed_ids_before, seed_active_before) =
        setup_two_activities_with_move_workspaces_test();

    let beta_active_pos_before = layout.active_view(&out1).active_position();
    let beta_active_before = layout.active_view(&out1).active();

    // Active is at position 1 (middle). Move it to position 0 — a distinct index.
    let target_idx = 0;
    assert_ne!(
        beta_active_pos_before, target_idx,
        "test setup: target_idx must differ from current active position",
    );
    layout.move_workspace_to_idx(None, target_idx);
    layout.verify_invariants();

    // Teeth: active position changes; focused id stays.
    let beta_view_after = layout.active_view(&out1);
    let beta_active_after_pos = beta_view_after.active_position();
    assert_eq!(
        beta_view_after.active(),
        beta_active_before,
        "move_workspace_to_idx must keep the same workspace focused (view.active id unchanged)",
    );
    assert_ne!(
        beta_active_after_pos, beta_active_pos_before,
        "move_workspace_to_idx must change the active workspace's position in ids() \
         (was {beta_active_pos_before}, still {beta_active_after_pos} — operation was a no-op)",
    );

    // Scoping check: seed's dormant view for out1 must be untouched.
    let seed_dormant = layout
        .activities
        .get(seed_id)
        .expect("seed must remain in pool after move_workspace_to_idx")
        .views()
        .get(&out1)
        .expect("seed must retain its dormant view for out1");
    assert_eq!(
        seed_dormant.ids(),
        seed_ids_before.as_slice(),
        "move_workspace_to_idx must not mutate seed's dormant ids for output1",
    );
    assert_eq!(
        seed_dormant.active(),
        seed_active_before,
        "move_workspace_to_idx must not mutate seed's dormant active workspace id for output1",
    );
}

// --- Config-reload activity reconciliation ---
//
// These tests exercise `Layout::reconcile_activities_on_reload_add` — the
// additive / same-name-preserving half of the reload reconciliation.
// Removals / exclusive-workspace rejection / active-cursor cascade are
// covered by Part 2.

#[test]
fn reload_adds_new_config_activity_appends_and_promotes_runtime_name_match() {
    // Seed pool: runtime "Default". Reload config declares `[Work, Default]`.
    // Expected post-reconcile:
    // - "Work" is a fresh config-declared id, position 0.
    // - "Default" keeps its id but flips to `is_config_declared == true`, position 1
    //   (config-declaration order).
    let mut layout = Layout::<TestWindow>::default();
    let default_id = layout.active_activity_id();
    assert!(
        !layout
            .activities
            .get(default_id)
            .expect("seed must exist")
            .is_config_declared(),
        "precondition: default seed is runtime",
    );

    let cfg = [
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("Work".to_owned()),
        },
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("Default".to_owned()),
        },
    ];
    layout.reconcile_activities_on_reload_add(&cfg, &[]);

    let names_in_order: Vec<&str> = layout.activities.iter().map(|a| a.name()).collect();
    assert_eq!(
        names_in_order,
        vec!["Work", "Default"],
        "pool must be ordered by config declaration",
    );
    assert_eq!(layout.activities.len(), 2);

    // The runtime "Default" id is preserved and promoted.
    let promoted = layout
        .activities
        .get(default_id)
        .expect("default must still be in pool with original id");
    assert_eq!(promoted.name(), "Default");
    assert!(
        promoted.is_config_declared(),
        "matching config name must promote existing runtime activity",
    );

    // "Work" is a fresh id distinct from `default_id`.
    let work = layout
        .activities
        .iter()
        .find(|a| a.name() == "Work")
        .expect("Work must be appended");
    assert_ne!(work.id(), default_id, "Work must be freshly minted");
    assert!(work.is_config_declared());

    layout.verify_invariants();
}

#[test]
fn reload_adds_new_config_activity_preserves_views_and_assignments_on_promotion() {
    // A runtime "Work" carrying a `views` entry and a workspace whose
    // `activities` set includes its id. Reload declares `activity "Work"`.
    // Expected: id preserved, views preserved, workspace set preserved.
    let ops = [
        Op::AddOutput(1),
        Op::AddNamedWorkspace {
            ws_name: 1,
            output_name: None,
            layout_config: None,
        },
    ];
    let mut layout = check_ops(ops);
    let seed_id = layout.active_activity_id();

    // Create a runtime activity "Work" and switch to it so it populates a
    // per-output view on the connected monitor.
    let work_id = layout
        .create_activity("Work".to_owned())
        .expect("create must succeed");
    layout.switch_activity(work_id);
    // Switch back to seed so that the reload flow does not have to juggle
    // the active cursor; the test only pins views/assignments.
    layout.switch_activity(seed_id);

    // Snapshot views and workspace sets referencing `work_id` before reload.
    let work_views_before: HashMap<OutputId, (Vec<WorkspaceId>, WorkspaceId, Option<WorkspaceId>)> =
        layout
            .activities
            .get(work_id)
            .expect("work must exist")
            .views()
            .iter()
            .map(|(out, v)| (out.clone(), (v.ids().to_vec(), v.active(), v.previous())))
            .collect();
    // Precondition: `Work` must hold at least one dormant view before the
    // reload, otherwise the later "views preserved across promotion" assert
    // degenerates into comparing two empty maps and stops catching
    // regressions.
    assert!(
        !work_views_before.is_empty(),
        "precondition: Work has a populated views map from the switch",
    );

    // Stamp a workspace so its `activities` set includes `work_id`. The
    // workspace is unnamed (dynamic), so the reload's workspace-reset step
    // will leave it alone.
    let dyn_ws_id: WorkspaceId = *layout
        .workspaces
        .iter()
        .find(|(_, ws)| ws.name().is_none())
        .expect("at least one dynamic workspace present")
        .0;
    layout
        .workspaces
        .get_mut(&dyn_ws_id)
        .unwrap()
        .activities
        .insert(work_id);
    let dyn_set_before = layout
        .workspaces
        .get(&dyn_ws_id)
        .unwrap()
        .activities()
        .clone();

    // Simulate the State::reload_config prewalk: clear the name off any
    // workspace whose name is absent from config_workspaces. "ws1" is not in
    // the config below, so it must be unnamed first — the debug-assert inside
    // reconcile_activities_on_reload_add enforces this precondition.
    layout.unname_workspace("ws1");

    let cfg = [jiji_config::ActivityDecl {
        name: jiji_config::ActivityName("Work".to_owned()),
    }];
    layout.reconcile_activities_on_reload_add(&cfg, &[]);

    // Id preserved.
    let work = layout
        .activities
        .get(work_id)
        .expect("work id must be preserved across promotion");
    assert_eq!(work.name(), "Work");
    assert!(work.is_config_declared());

    // Views preserved.
    let work_views_after: HashMap<OutputId, (Vec<WorkspaceId>, WorkspaceId, Option<WorkspaceId>)> =
        work.views()
            .iter()
            .map(|(out, v)| (out.clone(), (v.ids().to_vec(), v.active(), v.previous())))
            .collect();
    assert_eq!(
        work_views_after, work_views_before,
        "promotion must not touch views",
    );

    // Workspace set preserved on the dynamic (unnamed) workspace.
    assert_eq!(
        layout.workspaces.get(&dyn_ws_id).unwrap().activities(),
        &dyn_set_before,
        "dynamic workspace activities set must be preserved on promotion",
    );

    layout.verify_invariants();
}

#[test]
fn reload_reorders_to_match_config_declaration_order() {
    // Seed pool by creating runtime activities. Initial order: [Default, A, B, C].
    // Reload config names [C, B, A] → post-reorder the config-declared prefix is
    // [C, B, A], runtime-only "Default" falls to the trailer.
    let mut layout = Layout::<TestWindow>::default();
    let default_id = layout.active_activity_id();
    let a_id = layout
        .create_activity("A".to_owned())
        .expect("create A must succeed");
    let b_id = layout
        .create_activity("B".to_owned())
        .expect("create B must succeed");
    let c_id = layout
        .create_activity("C".to_owned())
        .expect("create C must succeed");
    // Sanity precondition on order.
    let order_before: Vec<ActivityId> = layout.activities.iter().map(|a| a.id()).collect();
    assert_eq!(order_before, vec![default_id, a_id, b_id, c_id]);

    let active_before = layout.active_activity_id();
    let previous_before = layout.activities.previous_id();

    let cfg = [
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("C".to_owned()),
        },
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("B".to_owned()),
        },
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("A".to_owned()),
        },
    ];
    layout.reconcile_activities_on_reload_add(&cfg, &[]);

    let names: Vec<&str> = layout.activities.iter().map(|a| a.name()).collect();
    assert_eq!(
        names,
        vec!["C", "B", "A", "Default"],
        "config-declared prefix must be in config order; runtime 'Default' trails",
    );
    assert_eq!(
        layout.active_activity_id(),
        active_before,
        "reorder must not flip the active cursor",
    );
    assert_eq!(
        layout.activities.previous_id(),
        previous_before,
        "reorder must not touch the previous cursor",
    );

    layout.verify_invariants();
}

#[test]
fn reload_preserves_active_and_previous_cursors_across_reorder() {
    // Same seed topology as test 3, but with `active = B, previous = A`
    // established before reload. Reload config `[C, B, A]` must preserve
    // both cursors.
    let mut layout = Layout::<TestWindow>::default();
    let _default_id = layout.active_activity_id();
    let a_id = layout
        .create_activity("A".to_owned())
        .expect("create A must succeed");
    let b_id = layout
        .create_activity("B".to_owned())
        .expect("create B must succeed");
    let _c_id = layout
        .create_activity("C".to_owned())
        .expect("create C must succeed");
    // Establish active=B, previous=A: switch to A first, then to B.
    layout.switch_activity(a_id);
    layout.switch_activity(b_id);
    assert_eq!(layout.active_activity_id(), b_id);
    assert_eq!(layout.activities.previous_id(), Some(a_id));

    let cfg = [
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("C".to_owned()),
        },
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("B".to_owned()),
        },
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("A".to_owned()),
        },
    ];
    layout.reconcile_activities_on_reload_add(&cfg, &[]);

    assert_eq!(
        layout.active_activity_id(),
        b_id,
        "reorder must preserve active_id across the move",
    );
    assert_eq!(
        layout.activities.previous_id(),
        Some(a_id),
        "reorder must preserve previous_id across the move",
    );

    layout.verify_invariants();
}

#[test]
fn reload_sticky_workspace_reexpands_activities_set_to_all_current() {
    // Pool starts with runtime "Default" and a sticky workspace whose
    // `activities` set was narrowed to `{Default}` (inconsistent with sticky
    // semantics, but we construct that by hand to pin the re-expansion
    // behavior). Reload adds config-declared `Work`. Sticky set must expand
    // to `{Default, Work}`.
    let ops = [
        Op::AddOutput(1),
        Op::AddNamedWorkspace {
            ws_name: 1,
            output_name: None,
            layout_config: None,
        },
    ];
    let mut layout = check_ops(ops);
    let default_id = layout.active_activity_id();
    let mon_out = layout.monitors[0].output_id();

    // Flip the output-bound named workspace sticky and narrow its activities.
    let sticky_id = {
        let sticky_id = layout
            .workspaces
            .values()
            .find(|ws| ws.output_id() == Some(&mon_out) && ws.name().is_some())
            .expect("at least one named workspace bound to the output")
            .id();
        let sticky = layout
            .workspaces
            .get_mut(&sticky_id)
            .expect("sticky candidate must be in the pool");
        sticky.is_sticky = true;
        sticky.activities = HashSet::from([default_id]);
        sticky_id
    };

    let cfg = [jiji_config::ActivityDecl {
        name: jiji_config::ActivityName("Work".to_owned()),
    }];
    layout.reconcile_activities_on_reload_add(&cfg, &[]);

    let work_id = layout
        .activities
        .iter()
        .find(|a| a.name() == "Work")
        .expect("Work must be appended")
        .id();

    assert_eq!(
        layout
            .workspaces
            .get(&sticky_id)
            .expect("sticky ws still in pool")
            .activities(),
        &HashSet::from([default_id, work_id]),
        "sticky workspace must be re-expanded to the full post-reload id universe",
    );

    layout.verify_invariants();
}

#[test]
fn reload_config_workspace_activities_reset_to_declared_values() {
    // Pool: runtime "Default" + runtime "Work" (via create_activity). A named
    // workspace "home" whose runtime activities set is `{Default, Work}`.
    // Reload declares `activity "Default"`, `activity "Work"`, and
    // `workspace "home" { activity "Work"; }` → home's activities must reset
    // to `{Work}`.
    let ops = [
        Op::AddOutput(1),
        Op::AddNamedWorkspace {
            ws_name: 7,
            output_name: None,
            layout_config: None,
        },
    ];
    let mut layout = check_ops(ops);
    let default_id = layout.active_activity_id();
    let work_id = layout
        .create_activity("Work".to_owned())
        .expect("create must succeed");

    // Find the named workspace and stamp it with both ids.
    let home_id = layout
        .workspaces
        .values()
        .find(|ws| ws.name().is_some())
        .expect("at least one named workspace")
        .id();
    let home_name = layout
        .workspaces
        .get(&home_id)
        .unwrap()
        .name()
        .cloned()
        .expect("named workspace has a name");
    layout.workspaces.get_mut(&home_id).unwrap().activities = HashSet::from([default_id, work_id]);

    let cfg_activities = [
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("Default".to_owned()),
        },
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("Work".to_owned()),
        },
    ];
    let cfg_workspaces = [WorkspaceConfig {
        name: WorkspaceName(home_name),
        open_on_output: None,
        layout: None,
        activities: vec!["Work".to_owned()],
        sticky: None,
    }];
    layout.reconcile_activities_on_reload_add(&cfg_activities, &cfg_workspaces);

    assert_eq!(
        layout
            .workspaces
            .get(&home_id)
            .expect("home ws still in pool")
            .activities(),
        &HashSet::from([work_id]),
        "config-declared workspace must have its activities set reset to config value",
    );

    layout.verify_invariants();
}

#[test]
fn reload_dynamic_workspace_keeps_runtime_activity_assignments() {
    // Pool: Default + Work. A dynamic (unnamed) workspace whose activities
    // set is `{Default, Work}`. Reload adds `Personal`. The dynamic
    // workspace's activities set must NOT change (not a config-declared
    // workspace, not sticky).
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let default_id = layout.active_activity_id();
    let work_id = layout
        .create_activity("Work".to_owned())
        .expect("create must succeed");

    let dyn_ws_id = *layout
        .workspaces
        .iter()
        .find(|(_, ws)| ws.name().is_none())
        .expect("at least one dynamic workspace present")
        .0;
    layout.workspaces.get_mut(&dyn_ws_id).unwrap().activities =
        HashSet::from([default_id, work_id]);
    let before = layout
        .workspaces
        .get(&dyn_ws_id)
        .unwrap()
        .activities()
        .clone();

    let cfg_activities = [
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("Default".to_owned()),
        },
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("Work".to_owned()),
        },
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("Personal".to_owned()),
        },
    ];
    layout.reconcile_activities_on_reload_add(&cfg_activities, &[]);

    assert_eq!(
        layout.workspaces.get(&dyn_ws_id).unwrap().activities(),
        &before,
        "dynamic workspace activities must be untouched on reload",
    );

    layout.verify_invariants();
}

#[test]
fn reload_no_change_is_semantically_noop_on_activities_and_workspaces() {
    // Build a layout from a config `C`, record all observable state, then
    // call reconcile with the same `C`. Activities list (id/name/flag) and
    // workspace state must be unchanged, and invariants must hold.
    let config = Config {
        activities: vec![
            jiji_config::ActivityDecl {
                name: jiji_config::ActivityName("Work".to_owned()),
            },
            jiji_config::ActivityDecl {
                name: jiji_config::ActivityName("Personal".to_owned()),
            },
        ],
        workspaces: vec![
            WorkspaceConfig {
                name: WorkspaceName("chat".to_owned()),
                open_on_output: None,
                layout: None,
                activities: vec!["Work".to_owned(), "Personal".to_owned()],
                sticky: None,
            },
            WorkspaceConfig {
                name: WorkspaceName("music".to_owned()),
                open_on_output: None,
                layout: None,
                activities: Vec::new(),
                sticky: None,
            },
        ],
        ..Config::default()
    };
    let mut layout = Layout::<TestWindow>::new(Clock::with_time(Duration::ZERO), &config);

    // Observable snapshot: (ordered (id, name, is_config_declared) tuples,
    // workspace activities sets keyed by id).
    type ActivityProjection = Vec<(ActivityId, String, bool)>;
    let activities_before: ActivityProjection = layout
        .activities
        .iter()
        .map(|a| (a.id(), a.name().to_owned(), a.is_config_declared()))
        .collect();
    let workspaces_before: HashMap<WorkspaceId, HashSet<ActivityId>> = layout
        .workspaces
        .iter()
        .map(|(id, ws)| (*id, ws.activities().clone()))
        .collect();
    let active_before = layout.active_activity_id();
    let previous_before = layout.activities.previous_id();

    layout.reconcile_activities_on_reload_add(&config.activities, &config.workspaces);

    let activities_after: ActivityProjection = layout
        .activities
        .iter()
        .map(|a| (a.id(), a.name().to_owned(), a.is_config_declared()))
        .collect();
    assert_eq!(
        activities_after, activities_before,
        "no-op reload must not mutate the activities pool",
    );

    let workspaces_after: HashMap<WorkspaceId, HashSet<ActivityId>> = layout
        .workspaces
        .iter()
        .map(|(id, ws)| (*id, ws.activities().clone()))
        .collect();
    assert_eq!(
        workspaces_after, workspaces_before,
        "no-op reload must not mutate any workspace's activities set",
    );

    assert_eq!(layout.active_activity_id(), active_before);
    assert_eq!(layout.activities.previous_id(), previous_before);

    layout.verify_invariants();
}

#[test]
fn reload_promotion_preserves_runtime_name_casing() {
    // Seed pool with a runtime activity whose name uses all-lowercase "work".
    // Reload config declares "WORK" (all-caps). Expected: stored name stays
    // "work" (runtime casing, NOT overwritten by config spelling), id
    // unchanged, is_config_declared flips to true. Pins bullet 1.
    let mut layout = Layout::<TestWindow>::default();
    let work_id = layout
        .create_activity("work".to_owned())
        .expect("create must succeed");
    assert_eq!(
        layout
            .activities
            .get(work_id)
            .expect("work must exist")
            .name(),
        "work",
        "precondition: runtime activity must carry lowercase name",
    );
    assert!(
        !layout
            .activities
            .get(work_id)
            .expect("work must exist")
            .is_config_declared(),
        "precondition: newly created activity is runtime",
    );

    // Reload declares "WORK" (all-caps) — case-insensitively matches "work".
    let cfg = [jiji_config::ActivityDecl {
        name: jiji_config::ActivityName("WORK".to_owned()),
    }];
    layout.reconcile_activities_on_reload_add(&cfg, &[]);

    let promoted = layout
        .activities
        .get(work_id)
        .expect("work must still be in pool with original id");
    assert_eq!(
        promoted.name(),
        "work",
        "runtime casing must NOT be overwritten by config spelling",
    );
    assert_eq!(promoted.id(), work_id, "id must be preserved on promotion",);
    assert!(
        promoted.is_config_declared(),
        "is_config_declared must flip to true on promotion",
    );

    layout.verify_invariants();
}

// `reconcile_activities_on_reload_add` must call `ensure_all_activity_views` so
// any newly-added or freshly-promoted config activity holds a bookend view for
// every connected monitor. Pin this for the "config-new activity added while a
// monitor is connected" path.
#[test]
fn reload_adds_new_config_activity_materializes_bookend_for_dormant_survivor_on_connected_monitor()
{
    // Start with one connected output and a runtime "Default" activity.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let mon_out = layout.monitors[0].output_id();

    let default_id = layout.active_activity_id();

    // Reload config declares [Work, Default]. "Work" is brand-new.
    let cfg = [
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("Work".to_owned()),
        },
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("Default".to_owned()),
        },
    ];
    layout.reconcile_activities_on_reload_add(&cfg, &[]);

    let work_id = layout
        .activities
        .iter()
        .find(|a| a.name() == "Work")
        .expect("Work must be present after reload")
        .id();

    // "Work" is a dormant survivor — it was never switched-to, so without the
    // materializer call its views map would be empty. Assert it has a bookend view
    // for the connected output.
    let work_view = layout
        .activities
        .get(work_id)
        .expect("work live")
        .views()
        .get(&mon_out)
        .expect("Work must hold a view for the connected monitor after reload_add")
        .clone();

    assert_eq!(
        work_view.len(),
        1,
        "freshly-materialised view must have exactly one trailing-empty bookend",
    );
    assert!(
        !layout
            .workspaces
            .get(work_view.ids().first().unwrap())
            .expect("bookend in pool")
            .has_windows_or_name(),
        "bookend must be empty and unnamed",
    );
    // Default (existing activity) must still have its view.
    assert!(
        layout
            .activities
            .get(default_id)
            .expect("default live")
            .views()
            .contains_key(&mon_out),
        "existing activity must still hold its view after reload_add",
    );

    layout.verify_invariants();
}

// --- Config-reload activity reconciliation — removal half ---
//
// These tests exercise `Layout::reconcile_activities_on_reload_remove` — the
// atomic validate-then-mutate entry that drops config-declared activities
// absent from the reloaded config. Every `Err` test snapshots pool-size,
// workspace-count, active cursor, and previous cursor before the call and
// `assert_eq!`'s them after — the atomicity contract is the load-bearing pin.

#[test]
fn reconcile_remove_no_op_when_config_retains_all_activities() {
    // Seed: runtime "Default", promoted to config-declared by an initial
    // reconcile_add. Reload config names `[Default]` (unchanged). Expected:
    // remove-set is empty, Ok(()), pool unchanged.
    let mut layout = Layout::<TestWindow>::default();
    let default_id = layout.active_activity_id();

    let cfg = [jiji_config::ActivityDecl {
        name: jiji_config::ActivityName("Default".to_owned()),
    }];
    layout.reconcile_activities_on_reload_add(&cfg, &[]);
    assert!(
        layout
            .activities
            .get(default_id)
            .expect("default must remain in pool")
            .is_config_declared(),
        "precondition: Default is config-declared after the seeding add-reconcile",
    );

    let pool_size_before = layout.activities.len();
    let workspaces_size_before = layout.workspaces.len();

    layout
        .reconcile_activities_on_reload_remove(&cfg)
        .expect("identical config must be Ok(())");

    assert_eq!(layout.activities.len(), pool_size_before, "pool unchanged");
    assert_eq!(
        layout.workspaces.len(),
        workspaces_size_before,
        "workspaces unchanged",
    );
    assert_eq!(layout.active_activity_id(), default_id);

    layout.verify_invariants();
}

#[test]
fn reconcile_remove_drops_config_activity_with_no_exclusive_workspaces() {
    // Seed: config-declared [A, B] (via reconcile_add). No workspaces are
    // exclusive to B. Reload config names [A]. Expected: B dropped from pool;
    // workspace count unchanged.
    let mut layout = Layout::<TestWindow>::default();
    let seed_id = layout.active_activity_id();

    let cfg_init = [
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("Default".to_owned()),
        },
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("Beta".to_owned()),
        },
    ];
    layout.reconcile_activities_on_reload_add(&cfg_init, &[]);
    let beta_id = layout
        .activities
        .iter()
        .find(|a| a.name() == "Beta")
        .expect("Beta was added")
        .id();

    let pool_before = layout.activities.len();
    let ws_before = layout.workspaces.len();

    let cfg_reload = [jiji_config::ActivityDecl {
        name: jiji_config::ActivityName("Default".to_owned()),
    }];
    layout
        .reconcile_activities_on_reload_remove(&cfg_reload)
        .expect("non-active config activity with no exclusives must drop");

    assert_eq!(
        layout.activities.len(),
        pool_before - 1,
        "exactly one activity dropped",
    );
    assert!(!layout.activities.contains(beta_id), "Beta dropped");
    assert_eq!(
        layout.workspaces.len(),
        ws_before,
        "no workspaces destroyed",
    );
    assert_eq!(layout.active_activity_id(), seed_id);

    layout.verify_invariants();
}

#[test]
fn reconcile_remove_destroys_empty_unnamed_exclusive_workspace() {
    // B exclusive over one empty unnamed workspace. Reload drops B. Expected:
    // workspace destroyed, B removed.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let seed_id = layout.active_activity_id();

    // Seed config-declared [Default, Beta].
    let cfg_init = [
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("Default".to_owned()),
        },
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("Beta".to_owned()),
        },
    ];
    layout.reconcile_activities_on_reload_add(&cfg_init, &[]);
    let beta_id = layout
        .activities
        .iter()
        .find(|a| a.name() == "Beta")
        .expect("Beta was added")
        .id();

    // Allocate a fresh exclusive unnamed workspace tagged to beta.
    let output = layout.monitors[0].output.clone();
    let beta_ws = Workspace::new(
        &output,
        HashSet::from([beta_id]),
        layout.clock.clone(),
        layout.options.clone(),
    );
    let beta_ws_id = beta_ws.id();
    assert!(
        layout.workspaces.insert(beta_ws_id, beta_ws).is_none(),
        "fresh id is unique",
    );
    let ws_baseline = layout.workspaces.len();

    let cfg_reload = [jiji_config::ActivityDecl {
        name: jiji_config::ActivityName("Default".to_owned()),
    }];
    layout
        .reconcile_activities_on_reload_remove(&cfg_reload)
        .expect("empty unnamed exclusive must be destroyed without error");

    assert!(!layout.activities.contains(beta_id), "Beta dropped");
    assert!(
        !layout.workspaces.contains_key(&beta_ws_id),
        "exclusive unnamed workspace destroyed",
    );
    assert_eq!(
        layout.workspaces.len(),
        ws_baseline - 2,
        "beta_ws plus the materializer's exclusive bookend were destroyed",
    );
    assert_eq!(layout.active_activity_id(), seed_id);

    layout.verify_invariants();
}

#[test]
fn reconcile_remove_destroys_empty_named_exclusive_workspace() {
    // Named exclusive workspaces are destroyed on reload (unlike IPC
    // RemoveActivity which rejects them via ExclusiveNamedWorkspace). This pins
    // the asymmetry: reload is user-initiated config churn; the caller
    // chose to drop the activity by editing the config.
    let ops = [
        Op::AddOutput(1),
        Op::AddNamedWorkspace {
            ws_name: 7,
            output_name: None,
            layout_config: None,
        },
    ];
    let mut layout = check_ops(ops);

    // Seed Beta directly via the pool API, mark it config-declared so it
    // qualifies for the remove-set. Skipping `reconcile_add` avoids the
    // add-path's debug-assert that every named non-sticky workspace must
    // appear in `config_workspaces` — this test exercises reload_remove in
    // isolation, not the full `State::reload_config` prewalk.
    let beta_activity = super::activity::Activity::new_config_declared("Beta".to_owned());
    let beta_id = beta_activity.id();
    test_insert_activity(&mut layout, beta_activity);

    // Flip the named workspace to exclusively beta.
    let named_ws_id = layout
        .workspaces
        .values()
        .find(|ws| ws.name() == Some(&"ws7".to_owned()))
        .expect("named workspace allocated")
        .id();
    layout
        .workspaces
        .get_mut(&named_ws_id)
        .expect("live pool key")
        .activities = std::iter::once(beta_id).collect();

    let cfg_reload = [jiji_config::ActivityDecl {
        name: jiji_config::ActivityName("Default".to_owned()),
    }];
    layout
        .reconcile_activities_on_reload_remove(&cfg_reload)
        .expect("empty named exclusive must be destroyed on reload path");

    assert!(!layout.activities.contains(beta_id), "Beta dropped");
    assert!(
        !layout.workspaces.contains_key(&named_ws_id),
        "named exclusive workspace destroyed (reload asymmetry vs IPC RemoveActivity)",
    );

    layout.verify_invariants();
}

#[test]
fn reconcile_remove_prunes_shared_workspaces() {
    // Workspace W has activities == {A, B}. Reload drops B. Expected: W
    // survives, W.activities shrinks to {A}.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);

    let cfg_init = [
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("Default".to_owned()),
        },
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("Beta".to_owned()),
        },
    ];
    layout.reconcile_activities_on_reload_add(&cfg_init, &[]);
    let default_id = layout
        .activities
        .iter()
        .find(|a| a.name() == "Default")
        .expect("Default is in pool")
        .id();
    let beta_id = layout
        .activities
        .iter()
        .find(|a| a.name() == "Beta")
        .expect("Beta was added")
        .id();

    // Union beta into the seed-Default workspace (NOT beta's materialized bookend — that one
    // is exclusive to beta and would itself become shared after the union, defeating the test).
    let shared_ws_id = layout
        .workspaces
        .values()
        .find(|ws| ws.activities() == &HashSet::from([default_id]))
        .map(|ws| ws.id())
        .expect("seed-Default workspace allocated");
    layout
        .workspaces
        .get_mut(&shared_ws_id)
        .expect("live pool key")
        .activities = [default_id, beta_id].into_iter().collect();

    let ws_before = layout.workspaces.len();

    let cfg_reload = [jiji_config::ActivityDecl {
        name: jiji_config::ActivityName("Default".to_owned()),
    }];
    layout
        .reconcile_activities_on_reload_remove(&cfg_reload)
        .expect("shared prune must succeed");

    assert!(!layout.activities.contains(beta_id), "Beta dropped");
    // The shared workspace is retained; beta's materializer bookend (exclusive to beta) is
    // destroyed as part of the reconcile remove.
    assert_eq!(
        layout.workspaces.len(),
        ws_before - 1,
        "shared workspace retained; only beta's materialized bookend was destroyed",
    );
    let shared_ws = layout
        .workspaces
        .get(&shared_ws_id)
        .expect("shared workspace must still exist");
    assert_eq!(
        shared_ws.activities(),
        &HashSet::from([default_id]),
        "shared workspace must have Beta pruned, Default retained",
    );

    layout.verify_invariants();
}

#[test]
fn reconcile_remove_rejects_on_exclusive_workspace_has_windows() {
    // B exclusive over a workspace with a window. Reload drops B. Expected:
    // Err(ExclusiveWorkspaceHasWindows), and the entire state — pool,
    // workspaces, active, previous — is byte-for-byte unchanged. This is the
    // atomicity pin inherited from Layout::remove_activity.
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(99),
        },
    ];
    let mut layout = check_ops(ops);

    let cfg_init = [
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("Default".to_owned()),
        },
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("Beta".to_owned()),
        },
    ];
    layout.reconcile_activities_on_reload_add(&cfg_init, &[]);
    let beta_id = layout
        .activities
        .iter()
        .find(|a| a.name() == "Beta")
        .expect("Beta was added")
        .id();

    let window_ws_id = layout
        .workspaces
        .values()
        .find(|ws| ws.has_windows())
        .expect("AddWindow allocated a window-carrying workspace")
        .id();
    layout
        .workspaces
        .get_mut(&window_ws_id)
        .expect("live pool key")
        .activities = std::iter::once(beta_id).collect();

    let pool_before = layout.activities.len();
    let ws_before = layout.workspaces.len();
    let active_before = layout.active_activity_id();
    let previous_before = layout.activities.previous_id();

    let cfg_reload = [jiji_config::ActivityDecl {
        name: jiji_config::ActivityName("Default".to_owned()),
    }];
    let err = layout
        .reconcile_activities_on_reload_remove(&cfg_reload)
        .expect_err("exclusive-with-windows must err");

    match err {
        ReloadActivityRemovalError::ExclusiveWorkspaceHasWindows {
            activity_name,
            workspace_id,
        } => {
            assert_eq!(activity_name, "Beta");
            assert_eq!(workspace_id, window_ws_id);
        }
        other => panic!("unexpected error variant: {other:?}"),
    }

    assert_eq!(layout.activities.len(), pool_before, "pool unchanged");
    assert_eq!(layout.workspaces.len(), ws_before, "workspaces unchanged");
    assert_eq!(
        layout.active_activity_id(),
        active_before,
        "active cursor unchanged on Err",
    );
    assert_eq!(
        layout.activities.previous_id(),
        previous_before,
        "previous cursor unchanged on Err",
    );
    assert!(layout.activities.contains(beta_id), "Beta survives on Err");

    layout.verify_invariants();
}

#[test]
fn reconcile_remove_rejects_would_empty_pool() {
    // Pool must be purely config-declared (runtime activities survive reload
    // per "Runtime activities on reload" and are not candidates for the
    // remove-set, so a runtime-seed pool never can hit WouldEmptyPool). Seed
    // via the startup `with_options_and_workspaces` path which uses
    // `Activities::from_config_or_default` — an explicit `activity "Solo"`
    // in the seed config makes the pool `[Solo (config-declared)]`. Reload
    // declares zero activities. Expected: Err(WouldEmptyPool), no mutation.
    let config = Config {
        activities: vec![jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("Solo".to_owned()),
        }],
        ..Config::default()
    };
    let mut layout = Layout::<TestWindow>::new(Clock::default(), &config);
    assert_eq!(layout.activities.len(), 1);
    let solo_id = layout.active_activity_id();
    assert!(
        layout
            .activities
            .get(solo_id)
            .expect("Solo in pool")
            .is_config_declared(),
        "precondition: Solo is config-declared",
    );

    let pool_before = layout.activities.len();
    let ws_before = layout.workspaces.len();
    let active_before = layout.active_activity_id();

    let cfg_reload: [jiji_config::ActivityDecl; 0] = [];
    let err = layout
        .reconcile_activities_on_reload_remove(&cfg_reload)
        .expect_err("would-empty-pool must err");

    match err {
        ReloadActivityRemovalError::WouldEmptyPool { activity_name } => {
            assert_eq!(activity_name, "Solo");
        }
        other => panic!("unexpected error variant: {other:?}"),
    }

    assert_eq!(layout.activities.len(), pool_before, "pool unchanged");
    assert_eq!(layout.workspaces.len(), ws_before, "workspaces unchanged");
    assert_eq!(layout.active_activity_id(), active_before);

    layout.verify_invariants();
}

#[test]
fn reconcile_remove_cascades_active_cursor_to_previous() {
    // Pool: [Default, Beta], active == Beta (via switch), previous == Default.
    // Reload drops Beta. Expected: active flips to Default (previous), Beta
    // removed.
    let mut layout = Layout::<TestWindow>::default();
    let cfg_init = [
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("Default".to_owned()),
        },
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("Beta".to_owned()),
        },
    ];
    layout.reconcile_activities_on_reload_add(&cfg_init, &[]);
    let default_id = layout
        .activities
        .iter()
        .find(|a| a.name() == "Default")
        .expect("Default in pool")
        .id();
    let beta_id = layout
        .activities
        .iter()
        .find(|a| a.name() == "Beta")
        .expect("Beta in pool")
        .id();

    layout.switch_activity(beta_id);
    assert_eq!(layout.active_activity_id(), beta_id);
    assert_eq!(layout.activities.previous_id(), Some(default_id));

    let cfg_reload = [jiji_config::ActivityDecl {
        name: jiji_config::ActivityName("Default".to_owned()),
    }];
    layout
        .reconcile_activities_on_reload_remove(&cfg_reload)
        .expect("cascade to previous must succeed");

    assert_eq!(
        layout.active_activity_id(),
        default_id,
        "cascade target was previous (Default)",
    );
    assert!(!layout.activities.contains(beta_id), "Beta dropped");
    assert_eq!(
        layout.activities.previous_id(),
        None,
        "previous pointed at the removed Beta and must be cleared",
    );

    layout.verify_invariants();
}

#[test]
fn reconcile_remove_cascades_active_cursor_to_first_when_previous_also_in_remove_set() {
    // Pool: [Default, Beta, Gamma]. Active == Gamma, previous == Beta.
    // Reload drops both Beta and Gamma. Expected: cascade target is Default
    // (first declaration-order id not in remove_set).
    let mut layout = Layout::<TestWindow>::default();
    let cfg_init = [
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("Default".to_owned()),
        },
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("Beta".to_owned()),
        },
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("Gamma".to_owned()),
        },
    ];
    layout.reconcile_activities_on_reload_add(&cfg_init, &[]);
    let default_id = layout
        .activities
        .iter()
        .find(|a| a.name() == "Default")
        .expect("Default in pool")
        .id();
    let beta_id = layout
        .activities
        .iter()
        .find(|a| a.name() == "Beta")
        .expect("Beta in pool")
        .id();
    let gamma_id = layout
        .activities
        .iter()
        .find(|a| a.name() == "Gamma")
        .expect("Gamma in pool")
        .id();

    layout.switch_activity(beta_id);
    layout.switch_activity(gamma_id);
    assert_eq!(layout.active_activity_id(), gamma_id);
    assert_eq!(layout.activities.previous_id(), Some(beta_id));

    let cfg_reload = [jiji_config::ActivityDecl {
        name: jiji_config::ActivityName("Default".to_owned()),
    }];
    layout
        .reconcile_activities_on_reload_remove(&cfg_reload)
        .expect("cascade to first surviving in declaration order must succeed");

    assert_eq!(
        layout.active_activity_id(),
        default_id,
        "previous (Beta) was itself in remove_set; cascade must skip to first non-remove-set id \
         in declaration order (Default)",
    );
    assert!(!layout.activities.contains(beta_id));
    assert!(!layout.activities.contains(gamma_id));
    assert_eq!(layout.activities.len(), 1);

    layout.verify_invariants();
}

#[test]
fn reconcile_remove_rebinds_orphan_workspace_into_cascade_target_view() {
    // Setup: seed `Default` (pool starts with it), promote via reload-add to
    // config-declared `[Default, alpha, beta]`. Switch active to alpha so
    // `active = alpha`, `previous = Default`. Mint a sentinel workspace via
    // `Workspace::new_no_outputs` (carries `OutputId("")`), tag it as
    // exclusively beta's, and splice it into alpha's view of monitor 1 just
    // before the trailing-empty bookend — mirroring what `Monitor::new`'s
    // lift loop does on first-monitor-attach when an orphan sits in
    // `disconnected_workspace_ids`.
    //
    // Reload drops `Default` and `alpha`, keeps `beta`. Both `active=alpha`
    // and `previous=Default` are in remove_set, so the cascade falls through
    // `previous.filter(...)` → None and lands on the first declaration-order
    // non-remove-set survivor (`beta`). Without the orphan-rebind pass, the
    // orphan's only anchoring view (alpha's view of mon_out) evaporates with
    // `self.activities.remove(alpha)`, breaking pool-keys-equal-union.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);

    let cfg_init = [
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("Default".to_owned()),
        },
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("alpha".to_owned()),
        },
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("beta".to_owned()),
        },
    ];
    layout.reconcile_activities_on_reload_add(&cfg_init, &[]);
    let default_id = layout
        .activities
        .iter()
        .find(|a| a.name() == "Default")
        .expect("Default in pool")
        .id();
    let alpha_id = layout
        .activities
        .iter()
        .find(|a| a.name() == "alpha")
        .expect("alpha in pool")
        .id();
    let beta_id = layout
        .activities
        .iter()
        .find(|a| a.name() == "beta")
        .expect("beta in pool")
        .id();

    layout.switch_activity(alpha_id);
    assert_eq!(layout.active_activity_id(), alpha_id);
    assert_eq!(layout.activities.previous_id(), Some(default_id));

    let mon_out = layout.monitors[0].output_id();

    // Mint the orphan with the empty-string sentinel. Named so the Monitor
    // invariant "non-active non-last workspaces must be empty-and-unnamed"
    // does not fire: an unnamed empty workspace at a non-last position would
    // violate it. This matches the production scenario the surface-C panic
    // reproduces from — a config-named workspace tagged to a non-active
    // activity, lifted by `Monitor::new` into the seed-active activity's
    // view at first-monitor-attach.
    let orphan_cfg = jiji_config::Workspace {
        name: jiji_config::workspace::WorkspaceName("ws_b".to_owned()),
        open_on_output: None,
        layout: None,
        activities: vec!["beta".to_owned()],
        sticky: None,
    };
    let orphan = Workspace::<TestWindow>::new_with_config_no_outputs(
        Some(orphan_cfg),
        HashSet::from([beta_id]),
        layout.clock.clone(),
        layout.options.clone(),
    );
    let orphan_id = orphan.id();
    assert_eq!(
        orphan.output_id().map(|oid| oid.as_str()),
        Some(""),
        "precondition: new_with_config_no_outputs seeds the empty-string sentinel",
    );
    assert!(
        layout.workspaces.insert(orphan_id, orphan).is_none(),
        "fresh id is unique",
    );

    // Splice the orphan into alpha's view of mon_out at position
    // `len() - 1` (just before the trailing-empty bookend).
    let alpha_view = layout
        .activities
        .get_mut(alpha_id)
        .expect("alpha live")
        .views_mut()
        .get_mut(&mon_out)
        .expect("alpha holds a view on mon_out post-add_output");
    let insert_pos = alpha_view.len() - 1;
    alpha_view.insert(insert_pos, orphan_id);

    layout.verify_invariants();

    // Reload: keep only beta — drops Default and alpha. Both active and
    // previous are in remove_set, so the cascade falls through to the
    // first-declaration-order non-remove-set survivor (beta).
    let cfg_reload = [jiji_config::ActivityDecl {
        name: jiji_config::ActivityName("beta".to_owned()),
    }];
    layout
        .reconcile_activities_on_reload_remove(&cfg_reload)
        .expect("cascade to beta with orphan rebind must succeed");

    // (a) Cascade target is beta.
    assert_eq!(
        layout.active_activity_id(),
        beta_id,
        "cascade-target arm: previous (Default) was itself in remove_set, so cascade falls \
         through `previous.filter(...)` → first declaration-order non-remove-set survivor (beta)",
    );

    // (b) Orphan in beta's view of mon_out at position `len() - 1`.
    let beta_view = layout
        .activities
        .get(beta_id)
        .expect("beta live")
        .views()
        .get(&mon_out)
        .expect("beta's view on mon_out must exist post-cascade");
    let orphan_pos = beta_view
        .position_of(orphan_id)
        .expect("orphan must be in beta's view post-rebind");
    assert!(
        beta_view.len() >= 2,
        "post-rebind view must have ≥ 2 ids (orphan + trailing bookend)",
    );
    assert_eq!(
        orphan_pos,
        beta_view.len() - 2,
        "rebind inserts immediately before the trailing-empty bookend",
    );

    // (b2) Active-cursor patch landed: beta's view on mon_out had `len == 1`
    // pre-insert (only the trailing-empty bookend, since beta was a fresh
    // branch whose `ensure_all_activity_views` synthesized a singleton view), so the
    // cascade promotes the rebound orphan to the active cursor — the user
    // lands on the orphan that was active under the removed activity, not on
    // the empty bookend.
    assert_eq!(
        beta_view.active(),
        orphan_id,
        "fresh-branch singleton path must shift active cursor onto the rebound orphan",
    );

    // (c) Sentinel rebind landed: orphan now carries the real OutputId.
    let orphan_oid = layout
        .workspaces
        .get(&orphan_id)
        .expect("orphan still in pool")
        .output_id()
        .cloned()
        .expect("orphan output_id must be Some");
    assert_eq!(
        orphan_oid, mon_out,
        "sentinel `OutputId(\"\")` must be rewritten to the real output id",
    );

    // (d) Pool still contains the orphan.
    assert!(
        layout.workspaces.contains_key(&orphan_id),
        "orphan must survive in the pool — it is not exclusive to alpha",
    );

    // (e) Cross-field invariants intact.
    layout.verify_invariants();
}

#[test]
fn reconcile_remove_predicate_skips_workspace_already_anchored_by_surviving_view() {
    // Pins the Pass-2 predicate's correct skip behaviour. The orphan is
    // pre-spliced into beta's view of mon_out before reconcile fires.
    //
    // Under the view-membership predicate, a workspace already present in a
    // surviving activity's view on the same output is part of
    // `surviving_anchored` and is therefore never emitted by Pass 2. The
    // rebind path is bypassed at the predicate, not at a downstream guard —
    // the orphan-rebind body never executes for this orphan, so beta's view
    // structure is unchanged after reconcile.
    //
    // `occurrences == 1` holds because Pass 2 did not emit the orphan (not
    // because a rebind happened then was deduped). The discriminating
    // `view.len() == beta_len_before` assertion pins the latter: if the rebind
    // body had run, it would have grown the view.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);

    let cfg_init = [
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("Default".to_owned()),
        },
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("alpha".to_owned()),
        },
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("beta".to_owned()),
        },
    ];
    layout.reconcile_activities_on_reload_add(&cfg_init, &[]);
    let alpha_id = layout
        .activities
        .iter()
        .find(|a| a.name() == "alpha")
        .expect("alpha in pool")
        .id();
    let beta_id = layout
        .activities
        .iter()
        .find(|a| a.name() == "beta")
        .expect("beta in pool")
        .id();
    // Switch beta active first so `ensure_all_activity_views` materializes a
    // well-formed view (trailing empty bookend) for beta on mon_out, then
    // switch to alpha for the test's pre-reload state. Required because we
    // need to splice the orphan into beta's view as a non-last entry, which
    // would violate the Monitor invariant if beta's view were a bare
    // singleton.
    layout.switch_activity(beta_id);
    layout.switch_activity(alpha_id);

    let mon_out = layout.monitors[0].output_id();

    // Named for the same Monitor-invariant reason as the rebind test.
    let orphan_cfg = jiji_config::Workspace {
        name: jiji_config::workspace::WorkspaceName("ws_b".to_owned()),
        open_on_output: None,
        layout: None,
        activities: vec!["beta".to_owned()],
        sticky: None,
    };
    let orphan = Workspace::<TestWindow>::new_with_config_no_outputs(
        Some(orphan_cfg),
        HashSet::from([beta_id]),
        layout.clock.clone(),
        layout.options.clone(),
    );
    let orphan_id = orphan.id();
    assert!(
        layout.workspaces.insert(orphan_id, orphan).is_none(),
        "fresh id is unique",
    );

    // Splice the orphan into BOTH alpha's view (so it is a genuine orphan when
    // alpha is removed) and beta's view (so that Pass 2's `surviving_anchored`
    // set contains it and the orphan is never emitted by Pass 2).
    {
        let alpha_view = layout
            .activities
            .get_mut(alpha_id)
            .expect("alpha live")
            .views_mut()
            .get_mut(&mon_out)
            .expect("alpha holds a view on mon_out post-add_output");
        let insert_pos = alpha_view.len() - 1;
        alpha_view.insert(insert_pos, orphan_id);
    }
    {
        // Beta already has a view on mon_out (we toggled `switch_activity`
        // through beta in the setup). Splice the orphan in just before the
        // trailing-empty bookend — this puts it into `surviving_anchored`
        // so Pass 2 will not emit it.
        let v = layout
            .activities
            .get_mut(beta_id)
            .expect("beta live")
            .views_mut()
            .get_mut(&mon_out)
            .expect("beta has a view on mon_out from the setup switch_activity");
        let pos = v.len() - 1;
        v.insert(pos, orphan_id);
    }

    // Capture beta's view state before reconcile. Pre-reconcile beta's view
    // holds [orphan, bookend] (`len == 2`). Since the orphan is in
    // `surviving_anchored`, Pass 2 does not emit it; the rebind body never
    // runs; view length and active cursor remain unchanged post-reconcile.
    let beta_active_before = layout
        .activities
        .get(beta_id)
        .expect("beta live")
        .views()
        .get(&mon_out)
        .expect("beta has a view on mon_out from the setup switch_activity")
        .active();
    let beta_len_before = layout
        .activities
        .get(beta_id)
        .expect("beta live")
        .views()
        .get(&mon_out)
        .expect("beta has a view on mon_out from the setup switch_activity")
        .len();
    assert!(
        beta_len_before >= 2,
        "predicate precondition: beta's view must already hold orphan + bookend \
         so that the orphan is in `surviving_anchored` and Pass 2 skips it",
    );

    let cfg_reload = [jiji_config::ActivityDecl {
        name: jiji_config::ActivityName("beta".to_owned()),
    }];
    layout
        .reconcile_activities_on_reload_remove(&cfg_reload)
        .expect("reconcile must succeed when orphan is already anchored by surviving view");

    let beta_view = layout
        .activities
        .get(beta_id)
        .expect("beta live")
        .views()
        .get(&mon_out)
        .expect("beta's view on mon_out must exist post-cascade");
    let occurrences = beta_view
        .ids()
        .iter()
        .filter(|id| **id == orphan_id)
        .count();
    assert_eq!(
        occurrences, 1,
        "orphan was pre-existing in beta's view; Pass 2 did not emit it, so count stays 1",
    );

    // The rebind body never executed, so the view structure is unchanged: no
    // insert occurred and the active cursor was not patched.
    assert_eq!(
        beta_view.len(),
        beta_len_before,
        "Pass 2 did not emit the orphan; rebind body was bypassed — view length unchanged",
    );
    assert_eq!(
        beta_view.active(),
        beta_active_before,
        "active-cursor patch must not fire; rebind body was bypassed at the predicate",
    );

    layout.verify_invariants();
}

#[test]
fn reconcile_remove_rebinds_orphan_when_workspace_tagged_to_both_removed_and_surviving_activity() {
    // Mixed-tag case: the orphan workspace carries `activities = {alpha, gamma}`
    // — alpha is in remove_set, gamma survives. With the old disjoint-set
    // predicate (`!ws.activities().iter().any(|aid| remove_set.contains(aid))`),
    // `disjoint` is `false` (alpha ∈ remove_set ∩ ws.activities), so the old
    // predicate would skip the orphan. The remove pass then prunes alpha from
    // the workspace's `activities`, leaving `{gamma}`, but gamma's view of mon_out never
    // contained it (it was only in alpha's view via the sentinel-output-id
    // lift path). The pool-keys-equal-union invariant is violated.
    //
    // The corrected predicate is view-membership-based: "not anchored by any
    // surviving activity's view of the same output" — gamma's view of mon_out
    // does NOT contain the orphan, so it is correctly emitted as an orphan and
    // rebound into the cascade target's view.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);

    let cfg_init = [
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("Default".to_owned()),
        },
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("alpha".to_owned()),
        },
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("gamma".to_owned()),
        },
    ];
    layout.reconcile_activities_on_reload_add(&cfg_init, &[]);
    let alpha_id = layout
        .activities
        .iter()
        .find(|a| a.name() == "alpha")
        .expect("alpha in pool")
        .id();
    let gamma_id = layout
        .activities
        .iter()
        .find(|a| a.name() == "gamma")
        .expect("gamma in pool")
        .id();

    // Switch alpha active so alpha is the seed-active activity before reload.
    layout.switch_activity(alpha_id);
    assert_eq!(layout.active_activity_id(), alpha_id);

    let mon_out = layout.monitors[0].output_id();

    // Mint orphan tagged to BOTH alpha and gamma — the mixed-tag case.
    let orphan_cfg = jiji_config::Workspace {
        name: jiji_config::workspace::WorkspaceName("ws_shared".to_owned()),
        open_on_output: None,
        layout: None,
        activities: vec!["alpha".to_owned(), "gamma".to_owned()],
        sticky: None,
    };
    let orphan = Workspace::<TestWindow>::new_with_config_no_outputs(
        Some(orphan_cfg),
        HashSet::from([alpha_id, gamma_id]),
        layout.clock.clone(),
        layout.options.clone(),
    );
    let orphan_id = orphan.id();
    assert_eq!(
        orphan.output_id().map(|oid| oid.as_str()),
        Some(""),
        "precondition: new_with_config_no_outputs seeds the empty-string sentinel",
    );
    assert!(
        layout.workspaces.insert(orphan_id, orphan).is_none(),
        "fresh id is unique",
    );

    // Splice the orphan into alpha's view of mon_out (the sentinel-lift path),
    // but NOT into gamma's view. This is the precondition that exposes the bug:
    // gamma is a surviving activity but its view on mon_out does not contain
    // the orphan, so the orphan is unanchored after the remove pass.
    {
        let alpha_view = layout
            .activities
            .get_mut(alpha_id)
            .expect("alpha live")
            .views_mut()
            .get_mut(&mon_out)
            .expect("alpha holds a view on mon_out post-add_output");
        let insert_pos = alpha_view.len() - 1;
        alpha_view.insert(insert_pos, orphan_id);
    }

    layout.verify_invariants();

    // Reload: keep only gamma — drops Default and alpha.
    // Cascade target: alpha was active, previous = Default (in remove_set) →
    // falls through to first-declaration-order non-remove-set survivor (gamma).
    let cfg_reload = [jiji_config::ActivityDecl {
        name: jiji_config::ActivityName("gamma".to_owned()),
    }];
    layout
        .reconcile_activities_on_reload_remove(&cfg_reload)
        .expect("mixed-tag orphan rebind must succeed");

    // (a) Cascade target is gamma.
    assert_eq!(
        layout.active_activity_id(),
        gamma_id,
        "cascade-target arm: previous_id == Some(Default) but Default ∈ remove_set, \
         so cascade falls through `previous.filter(...)` → first declaration-order \
         non-remove-set survivor (gamma)",
    );

    // (b) Orphan is now in gamma's view of mon_out.
    let gamma_view = layout
        .activities
        .get(gamma_id)
        .expect("gamma live")
        .views()
        .get(&mon_out)
        .expect("gamma's view on mon_out must exist post-cascade");
    let orphan_pos = gamma_view
        .position_of(orphan_id)
        .expect("orphan must be in gamma's view post-rebind (mixed-tag case)");
    assert!(
        gamma_view.len() >= 2,
        "post-rebind view must have ≥ 2 ids (orphan + trailing bookend)",
    );
    assert_eq!(
        orphan_pos,
        gamma_view.len() - 2,
        "rebind inserts immediately before the trailing-empty bookend",
    );

    // (b2) gamma's view was a fresh-branch singleton (len == 1) before the
    // insert, so the active-cursor patch must fire — the user lands on the
    // orphan, not on the empty bookend.
    assert_eq!(
        gamma_view.active(),
        orphan_id,
        "fresh-branch singleton path must shift active cursor onto the rebound orphan",
    );

    // (c) Orphan still in pool; sentinel rewritten.
    assert!(
        layout.workspaces.contains_key(&orphan_id),
        "orphan must survive in the pool — it is not exclusive to alpha",
    );
    let orphan_oid = layout
        .workspaces
        .get(&orphan_id)
        .expect("orphan still in pool")
        .output_id()
        .cloned()
        .expect("orphan output_id must be Some");
    assert_eq!(
        orphan_oid, mon_out,
        "sentinel `OutputId(\"\")` must be rewritten to the real output id",
    );

    // (d) Cross-field invariants intact.
    layout.verify_invariants();
}

#[test]
fn reconcile_remove_rejects_on_hard_block_when_cascade_required() {
    // Active in remove_set, plus an in-flight interactive_move: cascade is
    // required but is_activity_switch_hard_blocked returns Some(_). Expected:
    // Err(HardBlockedCascade), no mutation.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);

    let cfg_init = [
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("Default".to_owned()),
        },
        jiji_config::ActivityDecl {
            name: jiji_config::ActivityName("Beta".to_owned()),
        },
    ];
    layout.reconcile_activities_on_reload_add(&cfg_init, &[]);
    let beta_id = layout
        .activities
        .iter()
        .find(|a| a.name() == "Beta")
        .expect("Beta in pool")
        .id();
    layout.switch_activity(beta_id);
    assert_eq!(layout.active_activity_id(), beta_id);

    // Arm a DnD session via the public API (same pattern as
    // `is_activity_switch_hard_blocked_returns_some_during_dnd`). DnD is
    // preferred over `interactive_move_begin` because it does not require a
    // window to exist on the active activity — our test has no windows and
    // `interactive_move_begin` would fail to arm without one.
    let output = layout
        .outputs()
        .find(|o| o.name() == "output1")
        .cloned()
        .expect("output1 must exist");
    layout.dnd_update(output, Point::from((0., 0.)));
    assert!(
        matches!(
            layout.is_activity_switch_hard_blocked(),
            Some(super::ActivitySwitchBlock::Dnd),
        ),
        "precondition: hard-blocked via DnD",
    );

    let pool_before = layout.activities.len();
    let ws_before = layout.workspaces.len();
    let active_before = layout.active_activity_id();
    let previous_before = layout.activities.previous_id();

    let cfg_reload = [jiji_config::ActivityDecl {
        name: jiji_config::ActivityName("Default".to_owned()),
    }];
    let err = layout
        .reconcile_activities_on_reload_remove(&cfg_reload)
        .expect_err("hard-blocked cascade must err");

    match err {
        ReloadActivityRemovalError::HardBlockedCascade {
            activity_name,
            block,
        } => {
            assert_eq!(activity_name, "Beta");
            assert_eq!(block, super::ActivitySwitchBlock::Dnd);
        }
        other => panic!("unexpected error variant: {other:?}"),
    }

    assert_eq!(layout.activities.len(), pool_before, "pool unchanged");
    assert_eq!(layout.workspaces.len(), ws_before, "workspaces unchanged");
    assert_eq!(
        layout.active_activity_id(),
        active_before,
        "active cursor unchanged on Err",
    );
    assert_eq!(
        layout.activities.previous_id(),
        previous_before,
        "previous cursor unchanged on Err",
    );

    // Clean up the hard-block so verify_invariants runs under the normal gate.
    layout.dnd_end();
    layout.verify_invariants();
}
// -- pick_activity_for_hidden_window tests -------------------------

/// Build a no-monitor `Layout` with three activities (seed/alpha, beta, gamma),
/// pick one workspace, and overwrite its `activities` tag set. Returns
/// `(layout, seed_id, beta_id, gamma_id, ws_id)`. The caller then calls
/// `layout.pick_activity_for_hidden_window(ws_id, hint)` to exercise each tier.
fn prepare_picker_layout(
    ws_tags: &[&str],
) -> (
    Layout<TestWindow>,
    ActivityId,
    ActivityId,
    ActivityId,
    WorkspaceId,
) {
    // Use `AddOutput` so the pool contains at least one workspace — the
    // no-monitor default ctor builds an empty pool, and the picker needs a
    // pool workspace to read `activities` from.
    let mut layout = check_ops([Op::AddOutput(1)]);
    let seed_id = layout.active_activity_id();

    let beta = super::activity::Activity::new_runtime("beta".to_owned());
    let beta_id = beta.id();
    test_insert_activity(&mut layout, beta);

    let gamma = super::activity::Activity::new_runtime("gamma".to_owned());
    let gamma_id = gamma.id();
    test_insert_activity(&mut layout, gamma);

    // Pick any pool workspace (there is exactly one — seeded by the
    // no-monitor Default ctor via the disconnected-pool path).
    let ws_id = *layout
        .workspaces
        .keys()
        .next()
        .expect("default layout has at least one pool workspace");

    // Overwrite the `activities` tag set. `ws_tags` names the activities
    // the workspace should be tagged with — all three names resolve to the
    // ids captured above.
    let tag_set: HashSet<ActivityId> = ws_tags
        .iter()
        .map(|name| match *name {
            "alpha" => seed_id,
            "beta" => beta_id,
            "gamma" => gamma_id,
            other => panic!("unknown tag name in test helper: {other}"),
        })
        .collect();
    {
        let ws = layout
            .workspaces
            .get_mut(&ws_id)
            .expect("ws_id must be a pool key");
        ws.activities = tag_set;
    }

    (layout, seed_id, beta_id, gamma_id, ws_id)
}

#[test]
fn pick_activity_for_hidden_window_returns_hint_when_hint_in_ws_activities() {
    // Tier 1: hint is in `ws.activities` and is not the active activity.
    // Pin that the hint is honored even when `previous_id` and declaration-
    // order also have valid candidates.
    let (mut layout, alpha_id, beta_id, gamma_id, ws_id) =
        prepare_picker_layout(&["beta", "gamma"]);
    // Populate previous_id = gamma so tier 2 has a valid candidate; the test
    // must still pick beta via tier 1.
    layout.switch_activity(gamma_id);
    layout.switch_activity(alpha_id);
    assert_eq!(layout.activities.previous_id(), Some(gamma_id));

    // Hint = beta. Expected: beta (tier 1 wins over tier 2 / tier 3).
    let target = layout.pick_activity_for_hidden_window(ws_id, Some(beta_id));
    assert_eq!(
        target, beta_id,
        "tier 1: in-set hint must win over previous_id and declaration-order candidates",
    );
}

#[test]
fn pick_activity_for_hidden_window_falls_to_previous_when_hint_stale() {
    // Tier 2: hint is `None` or not in `ws.activities`, but `previous_id`
    // is valid. Pin that previous_id takes precedence over declaration-order.
    let (mut layout, alpha_id, beta_id, gamma_id, ws_id) =
        prepare_picker_layout(&["beta", "gamma"]);
    // Make previous = gamma, active = alpha.
    layout.switch_activity(gamma_id);
    layout.switch_activity(alpha_id);
    assert_eq!(layout.activities.previous_id(), Some(gamma_id));

    // Hint None → tier 1 skipped. previous = gamma, in ws.activities, != active → tier 2.
    let target = layout.pick_activity_for_hidden_window(ws_id, None);
    assert_eq!(
        target, gamma_id,
        "tier 2: previous_id wins when hint is absent and previous is a valid candidate",
    );

    // Hint = alpha (active) → filtered by tier 1's `hint != active_id()`
    // guard. Tier 2 still fires.
    let target = layout.pick_activity_for_hidden_window(ws_id, Some(alpha_id));
    assert_eq!(
        target, gamma_id,
        "tier 2: hint == active is filtered, previous wins",
    );

    // Hint = an id that is not in `ws.activities` at all (synthesize a
    // never-inserted id via `specific(u64::MAX)`). Tier 1 filter rejects it
    // via `activities.contains`; tier 2 still fires.
    let target =
        layout.pick_activity_for_hidden_window(ws_id, Some(ActivityId::specific(u64::MAX)));
    assert_eq!(
        target, gamma_id,
        "tier 2: out-of-set hint is filtered by `activities.contains`, previous wins",
    );

    // Silence the unused-variable warning on beta_id; no branch here uses
    // it, but the helper minted it for symmetry with other tests.
    let _ = beta_id;
}

#[test]
fn pick_activity_for_hidden_window_falls_to_display_order_when_previous_unavailable() {
    // Tier 3: hint absent, `previous_id` unavailable (either None or not in
    // `ws.activities`). The picker must return the first activity in
    // declaration order that is in `ws.activities` and != active.
    //
    // Declaration order on the helper is seed → beta → gamma (IndexMap
    // preserves insertion order). With ws tagged {beta, gamma} and active =
    // alpha, the first non-active candidate is beta.
    let (layout, alpha_id, beta_id, _gamma, ws_id) = prepare_picker_layout(&["beta", "gamma"]);
    // Fresh layout → previous_id is None.
    assert_eq!(layout.activities.previous_id(), None);
    assert_eq!(layout.active_activity_id(), alpha_id);

    let target = layout.pick_activity_for_hidden_window(ws_id, None);
    assert_eq!(
        target, beta_id,
        "tier 3: first declaration-order candidate in ws.activities wins",
    );
}

#[test]
fn pick_activity_for_hidden_window_skips_active_in_all_tiers() {
    // Pin that every tier excludes the currently-active activity. ws is
    // tagged with all three ids, so a bug that omits the `!= active` guard
    // on any tier would return alpha (the active id).
    let (mut layout, alpha_id, beta_id, gamma_id, ws_id) =
        prepare_picker_layout(&["alpha", "beta", "gamma"]);
    // Populate previous_id = gamma so tier 2 has a concrete non-active
    // candidate; we still want to check that no tier returns alpha.
    layout.switch_activity(gamma_id);
    layout.switch_activity(alpha_id);

    // Tier 1: hint = alpha (the active). Filtered → falls to tier 2
    // (previous = gamma, != active, in set).
    let target = layout.pick_activity_for_hidden_window(ws_id, Some(alpha_id));
    assert_ne!(target, alpha_id, "tier 1 must not return the active id");
    assert_eq!(target, gamma_id, "tier 2 fires after tier 1 filters alpha");

    // Tier 1 honored (non-active hint).
    let target = layout.pick_activity_for_hidden_window(ws_id, Some(beta_id));
    assert_ne!(target, alpha_id);
    assert_eq!(target, beta_id);
}

#[test]
fn pick_activity_for_hidden_window_degenerate_single_activity_on_hidden_ws() {
    // Degenerate arm: workspace tagged with *only* one hidden activity (beta).
    // Hint absent, previous_id absent. Tier 3 must return beta in declaration
    // order, not fall into the `unreachable!` tail (which is reserved for the
    // empty-set invariant break) or the trailing `active_id()` return.
    let (layout, alpha_id, beta_id, _gamma, ws_id) = prepare_picker_layout(&["beta"]);
    assert_eq!(layout.activities.previous_id(), None);
    assert_eq!(layout.active_activity_id(), alpha_id);

    let target = layout.pick_activity_for_hidden_window(ws_id, None);
    assert_eq!(
        target, beta_id,
        "single-activity hidden ws: tier 3 picks beta",
    );
}

// -- FocusWindow hard-block gate --------------------------

#[test]
fn focus_window_hard_block_gate_fires_before_switch() {
    // Pins the hard-block discipline at the Layout level:
    // when `is_activity_switch_hard_blocked()` is `Some(_)`, the caller is
    // expected to return `Err(block)` without mutating the active activity.
    // This test performs the structural inspection the spec allows
    // ("Layout<TestWindow>-level structural inspection") instead of wiring
    // up a full `do_action_inner` fixture because:
    //   (a) the gate is a two-line test that directly reads `is_activity_switch_hard_blocked`, and
    //   (b) the production `Action::FocusWindow` arm early-returns `Err(block)` before
    //       calling `switch_activity`, so no state mutation happens regardless.
    //
    // Steps:
    //   1. Two activities (alpha/seed, beta). Window added under alpha.
    //   2. Switch to beta — window is now on a hidden workspace.
    //   3. Arm DnD to trigger the hard block.
    //   4. Verify `is_activity_switch_hard_blocked()` is `Some(Dnd)`.
    //   5. Verify `window_ws_and_activity_hint` resolves the window's workspace.
    //   6. Verify `pick_activity_for_hidden_window` picks a non-active target (the production gate
    //      would fire before reaching `switch_activity`).
    //   7. Verify `active_activity_id()` is unchanged (simulates Err path — no mutation).
    let ops = [
        Op::AddOutput(1),
        Op::AddWindow {
            params: TestWindowParams::new(0),
        },
    ];
    let mut layout = check_ops(ops);
    let alpha_id = layout.active_activity_id();

    // Mint a beta activity and switch to it so the window is hidden.
    let beta = super::activity::Activity::new_runtime("beta".to_owned());
    let beta_id = beta.id();
    test_insert_activity(&mut layout, beta);
    layout.switch_activity(beta_id);
    assert_eq!(
        layout.active_activity_id(),
        beta_id,
        "precondition: active = beta"
    );

    // Arm DnD — no window required (same recipe as
    // `is_activity_switch_hard_blocked_returns_some_during_dnd` at line 5747).
    let output = layout
        .outputs()
        .find(|o| o.name() == "output1")
        .cloned()
        .expect("output1 must exist after AddOutput(1)");
    layout.dnd_update(output, Point::from((0., 0.)));
    assert_eq!(
        layout.is_activity_switch_hard_blocked(),
        Some(super::ActivitySwitchBlock::Dnd),
        "precondition: DnD must hard-block the activity switch",
    );

    // The window must be resolvable (mirrors production `window_ws_and_activity_hint`).
    // For TestWindow, W::Id = usize, so pass &0 to look up window 0.
    let ws_id = layout
        .window_ws_and_activity_hint(&0usize)
        .expect("window 0 must be tracked in the workspace pool");

    // Picker must return a non-active candidate (alpha) — the hard-block gate
    // in production fires AFTER picking the target, before `switch_activity`.
    let target = layout.pick_activity_for_hidden_window(ws_id, None);
    assert_ne!(
        target, beta_id,
        "picker must return a non-active activity for the hidden window",
    );
    assert_eq!(
        target, alpha_id,
        "tier 3 (declaration order) picks alpha for the hidden window under beta-active",
    );

    // No mutation: active activity must still be beta (caller would Err-return).
    assert_eq!(
        layout.active_activity_id(),
        beta_id,
        "hard-block gate simulation: active activity must be unchanged when Err is returned",
    );

    // Clean up so verify_invariants runs under the normal gate.
    layout.dnd_end();
    layout.verify_invariants();
}

// --- AddWorkspaceToActivity ------------------------------------------------

#[test]
fn add_workspace_to_activity_appends_to_dormant_view() {
    // Dormant activity `beta` has a pre-existing view on the output containing
    // a single id. Calling `add_workspace_to_activity(ws, beta)` for a
    // workspace bound to that output must append the id to beta's view and
    // union beta into the workspace's `activities` set.
    //
    // AddNamedWorkspace is load-bearing: it gives alpha's view ≥2 entries so
    // we can seed beta's view with one id and Add a distinct one.
    let ops = [
        Op::AddOutput(1),
        Op::AddNamedWorkspace {
            ws_name: 1,
            output_name: None,
            layout_config: None,
        },
    ];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let mon_out = layout.monitors[0].output_id();

    // Create beta (runtime, dormant) and give it a view on mon_out with a
    // single pre-existing workspace id (the first in alpha's view).
    let beta = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");
    let seed_ws_id = layout
        .active_view(&mon_out)
        .ids()
        .first()
        .copied()
        .expect("alpha's view must have a workspace");
    test_override_activity_view(
        &mut layout,
        beta,
        mon_out.clone(),
        WorkspaceView::new(vec![seed_ws_id], 0),
    );

    // Pick a workspace from the alpha view that is NOT in beta's view, so
    // Add appends a distinct id.
    let target_ws_id = {
        let ids = layout.active_view(&mon_out).ids().to_vec();
        *ids.iter()
            .find(|id| **id != seed_ws_id)
            .expect("alpha view must have >= 2 entries")
    };
    // Sanity: target is currently alpha-only.
    assert_eq!(
        layout
            .workspaces
            .get(&target_ws_id)
            .expect("target ws in pool")
            .activities(),
        &HashSet::from([alpha]),
    );

    let (ws_id, act_id) = layout
        .add_workspace_to_activity(
            Some(WorkspaceReference::Id(target_ws_id.get())),
            &ActivityReferenceArg::Id(beta.get()),
        )
        .expect("add must succeed on resolvable refs");
    assert_eq!(ws_id, target_ws_id);
    assert_eq!(act_id, beta);

    // ws.activities gained beta.
    assert_eq!(
        layout
            .workspaces
            .get(&target_ws_id)
            .expect("target ws live")
            .activities(),
        &HashSet::from([alpha, beta]),
    );
    // beta's view on the output now appends target_ws_id after seed_ws_id.
    assert_eq!(
        layout
            .activities
            .get(beta)
            .expect("beta live")
            .views()
            .get(&mon_out)
            .expect("beta has view")
            .ids(),
        &[seed_ws_id, target_ws_id],
    );

    layout.verify_invariants();
}

#[test]
fn add_workspace_to_activity_no_op_when_already_member() {
    // Workspace is already tagged with beta. A second Add must be a no-op:
    // `activities` unchanged, beta's view unchanged, invariants still green.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let mon_out = layout.monitors[0].output_id();

    let beta = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");
    let target_ws_id = layout
        .active_view(&mon_out)
        .ids()
        .first()
        .copied()
        .expect("alpha view has a workspace");

    // Seed beta into the workspace manually, plus a view entry of the same
    // id. Add must leave both untouched.
    layout
        .workspaces
        .get_mut(&target_ws_id)
        .expect("target ws live")
        .activities = [alpha, beta].into_iter().collect();
    test_override_activity_view(
        &mut layout,
        beta,
        mon_out.clone(),
        WorkspaceView::new(vec![target_ws_id], 0),
    );

    let before_set = layout
        .workspaces
        .get(&target_ws_id)
        .expect("live")
        .activities()
        .clone();
    let before_ids = layout
        .activities
        .get(beta)
        .expect("live")
        .views()
        .get(&mon_out)
        .expect("view")
        .ids()
        .to_vec();

    let (_, act_id) = layout
        .add_workspace_to_activity(
            Some(WorkspaceReference::Id(target_ws_id.get())),
            &ActivityReferenceArg::Id(beta.get()),
        )
        .expect("no-op add must return Ok");
    assert_eq!(act_id, beta);

    assert_eq!(
        layout
            .workspaces
            .get(&target_ws_id)
            .expect("live")
            .activities(),
        &before_set,
    );
    assert_eq!(
        layout
            .activities
            .get(beta)
            .expect("live")
            .views()
            .get(&mon_out)
            .expect("view")
            .ids(),
        &before_ids[..],
    );

    layout.verify_invariants();
}

#[test]
fn add_workspace_to_activity_appends_to_materialized_bookend_view() {
    // The per-activity bookend invariant materializes a bookend view for every newly-created
    // activity on every connected monitor. `add_workspace_to_activity` for a workspace bound
    // to that monitor must then append the id to beta's materialized view (just before the
    // trailing-empty bookend — preserved by the materializer at insert time). This replaces
    // the pre-invariant "Add does not fabricate a view; rebuild happens lazily on switch"
    // assertion: views are now eager.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let mon_out = layout.monitors[0].output_id();

    let beta = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");

    // Materialized view for beta on mon_out: a single fresh trailing empty.
    let beta_view_before = layout
        .activities
        .get(beta)
        .expect("beta live")
        .views()
        .get(&mon_out)
        .expect("materializer must install a view for beta on mon_out")
        .clone();
    assert_eq!(
        beta_view_before.len(),
        1,
        "fresh materialized view holds exactly one trailing-empty bookend id",
    );

    let target_ws_id = layout
        .active_view(&mon_out)
        .ids()
        .first()
        .copied()
        .expect("alpha view has a workspace");

    layout
        .add_workspace_to_activity(
            Some(WorkspaceReference::Id(target_ws_id.get())),
            &ActivityReferenceArg::Id(beta.get()),
        )
        .expect("add must succeed");

    assert_eq!(
        layout
            .workspaces
            .get(&target_ws_id)
            .expect("live")
            .activities(),
        &HashSet::from([alpha, beta]),
    );

    // Beta's view gained the target id (appended at the tail per `add_workspace_to_activity`'s
    // position-invariant insert).
    let beta_view_after = layout
        .activities
        .get(beta)
        .expect("beta live")
        .views()
        .get(&mon_out)
        .expect("beta view persisted");
    assert!(
        beta_view_after.ids().contains(&target_ws_id),
        "Add must union the workspace id into beta's view",
    );

    layout.verify_invariants();
}

#[test]
fn add_workspace_to_activity_activity_not_found() {
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let mon_out = layout.monitors[0].output_id();
    let target_ws_id = layout
        .active_view(&mon_out)
        .ids()
        .first()
        .copied()
        .expect("view has ws");

    let err = layout
        .add_workspace_to_activity(
            Some(WorkspaceReference::Id(target_ws_id.get())),
            &ActivityReferenceArg::Id(u64::MAX),
        )
        .expect_err("unknown activity must err");
    assert_eq!(err, AddWorkspaceToActivityError::ActivityNotFound);
}

#[test]
fn add_workspace_to_activity_workspace_not_found() {
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let _beta = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");
    let beta_id = layout
        .activities
        .iter()
        .find(|a| a.name() == "Beta")
        .expect("beta live")
        .id();

    let err = layout
        .add_workspace_to_activity(
            Some(WorkspaceReference::Id(u64::MAX)),
            &ActivityReferenceArg::Id(beta_id.get()),
        )
        .expect_err("unknown workspace must err");
    assert_eq!(err, AddWorkspaceToActivityError::WorkspaceNotFound);
}

// --- RemoveWorkspaceFromActivity -------------------------------------------

#[test]
fn remove_workspace_from_activity_drops_single_entry_view_inactive() {
    // Dormant activity beta has a view containing the target as its sole non-bookend entry;
    // its workspace is {alpha, beta}. Remove(ws, beta) must prune beta from ws.activities and
    // drop the entry from beta's view, leaving the materializer-side bookend in place (under
    // the per-activity bookend invariant the materializer re-installs a fresh bookend if the
    // single-entry path drops the view entirely).
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let mon_out = layout.monitors[0].output_id();

    let beta = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");
    let target_ws_id = layout
        .active_view(&mon_out)
        .ids()
        .first()
        .copied()
        .expect("alpha view has ws");

    // Seed shared membership + a dormant 2-entry view (target + materializer's bookend kept
    // as the trailing empty). Under the per-view bookend invariant, every view on a connected
    // monitor must end with an empty unnamed workspace.
    layout
        .workspaces
        .get_mut(&target_ws_id)
        .expect("live")
        .activities = [alpha, beta].into_iter().collect();
    let beta_bottom = test_mint_empty_for(&mut layout, 0, beta);
    test_override_activity_view(
        &mut layout,
        beta,
        mon_out.clone(),
        WorkspaceView::new(vec![target_ws_id, beta_bottom], 0),
    );

    layout
        .remove_workspace_from_activity(
            Some(WorkspaceReference::Id(target_ws_id.get())),
            &ActivityReferenceArg::Id(beta.get()),
        )
        .expect("remove must succeed");

    assert_eq!(
        layout
            .workspaces
            .get(&target_ws_id)
            .expect("live")
            .activities(),
        &HashSet::from([alpha]),
        "beta must be pruned from ws.activities",
    );
    // The target entry is gone from beta's view, but the trailing-empty bookend remains
    // because every view on a connected monitor must hold one under the per-activity bookend
    // invariant.
    let beta_view = layout
        .activities
        .get(beta)
        .expect("live")
        .views()
        .get(&mon_out)
        .expect("beta view persists (carries the bookend)");
    assert!(
        !beta_view.ids().contains(&target_ws_id),
        "target id must be removed from beta's view",
    );
    assert_eq!(
        beta_view.len(),
        1,
        "only the trailing-empty bookend remains"
    );

    layout.verify_invariants();
}

#[test]
fn remove_workspace_from_activity_patches_multi_entry_view() {
    // Dormant activity beta has a view with ids [a, target, b], active=b.
    // Remove(target, beta) must call WorkspaceView::remove_at which patches
    // active / previous per its contract (active is unaffected; the view
    // shrinks by one).
    //
    // Fixture: three named workspaces tagged {alpha, beta}. Named so they
    // don't trip the "non-terminal empty unnamed" bookend check at
    // monitor.rs:1772. Appended to alpha's active view AND placed in
    // beta's dormant view so pool-keys equality holds pre-call.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let mon_out = layout.monitors[0].output_id();
    let mon_output = layout.monitors[0].output.clone();

    let beta = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");

    // Allocate three named workspaces tagged {alpha, beta} via
    // `Workspace::new_with_config(&output, ...)` so their `output_id`
    // matches `mon_out` directly. Names clear the "non-terminal empty
    // unnamed" bookend check (monitor.rs:1772) when we insert them into
    // alpha's active view below — that placement is required because
    // the workspaces stay in alpha's set after Remove (ws.activities
    // becomes {alpha}) and the pool-keys equality invariant would
    // otherwise flag them as zombies.
    let make_named = |layout: &mut Layout<TestWindow>, name: &str| -> WorkspaceId {
        let ws = Workspace::<TestWindow>::new_with_config(
            &mon_output,
            Some(WorkspaceConfig {
                name: WorkspaceName(name.to_owned()),
                open_on_output: None,
                layout: None,
                activities: Vec::new(),
                sticky: None,
            }),
            [alpha, beta].into_iter().collect(),
            layout.clock.clone(),
            layout.options.clone(),
        );
        let id = ws.id();
        layout.workspaces.insert(id, ws);
        id
    };
    let a_id = make_named(&mut layout, "named-a");
    let target_id = make_named(&mut layout, "named-target");
    let b_id = make_named(&mut layout, "named-b");

    // Append into alpha's active view just before the trailing-empty
    // bookend so trailing-empty discipline is preserved.
    {
        let alpha_view = layout
            .activities
            .active_mut()
            .views_mut()
            .get_mut(&mon_out)
            .expect("alpha has view on mon_out");
        let tail = alpha_view.len() - 1;
        alpha_view.insert(tail, a_id);
        alpha_view.insert(tail + 1, target_id);
        alpha_view.insert(tail + 2, b_id);
    }

    // Give beta a 4-entry dormant view with active = b_id; the trailing empty satisfies the
    // per-view bookend invariant on beta's dormant view (every view on a connected monitor
    // must end with an empty unnamed workspace).
    let beta_bottom = test_mint_empty_for(&mut layout, 0, beta);
    test_override_activity_view(
        &mut layout,
        beta,
        mon_out.clone(),
        WorkspaceView::new(vec![a_id, target_id, b_id, beta_bottom], 2),
    );

    // Sanity: precondition invariants hold.
    layout.verify_invariants();

    layout
        .remove_workspace_from_activity(
            Some(WorkspaceReference::Id(target_id.get())),
            &ActivityReferenceArg::Id(beta.get()),
        )
        .expect("remove must succeed");

    let view = layout
        .activities
        .get(beta)
        .expect("live")
        .views()
        .get(&mon_out)
        .expect("view still present (len was 4)");
    assert_eq!(view.ids(), &[a_id, b_id, beta_bottom]);
    assert_eq!(view.active(), b_id);
    // target_id's pool entry still exists (still has alpha tag); only beta was pruned.
    assert_eq!(
        layout
            .workspaces
            .get(&target_id)
            .expect("live")
            .activities(),
        &HashSet::from([alpha]),
    );

    layout.verify_invariants();
}

#[test]
fn remove_workspace_from_activity_active_activity_recreates_view() {
    // Remove from the active activity in a way that drops its view's last
    // entry on a connected monitor. `ensure_all_activity_views` must reinstate
    // the view so `active.views.len() == monitors.len()` holds.
    //
    // Fixture shape: ws_id is tagged {alpha, beta}, appears in alpha's
    // active view (single entry) AND in beta's dormant view on mon_out (so
    // the pool-keys equality invariant holds at call time — every pool id
    // is reachable through at least one view).
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let mon_out = layout.monitors[0].output_id();
    let output = layout.monitors[0].output.clone();

    let beta = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");

    // Allocate a fresh workspace tagged {alpha, beta} so the Remove is
    // len_before=2 (not LastActivity).
    let ws = Workspace::new(
        &output,
        [alpha, beta].into_iter().collect(),
        layout.clock.clone(),
        layout.options.clone(),
    );
    let ws_id = ws.id();
    assert!(layout.workspaces.insert(ws_id, ws).is_none());

    // Destroy the original alpha bookend from the pool before overwriting
    // alpha's view, so we leave no orphan. The bookend was untagged from
    // any view once we overwrite, and it would stay in the pool as a
    // zombie and trip the pool-keys equality invariant.
    let original_ids: Vec<_> = layout.active_view(&mon_out).ids().to_vec();

    // Overwrite alpha's view on mon_out with [ws_id] and drop the original
    // bookend from the pool.
    layout
        .activities
        .active_mut()
        .views_mut()
        .insert(mon_out.clone(), WorkspaceView::new(vec![ws_id], 0));
    for id in &original_ids {
        // Only drop ids that no other view still references (here: all of
        // them — alpha was the only activity and we just overwrote its
        // view, and beta's view doesn't exist yet).
        let _ = layout.workspaces.remove(id);
    }

    // Place ws_id in beta's dormant view on mon_out too so pool-keys
    // equality holds pre-call.
    test_override_activity_view(
        &mut layout,
        beta,
        mon_out.clone(),
        WorkspaceView::new(vec![ws_id], 0),
    );

    // Sanity: precondition invariants hold.
    layout.verify_invariants();
    assert_eq!(layout.active_view(&mon_out).ids(), &[ws_id]);

    layout
        .remove_workspace_from_activity(
            Some(WorkspaceReference::Id(ws_id.get())),
            &ActivityReferenceArg::Id(alpha.get()),
        )
        .expect("remove from active activity must succeed");

    // After the mutation, alpha's view for mon_out must exist (ensure_all_activity_views reinstated
    // it) and must NOT contain ws_id (ws no longer carries alpha).
    let reinstated = layout
        .activities
        .active()
        .views()
        .get(&mon_out)
        .expect("ensure_all_activity_views must reinstate alpha's view on mon_out");
    assert!(!reinstated.ids().contains(&ws_id));
    assert!(!reinstated.ids().is_empty(), "reinstated view is non-empty");

    // ws.activities lost alpha but kept beta.
    assert_eq!(
        layout.workspaces.get(&ws_id).expect("live").activities(),
        &HashSet::from([beta]),
    );

    layout.verify_invariants();
}

#[test]
fn remove_workspace_from_activity_last_activity_errors() {
    // ws.activities = {alpha}. Remove(alpha) would empty the set — must Err
    // with LastActivity and mutate NOTHING (guard-before-mutate).
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let mon_out = layout.monitors[0].output_id();

    let target_ws_id = layout
        .active_view(&mon_out)
        .ids()
        .first()
        .copied()
        .expect("view has ws");

    let acts_before = layout
        .workspaces
        .get(&target_ws_id)
        .expect("live")
        .activities()
        .clone();
    assert_eq!(acts_before, HashSet::from([alpha]));
    let view_ids_before = layout.active_view(&mon_out).ids().to_vec();

    let err = layout
        .remove_workspace_from_activity(
            Some(WorkspaceReference::Id(target_ws_id.get())),
            &ActivityReferenceArg::Id(alpha.get()),
        )
        .expect_err("removing last activity must err");
    assert_eq!(err, RemoveWorkspaceFromActivityError::LastActivity);

    // Zero mutation.
    assert_eq!(
        layout
            .workspaces
            .get(&target_ws_id)
            .expect("live")
            .activities(),
        &acts_before,
    );
    assert_eq!(layout.active_view(&mon_out).ids(), &view_ids_before[..]);
}

#[test]
fn remove_workspace_from_activity_activity_not_found() {
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let mon_out = layout.monitors[0].output_id();
    let target_ws_id = layout
        .active_view(&mon_out)
        .ids()
        .first()
        .copied()
        .expect("view has ws");

    let err = layout
        .remove_workspace_from_activity(
            Some(WorkspaceReference::Id(target_ws_id.get())),
            &ActivityReferenceArg::Id(u64::MAX),
        )
        .expect_err("unknown activity must err");
    assert_eq!(err, RemoveWorkspaceFromActivityError::ActivityNotFound);
}

#[test]
fn remove_workspace_from_activity_workspace_not_found() {
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();

    let err = layout
        .remove_workspace_from_activity(
            Some(WorkspaceReference::Id(u64::MAX)),
            &ActivityReferenceArg::Id(alpha.get()),
        )
        .expect_err("unknown workspace must err");
    assert_eq!(err, RemoveWorkspaceFromActivityError::WorkspaceNotFound);
}

#[test]
fn remove_workspace_from_activity_snaps_animation() {
    // `WorkspaceSwitch::Animation` in flight on the active monitor. Removing
    // from the ACTIVE activity must snap the animation to None before
    // patching views; the mutator does not preserve the animation.
    let ops = [
        Op::AddOutput(1),
        Op::AddNamedWorkspace {
            ws_name: 1,
            output_name: None,
            layout_config: None,
        },
    ];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let mon_out = layout.monitors[0].output_id();

    // The named workspace is at pos 1 (trailing empty at pos 0 under this
    // setup, or the other way around). Pick it as the Remove target so
    // the post-remove view still carries its trailing empty and satisfies
    // the monitor-level bookend invariants at monitor.rs:1735.
    let target_ws_id = {
        let ids = layout.active_view(&mon_out).ids().to_vec();
        *ids.iter()
            .find(|id| {
                layout
                    .workspaces
                    .get(id)
                    .expect("view id in pool")
                    .name()
                    .is_some()
            })
            .expect("view must contain the named workspace")
    };

    // Share the named workspace with fresh beta BEFORE arming the
    // animation, so Remove is not LastActivity. Beta's dormant view on
    // mon_out covers target_ws_id for pool-keys equality after the
    // mutation drops it from alpha's view (though in this test, the
    // view.len() > 1 branch patches via remove_at instead of drop-entry).
    let beta = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");
    layout
        .workspaces
        .get_mut(&target_ws_id)
        .expect("live")
        .activities = [alpha, beta].into_iter().collect();
    // The override replaces beta's materialized fresh-empty view; include a freshly-minted
    // trailing-empty bookend so the per-view bookend invariant holds on beta's dormant view.
    let beta_bottom = test_mint_empty_for(&mut layout, 0, beta);
    test_override_activity_view(
        &mut layout,
        beta,
        mon_out.clone(),
        WorkspaceView::new(vec![target_ws_id, beta_bottom], 0),
    );
    layout.verify_invariants();

    // Arm a workspace-switch animation by switching to a different position.
    let target_pos = if layout.active_view(&mon_out).active_position() == 0 {
        1
    } else {
        0
    };
    layout.switch_workspace(target_pos);
    assert!(
        matches!(
            layout.monitors[0].workspace_switch,
            Some(super::monitor::WorkspaceSwitch::Animation(_)),
        ),
        "switch_workspace must arm an Animation",
    );

    layout
        .remove_workspace_from_activity(
            Some(WorkspaceReference::Id(target_ws_id.get())),
            &ActivityReferenceArg::Id(alpha.get()),
        )
        .expect("remove must succeed");

    // Snap pin: workspace_switch is None on every monitor after the call.
    for mon in &layout.monitors {
        assert!(
            mon.workspace_switch.is_none(),
            "Remove from active activity must snap in-flight animation",
        );
    }

    layout.verify_invariants();
}

#[test]
fn remove_workspace_from_activity_hard_blocked_by_gesture() {
    // Predicate-level pin: when a gesture is in flight on any monitor, the
    // caller-side gate `is_workspace_activity_assignment_blocked_by_gesture`
    // returns `Some(WorkspaceSwitchGesture)`. The dispatch-layer Err is
    // exercised indirectly via this predicate and the `ActivitySwitchBlock`
    // display pin.
    let ops = [Op::AddOutput(1), Op::AddOutput(2)];
    let mut layout = check_ops(ops);

    assert!(layout
        .is_workspace_activity_assignment_blocked_by_gesture()
        .is_none());

    let output2 = layout
        .outputs()
        .find(|o| o.name() == "output2")
        .cloned()
        .expect("output2 live");
    layout.workspace_switch_gesture_begin(&output2, true);

    assert_eq!(
        layout.is_workspace_activity_assignment_blocked_by_gesture(),
        Some(super::ActivitySwitchBlock::WorkspaceSwitchGesture),
    );

    // Animations alone must NOT trip the gesture predicate. Clean up the
    // gesture first to isolate the animation-only case.
    layout.workspace_switch_gesture_end(Some(true));
    assert!(layout
        .is_workspace_activity_assignment_blocked_by_gesture()
        .is_none());
}

// --- SetWorkspaceActivities ------------------------------------------------

#[test]
fn set_workspace_activities_diff_removes_from_dormant_view() {
    // Workspace initially tagged {alpha, beta}. Call Set(ws, [alpha]).
    // Symmetric diff yields to_remove = {beta} (dormant), to_add = {}.
    // Beta's single-entry dormant view must be dropped; alpha untouched.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let mon_out = layout.monitors[0].output_id();

    let beta = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");
    let target_ws_id = layout
        .active_view(&mon_out)
        .ids()
        .first()
        .copied()
        .expect("alpha view has ws");

    layout
        .workspaces
        .get_mut(&target_ws_id)
        .expect("live")
        .activities = [alpha, beta].into_iter().collect();
    let beta_bottom = test_mint_empty_for(&mut layout, 0, beta);
    test_override_activity_view(
        &mut layout,
        beta,
        mon_out.clone(),
        WorkspaceView::new(vec![target_ws_id, beta_bottom], 0),
    );

    let (ws_id, new_set, active_affected) = layout
        .set_workspace_activities(
            Some(WorkspaceReference::Id(target_ws_id.get())),
            &[ActivityReferenceArg::Id(alpha.get())],
        )
        .expect("set must succeed");
    assert_eq!(ws_id, target_ws_id);
    assert_eq!(new_set, HashSet::from([alpha]));
    assert!(
        !active_affected,
        "alpha stays on ws, beta drops — active activity (alpha) is NOT in symmetric diff",
    );

    assert_eq!(
        layout
            .workspaces
            .get(&target_ws_id)
            .expect("live")
            .activities(),
        &HashSet::from([alpha]),
    );
    // Beta's view loses target_ws_id; the trailing-empty bookend stays. Per the per-activity
    // bookend invariant the view is never absent on a connected monitor.
    let beta_view = layout
        .activities
        .get(beta)
        .expect("live")
        .views()
        .get(&mon_out)
        .expect("beta's view persists (carries the bookend)");
    assert!(!beta_view.ids().contains(&target_ws_id));
    assert_eq!(beta_view.len(), 1, "only trailing-empty bookend remains");

    layout.verify_invariants();
}

#[test]
fn set_workspace_activities_diff_adds_to_target_view() {
    // Workspace initially tagged {alpha}. Set(ws, [alpha, beta]) where
    // beta has a pre-existing dormant view on the output. to_add = {beta}.
    // Beta's view must append the ws id; alpha's view unchanged.
    let ops = [
        Op::AddOutput(1),
        Op::AddNamedWorkspace {
            ws_name: 1,
            output_name: None,
            layout_config: None,
        },
    ];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let mon_out = layout.monitors[0].output_id();

    let beta = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");
    // Seed beta's view with one id from alpha's view so Set appends a
    // distinct id.
    let seed_ws_id = layout
        .active_view(&mon_out)
        .ids()
        .first()
        .copied()
        .expect("alpha view has ws");
    test_override_activity_view(
        &mut layout,
        beta,
        mon_out.clone(),
        WorkspaceView::new(vec![seed_ws_id], 0),
    );

    // Pick a workspace from alpha's view NOT in beta's view — distinct.
    let target_ws_id = {
        let ids = layout.active_view(&mon_out).ids().to_vec();
        *ids.iter()
            .find(|id| **id != seed_ws_id)
            .expect("alpha view has >= 2 entries")
    };

    let (_, new_set, active_affected) = layout
        .set_workspace_activities(
            Some(WorkspaceReference::Id(target_ws_id.get())),
            &[
                ActivityReferenceArg::Id(alpha.get()),
                ActivityReferenceArg::Id(beta.get()),
            ],
        )
        .expect("set must succeed");
    assert_eq!(new_set, HashSet::from([alpha, beta]));
    assert!(
        !active_affected,
        "alpha stays, beta is added — active activity (alpha) not in symmetric diff",
    );

    assert_eq!(
        layout
            .workspaces
            .get(&target_ws_id)
            .expect("live")
            .activities(),
        &HashSet::from([alpha, beta]),
    );
    assert_eq!(
        layout
            .activities
            .get(beta)
            .expect("live")
            .views()
            .get(&mon_out)
            .expect("beta has view")
            .ids(),
        &[seed_ws_id, target_ws_id],
    );

    layout.verify_invariants();
}

#[test]
fn set_workspace_activities_drops_single_entry_view_for_removed_active_activity() {
    // Set removes the active activity from a workspace whose active-activity
    // view has exactly that workspace as its single entry on a connected
    // monitor. `ensure_all_activity_views` must reinstate the view.
    //
    // Fixture shape mirrors `remove_workspace_from_activity_active_activity_recreates_view`:
    // ws is tagged {alpha, beta}, appears in alpha's active view (single
    // entry) AND in beta's dormant view so pool-keys equality holds.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let mon_out = layout.monitors[0].output_id();
    let output = layout.monitors[0].output.clone();

    let beta = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");

    let ws = Workspace::new(
        &output,
        [alpha, beta].into_iter().collect(),
        layout.clock.clone(),
        layout.options.clone(),
    );
    let ws_id = ws.id();
    assert!(layout.workspaces.insert(ws_id, ws).is_none());

    let original_ids: Vec<_> = layout.active_view(&mon_out).ids().to_vec();
    layout
        .activities
        .active_mut()
        .views_mut()
        .insert(mon_out.clone(), WorkspaceView::new(vec![ws_id], 0));
    for id in &original_ids {
        let _ = layout.workspaces.remove(id);
    }
    test_override_activity_view(
        &mut layout,
        beta,
        mon_out.clone(),
        WorkspaceView::new(vec![ws_id], 0),
    );
    layout.verify_invariants();

    // Set(ws, [beta]) → to_remove = {alpha}, drops alpha's single-entry view.
    let (_, _, active_affected) = layout
        .set_workspace_activities(
            Some(WorkspaceReference::Id(ws_id.get())),
            &[ActivityReferenceArg::Id(beta.get())],
        )
        .expect("set must succeed");
    assert!(
        active_affected,
        "alpha (active) is in to_remove — active_affected must be true",
    );

    let reinstated = layout
        .activities
        .active()
        .views()
        .get(&mon_out)
        .expect("ensure_all_activity_views must reinstate alpha's view on mon_out");
    assert!(!reinstated.ids().contains(&ws_id));
    assert!(!reinstated.ids().is_empty(), "reinstated view non-empty");

    assert_eq!(
        layout.workspaces.get(&ws_id).expect("live").activities(),
        &HashSet::from([beta]),
    );

    layout.verify_invariants();
}

#[test]
fn set_workspace_activities_empty_list_errors_without_mutation() {
    // Passing an empty list must Err(EmptyActivityList) with zero mutation —
    // guard-before-mutate. Pre/post snapshot of the workspace's activities
    // and alpha's view is byte-identical after the call.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let mon_out = layout.monitors[0].output_id();

    let target_ws_id = layout
        .active_view(&mon_out)
        .ids()
        .first()
        .copied()
        .expect("alpha view has ws");

    let acts_before = layout
        .workspaces
        .get(&target_ws_id)
        .expect("live")
        .activities()
        .clone();
    assert_eq!(acts_before, HashSet::from([alpha]));
    let view_ids_before = layout.active_view(&mon_out).ids().to_vec();

    let err = layout
        .set_workspace_activities(Some(WorkspaceReference::Id(target_ws_id.get())), &[])
        .expect_err("empty list must err");
    assert_eq!(err, SetWorkspaceActivitiesError::EmptyActivityList);

    assert_eq!(
        layout
            .workspaces
            .get(&target_ws_id)
            .expect("live")
            .activities(),
        &acts_before,
    );
    assert_eq!(layout.active_view(&mon_out).ids(), &view_ids_before[..]);
}

#[test]
fn set_workspace_activities_activity_not_found_errors_without_mutation() {
    // List contains one unresolvable id. Must Err(ActivityNotFound) with
    // zero mutation.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let mon_out = layout.monitors[0].output_id();
    let target_ws_id = layout
        .active_view(&mon_out)
        .ids()
        .first()
        .copied()
        .expect("alpha view has ws");
    let acts_before = layout
        .workspaces
        .get(&target_ws_id)
        .expect("live")
        .activities()
        .clone();

    let err = layout
        .set_workspace_activities(
            Some(WorkspaceReference::Id(target_ws_id.get())),
            &[ActivityReferenceArg::Id(u64::MAX)],
        )
        .expect_err("unknown activity must err");
    assert_eq!(err, SetWorkspaceActivitiesError::ActivityNotFound);

    assert_eq!(
        layout
            .workspaces
            .get(&target_ws_id)
            .expect("live")
            .activities(),
        &acts_before,
    );
}

#[test]
fn set_workspace_activities_activity_not_found_precedence_over_empty() {
    // A single-element list `[unresolvable_id]` must yield ActivityNotFound,
    // not EmptyActivityList — precedence pin. Resolution happens before the
    // empty-list gate.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let mon_out = layout.monitors[0].output_id();
    let target_ws_id = layout
        .active_view(&mon_out)
        .ids()
        .first()
        .copied()
        .expect("alpha view has ws");

    let err = layout
        .set_workspace_activities(
            Some(WorkspaceReference::Id(target_ws_id.get())),
            &[ActivityReferenceArg::Id(u64::MAX)],
        )
        .expect_err("unknown activity must err");
    assert_eq!(
        err,
        SetWorkspaceActivitiesError::ActivityNotFound,
        "ActivityNotFound must precede EmptyActivityList",
    );
}

#[test]
fn set_workspace_activities_workspace_not_found_returns_err() {
    // Layout-level: surfaces WorkspaceNotFound. The dispatch-layer silent
    // no-op is an input/mod.rs concern; here we pin the Layout surface so
    // `DoActionError` match exhaustiveness stays load-bearing.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();

    let err = layout
        .set_workspace_activities(
            Some(WorkspaceReference::Id(u64::MAX)),
            &[ActivityReferenceArg::Id(alpha.get())],
        )
        .expect_err("unknown workspace must err at the Layout surface");
    assert_eq!(err, SetWorkspaceActivitiesError::WorkspaceNotFound);
}

#[test]
fn set_workspace_activities_no_op_when_set_equals_current() {
    // Call Set with new == old. Must return Ok without touching any state:
    // no animation clear, no view patching, and the result's
    // active_activity_affected flag is false.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let mon_out = layout.monitors[0].output_id();

    let target_ws_id = layout
        .active_view(&mon_out)
        .ids()
        .first()
        .copied()
        .expect("alpha view has ws");

    let view_ids_before = layout.active_view(&mon_out).ids().to_vec();

    let (_, new_set, active_affected) = layout
        .set_workspace_activities(
            Some(WorkspaceReference::Id(target_ws_id.get())),
            &[ActivityReferenceArg::Id(alpha.get())],
        )
        .expect("no-op set must succeed");
    assert_eq!(new_set, HashSet::from([alpha]));
    assert!(
        !active_affected,
        "identity set must not flag active_affected",
    );
    assert_eq!(
        layout
            .workspaces
            .get(&target_ws_id)
            .expect("live")
            .activities(),
        &HashSet::from([alpha]),
    );
    assert_eq!(layout.active_view(&mon_out).ids(), &view_ids_before[..]);

    layout.verify_invariants();
}

#[test]
fn set_workspace_activities_snaps_animation_only_when_active_affected() {
    // Arm an in-flight workspace-switch Animation. A Set where active is
    // NOT in the symmetric diff must leave the animation intact; a Set that
    // DOES touch the active activity must snap the animation on every
    // monitor ( snap+proceed).
    let ops = [
        Op::AddOutput(1),
        Op::AddNamedWorkspace {
            ws_name: 1,
            output_name: None,
            layout_config: None,
        },
    ];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let mon_out = layout.monitors[0].output_id();

    let beta = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");

    // Pick the named workspace (its remove_at patch preserves bookend).
    let target_ws_id = {
        let ids = layout.active_view(&mon_out).ids().to_vec();
        *ids.iter()
            .find(|id| {
                layout
                    .workspaces
                    .get(id)
                    .expect("view id in pool")
                    .name()
                    .is_some()
            })
            .expect("named workspace must be present")
    };

    // Place into beta's dormant view for pool-keys equality after the cross-activity changes
    // below. ws.activities stays {alpha} initially. Append a freshly-minted trailing-empty
    // bookend so the per-view bookend invariant holds on beta's dormant view.
    let beta_bottom = test_mint_empty_for(&mut layout, 0, beta);
    test_override_activity_view(
        &mut layout,
        beta,
        mon_out.clone(),
        WorkspaceView::new(vec![target_ws_id, beta_bottom], 0),
    );
    layout.verify_invariants();

    // Arm an animation.
    let target_pos = if layout.active_view(&mon_out).active_position() == 0 {
        1
    } else {
        0
    };
    layout.switch_workspace(target_pos);
    assert!(
        matches!(
            layout.monitors[0].workspace_switch,
            Some(super::monitor::WorkspaceSwitch::Animation(_)),
        ),
        "switch_workspace must arm an Animation",
    );

    // Path A: Set(ws, [alpha]) — new == old, identity case, no diff.
    // Animation must survive (no mutation, no snap).
    layout
        .set_workspace_activities(
            Some(WorkspaceReference::Id(target_ws_id.get())),
            &[ActivityReferenceArg::Id(alpha.get())],
        )
        .expect("identity set must succeed");
    assert!(
        matches!(
            layout.monitors[0].workspace_switch,
            Some(super::monitor::WorkspaceSwitch::Animation(_)),
        ),
        "no-op Set must not snap in-flight animation",
    );

    // Path B: Set(ws, [beta]) — to_remove = {alpha}, active_affected = true.
    // Animation must be snapped.
    layout
        .set_workspace_activities(
            Some(WorkspaceReference::Id(target_ws_id.get())),
            &[ActivityReferenceArg::Id(beta.get())],
        )
        .expect("set with active in diff must succeed");
    for mon in &layout.monitors {
        assert!(
            mon.workspace_switch.is_none(),
            "Set touching active activity must snap animation on every monitor",
        );
    }

    layout.verify_invariants();
}

#[test]
fn set_workspace_activities_active_affected_via_to_add_branch() {
    // Regression pin for the `to_add` branch of the active_affected predicate:
    //   active_activity_affected = to_remove.contains(&active_id)
    //                           || to_add.contains(&active_id)
    //
    // Dropping the `to_add` clause would not be caught by the other tests
    // (which all exercise active_affected=true via to_remove). Fixture:
    // ws starts in {beta} only; active=alpha; Set(ws, [alpha, beta]) →
    // to_add={alpha}, to_remove={}, active_affected=true.
    //
    // The fixture is a clean slate (single ws owns the active view) to avoid
    // bookend-ordering complications when to_add appends to alpha's view.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let mon_out = layout.monitors[0].output_id();
    let output = layout.monitors[0].output.clone();

    let beta = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");

    // Build a workspace that belongs ONLY to beta (not alpha).
    let ws = Workspace::new(
        &output,
        [beta].into_iter().collect(),
        layout.clock.clone(),
        layout.options.clone(),
    );
    let ws_id = ws.id();
    assert!(layout.workspaces.insert(ws_id, ws).is_none());

    // Clear alpha's active view to a single-entry view holding ws so the pool
    // satisfies the active-activity invariant (every monitor's view id is in the
    // pool). ws.activities remains {beta} — alpha is NOT yet a member.
    let old_ids: Vec<_> = layout.active_view(&mon_out).ids().to_vec();
    layout
        .activities
        .active_mut()
        .views_mut()
        .insert(mon_out.clone(), WorkspaceView::new(vec![ws_id], 0));
    for id in &old_ids {
        layout.workspaces.remove(id);
    }
    // Beta's dormant view also holds ws so pool-keys equality holds.
    test_override_activity_view(
        &mut layout,
        beta,
        mon_out.clone(),
        WorkspaceView::new(vec![ws_id], 0),
    );
    layout.verify_invariants();

    // Sanity: alpha is NOT in ws.activities yet (to_add branch precondition).
    assert!(
        !layout
            .workspaces
            .get(&ws_id)
            .expect("live")
            .activities()
            .contains(&alpha),
        "precondition: ws must not be in alpha's activities before the set call",
    );

    // Set(ws, [alpha, beta]): to_add = {alpha}, to_remove = {}, active_affected = true.
    let (_, new_set, active_affected) = layout
        .set_workspace_activities(
            Some(WorkspaceReference::Id(ws_id.get())),
            &[
                ActivityReferenceArg::Id(alpha.get()),
                ActivityReferenceArg::Id(beta.get()),
            ],
        )
        .expect("set must succeed");

    assert!(
        active_affected,
        "alpha (active) is in to_add — active_affected must be true via to_add branch",
    );
    assert_eq!(
        new_set,
        HashSet::from([alpha, beta]),
        "new_set must be {{alpha, beta}}",
    );

    layout.verify_invariants();
}

#[test]
fn set_workspace_activities_blocked_by_gesture_at_dispatch() {
    // Predicate-level pin: the dispatch arm's weaker gate fires on a
    // workspace-switch gesture in flight on any monitor, matching Remove's
    // recipe. Animations alone do NOT trip the gate.
    let ops = [Op::AddOutput(1), Op::AddOutput(2)];
    let mut layout = check_ops(ops);

    assert!(layout
        .is_workspace_activity_assignment_blocked_by_gesture()
        .is_none());

    let output2 = layout
        .outputs()
        .find(|o| o.name() == "output2")
        .cloned()
        .expect("output2 live");
    layout.workspace_switch_gesture_begin(&output2, true);

    assert_eq!(
        layout.is_workspace_activity_assignment_blocked_by_gesture(),
        Some(super::ActivitySwitchBlock::WorkspaceSwitchGesture),
        "Set shares Remove's weaker gesture gate — pinned by the predicate",
    );

    layout.workspace_switch_gesture_end(Some(true));
    assert!(layout
        .is_workspace_activity_assignment_blocked_by_gesture()
        .is_none());
}

// --- MoveWorkspaceToActivity -----------------------------------------------

#[test]
fn move_workspace_to_activity_atomic_add_then_remove() {
    // Workspace in {active(alpha)}; move to beta. Final state:
    // ws.activities = {beta}, alpha's view lost the id, beta's view
    // (dormant) gained it if a view existed; if beta had no view on the
    // output, Add fabricates none — final beta view may be absent (lazy).
    //
    // Fixture gives beta a pre-existing dormant view so we can assert
    // the append side of Add-then-Remove.
    let ops = [
        Op::AddOutput(1),
        Op::AddNamedWorkspace {
            ws_name: 1,
            output_name: None,
            layout_config: None,
        },
    ];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let mon_out = layout.monitors[0].output_id();

    let beta = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");
    // Seed beta's view with an id from alpha's view (distinct from the named one we'll move,
    // so Move appends to a non-empty view). Append a freshly-minted trailing-empty bookend so
    // the per-view bookend invariant holds on beta's dormant view.
    let seed_ws_id = layout
        .active_view(&mon_out)
        .ids()
        .first()
        .copied()
        .expect("alpha view has ws");
    let beta_bottom = test_mint_empty_for(&mut layout, 0, beta);
    test_override_activity_view(
        &mut layout,
        beta,
        mon_out.clone(),
        WorkspaceView::new(vec![seed_ws_id, beta_bottom], 0),
    );

    // Pick the named workspace (preserves bookend discipline under remove_at).
    let target_ws_id = {
        let ids = layout.active_view(&mon_out).ids().to_vec();
        *ids.iter()
            .find(|id| {
                layout
                    .workspaces
                    .get(id)
                    .expect("view id in pool")
                    .name()
                    .is_some()
            })
            .expect("named workspace present")
    };
    // Precondition: workspace belongs exclusively to alpha.
    assert_eq!(
        layout
            .workspaces
            .get(&target_ws_id)
            .expect("live")
            .activities(),
        &HashSet::from([alpha]),
    );

    let (ws_id, target_id, source_id) = layout
        .move_workspace_to_activity(
            Some(WorkspaceReference::Id(target_ws_id.get())),
            &ActivityReferenceArg::Id(beta.get()),
        )
        .expect("move must succeed");
    assert_eq!(ws_id, target_ws_id);
    assert_eq!(target_id, beta);
    assert_eq!(source_id, alpha);

    // Final set: {beta} (alpha pruned, beta added).
    assert_eq!(
        layout
            .workspaces
            .get(&target_ws_id)
            .expect("live")
            .activities(),
        &HashSet::from([beta]),
    );
    // Beta's view contains both seed_ws_id and target_ws_id.
    let beta_view_ids: Vec<_> = layout
        .activities
        .get(beta)
        .expect("live")
        .views()
        .get(&mon_out)
        .expect("beta view")
        .ids()
        .to_vec();
    assert!(beta_view_ids.contains(&seed_ws_id));
    assert!(beta_view_ids.contains(&target_ws_id));

    // Alpha's active view lost target_ws_id.
    assert!(!layout.active_view(&mon_out).ids().contains(&target_ws_id));

    layout.verify_invariants();
}

#[test]
fn move_workspace_to_activity_preserves_other_memberships() {
    // Workspace in {active(alpha), X, Y}; move to target. Final state:
    // {X, Y, target} — workspace leaves active but stays in X, Y.
    // "multi-activity semantics" pin.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let mon_out = layout.monitors[0].output_id();

    let x = layout.create_activity("X".to_owned()).expect("create x");
    let y = layout.create_activity("Y".to_owned()).expect("create y");
    let target = layout
        .create_activity("Target".to_owned())
        .expect("create target");

    // Seed a workspace tagged {alpha, x, y}. Use an existing one from
    // alpha's view. Place it in X's and Y's dormant views too — otherwise
    // the post-Remove pool-keys equality invariant fails (X/Y/target have
    // no view on mon_out, so after Remove prunes alpha the ws has no
    // reachable view → orphan).
    let target_ws_id = layout
        .active_view(&mon_out)
        .ids()
        .first()
        .copied()
        .expect("alpha view has ws");
    layout
        .workspaces
        .get_mut(&target_ws_id)
        .expect("live")
        .activities = [alpha, x, y].into_iter().collect();
    test_override_activity_view(
        &mut layout,
        x,
        mon_out.clone(),
        WorkspaceView::new(vec![target_ws_id], 0),
    );
    test_override_activity_view(
        &mut layout,
        y,
        mon_out.clone(),
        WorkspaceView::new(vec![target_ws_id], 0),
    );

    let (_, target_id, source_id) = layout
        .move_workspace_to_activity(
            Some(WorkspaceReference::Id(target_ws_id.get())),
            &ActivityReferenceArg::Id(target.get()),
        )
        .expect("move must succeed");
    assert_eq!(target_id, target);
    assert_eq!(source_id, alpha);

    assert_eq!(
        layout
            .workspaces
            .get(&target_ws_id)
            .expect("live")
            .activities(),
        &HashSet::from([x, y, target]),
        "move must leave alpha but keep x, y and add target",
    );

    layout.verify_invariants();
}

#[test]
fn move_workspace_to_activity_workspace_already_in_target() {
    // Workspace in {active(alpha), target}; move to target. Final state:
    // {target} — source removed, target retained.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let mon_out = layout.monitors[0].output_id();

    let target = layout
        .create_activity("Target".to_owned())
        .expect("create target");

    let target_ws_id = layout
        .active_view(&mon_out)
        .ids()
        .first()
        .copied()
        .expect("alpha view has ws");
    layout
        .workspaces
        .get_mut(&target_ws_id)
        .expect("live")
        .activities = [alpha, target].into_iter().collect();
    // Give target a dormant view containing the id so Add is a delegate
    // no-op.
    test_override_activity_view(
        &mut layout,
        target,
        mon_out.clone(),
        WorkspaceView::new(vec![target_ws_id], 0),
    );

    layout
        .move_workspace_to_activity(
            Some(WorkspaceReference::Id(target_ws_id.get())),
            &ActivityReferenceArg::Id(target.get()),
        )
        .expect("move must succeed");

    assert_eq!(
        layout
            .workspaces
            .get(&target_ws_id)
            .expect("live")
            .activities(),
        &HashSet::from([target]),
    );

    layout.verify_invariants();
}

#[test]
fn move_workspace_to_activity_target_equals_source_no_op() {
    // Call with target == active. Must return Ok without mutating any
    // state — the no-op branch subsumes the "No-op if workspace
    // already exclusively in target" row (source == target implies the
    // workspace stays put).
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let mon_out = layout.monitors[0].output_id();

    let target_ws_id = layout
        .active_view(&mon_out)
        .ids()
        .first()
        .copied()
        .expect("alpha view has ws");

    let acts_before = layout
        .workspaces
        .get(&target_ws_id)
        .expect("live")
        .activities()
        .clone();
    let view_ids_before = layout.active_view(&mon_out).ids().to_vec();

    let (_, target_id, source_id) = layout
        .move_workspace_to_activity(
            Some(WorkspaceReference::Id(target_ws_id.get())),
            &ActivityReferenceArg::Id(alpha.get()),
        )
        .expect("target == source move must succeed");
    assert_eq!(target_id, alpha);
    assert_eq!(source_id, alpha);

    assert_eq!(
        layout
            .workspaces
            .get(&target_ws_id)
            .expect("live")
            .activities(),
        &acts_before,
    );
    assert_eq!(layout.active_view(&mon_out).ids(), &view_ids_before[..]);

    layout.verify_invariants();
}

#[test]
fn move_workspace_to_activity_not_in_active_errors() {
    // Workspace tagged {x, y} (active=alpha not present). Move must
    // Err(WorkspaceNotInActiveActivity) without mutating any state.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let mon_out = layout.monitors[0].output_id();
    let output = layout.monitors[0].output.clone();

    let x = layout.create_activity("X".to_owned()).expect("create x");
    let y = layout.create_activity("Y".to_owned()).expect("create y");
    let target = layout
        .create_activity("Target".to_owned())
        .expect("create target");

    // Allocate a workspace tagged {x, y} only. Place it in x's view so
    // pool-keys equality holds.
    let ws = Workspace::new(
        &output,
        [x, y].into_iter().collect(),
        layout.clock.clone(),
        layout.options.clone(),
    );
    let ws_id = ws.id();
    assert!(layout.workspaces.insert(ws_id, ws).is_none());
    test_override_activity_view(
        &mut layout,
        x,
        mon_out.clone(),
        WorkspaceView::new(vec![ws_id], 0),
    );
    test_override_activity_view(
        &mut layout,
        y,
        mon_out.clone(),
        WorkspaceView::new(vec![ws_id], 0),
    );
    // alpha remains active. Verify precondition.
    assert_eq!(layout.active_activity_id(), alpha);
    assert!(!layout
        .workspaces
        .get(&ws_id)
        .expect("live")
        .activities()
        .contains(&alpha));

    let acts_before = layout
        .workspaces
        .get(&ws_id)
        .expect("live")
        .activities()
        .clone();

    let err = layout
        .move_workspace_to_activity(
            Some(WorkspaceReference::Id(ws_id.get())),
            &ActivityReferenceArg::Id(target.get()),
        )
        .expect_err("must err: workspace not in active");
    assert_eq!(
        err,
        MoveWorkspaceToActivityError::WorkspaceNotInActiveActivity
    );

    assert_eq!(
        layout.workspaces.get(&ws_id).expect("live").activities(),
        &acts_before,
    );
}

#[test]
fn move_workspace_to_activity_activity_not_found_errors() {
    // Unresolvable target. Must Err(ActivityNotFound) before the
    // workspace-resolution step runs — precedence pin.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let mon_out = layout.monitors[0].output_id();
    let target_ws_id = layout
        .active_view(&mon_out)
        .ids()
        .first()
        .copied()
        .expect("alpha view has ws");

    let err = layout
        .move_workspace_to_activity(
            Some(WorkspaceReference::Id(target_ws_id.get())),
            &ActivityReferenceArg::Id(u64::MAX),
        )
        .expect_err("unknown target activity must err");
    assert_eq!(err, MoveWorkspaceToActivityError::ActivityNotFound);
}

#[test]
fn move_workspace_to_activity_workspace_not_found_errors() {
    // Target resolves but workspace does not. Must Err(WorkspaceNotFound).
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let target = layout
        .create_activity("Target".to_owned())
        .expect("create target");

    let err = layout
        .move_workspace_to_activity(
            Some(WorkspaceReference::Id(u64::MAX)),
            &ActivityReferenceArg::Id(target.get()),
        )
        .expect_err("unknown workspace must err");
    assert_eq!(err, MoveWorkspaceToActivityError::WorkspaceNotFound);
}

#[test]
fn move_workspace_to_activity_focus_false_uses_weaker_gate() {
    // DnD armed → the full `is_activity_switch_hard_blocked` returns
    // Some(Dnd), but the weaker gesture-only gate returns None. The
    // `focus: false` dispatch path uses the weaker gate and therefore
    // does NOT block on DnD — the Layout call would succeed if run
    // through. A workspace-switch gesture, by contrast, IS caught by
    // the weaker gate.
    //
    // This test pins the predicate contrast at the Layout level — the
    // dispatch arm itself is exercised via the production `do_action`
    // flow in the fixture-level integration suite.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);

    // Precondition: no block.
    assert!(layout.is_activity_switch_hard_blocked().is_none());
    assert!(layout
        .is_workspace_activity_assignment_blocked_by_gesture()
        .is_none());

    // Arm DnD — hard-block via `is_activity_switch_hard_blocked`, but
    // the weaker gate is unaffected (matches the recipe from
    // `focus_window_hard_block_gate_fires_before_switch`).
    let output = layout
        .outputs()
        .find(|o| o.name() == "output1")
        .cloned()
        .expect("output1 exists");
    layout.dnd_update(output, Point::from((0., 0.)));
    assert_eq!(
        layout.is_activity_switch_hard_blocked(),
        Some(super::ActivitySwitchBlock::Dnd),
        "DnD must trip the full hard-block predicate",
    );
    assert!(
        layout
            .is_workspace_activity_assignment_blocked_by_gesture()
            .is_none(),
        "DnD must NOT trip the weaker gesture-only gate (Move focus:false \
         would proceed)",
    );

    layout.dnd_end();
}

#[test]
fn move_workspace_to_activity_focus_true_uses_stronger_gate() {
    // DnD armed. The `focus: true` dispatch path uses the full
    // `is_activity_switch_hard_blocked` gate because it chains into
    // `switch_activity`. This asymmetry is the critical review-stop
    // concern: collapsing to a single gate would leave the `focus: true`
    // branch unblocked during interactive_move / DnD, chaining into an
    // activity switch under an active user drag.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);

    let output = layout
        .outputs()
        .find(|o| o.name() == "output1")
        .cloned()
        .expect("output1 exists");
    layout.dnd_update(output, Point::from((0., 0.)));
    assert_eq!(
        layout.is_activity_switch_hard_blocked(),
        Some(super::ActivitySwitchBlock::Dnd),
        "Move with focus:true must consult this predicate — pinned here",
    );

    layout.dnd_end();
}

// --- SetWorkspaceSticky / UnsetWorkspaceSticky / ToggleWorkspaceSticky -----

#[test]
fn set_workspace_sticky_expands_activities_to_all_ids() {
    // A non-sticky workspace bound to {alpha}; alpha + beta live in the pool.
    // SetSticky must flip is_sticky and expand activities to {alpha, beta}.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let beta = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");

    let mon_out = layout.monitors[0].output_id();
    let target_ws_id = layout
        .active_view(&mon_out)
        .ids()
        .first()
        .copied()
        .expect("alpha view has at least one ws");

    // Sanity: not sticky, only in alpha.
    {
        let ws = layout
            .workspaces
            .get(&target_ws_id)
            .expect("target ws live");
        assert!(!ws.is_sticky());
        assert_eq!(ws.activities(), &HashSet::from([alpha]));
    }

    let (ws_id, active_affected) = layout
        .set_workspace_sticky(Some(WorkspaceReference::Id(target_ws_id.get())))
        .expect("set sticky must succeed");
    assert_eq!(ws_id, target_ws_id);
    // alpha was already in the set; only beta was added — alpha is the
    // active activity, but it's NOT in the symmetric diff, so
    // active_affected is false.
    assert!(
        !active_affected,
        "to_add = {{beta}} only; active activity (alpha) is not in the diff"
    );

    let ws = layout
        .workspaces
        .get(&target_ws_id)
        .expect("target ws live");
    assert!(ws.is_sticky(), "is_sticky must be true after SetSticky");
    assert_eq!(
        ws.activities(),
        &HashSet::from([alpha, beta]),
        "activities must equal the full live id set",
    );

    layout.verify_invariants();
}

#[test]
fn set_workspace_sticky_no_op_when_already_sticky_with_full_set() {
    // Pre-state: workspace already sticky with activities = {alpha, beta}
    // (the full live set). SetSticky must early-exit without churning state:
    // active_affected = false, animation untouched.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let beta = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");

    let mon_out = layout.monitors[0].output_id();
    let target_ws_id = layout
        .active_view(&mon_out)
        .ids()
        .first()
        .copied()
        .expect("alpha view has ws");

    // Hand-construct the "already sticky + full set" state.
    {
        let ws = layout
            .workspaces
            .get_mut(&target_ws_id)
            .expect("target ws live");
        ws.is_sticky = true;
        ws.activities = HashSet::from([alpha, beta]);
    }
    layout.verify_invariants();

    let (ws_id, active_affected) = layout
        .set_workspace_sticky(Some(WorkspaceReference::Id(target_ws_id.get())))
        .expect("no-op set must succeed");
    assert_eq!(ws_id, target_ws_id);
    assert!(!active_affected, "no-op set must not flag active_affected",);

    let ws = layout
        .workspaces
        .get(&target_ws_id)
        .expect("target ws live");
    assert!(ws.is_sticky());
    assert_eq!(ws.activities(), &HashSet::from([alpha, beta]));

    layout.verify_invariants();
}

#[test]
fn set_workspace_sticky_no_op_when_workspace_not_found() {
    // Layout-level: surfaces WorkspaceNotFound. The dispatch-layer silent
    // no-op is an input/mod.rs concern; this pin keeps the Layout surface
    // exhaustive for the dispatch arm's `match e`.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);

    let err = layout
        .set_workspace_sticky(Some(WorkspaceReference::Id(u64::MAX)))
        .expect_err("unknown workspace must err at the Layout surface");
    assert_eq!(err, SetWorkspaceStickyError::WorkspaceNotFound);
}

#[test]
fn set_workspace_sticky_re_expands_when_activities_was_narrowed_to_subset() {
    // Load-bearing pin for the "no strict equality invariant" decision in
    // workspace.rs `is_sticky()`'s rustdoc — calling SetSticky on a
    // workspace whose `is_sticky == true` AND `activities ⊊ all_live_ids`
    // must re-expand the activities set to the full live id set (rather
    // than no-op out on the `is_sticky` flag alone).
    //
    // The / is_sticky() contract explicitly permits the inconsistent
    // state `is_sticky == true ∧ activities ⊊ all_ids` to arise via runtime
    // narrowing (e.g. `SetWorkspaceActivities` / `RemoveWorkspaceFromActivity`
    // on a sticky workspace). This test hand-mutates the workspace into that
    // state directly rather than going through the narrowing API. Going
    // through the API would surface a pre-existing limitation in
    // `set_workspace_activities` step 9 / `add_workspace_to_activity` step
    // (mod.rs view.insert(pos = view.len(), ws_id) appends after the
    // monitor's unnamed-bookend, breaking monitor.rs "last must be unnamed"
    // when re-adding a named workspace to the *active* activity's
    // view; lazy view rebuild covers dormant activities only). That
    // limitation is orthogonal to the sticky-action triplet and is not
    // addressed here — see the implementer's notes in the landing commit body
    // for a follow-up candidate.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let beta = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");

    let mon_out = layout.monitors[0].output_id();
    let target_ws_id = layout
        .active_view(&mon_out)
        .ids()
        .first()
        .copied()
        .expect("alpha view has the seed unnamed workspace");

    // Hand-seed the inconsistent state: is_sticky = true, activities =
    // {alpha} (a strict subset of {alpha, beta}). This is what runtime
    // narrowing on a sticky workspace would leave behind.
    {
        let ws = layout
            .workspaces
            .get_mut(&target_ws_id)
            .expect("target ws live");
        ws.is_sticky = true;
        ws.activities = HashSet::from([alpha]);
    }
    layout.verify_invariants();

    // Sanity: pre-call invariants hold despite the inconsistent state.
    {
        let ws = layout.workspaces.get(&target_ws_id).expect("live");
        assert!(ws.is_sticky());
        assert_eq!(ws.activities(), &HashSet::from([alpha]));
    }

    // SetSticky: must re-expand to {alpha, beta}. The diff is purely
    // additive on beta (a dormant activity without a view on `mon_out`), so
    // alpha's view is untouched and the unnamed-last invariant is
    // preserved.
    let (ws_id, active_affected) = layout
        .set_workspace_sticky(Some(WorkspaceReference::Id(target_ws_id.get())))
        .expect("SetSticky must re-expand");
    assert_eq!(ws_id, target_ws_id);
    // alpha was already in the set; only beta is in to_add. alpha (the
    // active activity) is NOT in the symmetric diff.
    assert!(
        !active_affected,
        "to_add = {{beta}}; active activity (alpha) is not in the diff",
    );

    let ws = layout.workspaces.get(&target_ws_id).expect("live");
    assert!(ws.is_sticky(), "is_sticky must remain true");
    assert_eq!(
        ws.activities(),
        &HashSet::from([alpha, beta]),
        "SetSticky must re-expand to the full live id set even when \
         is_sticky was already true and activities was a strict subset",
    );

    layout.verify_invariants();
}

#[test]
fn set_workspace_sticky_reports_active_affected_when_active_enters_set() {
    // Pin the active_affected=true branch: when the active activity id is
    // NOT already in the workspace's activities set, set_workspace_sticky
    // adds it and returns active_affected=true.
    //
    // This is the cursor-warp / redraw trigger; a regression that always
    // returns false would silence the redraw and leave the workspace
    // invisible in the active view.
    //
    // We seed ws.activities = {beta} (beta is inactive, alpha is active)
    // by hand. Going through RemoveWorkspaceFromActivity would surface
    // the pre-existing "last must be unnamed" limitation (see the
    // re_expands test above for the full explanation).
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let beta = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");

    let mon_out = layout.monitors[0].output_id();
    let target_ws_id = layout
        .active_view(&mon_out)
        .ids()
        .first()
        .copied()
        .expect("alpha view has seed ws");

    // Seed: activities = {beta} only, so active (alpha) is NOT in the set.
    {
        let ws = layout.workspaces.get_mut(&target_ws_id).expect("live");
        ws.activities = HashSet::from([beta]);
    }

    let (ws_id, active_affected) = layout
        .set_workspace_sticky(Some(WorkspaceReference::Id(target_ws_id.get())))
        .expect("set must succeed");
    assert_eq!(ws_id, target_ws_id);
    assert!(
        active_affected,
        "to_add includes alpha (active activity); active_affected must be true",
    );

    let ws = layout.workspaces.get(&target_ws_id).expect("live");
    assert!(ws.is_sticky());
    assert_eq!(ws.activities(), &HashSet::from([alpha, beta]));
}

#[test]
fn unset_workspace_sticky_clears_flag_keeps_activities_set() {
    // "Toggling off … keeps the current `activities` set." Pin the
    // contract: is_sticky flips to false; activities is untouched (even if
    // it equals all_live_ids at call time).
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let beta = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");

    let mon_out = layout.monitors[0].output_id();
    let target_ws_id = layout
        .active_view(&mon_out)
        .ids()
        .first()
        .copied()
        .expect("alpha view has ws");

    {
        let ws = layout.workspaces.get_mut(&target_ws_id).expect("live");
        ws.is_sticky = true;
        ws.activities = HashSet::from([alpha, beta]);
    }
    layout.verify_invariants();

    let ws_id = layout
        .unset_workspace_sticky(Some(WorkspaceReference::Id(target_ws_id.get())))
        .expect("unset must succeed");
    assert_eq!(ws_id, target_ws_id);

    let ws = layout.workspaces.get(&target_ws_id).expect("live");
    assert!(!ws.is_sticky(), "is_sticky must flip to false");
    assert_eq!(
        ws.activities(),
        &HashSet::from([alpha, beta]),
        "activities set must be preserved verbatim",
    );

    layout.verify_invariants();
}

#[test]
fn unset_workspace_sticky_no_op_when_not_sticky() {
    // Already-not-sticky workspace: Unset must early-exit without touching
    // state. Pinned because the early-exit branch is the common case for
    // toggle-off-on-a-non-sticky-ws.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();

    let mon_out = layout.monitors[0].output_id();
    let target_ws_id = layout
        .active_view(&mon_out)
        .ids()
        .first()
        .copied()
        .expect("alpha view has ws");

    let activities_before = layout
        .workspaces
        .get(&target_ws_id)
        .expect("live")
        .activities()
        .clone();
    assert!(!layout
        .workspaces
        .get(&target_ws_id)
        .expect("live")
        .is_sticky());

    layout
        .unset_workspace_sticky(Some(WorkspaceReference::Id(target_ws_id.get())))
        .expect("unset on non-sticky must succeed (no-op)");

    let ws = layout.workspaces.get(&target_ws_id).expect("live");
    assert!(!ws.is_sticky());
    assert_eq!(ws.activities(), &activities_before);
    assert_eq!(ws.activities(), &HashSet::from([alpha]));

    layout.verify_invariants();
}

#[test]
fn unset_workspace_sticky_preserves_strict_subset_activities() {
    // Pin the sub-contract that unset leaves `activities` intact even when
    // it is a strict subset of all_live_ids. A plausible mis-fix for a
    // misread of would expand activities to all_ids on unset when
    // already non-sticky; this test would catch that regression.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let beta = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");

    let mon_out = layout.monitors[0].output_id();
    let target_ws_id = layout
        .active_view(&mon_out)
        .ids()
        .first()
        .copied()
        .expect("alpha view has ws");

    // Seed the workspace with activities = {alpha} — a strict subset of
    // all_live_ids = {alpha, beta}. is_sticky stays false (no-op path).
    {
        let ws = layout.workspaces.get_mut(&target_ws_id).expect("live");
        ws.activities = HashSet::from([alpha]);
        assert!(!ws.is_sticky());
    }

    layout
        .unset_workspace_sticky(Some(WorkspaceReference::Id(target_ws_id.get())))
        .expect("unset on non-sticky must succeed (no-op)");

    let ws = layout.workspaces.get(&target_ws_id).expect("live");
    assert!(!ws.is_sticky());
    // Strict subset {alpha} must be preserved — not expanded to {alpha, beta}.
    assert_eq!(ws.activities(), &HashSet::from([alpha]));
    assert!(!ws.activities().contains(&beta));
}

#[test]
fn unset_workspace_sticky_no_op_when_workspace_not_found() {
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);

    let err = layout
        .unset_workspace_sticky(Some(WorkspaceReference::Id(u64::MAX)))
        .expect_err("unknown workspace must err at the Layout surface");
    assert_eq!(err, UnsetWorkspaceStickyError::WorkspaceNotFound);
}

#[test]
fn toggle_workspace_sticky_dispatches_to_set_when_off() {
    // is_sticky=false → Toggle dispatches to SetSticky. Outcome carries
    // StickyOn outcome with active_affected bubbled up from the delegate.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let beta = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");

    let mon_out = layout.monitors[0].output_id();
    let target_ws_id = layout
        .active_view(&mon_out)
        .ids()
        .first()
        .copied()
        .expect("alpha view has ws");

    let outcome = layout
        .toggle_workspace_sticky(Some(WorkspaceReference::Id(target_ws_id.get())))
        .expect("toggle must succeed");
    let (out_ws_id, out_active_affected) = match outcome {
        ToggleWorkspaceStickyOutcome::StickyOn {
            ws_id,
            active_affected,
        } => (ws_id, active_affected),
        ToggleWorkspaceStickyOutcome::StickyOff { .. } => {
            panic!("expected StickyOn outcome for toggle-off → on")
        }
    };
    assert_eq!(out_ws_id, target_ws_id);
    // alpha already in set; to_add = {beta}; active_affected = false.
    assert!(!out_active_affected);

    let ws = layout.workspaces.get(&target_ws_id).expect("live");
    assert!(ws.is_sticky());
    assert_eq!(ws.activities(), &HashSet::from([alpha, beta]));

    layout.verify_invariants();
}

#[test]
fn toggle_workspace_sticky_dispatches_to_unset_when_on() {
    // is_sticky=true → Toggle dispatches to UnsetSticky. Outcome carries
    // StickyOff outcome; Unset never touches activities.
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);
    let alpha = layout.active_activity_id();
    let beta = layout
        .create_activity("Beta".to_owned())
        .expect("create beta");

    let mon_out = layout.monitors[0].output_id();
    let target_ws_id = layout
        .active_view(&mon_out)
        .ids()
        .first()
        .copied()
        .expect("alpha view has ws");

    {
        let ws = layout.workspaces.get_mut(&target_ws_id).expect("live");
        ws.is_sticky = true;
        ws.activities = HashSet::from([alpha, beta]);
    }
    layout.verify_invariants();

    let outcome = layout
        .toggle_workspace_sticky(Some(WorkspaceReference::Id(target_ws_id.get())))
        .expect("toggle must succeed");
    // StickyOff makes (StickyOff, active_affected: true) unrepresentable.
    let out_ws_id = match outcome {
        ToggleWorkspaceStickyOutcome::StickyOff { ws_id } => ws_id,
        ToggleWorkspaceStickyOutcome::StickyOn { .. } => {
            panic!("expected StickyOff outcome for toggle-on → off")
        }
    };
    assert_eq!(out_ws_id, target_ws_id);

    let ws = layout.workspaces.get(&target_ws_id).expect("live");
    assert!(!ws.is_sticky());
    // Toggle-off keeps activities set.
    assert_eq!(ws.activities(), &HashSet::from([alpha, beta]));

    layout.verify_invariants();
}

#[test]
fn toggle_workspace_sticky_no_op_when_workspace_not_found() {
    let ops = [Op::AddOutput(1)];
    let mut layout = check_ops(ops);

    let err = layout
        .toggle_workspace_sticky(Some(WorkspaceReference::Id(u64::MAX)))
        .expect_err("unknown workspace must err at the Layout surface");
    assert_eq!(err, ToggleWorkspaceStickyError::WorkspaceNotFound);
}

// ---- `open-on-activity` cross-activity helpers ----

/// Build a [`Config`] with two declared activities (`alpha` first → seed,
/// `beta` second → inactive) and the supplied per-activity workspace lists.
/// The `output_for` slice (same length as `alpha_ws + beta_ws`) routes each
/// workspace to a named output via `open-on-output`. `None` leaves the
/// workspace unbound. Mirrors the integration-test helper in
/// `tests/fixture.rs::config_with_two_activities`, but constructed
/// field-by-field so tests can name an output.
fn cross_activity_config(
    alpha_ws: &[(&str, Option<&str>)],
    beta_ws: &[(&str, Option<&str>)],
) -> jiji_config::Config {
    let mut workspaces = Vec::new();
    for (name, output) in alpha_ws {
        workspaces.push(WorkspaceConfig {
            name: WorkspaceName((*name).to_owned()),
            open_on_output: output.map(|s| s.to_owned()),
            layout: None,
            activities: vec!["alpha".to_owned()],
            sticky: None,
        });
    }
    for (name, output) in beta_ws {
        workspaces.push(WorkspaceConfig {
            name: WorkspaceName((*name).to_owned()),
            open_on_output: output.map(|s| s.to_owned()),
            layout: None,
            activities: vec!["beta".to_owned()],
            sticky: None,
        });
    }
    jiji_config::Config {
        activities: vec![
            jiji_config::ActivityDecl {
                name: jiji_config::ActivityName("alpha".to_owned()),
            },
            jiji_config::ActivityDecl {
                name: jiji_config::ActivityName("beta".to_owned()),
            },
        ],
        workspaces,
        ..jiji_config::Config::default()
    }
}

#[test]
fn activities_find_by_name_resolves_case_insensitively() {
    let layout = Layout::<TestWindow>::new(
        Clock::with_time(Duration::ZERO),
        &cross_activity_config(&[], &[]),
    );
    let beta_id = layout
        .activities()
        .find_by_name("BETA")
        .expect("case-insensitive match must resolve a config-declared activity")
        .id();
    let canonical = layout
        .activities()
        .find_by_name("beta")
        .expect("exact-case lookup must resolve too");
    assert_eq!(beta_id, canonical.id());
    assert!(layout.activities().find_by_name("gamma").is_none());
}

#[test]
fn find_workspace_in_activity_by_name_finds_hidden_workspace() {
    // A workspace tagged with the inactive `beta` activity is not in the
    // active view — `find_workspace_by_name` would miss it. Pin that the new
    // pool-walk helper resolves it.
    let mut layout = Layout::<TestWindow>::new(
        Clock::with_time(Duration::ZERO),
        &cross_activity_config(&[], &[("beta-ws", None)]),
    );
    layout.add_output(make_test_output("output1"), None);

    let beta_id = layout.activities().find_by_name("beta").unwrap().id();
    let alpha_id = layout.activities().find_by_name("alpha").unwrap().id();

    let found = layout
        .find_workspace_in_activity_by_name("beta-ws", beta_id)
        .expect("hidden-activity workspace must be reachable via the new helper");
    assert_eq!(found.name(), Some(&"beta-ws".to_owned()));
    assert!(found.activities().contains(&beta_id));

    // Same name + wrong activity → None (point 3 fallback: activity-scoped lookup).
    assert!(layout
        .find_workspace_in_activity_by_name("beta-ws", alpha_id)
        .is_none());
}

#[test]
fn find_workspace_in_activity_by_name_returns_none_for_active_activity_when_workspace_in_other_activity(
) {
    // point 3 fallback: `open-on-workspace ws` + `open-on-activity X`
    // where `ws` is not tagged with `X` must miss, so the precedence chain
    // can fall through to the next branch.
    let mut layout = Layout::<TestWindow>::new(
        Clock::with_time(Duration::ZERO),
        &cross_activity_config(&[("alpha-only", None)], &[]),
    );
    layout.add_output(make_test_output("output1"), None);

    let beta_id = layout.activities().find_by_name("beta").unwrap().id();
    assert!(layout
        .find_workspace_in_activity_by_name("alpha-only", beta_id)
        .is_none());
}

#[test]
fn monitor_for_workspace_in_activity_returns_none_for_workspace_not_in_activity() {
    // The point-3 lookup variant of the above: `monitor_for_workspace`
    // must filter by activity membership, not just name.
    let mut layout = Layout::<TestWindow>::new(
        Clock::with_time(Duration::ZERO),
        &cross_activity_config(
            &[("alpha-only", Some("output1"))],
            &[("beta-only", Some("output1"))],
        ),
    );
    layout.add_output(make_test_output("output1"), None);

    let alpha_id = layout.activities().find_by_name("alpha").unwrap().id();
    let beta_id = layout.activities().find_by_name("beta").unwrap().id();

    assert!(layout
        .monitor_for_workspace_in_activity("alpha-only", alpha_id)
        .is_some());
    assert!(layout
        .monitor_for_workspace_in_activity("alpha-only", beta_id)
        .is_none(),);
    assert!(layout
        .monitor_for_workspace_in_activity("beta-only", beta_id)
        .is_some());
    assert!(layout
        .monitor_for_workspace_in_activity("beta-only", alpha_id)
        .is_none());
}

#[test]
fn view_in_activity_or_materialize_is_idempotent_when_view_exists() {
    // alpha is the seed (active); beta is config-declared with no workspace bindings. Under
    // the per-activity bookend invariant, `add_output` materializes a single-bookend view for
    // beta on output1 at connect time. `view_in_activity_or_materialize` is then a pure
    // no-op (the contains-key skip fires before any allocation).
    let mut layout = Layout::<TestWindow>::new(
        Clock::with_time(Duration::ZERO),
        &cross_activity_config(&[], &[]),
    );
    layout.add_output(make_test_output("output1"), None);

    let beta_id = layout.activities().find_by_name("beta").unwrap().id();
    let output_id = OutputId::new(&layout.monitors[0].output);

    // Precondition: beta already has a single-bookend view materialized by add_output.
    let pre_view_ids: Vec<WorkspaceId> = layout
        .activities()
        .get(beta_id)
        .unwrap()
        .views()
        .get(&output_id)
        .expect("precondition: per-activity materializer installed beta's view")
        .ids()
        .to_vec();
    assert_eq!(
        pre_view_ids.len(),
        1,
        "freshly-materialized view holds exactly one trailing-empty bookend",
    );
    let pool_size_pre = layout.workspace_pool().len();

    layout.view_in_activity_or_materialize(beta_id, &output_id);

    // Post: no new workspace allocated; the view is byte-identical.
    let post_view_ids: Vec<WorkspaceId> = layout
        .activities()
        .get(beta_id)
        .unwrap()
        .views()
        .get(&output_id)
        .expect("post: beta still owns the view")
        .ids()
        .to_vec();
    assert_eq!(
        post_view_ids, pre_view_ids,
        "no-op materialize must leave the view unchanged",
    );
    assert_eq!(
        layout.workspace_pool().len(),
        pool_size_pre,
        "no-op materialize must not allocate a new workspace",
    );

    let id = post_view_ids[0];
    let ws = layout.workspace_pool().get(&id).unwrap();
    assert!(ws.activities().contains(&beta_id));
    assert!(!ws.has_windows_or_name());

    layout.verify_invariants();
}

#[test]
fn view_in_activity_or_materialize_lifts_pre_tagged_workspaces() {
    // beta has one config-declared workspace bound to output1. Materializing
    // beta's view must lift it (the lift branch in `ensure_view_for`)
    // and pad with a trailing empty so the "last must be empty/unnamed"
    // monitor invariants hold.
    let mut layout = Layout::<TestWindow>::new(
        Clock::with_time(Duration::ZERO),
        &cross_activity_config(&[], &[("beta-ws", Some("output1"))]),
    );
    layout.add_output(make_test_output("output1"), None);

    let beta_id = layout.activities().find_by_name("beta").unwrap().id();
    let output_id = OutputId::new(&layout.monitors[0].output);

    layout.view_in_activity_or_materialize(beta_id, &output_id);

    let view = layout
        .activities()
        .get(beta_id)
        .unwrap()
        .views()
        .get(&output_id)
        .expect("post: beta must own a view for output1");
    assert_eq!(
        view.len(),
        2,
        "lift branch (no EWAF) must produce body + trailing empty"
    );
    let body_id = view.ids()[0];
    let trailing_id = view.ids()[1];
    let body = layout.workspace_pool().get(&body_id).unwrap();
    let trailing = layout.workspace_pool().get(&trailing_id).unwrap();
    assert_eq!(body.name(), Some(&"beta-ws".to_owned()));
    assert!(trailing.name().is_none());
    assert!(!trailing.has_windows_or_name());
    assert_eq!(
        view.active_position(),
        0,
        "active stays on the lifted workspace"
    );

    layout.verify_invariants();
}

#[test]
fn view_in_activity_or_materialize_respects_ewaf_bookend() {
    // EWAF (`empty-workspace-above-first`) mirror of the lift branch above:
    // the materialized view must also have a leading-empty bookend, with
    // active_position shifted to 1.
    let mut config = cross_activity_config(&[], &[("beta-ws", Some("output1"))]);
    config.layout.empty_workspace_above_first = true;

    let mut layout = Layout::<TestWindow>::new(Clock::with_time(Duration::ZERO), &config);
    layout.add_output(make_test_output("output1"), None);

    let beta_id = layout.activities().find_by_name("beta").unwrap().id();
    let output_id = OutputId::new(&layout.monitors[0].output);

    layout.view_in_activity_or_materialize(beta_id, &output_id);

    let view = layout
        .activities()
        .get(beta_id)
        .unwrap()
        .views()
        .get(&output_id)
        .expect("post: beta must own a view for output1");
    assert_eq!(
        view.len(),
        3,
        "EWAF lift branch must produce leading + body + trailing"
    );
    assert_eq!(view.active_position(), 1, "EWAF active shifts to body");
    let leading = layout.workspace_pool().get(&view.ids()[0]).unwrap();
    let body = layout.workspace_pool().get(&view.ids()[1]).unwrap();
    let trailing = layout.workspace_pool().get(&view.ids()[2]).unwrap();
    assert!(leading.name().is_none());
    assert_eq!(body.name(), Some(&"beta-ws".to_owned()));
    assert!(trailing.name().is_none());

    layout.verify_invariants();
}

#[test]
fn view_in_activity_or_materialize_no_op_when_view_already_exists() {
    // Re-call must not allocate a second trailing empty / re-lift workspaces.
    let mut layout = Layout::<TestWindow>::new(
        Clock::with_time(Duration::ZERO),
        &cross_activity_config(&[], &[]),
    );
    layout.add_output(make_test_output("output1"), None);

    let beta_id = layout.activities().find_by_name("beta").unwrap().id();
    let output_id = OutputId::new(&layout.monitors[0].output);

    layout.view_in_activity_or_materialize(beta_id, &output_id);
    let after_first = layout
        .activities()
        .get(beta_id)
        .unwrap()
        .views()
        .get(&output_id)
        .unwrap()
        .ids()
        .to_vec();

    layout.view_in_activity_or_materialize(beta_id, &output_id);
    let after_second = layout
        .activities()
        .get(beta_id)
        .unwrap()
        .views()
        .get(&output_id)
        .unwrap()
        .ids()
        .to_vec();

    assert_eq!(after_first, after_second, "second call must be a no-op");
    layout.verify_invariants();
}

#[test]
fn add_window_to_hidden_activity_workspace_via_add_window_target_workspace() {
    // Pin the hidden-target shortcut in `Layout::add_window`. A workspace
    // tagged with the inactive `beta` activity is materialized via
    // `view_in_activity_or_materialize`; then `add_window` is called with
    // `AddWindowTarget::Workspace(beta_ws_id)`. The window must land in the
    // pool entry, the active activity's view for output1 must remain
    // unchanged, and the layout must stay invariant-clean.
    let mut layout = Layout::<TestWindow>::new(
        Clock::with_time(Duration::ZERO),
        &cross_activity_config(&[], &[("beta-ws", Some("output1"))]),
    );
    layout.add_output(make_test_output("output1"), None);

    let beta_id = layout.activities().find_by_name("beta").unwrap().id();
    let output_id = OutputId::new(&layout.monitors[0].output);

    // Materialize beta's view so the workspace is reachable; capture the
    // alpha-active view shape as a regression baseline.
    layout.view_in_activity_or_materialize(beta_id, &output_id);
    let alpha_view_before = layout.active_view(&output_id).ids().to_vec();
    let beta_ws_id = layout
        .find_workspace_in_activity_by_name("beta-ws", beta_id)
        .expect("beta-ws must be present in pool")
        .id();

    let window = TestWindow::new(TestWindowParams::new(0));
    let _output = layout.add_window(
        window,
        AddWindowTarget::Workspace(beta_ws_id),
        None,
        None,
        false,
        false,
        ActivateWindow::No,
    );

    // The window landed in the hidden workspace.
    let beta_ws = layout.workspace_pool().get(&beta_ws_id).unwrap();
    assert!(
        beta_ws.has_window(&0),
        "window must land in the hidden-activity workspace"
    );

    // The active activity's view for output1 is unchanged (no auto-switch,
    // no view mutation).
    assert_eq!(
        layout.active_view(&output_id).ids(),
        alpha_view_before.as_slice(),
        "alpha view for output1 must remain unchanged after adding to a hidden-activity workspace",
    );

    layout.verify_invariants();
}

#[test]
fn view_in_activity_or_materialize_then_switch_activity_does_not_double_allocate() {
    // Regression pin: `view_in_activity_or_materialize` followed by
    // `switch_activity` to the same target must not allocate a second trailing
    // empty. The `contains_key` early-exit in `ensure_all_activity_views` is what
    // prevents it; this test drives through the public `switch_activity` entry
    // to confirm the guard is reachable via that path.
    let mut layout = Layout::<TestWindow>::new(
        Clock::with_time(Duration::ZERO),
        &cross_activity_config(&[], &[]),
    );
    layout.add_output(make_test_output("output1"), None);

    let beta_id = layout.activities().find_by_name("beta").unwrap().id();
    let output_id = OutputId::new(&layout.monitors[0].output);

    layout.view_in_activity_or_materialize(beta_id, &output_id);
    let ids_after_materialize = layout
        .activities()
        .get(beta_id)
        .unwrap()
        .views()
        .get(&output_id)
        .unwrap()
        .ids()
        .to_vec();

    layout.switch_activity(beta_id);

    let ids_after_switch = layout
        .activities()
        .get(beta_id)
        .unwrap()
        .views()
        .get(&output_id)
        .unwrap()
        .ids()
        .to_vec();

    assert_eq!(
        ids_after_materialize, ids_after_switch,
        "switch_activity must not allocate a second trailing empty after materialize",
    );
    layout.verify_invariants();
}

#[test]
fn find_workspace_in_activity_by_name_resolves_case_insensitively() {
    // `find_workspace_in_activity_by_name` uses `eq_ignore_ascii_case`
    // internally; pin that upper-cased lookups still resolve. A future
    // refactor to a typed `WorkspaceName` could silently break this.
    let mut layout = Layout::<TestWindow>::new(
        Clock::with_time(Duration::ZERO),
        &cross_activity_config(&[], &[("beta-ws", None)]),
    );
    layout.add_output(make_test_output("output1"), None);

    let beta_id = layout.activities().find_by_name("beta").unwrap().id();

    let lower = layout.find_workspace_in_activity_by_name("beta-ws", beta_id);
    let upper = layout.find_workspace_in_activity_by_name("BETA-WS", beta_id);
    let mixed = layout.find_workspace_in_activity_by_name("Beta-Ws", beta_id);

    assert!(lower.is_some(), "lowercase must resolve");
    assert!(upper.is_some(), "uppercase must resolve (case-insensitive)");
    assert!(
        mixed.is_some(),
        "mixed-case must resolve (case-insensitive)"
    );
    assert_eq!(
        lower.unwrap().id(),
        upper.unwrap().id(),
        "all variants must resolve to the same workspace"
    );
    // Negative: a completely different name must not resolve.
    assert!(
        layout
            .find_workspace_in_activity_by_name("nonexistent", beta_id)
            .is_none(),
        "nonexistent name must not resolve"
    );
    // Negative: a substring of the workspace name must not resolve (guards against
    // a hypothetical future `contains(...)` regression replacing `eq_ignore_ascii_case`).
    assert!(
        layout
            .find_workspace_in_activity_by_name("eta-w", beta_id)
            .is_none(),
        "substring of the workspace name must not resolve"
    );
}

#[test]
fn monitor_for_workspace_in_activity_resolves_case_insensitively() {
    // `monitor_for_workspace_in_activity` uses `eq_ignore_ascii_case`
    // internally; pin that upper-cased lookups still resolve.
    let mut layout = Layout::<TestWindow>::new(
        Clock::with_time(Duration::ZERO),
        &cross_activity_config(&[("alpha-ws", Some("output1"))], &[]),
    );
    layout.add_output(make_test_output("output1"), None);

    let alpha_id = layout.activities().find_by_name("alpha").unwrap().id();

    assert!(
        layout
            .monitor_for_workspace_in_activity("alpha-ws", alpha_id)
            .is_some(),
        "lowercase must resolve"
    );
    assert!(
        layout
            .monitor_for_workspace_in_activity("ALPHA-WS", alpha_id)
            .is_some(),
        "uppercase must resolve (case-insensitive)"
    );
    assert!(
        layout
            .monitor_for_workspace_in_activity("Alpha-Ws", alpha_id)
            .is_some(),
        "mixed-case must resolve (case-insensitive)"
    );
    // Negative: a completely different name must not resolve.
    assert!(
        layout
            .monitor_for_workspace_in_activity("nonexistent", alpha_id)
            .is_none(),
        "nonexistent name must not resolve"
    );
    // Negative: a substring of the workspace name must not resolve (guards against
    // a hypothetical future `contains(...)` regression replacing `eq_ignore_ascii_case`).
    assert!(
        layout
            .monitor_for_workspace_in_activity("lpha-w", alpha_id)
            .is_none(),
        "substring of the workspace name must not resolve"
    );
}

// `add_window_to_hidden_workspace` must call `dormant_view_bookend_fixup` so that
// a dormant activity whose view has the target workspace as its trailing entry gets
// a fresh trailing empty appended after the window lands via the pool-direct hidden
// path (the workspace is NOT in any active view).
#[test]
fn add_window_to_hidden_workspace_with_dormant_trailing_appends_bookend() {
    // Two config activities (alpha + beta). "beta-ws" is exclusively beta's.
    let mut layout = Layout::<TestWindow>::new(
        Clock::with_time(Duration::ZERO),
        &cross_activity_config(&[], &[("beta-ws", Some("output1"))]),
    );
    layout.add_output(make_test_output("output1"), None);

    let beta_id = layout.activities().find_by_name("beta").unwrap().id();
    let output_id = OutputId::new(&layout.monitors[0].output);

    // Ensure beta's view is materialized.
    layout.view_in_activity_or_materialize(beta_id, &output_id);

    // Find beta-ws in the pool.
    let beta_ws_id = layout
        .find_workspace_in_activity_by_name("beta-ws", beta_id)
        .expect("beta-ws must be in pool")
        .id();

    // Hand-roll beta's view so beta-ws is the sole trailing entry (no bookend).
    // This is the precondition the fixup must correct: after the window lands on
    // beta-ws, the view must grow.
    test_override_activity_view(
        &mut layout,
        beta_id,
        output_id.clone(),
        WorkspaceView::new(vec![beta_ws_id], 0),
    );

    let beta_view_len_before = 1usize; // just beta_ws_id

    // beta-ws is NOT in alpha's active view, so add_window takes the hidden-target
    // path (`add_window_to_hidden_workspace`).
    let window = TestWindow::new(TestWindowParams::new(42));
    let _out = layout.add_window(
        window,
        AddWindowTarget::Workspace(beta_ws_id),
        None,
        None,
        false,
        false,
        ActivateWindow::No,
    );

    // beta's view must have grown: dormant_view_bookend_fixup must have appended a
    // fresh trailing empty after beta_ws_id.
    let beta_view_after = layout
        .activities
        .get(beta_id)
        .expect("beta live")
        .views()
        .get(&output_id)
        .expect("beta has view");
    assert!(
        beta_view_after.len() > beta_view_len_before,
        "dormant_view_bookend_fixup must extend beta's view after add_window_to_hidden_workspace",
    );
    assert_ne!(
        beta_view_after.ids().last(),
        Some(&beta_ws_id),
        "beta_ws_id must no longer be beta's trailing entry after the window was added",
    );

    layout.verify_invariants();
}

// `dormant_view_bookend_fixup` under `empty_workspace_above_first = true`: when a
// window lands on a workspace that sits at position 0 of a dormant activity's view,
// the fixup must prepend a fresh leading-empty bookend in addition to appending the
// trailing one. A single-entry view (`[shared_ws]`) where `shared_ws` is both first
// and last exercises both branches.
#[test]
fn add_window_under_ewaf_prepends_leading_empty_to_dormant_view_at_position_zero() {
    // Build a layout with EWAF enabled.
    let config = {
        let mut c = cross_activity_config(&[], &[("beta-ws", Some("output1"))]);
        c.layout.empty_workspace_above_first = true;
        c
    };
    let mut layout = Layout::<TestWindow>::new(Clock::with_time(Duration::ZERO), &config);
    layout.add_output(make_test_output("output1"), None);

    let beta_id = layout.activities().find_by_name("beta").unwrap().id();
    let output_id = OutputId::new(&layout.monitors[0].output);

    // Materialize beta's view so beta-ws is reachable.
    layout.view_in_activity_or_materialize(beta_id, &output_id);

    let beta_ws_id = layout
        .find_workspace_in_activity_by_name("beta-ws", beta_id)
        .expect("beta-ws must be in pool")
        .id();

    // Hand-roll beta's view as a single-entry [beta_ws_id]. Under EWAF, a single
    // workspace view is valid (it acts as the combined leading+trailing empty).
    // beta_ws_id is at position 0 (= last), so both `is_first` and `is_last` fire.
    test_override_activity_view(
        &mut layout,
        beta_id,
        output_id.clone(),
        WorkspaceView::new(vec![beta_ws_id], 0),
    );

    let beta_view_len_before = 1usize;

    // beta-ws is NOT in alpha's active view — add_window takes the hidden path.
    let window = TestWindow::new(TestWindowParams::new(55));
    let _out = layout.add_window(
        window,
        AddWindowTarget::Workspace(beta_ws_id),
        None,
        None,
        false,
        false,
        ActivateWindow::No,
    );

    // Beta's view must have grown by 2: trailing bookend appended and leading bookend
    // prepended, yielding [ewaf_leading, beta_ws_id, trailing_bookend].
    let beta_view_after = layout
        .activities
        .get(beta_id)
        .expect("beta live")
        .views()
        .get(&output_id)
        .expect("beta has view");
    assert!(
        beta_view_after.len() > beta_view_len_before,
        "EWAF fixup must grow beta's view after the window lands at position 0",
    );
    // beta_ws_id must no longer be the first entry (a new leading empty was prepended).
    assert_ne!(
        beta_view_after.ids().first(),
        Some(&beta_ws_id),
        "beta_ws_id must no longer be beta's first entry after EWAF leading-empty prepend",
    );
    // beta_ws_id must no longer be the last entry (a trailing bookend was appended).
    assert_ne!(
        beta_view_after.ids().last(),
        Some(&beta_ws_id),
        "beta_ws_id must no longer be beta's trailing entry after the window was added",
    );

    layout.verify_invariants();
}

#[test]
fn partial_disconnect_migrates_dormant_activity_views_to_primary() {
    // Partial-disconnect path: the dormant view for a non-active activity gets drained
    // for the disconnecting output, with named / window-bearing workspaces migrated into
    // that activity's view for the still-connected (primary) monitor. The pool entry's
    // `output_id` is preserved so reconnect-time reclaim can find it.
    let ops = [Op::AddOutput(1), Op::AddOutput(2)];
    let mut layout = check_ops(ops);
    let seed_id = layout.active_activity_id();
    let out1 = layout.monitors[0].output_id();
    let out2 = layout.monitors[1].output_id();

    let beta = super::activity::Activity::new_runtime("beta".to_owned());
    let beta_id = beta.id();
    test_insert_activity(&mut layout, beta);

    // Switch to beta so its view on out1 becomes active, plant a window in it, then
    // switch back to seed so beta's views become dormant.
    layout.switch_activity(beta_id);
    layout.verify_invariants();
    layout.add_window(
        TestWindow::new(TestWindowParams::new(60)),
        AddWindowTarget::Auto,
        None,
        None,
        false,
        false,
        ActivateWindow::default(),
    );
    layout.verify_invariants();

    // Identify the windowed workspace id on beta's out1 view before going dormant.
    let beta_out1_pre: Vec<WorkspaceId> = layout.active_view(&out1).ids().to_vec();
    let windowed_id: WorkspaceId = *beta_out1_pre
        .iter()
        .find(|id| {
            layout
                .workspaces
                .get(id)
                .expect("snapshot id must be live")
                .has_windows_or_name()
        })
        .expect("beta out1 view must contain at least one windowed workspace");

    layout.switch_activity(seed_id);
    layout.verify_invariants();

    // Drop output1 — beta is dormant; the partial-disconnect walk must drain beta's
    // out1 view and migrate the windowed workspace into beta's out2 view.
    check_ops_on_layout(&mut layout, [Op::RemoveOutput(1)]);

    // (i) No activity has a view keyed by out1.
    for act in layout.activities.iter() {
        assert!(
            !act.views().contains_key(&out1),
            "no activity may hold a view keyed by the disconnected output (offender: {:?})",
            act.id(),
        );
    }

    // (ii) The windowed workspace is now in beta's out2 view.
    let beta_after = layout
        .activities
        .get(beta_id)
        .expect("beta must remain present after partial disconnect");
    let beta_out2 = beta_after
        .views()
        .get(&out2)
        .expect("beta must hold a view for the still-connected output");
    assert!(
        beta_out2.ids().contains(&windowed_id),
        "windowed workspace {windowed_id:?} must appear in beta's view for the remaining monitor",
    );

    // (iii) Pool entry's output_id still equals the disconnecting output.
    let ws = layout
        .workspaces
        .get(&windowed_id)
        .expect("migrated workspace must remain a pool key");
    assert_eq!(
        ws.output_id().cloned(),
        Some(out1.clone()),
        "system motion preserves output_id; the migrated workspace must still point at out1",
    );

    // (iv) Layout invariants hold.
    layout.verify_invariants();
}

#[test]
fn partial_reconnect_reclaims_dormant_activity_workspaces_via_output_id() {
    // Partial-reconnect path: when output1 comes back, `ensure_view_for`'s lift branch
    // walks the pool for `output_id == out1 && activity == beta` and reclaims those
    // workspaces into a fresh beta.views[out1]. The new source-side dedup block must
    // drop those ids from beta.views[out2] in the same operation, so no workspace
    // appears in both views.
    let ops = [Op::AddOutput(1), Op::AddOutput(2)];
    let mut layout = check_ops(ops);
    let seed_id = layout.active_activity_id();
    let out1 = layout.monitors[0].output_id();
    let out2 = layout.monitors[1].output_id();

    let beta = super::activity::Activity::new_runtime("beta".to_owned());
    let beta_id = beta.id();
    test_insert_activity(&mut layout, beta);

    layout.switch_activity(beta_id);
    layout.verify_invariants();
    layout.add_window(
        TestWindow::new(TestWindowParams::new(70)),
        AddWindowTarget::Auto,
        None,
        None,
        false,
        false,
        ActivateWindow::default(),
    );
    layout.verify_invariants();

    let beta_out1_pre: Vec<WorkspaceId> = layout.active_view(&out1).ids().to_vec();
    let windowed_id: WorkspaceId = *beta_out1_pre
        .iter()
        .find(|id| {
            layout
                .workspaces
                .get(id)
                .expect("snapshot id must be live")
                .has_windows_or_name()
        })
        .expect("beta out1 view must contain at least one windowed workspace");

    layout.switch_activity(seed_id);
    layout.verify_invariants();
    check_ops_on_layout(&mut layout, [Op::RemoveOutput(1)]);
    // Reconnect; do NOT switch activity yet — the reclaim must fire from
    // ensure_all_activity_views, not from a switch path.
    check_ops_on_layout(&mut layout, [Op::AddOutput(1)]);

    let beta_after = layout.activities.get(beta_id).expect("beta must be live");

    // (i) Beta has a fresh view for out1.
    let beta_out1 = beta_after
        .views()
        .get(&out1)
        .expect("beta must have a freshly materialized view for the reconnected output");

    // (ii) The windowed workspace appears in beta.views[out1] (lift branch reclaimed).
    assert!(
        beta_out1.ids().contains(&windowed_id),
        "windowed workspace {windowed_id:?} must be reclaimed into beta's fresh out1 view",
    );

    // (iii) verify_invariants passes.
    layout.verify_invariants();

    // (iv) Critical dedup contract: beta.views[out1] ∩ beta.views[out2] is empty.
    let out1_ids: std::collections::HashSet<WorkspaceId> =
        beta_out1.ids().iter().copied().collect();
    let beta_out2 = beta_after
        .views()
        .get(&out2)
        .expect("beta must hold a view for the still-connected output");
    let out2_ids: std::collections::HashSet<WorkspaceId> =
        beta_out2.ids().iter().copied().collect();
    let intersection: Vec<WorkspaceId> = out1_ids.intersection(&out2_ids).copied().collect();
    assert!(
        intersection.is_empty(),
        "ensure_view_for's lift branch must drop reclaimed ids from sibling views; \
         found {intersection:?} present in both beta.views[out1] and beta.views[out2]",
    );
}

#[test]
fn cable_flap_preserves_workspace_identity_across_activities() {
    // A disconnect + reconnect cycle on output1 with a non-active activity holding a
    // windowed workspace pre-disconnect must preserve the WorkspaceId of that workspace
    // — system motion does not destroy and re-mint it. Empty unnamed bookends ARE
    // destroyed and re-minted on this path and we deliberately do not pin their ids.
    let ops = [Op::AddOutput(1), Op::AddOutput(2)];
    let mut layout = check_ops(ops);
    let seed_id = layout.active_activity_id();
    let out1 = layout.monitors[0].output_id();

    let beta = super::activity::Activity::new_runtime("beta".to_owned());
    let beta_id = beta.id();
    test_insert_activity(&mut layout, beta);

    layout.switch_activity(beta_id);
    layout.verify_invariants();
    layout.add_window(
        TestWindow::new(TestWindowParams::new(80)),
        AddWindowTarget::Auto,
        None,
        None,
        false,
        false,
        ActivateWindow::default(),
    );
    layout.verify_invariants();

    let beta_out1_pre: Vec<WorkspaceId> = layout.active_view(&out1).ids().to_vec();
    let windowed_id: WorkspaceId = *beta_out1_pre
        .iter()
        .find(|id| {
            layout
                .workspaces
                .get(id)
                .expect("snapshot id must be live")
                .has_windows_or_name()
        })
        .expect("beta out1 view must contain at least one windowed workspace");

    layout.switch_activity(seed_id);
    layout.verify_invariants();

    // Cable flap: disconnect then reconnect.
    check_ops_on_layout(&mut layout, [Op::RemoveOutput(1)]);
    check_ops_on_layout(&mut layout, [Op::AddOutput(1)]);

    // The snapshotted id must still be a live pool key — system motion preserves
    // workspace identity across the flap.
    assert!(
        layout.workspaces.contains_key(&windowed_id),
        "windowed workspace {windowed_id:?} must survive the cable flap as a live pool key",
    );
    layout.verify_invariants();
}

#[test]
fn disconnected_pool_unchanged_under_partial_disconnect() {
    // Partial disconnect must NOT feed the disconnected pool: the surviving monitor
    // absorbs the dormant activity's workspaces via in-views migration. The
    // `disconnected_workspace_ids` field is exclusively for the all-outputs-gone
    // (`NoOutputs`) path.
    let ops = [Op::AddOutput(1), Op::AddOutput(2)];
    let mut layout = check_ops(ops);
    let seed_id = layout.active_activity_id();

    let beta = super::activity::Activity::new_runtime("beta".to_owned());
    let beta_id = beta.id();
    test_insert_activity(&mut layout, beta);

    layout.switch_activity(beta_id);
    layout.verify_invariants();
    layout.add_window(
        TestWindow::new(TestWindowParams::new(90)),
        AddWindowTarget::Auto,
        None,
        None,
        false,
        false,
        ActivateWindow::default(),
    );
    layout.verify_invariants();
    layout.switch_activity(seed_id);
    layout.verify_invariants();

    assert!(
        layout.disconnected_workspace_ids.is_empty(),
        "pre-disconnect baseline: disconnected pool must be empty with both outputs connected",
    );

    check_ops_on_layout(&mut layout, [Op::RemoveOutput(1)]);

    assert!(
        layout.disconnected_workspace_ids.is_empty(),
        "partial disconnect must not push workspaces into disconnected_workspace_ids; \
         the surviving monitor absorbs them via the per-activity views migration",
    );
    layout.verify_invariants();
}

#[test]
fn partial_disconnect_dormant_walk_handles_multi_activity_membership() {
    // Regression for the CRIT-2 dedup bug: a workspace shared by two dormant activities
    // (A = alpha, B = beta, ws.activities = {alpha, beta}) must end up in *both*
    // alpha.views[out2] AND beta.views[out2] after out1 is disconnected. The old code
    // keyed `already_kept` by WorkspaceId alone, causing the second activity's iter to
    // skip the migration — leaving only one activity with the workspace in its view.
    //
    // Setup: switch to alpha, add a window (shared_ws), switch to beta, splice shared_ws
    // into beta's out1 view at a non-trailing slot (before its trailing bookend), then
    // patch the pool entry's activities set to include both {alpha, beta}. Finally switch
    // to seed so both activities are dormant and trigger partial disconnect.
    let ops = [Op::AddOutput(1), Op::AddOutput(2)];
    let mut layout = check_ops(ops);
    let seed_id = layout.active_activity_id();
    let out1 = layout.monitors[0].output_id();
    let out2 = layout.monitors[1].output_id();

    // Create two extra activities while seed is active — both get views on both outputs.
    let alpha = layout
        .create_activity("alpha".to_owned())
        .expect("create alpha");
    let beta = layout
        .create_activity("beta".to_owned())
        .expect("create beta");

    // Switch to alpha and plant a window so the active workspace has_windows.
    layout.switch_activity(alpha);
    layout.verify_invariants();
    layout.add_window(
        TestWindow::new(TestWindowParams::new(100)),
        AddWindowTarget::Auto,
        None,
        None,
        false,
        false,
        ActivateWindow::default(),
    );
    layout.verify_invariants();

    // Grab the windowed workspace id from alpha's out1 view.
    let shared_ws_id: WorkspaceId = *layout
        .active_view(&out1)
        .ids()
        .iter()
        .find(|id| {
            layout
                .workspaces
                .get(id)
                .expect("snapshot id must be live")
                .has_windows_or_name()
        })
        .expect("alpha out1 view must contain a windowed workspace");

    // Switch to seed so alpha is dormant; alpha.views[out1] now contains shared_ws_id.
    layout.switch_activity(seed_id);
    layout.verify_invariants();

    // Directly splice shared_ws_id into beta's dormant view for out1 *before* the trailing
    // bookend, so the bookend invariant holds (beta.views[out1] = [shared_ws_id, bookend]).
    // This bypasses set_workspace_activities (which would append after the bookend and
    // violate the trailing-empty invariant for a windowed workspace).
    //
    // Note: this multi-activity-shared windowed state is currently only producible via direct
    // pool+view splice. Production paths through `set_workspace_activities` would trip the
    // trailing-bookend invariant (the Add branch appends past the trailing-empty bookend —
    // a latent gap to be closed in a future pass). This test forward-proofs the
    // partial-disconnect walk's contract for the state once that latent gap is closed.
    {
        let beta_activity = layout.activities.get_mut(beta).expect("beta live");
        let beta_out1 = beta_activity
            .views_mut()
            .get_mut(&out1)
            .expect("beta has out1 view");
        // Insert before the trailing bookend (position len-1).
        let insert_pos = beta_out1.len() - 1;
        beta_out1.insert(insert_pos, shared_ws_id);
    }
    // Patch the pool entry's activities set to include both alpha and beta.
    {
        let ws = layout.workspaces.get_mut(&shared_ws_id).expect("pool key");
        ws.activities.insert(alpha);
        ws.activities.insert(beta);
    }

    // Verify the invariants still hold with our surgical splice.
    layout.verify_invariants();

    // Verify both dormant activities have shared_ws_id in their out1 views.
    let alpha_before = layout.activities.get(alpha).expect("alpha live");
    assert!(
        alpha_before
            .views()
            .get(&out1)
            .expect("alpha has out1 view")
            .ids()
            .contains(&shared_ws_id),
        "precondition: alpha's dormant out1 view must contain shared_ws_id",
    );
    let beta_before = layout.activities.get(beta).expect("beta live");
    assert!(
        beta_before
            .views()
            .get(&out1)
            .expect("beta has out1 view")
            .ids()
            .contains(&shared_ws_id),
        "precondition: beta's dormant out1 view must contain shared_ws_id",
    );

    // Partial-disconnect out1.
    check_ops_on_layout(&mut layout, [Op::RemoveOutput(1)]);

    // Both alpha and beta must have shared_ws_id in their out2 views.
    let alpha_after = layout
        .activities
        .get(alpha)
        .expect("alpha live after disconnect");
    assert!(
        alpha_after
            .views()
            .get(&out2)
            .expect("alpha must hold a view for the still-connected output")
            .ids()
            .contains(&shared_ws_id),
        "shared workspace must appear in alpha's view for the remaining monitor after partial disconnect",
    );
    let beta_after = layout
        .activities
        .get(beta)
        .expect("beta live after disconnect");
    assert!(
        beta_after
            .views()
            .get(&out2)
            .expect("beta must hold a view for the still-connected output")
            .ids()
            .contains(&shared_ws_id),
        "shared workspace must appear in beta's view for the remaining monitor after partial disconnect",
    );

    layout.verify_invariants();
}

#[test]
fn partial_disconnect_preserves_named_but_empty_dormant_workspaces() {
    // A named (but window-less) workspace in a dormant activity's view for the disconnecting
    // output must survive partial disconnect — it migrates to that activity's view for the
    // remaining monitor rather than being doomed. Mirrors
    // `removing_all_outputs_preserves_empty_named_workspaces` for the partial-disconnect side.
    let ops = [Op::AddOutput(1), Op::AddOutput(2)];
    let mut layout = check_ops(ops);
    let seed_id = layout.active_activity_id();
    let out1 = layout.monitors[0].output_id();
    let out2 = layout.monitors[1].output_id();

    let beta = super::activity::Activity::new_runtime("beta".to_owned());
    let beta_id = beta.id();
    test_insert_activity(&mut layout, beta);

    // Switch to beta so it acquires views on both outputs.
    layout.switch_activity(beta_id);
    layout.verify_invariants();

    // Name the active (empty) workspace on out1 so it has_windows_or_name() == true.
    let named_ws_id = layout.active_view(&out1).active();
    layout.set_workspace_name(
        "keep_me".to_owned(),
        Some(WorkspaceReference::Id(named_ws_id.get())),
    );
    layout.verify_invariants();

    // Switch back to seed so beta is dormant.
    layout.switch_activity(seed_id);
    layout.verify_invariants();

    // Verify the pool entry is named but has no windows.
    {
        let ws = layout.workspaces.get(&named_ws_id).expect("pool key");
        assert!(ws.name.is_some(), "precondition: workspace must be named");
        assert!(
            !ws.has_windows(),
            "precondition: workspace must have no windows"
        );
        assert!(
            ws.has_windows_or_name(),
            "precondition: has_windows_or_name must be true for named ws"
        );
    }

    // Partial-disconnect out1.
    check_ops_on_layout(&mut layout, [Op::RemoveOutput(1)]);

    // The named workspace must survive and appear in beta's out2 view.
    assert!(
        layout.workspaces.contains_key(&named_ws_id),
        "named empty workspace must survive partial disconnect as a pool key",
    );
    let beta_after = layout.activities.get(beta_id).expect("beta live");
    assert!(
        beta_after
            .views()
            .get(&out2)
            .expect("beta must hold a view for the still-connected output")
            .ids()
            .contains(&named_ws_id),
        "named empty workspace must appear in beta's view for the remaining monitor after partial disconnect",
    );

    layout.verify_invariants();
}

#[test]
fn partial_disconnect_of_primary_migrates_dormant_views_to_new_primary() {
    // When the primary monitor (index 0) is disconnected, `primary_idx` does a
    // `saturating_sub(1)` which stays at 0 — now pointing at the formerly secondary
    // monitor. Dormant activities' migrations must target this new primary, not the
    // removed one.
    let ops = [Op::AddOutput(1), Op::AddOutput(2)];
    let mut layout = check_ops(ops);
    let seed_id = layout.active_activity_id();
    let out1 = layout.monitors[0].output_id();
    let out2 = layout.monitors[1].output_id();

    // Verify out1 is primary (index 0).
    assert_eq!(layout.primary_idx, 0, "precondition: out1 must be primary");

    let beta = super::activity::Activity::new_runtime("beta".to_owned());
    let beta_id = beta.id();
    test_insert_activity(&mut layout, beta);

    // Switch to beta, plant a window on out1 (primary), then switch back to seed.
    layout.switch_activity(beta_id);
    layout.verify_invariants();
    layout.add_window(
        TestWindow::new(TestWindowParams::new(110)),
        AddWindowTarget::Auto,
        None,
        None,
        false,
        false,
        ActivateWindow::default(),
    );
    layout.verify_invariants();

    let windowed_id: WorkspaceId = *layout
        .active_view(&out1)
        .ids()
        .iter()
        .find(|id| {
            layout
                .workspaces
                .get(id)
                .expect("snapshot id must be live")
                .has_windows_or_name()
        })
        .expect("beta out1 view must contain a windowed workspace");

    layout.switch_activity(seed_id);
    layout.verify_invariants();

    // Disconnect out1 (the primary). primary_idx stays 0, now pointing at out2.
    check_ops_on_layout(&mut layout, [Op::RemoveOutput(1)]);

    assert_eq!(
        layout.primary_idx, 0,
        "after removing out1 (primary), primary_idx saturates to 0 = out2",
    );

    // Beta's windowed workspace must have migrated to the new primary (out2).
    let beta_after = layout
        .activities
        .get(beta_id)
        .expect("beta live after disconnect");
    assert!(
        !beta_after.views().contains_key(&out1),
        "beta must not hold a view keyed by the disconnected output",
    );
    let beta_out2 = beta_after
        .views()
        .get(&out2)
        .expect("beta must hold a view for the new primary (out2)");
    assert!(
        beta_out2.ids().contains(&windowed_id),
        "windowed workspace must appear in beta's view for the new primary (out2) after primary disconnect",
    );

    layout.verify_invariants();
}

#[test]
fn partial_disconnect_dooms_sentinel_output_id_dormant_workspace() {
    // A dormant activity's workspace whose output_id is the empty-string sentinel (produced
    // by `new_with_config_no_outputs`) must be doomed by the partial-disconnect walk rather
    // than migrated. Mirrors the sentinel coverage in the full-disconnect branch.
    let ops = [Op::AddOutput(1), Op::AddOutput(2)];
    let mut layout = check_ops(ops);
    let seed_id = layout.active_activity_id();
    let out1 = layout.monitors[0].output_id();
    let out2 = layout.monitors[1].output_id();
    let _ = out2; // referenced in verify_invariants path

    let beta = super::activity::Activity::new_runtime("beta".to_owned());
    let beta_id = beta.id();
    test_insert_activity(&mut layout, beta);

    // Switch to beta so it has views on both outputs, then switch back to seed (beta dormant).
    layout.switch_activity(beta_id);
    layout.verify_invariants();
    layout.switch_activity(seed_id);
    layout.verify_invariants();

    // Inject a sentinel-output-id workspace directly into beta's view for out1.
    // This simulates a workspace produced by new_with_config_no_outputs that somehow reached
    // a dormant view (e.g. via a config reload or future code path).
    let sentinel_ws = Workspace::<TestWindow>::new_with_config_no_outputs(
        None,
        HashSet::from([beta_id]),
        layout.clock.clone(),
        layout.options.clone(),
    );
    let sentinel_id = sentinel_ws.id();
    assert_eq!(
        sentinel_ws.output_id().map(|id| id.as_str()),
        Some(""),
        "precondition: new_with_config_no_outputs seeds empty-string sentinel",
    );
    layout.workspaces.insert(sentinel_id, sentinel_ws);

    // Splice into beta's dormant view for out1 before the trailing bookend.
    let beta_out1_view = layout
        .activities
        .get_mut(beta_id)
        .expect("beta live")
        .views_mut()
        .get_mut(&out1)
        .expect("beta has out1 view");
    let insert_pos = beta_out1_view.len() - 1;
    beta_out1_view.insert(insert_pos, sentinel_id);

    // Confirm the spliced state is itself invariant-clean before triggering the disconnect.
    layout.verify_invariants();

    // Partial-disconnect out1.
    check_ops_on_layout(&mut layout, [Op::RemoveOutput(1)]);

    // The sentinel workspace must have been doomed — it is no longer a pool key.
    assert!(
        !layout.workspaces.contains_key(&sentinel_id),
        "sentinel-output-id workspace must be destroyed by the partial-disconnect walk",
    );

    layout.verify_invariants();
}
