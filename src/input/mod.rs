use std::any::Any;
use std::collections::hash_map::Entry;
use std::collections::HashSet;
use std::time::{Duration, Instant};

use calloop::timer::{TimeoutAction, Timer};
use input::event::gesture::GestureEventCoordinates as _;
use jiji_config::utils::RegexEq;
use jiji_config::{
    flatten, key_to_wire_string, Action, Bind, Binds, Config, Key, LayerId, ModKey, Modifiers,
    MruDirection, ResolvedAppearanceOverride, SwitchBinds, Trigger, WorkspaceReference,
};
use jiji_ipc::{ActivityReferenceArg, LayoutSwitchTarget, NoOpReason};
use smithay::backend::input::{
    AbsolutePositionEvent, Axis, AxisSource, ButtonState, Device, DeviceCapability, Event,
    GestureBeginEvent, GestureEndEvent, GesturePinchUpdateEvent as _, GestureSwipeUpdateEvent as _,
    InputEvent, KeyState, KeyboardKeyEvent, Keycode, MouseButton, PointerAxisEvent,
    PointerButtonEvent, PointerMotionEvent, ProximityState, Switch, SwitchState, SwitchToggleEvent,
    TabletToolButtonEvent, TabletToolEvent, TabletToolProximityEvent, TabletToolTipEvent,
    TabletToolTipState, TouchEvent,
};
use smithay::backend::libinput::LibinputInputBackend;
use smithay::input::dnd::DnDGrab;
use smithay::input::keyboard::{keysyms, FilterResult, Keysym, Layout, ModifiersState};
use smithay::input::pointer::{
    AxisFrame, ButtonEvent, CursorIcon, CursorImageStatus, Focus, GestureHoldBeginEvent,
    GestureHoldEndEvent, GesturePinchBeginEvent, GesturePinchEndEvent, GesturePinchUpdateEvent,
    GestureSwipeBeginEvent, GestureSwipeEndEvent, GestureSwipeUpdateEvent,
    GrabStartData as PointerGrabStartData, MotionEvent, PointerGrab, RelativeMotionEvent,
};
use smithay::input::touch::{
    DownEvent, GrabStartData as TouchGrabStartData, MotionEvent as TouchMotionEvent, UpEvent,
};
use smithay::input::SeatHandler;
use smithay::output::Output;
use smithay::reexports::wayland_server::protocol::wl_data_source::WlDataSource;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::utils::{Logical, Point, Rectangle, Transform, SERIAL_COUNTER};
use smithay::wayland::keyboard_shortcuts_inhibit::KeyboardShortcutsInhibitor;
use smithay::wayland::pointer_constraints::{with_pointer_constraint, PointerConstraint};
use smithay::wayland::tablet_manager::{TabletDescriptor, TabletSeatTrait};
use touch_overview_grab::TouchOverviewGrab;

use self::move_grab::MoveGrab;
use self::pick_color_grab::PickColorGrab;
use self::pick_window_grab::PickWindowGrab;
use self::resize_grab::ResizeGrab;
use self::spatial_movement_grab::SpatialMovementGrab;
#[cfg(feature = "dbus")]
use crate::dbus::freedesktop_a11y::KbMonBlock;
use crate::ipc::server::role_title_to_tag_and_clean;
use crate::layout::activity::ActivityId;
use crate::layout::bookmarks::{
    AnchorWire, BookmarkJumpOutcome, BookmarkKey, BookmarkKeyError, BookmarkName, BookmarkRule,
    WalkDirection,
};
use crate::layout::scrolling::ScrollDirection;
use crate::layout::{
    ActivateWindow, DoActionError, DoActionOutcome, LayoutElement, MoveWindowToPoolOutcome,
};
use crate::niri::{CastTarget, PointerVisibility, State};
use crate::ui::bookmark_switcher::{ModeCommand, PressOutcome};
use crate::ui::confirm_dialog::ConfirmRequest;
use crate::ui::mru::{WindowMru, WindowMruUi};
use crate::ui::screenshot_ui::ScreenshotUi;
use crate::utils::spawning::{spawn, spawn_sh};
use crate::utils::{center, get_monotonic_time, with_toplevel_role, CastSessionId, ResizeEdge};

pub mod backend_ext;
pub mod mod_tap;
pub mod move_grab;
pub mod pick_color_grab;
pub mod pick_window_grab;
pub mod resize_grab;
pub mod scroll_swipe_gesture;
pub mod scroll_tracker;
pub mod spatial_movement_grab;
pub mod swipe_tracker;
pub mod touch_overview_grab;
pub mod touch_resize_grab;

use backend_ext::{NiriInputBackend as InputBackend, NiriInputDevice as _};

pub const DOUBLE_CLICK_TIME: Duration = Duration::from_millis(400);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TabletData {
    pub aspect_ratio: f64,
}

pub enum PointerOrTouchStartData<D: SeatHandler> {
    Pointer(PointerGrabStartData<D>),
    Touch(TouchGrabStartData<D>),
}

impl<D: SeatHandler> PointerOrTouchStartData<D> {
    pub fn location(&self) -> Point<f64, Logical> {
        match self {
            PointerOrTouchStartData::Pointer(x) => x.location,
            PointerOrTouchStartData::Touch(x) => x.location,
        }
    }

    pub fn unwrap_pointer(&self) -> &PointerGrabStartData<D> {
        match self {
            PointerOrTouchStartData::Pointer(x) => x,
            PointerOrTouchStartData::Touch(_) => panic!("start_data is not Pointer"),
        }
    }

    pub fn unwrap_touch(&self) -> &TouchGrabStartData<D> {
        match self {
            PointerOrTouchStartData::Pointer(_) => panic!("start_data is not Touch"),
            PointerOrTouchStartData::Touch(x) => x,
        }
    }

    pub fn is_pointer(&self) -> bool {
        matches!(self, Self::Pointer(_))
    }

    pub fn is_touch(&self) -> bool {
        matches!(self, Self::Touch(_))
    }
}

impl State {
    pub fn process_input_event<I: InputBackend + 'static>(&mut self, event: InputEvent<I>)
    where
        I::Device: 'static, // Needed for downcasting.
    {
        let _span = tracy_client::span!("process_input_event");

        let latency_warn = crate::niri::latency_warn_threshold();
        let ev_start = latency_warn.map(|_| Instant::now());
        let ev_label = ev_start.map(|_| input_event_label(&event));

        // Make sure some logic like workspace clean-up has a chance to run before doing actions.
        let anim_start = ev_start.map(|_| Instant::now());
        self.niri.advance_animations();
        let anim_elapsed = anim_start.map(|s| s.elapsed());

        if self.niri.monitors_active {
            // Notify the idle-notifier of activity.
            if should_notify_activity(&event) {
                self.niri.notify_activity();
            }
        } else {
            // Power on monitors if they were off.
            if should_activate_monitors(&event) {
                self.niri.activate_monitors(&mut self.backend);

                // Notify the idle-notifier of activity only if we're also powering on the
                // monitors.
                self.niri.notify_activity();
            }
        }

        if should_reset_pointer_inactivity_timer(&event) {
            self.niri.reset_pointer_inactivity_timer();
        }

        if should_disarm_mod_tap(&event) {
            self.niri.mod_tap.disarm();
        }

        // Each pair below couples its own event predicate; there is no
        // priority ordering or shared membership list to collapse through
        // `crate::niri::Niri::active_modal` here, and the hotkey overlay
        // isn't a `crate::niri::ModalKind` at all. See that type for the
        // modal-overlay priority/membership table.
        let hide_hotkey_overlay =
            self.niri.hotkey_overlay.is_open() && should_hide_hotkey_overlay(&event);

        let hide_confirm_dialog =
            self.niri.confirm_dialog.is_open() && should_hide_confirm_dialog(&event);

        let hide_bookmark_switcher =
            self.niri.bookmark_switcher.is_open() && should_hide_bookmark_switcher(&event);

        let mut consumed_by_a11y = false;
        use InputEvent::*;
        match event {
            DeviceAdded { device } => self.on_device_added(device),
            DeviceRemoved { device } => self.on_device_removed(device),
            Keyboard { event } => self.on_keyboard::<I>(event, &mut consumed_by_a11y),
            PointerMotion { event } => self.on_pointer_motion::<I>(event),
            PointerMotionAbsolute { event } => self.on_pointer_motion_absolute::<I>(event),
            PointerButton { event } => self.on_pointer_button::<I>(event),
            PointerAxis { event } => self.on_pointer_axis::<I>(event),
            TabletToolAxis { event } => self.on_tablet_tool_axis::<I>(event),
            TabletToolTip { event } => self.on_tablet_tool_tip::<I>(event),
            TabletToolProximity { event } => self.on_tablet_tool_proximity::<I>(event),
            TabletToolButton { event } => self.on_tablet_tool_button::<I>(event),
            GestureSwipeBegin { event } => self.on_gesture_swipe_begin::<I>(event),
            GestureSwipeUpdate { event } => self.on_gesture_swipe_update::<I>(event),
            GestureSwipeEnd { event } => self.on_gesture_swipe_end::<I>(event),
            GesturePinchBegin { event } => self.on_gesture_pinch_begin::<I>(event),
            GesturePinchUpdate { event } => self.on_gesture_pinch_update::<I>(event),
            GesturePinchEnd { event } => self.on_gesture_pinch_end::<I>(event),
            GestureHoldBegin { event } => self.on_gesture_hold_begin::<I>(event),
            GestureHoldEnd { event } => self.on_gesture_hold_end::<I>(event),
            TouchDown { event } => self.on_touch_down::<I>(event),
            TouchMotion { event } => self.on_touch_motion::<I>(event),
            TouchUp { event } => self.on_touch_up::<I>(event),
            TouchCancel { event } => self.on_touch_cancel::<I>(event),
            TouchFrame { event } => self.on_touch_frame::<I>(event),
            SwitchToggle { event } => self.on_switch_toggle::<I>(event),
            Special(_) => (),
        }

        if let (Some(start), Some(anim), Some(label), Some(threshold)) =
            (ev_start, anim_elapsed, ev_label, latency_warn)
        {
            let total = start.elapsed();
            if total >= threshold {
                let (pool, active_view, disconnected) = self.niri.layout.latency_debug_counts();
                warn!(
                    "input latency: {label} took {total:?} (advance_animations {anim:?}); \
                     pool={pool} active_view={active_view} disconnected={disconnected}"
                );
            }
        }

        // Don't hide overlays if consumed by a11y, so that you can use the screen reader
        // navigation keys.
        if consumed_by_a11y {
            return;
        }

        // Do this last so that screenshot still gets it.
        if hide_hotkey_overlay && self.niri.hotkey_overlay.hide() {
            self.niri.queue_redraw_all();
        }

        if hide_confirm_dialog && self.niri.confirm_dialog.hide() {
            self.niri.queue_redraw_all();
        }

        if hide_bookmark_switcher {
            self.niri.bookmark_switcher.close();
            self.niri.queue_redraw_all();
        }
    }

    pub fn process_libinput_event(&mut self, event: &mut InputEvent<LibinputInputBackend>) {
        let _span = tracy_client::span!("process_libinput_event");

        match event {
            InputEvent::DeviceAdded { device } => {
                self.niri.devices.insert(device.clone());

                if device.has_capability(input::DeviceCapability::TabletTool) {
                    match device.size() {
                        Some((w, h)) => {
                            let aspect_ratio = w / h;
                            let data = TabletData { aspect_ratio };
                            self.niri.tablets.insert(device.clone(), data);
                        }
                        None => {
                            warn!("tablet tool device has no size");
                        }
                    }
                }

                if device.has_capability(input::DeviceCapability::Keyboard) {
                    if let Some(led_state) = self
                        .niri
                        .seat
                        .get_keyboard()
                        .map(|keyboard| keyboard.led_state())
                    {
                        device.led_update(led_state.into());
                    }
                }

                if device.has_capability(input::DeviceCapability::Touch) {
                    self.niri.touch.insert(device.clone());
                }

                apply_libinput_settings(&self.niri.config.borrow().input, device);
            }
            InputEvent::DeviceRemoved { device } => {
                self.niri.touch.remove(device);
                self.niri.tablets.remove(device);
                self.niri.devices.remove(device);
            }
            _ => (),
        }
    }

    fn on_device_added(&mut self, device: impl Device) {
        if device.has_capability(DeviceCapability::TabletTool) {
            let tablet_seat = self.niri.seat.tablet_seat();

            let desc = TabletDescriptor::from(&device);
            tablet_seat.add_tablet::<Self>(&self.niri.display_handle, &desc);
        }
        if device.has_capability(DeviceCapability::Touch) && self.niri.seat.get_touch().is_none() {
            self.niri.seat.add_touch();
        }
    }

    fn on_device_removed(&mut self, device: impl Device) {
        if device.has_capability(DeviceCapability::TabletTool) {
            let tablet_seat = self.niri.seat.tablet_seat();

            let desc = TabletDescriptor::from(&device);
            tablet_seat.remove_tablet(&desc);

            // If there are no tablets in seat we can remove all tools
            if tablet_seat.count_tablets() == 0 {
                tablet_seat.clear_tools();
            }
        }
        if device.has_capability(DeviceCapability::Touch) && self.niri.touch.is_empty() {
            self.niri.seat.remove_touch();
        }
    }

    /// Computes the rectangle that covers all outputs in global space.
    fn global_bounding_rectangle(&self) -> Option<Rectangle<i32, Logical>> {
        self.niri.global_space.outputs().fold(
            None,
            |acc: Option<Rectangle<i32, Logical>>, output| {
                self.niri
                    .global_space
                    .output_geometry(output)
                    .map(|geo| acc.map(|acc| acc.merge(geo)).unwrap_or(geo))
            },
        )
    }

    /// Computes the cursor position for the tablet event.
    ///
    /// This function handles the tablet output mapping, as well as coordinate clamping and aspect
    /// ratio correction.
    fn compute_tablet_position<I: InputBackend>(
        &self,
        event: &(impl Event<I> + TabletToolEvent<I>),
    ) -> Option<Point<f64, Logical>>
    where
        I::Device: 'static,
    {
        let device_output = event.device().output(self);
        let device_output = device_output.filter(|output| self.niri.output_exists(output));
        let device_output = device_output.as_ref();
        let mapped_output = device_output.or_else(|| self.niri.output_for_tablet());

        // If the tablet is configured to map to the focused window, use that window's geometry on
        // the mapped output (or on the focused output if no specific output is mapped).
        let map_to_focused_window = self.niri.config.borrow().input.tablet.map_to_focused_window;
        // But only if the keyboard focus is on the layout, so that it doesn't trigger on the lock
        // screen and such.
        let window_target = if map_to_focused_window && self.niri.keyboard_focus.is_layout() {
            let output = mapped_output.or_else(|| self.niri.layout.active_output());
            output.and_then(|output| {
                let layout = &self.niri.layout;
                let monitor = layout.monitor_for_output(output)?;
                let view = layout.active_view(&monitor.output_id());
                let pool = layout.workspace_pool();
                let mut rect = monitor.active_window_visual_rectangle(pool, view)?;
                let output_geo = self.niri.global_space.output_geometry(output)?;
                rect.loc += output_geo.loc.to_f64();
                Some((rect, output))
            })
        } else {
            None
        };

        let (target_geo, keep_ratio, px, transform) = if let Some((rect, output)) = window_target {
            (
                rect,
                true,
                1. / output.current_scale().fractional_scale(),
                output.current_transform(),
            )
        } else if let Some(output) = mapped_output {
            let geo = self.niri.global_space.output_geometry(output).unwrap();
            (
                geo.to_f64(),
                true,
                1. / output.current_scale().fractional_scale(),
                output.current_transform(),
            )
        } else {
            let geo = self.global_bounding_rectangle()?.to_f64();

            // FIXME: this 1 px size should ideally somehow be computed for the rightmost output
            // corresponding to the position on the right when clamping.
            let output = self.niri.global_space.outputs().next().unwrap();
            let scale = output.current_scale().fractional_scale();

            // Do not keep ratio for the unified mode as this is what OpenTabletDriver expects.
            (geo, false, 1. / scale, Transform::Normal)
        };

        let mut pos = {
            let size = transform.invert().transform_size(target_geo.size);
            transform.transform_point_in(event.position_transformed(size.to_i32_round()), &size)
        };

        if keep_ratio {
            pos.x /= target_geo.size.w;
            pos.y /= target_geo.size.h;

            let device = event.device();
            if let Some(device) = (&device as &dyn Any).downcast_ref::<input::Device>() {
                if let Some(data) = self.niri.tablets.get(device) {
                    // This code does the same thing as mutter with "keep aspect ratio" enabled.
                    let size = transform.invert().transform_size(target_geo.size);
                    let output_aspect_ratio = size.w / size.h;
                    let ratio = data.aspect_ratio / output_aspect_ratio;

                    if ratio > 1. {
                        pos.x *= ratio;
                    } else {
                        pos.y /= ratio;
                    }
                }
            };

            pos.x *= target_geo.size.w;
            pos.y *= target_geo.size.h;
        }

        pos.x = pos.x.clamp(0.0, target_geo.size.w - px);
        pos.y = pos.y.clamp(0.0, target_geo.size.h - px);
        Some(pos + target_geo.loc)
    }

    fn is_inhibiting_shortcuts(&self) -> bool {
        self.niri
            .keyboard_focus
            .surface()
            .and_then(|surface| {
                self.niri
                    .keyboard_shortcuts_inhibiting_surfaces
                    .get(surface)
            })
            .is_some_and(KeyboardShortcutsInhibitor::is_active)
    }

    fn on_keyboard<I: InputBackend>(
        &mut self,
        event: I::KeyboardKeyEvent,
        consumed_by_a11y: &mut bool,
    ) {
        let mod_key = self.backend.mod_key(&self.niri.config.borrow());

        let serial = SERIAL_COUNTER.next_serial();
        let time = Event::time_msec(&event);
        let pressed = event.state() == KeyState::Pressed;
        let key_code = event.key_code();

        // Hoisted so the mod-tap tracking below and the `.input()` call further down share one
        // handle. `pressed_keys()` must be read before `.input()`, which is what applies this
        // event to the keyboard's internal state.
        let keyboard = self.niri.seat.get_keyboard().unwrap();

        // Track mod-tap arm/disarm before anything below (accessibility grabs, modal overlays)
        // gets a chance to intercept this event, so a press or release that a11y/modal consumes
        // still disarms/fires correctly. See `mod_tap` module docs for the full state machine.
        let other_keys_held;
        let tap_fire_candidate;
        if pressed {
            self.niri.mod_tap.on_key_press(key_code);
            other_keys_held = !keyboard.pressed_keys().is_empty();
            tap_fire_candidate = false;
        } else {
            other_keys_held = false;
            tap_fire_candidate = self.niri.mod_tap.on_key_release(key_code);
        }

        // Stop bind key repeat on any release. This won't work 100% correctly in cases like:
        // 1. Press Mod
        // 2. Press Left (repeat starts)
        // 3. Press PgDown (new repeat starts)
        // 4. Release Left (PgDown repeat stops)
        // But it's good enough for now.
        // FIXME: handle this properly.
        if !pressed {
            if let Some(token) = self.niri.bind_repeat_timer.take() {
                self.niri.event_loop.remove(token);
            }
        }

        if pressed {
            self.hide_cursor_if_needed();
        }

        let is_inhibiting_shortcuts = self.is_inhibiting_shortcuts();

        // Accessibility modifier grabs should override XKB state changes (e.g. Caps Lock), so we
        // need to process them before keyboard.input() below.
        //
        // Other accessibility-grabbed keys should still update our XKB state, but not cause any
        // other changes.
        #[cfg(feature = "dbus")]
        let block = {
            let block = self.a11y_process_key(
                Duration::from_millis(u64::from(time)),
                event.key_code(),
                event.state(),
            );
            if block != KbMonBlock::Pass {
                *consumed_by_a11y = true;
            }
            // The accessibility modifier first press must not change XKB state, so we return
            // early here.
            if block == KbMonBlock::ModifierFirstPress {
                return;
            }
            block
        };
        #[cfg(not(feature = "dbus"))]
        let _ = consumed_by_a11y;

        let mut mod_tap_bind: Option<Bind> = None;

        let bind_result = keyboard.input(
            self,
            key_code,
            event.state(),
            serial,
            time,
            |this, mods, keysym| {
                let key_code = event.key_code();
                let modified = keysym.modified_sym();
                let raw = keysym.raw_latin_sym_or_raw_current_sym();
                let modifiers = modifiers_from_state(*mods);

                // After updating XKB state from accessibility-grabbed keys, return right away and
                // don't handle them.
                #[cfg(feature = "dbus")]
                if block != KbMonBlock::Pass {
                    // HACK: there's a slight problem with this code. Here we filter out keys
                    // consumed by accessibility from getting sent to the Wayland client. However,
                    // the Wayland client can still receive these keys from the wl_keyboard
                    // enter/modifiers events. In particular, this can easily happen when opening
                    // the Orca actions menu with Orca + Shift + A: in most cases, when this menu
                    // opens, Shift is still held down, so the menu receives it in
                    // wl_keyboard.enter/modifiers. Then the menu won't react to Enter presses
                    // until the user taps Shift again to "release" it (since the initial Shift
                    // release will be intercepted here).
                    //
                    // I don't think there's any good way of dealing with this apart from keeping a
                    // separate xkb state for accessibility, so that we can track the pressed
                    // modifiers without accidentally leaking them to wl_keyboard.enter. So for now
                    // let's forward modifier releases to the clients here to deal with the most
                    // common case.
                    if !pressed
                        && matches!(
                            modified,
                            Keysym::Shift_L
                                | Keysym::Shift_R
                                | Keysym::Control_L
                                | Keysym::Control_R
                                | Keysym::Super_L
                                | Keysym::Super_R
                                | Keysym::Alt_L
                                | Keysym::Alt_R
                        )
                    {
                        return FilterResult::Forward;
                    } else {
                        return FilterResult::Intercept(None);
                    }
                }

                if this.niri.confirm_dialog.is_open() && pressed {
                    if raw == Some(Keysym::Return) {
                        match this.niri.confirm_dialog.confirm() {
                            Some(ConfirmRequest::Exit) => {
                                info!("quitting after confirming exit dialog");
                                this.niri.stop_signal.stop();
                            }
                            Some(ConfirmRequest::RemoveBookmark { id }) => {
                                match this.niri.layout.remove_bookmark(Some(id)) {
                                    Ok(()) => (),
                                    // The bookmark may have been pruned
                                    // (window closed) while the dialog was
                                    // open; that degrades the confirm to a
                                    // cancel rather than an error.
                                    Err(err @ DoActionError::BookmarkNotFound { .. }) => {
                                        debug!(
                                            "remove_bookmark on confirm: {err:?}, treating as cancel"
                                        );
                                    }
                                    Err(err) => {
                                        error!("remove_bookmark on confirm: {err:?}");
                                    }
                                }
                            }
                            None => unreachable!(
                                "the dialog is open, so show() must have set a pending request"
                            ),
                        }
                    }

                    // Don't send this press to any clients.
                    this.niri.suppressed_keys.insert(key_code);
                    return FilterResult::Intercept(None);
                }

                // While the bookmark switcher is open (the standalone hint
                // overlay, leader mode, or incremental search) it owns every
                // key press. In the hint/leader states: a hint letter jumps,
                // a command letter (mode only) runs its command, everything
                // else dismisses. In search: a printable character extends
                // the query, Backspace trims it, Enter jumps to the top
                // match (or holds open with none), and only Esc dismisses —
                // an otherwise-unmatched key holds the overlay open rather
                // than dismissing. Releases fall through to the
                // suppressed-keys logic below so they are swallowed too. The
                // confirm dialog above wins if both somehow race.
                //
                // `raw` reports the base (layout-unshifted) keysym, so a
                // shifted hint/command letter still matches its base sym;
                // `chorded` (any modifier held) routes a shifted or
                // otherwise chorded letter to a dismiss rather than a
                // jump/command.
                if this.niri.bookmark_switcher.is_open() && pressed {
                    match this.niri.bookmark_switcher.press_outcome(raw, modifiers) {
                        PressOutcome::HoldOpen => {
                            // A pure modifier keeps the overlay open (you
                            // might be reaching for a chord, or just resting
                            // a finger).
                            this.niri.suppressed_keys.insert(key_code);
                            return FilterResult::Intercept(None);
                        }
                        PressOutcome::Jump(id) => {
                            // Reuse the full jump arm (post-jump bookkeeping
                            // and all). Modality only suppresses input, not
                            // client activity: if the hinted window has
                            // closed since the overlay opened,
                            // `Layout::remove_window` already pruned its
                            // bookmark, so this id is stale and the jump
                            // yields `Err(BookmarkNotFound)`, discarded here
                            // (matching the MRU dispatch precedent) — a
                            // user-visible no-op, though the overlay still
                            // dismisses below. `press_outcome` debug-logs the
                            // matched bookmark id on every jump outcome
                            // (hint, mode, or search) so that no-op is
                            // diagnosable.
                            this.do_action(Action::JumpToBookmark(id), false);
                            this.niri.bookmark_switcher.close();
                            this.niri.queue_redraw_all();
                        }
                        PressOutcome::Command { cmd, sticky } => {
                            // Close (and redraw) before dispatching: `Add`
                            // (on a repress under the `remove` policy) and
                            // `RemoveFocused` both show the confirm dialog,
                            // which must not open behind a still-open
                            // switcher — an open switcher would otherwise
                            // keep intercepting keys ahead of the dialog,
                            // eating its confirm/cancel keystrokes. There is
                            // no headless harness for this input path (no
                            // `TestInputBackend` in this codebase), so this
                            // ordering isn't regression-pinned by a test —
                            // reordering it would silently reintroduce a
                            // switcher-eats-dialog-keys bug that manual
                            // testing might not catch immediately. When
                            // `sticky` (`bookmarks.mode-sticky`), the same
                            // discipline extends to the reopen below: it must
                            // happen strictly after `do_action` and only if
                            // no modal overlay (e.g. the confirm dialog this
                            // very command may have just opened) claimed
                            // keyboard focus in the meantime.
                            this.niri.bookmark_switcher.close();
                            this.niri.queue_redraw_all();
                            let action = match cmd {
                                ModeCommand::Add => Action::AddBookmark,
                                ModeCommand::RemoveFocused => Action::RemoveBookmark(false),
                                ModeCommand::WalkBackward => Action::WalkBookmarksBackward,
                                ModeCommand::WalkForward => Action::WalkBookmarksForward,
                            };
                            debug!(
                                "bookmark mode: command {cmd:?} matched, dispatching {action:?}"
                            );
                            this.do_action(action, false);
                            if sticky {
                                if this.niri.modal_overlay_blocks_bookmark_overlay() {
                                    debug!(
                                        "bookmark mode: sticky reopen skipped, a modal overlay \
                                         is open"
                                    );
                                } else {
                                    // Scope the config borrow to the open
                                    // call: the `RefCell` guard must be
                                    // dropped before `queue_redraw_all()`
                                    // below borrows `self.niri` again. This
                                    // resolves the *current* config, i.e. a
                                    // mid-open reload that flips
                                    // `mode-sticky` applies starting with
                                    // this reopened instance, not the one
                                    // that just dispatched.
                                    let opened = {
                                        let config = this.niri.config.borrow();
                                        this.niri
                                            .bookmark_switcher
                                            .open_mode(&this.niri.layout, &config.bookmarks)
                                    };
                                    if opened {
                                        this.niri.queue_redraw_all();
                                    }
                                    // A `false` return needs no extra
                                    // logging: `open_mode` only refuses when
                                    // the command sheet itself fails to
                                    // rasterise, and it already warns
                                    // internally on that path.
                                }
                            }
                        }
                        PressOutcome::KeyCandidate { bookmark_id, key } => {
                            // Validate through the exact same policy the
                            // typed `AssignBookmarkKey` path uses — see
                            // `validate_bookmark_key_candidate`'s doc for why
                            // this must be the only place either path checks
                            // collisions. This arm's dispatch is as
                            // input-seam-untestable as the `Command` arm
                            // above (no headless harness for this input
                            // path) — see that arm's comment.
                            let mod_key = this.backend.mod_key(&this.niri.config.borrow());
                            let candidate = {
                                let config = this.niri.config.borrow();
                                validate_bookmark_key_candidate(
                                    key,
                                    bookmark_id,
                                    config
                                        .binds
                                        .0
                                        .iter()
                                        .chain(config.recent_windows.binds.iter()),
                                    this.niri
                                        .layout
                                        .bookmarks()
                                        .list()
                                        .iter()
                                        .filter_map(|b| b.key().map(|k| (b.id().get(), k.key()))),
                                    mod_key,
                                )
                            };
                            match candidate {
                                Ok(bookmark_key) => match this
                                    .niri
                                    .layout
                                    .assign_bookmark_key(bookmark_id, bookmark_key)
                                {
                                    Ok(()) => {
                                        this.niri.bookmark_switcher.close();
                                    }
                                    Err(DoActionError::BookmarkNotFound { id }) => {
                                        // The bookmark was pruned (its window
                                        // closed) while the capture prompt was
                                        // open; tolerated as a cancel rather
                                        // than an error, the same discipline
                                        // the confirm-dialog `RemoveBookmark`
                                        // arm above applies.
                                        debug!(
                                            "capture-bookmark-key: bookmark {id} pruned while \
                                             capturing, treating as cancel"
                                        );
                                        this.niri.bookmark_switcher.close();
                                    }
                                    Err(DoActionError::BookmarkKeyCollision { key }) => {
                                        // The sibling-collision snapshot the
                                        // validator just read could only go
                                        // stale via reentrancy this
                                        // single-threaded dispatch path
                                        // doesn't have; handled explicitly
                                        // rather than assumed unreachable.
                                        debug!(
                                            "capture-bookmark-key: commit reported collision on \
                                             {key}, staying open"
                                        );
                                        this.niri
                                            .bookmark_switcher
                                            .capture_rejected("already bound to another bookmark");
                                    }
                                    Err(err) => unreachable!(
                                        "assign_bookmark_key only returns BookmarkNotFound or \
                                         BookmarkKeyCollision: {err:?}"
                                    ),
                                },
                                // Matched exhaustively on the typed
                                // `BookmarkKeyError` (rather than a wildcard)
                                // so a future keysym-reachable variant fails
                                // to compile here instead of silently falling
                                // into the wrong prompt text.
                                Err(BookmarkKeyRejection::Invalid(BookmarkKeyError::NoModifiers)) => {
                                    this.niri
                                        .bookmark_switcher
                                        .capture_rejected("needs a modifier");
                                }
                                Err(BookmarkKeyRejection::Invalid(BookmarkKeyError::NotAKeysym)) => {
                                    // The interactive capture surface always
                                    // builds a keysym candidate from a live
                                    // keyboard press; only the typed
                                    // `AssignBookmarkKey` path can name a
                                    // non-keysym trigger via a parsed string.
                                    unreachable!(
                                        "capture-bookmark-key candidate is always a keysym trigger"
                                    )
                                }
                                Err(BookmarkKeyRejection::Collision {
                                    with: BookmarkKeyCollidee::ConfigBind,
                                    ..
                                }) => {
                                    this.niri
                                        .bookmark_switcher
                                        .capture_rejected("already bound to a keybind");
                                }
                                Err(BookmarkKeyRejection::Collision {
                                    with: BookmarkKeyCollidee::SiblingBookmark,
                                    ..
                                }) => {
                                    this.niri
                                        .bookmark_switcher
                                        .capture_rejected("already bound to another bookmark");
                                }
                            }
                            this.niri.queue_redraw_all();
                        }
                        PressOutcome::Dismiss => {
                            this.niri.bookmark_switcher.close();
                            this.niri.queue_redraw_all();
                        }
                        PressOutcome::SearchUpdated => {
                            // Entering search or editing the query mutated the
                            // overlay in place; it stays open, so just redraw.
                            this.niri.queue_redraw_all();
                        }
                    }
                    this.niri.suppressed_keys.insert(key_code);
                    return FilterResult::Intercept(None);
                }

                // Check if all modifiers were released while the MRU UI was open. If so, close the
                // UI (which will also transfer the focus to the current MRU UI selection).
                if this.niri.window_mru_ui.is_open() && !pressed && modifiers.is_empty() {
                    this.do_action(Action::MruConfirm, false);

                    if this.niri.suppressed_keys.remove(&key_code) {
                        return FilterResult::Intercept(None);
                    } else {
                        return FilterResult::Forward;
                    }
                }

                if pressed && raw == Some(Keysym::Escape) {
                    // Cancel certain grabs on Escape.
                    let pointer = this.niri.seat.get_pointer().unwrap();
                    if pointer
                        .with_grab(|_, grab| Self::grab_can_be_cancelled_with_esc(grab))
                        .unwrap_or(false)
                    {
                        pointer.unset_grab(this, serial, time);
                        this.niri.suppressed_keys.insert(key_code);
                        return FilterResult::Intercept(None);
                    }
                }

                if let Some(Keysym::space) = raw {
                    this.niri.screenshot_ui.set_space_down(pressed);
                }

                let res = {
                    let config = this.niri.config.borrow();
                    let bindings =
                        make_binds_iter(&config, &mut this.niri.window_mru_ui, modifiers, &this.niri.bookmark_binds);

                    should_intercept_key(
                        &mut this.niri.suppressed_keys,
                        bindings,
                        mod_key,
                        key_code,
                        modified,
                        raw,
                        pressed,
                        *mods,
                        &this.niri.screenshot_ui,
                        this.niri.config.borrow().input.disable_power_key_handling,
                        is_inhibiting_shortcuts,
                    )
                };

                if matches!(res, FilterResult::Forward) {
                    // If we didn't find any bind, try other hardcoded keys.
                    if this.niri.keyboard_focus.is_overview() && pressed {
                        if let Some(bind) = raw.and_then(|raw| hardcoded_overview_bind(raw, *mods))
                        {
                            this.niri.suppressed_keys.insert(key_code);
                            return FilterResult::Intercept(Some(bind));
                        }
                    }

                    if pressed {
                        if mod_tap::is_bare_mod_press(mod_key, raw, modifiers, other_keys_held) {
                            this.niri.mod_tap.arm(key_code);
                        }
                    } else if tap_fire_candidate {
                        // A fire candidate implies the armed key's own release just emptied the
                        // modifier state (nothing else was pressed since arming) — implication
                        // only, not a biconditional, since an unrelated already-empty release can
                        // reach this arm too.
                        debug_assert!(
                            modifiers.is_empty(),
                            "mod-tap fire candidate with non-empty modifiers: {modifiers:?}"
                        );

                        // No `Key { trigger: Trigger::ModTap, .. }` literal is synthesized here;
                        // this is the same canonical lookup the wheel/touchpad triggers use, so a
                        // future non-empty-modifiers literal elsewhere can't silently shadow it.
                        let candidate = {
                            let config = this.niri.config.borrow();
                            find_configured_bind(&config.binds.0, mod_key, Trigger::ModTap, *mods)
                        };

                        if let Some(bind) = candidate {
                            if mod_tap::tap_fire_allowed(
                                this.niri.active_modal(),
                                &bind.action,
                                is_inhibiting_shortcuts,
                                bind.allow_inhibiting,
                            ) {
                                mod_tap_bind = Some(bind);
                            }
                        }
                    }

                    // Interaction with the active window, immediately update the active window's
                    // focus timestamp without waiting for a possible pending MRU lock-in delay.
                    this.niri.mru_apply_keyboard_commit();
                }

                res
            },
        );

        // Dispatched here, outside the filter closure, so the client has already received the
        // forwarded release via `input_forward` before the bound action runs. Checked before the
        // existing `Some(Some(bind))` handling below: a mod-tap fire is a release-time event, so
        // `bind_result` itself is never `Some(Some(_))` on this path (only a press can produce
        // that from `should_intercept_key`).
        if let Some(bind) = mod_tap_bind {
            self.handle_bind(bind);
            return;
        }

        let Some(Some(bind)) = bind_result else {
            return;
        };

        if !pressed {
            return;
        }

        self.handle_bind(bind.clone());

        self.start_key_repeat(bind);
    }

    fn start_key_repeat(&mut self, bind: Bind) {
        if !bind.repeat {
            return;
        }

        // Stop the previous key repeat if any.
        if let Some(token) = self.niri.bind_repeat_timer.take() {
            self.niri.event_loop.remove(token);
        }

        let config = self.niri.config.borrow();
        let config = &config.input.keyboard;

        let repeat_rate = config.repeat_rate;
        if repeat_rate == 0 {
            return;
        }
        let repeat_duration = Duration::from_secs_f64(1. / f64::from(repeat_rate));

        let repeat_timer =
            Timer::from_duration(Duration::from_millis(u64::from(config.repeat_delay)));

        let token = self
            .niri
            .event_loop
            .insert_source(repeat_timer, move |_, _, state| {
                state.handle_bind(bind.clone());
                TimeoutAction::ToDuration(repeat_duration)
            })
            .unwrap();

        self.niri.bind_repeat_timer = Some(token);
    }

    fn hide_cursor_if_needed(&mut self) {
        // If the pointer is already invisible, don't reset it back to Hidden causing one frame
        // of hover.
        if !self.niri.pointer_visibility.is_visible() {
            return;
        }

        if !self.niri.config.borrow().cursor.hide_when_typing {
            return;
        }

        // jiji keeps this set only while actively using a tablet, which means the cursor position
        // is likely to change almost immediately, causing pointer_visibility to just flicker back
        // and forth.
        if self.niri.tablet_cursor_location.is_some() {
            return;
        }

        self.niri.pointer_visibility = PointerVisibility::Hidden;
        self.niri.queue_redraw_all();
    }

    pub fn handle_bind(&mut self, bind: Bind) {
        let Some(cooldown) = bind.cooldown else {
            self.do_action(bind.action, bind.allow_when_locked);
            return;
        };

        // Check this first so that it doesn't trigger the cooldown.
        if self.niri.is_locked() && !(bind.allow_when_locked || allowed_when_locked(&bind.action)) {
            return;
        }

        match self.niri.bind_cooldown_timers.entry(bind.key) {
            // The bind is on cooldown.
            Entry::Occupied(_) => (),
            Entry::Vacant(entry) => {
                let timer = Timer::from_duration(cooldown);
                let token = self
                    .niri
                    .event_loop
                    .insert_source(timer, move |_, _, state| {
                        if state.niri.bind_cooldown_timers.remove(&bind.key).is_none() {
                            error!("bind cooldown timer entry disappeared");
                        }
                        TimeoutAction::Drop
                    })
                    .unwrap();
                entry.insert(token);

                self.do_action(bind.action, bind.allow_when_locked);
            }
        }
    }

    /// Silent-drop wrapper over [`Self::do_action_inner`].
    ///
    /// Keybinding-triggered actions (and other internal callers like switch
    /// events) inherit the contract: every [`DoActionError`] is dropped here
    /// on purpose, whatever its [`crate::layout::Disposition`] —
    /// [`crate::layout::Disposition::Park`] and
    /// [`crate::layout::Disposition::Terminal`] alike. The disposition
    /// classification only matters to the IPC drain-walk, which parks `Park`
    /// errors on a per-connection waiter queue and signals `Terminal` ones
    /// immediately; a keybinding-triggered call has no such waiter to park or
    /// signal. The keypress is discarded, not queued, and no error is
    /// surfaced to the user. Callers that can surface the `Err` to a client
    /// (e.g., the IPC `Request::Action` dispatch) call `do_action_inner`
    /// directly.
    ///
    /// The successful [`DoActionOutcome`] payload (`Handled` and `NoOp(reason)`
    /// alike) is also dropped here. A `NoOp(...)` breadcrumb produced via a
    /// keybinding-triggered path is not observable to any IPC consumer; the
    /// user already sees the on-screen effect (or lack thereof) directly. Only
    /// callers that go through the IPC dispatch site benefit from the typed
    /// reply.
    pub fn do_action(&mut self, action: Action, allow_when_locked: bool) {
        let _ = self.do_action_inner(action, allow_when_locked);
    }

    #[must_use = "IPC dispatch sites must surface this via the bounded channel; \
                  keybinding-path callers must discard explicitly via `let _ =` (see `do_action`)"]
    pub(crate) fn do_action_inner(
        &mut self,
        action: Action,
        allow_when_locked: bool,
    ) -> Result<DoActionOutcome, DoActionError> {
        if self.niri.is_locked() && !(allow_when_locked || allowed_when_locked(&action)) {
            return Ok(DoActionOutcome::Handled);
        }

        if let Some(touch) = self.niri.seat.get_touch() {
            touch.cancel(self);
        }

        match action {
            Action::Quit(skip_confirmation) => {
                if !skip_confirmation && self.niri.confirm_dialog.show(ConfirmRequest::Exit) {
                    self.niri.queue_redraw_all();
                    return Ok(DoActionOutcome::Handled);
                }

                info!("quitting as requested");
                self.niri.stop_signal.stop()
            }
            Action::ChangeVt(vt) => {
                self.backend.change_vt(vt);
                // Changing VT may not deliver the key releases, so clear the state.
                self.niri.suppressed_keys.clear();
                self.niri.mod_tap.disarm();
            }
            Action::Suspend => {
                self.backend.suspend();
                // Suspend may not deliver the key releases, so clear the state.
                self.niri.suppressed_keys.clear();
                self.niri.mod_tap.disarm();
            }
            Action::PowerOffMonitors => {
                self.niri.deactivate_monitors(&mut self.backend);
            }
            Action::PowerOnMonitors => {
                self.niri.activate_monitors(&mut self.backend);
            }
            Action::ToggleDebugTint => {
                self.backend.toggle_debug_tint();
                self.niri.queue_redraw_all();
            }
            Action::DebugToggleOpaqueRegions => {
                self.niri.debug_draw_opaque_regions = !self.niri.debug_draw_opaque_regions;
                self.niri.queue_redraw_all();
            }
            Action::DebugToggleDamage => {
                self.niri.debug_toggle_damage();
            }
            Action::Spawn(command) => {
                let (token, _) = self.niri.activation_state.create_external_token(None);
                spawn(command, Some(token.clone()));
            }
            Action::SpawnSh(command) => {
                let (token, _) = self.niri.activation_state.create_external_token(None);
                spawn_sh(command, Some(token.clone()));
            }
            Action::DoScreenTransition(delay_ms) => {
                self.backend.with_primary_renderer(|renderer| {
                    self.niri.do_screen_transition(renderer, delay_ms);
                });
            }
            Action::ScreenshotScreen(write_to_disk, show_pointer, path) => {
                let active = self.niri.layout.active_output().cloned();
                if let Some(active) = active {
                    self.backend.with_primary_renderer(|renderer| {
                        if let Err(err) = self.niri.screenshot(
                            renderer,
                            &active,
                            write_to_disk,
                            show_pointer,
                            path,
                        ) {
                            warn!("error taking screenshot: {err:?}");
                        }
                    });
                }
            }
            Action::ConfirmScreenshot { write_to_disk } => {
                self.confirm_screenshot(write_to_disk);
            }
            Action::CancelScreenshot => {
                if !self.niri.screenshot_ui.is_open() {
                    return Ok(DoActionOutcome::Handled);
                }

                self.niri.screenshot_ui.close();
                self.niri
                    .cursor_manager
                    .set_cursor_image(CursorImageStatus::default_named());
                self.niri.queue_redraw_all();
            }
            Action::ScreenshotTogglePointer => {
                self.niri.screenshot_ui.toggle_pointer();
                self.niri.queue_redraw_all();
            }
            Action::Screenshot(show_cursor, path) => {
                self.open_screenshot_ui(show_cursor, path);
                self.niri.cancel_mru();
            }
            Action::ScreenshotWindow(write_to_disk, show_pointer, path) => {
                let focus = self.niri.layout.focus_with_output();
                if let Some((mapped, output)) = focus {
                    self.backend.with_primary_renderer(|renderer| {
                        if let Err(err) = self.niri.screenshot_window(
                            renderer,
                            output,
                            mapped,
                            write_to_disk,
                            show_pointer,
                            path,
                        ) {
                            warn!("error taking screenshot: {err:?}");
                        }
                    });
                }
            }
            Action::ScreenshotWindowById {
                id,
                write_to_disk,
                show_pointer,
                path,
            } => {
                // Widen the id lookup from `windows()` (active-view
                // scope) to `windows_all()` (pool-span) so hidden-activity
                // windows are reachable. A window with no live monitor backing
                // its bound output is a silent no-op (design call (a) below).
                let window = self
                    .niri
                    .layout
                    .windows_all()
                    .find(|(_, m)| m.id().get() == id);
                if let Some((oid_opt, mapped)) = window {
                    // Resolve through the layout rather than using the pool
                    // workspace's `OutputId` alone: screenshotting needs the
                    // renderer's live `Output`, which only exists on a
                    // connected monitor.
                    let monitor =
                        oid_opt.and_then(|oid| self.niri.layout.monitor_for_output_id(oid));
                    if let Some(monitor) = monitor {
                        // Hoist `output` before the closure for readability — mirrors
                        // the style used by `focus_with_output()` callers.
                        let output = monitor.output();
                        self.backend.with_primary_renderer(|renderer| {
                            if let Err(err) = self.niri.screenshot_window(
                                renderer,
                                output,
                                mapped,
                                write_to_disk,
                                show_pointer,
                                path,
                            ) {
                                warn!("error taking screenshot: {err:?}");
                            }
                        });
                    } else if let Some(oid) = oid_opt {
                        // Design call (a): silent no-op + `debug!` when the
                        // window's bound output is disconnected. Alternative
                        // (b) — falling back to the primary monitor — was
                        // rejected because scale/transform mismatch would
                        // silently produce a wrong-for-that-output render.
                        // A future reader tempted to flip to (b) must come
                        // back to this comment first.
                        debug!(
                            "screenshot_window: id={id} bound output {oid:?} not connected, \
                             no-op"
                        );
                    } else {
                        // In-flight interactive-move window for which `windows_all`
                        // could not resolve a bound output id — either no live
                        // monitor exists for the move's output (triggers a `warn!`
                        // inside `windows_all`), or no pool workspace is currently
                        // bound to it.
                        debug!(
                            "screenshot_window: id={id} has no bound output \
                             (interactive-move in flight), no-op"
                        );
                    }
                } else {
                    debug!("screenshot_window: id={id} not found in pool, no-op");
                }
            }
            Action::ToggleKeyboardShortcutsInhibit => {
                if let Some(inhibitor) = self.niri.keyboard_focus.surface().and_then(|surface| {
                    self.niri
                        .keyboard_shortcuts_inhibiting_surfaces
                        .get(surface)
                }) {
                    if inhibitor.is_active() {
                        inhibitor.inactivate();
                    } else {
                        inhibitor.activate();
                    }
                }
            }
            Action::CloseWindow => {
                if let Some(mapped) = self.niri.layout.focus() {
                    mapped.toplevel().send_close();
                }
            }
            Action::CloseWindowById(id) => {
                let window = self
                    .niri
                    .layout
                    .windows_all()
                    .find(|(_, m)| m.id().get() == id);
                if let Some((_, mapped)) = window {
                    mapped.toplevel().send_close();
                }
            }
            Action::FullscreenWindow => {
                let focus = self.niri.layout.focus().map(|m| m.window.clone());
                if let Some(window) = focus {
                    self.niri.layout.toggle_fullscreen(&window);
                    // FIXME: granular
                    self.niri.queue_redraw_all();
                }
            }
            Action::FullscreenWindowById(id) => {
                let window = self
                    .niri
                    .layout
                    .windows_all()
                    .find(|(_, m)| m.id().get() == id);
                let window = window.map(|(_, m)| m.window.clone());
                if let Some(window) = window {
                    self.niri.layout.toggle_fullscreen(&window);
                    // FIXME: granular
                    self.niri.queue_redraw_all();
                }
            }
            Action::ToggleWindowedFullscreen => {
                let focus = self.niri.layout.focus().map(|m| m.window.clone());
                if let Some(window) = focus {
                    self.niri.layout.toggle_windowed_fullscreen(&window);
                    // FIXME: granular
                    self.niri.queue_redraw_all();
                }
            }
            Action::ToggleWindowedFullscreenById(id) => {
                let window = self
                    .niri
                    .layout
                    .windows_all()
                    .find(|(_, m)| m.id().get() == id);
                let window = window.map(|(_, m)| m.window.clone());
                if let Some(window) = window {
                    self.niri.layout.toggle_windowed_fullscreen(&window);
                    // FIXME: granular
                    self.niri.queue_redraw_all();
                }
            }
            Action::FocusWindow(id) => {
                // `FocusWindow { id }` auto-switches the active
                // activity when the resolved window lives on a dormant
                // workspace. We walk `windows_all()` (pool-span) rather than
                // `windows()` (active-view scope) so hidden-activity windows
                // are reachable; the visibility fast-path below preserves the
                // pre-1b behavior for windows already on the active view.
                //
                // Missing id surfaces `Err(DoActionError::WindowNotFound)` —
                // the wire contract. IPC callers see the envelope
                // `"window not found: id={id}"`; keybinding callers
                // silent-drop via the `do_action` wrapper.
                let Some((_oid, mapped)) = self
                    .niri
                    .layout
                    .windows_all()
                    .find(|(_, m)| m.id().get() == id)
                else {
                    debug!("focus_window: id={id} not found in pool");
                    return Err(DoActionError::WindowNotFound { id });
                };
                let window = mapped.window.clone();
                let hint = mapped.get_last_focused_activity();

                let Some(ws_id) = self.niri.layout.window_ws_and_activity_hint(&window) else {
                    // Silent no-op — interactive-move window is not "no longer
                    // exists"; that branch is handled at the genuinely-unknown-id
                    // case above. Resolved via windows_all but not in the pool
                    // means the window is mid-flight and not owned by any pool
                    // workspace.
                    debug!("focus_window: id={id} resolved via windows_all but not in pool (interactive-move window), no-op");
                    return Ok(DoActionOutcome::Handled);
                };

                // Visibility fast-path: if `ws_id` appears in any of the
                // active activity's per-output `WorkspaceView::ids()`, the
                // window is already reachable under the current activity
                // cursor — just focus it. This matches the spec contract
                // "its workspace id is in the active activity's
                // `views[&output_id].ids`" without panicking when the window's
                // bound output is disconnected or when the window has no
                // bound output at all.
                let is_visible = self
                    .niri
                    .layout
                    .activities()
                    .active()
                    .views()
                    .values()
                    .any(|view| view.ids().contains(&ws_id));

                if is_visible {
                    self.focus_window(&window);
                    return Ok(DoActionOutcome::Handled);
                }

                // Hidden workspace — pick the activity to switch into. The
                // picker excludes the currently-active activity in all three
                // tiers; a `target == active_id()` here would mean the
                // workspace is only tagged with the active activity, which
                // contradicts `is_visible == false` and is handled as a silent
                // no-op.
                let target = self
                    .niri
                    .layout
                    .pick_activity_for_hidden_window(ws_id, hint);
                if target == self.niri.layout.active_activity_id() {
                    debug!("focus_window: id={id} — picker returned active_id (ws tagged only with active activity, contradicts is_visible==false), no-op");
                    return Ok(DoActionOutcome::Handled);
                }

                // Hard-block gate — mirrors the SwitchActivity /
                // RemoveActivity arms. Returning `Err(block)` preserves the
                // Part 2 IPC-queue semantics: the ipc-server parks the
                // action and re-dispatches on drain, so the client sees
                // `Handled` once the switch performs.
                if let Some(block) = self.niri.layout.is_activity_switch_hard_blocked() {
                    debug!("focus_window: activity switch hard-blocked by {block:?}");
                    return Err(block.into());
                }

                self.niri.layout.switch_activity(target);
                self.activity_switch_epilogue();

                // `focus_window` performs the actual activate + cursor warp
                // under the newly-active activity.
                self.focus_window(&window);
            }
            Action::FocusWindowInColumn(index) => {
                self.niri.layout.focus_window_in_column(index);
                self.maybe_warp_cursor_to_focus();
                self.niri.layer_shell_on_demand_focus = None;
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::FocusWindowPrevious => {
                let current = self.niri.layout.focus().map(|win| win.id());
                if let Some(window) = self
                    .niri
                    .layout
                    .windows()
                    .map(|(_, win)| win)
                    .filter(|win| Some(win.id()) != current)
                    .max_by_key(|win| win.get_focus_timestamp())
                    .map(|win| win.window.clone())
                {
                    // Commit current focus so repeated focus-window-previous works as expected.
                    self.niri.mru_apply_keyboard_commit();

                    self.focus_window(&window);
                }
            }
            Action::SwitchLayout(action) => {
                let keyboard = &self.niri.seat.get_keyboard().unwrap();
                keyboard.with_xkb_state(self, |mut state| match action {
                    LayoutSwitchTarget::Next => state.cycle_next_layout(),
                    LayoutSwitchTarget::Prev => state.cycle_prev_layout(),
                    LayoutSwitchTarget::Index(layout) => {
                        let num_layouts = state.xkb().lock().unwrap().layouts().count();
                        if usize::from(layout) >= num_layouts {
                            warn!("requested layout doesn't exist")
                        } else {
                            state.set_layout(Layout(layout.into()))
                        }
                    }
                });
            }
            Action::MoveColumnLeft => {
                if self.niri.screenshot_ui.is_open() {
                    self.niri.screenshot_ui.move_left();
                } else {
                    self.niri.layout.move_left();
                    self.maybe_warp_cursor_to_focus();
                }

                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::MoveColumnRight => {
                if self.niri.screenshot_ui.is_open() {
                    self.niri.screenshot_ui.move_right();
                } else {
                    self.niri.layout.move_right();
                    self.maybe_warp_cursor_to_focus();
                }

                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::MoveColumnToFirst => {
                self.niri.layout.move_column_to_first();
                self.maybe_warp_cursor_to_focus();
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::MoveColumnToLast => {
                self.niri.layout.move_column_to_last();
                self.maybe_warp_cursor_to_focus();
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::MoveColumnLeftOrToMonitorLeft => {
                if self.niri.screenshot_ui.is_open() {
                    self.niri.screenshot_ui.move_left();
                } else if let Some(output) = self.niri.output_left() {
                    if self.niri.layout.move_column_left_or_to_output(&output)
                        && !self.maybe_warp_cursor_to_focus_centered()
                    {
                        self.move_cursor_to_output(&output);
                    } else {
                        self.maybe_warp_cursor_to_focus();
                    }
                } else {
                    self.niri.layout.move_left();
                    self.maybe_warp_cursor_to_focus();
                }

                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::MoveColumnRightOrToMonitorRight => {
                if self.niri.screenshot_ui.is_open() {
                    self.niri.screenshot_ui.move_right();
                } else if let Some(output) = self.niri.output_right() {
                    if self.niri.layout.move_column_right_or_to_output(&output)
                        && !self.maybe_warp_cursor_to_focus_centered()
                    {
                        self.move_cursor_to_output(&output);
                    } else {
                        self.maybe_warp_cursor_to_focus();
                    }
                } else {
                    self.niri.layout.move_right();
                    self.maybe_warp_cursor_to_focus();
                }

                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::MoveWindowDown => {
                if self.niri.screenshot_ui.is_open() {
                    self.niri.screenshot_ui.move_down();
                } else {
                    self.niri.layout.move_down();
                    self.maybe_warp_cursor_to_focus();
                }

                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::MoveWindowUp => {
                if self.niri.screenshot_ui.is_open() {
                    self.niri.screenshot_ui.move_up();
                } else {
                    self.niri.layout.move_up();
                    self.maybe_warp_cursor_to_focus();
                }

                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::MoveWindowDownOrToWorkspaceDown => {
                if self.niri.screenshot_ui.is_open() {
                    self.niri.screenshot_ui.move_down();
                } else {
                    self.niri.layout.move_down_or_to_workspace_down();
                    self.maybe_warp_cursor_to_focus();
                }
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::MoveWindowUpOrToWorkspaceUp => {
                if self.niri.screenshot_ui.is_open() {
                    self.niri.screenshot_ui.move_up();
                } else {
                    self.niri.layout.move_up_or_to_workspace_up();
                    self.maybe_warp_cursor_to_focus();
                }
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::ConsumeOrExpelWindowLeft => {
                self.niri.layout.consume_or_expel_window_left(None);
                self.maybe_warp_cursor_to_focus();
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::ConsumeOrExpelWindowLeftById(id) => {
                let window = self
                    .niri
                    .layout
                    .windows_all()
                    .find(|(_, m)| m.id().get() == id);
                let window = window.map(|(_, m)| m.window.clone());
                if let Some(window) = window {
                    self.niri.layout.consume_or_expel_window_left(Some(&window));
                    self.maybe_warp_cursor_to_focus();
                    // FIXME: granular
                    self.niri.queue_redraw_all();
                }
            }
            Action::ConsumeOrExpelWindowRight => {
                self.niri.layout.consume_or_expel_window_right(None);
                self.maybe_warp_cursor_to_focus();
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::ConsumeOrExpelWindowRightById(id) => {
                let window = self
                    .niri
                    .layout
                    .windows_all()
                    .find(|(_, m)| m.id().get() == id);
                let window = window.map(|(_, m)| m.window.clone());
                if let Some(window) = window {
                    self.niri
                        .layout
                        .consume_or_expel_window_right(Some(&window));
                    self.maybe_warp_cursor_to_focus();
                    // FIXME: granular
                    self.niri.queue_redraw_all();
                }
            }
            Action::FocusColumnLeft => {
                self.niri.layout.focus_left();
                self.maybe_warp_cursor_to_focus();
                self.niri.layer_shell_on_demand_focus = None;
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::FocusColumnLeftUnderMouse => {
                if let Some((output, ws)) = self.niri.workspace_under_cursor(true) {
                    let ws_id = ws.id();
                    let ws = {
                        let mut workspaces = self.niri.layout.workspaces_mut();
                        workspaces.find(|ws| ws.id() == ws_id).unwrap()
                    };
                    ws.focus_left();
                    self.maybe_warp_cursor_to_focus();
                    self.niri.layer_shell_on_demand_focus = None;
                    self.niri.queue_redraw(&output);
                }
            }
            Action::FocusColumnRight => {
                self.niri.layout.focus_right();
                self.maybe_warp_cursor_to_focus();
                self.niri.layer_shell_on_demand_focus = None;
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::FocusColumnRightUnderMouse => {
                if let Some((output, ws)) = self.niri.workspace_under_cursor(true) {
                    let ws_id = ws.id();
                    let ws = {
                        let mut workspaces = self.niri.layout.workspaces_mut();
                        workspaces.find(|ws| ws.id() == ws_id).unwrap()
                    };
                    ws.focus_right();
                    self.maybe_warp_cursor_to_focus();
                    self.niri.layer_shell_on_demand_focus = None;
                    self.niri.queue_redraw(&output);
                }
            }
            Action::MoveViewLeft => {
                self.niri.layout.move_view_left();
                self.maybe_warp_cursor_to_focus();
                self.niri.layer_shell_on_demand_focus = None;
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::MoveViewRight => {
                self.niri.layout.move_view_right();
                self.maybe_warp_cursor_to_focus();
                self.niri.layer_shell_on_demand_focus = None;
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::FocusColumnFirst => {
                self.niri.layout.focus_column_first();
                self.maybe_warp_cursor_to_focus();
                self.niri.layer_shell_on_demand_focus = None;
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::FocusColumnLast => {
                self.niri.layout.focus_column_last();
                self.maybe_warp_cursor_to_focus();
                self.niri.layer_shell_on_demand_focus = None;
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::FocusColumnRightOrFirst => {
                self.niri.layout.focus_column_right_or_first();
                self.maybe_warp_cursor_to_focus();
                self.niri.layer_shell_on_demand_focus = None;
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::FocusColumnLeftOrLast => {
                self.niri.layout.focus_column_left_or_last();
                self.maybe_warp_cursor_to_focus();
                self.niri.layer_shell_on_demand_focus = None;
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::FocusColumn(index) => {
                self.niri.layout.focus_column(index);
                self.maybe_warp_cursor_to_focus();
                self.niri.layer_shell_on_demand_focus = None;
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::FocusWindowOrMonitorUp => {
                if let Some(output) = self.niri.output_up() {
                    if self.niri.layout.focus_window_up_or_output(&output)
                        && !self.maybe_warp_cursor_to_focus_centered()
                    {
                        self.move_cursor_to_output(&output);
                    } else {
                        self.maybe_warp_cursor_to_focus();
                    }
                } else {
                    self.niri.layout.focus_up();
                    self.maybe_warp_cursor_to_focus();
                }
                self.niri.layer_shell_on_demand_focus = None;

                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::FocusWindowOrMonitorDown => {
                if let Some(output) = self.niri.output_down() {
                    if self.niri.layout.focus_window_down_or_output(&output)
                        && !self.maybe_warp_cursor_to_focus_centered()
                    {
                        self.move_cursor_to_output(&output);
                    } else {
                        self.maybe_warp_cursor_to_focus();
                    }
                } else {
                    self.niri.layout.focus_down();
                    self.maybe_warp_cursor_to_focus();
                }
                self.niri.layer_shell_on_demand_focus = None;

                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::FocusColumnOrMonitorLeft => {
                if let Some(output) = self.niri.output_left() {
                    if self.niri.layout.focus_column_left_or_output(&output)
                        && !self.maybe_warp_cursor_to_focus_centered()
                    {
                        self.move_cursor_to_output(&output);
                    } else {
                        self.maybe_warp_cursor_to_focus();
                    }
                } else {
                    self.niri.layout.focus_left();
                    self.maybe_warp_cursor_to_focus();
                }
                self.niri.layer_shell_on_demand_focus = None;

                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::FocusColumnOrMonitorRight => {
                if let Some(output) = self.niri.output_right() {
                    if self.niri.layout.focus_column_right_or_output(&output)
                        && !self.maybe_warp_cursor_to_focus_centered()
                    {
                        self.move_cursor_to_output(&output);
                    } else {
                        self.maybe_warp_cursor_to_focus();
                    }
                } else {
                    self.niri.layout.focus_right();
                    self.maybe_warp_cursor_to_focus();
                }
                self.niri.layer_shell_on_demand_focus = None;

                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::FocusWindowDown => {
                self.niri.layout.focus_down();
                self.maybe_warp_cursor_to_focus();
                self.niri.layer_shell_on_demand_focus = None;
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::FocusWindowUp => {
                self.niri.layout.focus_up();
                self.maybe_warp_cursor_to_focus();
                self.niri.layer_shell_on_demand_focus = None;
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::FocusWindowDownOrColumnLeft => {
                self.niri.layout.focus_down_or_left();
                self.maybe_warp_cursor_to_focus();
                self.niri.layer_shell_on_demand_focus = None;
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::FocusWindowDownOrColumnRight => {
                self.niri.layout.focus_down_or_right();
                self.maybe_warp_cursor_to_focus();
                self.niri.layer_shell_on_demand_focus = None;
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::FocusWindowUpOrColumnLeft => {
                self.niri.layout.focus_up_or_left();
                self.maybe_warp_cursor_to_focus();
                self.niri.layer_shell_on_demand_focus = None;
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::FocusWindowUpOrColumnRight => {
                self.niri.layout.focus_up_or_right();
                self.maybe_warp_cursor_to_focus();
                self.niri.layer_shell_on_demand_focus = None;
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::FocusWindowOrWorkspaceDown => {
                self.niri.layout.focus_window_or_workspace_down();
                self.maybe_warp_cursor_to_focus();
                self.niri.layer_shell_on_demand_focus = None;
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::FocusWindowOrWorkspaceUp => {
                self.niri.layout.focus_window_or_workspace_up();
                self.maybe_warp_cursor_to_focus();
                self.niri.layer_shell_on_demand_focus = None;
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::FocusWindowTop => {
                self.niri.layout.focus_window_top();
                self.maybe_warp_cursor_to_focus();
                self.niri.layer_shell_on_demand_focus = None;
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::FocusWindowBottom => {
                self.niri.layout.focus_window_bottom();
                self.maybe_warp_cursor_to_focus();
                self.niri.layer_shell_on_demand_focus = None;
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::FocusWindowDownOrTop => {
                self.niri.layout.focus_window_down_or_top();
                self.maybe_warp_cursor_to_focus();
                self.niri.layer_shell_on_demand_focus = None;
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::FocusWindowUpOrBottom => {
                self.niri.layout.focus_window_up_or_bottom();
                self.maybe_warp_cursor_to_focus();
                self.niri.layer_shell_on_demand_focus = None;
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::MoveWindowToWorkspaceDown(focus) => {
                self.niri.layout.move_to_workspace_down(focus);
                self.maybe_warp_cursor_to_focus();
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::MoveWindowToWorkspaceUp(focus) => {
                self.niri.layout.move_to_workspace_up(focus);
                self.maybe_warp_cursor_to_focus();
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::MoveWindowToNewWorkspaceDown(focus) => {
                self.niri.layout.move_to_new_workspace_down(focus);
                self.maybe_warp_cursor_to_focus();
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::MoveWindowToNewWorkspaceUp(focus) => {
                self.niri.layout.move_to_new_workspace_up(focus);
                self.maybe_warp_cursor_to_focus();
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::AddWorkspaceDown => {
                self.niri.layout.add_workspace_down();
                self.maybe_warp_cursor_to_focus();
                self.niri.layer_shell_on_demand_focus = None;
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::AddWorkspaceUp => {
                self.niri.layout.add_workspace_up();
                self.maybe_warp_cursor_to_focus();
                self.niri.layer_shell_on_demand_focus = None;
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::AddBookmark => {
                if let Some(id) = self.niri.layout.add_bookmark() {
                    // The window was already bookmarked and the configured
                    // `repress` policy is `remove`: prompt before removing,
                    // regardless of whether this add-bookmark came from a
                    // keybind or an IPC caller.
                    if self
                        .niri
                        .confirm_dialog
                        .show(ConfirmRequest::RemoveBookmark { id })
                    {
                        self.niri.queue_redraw_all();
                    } else {
                        warn!(
                            "confirm dialog unavailable; not removing bookmark without confirmation"
                        );
                    }
                }
            }
            Action::AddBookmarkRule { app_id, title } => {
                let compile = |src: Option<String>| -> Result<Option<RegexEq>, DoActionError> {
                    match src {
                        None => Ok(None),
                        Some(s) => s.parse::<RegexEq>().map(Some).map_err(|err| {
                            DoActionError::BookmarkRuleInvalid {
                                reason: format!("{err}"),
                            }
                        }),
                    }
                };
                let app_id_re = compile(app_id).map_err(|err| {
                    debug!("add_bookmark_rule: {err:?}, propagating");
                    err
                })?;
                let title_re = compile(title).map_err(|err| {
                    debug!("add_bookmark_rule: {err:?}, propagating");
                    err
                })?;
                let rule = BookmarkRule::new(app_id_re, title_re).map_err(|err| {
                    let err = DoActionError::BookmarkRuleInvalid {
                        reason: err.to_string(),
                    };
                    debug!("add_bookmark_rule: {err:?}, propagating");
                    err
                })?;
                self.niri.layout.add_bookmark_rule(rule);

                // Creation-time attach: sweep the current windows and attach the
                // rule to the first un-bookmarked match in iteration order. The
                // per-window app-id/raw title are snapshotted first so the
                // `&mut layout` attach call does not overlap the `&layout`
                // iteration.
                let candidates: Vec<_> = self
                    .niri
                    .layout
                    .windows_all()
                    .map(|(_, mapped)| {
                        let (app_id, title) = with_toplevel_role(mapped.toplevel(), |role| {
                            (role.app_id.clone(), role.title.clone())
                        });
                        (mapped.window.clone(), app_id, title)
                    })
                    .collect();
                for (window, app_id, title) in candidates {
                    if self
                        .niri
                        .layout
                        .try_attach_bookmark_rules(&window, app_id.as_deref(), title.as_deref())
                        .is_some()
                    {
                        break;
                    }
                }
            }
            Action::RemoveBookmark(skip_confirmation) => {
                if skip_confirmation {
                    // The id-less arm never errs (silent no-op when nothing
                    // matches); keep the propagation shape for uniformity
                    // with `RemoveBookmarkById`.
                    if let Err(err) = self.niri.layout.remove_bookmark(None) {
                        debug!("remove_bookmark: {err:?}, propagating");
                        return Err(err);
                    }
                } else if let Some(id) = self.niri.layout.bookmark_id_for_focused() {
                    if self
                        .niri
                        .confirm_dialog
                        .show(ConfirmRequest::RemoveBookmark { id })
                    {
                        self.niri.queue_redraw_all();
                    } else {
                        warn!(
                            "confirm dialog unavailable; not removing bookmark without confirmation"
                        );
                    }
                } else {
                    // No focused window, or the focused window has no
                    // bookmark: a boundary no-op, same class as walking off
                    // the end of the list. No dialog to show.
                    debug!(
                        "remove_bookmark: no focused window or focused window has no bookmark, no-op"
                    );
                }
            }
            Action::RemoveBookmarkById(id) => {
                if let Err(err) = self.niri.layout.remove_bookmark(Some(id)) {
                    debug!("remove_bookmark: {err:?}, propagating");
                    return Err(err);
                }
            }
            Action::WalkBookmarksForward => {
                let prev_output = self.niri.layout.active_output().cloned();
                match self.niri.layout.walk_bookmarks(WalkDirection::Forward) {
                    Err(block) => {
                        debug!("walk_bookmarks(forward): hard-blocked by {block:?}, parking for re-dispatch");
                        return Err(block.into());
                    }
                    Ok(BookmarkJumpOutcome::Noop) => {
                        debug!("walk_bookmarks(forward): boundary or empty list, no-op");
                    }
                    Ok(BookmarkJumpOutcome::Jumped { switched_activity }) => {
                        self.post_jump_bookkeeping(prev_output, switched_activity);
                    }
                }
            }
            Action::WalkBookmarksBackward => {
                let prev_output = self.niri.layout.active_output().cloned();
                match self.niri.layout.walk_bookmarks(WalkDirection::Backward) {
                    Err(block) => {
                        debug!("walk_bookmarks(backward): hard-blocked by {block:?}, parking for re-dispatch");
                        return Err(block.into());
                    }
                    Ok(BookmarkJumpOutcome::Noop) => {
                        debug!("walk_bookmarks(backward): boundary or empty list, no-op");
                    }
                    Ok(BookmarkJumpOutcome::Jumped { switched_activity }) => {
                        self.post_jump_bookkeeping(prev_output, switched_activity);
                    }
                }
            }
            Action::JumpToBookmark(id) => {
                let prev_output = self.niri.layout.active_output().cloned();
                match self.niri.layout.jump_to_bookmark(id) {
                    Err(err) => {
                        debug!("jump_to_bookmark: failed ({err:?}), propagating");
                        return Err(err);
                    }
                    Ok(BookmarkJumpOutcome::Noop) => {
                        // Structurally unreachable for a known bookmark
                        // (rustdoc-level invariant on Layout::jump_to_bookmark),
                        // but not panic-worthy: a breadcrumb keeps the path loud
                        // without turning a future benign refactor into a crash.
                        debug!("jump_to_bookmark: Noop outcome (unexpected for a known bookmark)");
                    }
                    Ok(BookmarkJumpOutcome::Jumped { switched_activity }) => {
                        self.post_jump_bookkeeping(prev_output, switched_activity);
                    }
                }
            }
            Action::MoveBookmark { id, pos } => {
                if let Err(err) = self.niri.layout.move_bookmark(id, pos) {
                    debug!("move_bookmark: {err:?}, propagating");
                    return Err(err);
                }
            }
            Action::AssignBookmarkKey { id, key: raw_key } => {
                let parsed: Key = match raw_key.parse() {
                    Ok(k) => k,
                    Err(err) => {
                        return Err(DoActionError::BookmarkKeyInvalid {
                            key: raw_key,
                            reason: format!("{err}"),
                        });
                    }
                };

                // The shared validator rejects collision against the static
                // config binds, the recent-windows binds, and every other
                // bookmark's key before touching layout state. MRU-open-only
                // binds (`opened_bindings`) are deliberately not checked:
                // bookmark binds are suppressed while the MRU is open (see
                // `make_binds_iter`), so they can never be live at the same
                // time. The sibling-bookmark check reads the layout list
                // directly (not the `Niri::bookmark_binds` synthetic mirror,
                // which is epoch-gated and only rebuilds once per calloop
                // dispatch iteration) so two `AssignBookmarkKey` actions
                // dispatched in the same iteration can't both see a stale,
                // pre-assign view and double-accept a collision.
                let mod_key = self.backend.mod_key(&self.niri.config.borrow());
                let candidate = {
                    let config = self.niri.config.borrow();
                    validate_bookmark_key_candidate(
                        parsed,
                        id,
                        config
                            .binds
                            .0
                            .iter()
                            .chain(config.recent_windows.binds.iter()),
                        self.niri
                            .layout
                            .bookmarks()
                            .list()
                            .iter()
                            .filter_map(|b| b.key().map(|k| (b.id().get(), k.key()))),
                        mod_key,
                    )
                };
                let bookmark_key = match candidate {
                    Ok(bookmark_key) => bookmark_key,
                    Err(BookmarkKeyRejection::Invalid(err)) => {
                        return Err(DoActionError::BookmarkKeyInvalid {
                            key: raw_key,
                            reason: err.to_string(),
                        });
                    }
                    Err(BookmarkKeyRejection::Collision { key, .. }) => {
                        return Err(DoActionError::BookmarkKeyCollision { key });
                    }
                };

                if let Err(err) = self.niri.layout.assign_bookmark_key(id, bookmark_key) {
                    debug!("assign_bookmark_key: {err:?}, propagating");
                    return Err(err);
                }
            }
            Action::UnassignBookmarkKey(id) => {
                if let Err(err) = self.niri.layout.unassign_bookmark_key(id) {
                    debug!("unassign_bookmark_key: {err:?}, propagating");
                    return Err(err);
                }
            }
            Action::CaptureBookmarkKey { id } => {
                // Resolve the id first: an unknown id stays a loud
                // `BookmarkNotFound` even when the modal-overlay gate below
                // would otherwise refuse the open.
                let Some(bookmark) = self.niri.layout.bookmarks().get_by_raw(id) else {
                    return Err(DoActionError::BookmarkNotFound { id });
                };

                // The prompt label: the bookmark's display name if set, else
                // (for an attached anchor) the clean tag-stripped window
                // title, else a description of the dangling rule that will
                // re-attach it — a dangling rule anchor is a legal capture
                // target, keys follow the bookmark, not its current window.
                let label = if let Some(name) = bookmark.name() {
                    name.as_str().to_owned()
                } else {
                    match bookmark.anchor().wire() {
                        AnchorWire::Attached { window, .. } => {
                            let mapped = self
                                .niri
                                .layout
                                .windows_all()
                                .find(|(_, w)| LayoutElement::id(*w) == window)
                                .map(|(_, w)| w)
                                .expect(
                                    "bookmark window must resolve via windows_all \
                                     (prune-on-close guarantee)",
                                );
                            let clean_title = with_toplevel_role(mapped.toplevel(), |role| {
                                role_title_to_tag_and_clean(&role.title).clean_title
                            });
                            clean_title
                                .filter(|title| !title.is_empty())
                                .unwrap_or_else(|| "(untitled)".to_owned())
                        }
                        AnchorWire::DanglingRule(rule) => bookmark_rule_capture_label(rule),
                    }
                };

                // Same modal-overlay refusal gate as `OpenBookmarkSwitcher`/
                // `EnterBookmarkMode`: only one modal overlay owns keyboard
                // focus at a time.
                if self.niri.modal_overlay_blocks_bookmark_overlay() {
                    debug!("capture-bookmark-key: another overlay is open, ignoring");
                } else if self.niri.bookmark_switcher.open_capture(id, label) {
                    // The overlay does not self-redraw; opening needs an
                    // explicit redraw.
                    self.niri.queue_redraw_all();
                }
                // A `false` return from `open_capture` means the prompt
                // failed to rasterise; it already warned internally.
            }
            Action::RenameBookmark { id, name: raw_name } => {
                let name = match raw_name {
                    None => None,
                    Some(raw) => match BookmarkName::new(&raw) {
                        Ok(name) => Some(name),
                        Err(err) => {
                            return Err(DoActionError::BookmarkNameInvalid {
                                name: raw,
                                reason: err.to_string(),
                            });
                        }
                    },
                };
                if let Err(err) = self.niri.layout.rename_bookmark(id, name) {
                    debug!("rename_bookmark: {err:?}, propagating");
                    return Err(err);
                }
            }
            Action::JumpToBookmarkViaKey(id) => {
                let prev_output = self.niri.layout.active_output().cloned();
                match self.niri.layout.jump_to_bookmark_via_key(id) {
                    Err(DoActionError::BookmarkNotFound { id }) => {
                        // A stale synthetic bind can race a bookmark-key
                        // unassign/prune within one refresh cycle; tolerated,
                        // not propagated.
                        debug!(
                            "jump_to_bookmark_via_key: bookmark {id} not found \
                             (stale synthetic bind), ignoring"
                        );
                    }
                    Err(err) => {
                        debug!("jump_to_bookmark_via_key: failed ({err:?}), propagating");
                        return Err(err);
                    }
                    Ok(BookmarkJumpOutcome::Noop) => {
                        // Structurally unreachable for a known bookmark, but not
                        // panic-worthy — see the `JumpToBookmark` arm's identical
                        // breadcrumb.
                        debug!(
                            "jump_to_bookmark_via_key: Noop outcome (unexpected for a known bookmark)"
                        );
                    }
                    Ok(BookmarkJumpOutcome::Jumped { switched_activity }) => {
                        self.post_jump_bookkeeping(prev_output, switched_activity);
                    }
                }
            }
            Action::OpenBookmarkSwitcher => {
                // Only one modal overlay owns keyboard focus at a time: refuse
                // to open behind a confirm dialog, the lock screen, the
                // screenshot UI, or the MRU switcher.
                if self.niri.modal_overlay_blocks_bookmark_overlay() {
                    debug!("open-bookmark-switcher: another overlay is open, ignoring");
                } else {
                    // Scope the config borrow to the open call: the `RefCell`
                    // guard must be dropped before `queue_redraw_all()` below
                    // borrows `self.niri` again.
                    let opened = {
                        let config = self.niri.config.borrow();
                        self.niri
                            .bookmark_switcher
                            .open(&self.niri.layout, &config.bookmarks)
                    };
                    if opened {
                        // The overlay does not self-redraw; opening (or the
                        // idempotent re-open refresh) needs an explicit redraw.
                        self.niri.queue_redraw_all();
                    }
                }
                // A `false` return means nothing was visible to tag (or every
                // hint failed to rasterise); `open` logged the reason and left
                // the overlay closed.
            }
            Action::EnterBookmarkMode => {
                // Same modal-overlay refusal gate as `OpenBookmarkSwitcher`:
                // only one modal overlay owns keyboard focus at a time.
                if self.niri.modal_overlay_blocks_bookmark_overlay() {
                    debug!("enter-bookmark-mode: another overlay is open, ignoring");
                } else {
                    // Scope the config borrow to the open call: the `RefCell`
                    // guard must be dropped before `queue_redraw_all()` below
                    // borrows `self.niri` again.
                    let opened = {
                        let config = self.niri.config.borrow();
                        self.niri
                            .bookmark_switcher
                            .open_mode(&self.niri.layout, &config.bookmarks)
                    };
                    if opened {
                        // The overlay does not self-redraw; opening (or the
                        // idempotent re-open refresh) needs an explicit redraw.
                        self.niri.queue_redraw_all();
                    }
                }
                // Unlike `OpenBookmarkSwitcher`, `open_mode` only refuses
                // entry when the command sheet itself fails to rasterise —
                // zero visible bookmarks still opens (the sheet is useful on
                // its own).
            }
            Action::MoveWindowToWorkspace(reference, focus) => {
                // Move-to-self short-circuit. Resolve source (the active
                // workspace, since this arm targets the active workspace's
                // focused tile) and target ids; if they match, emit a typed
                // `NoOp(AlreadyOnTarget)` on the wire rather than falling
                // through to the layout-mutator's silent equality
                // short-circuit. The `resolve_workspace_reference_to_id` `Id`
                // arm is pool-wide, so a cross-activity self-move (target id
                // names the same workspace the focused tile already lives on,
                // viewed via a dormant activity) is also caught here.
                let source_ws_id = self.niri.layout.active_workspace().map(|ws| ws.id().get());
                let target_ws_id = self
                    .niri
                    .resolve_workspace_reference_to_id(reference.clone());
                if let (Some(src), Some(tgt)) = (source_ws_id, target_ws_id) {
                    if src == tgt {
                        return Ok(DoActionOutcome::NoOp(NoOpReason::AlreadyOnTarget {
                            workspace_id: src,
                        }));
                    }
                }

                // Cross-activity dispatch for `Id`-reference moves. The CLI
                // sends `Id` for cross-activity routes (focus-drift guard);
                // `Name` / `Index` continue to follow the existing
                // active-view-scoped path, which silently drops on misses.
                //
                // For the `Id` reference the four-arm classifier below
                // either errors (unknown target / disconnected output),
                // delegates back to the existing index path (target is in an
                // active view), or moves the focused tile into a dormant
                // pool workspace and honors `focus:true` via the same
                // activation flow as `Action::FocusWindow`.
                if let WorkspaceReference::Id(raw) = reference {
                    // Gate the hard-block check before any mutating call when
                    // `focus:true` requests an activity switch. Mirrors the
                    // `Action::FocusWindow` precedent, which checks
                    // `is_activity_switch_hard_blocked` before acting, so an
                    // in-flight gesture cannot observe a moved window with no
                    // matching focus change.
                    if focus {
                        if let Some(block) = self.niri.layout.is_activity_switch_hard_blocked() {
                            return Err(block.into());
                        }
                    }
                    let activate = if focus {
                        ActivateWindow::Smart
                    } else {
                        ActivateWindow::No
                    };
                    // Capture the focused tile's last-focused-activity hint
                    // before the move; the hint informs the focus:true target
                    // activity pick and may shift after the detach.
                    let focus_hint = if focus {
                        self.niri
                            .layout
                            .active_workspace()
                            .and_then(|ws| ws.active_window())
                            .and_then(|w| w.get_last_focused_activity())
                    } else {
                        None
                    };
                    // Snapshot the focused window before the move so the
                    // `focus:true` follow-up can re-focus it under the
                    // newly-active activity.
                    let focused_window = if focus {
                        self.niri
                            .layout
                            .active_workspace()
                            .and_then(|ws| ws.active_window())
                            .map(|w| w.window.clone())
                    } else {
                        None
                    };
                    match self
                        .niri
                        .layout
                        .move_window_to_pool_workspace(None, raw, activate)
                    {
                        Ok(MoveWindowToPoolOutcome::DelegateToActiveView) => {
                            // Fall through to the existing path below.
                        }
                        Ok(MoveWindowToPoolOutcome::NothingToMove) => {
                            // Empty bookend slot — no tile to move. No activity
                            // switch, no focus change; reply Handled so the CLI
                            // does not print a false-error line.
                            return Ok(DoActionOutcome::Handled);
                        }
                        Ok(MoveWindowToPoolOutcome::MovedDormant {
                            ws_id: target_ws_id,
                        }) => {
                            if focus {
                                let target_activity = self
                                    .niri
                                    .layout
                                    .pick_activity_for_hidden_window(target_ws_id, focus_hint);
                                if target_activity != self.niri.layout.active_activity_id() {
                                    self.switch_activity_and_reconcile(target_activity);
                                }
                                if let Some(window) = focused_window.as_ref() {
                                    self.focus_window(window);
                                } else {
                                    self.maybe_warp_cursor_to_focus();
                                }
                            } else {
                                self.maybe_warp_cursor_to_focus();
                            }
                            self.niri.queue_redraw_all();
                            return Ok(DoActionOutcome::Handled);
                        }
                        Err(_) => {
                            return Err(DoActionError::MoveWindowTargetUnreachable { ws_id: raw });
                        }
                    }
                }

                // The token is built eagerly for every reference form, but only
                // the Name arm's value is ever consumed: Index always resolves
                // to Some (saturating clamp) and Id is either intercepted
                // upstream or resolves via the active view, so the `None`
                // error-return below — the only reader of this token — is never
                // reached for Index or Id. A harmless placeholder for those
                // arms is preferable to a panic on a path the dispatch
                // evaluates on every move.
                let name_token = match &reference {
                    jiji_config::WorkspaceReference::Name(n) => n.clone(),
                    jiji_config::WorkspaceReference::Index(i) => format!("{i}"),
                    jiji_config::WorkspaceReference::Id(id) => format!("id:{id}"),
                };
                if let Some((mut output, index)) =
                    self.niri.find_output_and_workspace_index(reference)
                {
                    // The source output is always the active output, so if the target output is
                    // also the active output, we don't need to use move_to_output().
                    if let Some(active) = self.niri.layout.active_output() {
                        if output.as_ref() == Some(active) {
                            output = None;
                        }
                    }

                    let activate = if focus {
                        ActivateWindow::Smart
                    } else {
                        ActivateWindow::No
                    };

                    if let Some(output) = output {
                        self.niri
                            .layout
                            .move_to_output(None, &output, Some(index), activate);

                        if focus {
                            if !self.maybe_warp_cursor_to_focus_centered() {
                                self.move_cursor_to_output(&output);
                            }
                        } else {
                            self.maybe_warp_cursor_to_focus();
                        }
                    } else {
                        self.niri.layout.move_to_workspace(None, index, activate);
                        self.maybe_warp_cursor_to_focus();
                    }

                    // FIXME: granular
                    self.niri.queue_redraw_all();
                } else {
                    return Err(DoActionError::MoveWindowTargetUnknownName { name: name_token });
                }
            }
            Action::MoveWindowToWorkspaceById {
                window_id: id,
                reference,
                focus,
            } => {
                // Fold the source-workspace lookup into the same
                // `windows_all()` scan that already resolves the window
                // handle, then check the move-to-self short-circuit before
                // resolving the target. Unknown window id preserves the
                // pre-existing silent exit-0 as `Ok(Handled)`: no source
                // workspace id is known, so no `AlreadyOnTarget { workspace_id
                // }` payload is constructible.
                let resolved = self
                    .niri
                    .layout
                    .windows_all()
                    .find(|(_, m)| m.id().get() == id);
                let resolved = resolved.and_then(|(_, mapped)| {
                    let win_id = LayoutElement::id(mapped);
                    let ws_id = self
                        .niri
                        .layout
                        .workspaces_all()
                        .find(|(_, ws)| ws.has_window(win_id))?
                        .1
                        .id();
                    let focus_hint = if focus {
                        mapped.get_last_focused_activity()
                    } else {
                        None
                    };
                    Some((mapped.window.clone(), ws_id, focus_hint))
                });
                let Some((window, source_ws_id, focus_hint)) = resolved else {
                    debug!(
                        window_id = id,
                        "MoveWindowToWorkspaceById: window_id not found in pool; \
                         preserving silent exit-0"
                    );
                    return Ok(DoActionOutcome::Handled);
                };

                let target_ws_id = self
                    .niri
                    .resolve_workspace_reference_to_id(reference.clone());
                if let Some(tgt) = target_ws_id {
                    if source_ws_id.get() == tgt {
                        return Ok(DoActionOutcome::NoOp(NoOpReason::AlreadyOnTarget {
                            workspace_id: source_ws_id.get(),
                        }));
                    }
                }

                // Dormant-source filter. The cross-activity pool entry point
                // assumes its source is live under the active view; if the
                // named window is on a dormant workspace, take the existing
                // fall-through (the layout-side `move_to_workspace` walk
                // emits its own `warn!` and clean-returns), which yields
                // `Ok(Handled)` — distinct from the new `Err` for unreachable
                // targets, so the `Err` never misattributes the dormant-source
                // case. This preserves the regression test pinned at
                // `move_window_to_workspace_by_id_reaches_hidden_activity_window_without_panic`.
                let source_is_active_view = self
                    .niri
                    .layout
                    .activities()
                    .active()
                    .views()
                    .values()
                    .any(|view| view.ids().contains(&source_ws_id));

                if source_is_active_view {
                    if let WorkspaceReference::Id(raw) = reference {
                        // Gate the hard-block check before any mutating call
                        // when `focus:true` requests an activity switch. Mirrors
                        // the `Action::FocusWindow` precedent and the None arm
                        // above.
                        if focus {
                            if let Some(block) = self.niri.layout.is_activity_switch_hard_blocked()
                            {
                                return Err(block.into());
                            }
                        }
                        let activate = if focus {
                            ActivateWindow::Smart
                        } else {
                            ActivateWindow::No
                        };
                        match self.niri.layout.move_window_to_pool_workspace(
                            Some(&window),
                            raw,
                            activate,
                        ) {
                            Ok(MoveWindowToPoolOutcome::DelegateToActiveView) => {
                                // Fall through to the existing path below.
                            }
                            Ok(MoveWindowToPoolOutcome::NothingToMove) => {
                                // The `Some(window)` arm always finds the window
                                // (caller verified presence); `NothingToMove` is
                                // unreachable on this arm by construction.
                                unreachable!(
                                    "move_window_to_pool_workspace: Some(window) arm \
                                     always has a tile to remove; NothingToMove \
                                     is only possible on the None arm"
                                )
                            }
                            Ok(MoveWindowToPoolOutcome::MovedDormant {
                                ws_id: target_ws_id,
                            }) => {
                                if focus {
                                    let target_activity = self
                                        .niri
                                        .layout
                                        .pick_activity_for_hidden_window(target_ws_id, focus_hint);
                                    if target_activity != self.niri.layout.active_activity_id() {
                                        self.switch_activity_and_reconcile(target_activity);
                                    }
                                    self.focus_window(&window);
                                } else {
                                    let new_focus = self.niri.layout.focus();
                                    if new_focus.is_some_and(|w| w.window == window) {
                                        self.maybe_warp_cursor_to_focus();
                                    }
                                }
                                self.niri.queue_redraw_all();
                                return Ok(DoActionOutcome::Handled);
                            }
                            Err(_) => {
                                return Err(DoActionError::MoveWindowTargetUnreachable {
                                    ws_id: raw,
                                });
                            }
                        }
                    }
                }
                // else: dormant source. Fall-through into the active-view-scoped
                // path yields `Ok(Handled)` when the target name resolves
                // (the `move_to_workspace` warn!/clean-return handles the
                // cross-activity source); an unresolvable target name reaches
                // the `else` below and errors with `MoveWindowTargetUnknownName`
                // regardless of source dormancy.

                // The token is built eagerly for every reference form, but only
                // the Name arm's value is ever consumed: Index always resolves
                // to Some (saturating clamp) and Id is either intercepted
                // upstream or resolves via the active view, so the `None`
                // error-return below — the only reader of this token — is never
                // reached for Index or Id. A harmless placeholder for those
                // arms is preferable to a panic on a path the dispatch
                // evaluates on every move.
                let name_token = match &reference {
                    jiji_config::WorkspaceReference::Name(n) => n.clone(),
                    jiji_config::WorkspaceReference::Index(i) => format!("{i}"),
                    jiji_config::WorkspaceReference::Id(id) => format!("id:{id}"),
                };
                if let Some((output, index)) = self.niri.find_output_and_workspace_index(reference)
                {
                    let target_was_active = self
                        .niri
                        .layout
                        .active_output()
                        .is_some_and(|active| output.as_ref() == Some(active));

                    let activate = if focus {
                        ActivateWindow::Smart
                    } else {
                        ActivateWindow::No
                    };

                    if let Some(output) = output {
                        self.niri.layout.move_to_output(
                            Some(&window),
                            &output,
                            Some(index),
                            activate,
                        );

                        // If the active output changed (window was moved and focused).
                        #[allow(clippy::collapsible_if)]
                        if !target_was_active && self.niri.layout.active_output() == Some(&output) {
                            if !self.maybe_warp_cursor_to_focus_centered() {
                                self.move_cursor_to_output(&output);
                            }
                        }
                    } else {
                        self.niri
                            .layout
                            .move_to_workspace(Some(&window), index, activate);

                        // If we focused the target window.
                        let new_focus = self.niri.layout.focus();
                        if new_focus.is_some_and(|win| win.window == window) {
                            self.maybe_warp_cursor_to_focus();
                        }
                    }

                    // FIXME: granular
                    self.niri.queue_redraw_all();
                } else {
                    return Err(DoActionError::MoveWindowTargetUnknownName { name: name_token });
                }
            }
            Action::MoveColumnToWorkspaceDown(focus) => {
                self.niri.layout.move_column_to_workspace_down(focus);
                self.maybe_warp_cursor_to_focus();
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::MoveColumnToWorkspaceUp(focus) => {
                self.niri.layout.move_column_to_workspace_up(focus);
                self.maybe_warp_cursor_to_focus();
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::MoveColumnToWorkspace(reference, focus) => {
                // Deliberate silent no-op on a Name miss: MoveColumnToWorkspace
                // has no by-name/by-id IPC consumer (only the Index keybind,
                // which clamps and cannot miss), so unlike move-window it stays
                // a silent no-op rather than surfacing an error.
                if let Some((mut output, index)) =
                    self.niri.find_output_and_workspace_index(reference)
                {
                    if let Some(active) = self.niri.layout.active_output() {
                        if output.as_ref() == Some(active) {
                            output = None;
                        }
                    }

                    if let Some(output) = output {
                        self.niri
                            .layout
                            .move_column_to_output(&output, Some(index), focus);
                        if focus && !self.maybe_warp_cursor_to_focus_centered() {
                            self.move_cursor_to_output(&output);
                        }
                    } else {
                        self.niri.layout.move_column_to_workspace(index, focus);
                        if focus {
                            self.maybe_warp_cursor_to_focus();
                        }
                    }

                    // FIXME: granular
                    self.niri.queue_redraw_all();
                }
            }
            Action::MoveColumnToIndex(idx) => {
                self.niri.layout.move_column_to_index(idx);
                self.maybe_warp_cursor_to_focus();
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::FocusWorkspaceDown => {
                self.niri.layout.switch_workspace_down();
                self.maybe_warp_cursor_to_focus();
                self.niri.layer_shell_on_demand_focus = None;
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::FocusWorkspaceDownUnderMouse => {
                if let Some(output) = self.niri.output_under_cursor() {
                    let output_id = crate::layout::workspace::OutputId::new(&output);
                    if self.niri.layout.monitor_for_output(&output).is_some() {
                        let (monitors, _, view) =
                            self.niri.layout.monitors_pool_view_mut(&output_id);
                        let mon = monitors
                            .iter_mut()
                            .find(|m| m.output() == &output)
                            .expect("monitor for output must exist");
                        mon.switch_workspace_down(view);
                        self.maybe_warp_cursor_to_focus();
                        self.niri.layer_shell_on_demand_focus = None;
                        self.niri.queue_redraw(&output);
                    }
                }
            }
            Action::FocusWorkspaceUp => {
                self.niri.layout.switch_workspace_up();
                self.maybe_warp_cursor_to_focus();
                self.niri.layer_shell_on_demand_focus = None;
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::FocusWorkspaceUpUnderMouse => {
                if let Some(output) = self.niri.output_under_cursor() {
                    let output_id = crate::layout::workspace::OutputId::new(&output);
                    if self.niri.layout.monitor_for_output(&output).is_some() {
                        let (monitors, _, view) =
                            self.niri.layout.monitors_pool_view_mut(&output_id);
                        let mon = monitors
                            .iter_mut()
                            .find(|m| m.output() == &output)
                            .expect("monitor for output must exist");
                        mon.switch_workspace_up(view);
                        self.maybe_warp_cursor_to_focus();
                        self.niri.layer_shell_on_demand_focus = None;
                        self.niri.queue_redraw(&output);
                    }
                }
            }
            Action::FocusWorkspace(reference, activity) => {
                if let Some(activity_ref) = activity {
                    // Resolve-everything-first: no observable state change
                    // (activity switch, workspace focus, cursor warp, redraw)
                    // until activity-resolve, hard-block, and workspace-resolve
                    // have all passed. A mid-sequence failure would otherwise
                    // strand the user in a switched-but-no-target state.
                    let arg: ActivityReferenceArg = activity_ref.into();
                    let Some(activity_id) = self.niri.layout.resolve_activity_ref(&arg) else {
                        warn!("focus_workspace: activity not found: {arg:?}");
                        return Err(DoActionError::SwitchActivity(
                            crate::layout::SwitchActivityError::NotFound,
                        ));
                    };
                    let needs_switch = activity_id != self.niri.layout.active_activity_id();
                    if needs_switch {
                        if let Some(block) = self.niri.layout.is_activity_switch_hard_blocked() {
                            debug!("focus_workspace: activity switch hard-blocked by {block:?}");
                            return Err(block.into());
                        }
                    }
                    let ws_id = self
                        .niri
                        .layout
                        .resolve_workspace_in_activity(activity_id, &reference)
                        .map_err(DoActionError::FocusWorkspaceInActivity)?;

                    // Only now mutate: switch activity, then reconcile inhibitor
                    // visibility with the new active set.
                    if needs_switch {
                        self.niri.layout.switch_activity(activity_id);
                        self.niri
                            .refresh_keyboard_shortcut_inhibitors_after_activity_switch();
                    }

                    // The workspace is in the active activity now; reuse the
                    // existing output-aware focus path (warp included; redraw
                    // is unconditional below).
                    if let Some((mut output, index)) = self
                        .niri
                        .find_output_and_workspace_index(WorkspaceReference::Id(ws_id.get()))
                    {
                        if let Some(active) = self.niri.layout.active_output() {
                            if output.as_ref() == Some(active) {
                                output = None;
                            }
                        }
                        if let Some(output) = output {
                            self.niri.layout.focus_output(&output);
                            self.niri.layout.switch_workspace(index);
                            if !self.maybe_warp_cursor_to_focus_centered() {
                                self.move_cursor_to_output(&output);
                            }
                        } else {
                            // No auto-back-and-forth on the explicit-activity
                            // path: the request names an exact destination.
                            self.niri.layout.switch_workspace(index);
                            self.maybe_warp_cursor_to_focus();
                        }
                    }
                    // Partial-success contract: if `find_output_and_workspace_index`
                    // returned None (workspace not reachable from any connected view
                    // — exclusive to a dormant activity or bound to a disconnected
                    // output), the activity switch above still stands — identical to
                    // a bare activity switch followed by a focus that found no
                    // connected output.
                    self.niri.layer_shell_on_demand_focus = None;
                    // FIXME: granular
                    self.niri.queue_redraw_all();
                } else {
                    // Capture a human-readable token before `reference` is
                    // consumed by `find_output_and_workspace_index`.
                    let reference_token = match &reference {
                        jiji_config::WorkspaceReference::Name(n) => n.clone(),
                        jiji_config::WorkspaceReference::Id(id) => format!("id:{id}"),
                        // find_output_and_workspace_index returns Some unconditionally for
                        // Index (it clamps and returns early), so the error `else` below is
                        // never reached for an Index reference. The token is built here
                        // regardless to keep the match exhaustive; it stays harmless if that
                        // invariant ever changes.
                        jiji_config::WorkspaceReference::Index(i) => format!("{i}"),
                    };
                    if let Some((mut output, index)) =
                        self.niri.find_output_and_workspace_index(reference)
                    {
                        if let Some(active) = self.niri.layout.active_output() {
                            if output.as_ref() == Some(active) {
                                output = None;
                            }
                        }

                        if let Some(output) = output {
                            self.niri.layout.focus_output(&output);
                            self.niri.layout.switch_workspace(index);
                            if !self.maybe_warp_cursor_to_focus_centered() {
                                self.move_cursor_to_output(&output);
                            }
                        } else {
                            let config = &self.niri.config;
                            if config.borrow().input.workspace_auto_back_and_forth {
                                self.niri.layout.switch_workspace_auto_back_and_forth(index);
                            } else {
                                self.niri.layout.switch_workspace(index);
                            }
                            self.maybe_warp_cursor_to_focus();
                        }
                        self.niri.layer_shell_on_demand_focus = None;

                        // FIXME: granular
                        self.niri.queue_redraw_all();
                    } else {
                        return Err(DoActionError::FocusWorkspaceTargetUnknown {
                            reference: reference_token,
                        });
                    }
                }
            }
            Action::FocusWorkspacePrevious => {
                self.niri.layout.switch_workspace_previous();
                self.maybe_warp_cursor_to_focus();
                self.niri.layer_shell_on_demand_focus = None;
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::SwitchActivity(reference) => {
                // Keybinding-triggered switches are silently dropped while
                // hard-blocked (no cursor warp, no redraw, no focus reset). IPC callers
                // receive `Reply::Err("activity switch blocked: ...")` with the block
                // reason; per-connection queue-and-await is implemented via
                // `drain_blocked_action_waiters` in ipc/server.rs.
                if let Some(block) = self.niri.layout.is_activity_switch_hard_blocked() {
                    debug!("switch_activity: hard-blocked by {block:?}, ignoring");
                    return Err(block.into());
                }
                // jiji-config holds its own ActivityReference to keep config
                // types independent of jiji-ipc's wire enums; layout's API
                // speaks the IPC type, so map variants at the boundary.
                let arg: ActivityReferenceArg = reference.into();
                match self.niri.layout.resolve_activity_ref(&arg) {
                    Some(id) => {
                        self.niri.layout.switch_activity(id);
                        self.activity_switch_epilogue();
                    }
                    None => {
                        warn!("switch_activity: activity not found: {arg:?}");
                        return Err(DoActionError::SwitchActivity(
                            crate::layout::SwitchActivityError::NotFound,
                        ));
                    }
                }
            }
            Action::SwitchActivityPrevious(depth) => {
                if let Some(block) = self.niri.layout.is_activity_switch_hard_blocked() {
                    debug!(
                        "switch_activity_previous: hard-blocked by {block:?}, ignoring \
                        "
                    );
                    return Err(block.into());
                }
                self.niri.layout.switch_activity_previous(depth);
                self.activity_switch_epilogue();
            }
            Action::CreateActivity(name) => {
                let name_str = name.clone();
                match self.niri.layout.create_activity(name) {
                    Ok(id) => {
                        debug!("CreateActivity: created {id:?} {name_str:?}");
                    }
                    Err(e) => {
                        warn!("CreateActivity: {e}: {name_str:?}");
                        return Err(DoActionError::CreateActivity(e));
                    }
                }
            }
            Action::RenameActivity(reference, name) => {
                // Rename is pure metadata — no cascade, no view patching, no
                // frame invalidation needed. Mirrors the `CreateActivity` arm
                // shape (debug on Ok, warn + Err on the failure path), not
                // `RemoveActivity`'s gated shape.
                let name_str = name.clone();
                let arg: ActivityReferenceArg = reference.into();
                match self.niri.layout.rename_activity(&arg, name) {
                    Ok(id) => {
                        debug!("RenameActivity: renamed {id:?} ({arg:?}) to {name_str:?}");
                    }
                    Err(e) => {
                        warn!("RenameActivity: {e}: {arg:?} new_name={name_str:?}");
                        return Err(DoActionError::RenameActivity(e));
                    }
                }
            }
            Action::RemoveActivity(reference) => {
                // "Animation blocking", keybinding-triggered
                // removes are silently dropped while hard-blocked — the cascade
                // branch calls `switch_activity` which debug-asserts the same
                // gate, so we must filter before dispatch. IPC per-connection
                // queueing is scope.
                if let Some(block) = self.niri.layout.is_activity_switch_hard_blocked() {
                    debug!("remove_activity: hard-blocked by {block:?}, ignoring");
                    return Err(block.into());
                }
                let arg: ActivityReferenceArg = reference.into();
                match self.niri.layout.remove_activity(&arg) {
                    Ok(id) => {
                        debug!("RemoveActivity: removed {id:?} ({arg:?})");
                        // The cascade branch may have flipped the active
                        // activity internally; run the epilogue unconditionally
                        // (not inside `Layout`, since the inhibitor tracking map
                        // lives on `Niri`) so the next frame rebinds cleanly
                        // either way.
                        self.activity_switch_epilogue();
                    }
                    Err(e) => {
                        warn!("remove_activity: {e}: {arg:?}");
                        return Err(DoActionError::RemoveActivity(e));
                    }
                }
            }
            Action::AddWorkspaceToActivity(ws_ref, activity_ref) => {
                // Add is safe un-gated: `add_workspace_to_activity` shifts any
                // in-flight workspace-switch (animation or gesture) when it patches
                // the active view, so no hard-block gate is needed here.
                let arg_act: ActivityReferenceArg = activity_ref.into();
                let arg_ws_log = ws_ref.clone();
                match self.niri.layout.add_workspace_to_activity(ws_ref, &arg_act) {
                    Ok((ws_id, act_id)) => {
                        debug!(
                            "AddWorkspaceToActivity: added {ws_id:?} to {act_id:?} \
                             ({arg_act:?})"
                        );
                        // No redraw: visibility only changes on the next
                        // activity switch. Event-stream emission for the
                        // workspace's `activities` field flows through the
                        // existing structural diff at ipc/server.rs
                        // diff_workspaces — no hand-rolled emit needed.
                    }
                    Err(e) => {
                        warn!(
                            "AddWorkspaceToActivity: {e}: workspace={arg_ws_log:?} \
                             activity={arg_act:?}"
                        );
                        return Err(DoActionError::AddWorkspaceToActivity(e));
                    }
                }
            }
            Action::RemoveWorkspaceFromActivity(ws_ref, activity_ref) => {
                // Remove is hard-blocked by an in-flight workspace-switch
                // gesture (removing from the
                // current activity's view would invalidate the gesture's
                // fractional targets). IPC callers are queued by the drain
                // path; keybinding callers see the same Err and the
                // keybinding dispatcher drops silently.
                if let Some(block) = self
                    .niri
                    .layout
                    .is_workspace_activity_assignment_blocked_by_gesture()
                {
                    debug!(
                        "RemoveWorkspaceFromActivity: hard-blocked by {block:?}, \
                         ignoring"
                    );
                    return Err(block.into());
                }
                let arg_act: ActivityReferenceArg = activity_ref.into();
                let arg_ws_log = ws_ref.clone();
                let active_before = self.niri.layout.active_activity_id();
                match self
                    .niri
                    .layout
                    .remove_workspace_from_activity(ws_ref, &arg_act)
                {
                    Ok((ws_id, act_id)) => {
                        debug!(
                            "RemoveWorkspaceFromActivity: removed {ws_id:?} from \
                             {act_id:?} ({arg_act:?})"
                        );
                        if act_id == active_before {
                            // Visibility of `ws_id` just flipped for the
                            // active activity. `is_in_active_activity` may
                            // have changed for workspaces that lost their
                            // active-activity tag — the structural diff at
                            // ipc/server.rs diff_workspaces emits
                            // `WorkspaceOpenedOrChanged` for those. Focus
                            // and frame may need refresh. No call
                            // here: the active activity id is unchanged —
                            // inhibitor reconciliation fires on flips only.
                            self.maybe_warp_cursor_to_focus();
                            self.niri.queue_redraw_all();
                        }
                    }
                    Err(e) => {
                        warn!(
                            "RemoveWorkspaceFromActivity: {e}: workspace={arg_ws_log:?} \
                             activity={arg_act:?}"
                        );
                        return Err(DoActionError::RemoveWorkspaceFromActivity(e));
                    }
                }
            }
            Action::MoveWorkspaceToActivity(ws_ref, activity_ref, focus) => {
                // Gate depth depends on `focus`.
                //   - focus: false → weaker gesture-only gate (matches the Remove leg of Move = Add
                //     + Remove).
                //   - focus: true → full `is_activity_switch_hard_blocked` predicate (same as
                //     `SwitchActivity`), because this path chains into `switch_activity` and
                //     interactive move / DnD must block it.
                // A single unified check here is a review-stop bug — the
                // `focus: true` path must be gated against DnD /
                // interactive_move.
                let block = if focus {
                    self.niri.layout.is_activity_switch_hard_blocked()
                } else {
                    self.niri
                        .layout
                        .is_workspace_activity_assignment_blocked_by_gesture()
                };
                if let Some(block) = block {
                    debug!(
                        "MoveWorkspaceToActivity: hard-blocked by {block:?}, \
                         ignoring (focus={focus})"
                    );
                    return Err(block.into());
                }
                let arg_act: ActivityReferenceArg = activity_ref.into();
                let arg_ws_log = ws_ref.clone();
                match self
                    .niri
                    .layout
                    .move_workspace_to_activity(ws_ref, &arg_act)
                {
                    Ok((ws_id, target_id, source_id)) => {
                        debug!(
                            "MoveWorkspaceToActivity: moved {ws_id:?} from \
                             {source_id:?} to {target_id:?} ({arg_act:?}, focus={focus})"
                        );
                        // The workspace just left the active activity's
                        // view (unless it was a no-op target == source, in
                        // which case the layout returned Ok without touching
                        // state). Visibility for the active activity may
                        // have flipped — fire cursor warp + redraw.
                        self.maybe_warp_cursor_to_focus();
                        self.niri.queue_redraw_all();
                        if focus {
                            // Chain into switch_activity. The outer gate at
                            // step 1 already guaranteed no hard-block state
                            // is live, so `switch_activity`'s debug_assert
                            // is satisfied — no inner re-check needed.
                            self.niri.layout.switch_activity(target_id);
                            self.activity_switch_epilogue();
                        }
                    }
                    Err(e) => {
                        warn!(
                            "MoveWorkspaceToActivity: {e}: workspace={arg_ws_log:?} \
                             activity={arg_act:?} focus={focus}"
                        );
                        return Err(DoActionError::MoveWorkspaceToActivity(e));
                    }
                }
            }
            Action::SetWorkspaceActivities(ws_ref, activity_refs) => {
                // Set is hard-blocked by an in-flight workspace-switch gesture
                // (symmetric-diff removes on the
                // active view invalidate gesture fractional targets). Same
                // gate as Remove. No call needed here — Set does NOT
                // flip the active activity cursor; inhibitor reconciliation
                // fires on activity flips only. (WHY-comment for future grep
                // sweeps.)
                if let Some(block) = self
                    .niri
                    .layout
                    .is_workspace_activity_assignment_blocked_by_gesture()
                {
                    debug!(
                        "SetWorkspaceActivities: hard-blocked by {block:?}, \
                         ignoring"
                    );
                    return Err(block.into());
                }
                // jiji-config holds its own ActivityReference; layout's API
                // speaks the IPC type, so map at the boundary.
                let arg_acts: Vec<ActivityReferenceArg> = activity_refs
                    .into_iter()
                    .map(ActivityReferenceArg::from)
                    .collect();
                let arg_ws_log = ws_ref.clone();
                match self.niri.layout.set_workspace_activities(ws_ref, &arg_acts) {
                    Ok((ws_id, new_set, active_affected)) => {
                        debug!(
                            "SetWorkspaceActivities: updated {ws_id:?} to \
                             {new_set:?} (active_affected={active_affected})"
                        );
                        if active_affected {
                            // Visibility for the workspace flipped in the
                            // active activity. The structural diff at
                            // ipc/server.rs diff_workspaces emits
                            // WorkspaceOpenedOrChanged for fields that
                            // changed. Focus and frame may need refresh.
                            self.maybe_warp_cursor_to_focus();
                            self.niri.queue_redraw_all();
                        }
                    }
                    Err(e) => {
                        warn!(
                            "SetWorkspaceActivities: {e}: workspace={arg_ws_log:?} \
                             activities={arg_acts:?}"
                        );
                        return Err(DoActionError::SetWorkspaceActivities(e));
                    }
                }
            }
            Action::ToggleWorkspaceSticky => {
                self.dispatch_toggle_workspace_sticky(None)?;
            }
            Action::ToggleWorkspaceStickyByRef(reference) => {
                self.dispatch_toggle_workspace_sticky(Some(reference))?;
            }
            Action::SetWorkspaceSticky => {
                self.dispatch_set_workspace_sticky(None)?;
            }
            Action::SetWorkspaceStickyByRef(reference) => {
                self.dispatch_set_workspace_sticky(Some(reference))?;
            }
            Action::UnsetWorkspaceSticky => {
                self.dispatch_unset_workspace_sticky(None)?;
            }
            Action::UnsetWorkspaceStickyByRef(reference) => {
                self.dispatch_unset_workspace_sticky(Some(reference))?;
            }
            Action::MoveWorkspaceDown => {
                self.niri.layout.move_workspace_down();
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::MoveWorkspaceUp => {
                self.niri.layout.move_workspace_up();
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::MoveWorkspaceToIndex(new_idx) => {
                let new_idx = new_idx.saturating_sub(1);
                self.niri.layout.move_workspace_to_idx(None, new_idx);
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::MoveWorkspaceToIndexByRef { new_idx, reference } => {
                if let Some(res) = self.niri.find_output_and_workspace_index(reference) {
                    let new_idx = new_idx.saturating_sub(1);
                    self.niri.layout.move_workspace_to_idx(Some(res), new_idx);
                    // FIXME: granular
                    self.niri.queue_redraw_all();
                }
            }
            Action::SetWorkspaceName(name) => {
                self.niri.layout.set_workspace_name(name, None);
            }
            Action::SetWorkspaceNameByRef { name, reference } => {
                self.niri.layout.set_workspace_name(name, Some(reference));
            }
            Action::UnsetWorkspaceName => {
                self.niri.layout.unset_workspace_name(None);
            }
            Action::UnsetWorkSpaceNameByRef(reference) => {
                self.niri.layout.unset_workspace_name(Some(reference));
            }
            Action::ConsumeWindowIntoColumn => {
                self.niri.layout.consume_into_column();
                // This does not cause immediate focus or window size change, so warping mouse to
                // focus won't do anything here.
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::ExpelWindowFromColumn => {
                self.niri.layout.expel_from_column();
                self.maybe_warp_cursor_to_focus();
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::SwapWindowRight => {
                self.niri
                    .layout
                    .swap_window_in_direction(ScrollDirection::Right);
                self.maybe_warp_cursor_to_focus();
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::SwapWindowLeft => {
                self.niri
                    .layout
                    .swap_window_in_direction(ScrollDirection::Left);
                self.maybe_warp_cursor_to_focus();
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::ToggleColumnTabbedDisplay => {
                self.niri.layout.toggle_column_tabbed_display();
                self.maybe_warp_cursor_to_focus();
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::SetColumnDisplay(display) => {
                self.niri.layout.set_column_display(display);
                self.maybe_warp_cursor_to_focus();
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::SwitchPresetColumnWidth => {
                self.niri.layout.toggle_width(true);
            }
            Action::SwitchPresetColumnWidthBack => {
                self.niri.layout.toggle_width(false);
            }
            Action::SwitchPresetWindowWidth => {
                self.niri.layout.toggle_window_width(None, true);
            }
            Action::SwitchPresetWindowWidthBack => {
                self.niri.layout.toggle_window_width(None, false);
            }
            Action::SwitchPresetWindowWidthById(id) => {
                let window = self
                    .niri
                    .layout
                    .windows_all()
                    .find(|(_, m)| m.id().get() == id);
                let window = window.map(|(_, m)| m.window.clone());
                if let Some(window) = window {
                    self.niri.layout.toggle_window_width(Some(&window), true);
                }
            }
            Action::SwitchPresetWindowWidthBackById(id) => {
                let window = self
                    .niri
                    .layout
                    .windows_all()
                    .find(|(_, m)| m.id().get() == id);
                let window = window.map(|(_, m)| m.window.clone());
                if let Some(window) = window {
                    self.niri.layout.toggle_window_width(Some(&window), false);
                }
            }
            Action::SwitchPresetWindowHeight => {
                self.niri.layout.toggle_window_height(None, true);
            }
            Action::SwitchPresetWindowHeightBack => {
                self.niri.layout.toggle_window_height(None, false);
            }
            Action::SwitchPresetWindowHeightById(id) => {
                let window = self
                    .niri
                    .layout
                    .windows_all()
                    .find(|(_, m)| m.id().get() == id);
                let window = window.map(|(_, m)| m.window.clone());
                if let Some(window) = window {
                    self.niri.layout.toggle_window_height(Some(&window), true);
                }
            }
            Action::SwitchPresetWindowHeightBackById(id) => {
                let window = self
                    .niri
                    .layout
                    .windows_all()
                    .find(|(_, m)| m.id().get() == id);
                let window = window.map(|(_, m)| m.window.clone());
                if let Some(window) = window {
                    self.niri.layout.toggle_window_height(Some(&window), false);
                }
            }
            Action::CenterColumn => {
                self.niri.layout.center_column();
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::CenterWindow => {
                self.niri.layout.center_window(None);
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::CenterWindowById(id) => {
                let window = self
                    .niri
                    .layout
                    .windows_all()
                    .find(|(_, m)| m.id().get() == id);
                let window = window.map(|(_, m)| m.window.clone());
                if let Some(window) = window {
                    self.niri.layout.center_window(Some(&window));
                    // FIXME: granular
                    self.niri.queue_redraw_all();
                }
            }
            Action::CenterVisibleColumns => {
                self.niri.layout.center_visible_columns();
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::MaximizeColumn => {
                self.niri.layout.toggle_full_width();
            }
            Action::MaximizeWindowToEdges => {
                let focus = self.niri.layout.focus().map(|m| m.window.clone());
                if let Some(window) = focus {
                    self.niri.layout.toggle_maximized(&window);
                    // FIXME: granular
                    self.niri.queue_redraw_all();
                }
            }
            Action::MaximizeWindowToEdgesById(id) => {
                let window = self
                    .niri
                    .layout
                    .windows_all()
                    .find(|(_, m)| m.id().get() == id);
                let window = window.map(|(_, m)| m.window.clone());
                if let Some(window) = window {
                    self.niri.layout.toggle_maximized(&window);
                    // FIXME: granular
                    self.niri.queue_redraw_all();
                }
            }
            Action::FocusMonitorLeft => {
                if let Some(output) = self.niri.output_left() {
                    self.niri.layout.focus_output(&output);
                    if !self.maybe_warp_cursor_to_focus_centered() {
                        self.move_cursor_to_output(&output);
                    }
                    self.niri.layer_shell_on_demand_focus = None;
                }
            }
            Action::FocusMonitorRight => {
                if let Some(output) = self.niri.output_right() {
                    self.niri.layout.focus_output(&output);
                    if !self.maybe_warp_cursor_to_focus_centered() {
                        self.move_cursor_to_output(&output);
                    }
                    self.niri.layer_shell_on_demand_focus = None;
                }
            }
            Action::FocusMonitorDown => {
                if let Some(output) = self.niri.output_down() {
                    self.niri.layout.focus_output(&output);
                    if !self.maybe_warp_cursor_to_focus_centered() {
                        self.move_cursor_to_output(&output);
                    }
                    self.niri.layer_shell_on_demand_focus = None;
                }
            }
            Action::FocusMonitorUp => {
                if let Some(output) = self.niri.output_up() {
                    self.niri.layout.focus_output(&output);
                    if !self.maybe_warp_cursor_to_focus_centered() {
                        self.move_cursor_to_output(&output);
                    }
                    self.niri.layer_shell_on_demand_focus = None;
                }
            }
            Action::FocusMonitorPrevious => {
                if let Some(output) = self.niri.output_previous() {
                    self.niri.layout.focus_output(&output);
                    if !self.maybe_warp_cursor_to_focus_centered() {
                        self.move_cursor_to_output(&output);
                    }
                    self.niri.layer_shell_on_demand_focus = None;
                }
            }
            Action::FocusMonitorNext => {
                if let Some(output) = self.niri.output_next() {
                    self.niri.layout.focus_output(&output);
                    if !self.maybe_warp_cursor_to_focus_centered() {
                        self.move_cursor_to_output(&output);
                    }
                    self.niri.layer_shell_on_demand_focus = None;
                }
            }
            Action::FocusMonitor(output) => {
                if let Some(output) = self.niri.output_by_name_match(&output).cloned() {
                    self.niri.layout.focus_output(&output);
                    if !self.maybe_warp_cursor_to_focus_centered() {
                        self.move_cursor_to_output(&output);
                    }
                    self.niri.layer_shell_on_demand_focus = None;
                }
            }
            Action::MoveWindowToMonitorLeft => {
                if let Some(current_output) = self.niri.screenshot_ui.selection_output() {
                    if let Some(target_output) = self.niri.output_left_of(current_output) {
                        self.move_cursor_to_output(&target_output);
                        self.niri.screenshot_ui.move_to_output(target_output);
                    }
                } else if let Some(output) = self.niri.output_left() {
                    self.niri
                        .layout
                        .move_to_output(None, &output, None, ActivateWindow::Smart);
                    self.niri.layout.focus_output(&output);
                    if !self.maybe_warp_cursor_to_focus_centered() {
                        self.move_cursor_to_output(&output);
                    }
                }
            }
            Action::MoveWindowToMonitorRight => {
                if let Some(current_output) = self.niri.screenshot_ui.selection_output() {
                    if let Some(target_output) = self.niri.output_right_of(current_output) {
                        self.move_cursor_to_output(&target_output);
                        self.niri.screenshot_ui.move_to_output(target_output);
                    }
                } else if let Some(output) = self.niri.output_right() {
                    self.niri
                        .layout
                        .move_to_output(None, &output, None, ActivateWindow::Smart);
                    self.niri.layout.focus_output(&output);
                    if !self.maybe_warp_cursor_to_focus_centered() {
                        self.move_cursor_to_output(&output);
                    }
                }
            }
            Action::MoveWindowToMonitorDown => {
                if let Some(current_output) = self.niri.screenshot_ui.selection_output() {
                    if let Some(target_output) = self.niri.output_down_of(current_output) {
                        self.move_cursor_to_output(&target_output);
                        self.niri.screenshot_ui.move_to_output(target_output);
                    }
                } else if let Some(output) = self.niri.output_down() {
                    self.niri
                        .layout
                        .move_to_output(None, &output, None, ActivateWindow::Smart);
                    self.niri.layout.focus_output(&output);
                    if !self.maybe_warp_cursor_to_focus_centered() {
                        self.move_cursor_to_output(&output);
                    }
                }
            }
            Action::MoveWindowToMonitorUp => {
                if let Some(current_output) = self.niri.screenshot_ui.selection_output() {
                    if let Some(target_output) = self.niri.output_up_of(current_output) {
                        self.move_cursor_to_output(&target_output);
                        self.niri.screenshot_ui.move_to_output(target_output);
                    }
                } else if let Some(output) = self.niri.output_up() {
                    self.niri
                        .layout
                        .move_to_output(None, &output, None, ActivateWindow::Smart);
                    self.niri.layout.focus_output(&output);
                    if !self.maybe_warp_cursor_to_focus_centered() {
                        self.move_cursor_to_output(&output);
                    }
                }
            }
            Action::MoveWindowToMonitorPrevious => {
                if let Some(current_output) = self.niri.screenshot_ui.selection_output() {
                    if let Some(target_output) = self.niri.output_previous_of(current_output) {
                        self.move_cursor_to_output(&target_output);
                        self.niri.screenshot_ui.move_to_output(target_output);
                    }
                } else if let Some(output) = self.niri.output_previous() {
                    self.niri
                        .layout
                        .move_to_output(None, &output, None, ActivateWindow::Smart);
                    self.niri.layout.focus_output(&output);
                    if !self.maybe_warp_cursor_to_focus_centered() {
                        self.move_cursor_to_output(&output);
                    }
                }
            }
            Action::MoveWindowToMonitorNext => {
                if let Some(current_output) = self.niri.screenshot_ui.selection_output() {
                    if let Some(target_output) = self.niri.output_next_of(current_output) {
                        self.move_cursor_to_output(&target_output);
                        self.niri.screenshot_ui.move_to_output(target_output);
                    }
                } else if let Some(output) = self.niri.output_next() {
                    self.niri
                        .layout
                        .move_to_output(None, &output, None, ActivateWindow::Smart);
                    self.niri.layout.focus_output(&output);
                    if !self.maybe_warp_cursor_to_focus_centered() {
                        self.move_cursor_to_output(&output);
                    }
                }
            }
            Action::MoveWindowToMonitor(output) => {
                if let Some(output) = self.niri.output_by_name_match(&output).cloned() {
                    if self.niri.screenshot_ui.is_open() {
                        self.move_cursor_to_output(&output);
                        self.niri.screenshot_ui.move_to_output(output);
                    } else {
                        self.niri
                            .layout
                            .move_to_output(None, &output, None, ActivateWindow::Smart);
                        self.niri.layout.focus_output(&output);
                        if !self.maybe_warp_cursor_to_focus_centered() {
                            self.move_cursor_to_output(&output);
                        }
                    }
                }
            }
            Action::MoveWindowToMonitorById { id, output } => {
                if let Some(output) = self.niri.output_by_name_match(&output).cloned() {
                    let window = self
                        .niri
                        .layout
                        .windows_all()
                        .find(|(_, m)| m.id().get() == id);
                    let window = window.map(|(_, m)| m.window.clone());

                    if let Some(window) = window {
                        let target_was_active = self
                            .niri
                            .layout
                            .active_output()
                            .is_some_and(|active| output == *active);

                        self.niri.layout.move_to_output(
                            Some(&window),
                            &output,
                            None,
                            ActivateWindow::Smart,
                        );

                        // If the active output changed (window was moved and focused).
                        #[allow(clippy::collapsible_if)]
                        if !target_was_active && self.niri.layout.active_output() == Some(&output) {
                            if !self.maybe_warp_cursor_to_focus_centered() {
                                self.move_cursor_to_output(&output);
                            }
                        }
                    }
                }
            }
            Action::MoveColumnToMonitorLeft => {
                if let Some(current_output) = self.niri.screenshot_ui.selection_output() {
                    if let Some(target_output) = self.niri.output_left_of(current_output) {
                        self.move_cursor_to_output(&target_output);
                        self.niri.screenshot_ui.move_to_output(target_output);
                    }
                } else if let Some(output) = self.niri.output_left() {
                    self.niri.layout.move_column_to_output(&output, None, true);
                    self.niri.layout.focus_output(&output);
                    if !self.maybe_warp_cursor_to_focus_centered() {
                        self.move_cursor_to_output(&output);
                    }
                }
            }
            Action::MoveColumnToMonitorRight => {
                if let Some(current_output) = self.niri.screenshot_ui.selection_output() {
                    if let Some(target_output) = self.niri.output_right_of(current_output) {
                        self.move_cursor_to_output(&target_output);
                        self.niri.screenshot_ui.move_to_output(target_output);
                    }
                } else if let Some(output) = self.niri.output_right() {
                    self.niri.layout.move_column_to_output(&output, None, true);
                    self.niri.layout.focus_output(&output);
                    if !self.maybe_warp_cursor_to_focus_centered() {
                        self.move_cursor_to_output(&output);
                    }
                }
            }
            Action::MoveColumnToMonitorDown => {
                if let Some(current_output) = self.niri.screenshot_ui.selection_output() {
                    if let Some(target_output) = self.niri.output_down_of(current_output) {
                        self.move_cursor_to_output(&target_output);
                        self.niri.screenshot_ui.move_to_output(target_output);
                    }
                } else if let Some(output) = self.niri.output_down() {
                    self.niri.layout.move_column_to_output(&output, None, true);
                    self.niri.layout.focus_output(&output);
                    if !self.maybe_warp_cursor_to_focus_centered() {
                        self.move_cursor_to_output(&output);
                    }
                }
            }
            Action::MoveColumnToMonitorUp => {
                if let Some(current_output) = self.niri.screenshot_ui.selection_output() {
                    if let Some(target_output) = self.niri.output_up_of(current_output) {
                        self.move_cursor_to_output(&target_output);
                        self.niri.screenshot_ui.move_to_output(target_output);
                    }
                } else if let Some(output) = self.niri.output_up() {
                    self.niri.layout.move_column_to_output(&output, None, true);
                    self.niri.layout.focus_output(&output);
                    if !self.maybe_warp_cursor_to_focus_centered() {
                        self.move_cursor_to_output(&output);
                    }
                }
            }
            Action::MoveColumnToMonitorPrevious => {
                if let Some(current_output) = self.niri.screenshot_ui.selection_output() {
                    if let Some(target_output) = self.niri.output_previous_of(current_output) {
                        self.move_cursor_to_output(&target_output);
                        self.niri.screenshot_ui.move_to_output(target_output);
                    }
                } else if let Some(output) = self.niri.output_previous() {
                    self.niri.layout.move_column_to_output(&output, None, true);
                    self.niri.layout.focus_output(&output);
                    if !self.maybe_warp_cursor_to_focus_centered() {
                        self.move_cursor_to_output(&output);
                    }
                }
            }
            Action::MoveColumnToMonitorNext => {
                if let Some(current_output) = self.niri.screenshot_ui.selection_output() {
                    if let Some(target_output) = self.niri.output_next_of(current_output) {
                        self.move_cursor_to_output(&target_output);
                        self.niri.screenshot_ui.move_to_output(target_output);
                    }
                } else if let Some(output) = self.niri.output_next() {
                    self.niri.layout.move_column_to_output(&output, None, true);
                    self.niri.layout.focus_output(&output);
                    if !self.maybe_warp_cursor_to_focus_centered() {
                        self.move_cursor_to_output(&output);
                    }
                }
            }
            Action::MoveColumnToMonitor(output) => {
                if let Some(output) = self.niri.output_by_name_match(&output).cloned() {
                    if self.niri.screenshot_ui.is_open() {
                        self.move_cursor_to_output(&output);
                        self.niri.screenshot_ui.move_to_output(output);
                    } else {
                        self.niri.layout.move_column_to_output(&output, None, true);
                        self.niri.layout.focus_output(&output);
                        if !self.maybe_warp_cursor_to_focus_centered() {
                            self.move_cursor_to_output(&output);
                        }
                    }
                }
            }
            Action::SetColumnWidth(change) => {
                if self.niri.screenshot_ui.is_open() {
                    self.niri.screenshot_ui.set_width(change);

                    // FIXME: granular
                    self.niri.queue_redraw_all();
                } else {
                    self.niri.layout.set_column_width(change);
                }
            }
            Action::SetWindowWidth(change) => {
                if self.niri.screenshot_ui.is_open() {
                    self.niri.screenshot_ui.set_width(change);

                    // FIXME: granular
                    self.niri.queue_redraw_all();
                } else {
                    self.niri.layout.set_window_width(None, change);
                }
            }
            Action::SetWindowWidthById { id, change } => {
                let window = self
                    .niri
                    .layout
                    .windows_all()
                    .find(|(_, m)| m.id().get() == id);
                let window = window.map(|(_, m)| m.window.clone());
                if let Some(window) = window {
                    self.niri.layout.set_window_width(Some(&window), change);
                }
            }
            Action::SetWindowHeight(change) => {
                if self.niri.screenshot_ui.is_open() {
                    self.niri.screenshot_ui.set_height(change);

                    // FIXME: granular
                    self.niri.queue_redraw_all();
                } else {
                    self.niri.layout.set_window_height(None, change);
                }
            }
            Action::SetWindowHeightById { id, change } => {
                let window = self
                    .niri
                    .layout
                    .windows_all()
                    .find(|(_, m)| m.id().get() == id);
                let window = window.map(|(_, m)| m.window.clone());
                if let Some(window) = window {
                    self.niri.layout.set_window_height(Some(&window), change);
                }
            }
            Action::ResetWindowHeight => {
                self.niri.layout.reset_window_height(None);
            }
            Action::ResetWindowHeightById(id) => {
                let window = self
                    .niri
                    .layout
                    .windows_all()
                    .find(|(_, m)| m.id().get() == id);
                let window = window.map(|(_, m)| m.window.clone());
                if let Some(window) = window {
                    self.niri.layout.reset_window_height(Some(&window));
                }
            }
            Action::ExpandColumnToAvailableWidth => {
                self.niri.layout.expand_column_to_available_width();
            }
            Action::ShowHotkeyOverlay => {
                if self.niri.hotkey_overlay.show() {
                    self.niri.queue_redraw_all();

                    #[cfg(feature = "dbus")]
                    self.niri.a11y_announce_hotkey_overlay();
                }
            }
            Action::MoveWorkspaceToMonitorLeft => {
                if let Some(output) = self.niri.output_left() {
                    self.niri.layout.move_workspace_to_output(&output);
                    if !self.maybe_warp_cursor_to_focus_centered() {
                        self.move_cursor_to_output(&output);
                    }
                }
            }
            Action::MoveWorkspaceToMonitorRight => {
                if let Some(output) = self.niri.output_right() {
                    self.niri.layout.move_workspace_to_output(&output);
                    if !self.maybe_warp_cursor_to_focus_centered() {
                        self.move_cursor_to_output(&output);
                    }
                }
            }
            Action::MoveWorkspaceToMonitorDown => {
                if let Some(output) = self.niri.output_down() {
                    self.niri.layout.move_workspace_to_output(&output);
                    if !self.maybe_warp_cursor_to_focus_centered() {
                        self.move_cursor_to_output(&output);
                    }
                }
            }
            Action::MoveWorkspaceToMonitorUp => {
                if let Some(output) = self.niri.output_up() {
                    self.niri.layout.move_workspace_to_output(&output);
                    if !self.maybe_warp_cursor_to_focus_centered() {
                        self.move_cursor_to_output(&output);
                    }
                }
            }
            Action::MoveWorkspaceToMonitorPrevious => {
                if let Some(output) = self.niri.output_previous() {
                    self.niri.layout.move_workspace_to_output(&output);
                    if !self.maybe_warp_cursor_to_focus_centered() {
                        self.move_cursor_to_output(&output);
                    }
                }
            }
            Action::MoveWorkspaceToMonitorNext => {
                if let Some(output) = self.niri.output_next() {
                    self.niri.layout.move_workspace_to_output(&output);
                    if !self.maybe_warp_cursor_to_focus_centered() {
                        self.move_cursor_to_output(&output);
                    }
                }
            }
            Action::MoveWorkspaceToMonitor(new_output) => {
                if let Some(new_output) = self.niri.output_by_name_match(&new_output).cloned() {
                    if self.niri.layout.move_workspace_to_output(&new_output)
                        && !self.maybe_warp_cursor_to_focus_centered()
                    {
                        self.move_cursor_to_output(&new_output);
                    }
                }
            }
            Action::MoveWorkspaceToMonitorByRef {
                output_name,
                reference,
            } => {
                if let Some((output, old_idx)) =
                    self.niri.find_output_and_workspace_index(reference)
                {
                    if let Some(new_output) = self.niri.output_by_name_match(&output_name).cloned()
                    {
                        if self.niri.layout.move_workspace_to_output_by_id(
                            old_idx,
                            output,
                            &new_output,
                        ) {
                            // Cursor warp already calls `queue_redraw_all`
                            if !self.maybe_warp_cursor_to_focus_centered() {
                                self.move_cursor_to_output(&new_output);
                            }
                        }
                    }
                }
            }
            Action::ToggleWindowFloating => {
                self.niri.layout.toggle_window_floating(None);
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::ToggleWindowFloatingById(id) => {
                let window = self
                    .niri
                    .layout
                    .windows_all()
                    .find(|(_, m)| m.id().get() == id);
                let window = window.map(|(_, m)| m.window.clone());
                if let Some(window) = window {
                    self.niri.layout.toggle_window_floating(Some(&window));
                    // FIXME: granular
                    self.niri.queue_redraw_all();
                }
            }
            Action::MoveWindowToFloating => {
                self.niri.layout.set_window_floating(None, true);
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::MoveWindowToFloatingById(id) => {
                let window = self
                    .niri
                    .layout
                    .windows_all()
                    .find(|(_, m)| m.id().get() == id);
                let window = window.map(|(_, m)| m.window.clone());
                if let Some(window) = window {
                    self.niri.layout.set_window_floating(Some(&window), true);
                    // FIXME: granular
                    self.niri.queue_redraw_all();
                }
            }
            Action::MoveWindowToTiling => {
                self.niri.layout.set_window_floating(None, false);
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::MoveWindowToTilingById(id) => {
                let window = self
                    .niri
                    .layout
                    .windows_all()
                    .find(|(_, m)| m.id().get() == id);
                let window = window.map(|(_, m)| m.window.clone());
                if let Some(window) = window {
                    self.niri.layout.set_window_floating(Some(&window), false);
                    // FIXME: granular
                    self.niri.queue_redraw_all();
                }
            }
            Action::FocusFloating => {
                self.niri.layout.focus_floating();
                self.maybe_warp_cursor_to_focus();
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::FocusTiling => {
                self.niri.layout.focus_tiling();
                self.maybe_warp_cursor_to_focus();
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::SwitchFocusBetweenFloatingAndTiling => {
                self.niri.layout.switch_focus_floating_tiling();
                self.maybe_warp_cursor_to_focus();
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::MoveFloatingWindowById { id, x, y } => {
                let window = if let Some(id) = id {
                    let window = self
                        .niri
                        .layout
                        .windows_all()
                        .find(|(_, m)| m.id().get() == id);
                    let window = window.map(|(_, m)| m.window.clone());
                    if window.is_none() {
                        return Ok(DoActionOutcome::Handled);
                    }
                    window
                } else {
                    None
                };

                self.niri
                    .layout
                    .move_floating_window(window.as_ref(), x, y, true);
                // FIXME: granular
                self.niri.queue_redraw_all();
            }
            Action::ToggleWindowRuleOpacity => {
                let active_window = self
                    .niri
                    .layout
                    .active_workspace_mut()
                    .and_then(|ws| ws.active_window_mut());
                if let Some(window) = active_window {
                    if window.rules().opacity.is_some_and(|o| o != 1.) {
                        window.toggle_ignore_opacity_window_rule();
                        // FIXME: granular
                        self.niri.queue_redraw_all();
                    }
                }
            }
            Action::ToggleWindowRuleOpacityById(id) => {
                let window = self
                    .niri
                    .layout
                    .workspaces_mut()
                    .find_map(|ws| ws.windows_mut().find(|w| w.id().get() == id));
                if let Some(window) = window {
                    if window.rules().opacity.is_some_and(|o| o != 1.) {
                        window.toggle_ignore_opacity_window_rule();
                        // FIXME: granular
                        self.niri.queue_redraw_all();
                    }
                }
            }
            Action::SetDynamicCastWindow => {
                let id = self
                    .niri
                    .layout
                    .active_workspace()
                    .and_then(|ws| ws.active_window())
                    .map(|mapped| mapped.id().get());
                if let Some(id) = id {
                    self.set_dynamic_cast_target(CastTarget::Window { id });
                }
            }
            Action::SetDynamicCastWindowById(id) => {
                let layout = &self.niri.layout;
                if layout
                    .windows_all()
                    .any(|(_, mapped)| mapped.id().get() == id)
                {
                    self.set_dynamic_cast_target(CastTarget::Window { id });
                }
            }
            Action::SetDynamicCastMonitor(output) => {
                let output = match output {
                    None => self.niri.layout.active_output(),
                    Some(name) => self.niri.output_by_name_match(&name),
                };
                if let Some(output) = output {
                    self.set_dynamic_cast_target(CastTarget::output(output));
                }
            }
            Action::ClearDynamicCastTarget => {
                self.set_dynamic_cast_target(CastTarget::Nothing);
            }
            Action::StopCast(session_id) => {
                self.niri.stop_cast(CastSessionId::from(session_id));
            }
            Action::ToggleOverview => {
                self.niri.layout.toggle_overview();
                self.niri.queue_redraw_all();
            }
            Action::OpenOverview => {
                if self.niri.layout.open_overview() {
                    self.niri.queue_redraw_all();
                }
            }
            Action::CloseOverview => {
                if self.niri.layout.close_overview() {
                    self.niri.queue_redraw_all();
                }
            }
            Action::ToggleWindowUrgent(id) => {
                let window = self
                    .niri
                    .layout
                    .workspaces_mut()
                    .find_map(|ws| ws.windows_mut().find(|w| w.id().get() == id));
                if let Some(window) = window {
                    let urgent = window.is_urgent();
                    window.set_urgent(!urgent);
                }
                self.niri.queue_redraw_all();
            }
            Action::SetWindowUrgent(id) => {
                let window = self
                    .niri
                    .layout
                    .workspaces_mut()
                    .find_map(|ws| ws.windows_mut().find(|w| w.id().get() == id));
                if let Some(window) = window {
                    window.set_urgent(true);
                }
                self.niri.queue_redraw_all();
            }
            Action::UnsetWindowUrgent(id) => {
                let window = self
                    .niri
                    .layout
                    .workspaces_mut()
                    .find_map(|ws| ws.windows_mut().find(|w| w.id().get() == id));
                if let Some(window) = window {
                    window.set_urgent(false);
                }
                self.niri.queue_redraw_all();
            }
            Action::SetAppearanceOverride { layer, r#override } => {
                let resolved = match ResolvedAppearanceOverride::try_from(&r#override) {
                    Ok(resolved) => resolved,
                    Err(reason) => {
                        warn!("SetAppearanceOverride: {reason}");
                        return Err(DoActionError::AppearanceOverrideInvalid { reason });
                    }
                };
                self.niri
                    .appearance_override
                    .insert(LayerId(layer), resolved);

                let config = self.niri.config.clone();
                let overrides = flatten(&self.niri.appearance_override);
                self.niri.layout.update_config(&config.borrow(), &overrides);
                self.niri.recompute_window_rules();
                self.niri.queue_redraw_all();
            }
            Action::ClearAppearanceOverride { layer } => {
                self.niri.appearance_override.remove(&LayerId(layer));

                let config = self.niri.config.clone();
                let overrides = flatten(&self.niri.appearance_override);
                self.niri.layout.update_config(&config.borrow(), &overrides);
                self.niri.recompute_window_rules();
                self.niri.queue_redraw_all();
            }
            Action::LoadConfigFile(path) => {
                if let Some(watcher) = &self.niri.config_file_watcher {
                    watcher.load_config(path);
                }
            }
            Action::MruConfirm => {
                self.confirm_mru();
            }
            Action::MruCancel => {
                self.niri.cancel_mru();
            }
            Action::MruAdvance {
                direction,
                scope,
                filter,
            } => {
                if self.niri.window_mru_ui.is_open() {
                    self.niri.window_mru_ui.advance(direction, filter);
                    self.niri.queue_redraw_mru_output();
                } else if self.niri.config.borrow().recent_windows.on {
                    self.niri.mru_apply_keyboard_commit();

                    let config = self.niri.config.borrow();
                    let scope = scope.unwrap_or(self.niri.window_mru_ui.scope());

                    let mut wmru = WindowMru::new(&self.niri);
                    if !wmru.is_empty() {
                        wmru.set_scope(scope);
                        if let Some(filter) = filter {
                            wmru.set_filter(filter);
                        }

                        if let Some(output) = self.niri.layout.active_output() {
                            self.niri.window_mru_ui.open(
                                self.niri.clock.clone(),
                                wmru,
                                output.clone(),
                            );

                            // Only select the *next* window if some window (which should be the
                            // first one) is already focused. If nothing is focused, keep the first
                            // window (which is logically the "previously selected" one).
                            let keep_first = direction == MruDirection::Forward
                                && self.niri.layout.focus().is_none();
                            if !keep_first {
                                self.niri.window_mru_ui.advance(direction, None);
                            }

                            drop(config);
                            self.niri.queue_redraw_all();
                        }
                    }
                }
            }
            Action::MruCloseCurrentWindow => {
                if self.niri.window_mru_ui.is_open() {
                    if let Some(id) = self.niri.window_mru_ui.current_window_id() {
                        if let Some(w) = self.niri.find_window_by_id(id) {
                            if let Some(tl) = w.toplevel() {
                                tl.send_close();
                            }
                        }
                    }
                }
            }
            Action::MruFirst => {
                if self.niri.window_mru_ui.is_open() {
                    self.niri.window_mru_ui.first();
                    self.niri.queue_redraw_mru_output();
                }
            }
            Action::MruLast => {
                if self.niri.window_mru_ui.is_open() {
                    self.niri.window_mru_ui.last();
                    self.niri.queue_redraw_mru_output();
                }
            }
            Action::MruSetScope(scope) => {
                if self.niri.window_mru_ui.is_open() {
                    self.niri.window_mru_ui.set_scope(scope);
                    self.niri.queue_redraw_mru_output();
                }
            }
            Action::MruCycleScope => {
                if self.niri.window_mru_ui.is_open() {
                    self.niri.window_mru_ui.cycle_scope();
                    self.niri.queue_redraw_mru_output();
                }
            }
        }

        Ok(DoActionOutcome::Handled)
    }

    /// Dispatch handler for `Action::ToggleWorkspaceSticky` /
    /// `Action::ToggleWorkspaceStickyByRef`.
    ///
    /// Sticky toggles are not hard-blocked during workspace-switch gestures:
    /// Toggle-on is append-only on views (delegates to `set_workspace_activities`
    /// which handles animation-snap internally); Toggle-off touches no views.
    /// No hard-block gate.
    fn dispatch_toggle_workspace_sticky(
        &mut self,
        reference: Option<WorkspaceReference>,
    ) -> Result<(), DoActionError> {
        let arg_ws_log = reference.clone();
        match self.niri.layout.toggle_workspace_sticky(reference) {
            Ok(outcome) => {
                match outcome {
                    crate::layout::ToggleWorkspaceStickyOutcome::StickyOn {
                        ws_id,
                        active_affected,
                    } => {
                        debug!(
                            "ToggleWorkspaceSticky: {ws_id:?} is_sticky=true \
                             (active_affected={active_affected})"
                        );
                        // Cursor-warp / redraw asymmetry: Toggle-on may flip
                        // workspace visibility in the active activity (when the
                        // symmetric diff touched the active id) — same
                        // precondition as SetWorkspaceActivities. No call:
                        // sticky toggles don't flip Activities.active_id.
                        if active_affected {
                            self.maybe_warp_cursor_to_focus();
                            self.niri.queue_redraw_all();
                        }
                    }
                    crate::layout::ToggleWorkspaceStickyOutcome::StickyOff { ws_id } => {
                        debug!("ToggleWorkspaceSticky: {ws_id:?} is_sticky=false");
                        // Toggle-off never touches views — no cursor-warp / redraw.
                    }
                }
            }
            Err(e) => {
                warn!("ToggleWorkspaceSticky: {e}: workspace={arg_ws_log:?}");
                return Err(DoActionError::ToggleWorkspaceSticky(e));
            }
        }
        Ok(())
    }

    /// Dispatch handler for `Action::SetWorkspaceSticky` /
    /// `Action::SetWorkspaceStickyByRef`.
    ///
    /// Sticky set is not hard-blocked during workspace-switch gestures:
    /// Set is append-only on views (delegates to `set_workspace_activities`
    /// which handles animation-snap internally). No hard-block gate.
    fn dispatch_set_workspace_sticky(
        &mut self,
        reference: Option<WorkspaceReference>,
    ) -> Result<(), DoActionError> {
        let arg_ws_log = reference.clone();
        match self.niri.layout.set_workspace_sticky(reference) {
            Ok((ws_id, active_affected)) => {
                debug!("SetWorkspaceSticky: {ws_id:?} (active_affected={active_affected})");
                // Mirrors SetWorkspaceActivities at input/mod.rs: cursor warp
                // + redraw fire when the symmetric diff touched the active
                // activity. No call: sticky toggles don't flip
                // Activities.active_id.
                if active_affected {
                    self.maybe_warp_cursor_to_focus();
                    self.niri.queue_redraw_all();
                }
            }
            Err(e) => {
                warn!("SetWorkspaceSticky: {e}: workspace={arg_ws_log:?}");
                return Err(DoActionError::SetWorkspaceSticky(e));
            }
        }
        Ok(())
    }

    /// Dispatch handler for `Action::UnsetWorkspaceSticky` /
    /// `Action::UnsetWorkspaceStickyByRef`.
    ///
    /// Sticky unset is not hard-blocked during workspace-switch gestures:
    /// Unset touches no views. No hard-block gate. No cursor-warp /
    /// redraw — visibility for the active activity does not change (precedent:
    /// see the `act_id == active_before` gate in the
    /// `Action::RemoveWorkspaceFromActivity` dispatch arm which only redraws
    /// when the active activity is the removal target). No call: sticky
    /// toggles don't flip Activities.active_id.
    fn dispatch_unset_workspace_sticky(
        &mut self,
        reference: Option<WorkspaceReference>,
    ) -> Result<(), DoActionError> {
        let arg_ws_log = reference.clone();
        match self.niri.layout.unset_workspace_sticky(reference) {
            Ok(ws_id) => {
                debug!("UnsetWorkspaceSticky: {ws_id:?}");
            }
            Err(e) => {
                warn!("UnsetWorkspaceSticky: {e}: workspace={arg_ws_log:?}");
                return Err(DoActionError::UnsetWorkspaceSticky(e));
            }
        }
        Ok(())
    }

    /// Perform the four post-switch steps shared by every dispatch arm that flips
    /// the active activity: cursor warp, layer-shell on-demand focus clear, redraw,
    /// and keyboard-shortcut inhibitor reconciliation.
    ///
    /// This deliberately excludes the switch call itself — callers diverge there
    /// (`Layout::switch_activity`, `Layout::switch_activity_previous`, or the
    /// cascade branch inside `Layout::remove_activity`) — so callers perform their
    /// own switch first, then call this. The inhibitor refresh must run on every
    /// path that flips `Activities.active_id`; centralizing it here means new
    /// activity-flipping arms get it by construction instead of by hand-maintained
    /// instruction.
    fn activity_switch_epilogue(&mut self) {
        self.maybe_warp_cursor_to_focus();
        self.niri.layer_shell_on_demand_focus = None;
        self.niri.queue_redraw_all();
        self.niri
            .refresh_keyboard_shortcut_inhibitors_after_activity_switch();
    }

    /// Switch to `target` and reconcile keyboard-shortcut inhibitor state, without
    /// the cursor warp / redraw steps.
    ///
    /// Used by the `MoveWindowToPoolOutcome::MovedDormant` arms, where warp/redraw
    /// are deferred to the caller — typically a following `focus_window` call. On
    /// the no-active-window path the caller instead falls back to
    /// `maybe_warp_cursor_to_focus()`, so warp coverage holds either way;
    /// duplicating it here would be redundant. See `activity_switch_epilogue` for
    /// the full four-step shape used by dispatch arms with no follow-up focus
    /// call.
    fn switch_activity_and_reconcile(&mut self, target: ActivityId) {
        self.niri.layout.switch_activity(target);
        self.niri.layer_shell_on_demand_focus = None;
        self.niri
            .refresh_keyboard_shortcut_inhibitors_after_activity_switch();
    }

    /// Perform the post-restore cursor-warp, redraw, and inhibitor bookkeeping
    /// the bookmark walk/jump dispatch arms need after a successful jump.
    ///
    /// `prev_output` is the active output *before* the layout mutation. When the
    /// jump crossed a monitor boundary, the cursor is warped to the new output;
    /// otherwise it is warped to the focused window per the standard
    /// `maybe_warp_cursor_to_focus` policy. The layer-shell on-demand focus slot
    /// is cleared unconditionally (any jump terminates an on-demand focus
    /// session), and `switched_activity` gates a reconcile of the
    /// keyboard-shortcut inhibitors to the new activity's visibility set.
    fn post_jump_bookkeeping(&mut self, prev_output: Option<Output>, switched_activity: bool) {
        let new_output = self.niri.layout.active_output().cloned();
        if new_output != prev_output {
            if !self.maybe_warp_cursor_to_focus_centered() {
                self.move_cursor_to_output(
                    &new_output.expect("a jump landed on a window, so an output is active"),
                );
            }
        } else {
            self.maybe_warp_cursor_to_focus();
        }
        self.niri.layer_shell_on_demand_focus = None;
        // FIXME: granular
        self.niri.queue_redraw_all();
        if switched_activity {
            self.niri
                .refresh_keyboard_shortcut_inhibitors_after_activity_switch();
        }
    }

    fn on_pointer_motion<I: InputBackend>(&mut self, event: I::PointerMotionEvent) {
        let was_inside_hot_corner = self.niri.pointer_inside_hot_corner;
        // Any of the early returns here mean that the pointer is not inside the hot corner.
        self.niri.pointer_inside_hot_corner = false;

        // We need an output to be able to move the pointer.
        if self.niri.global_space.outputs().next().is_none() {
            return;
        }

        let serial = SERIAL_COUNTER.next_serial();

        let pointer = self.niri.seat.get_pointer().unwrap();

        let pos = pointer.current_location();

        // We have an output, so we can compute the new location and focus.
        let mut new_pos = pos + event.delta();

        // We received an event for the regular pointer, so show it now.
        self.niri.pointer_visibility = PointerVisibility::Visible;
        self.niri.tablet_cursor_location = None;

        // Check if we have an active pointer constraint.
        //
        // FIXME: ideally this should use the pointer focus with up-to-date global location.
        let mut pointer_confined = None;
        if let Some(under) = &self.niri.pointer_contents.surface {
            // No need to check if the pointer focus surface matches, because here we're checking
            // for an already-active constraint, and the constraint is deactivated when the focused
            // surface changes.
            let pos_within_surface = pos - under.1;

            let mut pointer_locked = false;
            with_pointer_constraint(&under.0, &pointer, |constraint| {
                let Some(constraint) = constraint else { return };
                if !constraint.is_active() {
                    return;
                }

                // Constraint does not apply if not within region.
                if let Some(region) = constraint.region() {
                    if !region.contains(pos_within_surface.to_i32_round()) {
                        return;
                    }
                }

                match &*constraint {
                    PointerConstraint::Locked(_locked) => {
                        pointer_locked = true;
                    }
                    PointerConstraint::Confined(confine) => {
                        pointer_confined = Some((under.clone(), confine.region().cloned()));
                    }
                }
            });

            // If the pointer is locked, only send relative motion.
            if pointer_locked {
                pointer.relative_motion(
                    self,
                    Some(under.clone()),
                    &RelativeMotionEvent {
                        delta: event.delta(),
                        delta_unaccel: event.delta_unaccel(),
                        utime: event.time(),
                    },
                );

                pointer.frame(self);

                // I guess a redraw to hide the tablet cursor could be nice? Doesn't matter too
                // much here I think.
                return;
            }
        }

        // Warp pointer across the screen during the spatial movement grabs.
        let spatial_grab = pointer.with_grab(|_, grab| {
            let grab = grab.as_any();
            if let Some(grab) = grab.downcast_ref::<SpatialMovementGrab>() {
                if let Some(output) = grab.view_offset_output() {
                    return Some((output.clone(), true));
                } else if let Some(output) = grab.workspace_switch_output() {
                    return Some((output.clone(), false));
                }
            } else if let Some(grab) = grab.downcast_ref::<MoveGrab>() {
                if let Some(output) = grab.view_offset_output() {
                    return Some((output.clone(), true));
                }
            }
            None
        });
        if let Some((output, horizontal)) = spatial_grab.flatten() {
            if let Some(geo) = self.niri.global_space.output_geometry(&output) {
                let geo = geo.to_f64();
                if horizontal {
                    new_pos.x = (new_pos.x - geo.loc.x).rem_euclid(geo.size.w) + geo.loc.x;
                    new_pos.y = new_pos.y.clamp(geo.loc.y, geo.loc.y + geo.size.h - 1.);
                } else {
                    new_pos.x = new_pos.x.clamp(geo.loc.x, geo.loc.x + geo.size.w - 1.);
                    new_pos.y = (new_pos.y - geo.loc.y).rem_euclid(geo.size.h) + geo.loc.y;
                }
            }
        }

        if self
            .niri
            .global_space
            .output_under(new_pos)
            .next()
            .is_none()
        {
            // We ended up outside the outputs and need to clip the movement.
            if let Some(output) = self.niri.global_space.output_under(pos).next() {
                // The pointer was previously on some output. Clip the movement against its
                // boundaries.
                let geom = self.niri.global_space.output_geometry(output).unwrap();
                new_pos.x = new_pos
                    .x
                    .clamp(geom.loc.x as f64, (geom.loc.x + geom.size.w - 1) as f64);
                new_pos.y = new_pos
                    .y
                    .clamp(geom.loc.y as f64, (geom.loc.y + geom.size.h - 1) as f64);
            } else {
                // The pointer was not on any output in the first place. Find one for it.
                // Let's do the simple thing and just put it on the first output.
                let output = self.niri.global_space.outputs().next().unwrap();
                let geom = self.niri.global_space.output_geometry(output).unwrap();
                new_pos = center(geom).to_f64();
            }
        }

        if let Some(output) = self.niri.screenshot_ui.selection_output() {
            let geom = self.niri.global_space.output_geometry(output).unwrap();
            let point = (new_pos - geom.loc.to_f64())
                .to_physical(output.current_scale().fractional_scale())
                .to_i32_round::<i32>();

            self.niri.screenshot_ui.pointer_motion(point, None);
        }

        if let Some(mru_output) = self.niri.window_mru_ui.output() {
            if let Some((output, pos_within_output)) = self.niri.output_under(new_pos) {
                if mru_output == output {
                    self.niri.window_mru_ui.pointer_motion(pos_within_output);
                }
            }
        }

        let under = self.niri.contents_under(new_pos);

        // Handle confined pointer.
        if let Some((focus_surface, region)) = pointer_confined {
            let mut prevent = false;

            // Prevent the pointer from leaving the focused surface.
            if Some(&focus_surface.0) != under.surface.as_ref().map(|(s, _)| s) {
                prevent = true;
            }

            // Prevent the pointer from leaving the confine region, if any.
            if let Some(region) = region {
                let new_pos_within_surface = new_pos - focus_surface.1;
                if !region.contains(new_pos_within_surface.to_i32_round()) {
                    prevent = true;
                }
            }

            if prevent {
                pointer.relative_motion(
                    self,
                    Some(focus_surface),
                    &RelativeMotionEvent {
                        delta: event.delta(),
                        delta_unaccel: event.delta_unaccel(),
                        utime: event.time(),
                    },
                );

                pointer.frame(self);

                return;
            }
        }

        self.niri.handle_focus_follows_mouse(&under);

        self.niri.pointer_contents.clone_from(&under);

        pointer.motion(
            self,
            under.surface.clone(),
            &MotionEvent {
                location: new_pos,
                serial,
                time: event.time_msec(),
            },
        );

        pointer.relative_motion(
            self,
            under.surface,
            &RelativeMotionEvent {
                delta: event.delta(),
                delta_unaccel: event.delta_unaccel(),
                utime: event.time(),
            },
        );

        pointer.frame(self);

        // contents_under() will return no surface when the hot corner should trigger, so
        // pointer.motion() will set the current focus to None.
        if under.hot_corner && pointer.current_focus().is_none() {
            if !was_inside_hot_corner
                && pointer
                    .with_grab(|_, grab| grab_allows_hot_corner(grab))
                    .unwrap_or(true)
            {
                self.niri.layout.toggle_overview();
            }
            self.niri.pointer_inside_hot_corner = true;
        }

        // Activate a new confinement if necessary.
        self.niri.maybe_activate_pointer_constraint();

        // Inform the layout of an ongoing DnD operation.
        let is_dnd_grab = pointer
            .with_grab(|_, grab| Self::is_dnd_grab(grab.as_any()))
            .unwrap_or(false);
        if is_dnd_grab {
            if let Some((output, pos_within_output)) = self.niri.output_under(new_pos) {
                let output = output.clone();
                self.niri.layout.dnd_update(output, pos_within_output);
            }
        }

        // Redraw to update the cursor position.
        // FIXME: redraw only outputs overlapping the cursor.
        self.niri.queue_redraw_all();
    }

    fn on_pointer_motion_absolute<I: InputBackend>(
        &mut self,
        event: I::PointerMotionAbsoluteEvent,
    ) {
        let was_inside_hot_corner = self.niri.pointer_inside_hot_corner;
        // Any of the early returns here mean that the pointer is not inside the hot corner.
        self.niri.pointer_inside_hot_corner = false;

        let Some(pos) = self.compute_absolute_location(&event, None).or_else(|| {
            self.global_bounding_rectangle().map(|output_geo| {
                event.position_transformed(output_geo.size) + output_geo.loc.to_f64()
            })
        }) else {
            return;
        };

        let serial = SERIAL_COUNTER.next_serial();

        let pointer = self.niri.seat.get_pointer().unwrap();

        if let Some(output) = self.niri.screenshot_ui.selection_output() {
            let geom = self.niri.global_space.output_geometry(output).unwrap();
            let point = (pos - geom.loc.to_f64())
                .to_physical(output.current_scale().fractional_scale())
                .to_i32_round::<i32>();

            self.niri.screenshot_ui.pointer_motion(point, None);
        }

        if let Some(mru_output) = self.niri.window_mru_ui.output() {
            if let Some((output, pos_within_output)) = self.niri.output_under(pos) {
                if mru_output == output {
                    self.niri.window_mru_ui.pointer_motion(pos_within_output);
                }
            }
        }

        let under = self.niri.contents_under(pos);

        self.niri.handle_focus_follows_mouse(&under);

        self.niri.pointer_contents.clone_from(&under);

        pointer.motion(
            self,
            under.surface,
            &MotionEvent {
                location: pos,
                serial,
                time: event.time_msec(),
            },
        );

        pointer.frame(self);

        // contents_under() will return no surface when the hot corner should trigger, so
        // pointer.motion() will set the current focus to None.
        if under.hot_corner && pointer.current_focus().is_none() {
            if !was_inside_hot_corner
                && pointer
                    .with_grab(|_, grab| grab_allows_hot_corner(grab))
                    .unwrap_or(true)
            {
                self.niri.layout.toggle_overview();
            }
            self.niri.pointer_inside_hot_corner = true;
        }

        self.niri.maybe_activate_pointer_constraint();

        // We moved the pointer, show it.
        self.niri.pointer_visibility = PointerVisibility::Visible;

        // We moved the regular pointer, so show it now.
        self.niri.tablet_cursor_location = None;

        // Inform the layout of an ongoing DnD operation.
        let is_dnd_grab = pointer
            .with_grab(|_, grab| Self::is_dnd_grab(grab.as_any()))
            .unwrap_or(false);
        if is_dnd_grab {
            if let Some((output, pos_within_output)) = self.niri.output_under(pos) {
                let output = output.clone();
                self.niri.layout.dnd_update(output, pos_within_output);
            }
        }

        // Redraw to update the cursor position.
        // FIXME: redraw only outputs overlapping the cursor.
        self.niri.queue_redraw_all();
    }

    fn on_pointer_button<I: InputBackend>(&mut self, event: I::PointerButtonEvent) {
        let pointer = self.niri.seat.get_pointer().unwrap();

        let serial = SERIAL_COUNTER.next_serial();

        let button = event.button();

        let button_code = event.button_code();

        let button_state = event.state();

        let mod_key = self.backend.mod_key(&self.niri.config.borrow());

        // Ignore release events for mouse clicks that triggered a bind.
        if self.niri.suppressed_buttons.remove(&button_code) {
            return;
        }

        let mods = self.niri.seat.get_keyboard().unwrap().modifier_state();
        let modifiers = modifiers_from_state(mods);
        let mod_down = modifiers.contains(mod_key.to_modifiers());

        if ButtonState::Pressed == button_state {
            let mut is_mru_open = false;
            if let Some(mru_output) = self.niri.window_mru_ui.output() {
                is_mru_open = true;
                if let Some(MouseButton::Left) = button {
                    let location = pointer.current_location();
                    let (output, pos_within_output) = self.niri.output_under(location).unwrap();
                    if mru_output == output {
                        let id = self.niri.window_mru_ui.pointer_motion(pos_within_output);
                        if id.is_some() {
                            self.confirm_mru();
                        } else {
                            self.niri.cancel_mru();
                        }
                    } else {
                        self.niri.cancel_mru();
                    }

                    self.niri.suppressed_buttons.insert(button_code);
                    return;
                }
            }

            if is_mru_open || self.niri.mods_with_mouse_binds.contains(&modifiers) {
                if let Some(bind) = match button {
                    Some(MouseButton::Left) => Some(Trigger::MouseLeft),
                    Some(MouseButton::Right) => Some(Trigger::MouseRight),
                    Some(MouseButton::Middle) => Some(Trigger::MouseMiddle),
                    Some(MouseButton::Back) => Some(Trigger::MouseBack),
                    Some(MouseButton::Forward) => Some(Trigger::MouseForward),
                    _ => None,
                }
                .and_then(|trigger| {
                    let config = self.niri.config.borrow();
                    let bindings = make_binds_iter(
                        &config,
                        &mut self.niri.window_mru_ui,
                        modifiers,
                        &self.niri.bookmark_binds,
                    );
                    find_configured_bind(bindings, mod_key, trigger, mods)
                })
                .filter(|bind| {
                    !self.niri.screenshot_ui.is_open() || allowed_during_screenshot(&bind.action)
                }) {
                    self.niri.suppressed_buttons.insert(button_code);
                    self.handle_bind(bind.clone());
                    return;
                };
            }

            // We received an event for the regular pointer, so show it now.
            self.niri.pointer_visibility = PointerVisibility::Visible;
            self.niri.tablet_cursor_location = None;

            let is_overview_open = self.niri.layout.is_overview_open();

            if is_overview_open && !pointer.is_grabbed() && button == Some(MouseButton::Right) {
                if let Some((output, ws)) = self.niri.workspace_under_cursor(true) {
                    let ws_id = ws.id();
                    let ws_idx = self.niri.layout.find_workspace_by_id(ws_id).unwrap().0;

                    self.niri.layout.focus_output(&output);

                    let location = pointer.current_location();
                    let start_data = PointerGrabStartData {
                        focus: None,
                        button: button_code,
                        location,
                    };
                    self.niri
                        .layout
                        .view_offset_gesture_begin(&output, Some(ws_idx), false);
                    let grab = SpatialMovementGrab::new(start_data, output, ws_id, true);
                    pointer.set_grab(self, grab, serial, Focus::Clear);
                    self.niri
                        .cursor_manager
                        .set_cursor_image(CursorImageStatus::Named(CursorIcon::AllScroll));

                    // FIXME: granular.
                    self.niri.queue_redraw_all();
                    return;
                }
            }

            if button == Some(MouseButton::Middle) && !pointer.is_grabbed() && mod_down {
                let output_ws = if is_overview_open {
                    self.niri.workspace_under_cursor(true)
                } else {
                    // We don't want to accidentally "catch" the wrong workspace during
                    // animations.
                    let layout = &self.niri.layout;
                    self.niri.output_under_cursor().and_then(|output| {
                        let mon = layout.monitor_for_output(&output)?;
                        let view = layout.active_view(&mon.output_id());
                        Some((output, view.active_workspace_ref(layout.workspace_pool())))
                    })
                };

                if let Some((output, ws)) = output_ws {
                    let ws_id = ws.id();

                    self.niri.layout.focus_output(&output);

                    let location = pointer.current_location();
                    let start_data = PointerGrabStartData {
                        focus: None,
                        button: button_code,
                        location,
                    };
                    let grab = SpatialMovementGrab::new(start_data, output, ws_id, false);
                    pointer.set_grab(self, grab, serial, Focus::Clear);
                    self.niri
                        .cursor_manager
                        .set_cursor_image(CursorImageStatus::Named(CursorIcon::AllScroll));

                    // FIXME: granular.
                    self.niri.queue_redraw_all();

                    // Don't activate the window under the cursor to avoid unnecessary
                    // scrolling when e.g. Mod+MMB clicking on a partially off-screen window.
                    return;
                }
            }

            if let Some(mapped) = self.niri.window_under_cursor() {
                let window = mapped.window.clone();

                // Check if we need to start an interactive move.
                if button == Some(MouseButton::Left) && !pointer.is_grabbed() {
                    if is_overview_open || mod_down {
                        let location = pointer.current_location();

                        if !is_overview_open {
                            self.niri.layout.activate_window(&window);
                        }

                        let start_data = PointerGrabStartData {
                            focus: None,
                            button: button_code,
                            location,
                        };
                        let start_data = PointerOrTouchStartData::Pointer(start_data);
                        let icon = CursorIcon::Grabbing;
                        if let Some(grab) =
                            MoveGrab::new(self, start_data, window.clone(), false, Some(icon))
                        {
                            pointer.set_grab(self, grab, serial, Focus::Clear);

                            // Set the cursor to Grabbing right away for Mod+LMB since it doesn't
                            // do any other gesture.
                            //
                            // In the overview, we click to activate window and close the overview,
                            // in this case setting the cursor right away would be distracting.
                            if !is_overview_open {
                                self.niri
                                    .cursor_manager
                                    .set_cursor_image(CursorImageStatus::Named(icon));
                            }
                        }
                    }
                }
                // Check if we need to start an interactive resize.
                else if button == Some(MouseButton::Right) && !pointer.is_grabbed() && mod_down {
                    let location = pointer.current_location();
                    let (output, pos_within_output) = self.niri.output_under(location).unwrap();
                    let edges = self
                        .niri
                        .layout
                        .resize_edges_under(output, pos_within_output)
                        .unwrap_or(ResizeEdge::empty());

                    if !edges.is_empty() {
                        // See if we got a double resize-click gesture.
                        // FIXME: deduplicate with resize_request in xdg-shell somehow.
                        let time = get_monotonic_time();
                        let last_cell = mapped.last_interactive_resize_start();
                        let mut last = last_cell.get();
                        last_cell.set(Some((time, edges)));

                        // Floating windows don't have either of the double-resize-click
                        // gestures, so just allow it to resize.
                        if mapped.is_floating() {
                            last = None;
                            last_cell.set(None);
                        }

                        if let Some((last_time, last_edges)) = last {
                            if time.saturating_sub(last_time) <= DOUBLE_CLICK_TIME {
                                // Allow quick resize after a triple click.
                                last_cell.set(None);

                                let intersection = edges.intersection(last_edges);
                                if intersection.intersects(ResizeEdge::LEFT_RIGHT) {
                                    // FIXME: don't activate once we can pass specific windows
                                    // to actions.
                                    self.niri.layout.activate_window(&window);
                                    self.niri.layout.toggle_full_width();
                                }
                                if intersection.intersects(ResizeEdge::TOP_BOTTOM) {
                                    self.niri.layout.activate_window(&window);
                                    self.niri.layout.reset_window_height(Some(&window));
                                }
                                // FIXME: granular.
                                self.niri.queue_redraw_all();
                                return;
                            }
                        }

                        self.niri.layout.activate_window(&window);

                        if self
                            .niri
                            .layout
                            .interactive_resize_begin(window.clone(), edges)
                        {
                            let start_data = PointerGrabStartData {
                                focus: None,
                                button: button_code,
                                location,
                            };
                            let grab = ResizeGrab::new(start_data, window.clone());
                            pointer.set_grab(self, grab, serial, Focus::Clear);
                            self.niri
                                .cursor_manager
                                .set_cursor_image(CursorImageStatus::Named(edges.cursor_icon()));
                        }
                    }
                }

                if !is_overview_open {
                    self.niri.layout.activate_window(&window);
                }

                // FIXME: granular.
                self.niri.queue_redraw_all();
            } else if let Some((output, ws)) = is_overview_open
                .then(|| self.niri.workspace_under_cursor(false))
                .flatten()
            {
                let ws_idx = self.niri.layout.find_workspace_by_id(ws.id()).unwrap().0;

                self.niri.layout.focus_output(&output);
                self.niri.layout.toggle_overview_to_workspace(ws_idx);

                // FIXME: granular.
                self.niri.queue_redraw_all();
            } else if let Some(output) = self.niri.output_under_cursor() {
                self.niri.layout.focus_output(&output);

                // FIXME: granular.
                self.niri.queue_redraw_all();
            }
        };

        self.update_pointer_contents();

        if ButtonState::Pressed == button_state {
            let layer_under = self.niri.pointer_contents.layer.clone();
            self.niri.focus_layer_surface_if_on_demand(layer_under);
        }

        if button == Some(MouseButton::Left) && self.niri.screenshot_ui.is_open() {
            if button_state == ButtonState::Pressed {
                let pos = pointer.current_location();

                // If we'll be moving the existing selection, use the selection output.
                let output = if mod_down {
                    self.niri.screenshot_ui.selection_output()
                } else {
                    self.niri.output_under(pos).map(|(out, _)| out)
                };

                if let Some(output) = output.cloned() {
                    let geom = self.niri.global_space.output_geometry(&output).unwrap();
                    let point = (pos - geom.loc.to_f64())
                        .to_physical(output.current_scale().fractional_scale())
                        .to_i32_round();

                    if self
                        .niri
                        .screenshot_ui
                        .pointer_down(output, point, None, mod_down)
                    {
                        self.niri.queue_redraw_all();
                    }
                }
            } else if let Some(capture) = self.niri.screenshot_ui.pointer_up(None) {
                if capture {
                    self.confirm_screenshot(true);
                } else {
                    self.niri.queue_redraw_all();
                }
            }
        }

        pointer.button(
            self,
            &ButtonEvent {
                button: button_code,
                state: button_state,
                serial,
                time: event.time_msec(),
            },
        );
        pointer.frame(self);
    }

    fn on_pointer_axis<I: InputBackend>(&mut self, event: I::PointerAxisEvent) {
        let pointer = &self.niri.seat.get_pointer().unwrap();

        let source = event.source();

        let mod_key = self.backend.mod_key(&self.niri.config.borrow());

        // We received an event for the regular pointer, so show it now. This is also needed for
        // update_pointer_contents() below to return the real contents, necessary for the pointer
        // axis event to reach the window.
        self.niri.pointer_visibility = PointerVisibility::Visible;
        self.niri.tablet_cursor_location = None;

        let timestamp = Duration::from_micros(event.time());

        let horizontal_amount_v120 = event.amount_v120(Axis::Horizontal);
        let vertical_amount_v120 = event.amount_v120(Axis::Vertical);

        let is_overview_open = self.niri.layout.is_overview_open();

        // We should only handle scrolling in the overview if the pointer is not over a (top or
        // overlay) layer surface.
        let should_handle_in_overview = if is_overview_open {
            // FIXME: ideally this should happen after updating the pointer contents, which happens
            // below. However, our pointer actions are supposed to act on the old surface, before
            // updating the pointer contents.
            pointer
                .current_focus()
                .map(|surface| self.niri.find_root_shell_surface(&surface))
                .is_none_or(|root| {
                    !self
                        .niri
                        .mapped_layer_surfaces
                        .keys()
                        .any(|layer| *layer.wl_surface() == root)
                })
        } else {
            false
        };

        let is_mru_open = self.niri.window_mru_ui.is_open();

        // Handle wheel scroll bindings.
        if source == AxisSource::Wheel {
            // If we have a scroll bind with current modifiers, then accumulate and don't pass to
            // Wayland. If there's no bind, reset the accumulator.
            let mods = self.niri.seat.get_keyboard().unwrap().modifier_state();
            let modifiers = modifiers_from_state(mods);
            let should_handle = should_handle_in_overview
                || is_mru_open
                || self.niri.mods_with_wheel_binds.contains(&modifiers);
            if should_handle {
                let horizontal = horizontal_amount_v120.unwrap_or(0.);
                let ticks = self.niri.horizontal_wheel_tracker.accumulate(horizontal);
                if ticks != 0 {
                    let (bind_left, bind_right) =
                        if should_handle_in_overview && modifiers.is_empty() {
                            let bind_left = Some(Bind {
                                key: Key {
                                    trigger: Trigger::WheelScrollLeft,
                                    modifiers: Modifiers::empty(),
                                },
                                action: Action::FocusColumnLeftUnderMouse,
                                repeat: true,
                                cooldown: None,
                                allow_when_locked: false,
                                allow_inhibiting: false,
                                hotkey_overlay_title: None,
                            });
                            let bind_right = Some(Bind {
                                key: Key {
                                    trigger: Trigger::WheelScrollRight,
                                    modifiers: Modifiers::empty(),
                                },
                                action: Action::FocusColumnRightUnderMouse,
                                repeat: true,
                                cooldown: None,
                                allow_when_locked: false,
                                allow_inhibiting: false,
                                hotkey_overlay_title: None,
                            });
                            (bind_left, bind_right)
                        } else {
                            let config = self.niri.config.borrow();
                            let bindings = make_binds_iter(
                                &config,
                                &mut self.niri.window_mru_ui,
                                modifiers,
                                &self.niri.bookmark_binds,
                            );
                            let bind_left = find_configured_bind(
                                bindings.clone(),
                                mod_key,
                                Trigger::WheelScrollLeft,
                                mods,
                            )
                            .filter(|bind| {
                                !self.niri.screenshot_ui.is_open()
                                    || allowed_during_screenshot(&bind.action)
                            });
                            let bind_right = find_configured_bind(
                                bindings,
                                mod_key,
                                Trigger::WheelScrollRight,
                                mods,
                            )
                            .filter(|bind| {
                                !self.niri.screenshot_ui.is_open()
                                    || allowed_during_screenshot(&bind.action)
                            });
                            (bind_left, bind_right)
                        };

                    if let Some(right) = bind_right {
                        for _ in 0..ticks {
                            self.handle_bind(right.clone());
                        }
                    }
                    if let Some(left) = bind_left {
                        for _ in ticks..0 {
                            self.handle_bind(left.clone());
                        }
                    }
                }

                let vertical = vertical_amount_v120.unwrap_or(0.);
                let ticks = self.niri.vertical_wheel_tracker.accumulate(vertical);
                if ticks != 0 {
                    let (bind_up, bind_down) = if should_handle_in_overview && modifiers.is_empty()
                    {
                        let bind_up = Some(Bind {
                            key: Key {
                                trigger: Trigger::WheelScrollUp,
                                modifiers: Modifiers::empty(),
                            },
                            action: Action::FocusWorkspaceUpUnderMouse,
                            repeat: true,
                            cooldown: Some(Duration::from_millis(50)),
                            allow_when_locked: false,
                            allow_inhibiting: false,
                            hotkey_overlay_title: None,
                        });
                        let bind_down = Some(Bind {
                            key: Key {
                                trigger: Trigger::WheelScrollDown,
                                modifiers: Modifiers::empty(),
                            },
                            action: Action::FocusWorkspaceDownUnderMouse,
                            repeat: true,
                            cooldown: Some(Duration::from_millis(50)),
                            allow_when_locked: false,
                            allow_inhibiting: false,
                            hotkey_overlay_title: None,
                        });
                        (bind_up, bind_down)
                    } else if should_handle_in_overview && modifiers == Modifiers::SHIFT {
                        let bind_up = Some(Bind {
                            key: Key {
                                trigger: Trigger::WheelScrollUp,
                                modifiers: Modifiers::empty(),
                            },
                            action: Action::FocusColumnLeftUnderMouse,
                            repeat: true,
                            cooldown: Some(Duration::from_millis(50)),
                            allow_when_locked: false,
                            allow_inhibiting: false,
                            hotkey_overlay_title: None,
                        });
                        let bind_down = Some(Bind {
                            key: Key {
                                trigger: Trigger::WheelScrollDown,
                                modifiers: Modifiers::empty(),
                            },
                            action: Action::FocusColumnRightUnderMouse,
                            repeat: true,
                            cooldown: Some(Duration::from_millis(50)),
                            allow_when_locked: false,
                            allow_inhibiting: false,
                            hotkey_overlay_title: None,
                        });
                        (bind_up, bind_down)
                    } else {
                        let config = self.niri.config.borrow();
                        let bindings = make_binds_iter(
                            &config,
                            &mut self.niri.window_mru_ui,
                            modifiers,
                            &self.niri.bookmark_binds,
                        );
                        let bind_up = find_configured_bind(
                            bindings.clone(),
                            mod_key,
                            Trigger::WheelScrollUp,
                            mods,
                        )
                        .filter(|bind| {
                            !self.niri.screenshot_ui.is_open()
                                || allowed_during_screenshot(&bind.action)
                        });
                        let bind_down =
                            find_configured_bind(bindings, mod_key, Trigger::WheelScrollDown, mods)
                                .filter(|bind| {
                                    !self.niri.screenshot_ui.is_open()
                                        || allowed_during_screenshot(&bind.action)
                                });
                        (bind_up, bind_down)
                    };

                    if let Some(down) = bind_down {
                        for _ in 0..ticks {
                            self.handle_bind(down.clone());
                        }
                    }
                    if let Some(up) = bind_up {
                        for _ in ticks..0 {
                            self.handle_bind(up.clone());
                        }
                    }
                }

                return;
            } else {
                self.niri.horizontal_wheel_tracker.reset();
                self.niri.vertical_wheel_tracker.reset();
            }
        }

        let horizontal_amount = event.amount(Axis::Horizontal);
        let vertical_amount = event.amount(Axis::Vertical);

        // Handle touchpad and continuous scroll bindings.
        if source == AxisSource::Finger || source == AxisSource::Continuous {
            let mods = self.niri.seat.get_keyboard().unwrap().modifier_state();
            let modifiers = modifiers_from_state(mods);

            let horizontal = horizontal_amount.unwrap_or(0.);
            let vertical = vertical_amount.unwrap_or(0.);

            if should_handle_in_overview && modifiers.is_empty() {
                let mut redraw = false;

                let action = self
                    .niri
                    .overview_scroll_swipe_gesture
                    .update(horizontal, vertical);
                let is_vertical = self.niri.overview_scroll_swipe_gesture.is_vertical();

                if action.end() {
                    if is_vertical {
                        redraw |= self
                            .niri
                            .layout
                            .workspace_switch_gesture_end(Some(true))
                            .is_some();
                    } else {
                        redraw |= self
                            .niri
                            .layout
                            .view_offset_gesture_end(Some(true))
                            .is_some();
                    }
                } else {
                    // Maybe begin, then update.
                    if is_vertical {
                        if action.begin() {
                            if let Some(output) = self.niri.output_under_cursor() {
                                self.niri
                                    .layout
                                    .workspace_switch_gesture_begin(&output, true);
                                redraw = true;
                            }
                        }

                        let res = self
                            .niri
                            .layout
                            .workspace_switch_gesture_update(vertical, timestamp, true);
                        if let Some(Some(_)) = res {
                            redraw = true;
                        }
                    } else {
                        if action.begin() {
                            if let Some((output, ws)) = self.niri.workspace_under_cursor(true) {
                                let ws_id = ws.id();
                                let ws_idx =
                                    self.niri.layout.find_workspace_by_id(ws_id).unwrap().0;

                                self.niri.layout.view_offset_gesture_begin(
                                    &output,
                                    Some(ws_idx),
                                    true,
                                );
                                redraw = true;
                            }
                        }

                        let res = self
                            .niri
                            .layout
                            .view_offset_gesture_update(horizontal, timestamp, true);
                        if let Some(Some(_)) = res {
                            redraw = true;
                        }
                    }
                }

                if redraw {
                    self.niri.queue_redraw_all();
                }

                return;
            } else {
                let mut redraw = false;
                if self.niri.overview_scroll_swipe_gesture.reset() {
                    if self.niri.overview_scroll_swipe_gesture.is_vertical() {
                        redraw |= self
                            .niri
                            .layout
                            .workspace_switch_gesture_end(Some(true))
                            .is_some();
                    } else {
                        redraw |= self
                            .niri
                            .layout
                            .view_offset_gesture_end(Some(true))
                            .is_some();
                    }
                }
                if redraw {
                    self.niri.queue_redraw_all();
                }
            }

            if is_mru_open || self.niri.mods_with_finger_scroll_binds.contains(&modifiers) {
                let ticks = self
                    .niri
                    .horizontal_finger_scroll_tracker
                    .accumulate(horizontal);
                if ticks != 0 {
                    let config = self.niri.config.borrow();
                    let bindings = make_binds_iter(
                        &config,
                        &mut self.niri.window_mru_ui,
                        modifiers,
                        &self.niri.bookmark_binds,
                    );
                    let bind_left = find_configured_bind(
                        bindings.clone(),
                        mod_key,
                        Trigger::TouchpadScrollLeft,
                        mods,
                    )
                    .filter(|bind| {
                        !self.niri.screenshot_ui.is_open()
                            || allowed_during_screenshot(&bind.action)
                    });
                    let bind_right =
                        find_configured_bind(bindings, mod_key, Trigger::TouchpadScrollRight, mods)
                            .filter(|bind| {
                                !self.niri.screenshot_ui.is_open()
                                    || allowed_during_screenshot(&bind.action)
                            });
                    drop(config);

                    if let Some(right) = bind_right {
                        for _ in 0..ticks {
                            self.handle_bind(right.clone());
                        }
                    }
                    if let Some(left) = bind_left {
                        for _ in ticks..0 {
                            self.handle_bind(left.clone());
                        }
                    }
                }

                let ticks = self
                    .niri
                    .vertical_finger_scroll_tracker
                    .accumulate(vertical);
                if ticks != 0 {
                    let config = self.niri.config.borrow();
                    let bindings = make_binds_iter(
                        &config,
                        &mut self.niri.window_mru_ui,
                        modifiers,
                        &self.niri.bookmark_binds,
                    );
                    let bind_up = find_configured_bind(
                        bindings.clone(),
                        mod_key,
                        Trigger::TouchpadScrollUp,
                        mods,
                    )
                    .filter(|bind| {
                        !self.niri.screenshot_ui.is_open()
                            || allowed_during_screenshot(&bind.action)
                    });
                    let bind_down =
                        find_configured_bind(bindings, mod_key, Trigger::TouchpadScrollDown, mods)
                            .filter(|bind| {
                                !self.niri.screenshot_ui.is_open()
                                    || allowed_during_screenshot(&bind.action)
                            });
                    drop(config);

                    if let Some(down) = bind_down {
                        for _ in 0..ticks {
                            self.handle_bind(down.clone());
                        }
                    }
                    if let Some(up) = bind_up {
                        for _ in ticks..0 {
                            self.handle_bind(up.clone());
                        }
                    }
                }

                return;
            } else {
                self.niri.horizontal_finger_scroll_tracker.reset();
                self.niri.vertical_finger_scroll_tracker.reset();
            }
        }

        self.update_pointer_contents();

        let device_scroll_factor = {
            let config = self.niri.config.borrow();
            match source {
                AxisSource::Wheel => config.input.mouse.scroll_factor,
                AxisSource::Finger => config.input.touchpad.scroll_factor,
                _ => None,
            }
        };

        // Get window-specific scroll factor
        let window_scroll_factor = pointer
            .current_focus()
            .map(|focused| self.niri.find_root_shell_surface(&focused))
            .and_then(|root| self.niri.layout.find_window_and_output(&root).unzip().0)
            .and_then(|window| window.rules().scroll_factor)
            .unwrap_or(1.);

        // Determine final scroll factors based on configuration
        let (horizontal_factor, vertical_factor) = device_scroll_factor
            .map(|x| x.h_v_factors())
            .unwrap_or((1.0, 1.0));
        let (horizontal_factor, vertical_factor) = (
            horizontal_factor * window_scroll_factor,
            vertical_factor * window_scroll_factor,
        );

        let horizontal_amount = horizontal_amount.unwrap_or_else(|| {
            // Winit backend, discrete scrolling.
            horizontal_amount_v120.unwrap_or(0.0) / 120. * 15.
        }) * horizontal_factor;

        let vertical_amount = vertical_amount.unwrap_or_else(|| {
            // Winit backend, discrete scrolling.
            vertical_amount_v120.unwrap_or(0.0) / 120. * 15.
        }) * vertical_factor;

        let horizontal_amount_v120 = horizontal_amount_v120.map(|x| x * horizontal_factor);
        let vertical_amount_v120 = vertical_amount_v120.map(|x| x * vertical_factor);

        let mut frame = AxisFrame::new(event.time_msec()).source(source);
        if horizontal_amount != 0.0 {
            frame = frame
                .relative_direction(Axis::Horizontal, event.relative_direction(Axis::Horizontal));
            frame = frame.value(Axis::Horizontal, horizontal_amount);
            if let Some(v120) = horizontal_amount_v120 {
                frame = frame.v120(Axis::Horizontal, v120 as i32);
            }
        }
        if vertical_amount != 0.0 {
            frame =
                frame.relative_direction(Axis::Vertical, event.relative_direction(Axis::Vertical));
            frame = frame.value(Axis::Vertical, vertical_amount);
            if let Some(v120) = vertical_amount_v120 {
                frame = frame.v120(Axis::Vertical, v120 as i32);
            }
        }

        if source == AxisSource::Finger {
            if event.amount(Axis::Horizontal) == Some(0.0) {
                frame = frame.stop(Axis::Horizontal);
            }
            if event.amount(Axis::Vertical) == Some(0.0) {
                frame = frame.stop(Axis::Vertical);
            }
        }

        pointer.axis(self, frame);
        pointer.frame(self);
    }

    fn on_tablet_tool_axis<I: InputBackend>(&mut self, event: I::TabletToolAxisEvent)
    where
        I::Device: 'static, // Needed for downcasting.
    {
        let Some(pos) = self.compute_tablet_position(&event) else {
            return;
        };

        if let Some(output) = self.niri.screenshot_ui.selection_output() {
            let geom = self.niri.global_space.output_geometry(output).unwrap();
            let point = (pos - geom.loc.to_f64())
                .to_physical(output.current_scale().fractional_scale())
                .to_i32_round::<i32>();

            self.niri.screenshot_ui.pointer_motion(point, None);
        }

        if let Some(mru_output) = self.niri.window_mru_ui.output() {
            if let Some((output, pos_within_output)) = self.niri.output_under(pos) {
                if mru_output == output {
                    self.niri.window_mru_ui.pointer_motion(pos_within_output);
                }
            }
        }

        let under = self.niri.contents_under(pos);

        let tablet_seat = self.niri.seat.tablet_seat();
        let tablet = tablet_seat.get_tablet(&TabletDescriptor::from(&event.device()));
        let tool = tablet_seat.get_tool(&event.tool());
        if let (Some(tablet), Some(tool)) = (tablet, tool) {
            if event.pressure_has_changed() {
                tool.pressure(event.pressure());
            }
            if event.distance_has_changed() {
                tool.distance(event.distance());
            }
            if event.tilt_has_changed() {
                tool.tilt(event.tilt());
            }
            if event.slider_has_changed() {
                tool.slider_position(event.slider_position());
            }
            if event.rotation_has_changed() {
                tool.rotation(event.rotation());
            }
            if event.wheel_has_changed() {
                tool.wheel(event.wheel_delta(), event.wheel_delta_discrete());
            }

            tool.motion(
                pos,
                under.surface,
                &tablet,
                SERIAL_COUNTER.next_serial(),
                event.time_msec(),
            );

            self.niri.pointer_visibility = PointerVisibility::Visible;
            self.niri.tablet_cursor_location = Some(pos);
        }

        // Redraw to update the cursor position.
        // FIXME: redraw only outputs overlapping the cursor.
        self.niri.queue_redraw_all();
    }

    fn on_tablet_tool_tip<I: InputBackend>(&mut self, event: I::TabletToolTipEvent) {
        let tool = self.niri.seat.tablet_seat().get_tool(&event.tool());

        let Some(tool) = tool else {
            return;
        };
        let tip_state = event.tip_state();

        let is_overview_open = self.niri.layout.is_overview_open();

        match tip_state {
            TabletToolTipState::Down => {
                let serial = SERIAL_COUNTER.next_serial();
                tool.tip_down(serial, event.time_msec());

                if let Some(pos) = self.niri.tablet_cursor_location {
                    let under = self.niri.contents_under(pos);

                    if self.niri.screenshot_ui.is_open() {
                        let mod_key = self.backend.mod_key(&self.niri.config.borrow());
                        let mods = self.niri.seat.get_keyboard().unwrap().modifier_state();
                        let modifiers = modifiers_from_state(mods);
                        let mod_down = modifiers.contains(mod_key.to_modifiers());

                        // If we'll be moving the existing selection, use the selection output.
                        let output = if mod_down {
                            self.niri.screenshot_ui.selection_output()
                        } else {
                            under.output.as_ref()
                        };

                        if let Some(output) = output.cloned() {
                            let geom = self.niri.global_space.output_geometry(&output).unwrap();
                            let point = (pos - geom.loc.to_f64())
                                .to_physical(output.current_scale().fractional_scale())
                                .to_i32_round();

                            if self
                                .niri
                                .screenshot_ui
                                .pointer_down(output, point, None, mod_down)
                            {
                                self.niri.queue_redraw_all();
                            }
                        }
                    } else if let Some(mru_output) = self.niri.window_mru_ui.output() {
                        if let Some((output, pos_within_output)) = self.niri.output_under(pos) {
                            if mru_output == output {
                                let id = self.niri.window_mru_ui.pointer_motion(pos_within_output);
                                if id.is_some() {
                                    self.confirm_mru();
                                } else {
                                    self.niri.cancel_mru();
                                }
                            } else {
                                self.niri.cancel_mru();
                            }
                        }
                    } else if let Some((window, _)) = under.window {
                        if let Some(output) = is_overview_open.then_some(under.output).flatten() {
                            let mut workspaces = self.niri.layout.workspaces();
                            if let Some(ws_idx) = workspaces.find_map(|(_, ws_idx, ws)| {
                                ws.windows().any(|w| w.window == window).then_some(ws_idx)
                            }) {
                                drop(workspaces);
                                self.niri.layout.focus_output(&output);
                                self.niri.layout.toggle_overview_to_workspace(ws_idx);
                            }
                        }

                        self.niri.layout.activate_window(&window);

                        // FIXME: granular.
                        self.niri.queue_redraw_all();
                    } else if let Some((output, ws)) = is_overview_open
                        .then(|| self.niri.workspace_under(false, pos))
                        .flatten()
                    {
                        let ws_idx = self.niri.layout.find_workspace_by_id(ws.id()).unwrap().0;

                        self.niri.layout.focus_output(&output);
                        self.niri.layout.toggle_overview_to_workspace(ws_idx);

                        // FIXME: granular.
                        self.niri.queue_redraw_all();
                    } else if let Some(output) = under.output {
                        self.niri.layout.focus_output(&output);

                        // FIXME: granular.
                        self.niri.queue_redraw_all();
                    }
                    self.niri.focus_layer_surface_if_on_demand(under.layer);
                }
            }
            TabletToolTipState::Up => {
                if let Some(capture) = self.niri.screenshot_ui.pointer_up(None) {
                    if capture {
                        self.confirm_screenshot(true);
                    } else {
                        self.niri.queue_redraw_all();
                    }
                }

                tool.tip_up(event.time_msec());
            }
        }
    }

    fn on_tablet_tool_proximity<I: InputBackend>(&mut self, event: I::TabletToolProximityEvent)
    where
        I::Device: 'static, // Needed for downcasting.
    {
        let Some(pos) = self.compute_tablet_position(&event) else {
            return;
        };

        let under = self.niri.contents_under(pos);

        let tablet_seat = self.niri.seat.tablet_seat();
        let display_handle = self.niri.display_handle.clone();
        let tool = tablet_seat.add_tool::<Self>(self, &display_handle, &event.tool());
        let tablet = tablet_seat.get_tablet(&TabletDescriptor::from(&event.device()));
        if let Some(tablet) = tablet {
            match event.state() {
                ProximityState::In => {
                    if let Some(under) = under.surface {
                        tool.proximity_in(
                            pos,
                            under,
                            &tablet,
                            SERIAL_COUNTER.next_serial(),
                            event.time_msec(),
                        );
                    }
                    self.niri.pointer_visibility = PointerVisibility::Visible;
                    self.niri.tablet_cursor_location = Some(pos);
                }
                ProximityState::Out => {
                    tool.proximity_out(event.time_msec());

                    // Move the mouse pointer here to avoid discontinuity.
                    //
                    // Plus, Wayland SDL2 currently warps the pointer into some weird
                    // location on proximity out, so this should help it a little.
                    if let Some(pos) = self.niri.tablet_cursor_location {
                        self.move_cursor(pos);
                    }

                    self.niri.pointer_visibility = PointerVisibility::Visible;
                    self.niri.tablet_cursor_location = None;
                }
            }

            // FIXME: granular.
            self.niri.queue_redraw_all();
        }
    }

    fn on_tablet_tool_button<I: InputBackend>(&mut self, event: I::TabletToolButtonEvent) {
        let tool = self.niri.seat.tablet_seat().get_tool(&event.tool());

        if let Some(tool) = tool {
            tool.button(
                event.button(),
                event.button_state(),
                SERIAL_COUNTER.next_serial(),
                event.time_msec(),
            );
        }
    }

    fn on_gesture_swipe_begin<I: InputBackend>(&mut self, event: I::GestureSwipeBeginEvent) {
        if self.niri.window_mru_ui.is_open() {
            // Don't start swipe gestures while in the MRU.
            return;
        }

        if event.fingers() == 3 {
            self.niri.gesture_swipe_3f_cumulative = Some((0., 0.));

            // We handled this event.
            return;
        } else if event.fingers() == 4 {
            self.niri.layout.overview_gesture_begin();
            self.niri.queue_redraw_all();

            // We handled this event.
            return;
        }

        let serial = SERIAL_COUNTER.next_serial();
        let pointer = self.niri.seat.get_pointer().unwrap();

        if self.update_pointer_contents() {
            pointer.frame(self);
        }

        pointer.gesture_swipe_begin(
            self,
            &GestureSwipeBeginEvent {
                serial,
                time: event.time_msec(),
                fingers: event.fingers(),
            },
        );
    }

    fn on_gesture_swipe_update<I: InputBackend + 'static>(
        &mut self,
        event: I::GestureSwipeUpdateEvent,
    ) where
        I::Device: 'static,
    {
        let mut delta_x = event.delta_x();
        let mut delta_y = event.delta_y();

        if let Some(libinput_event) =
            (&event as &dyn Any).downcast_ref::<input::event::gesture::GestureSwipeUpdateEvent>()
        {
            delta_x = libinput_event.dx_unaccelerated();
            delta_y = libinput_event.dy_unaccelerated();
        }

        let uninverted_delta_y = delta_y;

        let device = event.device();
        if let Some(device) = (&device as &dyn Any).downcast_ref::<input::Device>() {
            if device.config_scroll_natural_scroll_enabled() {
                delta_x = -delta_x;
                delta_y = -delta_y;
            }
        }

        let is_overview_open = self.niri.layout.is_overview_open();

        if let Some((cx, cy)) = &mut self.niri.gesture_swipe_3f_cumulative {
            *cx += delta_x;
            *cy += delta_y;

            // Check if the gesture moved far enough to decide. Threshold copied from GNOME Shell.
            let (cx, cy) = (*cx, *cy);
            if cx * cx + cy * cy >= 16. * 16. {
                self.niri.gesture_swipe_3f_cumulative = None;

                if let Some(output) = self.niri.output_under_cursor() {
                    if cx.abs() > cy.abs() {
                        let output_ws = if is_overview_open {
                            self.niri.workspace_under_cursor(true)
                        } else {
                            // We don't want to accidentally "catch" the wrong workspace during
                            // animations.
                            let layout = &self.niri.layout;
                            self.niri.output_under_cursor().and_then(|output| {
                                let mon = layout.monitor_for_output(&output)?;
                                let view = layout.active_view(&mon.output_id());
                                Some((output, view.active_workspace_ref(layout.workspace_pool())))
                            })
                        };

                        if let Some((output, ws)) = output_ws {
                            let ws_idx = self.niri.layout.find_workspace_by_id(ws.id()).unwrap().0;
                            self.niri
                                .layout
                                .view_offset_gesture_begin(&output, Some(ws_idx), true);
                        }
                    } else {
                        self.niri
                            .layout
                            .workspace_switch_gesture_begin(&output, true);
                    }
                }
            }
        }

        let timestamp = Duration::from_micros(event.time());

        let mut handled = false;
        let res = self
            .niri
            .layout
            .workspace_switch_gesture_update(delta_y, timestamp, true);
        if let Some(output) = res {
            if let Some(output) = output {
                self.niri.queue_redraw(&output);
            }
            handled = true;
        }

        let res = self
            .niri
            .layout
            .view_offset_gesture_update(delta_x, timestamp, true);
        if let Some(output) = res {
            if let Some(output) = output {
                self.niri.queue_redraw(&output);
            }
            handled = true;
        }

        let res = self
            .niri
            .layout
            .overview_gesture_update(-uninverted_delta_y, timestamp);
        if let Some(redraw) = res {
            if redraw {
                self.niri.queue_redraw_all();
            }
            handled = true;
        }

        if handled {
            // We handled this event.
            return;
        }

        let pointer = self.niri.seat.get_pointer().unwrap();

        if self.update_pointer_contents() {
            pointer.frame(self);
        }

        pointer.gesture_swipe_update(
            self,
            &GestureSwipeUpdateEvent {
                time: event.time_msec(),
                delta: event.delta(),
            },
        );
    }

    fn on_gesture_swipe_end<I: InputBackend>(&mut self, event: I::GestureSwipeEndEvent) {
        self.niri.gesture_swipe_3f_cumulative = None;

        let mut handled = false;
        let res = self.niri.layout.workspace_switch_gesture_end(Some(true));
        if let Some(output) = res {
            self.niri.queue_redraw(&output);
            handled = true;
        }

        let res = self.niri.layout.view_offset_gesture_end(Some(true));
        if let Some(output) = res {
            self.niri.queue_redraw(&output);
            handled = true;
        }

        let res = self.niri.layout.overview_gesture_end();
        if res {
            self.niri.queue_redraw_all();
            handled = true;
        }

        if handled {
            // We handled this event.
            return;
        }

        let serial = SERIAL_COUNTER.next_serial();
        let pointer = self.niri.seat.get_pointer().unwrap();

        if self.update_pointer_contents() {
            pointer.frame(self);
        }

        pointer.gesture_swipe_end(
            self,
            &GestureSwipeEndEvent {
                serial,
                time: event.time_msec(),
                cancelled: event.cancelled(),
            },
        );
    }

    fn on_gesture_pinch_begin<I: InputBackend>(&mut self, event: I::GesturePinchBeginEvent) {
        let serial = SERIAL_COUNTER.next_serial();
        let pointer = self.niri.seat.get_pointer().unwrap();

        if self.update_pointer_contents() {
            pointer.frame(self);
        }

        pointer.gesture_pinch_begin(
            self,
            &GesturePinchBeginEvent {
                serial,
                time: event.time_msec(),
                fingers: event.fingers(),
            },
        );
    }

    fn on_gesture_pinch_update<I: InputBackend>(&mut self, event: I::GesturePinchUpdateEvent) {
        let pointer = self.niri.seat.get_pointer().unwrap();

        if self.update_pointer_contents() {
            pointer.frame(self);
        }

        pointer.gesture_pinch_update(
            self,
            &GesturePinchUpdateEvent {
                time: event.time_msec(),
                delta: event.delta(),
                scale: event.scale(),
                rotation: event.rotation(),
            },
        );
    }

    fn on_gesture_pinch_end<I: InputBackend>(&mut self, event: I::GesturePinchEndEvent) {
        let serial = SERIAL_COUNTER.next_serial();
        let pointer = self.niri.seat.get_pointer().unwrap();

        if self.update_pointer_contents() {
            pointer.frame(self);
        }

        pointer.gesture_pinch_end(
            self,
            &GesturePinchEndEvent {
                serial,
                time: event.time_msec(),
                cancelled: event.cancelled(),
            },
        );
    }

    fn on_gesture_hold_begin<I: InputBackend>(&mut self, event: I::GestureHoldBeginEvent) {
        let serial = SERIAL_COUNTER.next_serial();
        let pointer = self.niri.seat.get_pointer().unwrap();

        if self.update_pointer_contents() {
            pointer.frame(self);
        }

        pointer.gesture_hold_begin(
            self,
            &GestureHoldBeginEvent {
                serial,
                time: event.time_msec(),
                fingers: event.fingers(),
            },
        );
    }

    fn on_gesture_hold_end<I: InputBackend>(&mut self, event: I::GestureHoldEndEvent) {
        let serial = SERIAL_COUNTER.next_serial();
        let pointer = self.niri.seat.get_pointer().unwrap();

        if self.update_pointer_contents() {
            pointer.frame(self);
        }

        pointer.gesture_hold_end(
            self,
            &GestureHoldEndEvent {
                serial,
                time: event.time_msec(),
                cancelled: event.cancelled(),
            },
        );
    }

    fn compute_absolute_location<I: InputBackend>(
        &self,
        evt: &impl AbsolutePositionEvent<I>,
        fallback_output: Option<&Output>,
    ) -> Option<Point<f64, Logical>> {
        let output = evt.device().output(self);
        let output = output.filter(|output| self.niri.output_exists(output));
        let output = output.as_ref().or(fallback_output)?;
        let output_geo = self.niri.global_space.output_geometry(output).unwrap();
        let transform = output.current_transform();
        let size = transform.invert().transform_size(output_geo.size);
        Some(
            transform.transform_point_in(evt.position_transformed(size), &size.to_f64())
                + output_geo.loc.to_f64(),
        )
    }

    /// Computes the cursor position for the touch event.
    ///
    /// This function handles the touch output mapping, as well as coordinate transform
    fn compute_touch_location<I: InputBackend>(
        &self,
        evt: &impl AbsolutePositionEvent<I>,
    ) -> Option<Point<f64, Logical>> {
        self.compute_absolute_location(evt, self.niri.output_for_touch())
    }

    fn on_touch_down<I: InputBackend>(&mut self, evt: I::TouchDownEvent) {
        let Some(handle) = self.niri.seat.get_touch() else {
            return;
        };
        let Some(pos) = self.compute_touch_location(&evt) else {
            return;
        };
        let slot = evt.slot();

        let serial = SERIAL_COUNTER.next_serial();

        let under = self.niri.contents_under(pos);

        let mod_key = self.backend.mod_key(&self.niri.config.borrow());
        let mods = self.niri.seat.get_keyboard().unwrap().modifier_state();
        let mods = modifiers_from_state(mods);
        let mod_down = mods.contains(mod_key.to_modifiers());

        if self.niri.screenshot_ui.is_open() {
            // If we'll be moving the existing selection, use the selection output.
            let output = if mod_down {
                self.niri.screenshot_ui.selection_output()
            } else {
                under.output.as_ref()
            };

            if let Some(output) = output.cloned() {
                let geom = self.niri.global_space.output_geometry(&output).unwrap();
                let point = (pos - geom.loc.to_f64())
                    .to_physical(output.current_scale().fractional_scale())
                    .to_i32_round();

                if self
                    .niri
                    .screenshot_ui
                    .pointer_down(output, point, Some(slot), mod_down)
                {
                    self.niri.queue_redraw_all();
                }
            }
        } else if let Some(mru_output) = self.niri.window_mru_ui.output() {
            if let Some((output, pos_within_output)) = self.niri.output_under(pos) {
                if mru_output == output {
                    let id = self.niri.window_mru_ui.pointer_motion(pos_within_output);
                    if id.is_some() {
                        self.confirm_mru();
                    } else {
                        self.niri.cancel_mru();
                    }
                } else {
                    self.niri.cancel_mru();
                }
            }
        } else if !handle.is_grabbed() {
            if self.niri.layout.is_overview_open()
                && !mod_down
                && under.layer.is_none()
                && under.output.is_some()
            {
                let (output, pos_within_output) = self.niri.output_under(pos).unwrap();
                let output = output.clone();

                let mut matched_narrow = true;
                let mut ws = self.niri.workspace_under(false, pos);
                if ws.is_none() {
                    matched_narrow = false;
                    ws = self.niri.workspace_under(true, pos);
                }
                let ws_id = ws.map(|(_, ws)| ws.id());

                let mapped = self.niri.window_under(pos);
                let window = mapped.map(|mapped| mapped.window.clone());

                let start_data = TouchGrabStartData {
                    focus: None,
                    slot,
                    location: pos,
                };
                let start_timestamp = Duration::from_micros(evt.time());
                let grab = TouchOverviewGrab::new(
                    start_data,
                    start_timestamp,
                    output,
                    pos_within_output,
                    ws_id,
                    matched_narrow,
                    window,
                );
                handle.set_grab(self, grab, serial);
            } else if let Some((window, _)) = under.window {
                self.niri.layout.activate_window(&window);

                // Check if we need to start a touch move grab.
                if mod_down {
                    let start_data = TouchGrabStartData {
                        focus: None,
                        slot,
                        location: pos,
                    };
                    let start_data = PointerOrTouchStartData::Touch(start_data);
                    if let Some(grab) = MoveGrab::new(self, start_data, window.clone(), true, None)
                    {
                        handle.set_grab(self, grab, serial);
                    }
                }

                // FIXME: granular.
                self.niri.queue_redraw_all();
            } else if let Some(output) = under.output {
                self.niri.layout.focus_output(&output);

                // FIXME: granular.
                self.niri.queue_redraw_all();
            }
            self.niri.focus_layer_surface_if_on_demand(under.layer);
        };

        handle.down(
            self,
            under.surface,
            &DownEvent {
                slot,
                location: pos,
                serial,
                time: evt.time_msec(),
            },
        );

        // We're using touch, hide the pointer.
        self.niri.pointer_visibility = PointerVisibility::Disabled;
    }
    fn on_touch_up<I: InputBackend>(&mut self, evt: I::TouchUpEvent) {
        let Some(handle) = self.niri.seat.get_touch() else {
            return;
        };
        let slot = evt.slot();

        if let Some(capture) = self.niri.screenshot_ui.pointer_up(Some(slot)) {
            if capture {
                self.confirm_screenshot(true);
            } else {
                self.niri.queue_redraw_all();
            }
        }

        let serial = SERIAL_COUNTER.next_serial();
        handle.up(
            self,
            &UpEvent {
                slot,
                serial,
                time: evt.time_msec(),
            },
        )
    }
    fn on_touch_motion<I: InputBackend>(&mut self, evt: I::TouchMotionEvent) {
        let Some(handle) = self.niri.seat.get_touch() else {
            return;
        };
        let Some(pos) = self.compute_touch_location(&evt) else {
            return;
        };
        let slot = evt.slot();

        if let Some(output) = self.niri.screenshot_ui.selection_output().cloned() {
            let geom = self.niri.global_space.output_geometry(&output).unwrap();
            let point = (pos - geom.loc.to_f64())
                .to_physical(output.current_scale().fractional_scale())
                .to_i32_round::<i32>();

            self.niri.screenshot_ui.pointer_motion(point, Some(slot));
            self.niri.queue_redraw(&output);
        }

        let under = self.niri.contents_under(pos);
        handle.motion(
            self,
            under.surface,
            &TouchMotionEvent {
                slot,
                location: pos,
                time: evt.time_msec(),
            },
        );

        // Inform the layout of an ongoing DnD operation.
        let is_dnd_grab = handle
            .with_grab(|_, grab| Self::is_dnd_grab(grab.as_any()))
            .unwrap_or(false);
        if is_dnd_grab {
            if let Some((output, pos_within_output)) = self.niri.output_under(pos) {
                let output = output.clone();
                self.niri.layout.dnd_update(output, pos_within_output);
            }
        }
    }
    fn on_touch_frame<I: InputBackend>(&mut self, _evt: I::TouchFrameEvent) {
        let Some(handle) = self.niri.seat.get_touch() else {
            return;
        };
        handle.frame(self);
    }
    fn on_touch_cancel<I: InputBackend>(&mut self, _evt: I::TouchCancelEvent) {
        let Some(handle) = self.niri.seat.get_touch() else {
            return;
        };
        handle.cancel(self);
    }

    fn on_switch_toggle<I: InputBackend>(&mut self, evt: I::SwitchToggleEvent) {
        let Some(switch) = evt.switch() else {
            return;
        };

        if switch == Switch::Lid {
            let is_closed = evt.state() == SwitchState::On;
            trace!("lid switch {}", if is_closed { "closed" } else { "opened" });
            self.set_lid_closed(is_closed);
        }

        let action = {
            let bindings = &self.niri.config.borrow().switch_events;
            find_configured_switch_action(bindings, switch, evt.state())
        };

        if let Some(action) = action {
            self.do_action(action, true);
        }
    }

    pub fn is_dnd_grab(grab: &dyn Any) -> bool {
        // Normal DnD
        grab.is::<DnDGrab<Self, WlDataSource, WlSurface>>()
            // Null-source DnD: weston-dnd --self-only
            || grab.is::<DnDGrab<Self, WlSurface, WlSurface>>()
    }

    fn grab_can_be_cancelled_with_esc(grab: &(dyn PointerGrab<State> + 'static)) -> bool {
        let grab = grab.as_any();

        grab.is::<PickWindowGrab>() || grab.is::<PickColorGrab>() || Self::is_dnd_grab(grab)
    }
}

/// Check whether the key should be intercepted and mark intercepted
/// pressed keys as `suppressed`, thus preventing `releases` corresponding
/// to them from being delivered.
#[allow(clippy::too_many_arguments)]
fn should_intercept_key<'a>(
    suppressed_keys: &mut HashSet<Keycode>,
    bindings: impl IntoIterator<Item = &'a Bind>,
    mod_key: ModKey,
    key_code: Keycode,
    modified: Keysym,
    raw: Option<Keysym>,
    pressed: bool,
    mods: ModifiersState,
    screenshot_ui: &ScreenshotUi,
    disable_power_key_handling: bool,
    is_inhibiting_shortcuts: bool,
) -> FilterResult<Option<Bind>> {
    // Actions are only triggered on presses, release of the key
    // shouldn't try to intercept anything unless we have marked
    // the key to suppress.
    if !pressed && !suppressed_keys.contains(&key_code) {
        return FilterResult::Forward;
    }

    let mut final_bind = find_bind(
        bindings,
        mod_key,
        modified,
        raw,
        mods,
        disable_power_key_handling,
    );

    // Allow only a subset of compositor actions while the screenshot UI is open, since the user
    // cannot see the screen.
    if screenshot_ui.is_open() {
        let mut use_screenshot_ui_action = true;

        if let Some(bind) = &final_bind {
            if allowed_during_screenshot(&bind.action) {
                use_screenshot_ui_action = false;
            }
        }

        if use_screenshot_ui_action {
            if let Some(raw) = raw {
                final_bind = screenshot_ui.action(raw, mods).map(|action| Bind {
                    key: Key {
                        trigger: Trigger::Keysym(raw),
                        // Not entirely correct but it doesn't matter in how we currently use it.
                        modifiers: Modifiers::empty(),
                    },
                    action,
                    repeat: true,
                    cooldown: None,
                    allow_when_locked: false,
                    // The screenshot UI owns the focus anyway, so this doesn't really matter.
                    // But logically, nothing can inhibit its actions. Only opening it can be
                    // inhibited.
                    allow_inhibiting: false,
                    hotkey_overlay_title: None,
                });
            }
        }
    }

    match (final_bind, pressed) {
        (Some(bind), true) => {
            if is_inhibiting_shortcuts && bind.allow_inhibiting {
                FilterResult::Forward
            } else {
                suppressed_keys.insert(key_code);
                FilterResult::Intercept(Some(bind))
            }
        }
        (_, false) => {
            // By this point, we know that the key was suppressed on press. Even if we're inhibiting
            // shortcuts, we should still suppress the release.
            // But we don't need to check for shortcuts inhibition here, because
            // if it was inhibited on press (forwarded to the client), it wouldn't be suppressed,
            // so the release would already have been forwarded at the start of this function.
            suppressed_keys.remove(&key_code);
            FilterResult::Intercept(None)
        }
        (None, true) => FilterResult::Forward,
    }
}

fn find_bind<'a>(
    bindings: impl IntoIterator<Item = &'a Bind>,
    mod_key: ModKey,
    modified: Keysym,
    raw: Option<Keysym>,
    mods: ModifiersState,
    disable_power_key_handling: bool,
) -> Option<Bind> {
    use keysyms::*;

    // Handle hardcoded binds.
    #[allow(non_upper_case_globals)] // wat
    let hardcoded_action = match modified.raw() {
        modified @ KEY_XF86Switch_VT_1..=KEY_XF86Switch_VT_12 => {
            let vt = (modified - KEY_XF86Switch_VT_1 + 1) as i32;
            Some(Action::ChangeVt(vt))
        }
        KEY_XF86PowerOff if !disable_power_key_handling => Some(Action::Suspend),
        _ => None,
    };

    if let Some(action) = hardcoded_action {
        return Some(Bind {
            key: Key {
                // Not entirely correct but it doesn't matter in how we currently use it.
                trigger: Trigger::Keysym(modified),
                modifiers: Modifiers::empty(),
            },
            action,
            repeat: true,
            cooldown: None,
            allow_when_locked: false,
            // In a worst-case scenario, the user has no way to unlock the compositor and a
            // misbehaving client has a keyboard shortcuts inhibitor, "jailing" the user.
            // The user must always be able to change VTs to recover from such a situation.
            // It also makes no sense to inhibit the default power key handling.
            // Hardcoded binds must never be inhibited.
            allow_inhibiting: false,
            hotkey_overlay_title: None,
        });
    }

    let trigger = Trigger::Keysym(raw?);
    find_configured_bind(bindings, mod_key, trigger, mods)
}

fn find_configured_bind<'a>(
    bindings: impl IntoIterator<Item = &'a Bind>,
    mod_key: ModKey,
    trigger: Trigger,
    mods: ModifiersState,
) -> Option<Bind> {
    // Handle configured binds.
    let mut modifiers = modifiers_from_state(mods);

    let mod_down = modifiers_from_state(mods).contains(mod_key.to_modifiers());
    if mod_down {
        modifiers |= Modifiers::COMPOSITOR;
    }

    for bind in bindings {
        if bind.key.trigger != trigger {
            continue;
        }

        let mut bind_modifiers = bind.key.modifiers;
        if bind_modifiers.contains(Modifiers::COMPOSITOR) {
            bind_modifiers |= mod_key.to_modifiers();
        } else if bind_modifiers.contains(mod_key.to_modifiers()) {
            bind_modifiers |= Modifiers::COMPOSITOR;
        }

        if bind_modifiers == modifiers {
            return Some(bind.clone());
        }
    }

    None
}

/// Whether `a` and `b` bind to the same effective key under `mod_key`
/// normalization.
///
/// Mirrors [`find_configured_bind`]'s bidirectional `COMPOSITOR` ⇄ `mod_key`
/// modifier expansion, applied symmetrically to both sides rather than to one
/// side plus a live [`ModifiersState`]: the collision check compares two
/// static [`Key`]s (a candidate bookmark key against a config bind, or a
/// static bind against another) rather than a static key against a live
/// keyboard state.
fn keys_conflict(a: Key, b: Key, mod_key: ModKey) -> bool {
    if a.trigger != b.trigger {
        return false;
    }
    let normalize = |mut modifiers: Modifiers| -> Modifiers {
        if modifiers.contains(Modifiers::COMPOSITOR) {
            modifiers |= mod_key.to_modifiers();
        } else if modifiers.contains(mod_key.to_modifiers()) {
            modifiers |= Modifiers::COMPOSITOR;
        }
        modifiers
    };
    normalize(a.modifiers) == normalize(b.modifiers)
}

/// Whether `candidate` collides, under [`keys_conflict`] normalization, with
/// any bookmark key in `keyed_bookmarks` other than the one being
/// (re-)assigned (`excluding_id`).
///
/// `keyed_bookmarks` is `(bookmark id, assigned key)` pairs read live off
/// `Layout::bookmarks()` — not `Niri::bookmark_binds`, the synthetic bind
/// mirror, which only rebuilds once per calloop dispatch iteration (epoch-
/// gated in `State::refresh`) and can lag behind two `AssignBookmarkKey`
/// actions dispatched in the same iteration. The layout list is mutated
/// synchronously by `assign_bookmark_key`, so it is never stale within a
/// cycle. `excluding_id` lets re-pressing a bookmark's own current key stay
/// idempotent, not collide with itself.
fn bookmark_key_collides_with_siblings(
    keyed_bookmarks: impl Iterator<Item = (u64, Key)>,
    excluding_id: u64,
    candidate: Key,
    mod_key: ModKey,
) -> bool {
    keyed_bookmarks
        .filter(|&(id, _)| id != excluding_id)
        .any(|(_, key)| keys_conflict(key, candidate, mod_key))
}

/// Why [`validate_bookmark_key_candidate`] rejected a candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
enum BookmarkKeyRejection {
    /// Failed [`BookmarkKey::new`] validation: no modifiers, or a non-keysym
    /// trigger (the latter cannot arise from a live keyboard press — the
    /// interactive capture surface always builds a keysym candidate — but
    /// the typed `AssignBookmarkKey` path can still name one via a parsed
    /// string). Carries the typed [`BookmarkKeyError`] rather than its
    /// flattened `Display` string, so each consumer can match on the actual
    /// variant instead of trusting a comment about which one is reachable.
    Invalid(BookmarkKeyError),
    /// Matches an existing key. `key` is the canonical formatted key string;
    /// `with` names which side of the collision matched.
    Collision {
        key: String,
        with: BookmarkKeyCollidee,
    },
}

/// Which existing binding a rejected [`BookmarkKeyRejection::Collision`]
/// matched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BookmarkKeyCollidee {
    /// A static config bind or a recent-windows bind — checked as one
    /// chained set, as today.
    ConfigBind,
    /// Another bookmark's assigned key.
    SiblingBookmark,
}

/// Validates `candidate` as the new dynamic keybind for the bookmark
/// `excluding_id`.
///
/// The single policy shared by the typed `Action::AssignBookmarkKey`
/// dispatch arm and the interactive `Action::CaptureBookmarkKey` capture
/// surface, so the two paths cannot drift apart. Checks, in order:
/// [`BookmarkKey::new`] (keysym trigger, at least one modifier), collision
/// against `config_binds` under [`keys_conflict`] normalization, then
/// collision against `keyed_bookmarks` via
/// [`bookmark_key_collides_with_siblings`] (excluding `excluding_id`, so
/// re-capturing a bookmark's own current key stays idempotent).
///
/// # Errors
///
/// Returns [`BookmarkKeyRejection::Invalid`] or
/// [`BookmarkKeyRejection::Collision`] on the first check that fails.
fn validate_bookmark_key_candidate<'a>(
    candidate: Key,
    excluding_id: u64,
    mut config_binds: impl Iterator<Item = &'a Bind>,
    keyed_bookmarks: impl Iterator<Item = (u64, Key)>,
    mod_key: ModKey,
) -> Result<BookmarkKey, BookmarkKeyRejection> {
    let bookmark_key = BookmarkKey::new(candidate).map_err(BookmarkKeyRejection::Invalid)?;

    if config_binds.any(|bind| keys_conflict(bind.key, candidate, mod_key)) {
        return Err(BookmarkKeyRejection::Collision {
            key: key_to_wire_string(candidate),
            with: BookmarkKeyCollidee::ConfigBind,
        });
    }

    if bookmark_key_collides_with_siblings(keyed_bookmarks, excluding_id, candidate, mod_key) {
        return Err(BookmarkKeyRejection::Collision {
            key: key_to_wire_string(candidate),
            with: BookmarkKeyCollidee::SiblingBookmark,
        });
    }

    Ok(bookmark_key)
}

/// A human-readable capture-prompt label for a dangling rule-anchored
/// bookmark, composed from whichever of `app_id`/`title` the rule
/// constrains ([`BookmarkRule::new`] guarantees at least one).
fn bookmark_rule_capture_label(rule: &BookmarkRule) -> String {
    match (rule.app_id_source(), rule.title_source()) {
        (Some(app_id), Some(title)) => format!("app_id~{app_id}, title~{title}"),
        (Some(app_id), None) => format!("app_id~{app_id}"),
        (None, Some(title)) => format!("title~{title}"),
        (None, None) => unreachable!("BookmarkRule::new requires at least one of app_id/title"),
    }
}

fn find_configured_switch_action(
    bindings: &SwitchBinds,
    switch: Switch,
    state: SwitchState,
) -> Option<Action> {
    let switch_action = match (switch, state) {
        (Switch::Lid, SwitchState::Off) => &bindings.lid_open,
        (Switch::Lid, SwitchState::On) => &bindings.lid_close,
        (Switch::TabletMode, SwitchState::Off) => &bindings.tablet_mode_off,
        (Switch::TabletMode, SwitchState::On) => &bindings.tablet_mode_on,
        _ => unreachable!(),
    };
    switch_action
        .as_ref()
        .map(|switch_action| Action::Spawn(switch_action.spawn.clone()))
}

fn modifiers_from_state(mods: ModifiersState) -> Modifiers {
    let mut modifiers = Modifiers::empty();
    if mods.ctrl {
        modifiers |= Modifiers::CTRL;
    }
    if mods.shift {
        modifiers |= Modifiers::SHIFT;
    }
    if mods.alt {
        modifiers |= Modifiers::ALT;
    }
    if mods.logo {
        modifiers |= Modifiers::SUPER;
    }
    if mods.iso_level3_shift {
        modifiers |= Modifiers::ISO_LEVEL3_SHIFT;
    }
    if mods.iso_level5_shift {
        modifiers |= Modifiers::ISO_LEVEL5_SHIFT;
    }
    modifiers
}

fn should_activate_monitors<I: InputBackend>(event: &InputEvent<I>) -> bool {
    match event {
        InputEvent::Keyboard { event } if event.state() == KeyState::Pressed => true,
        InputEvent::PointerButton { event } if event.state() == ButtonState::Pressed => true,
        InputEvent::PointerMotion { .. }
        | InputEvent::PointerMotionAbsolute { .. }
        | InputEvent::PointerAxis { .. }
        | InputEvent::GestureSwipeBegin { .. }
        | InputEvent::GesturePinchBegin { .. }
        | InputEvent::GestureHoldBegin { .. }
        | InputEvent::TouchDown { .. }
        | InputEvent::TouchMotion { .. }
        | InputEvent::TabletToolAxis { .. }
        | InputEvent::TabletToolProximity { .. }
        | InputEvent::TabletToolTip { .. }
        | InputEvent::TabletToolButton { .. } => true,
        // Ignore events like device additions and removals, key releases, gesture ends.
        _ => false,
    }
}

fn should_hide_hotkey_overlay<I: InputBackend>(event: &InputEvent<I>) -> bool {
    match event {
        InputEvent::Keyboard { event } if event.state() == KeyState::Pressed => true,
        InputEvent::PointerButton { event } if event.state() == ButtonState::Pressed => true,
        InputEvent::PointerAxis { .. }
        | InputEvent::GestureSwipeBegin { .. }
        | InputEvent::GesturePinchBegin { .. }
        | InputEvent::TouchDown { .. }
        | InputEvent::TouchMotion { .. }
        | InputEvent::TabletToolTip { .. }
        | InputEvent::TabletToolButton { .. } => true,
        _ => false,
    }
}

fn should_hide_confirm_dialog<I: InputBackend>(event: &InputEvent<I>) -> bool {
    match event {
        InputEvent::Keyboard { event } if event.state() == KeyState::Pressed => true,
        InputEvent::PointerButton { event } if event.state() == ButtonState::Pressed => true,
        InputEvent::PointerAxis { .. }
        | InputEvent::GestureSwipeBegin { .. }
        | InputEvent::GesturePinchBegin { .. }
        | InputEvent::TouchDown { .. }
        | InputEvent::TouchMotion { .. }
        | InputEvent::TabletToolTip { .. }
        | InputEvent::TabletToolButton { .. } => true,
        _ => false,
    }
}

/// Pointer/touch/tablet events that dismiss the bookmark switcher.
///
/// Mirrors [`should_hide_confirm_dialog`] but omits the keyboard arm: while the
/// switcher is open, key presses are handled (and swallowed) inside the
/// `on_keyboard` filter closure, so routing them through here too would be
/// redundant.
fn should_hide_bookmark_switcher<I: InputBackend>(event: &InputEvent<I>) -> bool {
    match event {
        InputEvent::PointerButton { event } if event.state() == ButtonState::Pressed => true,
        InputEvent::PointerAxis { .. }
        | InputEvent::GestureSwipeBegin { .. }
        | InputEvent::GesturePinchBegin { .. }
        | InputEvent::TouchDown { .. }
        | InputEvent::TouchMotion { .. }
        | InputEvent::TabletToolTip { .. }
        | InputEvent::TabletToolButton { .. } => true,
        _ => false,
    }
}

fn should_notify_activity<I: InputBackend>(event: &InputEvent<I>) -> bool {
    !matches!(
        event,
        InputEvent::DeviceAdded { .. } | InputEvent::DeviceRemoved { .. }
    )
}

/// Stable short label per input-event kind for the latency canary log.
fn input_event_label<I: InputBackend>(event: &InputEvent<I>) -> &'static str {
    use InputEvent::*;
    match event {
        DeviceAdded { .. } => "device-added",
        DeviceRemoved { .. } => "device-removed",
        Keyboard { .. } => "keyboard",
        PointerMotion { .. } => "pointer-motion",
        PointerMotionAbsolute { .. } => "pointer-motion-absolute",
        PointerButton { .. } => "pointer-button",
        PointerAxis { .. } => "pointer-axis",
        TabletToolAxis { .. } => "tablet-tool-axis",
        TabletToolTip { .. } => "tablet-tool-tip",
        TabletToolProximity { .. } => "tablet-tool-proximity",
        TabletToolButton { .. } => "tablet-tool-button",
        GestureSwipeBegin { .. } => "gesture-swipe-begin",
        GestureSwipeUpdate { .. } => "gesture-swipe-update",
        GestureSwipeEnd { .. } => "gesture-swipe-end",
        GesturePinchBegin { .. } => "gesture-pinch-begin",
        GesturePinchUpdate { .. } => "gesture-pinch-update",
        GesturePinchEnd { .. } => "gesture-pinch-end",
        GestureHoldBegin { .. } => "gesture-hold-begin",
        GestureHoldEnd { .. } => "gesture-hold-end",
        TouchDown { .. } => "touch-down",
        TouchMotion { .. } => "touch-motion",
        TouchUp { .. } => "touch-up",
        TouchCancel { .. } => "touch-cancel",
        TouchFrame { .. } => "touch-frame",
        SwitchToggle { .. } => "switch-toggle",
        Special(_) => "special",
    }
}

fn should_reset_pointer_inactivity_timer<I: InputBackend>(event: &InputEvent<I>) -> bool {
    matches!(
        event,
        InputEvent::PointerAxis { .. }
            | InputEvent::PointerButton { .. }
            | InputEvent::PointerMotion { .. }
            | InputEvent::PointerMotionAbsolute { .. }
            | InputEvent::TabletToolAxis { .. }
            | InputEvent::TabletToolButton { .. }
            | InputEvent::TabletToolProximity { .. }
            | InputEvent::TabletToolTip { .. }
    )
}

/// Non-keyboard input that disarms a pending mod-tap: any signal that the
/// user has moved on to something else. Both `PointerButton` states are
/// included — even a release-only ordering (button down before the mod
/// press) still signals interaction, and disarming too eagerly only loses a
/// tap, never fires one spuriously. Motion events (`PointerMotion`,
/// `PointerMotionAbsolute`, `TouchMotion`, `TabletToolAxis`,
/// `TabletToolProximity`) are deliberately excluded so that moving the mouse
/// during a tap doesn't disarm it.
fn should_disarm_mod_tap<I: InputBackend>(event: &InputEvent<I>) -> bool {
    matches!(
        event,
        InputEvent::PointerButton { .. }
            | InputEvent::PointerAxis { .. }
            | InputEvent::TouchDown { .. }
            | InputEvent::TabletToolTip { .. }
            | InputEvent::TabletToolButton { .. }
            | InputEvent::GestureSwipeBegin { .. }
            | InputEvent::GesturePinchBegin { .. }
            | InputEvent::GestureHoldBegin { .. }
    )
}

fn allowed_when_locked(action: &Action) -> bool {
    matches!(
        action,
        Action::Quit(_)
            | Action::ChangeVt(_)
            | Action::Suspend
            | Action::PowerOffMonitors
            | Action::PowerOnMonitors
            | Action::SwitchLayout(_)
            | Action::ToggleKeyboardShortcutsInhibit
    )
}

fn allowed_during_screenshot(action: &Action) -> bool {
    matches!(
        action,
        Action::Quit(_)
            | Action::ChangeVt(_)
            | Action::Suspend
            | Action::PowerOffMonitors
            | Action::PowerOnMonitors
            // Intended for binds such as volume up/down, lock the screen, etc.
            | Action::Spawn(_)
            | Action::SpawnSh(_)
            // The screenshot UI can handle these.
            | Action::MoveColumnLeft
            | Action::MoveColumnLeftOrToMonitorLeft
            | Action::MoveColumnRight
            | Action::MoveColumnRightOrToMonitorRight
            | Action::MoveWindowUp
            | Action::MoveWindowUpOrToWorkspaceUp
            | Action::MoveWindowDown
            | Action::MoveWindowDownOrToWorkspaceDown
            | Action::MoveColumnToMonitorLeft
            | Action::MoveColumnToMonitorRight
            | Action::MoveColumnToMonitorUp
            | Action::MoveColumnToMonitorDown
            | Action::MoveColumnToMonitorPrevious
            | Action::MoveColumnToMonitorNext
            | Action::MoveColumnToMonitor(_)
            | Action::MoveWindowToMonitorLeft
            | Action::MoveWindowToMonitorRight
            | Action::MoveWindowToMonitorUp
            | Action::MoveWindowToMonitorDown
            | Action::MoveWindowToMonitorPrevious
            | Action::MoveWindowToMonitorNext
            | Action::MoveWindowToMonitor(_)
            | Action::SetWindowWidth(_)
            | Action::SetWindowHeight(_)
            | Action::SetColumnWidth(_)
    )
}

fn hardcoded_overview_bind(raw: Keysym, mods: ModifiersState) -> Option<Bind> {
    let mods = modifiers_from_state(mods);
    if !mods.is_empty() {
        return None;
    }

    let mut repeat = true;
    let action = match raw {
        Keysym::Escape | Keysym::Return => {
            repeat = false;
            Action::ToggleOverview
        }
        Keysym::Left => Action::FocusColumnLeft,
        Keysym::Right => Action::FocusColumnRight,
        Keysym::Up => Action::FocusWindowOrWorkspaceUp,
        Keysym::Down => Action::FocusWindowOrWorkspaceDown,
        _ => {
            return None;
        }
    };

    Some(Bind {
        key: Key {
            trigger: Trigger::Keysym(raw),
            modifiers: Modifiers::empty(),
        },
        action,
        repeat,
        cooldown: None,
        allow_when_locked: false,
        allow_inhibiting: false,
        hotkey_overlay_title: None,
    })
}

pub fn apply_libinput_settings(config: &jiji_config::Input, device: &mut input::Device) {
    // According to Mutter code, this setting is specific to touchpads.
    let is_touchpad = device.config_tap_finger_count() > 0;
    if is_touchpad {
        let c = &config.touchpad;
        let _ = device.config_send_events_set_mode(if c.off {
            input::SendEventsMode::DISABLED
        } else if c.disabled_on_external_mouse {
            input::SendEventsMode::DISABLED_ON_EXTERNAL_MOUSE
        } else {
            input::SendEventsMode::ENABLED
        });
        let _ = device.config_tap_set_enabled(c.tap);
        let _ = device.config_dwt_set_enabled(c.dwt);
        let _ = device.config_dwtp_set_enabled(c.dwtp);
        let _ = device.config_tap_set_drag_lock_enabled(if c.drag_lock {
            input::DragLockState::EnabledTimeout
        } else {
            input::DragLockState::Disabled
        });
        let _ = device.config_scroll_set_natural_scroll_enabled(c.natural_scroll);
        let _ = device.config_accel_set_speed(c.accel_speed.0);
        let _ = device.config_left_handed_set(c.left_handed);
        let _ = device.config_middle_emulation_set_enabled(c.middle_emulation);

        if let Some(drag) = c.drag {
            let _ = device.config_tap_set_drag_enabled(drag);
        } else {
            let default = device.config_tap_default_drag_enabled();
            let _ = device.config_tap_set_drag_enabled(default);
        }

        if let Some(accel_profile) = c.accel_profile {
            let _ = device.config_accel_set_profile(accel_profile.into());
        } else if let Some(default) = device.config_accel_default_profile() {
            let _ = device.config_accel_set_profile(default);
        }

        if let Some(method) = c.scroll_method {
            let _ = device.config_scroll_set_method(method.into());

            if method == jiji_config::ScrollMethod::OnButtonDown {
                if let Some(button) = c.scroll_button {
                    let _ = device.config_scroll_set_button(button);
                }
                let _ = device.config_scroll_set_button_lock(if c.scroll_button_lock {
                    input::ScrollButtonLockState::Enabled
                } else {
                    input::ScrollButtonLockState::Disabled
                });
            }
        } else if let Some(default) = device.config_scroll_default_method() {
            let _ = device.config_scroll_set_method(default);

            if default == input::ScrollMethod::OnButtonDown {
                if let Some(button) = c.scroll_button {
                    let _ = device.config_scroll_set_button(button);
                }
                let _ = device.config_scroll_set_button_lock(if c.scroll_button_lock {
                    input::ScrollButtonLockState::Enabled
                } else {
                    input::ScrollButtonLockState::Disabled
                });
            }
        }

        if let Some(tap_button_map) = c.tap_button_map {
            let _ = device.config_tap_set_button_map(tap_button_map.into());
        } else if let Some(default) = device.config_tap_default_button_map() {
            let _ = device.config_tap_set_button_map(default);
        }

        if let Some(method) = c.click_method {
            let _ = device.config_click_set_method(method.into());
        } else if let Some(default) = device.config_click_default_method() {
            let _ = device.config_click_set_method(default);
        }
    }

    // This is how Mutter tells apart mice.
    let mut is_trackball = false;
    let mut is_trackpoint = false;
    if let Some(udev_device) = unsafe { device.udev_device() } {
        if udev_device.property_value("ID_INPUT_TRACKBALL").is_some() {
            is_trackball = true;
        }
        if udev_device
            .property_value("ID_INPUT_POINTINGSTICK")
            .is_some()
        {
            is_trackpoint = true;
        }
    }

    let is_mouse = device.has_capability(input::DeviceCapability::Pointer)
        && !is_touchpad
        && !is_trackball
        && !is_trackpoint;
    if is_mouse {
        let c = &config.mouse;
        let _ = device.config_send_events_set_mode(if c.off {
            input::SendEventsMode::DISABLED
        } else {
            input::SendEventsMode::ENABLED
        });
        let _ = device.config_scroll_set_natural_scroll_enabled(c.natural_scroll);
        let _ = device.config_accel_set_speed(c.accel_speed.0);
        let _ = device.config_left_handed_set(c.left_handed);
        let _ = device.config_middle_emulation_set_enabled(c.middle_emulation);

        if let Some(accel_profile) = c.accel_profile {
            let _ = device.config_accel_set_profile(accel_profile.into());
        } else if let Some(default) = device.config_accel_default_profile() {
            let _ = device.config_accel_set_profile(default);
        }

        if let Some(method) = c.scroll_method {
            let _ = device.config_scroll_set_method(method.into());

            if method == jiji_config::ScrollMethod::OnButtonDown {
                if let Some(button) = c.scroll_button {
                    let _ = device.config_scroll_set_button(button);
                }
                let _ = device.config_scroll_set_button_lock(if c.scroll_button_lock {
                    input::ScrollButtonLockState::Enabled
                } else {
                    input::ScrollButtonLockState::Disabled
                });
            }
        } else if let Some(default) = device.config_scroll_default_method() {
            let _ = device.config_scroll_set_method(default);

            if default == input::ScrollMethod::OnButtonDown {
                if let Some(button) = c.scroll_button {
                    let _ = device.config_scroll_set_button(button);
                }
                let _ = device.config_scroll_set_button_lock(if c.scroll_button_lock {
                    input::ScrollButtonLockState::Enabled
                } else {
                    input::ScrollButtonLockState::Disabled
                });
            }
        }
    }

    if is_trackball {
        let c = &config.trackball;
        let _ = device.config_send_events_set_mode(if c.off {
            input::SendEventsMode::DISABLED
        } else {
            input::SendEventsMode::ENABLED
        });
        let _ = device.config_scroll_set_natural_scroll_enabled(c.natural_scroll);
        let _ = device.config_accel_set_speed(c.accel_speed.0);
        let _ = device.config_middle_emulation_set_enabled(c.middle_emulation);
        let _ = device.config_left_handed_set(c.left_handed);

        if let Some(accel_profile) = c.accel_profile {
            let _ = device.config_accel_set_profile(accel_profile.into());
        } else if let Some(default) = device.config_accel_default_profile() {
            let _ = device.config_accel_set_profile(default);
        }

        if let Some(method) = c.scroll_method {
            let _ = device.config_scroll_set_method(method.into());

            if method == jiji_config::ScrollMethod::OnButtonDown {
                if let Some(button) = c.scroll_button {
                    let _ = device.config_scroll_set_button(button);
                }
                let _ = device.config_scroll_set_button_lock(if c.scroll_button_lock {
                    input::ScrollButtonLockState::Enabled
                } else {
                    input::ScrollButtonLockState::Disabled
                });
            }
        } else if let Some(default) = device.config_scroll_default_method() {
            let _ = device.config_scroll_set_method(default);

            if default == input::ScrollMethod::OnButtonDown {
                if let Some(button) = c.scroll_button {
                    let _ = device.config_scroll_set_button(button);
                }
                let _ = device.config_scroll_set_button_lock(if c.scroll_button_lock {
                    input::ScrollButtonLockState::Enabled
                } else {
                    input::ScrollButtonLockState::Disabled
                });
            }
        }
    }

    if is_trackpoint {
        let c = &config.trackpoint;
        let _ = device.config_send_events_set_mode(if c.off {
            input::SendEventsMode::DISABLED
        } else {
            input::SendEventsMode::ENABLED
        });
        let _ = device.config_scroll_set_natural_scroll_enabled(c.natural_scroll);
        let _ = device.config_accel_set_speed(c.accel_speed.0);
        let _ = device.config_left_handed_set(c.left_handed);
        let _ = device.config_middle_emulation_set_enabled(c.middle_emulation);

        if let Some(accel_profile) = c.accel_profile {
            let _ = device.config_accel_set_profile(accel_profile.into());
        } else if let Some(default) = device.config_accel_default_profile() {
            let _ = device.config_accel_set_profile(default);
        }

        if let Some(method) = c.scroll_method {
            let _ = device.config_scroll_set_method(method.into());

            if method == jiji_config::ScrollMethod::OnButtonDown {
                if let Some(button) = c.scroll_button {
                    let _ = device.config_scroll_set_button(button);
                }
                let _ = device.config_scroll_set_button_lock(if c.scroll_button_lock {
                    input::ScrollButtonLockState::Enabled
                } else {
                    input::ScrollButtonLockState::Disabled
                });
            }
        } else if let Some(default) = device.config_scroll_default_method() {
            let _ = device.config_scroll_set_method(default);

            if default == input::ScrollMethod::OnButtonDown {
                if let Some(button) = c.scroll_button {
                    let _ = device.config_scroll_set_button(button);
                }
                let _ = device.config_scroll_set_button_lock(if c.scroll_button_lock {
                    input::ScrollButtonLockState::Enabled
                } else {
                    input::ScrollButtonLockState::Disabled
                });
            }
        }
    }

    let is_tablet = device.has_capability(input::DeviceCapability::TabletTool);
    if is_tablet {
        let c = &config.tablet;
        let _ = device.config_send_events_set_mode(if c.off {
            input::SendEventsMode::DISABLED
        } else {
            input::SendEventsMode::ENABLED
        });

        #[rustfmt::skip]
        const IDENTITY_MATRIX: [f32; 6] = [
            1., 0., 0.,
            0., 1., 0.,
        ];

        let _ = device.config_calibration_set_matrix(
            c.calibration_matrix
                .as_deref()
                .and_then(|m| m.try_into().ok())
                .or(device.config_calibration_default_matrix())
                .unwrap_or(IDENTITY_MATRIX),
        );

        let _ = device.config_left_handed_set(c.left_handed);
    }

    let is_touch = device.has_capability(input::DeviceCapability::Touch);
    if is_touch {
        let c = &config.touch;
        let _ = device.config_send_events_set_mode(if c.off {
            input::SendEventsMode::DISABLED
        } else {
            input::SendEventsMode::ENABLED
        });

        #[rustfmt::skip]
        const IDENTITY_MATRIX: [f32; 6] = [
            1., 0., 0.,
            0., 1., 0.,
        ];

        let _ = device.config_calibration_set_matrix(
            c.calibration_matrix
                .as_deref()
                .and_then(|m| m.try_into().ok())
                .or(device.config_calibration_default_matrix())
                .unwrap_or(IDENTITY_MATRIX),
        );
    }
}

pub fn mods_with_binds(mod_key: ModKey, binds: &Binds, triggers: &[Trigger]) -> HashSet<Modifiers> {
    let mut rv = HashSet::new();
    for bind in &binds.0 {
        if !triggers.contains(&bind.key.trigger) {
            continue;
        }

        let mut mods = bind.key.modifiers;
        if mods.contains(Modifiers::COMPOSITOR) {
            mods.remove(Modifiers::COMPOSITOR);
            mods.insert(mod_key.to_modifiers());
        }

        rv.insert(mods);
    }

    rv
}

pub fn mods_with_mouse_binds(mod_key: ModKey, binds: &Binds) -> HashSet<Modifiers> {
    mods_with_binds(
        mod_key,
        binds,
        &[
            Trigger::MouseLeft,
            Trigger::MouseRight,
            Trigger::MouseMiddle,
            Trigger::MouseBack,
            Trigger::MouseForward,
        ],
    )
}

pub fn mods_with_wheel_binds(mod_key: ModKey, binds: &Binds) -> HashSet<Modifiers> {
    mods_with_binds(
        mod_key,
        binds,
        &[
            Trigger::WheelScrollUp,
            Trigger::WheelScrollDown,
            Trigger::WheelScrollLeft,
            Trigger::WheelScrollRight,
        ],
    )
}

pub fn mods_with_finger_scroll_binds(mod_key: ModKey, binds: &Binds) -> HashSet<Modifiers> {
    mods_with_binds(
        mod_key,
        binds,
        &[
            Trigger::TouchpadScrollUp,
            Trigger::TouchpadScrollDown,
            Trigger::TouchpadScrollLeft,
            Trigger::TouchpadScrollRight,
        ],
    )
}

fn grab_allows_hot_corner(grab: &(dyn PointerGrab<State> + 'static)) -> bool {
    let grab = grab.as_any();

    // We lean on the blocklist approach here since it's not a terribly big deal if hot corner
    // works where it shouldn't, but it could prevent some workflows if the hot corner doesn't work
    // when it should.
    //
    // Some notable grabs not mentioned here:
    // - DnDGrab allows hot corner to DnD across workspaces.
    // - ClickGrab keeps pointer focus on the window, so the hot corner doesn't trigger.
    // - Touch grabs: touch doesn't trigger the hot corner.
    if grab.is::<ResizeGrab>() || grab.is::<SpatialMovementGrab>() {
        return false;
    }

    if let Some(grab) = grab.downcast_ref::<MoveGrab>() {
        // Window move allows hot corner to DnD across workspaces.
        if !grab.is_move() {
            return false;
        }
    }

    true
}

/// Returns an iterator over bindings.
///
/// Includes dynamically populated bindings like the MRU UI and the
/// synthetic per-bookmark keybinds.
fn make_binds_iter<'a>(
    config: &'a Config,
    mru: &'a mut WindowMruUi,
    mods: Modifiers,
    bookmark_binds: &'a [Bind],
) -> impl Iterator<Item = &'a Bind> + Clone {
    // Figure out the binds to use depending on whether the MRU is enabled and/or open.
    // Read once: `mru_open_binds` below captures `mru` by unique reference in
    // a closure, which would otherwise conflict with the later `is_open()`
    // reads needed for `bookmark_binds`.
    let is_open = mru.is_open();

    let general_binds = (!is_open).then_some(config.binds.0.iter());
    let general_binds = general_binds.into_iter().flatten();

    let mru_binds =
        (config.recent_windows.on || is_open).then_some(config.recent_windows.binds.iter());
    let mru_binds = mru_binds.into_iter().flatten();

    let mru_open_binds = is_open.then(|| mru.opened_bindings(mods));
    let mru_open_binds = mru_open_binds.into_iter().flatten();

    // Suppressed while the MRU is open, same as `general_binds`: a bookmark
    // bind and the MRU's own overlay binds can never be live together.
    let bookmark_binds = (!is_open).then_some(bookmark_binds.iter());
    let bookmark_binds = bookmark_binds.into_iter().flatten();

    // General binds take precedence over the MRU binds. Reject-on-collision at
    // assign time (against config binds and sibling bookmarks) and hot-reload
    // re-validation (same two checks) keep this chain order semantically
    // irrelevant for the bookmark binds specifically.
    general_binds
        .chain(mru_binds)
        .chain(mru_open_binds)
        .chain(bookmark_binds)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::animation::Clock;
    use crate::layout::bookmarks::BookmarkKeyError;

    #[test]
    fn bindings_suppress_keys() {
        let close_keysym = Keysym::q;
        let bindings = Binds(vec![Bind {
            key: Key {
                trigger: Trigger::Keysym(close_keysym),
                modifiers: Modifiers::COMPOSITOR | Modifiers::CTRL,
            },
            action: Action::CloseWindow,
            repeat: true,
            cooldown: None,
            allow_when_locked: false,
            allow_inhibiting: true,
            hotkey_overlay_title: None,
        }]);

        let comp_mod = ModKey::Super;
        let mut suppressed_keys = HashSet::new();

        let screenshot_ui = ScreenshotUi::new(Clock::default(), Default::default());
        let disable_power_key_handling = false;
        let is_inhibiting_shortcuts = Cell::new(false);

        // The key_code we pick is arbitrary, the only thing
        // that matters is that they are different between cases.

        let close_key_code = Keycode::from(close_keysym.raw() + 8u32);
        let close_key_event = |suppr: &mut HashSet<Keycode>, mods: ModifiersState, pressed| {
            should_intercept_key(
                suppr,
                &bindings.0,
                comp_mod,
                close_key_code,
                close_keysym,
                Some(close_keysym),
                pressed,
                mods,
                &screenshot_ui,
                disable_power_key_handling,
                is_inhibiting_shortcuts.get(),
            )
        };

        // Key event with the code which can't trigger any action.
        let none_key_event = |suppr: &mut HashSet<Keycode>, mods: ModifiersState, pressed| {
            should_intercept_key(
                suppr,
                &bindings.0,
                comp_mod,
                Keycode::from(Keysym::l.raw() + 8),
                Keysym::l,
                Some(Keysym::l),
                pressed,
                mods,
                &screenshot_ui,
                disable_power_key_handling,
                is_inhibiting_shortcuts.get(),
            )
        };

        let mut mods = ModifiersState {
            logo: true,
            ctrl: true,
            ..Default::default()
        };

        // Action press/release.

        let filter = close_key_event(&mut suppressed_keys, mods, true);
        assert!(matches!(
            filter,
            FilterResult::Intercept(Some(Bind {
                action: Action::CloseWindow,
                ..
            }))
        ));
        assert!(suppressed_keys.contains(&close_key_code));

        let filter = close_key_event(&mut suppressed_keys, mods, false);
        assert!(matches!(filter, FilterResult::Intercept(None)));
        assert!(suppressed_keys.is_empty());

        // Remove mod to make it for a binding.

        mods.shift = true;
        let filter = close_key_event(&mut suppressed_keys, mods, true);
        assert!(matches!(filter, FilterResult::Forward));

        mods.shift = false;
        let filter = close_key_event(&mut suppressed_keys, mods, false);
        assert!(matches!(filter, FilterResult::Forward));

        // Just none press/release.

        let filter = none_key_event(&mut suppressed_keys, mods, true);
        assert!(matches!(filter, FilterResult::Forward));

        let filter = none_key_event(&mut suppressed_keys, mods, false);
        assert!(matches!(filter, FilterResult::Forward));

        // Press action, press arbitrary, release action, release arbitrary.

        let filter = close_key_event(&mut suppressed_keys, mods, true);
        assert!(matches!(
            filter,
            FilterResult::Intercept(Some(Bind {
                action: Action::CloseWindow,
                ..
            }))
        ));

        let filter = none_key_event(&mut suppressed_keys, mods, true);
        assert!(matches!(filter, FilterResult::Forward));

        let filter = close_key_event(&mut suppressed_keys, mods, false);
        assert!(matches!(filter, FilterResult::Intercept(None)));

        let filter = none_key_event(&mut suppressed_keys, mods, false);
        assert!(matches!(filter, FilterResult::Forward));

        // Trigger and remove all mods.

        let filter = close_key_event(&mut suppressed_keys, mods, true);
        assert!(matches!(
            filter,
            FilterResult::Intercept(Some(Bind {
                action: Action::CloseWindow,
                ..
            }))
        ));

        mods = Default::default();
        let filter = close_key_event(&mut suppressed_keys, mods, false);
        assert!(matches!(filter, FilterResult::Intercept(None)));

        // Ensure that no keys are being suppressed.
        assert!(suppressed_keys.is_empty());

        // Now test shortcut inhibiting.

        // With inhibited shortcuts, we don't intercept our shortcut.
        is_inhibiting_shortcuts.set(true);

        mods = ModifiersState {
            logo: true,
            ctrl: true,
            ..Default::default()
        };

        let filter = close_key_event(&mut suppressed_keys, mods, true);
        assert!(matches!(filter, FilterResult::Forward));
        assert!(suppressed_keys.is_empty());

        let filter = close_key_event(&mut suppressed_keys, mods, false);
        assert!(matches!(filter, FilterResult::Forward));
        assert!(suppressed_keys.is_empty());

        // Toggle it off after pressing the shortcut.
        let filter = close_key_event(&mut suppressed_keys, mods, true);
        assert!(matches!(filter, FilterResult::Forward));
        assert!(suppressed_keys.is_empty());

        is_inhibiting_shortcuts.set(false);

        let filter = close_key_event(&mut suppressed_keys, mods, false);
        assert!(matches!(filter, FilterResult::Forward));
        assert!(suppressed_keys.is_empty());

        // Toggle it on after pressing the shortcut.
        let filter = close_key_event(&mut suppressed_keys, mods, true);
        assert!(matches!(
            filter,
            FilterResult::Intercept(Some(Bind {
                action: Action::CloseWindow,
                ..
            }))
        ));
        assert!(suppressed_keys.contains(&close_key_code));

        is_inhibiting_shortcuts.set(true);

        let filter = close_key_event(&mut suppressed_keys, mods, false);
        assert!(matches!(filter, FilterResult::Intercept(None)));
        assert!(suppressed_keys.is_empty());
    }

    #[test]
    fn comp_mod_handling() {
        let bindings = Binds(vec![
            Bind {
                key: Key {
                    trigger: Trigger::Keysym(Keysym::q),
                    modifiers: Modifiers::COMPOSITOR,
                },
                action: Action::CloseWindow,
                repeat: true,
                cooldown: None,
                allow_when_locked: false,
                allow_inhibiting: true,
                hotkey_overlay_title: None,
            },
            Bind {
                key: Key {
                    trigger: Trigger::Keysym(Keysym::h),
                    modifiers: Modifiers::SUPER,
                },
                action: Action::FocusColumnLeft,
                repeat: true,
                cooldown: None,
                allow_when_locked: false,
                allow_inhibiting: true,
                hotkey_overlay_title: None,
            },
            Bind {
                key: Key {
                    trigger: Trigger::Keysym(Keysym::j),
                    modifiers: Modifiers::empty(),
                },
                action: Action::FocusWindowDown,
                repeat: true,
                cooldown: None,
                allow_when_locked: false,
                allow_inhibiting: true,
                hotkey_overlay_title: None,
            },
            Bind {
                key: Key {
                    trigger: Trigger::Keysym(Keysym::k),
                    modifiers: Modifiers::COMPOSITOR | Modifiers::SUPER,
                },
                action: Action::FocusWindowUp,
                repeat: true,
                cooldown: None,
                allow_when_locked: false,
                allow_inhibiting: true,
                hotkey_overlay_title: None,
            },
            Bind {
                key: Key {
                    trigger: Trigger::Keysym(Keysym::l),
                    modifiers: Modifiers::SUPER | Modifiers::ALT,
                },
                action: Action::FocusColumnRight,
                repeat: true,
                cooldown: None,
                allow_when_locked: false,
                allow_inhibiting: true,
                hotkey_overlay_title: None,
            },
        ]);

        assert_eq!(
            find_configured_bind(
                &bindings.0,
                ModKey::Super,
                Trigger::Keysym(Keysym::q),
                ModifiersState {
                    logo: true,
                    ..Default::default()
                }
            )
            .as_ref(),
            Some(&bindings.0[0])
        );
        assert_eq!(
            find_configured_bind(
                &bindings.0,
                ModKey::Super,
                Trigger::Keysym(Keysym::q),
                ModifiersState::default(),
            ),
            None,
        );

        assert_eq!(
            find_configured_bind(
                &bindings.0,
                ModKey::Super,
                Trigger::Keysym(Keysym::h),
                ModifiersState {
                    logo: true,
                    ..Default::default()
                }
            )
            .as_ref(),
            Some(&bindings.0[1])
        );
        assert_eq!(
            find_configured_bind(
                &bindings.0,
                ModKey::Super,
                Trigger::Keysym(Keysym::h),
                ModifiersState::default(),
            ),
            None,
        );

        assert_eq!(
            find_configured_bind(
                &bindings.0,
                ModKey::Super,
                Trigger::Keysym(Keysym::j),
                ModifiersState {
                    logo: true,
                    ..Default::default()
                }
            ),
            None,
        );
        assert_eq!(
            find_configured_bind(
                &bindings.0,
                ModKey::Super,
                Trigger::Keysym(Keysym::j),
                ModifiersState::default(),
            )
            .as_ref(),
            Some(&bindings.0[2])
        );

        assert_eq!(
            find_configured_bind(
                &bindings.0,
                ModKey::Super,
                Trigger::Keysym(Keysym::k),
                ModifiersState {
                    logo: true,
                    ..Default::default()
                }
            )
            .as_ref(),
            Some(&bindings.0[3])
        );
        assert_eq!(
            find_configured_bind(
                &bindings.0,
                ModKey::Super,
                Trigger::Keysym(Keysym::k),
                ModifiersState::default(),
            ),
            None,
        );

        assert_eq!(
            find_configured_bind(
                &bindings.0,
                ModKey::Super,
                Trigger::Keysym(Keysym::l),
                ModifiersState {
                    logo: true,
                    alt: true,
                    ..Default::default()
                }
            )
            .as_ref(),
            Some(&bindings.0[4])
        );
        assert_eq!(
            find_configured_bind(
                &bindings.0,
                ModKey::Super,
                Trigger::Keysym(Keysym::l),
                ModifiersState {
                    logo: true,
                    ..Default::default()
                },
            ),
            None,
        );
    }

    #[test]
    fn keys_conflict_normalizes_compositor_modifier_both_directions() {
        let mod_plus_m = Key {
            trigger: Trigger::Keysym(Keysym::m),
            modifiers: Modifiers::COMPOSITOR,
        };
        let super_plus_m = Key {
            trigger: Trigger::Keysym(Keysym::m),
            modifiers: Modifiers::SUPER,
        };
        let alt_plus_m = Key {
            trigger: Trigger::Keysym(Keysym::m),
            modifiers: Modifiers::ALT,
        };

        assert!(
            keys_conflict(mod_plus_m, super_plus_m, ModKey::Super),
            "Mod+M and Super+M must conflict when the mod key is Super"
        );
        assert!(
            keys_conflict(super_plus_m, mod_plus_m, ModKey::Super),
            "the check must be symmetric"
        );
        assert!(
            !keys_conflict(mod_plus_m, alt_plus_m, ModKey::Super),
            "Mod+M and Alt+M must not conflict when the mod key is Super"
        );
    }

    #[test]
    fn keys_conflict_identical_keys_conflict() {
        let key = Key {
            trigger: Trigger::Keysym(Keysym::m),
            modifiers: Modifiers::CTRL | Modifiers::ALT,
        };
        assert!(keys_conflict(key, key, ModKey::Super));
    }

    #[test]
    fn keys_conflict_different_triggers_never_conflict() {
        let a = Key {
            trigger: Trigger::Keysym(Keysym::m),
            modifiers: Modifiers::COMPOSITOR,
        };
        let b = Key {
            trigger: Trigger::Keysym(Keysym::n),
            modifiers: Modifiers::COMPOSITOR,
        };
        assert!(!keys_conflict(a, b, ModKey::Super));
    }

    #[test]
    fn bookmark_key_collides_with_siblings_rejects_normalized_collision() {
        // Bookmark 1 holds Mod+M; assigning Super+M to bookmark 2 must be
        // rejected under ModKey::Super normalization even though the raw
        // `Key`s differ.
        let mod_plus_m = Key {
            trigger: Trigger::Keysym(Keysym::m),
            modifiers: Modifiers::COMPOSITOR,
        };
        let super_plus_m = Key {
            trigger: Trigger::Keysym(Keysym::m),
            modifiers: Modifiers::SUPER,
        };
        let keyed = vec![(1, mod_plus_m)];

        assert!(
            bookmark_key_collides_with_siblings(keyed.into_iter(), 2, super_plus_m, ModKey::Super),
            "a different bookmark's normalized-equal key must collide",
        );
    }

    #[test]
    fn bookmark_key_collides_with_siblings_excludes_own_id() {
        // Re-assigning bookmark 1's own current key must stay idempotent: it
        // is excluded from the sibling sweep by matching `excluding_id`.
        let key = Key {
            trigger: Trigger::Keysym(Keysym::m),
            modifiers: Modifiers::COMPOSITOR,
        };
        let keyed = vec![(1, key)];

        assert!(
            !bookmark_key_collides_with_siblings(keyed.into_iter(), 1, key, ModKey::Super),
            "a bookmark's own bind must not collide with itself",
        );
    }

    #[test]
    fn bookmark_key_collides_with_siblings_no_conflict_for_distinct_keys() {
        let m_key = Key {
            trigger: Trigger::Keysym(Keysym::m),
            modifiers: Modifiers::COMPOSITOR,
        };
        let n_key = Key {
            trigger: Trigger::Keysym(Keysym::n),
            modifiers: Modifiers::COMPOSITOR,
        };
        let keyed = vec![(1, m_key)];

        assert!(!bookmark_key_collides_with_siblings(
            keyed.into_iter(),
            2,
            n_key,
            ModKey::Super
        ));
    }

    #[test]
    fn bookmark_key_collides_with_siblings_catches_same_cycle_assign() {
        // Pins the fix for the staleness window: this asserts against the
        // live `(id, Key)` pairs the dispatch site now reads straight off
        // `Layout::bookmarks()`, not the once-per-dispatch-iteration
        // `Niri::bookmark_binds` mirror — so a second `AssignBookmarkKey`
        // action landing in the same calloop iteration as a first sees the
        // first's key immediately, with no rebuild lag.
        let mod_plus_m = Key {
            trigger: Trigger::Keysym(Keysym::m),
            modifiers: Modifiers::COMPOSITOR,
        };
        let super_plus_m = Key {
            trigger: Trigger::Keysym(Keysym::m),
            modifiers: Modifiers::SUPER,
        };
        // Simulates the layout list immediately after assigning Mod+M to
        // bookmark 1, before any `Niri::bookmark_binds` rebuild.
        let live_list = vec![(1, mod_plus_m)];

        assert!(
            bookmark_key_collides_with_siblings(
                live_list.into_iter(),
                2,
                super_plus_m,
                ModKey::Super
            ),
            "a same-cycle sibling assign must see the just-assigned key immediately",
        );
    }

    #[test]
    fn validate_bookmark_key_candidate_rejects_modifier_less_key() {
        let bare = Key {
            trigger: Trigger::Keysym(Keysym::m),
            modifiers: Modifiers::empty(),
        };

        let result = validate_bookmark_key_candidate(
            bare,
            1,
            std::iter::empty(),
            std::iter::empty(),
            ModKey::Super,
        );

        assert_eq!(
            result,
            Err(BookmarkKeyRejection::Invalid(BookmarkKeyError::NoModifiers))
        );
    }

    #[test]
    fn validate_bookmark_key_candidate_rejects_static_bind_collision_both_directions() {
        // A `Super`-held candidate must collide with a `COMPOSITOR`-held
        // static bind on the same trigger, and vice versa — the same
        // bidirectional `ModKey` normalization `keys_conflict` provides.
        let compositor_bind = Bind {
            key: Key {
                trigger: Trigger::Keysym(Keysym::m),
                modifiers: Modifiers::COMPOSITOR,
            },
            action: Action::CloseWindow,
            repeat: true,
            cooldown: None,
            allow_when_locked: false,
            allow_inhibiting: true,
            hotkey_overlay_title: None,
        };
        let super_candidate = Key {
            trigger: Trigger::Keysym(Keysym::m),
            modifiers: Modifiers::SUPER,
        };
        assert_eq!(
            validate_bookmark_key_candidate(
                super_candidate,
                1,
                std::iter::once(&compositor_bind),
                std::iter::empty(),
                ModKey::Super,
            ),
            Err(BookmarkKeyRejection::Collision {
                key: key_to_wire_string(super_candidate),
                with: BookmarkKeyCollidee::ConfigBind,
            })
        );

        let super_bind = Bind {
            key: Key {
                trigger: Trigger::Keysym(Keysym::m),
                modifiers: Modifiers::SUPER,
            },
            action: Action::CloseWindow,
            repeat: true,
            cooldown: None,
            allow_when_locked: false,
            allow_inhibiting: true,
            hotkey_overlay_title: None,
        };
        let compositor_candidate = Key {
            trigger: Trigger::Keysym(Keysym::m),
            modifiers: Modifiers::COMPOSITOR,
        };
        assert_eq!(
            validate_bookmark_key_candidate(
                compositor_candidate,
                1,
                std::iter::once(&super_bind),
                std::iter::empty(),
                ModKey::Super,
            ),
            Err(BookmarkKeyRejection::Collision {
                key: key_to_wire_string(compositor_candidate),
                with: BookmarkKeyCollidee::ConfigBind,
            })
        );
    }

    #[test]
    fn validate_bookmark_key_candidate_rejects_config_bind_collision_from_any_source() {
        // The validator treats `config_binds` as one opaque chained set — the
        // static/recent-windows split happens at the call site (both dispatch
        // arms chain `config.binds.0` with `config.recent_windows.binds`), so
        // a recent-windows-only bind collides exactly like a static one.
        let recent_windows_bind = Bind {
            key: Key {
                trigger: Trigger::Keysym(Keysym::n),
                modifiers: Modifiers::COMPOSITOR,
            },
            action: Action::FocusColumnLeft,
            repeat: true,
            cooldown: None,
            allow_when_locked: false,
            allow_inhibiting: true,
            hotkey_overlay_title: None,
        };
        let candidate = Key {
            trigger: Trigger::Keysym(Keysym::n),
            modifiers: Modifiers::COMPOSITOR,
        };

        assert_eq!(
            validate_bookmark_key_candidate(
                candidate,
                1,
                std::iter::once(&recent_windows_bind),
                std::iter::empty(),
                ModKey::Super,
            ),
            Err(BookmarkKeyRejection::Collision {
                key: key_to_wire_string(candidate),
                with: BookmarkKeyCollidee::ConfigBind,
            })
        );
    }

    #[test]
    fn validate_bookmark_key_candidate_rejects_sibling_but_not_own_id() {
        let key = Key {
            trigger: Trigger::Keysym(Keysym::m),
            modifiers: Modifiers::COMPOSITOR,
        };
        let keyed = vec![(1, key)];

        // Bookmark 2 cannot take bookmark 1's key.
        assert_eq!(
            validate_bookmark_key_candidate(
                key,
                2,
                std::iter::empty(),
                keyed.clone().into_iter(),
                ModKey::Super,
            ),
            Err(BookmarkKeyRejection::Collision {
                key: key_to_wire_string(key),
                with: BookmarkKeyCollidee::SiblingBookmark,
            })
        );

        // Bookmark 1 re-capturing its own current key is idempotent, not a
        // collision.
        assert!(validate_bookmark_key_candidate(
            key,
            1,
            std::iter::empty(),
            keyed.into_iter(),
            ModKey::Super,
        )
        .is_ok());
    }

    #[test]
    fn validate_bookmark_key_candidate_accepts_valid_chord() {
        let key = Key {
            trigger: Trigger::Keysym(Keysym::m),
            modifiers: Modifiers::COMPOSITOR,
        };

        assert_eq!(
            validate_bookmark_key_candidate(
                key,
                1,
                std::iter::empty(),
                std::iter::empty(),
                ModKey::Super,
            ),
            Ok(BookmarkKey::new(key).expect("a chorded keysym is a valid bookmark key"))
        );
    }

    #[test]
    fn bookmark_rule_capture_label_covers_reachable_arms() {
        let app_id_only =
            BookmarkRule::new(Some("^firefox$".parse().unwrap()), None).expect("valid rule");
        assert_eq!(
            bookmark_rule_capture_label(&app_id_only),
            "app_id~^firefox$"
        );

        let title_only =
            BookmarkRule::new(None, Some("^Inbox$".parse().unwrap())).expect("valid rule");
        assert_eq!(bookmark_rule_capture_label(&title_only), "title~^Inbox$");

        let both = BookmarkRule::new(
            Some("^firefox$".parse().unwrap()),
            Some("^Inbox$".parse().unwrap()),
        )
        .expect("valid rule");
        assert_eq!(
            bookmark_rule_capture_label(&both),
            "app_id~^firefox$, title~^Inbox$"
        );
    }
}
