//! A bare-`Mod`-tap input state machine: press and release the compositor
//! modifier alone, with nothing else pressed in between, to fire a bound
//! [`Trigger::ModTap`](jiji_config::Trigger::ModTap) action. No timeout is
//! involved; a held modifier stays armed indefinitely as long as nothing else
//! disarms it.
//!
//! - **Arm** on a clean press of the modifier: exactly the compositor's modifier keysym, no other
//!   modifiers held, and no other key already pressed. See [`is_bare_mod_press`].
//! - **Disarm** on any other keyboard input (another key's press, whether or not it goes on to be
//!   consumed by accessibility, a modal overlay, or a bind), or on select non-keyboard input that
//!   signals the user has moved on to something else (a pointer button, an axis event, a
//!   touch/tablet/gesture "begin"). Pointer motion alone does not disarm, so a tap survives
//!   incidental mouse movement while the modifier is held.
//! - **Fire** on a clean release of the armed modifier (no other modifiers accumulated since
//!   arming), gated through [`tap_fire_allowed`] for the active modal overlay, a
//!   keyboard-shortcuts-inhibiting client, and the bind's own `allow-inhibiting` flag.
//!
//! A mod press or release that happens mid-interactive-grab (e.g. during a
//! window move/resize) still arms and fires by this same logic; there is no
//! headless input harness in this codebase that reaches pointer grabs, so
//! that interplay is verified manually at deploy rather than pinned by a
//! test here.

use jiji_config::{Action, ModKey, Modifiers};
use smithay::backend::input::Keycode;
use smithay::input::keyboard::Keysym;

use crate::niri::ModalKind;

/// Tracks whether a bare `Mod` tap is armed and, if so, which physical key
/// armed it.
///
/// Invariant: `armed == Some(key_code)` implies `key_code` is physically
/// held. Every release of the armed key clears the state (via
/// [`Self::on_key_release`]) before any modal early-return in the input
/// pipeline can skip that clear, and every other-key press, plus the
/// non-keyboard disarm signals routed through [`Self::disarm`], also clear
/// it.
#[derive(Debug, Default)]
pub struct ModTapState {
    armed: Option<Keycode>,
}

impl ModTapState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Called on every key press, including ones later consumed by
    /// accessibility, a modal overlay, or a matched bind.
    ///
    /// A press of the already-armed key itself is left alone: it cannot
    /// physically occur while that key is held (hardware repeats are not
    /// delivered as separate input events; jiji synthesizes bind repeat
    /// itself). Any other key press disarms unconditionally.
    pub fn on_key_press(&mut self, key_code: Keycode) {
        if self.armed != Some(key_code) {
            self.armed = None;
        }
    }

    /// Arms the state machine on `key_code`. Callers must have already
    /// confirmed the press was a clean bare-modifier press (see
    /// [`is_bare_mod_press`]).
    pub fn arm(&mut self, key_code: Keycode) {
        self.armed = Some(key_code);
    }

    /// Called on every key release. Returns whether `key_code` was the armed
    /// key, i.e. whether this release is a fire candidate.
    ///
    /// Always clears the armed state, regardless of the match, so the
    /// physically-held invariant holds even if the caller's own dispatch of
    /// a fire candidate is later skipped by a modal gate.
    pub fn on_key_release(&mut self, key_code: Keycode) -> bool {
        let was_armed = self.armed == Some(key_code);
        self.armed = None;
        was_armed
    }

    /// Clears the armed state. Used for non-keyboard disarm signals (pointer
    /// button, axis, touch/tablet/gesture begin) and for the VT-switch/
    /// suspend clears, which may not deliver the key release that would
    /// otherwise clear it.
    pub fn disarm(&mut self) {
        self.armed = None;
    }
}

/// Whether a just-pressed key is a clean bare press of `mod_key`'s modifier:
/// no other key already held, the post-press modifier state is *exactly*
/// `mod_key`'s own modifier (rejecting chords), and `raw` is one of that
/// modifier's keysyms.
///
/// Exhaustively matched on `mod_key` (no wildcard arm) so that a future
/// `ModKey` variant fails to compile here instead of silently never arming.
pub fn is_bare_mod_press(
    mod_key: ModKey,
    raw: Option<Keysym>,
    modifiers: Modifiers,
    other_keys_held: bool,
) -> bool {
    if other_keys_held {
        return false;
    }

    if modifiers != mod_key.to_modifiers() {
        return false;
    }

    let Some(raw) = raw else {
        return false;
    };

    match mod_key {
        ModKey::Super => matches!(raw, Keysym::Super_L | Keysym::Super_R),
        ModKey::Alt => matches!(raw, Keysym::Alt_L | Keysym::Alt_R),
        ModKey::Ctrl => matches!(raw, Keysym::Control_L | Keysym::Control_R),
        ModKey::Shift => matches!(raw, Keysym::Shift_L | Keysym::Shift_R),
        ModKey::IsoLevel3Shift => raw == Keysym::ISO_Level3_Shift,
        ModKey::IsoLevel5Shift => raw == Keysym::ISO_Level5_Shift,
    }
}

/// Whether a mod-tap fire candidate is actually allowed to dispatch, given
/// the currently active modal overlay, a keyboard-shortcuts-inhibiting
/// client, and the matched bind's own `allow-inhibiting` flag.
///
/// Exhaustively matched on `active_modal` (all five [`ModalKind`] arms, no
/// wildcard): a modal gate that skips an arm here is exactly the fork's
/// precedent popup-grab defect class, so a future `ModalKind` variant must
/// force a decision at this call site rather than silently falling through.
pub fn tap_fire_allowed(
    active_modal: Option<ModalKind>,
    action: &Action,
    is_inhibiting_shortcuts: bool,
    allow_inhibiting: bool,
) -> bool {
    let modal_allows = match active_modal {
        None => true,
        // The locked-session gate lives in `handle_bind`/`do_action` via
        // `allow_when_locked`, like every other bind; blocking taps here too
        // would break `allow-when-locked` binds on this trigger.
        Some(ModalKind::LockScreen) => true,
        Some(ModalKind::ScreenshotUi) => super::allowed_during_screenshot(action),
        // Both own keyboard focus outright and can open mid-hold (e.g. via
        // an IPC-triggered action) without a disarming key press.
        Some(ModalKind::ConfirmDialog) | Some(ModalKind::BookmarkSwitcher) => false,
        // Defense-in-depth, not the primary gate: the MRU-open
        // all-modifiers-released check in `on_keyboard` intercepts an armed
        // release first, since an armed release implies empty modifiers. This
        // arm is only reached if that upstream ordering ever changes.
        Some(ModalKind::Mru) => false,
    };

    modal_allows && !(is_inhibiting_shortcuts && allow_inhibiting)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keycode(raw: u32) -> Keycode {
        Keycode::from(raw)
    }

    #[test]
    fn clean_tap_arms_then_fires_then_stays_disarmed() {
        let mut state = ModTapState::new();
        let mod_key = keycode(100);

        state.on_key_press(mod_key);
        state.arm(mod_key);

        assert!(state.on_key_release(mod_key));
        // The fire already cleared the armed state, so a second release of
        // the same (no longer held) key is not a fire candidate.
        assert!(!state.on_key_release(mod_key));
    }

    #[test]
    fn other_key_press_disarms_whether_or_not_it_later_fires_a_bind() {
        let mut state = ModTapState::new();
        let mod_key = keycode(100);
        let other_key = keycode(200);

        // Chord-abort: another key goes down while the mod is held, with no
        // bind ultimately matching it.
        state.on_key_press(mod_key);
        state.arm(mod_key);
        state.on_key_press(other_key);
        assert!(!state.on_key_release(mod_key));

        // Bind-fired: same disarm must hold even when the other key goes on
        // to match a config bind (`on_key_press` doesn't know or care).
        state.on_key_press(mod_key);
        state.arm(mod_key);
        state.on_key_press(other_key);
        state.on_key_release(other_key);
        assert!(!state.on_key_release(mod_key));
    }

    #[test]
    fn other_mod_variant_keycode_press_disarms() {
        let mut state = ModTapState::new();
        let super_key = keycode(100);
        let alt_key = keycode(101);

        state.on_key_press(super_key);
        state.arm(super_key);
        // Pressing a *different* modifier's keycode (e.g. Alt while Super is
        // armed) is still "some other key" to the state machine.
        state.on_key_press(alt_key);
        assert!(!state.on_key_release(super_key));
    }

    #[test]
    fn release_without_arm_returns_false() {
        let mut state = ModTapState::new();
        assert!(!state.on_key_release(keycode(100)));
    }

    #[test]
    fn disarm_prevents_fire() {
        let mut state = ModTapState::new();
        let mod_key = keycode(100);

        state.on_key_press(mod_key);
        state.arm(mod_key);
        // A pointer button, axis, or similar non-keyboard signal routes here.
        state.disarm();
        assert!(!state.on_key_release(mod_key));
    }

    #[test]
    fn press_of_the_armed_key_itself_does_not_disarm() {
        let mut state = ModTapState::new();
        let mod_key = keycode(100);

        state.on_key_press(mod_key);
        state.arm(mod_key);
        // A repeat-like press of the same key (not physically possible from
        // real hardware while held, but the method must not assume that).
        state.on_key_press(mod_key);
        assert!(state.on_key_release(mod_key));
    }

    #[test]
    fn accepts_left_and_right_variant_keysyms_per_mod_key() {
        // Hardcoded independently of `is_bare_mod_press`'s own match, per the
        // discipline against deriving a test oracle from the table under
        // test.
        let cases = [
            (ModKey::Super, Keysym::Super_L),
            (ModKey::Super, Keysym::Super_R),
            (ModKey::Alt, Keysym::Alt_L),
            (ModKey::Alt, Keysym::Alt_R),
            (ModKey::Ctrl, Keysym::Control_L),
            (ModKey::Ctrl, Keysym::Control_R),
            (ModKey::Shift, Keysym::Shift_L),
            (ModKey::Shift, Keysym::Shift_R),
            (ModKey::IsoLevel3Shift, Keysym::ISO_Level3_Shift),
            (ModKey::IsoLevel5Shift, Keysym::ISO_Level5_Shift),
        ];

        for (mod_key, raw) in cases {
            assert!(
                is_bare_mod_press(mod_key, Some(raw), mod_key.to_modifiers(), false),
                "{mod_key:?} should accept {raw:?}"
            );
        }
    }

    #[test]
    fn rejects_non_mod_keysym() {
        assert!(!is_bare_mod_press(
            ModKey::Super,
            Some(Keysym::q),
            ModKey::Super.to_modifiers(),
            false,
        ));
    }

    #[test]
    fn rejects_extra_modifiers() {
        let chorded = ModKey::Super.to_modifiers() | Modifiers::SHIFT;
        assert!(!is_bare_mod_press(
            ModKey::Super,
            Some(Keysym::Super_L),
            chorded,
            false,
        ));
    }

    #[test]
    fn rejects_other_keys_held() {
        assert!(!is_bare_mod_press(
            ModKey::Super,
            Some(Keysym::Super_L),
            ModKey::Super.to_modifiers(),
            true,
        ));
    }

    #[test]
    fn rejects_missing_raw_keysym() {
        // Some layouts genuinely have `raw_latin_sym_or_raw_current_sym()`
        // return `None`; the gate must fail closed rather than panic or fall
        // through to the match on `mod_key`.
        assert!(!is_bare_mod_press(
            ModKey::Super,
            None,
            ModKey::Super.to_modifiers(),
            false,
        ));
    }

    #[test]
    fn rejects_empty_modifiers_despite_mod_keysym() {
        // The state an a11y-grabbed or otherwise filtered press could
        // present: the raw keysym looks right but XKB modifier state hasn't
        // (or no longer) reflects it.
        assert!(!is_bare_mod_press(
            ModKey::Super,
            Some(Keysym::Super_L),
            Modifiers::empty(),
            false,
        ));
    }

    #[test]
    fn modal_gate_matrix() {
        let action = Action::ToggleOverview;
        assert!(tap_fire_allowed(None, &action, false, false));
        assert!(tap_fire_allowed(
            Some(ModalKind::LockScreen),
            &action,
            false,
            false
        ));
        assert!(!tap_fire_allowed(
            Some(ModalKind::ConfirmDialog),
            &action,
            false,
            false
        ));
        assert!(!tap_fire_allowed(
            Some(ModalKind::Mru),
            &action,
            false,
            false
        ));
        assert!(!tap_fire_allowed(
            Some(ModalKind::BookmarkSwitcher),
            &action,
            false,
            false
        ));
    }

    #[test]
    fn screenshot_ui_allows_only_allowed_during_screenshot_actions() {
        // `Action::Suspend` is in `allowed_during_screenshot`'s list;
        // `Action::ToggleOverview` is not.
        assert!(tap_fire_allowed(
            Some(ModalKind::ScreenshotUi),
            &Action::Suspend,
            false,
            false
        ));
        assert!(!tap_fire_allowed(
            Some(ModalKind::ScreenshotUi),
            &Action::ToggleOverview,
            false,
            false
        ));
    }

    #[test]
    fn inhibitor_denies_iff_allow_inhibiting() {
        let action = Action::ToggleOverview;
        assert!(!tap_fire_allowed(None, &action, true, true));
        assert!(tap_fire_allowed(None, &action, true, false));
    }
}
