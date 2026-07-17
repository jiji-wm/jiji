//! Pins `XdgShellHandler::grab`'s modal-gating behavior after routing it
//! through `Niri::active_modal`.
//!
//! One granted counterfactual (no modal open) plus one refusal each for
//! the two modal kinds that previously had no arm in the popup-grab gate
//! (`Mru`, `BookmarkSwitcher`) and so fell through to the no-modal
//! layer/toplevel checks instead of being refused, plus a refusal for the
//! overview, which owns keyboard focus without being a `ModalKind` and so
//! is gated in `grab()` separately from `Niri::active_modal`.

use jiji_config::{Action, MruDirection};
use wayland_client::protocol::wl_surface::WlSurface;

use super::client::ClientId;
use super::fixture::Fixture;
use crate::niri::ModalKind;

fn map_window(f: &mut Fixture, id: ClientId, w: u16, h: u16) -> WlSurface {
    let window = f.client(id).create_window();
    let surface = window.surface.clone();
    window.commit();
    f.roundtrip(id);

    let window = f.client(id).window(&surface);
    window.attach_new_buffer();
    window.set_size(w, h);
    window.ack_last_and_commit();
    f.roundtrip(id);

    surface
}

/// No modal open: the grab must be granted, not dismissed, and the
/// production handout tail must actually register a `popup_grab`. This
/// also empirically confirms the fabricated-serial probe: Smithay performs
/// no serial validation on popup grabs, so a made-up serial reaches the
/// same code path a real one would.
#[test]
fn popup_grab_granted_when_no_modal_open() {
    let mut f = Fixture::new();
    f.add_output(1, (1920, 1080));
    let id = f.add_client();

    let window_surface = map_window(&mut f, id, 100, 100);
    let parent = f.client(id).window(&window_surface).xdg_surface.clone();

    let popup_surface = f.client(id).create_popup(&parent).surface.clone();
    f.client(id).grab_popup(&popup_surface, 1);
    f.client(id).popup(&popup_surface).commit();
    f.double_roundtrip(id);

    assert!(!f.client(id).popup(&popup_surface).popup_done_received);
    assert!(f.niri().popup_grab.is_some());
}

/// The MRU switcher owns keyboard focus while open; a popup grab must be
/// refused (dismissed via `popup_done`) rather than falling through to the
/// no-modal layer/toplevel checks.
#[test]
fn popup_grab_refused_when_mru_open() {
    let mut f = Fixture::new();
    f.add_output(1, (1920, 1080));
    let id = f.add_client();

    let window_surface = map_window(&mut f, id, 100, 100);

    f.niri_state()
        .do_action_inner(
            Action::MruAdvance {
                direction: MruDirection::Forward,
                scope: None,
                filter: None,
            },
            false,
        )
        .unwrap();
    // Not just `is_open()`: pins the exact input `popup_grab_disposition`
    // consumes, so a higher-priority modal being open concurrently can't
    // make this pass through the wrong arm.
    assert_eq!(f.niri().active_modal(), Some(ModalKind::Mru));

    let parent = f.client(id).window(&window_surface).xdg_surface.clone();
    let popup_surface = f.client(id).create_popup(&parent).surface.clone();
    f.client(id).grab_popup(&popup_surface, 1);
    f.client(id).popup(&popup_surface).commit();
    f.double_roundtrip_no_refresh(id);

    // Asserted inside the pre-refresh window: this pins the proactive refusal
    // in `grab()` itself, not the reactive cleanup `update_keyboard_focus`
    // would otherwise perform on the next refresh.
    assert!(f.client(id).popup(&popup_surface).popup_done_received);
    assert!(f.niri().popup_grab.is_none());

    // Steady-state coverage: still refused after a normal refreshing
    // roundtrip.
    f.double_roundtrip(id);
    assert!(f.niri().popup_grab.is_none());
}

/// The overview owns keyboard focus while open, but is not a `ModalKind`;
/// a toplevel-rooted popup grab must still be refused proactively.
#[test]
fn popup_grab_refused_when_overview_open() {
    let mut f = Fixture::new();
    f.add_output(1, (1920, 1080));
    let id = f.add_client();

    let window_surface = map_window(&mut f, id, 100, 100);

    f.niri_state()
        .do_action_inner(Action::ToggleOverview, false)
        .unwrap();
    // Not just `is_overview_open()`: pins that this refusal is reached via
    // the overview gate and not `Niri::active_modal`, since the overview is
    // deliberately not a `ModalKind`.
    assert_eq!(f.niri().active_modal(), None);
    assert!(f.niri().layout.is_overview_open());

    let parent = f.client(id).window(&window_surface).xdg_surface.clone();
    let popup_surface = f.client(id).create_popup(&parent).surface.clone();
    f.client(id).grab_popup(&popup_surface, 1);
    f.client(id).popup(&popup_surface).commit();
    f.double_roundtrip_no_refresh(id);

    // Asserted inside the pre-refresh window: this pins the proactive refusal
    // in `grab()` itself, not the reactive cleanup `update_keyboard_focus`
    // would otherwise perform on the next refresh.
    assert!(f.client(id).popup(&popup_surface).popup_done_received);
    assert!(f.niri().popup_grab.is_none());

    // Steady-state coverage: still refused after a normal refreshing
    // roundtrip.
    f.double_roundtrip(id);
    assert!(f.niri().popup_grab.is_none());
}

/// The bookmark switcher owns keyboard focus while open; a popup grab must
/// be refused the same way.
#[test]
fn popup_grab_refused_when_bookmark_switcher_open() {
    let mut f = Fixture::new();
    f.add_output(1, (1920, 1080));
    let id = f.add_client();

    let window_surface = map_window(&mut f, id, 100, 100);

    // A fresh window has no bookmark yet, so this appends one (no confirm
    // dialog) and gives the switcher something visible to hint.
    f.niri_state()
        .do_action_inner(Action::AddBookmark, false)
        .unwrap();
    f.niri_state()
        .do_action_inner(Action::OpenBookmarkSwitcher, false)
        .unwrap();
    // Not just `is_open()`: pins the exact input `popup_grab_disposition`
    // consumes, so a higher-priority modal being open concurrently can't
    // make this pass through the wrong arm.
    assert_eq!(f.niri().active_modal(), Some(ModalKind::BookmarkSwitcher));

    let parent = f.client(id).window(&window_surface).xdg_surface.clone();
    let popup_surface = f.client(id).create_popup(&parent).surface.clone();
    f.client(id).grab_popup(&popup_surface, 1);
    f.client(id).popup(&popup_surface).commit();
    f.double_roundtrip_no_refresh(id);

    // Asserted inside the pre-refresh window: this pins the proactive refusal
    // in `grab()` itself, not the reactive cleanup `update_keyboard_focus`
    // would otherwise perform on the next refresh.
    assert!(f.client(id).popup(&popup_surface).popup_done_received);
    assert!(f.niri().popup_grab.is_none());

    // Steady-state coverage: still refused after a normal refreshing
    // roundtrip.
    f.double_roundtrip(id);
    assert!(f.niri().popup_grab.is_none());
}
