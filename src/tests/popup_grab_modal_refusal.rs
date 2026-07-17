//! Pins `XdgShellHandler::grab`'s modal-gating behavior.
//!
//! A granted counterfactual with no modal open, confirming the popup-grab
//! test harness itself (positioner completeness, explicit-parent rooting,
//! the client-side `grab` + `commit` request pair) reaches the production
//! handler and registers a real `popup_grab`.

use wayland_client::protocol::wl_surface::WlSurface;

use super::client::ClientId;
use super::fixture::Fixture;

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
