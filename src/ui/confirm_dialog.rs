use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Mutex;

use jiji_config::Config;
use ordered_float::NotNan;
use pangocairo::pango::Alignment;
use smithay::backend::renderer::element::utils::RescaleRenderElement;
use smithay::backend::renderer::element::Kind;
use smithay::output::Output;
use smithay::utils::Point;

use crate::animation::{Animation, Clock};
use crate::niri_render_elements;
use crate::render_helpers::memory::MemoryBuffer;
use crate::render_helpers::primary_gpu_texture::PrimaryGpuTextureRenderElement;
use crate::render_helpers::renderer::NiriRenderer;
use crate::render_helpers::solid_color::{SolidColorBuffer, SolidColorRenderElement};
use crate::render_helpers::text::{rasterize, TextBoxStyle};
use crate::render_helpers::texture::{TextureBuffer, TextureRenderElement};
use crate::utils::output_size;

const KEY_NAME: &str = "Enter";
const PADDING: i32 = 16;
const FONT: &str = "sans 14px";
const BORDER: i32 = 8;
const BACKDROP_COLOR: [f32; 4] = [0., 0., 0., 0.4];

/// What the confirm dialog is currently gating.
///
/// The `id` in `RemoveBookmark` is resolved when the request is shown, not
/// when it is confirmed: the open dialog intercepts every key and any
/// pointer press dismisses it, so focus cannot drift under the prompt, and
/// keying the pending request on the id makes it immune to list reorders
/// while the prompt is open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfirmRequest {
    Exit,
    RemoveBookmark { id: u64 },
}

impl ConfirmRequest {
    pub(crate) fn kind(self) -> ConfirmKind {
        match self {
            ConfirmRequest::Exit => ConfirmKind::Exit,
            ConfirmRequest::RemoveBookmark { .. } => ConfirmKind::RemoveBookmark,
        }
    }
}

/// The payload-free shape of a [`ConfirmRequest`], used as the rendered-text
/// and buffer-cache key. The bookmark id does not affect the rendered text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConfirmKind {
    Exit,
    RemoveBookmark,
}

/// Cache key: which dialog text, at which output scale.
type BufferKey = (ConfirmKind, NotNan<f64>);

pub struct ConfirmDialog {
    state: State,
    buffers: RefCell<HashMap<BufferKey, Option<MemoryBuffer>>>,

    clock: Clock,
    config: Rc<RefCell<Config>>,
}

niri_render_elements! {
    ConfirmDialogRenderElement => {
        Texture = RescaleRenderElement<PrimaryGpuTextureRenderElement>,
        SolidColor = SolidColorRenderElement,
    }
}

struct OutputData {
    backdrop: SolidColorBuffer,
}

/// The request lives inside the open/visible variants rather than in a
/// separate `Option<ConfirmRequest>` field, so "open with no pending
/// request" is unrepresentable: every state that can be rendered carries
/// enough to render it. `Hiding` only needs the [`ConfirmKind`] (for text),
/// not the full request, since a fading-out dialog can no longer be
/// re-confirmed.
enum State {
    Hidden,
    Showing(Animation, ConfirmRequest),
    Visible(ConfirmRequest),
    Hiding(Animation, ConfirmKind),
}

impl ConfirmDialog {
    pub fn new(clock: Clock, config: Rc<RefCell<Config>>) -> Self {
        let mut buffers = HashMap::new();
        for kind in [ConfirmKind::Exit, ConfirmKind::RemoveBookmark] {
            let buffer = match render(kind, 1.) {
                Ok(x) => Some(x),
                Err(err) => {
                    warn!("error creating the confirm dialog ({kind:?}): {err:?}");
                    None
                }
            };
            buffers.insert((kind, NotNan::new(1.).unwrap()), buffer);
        }

        Self {
            state: State::Hidden,
            buffers: RefCell::new(buffers),
            clock,
            config,
        }
    }

    pub fn can_show(&self, kind: ConfirmKind) -> bool {
        let buffers = self.buffers.borrow();
        let fallback = &buffers[&(kind, NotNan::new(1.).unwrap())];
        fallback.is_some()
    }

    fn animation(&self, from: f64, to: f64) -> Animation {
        let c = self.config.borrow();
        Animation::new(
            self.clock.clone(),
            from,
            to,
            0.,
            c.animations.exit_confirmation_open_close.0,
        )
    }

    fn value(&self) -> f64 {
        match &self.state {
            State::Hidden => 0.,
            State::Showing(anim, _) | State::Hiding(anim, _) => anim.value(),
            State::Visible(_) => 1.,
        }
    }

    /// Requests the dialog show `request`.
    ///
    /// While already open for the *same* kind, this refreshes the request in
    /// place (e.g. a follow-up repress retargeting which bookmark id would be
    /// removed). While open for a *different* kind, the request is rejected
    /// (returns false) rather than swapping the visible prompt out from under
    /// the user's next keypress.
    ///
    /// Returns true if the dialog will be shown (even if it was already
    /// shown).
    pub fn show(&mut self, request: ConfirmRequest) -> bool {
        if !self.can_show(request.kind()) {
            return false;
        }

        match &mut self.state {
            State::Showing(_, existing) | State::Visible(existing) => {
                if existing.kind() != request.kind() {
                    return false;
                }
                *existing = request;
                return true;
            }
            State::Hidden | State::Hiding(..) => (),
        }

        self.state = State::Showing(self.animation(self.value(), 1.), request);
        true
    }

    /// Returns true if started the hide animation.
    pub fn hide(&mut self) -> bool {
        let kind = match &self.state {
            State::Showing(_, request) | State::Visible(request) => request.kind(),
            State::Hidden | State::Hiding(..) => return false,
        };

        self.state = State::Hiding(self.animation(self.value(), 0.), kind);
        true
    }

    /// The kind of the currently-showing or fading-out request, or `None`
    /// while fully hidden. Used for the a11y label/description, which must
    /// stay accurate whether the dialog is visible or animating out.
    pub fn kind(&self) -> Option<ConfirmKind> {
        match &self.state {
            State::Hidden => None,
            State::Showing(_, request) | State::Visible(request) => Some(request.kind()),
            State::Hiding(_, kind) => Some(*kind),
        }
    }

    /// Accepts the pending request and begins the hide animation, returning
    /// the request that was confirmed. Only meaningful while [`is_open`]
    /// (`Showing` or `Visible`); returns `None` otherwise.
    ///
    /// [`is_open`]: Self::is_open
    pub fn confirm(&mut self) -> Option<ConfirmRequest> {
        let request = match &self.state {
            State::Showing(_, request) | State::Visible(request) => *request,
            State::Hidden | State::Hiding(..) => return None,
        };

        self.state = State::Hiding(self.animation(self.value(), 0.), request.kind());
        Some(request)
    }

    pub fn is_open(&self) -> bool {
        matches!(self.state, State::Showing(..) | State::Visible(_))
    }

    pub fn advance_animations(&mut self) {
        match &mut self.state {
            State::Hidden => (),
            State::Showing(anim, request) => {
                if anim.is_done() {
                    self.state = State::Visible(*request);
                }
            }
            State::Visible(_) => (),
            State::Hiding(anim, _) => {
                if anim.is_clamped_done() {
                    self.state = State::Hidden;
                }
            }
        }
    }

    pub fn are_animations_ongoing(&self) -> bool {
        matches!(self.state, State::Showing(..) | State::Hiding(..))
    }

    pub fn render<R: NiriRenderer>(
        &self,
        renderer: &mut R,
        output: &Output,
        push: &mut dyn FnMut(ConfirmDialogRenderElement),
    ) {
        let (value, clamped_value, kind) = match &self.state {
            State::Hidden => return,
            State::Showing(anim, request) => (anim.value(), anim.clamped_value(), request.kind()),
            State::Visible(request) => (1., 1., request.kind()),
            State::Hiding(anim, kind) => (anim.value(), anim.clamped_value(), *kind),
        };
        let _span = tracy_client::span!("ConfirmDialog::render");

        // Can be out of range when starting from past 0. or 1. from a spring bounce.
        let clamped_value = clamped_value.clamp(0., 1.);

        let scale = output.current_scale().fractional_scale();
        let output_size = output_size(output);

        let mut buffers = self.buffers.borrow_mut();
        let Some(fallback) = buffers[&(kind, NotNan::new(1.).unwrap())].clone() else {
            error!("confirm dialog opened without fallback buffer");
            return;
        };

        let buffer = buffers
            .entry((kind, NotNan::new(scale).unwrap()))
            .or_insert_with(|| {
                render(kind, scale)
                .inspect_err(|err| {
                    warn!("error creating the confirm dialog ({kind:?}) at scale {scale}: {err:?}")
                })
                .ok()
            });
        let buffer = buffer.as_ref().unwrap_or(&fallback);

        let size = buffer.logical_size();
        let Ok(buffer) = TextureBuffer::from_memory_buffer(renderer.as_gles_renderer(), buffer)
        else {
            return;
        };

        let location = (output_size.to_point() - size.to_point()).downscale(2.);
        let mut location = location.to_physical_precise_round(scale).to_logical(scale);
        location.x = f64::max(0., location.x);
        location.y = f64::max(0., location.y);

        let elem = TextureRenderElement::from_texture_buffer(
            buffer,
            location,
            clamped_value as f32,
            None,
            None,
            Kind::Unspecified,
        );
        let elem = PrimaryGpuTextureRenderElement(elem);
        let elem = RescaleRenderElement::from_element(
            elem,
            (location + size.downscale(2.)).to_physical_precise_round(scale),
            value.max(0.) * 0.2 + 0.8,
        );
        push(ConfirmDialogRenderElement::Texture(elem));

        // Backdrop.
        let data = output.user_data().get_or_insert(|| {
            Mutex::new(OutputData {
                backdrop: SolidColorBuffer::new(output_size, BACKDROP_COLOR),
            })
        });
        let mut data = data.lock().unwrap();
        data.backdrop.resize(output_size);

        let elem = SolidColorRenderElement::from_buffer(
            &data.backdrop,
            Point::new(0., 0.),
            clamped_value as f32,
            Kind::Unspecified,
        );
        push(ConfirmDialogRenderElement::SolidColor(elem));
    }
}

fn render(kind: ConfirmKind, scale: f64) -> anyhow::Result<MemoryBuffer> {
    let _span = tracy_client::span!("confirm_dialog::render");

    let markup = text(kind, true);

    rasterize(
        scale,
        FONT,
        Some(TextBoxStyle {
            padding: PADDING,
            border_width: BORDER,
            border_color: [1., 0.3, 0.3],
        }),
        None,
        |layout| {
            layout.set_alignment(Alignment::Center);
            layout.set_markup(&markup);
        },
    )
}

fn text(kind: ConfirmKind, markup: bool) -> String {
    let key = if markup {
        format!("<span face='mono' bgcolor='#2C2C2C'> {KEY_NAME} </span>")
    } else {
        String::from(KEY_NAME)
    };

    match kind {
        ConfirmKind::Exit => format!(
            "Are you sure you want to exit jiji?\n\n\
             Press {key} to confirm."
        ),
        ConfirmKind::RemoveBookmark => format!(
            "Remove the bookmark for the focused window?\n\n\
             Press {key} to confirm."
        ),
    }
}

#[cfg(feature = "dbus")]
pub fn a11y_node(kind: ConfirmKind) -> accesskit::Node {
    let mut node = accesskit::Node::new(accesskit::Role::AlertDialog);
    let label = match kind {
        ConfirmKind::Exit => "Exit jiji",
        ConfirmKind::RemoveBookmark => "Remove bookmark",
    };
    node.set_label(label);
    node.set_description(text(kind, false));
    node.set_modal();
    node
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn dialog() -> ConfirmDialog {
        ConfirmDialog::new(
            Clock::with_time(Duration::ZERO),
            Rc::new(RefCell::new(Config::default())),
        )
    }

    #[test]
    fn show_sets_kind_and_hide_starts_the_close_animation() {
        let mut dialog = dialog();
        assert_eq!(dialog.kind(), None);

        assert!(dialog.show(ConfirmRequest::Exit));
        assert_eq!(dialog.kind(), Some(ConfirmKind::Exit));

        assert!(dialog.hide());
        assert!(!dialog.is_open());
        // The kind is retained through the fade-out: `render()` needs it to
        // keep drawing the closing dialog rather than bailing blank.
        assert_eq!(dialog.kind(), Some(ConfirmKind::Exit));
    }

    #[test]
    fn confirm_returns_the_request_and_begins_hiding() {
        let mut dialog = dialog();
        dialog.show(ConfirmRequest::RemoveBookmark { id: 7 });

        assert_eq!(
            dialog.confirm(),
            Some(ConfirmRequest::RemoveBookmark { id: 7 })
        );
        assert!(!dialog.is_open(), "confirm begins the hide animation");
        // The kind survives into the fade-out (same guarantee as hide()).
        assert_eq!(dialog.kind(), Some(ConfirmKind::RemoveBookmark));
        assert_eq!(dialog.confirm(), None, "no double-confirm while hiding");
    }

    #[test]
    fn show_while_open_refreshes_the_id_for_the_same_kind() {
        let mut dialog = dialog();
        assert!(dialog.show(ConfirmRequest::RemoveBookmark { id: 1 }));
        assert!(dialog.is_open());

        // A second show() for the same kind while open (e.g. a follow-up
        // repress on a different window) refreshes the id rather than
        // stacking or being ignored.
        assert!(dialog.show(ConfirmRequest::RemoveBookmark { id: 2 }));
        assert_eq!(
            dialog.confirm(),
            Some(ConfirmRequest::RemoveBookmark { id: 2 })
        );
    }

    #[test]
    fn show_while_open_rejects_a_different_kind() {
        let mut dialog = dialog();
        assert!(dialog.show(ConfirmRequest::Exit));

        // A bookmark-removal repress must not swap out a visible "Exit
        // jiji?" prompt out from under the user's next Enter.
        assert!(!dialog.show(ConfirmRequest::RemoveBookmark { id: 1 }));
        assert!(dialog.is_open());
        assert_eq!(dialog.kind(), Some(ConfirmKind::Exit));
    }
}
