//! Shared pango/cairo text rasterizer for the fork's overlays.
//!
//! Several fork overlays (the bookmark hint/leader-mode switcher, the MRU
//! switcher's title and scope-panel textures, the exit/remove confirmation
//! dialog) each rasterize a pango layout into an ARGB32 [`MemoryBuffer`]
//! using the same measure-then-draw shape: lay the content out once on a
//! throwaway surface to measure it, then draw it again on a surface sized to
//! fit. [`rasterize`] is that shared body.
//!
//! The house style is fixed, not configurable: a dark `(0.1, 0.1, 0.1)` fill
//! and white text, drawn only when a [`TextBoxStyle`] is given — every
//! current adopter that opts into a filled box wants exactly this fill and
//! text color, so there is no per-caller color knob for either. The one
//! adopter that opts out ([`crate::ui::mru`]'s window-title texture) relies
//! on the surface's transparent-black default so the title floats over
//! whatever it's composited onto; drawing has no border in that case either.
//! Every adopter passes [`MAX_SURFACE_SIZE`] for `max_surface_size`: any
//! rasterized surface can end up uploaded as a GL texture, and at least one
//! shared rasterize path per component carries untrusted or unbounded text —
//! [`crate::ui::mru`]'s title texture and the bookmark switcher's search
//! query and interpolated window titles — even though other call sites on
//! the same components only ever draw short, fixed, developer-controlled
//! strings (the confirm dialog's labels, the MRU scope panel's labels). It's
//! simpler and safer to clamp unconditionally for every adopter than to
//! track which specific call is exposed, so the clamp is load-bearing
//! everywhere, not just on the title path. The parameter stays `Option`
//! rather than being applied unconditionally inside [`rasterize`] so the
//! clamp remains cheap to exercise with small test values instead of only
//! at the real GL-sized limit.

use anyhow::{ensure, Result};
use pangocairo::cairo::{self, ImageSurface};
use pangocairo::pango::{FontDescription, Layout};
use smithay::backend::allocator::Fourcc;
use smithay::utils::Transform;

use crate::render_helpers::memory::MemoryBuffer;
use crate::utils::to_physical_precise_round;

/// Upper bound for a rasterized surface's width and height, in physical pixels.
///
/// Stays strictly under the common 16384 GL `MAX_TEXTURE_SIZE` floor so a
/// texture built from a [`rasterize`]d surface never exceeds it on any
/// adopter's target hardware.
pub const MAX_SURFACE_SIZE: i32 = 16383;

/// Chrome drawn around rasterized text: padding, an even-width border, and
/// the border's color. Passing `None` for the `style` parameter of
/// [`rasterize`] skips both the padding and the border entirely, and also
/// skips the background fill (see the module docs for why the two are
/// linked).
pub struct TextBoxStyle {
    /// Logical padding on every side, scaled via [`to_physical_precise_round`].
    pub padding: i32,
    /// Logical border width; drawn rounded to the nearest even physical
    /// pixel (see the "Keep the border width even" comment at the border
    /// draw site) so the stroke doesn't blur.
    pub border_width: i32,
    /// RGB border color, `cairo::Context::set_source_rgb` component order.
    pub border_color: [f64; 3],
}

/// Rasterizes a pango layout into an ARGB32 [`MemoryBuffer`] at `scale`.
///
/// `set_content` is invoked twice — once against a throwaway sizing layout
/// used only to measure the content's pixel size, and once against the
/// layout that is actually drawn. The two are distinct `pango::Layout`
/// instances, so any configuration `set_content` needs on the drawn output
/// (single-paragraph mode, alignment, markup vs. plain text, ...) must be
/// set unconditionally inside the closure rather than assumed to carry over
/// from the sizing pass.
///
/// When `style` is `Some`, the box is filled with a dark background, the
/// text is drawn in white, and a border is stroked in `style.border_color`
/// around the padded box. When `style` is `None`, nothing is filled or
/// stroked — the surface starts fully transparent and only the (white) text
/// is drawn at the origin.
///
/// `max_surface_size` clamps both the width and the height of the rasterized
/// surface (after padding, before allocation), for callers whose content
/// length isn't bounded by the caller's own UI (e.g. a window title from a
/// client). Content wider or taller than the clamp is silently truncated by
/// the surface bounds, not scaled down.
///
/// # Errors
///
/// Returns `Err` if the measured content (width or height, including
/// padding) is zero, or if cairo surface/context creation fails.
pub fn rasterize(
    scale: f64,
    font: &str,
    style: Option<TextBoxStyle>,
    max_surface_size: Option<i32>,
    set_content: impl Fn(&Layout),
) -> Result<MemoryBuffer> {
    let padding = style
        .as_ref()
        .map_or(0, |s| to_physical_precise_round(scale, s.padding));

    let mut font = FontDescription::from_string(font);
    font.set_absolute_size(to_physical_precise_round(scale, font.size()));

    // Render to a dummy surface to determine the size.
    let surface = ImageSurface::create(cairo::Format::ARgb32, 0, 0)?;
    let cr = cairo::Context::new(&surface)?;
    let layout = pangocairo::functions::create_layout(&cr);
    layout.context().set_round_glyph_positions(false);
    layout.set_font_description(Some(&font));
    set_content(&layout);

    let (mut width, mut height) = layout.pixel_size();
    width += padding * 2;
    height += padding * 2;

    ensure!(width > 0 && height > 0);

    if let Some(max) = max_surface_size {
        width = width.min(max);
        height = height.min(max);
    }

    let surface = ImageSurface::create(cairo::Format::ARgb32, width, height)?;
    let cr = cairo::Context::new(&surface)?;
    if style.is_some() {
        cr.set_source_rgb(0.1, 0.1, 0.1);
        cr.paint()?;
    }

    cr.move_to(padding.into(), padding.into());
    let layout = pangocairo::functions::create_layout(&cr);
    layout.context().set_round_glyph_positions(false);
    layout.set_font_description(Some(&font));
    set_content(&layout);

    cr.set_source_rgb(1., 1., 1.);
    pangocairo::functions::show_layout(&cr, &layout);

    if let Some(style) = &style {
        cr.move_to(0., 0.);
        cr.line_to(width.into(), 0.);
        cr.line_to(width.into(), height.into());
        cr.line_to(0., height.into());
        cr.line_to(0., 0.);
        let [r, g, b] = style.border_color;
        cr.set_source_rgb(r, g, b);
        // Keep the border width even to avoid blurry edges.
        cr.set_line_width((f64::from(style.border_width) / 2. * scale).round() * 2.);
        cr.stroke()?;
    }

    drop(cr);

    let data = surface
        .take_data()
        .expect("surface data is owned and unique");
    let buffer = MemoryBuffer::new(
        data.to_vec(),
        Fourcc::Argb8888,
        (width, height),
        scale,
        Transform::Normal,
    );

    Ok(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FONT: &str = "sans 14px";

    #[test]
    fn boxed_rasterize_adds_exact_padding() {
        for scale in [1., 2.] {
            let padding = 8;
            let bare = rasterize(scale, FONT, None, None, |l| l.set_text("hello")).unwrap();
            let styled = rasterize(
                scale,
                FONT,
                Some(TextBoxStyle {
                    padding,
                    border_width: 4,
                    border_color: [0.9, 0.6, 0.1],
                }),
                None,
                |l| l.set_text("hello"),
            )
            .unwrap();

            // Hardcoded, not `2 * to_physical_precise_round(scale, padding)`:
            // that's the same rounding helper `rasterize` itself uses to size
            // the padding, so reusing it here would make this assertion
            // partly `f(x) == f(x)` and blind to a rounding bug in the
            // shared helper.
            let expected_delta = match scale as i32 {
                1 => 16,
                2 => 32,
                _ => unreachable!("test only exercises scale 1. and 2."),
            };
            assert_eq!(
                styled.size().w - bare.size().w,
                expected_delta,
                "width delta at scale {scale}"
            );
            assert_eq!(
                styled.size().h - bare.size().h,
                expected_delta,
                "height delta at scale {scale}"
            );
        }
    }

    #[test]
    fn rasterize_rejects_empty_content_and_clamps_oversized() {
        let err = rasterize(1., FONT, None, None, |l| l.set_text(""));
        assert!(err.is_err());

        let long = "x".repeat(1000);
        let buffer = rasterize(1., FONT, None, Some(8), |l| l.set_text(&long)).unwrap();
        assert!(buffer.size().w <= 8);
        assert!(buffer.size().h <= 8);

        // Padding and the clamp together: if the padding-add ever moved to
        // after the clamp block, the surface would exceed `max` here even
        // though the two cases above (both padding-less) would stay green.
        let styled = rasterize(
            1.,
            FONT,
            Some(TextBoxStyle {
                padding: 8,
                border_width: 4,
                border_color: [0.9, 0.6, 0.1],
            }),
            Some(8),
            |l| l.set_text(&long),
        )
        .unwrap();
        assert!(styled.size().w <= 8);
        assert!(styled.size().h <= 8);
    }
}
