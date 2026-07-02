//! Letter-hint overlay for jumping to a bookmarked window.
//!
//! When open, every currently visible bookmarked window is tagged with a
//! single-letter hint drawn over its tile's top-left corner; pressing a hint
//! jumps straight to that bookmark. The overlay carries no bookmark state of
//! its own: a hint is a stateless shortcut, an open-time snapshot of a
//! `(bookmark id, window id)` pair, and pressing it re-resolves through the
//! ordinary [`crate::layout::Layout::jump_to_bookmark`] path by bookmark id.
//! Modality only suppresses *input*, not client activity: if the hinted
//! window closes while the overlay is still open (its owning client can act
//! independently of what has keyboard focus), [`Layout::remove_window`]
//! prunes the bookmark anchored to it before the removal completes, so the
//! id the stale hint carries is no longer live. Pressing that hint then
//! yields `Err(DoActionError::BookmarkNotFound)` from the jump, which the
//! caller discards — a user-visible no-op, but the overlay still dismisses.
//!
//! There is no backdrop and no animation: the overlay shows and dismisses
//! instantly. It is purely a visual plus key-capture layer, so any pointer
//! press dismisses it (handled by the caller) and it never gates pointer
//! hit-testing.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use ordered_float::NotNan;
use pangocairo::cairo::{self, ImageSurface};
use pangocairo::pango::FontDescription;
use smithay::backend::renderer::element::utils::RescaleRenderElement;
use smithay::backend::renderer::element::Kind;
use smithay::desktop::Window;
use smithay::input::keyboard::Keysym;
use smithay::output::Output;
use smithay::reexports::gbm::Format as Fourcc;
use smithay::utils::Transform;

use crate::layout::{Layout, LayoutElement};
use crate::niri_render_elements;
use crate::render_helpers::memory::MemoryBuffer;
use crate::render_helpers::primary_gpu_texture::PrimaryGpuTextureRenderElement;
use crate::render_helpers::renderer::NiriRenderer;
use crate::render_helpers::texture::{TextureBuffer, TextureRenderElement};
use crate::utils::to_physical_precise_round;
use crate::window::Mapped;

/// Letters used for hints, home-row-first. Assigned to visible bookmarked
/// windows in bookmark-list order; visible bookmarks past the end of this
/// string get no hint (a boundary — curated bookmark lists are small).
const HINT_KEYS: &str = "asdfghjklqwertyuiopzxcvbnm";

const PADDING: i32 = 8;
const BORDER: i32 = 4;
const FONT: &str = "mono 16px";

/// Cache key: which hint letter, at which output scale.
type BufferKey = (char, NotNan<f64>);

/// One hint: the letter shown, the keysym that selects it, the bookmark it
/// jumps to, and the window it is drawn over.
///
/// `keysym` is precomputed at open ([`Keysym::from_char`] of `letter`) so
/// matching an incoming press is a pure comparison. Generic over the window id
/// type only so the assignment/matching logic can be unit-tested with a cheap
/// stand-in; production always uses [`Window`], the layout element id type.
///
/// Two bookmarks anchored on the same window can both receive a hint; that is
/// accepted (pressing either jumps to its own bookmark id) rather than an
/// invariant this module enforces.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Hint<Id> {
    letter: char,
    keysym: Keysym,
    bookmark_id: u64,
    window: Id,
}

/// A non-empty set of open hints, all of which rasterised successfully at
/// scale 1 (see [`BookmarkSwitcher::open`]'s pre-rasterise-and-retain step).
/// Keeps "matchable by [`match_keysym`]" and "drawable by
/// [`BookmarkSwitcher::render_output`]" in sync by construction: there is no
/// way to build a [`State::Open`] whose hint list is empty or contains an
/// undrawable hint.
struct Hints<Id>(Vec<Hint<Id>>);

impl<Id> Hints<Id> {
    fn new(hints: Vec<Hint<Id>>) -> Option<Self> {
        if hints.is_empty() {
            None
        } else {
            Some(Self(hints))
        }
    }

    fn as_slice(&self) -> &[Hint<Id>] {
        &self.0
    }
}

enum State<Id> {
    Closed,
    Open { hints: Hints<Id> },
}

impl<Id> State<Id> {
    fn is_open(&self) -> bool {
        matches!(self, State::Open { .. })
    }

    fn hint_for_keysym(&self, raw: Keysym) -> Option<u64> {
        let State::Open { hints } = self else {
            return None;
        };
        match_keysym(hints.as_slice(), raw)
    }
}

pub struct BookmarkSwitcher {
    state: State<Window>,
    /// Rasterised hint-letter textures, keyed by `(letter, scale)`. Populated
    /// lazily: [`Self::open`] pre-rasterises every needed letter at scale 1 as
    /// the fail-safe fallback; [`Self::render_output`] fills in the exact
    /// output scale on demand. A `None` value records a rasterisation failure
    /// so it is not retried every frame.
    ///
    /// The `or_insert_with` closures that populate this map (in `open` and
    /// `render_output`) call only `render_hint`, never anything that touches
    /// `self.buffers` again — re-borrowing it from inside the closure while
    /// the outer `borrow_mut()` guard is still held would panic.
    buffers: RefCell<HashMap<BufferKey, Option<MemoryBuffer>>>,
    /// `(letter, scale)` texture-upload failures already logged, so a
    /// persistently-failing GPU upload warns once instead of every frame.
    warned_uploads: RefCell<HashSet<BufferKey>>,
    /// Whether [`Self::render_output`] has already logged a
    /// zero-drawable-hints frame for the current open session. Reset on
    /// [`Self::open`] so the breadcrumb fires at most once per open, not
    /// every frame.
    warned_empty_frame: Cell<bool>,
}

niri_render_elements! {
    BookmarkSwitcherRenderElement => {
        Texture = RescaleRenderElement<PrimaryGpuTextureRenderElement>,
    }
}

impl Default for BookmarkSwitcher {
    fn default() -> Self {
        Self::new()
    }
}

impl BookmarkSwitcher {
    pub fn new() -> Self {
        Self {
            state: State::Closed,
            buffers: RefCell::new(HashMap::new()),
            warned_uploads: RefCell::new(HashSet::new()),
            warned_empty_frame: Cell::new(false),
        }
    }

    pub fn is_open(&self) -> bool {
        self.state.is_open()
    }

    pub fn close(&mut self) {
        self.state = State::Closed;
    }

    /// Builds the hint list from the layout and shows the overlay, returning
    /// `true` if it opened.
    ///
    /// Returns `false` (leaving the overlay closed) in two cases, so the
    /// overlay never becomes an invisible key-eater:
    /// - no bookmarked window is currently visible (nothing to tag);
    /// - rasterisation failed for *every* hint letter.
    ///
    /// Re-invoking while already open recomputes the assignment against the
    /// current layout (an idempotent refresh).
    pub fn open(&mut self, layout: &Layout<Mapped>) -> bool {
        // Collect the set of window ids with a visible tile this frame, across
        // every connected output. `tiles_with_render_positions` yields the
        // Incoming activity strip only, so an in-flight activity switch does
        // not double-count the outgoing strip.
        let mut visible: HashSet<Window> = HashSet::new();
        for mon in layout.monitors() {
            let ctx = layout.ctx_for(mon);
            for (ws, _geo) in mon.workspaces_with_render_geo(ctx) {
                for (tile, _pos, tile_visible) in ws.tiles_with_render_positions() {
                    if tile_visible {
                        visible.insert(LayoutElement::id(tile.window()).clone());
                    }
                }
            }
        }

        let mut hints = build_hints(
            layout
                .bookmarks()
                .list()
                .iter()
                .map(|bookmark| (bookmark.id().get(), bookmark.anchor().window().clone())),
            &visible,
        );

        if hints.is_empty() {
            if layout.bookmarks().list().is_empty() {
                debug!("bookmark switcher: no bookmarks exist, not opening");
            } else {
                debug!("bookmark switcher: bookmarks exist but none are visible, not opening");
            }
            return false;
        }

        // Pre-rasterise each hint letter at scale 1, the fallback texture used
        // whenever the exact-scale render (in `render_output`) fails or hasn't
        // been cached yet. A hint whose letter can't be rasterised at all would
        // stay matchable via `hint_for_keysym` but never drawable — an
        // invisible key-eater for that one hint — so such hints are dropped
        // here, before the overlay opens, keeping "matchable" and "drawable"
        // in sync by construction (see `Hints`).
        {
            let mut buffers = self.buffers.borrow_mut();
            hints.retain(|hint| {
                let key = (hint.letter, NotNan::new(1.).expect("1. is not NaN"));
                let buffer = buffers.entry(key).or_insert_with(|| {
                    render_hint(hint.letter, 1.)
                        .inspect_err(|err| {
                            warn!(
                                "bookmark hint '{}' failed to rasterise: {err:?}",
                                hint.letter
                            )
                        })
                        .ok()
                });
                buffer.is_some()
            });
        }

        let Some(hints) = Hints::new(hints) else {
            warn!("bookmark switcher: no hint letter could be rasterised, not opening");
            return false;
        };

        self.state = State::Open { hints };
        self.warned_empty_frame.set(false);
        true
    }

    /// Matches an incoming raw keysym against the open hints, returning the
    /// bookmark id to jump to. `None` when closed or when no hint matches.
    pub fn hint_for_keysym(&self, raw: Keysym) -> Option<u64> {
        let id = self.state.hint_for_keysym(raw)?;
        // A matched hint is about to dispatch a jump whose result the caller
        // discards (matching the MRU precedent); log which hint fired so a
        // press that turns out to be a no-op (e.g. the bookmarked window has
        // since become unresolvable) is diagnosable.
        if let State::Open { hints } = &self.state {
            if let Some(hint) = hints.as_slice().iter().find(|h| h.bookmark_id == id) {
                debug!(
                    "bookmark switcher: hint '{}' matched, dispatching jump to bookmark {id}",
                    hint.letter
                );
            }
        }
        Some(id)
    }

    pub fn render_output<R: NiriRenderer>(
        &self,
        layout: &Layout<Mapped>,
        output: &Output,
        renderer: &mut R,
        push: &mut dyn FnMut(BookmarkSwitcherRenderElement),
    ) {
        let State::Open { hints } = &self.state else {
            return;
        };
        let _span = tracy_client::span!("BookmarkSwitcher::render_output");

        let Some(mon) = layout.monitor_for_output(output) else {
            return;
        };
        let ctx = layout.ctx_for(mon);
        let zoom = mon.overview_zoom();
        let scale = output.current_scale().fractional_scale();

        let mut drew_any = false;

        for (ws, geo) in mon.workspaces_with_render_geo(ctx) {
            for (tile, tile_pos, visible) in ws.tiles_with_render_positions() {
                if !visible {
                    continue;
                }
                let window = LayoutElement::id(tile.window());
                let Some(hint) = hints.as_slice().iter().find(|hint| &hint.window == window) else {
                    continue;
                };

                // Hint anchor in output-local logical coordinates: the tile's
                // on-screen top-left. This is the inverse of the overview
                // branch of `Monitor::window_under`, which maps a pointer via
                // `(pos_within_output - geo.loc).downscale(zoom)`; here we go
                // the other way, `geo.loc + tile_pos * zoom`. The hint texture
                // is then rescaled by `zoom` around this point so it shrinks
                // together with its tile in the overview.
                let anchor = geo.loc + tile_pos.upscale(zoom);

                let buffer = {
                    let mut buffers = self.buffers.borrow_mut();
                    let fallback = buffers
                        .get(&(hint.letter, NotNan::new(1.).expect("1. is not NaN")))
                        .cloned()
                        .flatten();
                    let at_scale = buffers
                        .entry((hint.letter, NotNan::new(scale).expect("scale is not NaN")))
                        .or_insert_with(|| {
                            render_hint(hint.letter, scale)
                                .inspect_err(|err| {
                                    warn!(
                                        "bookmark hint '{}' failed to rasterise at scale {scale}: \
                                         {err:?}",
                                        hint.letter
                                    )
                                })
                                .ok()
                        })
                        .clone();
                    at_scale.or(fallback)
                };
                let Some(buffer) = buffer else {
                    continue;
                };

                let Ok(texture) =
                    TextureBuffer::from_memory_buffer(renderer.as_gles_renderer(), &buffer)
                else {
                    let key = (hint.letter, NotNan::new(scale).expect("scale is not NaN"));
                    if self.warned_uploads.borrow_mut().insert(key) {
                        warn!(
                            "bookmark hint '{}' failed to upload as a texture at scale {scale}",
                            hint.letter
                        );
                    }
                    continue;
                };

                let elem = TextureRenderElement::from_texture_buffer(
                    texture,
                    anchor,
                    1.,
                    None,
                    None,
                    Kind::Unspecified,
                );
                let elem = PrimaryGpuTextureRenderElement(elem);
                let elem = RescaleRenderElement::from_element(
                    elem,
                    anchor.to_physical_precise_round(scale),
                    zoom,
                );
                push(BookmarkSwitcherRenderElement::Texture(elem));
                drew_any = true;
            }
        }

        // Not necessarily a problem on its own: hints can legitimately live on
        // another output. But an open switcher intercepts every key press
        // (see the interception block in `input/mod.rs`) regardless of
        // whether anything is currently drawn, relying on any-key-dismiss to
        // self-heal if every hinted tile has become invisible since opening
        // (animation completed, window closed) — so a persistent all-outputs
        // zero would otherwise be a silent, invisible key-eater. Breadcrumb
        // it once per open rather than re-deriving true cross-output
        // zero-ness here, which would need bookkeeping shared across the
        // per-output calls this method doesn't otherwise need.
        if !drew_any && !self.warned_empty_frame.replace(true) {
            debug!(
                "bookmark switcher: open but drew no hints on output {} this frame",
                output.name()
            );
        }
    }
}

/// Assigns hint letters to visible bookmarks in list order.
///
/// `bookmarks` yields `(bookmark id, anchor window id)` in presentation order;
/// only those whose window is in `visible` are kept, and they are zipped with
/// [`HINT_KEYS`] so the letters are compact (no gaps for hidden bookmarks).
/// Bookmarks past the end of the alphabet get no hint.
fn build_hints<Id: Eq + Hash + Clone>(
    bookmarks: impl Iterator<Item = (u64, Id)>,
    visible: &HashSet<Id>,
) -> Vec<Hint<Id>> {
    bookmarks
        .filter(|(_id, window)| visible.contains(window))
        .zip(HINT_KEYS.chars())
        .map(|((bookmark_id, window), letter)| Hint {
            letter,
            keysym: Keysym::from_char(letter),
            bookmark_id,
            window,
        })
        .collect()
}

/// Resolves a raw keysym to the bookmark id of the matching hint, if any. Pure
/// over the window id type so it is unit-testable without a real [`Window`].
fn match_keysym<Id>(hints: &[Hint<Id>], raw: Keysym) -> Option<u64> {
    hints
        .iter()
        .find(|hint| hint.keysym == raw)
        .map(|hint| hint.bookmark_id)
}

/// True for the pure modifier and lock keysyms this module holds the overlay
/// open for, rather than treating as a dismiss: `Shift`/`Control`/`Super`/
/// `Alt` (both sides, resting or chording), `AltGr`/`ISO_Level3_Shift` and
/// `ISO_Level5_Shift` (so non-US layouts reaching for a third/fifth level are
/// first-class, not penalised), `Caps_Lock`/`Num_Lock`, and `Hyper`/`Meta`
/// (both sides). Pressing any of these while the overlay is open must keep it
/// open. A superset of the accessibility modifier-forwarding list in
/// `State::on_keyboard` (which covers only the 8 base modifiers): this list
/// additionally holds for the lock and level-shift keys above.
pub fn is_modifier_keysym(raw: Keysym) -> bool {
    matches!(
        raw,
        Keysym::Shift_L
            | Keysym::Shift_R
            | Keysym::Control_L
            | Keysym::Control_R
            | Keysym::Super_L
            | Keysym::Super_R
            | Keysym::Alt_L
            | Keysym::Alt_R
            | Keysym::ISO_Level3_Shift
            | Keysym::ISO_Level5_Shift
            | Keysym::Caps_Lock
            | Keysym::Num_Lock
            | Keysym::Hyper_L
            | Keysym::Hyper_R
            | Keysym::Meta_L
            | Keysym::Meta_R
    )
}

/// Rasterises a single hint letter into a padded, bordered, rounded-ish box.
///
/// Modelled on `confirm_dialog::render`: a dark fill, the letter centred in a
/// mono font, and a bright border. The returned [`MemoryBuffer`] is uploaded
/// to a texture at render time.
fn render_hint(letter: char, scale: f64) -> anyhow::Result<MemoryBuffer> {
    let _span = tracy_client::span!("bookmark_switcher::render_hint");

    let markup = format!(
        "<span face='mono'><b>{}</b></span>",
        letter.to_ascii_uppercase()
    );

    let padding: i32 = to_physical_precise_round(scale, PADDING);

    let mut font = FontDescription::from_string(FONT);
    font.set_absolute_size(to_physical_precise_round(scale, font.size()));

    let surface = ImageSurface::create(cairo::Format::ARgb32, 0, 0)?;
    let cr = cairo::Context::new(&surface)?;
    let layout = pangocairo::functions::create_layout(&cr);
    layout.context().set_round_glyph_positions(false);
    layout.set_font_description(Some(&font));
    layout.set_markup(&markup);

    let (mut width, mut height) = layout.pixel_size();
    width += padding * 2;
    height += padding * 2;

    let surface = ImageSurface::create(cairo::Format::ARgb32, width, height)?;
    let cr = cairo::Context::new(&surface)?;
    cr.set_source_rgb(0.1, 0.1, 0.1);
    cr.paint()?;

    cr.move_to(padding.into(), padding.into());
    let layout = pangocairo::functions::create_layout(&cr);
    layout.context().set_round_glyph_positions(false);
    layout.set_font_description(Some(&font));
    layout.set_markup(&markup);

    cr.set_source_rgb(1., 1., 1.);
    pangocairo::functions::show_layout(&cr, &layout);

    cr.move_to(0., 0.);
    cr.line_to(width.into(), 0.);
    cr.line_to(width.into(), height.into());
    cr.line_to(0., height.into());
    cr.line_to(0., 0.);
    cr.set_source_rgb(0.9, 0.6, 0.1);
    // Keep the border width even to avoid blurry edges.
    cr.set_line_width((f64::from(BORDER) / 2. * scale).round() * 2.);
    cr.stroke()?;
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

    /// A fixture bookmark list: bookmark id `n` anchored to window id `n`
    /// (using a plain `u64` as the window-id stand-in, so the pure assignment
    /// logic is exercised without constructing a real `Window`). Bookmark id
    /// and window id share the value only to keep the fixtures readable.
    fn fixture(ids: &[u64]) -> Vec<(u64, u64)> {
        ids.iter().map(|&id| (id, id)).collect()
    }

    fn visible(ids: &[u64]) -> HashSet<u64> {
        ids.iter().copied().collect()
    }

    #[test]
    fn hints_assigned_in_list_order_skipping_hidden_bookmarks() {
        // Bookmarks 1, 2, 3, 4 in order; only 1 and 3 are visible. The letters
        // must be compact (home-row first, no gap for the hidden 2).
        let list = fixture(&[1, 2, 3, 4]);

        let hints = build_hints(list.into_iter(), &visible(&[1, 3]));

        assert_eq!(hints.len(), 2);
        assert_eq!(hints[0].letter, 'a');
        assert_eq!(hints[0].bookmark_id, 1);
        assert_eq!(hints[1].letter, 's');
        assert_eq!(hints[1].bookmark_id, 3);
    }

    #[test]
    fn visible_bookmarks_beyond_the_alphabet_get_no_hint() {
        let ids: Vec<u64> = (1..=30).collect();
        let list = fixture(&ids);

        let hints = build_hints(list.into_iter(), &visible(&ids));

        // 26 letters in the alphabet; the 4 extra visible bookmarks drop off.
        assert_eq!(hints.len(), HINT_KEYS.len());
        assert_eq!(hints.len(), 26);
    }

    #[test]
    fn hint_for_keysym_maps_letters_to_ids_and_misses_to_none() {
        let list = fixture(&[10, 20]);
        let hints = build_hints(list.into_iter(), &visible(&[10, 20]));

        assert_eq!(match_keysym(&hints, Keysym::from_char('a')), Some(10));
        assert_eq!(match_keysym(&hints, Keysym::from_char('s')), Some(20));
        // A letter that was never assigned.
        assert_eq!(match_keysym(&hints, Keysym::from_char('z')), None);
    }

    #[test]
    fn empty_visible_set_yields_no_hints() {
        // The pure signal behind `open`'s "did not open" return: no visible
        // bookmarked window means no hints, so the overlay must not open.
        let list = fixture(&[1, 2, 3]);

        let hints = build_hints(list.into_iter(), &visible(&[]));

        assert!(hints.is_empty());
    }

    #[test]
    fn rebuild_recomputes_assignment_against_new_visibility() {
        // Reopening (idempotent refresh) recomputes letters from the current
        // visible set: the same bookmark can move to a different letter when a
        // preceding bookmark becomes visible.
        let first = build_hints(fixture(&[1, 2, 3]).into_iter(), &visible(&[3]));
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].letter, 'a');
        assert_eq!(first[0].bookmark_id, 3);

        let second = build_hints(fixture(&[1, 2, 3]).into_iter(), &visible(&[2, 3]));
        assert_eq!(second.len(), 2);
        // Now bookmark 2 takes 'a' and 3 is pushed to 's'.
        assert_eq!(second[1].letter, 's');
        assert_eq!(second[1].bookmark_id, 3);
    }

    #[test]
    fn hints_new_rejects_empty_and_accepts_nonempty() {
        // The guard `open` relies on to make an empty-hints `State::Open`
        // unrepresentable, exercised directly with the `u64` id stand-in so
        // it doesn't need a real `Window`.
        assert!(Hints::<u64>::new(Vec::new()).is_none());

        let hints = build_hints(fixture(&[1]).into_iter(), &visible(&[1]));
        assert!(Hints::new(hints).is_some());
    }

    #[test]
    fn close_resets_state_so_no_keysym_can_match() {
        // Goes through the real `Hints::new` guard (a `State::Open` can only
        // be built from a non-empty, already-validated hint list) rather than
        // hand-constructing an invalid open state; the `u64` id stand-in
        // keeps this independent of a real `Window`.
        let hints = build_hints(fixture(&[10]).into_iter(), &visible(&[10]));
        let mut state = State::Open {
            hints: Hints::new(hints).expect("nonempty hints must construct"),
        };
        assert!(state.is_open());

        state = State::Closed;

        assert!(!state.is_open());
        // A closed state must never resolve a hint.
        assert_eq!(state.hint_for_keysym(Keysym::from_char('a')), None);
    }

    #[test]
    fn modifier_keysyms_are_classified() {
        for raw in [
            Keysym::Shift_L,
            Keysym::Shift_R,
            Keysym::Control_L,
            Keysym::Control_R,
            Keysym::Super_L,
            Keysym::Super_R,
            Keysym::Alt_L,
            Keysym::Alt_R,
            Keysym::ISO_Level3_Shift,
            Keysym::ISO_Level5_Shift,
            Keysym::Caps_Lock,
            Keysym::Num_Lock,
            Keysym::Hyper_L,
            Keysym::Hyper_R,
            Keysym::Meta_L,
            Keysym::Meta_R,
        ] {
            assert!(is_modifier_keysym(raw), "{raw:?} must count as a modifier");
        }

        // A hint letter and Escape are not modifiers, so they drive the
        // jump/dismiss branches rather than being swallowed.
        assert!(!is_modifier_keysym(Keysym::from_char('a')));
        assert!(!is_modifier_keysym(Keysym::Escape));
    }

    #[test]
    fn keysym_from_char_is_case_sensitive() {
        // `build_hints` always assigns lowercase `HINT_KEYS` letters, and
        // `input/mod.rs` matches on the unshifted `raw` keysym; both rely on
        // `Keysym::from_char('a')` and `Keysym::from_char('A')` being
        // distinct so a shifted letter press doesn't accidentally match.
        assert_ne!(Keysym::from_char('a'), Keysym::from_char('A'));
    }
}
