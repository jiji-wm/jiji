//! Letter-hint overlay for jumping to a bookmarked window, with an optional
//! leader/prefix mode layered on top.
//!
//! In its base ([`BookmarkSwitcher::open`]) form, every currently visible
//! bookmarked window is tagged with a single-letter hint drawn over its
//! tile's top-left corner; pressing a hint jumps straight to that bookmark.
//! [`BookmarkSwitcher::open_mode`] opens the same hints plus a one-line
//! command sheet anchored at the bottom of each output, and reads a handful
//! of add / remove-focused / walk-backward-forward command letters (see
//! [`MODE_COMMANDS`], the single source of truth for the table) ahead of
//! hint matching — a leader chord that lets bookmark management happen
//! without leaving the keyboard. Mode entry does not require any bookmarks
//! to be visible: the command sheet is useful on its own (e.g. to add the
//! first bookmark).
//!
//! The overlay carries no bookmark state of its own: a hint is a stateless
//! shortcut, an open-time snapshot of a `(bookmark id, window id)` pair, and
//! pressing it re-resolves through the ordinary
//! [`crate::layout::Layout::jump_to_bookmark`] path by bookmark id. A
//! command letter re-resolves through the ordinary bookmark actions
//! (`AddBookmark`, `RemoveBookmark`, `WalkBookmarksForward`/`Backward`) the
//! same way a keybind would. Modality only suppresses *input*, not client
//! activity: if the hinted window closes while the overlay is still open
//! (its owning client can act independently of what has keyboard focus),
//! [`Layout::remove_window`] prunes the bookmark anchored to it before the
//! removal completes, so the id the stale hint carries is no longer live.
//! Pressing that hint then yields `Err(DoActionError::BookmarkNotFound)` from
//! the jump, which the caller discards — a user-visible no-op, but the
//! overlay still dismisses.
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
use smithay::utils::{Logical, Point, Transform};

use crate::layout::{Layout, LayoutElement};
use crate::niri_render_elements;
use crate::render_helpers::memory::MemoryBuffer;
use crate::render_helpers::primary_gpu_texture::PrimaryGpuTextureRenderElement;
use crate::render_helpers::renderer::NiriRenderer;
use crate::render_helpers::texture::{TextureBuffer, TextureRenderElement};
use crate::utils::{output_size, to_physical_precise_round};
use crate::window::Mapped;

/// Letters used for hints, home-row-first. Assigned to visible bookmarked
/// windows in bookmark-list order; visible bookmarks past the end of this
/// string get no hint (a boundary — curated bookmark lists are small). The
/// standalone switcher uses the full alphabet; [`mode_hint_alphabet`] strips
/// the [`MODE_COMMANDS`] letters out for leader mode.
const HINT_KEYS: &str = "asdfghjklqwertyuiopzxcvbnm";

const PADDING: i32 = 8;
const BORDER: i32 = 4;
const FONT: &str = "mono 16px";

/// A leader-mode command, reachable only while [`State::Mode`] is open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModeCommand {
    Add,
    RemoveFocused,
    WalkBackward,
    WalkForward,
}

/// Single source of truth for leader-mode routing: which letter triggers
/// which [`ModeCommand`], and the label shown for it on the command sheet.
/// Drives three things from one place so they cannot drift apart: keysym
/// matching in [`command_for_keysym`], the in-mode hint alphabet
/// ([`mode_hint_alphabet`]), and the sheet text ([`mode_sheet_markup`]).
const MODE_COMMANDS: &[(char, ModeCommand, &str)] = &[
    ('a', ModeCommand::Add, "add"),
    ('d', ModeCommand::RemoveFocused, "remove"),
    ('x', ModeCommand::RemoveFocused, "remove"),
    (',', ModeCommand::WalkBackward, "walk"),
    ('.', ModeCommand::WalkForward, "walk"),
];

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
/// Keeps "matchable by [`press_outcome_core`]" and "drawable by
/// [`BookmarkSwitcher::render_output`]" in sync by construction: there is no
/// way to build a [`State::Hints`] whose hint list is empty or contains an
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

/// The overlay's open states.
///
/// `Mode`'s hint list, unlike `Hints`', may legitimately be empty (the
/// command sheet is the visible artifact then), but every hint it does carry
/// still passed the same rasterise-at-scale-1 retain step, and `sheet` is
/// the scale-1 rasterised command-sheet buffer — carrying it in the variant
/// makes "mode open but nothing drawable" unrepresentable, the same
/// validated-construction discipline `Hints` applies to the standalone
/// switcher.
enum State<Id> {
    Closed,
    Hints {
        hints: Hints<Id>,
    },
    Mode {
        hints: Vec<Hint<Id>>,
        sheet: MemoryBuffer,
    },
}

impl<Id> State<Id> {
    fn is_open(&self) -> bool {
        !matches!(self, State::Closed)
    }
}

pub struct BookmarkSwitcher {
    state: State<Window>,
    /// Rasterised hint-letter textures, keyed by `(letter, scale)`. Populated
    /// lazily: [`Self::open`]/[`Self::open_mode`] pre-rasterise every needed
    /// letter at scale 1 as the fail-safe fallback; [`Self::render_output`]
    /// fills in the exact output scale on demand. A `None` value records a
    /// rasterisation failure so it is not retried every frame.
    ///
    /// The `or_insert_with` closures that populate this map (in
    /// [`Self::retain_rasterisable`] and [`Self::render_output`]) call only
    /// `render_hint`, never anything that touches `self.buffers` again —
    /// re-borrowing it from inside the closure while the outer
    /// `borrow_mut()` guard is still held would panic.
    buffers: RefCell<HashMap<BufferKey, Option<MemoryBuffer>>>,
    /// `(letter, scale)` texture-upload failures already logged, so a
    /// persistently-failing GPU upload warns once instead of every frame.
    warned_uploads: RefCell<HashSet<BufferKey>>,
    /// Rasterised mode command-sheet textures, keyed by output scale. The
    /// scale-1 buffer also lives in [`State::Mode`] (the fail-safe fallback,
    /// mirroring `buffers`' role for hints); this cache only ever needs to
    /// fill in the exact output scale on demand.
    sheet_buffers: RefCell<HashMap<NotNan<f64>, Option<MemoryBuffer>>>,
    /// Scales at which the sheet has already logged an upload failure, so a
    /// persistently-failing GPU upload warns once instead of every frame —
    /// mirrors `warned_uploads`' role for hints, keyed by scale alone since
    /// there is only ever one sheet.
    warned_sheet_uploads: RefCell<HashSet<NotNan<f64>>>,
    /// Whether [`Self::render_output`] has already logged a
    /// zero-drawable-hints frame for the current open session. Reset on
    /// [`Self::open`]/[`Self::open_mode`] so the breadcrumb fires at most
    /// once per open, not every frame.
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
            sheet_buffers: RefCell::new(HashMap::new()),
            warned_sheet_uploads: RefCell::new(HashSet::new()),
            warned_empty_frame: Cell::new(false),
        }
    }

    pub fn is_open(&self) -> bool {
        self.state.is_open()
    }

    pub fn close(&mut self) {
        self.state = State::Closed;
    }

    /// Every window id with a visible tile this frame, across every
    /// connected output. `tiles_with_render_positions` yields the Incoming
    /// activity strip only, so an in-flight activity switch does not
    /// double-count the outgoing strip. Shared by [`Self::open`] and
    /// [`Self::open_mode`].
    fn collect_visible_windows(layout: &Layout<Mapped>) -> HashSet<Window> {
        let mut visible = HashSet::new();
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
        visible
    }

    /// Pre-rasterises each hint letter at scale 1, the fallback texture used
    /// whenever the exact-scale render (in [`Self::render_output`]) fails or
    /// hasn't been cached yet, and drops any hint whose letter can't be
    /// rasterised at all — such a hint would stay matchable via
    /// [`Self::press_outcome`] but never drawable, an invisible key-eater
    /// for that one hint. Keeps "matchable" and "drawable" in sync by
    /// construction (see [`Hints`]). Shared by [`Self::open`] and
    /// [`Self::open_mode`].
    fn retain_rasterisable(&self, hints: &mut Vec<Hint<Window>>) {
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

    /// Builds the hint list from the layout and shows the overlay, returning
    /// `true` if it opened.
    ///
    /// Returns `false` (leaving the overlay closed) in two cases, so the
    /// overlay never becomes an invisible key-eater:
    /// - no bookmarked window is currently visible (nothing to tag);
    /// - rasterisation failed for *every* hint letter.
    ///
    /// Re-invoking while already open (in either variant) recomputes the
    /// assignment against the current layout into `Hints` (an idempotent
    /// refresh that also exits mode, if it was active — last action wins).
    pub fn open(&mut self, layout: &Layout<Mapped>) -> bool {
        let visible = Self::collect_visible_windows(layout);

        let mut hints = build_hints(
            layout
                .bookmarks()
                .list()
                .iter()
                .map(|bookmark| (bookmark.id().get(), bookmark.anchor().window().clone())),
            &visible,
            HINT_KEYS,
        );

        if hints.is_empty() {
            if layout.bookmarks().list().is_empty() {
                debug!("bookmark switcher: no bookmarks exist, not opening");
            } else {
                debug!("bookmark switcher: bookmarks exist but none are visible, not opening");
            }
            return false;
        }

        self.retain_rasterisable(&mut hints);

        let Some(hints) = Hints::new(hints) else {
            warn!("bookmark switcher: no hint letter could be rasterised, not opening");
            return false;
        };

        self.state = State::Hints { hints };
        self.warned_empty_frame.set(false);
        true
    }

    /// Builds the hint list (over [`mode_hint_alphabet`], not the full
    /// alphabet) and the command sheet, then opens the overlay in leader
    /// mode, returning `true` if it opened.
    ///
    /// Unlike [`Self::open`], an empty hint list does *not* refuse entry —
    /// the command sheet alone (e.g. to add the first bookmark) is a useful
    /// mode to enter. The only refusal is the command sheet itself failing
    /// to rasterise at scale 1.
    ///
    /// Re-invoking while already open (in either variant) recomputes into
    /// `Mode` — last action wins, mirroring [`Self::open`].
    pub fn open_mode(&mut self, layout: &Layout<Mapped>) -> bool {
        let visible = Self::collect_visible_windows(layout);

        let mut hints = build_hints(
            layout
                .bookmarks()
                .list()
                .iter()
                .map(|bookmark| (bookmark.id().get(), bookmark.anchor().window().clone())),
            &visible,
            &mode_hint_alphabet(),
        );
        self.retain_rasterisable(&mut hints);

        let sheet_markup = mode_sheet_markup();
        let sheet = {
            let mut sheet_buffers = self.sheet_buffers.borrow_mut();
            sheet_buffers
                .entry(NotNan::new(1.).expect("1. is not NaN"))
                .or_insert_with(|| {
                    render_markup(&sheet_markup, 1.)
                        .inspect_err(|err| {
                            warn!("bookmark mode: command sheet failed to rasterise: {err:?}")
                        })
                        .ok()
                })
                .clone()
        };

        let Some(sheet) = sheet else {
            warn!("bookmark mode: command sheet could not be rasterised, not opening");
            return false;
        };

        self.state = State::Mode { hints, sheet };
        self.warned_empty_frame.set(false);
        true
    }

    /// Routes an incoming key press while the overlay is open. Callers must
    /// gate on [`Self::is_open`] first; `Closed` panics rather than
    /// returning a meaningless [`PressOutcome::Dismiss`], because a caller
    /// that reaches here despite the gate has a bug worth surfacing loudly.
    pub fn press_outcome(&self, raw: Option<Keysym>, chorded: bool) -> PressOutcome {
        let (is_mode, hints) = match &self.state {
            State::Closed => {
                unreachable!("press_outcome requires is_open(); caller must gate on it")
            }
            State::Hints { hints } => (false, hints.as_slice()),
            State::Mode { hints, .. } => (true, hints.as_slice()),
        };

        let outcome = press_outcome_core(is_mode, hints, raw, chorded);

        // A matched hint is about to dispatch a jump whose result the caller
        // discards (matching the MRU precedent); log which hint fired so a
        // press that turns out to be a no-op (e.g. the bookmarked window has
        // since become unresolvable) is diagnosable.
        if let PressOutcome::Jump(id) = outcome {
            if let Some(hint) = hints.iter().find(|h| h.bookmark_id == id) {
                debug!(
                    "bookmark switcher: hint '{}' matched, dispatching jump to bookmark {id}",
                    hint.letter
                );
            }
        }

        outcome
    }

    pub fn render_output<R: NiriRenderer>(
        &self,
        layout: &Layout<Mapped>,
        output: &Output,
        renderer: &mut R,
        push: &mut dyn FnMut(BookmarkSwitcherRenderElement),
    ) {
        let (hints, sheet): (&[Hint<Window>], Option<&MemoryBuffer>) = match &self.state {
            State::Closed => return,
            State::Hints { hints } => (hints.as_slice(), None),
            State::Mode { hints, sheet } => (hints, Some(sheet)),
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
                let Some(hint) = hints.iter().find(|hint| &hint.window == window) else {
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

        if let Some(sheet) = sheet {
            if self.render_sheet(sheet, output, renderer, scale, push) {
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
        // per-output calls this method doesn't otherwise need. In `Mode`,
        // the sheet drawing counts too — it draws whenever its upload
        // succeeds, so it should not fire alongside a drawn sheet.
        if !drew_any && !self.warned_empty_frame.replace(true) {
            debug!(
                "bookmark switcher: open but drew no hints on output {} this frame",
                output.name()
            );
        }
    }

    /// Draws the mode command sheet anchored bottom-center in output-local
    /// logical coordinates, horizontally centered with a small bottom
    /// margin. Unlike hints, this is output chrome rather than a
    /// tile-anchored element, so it is *not* rescaled by the overview zoom
    /// (the `RescaleRenderElement` wrap below uses a fixed `1.0` factor,
    /// purely to satisfy the shared render-element type). `fallback` is the
    /// scale-1 buffer carried by [`State::Mode`], used whenever the
    /// exact-scale render below fails or hasn't been cached yet. Returns
    /// whether anything was actually pushed.
    fn render_sheet<R: NiriRenderer>(
        &self,
        fallback: &MemoryBuffer,
        output: &Output,
        renderer: &mut R,
        scale: f64,
        push: &mut dyn FnMut(BookmarkSwitcherRenderElement),
    ) -> bool {
        let buffer = {
            let mut sheet_buffers = self.sheet_buffers.borrow_mut();
            let at_scale = sheet_buffers
                .entry(NotNan::new(scale).expect("scale is not NaN"))
                .or_insert_with(|| {
                    render_markup(&mode_sheet_markup(), scale)
                        .inspect_err(|err| {
                            warn!(
                                "bookmark mode sheet failed to rasterise at scale {scale}: {err:?}"
                            )
                        })
                        .ok()
                })
                .clone();
            at_scale.unwrap_or_else(|| fallback.clone())
        };

        let output_sz = output_size(output);
        let size = buffer.logical_size();
        let x = ((output_sz.w - size.w) / 2.).max(0.);
        let y = (output_sz.h - size.h - f64::from(PADDING)).max(0.);
        let location: Point<f64, Logical> = Point::from((x, y));
        let location = location.to_physical_precise_round(scale).to_logical(scale);

        let Ok(texture) = TextureBuffer::from_memory_buffer(renderer.as_gles_renderer(), &buffer)
        else {
            let key = NotNan::new(scale).expect("scale is not NaN");
            if self.warned_sheet_uploads.borrow_mut().insert(key) {
                warn!("bookmark mode sheet failed to upload as a texture at scale {scale}");
            }
            return false;
        };

        let elem = TextureRenderElement::from_texture_buffer(
            texture,
            location,
            1.,
            None,
            None,
            Kind::Unspecified,
        );
        let elem = PrimaryGpuTextureRenderElement(elem);
        let elem = RescaleRenderElement::from_element(
            elem,
            location.to_physical_precise_round(scale),
            1.0,
        );
        push(BookmarkSwitcherRenderElement::Texture(elem));
        true
    }
}

/// Where an incoming key press routes while the overlay is open, computed
/// purely over the hint list plus (in [`State::Mode`]) the command table —
/// unit-testable without a real [`Window`] or a live compositor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PressOutcome {
    /// A pure modifier or lock key: keep the overlay open.
    HoldOpen,
    /// An un-chorded hint letter matched: jump to this bookmark id.
    Jump(u64),
    /// (`Mode` only) an un-chorded command letter matched.
    Command(ModeCommand),
    /// Anything else — Esc, an unmatched key, `raw == None`, or a chorded
    /// press: dismiss without acting. Outside `Mode` a character that is
    /// also a [`MODE_COMMANDS`] letter is not special-cased here; it
    /// dismisses only if it isn't assigned as a hint (see [`Jump`](Self::Jump)).
    Dismiss,
}

/// Pure routing core behind [`BookmarkSwitcher::press_outcome`], generic
/// over the hint id type so it is unit-testable with the `u64` stand-in
/// (see the `tests` module). `is_mode` selects whether command letters are
/// checked ([`State::Mode`]) or skipped entirely, falling through straight
/// to hint matching ([`State::Hints`]). Outside `Mode` there is no command
/// special-case at all: a character that also appears in [`MODE_COMMANDS`]
/// is just an ordinary member of the hint alphabet there, so an assigned
/// one still produces [`PressOutcome::Jump`].
fn press_outcome_core<Id>(
    is_mode: bool,
    hints: &[Hint<Id>],
    raw: Option<Keysym>,
    chorded: bool,
) -> PressOutcome {
    if raw.is_some_and(is_modifier_keysym) {
        return PressOutcome::HoldOpen;
    }
    if chorded {
        return PressOutcome::Dismiss;
    }
    let Some(raw) = raw else {
        return PressOutcome::Dismiss;
    };

    // Commands can't collide with hints by construction (`mode_hint_alphabet`
    // excludes every `MODE_COMMANDS` character), but checking commands first
    // costs nothing and keeps that guarantee from being load-bearing here.
    if is_mode {
        if let Some(cmd) = command_for_keysym(raw) {
            return PressOutcome::Command(cmd);
        }
    }

    match match_keysym(hints, raw) {
        Some(id) => PressOutcome::Jump(id),
        None => PressOutcome::Dismiss,
    }
}

/// Resolves a raw keysym to its [`ModeCommand`], derived from
/// [`MODE_COMMANDS`] rather than a separately maintained keysym table so the
/// two can never drift.
fn command_for_keysym(raw: Keysym) -> Option<ModeCommand> {
    MODE_COMMANDS
        .iter()
        .find_map(|&(ch, cmd, _)| (Keysym::from_char(ch) == raw).then_some(cmd))
}

/// The in-mode hint alphabet: [`HINT_KEYS`] with every [`MODE_COMMANDS`]
/// character removed, so a hint letter and a command letter can never
/// collide. The standalone switcher ([`BookmarkSwitcher::open`]) keeps the
/// full alphabet.
fn mode_hint_alphabet() -> String {
    HINT_KEYS
        .chars()
        .filter(|c| !MODE_COMMANDS.iter().any(|&(ch, _, _)| ch == *c))
        .collect()
}

/// One-line command-sheet markup, e.g. `a add · d/x remove · ,/. walk ·
/// letter jump · esc close`. Generated from [`MODE_COMMANDS`], grouping
/// consecutive entries that share a label (`d`/`x` both "remove") so the
/// sheet text can never drift from the routing table.
fn mode_sheet_markup() -> String {
    let mut groups: Vec<(String, &str)> = Vec::new();
    for &(ch, _, label) in MODE_COMMANDS {
        match groups.last_mut() {
            Some((chars, last_label)) if *last_label == label => {
                chars.push('/');
                chars.push(ch);
            }
            _ => groups.push((ch.to_string(), label)),
        }
    }

    let mut text = groups
        .into_iter()
        .map(|(chars, label)| format!("{chars} {label}"))
        .collect::<Vec<_>>()
        .join(" · ");
    // Always advertises "letter jump" even when the current open has zero
    // hints assigned — the sheet is static/cached per scale rather than
    // re-derived from the live hint count, so it can't reflect that. Accepted
    // tradeoff: keeping it static is what makes the upload cache safe to
    // reuse across frames.
    text.push_str(" · letter jump · esc close");

    format!("<span face='mono'>{text}</span>")
}

/// Assigns hint letters to visible bookmarks in list order.
///
/// `bookmarks` yields `(bookmark id, anchor window id)` in presentation
/// order; only those whose window is in `visible` are kept, and they are
/// zipped with `alphabet` (chars, in order) so the letters are compact (no
/// gaps for hidden bookmarks). Bookmarks past the end of `alphabet` get no
/// hint. Callers pass [`HINT_KEYS`] (standalone switcher) or
/// [`mode_hint_alphabet`] (leader mode, command letters excluded).
fn build_hints<Id: Eq + Hash + Clone>(
    bookmarks: impl Iterator<Item = (u64, Id)>,
    visible: &HashSet<Id>,
    alphabet: &str,
) -> Vec<Hint<Id>> {
    bookmarks
        .filter(|(_id, window)| visible.contains(window))
        .zip(alphabet.chars())
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

/// Rasterises pango markup into a padded, bordered box (dark fill, bright
/// border, white text). Shared by hint letters ([`render_hint`]) and the
/// leader-mode command sheet ([`mode_sheet_markup`]'s output); the returned
/// [`MemoryBuffer`] is uploaded to a texture at render time.
fn render_markup(markup: &str, scale: f64) -> anyhow::Result<MemoryBuffer> {
    let _span = tracy_client::span!("bookmark_switcher::render_markup");

    let padding: i32 = to_physical_precise_round(scale, PADDING);

    let mut font = FontDescription::from_string(FONT);
    font.set_absolute_size(to_physical_precise_round(scale, font.size()));

    let surface = ImageSurface::create(cairo::Format::ARgb32, 0, 0)?;
    let cr = cairo::Context::new(&surface)?;
    let layout = pangocairo::functions::create_layout(&cr);
    layout.context().set_round_glyph_positions(false);
    layout.set_font_description(Some(&font));
    layout.set_markup(markup);

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
    layout.set_markup(markup);

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

/// Rasterises a single hint letter. A thin wrapper over [`render_markup`].
fn render_hint(letter: char, scale: f64) -> anyhow::Result<MemoryBuffer> {
    let markup = format!(
        "<span face='mono'><b>{}</b></span>",
        letter.to_ascii_uppercase()
    );
    render_markup(&markup, scale)
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

        let hints = build_hints(list.into_iter(), &visible(&[1, 3]), HINT_KEYS);

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

        let hints = build_hints(list.into_iter(), &visible(&ids), HINT_KEYS);

        // 26 letters in the alphabet; the 4 extra visible bookmarks drop off.
        assert_eq!(hints.len(), HINT_KEYS.len());
        assert_eq!(hints.len(), 26);
    }

    #[test]
    fn match_keysym_maps_letters_to_ids_and_misses_to_none() {
        let list = fixture(&[10, 20]);
        let hints = build_hints(list.into_iter(), &visible(&[10, 20]), HINT_KEYS);

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

        let hints = build_hints(list.into_iter(), &visible(&[]), HINT_KEYS);

        assert!(hints.is_empty());
    }

    #[test]
    fn rebuild_recomputes_assignment_against_new_visibility() {
        // Reopening (idempotent refresh) recomputes letters from the current
        // visible set: the same bookmark can move to a different letter when a
        // preceding bookmark becomes visible.
        let first = build_hints(fixture(&[1, 2, 3]).into_iter(), &visible(&[3]), HINT_KEYS);
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].letter, 'a');
        assert_eq!(first[0].bookmark_id, 3);

        let second = build_hints(
            fixture(&[1, 2, 3]).into_iter(),
            &visible(&[2, 3]),
            HINT_KEYS,
        );
        assert_eq!(second.len(), 2);
        // Now bookmark 2 takes 'a' and 3 is pushed to 's'.
        assert_eq!(second[1].letter, 's');
        assert_eq!(second[1].bookmark_id, 3);
    }

    #[test]
    fn hints_new_rejects_empty_and_accepts_nonempty() {
        // The guard `open` relies on to make an empty-hints `State::Hints`
        // unrepresentable, exercised directly with the `u64` id stand-in so
        // it doesn't need a real `Window`.
        assert!(Hints::<u64>::new(Vec::new()).is_none());

        let hints = build_hints(fixture(&[1]).into_iter(), &visible(&[1]), HINT_KEYS);
        assert!(Hints::new(hints).is_some());
    }

    #[test]
    fn close_transitions_to_closed_state() {
        // Goes through the real `Hints::new` guard (a `State::Hints` can only
        // be built from a non-empty, already-validated hint list) rather than
        // hand-constructing an invalid open state; the `u64` id stand-in
        // keeps this independent of a real `Window`.
        let hints = build_hints(fixture(&[10]).into_iter(), &visible(&[10]), HINT_KEYS);
        let mut state = State::Hints {
            hints: Hints::new(hints).expect("nonempty hints must construct"),
        };
        assert!(state.is_open());

        state = State::Closed;

        assert!(!state.is_open());
    }

    #[test]
    #[should_panic(expected = "requires is_open")]
    fn press_outcome_panics_when_closed() {
        // `Closed` is unreachable in `press_outcome`; callers must gate on
        // `is_open()` first. Exercised through the real `BookmarkSwitcher`
        // entry point, not a reimplemented replica.
        let switcher = BookmarkSwitcher::new();
        assert!(!switcher.is_open());

        switcher.press_outcome(None, false);
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

    #[test]
    fn mode_hint_alphabet_excludes_every_command_char() {
        let alphabet = mode_hint_alphabet();
        for &(ch, _, _) in MODE_COMMANDS {
            assert!(
                !alphabet.contains(ch),
                "mode hint alphabet must not contain command char '{ch}'"
            );
        }
        // Every other HINT_KEYS letter survives: 'a', 'd', 'x' are removed
        // ('.' and ',' were never in HINT_KEYS to begin with).
        assert_eq!(alphabet.len(), HINT_KEYS.len() - 3);
    }

    #[test]
    fn standalone_alphabet_is_unaffected_by_mode_commands() {
        // The non-mode switcher keeps the full alphabet: HINT_KEYS itself,
        // passed directly to `build_hints`, is untouched by MODE_COMMANDS.
        assert_eq!(HINT_KEYS.len(), 26);
    }

    #[test]
    fn press_outcome_core_routes_modifier_to_hold_open() {
        let hints: Vec<Hint<u64>> = Vec::new();
        let outcome = press_outcome_core(true, &hints, Some(Keysym::Shift_L), false);
        assert_eq!(outcome, PressOutcome::HoldOpen);
    }

    #[test]
    fn press_outcome_core_routes_command_char_in_mode() {
        let hints: Vec<Hint<u64>> = Vec::new();
        let outcome = press_outcome_core(true, &hints, Some(Keysym::from_char('a')), false);
        assert_eq!(outcome, PressOutcome::Command(ModeCommand::Add));

        let outcome = press_outcome_core(true, &hints, Some(Keysym::from_char('d')), false);
        assert_eq!(outcome, PressOutcome::Command(ModeCommand::RemoveFocused));

        let outcome = press_outcome_core(true, &hints, Some(Keysym::from_char('x')), false);
        assert_eq!(outcome, PressOutcome::Command(ModeCommand::RemoveFocused));

        let outcome = press_outcome_core(true, &hints, Some(Keysym::from_char(',')), false);
        assert_eq!(outcome, PressOutcome::Command(ModeCommand::WalkBackward));

        let outcome = press_outcome_core(true, &hints, Some(Keysym::from_char('.')), false);
        assert_eq!(outcome, PressOutcome::Command(ModeCommand::WalkForward));
    }

    #[test]
    fn press_outcome_core_command_char_dismisses_outside_mode() {
        // Outside Mode (is_mode = false), a command letter is just an
        // unmatched hint letter: dismiss.
        let hints: Vec<Hint<u64>> = Vec::new();
        let outcome = press_outcome_core(false, &hints, Some(Keysym::from_char('a')), false);
        assert_eq!(outcome, PressOutcome::Dismiss);
    }

    #[test]
    fn press_outcome_core_chorded_command_char_in_mode_dismisses() {
        // A chorded press short-circuits before command matching even when
        // the raw keysym is a command letter and is_mode is true.
        let hints: Vec<Hint<u64>> = Vec::new();
        let outcome = press_outcome_core(true, &hints, Some(Keysym::from_char('a')), true);
        assert_eq!(outcome, PressOutcome::Dismiss);
    }

    #[test]
    fn press_outcome_core_modifier_and_none_dismiss_outside_mode_too() {
        // HoldOpen and the raw == None dismiss are checked ahead of the
        // is_mode branch, so they behave identically whether or not we're
        // in Mode.
        let hints: Vec<Hint<u64>> = Vec::new();
        let outcome = press_outcome_core(false, &hints, Some(Keysym::Shift_L), false);
        assert_eq!(outcome, PressOutcome::HoldOpen);

        let outcome = press_outcome_core(false, &hints, None, false);
        assert_eq!(outcome, PressOutcome::Dismiss);
    }

    #[test]
    fn press_outcome_core_routes_hint_letter_to_jump() {
        let hints = build_hints(fixture(&[10]).into_iter(), &visible(&[10]), HINT_KEYS);
        let outcome = press_outcome_core(false, &hints, Some(Keysym::from_char('a')), false);
        assert_eq!(outcome, PressOutcome::Jump(10));

        // In mode, a hint letter drawn from the mode alphabet (which never
        // collides with a command char) still jumps.
        let mode_hints = build_hints(
            fixture(&[10]).into_iter(),
            &visible(&[10]),
            &mode_hint_alphabet(),
        );
        let outcome = press_outcome_core(true, &mode_hints, Some(mode_hints[0].keysym), false);
        assert_eq!(outcome, PressOutcome::Jump(10));
    }

    #[test]
    fn press_outcome_core_chorded_letter_dismisses() {
        let hints = build_hints(fixture(&[10]).into_iter(), &visible(&[10]), HINT_KEYS);
        let outcome = press_outcome_core(false, &hints, Some(Keysym::from_char('a')), true);
        assert_eq!(outcome, PressOutcome::Dismiss);
    }

    #[test]
    fn press_outcome_core_escape_unmatched_and_none_dismiss() {
        let hints: Vec<Hint<u64>> = Vec::new();
        assert_eq!(
            press_outcome_core(true, &hints, Some(Keysym::Escape), false),
            PressOutcome::Dismiss
        );
        assert_eq!(
            press_outcome_core(true, &hints, Some(Keysym::from_char('z')), false),
            PressOutcome::Dismiss
        );
        assert_eq!(
            press_outcome_core(true, &hints, None, false),
            PressOutcome::Dismiss
        );
    }

    #[test]
    fn press_outcome_core_mode_constructible_with_zero_hints_still_routes_commands() {
        let hints: Vec<Hint<u64>> = Vec::new();
        let outcome = press_outcome_core(true, &hints, Some(Keysym::from_char('a')), false);
        assert_eq!(outcome, PressOutcome::Command(ModeCommand::Add));
    }

    #[test]
    fn mode_sheet_markup_contains_every_command_key_and_label() {
        let markup = mode_sheet_markup();
        for &(ch, _, label) in MODE_COMMANDS {
            assert!(
                markup.contains(ch),
                "sheet markup must mention command char '{ch}'"
            );
            assert!(
                markup.contains(label),
                "sheet markup must mention label '{label}'"
            );
        }
    }

    #[test]
    fn mode_sheet_markup_aliases_d_and_x_together() {
        // `d` and `x` share the "remove" label and must be grouped rather
        // than appearing as two separate "remove" entries.
        let markup = mode_sheet_markup();
        assert!(
            markup.contains("d/x remove"),
            "expected 'd/x remove' grouping in: {markup}"
        );
    }
}
