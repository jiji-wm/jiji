//! Letter-hint overlay for jumping to a bookmarked window, with an optional
//! leader/prefix mode layered on top.
//!
//! In its base ([`BookmarkSwitcher::open`]) form, every currently visible
//! bookmarked window is tagged with a single-letter hint drawn over its
//! tile's top-left corner; pressing a hint jumps straight to that bookmark.
//! [`BookmarkSwitcher::open_mode`] opens the same hints plus a one-line
//! command sheet anchored at the bottom of each output, and reads a handful
//! of add / remove-focused / walk-backward-forward command letters (resolved
//! from config into a [`ModeKeymap`] at open, the single source of truth for
//! that instance's table) ahead of hint matching — a leader chord that lets
//! bookmark management happen without leaving the keyboard. The command keys
//! and the hint alphabet are configurable (`bookmarks.mode-keys` /
//! `bookmarks.hint-alphabet`); the defaults are the letters `a` add, `d`/`x`
//! remove, `,`/`.` walk, and `/` search over the home-row-first alphabet.
//! Mode entry does not require any bookmarks
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
//! overlay still dismisses. The same open-time-snapshot staleness window
//! applies to [`State::Search`]'s Enter-jumps-to-top-match path: the search
//! snapshot is taken once on entry, so a window closing mid-search can leave
//! `Enter` targeting a since-pruned bookmark, with the identical
//! `BookmarkNotFound`-discarded outcome.
//!
//! There is no backdrop and no animation: the overlay shows and dismisses
//! instantly. It is purely a visual plus key-capture layer, so any pointer
//! press dismisses it (handled by the caller) and it never gates pointer
//! hit-testing.
//!
//! By default, a leader-mode command letter (add, remove, walk) dismisses the
//! overlay after dispatching, the same as any other terminal outcome. Setting
//! `bookmarks.mode-sticky` changes only that one continuation: after a
//! successful command dispatches, the overlay is dismissed and immediately
//! reopened (a fresh instance — new hints, new search snapshot — not a
//! kept-open one), so repeated adds/removes/walks stay in mode without a
//! re-press of the leader key. Nothing else is affected by the toggle: a hint
//! letter jump always exits mode even when sticky, the confirm-gated removal
//! path opens the dialog without reopening mode (before or after
//! confirm/cancel), search is unaffected (`Enter`-jump and `Esc` behave as
//! below regardless of sticky), and Esc / an unmatched key / a chorded press /
//! a pointer press still dismiss outright.
//!
//! Leader mode layers one more state on top: pressing the configured search
//! key (default `/`) switches from
//! [`State::Mode`] into [`State::Search`], an incremental, case-insensitive
//! substring filter over each visible bookmark's display name and clean
//! window title (the two fields matched independently, never as one
//! concatenated haystack). Typing narrows the set live, `Enter` jumps to the
//! first still-matching bookmark in list order, and `Esc` closes the overlay
//! outright (single-shot: there is no path back to [`State::Mode`];
//! `Backspace` covers correction). While searching, the hint letters carried
//! over from mode are drawn only on the currently matching tiles as *match
//! indicators* — they are not pressable, selection is `Enter`-only.
//!
//! Two deliberate limitations of this first search cut, recorded here rather
//! than in the code that embodies them:
//! - Search is reachable only via the configured search key (default `/`) from leader mode; there
//!   is no separate selector to open straight into it.
//! - Query characters come from the *base* (unshifted) keysym via [`Keysym::key_char`], so there
//!   are no capitals, shifted punctuation, or IME input in the query. This is harmless under
//!   case-insensitive matching (the lowercased query still matches), just visually lower-case.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::mem;

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

use crate::ipc::server::role_title_to_tag_and_clean;
use crate::layout::{Layout, LayoutElement};
use crate::niri_render_elements;
use crate::render_helpers::memory::MemoryBuffer;
use crate::render_helpers::primary_gpu_texture::PrimaryGpuTextureRenderElement;
use crate::render_helpers::renderer::NiriRenderer;
use crate::render_helpers::texture::{TextureBuffer, TextureRenderElement};
use crate::utils::{output_size, to_physical_precise_round, with_toplevel_role};
use crate::window::Mapped;

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

impl ModeCommand {
    /// The label shown for this command on the command sheet. Code-owned (the
    /// config remaps keys, never labels); both walk directions deliberately
    /// share `"walk"` so a walk-backward/forward pair groups into one sheet
    /// entry.
    fn label(self) -> &'static str {
        match self {
            ModeCommand::Add => "add",
            ModeCommand::RemoveFocused => "remove",
            ModeCommand::WalkBackward | ModeCommand::WalkForward => "walk",
        }
    }
}

/// The resolved leader-mode key table for one open instance, built from config
/// at [`BookmarkSwitcher::open_mode`] and carried by value in [`State::Mode`]
/// so the instance keeps its table for its whole lifetime (a config reload
/// takes effect on the next open, never mid-open).
///
/// Single source of truth for that instance's routing: keysym → command
/// ([`Self::command_for_keysym`]), the search-entry keysym, and the sheet text
/// ([`mode_sheet_markup`]) all read it, so they cannot drift apart. The
/// `mode_alphabet` is derived by stripping every command and search character
/// from the configured `full_alphabet`, so a hint letter and a command key can
/// never collide by construction; the standalone switcher keeps
/// `full_alphabet` unstripped.
struct ModeKeymap {
    /// `(char, keysym, command)`, in sheet order: add keys, remove keys,
    /// walk-backward keys, walk-forward keys. `keysym` is precomputed so
    /// routing an incoming press is a pure comparison.
    commands: Vec<(char, Keysym, ModeCommand)>,
    /// The `(char, keysym)` that switches into [`State::Search`].
    search: (char, Keysym),
    /// The configured hint alphabet, used unstripped by the standalone switcher.
    full_alphabet: String,
    /// `full_alphabet` with every command and search character removed — the
    /// leader-mode hint alphabet, disjoint from the command keys by
    /// construction.
    mode_alphabet: String,
    /// Not a key — `bookmarks.mode-sticky`, carried here because
    /// [`ModeKeymap`] is the per-open config-resolution artifact
    /// [`State::Mode`] already carries by value, so a successful command's
    /// dismiss-then-reopen continuation reads a table consistent with the
    /// rest of this instance's routing.
    sticky: bool,
}

impl ModeKeymap {
    fn from_config(cfg: &jiji_config::BookmarksConfig) -> Self {
        let mk = &cfg.mode_keys;
        let mut commands = Vec::new();
        let mut push = |chars: &[char], cmd: ModeCommand| {
            for &ch in chars {
                commands.push((ch, Keysym::from_char(ch), cmd));
            }
        };
        push(mk.add(), ModeCommand::Add);
        push(mk.remove(), ModeCommand::RemoveFocused);
        push(mk.walk_backward(), ModeCommand::WalkBackward);
        push(mk.walk_forward(), ModeCommand::WalkForward);

        let search = (mk.search(), Keysym::from_char(mk.search()));

        let full_alphabet = cfg.hint_alphabet.as_str().to_string();
        let reserved: HashSet<char> = commands
            .iter()
            .map(|&(ch, _, _)| ch)
            .chain(std::iter::once(mk.search()))
            .collect();
        let mode_alphabet = full_alphabet
            .chars()
            .filter(|c| !reserved.contains(c))
            .collect();

        Self {
            commands,
            search,
            full_alphabet,
            mode_alphabet,
            sticky: cfg.mode_sticky,
        }
    }

    /// Resolves a raw keysym to its [`ModeCommand`], if any command key matches.
    fn command_for_keysym(&self, raw: Keysym) -> Option<ModeCommand> {
        self.commands
            .iter()
            .find_map(|&(_, keysym, cmd)| (keysym == raw).then_some(cmd))
    }
}

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

/// One searchable bookmark, snapshotted when [`State::Search`] is entered.
///
/// The set is *all* visible bookmarked windows — a superset of the hinted
/// set, so a bookmark past the end of the hint alphabet is still searchable
/// (just unhinted). Generic over the window id type only so the pure
/// matching logic can be unit-tested with a cheap stand-in; production always
/// uses [`Window`].
///
/// `name_lower` and `title_lower` are the lowercased forms of the two match
/// fields, precomputed once at construction so each keystroke's filter is a
/// plain `contains`. They are private and only ever set by [`Self::new`] from
/// the same sources as `label`, so they cannot drift out of sync with it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SearchEntry<Id> {
    bookmark_id: u64,
    window: Id,
    /// Display text: the bookmark name if set, else the clean window title,
    /// else `"(untitled)"`.
    label: String,
    /// Lowercased bookmark name, if the bookmark has one.
    name_lower: Option<String>,
    /// Lowercased clean window title, `""` when the window is untitled.
    title_lower: String,
}

impl<Id> SearchEntry<Id> {
    /// Snapshots one bookmark's searchable form, precomputing the lowercased
    /// match fields and the display label from the same sources so the three
    /// stay consistent by construction. `name` is the bookmark's display name
    /// (already non-empty by [`BookmarkName`](crate::layout::bookmarks) 's
    /// validation); `title` is the *clean* (machine-tag-stripped) window
    /// title.
    fn new(bookmark_id: u64, window: Id, name: Option<&str>, title: Option<&str>) -> Self {
        let label = match name {
            Some(name) => name.to_owned(),
            None => match title {
                Some(title) if !title.is_empty() => title.to_owned(),
                _ => "(untitled)".to_owned(),
            },
        };
        SearchEntry {
            bookmark_id,
            window,
            label,
            name_lower: name.map(str::to_lowercase),
            title_lower: title.unwrap_or("").to_lowercase(),
        }
    }
}

/// Whether `entry` matches `query_lower` (already lowercased by the caller).
///
/// An empty query matches everything. A non-empty query matches if it is a
/// substring of the name **or** the title, each tested independently — the
/// two fields are never concatenated into one haystack, so a query cannot
/// span the name/title boundary.
fn entry_matches<Id>(entry: &SearchEntry<Id>, query_lower: &str) -> bool {
    if query_lower.is_empty() {
        return true;
    }
    entry
        .name_lower
        .as_deref()
        .is_some_and(|name| name.contains(query_lower))
        || entry.title_lower.contains(query_lower)
}

/// The first entry (in bookmark-list order) that matches `query_lower`, i.e.
/// what `Enter` jumps to. `None` when nothing matches.
fn top_match<'a, Id>(
    entries: &'a [SearchEntry<Id>],
    query_lower: &str,
) -> Option<&'a SearchEntry<Id>> {
    entries
        .iter()
        .find(|entry| entry_matches(entry, query_lower))
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
        /// All visible bookmarked windows, snapshotted at open, carried ready
        /// so the search key can enter [`State::Search`] without re-walking the
        /// layout.
        entries: Vec<SearchEntry<Id>>,
        sheet: MemoryBuffer,
        /// The resolved key table for this open instance. Carried by value so
        /// routing and the sheet text read one consistent table for the whole
        /// lifetime of the open, independent of any config reload.
        keymap: ModeKeymap,
    },
    /// Incremental search over the mode snapshot. Like `Mode`/`Hints`, this
    /// variant carries its scale-1 rasterised artifact — the query `line` —
    /// by value, so "search open but nothing drawable" is unrepresentable
    /// (the same validated-construction discipline `Hints` and `Mode.sheet`
    /// apply).
    Search {
        /// Carried from `Mode` unchanged. Indicator-only in this state: hint
        /// letters are drawn on matching tiles but are *not* pressable —
        /// selection is `Enter`-only.
        hints: Vec<Hint<Id>>,
        entries: Vec<SearchEntry<Id>>,
        query: String,
        /// Scale-1 rasterised query line, re-rendered on every edit; the
        /// fail-safe fallback for [`BookmarkSwitcher::render_output`]'s
        /// per-frame at-scale render.
        line: MemoryBuffer,
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

    /// Like [`Self::collect_visible_windows`], but also captures each visible
    /// window's *clean* (machine-tag-stripped) title, so leader-mode search
    /// matches the human-facing title rather than a Firefox-restore UUID tag.
    /// Titles pass through [`role_title_to_tag_and_clean`] — the same
    /// tag-stripping the IPC layer applies — never the raw role title. Used
    /// only by [`Self::open_mode`]; the standalone [`Self::open`] has no
    /// search and keeps the lighter set-only collector.
    fn collect_visible_titles(layout: &Layout<Mapped>) -> HashMap<Window, Option<String>> {
        let mut visible = HashMap::new();
        for mon in layout.monitors() {
            let ctx = layout.ctx_for(mon);
            for (ws, _geo) in mon.workspaces_with_render_geo(ctx) {
                for (tile, _pos, tile_visible) in ws.tiles_with_render_positions() {
                    if tile_visible {
                        let window = LayoutElement::id(tile.window()).clone();
                        let clean = with_toplevel_role(tile.window().toplevel(), |role| {
                            role_title_to_tag_and_clean(&role.title).clean_title
                        });
                        visible.insert(window, clean);
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
    pub fn open(&mut self, layout: &Layout<Mapped>, config: &jiji_config::BookmarksConfig) -> bool {
        let keymap = ModeKeymap::from_config(config);
        let visible = Self::collect_visible_windows(layout);

        let mut hints = build_hints(
            layout
                .bookmarks()
                .list()
                .iter()
                .map(|bookmark| (bookmark.id().get(), bookmark.anchor().window().clone())),
            &visible,
            &keymap.full_alphabet,
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

    /// Builds the hint list (over the resolved [`ModeKeymap::mode_alphabet`],
    /// not the full alphabet) and the command sheet, then opens the overlay in
    /// leader mode, returning `true` if it opened.
    ///
    /// Unlike [`Self::open`], an empty hint list does *not* refuse entry —
    /// the command sheet alone (e.g. to add the first bookmark) is a useful
    /// mode to enter. The only refusal is the command sheet itself failing
    /// to rasterise at scale 1.
    ///
    /// The key table is resolved from `config` here and carried by value in
    /// [`State::Mode`]: a config reload takes effect on the *next* open, and a
    /// live-open instance finishes with the table it opened with (no
    /// dismiss-on-reload). Because the sheet text depends on that table (not
    /// just on the output scale), the per-scale sheet caches are cleared here
    /// so a reopen under a changed table cannot serve stale sheet pixels.
    ///
    /// Re-invoking while already open (in either variant) recomputes into
    /// `Mode` — last action wins, mirroring [`Self::open`].
    pub fn open_mode(
        &mut self,
        layout: &Layout<Mapped>,
        config: &jiji_config::BookmarksConfig,
    ) -> bool {
        let keymap = ModeKeymap::from_config(config);
        if keymap.mode_alphabet.is_empty() && !keymap.full_alphabet.is_empty() {
            warn!(
                "bookmark leader mode: hint-alphabet is fully consumed by command/search keys; \
                 no jump hints will be shown"
            );
        }

        // The sheet text is a function of the resolved key table, not just the
        // output scale the cache keys on. Clear both sheet caches so a reopen
        // under a changed table re-rasterises rather than serving a stale sheet.
        self.sheet_buffers.borrow_mut().clear();
        self.warned_sheet_uploads.borrow_mut().clear();

        let titles = Self::collect_visible_titles(layout);
        let visible: HashSet<Window> = titles.keys().cloned().collect();

        let mut hints = build_hints(
            layout
                .bookmarks()
                .list()
                .iter()
                .map(|bookmark| (bookmark.id().get(), bookmark.anchor().window().clone())),
            &visible,
            &keymap.mode_alphabet,
        );
        self.retain_rasterisable(&mut hints);

        // Search over *all* visible bookmarked windows — a superset of the
        // hinted set, so bookmarks past the end of the hint alphabet are
        // searchable too (just unhinted). Snapshotted here so entering search
        // never re-walks the layout.
        let entries: Vec<SearchEntry<Window>> = layout
            .bookmarks()
            .list()
            .iter()
            .filter_map(|bookmark| {
                let window = bookmark.anchor().window();
                let title = titles.get(window)?;
                Some(SearchEntry::new(
                    bookmark.id().get(),
                    window.clone(),
                    bookmark.name().map(|name| name.as_str()),
                    title.as_deref(),
                ))
            })
            .collect();

        let sheet_markup = mode_sheet_markup(&keymap);
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

        self.state = State::Mode {
            hints,
            entries,
            sheet,
            keymap,
        };
        self.warned_empty_frame.set(false);
        true
    }

    /// Routes an incoming key press while the overlay is open. Callers must
    /// gate on [`Self::is_open`] first; `Closed` panics rather than
    /// returning a meaningless [`PressOutcome::Dismiss`], because a caller
    /// that reaches here despite the gate has a bug worth surfacing loudly.
    ///
    /// Takes `&mut self` because the [`State::Search`] edit outcomes
    /// (`/` enters search, a printable character extends the query, `Backspace`
    /// trims it) mutate the overlay in place. The pure routing decision is
    /// made by [`press_outcome_core`]; this wrapper applies the state-changing
    /// ones and reports [`PressOutcome::SearchUpdated`] so the caller redraws.
    pub fn press_outcome(&mut self, raw: Option<Keysym>, chorded: bool) -> PressOutcome {
        let core = match &self.state {
            State::Closed => {
                unreachable!("press_outcome requires is_open(); caller must gate on it")
            }
            State::Hints { hints } => press_outcome_core(
                RoutingContext::Hints {
                    hints: hints.as_slice(),
                },
                raw,
                chorded,
            ),
            State::Mode { hints, keymap, .. } => press_outcome_core(
                RoutingContext::Mode {
                    hints: hints.as_slice(),
                    keymap,
                },
                raw,
                chorded,
            ),
            State::Search { entries, query, .. } => {
                press_outcome_core(RoutingContext::Search { entries, query }, raw, chorded)
            }
        };

        // A matched hint/entry is about to dispatch a jump whose result the
        // caller discards (matching the MRU precedent); log so a press that
        // turns out to be a no-op (e.g. the bookmarked window has since become
        // unresolvable) is diagnosable.
        if let CoreOutcome::Jump(id) = core {
            debug!("bookmark switcher: matched, dispatching jump to bookmark {id}");
        }

        match core {
            CoreOutcome::HoldOpen => PressOutcome::HoldOpen,
            CoreOutcome::Jump(id) => PressOutcome::Jump(id),
            CoreOutcome::Command(cmd) => {
                let State::Mode { keymap, .. } = &self.state else {
                    unreachable!("CoreOutcome::Command is only routed from State::Mode")
                };
                PressOutcome::Command {
                    cmd,
                    sticky: keymap.sticky,
                }
            }
            CoreOutcome::Dismiss => PressOutcome::Dismiss,
            CoreOutcome::EnterSearch => self.enter_search(),
            CoreOutcome::Push(ch) => self.push_query_char(ch),
            CoreOutcome::Pop => self.pop_query_char(),
        }
    }

    /// Transitions [`State::Mode`] → [`State::Search`] on `/`.
    ///
    /// Rasterises the initial (empty-query) line at scale 1 first, before
    /// touching `self.state`: on failure it warns and returns, leaving
    /// `Mode` untouched, so a rasterise failure can never strand the overlay
    /// mid-transition or enter an undrawable search state. The command sheet
    /// is dropped on success — from search, `Esc` closes the overlay
    /// outright rather than returning to mode.
    fn enter_search(&mut self) -> PressOutcome {
        let State::Mode { entries, .. } = &self.state else {
            unreachable!("enter_search is only routed from State::Mode");
        };

        let query = String::new();
        let line = match render_query_line(&query, entries, 1.) {
            Ok(line) => line,
            Err(err) => {
                warn!(
                    "bookmark search: initial query line failed to rasterise: {err:?}, \
                     staying in leader mode"
                );
                return PressOutcome::HoldOpen;
            }
        };

        let State::Mode { hints, entries, .. } = mem::replace(&mut self.state, State::Closed)
        else {
            unreachable!("enter_search is only routed from State::Mode");
        };
        self.state = State::Search {
            hints,
            entries,
            query,
            line,
        };
        PressOutcome::SearchUpdated
    }

    /// Appends `ch` to the search query and re-rasterises the scale-1 line.
    ///
    /// On rasterise failure it logs (debug, not warn — the failure repeats on
    /// every keystroke of a held-down key, and the per-frame at-scale render
    /// in [`Self::render_output`] already surfaces a persistent failure at
    /// debug) and keeps the previous line buffer (stale but visible) — the
    /// query still advances, and that per-frame render may well succeed at
    /// the output scale even when scale 1 failed.
    fn push_query_char(&mut self, ch: char) -> PressOutcome {
        let State::Search {
            entries,
            query,
            line,
            ..
        } = &mut self.state
        else {
            unreachable!("push_query_char is only routed from State::Search");
        };

        query.push(ch);
        match render_query_line(query, entries, 1.) {
            Ok(new_line) => *line = new_line,
            Err(err) => debug!(
                "bookmark search: query line failed to rasterise: {err:?}, keeping previous line"
            ),
        }
        PressOutcome::SearchUpdated
    }

    /// Trims the last character from the search query. A `Backspace` on an
    /// already-empty query is a [`PressOutcome::HoldOpen`] no-op.
    fn pop_query_char(&mut self) -> PressOutcome {
        let State::Search {
            entries,
            query,
            line,
            ..
        } = &mut self.state
        else {
            unreachable!("pop_query_char is only routed from State::Search");
        };

        if query.pop().is_none() {
            return PressOutcome::HoldOpen;
        }
        match render_query_line(query, entries, 1.) {
            Ok(new_line) => *line = new_line,
            Err(err) => debug!(
                "bookmark search: query line failed to rasterise: {err:?}, keeping previous line"
            ),
        }
        PressOutcome::SearchUpdated
    }

    pub fn render_output<R: NiriRenderer>(
        &self,
        layout: &Layout<Mapped>,
        output: &Output,
        renderer: &mut R,
        push: &mut dyn FnMut(BookmarkSwitcherRenderElement),
    ) {
        // `hints` are the letters to draw; `matched`, present only while
        // searching, restricts drawing to the currently matching tiles. The
        // matched window set is computed once here (one `to_lowercase`, one
        // pass over the snapshot) rather than per tile.
        let hints: &[Hint<Window>];
        let matched: Option<HashSet<Window>>;
        match &self.state {
            State::Closed => return,
            State::Hints { hints: open_hints } => {
                hints = open_hints.as_slice();
                matched = None;
            }
            State::Mode {
                hints: open_hints, ..
            } => {
                hints = open_hints.as_slice();
                matched = None;
            }
            State::Search {
                hints: open_hints,
                entries,
                query,
                ..
            } => {
                let query_lower = query.to_lowercase();
                matched = Some(
                    entries
                        .iter()
                        .filter(|entry| entry_matches(entry, &query_lower))
                        .map(|entry| entry.window.clone())
                        .collect(),
                );
                hints = open_hints.as_slice();
            }
        }
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
                // While searching, a hint is drawn only if its window is still
                // in the matched set — the letters are match indicators here,
                // not pressable shortcuts.
                if let Some(matched) = &matched {
                    if !matched.contains(window) {
                        continue;
                    }
                }

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

        // Bottom-center chrome: the command sheet in `Mode`, the query line in
        // `Search`. Both count toward `drew_any` — a zero-match search still
        // draws its line, so the overlay is never an invisible key-eater.
        match &self.state {
            State::Mode { sheet, keymap, .. } => {
                let markup = mode_sheet_markup(keymap);
                if self.render_sheet(sheet, &markup, output, renderer, scale, push) {
                    drew_any = true;
                }
            }
            State::Search {
                entries,
                query,
                line,
                ..
            } => {
                // Resolve the query-line buffer here (fresh at the output
                // scale, falling back to the carried scale-1 `line`) so the
                // drawing helper stays a plain positioning/upload step.
                let buffer = match render_query_line(query, entries, scale) {
                    Ok(buffer) => buffer,
                    Err(err) => {
                        debug!(
                            "bookmark search line failed to rasterise at scale {scale}: {err:?}"
                        );
                        line.clone()
                    }
                };
                if self.render_search_line(&buffer, output, renderer, scale, push) {
                    drew_any = true;
                }
            }
            _ => {}
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
        // succeeds, so it should not fire alongside a drawn sheet. In
        // `Search`, the rendered query line itself counts toward `drew_any`
        // (see the `render_search_line` call above), so this breadcrumb
        // fires only when even the query line failed to upload — the
        // any-key-dismiss self-heal this comment describes does not apply to
        // `Search`, where `Esc` is the only guaranteed exit.
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
    /// exact-scale render below fails or hasn't been cached yet. `markup` is
    /// the sheet text derived from the state-carried [`ModeKeymap`] by the
    /// caller — this helper never re-derives it, so it renders exactly the
    /// open instance's table. Returns whether anything was actually pushed.
    fn render_sheet<R: NiriRenderer>(
        &self,
        fallback: &MemoryBuffer,
        markup: &str,
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
                    render_markup(markup, scale)
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

    /// Draws the incremental-search query line `buffer`, anchored bottom-center
    /// where the command sheet sits in [`State::Mode`]. Returns whether
    /// anything was pushed.
    ///
    /// Unlike hints and the sheet, this line has **no** per-scale cache: its
    /// text changes on every keystroke, so a `(query, scale)` cache would
    /// explode in key space for no reuse. The caller rasterises it fresh at the
    /// output scale each frame (a per-frame upload is already this module's
    /// norm), falling back to the scale-1 buffer carried by [`State::Search`]
    /// whenever the at-scale rasterise fails; this helper takes no cache borrow
    /// at all, keeping it clear of the `RefCell` re-borrow hazard the
    /// letter/sheet paths guard against.
    fn render_search_line<R: NiriRenderer>(
        &self,
        buffer: &MemoryBuffer,
        output: &Output,
        renderer: &mut R,
        scale: f64,
        push: &mut dyn FnMut(BookmarkSwitcherRenderElement),
    ) -> bool {
        let output_sz = output_size(output);
        let size = buffer.logical_size();
        let x = ((output_sz.w - size.w) / 2.).max(0.);
        let y = (output_sz.h - size.h - f64::from(PADDING)).max(0.);
        let location: Point<f64, Logical> = Point::from((x, y));
        let location = location.to_physical_precise_round(scale).to_logical(scale);

        let Ok(texture) = TextureBuffer::from_memory_buffer(renderer.as_gles_renderer(), buffer)
        else {
            debug!("bookmark search line failed to upload as a texture at scale {scale}");
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

/// What [`BookmarkSwitcher::press_outcome`] reports back to its caller. The
/// caller acts on the terminal outcomes (jump, run command, dismiss); the
/// search-edit transitions are applied inside `press_outcome` itself and
/// collapse to [`Self::SearchUpdated`], a "state changed, please redraw"
/// signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PressOutcome {
    /// A pure modifier or lock key, or an inert key while searching: keep the
    /// overlay open, unchanged.
    HoldOpen,
    /// An un-chorded hint letter matched (or, while searching, `Enter` with a
    /// live top match): jump to this bookmark id.
    Jump(u64),
    /// (`Mode` only) an un-chorded command letter matched. `sticky` is
    /// resolved from the open instance's [`ModeKeymap`]
    /// (`bookmarks.mode-sticky`): the caller dispatches the command as usual,
    /// then, if `sticky`, dismisses and immediately reopens the overlay
    /// (a fresh instance, not a kept-open one) instead of dismissing outright.
    Command { cmd: ModeCommand, sticky: bool },
    /// Anything else that ends the overlay: Esc always dismisses. In the
    /// hint/leader (`Mode`) states, an unmatched key, `raw == None`, or a
    /// chorded press also dismisses. `Search` narrows this: an unmatched
    /// printable becomes a query edit (`SearchUpdated`) and an unmatched
    /// non-printable is [`Self::HoldOpen`] — only Esc dismisses.
    Dismiss,
    /// (`Search` only) the query changed (entered, extended, or trimmed). The
    /// overlay stays open; the caller redraws.
    SearchUpdated,
}

/// The pure routing decision behind [`BookmarkSwitcher::press_outcome`],
/// before any state mutation. `EnterSearch` / `Push` / `Pop` are the
/// search-edit transitions the wrapper applies; the other four mirror the
/// public [`PressOutcome`] terminal cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CoreOutcome {
    HoldOpen,
    Jump(u64),
    Command(ModeCommand),
    Dismiss,
    /// (`Mode` only) `/` was pressed: enter [`State::Search`].
    EnterSearch,
    /// (`Search` only) append this character to the query.
    Push(char),
    /// (`Search` only) delete the last query character.
    Pop,
}

/// Which open state [`press_outcome_core`] is routing for, carrying just the
/// data that state's routing needs. Lets the pure core stay unit-testable
/// with the `u64` id stand-in, no real [`Window`] or live compositor.
enum RoutingContext<'a, Id> {
    /// Standalone hint overlay: hint letters only.
    Hints { hints: &'a [Hint<Id>] },
    /// Leader mode: command letters, then hint letters, plus the search key →
    /// search. The resolved key table is read from `keymap`.
    Mode {
        hints: &'a [Hint<Id>],
        keymap: &'a ModeKeymap,
    },
    /// Incremental search: the query drives matching; hints are not consulted
    /// (they are indicator-only while searching).
    Search {
        entries: &'a [SearchEntry<Id>],
        query: &'a str,
    },
}

/// Pure routing core behind [`BookmarkSwitcher::press_outcome`], generic over
/// the id type so it is unit-testable with the `u64` stand-in (see the
/// `tests` module).
///
/// A pure modifier keysym holds the overlay open in every state. Past that,
/// routing splits by [`RoutingContext`]:
/// - `Hints`: chorded or `raw == None` dismisses; otherwise a hint letter jumps and anything else
///   dismisses. The search key is just an unmatched key here.
/// - `Mode`: the configured search key (default `/`) enters search — checked *before* the chorded
///   early-return so a shift-chorded search key on a non-US layout still enters search. Then a
///   chorded press dismisses, a command letter runs its command, a hint letter jumps, and anything
///   else dismisses.
/// - `Search`: a chorded press or `raw == None` holds open (a habitual Shift must not destroy the
///   typed query); `Esc` dismisses; `Enter` jumps to the top match if one exists, else holds;
///   `Backspace` pops; an otherwise printable character is pushed; any other keysym (arrows,
///   F-keys) holds.
fn press_outcome_core<Id>(
    ctx: RoutingContext<Id>,
    raw: Option<Keysym>,
    chorded: bool,
) -> CoreOutcome {
    if raw.is_some_and(is_modifier_keysym) {
        return CoreOutcome::HoldOpen;
    }

    match ctx {
        RoutingContext::Hints { hints } => {
            if chorded {
                return CoreOutcome::Dismiss;
            }
            let Some(raw) = raw else {
                return CoreOutcome::Dismiss;
            };
            match match_keysym(hints, raw) {
                Some(id) => CoreOutcome::Jump(id),
                None => CoreOutcome::Dismiss,
            }
        }
        RoutingContext::Mode { hints, keymap } => {
            // The search key enters search, checked ahead of the chorded
            // early-return so a shift-chorded search key (non-US layouts) still
            // enters rather than dismissing.
            if raw == Some(keymap.search.1) {
                return CoreOutcome::EnterSearch;
            }
            if chorded {
                return CoreOutcome::Dismiss;
            }
            let Some(raw) = raw else {
                return CoreOutcome::Dismiss;
            };
            // Commands can't collide with hints by construction (the mode
            // alphabet excludes every command and search character), but
            // checking commands first costs nothing and keeps that guarantee
            // from being load-bearing here.
            if let Some(cmd) = keymap.command_for_keysym(raw) {
                return CoreOutcome::Command(cmd);
            }
            match match_keysym(hints, raw) {
                Some(id) => CoreOutcome::Jump(id),
                None => CoreOutcome::Dismiss,
            }
        }
        RoutingContext::Search { entries, query } => {
            // A chorded press must not destroy the typed query (a habitual
            // Shift is not a dismiss here), and neither must an unmapped
            // keysym.
            if chorded {
                return CoreOutcome::HoldOpen;
            }
            let Some(raw) = raw else {
                return CoreOutcome::HoldOpen;
            };
            if raw == Keysym::Escape {
                return CoreOutcome::Dismiss;
            }
            if raw == Keysym::Return || raw == Keysym::KP_Enter {
                let query_lower = query.to_lowercase();
                return match top_match(entries, &query_lower) {
                    Some(entry) => CoreOutcome::Jump(entry.bookmark_id),
                    None => CoreOutcome::HoldOpen,
                };
            }
            if raw == Keysym::BackSpace {
                return CoreOutcome::Pop;
            }
            // Base (unshifted) keysym → char; control chars (Tab, etc.) are
            // not query text. See the module doc's note on the capitals/IME
            // limitation this base-keysym mapping implies.
            if let Some(ch) = raw.key_char() {
                if !ch.is_control() {
                    return CoreOutcome::Push(ch);
                }
            }
            CoreOutcome::HoldOpen
        }
    }
}

/// One-line command-sheet markup, e.g. `a add · d/x remove · ,/. walk ·
/// letter jump · / search · esc close`. Generated from the resolved
/// [`ModeKeymap`], grouping consecutive command entries that share a label
/// (the two walk directions both "walk", the two default remove keys `d`/`x`)
/// so the command portion can never drift from the routing table. The trailing
/// `letter jump · <search> search · esc close` clause is hand-written and
/// interpolates the configured search character; its shape is pinned by the
/// `mode_sheet_markup_mentions_search_key` test rather than by construction.
fn mode_sheet_markup(keymap: &ModeKeymap) -> String {
    let mut groups: Vec<(String, &str)> = Vec::new();
    for &(ch, _, cmd) in &keymap.commands {
        let label = cmd.label();
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
    text.push_str(&format!(
        " · letter jump · {} search · esc close",
        keymap.search.0
    ));

    format!("<span face='mono'>{text}</span>")
}

/// Assigns hint letters to visible bookmarks in list order.
///
/// `bookmarks` yields `(bookmark id, anchor window id)` in presentation
/// order; only those whose window is in `visible` are kept, and they are
/// zipped with `alphabet` (chars, in order) so the letters are compact (no
/// gaps for hidden bookmarks). Bookmarks past the end of `alphabet` get no
/// hint. Callers pass the configured hint alphabet
/// ([`ModeKeymap::full_alphabet`], standalone switcher) or its command-and-
/// search-stripped form ([`ModeKeymap::mode_alphabet`], leader mode).
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

/// Rasterises a padded, bordered box (dark fill, bright border, white text)
/// around whatever `set_content` puts on the pango layout. The shared body of
/// [`render_markup`] (trusted markup) and [`render_text`] (untrusted plain
/// text); the returned [`MemoryBuffer`] is uploaded to a texture at render
/// time. `set_content` runs twice — once on the throwaway sizing layout and
/// once on the drawing layout — because the two are distinct pango layouts.
fn render_boxed(
    scale: f64,
    set_content: impl Fn(&pangocairo::pango::Layout),
) -> anyhow::Result<MemoryBuffer> {
    let _span = tracy_client::span!("bookmark_switcher::render_boxed");

    let padding: i32 = to_physical_precise_round(scale, PADDING);

    let mut font = FontDescription::from_string(FONT);
    font.set_absolute_size(to_physical_precise_round(scale, font.size()));

    let surface = ImageSurface::create(cairo::Format::ARgb32, 0, 0)?;
    let cr = cairo::Context::new(&surface)?;
    let layout = pangocairo::functions::create_layout(&cr);
    layout.context().set_round_glyph_positions(false);
    layout.set_font_description(Some(&font));
    set_content(&layout);

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
    set_content(&layout);

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

/// Rasterises pango **markup** into the shared box. Callers must pass trusted
/// markup only — hint letters ([`render_hint`]) and the leader-mode command
/// sheet ([`mode_sheet_markup`]), both module-generated.
fn render_markup(markup: &str, scale: f64) -> anyhow::Result<MemoryBuffer> {
    render_boxed(scale, |layout| layout.set_markup(markup))
}

/// Rasterises **plain text** into the shared box via `set_text` — never
/// `set_markup`. The search query and the window titles woven into the query
/// line are untrusted, so they must never be interpreted as pango markup (a
/// `&` or `<` in a title would corrupt or inject the layout). Mirrors the
/// `set_text` discipline in [`crate::ui::mru`]'s title rendering.
fn render_text(text: &str, scale: f64) -> anyhow::Result<MemoryBuffer> {
    render_boxed(scale, |layout| layout.set_text(text))
}

/// Rasterises the incremental-search query line at `scale`: filters `entries`
/// by `query` (lowercased once), formats the status text with
/// [`search_line_text`], and renders it as plain text. Shared by the edit-time
/// scale-1 rasterise ([`BookmarkSwitcher::enter_search`] and friends) and the
/// per-frame at-scale render ([`BookmarkSwitcher::render_search_line`]).
fn render_query_line<Id>(
    query: &str,
    entries: &[SearchEntry<Id>],
    scale: f64,
) -> anyhow::Result<MemoryBuffer> {
    let query_lower = query.to_lowercase();
    let matches: Vec<&SearchEntry<Id>> = entries
        .iter()
        .filter(|entry| entry_matches(entry, &query_lower))
        .collect();
    let top_label = matches.first().map(|entry| entry.label.as_str());
    let text = search_line_text(query, matches.len(), top_label);
    render_text(&text, scale)
}

/// The one-line search status text, e.g.
/// `/qu — 3 matches · enter → Mail — inbox · esc close`, or with no matches
/// `/quz — no matches · esc close`. Pure so it is unit-testable. `top_label`
/// is the display label of the top match, truncated char-boundary-safe.
fn search_line_text(query: &str, match_count: usize, top_label: Option<&str>) -> String {
    if match_count == 0 {
        return format!("/{query} — no matches · esc close");
    }
    let noun = if match_count == 1 { "match" } else { "matches" };
    let mut text = format!("/{query} — {match_count} {noun}");
    if let Some(label) = top_label {
        text.push_str(" · enter → ");
        text.push_str(&truncate_label(label));
    }
    text.push_str(" · esc close");
    text
}

/// Truncates a display label to at most 48 characters, appending `…` when it
/// had to cut. Truncates on *character* boundaries (`chars().take`), never
/// bytes — a byte slice would panic mid-UTF-8.
fn truncate_label(label: &str) -> String {
    const MAX_CHARS: usize = 48;
    if label.chars().count() > MAX_CHARS {
        let mut truncated: String = label.chars().take(MAX_CHARS).collect();
        truncated.push('…');
        truncated
    } else {
        label.to_owned()
    }
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

    /// The default hint alphabet, as a test fixture for the pure `build_hints`
    /// assignment tests. Mirrors `jiji_config::HintAlphabet::default`; the
    /// production alphabet lives in jiji-config, this is just a convenient
    /// literal for exercising letter assignment without a config round-trip.
    const HINT_KEYS: &str = "asdfghjklqwertyuiopzxcvbnm";

    /// The resolved keymap for the default config — the table the overlay used
    /// before the knob existed. Used by the leader-mode routing/sheet tests.
    fn default_keymap() -> ModeKeymap {
        ModeKeymap::from_config(&jiji_config::BookmarksConfig::default())
    }

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
        let mut switcher = BookmarkSwitcher::new();
        assert!(!switcher.is_open());

        switcher.press_outcome(None, false);
    }

    #[test]
    fn press_outcome_carries_sticky_from_keymap() {
        // The outer `press_outcome` wrapper (not just the pure
        // `press_outcome_core`) must read `sticky` off the live
        // `State::Mode` keymap when it builds `PressOutcome::Command`;
        // exercised through the real entry point rather than
        // `press_outcome_core` directly.
        let sticky_cfg = jiji_config::BookmarksConfig {
            mode_sticky: true,
            ..Default::default()
        };
        let sticky_keymap = ModeKeymap::from_config(&sticky_cfg);
        let sheet = render_markup(&mode_sheet_markup(&sticky_keymap), 1.)
            .expect("sheet markup must rasterise");
        let mut switcher = BookmarkSwitcher::new();
        switcher.state = State::Mode {
            hints: Vec::new(),
            entries: Vec::new(),
            sheet,
            keymap: sticky_keymap,
        };

        let outcome = switcher.press_outcome(Some(Keysym::from_char('a')), false);
        assert!(matches!(
            outcome,
            PressOutcome::Command { sticky: true, .. }
        ));

        let default_keymap = default_keymap();
        let sheet = render_markup(&mode_sheet_markup(&default_keymap), 1.)
            .expect("sheet markup must rasterise");
        switcher.state = State::Mode {
            hints: Vec::new(),
            entries: Vec::new(),
            sheet,
            keymap: default_keymap,
        };

        let outcome = switcher.press_outcome(Some(Keysym::from_char('a')), false);
        assert!(matches!(
            outcome,
            PressOutcome::Command { sticky: false, .. }
        ));
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
    fn default_keymap_pins_the_legacy_table() {
        // The default resolved table, hardcoded (not re-derived from config) so
        // a change to the jiji-config defaults trips this test.
        let keymap = default_keymap();
        let commands: Vec<(char, ModeCommand)> = keymap
            .commands
            .iter()
            .map(|&(ch, _, cmd)| (ch, cmd))
            .collect();
        assert_eq!(
            commands,
            vec![
                ('a', ModeCommand::Add),
                ('d', ModeCommand::RemoveFocused),
                ('x', ModeCommand::RemoveFocused),
                (',', ModeCommand::WalkBackward),
                ('.', ModeCommand::WalkForward),
            ]
        );
        assert_eq!(keymap.search.0, '/');
        assert_eq!(keymap.full_alphabet, "asdfghjklqwertyuiopzxcvbnm");
        assert_eq!(keymap.full_alphabet.len(), 26);
        assert!(!keymap.sticky);
    }

    #[test]
    fn keymap_resolves_sticky_from_config() {
        let keymap = ModeKeymap::from_config(&jiji_config::BookmarksConfig::default());
        assert!(!keymap.sticky);

        let cfg = jiji_config::BookmarksConfig {
            mode_sticky: true,
            ..Default::default()
        };
        let keymap = ModeKeymap::from_config(&cfg);
        assert!(keymap.sticky);
    }

    #[test]
    fn default_mode_alphabet_excludes_every_command_and_search_char() {
        let keymap = default_keymap();
        for &(ch, _, _) in &keymap.commands {
            assert!(
                !keymap.mode_alphabet.contains(ch),
                "mode alphabet must not contain command char '{ch}'"
            );
        }
        assert!(
            !keymap.mode_alphabet.contains(keymap.search.0),
            "mode alphabet must not contain the search char"
        );
        // 'a', 'd', 'x' are stripped from the 26-letter alphabet ('.' , ',',
        // and '/' were never in it to begin with).
        assert_eq!(keymap.mode_alphabet.len(), keymap.full_alphabet.len() - 3);
    }

    #[test]
    fn mode_alphabet_strips_configured_command_and_search_chars() {
        // Command and search keys drawn from inside the alphabet are all
        // stripped from the derived mode alphabet.
        let cfg = jiji_config::BookmarksConfig {
            mode_keys: jiji_config::ModeKeysConfig::new(
                vec!['q'],
                vec!['w', 'e'],
                vec!['r'],
                vec!['t'],
                'y',
            )
            .expect("valid mode-keys table"),
            ..Default::default()
        };
        let keymap = ModeKeymap::from_config(&cfg);
        for ch in ['q', 'w', 'e', 'r', 't', 'y'] {
            assert!(
                !keymap.mode_alphabet.contains(ch),
                "mode alphabet must strip configured key '{ch}'"
            );
        }
        // A non-command letter survives.
        assert!(keymap.mode_alphabet.contains('a'));
        assert_eq!(keymap.mode_alphabet.len(), keymap.full_alphabet.len() - 6);
    }

    #[test]
    fn mode_alphabet_can_derive_empty_and_stays_constructible() {
        // A hint alphabet made entirely of command/search keys strips to empty
        // — legal (the mode opens with zero hints), and `from_config` must not
        // panic building it.
        let cfg = jiji_config::BookmarksConfig {
            hint_alphabet: jiji_config::HintAlphabet::new("ad").expect("valid alphabet"),
            mode_keys: jiji_config::ModeKeysConfig::new(
                vec!['a'],
                vec!['d'],
                vec![','],
                vec!['.'],
                '/',
            )
            .expect("valid mode-keys table"),
            ..Default::default()
        };
        let keymap = ModeKeymap::from_config(&cfg);
        assert!(keymap.mode_alphabet.is_empty());
    }

    /// A `State::Mode` routing context over the given hints and keymap, for the
    /// pure `press_outcome_core` tests.
    fn mode_ctx<'a>(hints: &'a [Hint<u64>], keymap: &'a ModeKeymap) -> RoutingContext<'a, u64> {
        RoutingContext::Mode { hints, keymap }
    }

    /// A `State::Hints` (standalone) routing context.
    fn hints_ctx(hints: &[Hint<u64>]) -> RoutingContext<'_, u64> {
        RoutingContext::Hints { hints }
    }

    #[test]
    fn press_outcome_core_routes_modifier_to_hold_open() {
        let hints: Vec<Hint<u64>> = Vec::new();
        let keymap = default_keymap();
        let outcome = press_outcome_core(mode_ctx(&hints, &keymap), Some(Keysym::Shift_L), false);
        assert_eq!(outcome, CoreOutcome::HoldOpen);
    }

    #[test]
    fn press_outcome_core_routes_command_char_in_mode() {
        let hints: Vec<Hint<u64>> = Vec::new();
        let keymap = default_keymap();
        let outcome = press_outcome_core(
            mode_ctx(&hints, &keymap),
            Some(Keysym::from_char('a')),
            false,
        );
        assert_eq!(outcome, CoreOutcome::Command(ModeCommand::Add));

        let outcome = press_outcome_core(
            mode_ctx(&hints, &keymap),
            Some(Keysym::from_char('d')),
            false,
        );
        assert_eq!(outcome, CoreOutcome::Command(ModeCommand::RemoveFocused));

        let outcome = press_outcome_core(
            mode_ctx(&hints, &keymap),
            Some(Keysym::from_char('x')),
            false,
        );
        assert_eq!(outcome, CoreOutcome::Command(ModeCommand::RemoveFocused));

        let outcome = press_outcome_core(
            mode_ctx(&hints, &keymap),
            Some(Keysym::from_char(',')),
            false,
        );
        assert_eq!(outcome, CoreOutcome::Command(ModeCommand::WalkBackward));

        let outcome = press_outcome_core(
            mode_ctx(&hints, &keymap),
            Some(Keysym::from_char('.')),
            false,
        );
        assert_eq!(outcome, CoreOutcome::Command(ModeCommand::WalkForward));
    }

    #[test]
    fn press_outcome_core_command_char_dismisses_outside_mode() {
        // In the standalone hint overlay, a command letter is just an
        // unmatched hint letter: dismiss.
        let hints: Vec<Hint<u64>> = Vec::new();
        let outcome = press_outcome_core(hints_ctx(&hints), Some(Keysym::from_char('a')), false);
        assert_eq!(outcome, CoreOutcome::Dismiss);
    }

    #[test]
    fn press_outcome_core_slash_in_mode_enters_search_unchorded_and_chorded() {
        // The default search key `/` enters search from leader mode. It is
        // checked ahead of the chorded early-return, so a shift-chorded search
        // key (as on layouts where `/` sits on a shifted key) still enters
        // rather than dismissing.
        let hints: Vec<Hint<u64>> = Vec::new();
        let keymap = default_keymap();
        assert_eq!(
            press_outcome_core(
                mode_ctx(&hints, &keymap),
                Some(Keysym::from_char('/')),
                false
            ),
            CoreOutcome::EnterSearch
        );
        assert_eq!(
            press_outcome_core(
                mode_ctx(&hints, &keymap),
                Some(Keysym::from_char('/')),
                true
            ),
            CoreOutcome::EnterSearch
        );
    }

    #[test]
    fn press_outcome_core_slash_in_standalone_hints_dismisses() {
        // Outside leader mode there is no search: `/` is an ordinary unmatched
        // key and dismisses.
        let hints: Vec<Hint<u64>> = Vec::new();
        assert_eq!(
            press_outcome_core(hints_ctx(&hints), Some(Keysym::from_char('/')), false),
            CoreOutcome::Dismiss
        );
    }

    #[test]
    fn press_outcome_core_chorded_command_char_in_mode_dismisses() {
        // A chorded press short-circuits before command matching (a non-search
        // key), even when the raw keysym is a command letter.
        let hints: Vec<Hint<u64>> = Vec::new();
        let keymap = default_keymap();
        let outcome = press_outcome_core(
            mode_ctx(&hints, &keymap),
            Some(Keysym::from_char('a')),
            true,
        );
        assert_eq!(outcome, CoreOutcome::Dismiss);
    }

    #[test]
    fn press_outcome_core_modifier_and_none_dismiss_outside_mode_too() {
        // HoldOpen (modifier) is checked ahead of the per-context split, so it
        // behaves identically in every state; `raw == None` dismisses in the
        // standalone hint overlay.
        let hints: Vec<Hint<u64>> = Vec::new();
        let outcome = press_outcome_core(hints_ctx(&hints), Some(Keysym::Shift_L), false);
        assert_eq!(outcome, CoreOutcome::HoldOpen);

        let outcome = press_outcome_core(hints_ctx(&hints), None, false);
        assert_eq!(outcome, CoreOutcome::Dismiss);
    }

    #[test]
    fn press_outcome_core_routes_hint_letter_to_jump() {
        let hints = build_hints(fixture(&[10]).into_iter(), &visible(&[10]), HINT_KEYS);
        let outcome = press_outcome_core(hints_ctx(&hints), Some(Keysym::from_char('a')), false);
        assert_eq!(outcome, CoreOutcome::Jump(10));

        // In mode, a hint letter drawn from the mode alphabet (which never
        // collides with a command char) still jumps.
        let keymap = default_keymap();
        let mode_hints = build_hints(
            fixture(&[10]).into_iter(),
            &visible(&[10]),
            &keymap.mode_alphabet,
        );
        let outcome = press_outcome_core(
            mode_ctx(&mode_hints, &keymap),
            Some(mode_hints[0].keysym),
            false,
        );
        assert_eq!(outcome, CoreOutcome::Jump(10));
    }

    #[test]
    fn press_outcome_core_chorded_letter_dismisses() {
        let hints = build_hints(fixture(&[10]).into_iter(), &visible(&[10]), HINT_KEYS);
        let outcome = press_outcome_core(hints_ctx(&hints), Some(Keysym::from_char('a')), true);
        assert_eq!(outcome, CoreOutcome::Dismiss);
    }

    #[test]
    fn press_outcome_core_escape_unmatched_and_none_dismiss() {
        let hints: Vec<Hint<u64>> = Vec::new();
        let keymap = default_keymap();
        assert_eq!(
            press_outcome_core(mode_ctx(&hints, &keymap), Some(Keysym::Escape), false),
            CoreOutcome::Dismiss
        );
        assert_eq!(
            press_outcome_core(
                mode_ctx(&hints, &keymap),
                Some(Keysym::from_char('z')),
                false
            ),
            CoreOutcome::Dismiss
        );
        assert_eq!(
            press_outcome_core(mode_ctx(&hints, &keymap), None, false),
            CoreOutcome::Dismiss
        );
    }

    #[test]
    fn press_outcome_core_mode_constructible_with_zero_hints_still_routes_commands() {
        let hints: Vec<Hint<u64>> = Vec::new();
        let keymap = default_keymap();
        let outcome = press_outcome_core(
            mode_ctx(&hints, &keymap),
            Some(Keysym::from_char('a')),
            false,
        );
        assert_eq!(outcome, CoreOutcome::Command(ModeCommand::Add));
    }

    #[test]
    fn press_outcome_core_routes_remapped_command_and_old_char_is_ordinary() {
        // Remapping `add` to `b` routes `b` to Add, while the old `a` is no
        // longer a command — with no hint bound to it, it dismisses like any
        // other unmatched key.
        let cfg = jiji_config::BookmarksConfig {
            mode_keys: jiji_config::ModeKeysConfig::new(
                vec!['b'],
                vec!['d', 'x'],
                vec![','],
                vec!['.'],
                '/',
            )
            .expect("valid mode-keys table"),
            ..Default::default()
        };
        let keymap = ModeKeymap::from_config(&cfg);
        let hints: Vec<Hint<u64>> = Vec::new();
        assert_eq!(
            press_outcome_core(
                mode_ctx(&hints, &keymap),
                Some(Keysym::from_char('b')),
                false
            ),
            CoreOutcome::Command(ModeCommand::Add)
        );
        assert_eq!(
            press_outcome_core(
                mode_ctx(&hints, &keymap),
                Some(Keysym::from_char('a')),
                false
            ),
            CoreOutcome::Dismiss
        );
    }

    // --- Search routing and matching ---

    /// Builds search entries from `(bookmark id, name, title)` triples, using
    /// the `u64` id as its own window stand-in so the pure matching logic
    /// needs no real `Window`.
    fn search_entries(rows: &[(u64, Option<&str>, Option<&str>)]) -> Vec<SearchEntry<u64>> {
        rows.iter()
            .map(|&(id, name, title)| SearchEntry::new(id, id, name, title))
            .collect()
    }

    fn search_ctx<'a>(entries: &'a [SearchEntry<u64>], query: &'a str) -> RoutingContext<'a, u64> {
        RoutingContext::Search { entries, query }
    }

    #[test]
    fn entry_matches_by_name_and_title_independently() {
        let entry = SearchEntry::new(1, 1, Some("Mail"), Some("Inbox — Fastmail"));
        // Matches the name.
        assert!(entry_matches(&entry, "mail"));
        // Matches the title.
        assert!(entry_matches(&entry, "inbox"));
        // A query spanning the name/title boundary must NOT match: the two
        // fields are never concatenated into one haystack.
        assert!(!entry_matches(&entry, "mailinbox"));
    }

    #[test]
    fn entry_matches_is_case_insensitive() {
        let entry = SearchEntry::new(1, 1, Some("Mail"), None);
        // The query is lowercased by the caller; the stored fields are
        // lowercased at construction, so matching ignores case both ways.
        assert!(entry_matches(&entry, "mail"));
    }

    #[test]
    fn entry_matches_empty_query_matches_everything() {
        let named = SearchEntry::new(1, 1, Some("Mail"), None);
        let untitled = SearchEntry::new(2, 2, None, None);
        assert!(entry_matches(&named, ""));
        assert!(entry_matches(&untitled, ""));
    }

    #[test]
    fn top_match_is_first_in_list_order() {
        let entries = search_entries(&[
            (1, Some("Editor"), None),
            (2, Some("Mail client"), None),
            (3, Some("Mail archive"), None),
        ]);
        // Two entries match "mail"; the top match is the first in list order.
        let top = top_match(&entries, "mail").expect("a match exists");
        assert_eq!(top.bookmark_id, 2);
        // No match yields None.
        assert!(top_match(&entries, "zzz").is_none());
    }

    #[test]
    fn search_label_prefers_name_then_clean_title_then_untitled() {
        assert_eq!(
            SearchEntry::new(1, 1, Some("Mail"), Some("Inbox")).label,
            "Mail"
        );
        assert_eq!(SearchEntry::new(2, 2, None, Some("Inbox")).label, "Inbox");
        // An empty clean title is treated as untitled.
        assert_eq!(SearchEntry::new(3, 3, None, Some("")).label, "(untitled)");
        assert_eq!(SearchEntry::new(4, 4, None, None).label, "(untitled)");
    }

    #[test]
    fn press_outcome_core_search_pushes_printable_char() {
        let entries = search_entries(&[(1, Some("Mail"), None)]);
        let outcome = press_outcome_core(
            search_ctx(&entries, ""),
            Some(Keysym::from_char('m')),
            false,
        );
        assert_eq!(outcome, CoreOutcome::Push('m'));
    }

    #[test]
    fn press_outcome_core_search_non_char_keysym_holds_open() {
        // A function key (no `key_char`) must not dismiss or corrupt the query.
        let entries = search_entries(&[(1, Some("Mail"), None)]);
        let outcome = press_outcome_core(search_ctx(&entries, "ma"), Some(Keysym::F5), false);
        assert_eq!(outcome, CoreOutcome::HoldOpen);
    }

    #[test]
    fn press_outcome_core_search_control_char_holds_open() {
        // Tab does have a `key_char` (`Some('\t')`), unlike F5, so this
        // exercises the `is_control()` filter specifically rather than the
        // `key_char().is_none()` branch above — a regression that dropped
        // the `is_control` check would wrongly `Push('\t')` into the query.
        let entries = search_entries(&[(1, Some("Mail"), None)]);
        let outcome = press_outcome_core(search_ctx(&entries, "ma"), Some(Keysym::Tab), false);
        assert_eq!(outcome, CoreOutcome::HoldOpen);
    }

    #[test]
    fn press_outcome_core_search_backspace_pops() {
        let entries = search_entries(&[(1, Some("Mail"), None)]);
        let outcome =
            press_outcome_core(search_ctx(&entries, "ma"), Some(Keysym::BackSpace), false);
        assert_eq!(outcome, CoreOutcome::Pop);
    }

    #[test]
    fn press_outcome_core_search_enter_jumps_to_top_match_or_holds() {
        let entries = search_entries(&[(7, Some("Mail"), None), (8, Some("Music"), None)]);
        // "m" matches both; Enter jumps to the first in list order.
        assert_eq!(
            press_outcome_core(search_ctx(&entries, "m"), Some(Keysym::Return), false),
            CoreOutcome::Jump(7)
        );
        // A query with no match holds open rather than dismissing — the user
        // can keep correcting.
        assert_eq!(
            press_outcome_core(search_ctx(&entries, "zzz"), Some(Keysym::Return), false),
            CoreOutcome::HoldOpen
        );
    }

    #[test]
    fn press_outcome_core_search_chorded_holds_open_not_dismiss() {
        // A habitual Shift while typing must not destroy the query.
        let entries = search_entries(&[(1, Some("Mail"), None)]);
        let outcome = press_outcome_core(
            search_ctx(&entries, "ma"),
            Some(Keysym::from_char('m')),
            true,
        );
        assert_eq!(outcome, CoreOutcome::HoldOpen);
    }

    #[test]
    fn press_outcome_core_search_escape_dismisses() {
        let entries = search_entries(&[(1, Some("Mail"), None)]);
        let outcome = press_outcome_core(search_ctx(&entries, "ma"), Some(Keysym::Escape), false);
        assert_eq!(outcome, CoreOutcome::Dismiss);
    }

    #[test]
    fn search_line_text_zero_match_and_truncation() {
        // Zero matches: no "enter →" clause.
        assert_eq!(
            search_line_text("quz", 0, None),
            "/quz — no matches · esc close"
        );
        // Singular vs plural noun, and the top-match label is shown.
        assert_eq!(
            search_line_text("qu", 1, Some("Mail")),
            "/qu — 1 match · enter → Mail · esc close"
        );
        assert_eq!(
            search_line_text("qu", 3, Some("Mail")),
            "/qu — 3 matches · enter → Mail · esc close"
        );
        // An over-long label is truncated on a char boundary with an ellipsis.
        let long: String = "x".repeat(60);
        let line = search_line_text("q", 1, Some(&long));
        assert!(line.contains(&format!("{}…", "x".repeat(48))));
        assert!(!line.contains(&"x".repeat(49)));
    }

    #[test]
    fn truncate_label_multi_byte_char_boundary() {
        // Multi-byte characters (here: CJK, 3 bytes each in UTF-8) crossing
        // the 48-char boundary must be truncated on a *character* boundary —
        // a byte-slicing regression would panic mid-codepoint on this input.
        let long: String = "日".repeat(60);
        let truncated = truncate_label(&long);
        assert_eq!(truncated.chars().count(), 49); // 48 chars + '…'
        assert!(truncated.ends_with('…'));
        assert_eq!(truncated.chars().filter(|&c| c == '日').count(), 48);

        // A single trailing emoji (4 bytes, non-BMP) right at the boundary.
        let mixed: String = "a".repeat(47) + "🎉🎉";
        let truncated = truncate_label(&mixed);
        assert_eq!(truncated.chars().count(), 49); // 48 chars + '…'
        assert!(truncated.ends_with('…'));
    }

    #[test]
    fn mode_sheet_markup_contains_every_command_key_and_label() {
        let keymap = default_keymap();
        let markup = mode_sheet_markup(&keymap);
        for &(ch, _, cmd) in &keymap.commands {
            assert!(
                markup.contains(ch),
                "sheet markup must mention command char '{ch}'"
            );
            assert!(
                markup.contains(cmd.label()),
                "sheet markup must mention label '{}'",
                cmd.label()
            );
        }
    }

    #[test]
    fn mode_sheet_markup_aliases_d_and_x_together() {
        // `d` and `x` share the "remove" label and must be grouped rather
        // than appearing as two separate "remove" entries.
        let keymap = default_keymap();
        let markup = mode_sheet_markup(&keymap);
        assert!(
            markup.contains("d/x remove"),
            "expected 'd/x remove' grouping in: {markup}"
        );
    }

    #[test]
    fn mode_sheet_markup_mentions_search_key() {
        // The search key is a state transition into search, not a `ModeCommand`,
        // so its sheet mention is hand-written; pin that it interpolates the
        // configured character (default `/`).
        let keymap = default_keymap();
        let markup = mode_sheet_markup(&keymap);
        assert!(
            markup.contains("/ search"),
            "expected '/ search' mention in: {markup}"
        );
    }

    #[test]
    fn mode_sheet_markup_interpolates_custom_command_and_search_chars() {
        let cfg = jiji_config::BookmarksConfig {
            mode_keys: jiji_config::ModeKeysConfig::new(
                vec!['n'],
                vec!['m'],
                vec!['j'],
                vec!['k'],
                ';',
            )
            .expect("valid mode-keys table"),
            ..Default::default()
        };
        let keymap = ModeKeymap::from_config(&cfg);
        let markup = mode_sheet_markup(&keymap);
        assert!(markup.contains("n add"), "in: {markup}");
        assert!(markup.contains("m remove"), "in: {markup}");
        // The two walk directions share the "walk" label and group together.
        assert!(markup.contains("j/k walk"), "in: {markup}");
        assert!(markup.contains("; search"), "in: {markup}");
    }
}
