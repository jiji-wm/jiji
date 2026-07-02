//! Curated window bookmarks: a hand-maintained, cursor-walked list of windows
//! the user pins for quick return.
//!
//! This module owns the *pure* state machine — add / re-press, remove, reorder,
//! the walk-target arithmetic, focus observation (including MRU promotion), and
//! prune-on-close. It knows nothing about how a bookmark is restored (activity
//! switch, window focus); that lives in `Layout`, which computes a walk target
//! here, gates on an activity-switch hard block, and only then commits the
//! cursor and executes the restore. Splitting the two lets a hard-blocked walk
//! leave this state bit-identical so the parked-and-re-dispatched action does
//! not double-step.
//!
//! Unlike a helix-style jumplist, the list is uncapped, never truncates a
//! forward tail, and holds at most one bookmark per window (the window is the
//! identity; the activity is carried context for restore, not part of the key).

use std::fmt;

use jiji_config::utils::RegexEq;
use jiji_config::{Key, OrderMode, RepressPolicy, Trigger};

use super::activity::ActivityId;

/// A validated dynamic bookmark keybind.
///
/// Constructed only via [`BookmarkKey::new`], which rejects any trigger other
/// than a keysym (a bookmark bind is keyboard-only) and any key with no
/// modifiers (a bare keysym would eat plain typing). The private field keeps
/// every live `BookmarkKey` valid by construction — no caller can smuggle in
/// an unvalidated [`Key`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BookmarkKey(Key);

/// Why a [`Key`] was rejected by [`BookmarkKey::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BookmarkKeyError {
    /// The trigger is not a keysym (a mouse button, wheel, or touchpad
    /// gesture).
    NotAKeysym,
    /// The key carries no modifiers.
    NoModifiers,
}

impl fmt::Display for BookmarkKeyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotAKeysym => write!(f, "must be a keyboard key, not a mouse or wheel trigger"),
            Self::NoModifiers => write!(f, "must include at least one modifier"),
        }
    }
}

impl BookmarkKey {
    /// Validate `key` as a dynamic bookmark keybind.
    ///
    /// # Errors
    ///
    /// Returns [`BookmarkKeyError::NotAKeysym`] for a mouse/wheel/touchpad
    /// trigger, or [`BookmarkKeyError::NoModifiers`] for a bare keysym.
    pub fn new(key: Key) -> Result<Self, BookmarkKeyError> {
        if !matches!(key.trigger, Trigger::Keysym(_)) {
            return Err(BookmarkKeyError::NotAKeysym);
        }
        if key.modifiers.is_empty() {
            return Err(BookmarkKeyError::NoModifiers);
        }
        Ok(Self(key))
    }

    /// The validated inner key.
    pub fn key(self) -> Key {
        self.0
    }
}

/// A validated user-facing bookmark display name.
///
/// Constructed only via [`BookmarkName::new`], which trims leading/trailing
/// whitespace and rejects an empty-after-trim string or any control
/// character (defense against fuzzel/pango injection at the picker layer,
/// though `jiji-do` also sanitizes independently). Names are display labels
/// only — [`BookmarkId`] remains the sole resolution handle, so duplicate
/// names across bookmarks are legal and carry no invariant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BookmarkName(String);

/// Why a raw string was rejected by [`BookmarkName::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BookmarkNameError {
    /// The string was empty after trimming leading/trailing whitespace.
    Empty,
    /// The string contains a control character.
    ControlChars,
}

impl fmt::Display for BookmarkNameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "must not be empty"),
            Self::ControlChars => write!(f, "must not contain control characters"),
        }
    }
}

impl BookmarkName {
    /// Validate `raw` as a bookmark display name.
    ///
    /// Trims leading/trailing whitespace first, so a whitespace-only input is
    /// rejected as [`BookmarkNameError::Empty`], never coerced to a clear
    /// (clearing a name is done by passing `None` to
    /// [`Bookmarks::rename`], not an empty string).
    ///
    /// # Errors
    ///
    /// Returns [`BookmarkNameError::Empty`] for an empty-after-trim string,
    /// or [`BookmarkNameError::ControlChars`] if any character is
    /// [`char::is_control`].
    pub fn new(raw: &str) -> Result<Self, BookmarkNameError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(BookmarkNameError::Empty);
        }
        if trimmed.chars().any(char::is_control) {
            return Err(BookmarkNameError::ControlChars);
        }
        Ok(Self(trimmed.to_owned()))
    }

    /// The validated name.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Stable identity of a bookmark, minted monotonically and never reused.
///
/// A pruned bookmark's id is retired for good: [`Bookmarks::next_id`] only ever
/// grows, so a fresh add never collides with a stale id a client may still hold.
///
/// `BookmarkId` is confined to [`Bookmarks`]: callers above it hold the raw
/// `u64` instead, because an id can go stale (the window closes, pruning the
/// bookmark) across a gap such as an open confirm dialog. A raw id must be
/// revalidated via [`Bookmarks::id_for_raw`] before use, never re-wrapped
/// directly into a `BookmarkId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BookmarkId(u64);

impl BookmarkId {
    /// The raw wire value, as surfaced over IPC and accepted back on dispatch.
    pub fn get(self) -> u64 {
        self.0
    }
}

/// Why [`BookmarkRule::new`] rejected a rule.
#[must_use]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BookmarkRuleError {
    /// Neither `app_id` nor `title` was given; a rule that matches nothing is
    /// rejected.
    Empty,
}

impl fmt::Display for BookmarkRuleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "at least one of app-id or title is required"),
        }
    }
}

/// A durable window-matching rule for a rule-anchored bookmark.
///
/// Constructed only via [`BookmarkRule::new`], which requires at least one
/// field (a rule matching nothing is meaningless); the private fields keep
/// every live rule valid by construction — outside callers can't smuggle in
/// an empty rule, though a same-module literal (`BookmarkRule { app_id: None,
/// title: None }`) would still bypass the check, as with any private-field
/// invariant in this module. Matching reuses the window-rules semantics (see
/// `window_matches` in `src/window/mod.rs`): present fields are AND-ed, and
/// each is an [`regex::Regex::is_match`] test — on the app-id and on the
/// **raw** (machine-tagged) title, the same strings a user writes window-rule
/// regexes against. A rule that names a field the window lacks (e.g. a
/// `title` regex on a titleless window) does not match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BookmarkRule {
    app_id: Option<RegexEq>,
    title: Option<RegexEq>,
}

impl BookmarkRule {
    /// Build a rule from optional app-id and title regexes.
    ///
    /// # Errors
    ///
    /// Returns [`BookmarkRuleError::Empty`] when both fields are `None`.
    pub fn new(app_id: Option<RegexEq>, title: Option<RegexEq>) -> Result<Self, BookmarkRuleError> {
        if app_id.is_none() && title.is_none() {
            return Err(BookmarkRuleError::Empty);
        }
        Ok(Self { app_id, title })
    }

    /// Whether this rule matches a window with the given app-id and raw title.
    ///
    /// Present fields are AND-ed; an absent window field against a present rule
    /// field is a non-match.
    pub(crate) fn matches(&self, app_id: Option<&str>, title: Option<&str>) -> bool {
        if let Some(re) = &self.app_id {
            let Some(app_id) = app_id else {
                return false;
            };
            if !re.0.is_match(app_id) {
                return false;
            }
        }
        if let Some(re) = &self.title {
            let Some(title) = title else {
                return false;
            };
            if !re.0.is_match(title) {
                return false;
            }
        }
        true
    }

    /// The app-id regex source string, if the rule constrains app-id.
    pub(crate) fn app_id_source(&self) -> Option<&str> {
        self.app_id.as_ref().map(|re| re.0.as_str())
    }

    /// The title regex source string, if the rule constrains title.
    pub(crate) fn title_source(&self) -> Option<&str> {
        self.title.as_ref().map(|re| re.0.as_str())
    }
}

/// The window a rule anchor is currently attached to, plus the activity it was
/// attached under.
///
/// Kept as a single struct so `window` and `activity` can never be half-set: a
/// rule anchor is either dangling ([`BookmarkAnchor::Rule`] with `attached:
/// None`) or fully attached (both fields present).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowAttachment<Id> {
    window: Id,
    activity: ActivityId,
}

/// What a bookmark points at.
///
/// A [`BookmarkAnchor::Window`] pins a concrete window; a
/// [`BookmarkAnchor::Rule`] pins a durable matcher that attaches to the first
/// matching window mapped after it was created and dangles (retaining its slot,
/// id, name, and key) when that window closes, re-attaching to the next match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BookmarkAnchor<Id> {
    /// A concrete window plus the activity it was bookmarked under. The activity
    /// is carried context for restore (which activity to switch into), not part
    /// of the bookmark's identity — one window has at most one bookmark.
    Window { window: Id, activity: ActivityId },
    /// A rule that attaches to a matching window on map. `attached: None` is the
    /// dangling state (no live window yet, or the previous one closed); the slot
    /// survives so the bookmark's id, name, key, and position persist across the
    /// dangle.
    Rule {
        rule: BookmarkRule,
        attached: Option<WindowAttachment<Id>>,
    },
}

impl<Id> BookmarkAnchor<Id> {
    /// The anchored window, or `None` for a dangling rule anchor.
    pub(crate) fn window(&self) -> Option<&Id> {
        match self {
            BookmarkAnchor::Window { window, .. } => Some(window),
            BookmarkAnchor::Rule { attached, .. } => attached.as_ref().map(|a| &a.window),
        }
    }

    /// The `(window, activity)` pair for an attached anchor, or `None` while
    /// dangling.
    ///
    /// A single destructure of both halves together, so a caller that wants
    /// the pair can't observe (or need to `expect`-justify) the impossible
    /// half-set state that two independent single-field accessors would each
    /// leave implicit by returning their own `Option`s.
    pub(crate) fn attachment(&self) -> Option<(&Id, ActivityId)> {
        match self {
            BookmarkAnchor::Window { window, activity } => Some((window, *activity)),
            BookmarkAnchor::Rule { attached, .. } => {
                attached.as_ref().map(|a| (&a.window, a.activity))
            }
        }
    }

    /// The wire-boundary split of this anchor: attached (window + activity,
    /// plus the rule if this was a rule anchor — a rule can be attached and
    /// still report its matcher) or dangling.
    ///
    /// Unlike [`Self::attachment`] paired with a separate [`Self::rule`] call,
    /// this is a single match over the anchor, so the "a windowless entry
    /// always carries a rule" wire invariant holds by construction:
    /// [`AnchorWire::DanglingRule`] carries `&BookmarkRule` directly, not an
    /// `Option`, because only a dangling [`BookmarkAnchor::Rule`] can lack a
    /// window and that arm is the only one this variant is built from.
    pub(crate) fn wire(&self) -> AnchorWire<'_, Id> {
        match self {
            BookmarkAnchor::Window { window, activity } => AnchorWire::Attached {
                window,
                activity: *activity,
                rule: None,
            },
            BookmarkAnchor::Rule {
                rule,
                attached: Some(a),
            } => AnchorWire::Attached {
                window: &a.window,
                activity: a.activity,
                rule: Some(rule),
            },
            BookmarkAnchor::Rule {
                rule,
                attached: None,
            } => AnchorWire::DanglingRule(rule),
        }
    }
}

/// The wire-boundary split of a [`BookmarkAnchor`], returned by
/// [`BookmarkAnchor::wire`].
pub(crate) enum AnchorWire<'a, Id> {
    /// The anchor has a live window (a plain window anchor, or an attached
    /// rule anchor). `rule` is `Some` only for the latter.
    Attached {
        window: &'a Id,
        activity: ActivityId,
        rule: Option<&'a BookmarkRule>,
    },
    /// The anchor is a dangling rule: no window, but the rule that will
    /// re-attach it is always present.
    DanglingRule(&'a BookmarkRule),
}

/// One bookmark: a stable id, an anchor, an optional display name, and an
/// optional dynamic keybind.
///
/// `name` starts `None` at add; it is set or cleared only via
/// [`Bookmarks::rename`]. It is a display label, never a resolution handle —
/// [`BookmarkId`] stays the sole address.
///
/// `key` always starts `None` at add; it is armed only via
/// [`Bookmarks::assign_key`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bookmark<Id> {
    id: BookmarkId,
    anchor: BookmarkAnchor<Id>,
    name: Option<BookmarkName>,
    key: Option<BookmarkKey>,
}

impl<Id> Bookmark<Id> {
    /// The bookmark's stable id.
    pub(crate) fn id(&self) -> BookmarkId {
        self.id
    }

    /// The bookmark's anchor.
    pub(crate) fn anchor(&self) -> &BookmarkAnchor<Id> {
        &self.anchor
    }

    /// The bookmark's display name, if set.
    pub(crate) fn name(&self) -> Option<&BookmarkName> {
        self.name.as_ref()
    }

    /// The bookmark's dynamic keybind, if assigned.
    pub(crate) fn key(&self) -> Option<BookmarkKey> {
        self.key
    }
}

/// A list index validated as the landing spot of a walk step.
///
/// Minted only by [`Bookmarks::walk_target`], which is the sole place that
/// knows the index is in bounds; [`Bookmarks::commit_walk`] trusts it without
/// re-checking. The field is private so no caller can construct one from an
/// arbitrary `usize` and reopen the bounds gap this type exists to close.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WalkTarget(usize);

impl WalkTarget {
    /// The wrapped index, for callers (tests) that need to inspect it.
    pub(crate) fn index(self) -> usize {
        self.0
    }
}

/// Direction of a bookmark walk. Forward steps toward the end of the list,
/// backward toward the start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WalkDirection {
    /// Toward higher indices; from no cursor the first step lands on index 0.
    Forward,
    /// Toward lower indices; from no cursor the first step lands on the last
    /// index.
    Backward,
}

/// Outcome of [`Bookmarks::add_or_repress`].
#[must_use]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddOutcome {
    /// A new bookmark was appended.
    Added(BookmarkId),
    /// The window was already bookmarked; the existing bookmark moved to front.
    MovedToFront,
    /// The window was already bookmarked and already at the front — no change.
    AlreadyFront,
    /// The window was already bookmarked and [`RepressPolicy::Remove`] is
    /// configured: nothing was mutated. The caller must show a confirmation
    /// prompt and, on confirm, remove the bookmark via [`Bookmarks::remove_by_id`]
    /// — there is exactly one removal code path.
    RemovalNeedsConfirm(BookmarkId),
}

/// Outcome of [`Bookmarks::move_to_pos`].
#[must_use]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveOutcome {
    /// The bookmark moved to a new position.
    Moved,
    /// The bookmark was already at the (clamped) target position.
    SamePosition,
    /// No bookmark with that id.
    NotFound,
}

/// What a walk or jump did, for the `Layout` caller to translate into post-jump
/// side effects (cursor warp, redraw, inhibitor refresh).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BookmarkJumpOutcome {
    /// Nothing to jump to: empty list, or a boundary with wrap disabled.
    Noop,
    /// A jump landed on a bookmark.
    Jumped {
        /// Whether the restore had to switch the active activity.
        switched_activity: bool,
    },
}

/// The curated bookmark list plus its walk cursor and focus-observation memory.
///
/// Invariants (upheld by every mutator here; mirrored at the `Layout` level by
/// `verify_invariants`): every [`BookmarkId`] in `list` is unique; at most one
/// bookmark anchors any given window; `next_id` is strictly greater than every
/// listed id; when `walk_cursor` is `Some(i)`, `i < list.len()`; no two
/// bookmarks hold an equal [`BookmarkKey`].
#[derive(Debug)]
pub struct Bookmarks<Id: PartialEq + Clone> {
    /// The bookmarks, in presentation order.
    list: Vec<Bookmark<Id>>,
    /// Index into `list` of the current walk position, or `None` when not
    /// walking. Healed on any focus change (see [`Self::observe_focus`]).
    walk_cursor: Option<usize>,
    /// Window to return to after a keybind-driven jump landed on an
    /// already-focused bookmark. Armed by [`Self::commit_key_jump`]; cleared by
    /// [`Self::commit_walk`], [`Self::commit_jump`], [`Self::commit_return`],
    /// [`Self::observe_focus`], and [`Self::prune_window`] — any focus change
    /// or ordinary jump invalidates the bounce target.
    return_target: Option<Id>,
    /// Next id to mint. Only ever grows; a retired id is never reused.
    next_id: u64,
    /// The window the focus hook last recorded. The walk-filter: [`Self::commit_walk`]
    /// and jump-commit set this synchronously to the landed window so the focus
    /// hook sees no delta and a walk never resets its own cursor or triggers MRU.
    last_seen_focus: Option<Id>,
    /// Monotonic counter, bumped whenever the id→key mapping changes
    /// (assign, unassign, or a keyed entry's removal/prune). The `Niri`-side
    /// synthetic-bind mirror rebuilds whenever this differs from its cached
    /// value — see `State::refresh`.
    binds_epoch: u64,
}

// Manual impl: deriving `Default` would demand `Id: Default`, which the layout
// window id type does not provide and does not need (the list starts empty).
impl<Id: PartialEq + Clone> Default for Bookmarks<Id> {
    fn default() -> Self {
        Self {
            list: Vec::new(),
            walk_cursor: None,
            return_target: None,
            next_id: 0,
            last_seen_focus: None,
            binds_epoch: 0,
        }
    }
}

/// Why [`Bookmarks::assign_key`] rejected a key assignment.
#[must_use]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignKeyError {
    /// No bookmark with the given id.
    NotFound,
    /// The key already belongs to a different bookmark.
    Collision,
}

impl<Id: PartialEq + Clone> Bookmarks<Id> {
    /// Mint the next id, growing the counter.
    fn mint_id(&mut self) -> BookmarkId {
        let id = BookmarkId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Bump `binds_epoch`. The sole mutation site for the counter, so every
    /// id→key mapping change is greppable from one place.
    fn touch_binds_epoch(&mut self) {
        self.binds_epoch = self.binds_epoch.wrapping_add(1);
    }

    /// Position of the bookmark anchoring `window`, if any. Matches both a
    /// `Window` anchor and an attached `Rule` anchor; dangling rule anchors
    /// (no live window) never match.
    fn position_of_window(&self, window: &Id) -> Option<usize> {
        self.list
            .iter()
            .position(|b| b.anchor.window() == Some(window))
    }

    /// Append a dangling rule-anchored bookmark, minting a fresh id. The rule
    /// attaches to a window later via [`Self::attach_first_matching`].
    pub fn add_rule(&mut self, rule: BookmarkRule) -> BookmarkId {
        let id = self.mint_id();
        self.list.push(Bookmark {
            id,
            anchor: BookmarkAnchor::Rule {
                rule,
                attached: None,
            },
            name: None,
            key: None,
        });
        id
    }

    /// Attach the first dangling rule whose matcher accepts `window` to it, in
    /// list order (list order is priority). Returns the attached bookmark's id,
    /// or `None` if no dangling rule matched.
    ///
    /// No-op — returns `None` — when `window` is already bookmarked (a `Window`
    /// anchor or an already-attached rule): one window has at most one bookmark.
    /// Later matching rules stay dangling.
    pub fn attach_first_matching(
        &mut self,
        window: Id,
        activity: ActivityId,
        app_id: Option<&str>,
        title: Option<&str>,
    ) -> Option<BookmarkId> {
        if self.position_of_window(&window).is_some() {
            return None;
        }
        let pos = self.list.iter().position(|b| {
            matches!(
                &b.anchor,
                BookmarkAnchor::Rule { rule, attached }
                    if attached.is_none() && rule.matches(app_id, title)
            )
        })?;
        let id = self.list[pos].id;
        let BookmarkAnchor::Rule { attached, .. } = &mut self.list[pos].anchor else {
            unreachable!("position selected a dangling Rule anchor");
        };
        *attached = Some(WindowAttachment { window, activity });
        Some(id)
    }

    /// Bookmark a window, or re-press an already-bookmarked one.
    ///
    /// A window has at most one bookmark (the window is the identity; the
    /// activity is not). When the window is already bookmarked, `policy` decides
    /// what re-pressing does; otherwise a fresh bookmark is appended to the end
    /// (a pruned slot is never reclaimed, so ids and positions stay monotonic).
    pub fn add_or_repress(
        &mut self,
        window: Id,
        activity: ActivityId,
        policy: RepressPolicy,
    ) -> AddOutcome {
        if let Some(pos) = self.position_of_window(&window) {
            match policy {
                RepressPolicy::MoveToFront => {
                    if pos == 0 {
                        AddOutcome::AlreadyFront
                    } else {
                        let bm = self.list.remove(pos);
                        self.list.insert(0, bm);
                        AddOutcome::MovedToFront
                    }
                }
                RepressPolicy::Remove => AddOutcome::RemovalNeedsConfirm(self.list[pos].id),
            }
        } else {
            let id = self.mint_id();
            self.list.push(Bookmark {
                id,
                anchor: BookmarkAnchor::Window { window, activity },
                name: None,
                key: None,
            });
            AddOutcome::Added(id)
        }
    }

    /// Remove the bookmark with `id`, returning it. `None` if no such bookmark.
    ///
    /// Compacts the list and adjusts `walk_cursor`: a removal strictly before
    /// the cursor decrements it; removing the cursor's own entry (or emptying
    /// the list) returns the cursor to `None`.
    pub fn remove_by_id(&mut self, id: BookmarkId) -> Option<Bookmark<Id>> {
        let pos = self.list.iter().position(|b| b.id == id)?;
        let removed = self.list.remove(pos);
        self.walk_cursor = match self.walk_cursor {
            Some(c) if pos < c => Some(c - 1),
            Some(c) if pos == c => None,
            other => other,
        };
        // Clamp for the emptied / now-out-of-bounds case.
        if let Some(c) = self.walk_cursor {
            if c >= self.list.len() {
                self.walk_cursor = None;
            }
        }
        if removed.key.is_some() {
            self.touch_binds_epoch();
        }
        Some(removed)
    }

    /// Assign `key` as the dynamic keybind for the bookmark with `id`.
    ///
    /// Re-assigning a bookmark's own current key is idempotent (`Ok`, no
    /// epoch bump). Assigning a key already held by a *different* bookmark is
    /// [`AssignKeyError::Collision`]. An unknown id is
    /// [`AssignKeyError::NotFound`].
    pub fn assign_key(&mut self, id: BookmarkId, key: BookmarkKey) -> Result<(), AssignKeyError> {
        let pos = self
            .list
            .iter()
            .position(|b| b.id == id)
            .ok_or(AssignKeyError::NotFound)?;
        if self.list[pos].key == Some(key) {
            return Ok(());
        }
        if self.list.iter().any(|b| b.id != id && b.key == Some(key)) {
            return Err(AssignKeyError::Collision);
        }
        self.list[pos].key = Some(key);
        self.touch_binds_epoch();
        Ok(())
    }

    /// Clear the dynamic keybind for the bookmark with `id`, if any.
    ///
    /// A bookmark with no assigned key is a boundary no-op (`Ok`, no epoch
    /// bump). An unknown id is [`AssignKeyError::NotFound`].
    pub fn unassign_key(&mut self, id: BookmarkId) -> Result<(), AssignKeyError> {
        let pos = self
            .list
            .iter()
            .position(|b| b.id == id)
            .ok_or(AssignKeyError::NotFound)?;
        if self.list[pos].key.take().is_some() {
            self.touch_binds_epoch();
        }
        Ok(())
    }

    /// Set or clear the display name of the bookmark with `id`. Returns
    /// `false` if no bookmark has that id.
    ///
    /// Does not bump `binds_epoch` (that counter tracks id→key mappings
    /// only) and does not touch `walk_cursor`, list order, or
    /// `return_target`.
    pub fn rename(&mut self, id: BookmarkId, name: Option<BookmarkName>) -> bool {
        let Some(bm) = self.list.iter_mut().find(|b| b.id == id) else {
            return false;
        };
        bm.name = name;
        true
    }

    /// Move the bookmark with `id` to `pos`, clamping `pos` to the last index.
    /// A move to the current position is a no-op.
    pub fn move_to_pos(&mut self, id: BookmarkId, pos: usize) -> MoveOutcome {
        let Some(cur) = self.list.iter().position(|b| b.id == id) else {
            return MoveOutcome::NotFound;
        };
        // `list` is non-empty (we just found `cur`), so `len - 1` is valid.
        let target = pos.min(self.list.len() - 1);
        if target == cur {
            return MoveOutcome::SamePosition;
        }
        let bm = self.list.remove(cur);
        self.list.insert(target, bm);
        MoveOutcome::Moved
    }

    /// The index a step in `direction` would land on, or `None` at a boundary
    /// (with wrap disabled) or for an empty list. Pure — does not mutate.
    ///
    /// If `walk_cursor` is a live continuation — `Some(i)`, `i` in bounds, and
    /// `list[i]` anchors the currently-focused window — the step continues from
    /// `i`. Otherwise the base is the focused window's own bookmark position (if
    /// it has one); failing that there is no current position, and the first
    /// step lands directly on the boundary entry (last for backward, first for
    /// forward). This stale-cursor guard makes correctness independent of
    /// refresh timing.
    ///
    /// Dangling rule anchors (no live window) are transparent to the walk: the
    /// step continues past them in `direction`, honoring `wrap`, until it lands
    /// on an attached entry or exhausts the list (an all-dangling list yields
    /// `None`).
    pub fn walk_target(
        &self,
        direction: WalkDirection,
        focused: Option<&Id>,
        wrap: bool,
    ) -> Option<WalkTarget> {
        let len = self.list.len();
        if len == 0 {
            return None;
        }
        let mut candidate = match self.walk_base(focused) {
            Some(base) => step(base, direction, len, wrap)?,
            None => match direction {
                WalkDirection::Backward => len - 1,
                WalkDirection::Forward => 0,
            },
        };
        // Advance past dangling entries. Bounded by `len` so an all-dangling
        // list terminates rather than looping under wrap.
        for _ in 0..len {
            if self.list[candidate].anchor.window().is_some() {
                return Some(WalkTarget(candidate));
            }
            candidate = step(candidate, direction, len, wrap)?;
        }
        None
    }

    /// The base index a walk steps from, or `None` when there is no current
    /// position (first step lands on a boundary).
    fn walk_base(&self, focused: Option<&Id>) -> Option<usize> {
        if let Some(i) = self.walk_cursor {
            if i < self.list.len() && self.list[i].anchor.window() == focused {
                return Some(i);
            }
        }
        focused.and_then(|w| self.position_of_window(w))
    }

    /// Commit a walk onto `target`: park the cursor there, record the landed
    /// window as the last-seen focus, and clear the return-to-previous target
    /// (a walk is a focus change like any other, so any pending bounce is
    /// invalidated).
    ///
    /// Recording the focus synchronously is the walk-filter: the focus hook then
    /// sees no delta when this window becomes focused, so a walk never resets its
    /// own cursor and never triggers MRU promotion. That same synchronous update
    /// is why the return-target clear must live here rather than relying on
    /// `observe_focus`: the walk-filter suppresses the hook for this window.
    pub fn commit_walk(&mut self, target: WalkTarget) {
        let target = target.0;
        debug_assert!(
            target < self.list.len(),
            "walk target is minted only from a validated list index"
        );
        let window = self.list[target]
            .anchor
            .window()
            .expect("walk_target skips dangling entries, so the target is attached")
            .clone();
        self.walk_cursor = Some(target);
        self.last_seen_focus = Some(window);
        self.return_target = None;
    }

    /// Commit a jump onto `window`: clear the walk cursor, record the landed
    /// window as the last-seen focus, clear the return-to-previous target, and
    /// — under [`OrderMode::Mru`] — promote the bookmark to the front. A jump
    /// *is* an activation; recording the focus synchronously keeps the focus
    /// hook from double-promoting it (and, per `commit_walk`, from re-clearing
    /// the return target it already cleared here).
    pub fn commit_jump(&mut self, window: &Id, order: OrderMode) {
        self.walk_cursor = None;
        self.last_seen_focus = Some(window.clone());
        self.return_target = None;
        if order == OrderMode::Mru {
            self.promote_to_front(window);
        }
    }

    /// Commit a keybind-driven jump onto `window`, arming `return_target =
    /// left` so a subsequent keybind jump onto the now-focused `window` can
    /// bounce back to `left`.
    ///
    /// Otherwise identical to [`Self::commit_jump`]: clears the walk cursor,
    /// records the landed window as the last-seen focus, and promotes under
    /// [`OrderMode::Mru`].
    pub fn commit_key_jump(&mut self, window: &Id, left: Option<Id>, order: OrderMode) {
        self.walk_cursor = None;
        self.last_seen_focus = Some(window.clone());
        self.return_target = left;
        if order == OrderMode::Mru {
            self.promote_to_front(window);
        }
    }

    /// Commit a return-to-previous bounce onto `window`: clear the
    /// return-target slot, record the landed window as the last-seen focus,
    /// and — under [`OrderMode::Mru`] — promote the bookmark to the front (a
    /// no-op if `window` is not itself bookmarked).
    pub fn commit_return(&mut self, window: &Id, order: OrderMode) {
        self.return_target = None;
        self.last_seen_focus = Some(window.clone());
        if order == OrderMode::Mru {
            self.promote_to_front(window);
        }
    }

    /// Observe a focus change. Returns whether anything mutated.
    ///
    /// A no-op when `current` matches the last-seen focus (the walk-filter path).
    /// Otherwise it records the new focus, resets the walk cursor, clears the
    /// return target, and — under [`OrderMode::Mru`] — promotes the newly-focused
    /// window's bookmark to the front.
    pub fn observe_focus(&mut self, current: Option<&Id>, order: OrderMode) -> bool {
        if current == self.last_seen_focus.as_ref() {
            return false;
        }
        self.last_seen_focus = current.cloned();
        self.walk_cursor = None;
        self.return_target = None;
        if order == OrderMode::Mru {
            if let Some(w) = current {
                self.promote_to_front(w);
            }
        }
        true
    }

    /// Move the bookmark anchoring `window` to the front, if it exists and is not
    /// already there.
    fn promote_to_front(&mut self, window: &Id) {
        if let Some(pos) = self.position_of_window(window) {
            if pos != 0 {
                let bm = self.list.remove(pos);
                self.list.insert(0, bm);
            }
        }
    }

    /// React to `window` closing: remove a `Window` anchor pointing at it, but
    /// *dangle* an attached `Rule` anchor in place (clear its attachment while
    /// keeping the entry), and clear `return_target` if it held the closed
    /// window.
    ///
    /// A dangled rule keeps its slot — id, name, key, and list position all
    /// survive, and the walk cursor for it is left unchanged (the entry did not
    /// move). `binds_epoch` is deliberately *not* bumped for a dangle: the
    /// id→key mapping is unchanged, so the synthetic bind stays armed and
    /// re-becomes functional when the rule re-attaches.
    ///
    /// A removed `Window` anchor adjusts the cursor with the same
    /// snapshot-then-subtract discipline as [`Self::remove_by_id`]: pruning the
    /// cursor's own entry returns the cursor to `None`, and each removal
    /// strictly before the cursor decrements it.
    pub fn prune_window(&mut self, window: &Id) {
        let original_cursor = self.walk_cursor;
        let mut kept = Vec::with_capacity(self.list.len());
        let mut removed_cursor_entry = false;
        let mut removed_before_cursor = 0usize;
        let mut removed_key = false;
        for (idx, mut bm) in self.list.drain(..).enumerate() {
            let is_window_anchor_match = matches!(
                &bm.anchor,
                BookmarkAnchor::Window { window: w, .. } if w == window
            );
            let is_attached_rule_match = matches!(
                &bm.anchor,
                BookmarkAnchor::Rule { attached: Some(a), .. } if &a.window == window
            );

            if is_window_anchor_match {
                if bm.key.is_some() {
                    removed_key = true;
                }
                if let Some(c) = original_cursor {
                    if idx == c {
                        removed_cursor_entry = true;
                    } else if idx < c {
                        removed_before_cursor += 1;
                    }
                }
                // Dropped: not pushed to `kept`.
            } else if is_attached_rule_match {
                // Dangle in place: keep the slot, drop only the attachment. No
                // epoch bump (id→key mapping unchanged), no cursor adjustment
                // (the entry did not move).
                if let BookmarkAnchor::Rule { attached, .. } = &mut bm.anchor {
                    *attached = None;
                }
                kept.push(bm);
            } else {
                kept.push(bm);
            }
        }
        self.list = kept;
        if removed_key {
            self.touch_binds_epoch();
        }
        self.walk_cursor = match original_cursor {
            Some(_) if removed_cursor_entry => None,
            // The cursor's entry survived, so `c - removed_before_cursor` still
            // indexes it in the compacted list.
            Some(c) => Some(c - removed_before_cursor),
            None => None,
        };
        if self.return_target.as_ref() == Some(window) {
            self.return_target = None;
        }
    }

    /// The bookmark id anchoring `window`, if any.
    pub(crate) fn id_for_window(&self, window: &Id) -> Option<BookmarkId> {
        self.list
            .iter()
            .find(|b| b.anchor.window() == Some(window))
            .map(|b| b.id)
    }

    /// The bookmark id whose raw wire value is `raw`, if any.
    pub(crate) fn id_for_raw(&self, raw: u64) -> Option<BookmarkId> {
        self.list.iter().map(|b| b.id).find(|id| id.get() == raw)
    }

    /// The bookmark whose raw wire value is `raw`, if any.
    pub(crate) fn get_by_raw(&self, raw: u64) -> Option<&Bookmark<Id>> {
        self.list.iter().find(|b| b.id.get() == raw)
    }

    /// Read accessor for the bookmark list, in presentation order.
    ///
    /// Available crate-wide: serves `Layout::verify_invariants` (debug builds),
    /// tests, and the IPC read surface (all builds).
    pub(crate) fn list(&self) -> &[Bookmark<Id>] {
        &self.list
    }

    /// Read accessor for the current walk cursor. `None` = not walking.
    pub(crate) fn walk_cursor(&self) -> Option<usize> {
        self.walk_cursor
    }

    /// Read accessor for the return-to-previous target.
    pub(crate) fn return_target(&self) -> Option<&Id> {
        self.return_target.as_ref()
    }

    /// Read accessor for the id→key mapping epoch. The `Niri`-side synthetic
    /// bind mirror rebuilds whenever this differs from its cached value.
    pub(crate) fn binds_epoch(&self) -> u64 {
        self.binds_epoch
    }

    /// Read accessor for the focus hook's last-seen window, for tests pinning
    /// bit-identity across the walk-filter and hard-block gates.
    #[cfg(test)]
    pub(crate) fn last_seen_focus(&self) -> Option<&Id> {
        self.last_seen_focus.as_ref()
    }

    /// Read accessor for `next_id`, for the monotonicity invariant.
    pub(crate) fn next_id(&self) -> u64 {
        self.next_id
    }
}

/// Step one index in `direction` within a list of length `len`, wrapping past a
/// boundary only when `wrap`.
fn step(base: usize, direction: WalkDirection, len: usize, wrap: bool) -> Option<usize> {
    match direction {
        WalkDirection::Forward => {
            if base + 1 < len {
                Some(base + 1)
            } else if wrap {
                Some(0)
            } else {
                None
            }
        }
        WalkDirection::Backward => {
            if base > 0 {
                Some(base - 1)
            } else if wrap {
                Some(len - 1)
            } else {
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use jiji_config::Modifiers;

    use super::*;
    use crate::layout::activity::{Activities, Activity};

    // Two distinct ActivityIds for tests.
    fn two_activities() -> (ActivityId, ActivityId) {
        let mut acts = Activities::new(Activity::new_runtime("a".to_owned()));
        let a = acts.active_id();
        let beta = Activity::new_runtime("b".to_owned());
        let b = beta.id();
        acts.insert(beta);
        (a, b)
    }

    fn windows(bm: &Bookmarks<usize>) -> Vec<usize> {
        bm.list()
            .iter()
            .filter_map(|b| b.anchor().window().copied())
            .collect()
    }

    /// Test-only setup helper: park the walk cursor on `idx` via a real
    /// `walk_target` call (landing there in one step from the boundary or
    /// from the adjacent entry) rather than casting a raw index — `WalkTarget`
    /// is only ever minted by `walk_target`, and this helper keeps that true
    /// for tests too.
    fn commit_walk_at(bm: &mut Bookmarks<usize>, idx: usize) {
        let len = bm.list().len();
        let target = if idx == 0 {
            bm.walk_target(WalkDirection::Forward, None, false)
        } else if idx == len - 1 {
            bm.walk_target(WalkDirection::Backward, None, false)
        } else {
            let prev_window = *bm.list()[idx - 1]
                .anchor()
                .window()
                .expect("test helper only used with window-anchored bookmarks");
            bm.walk_target(WalkDirection::Forward, Some(&prev_window), false)
        }
        .expect("idx must be in bounds");
        assert_eq!(target.index(), idx, "test helper miscomputed the target");
        bm.commit_walk(target);
    }

    #[test]
    fn append_mints_monotonic_ids() {
        let (a, _) = two_activities();
        let mut bm = Bookmarks::<usize>::default();
        let AddOutcome::Added(id1) = bm.add_or_repress(1, a, RepressPolicy::MoveToFront) else {
            panic!("first add must append");
        };
        let AddOutcome::Added(id2) = bm.add_or_repress(2, a, RepressPolicy::MoveToFront) else {
            panic!("second add must append");
        };
        assert_eq!(id1.get(), 0);
        assert_eq!(id2.get(), 1);
        assert_eq!(windows(&bm), [1, 2], "appended to the end in order");
        assert!(bm.next_id() > id2.get(), "next_id stays ahead of every id");
    }

    #[test]
    fn repress_moves_to_front() {
        let (a, _) = two_activities();
        let mut bm = Bookmarks::<usize>::default();
        let _ = bm.add_or_repress(1, a, RepressPolicy::MoveToFront);
        let _ = bm.add_or_repress(2, a, RepressPolicy::MoveToFront);
        let _ = bm.add_or_repress(3, a, RepressPolicy::MoveToFront);
        // Re-press window 3 (currently last) → to front.
        let out = bm.add_or_repress(3, a, RepressPolicy::MoveToFront);
        assert_eq!(out, AddOutcome::MovedToFront);
        assert_eq!(windows(&bm), [3, 1, 2]);
    }

    #[test]
    fn repress_already_front_is_noop() {
        let (a, _) = two_activities();
        let mut bm = Bookmarks::<usize>::default();
        let _ = bm.add_or_repress(1, a, RepressPolicy::MoveToFront);
        let _ = bm.add_or_repress(2, a, RepressPolicy::MoveToFront);
        let out = bm.add_or_repress(1, a, RepressPolicy::MoveToFront);
        assert_eq!(out, AddOutcome::AlreadyFront);
        assert_eq!(windows(&bm), [1, 2], "order unchanged");
    }

    #[test]
    fn repress_remove_policy_needs_confirm_and_mutates_nothing() {
        let (a, _) = two_activities();
        let mut bm = Bookmarks::<usize>::default();
        let _ = bm.add_or_repress(1, a, RepressPolicy::MoveToFront);
        let _ = bm.add_or_repress(2, a, RepressPolicy::MoveToFront);
        let id2 = bm.id_for_window(&2).unwrap();
        let next_id_before = bm.next_id();

        let out = bm.add_or_repress(2, a, RepressPolicy::Remove);

        assert_eq!(out, AddOutcome::RemovalNeedsConfirm(id2));
        assert_eq!(windows(&bm), [1, 2], "list order unchanged");
        assert_eq!(bm.walk_cursor(), None, "cursor unchanged");
        assert_eq!(bm.next_id(), next_id_before, "next_id unchanged");
        assert_eq!(
            bm.id_for_window(&2),
            Some(id2),
            "the bookmark itself is untouched until the caller confirms"
        );
    }

    #[test]
    fn add_never_reclaims_pruned_slot() {
        let (a, _) = two_activities();
        let mut bm = Bookmarks::<usize>::default();
        let AddOutcome::Added(id1) = bm.add_or_repress(1, a, RepressPolicy::MoveToFront) else {
            unreachable!()
        };
        let _ = bm.add_or_repress(2, a, RepressPolicy::MoveToFront);
        bm.prune_window(&1);
        // A fresh add appends with a *new* id, never reusing id1's value.
        let AddOutcome::Added(id3) = bm.add_or_repress(3, a, RepressPolicy::MoveToFront) else {
            unreachable!()
        };
        assert!(id3.get() > id1.get(), "new id must exceed the pruned one");
        assert_eq!(windows(&bm), [2, 3], "appended to the end");
    }

    #[test]
    fn remove_by_id_adjusts_cursor() {
        let (a, _) = two_activities();
        let build = || {
            let mut bm = Bookmarks::<usize>::default();
            for w in [1, 2, 3, 4] {
                let _ = bm.add_or_repress(w, a, RepressPolicy::MoveToFront);
            }
            bm
        };

        // Remove before the cursor → cursor decrements.
        let mut bm = build();
        commit_walk_at(&mut bm, 2); // cursor on index 2 (window 3)
        let id1 = bm.id_for_window(&1).unwrap();
        bm.remove_by_id(id1);
        assert_eq!(bm.walk_cursor(), Some(1), "cursor decremented");
        assert_eq!(windows(&bm), [2, 3, 4]);

        // Remove the cursor's own entry → cursor to None.
        let mut bm = build();
        commit_walk_at(&mut bm, 2);
        let id3 = bm.id_for_window(&3).unwrap();
        bm.remove_by_id(id3);
        assert_eq!(bm.walk_cursor(), None, "cursor cleared");

        // Remove after the cursor → cursor unchanged.
        let mut bm = build();
        commit_walk_at(&mut bm, 1);
        let id4 = bm.id_for_window(&4).unwrap();
        bm.remove_by_id(id4);
        assert_eq!(bm.walk_cursor(), Some(1), "cursor unchanged");
    }

    #[test]
    fn move_to_pos_clamps_and_same_pos_noops() {
        let (a, _) = two_activities();
        let mut bm = Bookmarks::<usize>::default();
        for w in [1, 2, 3] {
            let _ = bm.add_or_repress(w, a, RepressPolicy::MoveToFront);
        }
        let id1 = bm.id_for_window(&1).unwrap();
        // Clamp: pos 99 → last index.
        assert_eq!(bm.move_to_pos(id1, 99), MoveOutcome::Moved);
        assert_eq!(windows(&bm), [2, 3, 1]);
        // Same position (window 1 now at last index) → no-op.
        let id1 = bm.id_for_window(&1).unwrap();
        assert_eq!(bm.move_to_pos(id1, 99), MoveOutcome::SamePosition);
        assert_eq!(windows(&bm), [2, 3, 1]);
        // Unknown id.
        assert_eq!(bm.move_to_pos(BookmarkId(999), 0), MoveOutcome::NotFound);
    }

    #[test]
    fn walk_init_from_focused_bookmark() {
        let (a, _) = two_activities();
        let mut bm = Bookmarks::<usize>::default();
        for w in [1, 2, 3] {
            let _ = bm.add_or_repress(w, a, RepressPolicy::MoveToFront);
        }
        // Focused on window 2 (index 1), no cursor: forward steps to index 2.
        assert_eq!(
            bm.walk_target(WalkDirection::Forward, Some(&2), false)
                .map(WalkTarget::index),
            Some(2)
        );
        // Backward from focused window 2 steps to index 0.
        assert_eq!(
            bm.walk_target(WalkDirection::Backward, Some(&2), false)
                .map(WalkTarget::index),
            Some(0)
        );
    }

    #[test]
    fn walk_init_from_ends_when_no_focused_bookmark() {
        let (a, _) = two_activities();
        let mut bm = Bookmarks::<usize>::default();
        for w in [1, 2, 3] {
            let _ = bm.add_or_repress(w, a, RepressPolicy::MoveToFront);
        }
        // Focused window is not bookmarked, no cursor: backward lands on the last
        // entry, forward on the first.
        assert_eq!(
            bm.walk_target(WalkDirection::Backward, Some(&99), false)
                .map(WalkTarget::index),
            Some(2)
        );
        assert_eq!(
            bm.walk_target(WalkDirection::Forward, Some(&99), false)
                .map(WalkTarget::index),
            Some(0)
        );
        // No focused window at all: same boundary behavior.
        assert_eq!(
            bm.walk_target(WalkDirection::Backward, None, false)
                .map(WalkTarget::index),
            Some(2)
        );
        assert_eq!(
            bm.walk_target(WalkDirection::Forward, None, false)
                .map(WalkTarget::index),
            Some(0)
        );
    }

    #[test]
    fn walk_wrap_boundary_behavior() {
        let (a, _) = two_activities();
        let mut bm = Bookmarks::<usize>::default();
        for w in [1, 2, 3] {
            let _ = bm.add_or_repress(w, a, RepressPolicy::MoveToFront);
        }
        // Cursor on last index; focused on that window.
        commit_walk_at(&mut bm, 2);
        // Forward off the end: None without wrap, index 0 with wrap.
        assert_eq!(
            bm.walk_target(WalkDirection::Forward, Some(&3), false)
                .map(WalkTarget::index),
            None
        );
        assert_eq!(
            bm.walk_target(WalkDirection::Forward, Some(&3), true)
                .map(WalkTarget::index),
            Some(0)
        );

        // Cursor on first index; focused on that window.
        commit_walk_at(&mut bm, 0);
        assert_eq!(
            bm.walk_target(WalkDirection::Backward, Some(&1), false)
                .map(WalkTarget::index),
            None
        );
        assert_eq!(
            bm.walk_target(WalkDirection::Backward, Some(&1), true)
                .map(WalkTarget::index),
            Some(2)
        );
    }

    #[test]
    fn walk_empty_list_is_none() {
        let bm = Bookmarks::<usize>::default();
        assert_eq!(
            bm.walk_target(WalkDirection::Forward, Some(&1), true)
                .map(WalkTarget::index),
            None
        );
        assert_eq!(
            bm.walk_target(WalkDirection::Backward, None, true)
                .map(WalkTarget::index),
            None
        );
    }

    #[test]
    fn stale_cursor_self_heals() {
        let (a, _) = two_activities();
        let mut bm = Bookmarks::<usize>::default();
        for w in [1, 2, 3] {
            let _ = bm.add_or_repress(w, a, RepressPolicy::MoveToFront);
        }
        // Cursor parked on index 2 (window 3), but focus is now elsewhere (window
        // 1, index 0). The cursor is stale: the walk must re-initialise from the
        // focused window's bookmark, not continue from the stale cursor.
        commit_walk_at(&mut bm, 2);
        // Focused on window 1: forward re-initialises from index 0 → index 1.
        assert_eq!(
            bm.walk_target(WalkDirection::Forward, Some(&1), false)
                .map(WalkTarget::index),
            Some(1)
        );
    }

    #[test]
    fn prune_multi_entry_cursor_snapshot_discipline() {
        let (a, _) = two_activities();
        let mut bm = Bookmarks::<usize>::default();
        for w in [1, 2, 3, 4] {
            let _ = bm.add_or_repress(w, a, RepressPolicy::MoveToFront);
        }
        // Cursor on index 2 (window 3); prune window 1 (index 0, before cursor).
        commit_walk_at(&mut bm, 2);
        bm.prune_window(&1);
        assert_eq!(windows(&bm), [2, 3, 4]);
        assert_eq!(
            bm.walk_cursor(),
            Some(1),
            "cursor decremented by one removal"
        );

        // Prune the cursor's own window → cursor to None.
        let mut bm = Bookmarks::<usize>::default();
        for w in [1, 2, 3] {
            let _ = bm.add_or_repress(w, a, RepressPolicy::MoveToFront);
        }
        commit_walk_at(&mut bm, 1); // on window 2
        bm.prune_window(&2);
        assert_eq!(bm.walk_cursor(), None);
    }

    #[test]
    fn prune_clears_matching_return_target() {
        let (a, _) = two_activities();
        let mut bm = Bookmarks::<usize>::default();
        let _ = bm.add_or_repress(1, a, RepressPolicy::MoveToFront);
        // Arm the return target directly (no public arming path yet) to pin that
        // prune clears it when the closed window matches.
        bm.return_target = Some(1);
        bm.prune_window(&1);
        assert_eq!(bm.return_target(), None, "return target cleared on prune");
    }

    #[test]
    fn observe_focus_resets_cursor_and_clears_return_target() {
        let (a, _) = two_activities();
        let mut bm = Bookmarks::<usize>::default();
        for w in [1, 2] {
            let _ = bm.add_or_repress(w, a, RepressPolicy::MoveToFront);
        }
        commit_walk_at(&mut bm, 1);
        bm.return_target = Some(2);
        // Focus moves to a genuinely new window → reset.
        let mutated = bm.observe_focus(Some(&99), OrderMode::Manual);
        assert!(mutated);
        assert_eq!(bm.walk_cursor(), None);
        assert_eq!(bm.return_target(), None);
    }

    #[test]
    fn observe_focus_mru_promotes() {
        let (a, _) = two_activities();
        let mut bm = Bookmarks::<usize>::default();
        for w in [1, 2, 3] {
            let _ = bm.add_or_repress(w, a, RepressPolicy::MoveToFront);
        }
        // Focusing window 3 (index 2) under MRU promotes it to front.
        bm.observe_focus(Some(&3), OrderMode::Mru);
        assert_eq!(windows(&bm), [3, 1, 2]);
        // Manual order leaves the list alone.
        let mut bm2 = Bookmarks::<usize>::default();
        for w in [1, 2, 3] {
            let _ = bm2.add_or_repress(w, a, RepressPolicy::MoveToFront);
        }
        bm2.observe_focus(Some(&3), OrderMode::Manual);
        assert_eq!(windows(&bm2), [1, 2, 3]);
    }

    #[test]
    fn walk_does_not_promote_under_mru() {
        let (a, _) = two_activities();
        let mut bm = Bookmarks::<usize>::default();
        for w in [1, 2, 3] {
            let _ = bm.add_or_repress(w, a, RepressPolicy::MoveToFront);
        }
        // Commit a walk onto index 2 (window 3): commit_walk records the landed
        // window as last-seen focus, so the subsequent focus observation for that
        // same window is a no-op and MRU does NOT reorder.
        commit_walk_at(&mut bm, 2);
        let mutated = bm.observe_focus(Some(&3), OrderMode::Mru);
        assert!(!mutated, "walk-driven focus is filtered, not observed");
        assert_eq!(windows(&bm), [1, 2, 3], "walk did not promote under MRU");
    }

    fn key(modifiers: Modifiers, sym: smithay::input::keyboard::Keysym) -> BookmarkKey {
        BookmarkKey::new(Key {
            trigger: Trigger::Keysym(sym),
            modifiers,
        })
        .expect("test-constructed key must be valid")
    }

    #[test]
    fn bookmark_key_rejects_non_keysym_trigger() {
        let err = BookmarkKey::new(Key {
            trigger: Trigger::MouseLeft,
            modifiers: Modifiers::SUPER,
        })
        .unwrap_err();
        assert_eq!(err, BookmarkKeyError::NotAKeysym);
    }

    #[test]
    fn bookmark_key_rejects_no_modifiers() {
        let err = BookmarkKey::new(Key {
            trigger: Trigger::Keysym(smithay::input::keyboard::Keysym::m),
            modifiers: Modifiers::empty(),
        })
        .unwrap_err();
        assert_eq!(err, BookmarkKeyError::NoModifiers);
    }

    #[test]
    fn bookmark_key_accepts_keysym_with_modifier() {
        assert!(BookmarkKey::new(Key {
            trigger: Trigger::Keysym(smithay::input::keyboard::Keysym::m),
            modifiers: Modifiers::SUPER,
        })
        .is_ok());
    }

    #[test]
    fn assign_key_rejects_unknown_id() {
        let (a, _) = two_activities();
        let mut bm = Bookmarks::<usize>::default();
        let _ = bm.add_or_repress(1, a, RepressPolicy::MoveToFront);
        let err = bm
            .assign_key(
                BookmarkId(999),
                key(Modifiers::SUPER, smithay::input::keyboard::Keysym::m),
            )
            .unwrap_err();
        assert_eq!(err, AssignKeyError::NotFound);
    }

    #[test]
    fn assign_key_rejects_collision_with_another_bookmark() {
        let (a, _) = two_activities();
        let mut bm = Bookmarks::<usize>::default();
        let AddOutcome::Added(id1) = bm.add_or_repress(1, a, RepressPolicy::MoveToFront) else {
            unreachable!()
        };
        let AddOutcome::Added(id2) = bm.add_or_repress(2, a, RepressPolicy::MoveToFront) else {
            unreachable!()
        };
        let k = key(Modifiers::SUPER, smithay::input::keyboard::Keysym::m);
        bm.assign_key(id1, k).unwrap();
        let err = bm.assign_key(id2, k).unwrap_err();
        assert_eq!(err, AssignKeyError::Collision);
    }

    #[test]
    fn assign_key_reassigning_own_current_key_is_idempotent() {
        let (a, _) = two_activities();
        let mut bm = Bookmarks::<usize>::default();
        let AddOutcome::Added(id1) = bm.add_or_repress(1, a, RepressPolicy::MoveToFront) else {
            unreachable!()
        };
        let k = key(Modifiers::SUPER, smithay::input::keyboard::Keysym::m);
        bm.assign_key(id1, k).unwrap();
        let epoch_after_first = bm.binds_epoch();
        assert!(bm.assign_key(id1, k).is_ok());
        assert_eq!(
            bm.binds_epoch(),
            epoch_after_first,
            "re-assigning the same key must not bump the epoch"
        );
    }

    #[test]
    fn assign_key_bumps_epoch_unassign_bumps_again() {
        let (a, _) = two_activities();
        let mut bm = Bookmarks::<usize>::default();
        let AddOutcome::Added(id1) = bm.add_or_repress(1, a, RepressPolicy::MoveToFront) else {
            unreachable!()
        };
        let epoch0 = bm.binds_epoch();
        bm.assign_key(
            id1,
            key(Modifiers::SUPER, smithay::input::keyboard::Keysym::m),
        )
        .unwrap();
        let epoch1 = bm.binds_epoch();
        assert_ne!(epoch0, epoch1, "assign_key must bump the epoch");
        bm.unassign_key(id1).unwrap();
        let epoch2 = bm.binds_epoch();
        assert_ne!(epoch1, epoch2, "unassign_key must bump the epoch");
    }

    #[test]
    fn unassign_key_rejects_unknown_id_and_noops_when_unset() {
        let (a, _) = two_activities();
        let mut bm = Bookmarks::<usize>::default();
        let AddOutcome::Added(id1) = bm.add_or_repress(1, a, RepressPolicy::MoveToFront) else {
            unreachable!()
        };
        assert_eq!(
            bm.unassign_key(BookmarkId(999)),
            Err(AssignKeyError::NotFound)
        );
        let epoch_before = bm.binds_epoch();
        assert!(
            bm.unassign_key(id1).is_ok(),
            "no key assigned: boundary no-op"
        );
        assert_eq!(
            bm.binds_epoch(),
            epoch_before,
            "unassigning an already-unset key must not bump the epoch"
        );
    }

    #[test]
    fn remove_by_id_bumps_epoch_only_when_keyed() {
        let (a, _) = two_activities();
        let mut bm = Bookmarks::<usize>::default();
        let AddOutcome::Added(id1) = bm.add_or_repress(1, a, RepressPolicy::MoveToFront) else {
            unreachable!()
        };
        let AddOutcome::Added(id2) = bm.add_or_repress(2, a, RepressPolicy::MoveToFront) else {
            unreachable!()
        };
        bm.assign_key(
            id1,
            key(Modifiers::SUPER, smithay::input::keyboard::Keysym::m),
        )
        .unwrap();

        let epoch0 = bm.binds_epoch();
        bm.remove_by_id(id2);
        assert_eq!(
            bm.binds_epoch(),
            epoch0,
            "removing an unkeyed entry must not bump the epoch"
        );

        bm.remove_by_id(id1);
        assert_ne!(
            bm.binds_epoch(),
            epoch0,
            "removing a keyed entry must bump the epoch"
        );
    }

    #[test]
    fn prune_window_bumps_epoch_only_when_keyed() {
        let (a, _) = two_activities();
        let mut bm = Bookmarks::<usize>::default();
        let AddOutcome::Added(id1) = bm.add_or_repress(1, a, RepressPolicy::MoveToFront) else {
            unreachable!()
        };
        let _ = bm.add_or_repress(2, a, RepressPolicy::MoveToFront);
        bm.assign_key(
            id1,
            key(Modifiers::SUPER, smithay::input::keyboard::Keysym::m),
        )
        .unwrap();

        let epoch0 = bm.binds_epoch();
        bm.prune_window(&2);
        assert_eq!(
            bm.binds_epoch(),
            epoch0,
            "pruning an unkeyed window must not bump the epoch"
        );

        bm.prune_window(&1);
        assert_ne!(
            bm.binds_epoch(),
            epoch0,
            "pruning a keyed window must bump the epoch"
        );
    }

    #[test]
    fn commit_key_jump_arms_return_target() {
        let (a, _) = two_activities();
        let mut bm = Bookmarks::<usize>::default();
        for w in [1, 2] {
            let _ = bm.add_or_repress(w, a, RepressPolicy::MoveToFront);
        }
        bm.commit_key_jump(&2, Some(1), OrderMode::Manual);
        assert_eq!(bm.return_target(), Some(&1));
    }

    #[test]
    fn commit_walk_and_commit_jump_clear_return_target() {
        let (a, _) = two_activities();
        let mut bm = Bookmarks::<usize>::default();
        for w in [1, 2, 3] {
            let _ = bm.add_or_repress(w, a, RepressPolicy::MoveToFront);
        }

        bm.commit_key_jump(&2, Some(1), OrderMode::Manual);
        assert_eq!(bm.return_target(), Some(&1));
        commit_walk_at(&mut bm, 0);
        assert_eq!(
            bm.return_target(),
            None,
            "commit_walk must clear an armed return target"
        );

        bm.commit_key_jump(&2, Some(1), OrderMode::Manual);
        assert_eq!(bm.return_target(), Some(&1));
        bm.commit_jump(&3, OrderMode::Manual);
        assert_eq!(
            bm.return_target(),
            None,
            "commit_jump must clear an armed return target"
        );
    }

    #[test]
    fn commit_return_clears_and_lands_and_promotes_under_mru() {
        let (a, _) = two_activities();
        let mut bm = Bookmarks::<usize>::default();
        for w in [1, 2, 3] {
            let _ = bm.add_or_repress(w, a, RepressPolicy::MoveToFront);
        }
        // Jumping to 2 under MRU promotes it to front and arms a bounce to 1.
        bm.commit_key_jump(&2, Some(1), OrderMode::Mru);
        assert_eq!(bm.return_target(), Some(&1));
        assert_eq!(
            windows(&bm),
            [2, 1, 3],
            "commit_key_jump promotes the landed bookmark"
        );

        // Bouncing back to 1 clears the slot and re-promotes 1 to front.
        bm.commit_return(&1, OrderMode::Mru);
        assert_eq!(bm.return_target(), None, "bounce clears the armed slot");
        assert_eq!(
            windows(&bm),
            [1, 2, 3],
            "commit_return promotes the bounce target under MRU"
        );
    }

    #[test]
    fn bookmark_name_trims_and_accepts() {
        let name = BookmarkName::new("  mail  ").expect("valid after trim");
        assert_eq!(name.as_str(), "mail");
    }

    #[test]
    fn bookmark_name_rejects_empty_after_trim() {
        assert_eq!(BookmarkName::new(""), Err(BookmarkNameError::Empty));
        assert_eq!(BookmarkName::new("   "), Err(BookmarkNameError::Empty));
    }

    #[test]
    fn bookmark_name_rejects_control_chars() {
        assert_eq!(
            BookmarkName::new("mail\nbox"),
            Err(BookmarkNameError::ControlChars)
        );
        assert_eq!(
            BookmarkName::new("mail\tbox"),
            Err(BookmarkNameError::ControlChars)
        );
    }

    #[test]
    fn rename_sets_reads_back_and_clears() {
        let (a, _) = two_activities();
        let mut bm = Bookmarks::<usize>::default();
        let AddOutcome::Added(id1) = bm.add_or_repress(1, a, RepressPolicy::MoveToFront) else {
            unreachable!()
        };
        assert_eq!(bm.list()[0].name(), None, "unnamed at add");

        let name = BookmarkName::new("mail").unwrap();
        assert!(bm.rename(id1, Some(name.clone())));
        assert_eq!(bm.list()[0].name().map(BookmarkName::as_str), Some("mail"));

        assert!(bm.rename(id1, None), "clearing is a rename to None");
        assert_eq!(bm.list()[0].name(), None, "cleared");
    }

    #[test]
    fn rename_unknown_id_returns_false() {
        let mut bm = Bookmarks::<usize>::default();
        let name = BookmarkName::new("mail").unwrap();
        assert!(!bm.rename(BookmarkId(999), Some(name)));
    }

    #[test]
    fn rename_survives_reorder_and_does_not_touch_epoch_cursor_or_order() {
        let (a, _) = two_activities();
        let mut bm = Bookmarks::<usize>::default();
        for w in [1, 2, 3] {
            let _ = bm.add_or_repress(w, a, RepressPolicy::MoveToFront);
        }
        let id1 = bm.id_for_window(&1).unwrap();
        let name = BookmarkName::new("mail").unwrap();
        let epoch_before = bm.binds_epoch();
        let cursor_before = bm.walk_cursor();
        assert!(bm.rename(id1, Some(name)));
        assert_eq!(bm.binds_epoch(), epoch_before, "rename must not bump epoch");
        assert_eq!(
            bm.walk_cursor(),
            cursor_before,
            "rename must not touch cursor"
        );
        assert_eq!(windows(&bm), [1, 2, 3], "rename must not touch order");

        // Reorder, then confirm the name is still keyed by id, not position.
        let _ = bm.move_to_pos(id1, 99);
        assert_eq!(windows(&bm), [2, 3, 1]);
        assert_eq!(
            bm.id_for_window(&1)
                .and_then(|id| bm.list().iter().find(|b| b.id() == id))
                .and_then(|b| b.name())
                .map(BookmarkName::as_str),
            Some("mail"),
            "name survives reorder, keyed by id"
        );
    }

    // --- Rule-anchored bookmarks ---

    fn rule(app_id: Option<&str>, title: Option<&str>) -> BookmarkRule {
        BookmarkRule::new(
            app_id.map(|s| s.parse().expect("valid test regex")),
            title.map(|s| s.parse().expect("valid test regex")),
        )
        .expect("at least one field given")
    }

    fn is_dangling(bm: &Bookmarks<usize>, id: BookmarkId) -> bool {
        bm.list()
            .iter()
            .find(|b| b.id() == id)
            .expect("bookmark exists")
            .anchor()
            .window()
            .is_none()
    }

    #[test]
    fn rule_ctor_rejects_zero_fields() {
        assert_eq!(
            BookmarkRule::new(None, None),
            Err(BookmarkRuleError::Empty),
            "a rule with no fields matches nothing and is rejected",
        );
        assert!(BookmarkRule::new(Some("^x$".parse().unwrap()), None).is_ok());
    }

    #[test]
    fn add_rule_appends_dangling_with_fresh_id() {
        let mut bm = Bookmarks::<usize>::default();
        let id = bm.add_rule(rule(Some("^firefox$"), None));
        assert_eq!(id.get(), 0, "fresh id minted");
        assert!(bm.next_id() > id.get(), "next_id stays ahead");
        assert_eq!(bm.list().len(), 1);
        assert!(is_dangling(&bm, id), "a fresh rule starts dangling");
    }

    #[test]
    fn attach_first_matching_honors_list_order_priority() {
        let (a, _) = two_activities();
        let mut bm = Bookmarks::<usize>::default();
        // Two dangling rules both match window 1; the earlier one wins.
        let first = bm.add_rule(rule(Some("^app$"), None));
        let second = bm.add_rule(rule(Some("^app$"), None));
        let attached = bm.attach_first_matching(1, a, Some("app"), None);
        assert_eq!(attached, Some(first), "list order is priority");
        assert!(!is_dangling(&bm, first), "the winner attached");
        assert!(is_dangling(&bm, second), "the loser stays dangling");
    }

    #[test]
    fn attach_skips_already_bookmarked_window() {
        let (a, _) = two_activities();
        let mut bm = Bookmarks::<usize>::default();
        let _ = bm.add_or_repress(1, a, RepressPolicy::MoveToFront);
        let rid = bm.add_rule(rule(Some("^app$"), None));
        // Window 1 is already bookmarked (a Window anchor), so no attach.
        assert_eq!(
            bm.attach_first_matching(1, a, Some("app"), None),
            None,
            "one window, one bookmark",
        );
        assert!(is_dangling(&bm, rid), "the rule stays dangling");
    }

    #[test]
    fn attach_sets_window_and_activity_together() {
        let (_, b) = two_activities();
        let mut bm = Bookmarks::<usize>::default();
        let rid = bm.add_rule(rule(Some("^app$"), None));
        assert_eq!(bm.attach_first_matching(7, b, Some("app"), None), Some(rid));
        let anchor = bm.list()[0].anchor();
        assert_eq!(anchor.window(), Some(&7));
        assert_eq!(anchor.attachment().map(|(_, activity)| activity), Some(b));
    }

    #[test]
    fn matching_is_and_over_present_fields_on_app_id_and_raw_title() {
        let r = rule(Some("^fire"), Some("tab$"));
        assert!(r.matches(Some("firefox"), Some("mytab")), "both match");
        assert!(
            !r.matches(Some("firefox"), Some("nope")),
            "title must match"
        );
        assert!(
            !r.matches(Some("chromium"), Some("mytab")),
            "app-id must match"
        );
        // A present rule field against an absent window field is a non-match.
        assert!(!r.matches(None, Some("mytab")));
        assert!(!r.matches(Some("firefox"), None));
        // App-id-only rule ignores the title entirely.
        let r = rule(Some("^fire"), None);
        assert!(r.matches(Some("firefox"), None));
        assert!(r.matches(Some("firefox"), Some("anything")));
    }

    #[test]
    fn prune_dangles_rule_in_place_keeping_slot_and_epoch() {
        let (a, _) = two_activities();
        let mut bm = Bookmarks::<usize>::default();
        let rid = bm.add_rule(rule(Some("^app$"), None));
        assert_eq!(bm.attach_first_matching(1, a, Some("app"), None), Some(rid));
        let name = BookmarkName::new("mail").unwrap();
        assert!(bm.rename(rid, Some(name)));
        bm.assign_key(
            rid,
            key(Modifiers::SUPER, smithay::input::keyboard::Keysym::m),
        )
        .unwrap();
        let epoch_before = bm.binds_epoch();

        bm.prune_window(&1);

        assert_eq!(bm.list().len(), 1, "the slot survives the dangle");
        assert!(is_dangling(&bm, rid), "the rule dangled in place");
        assert_eq!(
            bm.list()[0].name().map(BookmarkName::as_str),
            Some("mail"),
            "name survives the dangle",
        );
        assert!(bm.list()[0].key().is_some(), "key survives the dangle");
        assert_eq!(
            bm.binds_epoch(),
            epoch_before,
            "a dangle keeps the id→key mapping, so no epoch bump",
        );

        // Re-attach reuses the same id and re-becomes functional.
        assert_eq!(
            bm.attach_first_matching(2, a, Some("app"), None),
            Some(rid),
            "re-attach reuses the same bookmark id",
        );
        assert!(!is_dangling(&bm, rid));
    }

    #[test]
    fn prune_still_removes_window_anchors() {
        let (a, _) = two_activities();
        let mut bm = Bookmarks::<usize>::default();
        let _ = bm.add_or_repress(1, a, RepressPolicy::MoveToFront);
        let _ = bm.add_or_repress(2, a, RepressPolicy::MoveToFront);
        bm.prune_window(&1);
        assert_eq!(windows(&bm), [2], "a Window anchor is removed entirely");
        assert_eq!(bm.list().len(), 1);
    }

    #[test]
    fn prune_cursor_unchanged_for_dangled_rule_but_decremented_for_removed_window() {
        let (a, _) = two_activities();

        // A rule entry before the cursor dangling in place leaves the cursor put.
        let mut bm = Bookmarks::<usize>::default();
        let r0 = bm.add_rule(rule(Some("^app$"), None));
        let _ = bm.add_or_repress(2, a, RepressPolicy::MoveToFront);
        let _ = bm.add_or_repress(3, a, RepressPolicy::MoveToFront);
        assert_eq!(bm.attach_first_matching(1, a, Some("app"), None), Some(r0));
        // Layout: [rule→1, win 2, win 3]. Cursor on index 2 (window 3).
        commit_walk_at(&mut bm, 2);
        bm.prune_window(&1);
        assert_eq!(
            bm.walk_cursor(),
            Some(2),
            "dangling a rule in place keeps every entry, so the cursor is unchanged",
        );

        // A Window anchor before the cursor being removed decrements the cursor.
        let mut bm = Bookmarks::<usize>::default();
        for w in [1, 2, 3] {
            let _ = bm.add_or_repress(w, a, RepressPolicy::MoveToFront);
        }
        commit_walk_at(&mut bm, 2);
        bm.prune_window(&1);
        assert_eq!(
            bm.walk_cursor(),
            Some(1),
            "removing a Window anchor before the cursor decrements it",
        );
    }

    #[test]
    fn walk_skips_dangling_entries_including_wrap() {
        let (a, _) = two_activities();
        let mut bm = Bookmarks::<usize>::default();
        // Layout: [win 1, dangling rule, win 3].
        let _ = bm.add_or_repress(1, a, RepressPolicy::MoveToFront);
        let _ = bm.add_rule(rule(Some("^never$"), None));
        let _ = bm.add_or_repress(3, a, RepressPolicy::MoveToFront);

        // Forward from window 1 (index 0) steps past the dangling index 1 to index 2.
        assert_eq!(
            bm.walk_target(WalkDirection::Forward, Some(&1), false)
                .map(WalkTarget::index),
            Some(2),
        );
        // Backward from window 3 (index 2) skips index 1 down to index 0.
        assert_eq!(
            bm.walk_target(WalkDirection::Backward, Some(&3), false)
                .map(WalkTarget::index),
            Some(0),
        );
        // Forward off the end from window 3 with wrap lands on window 1 (index 0),
        // skipping the dangling entry on the way.
        commit_walk_at(&mut bm, 2);
        assert_eq!(
            bm.walk_target(WalkDirection::Forward, Some(&3), true)
                .map(WalkTarget::index),
            Some(0),
        );
    }

    #[test]
    fn all_dangling_walk_is_none() {
        let mut bm = Bookmarks::<usize>::default();
        let _ = bm.add_rule(rule(Some("^a$"), None));
        let _ = bm.add_rule(rule(Some("^b$"), None));
        assert_eq!(
            bm.walk_target(WalkDirection::Forward, None, true)
                .map(WalkTarget::index),
            None,
            "an all-dangling list has no walk target even with wrap",
        );
        assert_eq!(
            bm.walk_target(WalkDirection::Backward, None, false)
                .map(WalkTarget::index),
            None,
        );
    }

    #[test]
    fn repress_on_rule_attached_window_applies_policy() {
        let (a, _) = two_activities();
        let mut bm = Bookmarks::<usize>::default();
        let _ = bm.add_or_repress(9, a, RepressPolicy::MoveToFront);
        let rid = bm.add_rule(rule(Some("^app$"), None));
        assert_eq!(bm.attach_first_matching(1, a, Some("app"), None), Some(rid));
        // Layout: [win 9, rule→1]. Re-press window 1 under MoveToFront promotes it.
        let out = bm.add_or_repress(1, a, RepressPolicy::MoveToFront);
        assert_eq!(out, AddOutcome::MovedToFront);
        assert_eq!(windows(&bm), [1, 9], "the rule bookmark moved to front");

        // Under Remove, re-press asks for confirmation naming the rule bookmark's id.
        let out = bm.add_or_repress(1, a, RepressPolicy::Remove);
        assert_eq!(out, AddOutcome::RemovalNeedsConfirm(rid));
        // Confirming destroys the rule entirely.
        assert!(bm.remove_by_id(rid).is_some());
        assert!(
            bm.list().iter().all(|b| b.id() != rid),
            "removing a rule bookmark kills the rule",
        );
    }
}
