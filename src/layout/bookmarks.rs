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

use jiji_config::{OrderMode, RepressPolicy};

use super::activity::ActivityId;

/// Stable identity of a bookmark, minted monotonically and never reused.
///
/// A pruned bookmark's id is retired for good: [`Bookmarks::next_id`] only ever
/// grows, so a fresh add never collides with a stale id a client may still hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BookmarkId(u64);

impl BookmarkId {
    /// The raw wire value, as surfaced over IPC and accepted back on dispatch.
    pub fn get(self) -> u64 {
        self.0
    }
}

/// What a bookmark points at.
///
/// Only the window anchor exists today: a bookmark pins a concrete window under
/// the activity it was created in. Future anchor kinds (a rule that re-resolves
/// to whatever window currently matches, or a dangling anchor whose window has
/// closed but whose slot is retained) would join as sibling variants; keeping
/// this an enum from the outset means adding one does not disturb the window
/// case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BookmarkAnchor<Id> {
    /// A concrete window plus the activity it was bookmarked under. The activity
    /// is carried context for restore (which activity to switch into), not part
    /// of the bookmark's identity — one window has at most one bookmark.
    Window { window: Id, activity: ActivityId },
}

impl<Id> BookmarkAnchor<Id> {
    /// The anchored window.
    pub(crate) fn window(&self) -> &Id {
        match self {
            BookmarkAnchor::Window { window, .. } => window,
        }
    }

    /// The activity the bookmark was created under.
    pub(crate) fn activity(&self) -> ActivityId {
        match self {
            BookmarkAnchor::Window { activity, .. } => *activity,
        }
    }
}

/// One bookmark: a stable id, an anchor, and a reserved name slot.
///
/// `name` is always `None` today. A user-facing rename surface is a later
/// addition; reserving the field now keeps the struct shape stable so adding
/// the surface does not migrate stored bookmarks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bookmark<Id> {
    id: BookmarkId,
    anchor: BookmarkAnchor<Id>,
    name: Option<String>,
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
/// listed id; when `walk_cursor` is `Some(i)`, `i < list.len()`.
#[derive(Debug)]
pub struct Bookmarks<Id: PartialEq + Clone> {
    /// The bookmarks, in presentation order.
    list: Vec<Bookmark<Id>>,
    /// Index into `list` of the current walk position, or `None` when not
    /// walking. Healed on any focus change (see [`Self::observe_focus`]).
    walk_cursor: Option<usize>,
    /// Window to return to after a keybind-driven jump. Reserved: this field is
    /// cleared by the focus hook but never armed today (arming arrives with the
    /// keybind-jump registry). One-sided wiring is intentional.
    return_target: Option<Id>,
    /// Next id to mint. Only ever grows; a retired id is never reused.
    next_id: u64,
    /// The window the focus hook last recorded. The walk-filter: [`Self::commit_walk`]
    /// and jump-commit set this synchronously to the landed window so the focus
    /// hook sees no delta and a walk never resets its own cursor or triggers MRU.
    last_seen_focus: Option<Id>,
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
        }
    }
}

impl<Id: PartialEq + Clone> Bookmarks<Id> {
    /// Mint the next id, growing the counter.
    fn mint_id(&mut self) -> BookmarkId {
        let id = BookmarkId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Position of the bookmark anchoring `window`, if any.
    fn position_of_window(&self, window: &Id) -> Option<usize> {
        self.list.iter().position(|b| b.anchor.window() == window)
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
            }
        } else {
            let id = self.mint_id();
            self.list.push(Bookmark {
                id,
                anchor: BookmarkAnchor::Window { window, activity },
                name: None,
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
        Some(removed)
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
        let target = match self.walk_base(focused) {
            Some(base) => step(base, direction, len, wrap),
            None => match direction {
                WalkDirection::Backward => Some(len - 1),
                WalkDirection::Forward => Some(0),
            },
        };
        target.map(WalkTarget)
    }

    /// The base index a walk steps from, or `None` when there is no current
    /// position (first step lands on a boundary).
    fn walk_base(&self, focused: Option<&Id>) -> Option<usize> {
        if let Some(i) = self.walk_cursor {
            if i < self.list.len() && Some(self.list[i].anchor.window()) == focused {
                return Some(i);
            }
        }
        focused.and_then(|w| self.position_of_window(w))
    }

    /// Commit a walk onto `target`: park the cursor there and record the landed
    /// window as the last-seen focus.
    ///
    /// Recording the focus synchronously is the walk-filter: the focus hook then
    /// sees no delta when this window becomes focused, so a walk never resets its
    /// own cursor and never triggers MRU promotion.
    pub fn commit_walk(&mut self, target: WalkTarget) {
        let target = target.0;
        debug_assert!(
            target < self.list.len(),
            "walk target is minted only from a validated list index"
        );
        self.walk_cursor = Some(target);
        self.last_seen_focus = Some(self.list[target].anchor.window().clone());
    }

    /// Commit a jump onto `window`: clear the walk cursor, record the landed
    /// window as the last-seen focus, and — under [`OrderMode::Mru`] — promote
    /// the bookmark to the front. A jump *is* an activation; recording the focus
    /// synchronously keeps the focus hook from double-promoting it.
    pub fn commit_jump(&mut self, window: &Id, order: OrderMode) {
        self.walk_cursor = None;
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

    /// Drop the bookmark anchoring `window` (if any), keeping the list free of
    /// dead windows, and clear `return_target` if it held the closed window.
    ///
    /// At most one bookmark anchors a window, so this removes at most one entry.
    /// The cursor is adjusted with a snapshot-then-subtract discipline so the
    /// shape stays correct if the one-per-window invariant is ever relaxed:
    /// pruning the cursor's own entry returns the cursor to `None` (matching
    /// [`Self::remove_by_id`]), and each removal strictly before the cursor
    /// decrements it.
    pub fn prune_window(&mut self, window: &Id) {
        let original_cursor = self.walk_cursor;
        let mut kept = Vec::with_capacity(self.list.len());
        let mut removed_cursor_entry = false;
        let mut removed_before_cursor = 0usize;
        for (idx, bm) in self.list.drain(..).enumerate() {
            if bm.anchor.window() == window {
                if let Some(c) = original_cursor {
                    if idx == c {
                        removed_cursor_entry = true;
                    } else if idx < c {
                        removed_before_cursor += 1;
                    }
                }
            } else {
                kept.push(bm);
            }
        }
        self.list = kept;
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
            .find(|b| b.anchor.window() == window)
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

    /// Read accessor for the return target. Reserved (never armed today).
    pub(crate) fn return_target(&self) -> Option<&Id> {
        self.return_target.as_ref()
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
        bm.list().iter().map(|b| *b.anchor().window()).collect()
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
            let prev_window = *bm.list()[idx - 1].anchor().window();
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
}
