//! Activity / workspace-view types.
//!
//! `Workspace<W>` values live in `Layout.workspaces: HashMap<WorkspaceId, Workspace<W>>`.
//! Per-output [`WorkspaceView`]s live in `Activity.views: HashMap<OutputId, WorkspaceView>`.
//! For the active activity, the `views` key domain equals `{ OutputId::new(&mon.output) | mon ∈
//! Layout.monitors }`; every id in any view's `ids()` is a key in `Layout.workspaces`.
//! Inactive activities carry a dormant snapshot of their views across activity switches.
//! The active-activity invariant is enforced in `Layout::verify_invariants`.

use std::collections::HashMap;

use indexmap::IndexMap;

use super::workspace::{OutputId, WorkspaceId};
use crate::utils::id::IdCounter;

/// An ordered list of workspace IDs with an active / previous cursor.
///
/// Invariants:
/// - `ids` is non-empty once constructed.
/// - `ids` contains no duplicates.
/// - `active` is always an id present in `ids`.
/// - `previous`, if `Some`, is an id present in `ids`.
#[derive(Debug, Clone)]
// No `is_empty` — a `WorkspaceView` is never empty by construction.
#[allow(clippy::len_without_is_empty)]
pub struct WorkspaceView {
    ids: Vec<WorkspaceId>,
    active: WorkspaceId,
    previous: Option<WorkspaceId>,
}

impl WorkspaceView {
    /// Panics if `ids` is empty or `active_pos >= ids.len()`.
    pub fn new(ids: Vec<WorkspaceId>, active_pos: usize) -> Self {
        assert!(!ids.is_empty(), "WorkspaceView must have at least one id");
        assert!(
            active_pos < ids.len(),
            "active_pos {active_pos} out of bounds for ids.len() = {}",
            ids.len()
        );
        let active = ids[active_pos];
        Self {
            ids,
            active,
            previous: None,
        }
    }

    pub fn ids(&self) -> &[WorkspaceId] {
        &self.ids
    }

    pub fn active(&self) -> WorkspaceId {
        self.active
    }

    pub fn previous(&self) -> Option<WorkspaceId> {
        self.previous
    }

    pub fn active_position(&self) -> usize {
        self.position_of(self.active)
            .expect("active id must be present in ids")
    }

    pub fn previous_position(&self) -> Option<usize> {
        self.previous.and_then(|id| self.position_of(id))
    }

    pub fn position_of(&self, id: WorkspaceId) -> Option<usize> {
        self.ids.iter().position(|i| *i == id)
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Activate the workspace at `pos`.
    ///
    /// If `ids[pos]` is already the active id, this is a no-op: `previous` is
    /// left untouched and `false` is returned. Otherwise `previous` is set to
    /// the old active id and `true` is returned. The no-op identity is
    /// id-based, so `pos` may differ from `active_position()` and still be a
    /// no-op if an intervening rearrangement moved the active id there.
    pub fn activate(&mut self, pos: usize) -> bool {
        assert!(
            pos < self.ids.len(),
            "activate pos {pos} out of bounds for len {}",
            self.ids.len()
        );
        let new_active = self.ids[pos];
        if new_active == self.active {
            return false;
        }
        self.previous = Some(self.active);
        self.active = new_active;
        true
    }

    /// Change `active` to the id at `pos` without touching `previous`.
    ///
    /// Used when shifting the active cursor as part of a rearrangement that is
    /// not a user-visible workspace switch (e.g., pinning focus to the first
    /// non-empty workspace after an output consolidation).
    pub fn set_active_at(&mut self, pos: usize) {
        assert!(
            pos < self.ids.len(),
            "set_active_at pos {pos} out of bounds for len {}",
            self.ids.len()
        );
        self.active = self.ids[pos];
    }

    /// Overwrite `previous`. Used to restore a saved previous across a
    /// sequence of mutations (e.g., `move_workspace_up/down`).
    ///
    /// If `Some`, the id must be present in `ids`. `previous == active` is
    /// permitted (and harmless — `activate` on the same id is a no-op).
    pub fn set_previous(&mut self, previous: Option<WorkspaceId>) {
        if let Some(id) = previous {
            assert!(self.ids.contains(&id), "previous id must be present in ids");
        }
        self.previous = previous;
    }

    pub fn insert(&mut self, pos: usize, id: WorkspaceId) {
        assert!(
            pos <= self.ids.len(),
            "insert pos {pos} out of bounds for len {}",
            self.ids.len()
        );
        assert!(!self.ids.contains(&id), "inserting duplicate id");
        self.ids.insert(pos, id);
    }

    /// Remove the id at `pos`, returning it.
    ///
    /// If the removed id was `active`, the new active is the id at
    /// `pos.saturating_sub(1)`. If it was `previous`, `previous` becomes
    /// `None`.
    ///
    /// Panics if `pos` is out of bounds, or if removing would leave the view
    /// empty.
    pub fn remove_at(&mut self, pos: usize) -> WorkspaceId {
        assert!(
            pos < self.ids.len(),
            "remove_at pos {pos} out of bounds for len {}",
            self.ids.len()
        );
        assert!(self.ids.len() > 1, "cannot remove the last id from a view");

        let removed = self.ids.remove(pos);

        if removed == self.active {
            let new_pos = pos.saturating_sub(1);
            self.active = self.ids[new_pos];
        }
        if self.previous == Some(removed) {
            self.previous = None;
        }
        removed
    }

    /// The `active` and `previous` ids are unchanged; their positions may swap
    /// if they referred to one of the swapped entries.
    pub fn swap(&mut self, a: usize, b: usize) {
        self.ids.swap(a, b);
    }

    /// `active` and `previous` ids are unchanged; their positions adjust
    /// naturally with the move.
    pub fn move_within(&mut self, old_pos: usize, new_pos: usize) {
        if old_pos == new_pos {
            return;
        }
        let id = self.ids.remove(old_pos);
        debug_assert!(
            new_pos <= self.ids.len(),
            "move_within new_pos {new_pos} out of bounds for len {}",
            self.ids.len()
        );
        self.ids.insert(new_pos, id);
    }
}

static ACTIVITY_ID_COUNTER: IdCounter = IdCounter::new();

/// Stable, process-unique identifier for an [`Activity`]. Will be exposed to
/// IPC as its `u64` when the IPC types land (mirrors [`WorkspaceId`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActivityId(u64);

impl ActivityId {
    fn next() -> ActivityId {
        ActivityId(ACTIVITY_ID_COUNTER.next())
    }

    pub fn get(self) -> u64 {
        self.0
    }

    #[cfg(test)]
    pub fn specific(id: u64) -> Self {
        Self(id)
    }
}

/// A named collection of per-output [`WorkspaceView`]s — the grouping
/// dimension above `Workspace`.
///
/// `is_config_declared` distinguishes activities the user named in config
/// (stable across reload) from runtime-created ones. The promotion rules
/// (rename / config reload) land with the action handlers that need them.
///
/// The `views` map backs [`Layout::active_view`] when this activity is the active one. For the
/// active activity the key domain equals the connected monitors' `OutputId`s; inactive activities
/// carry a dormant snapshot across activity switches. The Layout-level invariant is enforced in
/// `Layout::verify_invariants`.
#[derive(Debug)]
pub struct Activity {
    id: ActivityId,
    name: String,
    is_config_declared: bool,
    /// Per-output workspace views. For the active activity, the key domain equals connected
    /// monitors' `OutputId`s; for inactive activities this is a dormant snapshot. See struct doc.
    views: HashMap<OutputId, WorkspaceView>,
}

impl Activity {
    pub fn new_runtime(name: String) -> Self {
        Self {
            id: ActivityId::next(),
            name,
            is_config_declared: false,
            views: HashMap::new(),
        }
    }

    pub fn new_config_declared(name: String) -> Self {
        Self {
            id: ActivityId::next(),
            name,
            is_config_declared: true,
            views: HashMap::new(),
        }
    }

    pub fn id(&self) -> ActivityId {
        self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn is_config_declared(&self) -> bool {
        self.is_config_declared
    }

    /// Per-output views owned by this activity. For the active activity, the key domain equals
    /// connected monitors' `OutputId`s and this is what [`Layout::active_view`] reads. For
    /// inactive activities the map is a dormant snapshot preserved across activity switches.
    pub fn views(&self) -> &HashMap<OutputId, WorkspaceView> {
        &self.views
    }

    /// Mutable access to per-output views. Same key-domain semantics as [`Self::views`].
    pub fn views_mut(&mut self) -> &mut HashMap<OutputId, WorkspaceView> {
        &mut self.views
    }

    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }
}

/// Ordered pool of [`Activity`]s plus active / previous cursors.
///
/// Invariants:
/// - `map` is non-empty — guaranteed by [`Activities::new`] taking a seed `Activity`; no `Default`,
///   no push-to-empty API.
/// - `active` is always a key in `map`.
/// - `previous`, if `Some`, is always a key in `map`.
/// - `previous`, if `Some`, is never equal to `active` (distinctness). The
///   no-op fast-path in [`Activities::set_active`] preserves this: when
///   `target == active` the call returns early without touching `previous`;
///   otherwise `previous` is set to the old active (guaranteed `!= target`),
///   so after the write `previous != active` still holds.
/// - Each stored `Activity`'s `id` equals its key in `map` (enforced by
///   private `Activity.id` + construction-only id minting; inserts go through
///   `map.insert(activity.id, activity)` exclusively).
#[derive(Debug)]
// No `is_empty` — `Activities` is never empty by construction.
#[allow(clippy::len_without_is_empty)]
pub struct Activities {
    map: IndexMap<ActivityId, Activity>,
    active: ActivityId,
    previous: Option<ActivityId>,
}

impl Activities {
    /// Seed with the first (default) activity. After construction,
    /// `active_id() == seed.id` and `previous_id() == None`. Mirrors
    /// [`WorkspaceView::new`]'s non-empty-by-construction discipline.
    pub fn new(seed: Activity) -> Self {
        let active = seed.id;
        let mut map = IndexMap::new();
        map.insert(seed.id, seed);
        Self {
            map,
            active,
            previous: None,
        }
    }

    /// Build an `Activities` pool from the parsed `config.activities` list.
    ///
    /// Empty input yields a single runtime "Default" activity (DD §6.5
    /// backwards-compat: a config with no `activity` blocks must behave
    /// identically to today's single-activity world).
    ///
    /// Non-empty input: the first entry becomes the seed (active cursor), the
    /// rest are inserted in declaration order via [`Self::insert`]. All entries
    /// are flagged `is_config_declared`. After construction, `active_id()`
    /// equals the id of the `Activity` minted from the first config entry, and
    /// `previous_id() == None`.
    pub fn from_config_or_default(config_activities: &[niri_config::Activity]) -> Self {
        if config_activities.is_empty() {
            return Self::new(Activity::new_runtime("Default".to_owned()));
        }

        let mut iter = config_activities.iter();
        let first = iter
            .next()
            .expect("non-empty check above guarantees at least one entry");
        let mut pool = Self::new(Activity::new_config_declared(first.name.0.clone()));
        for entry in iter {
            pool.insert(Activity::new_config_declared(entry.name.0.clone()));
        }
        pool
    }

    /// Insert an additional activity into the pool, preserving the
    /// id-equals-key invariant. The only other inserter is [`Self::new`]
    /// (seed); runtime multi-activity population will later arrive via the
    /// `CreateActivity` action (Phase 1b scope), and config-declared
    /// multi-activity population arrives via [`Self::from_config_or_default`].
    ///
    /// Panics if `activity.id()` is already present in the pool — ids are
    /// monotonic and minted by `ActivityId::next()`, so a collision indicates
    /// a logic bug upstream (e.g., a double-insert).
    pub(super) fn insert(&mut self, activity: Activity) {
        assert!(
            self.map.insert(activity.id, activity).is_none(),
            "Activities::insert: id already present",
        );
    }

    pub fn active(&self) -> &Activity {
        self.map
            .get(&self.active)
            .expect("active id must be a key in the map")
    }

    pub fn active_mut(&mut self) -> &mut Activity {
        self.map
            .get_mut(&self.active)
            .expect("active id must be a key in the map")
    }

    pub fn active_id(&self) -> ActivityId {
        self.active
    }

    pub fn previous_id(&self) -> Option<ActivityId> {
        self.previous
    }

    pub fn get(&self, id: ActivityId) -> Option<&Activity> {
        self.map.get(&id)
    }

    pub fn get_mut(&mut self, id: ActivityId) -> Option<&mut Activity> {
        self.map.get_mut(&id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Activity> {
        self.map.values()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn contains(&self, id: ActivityId) -> bool {
        self.map.contains_key(&id)
    }

    /// Flip the active cursor to `target`, recording the previously-active id
    /// in `previous` on a real flip.
    ///
    /// Contract: `target` must be a live key in `map`. Caller validates before
    /// dispatch (see [`Self::contains`]); this method is debug-asserted only
    /// and does not perform user-input validation. Folding validation here
    /// would duplicate the single caller-side error-log site at
    /// `Layout::switch_activity`.
    ///
    /// No-op fast-path: if `target == active_id()`, returns immediately
    /// without touching `previous`. Otherwise `previous = Some(old_active)`
    /// and `active = target`; since `target != old_active` in that branch,
    /// the `previous != active` distinctness invariant is re-established.
    ///
    /// Panics (debug only) if `target` is not a live key in `map`.
    pub(super) fn set_active(&mut self, target: ActivityId) {
        debug_assert!(
            self.map.contains_key(&target),
            "set_active target must be a live key in the map",
        );
        if target == self.active {
            return;
        }
        self.previous = Some(self.active);
        self.active = target;
        debug_assert!(
            self.previous != Some(self.active),
            "set_active must preserve previous != Some(active) distinctness",
        );
    }

    #[cfg(debug_assertions)]
    pub(super) fn verify_invariants(&self) {
        assert!(
            self.map.contains_key(&self.active),
            "Activities.active {:?} must be a live key in the map",
            self.active,
        );
        if let Some(prev) = self.previous {
            assert!(
                self.map.contains_key(&prev),
                "Activities.previous {:?} must be a live key in the map",
                prev,
            );
            assert_ne!(
                Some(prev),
                Some(self.active),
                "Activities.previous must not equal active (distinctness invariant)",
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ws(n: u64) -> WorkspaceId {
        WorkspaceId::specific(n)
    }

    #[test]
    fn new_and_basic_accessors() {
        let v = WorkspaceView::new(vec![ws(1), ws(2), ws(3)], 1);
        assert_eq!(v.ids(), &[ws(1), ws(2), ws(3)]);
        assert_eq!(v.active(), ws(2));
        assert_eq!(v.active_position(), 1);
        assert_eq!(v.previous(), None);
        assert_eq!(v.len(), 3);
        assert_eq!(v.position_of(ws(3)), Some(2));
        assert_eq!(v.position_of(ws(99)), None);
    }

    #[test]
    #[should_panic]
    fn new_empty_panics() {
        let _ = WorkspaceView::new(vec![], 0);
    }

    #[test]
    #[should_panic]
    fn new_out_of_bounds_panics() {
        let _ = WorkspaceView::new(vec![ws(1)], 2);
    }

    #[test]
    fn activate_changes_active_and_previous() {
        let mut v = WorkspaceView::new(vec![ws(1), ws(2), ws(3)], 0);
        assert!(v.activate(2));
        assert_eq!(v.active(), ws(3));
        assert_eq!(v.previous(), Some(ws(1)));
        assert_eq!(v.active_position(), 2);

        // No-op when activating the already-active id.
        assert!(!v.activate(2));
        assert_eq!(v.previous(), Some(ws(1)));
    }

    #[test]
    fn activate_same_position_does_not_update_previous() {
        let mut v = WorkspaceView::new(vec![ws(1), ws(2)], 0);
        assert!(!v.activate(0));
        assert_eq!(v.previous(), None);
    }

    #[test]
    fn set_active_at_preserves_previous() {
        let mut v = WorkspaceView::new(vec![ws(1), ws(2), ws(3)], 0);
        v.activate(2);
        assert_eq!(v.previous(), Some(ws(1)));

        v.set_active_at(1);
        assert_eq!(v.active(), ws(2));
        assert_eq!(v.previous(), Some(ws(1)));
    }

    #[test]
    #[should_panic]
    fn set_active_at_out_of_bounds_panics() {
        let mut v = WorkspaceView::new(vec![ws(1)], 0);
        v.set_active_at(1);
    }

    #[test]
    fn set_previous_accepts_id_in_view() {
        let mut v = WorkspaceView::new(vec![ws(1), ws(2), ws(3)], 0);
        v.set_previous(Some(ws(3)));
        assert_eq!(v.previous(), Some(ws(3)));
        v.set_previous(None);
        assert_eq!(v.previous(), None);
    }

    #[test]
    #[should_panic]
    fn set_previous_rejects_unknown_id() {
        let mut v = WorkspaceView::new(vec![ws(1), ws(2)], 0);
        v.set_previous(Some(ws(99)));
    }

    #[test]
    fn insert_shifts_positions_but_keeps_active() {
        let mut v = WorkspaceView::new(vec![ws(1), ws(2), ws(3)], 2);
        v.insert(0, ws(0));
        assert_eq!(v.ids(), &[ws(0), ws(1), ws(2), ws(3)]);
        assert_eq!(v.active(), ws(3));
        assert_eq!(v.active_position(), 3);
    }

    #[test]
    fn insert_after_active_does_not_shift_active_position() {
        let mut v = WorkspaceView::new(vec![ws(1), ws(2), ws(3)], 1);
        v.insert(3, ws(4));
        assert_eq!(v.ids(), &[ws(1), ws(2), ws(3), ws(4)]);
        assert_eq!(v.active(), ws(2));
        assert_eq!(v.active_position(), 1);
    }

    #[test]
    fn insert_into_singleton_preserves_active() {
        let mut v = WorkspaceView::new(vec![ws(1)], 0);
        v.insert(0, ws(0));
        assert_eq!(v.ids(), &[ws(0), ws(1)]);
        assert_eq!(v.active(), ws(1));
        assert_eq!(v.active_position(), 1);
    }

    #[test]
    #[should_panic]
    fn insert_rejects_duplicate() {
        let mut v = WorkspaceView::new(vec![ws(1), ws(2)], 0);
        v.insert(1, ws(1));
    }

    #[test]
    #[should_panic]
    fn insert_out_of_bounds_panics() {
        let mut v = WorkspaceView::new(vec![ws(1), ws(2)], 0);
        v.insert(3, ws(3));
    }

    #[test]
    fn remove_before_active_shifts_active_position() {
        let mut v = WorkspaceView::new(vec![ws(1), ws(2), ws(3)], 2);
        v.remove_at(0);
        assert_eq!(v.ids(), &[ws(2), ws(3)]);
        assert_eq!(v.active(), ws(3));
        assert_eq!(v.active_position(), 1);
    }

    #[test]
    fn remove_active_shifts_to_previous_position() {
        let mut v = WorkspaceView::new(vec![ws(1), ws(2), ws(3)], 2);
        v.remove_at(2);
        assert_eq!(v.ids(), &[ws(1), ws(2)]);
        assert_eq!(v.active(), ws(2));
        assert_eq!(v.active_position(), 1);
    }

    #[test]
    fn remove_active_at_zero_stays_at_zero() {
        let mut v = WorkspaceView::new(vec![ws(1), ws(2), ws(3)], 0);
        v.remove_at(0);
        assert_eq!(v.ids(), &[ws(2), ws(3)]);
        assert_eq!(v.active(), ws(2));
        assert_eq!(v.active_position(), 0);
    }

    #[test]
    fn remove_clears_previous_if_needed() {
        let mut v = WorkspaceView::new(vec![ws(1), ws(2), ws(3)], 2);
        v.set_previous(Some(ws(1)));
        v.remove_at(0);
        assert_eq!(v.previous(), None);
    }

    #[test]
    fn remove_keeps_previous_when_removing_other() {
        let mut v = WorkspaceView::new(vec![ws(1), ws(2), ws(3)], 2);
        v.set_previous(Some(ws(1)));
        v.remove_at(1); // removes ws(2), not ws(1)
        assert_eq!(v.ids(), &[ws(1), ws(3)]);
        assert_eq!(v.previous(), Some(ws(1)));
    }

    #[test]
    #[should_panic]
    fn remove_last_panics() {
        let mut v = WorkspaceView::new(vec![ws(1)], 0);
        v.remove_at(0);
    }

    #[test]
    fn swap_keeps_active_id() {
        let mut v = WorkspaceView::new(vec![ws(1), ws(2), ws(3)], 2);
        v.swap(0, 2);
        assert_eq!(v.ids(), &[ws(3), ws(2), ws(1)]);
        assert_eq!(v.active(), ws(3));
        assert_eq!(v.active_position(), 0);
    }

    #[test]
    fn move_within_preserves_active_id() {
        let mut v = WorkspaceView::new(vec![ws(1), ws(2), ws(3), ws(4)], 1);
        v.move_within(0, 3);
        assert_eq!(v.ids(), &[ws(2), ws(3), ws(4), ws(1)]);
        assert_eq!(v.active(), ws(2));
        assert_eq!(v.active_position(), 0);
    }

    #[test]
    fn move_within_same_pos_is_noop() {
        let mut v = WorkspaceView::new(vec![ws(1), ws(2)], 0);
        v.move_within(0, 0);
        assert_eq!(v.ids(), &[ws(1), ws(2)]);
    }

    #[test]
    fn new_seeds_with_active_in_map() {
        let seed = Activity::new_runtime("work".into());
        let seed_id = seed.id();
        let acts = Activities::new(seed);
        assert_eq!(acts.active_id(), seed_id);
        assert_eq!(acts.len(), 1);
        assert_eq!(acts.previous_id(), None);
        assert!(acts.contains(seed_id));
        assert_eq!(acts.active().name(), "work");
        assert!(acts.active().views().is_empty());
    }

    #[test]
    fn active_returns_seed_by_ref() {
        let seed = Activity::new_runtime("work".into());
        let seed_id = seed.id();
        let mut acts = Activities::new(seed);
        assert_eq!(acts.active().id(), seed_id);
        acts.active_mut().set_name("play".into());
        assert_eq!(acts.active().name(), "play");
    }

    #[test]
    fn get_behavior() {
        let seed = Activity::new_runtime("work".into());
        let seed_id = seed.id();
        let mut acts = Activities::new(seed);
        assert_eq!(acts.get(seed_id).map(|a| a.id()), Some(seed_id));
        assert!(acts.get(ActivityId::specific(99_999)).is_none());
        // get_mut: mutate through known id, confirm None for unknown.
        acts.get_mut(seed_id)
            .expect("seed id must be present")
            .set_name("mutated".into());
        assert_eq!(acts.active().name(), "mutated");
        assert!(acts.get_mut(ActivityId::specific(99_999)).is_none());
    }

    #[test]
    fn iter_yields_single_seed() {
        let seed = Activity::new_runtime("work".into());
        let seed_id = seed.id();
        let acts = Activities::new(seed);
        let collected: Vec<_> = acts.iter().map(|a| a.id()).collect();
        assert_eq!(collected, vec![seed_id]);
        assert_eq!(acts.iter().count(), 1);
    }

    #[test]
    fn contains_tracks_seed() {
        let seed = Activity::new_runtime("work".into());
        let seed_id = seed.id();
        let acts = Activities::new(seed);
        assert!(acts.contains(seed_id));
        assert!(!acts.contains(ActivityId::specific(u64::MAX)));
    }

    #[test]
    fn config_declared_flag_preserved() {
        let runtime = Activity::new_runtime("a".into());
        let declared = Activity::new_config_declared("b".into());
        assert!(!runtime.is_config_declared());
        assert!(declared.is_config_declared());
    }

    #[test]
    fn runtime_and_config_ids_are_unique() {
        let a = Activity::new_runtime("a".into());
        let b = Activity::new_config_declared("b".into());
        assert_ne!(a.id(), b.id());
    }

    #[test]
    fn activity_id_specific_roundtrips() {
        assert_eq!(ActivityId::specific(7).get(), 7);
    }

    #[test]
    fn set_active_no_op_preserves_previous() {
        let seed = Activity::new_runtime("work".into());
        let seed_id = seed.id();
        let mut acts = Activities::new(seed);

        // Precondition: fresh pool has no previous.
        assert_eq!(acts.previous_id(), None);

        acts.set_active(seed_id);

        assert_eq!(acts.active_id(), seed_id);
        // No-op must leave `previous` untouched (not overwrite with Some(seed_id)).
        assert_eq!(acts.previous_id(), None);
    }

    #[test]
    fn set_active_no_op_with_existing_previous_preserves_it() {
        let seed = Activity::new_runtime("work".into());
        let seed_id = seed.id();
        let other = Activity::new_runtime("play".into());
        let other_id = other.id();

        let mut acts = Activities::new(seed);
        acts.insert(other);

        // Establish a non-None previous.
        acts.set_active(other_id); // previous = Some(seed_id), active = other_id

        // No-op: must not clear `previous`.
        acts.set_active(other_id);

        assert_eq!(acts.previous_id(), Some(seed_id));
        assert_eq!(acts.active_id(), other_id);
    }

    #[test]
    fn set_active_flip_updates_cursors() {
        let seed = Activity::new_runtime("work".into());
        let seed_id = seed.id();
        let other = Activity::new_runtime("play".into());
        let other_id = other.id();

        let mut acts = Activities::new(seed);
        acts.insert(other);

        acts.set_active(other_id);

        assert_eq!(acts.active_id(), other_id);
        assert_eq!(acts.previous_id(), Some(seed_id));
        // Distinctness invariant on the flip path.
        assert_ne!(acts.previous_id(), Some(acts.active_id()));
    }

    #[test]
    #[should_panic]
    #[cfg(debug_assertions)]
    fn set_active_unknown_debug_asserts() {
        let seed = Activity::new_runtime("work".into());
        let mut acts = Activities::new(seed);
        // `u64::MAX` cannot collide with a runtime-minted id (counter starts at 0).
        acts.set_active(ActivityId::specific(u64::MAX));
    }

    #[test]
    fn activities_from_config_or_default_empty_seeds_runtime_default() {
        let acts = Activities::from_config_or_default(&[]);
        assert_eq!(acts.len(), 1);
        let only = acts.active();
        assert_eq!(only.name(), "Default");
        assert!(!only.is_config_declared());
        assert_eq!(acts.active_id(), only.id());
        assert_eq!(acts.previous_id(), None);
    }

    #[test]
    fn activities_from_config_or_default_multiple_preserves_order_and_first_active() {
        let cfg = vec![
            niri_config::Activity {
                name: niri_config::ActivityName("Work".to_owned()),
            },
            niri_config::Activity {
                name: niri_config::ActivityName("Personal".to_owned()),
            },
        ];
        let acts = Activities::from_config_or_default(&cfg);
        assert_eq!(acts.len(), 2);
        let names: Vec<&str> = acts.iter().map(|a| a.name()).collect();
        assert_eq!(names, vec!["Work", "Personal"]);
        assert!(acts.iter().all(|a| a.is_config_declared()));
        assert_eq!(acts.active().name(), "Work");
        assert_eq!(acts.previous_id(), None);
    }
}
