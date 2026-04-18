//! Activity / workspace-view types. See `docs/activities-design.md`.
//!
//! While `Monitor` still owns `Vec<Workspace<W>>`, the invariant
//! `view.ids()[i] == monitor.workspaces[i].id()` is upheld by `Monitor`'s
//! mutating methods.

use super::workspace::WorkspaceId;

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
}
