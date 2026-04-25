//! Activity / workspace-view types.
//!
//! `Workspace<W>` values live in `Layout.workspaces: HashMap<WorkspaceId, Workspace<W>>`.
//! Per-output [`WorkspaceView`]s live in `Activity.views: HashMap<OutputId, WorkspaceView>`.
//! For the active activity, the `views` key domain equals `{ OutputId::new(&mon.output) | mon ∈
//! Layout.monitors }`; every id in any view's `ids()` is a key in `Layout.workspaces`.
//! Inactive activities carry a dormant snapshot of their views across activity switches.
//! The active-activity invariant is enforced in `Layout::verify_invariants`.

use std::collections::{HashMap, HashSet};
use std::fmt;

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
    pub(super) fn views_mut(&mut self) -> &mut HashMap<OutputId, WorkspaceView> {
        &mut self.views
    }

    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }
}

/// Validation failure for runtime activity creation via [`Activities::create_runtime`]
/// / `Layout::create_activity`. The two variants mirror the rejection rules enforced
/// inside `create_runtime`; callers surface them as log messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CreateActivityError {
    /// The requested name was empty after trimming whitespace.
    EmptyName,
    /// The requested name collided (case-insensitively) with an existing activity's
    /// name. Mirrors the uniqueness policy in
    /// [`Activities::resolve_config_names`].
    DuplicateName,
}

impl fmt::Display for CreateActivityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyName => f.write_str("activity name must not be empty"),
            Self::DuplicateName => f.write_str("activity name already exists"),
        }
    }
}

impl std::error::Error for CreateActivityError {}

/// Validation failure for runtime activity removal via [`Activities::remove`] /
/// `Layout::remove_activity`. Each variant corresponds to one of the rejection
/// rules the outer `Layout::remove_activity` evaluates before any mutation
///. Callers surface them as log messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoveActivityError {
    /// No activity in the pool matches the supplied reference.
    NotFound,
    /// Target is flagged `is_config_declared` — config-declared activities are
    /// removed by editing the config file and reloading, not via the runtime
    /// action ( bullet 1).
    ConfigDeclared,
    /// Target is the only activity in the pool; at least one activity must
    /// always exist ( bullet 3).
    LastRemaining,
    /// At least one workspace exclusively belonging to this activity still has
    /// windows. The caller must close / move those windows first (
    /// bullet 2).
    ExclusiveWorkspaceHasWindows,
    /// At least one workspace exclusively belonging to this activity is named
    /// (even if empty). Named-empty exclusive workspaces are preserved: the
    /// caller must unname them first ( "Exclusive workspace
    /// destruction semantics").
    ExclusiveNamedWorkspace,
}

impl fmt::Display for RemoveActivityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound => f.write_str("activity not found"),
            Self::ConfigDeclared => {
                f.write_str("activity is config-declared; edit config and reload to remove")
            }
            Self::LastRemaining => f.write_str("cannot remove the last remaining activity"),
            Self::ExclusiveWorkspaceHasWindows => f.write_str(
                "activity owns an exclusive workspace with windows; close or move them first",
            ),
            Self::ExclusiveNamedWorkspace => f.write_str(
                "activity owns a named exclusive workspace (even if empty); unname it first",
            ),
        }
    }
}

impl std::error::Error for RemoveActivityError {}

/// Validation failure for activity switching via `Layout::switch_activity` —
/// surfaced through the dispatch layer as
/// `DoActionError::SwitchActivity(SwitchActivityError::NotFound)`.
///
/// `Layout::switch_activity` itself takes an [`ActivityId`] and never fails
/// — the rejection happens at the dispatch boundary when
/// `resolve_activity_ref` returns `None`. This single-variant enum exists so
/// every activity-action outer variant on [`super::DoActionError`] carries a
/// layout-side `*Error` payload (cohort symmetry); it is constructed only by
/// the dispatch arm in `input/mod.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwitchActivityError {
    /// No activity in the pool matches the supplied reference.
    NotFound,
}

impl fmt::Display for SwitchActivityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound => f.write_str("activity not found"),
        }
    }
}

impl std::error::Error for SwitchActivityError {}

/// Validation failure for runtime activity rename via [`Activities::rename_runtime`]
/// / `Layout::rename_activity`. Each variant corresponds to one of the rejection
/// rules the outer `Layout::rename_activity` and `rename_runtime` evaluate before
/// any mutation. Callers surface them as log messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenameActivityError {
    /// No activity in the pool matches the supplied reference.
    NotFound,
    /// Target is flagged `is_config_declared` — config-declared activities are
    /// renamed by editing the config file and reloading, not via the runtime
    /// action.
    ConfigDeclared,
    /// The requested new name was empty after trimming whitespace.
    EmptyName,
    /// The requested new name collides (case-insensitively) with a *different*
    /// activity's name. Renaming to a case variant of the target's own
    /// current name succeeds.
    DuplicateName,
}

impl fmt::Display for RenameActivityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound => f.write_str("activity not found"),
            Self::ConfigDeclared => {
                f.write_str("activity is config-declared; edit config and reload to rename")
            }
            Self::EmptyName => f.write_str("activity name must not be empty"),
            Self::DuplicateName => f.write_str("activity name already exists"),
        }
    }
}

impl std::error::Error for RenameActivityError {}

/// Validation failure for adding a workspace to an activity via
/// `Layout::add_workspace_to_activity`. Each variant corresponds to one of the
/// rejection rules the outer entry point evaluates before any mutation.
///
/// Precedence on unresolvable references: `ActivityNotFound` is returned before
/// `WorkspaceNotFound` (activity is resolved first). The no-op "already a
/// member" path returns `Ok` and is not surfaced here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddWorkspaceToActivityError {
    /// No activity in the pool matches the supplied reference.
    ActivityNotFound,
    /// No workspace in the pool matches the supplied reference (or `None`
    /// was supplied and there is no active workspace, i.e. zero connected
    /// monitors).
    WorkspaceNotFound,
}

impl fmt::Display for AddWorkspaceToActivityError {
    /// Plain lowercase tokens. The token is the entire envelope;
    /// `format_do_action_error` does no further wrapping. Token drift will
    /// fail the `do_action_error_display_matches_wire_contract` pin test.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ActivityNotFound => f.write_str("activity not found"),
            Self::WorkspaceNotFound => f.write_str("workspace not found"),
        }
    }
}

impl std::error::Error for AddWorkspaceToActivityError {}

/// Validation failure for removing a workspace from an activity via
/// `Layout::remove_workspace_from_activity`. Variants mirror the
/// wire-contract table plus the non-empty-activities invariant guard.
///
/// Precedence: `ActivityNotFound` > `WorkspaceNotFound` > `LastActivity`.
/// Resolution runs before membership inspection so an unresolvable reference
/// never leaks into an "already not a member" no-op path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoveWorkspaceFromActivityError {
    /// No activity in the pool matches the supplied reference.
    ActivityNotFound,
    /// No workspace in the pool matches the supplied reference (or `None`
    /// was supplied and there is no active workspace).
    WorkspaceNotFound,
    /// Removing the activity id would leave the workspace's `activities` set
    /// empty — every workspace must belong to at least one activity.
    LastActivity,
}

impl fmt::Display for RemoveWorkspaceFromActivityError {
    /// Plain lowercase tokens. The token is the entire envelope;
    /// `format_do_action_error` does no further wrapping.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ActivityNotFound => f.write_str("activity not found"),
            Self::WorkspaceNotFound => f.write_str("workspace not found"),
            Self::LastActivity => f.write_str("workspace would be left with no activities"),
        }
    }
}

impl std::error::Error for RemoveWorkspaceFromActivityError {}

/// Validation failure for replacing a workspace's activity set via
/// `Layout::set_workspace_activities`. Variants mirror the wire
/// contract plus the non-empty-activities invariant guard.
///
/// Precedence: `ActivityNotFound` > `EmptyActivityList` > `WorkspaceNotFound`.
/// Activity refs are resolved first — an unresolvable ref in the list
/// short-circuits to `ActivityNotFound` regardless of list length, matching
/// the `resolve_activity_ref` precedence of `Add` / `Remove`. Concretely:
/// `[unresolvable_id]` (length 1, unresolvable) → `ActivityNotFound`, not
/// `EmptyActivityList`; the empty-list guard is only reached when the list
/// resolves to zero distinct live activities.
///
/// `WorkspaceNotFound` is wire-surfaced via
/// `DoActionError::SetWorkspaceActivities(SetWorkspaceActivitiesError::WorkspaceNotFound)`;
/// the dispatch arm in `input/mod.rs` returns it directly. (Pre-harmonization
/// the dispatch layer intercepted this variant and returned `Ok(())`;
/// the silent intercept was dropped to harmonize the workspace-miss contract
/// across the activity-action cohort.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetWorkspaceActivitiesError {
    /// At least one supplied activity reference does not resolve to a live
    /// activity in the pool.
    ActivityNotFound,
    /// The supplied `activities` list was empty — every workspace
    /// must belong to at least one activity.
    EmptyActivityList,
    /// No workspace in the pool matches the supplied reference (or `None`
    /// was supplied and there is no active workspace).
    WorkspaceNotFound,
}

impl fmt::Display for SetWorkspaceActivitiesError {
    /// Plain lowercase tokens. The token is the entire envelope;
    /// `format_do_action_error` does no further wrapping.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ActivityNotFound => f.write_str("activity not found"),
            Self::EmptyActivityList => f.write_str("activities list is empty"),
            Self::WorkspaceNotFound => f.write_str("workspace not found"),
        }
    }
}

impl std::error::Error for SetWorkspaceActivitiesError {}

/// Validation failure for `Layout::move_workspace_to_activity`.
/// defines Move as "Add to target + Remove from active" with an explicit
/// requirement that the workspace be a member of the active activity
/// (the move verb requires a well-defined source).
///
/// Precedence: `ActivityNotFound` > `WorkspaceNotFound` >
/// `WorkspaceNotInActiveActivity`. Resolution happens before membership
/// inspection, matching the Part 1 `Remove` precedent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveWorkspaceToActivityError {
    /// Target activity reference does not resolve to a live activity.
    ActivityNotFound,
    /// Workspace reference does not resolve to a live workspace (or
    /// `None` was supplied and there is no active workspace).
    /// Wire-surfaced via
    /// `DoActionError::MoveWorkspaceToActivity(MoveWorkspaceToActivityError::WorkspaceNotFound)`.
    /// (Pre-harmonization the dispatch layer intercepted this and returned
    /// `Ok(())` as a silent no-op; the intercept was dropped to harmonize
    /// the workspace-miss contract across the activity-action cohort.)
    WorkspaceNotFound,
    /// Workspace is not a member of the currently-active activity.
    /// Move requires a well-defined source.
    WorkspaceNotInActiveActivity,
}

impl fmt::Display for MoveWorkspaceToActivityError {
    /// Plain lowercase tokens. The token is the entire envelope;
    /// `format_do_action_error` does no further wrapping.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ActivityNotFound => f.write_str("activity not found"),
            Self::WorkspaceNotFound => f.write_str("workspace not found"),
            Self::WorkspaceNotInActiveActivity => f.write_str("workspace not in active activity"),
        }
    }
}

impl std::error::Error for MoveWorkspaceToActivityError {}

/// Validation failure for `Layout::set_workspace_sticky`. Single-variant: the
/// only failure mode is a workspace reference that does not resolve. Wire-
/// surfaced via
/// `DoActionError::SetWorkspaceSticky(SetWorkspaceStickyError::WorkspaceNotFound)`.
/// (Pre-harmonization the dispatch layer intercepted this as a silent no-op;
/// the intercept was dropped to harmonize the workspace-miss contract across
/// the activity-action cohort.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetWorkspaceStickyError {
    /// No workspace in the pool matches the supplied reference (or `None`
    /// was supplied and there is no active workspace, i.e. zero connected
    /// monitors).
    WorkspaceNotFound,
}

impl fmt::Display for SetWorkspaceStickyError {
    /// Plain lowercase token. The token is the entire envelope;
    /// `format_do_action_error` does no further wrapping. Token drift will
    /// fail `do_action_error_envelope_matches_wire_contract`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WorkspaceNotFound => f.write_str("workspace not found"),
        }
    }
}

impl std::error::Error for SetWorkspaceStickyError {}

/// Validation failure for `Layout::unset_workspace_sticky`. Single-variant —
/// see [`SetWorkspaceStickyError`] for the workspace-miss harmonization rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnsetWorkspaceStickyError {
    /// No workspace in the pool matches the supplied reference (or `None`
    /// was supplied and there is no active workspace).
    WorkspaceNotFound,
}

impl fmt::Display for UnsetWorkspaceStickyError {
    /// Plain lowercase token. The token is the entire envelope;
    /// `format_do_action_error` does no further wrapping. Token drift will
    /// fail `do_action_error_envelope_matches_wire_contract`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WorkspaceNotFound => f.write_str("workspace not found"),
        }
    }
}

impl std::error::Error for UnsetWorkspaceStickyError {}

/// Validation failure for `Layout::toggle_workspace_sticky`. Single-variant —
/// see [`SetWorkspaceStickyError`] for the workspace-miss harmonization rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToggleWorkspaceStickyError {
    /// No workspace in the pool matches the supplied reference (or `None`
    /// was supplied and there is no active workspace).
    WorkspaceNotFound,
}

impl fmt::Display for ToggleWorkspaceStickyError {
    /// Plain lowercase token. The token is the entire envelope;
    /// `format_do_action_error` does no further wrapping. Token drift will
    /// fail `do_action_error_envelope_matches_wire_contract`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WorkspaceNotFound => f.write_str("workspace not found"),
        }
    }
}

impl std::error::Error for ToggleWorkspaceStickyError {}

/// Validation failure for config-reload activity removal via
/// `Layout::reconcile_activities_on_reload_remove`. Each variant corresponds
/// to one of the rejection rules the outer entry point evaluates before any
/// mutation ( bullet 2). Callers surface them via `warn!` + the
/// config-error notification, then early-return the entire reload — the
/// atomicity contract mirrors `RemoveActivityError` on the IPC path.
///
/// Payloads carry owned `String` names (resolved at validation time while the
/// pool is still intact), so this enum is `Clone` but not `Copy` — sibling
/// `RemoveActivityError` is `Copy` because its payload is unit-per-variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ReloadActivityRemovalError {
    /// At least one workspace exclusively belonging to an about-to-be-removed
    /// activity has windows. The user must close / move those windows first;
    /// the entire reload is rejected ( bullet 2).
    ExclusiveWorkspaceHasWindows {
        activity_name: String,
        workspace_id: super::workspace::WorkspaceId,
    },
    /// Removing all in-remove-set activities would leave the pool empty — no
    /// runtime activities survive to absorb the active-cursor cascade. Parallel
    /// to [`RemoveActivityError::LastRemaining`] on the IPC path, but evaluated
    /// across the whole remove-set rather than a single target.
    WouldEmptyPool { activity_name: String },
    /// The active activity is in the remove-set (so an active-cursor cascade
    /// is required) but [`Layout::is_activity_switch_hard_blocked`] returns
    /// `Some(_)` — an interactive move, DnD, or workspace-switch gesture is in
    /// flight. Parallel to the caller-side gate the IPC `switch_activity`
    /// dispatcher enforces.
    HardBlockedCascade {
        activity_name: String,
        block: super::ActivitySwitchBlock,
    },
}

impl fmt::Display for ReloadActivityRemovalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExclusiveWorkspaceHasWindows {
                activity_name,
                workspace_id,
            } => write!(
                f,
                "activity {activity_name:?} owns exclusive workspace {workspace_id:?} with \
                 windows; close or move them first",
            ),
            Self::WouldEmptyPool { activity_name } => write!(
                f,
                "removing activity {activity_name:?} would empty the pool; at least one \
                 activity must remain",
            ),
            Self::HardBlockedCascade {
                activity_name,
                block,
            } => write!(
                f,
                "cannot cascade off active activity {activity_name:?}: activity switch blocked \
                 by {block}",
            ),
        }
    }
}

impl std::error::Error for ReloadActivityRemovalError {}

/// Ordered pool of [`Activity`]s plus active / previous cursors.
///
/// Invariants:
/// - `map` is non-empty — guaranteed by [`Activities::new`] taking a seed `Activity`; no `Default`,
///   no push-to-empty API.
/// - `active` is always a key in `map`.
/// - `previous`, if `Some`, is always a key in `map`.
/// - `previous`, if `Some`, is never equal to `active` (distinctness). The no-op fast-path in
///   [`Activities::set_active`] preserves this: when `target == active` the call returns early
///   without touching `previous`; otherwise `previous` is set to the old active (guaranteed `!=
///   target`), so after the write `previous != active` still holds.
/// - Each stored `Activity`'s `id` equals its key in `map` (enforced by private `Activity.id` +
///   construction-only id minting; inserts go through `map.insert(activity.id, activity)`
///   exclusively).
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
    /// Empty input yields a single runtime "Default" activity (
    /// backwards-compat: a config with no `activity` blocks must behave
    /// identically to today's single-activity world).
    ///
    /// Non-empty input: the first entry becomes the seed (active cursor), the
    /// rest are inserted in declaration order via [`Self::insert`]. All entries
    /// are flagged `is_config_declared`. After construction, `active_id()`
    /// equals the id of the `Activity` minted from the first config entry, and
    /// `previous_id() == None`.
    pub fn from_config_or_default(config_activities: &[niri_config::ActivityDecl]) -> Self {
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
    /// (seed); runtime multi-activity population arrives via the
    /// `CreateActivity` action, and config-declared multi-activity population
    /// arrives via [`Self::from_config_or_default`].
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

    /// Validate `name` and, on success, mint a fresh runtime [`Activity`] and insert
    /// it into the pool, returning the new id.
    ///
    /// Rejection rules (both are pure inspections; on either error the pool is
    /// left untouched):
    ///
    /// - `name.trim().is_empty()` → [`CreateActivityError::EmptyName`]. Trimming before the check
    ///   matches user intent — a whitespace-only name is not a legitimate distinct activity
    ///   identifier.
    /// - Any existing activity's name equals `name` case-insensitively (via
    ///   `str::eq_ignore_ascii_case`) → [`CreateActivityError::DuplicateName`]. Mirrors the
    ///   collision policy in [`Self::resolve_config_names`] and
    ///   `niri_config::ActivityName::raw_decode`.
    ///
    /// On success, delegates insertion to [`Self::insert`]; the new activity is
    /// `is_config_declared == false` and carries an empty `views` map. The pool's
    /// `active` / `previous` cursors are not touched — creation is independent of
    /// focus transitions.
    pub(super) fn create_runtime(
        &mut self,
        name: String,
    ) -> Result<ActivityId, CreateActivityError> {
        if name.trim().is_empty() {
            return Err(CreateActivityError::EmptyName);
        }
        if self
            .map
            .values()
            .any(|a| a.name().eq_ignore_ascii_case(&name))
        {
            return Err(CreateActivityError::DuplicateName);
        }
        let activity = Activity::new_runtime(name);
        let id = activity.id();
        self.insert(activity);
        Ok(id)
    }

    /// Validate `name` and rename the activity identified by `id` in place.
    ///
    /// Rejection rules (both are pure inspections; on either error the pool is
    /// left untouched):
    ///
    /// - `name.trim().is_empty()` → [`RenameActivityError::EmptyName`]. Mirrors the trim-then-check
    ///   rule in [`Self::create_runtime`].
    /// - Any *other* existing activity's name equals `name` case-insensitively (via
    ///   `str::eq_ignore_ascii_case`) → [`RenameActivityError::DuplicateName`]. The target activity
    ///   itself is excluded from the scan, so renaming to a case variant of its current name (e.g.
    ///   "Beta" → "beta") or to its exact current name (no-op) succeeds.
    ///
    /// Config-declared and not-found checks are enforced by the outer
    /// [`Layout::rename_activity`] wrapper; this method mutates whatever
    /// activity `id` resolves to and trusts `id` to be a live key in the pool.
    ///
    /// Panics if `id` is not a live key in the map — caller must resolve before
    /// calling this method.
    pub(super) fn rename_runtime(
        &mut self,
        id: ActivityId,
        name: String,
    ) -> Result<(), RenameActivityError> {
        if name.trim().is_empty() {
            return Err(RenameActivityError::EmptyName);
        }
        // Exclude the target id from the collision scan so that renaming to a
        // case variant of the target's own name (or its exact current name)
        // succeeds — otherwise every rename would self-collide.
        let is_dup = self
            .map
            .iter()
            .any(|(other_id, other)| *other_id != id && other.name().eq_ignore_ascii_case(&name));
        if is_dup {
            return Err(RenameActivityError::DuplicateName);
        }
        self.map
            .get_mut(&id)
            .expect("id must be a live key in the map — caller must resolve before rename")
            .set_name(name);
        Ok(())
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

    /// Find an activity whose name matches `name` case-insensitively
    /// (`str::eq_ignore_ascii_case`), in declaration order.
    ///
    /// Mirrors the matching rule used by [`Self::resolve_config_names`] and
    /// the `niri_config::ActivityName` duplicate detector. Returns `None` if
    /// no activity carries the requested name; callers (e.g. the
    /// `open-on-activity` window-rule resolver in `send_initial_configure`)
    /// own the silent-fallback to the active activity (liberal-accept,
    /// mirroring `open-on-output`'s precedent for unknown output names).
    pub fn find_by_name(&self, name: &str) -> Option<&Activity> {
        self.map
            .values()
            .find(|a| a.name().eq_ignore_ascii_case(name))
    }

    pub fn get_mut(&mut self, id: ActivityId) -> Option<&mut Activity> {
        self.map.get_mut(&id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Activity> {
        self.map.values()
    }

    /// Mutable view over every [`Activity`] in declaration order. Mirrors
    /// [`Self::iter`] for the shared case. Used by `Layout::remove_activity`
    /// to patch per-activity `views` during exclusive-workspace destruction,
    /// so every activity's stale `WorkspaceView` entries drop in the same pass.
    pub(super) fn iter_mut(&mut self) -> impl Iterator<Item = &mut Activity> + '_ {
        self.map.values_mut()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn contains(&self, id: ActivityId) -> bool {
        self.map.contains_key(&id)
    }

    /// Resolve a list of config-declared activity names against this pool.
    ///
    /// Matching is case-insensitive (`str::eq_ignore_ascii_case`), mirroring
    /// the duplicate-detection rule in `niri_config::ActivityName::raw_decode`.
    /// Duplicate names in `names` collapse into the returned `HashSet`.
    ///
    /// Returns `(resolved_ids, unknown_names)`: resolved ids for entries that
    /// matched, and the untouched unknown names (preserving their input
    /// spelling) for the caller to `warn!` about. This method is pure and
    /// logs nothing itself — callers own the diagnostic surface.
    pub(super) fn resolve_config_names(
        &self,
        names: &[String],
    ) -> (HashSet<ActivityId>, Vec<String>) {
        let mut resolved = HashSet::new();
        let mut unknown = Vec::new();
        for name in names {
            match self
                .map
                .values()
                .find(|a| a.name().eq_ignore_ascii_case(name))
            {
                Some(activity) => {
                    resolved.insert(activity.id());
                }
                None => unknown.push(name.clone()),
            }
        }
        (resolved, unknown)
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

    /// Append a fresh config-declared [`Activity`] with `name` to the pool,
    /// returning the new id.
    ///
    /// No validation: callers (`Layout::reconcile_activities_on_reload_add`)
    /// have already resolved the name against the existing pool via
    /// [`Self::resolve_config_names`] and confirmed it is unknown. Parse-time
    /// `ActivityNameSet` is the primary defense against collisions among
    /// config-declared names; this method adds a debug-assert as secondary
    /// defense.
    ///
    /// The new activity carries `is_config_declared == true`, an empty
    /// `views` map, and has no effect on the pool's `active` / `previous`
    /// cursors — the reload cursor cascade is performed by the removal-side
    /// reload entry point.
    ///
    /// Panics (debug only) if any existing activity's name matches `name`
    /// case-insensitively.
    pub(super) fn add_config_declared(&mut self, name: String) -> ActivityId {
        debug_assert!(
            !self
                .map
                .values()
                .any(|a| a.name().eq_ignore_ascii_case(&name)),
            "add_config_declared: name {name:?} collides with an existing activity; \
             caller must resolve first",
        );
        let activity = Activity::new_config_declared(name);
        let id = activity.id();
        self.insert(activity);
        id
    }

    /// Flip `is_config_declared` to `true` on the activity identified by `id`.
    ///
    /// Used by the config-reload path when an existing runtime activity's
    /// name matches a newly-declared config activity: its id, stored name
    /// (with its runtime casing — NOT overwritten with the config entry's
    /// spelling), per-output `views`, and every workspace's `activities` set
    /// referencing it are preserved unchanged ( bullet 1). Only the
    /// config-declared flag flips.
    ///
    /// Panics (debug only) if `id` is not a live key in the pool — caller
    /// must resolve before calling.
    pub(super) fn promote_to_config_declared(&mut self, id: ActivityId) {
        debug_assert!(
            self.map.contains_key(&id),
            "promote_to_config_declared: id {id:?} is not a live key in the map",
        );
        self.map
            .get_mut(&id)
            .expect("id must be a live key in the map — caller must resolve before promote")
            .is_config_declared = true;
    }

    /// Reorder the pool so ids in `config_order` occupy positions `[0, N)`
    /// in the given order, preserving the relative order of any activity not
    /// in `config_order` after the prefix.
    ///
    /// **Precondition (caller regime):** every id in `config_order` must
    /// already sit at or after its target position in the map — i.e.
    /// `old_pos >= target_pos` for each entry. This is satisfied by the
    /// additive reload path, which walks config entries in order and places
    /// each at a monotonically advancing target; a different caller that can
    /// move entries *forward* must not reuse this method without analysis.
    ///
    /// Under that precondition, after `move_index(old, i)` the entry at `old`
    /// is relocated to position `i` and entries in `[i, old)` shift right by
    /// one (no other relative order is disturbed). Walking `config_order` with
    /// an incrementing target position `i = 0..config_order.len()` therefore
    /// yields the desired `[prefix, trailer]` layout.
    ///
    /// Debug-asserts every id in `config_order` is a live, config-declared
    /// key listed exactly once, and that `config_order` covers all
    /// config-declared activities.
    pub(super) fn reorder_to_match_config(&mut self, config_order: &[ActivityId]) {
        // Uniqueness and presence checks run in all builds (not just debug)
        // because a duplicate id would cause move_index to silently relocate
        // the same entry twice and corrupt the order. N ≤ activity count,
        // so the O(N) cost is negligible.
        let mut seen: HashSet<ActivityId> = HashSet::with_capacity(config_order.len());
        for id in config_order {
            assert!(
                self.map.contains_key(id),
                "reorder_to_match_config: id {id:?} is not a live key in the map",
            );
            assert!(
                seen.insert(*id),
                "reorder_to_match_config: id {id:?} listed more than once",
            );
        }

        // Additional invariant assertions that are only cheap enough in debug.
        #[cfg(debug_assertions)]
        {
            for id in config_order {
                let a = self.map.get(id).unwrap();
                assert!(
                    a.is_config_declared(),
                    "reorder_to_match_config: id {id:?} is not config-declared",
                );
            }
            // config_order must list every config-declared activity exactly once
            // so that no config-declared activity is left in the trailer.
            let config_declared_count =
                self.map.values().filter(|a| a.is_config_declared()).count();
            assert_eq!(
                config_order.len(),
                config_declared_count,
                "reorder_to_match_config: config_order has {} ids but {} config-declared activities exist",
                config_order.len(),
                config_declared_count,
            );
        }

        for (target_pos, id) in config_order.iter().enumerate() {
            let old_pos = self
                .map
                .get_index_of(id)
                .expect("id must be live — confirmed by the uniqueness/presence walk above");
            if old_pos != target_pos {
                self.map.move_index(old_pos, target_pos);
            }
        }
    }

    /// Remove the activity identified by `id` from the pool, returning the
    /// extracted [`Activity`].
    ///
    /// Pure pool mutator: the caller owns all upstream sequencing
    /// (`Layout::remove_activity` cascades the active cursor via
    /// [`Layout::switch_activity`], destroys exclusive workspaces, and prunes
    /// shared ones before calling here).
    ///
    /// Uses `IndexMap::shift_remove` (not `swap_remove`) so declaration order
    /// is preserved — matches the insertion contract of
    /// [`Self::from_config_or_default`] / [`Self::insert`].
    ///
    /// Clears [`Self::previous_id`] if it pointed at the removed id. This
    /// covers both the cascade case (where `Layout::remove_activity` calls
    /// `switch_activity` away from the target, leaving `previous == Some(id)`)
    /// and any non-active case where `previous` already equaled `id`. The
    /// post-condition after this call is that `previous`, if `Some`, is still
    /// a live key in the map and is still `!= active` — preserving the pool
    /// invariants in every branch.
    ///
    /// Panics (debug only) if `id == active_id()` (the caller must cascade
    /// first), if `id` is not a live key in the map, or if removal would
    /// leave the pool empty.
    pub(super) fn remove(&mut self, id: ActivityId) -> Activity {
        debug_assert!(
            id != self.active,
            "Activities::remove: caller must cascade off the target before removing \
             (active == {:?})",
            id,
        );
        debug_assert!(
            self.map.contains_key(&id),
            "Activities::remove: id {:?} is not a live key in the map",
            id,
        );
        debug_assert!(
            self.map.len() > 1,
            "Activities::remove: cannot empty the pool (len == 1)",
        );

        let activity = self
            .map
            .shift_remove(&id)
            .expect("Activities::remove: id must be a live key (precondition: caller validates before calling)");
        if self.previous == Some(id) {
            self.previous = None;
        }
        activity
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
    fn resolve_config_names_case_insensitive_match() {
        let cfg = vec![
            niri_config::ActivityDecl {
                name: niri_config::ActivityName("Work".to_owned()),
            },
            niri_config::ActivityDecl {
                name: niri_config::ActivityName("Personal".to_owned()),
            },
        ];
        let acts = Activities::from_config_or_default(&cfg);
        let work_id = acts.iter().find(|a| a.name() == "Work").unwrap().id();
        let personal_id = acts.iter().find(|a| a.name() == "Personal").unwrap().id();

        let (resolved, unknown) =
            acts.resolve_config_names(&["work".to_owned(), "personal".to_owned()]);
        assert_eq!(resolved, HashSet::from([work_id, personal_id]));
        assert!(unknown.is_empty());
    }

    #[test]
    fn resolve_config_names_unknown_returns_in_unknowns_list() {
        let cfg = vec![
            niri_config::ActivityDecl {
                name: niri_config::ActivityName("Work".to_owned()),
            },
            niri_config::ActivityDecl {
                name: niri_config::ActivityName("Personal".to_owned()),
            },
        ];
        let acts = Activities::from_config_or_default(&cfg);
        let work_id = acts.iter().find(|a| a.name() == "Work").unwrap().id();

        let (resolved, unknown) =
            acts.resolve_config_names(&["Work".to_owned(), "DoesNotExist".to_owned()]);
        assert_eq!(resolved, HashSet::from([work_id]));
        assert_eq!(unknown, vec!["DoesNotExist".to_owned()]);
    }

    #[test]
    fn activities_from_config_or_default_multiple_preserves_order_and_first_active() {
        let cfg = vec![
            niri_config::ActivityDecl {
                name: niri_config::ActivityName("Work".to_owned()),
            },
            niri_config::ActivityDecl {
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
