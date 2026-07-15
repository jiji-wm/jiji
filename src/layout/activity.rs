//! Activity / workspace-view types.
//!
//! `Workspace<W>` values live in `Layout.workspaces: HashMap<WorkspaceId, Workspace<W>>`.
//! Per-output [`WorkspaceView`]s live in `Activity.views: HashMap<OutputId, WorkspaceView>`.
//! For the active activity, the `views` key domain equals `{ OutputId::new(&mon.output) | mon ∈
//! Layout.monitors }`; every id in any view's `ids()` is a key in `Layout.workspaces`.
//! Inactive activities carry a dormant snapshot of their views across activity switches.
//! The active-activity invariant is enforced in `Layout::verify_invariants`.
//! `Layout`'s activity-orchestration `impl` block — the driver that operates on
//! these types — is also hosted in this file.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::rc::Rc;

use indexmap::IndexMap;
use jiji_config::WorkspaceReference;
use jiji_ipc::ActivityReferenceArg;
use smithay::output::Output;

use super::monitor::{ActivitySwitch, SlideDirection, WorkspaceSwitch};
use super::workspace::{OutputId, Workspace, WorkspaceId};
use super::{
    ActivitySwitchBlock, FocusWorkspaceInActivityError, Layout, LayoutElement, Options,
    ToggleWorkspaceStickyOutcome,
};
use crate::animation::Animation;
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

    /// Borrow the workspace at position `pos` in this view.
    ///
    /// Panics if `pos` is out of bounds for `self.ids()` or if the id is absent from `pool`;
    /// both indicate a broken pool/view invariant, not user error.
    pub fn workspace_at<'a, W: LayoutElement>(
        &self,
        pool: &'a HashMap<WorkspaceId, Workspace<W>>,
        pos: usize,
    ) -> &'a Workspace<W> {
        let id = self.ids()[pos];
        pool.get(&id).expect("view id must be a key in the pool")
    }

    /// Mutably borrow the workspace at position `pos` in this view.
    ///
    /// Panics on the same conditions as [`workspace_at`](Self::workspace_at).
    pub fn workspace_at_mut<'a, W: LayoutElement>(
        &self,
        pool: &'a mut HashMap<WorkspaceId, Workspace<W>>,
        pos: usize,
    ) -> &'a mut Workspace<W> {
        let id = self.ids()[pos];
        pool.get_mut(&id)
            .expect("view id must be a key in the pool")
    }

    pub fn active_workspace_ref<'a, W: LayoutElement>(
        &self,
        pool: &'a HashMap<WorkspaceId, Workspace<W>>,
    ) -> &'a Workspace<W> {
        self.workspace_at(pool, self.active_position())
    }

    /// Finds the workspace in this view whose name matches `workspace_name`, case-insensitively
    /// (`str::eq_ignore_ascii_case`).
    pub fn find_named_workspace<'a, W: LayoutElement>(
        &self,
        pool: &'a HashMap<WorkspaceId, Workspace<W>>,
        workspace_name: &str,
    ) -> Option<&'a Workspace<W>> {
        self.ids().iter().find_map(|id| {
            let ws = pool.get(id).expect("view id must be a key in the pool");
            ws.name
                .as_ref()
                .is_some_and(|name| name.eq_ignore_ascii_case(workspace_name))
                .then_some(ws)
        })
    }

    pub fn active_workspace<'a, W: LayoutElement>(
        &self,
        pool: &'a mut HashMap<WorkspaceId, Workspace<W>>,
    ) -> &'a mut Workspace<W> {
        self.workspace_at_mut(pool, self.active_position())
    }

    pub fn windows<'a, W: LayoutElement>(
        &'a self,
        pool: &'a HashMap<WorkspaceId, Workspace<W>>,
    ) -> impl Iterator<Item = &'a W> + 'a {
        self.ids()
            .iter()
            .map(move |id| pool.get(id).expect("view id must be a key in the pool"))
            .flat_map(|ws| ws.windows())
    }

    pub fn has_window<W: LayoutElement>(
        &self,
        pool: &HashMap<WorkspaceId, Workspace<W>>,
        window: &W::Id,
    ) -> bool {
        self.windows(pool).any(|win| win.id() == window)
    }

    pub fn active_window<'a, W: LayoutElement>(
        &self,
        pool: &'a HashMap<WorkspaceId, Workspace<W>>,
    ) -> Option<&'a W> {
        self.active_workspace_ref(pool).active_window()
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
///
/// `last_active_seq` records the [`Activities::activation_counter`] value at the time this
/// activity last became active. `0` means never activated (only possible for non-seed activities
/// that have not yet been switched to this session). The seed activity is stamped `1` at pool
/// construction. Higher values indicate more-recent activation; clients sort descending for MRU
/// order.
#[derive(Debug)]
pub struct Activity {
    id: ActivityId,
    name: String,
    is_config_declared: bool,
    /// Per-output workspace views. For the active activity, the key domain equals connected
    /// monitors' `OutputId`s; for inactive activities this is a dormant snapshot. See struct doc.
    views: HashMap<OutputId, WorkspaceView>,
    /// Monotonic activation sequence number. Set to the pool's `activation_counter` value each
    /// time this activity becomes active via [`Activities::set_active`]. `0` = not yet activated
    /// this session. The seed is initialized to `1`; subsequent flips stamp `>= 2`.
    last_active_seq: u64,
}

impl Activity {
    pub fn new_runtime(name: String) -> Self {
        Self {
            id: ActivityId::next(),
            name,
            is_config_declared: false,
            views: HashMap::new(),
            last_active_seq: 0,
        }
    }

    pub fn new_config_declared(name: String) -> Self {
        Self {
            id: ActivityId::next(),
            name,
            is_config_declared: true,
            views: HashMap::new(),
            last_active_seq: 0,
        }
    }

    /// Returns the monotonic activation sequence number for this activity.
    ///
    /// `0` means the activity has not been activated this session. Higher values
    /// indicate more-recent activation; sort descending for MRU order.
    pub fn last_active_seq(&self) -> u64 {
        self.last_active_seq
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

/// Validation failure shared by `Layout::set_workspace_sticky`,
/// `Layout::unset_workspace_sticky`, and `Layout::toggle_workspace_sticky`.
/// Single-variant: the only failure mode across all three entry points is a
/// workspace reference that does not resolve (or `None` was supplied and
/// there is no active workspace, i.e. zero connected monitors). The wrapping
/// `DoActionError` outer variant (`SetWorkspaceSticky` /
/// `UnsetWorkspaceSticky` / `ToggleWorkspaceSticky`), not this payload,
/// identifies which verb failed.
///
/// (Pre-harmonization the dispatch layer intercepted this as a silent no-op;
/// the intercept was dropped to harmonize the workspace-miss contract across
/// the activity-action cohort.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkspaceStickyError {
    /// No workspace in the pool matches the supplied reference (or `None`
    /// was supplied and there is no active workspace, i.e. zero connected
    /// monitors).
    WorkspaceNotFound,
}

impl fmt::Display for WorkspaceStickyError {
    /// Plain lowercase token. The token is the entire envelope;
    /// `format_do_action_error` does no further wrapping. Token drift will
    /// fail `do_action_error_envelope_matches_wire_contract`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WorkspaceNotFound => f.write_str("workspace not found"),
        }
    }
}

impl std::error::Error for WorkspaceStickyError {}

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
/// - `activation_counter` starts at `1` and is incremented on every real flip in
///   [`Self::set_active`]. The seed activity's `last_active_seq` is initialized to `1` at
///   construction time, so the first real flip stamps `2` — no seq value is ever reused.
#[derive(Debug)]
// No `is_empty` — `Activities` is never empty by construction.
#[allow(clippy::len_without_is_empty)]
pub struct Activities {
    map: IndexMap<ActivityId, Activity>,
    active: ActivityId,
    previous: Option<ActivityId>,
    /// Monotonically increasing counter bumped on every real activity flip. Starts at `1`;
    /// the seed activity's `last_active_seq` is set to `1` at construction. Each subsequent
    /// real flip increments this first, then stamps the new active activity.
    activation_counter: u64,
}

impl Activities {
    /// Seed with the first (default) activity. After construction,
    /// `active_id() == seed.id`, `previous_id() == None`, and the seed
    /// holds `last_active_seq == 1`. Mirrors
    /// [`WorkspaceView::new`]'s non-empty-by-construction discipline.
    pub fn new(mut seed: Activity) -> Self {
        // Start the counter at 1 and stamp the seed so it always holds the maximum
        // seq in a fresh pool. The first real flip will bump the counter to 2 and
        // stamp the target — no seq value is ever shared between two activities.
        let activation_counter = 1;
        seed.last_active_seq = activation_counter;
        let active = seed.id;
        let mut map = IndexMap::new();
        map.insert(seed.id, seed);
        Self {
            map,
            active,
            previous: None,
            activation_counter,
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
    pub fn from_config_or_default(config_activities: &[jiji_config::ActivityDecl]) -> Self {
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
    ///   `jiji_config::ActivityName::raw_decode`.
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
    /// the `jiji_config::ActivityName` duplicate detector. Returns `None` if
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

    /// Position of `id` in the pool, in declaration / creation order.
    ///
    /// Reflects the `IndexMap` insertion order — the same order the user sees
    /// in IPC listings and that determines slide direction on an activity switch.
    /// Returns `None` when `id` is not a live key.
    pub(super) fn position_of(&self, id: ActivityId) -> Option<usize> {
        self.map.get_index_of(&id)
    }

    /// Resolve a list of config-declared activity names against this pool.
    ///
    /// Matching is case-insensitive (`str::eq_ignore_ascii_case`), mirroring
    /// the duplicate-detection rule in `jiji_config::ActivityName::raw_decode`.
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
    /// without touching `previous` or `activation_counter`. Otherwise:
    /// `activation_counter` is incremented, `target.last_active_seq` is
    /// stamped with the new counter value, `previous = Some(old_active)`, and
    /// `active = target`; since `target != old_active` in that branch, the
    /// `previous != active` distinctness invariant is re-established.
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
        self.activation_counter += 1;
        self.map
            .get_mut(&target)
            .expect("set_active target must be a live key in the map")
            .last_active_seq = self.activation_counter;
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

        // No recency cleanup needed: `shift_remove` drops the entry entirely,
        // so its `last_active_seq` is gone with it. The remaining activities'
        // seq values stay valid — they are compared relative to each other and
        // `activation_counter` is never reset, so no renumbering is needed.
        let activity = self
            .map
            .shift_remove(&id)
            .expect("Activities::remove: id must be a live key (precondition: caller validates before calling)");
        if self.previous == Some(id) {
            self.previous = None;
        }
        activity
    }

    /// Returns activity ids sorted by `last_active_seq` descending (most-recently-activated
    /// first).
    ///
    /// Ties (both activities have the same `last_active_seq`) are broken by declaration order —
    /// `IndexMap` iteration is insertion order, and `sort_by` is stable, so equal-seq entries
    /// preserve their relative insertion order. In practice this only occurs when multiple
    /// activities share `seq == 0` (never activated this session); the active activity always
    /// holds the unique maximum seq, so it is always `result[0]`.
    ///
    /// Declaration order is not reordered for recency: the `IndexMap` pool remains in
    /// insertion order; recency is computed on read via this method.
    pub fn recency_ordered(&self) -> Vec<ActivityId> {
        let mut ids = Vec::from_iter(self.map.keys().copied());
        ids.sort_by(|&a, &b| {
            let seq_a = self.map[&a].last_active_seq;
            let seq_b = self.map[&b].last_active_seq;
            seq_b.cmp(&seq_a)
        });
        ids
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

        // Recency invariant: the active activity always holds the maximum last_active_seq
        // (guaranteed by the monotonically-increasing counter in `set_active`; newest stamp
        // wins), so it must be first in the recency-ordered list.
        {
            let ordered = self.recency_ordered();
            assert_eq!(
                ordered.first().copied(),
                Some(self.active),
                "active must hold max last_active_seq (ordered[0] == active)",
            );
        }

        // Corollary: when `previous` is set and at least two activities carry a non-zero seq
        // (i.e., have been activated at least once this session), the entry with the
        // second-highest seq must equal `previous`. The assertion catches any future mutation
        // path that forgets to stamp.
        if self.previous.is_some() {
            let activated_count = self.map.values().filter(|a| a.last_active_seq > 0).count();
            if activated_count >= 2 {
                let ordered = self.recency_ordered();
                // ordered[0] == active (max seq); ordered[1] == previous (second-highest seq).
                assert_eq!(
                    ordered.get(1).copied(),
                    self.previous,
                    "second-highest last_active_seq entry must equal previous \
                     (Activities recency invariant)",
                );
            }
        }
    }
}

/// Outcome of [`Layout::detach_ws_from_activity_view`]: what the holding-view
/// scan found and, if a view was patched, how. The payload asymmetry between
/// the two patched arms is deliberate: only `DroppedView` gates on output
/// connectivity (a whole-view drop is only reinstate-worthy on a connected
/// monitor), while `RemovedAt` always records a collapse candidate (a
/// still-multi-entry view is always eligible for the post-loop bookend
/// normalization regardless of connectivity).
#[derive(Debug)]
enum DetachOutcome {
    /// No view in the activity held the workspace. Legitimate only when
    /// `self.monitors` is empty — the coherence `debug_assert!` inside
    /// `detach_ws_from_activity_view` guards the connected-monitor case.
    NoView,
    /// The workspace was the view's sole entry, so the whole view was
    /// dropped. `connected` mirrors whether the view's output was a
    /// currently-connected monitor at the time of the drop — the caller
    /// gates its reinstate flag on it.
    DroppedView { connected: bool },
    /// `WorkspaceView::remove_at` ran on a still-multi-entry view. The
    /// caller records the view's output as a collapse candidate.
    RemovedAt(OutputId),
}

impl<W: LayoutElement> Layout<W> {
    /// Locate the pool workspace owning `window` and return its id plus a
    /// clone of its `activities` set. Returns `None` if the window id is not
    /// present anywhere in the pool. Used by the `Action::FocusWindow`
    /// dispatcher to compute a target activity for a window that lives on a
    /// dormant workspace.
    ///
    /// Trait-agnostic by design: the `Mapped`-specific `last_focused_activity`
    /// hint is read at the call site and passed into
    /// [`Self::pick_activity_for_hidden_window`] separately, so that
    /// `Layout<TestWindow>` tests of the picker do not need a
    /// `last_focused_activity` plumb on `TestWindow`.
    pub(crate) fn window_ws_and_activity_hint(&self, window: &W::Id) -> Option<WorkspaceId> {
        self.workspaces.values().find_map(|ws| {
            if ws.windows().any(|w| w.id() == window) {
                Some(ws.id())
            } else {
                None
            }
        })
    }

    /// Pick a non-active target activity for an `Action::FocusWindow { id }`
    /// dispatch that resolved a window on a hidden workspace.
    ///
    /// Three-tier tiebreaker, in order — each tier excludes the currently-active
    /// activity so the picker never returns `active_id()` from a productive path
    /// (the caller already resolved the "visible" fast-path before reaching here):
    /// 1. `hint` if `ws.activities.contains(hint) && hint != active_id()`. `hint` is the callee's
    ///    `Mapped.last_focused_activity` — the activity that was active the last time the user
    ///    focused *this specific* window.
    /// 2. `Activities::previous_id()` if that id is in `ws.activities` and is not the current
    ///    active.
    /// 3. First activity in `self.activities.iter()` (IndexMap display order) whose id is in
    ///    `ws.activities` and is not the current active.
    ///
    /// Invariant relied on: every workspace's `activities` set is non-empty
    /// (seeded at construction, enforced by `Layout::verify_invariants`). If
    /// the set is empty we hit an `unreachable!` with that contract named —
    /// a silent `.first()?` or `_ => active_id()` fallback is a review-stop
    /// bug (fork CLAUDE.md).
    ///
    /// Degenerate tail: if the workspace is tagged *only* with the currently
    /// active activity (and the caller invoked the picker anyway instead of
    /// short-circuiting on the visibility fast-path), every tier filters the
    /// lone candidate out and the function returns `active_id()`. This arm is
    /// not reached from the production dispatch path — the caller is expected
    /// to detect `target == active_id()` and no-op.
    pub(crate) fn pick_activity_for_hidden_window(
        &self,
        ws_id: WorkspaceId,
        hint: Option<ActivityId>,
    ) -> ActivityId {
        let active = self.activities.active_id();
        let ws = self
            .workspaces
            .get(&ws_id)
            .expect("ws_id must be a key in the pool (caller resolved it via windows_all)");
        let activities = ws.activities();

        if activities.is_empty() {
            unreachable!(
                "workspace.activities must be non-empty (Layout invariant, verify_invariants)"
            );
        }

        // Tier 1: MRU hint.
        if let Some(h) = hint {
            if h != active && activities.contains(&h) {
                return h;
            }
        }

        // Tier 2: Activities::previous_id.
        if let Some(prev) = self.activities.previous_id() {
            if prev != active && activities.contains(&prev) {
                return prev;
            }
        }

        // Tier 3: first activity in declaration order whose id is in ws.activities.
        for act in self.activities.iter() {
            let id = act.id();
            if id != active && activities.contains(&id) {
                return id;
            }
        }

        // Degenerate tail — see rustdoc above. Not reached from the production
        // dispatch path; the caller short-circuits on `target == active_id()`.
        active
    }

    /// Returns `Some(reason)` when the three live-input conditions in the
    /// activities design forbid switching activities — interactive window move, DnD, or a
    /// workspace-switch gesture in flight on *any* monitor. Returns `None`
    /// otherwise; in particular, an in-flight `WorkspaceSwitch::Animation` is
    /// not a hard block (it is snapped by `switch_activity` itself).
    ///
    /// The keybinding and IPC dispatch sites consult this reader and drop /
    /// queue the action accordingly. The `debug_assert!` at the top of
    /// [`Self::switch_activity`] pins that no caller bypasses the gate.
    pub(crate) fn is_activity_switch_hard_blocked(&self) -> Option<ActivitySwitchBlock> {
        if self.interactive_move.is_some() {
            return Some(ActivitySwitchBlock::InteractiveMove);
        }
        if self.dnd.is_some() {
            return Some(ActivitySwitchBlock::Dnd);
        }
        // Gesture can be in flight on any monitor, not just the active one.
        for mon in &self.monitors {
            if matches!(mon.workspace_switch, Some(WorkspaceSwitch::Gesture(_))) {
                return Some(ActivitySwitchBlock::WorkspaceSwitchGesture);
            }
        }
        None
    }

    /// Flip the active activity cursor to `target`.
    ///
    /// Hard-block conditions from (interactive move, DnD, workspace-
    /// switch gesture) MUST be filtered by the caller via
    /// [`Self::is_activity_switch_hard_blocked`]; the entry `debug_assert!`
    /// below pins this contract.
    ///
    /// Ordering:
    /// 1. No-op fast-path on `ActivityId` equality — avoids a `HashMap` lookup on the dominant
    ///    real-input path (`target == active_activity_id()`). Load-bearing for the hot path.
    /// 2. Reject unknown `target` with `error!` + early return, follows the
    ///    early-return-with-`error!` pattern from `update_output_size` and
    ///    `update_render_elements`, with the offending id in the message for log correlation. Step
    ///    2→3 ordering is also pinned by the `debug_assert!` inside [`Activities::set_active`].
    /// 3. Flip cursors via [`Activities::set_active`].
    /// 4. Snap any in-flight `WorkspaceSwitch` on every monitor by setting it to `None`. Per this
    ///    is the load-bearing snap step of the snap+proceed contract for
    ///    `WorkspaceSwitch::Animation`. Gestures are filtered out by the caller-side hard-block
    ///    guard, so the only `Some(_)` reachable here is `Animation(_)`; the unconditional
    ///    `mon.workspace_switch = None` matches every other clear site in the codebase. Fractional
    ///    positions inside `WorkspaceSwitch` refer to positions in the previously-active activity's
    ///    view, so leaving them around after the flip would be incoherent.
    /// 5. Lazily populate the target activity's per-output views via
    ///    [`Self::ensure_all_activity_views`] — see step 3.
    ///
    /// # Focus restoration
    ///
    /// No focus is poked directly from here; the read path picks it up on the
    /// next `State::refresh`. [`Self::focus`] reads the active activity's view
    /// for the active monitor, and each workspace's own persisted
    /// `active_column_idx` / `active_tile_idx` selects the window. Global
    /// keyboard focus is reassigned by `State::refresh` →
    /// `update_keyboard_focus` before rendering; `active_monitor_idx` is not
    /// mutated by the switch, so focus returns to the output that held it.
    ///
    /// At the end of the switch, two post-conditions are pinned as
    /// `debug_assert!`s (release builds pay nothing):
    ///
    /// - `view.active ∈ view.ids()` for every entry in the active activity's `views()`. Today
    ///   [`WorkspaceView`] upholds this at construction and in every mutator; the assertion
    ///   documents the contract that the only public entry point flipping activities is required to
    ///   preserve.
    /// - `active_monitor_idx < monitors.len()` when monitors are non-empty. Monitors don't mutate
    ///   during a switch, so this is also a pin, not a fix — a drift would be a caller-side bug.
    pub fn switch_activity(&mut self, target: ActivityId) {
        debug_assert!(
            self.is_activity_switch_hard_blocked().is_none(),
            "switch_activity called while hard-blocked: caller must filter via \
             is_activity_switch_hard_blocked",
        );
        // Capture before set_active so we can record the departing strip.
        let outgoing = self.activities.active_id();
        if target == outgoing {
            return;
        }
        if !self.activities.contains(target) {
            error!("switch_activity: target {target:?} is not a live activity id");
            return;
        }
        self.activities.set_active(target);
        // Snap any in-flight WorkspaceSwitch::Animation (snap+proceed contract).
        // Gestures are filtered by the caller-side hard-block guard, so the only Some(_)
        // reachable here is Animation(_); the unconditional clear matches every other clear
        // site in the codebase. Not reached on the no-op or unknown-id paths above, which
        // both early-return before touching monitor state.
        for mon in &mut self.monitors {
            mon.workspace_switch = None;
        }
        self.ensure_all_activity_views();

        // Arm the activity-switch transition on each connected monitor.
        // Config gate: if the activity-switch animation is marked off, skip (zero new state).
        // Overview gate: skip when the overview is open — the overview already gives a
        // full spatial map; a simultaneous strip slide would compound confusingly.
        // Assignment over an existing Some(_) is the snap+restart contract — no continuity math.
        if !self.options.animations.activity_switch.0.off && !self.overview_open {
            // Direction: higher pool index → Left (incoming strip from the right).
            // Both positions are `.expect`-safe: `target` passed `contains` above;
            // `outgoing` was the active id, which is always a live key.
            let target_pos = self
                .activities
                .position_of(target)
                .expect("target passed contains check and must be in the pool");
            let outgoing_pos = self
                .activities
                .position_of(outgoing)
                .expect("outgoing was the active id and must be in the pool");
            let dir = if target_pos > outgoing_pos {
                SlideDirection::Left
            } else {
                SlideDirection::Right
            };
            for mon in &mut self.monitors {
                mon.activity_switch = Some(ActivitySwitch {
                    from: outgoing,
                    anim: Animation::new(
                        mon.clock.clone(),
                        0.,
                        1.,
                        0.,
                        self.options.animations.activity_switch.0,
                    ),
                    dir,
                });
            }
        }

        // Post-condition pins. See the "Focus restoration" rustdoc paragraph for rationale.
        #[cfg(debug_assertions)]
        {
            for (output_id, view) in self.activities.active().views() {
                debug_assert!(
                    view.ids().contains(&view.active()),
                    "switch_activity post-condition: view.active ({:?}) must be in view.ids \
                     for output {:?}",
                    view.active(),
                    output_id,
                );
            }
            if !self.monitors.is_empty() {
                debug_assert!(
                    self.active_monitor_idx < self.monitors.len(),
                    "switch_activity post-condition: active_monitor_idx {} out of range \
                     (monitors.len() = {})",
                    self.active_monitor_idx,
                    self.monitors.len(),
                );
            }
        }
    }

    /// Repair every activity's `views` so they agree with pool membership, in both directions.
    ///
    /// Total and repair-oriented (debug-loud, release-healable): it installs a view entry for
    /// every (activity, workspace) membership that lacks one, and removes every view entry whose
    /// workspace no longer carries that activity in its `activities` set. Callers run it after a
    /// membership-mutating pass (config-reload reset, sticky (re-)expansion, activity creation)
    /// has left views and membership out of agreement.
    ///
    /// Ordering is load-bearing:
    ///
    /// - **Installs run before removals.** A workspace that moved wholesale from activity `L` to
    ///   activity `A` must resolve its holding output via [`Self::workspace_holding_output`] while
    ///   `L`'s view still holds it. Removing from `L` first would strand the id in no view, the
    ///   resolver would return `None`, and the install would skip — minting the very
    ///   membership-without-view incoherence this method exists to close.
    /// - **Bookend repair for windowed/named installs runs after removals** so
    ///   [`Self::normalize_view_bookends`] sees final view positions.
    ///
    /// The install worklist is deterministic — activities walked in declaration order, workspaces
    /// sorted by `WorkspaceId.get()` (creation order) — so view order does not flap across runs.
    ///
    /// Every skip is `trace!` / `debug_assert!`-loud. Does not call `Self::verify_invariants` —
    /// callers do.
    pub(super) fn reconcile_views_with_membership(&mut self) {
        if self.monitors.is_empty() {
            // In the fully-disconnected window every activity's `views` map is empty and parked
            // workspaces live directly in `disconnected_workspace_ids` — nothing to reconcile.
            return;
        }

        let disconnected: HashSet<WorkspaceId> =
            self.disconnected_workspace_ids.iter().copied().collect();

        // Compute both worklists read-only before mutating anything.
        let mut installs: Vec<(WorkspaceId, ActivityId)> = Vec::new();
        let mut removals: Vec<(ActivityId, OutputId, WorkspaceId)> = Vec::new();
        for activity in self.activities.iter() {
            let act_id = activity.id();

            // Installs (membership → view): pool workspaces that carry this activity but are
            // absent from every one of its views. Sorted by creation order (mirrors the
            // `ensure_view_for` lift sort) so the resulting view order is deterministic.
            let union: HashSet<WorkspaceId> = activity
                .views()
                .values()
                .flat_map(|view| view.ids().iter().copied())
                .collect();
            let mut per_activity: Vec<WorkspaceId> = self
                .workspaces
                .iter()
                .filter(|(ws_id, ws)| {
                    ws.activities().contains(&act_id)
                        && !disconnected.contains(ws_id)
                        && !union.contains(ws_id)
                })
                .map(|(ws_id, _)| *ws_id)
                .collect();
            per_activity.sort_by_key(|id| id.get());
            for ws_id in per_activity {
                installs.push((ws_id, act_id));
            }

            // Removals (view → membership): view entries whose workspace no longer carries this
            // activity. Positions are re-resolved at removal time, so only the ids are recorded.
            for (out_id, view) in activity.views() {
                for &ws_id in view.ids() {
                    debug_assert!(
                        self.workspaces.contains_key(&ws_id),
                        "reconcile_views_with_membership: view id {ws_id:?} for activity \
                         {act_id:?} has no pool entry — zombie view id in reconcile removals",
                    );
                    let lacks_membership = self
                        .workspaces
                        .get(&ws_id)
                        .is_some_and(|ws| !ws.activities().contains(&act_id));
                    if lacks_membership {
                        removals.push((act_id, out_id.clone(), ws_id));
                    }
                }
            }
        }

        // Snap any in-flight animation on every monitor when the active activity's view length or
        // positions are about to change — the mutation can shrink or reshuffle the active view,
        // and a stale fractional switch target would trip the animation-bounds check in
        // `Monitor::verify_invariants` (`before_idx`/`after_idx < active_view.len()`) after it.
        let active_id = self.activities.active_id();
        let active_affected = installs.iter().any(|(_, act_id)| *act_id == active_id)
            || removals.iter().any(|(act_id, _, _)| *act_id == active_id);
        if active_affected {
            for mon in &mut self.monitors {
                if matches!(mon.workspace_switch, Some(WorkspaceSwitch::Animation(_))) {
                    mon.workspace_switch = None;
                }
            }
        }

        // Installs first: each membership without a view gets one, keyed by the output whose view
        // currently holds the workspace (in the losing activity, before any removal runs).
        let mut installed_windowed: Vec<OutputId> = Vec::new();
        for (ws_id, act_id) in installs {
            let Some(out_id) = self.workspace_holding_output(ws_id) else {
                trace!(
                    "reconcile_views_with_membership: {ws_id:?} has no holding view; \
                     membership→view install into {act_id:?} skipped",
                );
                continue;
            };
            let inserted = {
                let pool = &mut self.workspaces;
                let activities = &mut self.activities;
                let activity = activities
                    .get_mut(act_id)
                    .expect("act_id came from self.activities.iter()");
                if let Some(view) = activity.views_mut().get_mut(&out_id) {
                    Self::view_insert_above_trailing_bookend(pool, view, ws_id);
                    true
                } else {
                    // `out_id` names a currently-connected output (`workspace_holding_output`
                    // only returns view keys, and every view key is a connected output's id per
                    // the connected-keyspace invariant), and the per-activity bookend invariant
                    // guarantees every activity holds a view for every connected output — so this
                    // activity must have a view keyed by `out_id`. Reaching here means that
                    // invariant is violated (membership↔view coherence bug); skip rather than
                    // mint a membership-without-view incoherence on top of it.
                    debug_assert!(
                        false,
                        "reconcile_views_with_membership: activity {act_id:?} has no view for \
                         {out_id:?}, a connected output — per-activity bookend invariant \
                         violated (membership↔view coherence bug)",
                    );
                    false
                }
            };
            if inserted
                && self
                    .workspaces
                    .get(&ws_id)
                    .is_some_and(|ws| ws.has_windows_or_name())
            {
                installed_windowed.push(out_id);
            }
        }

        // Removals: drop view entries whose workspace lost the membership. Positions are
        // re-resolved per removal because an earlier removal in the same view shifts them.
        let mut dropped_any_view_entry = false;
        let mut removed_ids: Vec<WorkspaceId> = Vec::new();
        let mut affected_pairs: Vec<(ActivityId, OutputId)> = Vec::new();
        for (act_id, out_id, ws_id) in removals {
            // Every view key is a currently-connected output per the connected-keyspace invariant;
            // the check mirrors the runtime Remove patches' is-connected guard for the
            // drop-to-zero flag.
            let is_connected = self.monitors.iter().any(|m| m.output_id() == out_id);
            let activity = self
                .activities
                .get_mut(act_id)
                .expect("act_id came from self.activities.iter()");
            let Some(view) = activity.views_mut().get_mut(&out_id) else {
                trace!(
                    "reconcile_views_with_membership: view for {out_id:?} in {act_id:?} already \
                     gone; removal of {ws_id:?} skipped",
                );
                continue;
            };
            let Some(pos) = view.position_of(ws_id) else {
                trace!(
                    "reconcile_views_with_membership: {ws_id:?} already absent from {act_id:?}'s \
                     {out_id:?} view; removal skipped",
                );
                continue;
            };
            if view.len() == 1 {
                // Drop the single-entry view outright — mirrors the
                // `destroy_workspaces_cross_activity` single-entry retain-drop path.
                activity.views_mut().remove(&out_id);
                if is_connected {
                    dropped_any_view_entry = true;
                }
            } else {
                view.remove_at(pos);
            }
            removed_ids.push(ws_id);
            affected_pairs.push((act_id, out_id));
        }

        // Normalize any all-empty exclusive EWAF len-2 view a removal left behind. The collapse is
        // idempotent (a non-len-2 view early-returns), so duplicate pairs are harmless.
        for (act_id, out_id) in &affected_pairs {
            self.collapse_empty_exclusive_ewaf_len2_view(*act_id, out_id);
        }

        // A whole-view drop on a connected output breaks `active.views.len() == monitors.len()`;
        // the materializer reinstates the per-activity bookend view.
        if dropped_any_view_entry {
            self.ensure_all_activity_views();
        }

        // Bookend repair for windowed/named installs, after removals so the sweep sees final
        // positions. An empty install is itself a valid bookend and needs no repair.
        for out_id in &installed_windowed {
            if self.monitors.iter().all(|m| m.output_id() != *out_id) {
                debug_assert!(
                    false,
                    "reconcile_views_with_membership: installed view output {out_id:?} must be \
                     connected (connected-keyspace invariant)",
                );
            }
        }
        if !installed_windowed.is_empty() {
            self.normalize_view_bookends();
        }

        // A removal may have narrowed a workspace to a single activity while it sits in an illegal
        // middle position; reclaim any such orphaned pinned empties. Dedup so an id removed from
        // two activities isn't reclaimed twice (the second pass would `debug_assert!` on the
        // already-culled id).
        removed_ids.sort_by_key(|id| id.get());
        removed_ids.dedup();
        self.reclaim_unpinned_empty_workspaces(removed_ids);
    }

    /// Cull the second entry of an all-empty exclusive EWAF length-2 view, restoring the
    /// "1 or 3+ unless second shared" bookend rule.
    ///
    /// Shared by [`Self::reconcile_views_with_membership`] and the runtime membership-removal
    /// patches so all three sites apply one policy. Mirrors `clean_up_workspaces_on`'s EWAF
    /// special case and [`Self::reclaim_unpinned_empty_workspaces`]' EWAF resolution: resolve the
    /// EWAF flag from the connected monitor's merged options (layout-root fallback for
    /// dormant/disconnected views); if the view exists, is length 2, both entries are empty
    /// unnamed, and the second entry is safe to reclaim (exclusive to a single activity), snap the
    /// holding monitor's animation when `act_id` is active and destroy the second entry.
    ///
    /// Guards, not asserts: this is a repair path and must stay total. A windowed first entry
    /// under EWAF is already-illegal input, left for `Self::verify_invariants` to name.
    pub(super) fn collapse_empty_exclusive_ewaf_len2_view(
        &mut self,
        act_id: ActivityId,
        out_id: &OutputId,
    ) {
        let ewaf = self
            .monitors
            .iter()
            .find(|m| &m.output_id() == out_id)
            .map(|m| m.options.layout.empty_workspace_above_first)
            .unwrap_or(self.options.layout.empty_workspace_above_first);
        if !ewaf {
            return;
        }

        let ids = match self
            .activities
            .get(act_id)
            .and_then(|activity| activity.views().get(out_id))
        {
            Some(view) if view.len() == 2 => [view.ids()[0], view.ids()[1]],
            _ => return,
        };

        let both_empty = ids.iter().all(|id| {
            self.workspaces
                .get(id)
                .is_some_and(|ws| !ws.has_windows_or_name())
        });
        if !both_empty {
            return;
        }
        // A shared second entry is the legal minimal EWAF shape; only an exclusive one collapses.
        if !Self::workspace_is_safe_to_reclaim(&self.workspaces, ids[1]) {
            return;
        }

        if act_id == self.activities.active_id() {
            for mon in &mut self.monitors {
                if &mon.output_id() == out_id
                    && matches!(mon.workspace_switch, Some(WorkspaceSwitch::Animation(_)))
                {
                    mon.workspace_switch = None;
                }
            }
        }

        Self::destroy_workspaces_cross_activity(
            &mut self.activities,
            &mut self.workspaces,
            [ids[1]],
        );
    }

    // Make every activity's `views` map cover every connected monitor. Required by the
    // per-activity bookend invariant: every (activity, connected-monitor) pair must hold a
    // `WorkspaceView` whose first and last workspaces satisfy the trailing-empty (and EWAF
    // leading-empty) rule.
    //
    // Connected-keyspace invariant: in the connected steady state every activity's view keys are
    // currently-connected outputs. `remove_output`'s dormant-view drain (both its partial- and
    // full-disconnect branches) removes every activity's view entry for the disconnecting output,
    // so no dormant view keyed by a disconnected output survives to be re-materialized here — the
    // skip-existing guard below therefore only ever short-circuits on a view for a still-connected
    // output, never resurrects a stale one. Any residual disconnected-output view (a coherence
    // bug, not a steady state) is still caught by `Layout::verify_invariants`' disconnected-output
    // bookend pass.
    //
    // No-op when no monitors are connected: the `monitors.is_empty()` branch of
    // `verify_invariants` requires every activity's `views` map to be empty in that state, and
    // `disconnected_workspace_ids` then holds workspaces directly without per-activity views.
    //
    // Borrow discipline: we read `&self.monitors` once into a `Vec` of triples
    // `(OutputId, Option<Output>, Rc<Options>)` (`Option<Output>` is always `Some` today; the
    // `None` shape on `ensure_view_for` is reserved for a future caller that mints a view
    // without a live `Output` handle). After the snapshot we drop the borrow and only then
    // dispatch to [`Self::ensure_view_for`], which owns the pool/activities split-borrow
    // internally.
    pub(super) fn ensure_all_activity_views(&mut self) {
        if self.monitors.is_empty() {
            return;
        }

        let known_outputs: Vec<(OutputId, Option<Output>, Rc<Options>)> = self
            .monitors
            .iter()
            .map(|m| (m.output_id(), Some(m.output.clone()), m.options.clone()))
            .collect();

        let activity_ids: Vec<ActivityId> = self.activities.iter().map(|a| a.id()).collect();

        for activity_id in activity_ids {
            for (output_id, output_opt, options) in &known_outputs {
                let already = self
                    .activities
                    .get(activity_id)
                    .is_some_and(|a| a.views().contains_key(output_id));
                if already {
                    continue;
                }
                self.ensure_view_for(activity_id, output_id.clone(), output_opt.as_ref(), options);
            }
        }
    }

    /// Materialize a `WorkspaceView` entry on `activity_id`'s `views` map for
    /// `output_id` if and only if no entry exists yet.
    ///
    /// The body mirrors the per-output allocation discipline of
    /// [`Self::ensure_all_activity_views`] — pre-tagged candidates from the
    /// pool are lifted into the new view (sorted by `WorkspaceId.get()` as a
    /// placeholder for `cmp_by_config_then_creation`), padded with a fresh
    /// trailing empty so the bookend "last must be empty / unnamed" invariant
    /// holds; under `empty_workspace_above_first` for this output, a fresh
    /// leading empty is also prepended so the "first must be empty / unnamed"
    /// and "1 or 3+" length rules hold. The active index becomes 1 in the
    /// EWAF lift branch so the first lifted workspace stays selected.
    ///
    /// `output` is `Some(&Output)` for outputs that are currently connected
    /// (`mon_options` is then the per-monitor merged options, mirroring
    /// `Monitor::new`'s ctor — using `self.options` would silently violate
    /// the EWAF first/last-empty invariants for outputs whose
    /// `layout_config` flips the flag) and `None` for outputs that are known
    /// only because some activity's view is keyed by them (e.g. a stale
    /// dormant view for an output that disconnected while the activity was
    /// inactive). On the `None` branch we use `Workspace::new_no_outputs`
    /// and then patch `ws.output_id` so the materialized bookend can be
    /// reclaimed by a future `bind_output` (mirrors the established pattern
    /// in `reconcile_activities_on_reload_remove` at the orphan-rebind site).
    ///
    /// Used by [`Self::ensure_all_activity_views`] and by
    /// [`Self::view_in_activity_or_materialize`] (target = arbitrary
    /// activity, e.g. an inactive activity addressed by an
    /// `open-on-activity` window rule).
    ///
    /// Caller must ensure the entry does not yet exist (else the final
    /// insert asserts).
    pub(crate) fn ensure_view_for(
        &mut self,
        activity_id: ActivityId,
        output_id: OutputId,
        output: Option<&Output>,
        mon_options: &Rc<Options>,
    ) {
        assert!(
            !output_id.as_str().is_empty(),
            "ensure_view_for requires a non-sentinel OutputId",
        );
        let clock = self.clock.clone();

        // Helper: mint a fresh empty workspace for this (activity, output) pair. The `Some`
        // branch is the only one fired today — `ensure_all_activity_views` narrows
        // `known_outputs` to connected monitors. The `None` branch (using
        // `Workspace::new_no_outputs` and patching `output_id` to the known OutputId — same
        // pattern as the orphan-rebind in `reconcile_activities_on_reload_remove`) is reserved
        // for a future caller that needs to materialize a view without a live `Output` handle.
        let mint_empty = |workspaces: &mut HashMap<WorkspaceId, Workspace<W>>| -> WorkspaceId {
            let mut ws = match output {
                Some(o) => Workspace::new(
                    o,
                    HashSet::from([activity_id]),
                    clock.clone(),
                    mon_options.clone(),
                ),
                None => Workspace::new_no_outputs(
                    HashSet::from([activity_id]),
                    clock.clone(),
                    mon_options.clone(),
                ),
            };
            if output.is_none() {
                ws.output_id = Some(output_id.clone());
            }
            let id = ws.id();
            assert!(
                workspaces.insert(id, ws).is_none(),
                "fresh id must be unique",
            );
            id
        };

        // Collect pre-tagged candidates under a shared borrow of the pool, so we can
        // later take `&mut self.workspaces` without a conflicting active borrow.
        let mut tagged: Vec<WorkspaceId> = self
            .workspaces
            .values()
            .filter(|ws| {
                ws.output_id() == Some(&output_id) && ws.activities().contains(&activity_id)
            })
            .map(|ws| ws.id())
            .collect();
        // sort by WorkspaceId (monotonic creation counter) — placeholder for
        // cmp_by_config_then_creation which requires is_config_declared on Workspace.
        tagged.sort_by_key(|id| id.get());

        // Source-side dedup: every workspace we are about to lift into this fresh
        // (activity_id, output_id) view may already appear in another view of the same
        // activity. Typical case: the partial-disconnect walk in `remove_output` migrated
        // these ids into another `(activity_id, other_output)` view with `output_id`
        // preserved on the pool entry; this call is now materializing the
        // reconnecting output's view via `ensure_all_activity_views`. Without this drop the
        // workspace appears in both views and the primary-monitor "own monitor exists"
        // invariant fires at `verify_invariants` on the next `switch_activity`.
        //
        // Diverges intentionally from `move_workspace_to_output_by_id`'s source-side drop
        // pattern: that path collapses a single-entry view by removing the map entry
        // (`activity.views_mut().remove(&source_out_id)`); here we never do that. The
        // sibling view's trailing-empty bookend (and, under EWAF, the leading-empty
        // bookend) must survive in place, since that view is the canonical per-activity
        // bookend for `(activity_id, other_output)` and removing the map entry would
        // violate the per-activity bookend invariant. A single-entry view consisting only
        // of the trailing-empty bookend is structurally valid — `assert_view_bookends`
        // accepts last==empty-unnamed at len=1 and EWAF's "1 or 3+" rule permits len=1.
        // Empty case (fresh-mint path) has no lifted ids and therefore no dedup work.
        if !tagged.is_empty() {
            let activity = self
                .activities
                .get_mut(activity_id)
                .expect("activity id must be a live key");
            // Clone the keyset to release the inner shared borrow before taking the
            // mutable borrow on each entry inside the loop — iterating `views().keys()`
            // while calling `views_mut().get_mut(...)` would mix shared and mutable
            // borrows of the same `HashMap`.
            let other_output_ids: Vec<OutputId> = activity
                .views()
                .keys()
                .filter(|oid| **oid != output_id)
                .cloned()
                .collect();
            for other_out in &other_output_ids {
                let view = activity
                    .views_mut()
                    .get_mut(other_out)
                    .expect("collected key must still be present");
                for ws_id in &tagged {
                    if let Some(pos) = view.position_of(*ws_id) {
                        debug_assert!(
                            view.len() > 1,
                            "sibling view must retain at least one bookend after dedup — \
                             `tagged` must never contain a bookend of this view",
                        );
                        view.remove_at(pos);
                    }
                }
            }
        }

        let ewaf = mon_options.layout.empty_workspace_above_first;

        let view = if tagged.is_empty() {
            // Fresh branch: a single trailing empty satisfies all EWAF invariants at len=1.
            let id = mint_empty(&mut self.workspaces);
            WorkspaceView::new(vec![id], 0)
        } else {
            // Lift branch: pre-tagged candidates form the body. Append a fresh trailing empty
            // so the trailing-empty / trailing-unnamed bookend holds. Under EWAF for this
            // output, also prepend a fresh leading empty so the leading-empty / leading-unnamed
            // / "1 or 3+" rules hold; the active index becomes 1 so the first lifted workspace
            // stays selected.
            let bottom_id = mint_empty(&mut self.workspaces);

            let mut ids = tagged;
            let mut active_idx = 0usize;

            if ewaf {
                let top_id = mint_empty(&mut self.workspaces);
                ids.insert(0, top_id);
                active_idx = 1;
            }

            ids.push(bottom_id);

            WorkspaceView::new(ids, active_idx)
        };

        // Explicit contains_key → insert rather than `.entry().or_insert_with()`: the
        // closure arg would need to capture `&mut self.workspaces` via `||`, conflicting
        // with this outer `&mut self.activities` reborrow.
        let activity = self
            .activities
            .get_mut(activity_id)
            .expect("activity id must be a live key");
        assert!(
            activity.views_mut().insert(output_id, view).is_none(),
            "caller must ensure the entry does not yet exist",
        );
    }

    /// Ensure `activity_id`'s `views` map has a `WorkspaceView` entry for the
    /// connected monitor identified by `output_id`, materializing one through
    /// [`Self::ensure_view_for`] if needed.
    ///
    /// Used by `xdg_shell::send_initial_configure` to address a hidden
    /// activity targeted by an `open-on-activity` window rule.
    /// Side-effecting only; the materialized view can be read back via
    /// `self.activities.get(activity_id).views()[output_id]`. Returns `()`
    /// rather than `&WorkspaceView` so callers can keep their subsequent
    /// reads on `&self.layout` instead of holding a `&mut self.layout`
    /// borrow across send_configure (keeps the call site free of an
    /// outstanding `&mut self.layout` borrow).
    ///
    /// No-op when `output_id` is not connected: the `monitor_for_output_id`
    /// lookup returns `None` and we skip the materialize. Caller is
    /// responsible for handling the missing-entry case after the call (in
    /// practice: `send_initial_configure` will then fall through its
    /// precedence chain to a different target).
    pub(crate) fn view_in_activity_or_materialize(
        &mut self,
        activity_id: ActivityId,
        output_id: &OutputId,
    ) {
        if self
            .activities
            .get(activity_id)
            .is_some_and(|a| a.views().contains_key(output_id))
        {
            return;
        }
        let Some((output, options)) = self
            .monitor_for_output_id(output_id)
            .map(|mon| (mon.output.clone(), mon.options.clone()))
        else {
            // Output not connected — nothing to materialize against.
            return;
        };
        self.ensure_view_for(activity_id, output_id.clone(), Some(&output), &options);
    }

    /// Switch to a recently-active activity at recency position `depth`.
    ///
    /// `depth = 0` names the currently-active activity — explicit no-op, returns immediately.
    /// `depth = 1` (the legacy default) switches to the previously-active activity.
    /// `depth = N` switches to the activity at recency position N in the MRU list computed by
    /// [`Activities::recency_ordered`]. If `depth` exceeds the number of activated activities
    /// minus one, it is clamped to the oldest-activated activity (no error is returned).
    ///
    /// Activities with `last_active_seq == 0` (never activated this session) are excluded from
    /// the reachable target set; the clamp always lands on an activity that has been activated
    /// at least once.
    pub(crate) fn switch_activity_previous(&mut self, depth: u32) {
        // depth = 0 means "current active" by definition — nothing to do.
        if depth == 0 {
            return;
        }
        let ordered = self.activities.recency_ordered();
        let activated_count = ordered
            .iter()
            .filter(|&&id| {
                self.activities
                    .get(id)
                    .expect("recency_ordered ids must be live keys in the pool")
                    .last_active_seq()
                    > 0
            })
            .count();
        // Fewer than 2 activated activities means there is no other activated activity to
        // switch to — toggling on a pristine pool (only the seed ever stamped) must not
        // error, mirroring the pre-depth no-previous early-return.
        if activated_count <= 1 {
            return;
        }
        // Clamp depth to the index of the oldest-activated entry in the ordered list.
        let idx = depth.min(activated_count.saturating_sub(1) as u32) as usize;
        let target = ordered[idx];
        // depth >= 1 (checked above) + activated_count >= 2 (checked above) guarantees
        // idx >= 1, so target is never ordered[0] (the active activity, which holds the
        // maximum last_active_seq by the recency invariant).
        if target == self.activities.active_id() {
            unreachable!(
                "activated_count >= 2 and depth >= 1 guarantee idx >= 1; \
                 ordered[0] is always the active activity by the max-seq invariant, \
                 so ordered[idx] cannot equal active_id()"
            );
        }
        self.switch_activity(target);
    }

    /// Create a fresh runtime activity with `name`, returning its id.
    ///
    /// Validation is delegated to [`Activities::create_runtime`] — empty /
    /// whitespace-only names and case-insensitive duplicates are rejected
    /// without mutating any pool state (see [`CreateActivityError`]).
    ///
    /// On success, the new activity id is unioned into every sticky workspace's
    /// `activities` set. Sticky workspaces are "present on every activity" by
    /// definition; whenever the activity pool grows, those sets must grow with
    /// it, or the pool-level union invariant (each workspace's `activities` set
    /// is a subset of live activity ids) would diverge from the intended
    /// semantics. The new activity's `views` map is populated eagerly, not
    /// lazily on first switch: [`Self::ensure_all_activity_views`] materializes
    /// a bookend view for every connected monitor, then
    /// [`Self::reconcile_views_with_membership`] sweeps in any stale-tagged
    /// sticky workspaces the materializer's tag-filtered lift misses.
    ///
    /// The `active` / `previous` cursors on [`Activities`] are not touched; the
    /// caller remains on whatever activity was active before the call.
    pub(crate) fn create_activity(
        &mut self,
        name: String,
    ) -> Result<ActivityId, CreateActivityError> {
        let id = self.activities.create_runtime(name)?;
        for ws in self.workspaces.values_mut() {
            if ws.is_sticky() {
                ws.activities.insert(id);
            }
        }
        // Materialize a bookend view for the new activity on every connected monitor; the
        // per-activity bookend invariant requires it eagerly, not lazily on first switch.
        self.ensure_all_activity_views();

        // Sticky expansion above widened membership; the materializer's tag-filtered lift misses
        // stale-tagged sticky workspaces, so sweep them into the new activity's views. Only the
        // install direction can fire here — creation only widens membership.
        self.reconcile_views_with_membership();

        #[cfg(debug_assertions)]
        self.verify_invariants();

        Ok(id)
    }

    /// Remove the runtime activity identified by `reference`, returning its id
    /// on success.
    ///
    /// Rejection rules (all evaluated before any mutation — see):
    ///
    /// - Unknown reference → [`RemoveActivityError::NotFound`].
    /// - Target is `is_config_declared` → [`RemoveActivityError::ConfigDeclared`].
    /// - Target is the only activity in the pool → [`RemoveActivityError::LastRemaining`].
    /// - Any exclusive workspace (one whose `activities` set is `{target}`) has windows →
    ///   [`RemoveActivityError::ExclusiveWorkspaceHasWindows`].
    /// - Any exclusive workspace is named → [`RemoveActivityError::ExclusiveNamedWorkspace`].
    ///
    /// Precedence on simultaneous violations:
    /// `ConfigDeclared` > `LastRemaining` > `ExclusiveWorkspaceHasWindows` >
    /// `ExclusiveNamedWorkspace`. The exclusive-workspace walk classifies
    /// every candidate before choosing an error, so the result does not depend
    /// on `HashMap` iteration order (non-deterministic across runs).
    ///
    /// On success, the mutation sequence is:
    ///
    /// 1. **Cascade the active cursor** if the target was active, via [`Self::switch_activity`] to
    ///    `previous_id()` (or, failing that, to the first other live activity in declaration
    ///    order). This is the only path that snaps in-flight workspace-switch animations and
    ///    lazy-populates views on the new active activity.
    /// 2. **Destroy exclusive unnamed-empty workspaces** — drop each from the pool, and from every
    ///    activity's per-output `views`. A view with a single entry pointing at the removed id
    ///    drops entirely (mirrors the "view removal when empty" rule); otherwise
    ///    [`WorkspaceView::remove_at`] shifts the active / previous cursors.
    /// 3. **Prune shared workspaces** — drop the target id from every workspace's `activities` set
    ///    where it still appears. At least one other activity id remains in each set (those are
    ///    shared by definition), preserving the non-empty-activities invariant.
    /// 4. **Remove the activity from the pool** via [`Activities::remove`]. That call also clears
    ///    `previous` if it pointed at the removed id — covering both the post-cascade case (where
    ///    step 1 just set `previous = Some(target)`) and the non-active case where `previous`
    ///    already equaled the target.
    pub(crate) fn remove_activity(
        &mut self,
        reference: &ActivityReferenceArg,
    ) -> Result<ActivityId, RemoveActivityError> {
        let target = self
            .resolve_activity_ref(reference)
            .ok_or(RemoveActivityError::NotFound)?;

        // Validation pass — read-only. No field on `self` is mutated until
        // every error class has been ruled out; this mirrors the atomicity
        // contract that `create_activity` inherits from `create_runtime`.
        {
            let target_act = self
                .activities
                .get(target)
                .expect("resolve_activity_ref returned a live id");
            if target_act.is_config_declared() {
                return Err(RemoveActivityError::ConfigDeclared);
            }
        }
        if self.activities.len() == 1 {
            return Err(RemoveActivityError::LastRemaining);
        }

        // Walk all workspaces once, classifying exclusive candidates. Both
        // error conditions are accumulated before we choose which to return,
        // because `HashMap::values()` iteration is non-deterministic — an
        // early-return on the first named-empty would mask a has-windows
        // violation encountered later and flip the user-facing error based on
        // hash order.
        let mut has_windows_violation = false;
        let mut named_violation = false;
        let mut destroy_ids: Vec<WorkspaceId> = Vec::new();
        for ws in self.workspaces.values() {
            if ws.activities().len() == 1 && ws.activities().contains(&target) {
                if ws.has_windows() {
                    has_windows_violation = true;
                } else if ws.name().is_some() {
                    named_violation = true;
                } else {
                    destroy_ids.push(ws.id());
                }
            }
        }
        if has_windows_violation {
            return Err(RemoveActivityError::ExclusiveWorkspaceHasWindows);
        }
        if named_violation {
            return Err(RemoveActivityError::ExclusiveNamedWorkspace);
        }

        // Mutation phase. Order matters: cascade first (so `switch_activity`'s
        // hard-block `debug_assert!` is satisfied by the same outer gate as
        // the keybinding dispatcher), then workspace destruction, then pool
        // prune, then the activity-pool removal.
        if target == self.activities.active_id() {
            let cascade_target = self
                .activities
                .previous_id()
                .or_else(|| {
                    self.activities
                        .iter()
                        .map(|a| a.id())
                        .find(|id| *id != target)
                })
                .expect("LastRemaining was rejected above, so at least one other id exists");
            self.switch_activity(cascade_target);
        }

        // Destroy exclusive unnamed-empty workspaces. The `retain` guard drops
        // the view entry entirely when removing would empty it — `WorkspaceView`
        // cannot be zero-sized, and "View removal when empty" specifies
        // the entry is dropped in that case. Otherwise `remove_at` patches
        // `active` / `previous` and shifts the id list.
        for ws_id in destroy_ids {
            for activity in self.activities.iter_mut() {
                activity.views_mut().retain(|_output_id, view| {
                    let Some(pos) = view.position_of(ws_id) else {
                        return true;
                    };
                    if view.len() == 1 {
                        return false;
                    }
                    view.remove_at(pos);
                    true
                });
            }
            assert!(
                self.workspaces.remove(&ws_id).is_some(),
                "destroy id {ws_id:?} must be a live pool key",
            );
        }

        // Prune shared workspaces — ones where the target coexists with at
        // least one other activity. Exclusives were destroyed or errored out
        // above, so every remaining membership is shared and the set stays
        // non-empty after the remove. Collect pruned ids for the post-remove
        // reclaim pass below.
        let mut pruned_ids: Vec<WorkspaceId> = Vec::new();
        for ws in self.workspaces.values_mut() {
            if ws.activities().contains(&target) {
                debug_assert!(
                    ws.activities().len() > 1,
                    "exclusive workspaces were destroyed or errored in the validation pass",
                );
                ws.activities.remove(&target);
                pruned_ids.push(ws.id());
            }
        }

        // Remove from the pool. `Activities::remove` clears `previous` when it
        // pointed at `target`, covering both the post-cascade case (step 1 set
        // previous = Some(target)) and any non-active case where previous
        // happened to already equal target.
        let _ = self.activities.remove(target);

        // Any monitor whose activity-switch transition was departing from `target` now holds a
        // dead id in `from`. Snap those before the invariant check fires.
        self.snap_stale_activity_switches();

        // Destroying exclusive workspaces above may have dropped single-entry views in other
        // activities (cross-activity `retain` at `destroy_workspaces_cross_activity`). Re-run
        // the materializer so every remaining activity holds a bookend view for every
        // connected monitor.
        self.ensure_all_activity_views();

        // The membership prune above shrank each pruned workspace's activities set
        // by one. Any that became exclusive to a single activity may now sit in an
        // illegal middle position (previously legal as a pinned shared workspace).
        self.reclaim_unpinned_empty_workspaces(pruned_ids);

        #[cfg(debug_assertions)]
        self.verify_invariants();

        Ok(target)
    }

    /// Clear any in-flight activity-switch transition whose outgoing activity is no longer a live
    /// pool key.
    ///
    /// Defensive: does not assume the caller redirected any in-flight transition before deleting
    /// an activity. Any `from` id that is no longer a live pool key is cleared. Without this
    /// sweep the monitor state would hold a permanently stale dead id; in debug builds,
    /// `verify_invariants` (which checks `from` liveness) would panic on the next refresh.
    pub(super) fn snap_stale_activity_switches(&mut self) {
        let activities = &self.activities;
        for mon in &mut self.monitors {
            if mon
                .activity_switch
                .as_ref()
                .is_some_and(|s| !activities.contains(s.from))
            {
                mon.activity_switch = None;
            }
        }
    }

    /// Clear every monitor's in-flight activity-switch transition, snapping it to its end.
    ///
    /// Used when entering or leaving the overview: a strip slide and the overview spatial map
    /// must not run simultaneously. The close direction is a structural no-op (nothing can be
    /// armed while the overview is open, enforced by the overview/slide exclusion invariant), so
    /// the unconditional clear is simply simpler than gating on direction.
    pub(super) fn snap_all_activity_switches(&mut self) {
        for mon in &mut self.monitors {
            mon.activity_switch = None;
        }
    }

    /// Rename the runtime activity identified by `reference`, returning its id
    /// on success.
    ///
    /// Rejection rules (all evaluated before any mutation):
    ///
    /// - Unknown reference → [`RenameActivityError::NotFound`].
    /// - Target is `is_config_declared` → [`RenameActivityError::ConfigDeclared`].
    /// - `name.trim().is_empty()` → [`RenameActivityError::EmptyName`].
    /// - `name` collides case-insensitively with a *different* activity's name →
    ///   [`RenameActivityError::DuplicateName`]. Renaming to a case variant of the target's own
    ///   current name (or its exact current name) succeeds — the target is excluded from the
    ///   duplicate scan.
    ///
    /// Precedence on overlapping violations:
    /// `NotFound` > `ConfigDeclared` > `EmptyName` / `DuplicateName`. The
    /// outer check rejects config-declared before delegating to
    /// [`Activities::rename_runtime`], so an empty or duplicate rename of a
    /// config-declared activity surfaces as `ConfigDeclared`, never as the
    /// inner validation error.
    ///
    /// Rename is pure metadata: no view patching, no cascade, no workspace-set
    /// changes. The [`Activities`] pool's `active` / `previous` cursors are
    /// unaffected.
    pub(crate) fn rename_activity(
        &mut self,
        reference: &ActivityReferenceArg,
        name: String,
    ) -> Result<ActivityId, RenameActivityError> {
        let target = self
            .resolve_activity_ref(reference)
            .ok_or(RenameActivityError::NotFound)?;

        if self
            .activities
            .get(target)
            .expect("resolve_activity_ref returned a live id")
            .is_config_declared()
        {
            return Err(RenameActivityError::ConfigDeclared);
        }

        self.activities.rename_runtime(target, name)?;

        #[cfg(debug_assertions)]
        self.verify_invariants();

        Ok(target)
    }

    /// Returns `Some(ActivitySwitchBlock::WorkspaceSwitchGesture)` if any
    /// monitor carries an in-flight `WorkspaceSwitch::Gesture(_)`, else `None`.
    ///
    /// `RemoveWorkspaceFromActivity` and `SetWorkspaceActivities` are
    /// hard-blocked by an in-flight workspace-switch *gesture* — removing an id
    /// from the current activity's `view.ids` would invalidate the gesture's
    /// fractional position targets.
    /// `AddWorkspaceToActivity` is not gated: the insert adjusts any in-flight
    /// workspace switch on the target monitor when the active view is patched,
    /// and is safe during animations and gestures alike.
    ///
    /// This predicate is weaker than [`Self::is_activity_switch_hard_blocked`]:
    /// it does **not** consider `self.interactive_move` / `self.dnd` (those are
    /// orthogonal to workspace-activity membership) and it does **not** block
    /// on `WorkspaceSwitch::Animation(_)` — animations are snapped by the
    /// mutator before proceeding, matching `switch_activity`'s snap+proceed
    /// contract.
    pub(crate) fn is_workspace_activity_assignment_blocked_by_gesture(
        &self,
    ) -> Option<ActivitySwitchBlock> {
        for mon in &self.monitors {
            if matches!(mon.workspace_switch, Some(WorkspaceSwitch::Gesture(_))) {
                return Some(ActivitySwitchBlock::WorkspaceSwitchGesture);
            }
        }
        None
    }

    /// Detach `ws_id` from the view within `act_id` that holds it, patching
    /// the view according to whether the workspace was its sole entry.
    ///
    /// Scans `act_id`'s own views (not the layout-wide
    /// [`Self::workspace_holding_output`]) for the entry holding `ws_id` — the
    /// caller's membership check already guarantees `ws_id` was a member of
    /// `act_id`, and the per-activity bookend invariant guarantees every
    /// activity holds a view for every connected output, so the holding view
    /// (if any) must belong to `act_id` itself, not some other activity.
    ///
    /// With any monitor connected, finding no holding view is a
    /// membership↔view coherence bug, not a legitimate outcome — mirrors
    /// [`Self::workspace_holding_output`]'s own debug-loud discipline; the
    /// coherence `debug_assert!` below fires in that case.
    ///
    /// `site` names the calling verb and appears verbatim in the assert
    /// message, so callers should pass their own function name.
    fn detach_ws_from_activity_view(
        &mut self,
        act_id: ActivityId,
        ws_id: WorkspaceId,
        site: &str,
    ) -> DetachOutcome {
        let holding = {
            let activity = self
                .activities
                .get(act_id)
                .expect("resolve_activity_ref returned a live id");
            activity
                .views()
                .iter()
                .find_map(|(out_id, view)| view.position_of(ws_id).map(|pos| (out_id.clone(), pos)))
        };
        debug_assert!(
            holding.is_some() || self.monitors.is_empty(),
            "{site}: {ws_id:?} was a member of {act_id:?} but no view in that activity holds \
             it while a monitor is connected — membership↔view coherence bug",
        );
        let Some((out_id, pos)) = holding else {
            return DetachOutcome::NoView;
        };
        // is_connected is always true here: every view key is a currently-connected output's
        // OutputId, per the connected-keyspace invariant (dormant activities' views for a
        // disconnecting output are migrated away by `remove_output`'s partial-disconnect walk).
        let is_connected = self.monitors.iter().any(|m| m.output_id() == out_id);
        let activity = self
            .activities
            .get_mut(act_id)
            .expect("resolve_activity_ref returned a live id");
        let view = activity
            .views_mut()
            .get_mut(&out_id)
            .expect("holding view was found in the shared-borrow scan above");
        if view.len() == 1 {
            // Drop the single-entry view outright — mirrors the
            // `destroy_workspaces_cross_activity` single-entry retain-drop path.
            activity.views_mut().remove(&out_id);
            DetachOutcome::DroppedView {
                connected: is_connected,
            }
        } else {
            view.remove_at(pos);
            DetachOutcome::RemovedAt(out_id)
        }
    }

    /// Insert `ws_id` into `activity`'s view for `out_id`, keeping a
    /// trailing-empty bookend at the tail — the shared write performed by
    /// [`Self::add_workspace_to_activity`] and [`Self::set_workspace_activities`]'s
    /// `to_add` loop, after each has already written the membership set.
    ///
    /// Guarded by a widened absence check across every view of `activity`,
    /// not just the `out_id` one: under each caller's precondition (an
    /// explicit no-op check, or exclusion from the just-written membership
    /// diff) the workspace cannot legitimately already be in one of the
    /// activity's views (per-view uniqueness is derived from pool
    /// membership), so this is belt-and-braces against a pre-existing
    /// membership↔view incoherence elsewhere in the same activity (a
    /// separate, not-yet-fixed producer) being compounded by a
    /// double-install. Returns `None` without mutating when the guard
    /// trips.
    ///
    /// `out_id` names a currently-connected output, and the per-activity
    /// bookend invariant guarantees every activity holds a view for every
    /// connected output — so `activity` must have a view keyed by `out_id`.
    /// If it doesn't, the caller's membership write already ran, so silently
    /// skipping the view patch would mint a membership-without-view
    /// incoherence; the missing-view arm asserts loudly instead. `site`
    /// names the calling verb and appears verbatim in the assert message.
    ///
    /// Returns the insertion position on success, so a caller patching the
    /// view its active monitor is rendering can shift an in-flight
    /// workspace switch.
    fn install_ws_into_view(
        pool: &HashMap<WorkspaceId, Workspace<W>>,
        activity: &mut Activity,
        ws_id: WorkspaceId,
        out_id: &OutputId,
        site: &str,
    ) -> Option<usize> {
        let already_present = activity
            .views()
            .values()
            .any(|view| view.ids().contains(&ws_id));
        if already_present {
            return None;
        }
        if let Some(view) = activity.views_mut().get_mut(out_id) {
            Some(Self::view_insert_above_trailing_bookend(pool, view, ws_id))
        } else {
            debug_assert!(
                false,
                "{site}: activity {:?} has no view for {out_id:?}, a connected output — \
                 per-activity bookend invariant violated (membership↔view coherence bug)",
                activity.id(),
            );
            None
        }
    }

    /// Add `workspace` to `activity_ref`'s membership set and, if the activity
    /// has a view for the output currently holding the workspace, insert the id above the
    /// view's trailing-empty bookend (or append when the workspace is itself
    /// empty-unnamed, since it is a valid bookend position).
    ///
    /// Resolution order:
    /// 1. `activity_ref` → `AddWorkspaceToActivityError::ActivityNotFound`.
    /// 2. `workspace` (via [`Self::find_workspace_by_ref`] or, when `None`, the active workspace) →
    ///    `AddWorkspaceToActivityError::WorkspaceNotFound`.
    ///
    /// Semantics:
    /// - No-op when the workspace's `activities` set already contains the target id (returns `Ok`,
    ///   no state touched).
    /// - Otherwise `activity_id` is inserted into `ws.activities`. The workspace's holding view is
    ///   located via [`Self::workspace_holding_output`] — **not** `ws.output_id()`, which is a
    ///   reclaim tag that can go stale after a partial disconnect. If the target activity already
    ///   has a view for the holding output, the id is inserted above the trailing-empty bookend (or
    ///   appended when the workspace is itself empty-unnamed). Every activity holds a view for
    ///   every connected output eagerly (the per-activity bookend invariant), so there is no
    ///   "dormant activity without a view" case to fabricate around. The pre-insert absence guard
    ///   checks every view of the target activity, not just the holding-output one, so a
    ///   pre-existing membership↔view incoherence elsewhere in the same activity can't be
    ///   compounded by a double-install.
    ///
    /// Not gated. The insert adjusts any in-flight workspace switch on the
    /// target monitor when the active view is patched, so this is safe during
    /// both workspace-switch animations and gestures.
    pub(crate) fn add_workspace_to_activity(
        &mut self,
        workspace: Option<WorkspaceReference>,
        activity_ref: &ActivityReferenceArg,
    ) -> Result<(WorkspaceId, ActivityId), AddWorkspaceToActivityError> {
        let activity_id = self
            .resolve_activity_ref(activity_ref)
            .ok_or(AddWorkspaceToActivityError::ActivityNotFound)?;

        let ws_id = self
            .resolve_workspace_ref_or_active(workspace)
            .ok_or(AddWorkspaceToActivityError::WorkspaceNotFound)?;

        // No-op early-exit: the workspace already belongs to the activity.
        // Do not touch views or invariants — the state is unchanged.
        {
            let ws = self
                .workspaces
                .get(&ws_id)
                .expect("resolved ws_id must be a live pool key");
            if ws.activities().contains(&activity_id) {
                return Ok((ws_id, activity_id));
            }
        }

        // Resolve the holding view's output before entering the split-borrow scope below —
        // `workspace_holding_output` takes `&self`, which cannot coexist with the
        // `&mut self.workspaces` / `&mut self.activities` split that follows.
        let holding_out_id = self.workspace_holding_output(ws_id);

        // Patch ws.activities + the target activity's view in a tight scope so
        // the split mutable borrows on `self.workspaces` / `self.activities`
        // die before the bookend sweep below needs `&mut self`.
        {
            let pool = &mut self.workspaces;
            let activities = &mut self.activities;
            let active_id = activities.active_id();

            let ws = pool
                .get_mut(&ws_id)
                .expect("resolved ws_id must be a live pool key");
            ws.activities.insert(activity_id);

            if let Some(out_id_ref) = holding_out_id.as_ref() {
                if let Some(activity) = activities.get_mut(activity_id) {
                    if let Some(pos) = Self::install_ws_into_view(
                        pool,
                        activity,
                        ws_id,
                        out_id_ref,
                        "add_workspace_to_activity",
                    ) {
                        // A mid-view insert into the view the monitor is
                        // rendering must shift an in-flight workspace
                        // switch, mirroring `add_workspace_at_on`.
                        if activity_id == active_id {
                            if let Some(mon) = self
                                .monitors
                                .iter_mut()
                                .find(|m| &m.output_id() == out_id_ref)
                            {
                                if let Some(switch) = &mut mon.workspace_switch {
                                    if pos as f64 <= switch.target_idx() {
                                        switch.offset(1);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Per-activity bookend invariant: every (activity, connected output)
        // view must end in an empty unnamed workspace. The insert above keeps
        // an existing trailing-empty bookend at the tail, so the only rule a
        // windowed-or-named `ws_id` can still break is EWAF's leading empty —
        // when the target view held a single empty entry (doubling as both
        // bookends) and `ws_id` landed at position 0. Both active and dormant
        // views are repaired by the sweep's leading-empty arm. An empty unnamed
        // `ws_id` is itself a valid bookend and the entire repair block
        // short-circuits.
        //
        // Silent-skip case: when `ws_id` has no holding view in any activity — the
        // fully-disconnected window, where every workspace is parked in
        // `disconnected_workspace_ids` and no activity holds any view — no
        // sweep runs here. The bookend rule is re-asserted on reconnect via
        // `Monitor::new`'s materializer. Matches the `set_workspace_name` precedent — no
        // warn!, no eager repair, no rendering happens against a disconnected output.
        let ws_needs_repair = self
            .workspaces
            .get(&ws_id)
            .expect("resolved ws_id must be a live pool key")
            .has_windows_or_name();
        if ws_needs_repair {
            if let Some(out_id) = holding_out_id {
                if self.monitors.iter().any(|m| m.output_id() == out_id) {
                    self.normalize_view_bookends();
                }
            }
        }

        #[cfg(debug_assertions)]
        self.verify_invariants();

        Ok((ws_id, activity_id))
    }

    /// Remove `workspace` from `activity_ref`'s membership set and patch the
    /// view actually holding the workspace within that activity (if any).
    ///
    /// Resolution order:
    /// 1. `activity_ref` → `RemoveWorkspaceFromActivityError::ActivityNotFound`.
    /// 2. `workspace` → `RemoveWorkspaceFromActivityError::WorkspaceNotFound`.
    ///
    /// Semantics:
    /// - No-op when the workspace's `activities` set does not contain the target id (returns `Ok`).
    /// - If removing would empty `ws.activities`, returns
    ///   `RemoveWorkspaceFromActivityError::LastActivity` without mutating any state
    ///   (guard-before-mutate discipline; invariant is enforced by `Layout::verify_invariants`).
    /// - Otherwise: if the active activity has an in-flight `WorkspaceSwitch::Animation` on any
    ///   monitor AND the target activity is the active one, the animation is snapped on every
    ///   monitor first (matching `switch_activity`'s snap+proceed contract). Then `activity_id` is
    ///   removed from `ws.activities`. Finally, `activity_id`'s own views are scanned for the entry
    ///   holding `ws_id` — **not** `ws.output_id()`, which is a reclaim tag that can go stale after
    ///   a partial disconnect (see [`Self::workspace_holding_output`]'s rustdoc) — and the holding
    ///   view (if any) is patched: if its single entry was `ws_id` it is dropped entirely
    ///   (mirroring `destroy_workspaces_cross_activity` behavior); otherwise
    ///   `WorkspaceView::remove_at` shifts the active / previous cursors.
    /// - Active-activity special case: when dropping the last view entry on a connected monitor's
    ///   output for the active activity, the cross-field invariant `active.views.len() ==
    ///   monitors.len()` would be violated — [`Self::ensure_all_activity_views`] is called
    ///   immediately after the drop to reinstate the view (fresh trailing empty + EWAF leading
    ///   empty if applicable).
    ///
    /// Callers that dispatch via IPC must first consult
    /// [`Self::is_workspace_activity_assignment_blocked_by_gesture`] — a
    /// gesture in flight on any monitor blocks the call.
    /// This method does not re-check that predicate; the caller owns the
    /// decision to queue vs. proceed.
    pub(crate) fn remove_workspace_from_activity(
        &mut self,
        workspace: Option<WorkspaceReference>,
        activity_ref: &ActivityReferenceArg,
    ) -> Result<(WorkspaceId, ActivityId), RemoveWorkspaceFromActivityError> {
        let activity_id = self
            .resolve_activity_ref(activity_ref)
            .ok_or(RemoveWorkspaceFromActivityError::ActivityNotFound)?;

        let ws_id = self
            .resolve_workspace_ref_or_active(workspace)
            .ok_or(RemoveWorkspaceFromActivityError::WorkspaceNotFound)?;

        // Read-only inspection phase — no mutation until every error class
        // has been ruled out ( guard-before-mutate).
        let (is_member, len_before) = {
            let ws = self
                .workspaces
                .get(&ws_id)
                .expect("resolved ws_id must be a live pool key");
            (
                ws.activities().contains(&activity_id),
                ws.activities().len(),
            )
        };

        if !is_member {
            // No-op: the workspace is already not a member. "No-op
            // if already not in activity".
            return Ok((ws_id, activity_id));
        }

        if len_before == 1 {
            // `activities.len() - 1 == 0` — removing would violate.
            return Err(RemoveWorkspaceFromActivityError::LastActivity);
        }

        // Active-activity branch: snap any in-flight workspace-switch
        // animation on every monitor before patching views. Mirrors
        // `switch_activity`'s step-4 clear. Only runs when the mutation
        // would affect the active activity's view that the animation's
        // fractional positions refer to.
        if activity_id == self.activities.active_id() {
            for mon in &mut self.monitors {
                if matches!(mon.workspace_switch, Some(WorkspaceSwitch::Animation(_))) {
                    mon.workspace_switch = None;
                }
            }
        }

        {
            let ws = self
                .workspaces
                .get_mut(&ws_id)
                .expect("resolved ws_id must be a live pool key");
            ws.activities.remove(&activity_id);
        }

        // Patch the view actually holding `ws_id` within `activity_id` (if any). Track any
        // view-entry drop on a connected monitor so the materializer can reinstate the dropped
        // view via ensure_all_activity_views below.
        let mut dropped_any_view_entry = false;
        let mut collapse_out_id: Option<OutputId> = None;
        match self.detach_ws_from_activity_view(
            activity_id,
            ws_id,
            "remove_workspace_from_activity",
        ) {
            DetachOutcome::NoView => {}
            DetachOutcome::DroppedView { connected } => {
                if connected {
                    dropped_any_view_entry = true;
                }
            }
            DetachOutcome::RemovedAt(out_id) => {
                // A `remove_at` on a length-3 view leaves length 2, which the collapse helper
                // normalizes below when both survivors are empty and the second is exclusive.
                collapse_out_id = Some(out_id);
            }
        }

        // Reinstate dropped views via the materializer. `ensure_all_activity_views` takes
        // `&mut self`, so every nested borrow above must already be released (they are — the
        // scope above ended).
        if dropped_any_view_entry {
            self.ensure_all_activity_views();
        }

        // Normalize any all-empty exclusive EWAF len-2 view the removal left behind, before the
        // narrowed-workspace reclaim and the tail verify (the len-2 shape trips
        // `assert_view_bookends`). Guarded and idempotent — a non-len-2 view early-returns.
        if let Some(out_id) = collapse_out_id {
            self.collapse_empty_exclusive_ewaf_len2_view(activity_id, &out_id);
        }

        // The narrowing may have left ws_id exclusively in one activity's view
        // while it sits in an illegal middle position (previously legal as a
        // pinned shared workspace). Reclaim if so.
        self.reclaim_unpinned_empty_workspaces([ws_id]);

        #[cfg(debug_assertions)]
        self.verify_invariants();

        Ok((ws_id, activity_id))
    }

    /// Replace the `activities` set of `workspace` with `activity_refs`,
    /// patching every affected activity's view holding the workspace (if any).
    ///
    /// Resolution order (guard-before-mutate — no mutation happens until
    /// every error class is ruled out):
    /// 1. Every ref in `activity_refs` is resolved. The first unresolvable ref short-circuits to
    ///    `SetWorkspaceActivitiesError::ActivityNotFound` — precedence over every subsequent check,
    ///    matching the `Add` / `Remove` precedent.
    /// 2. If the resolved set is empty → `EmptyActivityList`.
    /// 3. Resolve the workspace reference. On miss →
    ///    `SetWorkspaceActivitiesError::WorkspaceNotFound`.
    ///
    /// Wire-surfaces via
    /// `DoActionError::SetWorkspaceActivities(SetWorkspaceActivitiesError::WorkspaceNotFound)`.
    ///
    /// Semantics:
    /// - Symmetric diff: `to_remove = old ∖ new`, `to_add = new ∖ old`. Removes scan each removed
    ///   activity's own views for the entry holding the workspace, then apply the same single-entry
    ///   drop / `remove_at` patch as [`Self::remove_workspace_from_activity`]. Adds insert above
    ///   the trailing-empty bookend in the target activity's view for the workspace's layout-wide
    ///   holding output (via [`Self::workspace_holding_output`] — **not** `ws.output_id()`, which
    ///   can go stale after a partial disconnect), or append when the workspace is itself
    ///   empty-unnamed.
    /// - No-op when `new == old` — returns without mutating any state.
    /// - If the active activity id is in the symmetric diff AND any monitor has a
    ///   `WorkspaceSwitch::Animation`, the animation is snapped on every monitor before patching (
    ///   snap+proceed, mirroring `remove_workspace_from_activity`).
    /// - If the active activity's view for a connected monitor is emptied by a single-entry drop on
    ///   the Remove side, [`Self::ensure_all_activity_views`] is called to reinstate the
    ///   cross-field invariant `active.views.len() == monitors.len()`.
    ///
    /// Does NOT destroy the workspace when `to_remove` prunes its last
    /// activity — the `EmptyActivityList` gate guarantees `new` is
    /// non-empty, so `ws.activities` stays non-empty post-call.
    /// distinguishes "prune the activity tag" from "destroy the workspace";
    /// only the latter goes through `destroy_workspaces_cross_activity`.
    ///
    /// The `focus: bool` parameter of the IPC action is NOT surfaced here —
    /// this method does not flip the active activity. Callers that chain
    /// into `switch_activity` handle the cursor move and the
    /// inhibitor refresh at the dispatch layer (`MoveWorkspaceToActivity`
    /// is the only such caller today).
    ///
    /// Returns `(WorkspaceId, new_activities_set, active_activity_affected)`.
    /// The `active_activity_affected` flag is `true` iff the active
    /// activity's id is in the symmetric diff — the dispatch layer uses it
    /// to decide whether a cursor warp / redraw is needed.
    ///
    /// Callers that dispatch via IPC must first consult
    /// [`Self::is_workspace_activity_assignment_blocked_by_gesture`] — a
    /// gesture in flight on any monitor blocks the call.
    pub(crate) fn set_workspace_activities(
        &mut self,
        workspace: Option<WorkspaceReference>,
        activity_refs: &[ActivityReferenceArg],
    ) -> Result<(WorkspaceId, HashSet<ActivityId>, bool), SetWorkspaceActivitiesError> {
        // Resolve every activity ref. First unresolvable ref → ActivityNotFound
        // (precedence over EmptyActivityList — matches the `resolve_activity_ref`
        // precedence of `Add` / `Remove` so a `[unresolvable_id]` of length 1
        // yields `ActivityNotFound`, not `EmptyActivityList`).
        let mut new_set: HashSet<ActivityId> = HashSet::with_capacity(activity_refs.len());
        for r in activity_refs {
            let id = self
                .resolve_activity_ref(r)
                .ok_or(SetWorkspaceActivitiesError::ActivityNotFound)?;
            new_set.insert(id);
        }

        if new_set.is_empty() {
            return Err(SetWorkspaceActivitiesError::EmptyActivityList);
        }

        // Resolve the workspace. On miss → WorkspaceNotFound (wire-surfaced).
        let ws_id = self
            .resolve_workspace_ref_or_active(workspace)
            .ok_or(SetWorkspaceActivitiesError::WorkspaceNotFound)?;

        // Snapshot the old membership set before any mutation. Resolve the layout-wide holding
        // output before the split-borrow scope below — `workspace_holding_output` takes `&self`,
        // which cannot coexist with the `&mut self.workspaces` write that follows.
        let old_set = self
            .workspaces
            .get(&ws_id)
            .expect("resolved ws_id must be a live pool key")
            .activities()
            .clone();
        let holding_out_id = self.workspace_holding_output(ws_id);

        let to_remove: Vec<ActivityId> = old_set.difference(&new_set).copied().collect();
        let to_add: Vec<ActivityId> = new_set.difference(&old_set).copied().collect();
        let active_id = self.activities.active_id();
        let active_activity_affected =
            to_remove.contains(&active_id) || to_add.contains(&active_id);

        // Identity case: new == old, no mutation.
        if to_remove.is_empty() && to_add.is_empty() {
            return Ok((ws_id, new_set, active_activity_affected));
        }

        // Snap any in-flight animation on every monitor if the active activity
        // is in the diff — the animation's fractional targets refer to the
        // active activity's view, which may change length (snap+proceed).
        if active_activity_affected {
            for mon in &mut self.monitors {
                if matches!(mon.workspace_switch, Some(WorkspaceSwitch::Animation(_))) {
                    mon.workspace_switch = None;
                }
            }
        }

        {
            let ws = self
                .workspaces
                .get_mut(&ws_id)
                .expect("resolved ws_id must be a live pool key");
            ws.activities = new_set.clone();
        }

        // Patch each affected activity's view. Track any view-entry drop on a connected monitor
        // so the materializer can reinstate the dropped view via ensure_all_activity_views below.
        let mut dropped_any_view_entry = false;
        // (activity, output) pairs a Remove narrowed to length 2 — normalized after the loop.
        let mut collapse_pairs: Vec<(ActivityId, OutputId)> = Vec::new();

        // Removes first — mirrors `remove_workspace_from_activity`'s single-entry drop vs
        // `remove_at` branch. Each removed activity is scanned independently for its own holding
        // view (not the layout-wide `holding_out_id`): under a stale-tag / partial-disconnect
        // history, different activities in `to_remove` can hold `ws_id` in views keyed by
        // different outputs. The drop-to-zero flag fires for any activity (active or dormant)
        // hitting the single-entry path on a connected output.
        for act_id in &to_remove {
            match self.detach_ws_from_activity_view(*act_id, ws_id, "set_workspace_activities") {
                DetachOutcome::NoView => {}
                DetachOutcome::DroppedView { connected } => {
                    if connected {
                        dropped_any_view_entry = true;
                    }
                }
                DetachOutcome::RemovedAt(out_id) => {
                    // Length-3 → length-2: a candidate for the all-empty exclusive EWAF collapse
                    // applied after the loop.
                    collapse_pairs.push((*act_id, out_id));
                }
            }
        }

        // Adds: insert into the target activity's view for the layout-wide holding output,
        // keeping a trailing-empty bookend at the tail. Every activity holds a view for every
        // connected output eagerly (the per-activity bookend invariant), so there is no "dormant
        // activity without a view" case to fabricate around. No in-flight switch shift is needed
        // even when the active view is patched: any switch was snapped above (the active
        // activity is in the diff).
        if let Some(out_id_ref) = holding_out_id.as_ref() {
            for act_id in &to_add {
                let activity = self
                    .activities
                    .get_mut(*act_id)
                    .expect("resolve_activity_ref returned a live id");
                Self::install_ws_into_view(
                    &self.workspaces,
                    activity,
                    ws_id,
                    out_id_ref,
                    "set_workspace_activities",
                );
            }
        }

        // Whether the post-loop bookend repair can run at all, keyed by the holding output
        // being currently connected.
        let holding_output_connected = holding_out_id
            .as_ref()
            .is_some_and(|out_id| self.monitors.iter().any(|m| &m.output_id() == out_id));

        // Per-activity bookend invariant: every (activity, connected output)
        // view must end in an empty unnamed workspace. The Add loop above
        // keeps an existing trailing-empty bookend at the tail, so the only
        // rule a windowed-or-named `ws_id` can still break is EWAF's leading
        // empty — when a target view held a single empty entry (doubling as
        // both bookends) and `ws_id` landed at position 0. Both active and
        // dormant views are repaired by the sweep's leading-empty arm. An
        // empty unnamed `ws_id` is itself a valid bookend and the entire
        // repair block short-circuits.
        //
        // Silent-skip case: when `ws_id` has no holding view in any activity — the
        // fully-disconnected window, where every workspace is parked in
        // `disconnected_workspace_ids` and no activity holds any view — no
        // sweep runs here. The bookend rule is re-asserted on reconnect via
        // `Monitor::new`'s materializer. Matches the `set_workspace_name` precedent — no
        // warn!, no eager repair, no rendering happens against a disconnected output.
        let ws_needs_repair = self
            .workspaces
            .get(&ws_id)
            .expect("resolved ws_id must be a live pool key")
            .has_windows_or_name();
        if ws_needs_repair && holding_output_connected {
            // Safe against the sweep's stricter missing-view assert: ws_needs_repair means
            // ws_id is windowed-or-named, and the bookend invariant forbids that from being
            // a view's sole (len-1) entry — the only shape the to_remove loop above drops —
            // so no activity here can be missing a view when this runs.
            self.normalize_view_bookends();
        }

        // Reinstate any dropped views via the materializer — mirrors the
        // `remove_workspace_from_activity` recipe.
        if dropped_any_view_entry {
            self.ensure_all_activity_views();
        }

        // Normalize any all-empty exclusive EWAF len-2 view a Remove left behind, before the
        // narrowed-workspace reclaim and the tail verify (the len-2 shape trips
        // `assert_view_bookends`). Guarded and idempotent — duplicate pairs and non-len-2 views
        // early-return.
        for (act_id, out_id) in &collapse_pairs {
            self.collapse_empty_exclusive_ewaf_len2_view(*act_id, out_id);
        }

        // When to_remove is non-empty, ws_id may have been narrowed to a single
        // activity. If it now sits in an illegal middle position under its sole
        // remaining owner, reclaim it.
        if !to_remove.is_empty() {
            self.reclaim_unpinned_empty_workspaces([ws_id]);
        }

        #[cfg(debug_assertions)]
        self.verify_invariants();

        Ok((ws_id, new_set, active_activity_affected))
    }

    /// Move `workspace` from the currently-active activity to
    /// `activity_ref`.
    ///
    /// Sugar for `AddWorkspaceToActivity(target) +
    /// RemoveWorkspaceFromActivity(active)`, applied atomically in that
    /// order. The Add-then-Remove ordering guarantees that the
    /// workspace is never transiently outside every activity, so the
    /// non-empty invariant is never violated — even when the workspace was
    /// previously exclusive to the active activity.
    ///
    /// Multi-activity semantics: if the workspace belongs to
    /// `{active, X, Y}`, after the move it belongs to `{X, Y, target}` —
    /// the workspace leaves the active activity but stays in the others.
    ///
    /// Resolution order:
    /// 1. `activity_ref` → `ActivityNotFound`.
    /// 2. `workspace` → `WorkspaceNotFound`.
    /// 3. If workspace's `activities` set does not contain the active activity id →
    ///    `WorkspaceNotInActiveActivity`.
    ///
    /// No-op cases (return `Ok` without mutating):
    /// - `target == active_id`.
    ///
    /// Implementation strategy: this method delegates to
    /// [`Self::add_workspace_to_activity`] + [`Self::remove_workspace_from_activity`]
    /// rather than reimplementing view-patching inline. Each delegate runs
    /// its own `verify_invariants`; the intermediate state (workspace
    /// tagged with both `active` and `target`) satisfies every invariant
    /// individually, so chaining is correctness-preserving.
    /// "Sugar for Add + Remove" is the load-bearing contract —
    /// inlining view-patching here would make the two paths drift on the
    /// next bug fix.
    ///
    /// If the workspace already has `target` in its activities set (but
    /// `source != target`), the Add step is a no-op at
    /// [`Self::add_workspace_to_activity`]; the Remove step still prunes
    /// source.
    ///
    /// The `focus: bool` discriminator is NOT consumed here — the dispatch
    /// layer chains into [`Self::switch_activity`] after a successful
    /// move, and fires the keyboard-shortcut-inhibitor refresh on
    /// that path. Keeping focus-flipping out of the Layout method keeps
    /// this API orthogonal to cursor movement.
    ///
    /// Returns `(workspace_id, target_activity_id, source_activity_id)`
    /// where `source_activity_id` is always the currently-active activity
    /// id at call time — logged by the dispatch layer.
    ///
    /// Callers that dispatch via IPC must first consult the appropriate
    /// hard-block predicate:
    /// - `focus: false` → [`Self::is_workspace_activity_assignment_blocked_by_gesture`]
    /// - `focus: true` → [`Self::is_activity_switch_hard_blocked`]
    pub(crate) fn move_workspace_to_activity(
        &mut self,
        workspace: Option<WorkspaceReference>,
        activity_ref: &ActivityReferenceArg,
    ) -> Result<(WorkspaceId, ActivityId, ActivityId), MoveWorkspaceToActivityError> {
        let target_id = self
            .resolve_activity_ref(activity_ref)
            .ok_or(MoveWorkspaceToActivityError::ActivityNotFound)?;

        let ws_id = self
            .resolve_workspace_ref_or_active(workspace)
            .ok_or(MoveWorkspaceToActivityError::WorkspaceNotFound)?;

        // Move requires the workspace to be a member of the currently-active
        // activity.
        let source_id = self.activities.active_id();
        {
            let ws = self
                .workspaces
                .get(&ws_id)
                .expect("resolved ws_id must be a live pool key");
            if !ws.activities().contains(&source_id) {
                return Err(MoveWorkspaceToActivityError::WorkspaceNotInActiveActivity);
            }
        }

        // No-op: target == source implies the workspace stays put.
        // Subsumes the "No-op if workspace already exclusively
        // in target" row — when source == target, the workspace's
        // activities cannot change. Leaves verify_invariants untouched.
        if target_id == source_id {
            return Ok((ws_id, target_id, source_id));
        }

        // Delegate to Add-then-Remove. Both operations are `&mut self`
        // and run their own `verify_invariants`. The intermediate state
        // (workspace tagged with both source and target) is invariant-
        // preserving — this is the correctness argument for "sugar for
        // Add + Remove".
        //
        // If the workspace already has target (but source != target), the
        // Add is a no-op at the delegate; the Remove still prunes source.
        // After Add the workspace has both {source, target, ...} — len >= 2,
        // so Remove's LastActivity guard will not fire.
        self.add_workspace_to_activity(
            Some(WorkspaceReference::Id(ws_id.get())),
            &ActivityReferenceArg::Id(target_id.get()),
        )
        .expect(
            "Add cannot fail: target was just resolved and ws_id is live; the delegate's only \
             error classes (ActivityNotFound, WorkspaceNotFound) are already ruled out",
        );
        self.remove_workspace_from_activity(
            Some(WorkspaceReference::Id(ws_id.get())),
            &ActivityReferenceArg::Id(source_id.get()),
        )
        .expect(
            "Remove cannot fail: source is the active activity (must resolve), ws_id is live, \
             and Add guaranteed ws.activities.len() >= 2 so LastActivity is not reachable",
        );

        // Delegates already ran `verify_invariants` in debug builds — no
        // extra chain needed here. Return the triple the dispatch layer
        // logs.
        Ok((ws_id, target_id, source_id))
    }

    /// Set `workspace.is_sticky = true` and expand its `activities` set to all
    /// live activity ids.
    ///
    /// Resolution order:
    /// 1. `workspace` → `WorkspaceNotFound`. Wire-surfaced via
    ///    `DoActionError::SetWorkspaceSticky(WorkspaceStickyError::WorkspaceNotFound)`.
    ///
    /// No-op cases (return `Ok((ws_id, false))` without mutating):
    /// - `ws.is_sticky() == true` AND `ws.activities() == all_live_ids` — the typical state for an
    ///   already-sticky workspace.
    ///
    /// Implementation strategy: delegate to [`Self::set_workspace_activities`]
    /// with the full live id set as the target activity list, then flip
    /// `is_sticky = true`. The flag flip is deliberately ordered AFTER the
    /// delegate's symmetric-diff machinery so any intermediate
    /// `verify_invariants` chain sees a coherent state. The flag flip alone,
    /// with `activities` already at all_ids, is invariant-preserving by
    /// construction.
    ///
    /// Returns `(workspace_id, active_affected)` where `active_affected`
    /// bubbles up from the delegate — `true` when the symmetric diff added or
    /// removed the active activity from the workspace's set, signalling that
    /// the dispatch layer must redraw.
    pub(crate) fn set_workspace_sticky(
        &mut self,
        workspace: Option<WorkspaceReference>,
    ) -> Result<(WorkspaceId, bool), WorkspaceStickyError> {
        let ws_id = self
            .resolve_workspace_ref_or_active(workspace)
            .ok_or(WorkspaceStickyError::WorkspaceNotFound)?;

        // Two views of the full live id set: a `HashSet<ActivityId>` for the
        // no-op equality check, and a `Vec<ActivityReferenceArg>` for the
        // delegate call.
        let all_live_ids: HashSet<ActivityId> = self.activities.iter().map(|a| a.id()).collect();

        // No-op early-exit on the typical "already sticky + already expanded"
        // state. Skip the entire delegate / flag-flip path so we don't churn
        // animation state or rerun verify_invariants chains.
        {
            let ws = self
                .workspaces
                .get(&ws_id)
                .expect("resolved ws_id must be a live pool key");
            if ws.is_sticky() && ws.activities() == &all_live_ids {
                return Ok((ws_id, false));
            }
        }

        // Delegate to set_workspace_activities. The delegate handles animation
        // snap, view patching, and ensure_all_activity_views. All three delegate
        // error classes are statically impossible here:
        //   - ActivityNotFound: ids came from the live pool.
        //   - EmptyActivityList: Activities is non-empty by type.
        //   - WorkspaceNotFound: ws_id was resolved above.
        let all_ids_arg: Vec<ActivityReferenceArg> = all_live_ids
            .iter()
            .map(|id| ActivityReferenceArg::Id(id.get()))
            .collect();
        let (_, _, active_affected) = self
            .set_workspace_activities(Some(WorkspaceReference::Id(ws_id.get())), &all_ids_arg)
            .expect(
                "set_workspace_activities cannot fail: ids come from the live activity pool \
                 (ActivityNotFound impossible), Activities is non-empty \
                 (EmptyActivityList impossible), and ws_id was just resolved \
                 (WorkspaceNotFound impossible)",
            );

        // Flip is_sticky AFTER the delegate so any intermediate
        // verify_invariants run sees a coherent intermediate state. The flip
        // alone (with activities already at all_ids) is invariant-preserving.
        self.workspaces
            .get_mut(&ws_id)
            .expect("resolved ws_id must be a live pool key")
            .is_sticky = true;

        #[cfg(debug_assertions)]
        self.verify_invariants();

        Ok((ws_id, active_affected))
    }

    /// Clear `workspace.is_sticky` without touching its `activities` set
    /// (`is_sticky` is the auto-expansion trigger; toggling off
    /// does not narrow `activities`).
    ///
    /// Resolution order:
    /// 1. `workspace` → `WorkspaceNotFound`. Wire-surfaced via
    ///    `DoActionError::UnsetWorkspaceSticky(WorkspaceStickyError::WorkspaceNotFound)`.
    ///
    /// No-op cases (return `Ok(ws_id)` without mutating):
    /// - `ws.is_sticky() == false` — already not sticky.
    ///
    /// The workspace's `activities` set is intentionally left intact: the
    /// user can narrow it via `RemoveWorkspaceFromActivity` /
    /// `SetWorkspaceActivities` afterwards. Config-declared `sticky true`
    /// workspaces re-expand on the next config reload —
    /// session-only-effect is the documented contract.
    pub(crate) fn unset_workspace_sticky(
        &mut self,
        workspace: Option<WorkspaceReference>,
    ) -> Result<WorkspaceId, WorkspaceStickyError> {
        let ws_id = self
            .resolve_workspace_ref_or_active(workspace)
            .ok_or(WorkspaceStickyError::WorkspaceNotFound)?;

        let ws = self
            .workspaces
            .get_mut(&ws_id)
            .expect("resolved ws_id must be a live pool key");
        if !ws.is_sticky() {
            return Ok(ws_id);
        }
        ws.is_sticky = false;

        #[cfg(debug_assertions)]
        self.verify_invariants();

        Ok(ws_id)
    }

    /// Toggle `workspace.is_sticky`, dispatching to
    /// [`Self::set_workspace_sticky`] / [`Self::unset_workspace_sticky`].
    ///
    /// Resolution order:
    /// 1. `workspace` → `WorkspaceNotFound`. Wire-surfaced via
    ///    `DoActionError::ToggleWorkspaceSticky(WorkspaceStickyError::WorkspaceNotFound)`.
    ///
    /// Dispatches on `is_sticky` alone. The non-empty `activities`
    /// invariant makes the `sticky == true ∧ activities == ∅` state
    /// unreachable, so the flag is a faithful signal of the current sticky
    /// state.
    pub(crate) fn toggle_workspace_sticky(
        &mut self,
        workspace: Option<WorkspaceReference>,
    ) -> Result<ToggleWorkspaceStickyOutcome, WorkspaceStickyError> {
        let ws_id = self
            .resolve_workspace_ref_or_active(workspace)
            .ok_or(WorkspaceStickyError::WorkspaceNotFound)?;

        // Read current flag in a narrow scope so the shared borrow releases
        // before the delegate call.
        let currently_sticky = {
            self.workspaces
                .get(&ws_id)
                .expect("resolved ws_id must be a live pool key")
                .is_sticky()
        };

        if currently_sticky {
            self.unset_workspace_sticky(Some(WorkspaceReference::Id(ws_id.get())))
                .expect(
                    "unset cannot fail: ws_id was just resolved \
                     (WorkspaceNotFound impossible)",
                );
            Ok(ToggleWorkspaceStickyOutcome::StickyOff { ws_id })
        } else {
            let (_, active_affected) = self
                .set_workspace_sticky(Some(WorkspaceReference::Id(ws_id.get())))
                .expect(
                    "set cannot fail: ws_id was just resolved \
                     (WorkspaceNotFound impossible)",
                );
            Ok(ToggleWorkspaceStickyOutcome::StickyOn {
                ws_id,
                active_affected,
            })
        }
    }

    /// Resolve `workspace` to an id: an explicit reference resolves via
    /// [`Self::find_workspace_by_ref`], `None` falls back to the currently
    /// active workspace.
    ///
    /// Returns the id only, not a `&mut Workspace` — the `None` arm goes
    /// through [`Self::active_workspace_mut`], so returning a reference would
    /// keep the `&mut self` borrow alive past the call. Callers that need to
    /// split into `&mut self.workspaces` / `&mut self.activities` afterwards
    /// depend on the borrow having already dropped.
    fn resolve_workspace_ref_or_active(
        &mut self,
        workspace: Option<WorkspaceReference>,
    ) -> Option<WorkspaceId> {
        match workspace {
            Some(r) => self.find_workspace_by_ref(r).map(|ws| ws.id()),
            None => self.active_workspace_mut().map(|ws| ws.id()),
        }
    }

    /// Returns `Some(id)` iff `raw` matches a workspace id in the canonical
    /// pool — including workspaces that belong only to dormant activities
    /// (mirror of [`Layout::resolve_activity_ref`]).
    ///
    /// Callers that need active-view membership must compose with
    /// [`Layout::find_workspace_by_id`], which is active-view +
    /// disconnected-pool scoped.
    ///
    /// Do not collapse the chain at the call site (the `Id` arm of
    /// `Niri::find_output_and_workspace_index`) into a single filter. The two
    /// scopes (pool-membership vs. active-view + disconnected-pool) are not
    /// equivalent — a workspace exclusive to a dormant activity is visible to
    /// the resolver but invisible to the finder, and the chain's
    /// `?`-propagation depends on this asymmetry. Pinned by
    /// `resolve_workspace_id_finds_dormant_activity_workspace` in
    /// `src/layout/tests.rs`.
    pub(crate) fn resolve_workspace_id(&self, raw: u64) -> Option<WorkspaceId> {
        self.workspaces
            .values()
            .find(|ws| ws.id().get() == raw)
            .map(|ws| ws.id())
    }

    /// Resolve a workspace reference scoped to a specific activity.
    ///
    /// Scans the pool for workspaces that both match `reference` and belong to
    /// `activity` (via [`Workspace::activities`]). The scan is pool-wide, so
    /// workspaces belonging to dormant activities also resolve — the caller is
    /// responsible for any further active-view filtering.
    ///
    /// # Errors
    ///
    /// - [`FocusWorkspaceInActivityError::IndexUnsupported`] — positional indices are not
    ///   meaningful for activity-scoped lookup; callers must use `WorkspaceReference::Name` or
    ///   `WorkspaceReference::Id`.
    /// - [`FocusWorkspaceInActivityError::WorkspaceNotInActivity`] — no pool workspace matches the
    ///   reference within the given activity's membership set.
    pub(crate) fn resolve_workspace_in_activity(
        &self,
        activity: ActivityId,
        reference: &WorkspaceReference,
    ) -> Result<WorkspaceId, FocusWorkspaceInActivityError> {
        if matches!(reference, WorkspaceReference::Index(_)) {
            return Err(FocusWorkspaceInActivityError::IndexUnsupported);
        }
        self.workspaces
            .values()
            .find(|ws| {
                ws.activities().contains(&activity)
                    && match reference {
                        WorkspaceReference::Id(raw) => ws.id().get() == *raw,
                        WorkspaceReference::Name(name) => ws
                            .name
                            .as_ref()
                            .is_some_and(|n| n.eq_ignore_ascii_case(name)),
                        WorkspaceReference::Index(_) => {
                            unreachable!("index arm guarded above")
                        }
                    }
            })
            .map(|ws| ws.id())
            .ok_or(FocusWorkspaceInActivityError::WorkspaceNotInActivity)
    }

    /// Resolve an [`ActivityReferenceArg`] to an [`ActivityId`] if the pool
    /// contains a matching activity, `None` otherwise.
    ///
    /// Names are unique across the activity pool, so the
    /// first-match walk is deterministic. `ActivityId` is opaque: the `Id`
    /// variant scans `iter()` comparing raw `u64` values rather than hashing
    /// into the underlying `IndexMap`.
    pub(crate) fn resolve_activity_ref(&self, r: &ActivityReferenceArg) -> Option<ActivityId> {
        match r {
            ActivityReferenceArg::Id(raw) => self
                .activities
                .iter()
                .find(|a| a.id().get() == *raw)
                .map(|a| a.id()),
            ActivityReferenceArg::Name(s) => self
                .activities
                .iter()
                .find(|a| a.name() == s)
                .map(|a| a.id()),
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
            jiji_config::ActivityDecl {
                name: jiji_config::ActivityName("Work".to_owned()),
            },
            jiji_config::ActivityDecl {
                name: jiji_config::ActivityName("Personal".to_owned()),
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
            jiji_config::ActivityDecl {
                name: jiji_config::ActivityName("Work".to_owned()),
            },
            jiji_config::ActivityDecl {
                name: jiji_config::ActivityName("Personal".to_owned()),
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
            jiji_config::ActivityDecl {
                name: jiji_config::ActivityName("Work".to_owned()),
            },
            jiji_config::ActivityDecl {
                name: jiji_config::ActivityName("Personal".to_owned()),
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

    // --- Recency stamping tests ---

    fn activity(name: &str) -> Activity {
        Activity::new_runtime(name.to_owned())
    }

    #[test]
    fn seed_has_seq_one_and_counter_one() {
        let seed = activity("seed");
        let pool = Activities::new(seed);
        // Seed must be stamped with seq = 1 (same as activation_counter at construction).
        assert_eq!(pool.active().last_active_seq(), 1);
        assert_eq!(pool.activation_counter, 1);
        assert_eq!(pool.previous_id(), None);
    }

    #[test]
    fn set_active_real_flip_bumps_counter_and_stamps_target() {
        let seed = activity("seed");
        let seed_id = seed.id();
        let mut pool = Activities::new(seed);

        let beta = activity("beta");
        let beta_id = beta.id();
        pool.insert(beta);

        // Beta starts with seq = 0.
        assert_eq!(pool.map[&beta_id].last_active_seq, 0);
        assert_eq!(pool.activation_counter, 1);

        pool.set_active(beta_id);
        assert_eq!(pool.activation_counter, 2);
        assert_eq!(pool.map[&beta_id].last_active_seq, 2);
        // Seed seq unchanged.
        assert_eq!(pool.map[&seed_id].last_active_seq, 1);
    }

    #[test]
    fn set_active_no_op_does_not_bump_counter_or_seq() {
        let seed = activity("seed");
        let seed_id = seed.id();
        let mut pool = Activities::new(seed);

        // No-op: switching to the already-active activity.
        pool.set_active(seed_id);
        assert_eq!(pool.activation_counter, 1);
        assert_eq!(pool.map[&seed_id].last_active_seq, 1);
    }

    #[test]
    fn active_always_holds_max_seq() {
        let seed = activity("seed");
        let seed_id = seed.id();
        let mut pool = Activities::new(seed);

        let beta = activity("beta");
        let beta_id = beta.id();
        pool.insert(beta);

        // Switch seed → beta → seed → beta. At each step, the active entry holds max seq.
        pool.set_active(beta_id);
        assert_eq!(pool.map[&beta_id].last_active_seq, pool.activation_counter);

        pool.set_active(seed_id);
        assert_eq!(pool.map[&seed_id].last_active_seq, pool.activation_counter);

        pool.set_active(beta_id);
        assert_eq!(pool.map[&beta_id].last_active_seq, pool.activation_counter);
    }

    #[test]
    fn recency_ordered_descending_with_stable_declaration_fallback() {
        // 4-activity sub-test: alpha (seed, seq=1), beta (seq=0), gamma (seq=3),
        // delta (seq=2). Sequence of switches: seed→gamma, seed→gamma→delta.
        let alpha = activity("alpha");
        let alpha_id = alpha.id();
        let mut pool = Activities::new(alpha);

        let beta = activity("beta");
        let beta_id = beta.id();
        pool.insert(beta);

        let gamma = activity("gamma");
        let gamma_id = gamma.id();
        pool.insert(gamma);

        let delta = activity("delta");
        let delta_id = delta.id();
        pool.insert(delta);

        // Switch: alpha → gamma → alpha → delta → gamma.
        pool.set_active(gamma_id); // gamma seq=2
        pool.set_active(alpha_id); // alpha seq=3
        pool.set_active(delta_id); // delta seq=4
        pool.set_active(gamma_id); // gamma seq=5

        // Expected MRU order: gamma(5) > delta(4) > alpha(3) > beta(0).
        let ordered = pool.recency_ordered();
        assert_eq!(ordered, vec![gamma_id, delta_id, alpha_id, beta_id]);

        // Pin seq-0 stable tie-break: insert two more activities that are never activated.
        let mut pool2 = Activities::new(activity("a"));
        let a_id = pool2.active_id();
        let b = activity("b");
        let b_id = b.id();
        pool2.insert(b);
        let c = activity("c");
        let c_id = c.id();
        pool2.insert(c);
        // b and c both have seq=0. Stable sort must preserve insertion order: b before c.
        let ordered2 = pool2.recency_ordered();
        assert_eq!(ordered2[0], a_id, "seed (seq=1) is first");
        // b and c both have seq=0; stable sort preserves their insertion order.
        assert_eq!(ordered2[1], b_id, "b inserted before c, same seq=0");
        assert_eq!(ordered2[2], c_id, "c inserted after b, same seq=0");
    }

    #[test]
    fn verify_invariants_holds_after_normal_flip_sequence() {
        let seed = activity("seed");
        let mut pool = Activities::new(seed);

        let beta = activity("beta");
        let beta_id = beta.id();
        pool.insert(beta);

        let seed_id = pool.active_id();

        // After each flip, verify_invariants must pass.
        pool.set_active(beta_id);
        pool.verify_invariants();

        pool.set_active(seed_id);
        pool.verify_invariants();

        pool.set_active(beta_id);
        pool.verify_invariants();
    }
}
