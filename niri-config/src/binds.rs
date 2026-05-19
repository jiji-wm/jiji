use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::str::FromStr;
use std::time::Duration;

use bitflags::bitflags;
use jiji_ipc::{
    ActivityReferenceArg, ColumnDisplay, LayoutSwitchTarget, PositionChange, SizeChange,
    WorkspaceReferenceArg,
};
use knuffel::errors::DecodeError;
use miette::miette;
use smithay::input::keyboard::keysyms::KEY_NoSymbol;
use smithay::input::keyboard::xkb::{keysym_from_name, KEYSYM_CASE_INSENSITIVE, KEYSYM_NO_FLAGS};
use smithay::input::keyboard::Keysym;

use crate::recent_windows::{MruDirection, MruFilter, MruScope};
use crate::utils::{expect_only_children, MergeWith};

#[derive(Debug, Default, PartialEq)]
pub struct Binds(pub Vec<Bind>);

#[derive(Debug, Clone, PartialEq)]
pub struct Bind {
    pub key: Key,
    pub action: Action,
    pub repeat: bool,
    pub cooldown: Option<Duration>,
    pub allow_when_locked: bool,
    pub allow_inhibiting: bool,
    pub hotkey_overlay_title: Option<Option<String>>,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct Key {
    pub trigger: Trigger,
    pub modifiers: Modifiers,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum Trigger {
    Keysym(Keysym),
    MouseLeft,
    MouseRight,
    MouseMiddle,
    MouseBack,
    MouseForward,
    WheelScrollDown,
    WheelScrollUp,
    WheelScrollLeft,
    WheelScrollRight,
    TouchpadScrollDown,
    TouchpadScrollUp,
    TouchpadScrollLeft,
    TouchpadScrollRight,
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct Modifiers : u8 {
        const CTRL = 1;
        const SHIFT = 1 << 1;
        const ALT = 1 << 2;
        const SUPER = 1 << 3;
        const ISO_LEVEL3_SHIFT = 1 << 4;
        const ISO_LEVEL5_SHIFT = 1 << 5;
        const COMPOSITOR = 1 << 6;
    }
}

#[derive(knuffel::Decode, Debug, Default, Clone, PartialEq)]
pub struct SwitchBinds {
    #[knuffel(child)]
    pub lid_open: Option<SwitchAction>,
    #[knuffel(child)]
    pub lid_close: Option<SwitchAction>,
    #[knuffel(child)]
    pub tablet_mode_on: Option<SwitchAction>,
    #[knuffel(child)]
    pub tablet_mode_off: Option<SwitchAction>,
}

impl MergeWith<SwitchBinds> for SwitchBinds {
    fn merge_with(&mut self, part: &SwitchBinds) {
        merge_clone_opt!(
            (self, part),
            lid_open,
            lid_close,
            tablet_mode_on,
            tablet_mode_off,
        );
    }
}

#[derive(knuffel::Decode, Debug, Clone, PartialEq)]
pub struct SwitchAction {
    #[knuffel(child, unwrap(arguments))]
    pub spawn: Vec<String>,
}

// Remember to add new actions to the CLI enum too.
#[derive(knuffel::Decode, Debug, Clone, PartialEq)]
pub enum Action {
    Quit(#[knuffel(property(name = "skip-confirmation"), default)] bool),
    #[knuffel(skip)]
    ChangeVt(i32),
    Suspend,
    PowerOffMonitors,
    PowerOnMonitors,
    ToggleDebugTint,
    DebugToggleOpaqueRegions,
    DebugToggleDamage,
    Spawn(#[knuffel(arguments)] Vec<String>),
    SpawnSh(#[knuffel(argument)] String),
    DoScreenTransition(#[knuffel(property(name = "delay-ms"))] Option<u16>),
    #[knuffel(skip)]
    ConfirmScreenshot {
        write_to_disk: bool,
    },
    #[knuffel(skip)]
    CancelScreenshot,
    #[knuffel(skip)]
    ScreenshotTogglePointer,
    Screenshot(
        #[knuffel(property(name = "show-pointer"), default = true)] bool,
        // Path; not settable from knuffel
        Option<String>,
    ),
    ScreenshotScreen(
        #[knuffel(property(name = "write-to-disk"), default = true)] bool,
        #[knuffel(property(name = "show-pointer"), default = true)] bool,
        // Path; not settable from knuffel
        Option<String>,
    ),
    ScreenshotWindow(
        #[knuffel(property(name = "write-to-disk"), default = true)] bool,
        #[knuffel(property(name = "show-pointer"), default = false)] bool,
        // Path; not settable from knuffel
        Option<String>,
    ),
    #[knuffel(skip)]
    ScreenshotWindowById {
        id: u64,
        write_to_disk: bool,
        show_pointer: bool,
        path: Option<String>,
    },
    ToggleKeyboardShortcutsInhibit,
    CloseWindow,
    #[knuffel(skip)]
    CloseWindowById(u64),
    FullscreenWindow,
    #[knuffel(skip)]
    FullscreenWindowById(u64),
    ToggleWindowedFullscreen,
    #[knuffel(skip)]
    ToggleWindowedFullscreenById(u64),
    #[knuffel(skip)]
    FocusWindow(u64),
    FocusWindowInColumn(#[knuffel(argument)] u8),
    FocusWindowPrevious,
    FocusColumnLeft,
    #[knuffel(skip)]
    FocusColumnLeftUnderMouse,
    FocusColumnRight,
    #[knuffel(skip)]
    FocusColumnRightUnderMouse,
    FocusColumnFirst,
    FocusColumnLast,
    FocusColumnRightOrFirst,
    FocusColumnLeftOrLast,
    FocusColumn(#[knuffel(argument)] usize),
    FocusWindowOrMonitorUp,
    FocusWindowOrMonitorDown,
    FocusColumnOrMonitorLeft,
    FocusColumnOrMonitorRight,
    FocusWindowDown,
    FocusWindowUp,
    FocusWindowDownOrColumnLeft,
    FocusWindowDownOrColumnRight,
    FocusWindowUpOrColumnLeft,
    FocusWindowUpOrColumnRight,
    FocusWindowOrWorkspaceDown,
    FocusWindowOrWorkspaceUp,
    FocusWindowTop,
    FocusWindowBottom,
    FocusWindowDownOrTop,
    FocusWindowUpOrBottom,
    MoveColumnLeft,
    MoveColumnRight,
    MoveColumnToFirst,
    MoveColumnToLast,
    MoveColumnLeftOrToMonitorLeft,
    MoveColumnRightOrToMonitorRight,
    MoveColumnToIndex(#[knuffel(argument)] usize),
    MoveWindowDown,
    MoveWindowUp,
    MoveWindowDownOrToWorkspaceDown,
    MoveWindowUpOrToWorkspaceUp,
    ConsumeOrExpelWindowLeft,
    #[knuffel(skip)]
    ConsumeOrExpelWindowLeftById(u64),
    ConsumeOrExpelWindowRight,
    #[knuffel(skip)]
    ConsumeOrExpelWindowRightById(u64),
    ConsumeWindowIntoColumn,
    ExpelWindowFromColumn,
    SwapWindowLeft,
    SwapWindowRight,
    ToggleColumnTabbedDisplay,
    SetColumnDisplay(#[knuffel(argument, str)] ColumnDisplay),
    CenterColumn,
    CenterWindow,
    #[knuffel(skip)]
    CenterWindowById(u64),
    CenterVisibleColumns,
    FocusWorkspaceDown,
    #[knuffel(skip)]
    FocusWorkspaceDownUnderMouse,
    FocusWorkspaceUp,
    #[knuffel(skip)]
    FocusWorkspaceUpUnderMouse,
    FocusWorkspace(#[knuffel(argument)] WorkspaceReference),
    FocusWorkspacePrevious,
    MoveWindowToWorkspaceDown(#[knuffel(property(name = "focus"), default = true)] bool),
    MoveWindowToWorkspaceUp(#[knuffel(property(name = "focus"), default = true)] bool),
    MoveWindowToWorkspace(
        #[knuffel(argument)] WorkspaceReference,
        #[knuffel(property(name = "focus"), default = true)] bool,
    ),
    #[knuffel(skip)]
    MoveWindowToWorkspaceById {
        window_id: u64,
        reference: WorkspaceReference,
        focus: bool,
    },
    MoveColumnToWorkspaceDown(#[knuffel(property(name = "focus"), default = true)] bool),
    MoveColumnToWorkspaceUp(#[knuffel(property(name = "focus"), default = true)] bool),
    MoveColumnToWorkspace(
        #[knuffel(argument)] WorkspaceReference,
        #[knuffel(property(name = "focus"), default = true)] bool,
    ),
    MoveWorkspaceDown,
    MoveWorkspaceUp,
    MoveWorkspaceToIndex(#[knuffel(argument)] usize),
    #[knuffel(skip)]
    MoveWorkspaceToIndexByRef {
        new_idx: usize,
        reference: WorkspaceReference,
    },
    #[knuffel(skip)]
    MoveWorkspaceToMonitorByRef {
        output_name: String,
        reference: WorkspaceReference,
    },
    MoveWorkspaceToMonitor(#[knuffel(argument)] String),
    SetWorkspaceName(#[knuffel(argument)] String),
    #[knuffel(skip)]
    SetWorkspaceNameByRef {
        name: String,
        reference: WorkspaceReference,
    },
    UnsetWorkspaceName,
    #[knuffel(skip)]
    UnsetWorkSpaceNameByRef(#[knuffel(argument)] WorkspaceReference),
    FocusMonitorLeft,
    FocusMonitorRight,
    FocusMonitorDown,
    FocusMonitorUp,
    FocusMonitorPrevious,
    FocusMonitorNext,
    FocusMonitor(#[knuffel(argument)] String),
    MoveWindowToMonitorLeft,
    MoveWindowToMonitorRight,
    MoveWindowToMonitorDown,
    MoveWindowToMonitorUp,
    MoveWindowToMonitorPrevious,
    MoveWindowToMonitorNext,
    MoveWindowToMonitor(#[knuffel(argument)] String),
    #[knuffel(skip)]
    MoveWindowToMonitorById {
        id: u64,
        output: String,
    },
    MoveColumnToMonitorLeft,
    MoveColumnToMonitorRight,
    MoveColumnToMonitorDown,
    MoveColumnToMonitorUp,
    MoveColumnToMonitorPrevious,
    MoveColumnToMonitorNext,
    MoveColumnToMonitor(#[knuffel(argument)] String),
    SetWindowWidth(#[knuffel(argument, str)] SizeChange),
    #[knuffel(skip)]
    SetWindowWidthById {
        id: u64,
        change: SizeChange,
    },
    SetWindowHeight(#[knuffel(argument, str)] SizeChange),
    #[knuffel(skip)]
    SetWindowHeightById {
        id: u64,
        change: SizeChange,
    },
    ResetWindowHeight,
    #[knuffel(skip)]
    ResetWindowHeightById(u64),
    SwitchPresetColumnWidth,
    SwitchPresetColumnWidthBack,
    SwitchPresetWindowWidth,
    SwitchPresetWindowWidthBack,
    #[knuffel(skip)]
    SwitchPresetWindowWidthById(u64),
    #[knuffel(skip)]
    SwitchPresetWindowWidthBackById(u64),
    SwitchPresetWindowHeight,
    SwitchPresetWindowHeightBack,
    #[knuffel(skip)]
    SwitchPresetWindowHeightById(u64),
    #[knuffel(skip)]
    SwitchPresetWindowHeightBackById(u64),
    MaximizeColumn,
    MaximizeWindowToEdges,
    #[knuffel(skip)]
    MaximizeWindowToEdgesById(u64),
    SetColumnWidth(#[knuffel(argument, str)] SizeChange),
    ExpandColumnToAvailableWidth,
    SwitchLayout(#[knuffel(argument, str)] LayoutSwitchTarget),
    ShowHotkeyOverlay,
    MoveWorkspaceToMonitorLeft,
    MoveWorkspaceToMonitorRight,
    MoveWorkspaceToMonitorDown,
    MoveWorkspaceToMonitorUp,
    MoveWorkspaceToMonitorPrevious,
    MoveWorkspaceToMonitorNext,
    ToggleWindowFloating,
    #[knuffel(skip)]
    ToggleWindowFloatingById(u64),
    MoveWindowToFloating,
    #[knuffel(skip)]
    MoveWindowToFloatingById(u64),
    MoveWindowToTiling,
    #[knuffel(skip)]
    MoveWindowToTilingById(u64),
    FocusFloating,
    FocusTiling,
    SwitchFocusBetweenFloatingAndTiling,
    #[knuffel(skip)]
    MoveFloatingWindowById {
        id: Option<u64>,
        x: PositionChange,
        y: PositionChange,
    },
    ToggleWindowRuleOpacity,
    #[knuffel(skip)]
    ToggleWindowRuleOpacityById(u64),
    SetDynamicCastWindow,
    #[knuffel(skip)]
    SetDynamicCastWindowById(u64),
    SetDynamicCastMonitor(#[knuffel(argument)] Option<String>),
    ClearDynamicCastTarget,
    #[knuffel(skip)]
    StopCast(u64),
    ToggleOverview,
    OpenOverview,
    CloseOverview,
    #[knuffel(skip)]
    ToggleWindowUrgent(u64),
    #[knuffel(skip)]
    SetWindowUrgent(u64),
    #[knuffel(skip)]
    UnsetWindowUrgent(u64),
    #[knuffel(skip)]
    LoadConfigFile(#[knuffel(argument)] Option<String>),
    #[knuffel(skip)]
    MruAdvance {
        direction: MruDirection,
        scope: Option<MruScope>,
        filter: Option<MruFilter>,
    },
    #[knuffel(skip)]
    MruConfirm,
    #[knuffel(skip)]
    MruCancel,
    #[knuffel(skip)]
    MruCloseCurrentWindow,
    #[knuffel(skip)]
    MruFirst,
    #[knuffel(skip)]
    MruLast,
    #[knuffel(skip)]
    MruSetScope(MruScope),
    #[knuffel(skip)]
    MruCycleScope,
    #[knuffel(skip)]
    CreateActivity(String),
    #[knuffel(skip)]
    RemoveActivity(ActivityReference),
    #[knuffel(skip)]
    RenameActivity(ActivityReference, String),
    #[knuffel(skip)]
    SwitchActivity(ActivityReference),
    SwitchActivityPrevious,
    #[knuffel(skip)]
    AddWorkspaceToActivity(Option<WorkspaceReference>, ActivityReference),
    #[knuffel(skip)]
    RemoveWorkspaceFromActivity(Option<WorkspaceReference>, ActivityReference),
    // IPC-only: `Vec<ActivityReference>` does not KDL-encode, so there is no
    // keybinding surface for this action.
    #[knuffel(skip)]
    SetWorkspaceActivities(Option<WorkspaceReference>, Vec<ActivityReference>),
    // IPC-only: the `focus: bool` parameter distinguishes two hard-block
    // gates at dispatch (weaker gesture-only vs full
    // `is_activity_switch_hard_blocked`), so the action is not surfaced as
    // a KDL bind to avoid a silent mode-selection.
    #[knuffel(skip)]
    MoveWorkspaceToActivity(Option<WorkspaceReference>, ActivityReference, bool),
    ToggleWorkspaceSticky,
    #[knuffel(skip)]
    ToggleWorkspaceStickyByRef(#[knuffel(argument)] WorkspaceReference),
    SetWorkspaceSticky,
    #[knuffel(skip)]
    SetWorkspaceStickyByRef(#[knuffel(argument)] WorkspaceReference),
    UnsetWorkspaceSticky,
    #[knuffel(skip)]
    UnsetWorkspaceStickyByRef(#[knuffel(argument)] WorkspaceReference),
}

impl From<jiji_ipc::Action> for Action {
    fn from(value: jiji_ipc::Action) -> Self {
        match value {
            jiji_ipc::Action::Quit { skip_confirmation } => Self::Quit(skip_confirmation),
            jiji_ipc::Action::PowerOffMonitors {} => Self::PowerOffMonitors,
            jiji_ipc::Action::PowerOnMonitors {} => Self::PowerOnMonitors,
            jiji_ipc::Action::Spawn { command } => Self::Spawn(command),
            jiji_ipc::Action::SpawnSh { command } => Self::SpawnSh(command),
            jiji_ipc::Action::DoScreenTransition { delay_ms } => Self::DoScreenTransition(delay_ms),
            jiji_ipc::Action::Screenshot { show_pointer, path } => {
                Self::Screenshot(show_pointer, path)
            }
            jiji_ipc::Action::ScreenshotScreen {
                write_to_disk,
                show_pointer,
                path,
            } => Self::ScreenshotScreen(write_to_disk, show_pointer, path),
            jiji_ipc::Action::ScreenshotWindow {
                id: None,
                write_to_disk,
                show_pointer,
                path,
            } => Self::ScreenshotWindow(write_to_disk, show_pointer, path),
            jiji_ipc::Action::ScreenshotWindow {
                id: Some(id),
                write_to_disk,
                show_pointer,
                path,
            } => Self::ScreenshotWindowById {
                id,
                write_to_disk,
                show_pointer,
                path,
            },
            jiji_ipc::Action::ToggleKeyboardShortcutsInhibit {} => {
                Self::ToggleKeyboardShortcutsInhibit
            }
            jiji_ipc::Action::CloseWindow { id: None } => Self::CloseWindow,
            jiji_ipc::Action::CloseWindow { id: Some(id) } => Self::CloseWindowById(id),
            jiji_ipc::Action::FullscreenWindow { id: None } => Self::FullscreenWindow,
            jiji_ipc::Action::FullscreenWindow { id: Some(id) } => Self::FullscreenWindowById(id),
            jiji_ipc::Action::ToggleWindowedFullscreen { id: None } => {
                Self::ToggleWindowedFullscreen
            }
            jiji_ipc::Action::ToggleWindowedFullscreen { id: Some(id) } => {
                Self::ToggleWindowedFullscreenById(id)
            }
            jiji_ipc::Action::FocusWindow { id } => Self::FocusWindow(id),
            jiji_ipc::Action::FocusWindowInColumn { index } => Self::FocusWindowInColumn(index),
            jiji_ipc::Action::FocusWindowPrevious {} => Self::FocusWindowPrevious,
            jiji_ipc::Action::FocusColumnLeft {} => Self::FocusColumnLeft,
            jiji_ipc::Action::FocusColumnRight {} => Self::FocusColumnRight,
            jiji_ipc::Action::FocusColumnFirst {} => Self::FocusColumnFirst,
            jiji_ipc::Action::FocusColumnLast {} => Self::FocusColumnLast,
            jiji_ipc::Action::FocusColumnRightOrFirst {} => Self::FocusColumnRightOrFirst,
            jiji_ipc::Action::FocusColumnLeftOrLast {} => Self::FocusColumnLeftOrLast,
            jiji_ipc::Action::FocusColumn { index } => Self::FocusColumn(index),
            jiji_ipc::Action::FocusWindowOrMonitorUp {} => Self::FocusWindowOrMonitorUp,
            jiji_ipc::Action::FocusWindowOrMonitorDown {} => Self::FocusWindowOrMonitorDown,
            jiji_ipc::Action::FocusColumnOrMonitorLeft {} => Self::FocusColumnOrMonitorLeft,
            jiji_ipc::Action::FocusColumnOrMonitorRight {} => Self::FocusColumnOrMonitorRight,
            jiji_ipc::Action::FocusWindowDown {} => Self::FocusWindowDown,
            jiji_ipc::Action::FocusWindowUp {} => Self::FocusWindowUp,
            jiji_ipc::Action::FocusWindowDownOrColumnLeft {} => Self::FocusWindowDownOrColumnLeft,
            jiji_ipc::Action::FocusWindowDownOrColumnRight {} => Self::FocusWindowDownOrColumnRight,
            jiji_ipc::Action::FocusWindowUpOrColumnLeft {} => Self::FocusWindowUpOrColumnLeft,
            jiji_ipc::Action::FocusWindowUpOrColumnRight {} => Self::FocusWindowUpOrColumnRight,
            jiji_ipc::Action::FocusWindowOrWorkspaceDown {} => Self::FocusWindowOrWorkspaceDown,
            jiji_ipc::Action::FocusWindowOrWorkspaceUp {} => Self::FocusWindowOrWorkspaceUp,
            jiji_ipc::Action::FocusWindowTop {} => Self::FocusWindowTop,
            jiji_ipc::Action::FocusWindowBottom {} => Self::FocusWindowBottom,
            jiji_ipc::Action::FocusWindowDownOrTop {} => Self::FocusWindowDownOrTop,
            jiji_ipc::Action::FocusWindowUpOrBottom {} => Self::FocusWindowUpOrBottom,
            jiji_ipc::Action::MoveColumnLeft {} => Self::MoveColumnLeft,
            jiji_ipc::Action::MoveColumnRight {} => Self::MoveColumnRight,
            jiji_ipc::Action::MoveColumnToFirst {} => Self::MoveColumnToFirst,
            jiji_ipc::Action::MoveColumnToLast {} => Self::MoveColumnToLast,
            jiji_ipc::Action::MoveColumnToIndex { index } => Self::MoveColumnToIndex(index),
            jiji_ipc::Action::MoveColumnLeftOrToMonitorLeft {} => {
                Self::MoveColumnLeftOrToMonitorLeft
            }
            jiji_ipc::Action::MoveColumnRightOrToMonitorRight {} => {
                Self::MoveColumnRightOrToMonitorRight
            }
            jiji_ipc::Action::MoveWindowDown {} => Self::MoveWindowDown,
            jiji_ipc::Action::MoveWindowUp {} => Self::MoveWindowUp,
            jiji_ipc::Action::MoveWindowDownOrToWorkspaceDown {} => {
                Self::MoveWindowDownOrToWorkspaceDown
            }
            jiji_ipc::Action::MoveWindowUpOrToWorkspaceUp {} => Self::MoveWindowUpOrToWorkspaceUp,
            jiji_ipc::Action::ConsumeOrExpelWindowLeft { id: None } => {
                Self::ConsumeOrExpelWindowLeft
            }
            jiji_ipc::Action::ConsumeOrExpelWindowLeft { id: Some(id) } => {
                Self::ConsumeOrExpelWindowLeftById(id)
            }
            jiji_ipc::Action::ConsumeOrExpelWindowRight { id: None } => {
                Self::ConsumeOrExpelWindowRight
            }
            jiji_ipc::Action::ConsumeOrExpelWindowRight { id: Some(id) } => {
                Self::ConsumeOrExpelWindowRightById(id)
            }
            jiji_ipc::Action::ConsumeWindowIntoColumn {} => Self::ConsumeWindowIntoColumn,
            jiji_ipc::Action::ExpelWindowFromColumn {} => Self::ExpelWindowFromColumn,
            jiji_ipc::Action::SwapWindowRight {} => Self::SwapWindowRight,
            jiji_ipc::Action::SwapWindowLeft {} => Self::SwapWindowLeft,
            jiji_ipc::Action::ToggleColumnTabbedDisplay {} => Self::ToggleColumnTabbedDisplay,
            jiji_ipc::Action::SetColumnDisplay { display } => Self::SetColumnDisplay(display),
            jiji_ipc::Action::CenterColumn {} => Self::CenterColumn,
            jiji_ipc::Action::CenterWindow { id: None } => Self::CenterWindow,
            jiji_ipc::Action::CenterWindow { id: Some(id) } => Self::CenterWindowById(id),
            jiji_ipc::Action::CenterVisibleColumns {} => Self::CenterVisibleColumns,
            jiji_ipc::Action::FocusWorkspaceDown {} => Self::FocusWorkspaceDown,
            jiji_ipc::Action::FocusWorkspaceUp {} => Self::FocusWorkspaceUp,
            jiji_ipc::Action::FocusWorkspace { reference } => {
                Self::FocusWorkspace(WorkspaceReference::from(reference))
            }
            jiji_ipc::Action::FocusWorkspacePrevious {} => Self::FocusWorkspacePrevious,
            jiji_ipc::Action::MoveWindowToWorkspaceDown { focus } => {
                Self::MoveWindowToWorkspaceDown(focus)
            }
            jiji_ipc::Action::MoveWindowToWorkspaceUp { focus } => {
                Self::MoveWindowToWorkspaceUp(focus)
            }
            jiji_ipc::Action::MoveWindowToWorkspace {
                window_id: None,
                reference,
                focus,
            } => Self::MoveWindowToWorkspace(WorkspaceReference::from(reference), focus),
            jiji_ipc::Action::MoveWindowToWorkspace {
                window_id: Some(window_id),
                reference,
                focus,
            } => Self::MoveWindowToWorkspaceById {
                window_id,
                reference: WorkspaceReference::from(reference),
                focus,
            },
            jiji_ipc::Action::MoveColumnToWorkspaceDown { focus } => {
                Self::MoveColumnToWorkspaceDown(focus)
            }
            jiji_ipc::Action::MoveColumnToWorkspaceUp { focus } => {
                Self::MoveColumnToWorkspaceUp(focus)
            }
            jiji_ipc::Action::MoveColumnToWorkspace { reference, focus } => {
                Self::MoveColumnToWorkspace(WorkspaceReference::from(reference), focus)
            }
            jiji_ipc::Action::MoveWorkspaceDown {} => Self::MoveWorkspaceDown,
            jiji_ipc::Action::MoveWorkspaceUp {} => Self::MoveWorkspaceUp,
            jiji_ipc::Action::SetWorkspaceName {
                name,
                workspace: None,
            } => Self::SetWorkspaceName(name),
            jiji_ipc::Action::SetWorkspaceName {
                name,
                workspace: Some(reference),
            } => Self::SetWorkspaceNameByRef {
                name,
                reference: WorkspaceReference::from(reference),
            },
            jiji_ipc::Action::UnsetWorkspaceName { reference: None } => Self::UnsetWorkspaceName,
            jiji_ipc::Action::UnsetWorkspaceName {
                reference: Some(reference),
            } => Self::UnsetWorkSpaceNameByRef(WorkspaceReference::from(reference)),
            jiji_ipc::Action::FocusMonitorLeft {} => Self::FocusMonitorLeft,
            jiji_ipc::Action::FocusMonitorRight {} => Self::FocusMonitorRight,
            jiji_ipc::Action::FocusMonitorDown {} => Self::FocusMonitorDown,
            jiji_ipc::Action::FocusMonitorUp {} => Self::FocusMonitorUp,
            jiji_ipc::Action::FocusMonitorPrevious {} => Self::FocusMonitorPrevious,
            jiji_ipc::Action::FocusMonitorNext {} => Self::FocusMonitorNext,
            jiji_ipc::Action::FocusMonitor { output } => Self::FocusMonitor(output),
            jiji_ipc::Action::MoveWindowToMonitorLeft {} => Self::MoveWindowToMonitorLeft,
            jiji_ipc::Action::MoveWindowToMonitorRight {} => Self::MoveWindowToMonitorRight,
            jiji_ipc::Action::MoveWindowToMonitorDown {} => Self::MoveWindowToMonitorDown,
            jiji_ipc::Action::MoveWindowToMonitorUp {} => Self::MoveWindowToMonitorUp,
            jiji_ipc::Action::MoveWindowToMonitorPrevious {} => Self::MoveWindowToMonitorPrevious,
            jiji_ipc::Action::MoveWindowToMonitorNext {} => Self::MoveWindowToMonitorNext,
            jiji_ipc::Action::MoveWindowToMonitor { id: None, output } => {
                Self::MoveWindowToMonitor(output)
            }
            jiji_ipc::Action::MoveWindowToMonitor {
                id: Some(id),
                output,
            } => Self::MoveWindowToMonitorById { id, output },
            jiji_ipc::Action::MoveColumnToMonitorLeft {} => Self::MoveColumnToMonitorLeft,
            jiji_ipc::Action::MoveColumnToMonitorRight {} => Self::MoveColumnToMonitorRight,
            jiji_ipc::Action::MoveColumnToMonitorDown {} => Self::MoveColumnToMonitorDown,
            jiji_ipc::Action::MoveColumnToMonitorUp {} => Self::MoveColumnToMonitorUp,
            jiji_ipc::Action::MoveColumnToMonitorPrevious {} => Self::MoveColumnToMonitorPrevious,
            jiji_ipc::Action::MoveColumnToMonitorNext {} => Self::MoveColumnToMonitorNext,
            jiji_ipc::Action::MoveColumnToMonitor { output } => Self::MoveColumnToMonitor(output),
            jiji_ipc::Action::SetWindowWidth { id: None, change } => Self::SetWindowWidth(change),
            jiji_ipc::Action::SetWindowWidth {
                id: Some(id),
                change,
            } => Self::SetWindowWidthById { id, change },
            jiji_ipc::Action::SetWindowHeight { id: None, change } => Self::SetWindowHeight(change),
            jiji_ipc::Action::SetWindowHeight {
                id: Some(id),
                change,
            } => Self::SetWindowHeightById { id, change },
            jiji_ipc::Action::ResetWindowHeight { id: None } => Self::ResetWindowHeight,
            jiji_ipc::Action::ResetWindowHeight { id: Some(id) } => Self::ResetWindowHeightById(id),
            jiji_ipc::Action::SwitchPresetColumnWidth {} => Self::SwitchPresetColumnWidth,
            jiji_ipc::Action::SwitchPresetColumnWidthBack {} => Self::SwitchPresetColumnWidthBack,
            jiji_ipc::Action::SwitchPresetWindowWidth { id: None } => Self::SwitchPresetWindowWidth,
            jiji_ipc::Action::SwitchPresetWindowWidthBack { id: None } => {
                Self::SwitchPresetWindowWidthBack
            }
            jiji_ipc::Action::SwitchPresetWindowWidth { id: Some(id) } => {
                Self::SwitchPresetWindowWidthById(id)
            }
            jiji_ipc::Action::SwitchPresetWindowWidthBack { id: Some(id) } => {
                Self::SwitchPresetWindowWidthBackById(id)
            }
            jiji_ipc::Action::SwitchPresetWindowHeight { id: None } => {
                Self::SwitchPresetWindowHeight
            }
            jiji_ipc::Action::SwitchPresetWindowHeightBack { id: None } => {
                Self::SwitchPresetWindowHeightBack
            }
            jiji_ipc::Action::SwitchPresetWindowHeight { id: Some(id) } => {
                Self::SwitchPresetWindowHeightById(id)
            }
            jiji_ipc::Action::SwitchPresetWindowHeightBack { id: Some(id) } => {
                Self::SwitchPresetWindowHeightBackById(id)
            }
            jiji_ipc::Action::MaximizeColumn {} => Self::MaximizeColumn,
            jiji_ipc::Action::MaximizeWindowToEdges { id: None } => Self::MaximizeWindowToEdges,
            jiji_ipc::Action::MaximizeWindowToEdges { id: Some(id) } => {
                Self::MaximizeWindowToEdgesById(id)
            }
            jiji_ipc::Action::SetColumnWidth { change } => Self::SetColumnWidth(change),
            jiji_ipc::Action::ExpandColumnToAvailableWidth {} => Self::ExpandColumnToAvailableWidth,
            jiji_ipc::Action::SwitchLayout { layout } => Self::SwitchLayout(layout),
            jiji_ipc::Action::ShowHotkeyOverlay {} => Self::ShowHotkeyOverlay,
            jiji_ipc::Action::MoveWorkspaceToMonitorLeft {} => Self::MoveWorkspaceToMonitorLeft,
            jiji_ipc::Action::MoveWorkspaceToMonitorRight {} => Self::MoveWorkspaceToMonitorRight,
            jiji_ipc::Action::MoveWorkspaceToMonitorDown {} => Self::MoveWorkspaceToMonitorDown,
            jiji_ipc::Action::MoveWorkspaceToMonitorUp {} => Self::MoveWorkspaceToMonitorUp,
            jiji_ipc::Action::MoveWorkspaceToMonitorPrevious {} => {
                Self::MoveWorkspaceToMonitorPrevious
            }
            jiji_ipc::Action::MoveWorkspaceToIndex {
                index,
                reference: Some(reference),
            } => Self::MoveWorkspaceToIndexByRef {
                new_idx: index,
                reference: WorkspaceReference::from(reference),
            },
            jiji_ipc::Action::MoveWorkspaceToIndex {
                index,
                reference: None,
            } => Self::MoveWorkspaceToIndex(index),
            jiji_ipc::Action::MoveWorkspaceToMonitor {
                output,
                reference: Some(reference),
            } => Self::MoveWorkspaceToMonitorByRef {
                output_name: output,
                reference: WorkspaceReference::from(reference),
            },
            jiji_ipc::Action::MoveWorkspaceToMonitor {
                output,
                reference: None,
            } => Self::MoveWorkspaceToMonitor(output),
            jiji_ipc::Action::MoveWorkspaceToMonitorNext {} => Self::MoveWorkspaceToMonitorNext,
            jiji_ipc::Action::ToggleDebugTint {} => Self::ToggleDebugTint,
            jiji_ipc::Action::DebugToggleOpaqueRegions {} => Self::DebugToggleOpaqueRegions,
            jiji_ipc::Action::DebugToggleDamage {} => Self::DebugToggleDamage,
            jiji_ipc::Action::ToggleWindowFloating { id: None } => Self::ToggleWindowFloating,
            jiji_ipc::Action::ToggleWindowFloating { id: Some(id) } => {
                Self::ToggleWindowFloatingById(id)
            }
            jiji_ipc::Action::MoveWindowToFloating { id: None } => Self::MoveWindowToFloating,
            jiji_ipc::Action::MoveWindowToFloating { id: Some(id) } => {
                Self::MoveWindowToFloatingById(id)
            }
            jiji_ipc::Action::MoveWindowToTiling { id: None } => Self::MoveWindowToTiling,
            jiji_ipc::Action::MoveWindowToTiling { id: Some(id) } => {
                Self::MoveWindowToTilingById(id)
            }
            jiji_ipc::Action::FocusFloating {} => Self::FocusFloating,
            jiji_ipc::Action::FocusTiling {} => Self::FocusTiling,
            jiji_ipc::Action::SwitchFocusBetweenFloatingAndTiling {} => {
                Self::SwitchFocusBetweenFloatingAndTiling
            }
            jiji_ipc::Action::MoveFloatingWindow { id, x, y } => {
                Self::MoveFloatingWindowById { id, x, y }
            }
            jiji_ipc::Action::ToggleWindowRuleOpacity { id: None } => Self::ToggleWindowRuleOpacity,
            jiji_ipc::Action::ToggleWindowRuleOpacity { id: Some(id) } => {
                Self::ToggleWindowRuleOpacityById(id)
            }
            jiji_ipc::Action::SetDynamicCastWindow { id: None } => Self::SetDynamicCastWindow,
            jiji_ipc::Action::SetDynamicCastWindow { id: Some(id) } => {
                Self::SetDynamicCastWindowById(id)
            }
            jiji_ipc::Action::SetDynamicCastMonitor { output } => {
                Self::SetDynamicCastMonitor(output)
            }
            jiji_ipc::Action::ClearDynamicCastTarget {} => Self::ClearDynamicCastTarget,
            jiji_ipc::Action::StopCast { session_id } => Self::StopCast(session_id),
            jiji_ipc::Action::ToggleOverview {} => Self::ToggleOverview,
            jiji_ipc::Action::OpenOverview {} => Self::OpenOverview,
            jiji_ipc::Action::CloseOverview {} => Self::CloseOverview,
            jiji_ipc::Action::ToggleWindowUrgent { id } => Self::ToggleWindowUrgent(id),
            jiji_ipc::Action::SetWindowUrgent { id } => Self::SetWindowUrgent(id),
            jiji_ipc::Action::UnsetWindowUrgent { id } => Self::UnsetWindowUrgent(id),
            jiji_ipc::Action::LoadConfigFile { path } => Self::LoadConfigFile(path),
            jiji_ipc::Action::CreateActivity { name } => Self::CreateActivity(name),
            jiji_ipc::Action::RemoveActivity { activity } => Self::RemoveActivity(activity.into()),
            jiji_ipc::Action::RenameActivity { activity, name } => {
                Self::RenameActivity(activity.into(), name)
            }
            jiji_ipc::Action::SwitchActivity { activity } => Self::SwitchActivity(activity.into()),
            jiji_ipc::Action::SwitchActivityPrevious {} => Self::SwitchActivityPrevious,
            jiji_ipc::Action::AddWorkspaceToActivity {
                workspace,
                activity,
            } => Self::AddWorkspaceToActivity(
                workspace.map(WorkspaceReference::from),
                activity.into(),
            ),
            jiji_ipc::Action::RemoveWorkspaceFromActivity {
                workspace,
                activity,
            } => Self::RemoveWorkspaceFromActivity(
                workspace.map(WorkspaceReference::from),
                activity.into(),
            ),
            jiji_ipc::Action::SetWorkspaceActivities {
                workspace,
                activities,
            } => Self::SetWorkspaceActivities(
                workspace.map(WorkspaceReference::from),
                activities
                    .into_iter()
                    .map(ActivityReference::from)
                    .collect(),
            ),
            jiji_ipc::Action::MoveWorkspaceToActivity {
                workspace,
                activity,
                focus,
            } => Self::MoveWorkspaceToActivity(
                workspace.map(WorkspaceReference::from),
                activity.into(),
                focus,
            ),
            jiji_ipc::Action::ToggleWorkspaceSticky { workspace: None } => {
                Self::ToggleWorkspaceSticky
            }
            jiji_ipc::Action::ToggleWorkspaceSticky {
                workspace: Some(reference),
            } => Self::ToggleWorkspaceStickyByRef(WorkspaceReference::from(reference)),
            jiji_ipc::Action::SetWorkspaceSticky { workspace: None } => Self::SetWorkspaceSticky,
            jiji_ipc::Action::SetWorkspaceSticky {
                workspace: Some(reference),
            } => Self::SetWorkspaceStickyByRef(WorkspaceReference::from(reference)),
            jiji_ipc::Action::UnsetWorkspaceSticky { workspace: None } => {
                Self::UnsetWorkspaceSticky
            }
            jiji_ipc::Action::UnsetWorkspaceSticky {
                workspace: Some(reference),
            } => Self::UnsetWorkspaceStickyByRef(WorkspaceReference::from(reference)),
            // jiji_ipc::Action is #[non_exhaustive]: any new variant added to
            // jiji_ipc without a matching arm here is a coding error that
            // surfaces as a panic when the unmapped action is dispatched.
            _ => unreachable!("unhandled jiji_ipc::Action variant; add a match arm above"),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum WorkspaceReference {
    Id(u64),
    Index(u8),
    Name(String),
}

impl From<WorkspaceReferenceArg> for WorkspaceReference {
    fn from(reference: WorkspaceReferenceArg) -> WorkspaceReference {
        match reference {
            WorkspaceReferenceArg::Id(id) => Self::Id(id),
            WorkspaceReferenceArg::Index(i) => Self::Index(i),
            WorkspaceReferenceArg::Name(n) => Self::Name(n),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ActivityReference {
    Id(u64),
    Name(String),
}

impl From<ActivityReferenceArg> for ActivityReference {
    fn from(reference: ActivityReferenceArg) -> ActivityReference {
        match reference {
            ActivityReferenceArg::Id(id) => Self::Id(id),
            ActivityReferenceArg::Name(n) => Self::Name(n),
        }
    }
}

impl From<ActivityReference> for ActivityReferenceArg {
    fn from(r: ActivityReference) -> Self {
        match r {
            ActivityReference::Id(id) => Self::Id(id),
            ActivityReference::Name(n) => Self::Name(n),
        }
    }
}

impl<S: knuffel::traits::ErrorSpan> knuffel::DecodeScalar<S> for ActivityReference {
    fn type_check(
        type_name: &Option<knuffel::span::Spanned<knuffel::ast::TypeName, S>>,
        ctx: &mut knuffel::decode::Context<S>,
    ) {
        if let Some(type_name) = &type_name {
            ctx.emit_error(DecodeError::unexpected(
                type_name,
                "type name",
                "no type name expected for this node",
            ));
        }
    }

    fn raw_decode(
        val: &knuffel::span::Spanned<knuffel::ast::Literal, S>,
        ctx: &mut knuffel::decode::Context<S>,
    ) -> Result<ActivityReference, DecodeError<S>> {
        match &**val {
            knuffel::ast::Literal::String(ref s) => Ok(ActivityReference::Name(s.clone().into())),
            knuffel::ast::Literal::Int(ref value) => match value.try_into() {
                Ok(v) => Ok(ActivityReference::Id(v)),
                Err(e) => {
                    ctx.emit_error(DecodeError::conversion(val, e));
                    Ok(ActivityReference::Id(0))
                }
            },
            _ => {
                ctx.emit_error(DecodeError::unsupported(
                    val,
                    "Unsupported value, only numbers and strings are recognized",
                ));
                Ok(ActivityReference::Id(0))
            }
        }
    }
}

impl<S: knuffel::traits::ErrorSpan> knuffel::DecodeScalar<S> for WorkspaceReference {
    fn type_check(
        type_name: &Option<knuffel::span::Spanned<knuffel::ast::TypeName, S>>,
        ctx: &mut knuffel::decode::Context<S>,
    ) {
        if let Some(type_name) = &type_name {
            ctx.emit_error(DecodeError::unexpected(
                type_name,
                "type name",
                "no type name expected for this node",
            ));
        }
    }

    fn raw_decode(
        val: &knuffel::span::Spanned<knuffel::ast::Literal, S>,
        ctx: &mut knuffel::decode::Context<S>,
    ) -> Result<WorkspaceReference, DecodeError<S>> {
        match &**val {
            knuffel::ast::Literal::String(ref s) => Ok(WorkspaceReference::Name(s.clone().into())),
            knuffel::ast::Literal::Int(ref value) => match value.try_into() {
                Ok(v) => Ok(WorkspaceReference::Index(v)),
                Err(e) => {
                    ctx.emit_error(DecodeError::conversion(val, e));
                    Ok(WorkspaceReference::Index(0))
                }
            },
            _ => {
                ctx.emit_error(DecodeError::unsupported(
                    val,
                    "Unsupported value, only numbers and strings are recognized",
                ));
                Ok(WorkspaceReference::Index(0))
            }
        }
    }
}

impl<S> knuffel::Decode<S> for Binds
where
    S: knuffel::traits::ErrorSpan,
{
    fn decode_node(
        node: &knuffel::ast::SpannedNode<S>,
        ctx: &mut knuffel::decode::Context<S>,
    ) -> Result<Self, DecodeError<S>> {
        expect_only_children(node, ctx);

        let mut seen_keys: HashMap<Key, &knuffel::ast::SpannedNode<S>> = HashMap::new();

        let mut binds = Vec::new();

        for child in node.children() {
            match Bind::decode_node(child, ctx) {
                Err(e) => {
                    ctx.emit_error(e);
                }
                Ok(bind) => {
                    match seen_keys.entry(bind.key) {
                        Entry::Occupied(entry) => {
                            // Even though it's technically incorrect, we use
                            // `DecodeError::Missing` here because it labels the bind with
                            // "node starts here", which is the least bad option
                            ctx.emit_error(DecodeError::missing(
                                entry.get(),
                                "keybind first defined here",
                            ));

                            ctx.emit_error(DecodeError::unexpected(
                                &child.node_name,
                                "keybind",
                                "duplicate keybind later defined here",
                            ));
                        }
                        Entry::Vacant(entry) => {
                            entry.insert(child);
                            binds.push(bind);
                        }
                    }
                }
            }
        }

        Ok(Self(binds))
    }
}

impl<S> knuffel::Decode<S> for Bind
where
    S: knuffel::traits::ErrorSpan,
{
    fn decode_node(
        node: &knuffel::ast::SpannedNode<S>,
        ctx: &mut knuffel::decode::Context<S>,
    ) -> Result<Self, DecodeError<S>> {
        if let Some(type_name) = &node.type_name {
            ctx.emit_error(DecodeError::unexpected(
                type_name,
                "type name",
                "no type name expected for this node",
            ));
        }

        for val in node.arguments.iter() {
            ctx.emit_error(DecodeError::unexpected(
                &val.literal,
                "argument",
                "no arguments expected for this node",
            ));
        }

        let key = node
            .node_name
            .parse::<Key>()
            .map_err(|e| DecodeError::conversion(&node.node_name, e.wrap_err("invalid keybind")))?;

        let mut repeat = true;
        let mut cooldown = None;
        let mut allow_when_locked = false;
        let mut allow_when_locked_node = None;
        let mut allow_inhibiting = true;
        let mut hotkey_overlay_title = None;
        for (name, val) in &node.properties {
            match &***name {
                "repeat" => {
                    repeat = knuffel::traits::DecodeScalar::decode(val, ctx)?;
                }
                "cooldown-ms" => {
                    cooldown = Some(Duration::from_millis(
                        knuffel::traits::DecodeScalar::decode(val, ctx)?,
                    ));
                }
                "allow-when-locked" => {
                    allow_when_locked = knuffel::traits::DecodeScalar::decode(val, ctx)?;
                    allow_when_locked_node = Some(name);
                }
                "allow-inhibiting" => {
                    allow_inhibiting = knuffel::traits::DecodeScalar::decode(val, ctx)?;
                }
                "hotkey-overlay-title" => {
                    hotkey_overlay_title = Some(knuffel::traits::DecodeScalar::decode(val, ctx)?);
                }
                name_str => {
                    ctx.emit_error(DecodeError::unexpected(
                        name,
                        "property",
                        format!("unexpected property `{}`", name_str.escape_default()),
                    ));
                }
            }
        }

        let mut children = node.children();

        // If the action is invalid but the key is fine, we still want to return something.
        // That way, the parent can handle the existence of duplicate keybinds,
        // even if their contents are not valid.
        let dummy = Self {
            key,
            action: Action::Spawn(vec![]),
            repeat: true,
            cooldown: None,
            allow_when_locked: false,
            allow_inhibiting: true,
            hotkey_overlay_title: None,
        };

        if let Some(child) = children.next() {
            for unwanted_child in children {
                ctx.emit_error(DecodeError::unexpected(
                    unwanted_child,
                    "node",
                    "only one action is allowed per keybind",
                ));
            }
            match Action::decode_node(child, ctx) {
                Ok(action) => {
                    if !matches!(action, Action::Spawn(_) | Action::SpawnSh(_)) {
                        if let Some(node) = allow_when_locked_node {
                            ctx.emit_error(DecodeError::unexpected(
                                node,
                                "property",
                                "allow-when-locked can only be set on spawn binds",
                            ));
                        }
                    }

                    // The toggle-inhibit action must always be uninhibitable.
                    // Otherwise, it would be impossible to trigger it.
                    if matches!(action, Action::ToggleKeyboardShortcutsInhibit) {
                        allow_inhibiting = false;
                    }

                    Ok(Self {
                        key,
                        action,
                        repeat,
                        cooldown,
                        allow_when_locked,
                        allow_inhibiting,
                        hotkey_overlay_title,
                    })
                }
                Err(e) => {
                    ctx.emit_error(e);
                    Ok(dummy)
                }
            }
        } else {
            ctx.emit_error(DecodeError::missing(
                node,
                "expected an action for this keybind",
            ));
            Ok(dummy)
        }
    }
}

impl FromStr for Key {
    type Err = miette::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut modifiers = Modifiers::empty();

        let mut split = s.split('+');
        let key = split.next_back().unwrap();

        for part in split {
            let part = part.trim();
            if part.eq_ignore_ascii_case("mod") {
                modifiers |= Modifiers::COMPOSITOR
            } else if part.eq_ignore_ascii_case("ctrl") || part.eq_ignore_ascii_case("control") {
                modifiers |= Modifiers::CTRL;
            } else if part.eq_ignore_ascii_case("shift") {
                modifiers |= Modifiers::SHIFT;
            } else if part.eq_ignore_ascii_case("alt") {
                modifiers |= Modifiers::ALT;
            } else if part.eq_ignore_ascii_case("super") || part.eq_ignore_ascii_case("win") {
                modifiers |= Modifiers::SUPER;
            } else if part.eq_ignore_ascii_case("iso_level3_shift")
                || part.eq_ignore_ascii_case("mod5")
            {
                modifiers |= Modifiers::ISO_LEVEL3_SHIFT;
            } else if part.eq_ignore_ascii_case("iso_level5_shift")
                || part.eq_ignore_ascii_case("mod3")
            {
                modifiers |= Modifiers::ISO_LEVEL5_SHIFT;
            } else {
                return Err(miette!("invalid modifier: {part}"));
            }
        }

        let trigger = if key.eq_ignore_ascii_case("MouseLeft") {
            Trigger::MouseLeft
        } else if key.eq_ignore_ascii_case("MouseRight") {
            Trigger::MouseRight
        } else if key.eq_ignore_ascii_case("MouseMiddle") {
            Trigger::MouseMiddle
        } else if key.eq_ignore_ascii_case("MouseBack") {
            Trigger::MouseBack
        } else if key.eq_ignore_ascii_case("MouseForward") {
            Trigger::MouseForward
        } else if key.eq_ignore_ascii_case("WheelScrollDown") {
            Trigger::WheelScrollDown
        } else if key.eq_ignore_ascii_case("WheelScrollUp") {
            Trigger::WheelScrollUp
        } else if key.eq_ignore_ascii_case("WheelScrollLeft") {
            Trigger::WheelScrollLeft
        } else if key.eq_ignore_ascii_case("WheelScrollRight") {
            Trigger::WheelScrollRight
        } else if key.eq_ignore_ascii_case("TouchpadScrollDown") {
            Trigger::TouchpadScrollDown
        } else if key.eq_ignore_ascii_case("TouchpadScrollUp") {
            Trigger::TouchpadScrollUp
        } else if key.eq_ignore_ascii_case("TouchpadScrollLeft") {
            Trigger::TouchpadScrollLeft
        } else if key.eq_ignore_ascii_case("TouchpadScrollRight") {
            Trigger::TouchpadScrollRight
        } else {
            let mut keysym = keysym_from_name(key, KEYSYM_CASE_INSENSITIVE);
            // The keyboard event handling code can receive either
            // XF86ScreenSaver or XF86Screensaver, because there is no
            // case mapping defined between these keysyms. If we just
            // use the case-insensitive version of keysym_from_name it
            // is not possible to bind the uppercase version, because the
            // case-insensitive match prefers the lowercase version when
            // there is a choice.
            //
            // Therefore, when we match this key with the initial
            // case-insensitive match we try a further case-sensitive match
            // (so that either key can be bound). If that fails, we change
            // to the uppercase version because:
            //
            // - A comment in xkb_keysym_from_name (in libxkbcommon) tells us that the uppercase
            //   version is the "best" of the two. [0]
            // - The xkbcommon crate only has a constant for ScreenSaver. [1]
            //
            // [0]: https://github.com/xkbcommon/libxkbcommon/blob/45a118d5325b051343b4b174f60c1434196fa7d4/src/keysym.c#L276
            // [1]: https://docs.rs/xkbcommon/latest/xkbcommon/xkb/keysyms/index.html#:~:text=KEY%5FXF86ScreenSaver
            //
            // See https://github.com/niri-wm/niri/issues/1969
            if keysym == Keysym::XF86_Screensaver {
                keysym = keysym_from_name(key, KEYSYM_NO_FLAGS);
                if keysym.raw() == KEY_NoSymbol {
                    keysym = Keysym::XF86_ScreenSaver;
                }
            }
            if keysym.raw() == KEY_NoSymbol {
                return Err(miette!("invalid key: {key}"));
            }
            Trigger::Keysym(keysym)
        };

        Ok(Key { trigger, modifiers })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_xf86_screensaver() {
        assert_eq!(
            "XF86ScreenSaver".parse::<Key>().unwrap(),
            Key {
                trigger: Trigger::Keysym(Keysym::XF86_ScreenSaver),
                modifiers: Modifiers::empty(),
            },
        );
        assert_eq!(
            "XF86Screensaver".parse::<Key>().unwrap(),
            Key {
                trigger: Trigger::Keysym(Keysym::XF86_Screensaver),
                modifiers: Modifiers::empty(),
            }
        );
        assert_eq!(
            "xf86screensaver".parse::<Key>().unwrap(),
            Key {
                trigger: Trigger::Keysym(Keysym::XF86_ScreenSaver),
                modifiers: Modifiers::empty(),
            }
        );
    }

    #[test]
    fn parse_iso_level_shifts() {
        assert_eq!(
            "ISO_Level3_Shift+A".parse::<Key>().unwrap(),
            Key {
                trigger: Trigger::Keysym(Keysym::a),
                modifiers: Modifiers::ISO_LEVEL3_SHIFT
            },
        );
        assert_eq!(
            "Mod5+A".parse::<Key>().unwrap(),
            Key {
                trigger: Trigger::Keysym(Keysym::a),
                modifiers: Modifiers::ISO_LEVEL3_SHIFT
            },
        );

        assert_eq!(
            "ISO_Level5_Shift+A".parse::<Key>().unwrap(),
            Key {
                trigger: Trigger::Keysym(Keysym::a),
                modifiers: Modifiers::ISO_LEVEL5_SHIFT
            },
        );
        assert_eq!(
            "Mod3+A".parse::<Key>().unwrap(),
            Key {
                trigger: Trigger::Keysym(Keysym::a),
                modifiers: Modifiers::ISO_LEVEL5_SHIFT
            },
        );
    }
}
