use std::collections::HashSet;

use knuffel::errors::DecodeError;

use crate::utils::MergeWith;

/// Configuration for the curated window-bookmark surface.
///
/// Unlike the recent-windows switcher this section carries no keybinds of its
/// own — bookmark actions are ordinary binds in the `binds {}` block — so it is
/// a plain scalar section with no first-encounter bind-clearing logic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BookmarksConfig {
    pub repress: RepressPolicy,
    pub order: OrderMode,
    pub walk_wrap: bool,
    /// Whether a keybind-driven jump to an already-focused bookmark bounces
    /// back to the window that was focused before the jump that landed on it.
    pub return_to_previous: bool,
    /// Letters offered as jump hints, in assignment order. This [`Default`]
    /// impl is the single source of truth for the hint alphabet; the overlay
    /// reads it rather than carrying its own copy.
    pub hint_alphabet: HintAlphabet,
    /// The in-leader-mode command key table. This [`Default`] impl is the
    /// single source of truth for the default command keys; the overlay reads
    /// it rather than carrying its own copy.
    pub mode_keys: ModeKeysConfig,
    /// Whether a successful in-leader-mode command (add, remove, walk) closes
    /// and immediately reopens the overlay (a fresh instance, not a kept-open
    /// one) instead of dismissing outright. Named `mode-sticky` rather than
    /// `sticky` to avoid colliding with this crate's unrelated
    /// sticky-workspaces terminology.
    pub mode_sticky: bool,
}

impl Default for BookmarksConfig {
    fn default() -> Self {
        Self {
            repress: RepressPolicy::default(),
            order: OrderMode::default(),
            walk_wrap: true,
            return_to_previous: true,
            hint_alphabet: HintAlphabet::default(),
            mode_keys: ModeKeysConfig::default(),
            mode_sticky: false,
        }
    }
}

#[derive(knuffel::Decode, Debug, Default, PartialEq)]
pub struct BookmarksPart {
    #[knuffel(child, unwrap(argument))]
    pub repress: Option<RepressPolicy>,
    #[knuffel(child, unwrap(argument))]
    pub order: Option<OrderMode>,
    #[knuffel(child, unwrap(argument))]
    pub walk_wrap: Option<bool>,
    // `return` is a Rust keyword; the raw identifier kebab-cases to the KDL
    // child name `return` the same way a plain identifier would (`child`
    // does not support a `name = ...` override, unlike `property`/`children`).
    #[knuffel(child, unwrap(argument))]
    pub r#return: Option<bool>,
    #[knuffel(child, unwrap(argument))]
    pub hint_alphabet: Option<HintAlphabet>,
    #[knuffel(child)]
    pub mode_keys: Option<ModeKeysConfig>,
    #[knuffel(child, unwrap(argument))]
    pub mode_sticky: Option<bool>,
}

impl MergeWith<BookmarksPart> for BookmarksConfig {
    fn merge_with(&mut self, part: &BookmarksPart) {
        merge_clone!(
            (self, part),
            repress,
            order,
            walk_wrap,
            hint_alphabet,
            mode_keys,
            mode_sticky
        );
        if let Some(x) = &part.r#return {
            self.return_to_previous.clone_from(x);
        }
    }
}

/// What re-bookmarking an already-bookmarked window does.
#[derive(knuffel::DecodeScalar, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RepressPolicy {
    /// Move the existing bookmark to the front of the list.
    #[default]
    MoveToFront,
    /// Remove the existing bookmark instead of moving it. Opt-in, and always
    /// gated behind a confirmation prompt, since re-pressing an existing
    /// bookmark is a destructive action under this policy.
    Remove,
}

/// The order the bookmark list presents and walks in.
#[derive(knuffel::DecodeScalar, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OrderMode {
    /// User-curated order: bookmarks stay where they were added or moved.
    #[default]
    Manual,
    /// Most-recently-used: focusing a bookmarked window promotes it to front.
    Mru,
}

/// The letters offered as jump hints, in assignment order.
///
/// Validated at parse time (see the [`knuffel::DecodeScalar`] impl): non-empty,
/// no duplicate character, no control or whitespace character. The [`Default`]
/// impl carries the home-row-first alphabet the overlay used before the knob
/// existed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HintAlphabet(String);

impl HintAlphabet {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Builds a hint alphabet, enforcing the same invariants config parsing
    /// does: non-empty, no duplicate character, no control or whitespace
    /// character. `Err` carries a human-readable reason.
    pub fn new(alphabet: impl Into<String>) -> Result<Self, String> {
        let alphabet = alphabet.into();
        if alphabet.is_empty() {
            return Err("hint-alphabet must not be empty".to_string());
        }
        let mut seen = HashSet::new();
        for c in alphabet.chars() {
            if c.is_control() || c.is_whitespace() {
                return Err(
                    "hint-alphabet must not contain control or whitespace characters".to_string(),
                );
            }
            if !seen.insert(c) {
                return Err(format!("hint-alphabet has a duplicate character '{c}'"));
            }
        }
        Ok(Self(alphabet))
    }
}

impl Default for HintAlphabet {
    fn default() -> Self {
        Self("asdfghjklqwertyuiopzxcvbnm".to_string())
    }
}

impl<S: knuffel::traits::ErrorSpan> knuffel::DecodeScalar<S> for HintAlphabet {
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
    ) -> Result<HintAlphabet, DecodeError<S>> {
        match &**val {
            knuffel::ast::Literal::String(ref s) => match HintAlphabet::new(s.to_string()) {
                Ok(alphabet) => Ok(alphabet),
                Err(msg) => {
                    ctx.emit_error(DecodeError::conversion(val, msg));
                    // SENTINEL — invalid alphabet, returned only for parse-continuation.
                    Ok(Self(String::new()))
                }
            },
            _ => {
                ctx.emit_error(DecodeError::unsupported(
                    val,
                    "hint-alphabet must be a string",
                ));
                // SENTINEL — invalid alphabet, returned only for parse-continuation.
                Ok(Self(String::new()))
            }
        }
    }
}

/// The leader-mode command key table.
///
/// Each command may be bound to one or more single-character keys; `search`
/// (the key that switches from leader mode into incremental search) is exactly
/// one. Absent commands fall back to this type's [`Default`] impl — the single
/// source of truth for the default keys — and validation runs on the effective
/// (post-default) table, so a configured key colliding with a default of an
/// omitted command is still a load error. No character may be bound to more
/// than one command. Keys are matched against the base (layout-unshifted)
/// keysym, so a shifted character (e.g. a capital letter) validates but never
/// fires.
///
/// Fields are private; construct through [`Self::new`], which enforces the
/// same invariants config parsing does (no duplicate character across
/// commands, no control or whitespace character).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModeKeysConfig {
    add: Vec<char>,
    remove: Vec<char>,
    walk_backward: Vec<char>,
    walk_forward: Vec<char>,
    search: char,
}

impl ModeKeysConfig {
    pub fn add(&self) -> &[char] {
        &self.add
    }

    pub fn remove(&self) -> &[char] {
        &self.remove
    }

    pub fn walk_backward(&self) -> &[char] {
        &self.walk_backward
    }

    pub fn walk_forward(&self) -> &[char] {
        &self.walk_forward
    }

    pub fn search(&self) -> char {
        self.search
    }

    /// Builds a mode-keys table, enforcing the same invariants config parsing
    /// does: no character bound to more than one command, no control or
    /// whitespace character. `Err` carries a human-readable reason.
    pub fn new(
        add: Vec<char>,
        remove: Vec<char>,
        walk_backward: Vec<char>,
        walk_forward: Vec<char>,
        search: char,
    ) -> Result<Self, String> {
        let all = add
            .iter()
            .chain(&remove)
            .chain(&walk_backward)
            .chain(&walk_forward)
            .copied()
            .chain(std::iter::once(search));
        let mut seen = HashSet::new();
        for c in all {
            if c.is_control() || c.is_whitespace() {
                return Err(
                    "mode-keys must not contain control or whitespace characters".to_string(),
                );
            }
            if !seen.insert(c) {
                return Err(format!(
                    "mode-keys character '{c}' is bound to more than one command"
                ));
            }
        }
        Ok(Self {
            add,
            remove,
            walk_backward,
            walk_forward,
            search,
        })
    }
}

impl Default for ModeKeysConfig {
    fn default() -> Self {
        Self {
            add: vec!['a'],
            remove: vec!['d', 'x'],
            walk_backward: vec![','],
            walk_forward: vec!['.'],
            search: '/',
        }
    }
}

impl<S: knuffel::traits::ErrorSpan> knuffel::Decode<S> for ModeKeysConfig {
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
        for val in &node.arguments {
            ctx.emit_error(DecodeError::unexpected(
                &val.literal,
                "argument",
                "no arguments expected for `mode-keys`",
            ));
        }
        for name in node.properties.keys() {
            ctx.emit_error(DecodeError::unexpected(
                name,
                "property",
                "no properties expected for `mode-keys`",
            ));
        }

        let mut add = None;
        let mut remove = None;
        let mut walk_backward = None;
        let mut walk_forward = None;
        let mut search = None;

        for child in node.children() {
            let name = &**child.node_name;
            match name {
                "add" => add = Some(decode_mode_key_chars(child, ctx)),
                "remove" => remove = Some(decode_mode_key_chars(child, ctx)),
                "walk-backward" => walk_backward = Some(decode_mode_key_chars(child, ctx)),
                "walk-forward" => walk_forward = Some(decode_mode_key_chars(child, ctx)),
                "search" => {
                    let chars = decode_mode_key_chars(child, ctx);
                    if chars.len() > 1 {
                        ctx.emit_error(DecodeError::unexpected(
                            child,
                            "node",
                            "`search` takes exactly one key",
                        ));
                    }
                    search = chars.first().copied();
                }
                _ => ctx.emit_error(DecodeError::unexpected(
                    child,
                    "node",
                    format!("node `{name}` is not allowed inside `mode-keys`"),
                )),
            }
        }

        let defaults = ModeKeysConfig::default();
        // Validation (cross-command duplicates, char validity) runs on the
        // effective (post-default) table, so a configured key colliding with a
        // default of an omitted command is caught too.
        match ModeKeysConfig::new(
            add.unwrap_or(defaults.add),
            remove.unwrap_or(defaults.remove),
            walk_backward.unwrap_or(defaults.walk_backward),
            walk_forward.unwrap_or(defaults.walk_forward),
            search.unwrap_or(defaults.search),
        ) {
            Ok(effective) => Ok(effective),
            Err(msg) => {
                ctx.emit_error(DecodeError::unexpected(node, "node", msg));
                // SENTINEL — invalid table, returned only for parse-continuation.
                Ok(ModeKeysConfig::default())
            }
        }
    }
}

/// Extracts the single-character key arguments from a `mode-keys` command child
/// (`add "a"`, `remove "d" "x"`, ...). Each argument must be a string holding
/// exactly one non-control, non-whitespace character. Emits a span-carrying
/// error for any violation and skips the offending argument, so the surrounding
/// parse still fails while continuing to collect diagnostics.
fn decode_mode_key_chars<S: knuffel::traits::ErrorSpan>(
    node: &knuffel::ast::SpannedNode<S>,
    ctx: &mut knuffel::decode::Context<S>,
) -> Vec<char> {
    if let Some(type_name) = &node.type_name {
        ctx.emit_error(DecodeError::unexpected(
            type_name,
            "type name",
            "no type name expected for this node",
        ));
    }
    for name in node.properties.keys() {
        ctx.emit_error(DecodeError::unexpected(
            name,
            "property",
            "no properties expected for a mode-keys command",
        ));
    }
    for child in node.children() {
        ctx.emit_error(DecodeError::unexpected(
            child,
            "node",
            "no children expected for a mode-keys command",
        ));
    }

    let mut chars = Vec::new();
    for val in &node.arguments {
        if let Some(typ) = &val.type_name {
            ctx.emit_error(DecodeError::unexpected(
                typ,
                "type name",
                "no type name expected for a mode-keys entry",
            ));
        }
        match &*val.literal {
            knuffel::ast::Literal::String(ref s) => match single_key_char(s) {
                Some(c) => chars.push(c),
                None => ctx.emit_error(DecodeError::conversion(
                    &val.literal,
                    "each mode-keys entry must be exactly one non-whitespace character",
                )),
            },
            _ => ctx.emit_error(DecodeError::unsupported(
                &val.literal,
                "mode-keys entries must be strings",
            )),
        }
    }

    if node.arguments.is_empty() {
        ctx.emit_error(DecodeError::missing(
            node,
            "a mode-keys command needs at least one key",
        ));
    }
    chars
}

/// The single character of `s`, or `None` if `s` is empty, longer than one
/// character, or that character is a control or whitespace character.
fn single_key_char(s: &str) -> Option<char> {
    let mut chars = s.chars();
    let c = chars.next()?;
    if chars.next().is_some() {
        return None;
    }
    if c.is_control() || c.is_whitespace() {
        return None;
    }
    Some(c)
}
