use crate::utils::MergeWith;

/// Configuration for the curated window-bookmark surface.
///
/// Unlike the recent-windows switcher this section carries no keybinds of its
/// own — bookmark actions are ordinary binds in the `binds {}` block — so it is
/// a plain scalar section with no first-encounter bind-clearing logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BookmarksConfig {
    pub repress: RepressPolicy,
    pub order: OrderMode,
    pub walk_wrap: bool,
}

impl Default for BookmarksConfig {
    fn default() -> Self {
        Self {
            repress: RepressPolicy::default(),
            order: OrderMode::default(),
            walk_wrap: true,
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
}

impl MergeWith<BookmarksPart> for BookmarksConfig {
    fn merge_with(&mut self, part: &BookmarksPart) {
        merge_clone!((self, part), repress, order, walk_wrap);
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
