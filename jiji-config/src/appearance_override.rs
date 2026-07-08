//! Resolved (validated) appearance-override types.
//!
//! [`ResolvedAppearanceOverride`] is converted from the wire-level
//! [`jiji_ipc::AppearanceOverride`] payload via [`TryFrom`], parsing every
//! color and regex up front so composing already-resolved layers with
//! [`flatten`] cannot fail.

use std::collections::BTreeMap;
use std::str::FromStr;

use jiji_ipc::{AppearanceMatch, AppearanceOverride, AppearanceRuleOverride, FocusRingOverride};

use crate::utils::{MergeWith as _, RegexEq};
use crate::window_rule::Match;
use crate::{BorderRule, Color};

/// Identifier for a composed appearance-override layer.
///
/// Layers are folded by [`flatten`] in ascending order, so this identifier
/// doubles as the tiebreak key: a lexically-greater layer id wins a
/// same-field collision.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LayerId(pub String);

/// A validated, resolved appearance override for one layer.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ResolvedAppearanceOverride {
    /// Global (rule-independent) overrides.
    pub global: ResolvedGlobalAppearance,
    /// Per-window-rule overrides. Stored and validated, but not yet composed
    /// by [`flatten`] — evaluation against static window rules lands
    /// separately.
    pub rules: Vec<ResolvedAppearanceRule>,
}

/// Resolved [`jiji_ipc::GlobalAppearanceOverride`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ResolvedGlobalAppearance {
    /// Focus ring overrides, as a partial [`BorderRule`] patch. `off`/`on`
    /// and the gradient fields stay at their `BorderRule::default()` value —
    /// the wire type carries no gradient or on/off surface, so those fields
    /// are always identity under [`MergeWith::merge_with`].
    pub focus_ring: BorderRule,
    /// Background color override.
    pub background_color: Option<Color>,
}

/// Resolved [`AppearanceRuleOverride`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ResolvedAppearanceRule {
    /// Window matchers this rule applies under (AND-ed).
    pub matches: Vec<Match>,
    /// Focus ring overrides to apply when this rule matches.
    pub focus_ring: BorderRule,
}

/// Result of folding every layer's global overrides together, ready to apply
/// on top of the base [`crate::Layout`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FlattenedAppearance {
    /// Composed focus ring override.
    pub focus_ring: BorderRule,
    /// Composed background color override.
    pub background_color: Option<Color>,
}

impl TryFrom<&AppearanceOverride> for ResolvedAppearanceOverride {
    type Error = String;

    fn try_from(wire: &AppearanceOverride) -> Result<Self, Self::Error> {
        let global = ResolvedGlobalAppearance {
            focus_ring: resolve_focus_ring("global.focus_ring", &wire.global.focus_ring)?,
            background_color: wire
                .global
                .background_color
                .as_deref()
                .map(|raw| resolve_color("global.background_color", raw))
                .transpose()?,
        };

        let rules = wire
            .rules
            .iter()
            .enumerate()
            .map(|(i, rule)| resolve_rule(i, rule))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self { global, rules })
    }
}

/// Fold every layer's global overrides together in ascending [`LayerId`]
/// order.
///
/// Per-field, a later layer only wins the fields it actually sets — see
/// `BorderRule::merge_with`. For `background_color`, the last layer that sets
/// it wins.
pub fn flatten(layers: &BTreeMap<LayerId, ResolvedAppearanceOverride>) -> FlattenedAppearance {
    let mut result = FlattenedAppearance::default();
    for layer in layers.values() {
        result.focus_ring.merge_with(&layer.global.focus_ring);
        if layer.global.background_color.is_some() {
            result.background_color = layer.global.background_color;
        }
    }
    result
}

fn resolve_focus_ring(field: &str, wire: &FocusRingOverride) -> Result<BorderRule, String> {
    let mut resolved = BorderRule::default();
    if let Some(raw) = &wire.active_color {
        resolved.active_color = Some(resolve_color(&format!("{field}.active_color"), raw)?);
    }
    if let Some(raw) = &wire.inactive_color {
        resolved.inactive_color = Some(resolve_color(&format!("{field}.inactive_color"), raw)?);
    }
    if let Some(raw) = &wire.urgent_color {
        resolved.urgent_color = Some(resolve_color(&format!("{field}.urgent_color"), raw)?);
    }
    if let Some(width) = wire.width {
        resolved.width = Some(crate::FloatOrInt(width));
    }
    Ok(resolved)
}

fn resolve_rule(
    index: usize,
    wire: &AppearanceRuleOverride,
) -> Result<ResolvedAppearanceRule, String> {
    let matches = wire
        .matches
        .iter()
        .enumerate()
        .map(|(m_index, m)| resolve_match(index, m_index, m))
        .collect::<Result<Vec<_>, _>>()?;
    let focus_ring = resolve_focus_ring(&format!("rules[{index}].focus_ring"), &wire.focus_ring)?;
    Ok(ResolvedAppearanceRule {
        matches,
        focus_ring,
    })
}

fn resolve_match(
    rule_index: usize,
    match_index: usize,
    wire: &AppearanceMatch,
) -> Result<Match, String> {
    let mut resolved = Match::default();
    if let Some(raw) = &wire.title {
        let field = format!("rules[{rule_index}].match[{match_index}].title");
        resolved.title = Some(resolve_regex(&field, raw)?);
    }
    if let Some(raw) = &wire.app_id {
        let field = format!("rules[{rule_index}].match[{match_index}].app_id");
        resolved.app_id = Some(resolve_regex(&field, raw)?);
    }
    Ok(resolved)
}

fn resolve_color(field: &str, raw: &str) -> Result<Color, String> {
    Color::from_str(raw).map_err(|e| format!("{field}: invalid color {raw:?}: {e}"))
}

fn resolve_regex(field: &str, raw: &str) -> Result<RegexEq, String> {
    RegexEq::from_str(raw).map_err(|e| format!("{field}: invalid regex {raw:?}: {e}"))
}

#[cfg(test)]
mod tests {
    use jiji_ipc::{AppearanceMatch, AppearanceRuleOverride};

    use super::*;

    fn full_example() -> AppearanceOverride {
        // `#[non_exhaustive]` on these wire types forbids struct-literal
        // construction outside `jiji-ipc`, even with `..Default::default()`,
        // so build via `Default` + field assignment instead.
        let mut wire = AppearanceOverride::default();
        wire.global.focus_ring.active_color = Some("#7fc8ff".to_string());
        wire.global.focus_ring.inactive_color = Some("#505050".to_string());
        wire.global.focus_ring.urgent_color = Some("#ff0000".to_string());
        wire.global.focus_ring.width = Some(4.0);
        wire.global.background_color = Some("#1e1e2e".to_string());

        let mut rule_match = AppearanceMatch::default();
        rule_match.app_id = Some("firefox".to_string());
        let mut rule = AppearanceRuleOverride::default();
        rule.matches = vec![rule_match];
        rule.focus_ring.active_color = Some("#00ff00".to_string());
        wire.rules = vec![rule];

        wire
    }

    #[test]
    fn try_from_accepts_full_example() {
        let resolved = ResolvedAppearanceOverride::try_from(&full_example())
            .expect("full example must be valid");
        assert_eq!(
            resolved.global.background_color,
            Some(Color::from_str("#1e1e2e").unwrap())
        );
        assert_eq!(resolved.rules.len(), 1);
        assert_eq!(
            resolved.rules[0].matches[0].app_id,
            Some(RegexEq::from_str("firefox").unwrap())
        );
    }

    #[test]
    fn try_from_rejects_bad_color() {
        let mut wire = full_example();
        wire.global.background_color = Some("not-a-color".to_string());
        let err =
            ResolvedAppearanceOverride::try_from(&wire).expect_err("bad color must be rejected");
        assert!(err.contains("global.background_color"), "{err}");
    }

    #[test]
    fn try_from_rejects_bad_regex() {
        let mut wire = full_example();
        wire.rules[0].matches[0].app_id = Some("(unterminated".to_string());
        let err =
            ResolvedAppearanceOverride::try_from(&wire).expect_err("bad regex must be rejected");
        assert!(err.contains("rules[0].match[0].app_id"), "{err}");
    }

    #[test]
    fn flatten_lexical_tiebreak_on_same_field() {
        let mut layers = BTreeMap::new();
        let mut a = ResolvedAppearanceOverride::default();
        a.global.focus_ring.active_color = Some(Color::from_str("#111111").unwrap());
        let mut b = ResolvedAppearanceOverride::default();
        b.global.focus_ring.active_color = Some(Color::from_str("#222222").unwrap());
        layers.insert(LayerId("a".to_string()), a);
        layers.insert(LayerId("z".to_string()), b);

        let flattened = flatten(&layers);
        assert_eq!(
            flattened.focus_ring.active_color,
            Some(Color::from_str("#222222").unwrap()),
            "lexically-greatest layer id must win a same-field collision",
        );
    }

    #[test]
    fn flatten_ignores_rules() {
        // `rules` is stored and validated (see `try_from_accepts_full_example`)
        // but intentionally not composed by `flatten` yet — rule evaluation
        // lands in a later unit. This test must be revisited (and the
        // `background_color`/`focus_ring` assertions below extended to cover
        // rule-derived fields) once that composition exists.
        let mut layers = BTreeMap::new();
        let mut with_rules = ResolvedAppearanceOverride::default();
        with_rules.rules.push(ResolvedAppearanceRule {
            matches: vec![Match {
                app_id: Some(RegexEq::from_str("firefox").unwrap()),
                ..Default::default()
            }],
            focus_ring: BorderRule {
                active_color: Some(Color::from_str("#00ff00").unwrap()),
                ..Default::default()
            },
        });
        layers.insert(LayerId("a".to_string()), with_rules);

        let flattened = flatten(&layers);
        assert_eq!(
            flattened,
            FlattenedAppearance::default(),
            "a layer with only rules (no global fields) must flatten to the default; \
             rules are not yet composed",
        );
    }

    #[test]
    fn flatten_background_color_lexical_tiebreak() {
        let mut layers = BTreeMap::new();
        let mut a = ResolvedAppearanceOverride::default();
        a.global.background_color = Some(Color::from_str("#111111").unwrap());
        let mut z = ResolvedAppearanceOverride::default();
        z.global.background_color = Some(Color::from_str("#222222").unwrap());
        layers.insert(LayerId("a".to_string()), a);
        layers.insert(LayerId("z".to_string()), z);

        let flattened = flatten(&layers);
        assert_eq!(
            flattened.background_color,
            Some(Color::from_str("#222222").unwrap()),
            "lexically-greatest layer id must win background_color too",
        );
    }

    #[test]
    fn flatten_background_color_pass_through() {
        let mut layers = BTreeMap::new();
        let mut a = ResolvedAppearanceOverride::default();
        a.global.background_color = Some(Color::from_str("#111111").unwrap());
        let b = ResolvedAppearanceOverride::default();
        layers.insert(LayerId("a".to_string()), a);
        layers.insert(LayerId("b".to_string()), b);

        let flattened = flatten(&layers);
        assert_eq!(
            flattened.background_color,
            Some(Color::from_str("#111111").unwrap()),
            "layer a's background_color must survive a layer that doesn't set it",
        );
    }

    #[test]
    fn flatten_per_field_pass_through() {
        let mut layers = BTreeMap::new();
        let mut a = ResolvedAppearanceOverride::default();
        a.global.focus_ring.active_color = Some(Color::from_str("#111111").unwrap());
        a.global.focus_ring.inactive_color = Some(Color::from_str("#333333").unwrap());
        let mut b = ResolvedAppearanceOverride::default();
        b.global.focus_ring.width = Some(crate::FloatOrInt(8.0));
        layers.insert(LayerId("a".to_string()), a);
        layers.insert(LayerId("b".to_string()), b);

        let flattened = flatten(&layers);
        assert_eq!(
            flattened.focus_ring.active_color,
            Some(Color::from_str("#111111").unwrap()),
            "layer a's colors must survive a layer that only sets width",
        );
        assert_eq!(
            flattened.focus_ring.inactive_color,
            Some(Color::from_str("#333333").unwrap()),
        );
        assert_eq!(flattened.focus_ring.width, Some(crate::FloatOrInt(8.0)));
    }
}
