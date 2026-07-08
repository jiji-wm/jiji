//! Pins `Action::SetAppearanceOverride` / `Action::ClearAppearanceOverride`
//! dispatch through `do_action_inner`: the override must show up in the
//! composed [`crate::layout::Options`] immediately, survive `reload_config`,
//! and disappear on clear. An invalid payload must be rejected with a
//! terminal error and must not mutate state.

use std::str::FromStr as _;

use jiji_config::{Action, Color, Config};
use jiji_ipc::AppearanceOverride;

use super::client;
use super::fixture::Fixture;
use crate::layout::{DoActionOutcome, LayoutElement as _};

fn override_payload(active_color: &str, background_color: &str) -> AppearanceOverride {
    // `#[non_exhaustive]` on these wire types forbids struct-literal
    // construction outside `jiji-ipc`, even with `..Default::default()`, so
    // build via `Default` + field assignment instead.
    let mut wire = AppearanceOverride::default();
    wire.global.focus_ring.active_color = Some(active_color.to_string());
    wire.global.background_color = Some(background_color.to_string());
    wire
}

#[test]
fn set_appearance_override_updates_composed_options() {
    let mut f = Fixture::new();

    let result = f.niri_state().do_action_inner(
        Action::SetAppearanceOverride {
            layer: "test".to_string(),
            r#override: override_payload("#00ff00", "#112233"),
        },
        false,
    );
    assert_eq!(result, Ok(DoActionOutcome::Handled));

    let options = f.niri().layout.options();
    assert_eq!(
        options.layout.focus_ring.active_color,
        Color::from_str("#00ff00").unwrap(),
    );
    assert_eq!(
        options.layout.background_color,
        Color::from_str("#112233").unwrap(),
    );
}

#[test]
fn appearance_override_survives_reload_config() {
    let mut f = Fixture::new();

    f.niri_state()
        .do_action_inner(
            Action::SetAppearanceOverride {
                layer: "test".to_string(),
                r#override: override_payload("#00ff00", "#112233"),
            },
            false,
        )
        .expect("SetAppearanceOverride must succeed");

    // Reload with a base config that changes an unrelated layout field, to
    // confirm the override is recomposed onto the *new* base rather than
    // just surviving because nothing changed.
    let mut new_config = Config::default();
    new_config.layout.gaps = 32.;
    f.reload_config(new_config);

    let options = f.niri().layout.options();
    assert_eq!(
        options.layout.focus_ring.active_color,
        Color::from_str("#00ff00").unwrap(),
        "override must still be applied after reload_config",
    );
    assert_eq!(
        options.layout.background_color,
        Color::from_str("#112233").unwrap(),
    );
    assert_eq!(
        options.layout.gaps, 32.,
        "the new base config must apply too"
    );
}

#[test]
fn clear_appearance_override_restores_base_values() {
    let mut f = Fixture::new();
    let base_active_color = f.niri().layout.options().layout.focus_ring.active_color;
    let base_background_color = f.niri().layout.options().layout.background_color;

    f.niri_state()
        .do_action_inner(
            Action::SetAppearanceOverride {
                layer: "test".to_string(),
                r#override: override_payload("#00ff00", "#112233"),
            },
            false,
        )
        .expect("SetAppearanceOverride must succeed");

    let result = f.niri_state().do_action_inner(
        Action::ClearAppearanceOverride {
            layer: "test".to_string(),
        },
        false,
    );
    assert_eq!(result, Ok(DoActionOutcome::Handled));

    let options = f.niri().layout.options();
    assert_eq!(options.layout.focus_ring.active_color, base_active_color);
    assert_eq!(options.layout.background_color, base_background_color);
}

#[test]
fn clear_appearance_override_on_absent_layer_is_a_no_op() {
    let mut f = Fixture::new();
    let base_active_color = f.niri().layout.options().layout.focus_ring.active_color;
    let base_background_color = f.niri().layout.options().layout.background_color;

    let result = f.niri_state().do_action_inner(
        Action::ClearAppearanceOverride {
            layer: "never-set".to_string(),
        },
        false,
    );
    assert_eq!(result, Ok(DoActionOutcome::Handled));

    let options = f.niri().layout.options();
    assert_eq!(options.layout.focus_ring.active_color, base_active_color);
    assert_eq!(options.layout.background_color, base_background_color);
}

#[test]
fn clearing_one_layer_preserves_other_layers_contribution() {
    let mut f = Fixture::new();

    f.niri_state()
        .do_action_inner(
            Action::SetAppearanceOverride {
                layer: "a".to_string(),
                r#override: override_payload("#00ff00", "#112233"),
            },
            false,
        )
        .expect("SetAppearanceOverride for layer a must succeed");
    f.niri_state()
        .do_action_inner(
            Action::SetAppearanceOverride {
                layer: "b".to_string(),
                r#override: override_payload("#ff00ff", "#445566"),
            },
            false,
        )
        .expect("SetAppearanceOverride for layer b must succeed");

    let result = f.niri_state().do_action_inner(
        Action::ClearAppearanceOverride {
            layer: "a".to_string(),
        },
        false,
    );
    assert_eq!(result, Ok(DoActionOutcome::Handled));

    // Layer "b" lexically follows "a", so its fields must still win after
    // "a" is cleared — this guards against a clear handler or `flatten` bug
    // that nukes the whole layer map instead of removing just one entry.
    let options = f.niri().layout.options();
    assert_eq!(
        options.layout.focus_ring.active_color,
        Color::from_str("#ff00ff").unwrap(),
    );
    assert_eq!(
        options.layout.background_color,
        Color::from_str("#445566").unwrap(),
    );
}

#[test]
fn appearance_override_wins_over_reload_setting_the_same_field() {
    let mut f = Fixture::new();

    f.niri_state()
        .do_action_inner(
            Action::SetAppearanceOverride {
                layer: "test".to_string(),
                r#override: override_payload("#00ff00", "#112233"),
            },
            false,
        )
        .expect("SetAppearanceOverride must succeed");

    // Unlike `appearance_override_survives_reload_config`, the new base
    // config sets the *same* field the override sets, to confirm the
    // override still wins rather than merely surviving because the base
    // never contended for it.
    let mut new_config = Config::default();
    new_config.layout.focus_ring.active_color = jiji_config::Color::from_str("#ffffff").unwrap();
    f.reload_config(new_config);

    let options = f.niri().layout.options();
    assert_eq!(
        options.layout.focus_ring.active_color,
        Color::from_str("#00ff00").unwrap(),
        "override must still win over a new base config value for the same field",
    );
}

#[test]
fn set_appearance_override_invalid_payload_returns_terminal_error() {
    let mut f = Fixture::new();

    let result = f.niri_state().do_action_inner(
        Action::SetAppearanceOverride {
            layer: "test".to_string(),
            r#override: override_payload("not-a-color", "#112233"),
        },
        false,
    );

    match result {
        Err(crate::layout::DoActionError::AppearanceOverrideInvalid { reason }) => {
            assert!(reason.contains("active_color"), "{reason}");
        }
        other => panic!("expected AppearanceOverrideInvalid, got {other:?}"),
    }

    // No state mutation on the rejected payload.
    let options = f.niri().layout.options();
    assert_ne!(
        options.layout.focus_ring.active_color,
        Color::from_str("#00ff00").unwrap(),
    );
}

fn rule_payload(title_pattern: &str, active_color: &str) -> AppearanceOverride {
    let mut wire = AppearanceOverride::default();
    let mut m = jiji_ipc::AppearanceMatch::default();
    m.title = Some(title_pattern.to_string());
    let mut rule = jiji_ipc::AppearanceRuleOverride::default();
    rule.matches = vec![m];
    rule.focus_ring.active_color = Some(active_color.to_string());
    wire.rules = vec![rule];
    wire
}

fn two_entry_rule_payload(
    non_matching_title: &str,
    matching_title: &str,
    active_color: &str,
) -> AppearanceOverride {
    let mut wire = AppearanceOverride::default();
    let mut m1 = jiji_ipc::AppearanceMatch::default();
    m1.title = Some(non_matching_title.to_string());
    let mut m2 = jiji_ipc::AppearanceMatch::default();
    m2.title = Some(matching_title.to_string());
    let mut rule = jiji_ipc::AppearanceRuleOverride::default();
    rule.matches = vec![m1, m2];
    rule.focus_ring.active_color = Some(active_color.to_string());
    wire.rules = vec![rule];
    wire
}

// Sets up a fixture with one output and a window-rule that matches
// `title="^target$"` and sets `inactive-color` on the static path, so a
// per-window appearance-rule assertion isn't vacuous (it must land on top of
// a fixture that already exercises the earlier static-rule resolution
// stage).
fn set_up_with_static_rule() -> (Fixture, client::ClientId) {
    let config = jiji_config::Config::parse_mem(
        r##"
window-rule {
    match title="^target$"
    focus-ring {
        inactive-color "#111111"
    }
}
"##,
    )
    .unwrap();

    let mut f = Fixture::with_config(config);
    f.add_output(1, (1280, 720));
    let id = f.add_client();
    (f, id)
}

fn map_window_with_title(
    f: &mut Fixture,
    id: client::ClientId,
    title: &str,
) -> wayland_client::protocol::wl_surface::WlSurface {
    let window = f.client(id).create_window();
    let surface = window.surface.clone();
    window.set_title(title);
    window.commit();
    f.roundtrip(id);

    let window = f.client(id).window(&surface);
    window.attach_new_buffer();
    window.ack_last_and_commit();
    f.double_roundtrip(id);

    surface
}

#[test]
fn appearance_rule_applies_to_matching_window() {
    let (mut f, id) = set_up_with_static_rule();

    f.niri_state()
        .do_action_inner(
            Action::SetAppearanceOverride {
                layer: "test".to_string(),
                r#override: rule_payload("^target$", "#00ff00"),
            },
            false,
        )
        .expect("SetAppearanceOverride must succeed");

    let _surface = map_window_with_title(&mut f, id, "target");

    let mapped = f.niri().layout.windows().next().unwrap().1;
    let rules = mapped.rules();
    assert_eq!(
        rules.focus_ring.inactive_color,
        Some(Color::from_str("#111111").unwrap()),
        "the static window-rule hit must still apply",
    );
    assert_eq!(
        rules.focus_ring.active_color,
        Some(Color::from_str("#00ff00").unwrap()),
        "the appearance rule must apply on top of the static rule",
    );
}

#[test]
fn appearance_rule_skips_non_matching_window() {
    let (mut f, id) = set_up_with_static_rule();

    f.niri_state()
        .do_action_inner(
            Action::SetAppearanceOverride {
                layer: "test".to_string(),
                r#override: rule_payload("^target$", "#00ff00"),
            },
            false,
        )
        .expect("SetAppearanceOverride must succeed");

    let _surface = map_window_with_title(&mut f, id, "unrelated");

    let mapped = f.niri().layout.windows().next().unwrap().1;
    let rules = mapped.rules();
    assert_eq!(
        rules.focus_ring,
        jiji_config::BorderRule::default(),
        "neither the static rule nor the appearance rule matches this window's title",
    );
}

#[test]
fn appearance_rule_wins_over_static_same_field() {
    let config = jiji_config::Config::parse_mem(
        r##"
window-rule {
    match title="^target$"
    focus-ring {
        active-color "#111111"
    }
}
"##,
    )
    .unwrap();

    let mut f = Fixture::with_config(config);
    f.add_output(1, (1280, 720));
    let id = f.add_client();

    f.niri_state()
        .do_action_inner(
            Action::SetAppearanceOverride {
                layer: "test".to_string(),
                r#override: rule_payload("^target$", "#00ff00"),
            },
            false,
        )
        .expect("SetAppearanceOverride must succeed");

    let _surface = map_window_with_title(&mut f, id, "target");

    let mapped = f.niri().layout.windows().next().unwrap().1;
    assert_eq!(
        mapped.rules().focus_ring.active_color,
        Some(Color::from_str("#00ff00").unwrap()),
        "the appearance rule must win over the static rule for the same field",
    );
}

#[test]
fn appearance_rule_or_across_entries() {
    let (mut f, id) = set_up_with_static_rule();

    f.niri_state()
        .do_action_inner(
            Action::SetAppearanceOverride {
                layer: "test".to_string(),
                r#override: two_entry_rule_payload("^nope$", "^target2$", "#00ff00"),
            },
            false,
        )
        .expect("SetAppearanceOverride must succeed");

    let _surface = map_window_with_title(&mut f, id, "target2");

    let mapped = f.niri().layout.windows().next().unwrap().1;
    assert_eq!(
        mapped.rules().focus_ring.active_color,
        Some(Color::from_str("#00ff00").unwrap()),
        "a window matching only the second match entry must still get the override",
    );
}

#[test]
fn appearance_rule_applies_on_title_change() {
    let (mut f, id) = set_up_with_static_rule();

    f.niri_state()
        .do_action_inner(
            Action::SetAppearanceOverride {
                layer: "test".to_string(),
                r#override: rule_payload("^target$", "#00ff00"),
            },
            false,
        )
        .expect("SetAppearanceOverride must succeed");

    let surface = map_window_with_title(&mut f, id, "not-yet-matching");

    let mapped = f.niri().layout.windows().next().unwrap().1;
    assert_eq!(mapped.rules().focus_ring.active_color, None);

    let window = f.client(id).window(&surface);
    window.set_title("target");
    window.commit();
    f.roundtrip(id);

    let mapped = f.niri().layout.windows().next().unwrap().1;
    assert_eq!(
        mapped.rules().focus_ring.active_color,
        Some(Color::from_str("#00ff00").unwrap()),
        "the appearance rule must apply once the title changes to match",
    );
}

#[test]
fn appearance_rule_survives_reload() {
    let (mut f, id) = set_up_with_static_rule();

    f.niri_state()
        .do_action_inner(
            Action::SetAppearanceOverride {
                layer: "test".to_string(),
                r#override: rule_payload("^target$", "#00ff00"),
            },
            false,
        )
        .expect("SetAppearanceOverride must succeed");

    let _surface = map_window_with_title(&mut f, id, "target");

    // Reload with a *changed* window-rule set, forcing the recompute path,
    // and confirm the appearance rule still merges on top of the fresh
    // static resolution.
    let new_config = jiji_config::Config::parse_mem(
        r##"
window-rule {
    match title="^target$"
    focus-ring {
        inactive-color "#222222"
    }
}
"##,
    )
    .unwrap();
    f.reload_config(new_config);

    let mapped = f.niri().layout.windows().next().unwrap().1;
    let rules = mapped.rules();
    assert_eq!(
        rules.focus_ring.inactive_color,
        Some(Color::from_str("#222222").unwrap()),
        "the new base static rule must apply",
    );
    assert_eq!(
        rules.focus_ring.active_color,
        Some(Color::from_str("#00ff00").unwrap()),
        "the appearance rule must still merge on top of the fresh static resolution",
    );
}

#[test]
fn clear_appearance_override_removes_rule_contribution() {
    let (mut f, id) = set_up_with_static_rule();

    f.niri_state()
        .do_action_inner(
            Action::SetAppearanceOverride {
                layer: "test".to_string(),
                r#override: rule_payload("^target$", "#00ff00"),
            },
            false,
        )
        .expect("SetAppearanceOverride must succeed");

    let _surface = map_window_with_title(&mut f, id, "target");

    f.niri_state()
        .do_action_inner(
            Action::ClearAppearanceOverride {
                layer: "test".to_string(),
            },
            false,
        )
        .expect("ClearAppearanceOverride must succeed");

    let mapped = f.niri().layout.windows().next().unwrap().1;
    let rules = mapped.rules();
    assert_eq!(
        rules.focus_ring.inactive_color,
        Some(Color::from_str("#111111").unwrap()),
        "the static rule contribution must remain",
    );
    assert_eq!(
        rules.focus_ring.active_color, None,
        "the appearance rule contribution must be gone after clear",
    );
}

#[test]
fn set_appearance_override_rejects_empty_matcher_and_does_not_mutate_state() {
    let mut f = Fixture::new();

    let mut wire = AppearanceOverride::default();
    let mut rule = jiji_ipc::AppearanceRuleOverride::default();
    rule.matches = vec![];
    wire.rules = vec![rule];

    let result = f.niri_state().do_action_inner(
        Action::SetAppearanceOverride {
            layer: "test".to_string(),
            r#override: wire,
        },
        false,
    );

    match result {
        Err(crate::layout::DoActionError::AppearanceOverrideInvalid { reason }) => {
            assert!(reason.contains("rules[0]"), "{reason}");
        }
        other => panic!("expected AppearanceOverrideInvalid, got {other:?}"),
    }

    assert!(
        f.niri().appearance_override.is_empty(),
        "a rejected payload must not mutate appearance_override state",
    );
}

#[test]
fn appearance_rule_applies_immediately_on_set() {
    let (mut f, id) = set_up_with_static_rule();

    let _surface = map_window_with_title(&mut f, id, "target");

    let mapped = f.niri().layout.windows().next().unwrap().1;
    assert_eq!(
        mapped.rules().focus_ring.active_color,
        None,
        "no override is set yet, so the window must carry no appearance contribution",
    );

    f.niri_state()
        .do_action_inner(
            Action::SetAppearanceOverride {
                layer: "test".to_string(),
                r#override: rule_payload("^target$", "#00ff00"),
            },
            false,
        )
        .expect("SetAppearanceOverride must succeed");

    let mapped = f.niri().layout.windows().next().unwrap().1;
    assert_eq!(
        mapped.rules().focus_ring.active_color,
        Some(Color::from_str("#00ff00").unwrap()),
        "the appearance rule must land immediately on Set, with no title change or reload",
    );
}

#[test]
fn appearance_rule_cross_layer_lexical_tiebreak() {
    let (mut f, id) = set_up_with_static_rule();

    // Insert "z" before "a" so a bug that resolved ties by insertion order
    // (rather than by ascending `LayerId`/lexical order) would still pass a
    // naively-ordered test.
    f.niri_state()
        .do_action_inner(
            Action::SetAppearanceOverride {
                layer: "z".to_string(),
                r#override: rule_payload("^target$", "#00ff00"),
            },
            false,
        )
        .expect("SetAppearanceOverride for layer z must succeed");
    f.niri_state()
        .do_action_inner(
            Action::SetAppearanceOverride {
                layer: "a".to_string(),
                r#override: rule_payload("^target$", "#ff00ff"),
            },
            false,
        )
        .expect("SetAppearanceOverride for layer a must succeed");

    let _surface = map_window_with_title(&mut f, id, "target");

    let mapped = f.niri().layout.windows().next().unwrap().1;
    assert_eq!(
        mapped.rules().focus_ring.active_color,
        Some(Color::from_str("#00ff00").unwrap()),
        "layer \"z\" is lexically greater than \"a\" and must win, even though \"a\" was \
         inserted last (ruling out an insertion-order-based tiebreak)",
    );
}
