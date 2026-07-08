//! Pins `Action::SetAppearanceOverride` / `Action::ClearAppearanceOverride`
//! dispatch through `do_action_inner`: the override must show up in the
//! composed [`crate::layout::Options`] immediately, survive `reload_config`,
//! and disappear on clear. An invalid payload must be rejected with a
//! terminal error and must not mutate state.

use std::str::FromStr as _;

use jiji_config::{Action, Color, Config};
use jiji_ipc::AppearanceOverride;

use super::fixture::Fixture;
use crate::layout::DoActionOutcome;

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
