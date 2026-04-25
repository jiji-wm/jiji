use knuffel::errors::DecodeError;

/// Top-level `activity "Name"` config block.
///
/// Activities are compositor-level groupings of workspaces (KDE-like). Each activity
/// is declared once at the top level. Workspaces then reference activities by name
/// via the `activity "Name"` child of the `workspace` block.
///
/// Cross-reference validation (a workspace naming an activity that doesn't exist)
/// happens at layout consumption time, not during parsing — parsing is liberal.
#[derive(knuffel::Decode, Debug, Clone, PartialEq)]
pub struct Activity {
    #[knuffel(argument)]
    pub name: ActivityName,
}

/// A case-insensitively-unique activity name.
///
/// Duplicate detection is performed during `DecodeScalar::raw_decode` via a
/// `ActivityNameSet` side table on the knuffel context, mirroring
/// `WorkspaceName` / `WorkspaceNameSet`. On duplicate, a decode error is
/// emitted and an empty-string placeholder is returned so parsing can
/// continue and surface further errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivityName(pub String);

impl<S: knuffel::traits::ErrorSpan> knuffel::DecodeScalar<S> for ActivityName {
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
    ) -> Result<ActivityName, DecodeError<S>> {
        // Distinct from `WorkspaceNameSet` (same shape, different identity): the
        // knuffel context keys on type, so two separate structs let activity and
        // workspace name sets coexist without collision.
        #[derive(Debug)]
        struct ActivityNameSet(Vec<String>);
        match &**val {
            knuffel::ast::Literal::String(ref s) => {
                let mut name_set: Vec<String> = match ctx.get::<ActivityNameSet>() {
                    Some(h) => h.0.clone(),
                    None => Vec::new(),
                };

                if name_set.iter().any(|name| name.eq_ignore_ascii_case(s)) {
                    ctx.emit_error(DecodeError::unexpected(
                        val,
                        "activity",
                        format!("duplicate activity: {s}"),
                    ));
                    // SENTINEL — not a valid name; returned only on duplicate-decode error for
                    // parse-continuation
                    return Ok(Self(String::new()));
                }

                name_set.push(s.to_string());
                ctx.set(ActivityNameSet(name_set));
                Ok(Self(s.clone().into()))
            }
            _ => {
                ctx.emit_error(DecodeError::unsupported(
                    val,
                    "activity names must be strings",
                ));
                // SENTINEL — not a valid name; returned only on unsupported-literal-type error for
                // parse-continuation
                Ok(Self(String::new()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::Config;

    #[test]
    fn activity_name_does_not_clash_with_workspace_name() {
        Config::parse_mem("activity \"Work\"\nworkspace \"Work\"")
            .expect("same name for an activity and a workspace must parse cleanly");
    }

    #[test]
    fn activity_duplicate_name_case_insensitive_errors() {
        // Same name in different case is still a duplicate.
        let err = Config::parse_mem(
            r#"
            activity "Work"
            activity "work"
            "#,
        )
        .expect_err("parsing must fail when an activity name is declared twice");

        let rendered = format!("{:?}", err);
        assert!(
            rendered.contains("duplicate activity"),
            "expected duplicate-activity error, got: {rendered}"
        );
    }
}
