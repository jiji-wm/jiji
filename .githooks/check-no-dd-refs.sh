#!/usr/bin/env bash
# Reject DD references (Phase markers, §X.X, sub-phase, sub-step, "DD",
# "design.md", Box N, Appendix X, "Reviewed: <date>") in commit messages
# and code lines.
#
# Pattern set mirrors scripts/rewrite-commit-refs.py — case-sensitive on the
# distinctive caps (Phase, DD) so lowercase English ("animation phase 2",
# "let dd = ...") is not flagged.
#
# Usage:
#   check-no-dd-refs.sh --message <file>   commit-msg hook ($1 = COMMIT_EDITMSG)
#   check-no-dd-refs.sh --staged           pre-commit hook (scans staged diff)
#
# Exempt paths (DD lives here, references are legitimate):
#   docs/design.md, docs/activities/
# A commit-message check is also skipped when the staged change set touches
# only exempt paths.
#
# Optional Stage B: if `claude` is on PATH, paraphrased references are
# adjudicated via Haiku. Uses `claude -p` with subscription auth — no
# ANTHROPIC_API_KEY required. Fails open on API errors so a flaky network
# never blocks a commit.
#
# To bypass for a legitimate meta-commit: git commit --no-verify

set -euo pipefail

EXEMPT_PATHS=(
    'docs/design.md'
    'docs/activities/'
)

is_exempt_path() {
    local f="$1"
    for path in "${EXEMPT_PATHS[@]}"; do
        case "$f" in "$path"*) return 0 ;; esac
    done
    return 1
}

only_exempt_files() {
    local files
    files=$(git diff --cached --name-only)
    [[ -z "$files" ]] && return 1
    while IFS= read -r f; do
        is_exempt_path "$f" || return 1
    done <<< "$files"
    return 0
}

case "${1:-}" in
    --message)  MODE=message; ARG="${2:?missing message file}" ;;
    --staged)   MODE=staged ;;
    *)
        echo "usage: $(basename "$0") --message <file> | --staged" >&2
        exit 2
        ;;
esac

# Skip auto-generated merge / squash messages.
if [[ "$MODE" == message ]]; then
    GIT_DIR=$(git rev-parse --git-dir)
    if [[ -f "$GIT_DIR/MERGE_MSG" || -f "$GIT_DIR/SQUASH_MSG" ]]; then
        exit 0
    fi
fi

# Gather text to scan.
case "$MODE" in
    message)
        only_exempt_files && exit 0
        # Strip git's commented template lines.
        TEXT=$(grep -v '^#' "$ARG" || true)
        ;;
    staged)
        # All added lines from staged diff, excluding exempt paths.
        EXCLUDES=()
        for path in "${EXEMPT_PATHS[@]}"; do
            EXCLUDES+=(":(exclude,glob)${path}**")
            EXCLUDES+=(":(exclude)${path}")
        done
        TEXT=$(git diff --cached --diff-filter=AM -U0 -- . "${EXCLUDES[@]}" \
            | grep -E '^\+[^+]' \
            | sed 's/^+//' \
            || true)
        ;;
esac

[[ -z "${TEXT// /}" ]] && exit 0

# ---- Stage A: regex pre-pass (definitive rejections) ----
# Case-sensitive on Phase/DD; word-boundary discipline avoids identifier hits.
PATTERNS=(
    '\bPhase[ -][0-9]'                    # Phase 1, Phase 0b-2, Phase-2
    '\b[Ss]ub-?[Pp]hase\b'                # sub-phase, subphase, Sub-Phase
    '\b[Ss]ub-?[Ss]tep\b'                 # sub-step, substep
    '§[0-9]+(\.[0-9a-z]+)*'               # §3.2, §5.17a, §5.11
    '\bDD\b'                              # DD (the abbreviation)
    '\bdesign\.md\b'
    '\b[Dd]esign[- ][Dd]oc(ument)?s?\b'   # design-doc, design document
    '\bAppendix [A-Z]\b'                  # Appendix C
    '\bBox [0-9]+\b'                      # Box 3 (NB: Box<T> has no space)
    '^Reviewed:[[:space:]]+[0-9]{4}-[0-9]{2}-[0-9]{2}'  # DD scribe trailer
)

ALT=
for p in "${PATTERNS[@]}"; do
    ALT="${ALT:+$ALT|}$p"
done

HITS=$(echo "$TEXT" | grep -nE "$ALT" || true)

if [[ -n "$HITS" ]]; then
    cat >&2 <<EOF

✖ DD reference detected in $MODE:

$HITS

This repo's commits and source must not reference the design document
(phases, sub-phases, sub-steps, §X.X, Box N, Appendix X, "DD", "design.md",
"Reviewed: YYYY-MM-DD"). The DD lives elsewhere; keep it out of the code.

To bypass for a legitimate meta-commit (e.g. "scrub DD refs from history"),
use:  git commit --no-verify
EOF
    exit 1
fi

# ---- Stage B: optional Haiku adjudication for paraphrased references ----
command -v claude >/dev/null 2>&1 || exit 0

SYSTEM_PROMPT='You are a strict commit-message and code linter for a project that
forbids any reference to its internal "Design Document" (DD).

Reject text that mentions:
- numbered phases ("Phase 2.6", "Phase 1a", "Phase 0b-2")
- sub-phases or sub-steps
- Box / Appendix references in a design-document sense
- "DD", "design.md", "design doc", section markers like "§X.X"
- paraphrases such as "as outlined in the design plan", "per the design",
  "in the next sub-phase", "the architecture document says"

Allow:
- "animation phase", "transition phase" (mechanical sense)
- "step 1: ...", "step 2: ..." (algorithm description)
- "Box<T>", "Box::new(...)", "boxed" (Rust syntax)
- "by design", "intentional design choice" (English usage)
- bare commit SHAs (those are out of scope for this linter)

Reply on a SINGLE line with EXACTLY one of:
  OK
  REJECT: <one-sentence reason naming the offending phrase>'

VERDICT=$(printf '%s' "$TEXT" | timeout 20 claude -p \
    --model haiku \
    --output-format text \
    --tools '' \
    --disable-slash-commands \
    --no-session-persistence \
    --setting-sources user \
    --append-system-prompt "$SYSTEM_PROMPT" \
    2>/dev/null || echo "OK")

VERDICT=$(echo "$VERDICT" | head -1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

if [[ "$VERDICT" == REJECT:* ]]; then
    cat >&2 <<EOF

✖ ${VERDICT}
  (Haiku-detected paraphrased DD reference in $MODE)

To bypass for a legitimate meta-commit, use:  git commit --no-verify
EOF
    exit 1
fi

exit 0
