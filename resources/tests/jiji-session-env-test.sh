#!/bin/sh
# Hermetic test for the environment-import logic in resources/jiji-session.
#
# Runs the real, unmodified jiji-session script with shim systemctl /
# systemd-path / dbus-update-activation-environment binaries on PATH and a
# fake environment.d generator, then asserts on the environment and variable
# list that the script hands to `systemctl --user import-environment`.
# Requires only a POSIX shell and coreutils — no systemd, no root.
#
# Usage: sh resources/tests/jiji-session-env-test.sh [shell]
# The optional argument selects the shell interpreting jiji-session
# (e.g. dash, bash, "busybox sh") so the script can be tested under several.
#
# jiji-session is invoked with its -l flag so that its login-shell re-exec
# is skipped: the trampoline would source the developer's real shell
# profile, which is exactly what a hermetic test must not do.

TEST_SH=${1:-sh}

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SESSION_SCRIPT=$script_dir/../jiji-session

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT INT TERM

failures=0
case_name=

fail() {
    echo "FAIL[$case_name]: $*" >&2
    failures=$((failures + 1))
}

# --- shim binaries ------------------------------------------------------

shims=$tmp/shims
mkdir -p "$shims"

cat > "$shims/systemctl" <<'SHIM'
#!/bin/sh
case "$*" in
    "--user -q is-active jiji.service")
        [ -n "${JIJI_TEST_ACTIVE:-}" ] && exit 0
        exit 1 ;;
    "--user reset-failed")
        exit 0 ;;
    "--user show-environment")
        [ -n "${JIJI_TEST_MANAGER_ENV:-}" ] && [ -f "$JIJI_TEST_MANAGER_ENV" ] &&
            cat "$JIJI_TEST_MANAGER_ENV"
        exit 0 ;;
    "--user import-environment"*)
        shift 2
        printf '%s\n' "$@" > "$JIJI_TEST_RECORD/import-args"
        env > "$JIJI_TEST_RECORD/import-env"
        exit 0 ;;
    "--user --wait start jiji.service")
        exit 0 ;;
    "--user start --job-mode=replace-irreversibly jiji-shutdown.target")
        exit 0 ;;
    "--user unset-environment"*)
        exit 0 ;;
esac
echo "systemctl shim: unexpected invocation: $*" >&2
exit 1
SHIM

cat > "$shims/systemd-path" <<'SHIM'
#!/bin/sh
if [ "$1" = systemd-search-user-environment-generator ]; then
    printf '%s\n' "$JIJI_TEST_GENDIR"
    exit 0
fi
exit 1
SHIM

cat > "$shims/dbus-update-activation-environment" <<'SHIM'
#!/bin/sh
printf '%s\n' "$@" > "$JIJI_TEST_RECORD/dbus-args"
exit 0
SHIM

chmod +x "$shims/systemctl" "$shims/systemd-path" \
    "$shims/dbus-update-activation-environment"

# Fake environment.d generator. Like the real one, it expands ${PATH}
# references against the environment it inherits from the caller.
gendir=$tmp/gendir
mkdir -p "$gendir"
cat > "$gendir/30-systemd-environment-d-generator" <<'GEN'
#!/bin/sh
printf 'PATH=/envd/bin:%s\n' "$PATH"
printf 'LANG=cs_CZ.UTF-8\n'
printf 'NO_AT_BRIDGE=1\n'
GEN
chmod +x "$gendir/30-systemd-environment-d-generator"

# A generator that emits nothing (machine with no environment.d configs).
gendir_empty_output=$tmp/gendir-empty-output
mkdir -p "$gendir_empty_output"
printf '#!/bin/sh\nexit 0\n' > \
    "$gendir_empty_output/30-systemd-environment-d-generator"
chmod +x "$gendir_empty_output/30-systemd-environment-d-generator"

# A directory with no generator at all (systemd without environment.d).
gendir_missing=$tmp/gendir-missing
mkdir -p "$gendir_missing"

# --- scenario runner ----------------------------------------------------

LOGIN_PATH=$shims:/usr/bin:/bin

# run_case <name> <generator-dir> <manager-env-file-or-empty>
run_case() {
    case_name=$1
    rec=$tmp/rec-$1
    mkdir -p "$rec"

    env -i \
        PATH="$LOGIN_PATH" \
        HOME="$tmp/home" \
        USER=tester \
        LOGNAME=tester \
        XDG_RUNTIME_DIR="$tmp/run" \
        XDG_SESSION_ID=3 \
        XDG_VTNR=2 \
        XDG_SEAT=seat0 \
        XDG_SESSION_TYPE=wayland \
        XDG_SESSION_CLASS=user \
        XDG_CURRENT_DESKTOP=jiji \
        DESKTOP_SESSION=jiji \
        LANG=en_US.UTF-8 \
        XDG_DATA_DIRS=/usr/local/share:/usr/share \
        PROFILE_VAR=from-profile \
        SHLVL=1 \
        OLDPWD=/ \
        TERM=linux \
        MAIL=/var/mail/tester \
        LS_COLORS='di=01;34' \
        WAYLAND_DISPLAY=wayland-9 \
        DISPLAY=:9 \
        NOTIFY_SOCKET=/fake/notify \
        WATCHDOG_USEC=10000000 \
        CREDENTIALS_DIRECTORY=/fake/creds \
        AWKPATH=/fake/awkpath \
        JIJI_TEST_RECORD="$rec" \
        JIJI_TEST_GENDIR="$2" \
        JIJI_TEST_MANAGER_ENV="$3" \
        $TEST_SH "$SESSION_SCRIPT" -l \
        || fail "jiji-session exited with status $?"
}

assert_env() {
    grep -qxF "$1" "$rec/import-env" ||
        fail "expected environment at import time to contain '$1'"
}

assert_imported() {
    grep -qxF "$1" "$rec/import-args" ||
        fail "expected '$1' in the imported variable list"
}

assert_not_imported() {
    if grep -qxF "$1" "$rec/import-args"; then
        fail "'$1' must not be in the imported variable list"
    fi
}

# The dbus activation environment must receive exactly the same variable
# list as the systemd user manager, in every scenario.
assert_dbus_matches_import() {
    if [ -f "$rec/dbus-args" ]; then
        cmp -s "$rec/import-args" "$rec/dbus-args" ||
            fail "dbus-update-activation-environment got a different variable list"
    else
        fail "dbus-update-activation-environment was not invoked"
    fi
}

# --- case 1: generator present (environment.d merge) --------------------

run_case generator "$gendir" ""

# environment.d wins / merges over the login environment.
assert_env "PATH=/envd/bin:$LOGIN_PATH"
assert_env "LANG=cs_CZ.UTF-8"
assert_env "NO_AT_BRIDGE=1"

# Session, profile, and ordinary shell variables are imported (the skip
# list is protocol-scoped only — shell trivia like SHLVL/TERM passes
# through, matching the historic blanket-import behavior for them).
for v in PATH LANG NO_AT_BRIDGE PROFILE_VAR XDG_DATA_DIRS HOME USER \
    XDG_SESSION_ID XDG_VTNR XDG_SEAT XDG_SESSION_TYPE XDG_SESSION_CLASS \
    XDG_CURRENT_DESKTOP DESKTOP_SESSION XDG_RUNTIME_DIR SHLVL TERM PWD; do
    assert_imported "$v"
done

# systemd execution-scoped variables and parent-session display handles
# are not, and neither is AWKPATH (gawk injects it into ENVIRON even when
# unset, which would make systemctl warn about a variable that isn't
# there).
for v in WAYLAND_DISPLAY DISPLAY NOTIFY_SOCKET WATCHDOG_USEC \
    CREDENTIALS_DIRECTORY AWKPATH; do
    assert_not_imported "$v"
done

assert_dbus_matches_import

# --- case 2: no generator, manager PATH merged from show-environment ----

# The empty entry between the first two components must be dropped by the
# merge (an empty PATH entry means "current directory").
manager_env=$tmp/manager-env
cat > "$manager_env" <<EOF
PATH=/envd/bin::/managed/only:/usr/bin
LANG=de_DE.UTF-8
EOF

run_case no-generator "$gendir_missing" "$manager_env"

# Manager-only PATH entries are prepended; login ordering is preserved.
assert_env "PATH=/envd/bin:/managed/only:$LOGIN_PATH"
# Scalars keep last-writer-wins semantics in the fallback: the login
# value is imported as-is.
assert_env "LANG=en_US.UTF-8"
assert_imported PATH
assert_not_imported NOTIFY_SOCKET

# --- case 3: no generator, no manager PATH ------------------------------

run_case bare "$gendir_missing" ""

# Degrades to today's behavior for PATH...
assert_env "PATH=$LOGIN_PATH"
# ...while the import list is still filtered.
assert_imported XDG_SESSION_ID
assert_not_imported NOTIFY_SOCKET

# --- case 4: generator present but no environment.d configs -------------

run_case empty-output "$gendir_empty_output" ""

assert_env "PATH=$LOGIN_PATH"
assert_env "LANG=en_US.UTF-8"
assert_imported PROFILE_VAR
assert_not_imported WAYLAND_DISPLAY

# --- case 5: external session management (run as a user service) --------
# When the script itself is the ExecStart of a systemd user unit, it must
# exec `jiji --session` directly and never touch the manager environment.

case_name=external
rec=$tmp/rec-external
mkdir -p "$rec"
if command -v bash >/dev/null 2>&1; then
    cat > "$shims/jiji" <<'SHIM'
#!/bin/sh
printf '%s\n' "$@" > "$JIJI_TEST_RECORD/jiji-args"
exit 0
SHIM
    chmod +x "$shims/jiji"

    # A decoy manager process whose command line looks like `systemd --user`.
    bash -c 'exec -a "/usr/lib/systemd/systemd --user" sleep 30' &
    mgr_pid=$!

    # exec the script from the intermediate shell so SYSTEMD_EXEC_PID
    # matches the script's own PID, as it would under systemd.
    env -i \
        PATH="$LOGIN_PATH" \
        HOME="$tmp/home" \
        JIJI_TEST_RECORD="$rec" \
        MANAGERPID="$mgr_pid" \
        sh -c 'SYSTEMD_EXEC_PID=$$; export SYSTEMD_EXEC_PID; exec sh "$1"' \
        _ "$SESSION_SCRIPT" \
        || fail "jiji-session exited with status $?"
    kill "$mgr_pid" 2>/dev/null

    grep -qxF -- --session "$rec/jiji-args" 2>/dev/null ||
        fail "expected direct exec of 'jiji --session'"
    [ ! -f "$rec/import-args" ] ||
        fail "external-management path must not import the environment"
else
    echo "SKIP[external]: bash unavailable for the decoy manager" >&2
fi

# --- case 6: a session is already running --------------------------------

case_name=already-running
rec=$tmp/rec-already-running
mkdir -p "$rec"
if env -i \
    PATH="$LOGIN_PATH" \
    HOME="$tmp/home" \
    JIJI_TEST_RECORD="$rec" \
    JIJI_TEST_GENDIR="$gendir" \
    JIJI_TEST_MANAGER_ENV="" \
    JIJI_TEST_ACTIVE=1 \
    $TEST_SH "$SESSION_SCRIPT" -l >/dev/null; then
    fail "expected a non-zero exit when a session is already running"
fi
[ ! -f "$rec/import-args" ] ||
    fail "must not import the environment when a session is already running"

# --- result --------------------------------------------------------------

if [ "$failures" -eq 0 ]; then
    echo "OK: all jiji-session environment-import cases passed ($TEST_SH)"
else
    echo "$failures assertion(s) failed ($TEST_SH)" >&2
    exit 1
fi
