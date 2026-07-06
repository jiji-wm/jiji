#!/bin/sh
# Runs inside a systemd container as root (PID 1 is systemd). Sets up a
# user with an environment.d config and stub jiji units, runs /jiji-session
# with a fabricated display-manager login environment, then asserts on the
# user manager's resulting environment block.
set -eu

# Wait for systemd to finish booting. `is-system-running` exits non-zero
# for "degraded", which is normal in a container (some units can't run) and
# fine for this test — go by the printed state, not the exit code.
i=60
state=
while :; do
    state=$(systemctl is-system-running 2>/dev/null) || :
    case "$state" in
        running | degraded) break ;;
    esac
    i=$((i - 1))
    if [ "$i" -le 0 ]; then
        echo "systemd did not come up (state: ${state:-unknown})" >&2
        exit 1
    fi
    sleep 1
done

useradd -m tester 2>/dev/null || true
uid=$(id -u tester)
home=$(getent passwd tester | cut -d: -f6)

# environment.d config — written before the user manager starts so the
# generator picks it up.
install -d "$home/.config/environment.d" "$home/.config/systemd/user"
cat > "$home/.config/environment.d/50-test.conf" <<'EOF'
PATH=/envd/bin:${PATH}
ENVD_SCALAR=from-environment-d
EOF

# Stub units so jiji-session has something to start.
cat > "$home/.config/systemd/user/jiji.service" <<'EOF'
[Service]
Type=oneshot
ExecStart=/bin/true
EOF
cat > "$home/.config/systemd/user/jiji-shutdown.target" <<'EOF'
[Unit]
Description=Stub shutdown target
EOF
chown -R tester:tester "$home/.config"

# Start the user manager directly instead of via loginctl enable-linger:
# logind is unreliable in containers, and lingering is just user@.service.
systemctl start "user@$uid.service"
i=30
until [ -S "/run/user/$uid/systemd/private" ]; do
    i=$((i - 1))
    if [ "$i" -le 0 ]; then
        echo "user manager did not start" >&2
        exit 1
    fi
    sleep 1
done

as_tester() {
    runuser -u tester -- env "XDG_RUNTIME_DIR=/run/user/$uid" "$@"
}

login_path=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Fabricated DM login environment. SHELL is left unset and -l is passed so
# the login-shell re-exec in jiji-session is skipped.
as_tester env -i \
    PATH="$login_path" \
    HOME="$home" USER=tester LOGNAME=tester \
    "XDG_RUNTIME_DIR=/run/user/$uid" \
    XDG_SESSION_ID=7 XDG_VTNR=2 XDG_SEAT=seat0 \
    XDG_SESSION_TYPE=wayland XDG_SESSION_CLASS=user \
    XDG_CURRENT_DESKTOP=jiji DESKTOP_SESSION=jiji \
    PROFILE_VAR=from-profile \
    SHLVL=1 TERM=linux NOTIFY_SOCKET=/fake/notify \
    sh /jiji-session -l

mgr_env=$(as_tester systemctl --user show-environment)

failures=0
expect() {
    echo "$mgr_env" | grep -qxF "$1" || {
        echo "FAIL: manager environment lacks '$1'" >&2
        failures=$((failures + 1))
    }
}
expect_absent() {
    if echo "$mgr_env" | grep -q "^$1="; then
        echo "FAIL: manager environment must not contain '$1'" >&2
        failures=$((failures + 1))
    fi
}
expect_name() {
    echo "$mgr_env" | grep -q "^$1=" || {
        echo "FAIL: manager environment lacks a '$1' entry" >&2
        failures=$((failures + 1))
    }
}

# environment.d PATH entries survive the import, merged with the login PATH.
expect "PATH=/envd/bin:$login_path"
expect "ENVD_SCALAR=from-environment-d"
# Login/profile variables (including ordinary shell ones) still arrive.
# SHLVL is asserted by name only: on distros where /bin/sh is bash, the
# shell increments it, so the value is not portable.
expect "PROFILE_VAR=from-profile"
expect "XDG_SESSION_ID=7"
expect_name SHLVL
expect "TERM=linux"
# systemd execution-scoped variables do not.
expect_absent NOTIFY_SOCKET

# The dbus activation environment has no query interface, so verify it
# end-to-end: activate a throwaway service that dumps its environment and
# assert on what a dbus-activated application would actually inherit.
# Skipped where the tooling or the user bus is unavailable.
dump=/tmp/dbus-activation-env
if command -v dbus-send >/dev/null 2>&1 && [ -S "/run/user/$uid/bus" ]; then
    install -d -o tester -g tester \
        "$home/.local/share" "$home/.local/share/dbus-1" \
        "$home/.local/share/dbus-1/services"
    cat > "$home/.local/share/dbus-1/services/org.jiji.EnvDump.service" <<EOF
[D-BUS Service]
Name=org.jiji.EnvDump
Exec=/bin/sh -c 'env > $dump'
EOF
    chown tester:tester \
        "$home/.local/share/dbus-1/services/org.jiji.EnvDump.service"
    # The dump service never claims the bus name, so activation itself
    # reports failure — the Exec still runs, which is all we need.
    as_tester env "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$uid/bus" \
        dbus-send --session --print-reply --reply-timeout=5000 \
        --dest=org.freedesktop.DBus /org/freedesktop/DBus \
        org.freedesktop.DBus.StartServiceByName \
        string:org.jiji.EnvDump uint32:0 >/dev/null 2>&1 || true
    i=10
    until [ -s "$dump" ]; do
        i=$((i - 1))
        [ "$i" -gt 0 ] || break
        sleep 1
    done
    if [ -s "$dump" ]; then
        grep -qxF "PATH=/envd/bin:$login_path" "$dump" || {
            echo "FAIL: dbus-activated service got wrong PATH" >&2
            failures=$((failures + 1))
        }
        grep -qxF "PROFILE_VAR=from-profile" "$dump" || {
            echo "FAIL: dbus-activated service lacks PROFILE_VAR" >&2
            failures=$((failures + 1))
        }
        grep -q "^NOTIFY_SOCKET=/fake" "$dump" && {
            echo "FAIL: dbus-activated service must not see the leaked" \
                "NOTIFY_SOCKET" >&2
            failures=$((failures + 1))
        }
        echo "inner-test: dbus activation environment verified"
    else
        echo "inner-test: WARNING: dbus activation produced no env dump," \
            "skipping dbus assertions" >&2
    fi
else
    echo "inner-test: WARNING: no dbus-send or user bus," \
        "skipping dbus assertions" >&2
fi

# --- Full chain: login-shell trampoline + profile.d + environment.d -----
# Without -l and with a valid SHELL, jiji-session re-execs through a login
# shell, so /etc/profile and /etc/profile.d contribute variables (the
# profile.d channel), and the environment.d generator then merges the user
# configuration on top. This is the combination a real display-manager
# login produces.

cat > /etc/profile.d/99-jiji-test.sh <<'EOF'
PATH="$PATH:/profiled/bin"
PROFILE_D_VAR=from-profile-d
export PATH PROFILE_D_VAR
EOF
# Default user dotfiles (skel) can rewrite PATH; remove them so the chain
# under test is exactly /etc/profile + profile.d + environment.d.
rm -f "$home/.profile" "$home/.bashrc" "$home/.bash_profile" "$home/.bash_login"

as_tester env -i \
    PATH="$login_path" \
    HOME="$home" USER=tester LOGNAME=tester SHELL=/bin/bash \
    "XDG_RUNTIME_DIR=/run/user/$uid" \
    XDG_SESSION_ID=8 XDG_VTNR=2 XDG_SEAT=seat0 \
    XDG_SESSION_TYPE=wayland XDG_SESSION_CLASS=user \
    XDG_CURRENT_DESKTOP=jiji DESKTOP_SESSION=jiji \
    SHLVL=1 TERM=linux NOTIFY_SOCKET=/fake/notify \
    sh /jiji-session

mgr_env=$(as_tester systemctl --user show-environment)

# /etc/profile may reset PATH to a distro default before profile.d runs,
# so assert structurally rather than on the exact value: environment.d
# entries first, the profile.d entry present, noise absent.
path_line=$(echo "$mgr_env" | sed -n 's/^PATH=//p')
case "$path_line" in
    /envd/bin:*) : ;;
    *)
        echo "FAIL: environment.d PATH entry not first: '$path_line'" >&2
        failures=$((failures + 1)) ;;
esac
case ":$path_line:" in
    *:/profiled/bin:*) : ;;
    *)
        echo "FAIL: profile.d PATH entry missing: '$path_line'" >&2
        failures=$((failures + 1)) ;;
esac
expect "PROFILE_D_VAR=from-profile-d"
expect "ENVD_SCALAR=from-environment-d"
expect "XDG_SESSION_ID=8"
expect_absent NOTIFY_SOCKET

if [ "$failures" -eq 0 ]; then
    echo "inner-test: OK"
else
    echo "inner-test: $failures assertion(s) failed" >&2
    exit 1
fi
