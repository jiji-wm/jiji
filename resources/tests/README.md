# jiji-session tests

Tests for the environment handling in `resources/jiji-session`.

## Hermetic unit test

```sh
sh resources/tests/jiji-session-env-test.sh          # system sh
sh resources/tests/jiji-session-env-test.sh dash     # or bash, "busybox sh"
```

Runs the real script against shim `systemctl` / `systemd-path` /
`dbus-update-activation-environment` / `jiji` binaries and a fake
environment.d generator. No systemd, root, or network needed — suitable
for CI. Cases:

1. **generator** — environment.d generator present: its PATH entry is
   prepended to the login PATH, its scalars (LANG) win over the login
   values, environment.d-only variables are exported and imported.
2. **no-generator** — fallback merge from `systemctl --user
   show-environment`: manager-only PATH entries are prepended, login
   ordering preserved; scalars keep last-writer-wins.
3. **bare** — no generator, no manager PATH: PATH passes through
   unchanged (old behavior), import list still filtered.
4. **empty-output** — generator present but no environment.d configs
   (the common case): PATH and LANG pass through unchanged.
5. **external** — invoked as a systemd user service (`MANAGERPID` +
   `SYSTEMD_EXEC_PID` detection): execs `jiji --session` directly,
   never touches the manager environment.
6. **already-running** — `jiji.service` active: exits non-zero without
   importing.

Every import case also asserts that the skip list held (systemd
execution-scoped variables like NOTIFY_SOCKET, WATCHDOG_USEC and
CREDENTIALS_DIRECTORY, parent-session WAYLAND_DISPLAY/DISPLAY, and the
gawk-fabricated AWKPATH are not imported), that everything else —
session identity, profile-style variables, ordinary shell variables
like SHLVL/TERM — is imported, and that
`dbus-update-activation-environment` received exactly the same variable
list as `import-environment`.

jiji-session is invoked with its `-l` flag in cases 1–4 and 6 so that
its login-shell re-exec is skipped: the trampoline would source the
developer's real shell profile, which a hermetic test must not do. The
trampoline itself is covered by the container test below.

## Container integration test

```sh
sh resources/tests/containers/run.sh                 # debian fedora arch
sh resources/tests/containers/run.sh debian          # single distro
BUILD_ARGS=--network=host sh resources/tests/containers/run.sh  # DNS-less bridge
```

Boots real systemd in a container (podman preferred, docker fallback),
starts a real user manager (`user@.service`) with an environment.d
config and stub `jiji.service` / `jiji-shutdown.target` units, then runs
`jiji-session` and asserts on `systemctl --user show-environment`. This
exercises each distro's actual `systemd-environment-d-generator`,
`systemd-path`, awk, and `import-environment` behavior. Scenarios:

1. **Direct import** (`-l`, no SHELL): a fabricated display-manager
   login environment; asserts the environment.d PATH prepend, scalar
   propagation, session identity import, and exclusion of
   execution-scoped variables against the real user manager.
2. **dbus activation environment**: activates a throwaway D-Bus service
   that dumps its environment, asserting what a dbus-activated
   application actually inherits (merged PATH, profile variables, no
   leaked NOTIFY_SOCKET). Skipped with a warning where `dbus-send` or
   the user bus is unavailable.
3. **Full chain**: no `-l`, SHELL=/bin/bash — the login-shell
   trampoline re-execs through a login shell, `/etc/profile.d`
   contributes PATH entries and variables, and the environment.d
   generator merges the user config on top. Asserts structurally:
   environment.d entries first, profile.d entries present, noise
   absent.

## Not covered

- Real display-manager PAM stacks (GDM/SDDM/greetd greeter flows) and
  non-FHS distros (NixOS) cannot run in containers; they need full VMs
  (e.g. mkosi images with a DM installed, or the NixOS test framework).
  The part those environments add is the PAM/DM-specific contents of
  the login environment; what jiji-session does with it is covered
  above.
- The dinit branch of jiji-session (unchanged by the environment work).
