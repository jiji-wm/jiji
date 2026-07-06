#!/bin/sh
# Run the full jiji-session test suite: the hermetic cases under every
# installed shell flavor, then the container matrix (needs podman or
# docker; pass distro names to restrict it, e.g. `run-all.sh debian`).
set -eu

cd -- "$(dirname -- "$0")"

for shell in sh dash bash "busybox sh"; do
    if command -v "${shell%% *}" >/dev/null 2>&1; then
        sh jiji-session-env-test.sh "$shell"
    else
        echo "skip: $shell is not installed"
    fi
done

sh containers/run.sh "$@"
