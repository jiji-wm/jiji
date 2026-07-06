#!/bin/sh
# Run the jiji-session environment-import test against real systemd user
# managers in containers. Complements ../jiji-session-env-test.sh (hermetic,
# shim-based): here the real systemd-environment-d-generator, systemd-path
# and import-environment machinery of each distro is exercised.
#
# Requires podman (preferred; uses --systemd=always) or docker (privileged),
# and network access to pull base images.
#
# Usage: sh run.sh [debian|fedora|arch ...]
set -eu

cd -- "$(dirname -- "$0")"

engine=${ENGINE:-}
if [ -z "$engine" ]; then
    if command -v podman >/dev/null 2>&1; then
        engine=podman
    elif command -v docker >/dev/null 2>&1; then
        engine=docker
    else
        echo "podman or docker is required" >&2
        exit 1
    fi
fi

distros=${*:-debian fedora arch}
rc=0
for distro in $distros; do
    img=jiji-session-env-test-$distro
    echo "=== $distro ($engine) ==="
    # BUILD_ARGS: extra flags for image builds. When the build fails on the
    # default network, retry once with --network=host — docker's default
    # bridge has no working DNS on some hosts (e.g. with a systemd-resolved
    # stub resolver).
    if ! $engine build -q ${BUILD_ARGS:-} -t "$img" -f "Dockerfile.$distro" . >/dev/null; then
        echo "image build failed; retrying with --network=host" >&2
        if ! $engine build -q ${BUILD_ARGS:-} --network=host -t "$img" \
                -f "Dockerfile.$distro" . >/dev/null; then
            echo "FAIL: $distro (image build)"
            rc=1
            continue
        fi
    fi
    if [ "$engine" = podman ]; then
        cid=$($engine run -d --rm --systemd=always "$img")
    else
        cid=$($engine run -d --rm --privileged --cgroupns=host \
            --tmpfs /run --tmpfs /tmp "$img")
    fi
    if ! { $engine cp ../../jiji-session "$cid:/jiji-session" >/dev/null &&
           $engine cp inner-test.sh "$cid:/inner-test.sh" >/dev/null; }; then
        echo "FAIL: $distro (copy into container)"
        $engine stop -t 1 "$cid" >/dev/null 2>&1 || true
        rc=1
        continue
    fi
    if $engine exec "$cid" sh /inner-test.sh; then
        echo "PASS: $distro"
    else
        echo "FAIL: $distro"
        rc=1
    fi
    $engine stop -t 1 "$cid" >/dev/null 2>&1 || true
done
exit $rc
