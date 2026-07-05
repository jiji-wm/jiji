#!/bin/sh
# Enable this repo's versioned git hooks (idempotent).
set -e
cd "$(dirname "$0")/.."
git config core.hooksPath .githooks
echo "git hooks enabled: $(git rev-parse --show-toplevel)/.githooks"
