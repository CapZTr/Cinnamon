#!/usr/bin/env bash
# Builds the PuD tutorial image.
#
#   tutorial/build-image.sh [tag]      default tag: cinnamon-pud-tutorial:esweek
#
# gem5 comes from MIMDRAM, which sits next to this repository (or at $MIMDRAM).
# Its committed HEAD is exported and passed as the `mimdram` build context: a
# checkout carries gigabytes of local gem5 build, and a commit is what the
# image can say it was built from. Uncommitted MIMDRAM changes are not in it.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MIMDRAM=${MIMDRAM:-$ROOT/../MIMDRAM}
TAG=${1:-cinnamon-pud-tutorial:esweek}

rev=$(git -C "$MIMDRAM" rev-parse --short HEAD)
if [[ -n $(git -C "$MIMDRAM" status --porcelain --untracked-files=no) ]]; then
  echo "note: MIMDRAM has uncommitted changes; building from $rev without them" >&2
fi
ctx=$(mktemp -d)
trap 'rm -rf "$ctx"' EXIT
git -C "$MIMDRAM" archive HEAD | tar -x -C "$ctx"

echo "building $TAG with MIMDRAM $rev"
docker build --platform linux/amd64 \
  --build-arg JOBS="${JOBS:-$(nproc)}" \
  --build-context mimdram="$ctx" \
  --label org.opencontainers.image.source=https://github.com/CapZTr/Cinnamon \
  --label pud.mimdram.revision="$rev" \
  --label pud.cinnamon.revision="$(git -C "$ROOT" rev-parse --short HEAD)" \
  -t "$TAG" "$ROOT"
