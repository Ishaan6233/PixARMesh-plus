#!/usr/bin/env bash
# Refresh graphify when the repo file structure changes.
#
# This intentionally stores its state under graphify-out/, which is ignored by
# git. It tracks repo-visible files only: tracked files, untracked non-ignored
# files, and tracked deletions.

set -euo pipefail
cd "$(dirname "$0")/.."

GRAPHIFY="${GRAPHIFY:-graphify}"
OUT_DIR="${GRAPHIFY_OUT:-graphify-out}"
STATE_FILE="${OUT_DIR}/structure.sha256"
LIST_FILE="${OUT_DIR}/structure.files"
FORCE=false
CHECK=false

for arg in "$@"; do
    case "$arg" in
        --force)
            FORCE=true
            ;;
        --check)
            CHECK=true
            ;;
        *)
            echo "usage: bash scripts/graphify_refresh.sh [--force] [--check]" >&2
            exit 2
            ;;
    esac
done

if ! command -v "$GRAPHIFY" >/dev/null 2>&1; then
    echo "[graphify] ERROR: graphify command not found. Set GRAPHIFY=/path/to/graphify." >&2
    exit 127
fi

mkdir -p "$OUT_DIR"
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

{
    git ls-files -co --exclude-standard | sed 's/^/PATH\t/'
    git ls-files -d | sed 's/^/DELETED\t/'
} | LC_ALL=C sort -u > "$tmp"

current_hash="$(sha256sum "$tmp" | awk '{print $1}')"
previous_hash=""
if [[ -f "$STATE_FILE" ]]; then
    previous_hash="$(cat "$STATE_FILE")"
fi

needs_refresh=false
if [[ "$FORCE" == "true" || ! -f graphify-out/graph.json || "$current_hash" != "$previous_hash" ]]; then
    needs_refresh=true
fi

if [[ "$CHECK" == "true" ]]; then
    if [[ "$needs_refresh" == "true" ]]; then
        echo "[graphify] refresh needed"
        exit 1
    fi
    echo "[graphify] up to date"
    exit 0
fi

if [[ "$needs_refresh" != "true" ]]; then
    echo "[graphify] up to date"
    exit 0
fi

echo "[graphify] refreshing code graph"
"$GRAPHIFY" update . --force --no-cluster
cp "$tmp" "$LIST_FILE"
printf '%s\n' "$current_hash" > "$STATE_FILE"
echo "[graphify] refreshed graphify-out/graph.json"
