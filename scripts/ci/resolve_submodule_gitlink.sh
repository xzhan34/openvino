#!/usr/bin/env bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Resolve a private submodule's gitlink SHA and URL for openvino.mx.
#
# Rules:
#   - Any PR (fork or same-repo): read SHA + URL from the PR head commit
#   - No PR: read from the current HEAD
#   - If mx_url is provided, override the URL from .gitmodules
#
# Reading gitlink SHA + .gitmodules URL from a fork PR is safe — it only
# parses tree objects (a 40-char hash and a URL string) without executing
# any code from the fork.
#
# Usage:
#   resolve_submodule_gitlink.sh <submodule_path>
#   resolve_submodule_gitlink.sh <submodule_path> <pr_number> false
#   resolve_submodule_gitlink.sh <submodule_path> <pr_number> true
#   resolve_submodule_gitlink.sh <submodule_path> <pr_number> false <mx_url>
#
# Output (→ GITHUB_OUTPUT when set, otherwise stdout):
#   sha=<40-char commit SHA>
#   url=<submodule remote URL>

set -euo pipefail

SUBMODULE_PATH="${1:?Usage: $0 <submodule_path> [pr_number] [is_fork] [mx_url]}"
PR_NUMBER="${2:-}"
IS_FORK="${3:-false}"
MX_URL="${4:-}"

_get_base_values() {
    SHA=$(git rev-parse "HEAD:${SUBMODULE_PATH}" 2>/dev/null || echo "unknown")
    URL=$(git config -f .gitmodules --get "submodule.${SUBMODULE_PATH}.url")
}

_get_pr_values() {
    local pr_number="$1"
    local fetch_ref="refs/pull/${pr_number}/head"
    local local_ref="refs/remotes/origin/pr/${pr_number}"

    echo "Fetching PR ${pr_number} from ${fetch_ref}"
    git fetch origin "+${fetch_ref}:${local_ref}"

    # MSYS_NO_PATHCONV prevents Git-for-Windows (MSYS2) from mangling the
    # colon-separated "ref:path" syntax into Windows paths.
    SHA=$(MSYS_NO_PATHCONV=1 git rev-parse "${local_ref}:${SUBMODULE_PATH}")

    local tmpfile
    tmpfile=$(mktemp)
    MSYS_NO_PATHCONV=1 git show "${local_ref}:.gitmodules" > "$tmpfile"
    URL=$(git config -f "$tmpfile" --get "submodule.${SUBMODULE_PATH}.url")
    rm -f "$tmpfile"
}

if [[ -n "$PR_NUMBER" ]]; then
    _get_pr_values "$PR_NUMBER"
else
    _get_base_values
fi

# Override URL if mx_url is provided
if [[ -n "$MX_URL" ]]; then
    echo "Overriding URL: ${URL} → ${MX_URL}"
    URL="$MX_URL"
fi

echo "Resolved ${SUBMODULE_PATH} SHA: ${SHA}"
echo "Resolved ${SUBMODULE_PATH} URL: ${URL}"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    echo "sha=${SHA}" >> "$GITHUB_OUTPUT"
    echo "url=${URL}" >> "$GITHUB_OUTPUT"
else
    echo "sha=${SHA}"
    echo "url=${URL}"
fi
