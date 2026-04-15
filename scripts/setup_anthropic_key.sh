#!/usr/bin/env bash
# One-shot Anthropic API key setup for ClaimGuard-CXR.
#
# Pasts the key ONCE in a hidden prompt, validates the format,
# tests it against api.anthropic.com, and ONLY THEN writes it to
# the local file + Modal secret.  If anything fails, nothing
# changes on disk so you can rerun safely.
#
# Usage:
#   bash scripts/setup_anthropic_key.sh

set -uo pipefail

# Don't trace the key value
set +x

echo
echo "===================================================================="
echo " ClaimGuard-CXR Anthropic API key setup"
echo "===================================================================="
echo
echo "1. Open https://console.anthropic.com/settings/keys in your browser"
echo "2. Click the CLIPBOARD ICON next to your key (NOT select-and-copy"
echo "   the displayed text — the displayed text is truncated with '...')"
echo "3. Paste the key below.  It will be hidden as you paste."
echo "   Press Enter when done."
echo
echo "Note: Real keys start with 'sk-ant-api03-' and are exactly 108"
echo "characters long.  If yours is shorter, you copied a truncated"
echo "display string, not the real key."
echo

# Hidden prompt (-s suppresses echo)
read -r -s -p "Paste key here: " NEW_KEY
echo
echo

# Strip any accidental whitespace
NEW_KEY="$(echo -n "$NEW_KEY" | tr -d '[:space:]')"

if [[ -z "$NEW_KEY" ]]; then
    echo "ERROR: empty input.  Aborting." >&2
    exit 1
fi

KEY_LEN=${#NEW_KEY}
KEY_PREFIX="${NEW_KEY:0:14}"
KEY_TAIL="${NEW_KEY: -5}"

echo "Key length: $KEY_LEN"
echo "Prefix:     $KEY_PREFIX"
echo "Last 5:     $KEY_TAIL  (should be alphanumeric, NOT ...)"
echo

# Format validation
if [[ "$KEY_PREFIX" != "sk-ant-api03-"* ]] && [[ "$KEY_PREFIX" != "sk-ant-api"* ]]; then
    echo "ERROR: key does not start with 'sk-ant-api'." >&2
    echo "Got: $KEY_PREFIX" >&2
    echo "If you got 'sk-ant-oat...' that is an OAuth token, NOT an API key." >&2
    echo "If you got 'sk-ant-admin...' that is an admin token, NOT a regular key." >&2
    echo "Aborting; nothing changed on disk." >&2
    exit 2
fi

if [[ "$KEY_TAIL" == *"..."* ]]; then
    echo "ERROR: key ends with literal '...' — this is a truncated display" >&2
    echo "string from the Anthropic console, NOT a real key." >&2
    echo "Use the CLIPBOARD button next to the key, not select-and-copy." >&2
    echo "Aborting; nothing changed on disk." >&2
    exit 3
fi

if [[ "$NEW_KEY" =~ [^a-zA-Z0-9_-] ]]; then
    echo "ERROR: key contains characters that aren't [a-zA-Z0-9_-]." >&2
    echo "This usually means whitespace or punctuation got in.  Try again." >&2
    echo "Aborting; nothing changed on disk." >&2
    exit 4
fi

if (( KEY_LEN < 100 )); then
    echo "WARN: key length $KEY_LEN is shorter than expected (~108 chars)." >&2
    echo "It may still be valid, but if it fails the live test below," >&2
    echo "you probably truncated it." >&2
fi

# Live test against api.anthropic.com
echo "Testing key against api.anthropic.com (Sonnet 4.5)..."
TEST_OUTPUT=$(ANTHROPIC_API_KEY="$NEW_KEY" python3 -c "
import sys
try:
    import anthropic
except ImportError:
    print('NEED_INSTALL', file=sys.stderr)
    sys.exit(99)
try:
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model='claude-sonnet-4-5',
        max_tokens=10,
        messages=[{'role': 'user', 'content': 'Reply with only OK'}],
    )
    text = ''.join(b.text for b in resp.content if hasattr(b, 'text'))
    print(f'AUTH_OK:{text}')
except anthropic.AuthenticationError:
    print('AUTH_FAIL_401', file=sys.stderr)
    sys.exit(1)
except anthropic.NotFoundError:
    print('MODEL_NOT_FOUND', file=sys.stderr)
    sys.exit(2)
except Exception as e:
    print(f'OTHER_ERR:{type(e).__name__}:{e}', file=sys.stderr)
    sys.exit(3)
" 2>&1)
TEST_EXIT=$?

if [[ $TEST_EXIT -eq 99 ]]; then
    echo "  installing anthropic SDK first..."
    pip install --quiet anthropic >/dev/null 2>&1
    # Retry
    TEST_OUTPUT=$(ANTHROPIC_API_KEY="$NEW_KEY" python3 -c "
import anthropic
try:
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model='claude-sonnet-4-5', max_tokens=10,
        messages=[{'role':'user','content':'OK'}])
    print('AUTH_OK')
except anthropic.AuthenticationError:
    import sys; sys.exit(1)
" 2>&1)
    TEST_EXIT=$?
fi

if [[ $TEST_EXIT -ne 0 ]]; then
    echo "ERROR: live test FAILED.  Output: $TEST_OUTPUT" >&2
    echo "" >&2
    echo "The key was NOT written to disk and the Modal secret was NOT" >&2
    echo "updated.  Try again with a fresh key from the console." >&2
    exit 5
fi

echo "  $TEST_OUTPUT"
echo "  Key authenticates against api.anthropic.com — OK"
echo

# Write to local file
LOCAL_FILE="$HOME/.config/claimguard/anthropic_key"
mkdir -p "$(dirname "$LOCAL_FILE")"
printf '%s\n' "$NEW_KEY" > "$LOCAL_FILE"
chmod 600 "$LOCAL_FILE"
echo "Wrote local file: $LOCAL_FILE  (mode 600, $(stat -f%z "$LOCAL_FILE") bytes)"

# Update Modal secret (delete + recreate)
echo "Updating Modal secret 'anthropic'..."
modal secret delete anthropic >/dev/null 2>&1 || true
if modal secret create anthropic "ANTHROPIC_API_KEY=$NEW_KEY" >/dev/null 2>&1; then
    echo "  Modal secret 'anthropic' updated"
else
    echo "ERROR: 'modal secret create anthropic' failed" >&2
    echo "Local file was updated successfully — you can retry the modal" >&2
    echo "secret manually with: modal secret create anthropic ANTHROPIC_API_KEY=...." >&2
    exit 6
fi

# Final verification
echo
echo "===================================================================="
echo " Setup COMPLETE"
echo "===================================================================="
echo
echo "Both wires are good:"
echo "  - Local file:    $LOCAL_FILE"
echo "  - Modal secret:  anthropic"
echo
echo "Now go back to Claude Code and say:"
echo
echo "    key fixed, fire 1 and 3"
echo
