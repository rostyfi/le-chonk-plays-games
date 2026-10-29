#!/bin/bash
# Le Chonk plays games - Quick launcher
#
# Usage:
#   ./play.sh                  # Screenshot + AI advice (terminal)
#   ./play.sh --gui            # Native desktop UI with window selector
#   ./play.sh --auto           # Screenshot + AI advice (same as default, via auto_play.py)
#   ./play.sh --auto --loop 5  # Take 5 screenshots in a row, get advice each time
#
# Set your Mistral API key first:
#   export MISTRAL_API_KEY="your-key-here"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Make the lechonk package importable when running from a source checkout
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# Check for API key (skip for --help)
if [[ "$1" != "--help" && "$1" != "-h" && -z "$MISTRAL_API_KEY" ]]; then
    echo "Error: MISTRAL_API_KEY not set."
    echo "Get a key at https://console.mistral.ai/api-keys/ and run:"
    echo "  export MISTRAL_API_KEY='your-key-here'"
    exit 1
fi

# Route to the right entry point
if [[ "$1" == "--gui" ]]; then
    shift
    python3 -m lechonk.gui "$@"
elif [[ "$1" == "--auto" ]]; then
    shift
    python3 -m lechonk.auto_play "$@"
else
    python3 -m lechonk.advisor "$@"
fi
