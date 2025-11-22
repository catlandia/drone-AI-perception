#!/bin/bash
#
# Perception AI - Image Labeler Launcher
#
# Just double-click or run this script!
#

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment if exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Launch the GUI launcher
python3 tools/labeler_launcher.py
