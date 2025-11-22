#!/bin/bash
#
# Perception AI - Image Labeler Launcher
#
# Quick launcher for the image labeling tool.
# Double-click or run from terminal.
#

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  PERCEPTION AI - IMAGE LABELER"
echo "========================================"
echo ""

# Activate virtual environment if exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "Virtual environment activated"
fi

# Set default paths
IMAGES_DIR="${1:-data/images}"
LABELS_DIR="${2:-data/labels}"

# Create directories if they don't exist
mkdir -p "$IMAGES_DIR"
mkdir -p "$LABELS_DIR"

# Check if there are images
IMAGE_COUNT=$(find "$IMAGES_DIR" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) 2>/dev/null | wc -l)

if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo "No images found in $IMAGES_DIR"
    echo ""
    read -p "Create sample images to try the labeler? [Y/n]: " CREATE_SAMPLES
    CREATE_SAMPLES=${CREATE_SAMPLES:-Y}

    if [[ $CREATE_SAMPLES =~ ^[Yy]$ ]]; then
        echo ""
        echo "Creating 20 sample images..."
        python3 tools/image_collector.py sample -o "$IMAGES_DIR" -n 20
        echo ""
    else
        echo ""
        echo "Please add images to: $IMAGES_DIR"
        echo "Then run this launcher again."
        echo ""
        echo "You can also extract frames from video:"
        echo "  python tools/image_collector.py video -i your_video.mp4 -o $IMAGES_DIR"
        echo ""
        exit 0
    fi
fi

# Count images again
IMAGE_COUNT=$(find "$IMAGES_DIR" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) 2>/dev/null | wc -l)

echo "Images directory: $IMAGES_DIR ($IMAGE_COUNT images)"
echo "Labels directory: $LABELS_DIR"
echo ""
echo "Starting labeler..."
echo ""

# Launch labeler
python3 tools/labeler.py --images "$IMAGES_DIR" --output "$LABELS_DIR"

echo ""
echo "Labeler closed."
