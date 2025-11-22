# Perception AI Tools

Tools for collecting and labeling training data.

## Quick Start

```bash
# 1. Create sample images to try the labeler
python tools/image_collector.py sample -o data/images -n 20

# 2. Open the labeling tool
python tools/labeler.py -i data/images -o data/labels
```

## Image Labeler

Interactive GUI for drawing bounding boxes on images.

```bash
python tools/labeler.py --images data/images --output data/labels
```

### Controls

| Key/Action | Function |
|------------|----------|
| Left Click + Drag | Draw bounding box |
| Right Click | Delete box under cursor |
| 1-8 | Select object class |
| A | Previous image |
| D | Next image |
| Space | Next image (auto-save) |
| S | Save annotations |
| Q | Quit |

### Classes

| Key | Class | Color |
|-----|-------|-------|
| 1 | tree | Green |
| 2 | building | Gray |
| 3 | person | Red |
| 4 | vehicle | Blue |
| 5 | pole | Brown |
| 6 | wire | Gold |
| 7 | bird | Pink |
| 8 | unknown | Purple |

### Output Format

Annotations are saved in two formats:

1. **JSON** (`annotations.json`) - Full annotation data
2. **YOLO** (`yolo/`) - For training (click "Export YOLO Format")

## Image Collector

Tool for gathering and preparing images.

### Extract frames from drone video

```bash
python tools/image_collector.py video \
    --input drone_flight.mp4 \
    --output data/images \
    --interval 30 \
    --max 500
```

This extracts every 30th frame (about 1 per second at 30fps).

### Resize images

```bash
python tools/image_collector.py resize \
    --input raw_images/ \
    --output data/images \
    --size 640x480
```

### Split into train/val

```bash
python tools/image_collector.py split \
    --input data/images \
    --ratio 0.8
```

Creates `data/train/` (80%) and `data/val/` (20%).

### Create sample images

```bash
python tools/image_collector.py sample \
    --output data/images \
    --count 20
```

Creates synthetic images for testing the labeler.

## Workflow

1. **Collect images**
   - Extract from drone video: `python tools/image_collector.py video -i flight.mp4 -o data/images`
   - Or download VisDrone dataset
   - Or use your own drone images

2. **Resize if needed**
   - `python tools/image_collector.py resize -i raw/ -o data/images -s 640x480`

3. **Label images**
   - `python tools/labeler.py -i data/images -o data/labels`
   - Draw boxes around all obstacles
   - Press Space to save and move to next image

4. **Export for training**
   - Click "Export YOLO Format" in the labeler
   - Labels will be in `data/labels/yolo/`

5. **Split dataset**
   - `python tools/image_collector.py split -i data/images -r 0.8`

## Requirements

```bash
pip install Pillow
pip install opencv-python  # For video extraction
```

## Tips

- **Quality over quantity**: 500 well-labeled images beats 5000 poorly-labeled ones
- **Variety**: Include different lighting, altitudes, angles
- **Tight boxes**: Draw boxes close to object boundaries
- **Label everything**: Mark all visible obstacles, even partially visible ones
- **Consistent classes**: A tree should always be labeled as "tree"
