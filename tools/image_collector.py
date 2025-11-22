#!/usr/bin/env python3
"""
Perception AI - Image Collection Tool

Helps collect and prepare images for labeling.

Features:
- Extract frames from video files
- Download sample images from datasets
- Resize/crop images for consistency
- Organize into train/val splits

Usage:
    python tools/image_collector.py video --input drone_flight.mp4 --output data/images
    python tools/image_collector.py resize --input raw/ --output data/images --size 640x480
    python tools/image_collector.py split --input data/images --ratio 0.8
"""

import os
import sys
import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow not found. Install with: pip install Pillow")
    sys.exit(1)


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    frame_interval: int = 30,
    max_frames: int = 1000
) -> int:
    """
    Extract frames from video file.

    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        frame_interval: Extract every Nth frame
        max_frames: Maximum frames to extract

    Returns:
        Number of frames extracted
    """
    try:
        import cv2
    except ImportError:
        print("Error: OpenCV not found. Install with: pip install opencv-python")
        return 0

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video: {total_frames} frames at {fps:.1f} FPS")
    print(f"Extracting every {frame_interval} frames (max {max_frames})...")

    frame_count = 0
    saved_count = 0

    while saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = output_path / f"frame_{saved_count:05d}.jpg"
            cv2.imwrite(str(filename), frame)
            saved_count += 1

            if saved_count % 50 == 0:
                print(f"  Saved {saved_count} frames...")

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames to {output_dir}")
    return saved_count


def resize_images(
    input_dir: str,
    output_dir: str,
    target_size: Tuple[int, int] = (640, 480),
    maintain_aspect: bool = True
) -> int:
    """
    Resize images to target size.

    Args:
        input_dir: Directory containing source images
        output_dir: Directory to save resized images
        target_size: (width, height) target dimensions
        maintain_aspect: If True, pad to maintain aspect ratio

    Returns:
        Number of images processed
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    images = [f for f in input_path.iterdir()
              if f.suffix.lower() in extensions]

    print(f"Found {len(images)} images")
    print(f"Resizing to {target_size[0]}x{target_size[1]}...")

    count = 0
    for img_path in images:
        try:
            img = Image.open(img_path)

            if maintain_aspect:
                # Calculate scaling to fit in target while maintaining aspect
                ratio = min(target_size[0] / img.width, target_size[1] / img.height)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

                # Create padded image
                padded = Image.new('RGB', target_size, (0, 0, 0))
                offset = ((target_size[0] - img.width) // 2,
                         (target_size[1] - img.height) // 2)
                padded.paste(img, offset)
                img = padded
            else:
                img = img.resize(target_size, Image.Resampling.LANCZOS)

            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')

            output_file = output_path / f"{img_path.stem}.jpg"
            img.save(output_file, 'JPEG', quality=95)
            count += 1

            if count % 50 == 0:
                print(f"  Processed {count} images...")

        except Exception as e:
            print(f"  Error processing {img_path.name}: {e}")

    print(f"Resized {count} images to {output_dir}")
    return count


def split_dataset(
    input_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[int, int]:
    """
    Split dataset into train/val sets.

    Args:
        input_dir: Directory containing images
        train_ratio: Fraction for training set
        seed: Random seed for reproducibility

    Returns:
        (train_count, val_count)
    """
    input_path = Path(input_dir)
    train_path = input_path.parent / 'train'
    val_path = input_path.parent / 'val'

    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)

    extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = [f for f in input_path.iterdir()
              if f.suffix.lower() in extensions]

    random.seed(seed)
    random.shuffle(images)

    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    print(f"Splitting {len(images)} images: {len(train_images)} train, {len(val_images)} val")

    for img in train_images:
        shutil.copy2(img, train_path / img.name)

    for img in val_images:
        shutil.copy2(img, val_path / img.name)

    print(f"Train set: {train_path}")
    print(f"Val set: {val_path}")

    return len(train_images), len(val_images)


def create_sample_images(output_dir: str, count: int = 10) -> int:
    """
    Create sample synthetic images for testing the labeler.

    Args:
        output_dir: Directory to save images
        count: Number of images to create

    Returns:
        Number of images created
    """
    import random

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Creating {count} sample synthetic images...")

    for i in range(count):
        # Create a simple scene
        img = Image.new('RGB', (640, 480), (135, 206, 235))  # Sky blue

        # Add ground
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 280, 640, 480], fill=(34, 139, 34))  # Green ground

        # Add some random "trees"
        for _ in range(random.randint(1, 5)):
            x = random.randint(50, 590)
            y = random.randint(200, 270)
            w = random.randint(30, 60)
            h = random.randint(80, 150)
            draw.rectangle([x, y, x+w, y+h], fill=(0, 100, 0))  # Dark green

        # Add random "buildings"
        for _ in range(random.randint(0, 2)):
            x = random.randint(50, 500)
            y = random.randint(180, 250)
            w = random.randint(80, 140)
            h = random.randint(60, 120)
            draw.rectangle([x, y, x+w, y+h], fill=(128, 128, 128))  # Gray

        filename = output_path / f"sample_{i:03d}.jpg"
        img.save(filename, 'JPEG', quality=95)

    print(f"Created {count} sample images in {output_dir}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Perception AI Image Collection Tool")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Video extraction
    video_parser = subparsers.add_parser('video', help='Extract frames from video')
    video_parser.add_argument('--input', '-i', required=True, help='Input video file')
    video_parser.add_argument('--output', '-o', default='data/images', help='Output directory')
    video_parser.add_argument('--interval', '-n', type=int, default=30,
                             help='Extract every Nth frame')
    video_parser.add_argument('--max', '-m', type=int, default=1000,
                             help='Maximum frames to extract')

    # Resize
    resize_parser = subparsers.add_parser('resize', help='Resize images')
    resize_parser.add_argument('--input', '-i', required=True, help='Input directory')
    resize_parser.add_argument('--output', '-o', required=True, help='Output directory')
    resize_parser.add_argument('--size', '-s', default='640x480',
                              help='Target size (WxH)')

    # Split
    split_parser = subparsers.add_parser('split', help='Split into train/val')
    split_parser.add_argument('--input', '-i', required=True, help='Input directory')
    split_parser.add_argument('--ratio', '-r', type=float, default=0.8,
                             help='Train ratio (0.8 = 80% train, 20% val)')

    # Sample
    sample_parser = subparsers.add_parser('sample', help='Create sample images')
    sample_parser.add_argument('--output', '-o', default='data/images',
                              help='Output directory')
    sample_parser.add_argument('--count', '-n', type=int, default=10,
                              help='Number of images')

    args = parser.parse_args()

    if args.command == 'video':
        extract_frames_from_video(args.input, args.output, args.interval, args.max)

    elif args.command == 'resize':
        w, h = map(int, args.size.lower().split('x'))
        resize_images(args.input, args.output, (w, h))

    elif args.command == 'split':
        split_dataset(args.input, args.ratio)

    elif args.command == 'sample':
        create_sample_images(args.output, args.count)

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python tools/image_collector.py video -i drone.mp4 -o data/images")
        print("  python tools/image_collector.py resize -i raw/ -o data/images -s 640x480")
        print("  python tools/image_collector.py split -i data/images -r 0.8")
        print("  python tools/image_collector.py sample -o data/images -n 20")


if __name__ == '__main__':
    main()
