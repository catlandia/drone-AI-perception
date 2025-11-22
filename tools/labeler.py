#!/usr/bin/env python3
"""
Perception AI - Image Labeling Tool

A simple GUI tool for annotating aerial/drone images with bounding boxes.
Makes it easy to create training data for the object detector.

Features:
- Draw bounding boxes with mouse
- Quick keyboard shortcuts for classes
- Auto-save progress
- Export to YOLO format (for training)
- Track labeling progress

Usage:
    python tools/labeler.py --images path/to/images --output path/to/labels

Controls:
    Left Click + Drag : Draw bounding box
    Right Click       : Delete nearest box
    1-8              : Select class (tree, building, person, etc.)
    A / D            : Previous / Next image
    S                : Save current annotations
    Space            : Next image (auto-save)
    Q                : Quit
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except ImportError:
    print("Error: tkinter not found. Install with:")
    print("  Ubuntu/Debian: sudo apt-get install python3-tk")
    print("  Fedora: sudo dnf install python3-tkinter")
    sys.exit(1)

try:
    from PIL import Image, ImageTk
except ImportError:
    print("Error: Pillow not found. Install with:")
    print("  pip install Pillow")
    sys.exit(1)


# Object classes for drone perception
CLASSES = [
    ("1", "tree", "#228B22"),       # Forest green
    ("2", "building", "#808080"),   # Gray
    ("3", "person", "#FF6347"),     # Tomato red
    ("4", "vehicle", "#4169E1"),    # Royal blue
    ("5", "pole", "#8B4513"),       # Saddle brown
    ("6", "wire", "#FFD700"),       # Gold
    ("7", "bird", "#FF69B4"),       # Hot pink
    ("8", "unknown", "#9932CC"),    # Dark orchid
]


@dataclass
class BoundingBox:
    """A bounding box annotation."""
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int
    class_name: str


@dataclass
class ImageAnnotation:
    """Annotations for a single image."""
    filename: str
    width: int
    height: int
    boxes: List[BoundingBox] = field(default_factory=list)


class LabelingTool:
    """Main labeling application."""

    def __init__(self, image_dir: str, output_dir: str):
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Find all images
        self.image_files = self._find_images()
        if not self.image_files:
            messagebox.showerror("Error", f"No images found in {image_dir}")
            sys.exit(1)

        self.current_index = 0
        self.current_class = 0  # tree by default
        self.annotations: Dict[str, ImageAnnotation] = {}

        # Drawing state
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_rect = None

        # Display scaling
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

        # Load existing annotations
        self._load_all_annotations()

        # Setup GUI
        self._setup_gui()

    def _find_images(self) -> List[Path]:
        """Find all image files in directory."""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        images = []
        for ext in extensions:
            images.extend(self.image_dir.glob(f'*{ext}'))
            images.extend(self.image_dir.glob(f'*{ext.upper()}'))
        return sorted(images)

    def _load_all_annotations(self):
        """Load existing annotations from output directory."""
        json_file = self.output_dir / "annotations.json"
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                for filename, ann_data in data.items():
                    boxes = [BoundingBox(**box) for box in ann_data.get('boxes', [])]
                    self.annotations[filename] = ImageAnnotation(
                        filename=filename,
                        width=ann_data['width'],
                        height=ann_data['height'],
                        boxes=boxes
                    )
                print(f"Loaded {len(self.annotations)} existing annotations")
            except Exception as e:
                print(f"Warning: Could not load annotations: {e}")

    def _save_all_annotations(self):
        """Save all annotations to JSON file."""
        json_file = self.output_dir / "annotations.json"
        data = {}
        for filename, ann in self.annotations.items():
            data[filename] = {
                'filename': ann.filename,
                'width': ann.width,
                'height': ann.height,
                'boxes': [asdict(box) for box in ann.boxes]
            }
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _export_yolo_format(self):
        """Export annotations in YOLO format."""
        yolo_dir = self.output_dir / "yolo"
        yolo_dir.mkdir(exist_ok=True)

        for filename, ann in self.annotations.items():
            if not ann.boxes:
                continue

            # YOLO format: class_id x_center y_center width height (normalized)
            label_file = yolo_dir / (Path(filename).stem + ".txt")
            with open(label_file, 'w') as f:
                for box in ann.boxes:
                    x_center = ((box.x1 + box.x2) / 2) / ann.width
                    y_center = ((box.y1 + box.y2) / 2) / ann.height
                    width = (box.x2 - box.x1) / ann.width
                    height = (box.y2 - box.y1) / ann.height
                    f.write(f"{box.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        # Write classes file
        with open(yolo_dir / "classes.txt", 'w') as f:
            for _, name, _ in CLASSES:
                f.write(f"{name}\n")

        messagebox.showinfo("Export Complete",
                          f"Exported {len(self.annotations)} annotations to YOLO format\n{yolo_dir}")

    def _setup_gui(self):
        """Setup the GUI."""
        self.root = tk.Tk()
        self.root.title("Perception AI - Image Labeler")
        self.root.geometry("1200x800")

        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel - controls
        left_panel = ttk.Frame(main_frame, width=200)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Class selection
        ttk.Label(left_panel, text="Object Class:", font=('Arial', 12, 'bold')).pack(pady=(10, 5))

        self.class_var = tk.IntVar(value=0)
        for i, (key, name, color) in enumerate(CLASSES):
            frame = ttk.Frame(left_panel)
            frame.pack(fill=tk.X, pady=2)

            rb = ttk.Radiobutton(frame, text=f"[{key}] {name}",
                                variable=self.class_var, value=i,
                                command=self._on_class_change)
            rb.pack(side=tk.LEFT)

            # Color indicator
            canvas = tk.Canvas(frame, width=20, height=20, bg=color,
                             highlightthickness=1)
            canvas.pack(side=tk.RIGHT, padx=5)

        # Separator
        ttk.Separator(left_panel, orient='horizontal').pack(fill=tk.X, pady=10)

        # Stats
        ttk.Label(left_panel, text="Progress:", font=('Arial', 12, 'bold')).pack(pady=(10, 5))
        self.progress_label = ttk.Label(left_panel, text="0 / 0 images")
        self.progress_label.pack()

        self.boxes_label = ttk.Label(left_panel, text="0 boxes in image")
        self.boxes_label.pack()

        self.total_label = ttk.Label(left_panel, text="0 total boxes")
        self.total_label.pack()

        # Progress bar
        self.progress_bar = ttk.Progressbar(left_panel, length=180, mode='determinate')
        self.progress_bar.pack(pady=10)

        # Separator
        ttk.Separator(left_panel, orient='horizontal').pack(fill=tk.X, pady=10)

        # Buttons
        ttk.Button(left_panel, text="< Previous (A)", command=self._prev_image).pack(fill=tk.X, pady=2)
        ttk.Button(left_panel, text="Next (D) >", command=self._next_image).pack(fill=tk.X, pady=2)
        ttk.Button(left_panel, text="Save (S)", command=self._save_current).pack(fill=tk.X, pady=2)
        ttk.Button(left_panel, text="Clear All Boxes", command=self._clear_boxes).pack(fill=tk.X, pady=2)

        ttk.Separator(left_panel, orient='horizontal').pack(fill=tk.X, pady=10)

        ttk.Button(left_panel, text="Export YOLO Format", command=self._export_yolo_format).pack(fill=tk.X, pady=2)

        # Separator
        ttk.Separator(left_panel, orient='horizontal').pack(fill=tk.X, pady=10)

        # Instructions
        ttk.Label(left_panel, text="Controls:", font=('Arial', 10, 'bold')).pack(pady=(10, 5))
        instructions = """
Left Click+Drag: Draw box
Right Click: Delete box
1-8: Select class
A/D: Prev/Next image
Space: Next (auto-save)
S: Save
Q: Quit
"""
        ttk.Label(left_panel, text=instructions, justify=tk.LEFT).pack()

        # Canvas for image
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(canvas_frame, bg='#2b2b2b', cursor='crosshair')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Filename label
        self.filename_label = ttk.Label(canvas_frame, text="", font=('Arial', 10))
        self.filename_label.pack()

        # Bind events
        self.canvas.bind('<Button-1>', self._on_mouse_down)
        self.canvas.bind('<B1-Motion>', self._on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_mouse_up)
        self.canvas.bind('<Button-3>', self._on_right_click)
        self.canvas.bind('<Configure>', self._on_resize)

        self.root.bind('<Key>', self._on_key)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Load first image
        self._load_current_image()
        self._update_stats()

    def _on_class_change(self):
        """Handle class selection change."""
        self.current_class = self.class_var.get()

    def _load_current_image(self):
        """Load and display current image."""
        if not self.image_files:
            return

        image_path = self.image_files[self.current_index]
        self.current_image = Image.open(image_path)
        self.image_width, self.image_height = self.current_image.size

        # Initialize annotation if not exists
        filename = image_path.name
        if filename not in self.annotations:
            self.annotations[filename] = ImageAnnotation(
                filename=filename,
                width=self.image_width,
                height=self.image_height,
                boxes=[]
            )

        self._display_image()
        self.filename_label.config(text=f"{image_path.name} ({self.image_width}x{self.image_height})")

    def _display_image(self):
        """Display image on canvas with scaling."""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            return

        # Calculate scale to fit
        scale_x = canvas_width / self.image_width
        scale_y = canvas_height / self.image_height
        self.scale = min(scale_x, scale_y, 1.0)  # Don't upscale

        new_width = int(self.image_width * self.scale)
        new_height = int(self.image_height * self.scale)

        # Center image
        self.offset_x = (canvas_width - new_width) // 2
        self.offset_y = (canvas_height - new_height) // 2

        # Resize and display
        resized = self.current_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized)

        self.canvas.delete('all')
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.photo)

        # Draw existing boxes
        self._draw_boxes()

    def _draw_boxes(self):
        """Draw all bounding boxes for current image."""
        if not self.image_files:
            return

        filename = self.image_files[self.current_index].name
        if filename not in self.annotations:
            return

        for box in self.annotations[filename].boxes:
            self._draw_box(box)

    def _draw_box(self, box: BoundingBox):
        """Draw a single bounding box."""
        # Convert to canvas coordinates
        x1 = int(box.x1 * self.scale) + self.offset_x
        y1 = int(box.y1 * self.scale) + self.offset_y
        x2 = int(box.x2 * self.scale) + self.offset_x
        y2 = int(box.y2 * self.scale) + self.offset_y

        color = CLASSES[box.class_id][2]

        # Draw rectangle
        self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags='box')

        # Draw label background
        self.canvas.create_rectangle(x1, y1-20, x1+80, y1, fill=color, outline=color, tags='box')

        # Draw label text
        self.canvas.create_text(x1+40, y1-10, text=box.class_name,
                               fill='white', font=('Arial', 10, 'bold'), tags='box')

    def _canvas_to_image(self, cx: int, cy: int) -> Tuple[int, int]:
        """Convert canvas coordinates to image coordinates."""
        ix = int((cx - self.offset_x) / self.scale)
        iy = int((cy - self.offset_y) / self.scale)
        # Clamp to image bounds
        ix = max(0, min(ix, self.image_width - 1))
        iy = max(0, min(iy, self.image_height - 1))
        return ix, iy

    def _on_mouse_down(self, event):
        """Start drawing box."""
        self.drawing = True
        self.start_x, self.start_y = self._canvas_to_image(event.x, event.y)

    def _on_mouse_drag(self, event):
        """Update box while dragging."""
        if not self.drawing:
            return

        # Delete previous temp rectangle
        if self.current_rect:
            self.canvas.delete(self.current_rect)

        end_x, end_y = self._canvas_to_image(event.x, event.y)

        # Draw temp rectangle
        x1 = int(min(self.start_x, end_x) * self.scale) + self.offset_x
        y1 = int(min(self.start_y, end_y) * self.scale) + self.offset_y
        x2 = int(max(self.start_x, end_x) * self.scale) + self.offset_x
        y2 = int(max(self.start_y, end_y) * self.scale) + self.offset_y

        color = CLASSES[self.current_class][2]
        self.current_rect = self.canvas.create_rectangle(
            x1, y1, x2, y2, outline=color, width=2, dash=(4, 4)
        )

    def _on_mouse_up(self, event):
        """Finish drawing box."""
        if not self.drawing:
            return

        self.drawing = False
        if self.current_rect:
            self.canvas.delete(self.current_rect)
            self.current_rect = None

        end_x, end_y = self._canvas_to_image(event.x, event.y)

        # Create bounding box (ensure x1 < x2, y1 < y2)
        x1, x2 = min(self.start_x, end_x), max(self.start_x, end_x)
        y1, y2 = min(self.start_y, end_y), max(self.start_y, end_y)

        # Minimum size check
        if x2 - x1 < 10 or y2 - y1 < 10:
            return

        box = BoundingBox(
            x1=x1, y1=y1, x2=x2, y2=y2,
            class_id=self.current_class,
            class_name=CLASSES[self.current_class][1]
        )

        # Add to annotations
        filename = self.image_files[self.current_index].name
        self.annotations[filename].boxes.append(box)

        # Redraw
        self._display_image()
        self._update_stats()

    def _on_right_click(self, event):
        """Delete nearest box."""
        ix, iy = self._canvas_to_image(event.x, event.y)

        filename = self.image_files[self.current_index].name
        if filename not in self.annotations:
            return

        # Find box containing click point
        boxes = self.annotations[filename].boxes
        for i, box in enumerate(boxes):
            if box.x1 <= ix <= box.x2 and box.y1 <= iy <= box.y2:
                boxes.pop(i)
                self._display_image()
                self._update_stats()
                return

    def _on_resize(self, event):
        """Handle canvas resize."""
        if hasattr(self, 'current_image'):
            self._display_image()

    def _on_key(self, event):
        """Handle keyboard input."""
        key = event.keysym.lower()

        # Number keys for class selection
        if event.char in '12345678':
            idx = int(event.char) - 1
            self.class_var.set(idx)
            self.current_class = idx

        # Navigation
        elif key == 'a':
            self._prev_image()
        elif key == 'd' or key == 'space':
            self._save_current()
            self._next_image()
        elif key == 's':
            self._save_current()
        elif key == 'q':
            self._on_close()

    def _prev_image(self):
        """Go to previous image."""
        if self.current_index > 0:
            self.current_index -= 1
            self._load_current_image()
            self._update_stats()

    def _next_image(self):
        """Go to next image."""
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self._load_current_image()
            self._update_stats()

    def _save_current(self):
        """Save current annotations."""
        self._save_all_annotations()
        print(f"Saved annotations for {len(self.annotations)} images")

    def _clear_boxes(self):
        """Clear all boxes from current image."""
        filename = self.image_files[self.current_index].name
        if filename in self.annotations:
            self.annotations[filename].boxes = []
            self._display_image()
            self._update_stats()

    def _update_stats(self):
        """Update progress statistics."""
        total_images = len(self.image_files)
        labeled_images = sum(1 for ann in self.annotations.values() if ann.boxes)
        total_boxes = sum(len(ann.boxes) for ann in self.annotations.values())

        current_boxes = 0
        if self.image_files:
            filename = self.image_files[self.current_index].name
            if filename in self.annotations:
                current_boxes = len(self.annotations[filename].boxes)

        self.progress_label.config(text=f"{self.current_index + 1} / {total_images} images")
        self.boxes_label.config(text=f"{current_boxes} boxes in image")
        self.total_label.config(text=f"{total_boxes} total boxes ({labeled_images} labeled)")

        progress = (labeled_images / total_images * 100) if total_images > 0 else 0
        self.progress_bar['value'] = progress

    def _on_close(self):
        """Handle window close."""
        if messagebox.askyesno("Save", "Save annotations before closing?"):
            self._save_all_annotations()
        self.root.destroy()

    def run(self):
        """Start the application."""
        print(f"Loaded {len(self.image_files)} images from {self.image_dir}")
        print("Starting labeling tool...")
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Perception AI Image Labeling Tool")
    parser.add_argument('--images', '-i', type=str, default='data/images',
                       help='Directory containing images to label')
    parser.add_argument('--output', '-o', type=str, default='data/labels',
                       help='Directory to save annotations')
    args = parser.parse_args()

    # Create sample images directory if it doesn't exist
    images_dir = Path(args.images)
    if not images_dir.exists():
        images_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created images directory: {images_dir}")
        print("Please add images to this directory and run again.")
        print("\nTip: You can download drone datasets from:")
        print("  - VisDrone: https://github.com/VisDrone/VisDrone-Dataset")
        print("  - UAV123: https://cemse.kaust.edu.sa/ivul/uav123")
        return

    app = LabelingTool(args.images, args.output)
    app.run()


if __name__ == '__main__':
    main()
