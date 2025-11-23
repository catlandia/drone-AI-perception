#!/usr/bin/env python3
"""
Perception AI - Image Labeling Tool (Friendly Edition)

A simple, easy-to-use GUI for labeling drone/aerial images.
Designed to be intuitive even for first-time users.

Just run:
    python tools/labeler.py

Or use the launcher:
    ./launch_labeler.sh
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

# Check dependencies before starting
def check_dependencies():
    """Check if required packages are installed."""
    missing = []

    try:
        import tkinter
    except ImportError:
        missing.append("tkinter")
        print("=" * 50)
        print("MISSING: tkinter (comes with Python)")
        print("=" * 50)
        print("\nInstall with:")
        print("  Ubuntu/Debian: sudo apt-get install python3-tk")
        print("  Fedora: sudo dnf install python3-tkinter")
        print("  macOS: brew install python-tk")
        print()

    try:
        from PIL import Image
    except ImportError:
        missing.append("Pillow")
        print("=" * 50)
        print("MISSING: Pillow (image library)")
        print("=" * 50)
        print("\nInstall with:")
        print("  pip install Pillow")
        print()

    if missing:
        print("Please install missing packages and try again.")
        sys.exit(1)

check_dependencies()

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw


# ============================================================
# CONFIGURATION
# ============================================================

# Object classes - what you'll be labeling
CLASSES = [
    {"key": "1", "name": "tree",     "color": "#228B22", "icon": "🌲"},
    {"key": "2", "name": "building", "color": "#708090", "icon": "🏢"},
    {"key": "3", "name": "person",   "color": "#FF4444", "icon": "🚶"},
    {"key": "4", "name": "vehicle",  "color": "#4169E1", "icon": "🚗"},
    {"key": "5", "name": "pole",     "color": "#8B4513", "icon": "📍"},
    {"key": "6", "name": "wire",     "color": "#FFD700", "icon": "〰️"},
    {"key": "7", "name": "bird",     "color": "#FF69B4", "icon": "🐦"},
    {"key": "8", "name": "unknown",  "color": "#9932CC", "icon": "❓"},
]


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class BoundingBox:
    """A labeled bounding box."""
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int
    class_name: str


@dataclass
class ImageAnnotation:
    """All labels for one image."""
    filename: str
    width: int
    height: int
    boxes: List[BoundingBox] = field(default_factory=list)


# ============================================================
# MAIN APPLICATION
# ============================================================

class FriendlyLabeler:
    """
    A friendly, easy-to-use image labeling tool.
    """

    def __init__(self, image_dir: str, output_dir: str):
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Directory for images with boxes drawn on them
        self.marked_dir = self.output_dir / "marked_images"
        self.marked_dir.mkdir(parents=True, exist_ok=True)

        # Directory for cropped individual objects
        self.crops_dir = self.output_dir / "cropped_objects"
        self.crops_dir.mkdir(parents=True, exist_ok=True)

        # Counter for each class type (for naming: tree_001, person_002, etc.)
        self.class_counters = self._load_class_counters()

        # Find images
        self.image_files = self._find_images()
        self.current_index = 0
        self.current_class = 0
        self.annotations: Dict[str, ImageAnnotation] = {}

        # Drawing state
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.temp_rect = None

        # Display
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self._initialized = False  # Prevent double-load on startup

        # Undo stack
        self.undo_stack: List[Tuple[str, BoundingBox]] = []

        # Load existing work
        self._load_annotations()

        # Build the GUI
        self._create_gui()

    def _find_images(self) -> List[Path]:
        """Find all images in the folder."""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        images = []
        for ext in extensions:
            images.extend(self.image_dir.glob(f'*{ext}'))
            images.extend(self.image_dir.glob(f'*{ext.upper()}'))
        return sorted(images)

    def _load_annotations(self):
        """Load any existing labels."""
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
            except Exception as e:
                print(f"Note: Could not load previous annotations: {e}")

    def _load_class_counters(self) -> Dict[str, int]:
        """Load or initialize counters for each class."""
        counters = {cls['name']: 0 for cls in CLASSES}
        counter_file = self.output_dir / "class_counters.json"

        if counter_file.exists():
            try:
                with open(counter_file, 'r') as f:
                    saved = json.load(f)
                    counters.update(saved)
            except:
                pass

        # Also scan existing cropped files to get accurate counts
        if self.crops_dir.exists():
            for cls in CLASSES:
                name = cls['name']
                pattern = f"{name}_*.jpg"
                existing = list(self.crops_dir.glob(pattern))
                if existing:
                    # Find highest number
                    max_num = 0
                    for f in existing:
                        try:
                            num = int(f.stem.split('_')[-1])
                            max_num = max(max_num, num)
                        except:
                            pass
                    counters[name] = max(counters[name], max_num)

        return counters

    def _save_class_counters(self):
        """Save class counters to file."""
        counter_file = self.output_dir / "class_counters.json"
        with open(counter_file, 'w') as f:
            json.dump(self.class_counters, f)

    def _crop_and_save_object(self, img_path: Path, box: BoundingBox) -> str:
        """
        Crop a single object from image and save it.

        Returns the filename of saved crop (e.g., "tree_001.jpg")
        """
        try:
            img = Image.open(img_path).convert('RGB')

            # Crop the region (with small padding)
            padding = 5
            x1 = max(0, box.x1 - padding)
            y1 = max(0, box.y1 - padding)
            x2 = min(img.width, box.x2 + padding)
            y2 = min(img.height, box.y2 + padding)

            cropped = img.crop((x1, y1, x2, y2))

            # Get next number for this class
            class_name = box.class_name
            self.class_counters[class_name] += 1
            num = self.class_counters[class_name]

            # Save with name like "tree_001.jpg"
            crop_filename = f"{class_name}_{num:03d}.jpg"
            crop_path = self.crops_dir / crop_filename
            cropped.save(crop_path, quality=95)

            return crop_filename

        except Exception as e:
            print(f"Error cropping object: {e}")
            return ""

    def _save_annotations(self):
        """Save all labels to file."""
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

    def _create_gui(self):
        """Build the user interface."""
        self.root = tk.Tk()
        self.root.title("🏷️ Perception AI - Image Labeler")
        self.root.geometry("1300x850")
        self.root.configure(bg='#2b2b2b')

        # Style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#2b2b2b')
        style.configure('TLabel', background='#2b2b2b', foreground='white')
        style.configure('TButton', padding=10)
        style.configure('Header.TLabel', font=('Arial', 14, 'bold'))
        style.configure('Big.TButton', font=('Arial', 12), padding=15)

        # Main container
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ========== LEFT PANEL ==========
        left = ttk.Frame(main, width=280)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left.pack_propagate(False)

        # --- Welcome / Help Section ---
        help_frame = ttk.LabelFrame(left, text=" 📖 How to Use ", padding=10)
        help_frame.pack(fill=tk.X, pady=(0, 10))

        help_text = """
1. Click & drag to draw a box
2. Press 1-8 to pick object type
3. Press SPACE for next image
4. Right-click to delete a box

Click "Crop All Objects" to save
each box as: tree_001.jpg, etc.
        """
        ttk.Label(help_frame, text=help_text.strip(),
                 font=('Arial', 10), justify=tk.LEFT).pack()

        # --- Class Selection ---
        class_frame = ttk.LabelFrame(left, text=" 🏷️ What are you labeling? ", padding=10)
        class_frame.pack(fill=tk.X, pady=(0, 10))

        self.class_var = tk.IntVar(value=0)
        self.class_buttons = []

        for i, cls in enumerate(CLASSES):
            btn_frame = ttk.Frame(class_frame)
            btn_frame.pack(fill=tk.X, pady=2)

            # Color box
            color_label = tk.Label(btn_frame, text="  ", bg=cls['color'],
                                  width=2, relief=tk.RAISED)
            color_label.pack(side=tk.LEFT, padx=(0, 5))

            # Radio button with icon
            rb = ttk.Radiobutton(
                btn_frame,
                text=f" [{cls['key']}]  {cls['icon']} {cls['name'].upper()}",
                variable=self.class_var,
                value=i,
                command=lambda idx=i: self._select_class(idx)
            )
            rb.pack(side=tk.LEFT, fill=tk.X)
            self.class_buttons.append(rb)

        # --- Progress Section ---
        progress_frame = ttk.LabelFrame(left, text=" 📊 Progress ", padding=10)
        progress_frame.pack(fill=tk.X, pady=(0, 10))

        self.progress_text = tk.StringVar(value="Image 0 of 0")
        ttk.Label(progress_frame, textvariable=self.progress_text,
                 font=('Arial', 12, 'bold')).pack()

        self.progress_bar = ttk.Progressbar(progress_frame, length=240, mode='determinate')
        self.progress_bar.pack(pady=5)

        self.boxes_text = tk.StringVar(value="0 boxes drawn")
        ttk.Label(progress_frame, textvariable=self.boxes_text).pack()

        self.total_text = tk.StringVar(value="0 total labels")
        ttk.Label(progress_frame, textvariable=self.total_text).pack()

        # --- Navigation Buttons ---
        nav_frame = ttk.LabelFrame(left, text=" 🧭 Navigation ", padding=10)
        nav_frame.pack(fill=tk.X, pady=(0, 10))

        btn_row1 = ttk.Frame(nav_frame)
        btn_row1.pack(fill=tk.X, pady=2)

        self.prev_btn = ttk.Button(btn_row1, text="⬅️ Previous (A)",
                                   command=self._prev_image, style='Big.TButton')
        self.prev_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))

        self.next_btn = ttk.Button(btn_row1, text="Next (D) ➡️",
                                   command=self._next_image, style='Big.TButton')
        self.next_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)

        ttk.Button(nav_frame, text="⏭️ NEXT + SAVE (SPACE)",
                  command=self._next_and_save, style='Big.TButton').pack(fill=tk.X, pady=(10, 0))

        # --- Actions ---
        action_frame = ttk.LabelFrame(left, text=" ⚡ Actions ", padding=10)
        action_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(action_frame, text="↩️ Undo Last Box (Ctrl+Z)",
                  command=self._undo).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="🗑️ Clear All Boxes",
                  command=self._clear_boxes).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="💾 Save Now (S)",
                  command=self._save_now).pack(fill=tk.X, pady=2)

        # --- Export ---
        export_frame = ttk.LabelFrame(left, text=" 📤 Export ", padding=10)
        export_frame.pack(fill=tk.X)

        ttk.Button(export_frame, text="✂️ Crop All Objects",
                  command=self._export_all_cropped_objects).pack(fill=tk.X, pady=2)
        ttk.Button(export_frame, text="🖼️ Save All Marked Images",
                  command=self._export_all_marked_images).pack(fill=tk.X, pady=2)
        ttk.Button(export_frame, text="📁 Export for Training (YOLO)",
                  command=self._export_yolo).pack(fill=tk.X, pady=2)

        # ========== RIGHT PANEL (Image) ==========
        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Image filename
        self.filename_var = tk.StringVar(value="No image loaded")
        filename_label = ttk.Label(right, textvariable=self.filename_var,
                                  font=('Arial', 11), anchor=tk.CENTER)
        filename_label.pack(fill=tk.X)

        # Canvas for image
        canvas_frame = ttk.Frame(right)
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.canvas = tk.Canvas(canvas_frame, bg='#1a1a1a',
                               highlightthickness=2, highlightbackground='#444')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status_var = tk.StringVar(value="Ready! Draw boxes around obstacles.")
        status_bar = ttk.Label(right, textvariable=self.status_var,
                              font=('Arial', 10), anchor=tk.W,
                              foreground='#888')
        status_bar.pack(fill=tk.X)

        # ========== BINDINGS ==========
        self.canvas.bind('<Button-1>', self._on_click)
        self.canvas.bind('<B1-Motion>', self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        self.canvas.bind('<Button-3>', self._on_right_click)
        self.canvas.bind('<Configure>', self._on_resize)

        self.root.bind('<Key>', self._on_key)
        self.root.bind('<Control-z>', lambda e: self._undo())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Load first image after window is ready
        if self.image_files:
            self._load_image()
            # Schedule display after window is fully shown
            self.root.after(100, self._initial_display)
        else:
            self._show_no_images()

        self._update_stats()

    def _initial_display(self):
        """Display image after window is ready."""
        self._initialized = True
        self._display_image()

    def _show_no_images(self):
        """Show message when no images found."""
        self.canvas.delete('all')
        self.canvas.create_text(
            400, 300,
            text="📂 No images found!\n\n"
                 "Add images to:\n"
                 f"{self.image_dir}\n\n"
                 "Or create samples:\n"
                 "python tools/image_collector.py sample",
            font=('Arial', 14),
            fill='white',
            justify=tk.CENTER
        )

    def _select_class(self, idx: int):
        """Select object class."""
        self.current_class = idx
        cls = CLASSES[idx]
        self.status_var.set(f"Selected: {cls['icon']} {cls['name'].upper()} - Draw a box!")

    def _load_image(self):
        """Load and display current image."""
        if not self.image_files:
            return

        path = self.image_files[self.current_index]
        self.current_image = Image.open(path)
        self.image_width, self.image_height = self.current_image.size

        # Initialize annotation
        filename = path.name
        if filename not in self.annotations:
            self.annotations[filename] = ImageAnnotation(
                filename=filename,
                width=self.image_width,
                height=self.image_height,
                boxes=[]
            )

        self._display_image()
        self.filename_var.set(f"📷 {path.name}  ({self.image_width} × {self.image_height})")

    def _display_image(self):
        """Render image on canvas."""
        if not hasattr(self, 'current_image'):
            return

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return

        # Scale to fit
        scale_x = cw / self.image_width
        scale_y = ch / self.image_height
        self.scale = min(scale_x, scale_y, 1.0)

        new_w = int(self.image_width * self.scale)
        new_h = int(self.image_height * self.scale)

        self.offset_x = (cw - new_w) // 2
        self.offset_y = (ch - new_h) // 2

        resized = self.current_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized)

        self.canvas.delete('all')
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.photo)

        self._draw_all_boxes()

    def _draw_all_boxes(self):
        """Draw all bounding boxes."""
        if not self.image_files:
            return

        filename = self.image_files[self.current_index].name
        if filename not in self.annotations:
            return

        for box in self.annotations[filename].boxes:
            self._draw_box(box)

    def _draw_box(self, box: BoundingBox):
        """Draw a single box with label."""
        x1 = int(box.x1 * self.scale) + self.offset_x
        y1 = int(box.y1 * self.scale) + self.offset_y
        x2 = int(box.x2 * self.scale) + self.offset_x
        y2 = int(box.y2 * self.scale) + self.offset_y

        cls = CLASSES[box.class_id]
        color = cls['color']

        # Box
        self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=3, tags='box')

        # Label background
        label_text = f" {cls['icon']} {box.class_name} "
        self.canvas.create_rectangle(x1, y1-25, x1+100, y1, fill=color, outline=color, tags='box')
        self.canvas.create_text(x1+50, y1-12, text=label_text, fill='white',
                               font=('Arial', 10, 'bold'), tags='box')

    def _canvas_to_image(self, cx: int, cy: int) -> Tuple[int, int]:
        """Convert canvas coords to image coords."""
        ix = int((cx - self.offset_x) / self.scale)
        iy = int((cy - self.offset_y) / self.scale)
        ix = max(0, min(ix, self.image_width - 1))
        iy = max(0, min(iy, self.image_height - 1))
        return ix, iy

    def _on_click(self, event):
        """Start drawing box."""
        if not self.image_files:
            return
        self.drawing = True
        self.start_x, self.start_y = self._canvas_to_image(event.x, event.y)
        self.status_var.set("Drawing... release to finish")

    def _on_drag(self, event):
        """Update box while dragging."""
        if not self.drawing:
            return

        if self.temp_rect:
            self.canvas.delete(self.temp_rect)

        end_x, end_y = self._canvas_to_image(event.x, event.y)

        x1 = int(min(self.start_x, end_x) * self.scale) + self.offset_x
        y1 = int(min(self.start_y, end_y) * self.scale) + self.offset_y
        x2 = int(max(self.start_x, end_x) * self.scale) + self.offset_x
        y2 = int(max(self.start_y, end_y) * self.scale) + self.offset_y

        color = CLASSES[self.current_class]['color']
        self.temp_rect = self.canvas.create_rectangle(
            x1, y1, x2, y2, outline=color, width=2, dash=(5, 5)
        )

    def _on_release(self, event):
        """Finish drawing box."""
        if not self.drawing:
            return
        self.drawing = False

        if self.temp_rect:
            self.canvas.delete(self.temp_rect)
            self.temp_rect = None

        end_x, end_y = self._canvas_to_image(event.x, event.y)

        x1, x2 = min(self.start_x, end_x), max(self.start_x, end_x)
        y1, y2 = min(self.start_y, end_y), max(self.start_y, end_y)

        # Too small?
        if x2 - x1 < 15 or y2 - y1 < 15:
            self.status_var.set("Box too small - try again!")
            return

        cls = CLASSES[self.current_class]
        box = BoundingBox(
            x1=x1, y1=y1, x2=x2, y2=y2,
            class_id=self.current_class,
            class_name=cls['name']
        )

        filename = self.image_files[self.current_index].name
        self.annotations[filename].boxes.append(box)
        self.undo_stack.append((filename, box))

        self._display_image()
        self._update_stats()
        self.status_var.set(f"✅ Added {cls['icon']} {cls['name']}! Draw more or press SPACE for next image.")

    def _on_right_click(self, event):
        """Delete box under cursor."""
        if not self.image_files:
            return

        ix, iy = self._canvas_to_image(event.x, event.y)
        filename = self.image_files[self.current_index].name

        if filename not in self.annotations:
            return

        boxes = self.annotations[filename].boxes
        for i, box in enumerate(boxes):
            if box.x1 <= ix <= box.x2 and box.y1 <= iy <= box.y2:
                removed = boxes.pop(i)
                self._display_image()
                self._update_stats()
                self.status_var.set(f"🗑️ Deleted {removed.class_name}")
                return

        self.status_var.set("No box under cursor")

    def _on_resize(self, event):
        """Handle window resize."""
        if not self._initialized:
            # First resize after window appears - do initial load
            if self.image_files and event.width > 100:
                self._initialized = True
                self._display_image()
        elif hasattr(self, 'current_image'):
            self._display_image()

    def _on_key(self, event):
        """Handle keyboard input."""
        key = event.keysym.lower()
        char = event.char

        # Number keys 1-8 for class selection
        if char in '12345678':
            idx = int(char) - 1
            self.class_var.set(idx)
            self._select_class(idx)

        elif key == 'a':
            self._prev_image()
        elif key == 'd':
            self._next_image()
        elif key == 'space':
            self._next_and_save()
        elif key == 's':
            self._save_now()

    def _prev_image(self):
        """Go to previous image."""
        if self.current_index > 0:
            self.current_index -= 1
            self._load_image()
            self._update_stats()

    def _next_image(self):
        """Go to next image."""
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self._load_image()
            self._update_stats()

    def _next_and_save(self):
        """Save and go to next image."""
        self._save_annotations()
        # Also save marked image if there are boxes
        if self.image_files:
            filename = self.image_files[self.current_index].name
            self._save_marked_image(filename)
        self.status_var.set("💾 Saved + marked image!")
        self._next_image()

    def _save_now(self):
        """Save immediately."""
        self._save_annotations()
        # Also save marked image
        if self.image_files:
            filename = self.image_files[self.current_index].name
            self._save_marked_image(filename)
        self.status_var.set("💾 Saved successfully!")

    def _undo(self):
        """Undo last box."""
        if not self.undo_stack:
            self.status_var.set("Nothing to undo")
            return

        filename, box = self.undo_stack.pop()
        if filename in self.annotations:
            boxes = self.annotations[filename].boxes
            if box in boxes:
                boxes.remove(box)

        self._display_image()
        self._update_stats()
        self.status_var.set(f"↩️ Undid {box.class_name}")

    def _clear_boxes(self):
        """Clear all boxes from current image."""
        if not self.image_files:
            return

        if not messagebox.askyesno("Clear All?", "Delete all boxes from this image?"):
            return

        filename = self.image_files[self.current_index].name
        if filename in self.annotations:
            self.annotations[filename].boxes = []

        self._display_image()
        self._update_stats()
        self.status_var.set("🗑️ Cleared all boxes")

    def _update_stats(self):
        """Update progress display."""
        total_images = len(self.image_files)
        labeled = sum(1 for a in self.annotations.values() if a.boxes)
        total_boxes = sum(len(a.boxes) for a in self.annotations.values())

        current_boxes = 0
        if self.image_files:
            filename = self.image_files[self.current_index].name
            if filename in self.annotations:
                current_boxes = len(self.annotations[filename].boxes)

        self.progress_text.set(f"Image {self.current_index + 1} of {total_images}")
        self.boxes_text.set(f"📦 {current_boxes} boxes on this image")
        self.total_text.set(f"📊 {total_boxes} total labels ({labeled} images done)")

        if total_images > 0:
            self.progress_bar['value'] = (labeled / total_images) * 100

        # Update nav buttons
        self.prev_btn.state(['!disabled'] if self.current_index > 0 else ['disabled'])
        self.next_btn.state(['!disabled'] if self.current_index < total_images - 1 else ['disabled'])

    def _save_marked_image(self, filename: str) -> bool:
        """
        Save a copy of the image with boxes drawn on it.

        Args:
            filename: The image filename

        Returns:
            True if saved successfully
        """
        if filename not in self.annotations:
            return False

        ann = self.annotations[filename]
        if not ann.boxes:
            return False

        # Find the original image
        img_path = self.image_dir / filename
        if not img_path.exists():
            return False

        try:
            # Open original image
            img = Image.open(img_path).convert('RGB')
            draw = ImageDraw.Draw(img)

            # Draw each box
            for box in ann.boxes:
                cls = CLASSES[box.class_id]
                color = cls['color']

                # Draw rectangle (thick border)
                for i in range(3):  # 3px thick
                    draw.rectangle(
                        [box.x1 + i, box.y1 + i, box.x2 - i, box.y2 - i],
                        outline=color
                    )

                # Draw label background
                label = f" {cls['name']} "
                # Approximate text size
                text_w = len(label) * 8
                text_h = 16
                draw.rectangle(
                    [box.x1, box.y1 - text_h - 4, box.x1 + text_w, box.y1],
                    fill=color
                )

                # Draw label text
                draw.text((box.x1 + 4, box.y1 - text_h - 2), label, fill='white')

            # Save marked image
            output_path = self.marked_dir / filename
            img.save(output_path, quality=95)
            return True

        except Exception as e:
            print(f"Error saving marked image {filename}: {e}")
            return False

    def _save_current_marked_image(self):
        """Save marked version of current image."""
        if not self.image_files:
            return

        filename = self.image_files[self.current_index].name
        if self._save_marked_image(filename):
            self.status_var.set(f"🖼️ Saved marked image to {self.marked_dir.name}/")

    def _export_all_marked_images(self):
        """Export all labeled images with boxes drawn on them."""
        count = 0
        for filename, ann in self.annotations.items():
            if ann.boxes:
                if self._save_marked_image(filename):
                    count += 1

        messagebox.showinfo(
            "Marked Images Saved! 🖼️",
            f"Saved {count} images with boxes drawn to:\n\n"
            f"📁 {self.marked_dir}\n\n"
            "These images show your labels visually!"
        )
        self.status_var.set(f"🖼️ Exported {count} marked images")

    def _export_all_cropped_objects(self):
        """
        Crop and save each labeled object as a separate image.

        Creates files like:
            tree_001.jpg, tree_002.jpg, ...
            person_001.jpg, person_002.jpg, ...
            building_001.jpg, ...
        """
        total_count = 0
        class_counts = {cls['name']: 0 for cls in CLASSES}

        for filename, ann in self.annotations.items():
            if not ann.boxes:
                continue

            img_path = self.image_dir / filename
            if not img_path.exists():
                continue

            for box in ann.boxes:
                crop_name = self._crop_and_save_object(img_path, box)
                if crop_name:
                    total_count += 1
                    class_counts[box.class_name] += 1

        # Save updated counters
        self._save_class_counters()

        # Build summary
        summary_lines = []
        for cls in CLASSES:
            name = cls['name']
            count = class_counts[name]
            if count > 0:
                total = self.class_counters[name]
                summary_lines.append(f"  {cls['icon']} {name}: {count} new (total: {total})")

        summary = "\n".join(summary_lines) if summary_lines else "  No objects to export"

        messagebox.showinfo(
            "Objects Cropped! ✂️",
            f"Saved {total_count} cropped objects to:\n\n"
            f"📁 {self.crops_dir}\n\n"
            f"Files created:\n{summary}\n\n"
            "Each object saved as: type_001.jpg, type_002.jpg, etc."
        )
        self.status_var.set(f"✂️ Exported {total_count} cropped objects")

    def _export_yolo(self):
        """Export to YOLO format for training."""
        yolo_dir = self.output_dir / "yolo"
        yolo_dir.mkdir(exist_ok=True)

        count = 0
        for filename, ann in self.annotations.items():
            if not ann.boxes:
                continue

            label_file = yolo_dir / (Path(filename).stem + ".txt")
            with open(label_file, 'w') as f:
                for box in ann.boxes:
                    x_center = ((box.x1 + box.x2) / 2) / ann.width
                    y_center = ((box.y1 + box.y2) / 2) / ann.height
                    width = (box.x2 - box.x1) / ann.width
                    height = (box.y2 - box.y1) / ann.height
                    f.write(f"{box.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            count += 1

        # Classes file
        with open(yolo_dir / "classes.txt", 'w') as f:
            for cls in CLASSES:
                f.write(f"{cls['name']}\n")

        messagebox.showinfo(
            "Export Complete! 🎉",
            f"Exported {count} labeled images to:\n{yolo_dir}\n\n"
            "Files created:\n"
            "• One .txt file per image\n"
            "• classes.txt with class names\n\n"
            "Ready for training!"
        )

    def _on_close(self):
        """Handle window close."""
        if messagebox.askyesno("Save Before Exit?", "Save your work before closing?"):
            self._save_annotations()
        self.root.destroy()

    def run(self):
        """Start the application."""
        print(f"Loaded {len(self.image_files)} images")
        print("Starting labeler...")
        self.root.mainloop()


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="🏷️ Perception AI - Image Labeling Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/labeler.py
  python tools/labeler.py --images my_photos --output my_labels

Controls:
  Left Click + Drag  = Draw bounding box
  Right Click        = Delete box
  1-8                = Select object class
  SPACE              = Save & next image
  A / D              = Previous / Next
  Ctrl+Z             = Undo
        """
    )
    parser.add_argument('--images', '-i', default='data/images',
                       help='Folder with images (default: data/images)')
    parser.add_argument('--output', '-o', default='data/labels',
                       help='Folder for labels (default: data/labels)')

    args = parser.parse_args()

    # Create directories
    Path(args.images).mkdir(parents=True, exist_ok=True)
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Start app
    app = FriendlyLabeler(args.images, args.output)
    app.run()


if __name__ == '__main__':
    main()
