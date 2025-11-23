#!/usr/bin/env python3
"""
Perception AI - Image Labeler Launcher

A friendly GUI launcher that makes it easy to start labeling images.
No command line needed - just double-click and go!
"""

import sys
from pathlib import Path

# Check dependencies
def check_deps():
    missing = []
    try:
        import tkinter
    except ImportError:
        missing.append(("tkinter", "sudo apt-get install python3-tk"))
    try:
        from PIL import Image
    except ImportError:
        missing.append(("Pillow", "pip install Pillow"))

    if missing:
        print("=" * 50)
        print("Missing packages:")
        for name, cmd in missing:
            print(f"  {name}: {cmd}")
        print("=" * 50)
        sys.exit(1)

check_deps()

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import subprocess
import os


class LauncherApp:
    """Friendly launcher for the image labeling tool."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🏷️ Perception AI - Image Labeler")
        self.root.geometry("600x580")
        self.root.resizable(False, False)
        self.root.configure(bg='#1e3a5f')

        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() - 600) // 2
        y = (self.root.winfo_screenheight() - 580) // 2
        self.root.geometry(f"600x580+{x}+{y}")

        # Default paths
        self.script_dir = Path(__file__).parent.parent
        self.images_path = tk.StringVar(value=str(self.script_dir / "data" / "images"))
        self.output_path = tk.StringVar(value=str(self.script_dir / "data" / "labels"))

        self._create_ui()

    def _create_ui(self):
        """Build the launcher interface."""
        # Main container
        main = tk.Frame(self.root, bg='#1e3a5f')
        main.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

        # ===== HEADER =====
        header = tk.Frame(main, bg='#1e3a5f')
        header.pack(fill=tk.X, pady=(0, 20))

        tk.Label(
            header,
            text="🏷️",
            font=('Arial', 48),
            bg='#1e3a5f',
            fg='white'
        ).pack()

        tk.Label(
            header,
            text="Image Labeler",
            font=('Arial', 28, 'bold'),
            bg='#1e3a5f',
            fg='white'
        ).pack()

        tk.Label(
            header,
            text="Draw boxes around obstacles for AI training",
            font=('Arial', 12),
            bg='#1e3a5f',
            fg='#8eb8e5'
        ).pack()

        # ===== FOLDER SELECTION =====
        folders = tk.LabelFrame(
            main,
            text=" 📁 Select Folders ",
            font=('Arial', 11, 'bold'),
            bg='#2a4a6f',
            fg='white',
            padx=15,
            pady=15
        )
        folders.pack(fill=tk.X, pady=(0, 15))

        # Images folder
        img_frame = tk.Frame(folders, bg='#2a4a6f')
        img_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            img_frame,
            text="📷 Images folder (input):",
            font=('Arial', 10),
            bg='#2a4a6f',
            fg='white',
            anchor='w'
        ).pack(fill=tk.X)

        img_row = tk.Frame(img_frame, bg='#2a4a6f')
        img_row.pack(fill=tk.X, pady=(5, 0))

        img_entry = tk.Entry(
            img_row,
            textvariable=self.images_path,
            font=('Arial', 10),
            width=45
        )
        img_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Button(
            img_row,
            text="Browse...",
            command=self._browse_images,
            font=('Arial', 9),
            padx=10
        ).pack(side=tk.LEFT, padx=(10, 0))

        # Output folder
        out_frame = tk.Frame(folders, bg='#2a4a6f')
        out_frame.pack(fill=tk.X)

        tk.Label(
            out_frame,
            text="🖼️ Output folder (labels + marked images):",
            font=('Arial', 10),
            bg='#2a4a6f',
            fg='white',
            anchor='w'
        ).pack(fill=tk.X)

        out_row = tk.Frame(out_frame, bg='#2a4a6f')
        out_row.pack(fill=tk.X, pady=(5, 0))

        out_entry = tk.Entry(
            out_row,
            textvariable=self.output_path,
            font=('Arial', 10),
            width=45
        )
        out_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Button(
            out_row,
            text="Browse...",
            command=self._browse_output,
            font=('Arial', 9),
            padx=10
        ).pack(side=tk.LEFT, padx=(10, 0))

        # ===== INFO BOX =====
        info = tk.LabelFrame(
            main,
            text=" ℹ️ How it works ",
            font=('Arial', 11, 'bold'),
            bg='#2a4a6f',
            fg='white',
            padx=15,
            pady=10
        )
        info.pack(fill=tk.X, pady=(0, 15))

        info_text = """
1. Put your images in the Images folder
2. Click "Start Labeling" below
3. Draw boxes around objects (trees, buildings, people, etc.)
4. Press SPACE to save and go to next image
5. Your marked images will be saved automatically!
        """
        tk.Label(
            info,
            text=info_text.strip(),
            font=('Arial', 10),
            bg='#2a4a6f',
            fg='#c0d8f0',
            justify=tk.LEFT
        ).pack(anchor='w')

        # ===== BUTTONS =====
        buttons = tk.Frame(main, bg='#1e3a5f')
        buttons.pack(fill=tk.X, pady=(15, 10))

        # Sample images button
        tk.Button(
            buttons,
            text="🎨 Create Sample Images",
            command=self._create_samples,
            font=('Arial', 11),
            bg='#3a6a9f',
            fg='white',
            padx=20,
            pady=8,
            cursor='hand2'
        ).pack(side=tk.LEFT)

        # Start button - BIG AND GREEN
        start_btn = tk.Button(
            buttons,
            text="▶️  START LABELING",
            command=self._start_labeling,
            font=('Arial', 16, 'bold'),
            bg='#28a745',
            fg='white',
            padx=35,
            pady=12,
            cursor='hand2',
            activebackground='#34c759',
            activeforeground='white'
        )
        start_btn.pack(side=tk.RIGHT)

        # ===== FOOTER =====
        footer = tk.Frame(main, bg='#1e3a5f')
        footer.pack(fill=tk.X, side=tk.BOTTOM)

        tk.Label(
            footer,
            text="Perception AI - Part of the Drone AI System",
            font=('Arial', 9),
            bg='#1e3a5f',
            fg='#6a8ab0'
        ).pack()

    def _browse_images(self):
        """Open folder picker for images."""
        folder = filedialog.askdirectory(
            title="Select folder containing images",
            initialdir=self.images_path.get()
        )
        if folder:
            self.images_path.set(folder)

    def _browse_output(self):
        """Open folder picker for output."""
        folder = filedialog.askdirectory(
            title="Select folder for output",
            initialdir=self.output_path.get()
        )
        if folder:
            self.output_path.set(folder)

    def _create_samples(self):
        """Create sample images for testing."""
        images_dir = Path(self.images_path.get())
        images_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Try to run the image collector
            collector_path = Path(__file__).parent / "image_collector.py"
            if collector_path.exists():
                subprocess.run([
                    sys.executable,
                    str(collector_path),
                    "sample",
                    "-o", str(images_dir),
                    "-n", "20"
                ], check=True)
                messagebox.showinfo(
                    "Samples Created! 🎨",
                    f"Created 20 sample images in:\n{images_dir}\n\n"
                    "Click 'Start Labeling' to begin!"
                )
            else:
                # Fallback: create simple samples
                self._create_simple_samples(images_dir, 10)
                messagebox.showinfo(
                    "Samples Created! 🎨",
                    f"Created 10 sample images in:\n{images_dir}\n\n"
                    "Click 'Start Labeling' to begin!"
                )
        except Exception as e:
            messagebox.showerror("Error", f"Could not create samples:\n{e}")

    def _create_simple_samples(self, output_dir: Path, count: int):
        """Create simple sample images."""
        from PIL import Image, ImageDraw
        import random

        for i in range(count):
            img = Image.new('RGB', (640, 480), (135, 206, 235))
            draw = ImageDraw.Draw(img)

            # Ground
            draw.rectangle([0, 300, 640, 480], fill=(34, 139, 34))

            # Random shapes
            for _ in range(random.randint(2, 5)):
                x = random.randint(50, 550)
                y = random.randint(150, 280)
                w = random.randint(40, 100)
                h = random.randint(60, 150)
                color = random.choice([
                    (0, 100, 0),    # Tree
                    (128, 128, 128), # Building
                    (139, 69, 19),   # Pole
                ])
                draw.rectangle([x, y, x+w, y+h], fill=color)

            img.save(output_dir / f"sample_{i:03d}.jpg", quality=95)

    def _start_labeling(self):
        """Launch the labeling tool."""
        images_dir = Path(self.images_path.get())
        output_dir = Path(self.output_path.get())

        # Create directories
        images_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check for images
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        images = [f for f in images_dir.iterdir()
                 if f.suffix.lower() in extensions]

        if not images:
            result = messagebox.askyesno(
                "No Images Found",
                f"No images found in:\n{images_dir}\n\n"
                "Would you like to create sample images to try the tool?"
            )
            if result:
                self._create_samples()
                return
            else:
                return

        # Launch labeler
        try:
            labeler_path = Path(__file__).parent / "labeler.py"
            self.root.destroy()  # Close launcher

            subprocess.run([
                sys.executable,
                str(labeler_path),
                "--images", str(images_dir),
                "--output", str(output_dir)
            ])

        except Exception as e:
            messagebox.showerror("Error", f"Could not start labeler:\n{e}")

    def run(self):
        """Start the launcher."""
        self.root.mainloop()


def main():
    app = LauncherApp()
    app.run()


if __name__ == '__main__':
    main()
