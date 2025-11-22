"""
Perception AI Setup Script

Install with: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="perception-ai",
    version="0.1.0",
    description="Perception AI for Drone Obstacle Detection - Layer 2 of Drone AI System",
    author="Drone AI Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "torch": [
            "torch>=1.9.0",
            "torchvision>=0.10.0",
        ],
        "dev": [
            "pytest>=6.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
