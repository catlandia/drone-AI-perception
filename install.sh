#!/bin/bash
#
# Perception AI - Installer Script
#
# This script installs all dependencies needed for Perception AI
# and the image labeling tools.
#
# Usage:
#   chmod +x install.sh
#   ./install.sh
#

set -e

echo "========================================"
echo "  PERCEPTION AI - INSTALLER"
echo "  Layer 2 of Drone AI System"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo -e "${RED}Error: Python not found!${NC}"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

PYTHON_VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "${GREEN}Found Python $PYTHON_VERSION${NC}"

# Check if version is adequate
MAJOR=$($PYTHON -c 'import sys; print(sys.version_info.major)')
MINOR=$($PYTHON -c 'import sys; print(sys.version_info.minor)')

if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 8 ]); then
    echo -e "${RED}Error: Python 3.8+ required, found $PYTHON_VERSION${NC}"
    exit 1
fi

# Check pip
echo ""
echo "Checking pip..."
if ! $PYTHON -m pip --version &> /dev/null; then
    echo -e "${YELLOW}Installing pip...${NC}"
    $PYTHON -m ensurepip --upgrade
fi
echo -e "${GREEN}pip is available${NC}"

# Create virtual environment (optional)
echo ""
read -p "Create virtual environment? (recommended) [Y/n]: " CREATE_VENV
CREATE_VENV=${CREATE_VENV:-Y}

if [[ $CREATE_VENV =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv venv

    # Activate
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        PYTHON=python
        echo -e "${GREEN}Virtual environment created and activated${NC}"
        echo "To activate later: source venv/bin/activate"
    else
        echo -e "${YELLOW}Could not activate venv, continuing with system Python${NC}"
    fi
fi

# Install core dependencies
echo ""
echo "Installing core dependencies..."
$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install numpy scipy

# Install Pillow (required for labeler)
echo ""
echo "Installing Pillow (image processing)..."
$PYTHON -m pip install Pillow

# Check tkinter (usually comes with Python)
echo ""
echo "Checking tkinter..."
if $PYTHON -c "import tkinter" 2>/dev/null; then
    echo -e "${GREEN}tkinter is available${NC}"
else
    echo -e "${YELLOW}tkinter not found!${NC}"
    echo "Installing tkinter..."

    # Detect OS and install
    if [ -f /etc/debian_version ]; then
        echo "Detected Debian/Ubuntu"
        sudo apt-get update
        sudo apt-get install -y python3-tk
    elif [ -f /etc/redhat-release ]; then
        echo "Detected RedHat/Fedora"
        sudo dnf install -y python3-tkinter
    elif [ -f /etc/arch-release ]; then
        echo "Detected Arch Linux"
        sudo pacman -S tk
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Detected macOS"
        echo "tkinter should come with Python. Try reinstalling Python:"
        echo "  brew install python-tk"
    else
        echo "Please install tkinter manually for your system"
    fi
fi

# Install PyTorch (optional but recommended)
echo ""
read -p "Install PyTorch? (needed for training/inference) [Y/n]: " INSTALL_TORCH
INSTALL_TORCH=${INSTALL_TORCH:-Y}

if [[ $INSTALL_TORCH =~ ^[Yy]$ ]]; then
    echo "Installing PyTorch..."

    # Check for CUDA
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}NVIDIA GPU detected, installing CUDA version${NC}"
        $PYTHON -m pip install torch torchvision
    else
        echo "No NVIDIA GPU detected, installing CPU version"
        $PYTHON -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
fi

# Install OpenCV (optional, for video extraction)
echo ""
read -p "Install OpenCV? (needed for video frame extraction) [Y/n]: " INSTALL_CV
INSTALL_CV=${INSTALL_CV:-Y}

if [[ $INSTALL_CV =~ ^[Yy]$ ]]; then
    echo "Installing OpenCV..."
    $PYTHON -m pip install opencv-python
fi

# Install the perception package
echo ""
echo "Installing Perception AI package..."
$PYTHON -m pip install -e .

# Install test dependencies
$PYTHON -m pip install pytest

# Create data directories
echo ""
echo "Creating directories..."
mkdir -p data/images
mkdir -p data/labels
mkdir -p models

# Verify installation
echo ""
echo "Verifying installation..."
echo ""

ERRORS=0

# Check imports
echo -n "  numpy: "
if $PYTHON -c "import numpy; print(numpy.__version__)" 2>/dev/null; then
    :
else
    echo -e "${RED}FAILED${NC}"
    ERRORS=$((ERRORS+1))
fi

echo -n "  scipy: "
if $PYTHON -c "import scipy; print(scipy.__version__)" 2>/dev/null; then
    :
else
    echo -e "${RED}FAILED${NC}"
    ERRORS=$((ERRORS+1))
fi

echo -n "  PIL: "
if $PYTHON -c "import PIL; print(PIL.__version__)" 2>/dev/null; then
    :
else
    echo -e "${RED}FAILED${NC}"
    ERRORS=$((ERRORS+1))
fi

echo -n "  tkinter: "
if $PYTHON -c "import tkinter; print('OK')" 2>/dev/null; then
    :
else
    echo -e "${RED}FAILED${NC}"
    ERRORS=$((ERRORS+1))
fi

echo -n "  perception: "
if $PYTHON -c "from perception import PerceptionAI; print('OK')" 2>/dev/null; then
    :
else
    echo -e "${RED}FAILED${NC}"
    ERRORS=$((ERRORS+1))
fi

# Summary
echo ""
echo "========================================"
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}INSTALLATION COMPLETE!${NC}"
    echo "========================================"
    echo ""
    echo "Quick start:"
    echo ""
    echo "  1. Create sample images:"
    echo "     python tools/image_collector.py sample -n 20"
    echo ""
    echo "  2. Launch labeling tool:"
    echo "     ./launch_labeler.sh"
    echo "     or: python tools/labeler.py"
    echo ""
    echo "  3. Run demo:"
    echo "     python examples/demo.py"
    echo ""
    if [[ $CREATE_VENV =~ ^[Yy]$ ]]; then
        echo "Remember to activate the virtual environment:"
        echo "  source venv/bin/activate"
        echo ""
    fi
else
    echo -e "${RED}INSTALLATION HAD $ERRORS ERRORS${NC}"
    echo "========================================"
    echo "Please check the errors above and try again."
fi
