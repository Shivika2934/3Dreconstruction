#!/bin/bash

# Gradio Dashboard Launcher for macOS/Linux
# ==========================================
#
# This script will:
# 1. Check if Python is installed
# 2. Create a virtual environment (if needed)
# 3. Install dependencies
# 4. Start the Gradio dashboard
#

set -e  # Exit on error

echo ""
echo "============================================================"
echo "  3D Reconstruction Gradio Dashboard - macOS/Linux Launcher"
echo "============================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: Python 3 is not installed${NC}"
    echo ""
    echo "Install Python with:"
    echo "  macOS: brew install python3"
    echo "  Ubuntu: sudo apt-get install python3 python3-pip"
    echo "  Fedora: sudo dnf install python3 python3-pip"
    echo ""
    exit 1
fi

echo -e "${GREEN}[OK]${NC} Python is installed"
python3 --version

# Check if we're already in a virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo -e "${GREEN}[OK]${NC} Virtual environment is active: $VIRTUAL_ENV"
else
    echo ""
    echo "[INFO] Setting up virtual environment..."
    
    if [ -d "venv" ]; then
        echo -e "${GREEN}[OK]${NC} Virtual environment already exists"
    else
        echo "[ACTION] Creating new virtual environment..."
        python3 -m venv venv
        if [ $? -ne 0 ]; then
            echo -e "${RED}ERROR: Failed to create virtual environment${NC}"
            exit 1
        fi
        echo -e "${GREEN}[OK]${NC} Virtual environment created"
    fi
    
    echo "[ACTION] Activating virtual environment..."
    source venv/bin/activate
    if [ $? -ne 0 ]; then
        echo -e "${RED}ERROR: Failed to activate virtual environment${NC}"
        exit 1
    fi
    echo -e "${GREEN}[OK]${NC} Virtual environment activated"
fi

# Check if requirements are installed
echo ""
echo "[ACTION] Checking dependencies..."

if ! pip list | grep -q gradio; then
    echo -e "${YELLOW}[WARNING]${NC} Missing dependencies detected"
    echo ""
    echo "[ACTION] Installing required packages..."
    echo "This may take several minutes..."
    echo ""
    
    pip install -r gradio_requirements.txt
    
    if [ $? -ne 0 ]; then
        echo ""
        echo -e "${RED}ERROR: Failed to install dependencies${NC}"
        echo ""
        echo "Try running manually:"
        echo "  pip install -r gradio_requirements.txt"
        echo ""
        exit 1
    fi
    
    echo ""
    echo -e "${GREEN}[OK]${NC} Dependencies installed successfully"
else
    echo -e "${GREEN}[OK]${NC} All dependencies are installed"
fi

# Check for required models
echo ""
echo "[ACTION] Checking for required models..."

if [ ! -f "yolov8n.pt" ]; then
    echo -e "${YELLOW}[WARNING]${NC} YOLO model not found"
    echo "[ACTION] Downloading YOLO model on first run..."
    echo "(This will happen when you start the dashboard)"
fi

echo ""
echo "============================================================"
echo "  Starting Gradio Dashboard"
echo "============================================================"
echo ""
echo -e "${GREEN}Once the server starts, open your browser to:${NC}"
echo ""
echo -e "${YELLOW}  http://localhost:7860${NC}"
echo ""
echo "Press ${YELLOW}CTRL+C${NC} in this window to stop the server"
echo ""
echo "============================================================"
echo ""

# Start the dashboard
python gradio_dashboard.py

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}ERROR: Dashboard failed to start${NC}"
    echo ""
    echo "Check the error messages above for details"
    echo ""
    exit 1
fi
