#!/bin/bash
# install.sh - Installation script for Unix/Linux systems

echo "🎯 Face Recognition Attendance System - Installation Script"
echo "=========================================================="

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1)
if [[ $? -eq 0 ]]; then
    echo "✅ Python found: $python_version"
else
    echo "❌ Python 3 not found. Please install Python 3.8+ first."
    exit 1
fi

# Check pip
echo "📋 Checking pip..."
if command -v pip3 &> /dev/null; then
    echo "✅ pip3 found"
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    echo "✅ pip found"
    PIP_CMD="pip"
else
    echo "❌ pip not found. Please install pip first."
    exit 1
fi

# Update pip
echo "🔄 Updating pip..."
$PIP_CMD install --upgrade pip

# Install system dependencies
echo "📦 Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    # Ubuntu/Debian
    sudo apt-get update
    sudo apt-get install -y cmake libopencv-dev python3-dev build-essential
elif command -v yum &> /dev/null; then
    # RHEL/CentOS
    sudo yum install -y cmake opencv-devel python3-devel gcc-c++
elif command -v brew &> /dev/null; then
    # macOS
    brew install cmake opencv
else
    echo "⚠️ Unknown package manager. Please install cmake and opencv manually."
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
$PIP_CMD install -r requirements.txt

# Download facial landmark predictor if not exists
if [ ! -f "shape_predictor_68_face_landmarks.dat" ]; then
    echo "📥 Downloading facial landmark predictor..."
    wget -q http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    if [ $? -eq 0 ]; then
        echo "📦 Extracting predictor..."
        bunzip2 shape_predictor_68_face_landmarks.dat.bz2
        echo "✅ Facial landmark predictor downloaded and extracted"
    else
        echo "❌ Failed to download predictor. Please download manually from:"
        echo "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    fi
fi

# Create directories
echo "📁 Creating directories..."
mkdir -p known_faces
mkdir -p Attendance_Reports

# Test installation
echo "🧪 Testing installation..."
python3 launcher.py --check-deps

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Installation completed successfully!"
    echo ""
    echo "To run the system:"
    echo "  Desktop GUI:     python3 launcher.py --mode tkinter"
    echo "  Web Interface:   python3 launcher.py --mode gradio"
    echo "  Console:         python3 launcher.py --mode console"
    echo "  Auto-detect:     python3 launcher.py"
else
    echo "❌ Installation test failed. Please check the errors above."
    exit 1
fi