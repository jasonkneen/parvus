#!/bin/bash

echo "=== Removing non-stock Python installations ==="

# Remove Python.org installations
if [ -d "/Library/Frameworks/Python.framework" ]; then
    echo "Removing Python.org installations..."
    sudo rm -rf /Library/Frameworks/Python.framework
    sudo rm -rf /Applications/Python*
    sudo rm -f /usr/local/bin/python3 /usr/local/bin/pip3
fi

# Uninstall Homebrew-installed Python
if brew list | grep -q "python"; then
    echo "Uninstalling Homebrew Python versions..."
    brew uninstall --ignore-dependencies python python@3.11 python@3.10
fi

# Verify no conflicting Python installations remain
echo "Checking Python installations..."
if which python3 >/dev/null 2>&1; then
    echo "Warning: Python 3 still found at $(which python3)"
else
    echo "No non-stock Python installations found."
fi

echo "=== Ensuring Homebrew is ARM-native ==="

# Check if Homebrew is ARM-native
BREW_PREFIX=$(brew --prefix)
if [[ "$BREW_PREFIX" == "/opt/homebrew" ]]; then
    echo "Homebrew is ARM-native."
else
    echo "Homebrew is Intel-based. Reinstalling for ARM..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/uninstall.sh)"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

echo "=== Installing ARM Python 3.11 via Homebrew ==="

# Install Python 3.11
brew install python@3.11

# Verify Python installation
if /opt/homebrew/bin/python3.11 -c "import platform; print(platform.platform())" | grep -q "arm64"; then
    echo "Python 3.11 ARM installed successfully."
else
    echo "Error: Failed to install Python 3.11 ARM."
    exit 1
fi

echo "=== Setting up Python environment ==="

# Create virtual environment
rm -rf env
/opt/homebrew/bin/python3.11 -m venv env
source env/bin/activate

# Install required dependencies
pip install --upgrade pip
pip install numpy
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers

# Verify installations
echo "=== Installed packages ==="
pip list | grep -E "numpy|torch|sentence-transformers"

echo "=== Setup Complete ==="