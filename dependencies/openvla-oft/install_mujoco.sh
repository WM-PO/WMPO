#!/bin/bash
set -e

# 1. Define installation paths
MUJOCO_DIR="$HOME/.mujoco"
MUJOCO_VERSION="2.1.0"
MUJOCO_TARBALL="mujoco${MUJOCO_VERSION}-linux-x86_64.tar.gz"
MUJOCO_URL="https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz"

echo "=== MuJoCo installation script ==="

# 2. Create installation directory
mkdir -p "$MUJOCO_DIR"

# 3. Download MuJoCo 2.1.0
echo "Downloading MuJoCo ${MUJOCO_VERSION}..."
wget -O "$MUJOCO_TARBALL" "$MUJOCO_URL"

# 4. Extract to ~/.mujoco/mujoco210
echo "Extracting to $MUJOCO_DIR/mujoco210 ..."
tar -xzf "$MUJOCO_TARBALL" -C "$MUJOCO_DIR"
mv "$MUJOCO_DIR/mujoco210" "$MUJOCO_DIR/mujoco210" 2>/dev/null || true

# 5. Remove tarball
rm "$MUJOCO_TARBALL"

# 6. Install system dependencies (Ubuntu/Debian)
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    libosmesa6-dev \
    libgl1-mesa-dev \
    libglfw3 \
    patchelf

# 7. Configure environment variables (append to ~/.bashrc)
echo "Configuring environment variables..."
if ! grep -q "MUJOCO_PY_MUJOCO_PATH" "$HOME/.bashrc"; then
    echo "export MUJOCO_PY_MUJOCO_PATH=$MUJOCO_DIR/mujoco210" >> "$HOME/.bashrc"
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$MUJOCO_DIR/mujoco210/bin" >> "$HOME/.bashrc"
fi

# 8. Apply environment variables immediately
export MUJOCO_PY_MUJOCO_PATH=$MUJOCO_DIR/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MUJOCO_DIR/mujoco210/bin

# 9. Install mujoco-py
echo "Installing mujoco-py ..."
pip install -U pip setuptools wheel
pip install mujoco-py==2.1.2.14

echo "✅ MuJoCo installation completed!"

# export MUJOCO_GL=osmesa
