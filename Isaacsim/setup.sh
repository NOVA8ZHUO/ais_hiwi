#!/bin/bash
# Isaac Sim - Setup Script
# Author: Moritz Giggenbach
# Target: Ubuntu/Linux | Isaac Sim 5.1.0
# Compatibility: Scalable from RTX 2060 (6GB) to High-End GPUs

set -e  # Exit on error

echo "============================================="
echo "  Isaac Sim — Setup"
echo "============================================="

# ---------------------------------------------------------------------------
# 1. Check NVIDIA Driver
# ---------------------------------------------------------------------------
echo ""
echo "[1/7] Checking NVIDIA driver..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "  ✗ ERROR: No NVIDIA driver found. Isaac Sim requires an NVIDIA GPU."
    echo "  Install driver ≥ 535.x from: https://www.nvidia.com/drivers"
    exit 1
fi

DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "  ✓ NVIDIA driver found: v${DRIVER_VERSION}"

# Check VRAM
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "  ✓ GPU VRAM: ${VRAM_MB} MB"
if [ "$VRAM_MB" -lt 6000 ]; then
    echo "  ⚠ WARNING: Less than 6 GB VRAM detected. Isaac Sim may struggle."
    echo "    Consider using --headless mode for training."
fi

# ---------------------------------------------------------------------------
# 2. Check / Install Conda
# ---------------------------------------------------------------------------
echo ""
echo "[2/7] Checking Conda..."
if ! command -v conda &> /dev/null; then
    echo "  Conda not found. Installing Miniconda..."
    mkdir -p ~/miniconda3
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
         -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
    source ~/miniconda3/bin/activate
    conda init bash
    echo "  ✓ Miniconda installed. Please restart your terminal after setup."
else
    echo "  ✓ Conda is already installed."
fi

# ---------------------------------------------------------------------------
# 3. Install System Dependencies
# ---------------------------------------------------------------------------
echo ""
echo "[3/7] Installing system dependencies (sudo required)..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    libglu1-mesa libvulkan1 vulkan-tools libnss3 libasound2 \
    libxcomposite1 libxcursor1 libxi6 libxtst6 libxrandr2
echo "  ✓ System dependencies installed."

# ---------------------------------------------------------------------------
# 4. Create Conda Environment
# ---------------------------------------------------------------------------
echo ""
echo "[4/7] Creating Conda environment 'Xinyu_isaacsim' (Python 3.11)..."
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | grep -q "Xinyu_isaacsim"; then
    echo "  ✓ Environment 'Xinyu_isaacsim' already exists."
else
    conda create -y -n Xinyu_isaacsim python=3.11
    echo "  ✓ Environment created."
fi
conda activate Xinyu_isaacsim

# ---------------------------------------------------------------------------
# 5. Install Isaac Sim
# ---------------------------------------------------------------------------
echo ""
echo "[5/7] Installing Isaac Sim 5.1.0 via pip..."
pip install --upgrade pip
pip install isaacsim[all,extscache]==5.1.0 \
    --extra-index-url https://pypi.nvidia.com
export OMNI_KIT_ACCEPT_EULA=YES
echo "  ✓ Isaac Sim 5.1.0 installed."

# ---------------------------------------------------------------------------
# 6. Install Isaac Lab
# ---------------------------------------------------------------------------
echo ""
echo "[6/7] Installing Isaac Lab (Multi-Agent RL framework)..."
if [ ! -d "IsaacLab" ]; then
    echo "  Cloning Isaac Lab repository..."
    git clone https://github.com/isaac-sim/IsaacLab.git
fi
cd IsaacLab
echo "  Running Isaac Lab setup script..."
./isaaclab.sh --install
cd ..
echo "  ✓ Isaac Lab installed."

echo "  Installing additional RL dependencies (pettingzoo, skrl)..."
pip install tensorboard
echo "  ✓ RL dependencies installed."

# ---------------------------------------------------------------------------
# 7. Validate Installation
# ---------------------------------------------------------------------------
echo ""
echo "[7/7] Validating installation..."

# Check Isaac Sim import
python -c "from isaacsim import SimulationApp; print('  ✓ Isaac Sim import OK')" 2>/dev/null || {
    echo "  ✗ Isaac Sim import failed. Check installation."
    exit 1
}

