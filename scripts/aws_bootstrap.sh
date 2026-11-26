#!/usr/bin/env bash
##################################################################################
# One-time setup of a fresh g6e.xlarge (1x L40S) for the IsaacLab runs.
# AMI: Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04 or 22.04)
# Stack (.python-version 3.11.14): torch 2.9.1+cu128, Isaac Sim 5.1.0, Isaac Lab 2.3.0
#
# usage (from the repo root): bash scripts/aws_bootstrap.sh && source ~/.bashrc
##################################################################################
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== GPU / driver"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | cut -d. -f1)
if [ "$driver_major" -lt 570 ]; then
    echo "NVIDIA driver ${driver_major}.x is too old for torch 2.9.1 (CUDA 12.8) / Isaac Sim 5.1. Use a newer DLAMI." >&2
    exit 1
fi

echo "== System packages"
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential git tmux htop curl
# Isaac Sim (Kit) runtime libraries; some names differ between 22.04 and 24.04, so install one by one
for pkg in libvulkan1 vulkan-tools libglu1-mesa libgl1 libegl1 libxt6 libxt6t64 libxrandr2 libxinerama1 libxcursor1 libsm6; do
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "$pkg" >/dev/null 2>&1 || echo "  (skipped $pkg)"
done
vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverVersion" \
    || echo "WARNING: vulkaninfo sees no GPU; Isaac Sim may fail to start" >&2

echo "== 16 GB swap (safety margin for 32 GiB RAM during compile + Isaac Sim)"
if ! swapon --show | grep -q /swapfile; then
    sudo fallocate -l 16G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
fi

echo "== Environment variables"
if ! grep -q "# flashsac-env" ~/.bashrc; then
    cat >> ~/.bashrc <<'EOF'
# flashsac-env
export PATH="$HOME/.local/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=YES
export UV_NO_SYNC=1
export CUDA_HOME=/usr/local/cuda
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export MKL_SERVICE_FORCE_INTEL=0
EOF
fi
export PATH="$HOME/.local/bin:$PATH" OMNI_KIT_ACCEPT_EULA=YES

echo "== uv, Python $(cat .python-version), locked deps incl. Isaac Sim / Isaac Lab (~20-30 GB)"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install "$(cat .python-version)"
uv sync --locked --extra isaaclab

echo "== Checks (first Isaac launch builds caches; can take several minutes)"
uv run --no-sync python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
uv run --no-sync python -c "from isaaclab.app import AppLauncher; app = AppLauncher(headless=True).app; import isaaclab_tasks; print('Isaac Sim + Isaac Lab OK'); app.close()"
echo "Bootstrap done. Now: source ~/.bashrc"