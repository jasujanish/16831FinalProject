#!/bin/bash
# Run two architecture experiments against the same humanoid environments/seeds.
#
# usage:
#   SEEDS="0" ENVS="Isaac-Velocity-Flat-G1-v0" bash scripts/run_architecture_variants.sh
#   SEEDS="0 1000 2000" bash scripts/run_architecture_variants.sh
#   EXP_PREFIX=mlp_arch bash scripts/run_architecture_variants.sh
#
# Extra run_humanoid agent overrides can be passed as arguments.
set -euo pipefail
cd "$(dirname "$0")/.."

exp_prefix="${EXP_PREFIX:-architecture}"
shutdown="${SHUTDOWN:-0}"

# Reduce the expansion from 4x to 2x and remove the post-projection ReLU.
EXP="${exp_prefix}_linear_bottleneck" SHUTDOWN=0 \
  bash scripts/run_variant.sh \
  agent.block_type=linear_bottleneck "$@"

# Test a SwiGLU gate at roughly the baseline block's parameter count.
EXP="${exp_prefix}_swiglu" SHUTDOWN=0 \
  bash scripts/run_variant.sh \
  agent.block_type=swiglu "$@"

if [[ "${shutdown}" == 1 ]]; then
  sudo shutdown -h now
fi
