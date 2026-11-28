#!/bin/bash
##################################################################################
# FlashSAC baseline sweep on one GPU: uniform replay, fixed actor delay d=2, default LR schedule.
# It wraps scripts/run_humanoid.sh with no extra overrides, so every setting matches the other arms
# (LaBER, SWD, actor delay). exp_name is uniform_d2.
#
# usage: SEEDS="0 1000 2000" ENVS="..." MAX_HOURS_PER_RUN=9 SHUTDOWN=0 scripts/run_baseline.sh
# Order: all tasks for the first seed, then the next seed, so a partial sweep still covers every task.
# Then: uv run --no-sync python scripts/compare_baseline.py
##################################################################################
set -u
cd "$(dirname "$0")/.."
export UV_NO_SYNC=1 OMNI_KIT_ACCEPT_EULA=YES

exp_name=uniform_d2
read -r -a seeds <<< "${SEEDS:-0 1000 2000}"
read -r -a env_names <<< "${ENVS:-Isaac-Velocity-Flat-G1-v0 Isaac-Velocity-Rough-G1-v0 Isaac-Velocity-Flat-H1-v0 Isaac-Velocity-Rough-H1-v0}"

mkdir -p stdout_logs
# record what ran before the sweep starts, so a crashed or stopped sweep still has it
{ date; git rev-parse HEAD; git status --short; nvidia-smi; uv pip list; } > "stdout_logs/provenance_${exp_name}.txt" 2>&1

nvidia-smi --query-gpu=timestamp,name,memory.used,utilization.gpu --format=csv -l 60 >> "stdout_logs/nvidia_smi_${exp_name}.csv" &
smi_pid=$!
trap 'kill "${smi_pid}" 2>/dev/null' EXIT

for seed in "${seeds[@]}"; do
    for env_name in "${env_names[@]}"; do
        # skip a task/seed that already has a complete run (its metrics.csv reaches the final env step)
        if grep -qs "^50000896," logs/humanoid/${exp_name}/${env_name}/seed${seed}_*/metrics.csv; then
            echo "[$(date)] skip ${env_name} seed ${seed} (complete run exists)" | tee -a "stdout_logs/sweep_${exp_name}.log"
            continue
        fi
        log="stdout_logs/${exp_name}_${env_name}_seed${seed}_$(date +%m%d-%H%M%S).log"
        echo "[$(date)] start ${env_name} seed ${seed}" | tee -a "stdout_logs/sweep_${exp_name}.log"
        SEEDS="${seed}" ENVS="${env_name}" timeout --kill-after=5m "${MAX_HOURS_PER_RUN:-9}h" \
            bash scripts/run_humanoid.sh "${exp_name}" > "${log}" 2>&1
        rc=$?
        echo "[$(date)] end ${env_name} seed ${seed} exit=${rc}" | tee -a "stdout_logs/sweep_${exp_name}.log"
    done
done

tar czf "${exp_name}_results_$(date +%m%d-%H%M).tar.gz" \
    "logs/humanoid/${exp_name}" "runs/humanoid/${exp_name}" "models/humanoid/${exp_name}" stdout_logs 2>/dev/null
[ "${SHUTDOWN:-0}" = 1 ] && sudo shutdown -h now
exit 0
