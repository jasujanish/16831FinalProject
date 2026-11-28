#!/bin/bash
##################################################################################
# Sweep of one arm on one GPU. Default: the FlashSAC baseline (uniform replay, fixed actor delay d=2, default LR
# schedule), exp_name uniform_d2. It wraps scripts/run_humanoid.sh, so every other setting matches across arms.
#
# usage: EXP=uniform_d2 SEEDS="0 1000 2000" ENVS="..." MAX_HOURS_PER_RUN=9 ATTEMPTS=2 SHUTDOWN=0 \
#        scripts/run_baseline.sh [agent overrides...]
#   EXP=window_1to10M scripts/run_baseline.sh agent.sampler=window
#   EXP=buffer_1M     scripts/run_baseline.sh agent.buffer_max_length=1_000_000
# A run counts as complete only if its metrics.csv reaches the final env step. The exit code is not reliable: Isaac Sim
# can crash and still exit with 0. Incomplete runs are repeated up to ATTEMPTS times, except after a timeout.
# Order: all tasks for the first seed, then the next seed, so a partial sweep still covers every task.
# Then: uv run --no-sync python scripts/compare_baseline.py --exp <EXP>
##################################################################################
set -u
cd "$(dirname "$0")/.."
export UV_NO_SYNC=1 OMNI_KIT_ACCEPT_EULA=YES

exp_name=${EXP:-uniform_d2}
if [ "$#" -gt 0 ] && [ "${exp_name}" = uniform_d2 ]; then
    echo "set EXP=<name>: overrides must not run under the baseline's name" >&2
    exit 1
fi
is_complete() { grep -qs "^50000896," logs/humanoid/"${exp_name}"/"$1"/seed"$2"_*/metrics.csv; }
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
        if is_complete "${env_name}" "${seed}"; then
            echo "[$(date)] skip ${env_name} seed ${seed} (complete run exists)" | tee -a "stdout_logs/sweep_${exp_name}.log"
            continue
        fi
        for attempt in $(seq 1 "${ATTEMPTS:-2}"); do
            log="stdout_logs/${exp_name}_${env_name}_seed${seed}_$(date +%m%d-%H%M%S).log"
            echo "[$(date)] start ${env_name} seed ${seed} attempt ${attempt}" | tee -a "stdout_logs/sweep_${exp_name}.log"
            SEEDS="${seed}" ENVS="${env_name}" timeout --kill-after=5m "${MAX_HOURS_PER_RUN:-9}h" \
                bash scripts/run_humanoid.sh "${exp_name}" "$@" > "${log}" 2>&1
            rc=$?
            complete=no; is_complete "${env_name}" "${seed}" && complete=yes
            echo "[$(date)] end ${env_name} seed ${seed} exit=${rc} complete=${complete}" | tee -a "stdout_logs/sweep_${exp_name}.log"
            [ "${complete}" = yes ] && break
            { [ "${rc}" -eq 124 ] || [ "${rc}" -eq 137 ]; } && break
        done
    done
done

tar czf "${exp_name}_results_$(date +%m%d-%H%M).tar.gz" \
    "logs/humanoid/${exp_name}" "runs/humanoid/${exp_name}" "models/humanoid/${exp_name}" stdout_logs 2>/dev/null
[ "${SHUTDOWN:-0}" = 1 ] && sudo shutdown -h now
exit 0
