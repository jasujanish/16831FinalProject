#!/bin/bash
##################################################################################
# LaBER sweep on one GPU (wraps scripts/run_humanoid.sh, so every setting matches the baseline).
#
# usage: SEEDS="0" ENVS="..." MAX_HOURS_PER_RUN=9 SHUTDOWN=1 scripts/run_laber.sh
# Order: all tasks for the first seed, then the next seed.
##################################################################################
set -u
cd "$(dirname "$0")/.."
export UV_NO_SYNC=1 OMNI_KIT_ACCEPT_EULA=YES

read -r -a seeds <<< "${SEEDS:-0 1000 2000}"
read -r -a env_names <<< "${ENVS:-Isaac-Velocity-Flat-G1-v0 Isaac-Velocity-Rough-G1-v0 Isaac-Velocity-Flat-H1-v0 Isaac-Velocity-Rough-H1-v0}"

mkdir -p stdout_logs
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 60 >> stdout_logs/nvidia_smi.csv &
smi_pid=$!

for seed in "${seeds[@]}"; do
    for env_name in "${env_names[@]}"; do
        log="stdout_logs/laber_d2_${env_name}_seed${seed}_$(date +%m%d-%H%M%S).log"
        echo "[$(date)] start ${env_name} seed ${seed}" | tee -a stdout_logs/sweep.log
        SEEDS="${seed}" ENVS="${env_name}" timeout --kill-after=5m "${MAX_HOURS_PER_RUN:-9}h" \
            bash scripts/run_humanoid.sh laber_d2 agent.sampler=laber > "${log}" 2>&1
        rc=$?
        echo "[$(date)] end ${env_name} seed ${seed} exit=${rc}" | tee -a stdout_logs/sweep.log
    done
done

kill "${smi_pid}"
{ git rev-parse HEAD; git status --short; uv pip list; nvidia-smi; } > stdout_logs/provenance.txt 2>&1
tar czf "laber_results_$(date +%m%d-%H%M).tar.gz" \
    logs/humanoid/laber_d2 runs/humanoid/laber_d2 models/humanoid/laber_d2 stdout_logs
[ "${SHUTDOWN:-1}" = 1 ] && sudo shutdown -h now