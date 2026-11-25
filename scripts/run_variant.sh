#!/bin/bash
# usage: EXP=<name> SEEDS="0 1000 2000" ENVS="..." MAX_HOURS_PER_RUN=3 SHUTDOWN=1 \
#        bash scripts/run_variant.sh <override> [<override> ...]
# e.g.   EXP=gamma995 bash scripts/run_variant.sh gamma=0.995
set -u
cd "$(dirname "$0")/.."
export UV_NO_SYNC=1 OMNI_KIT_ACCEPT_EULA=YES
exp="${EXP:?set EXP=<experiment name>}"
read -r -a seeds <<< "${SEEDS:-0 1000 2000}"
read -r -a env_names <<< "${ENVS:-Isaac-Velocity-Flat-G1-v0 Isaac-Velocity-Rough-G1-v0 Isaac-Velocity-Flat-H1-v0 Isaac-Velocity-Rough-H1-v0}"
done_run() { ls "models/humanoid/${exp}/$1/" 2>/dev/null | grep -q "^seed$2-"; }

mkdir -p stdout_logs failed_runs
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 60 >> "stdout_logs/nvidia_smi_${exp}.csv" &
smi_pid=$!

for seed in "${seeds[@]}"; do
  for env_name in "${env_names[@]}"; do
    for attempt in 1 2 3; do
      done_run "$env_name" "$seed" && break
      [ "$attempt" -gt 1 ] && rm -rf "/tmp/torchinductor_$(whoami)"
      log="stdout_logs/${exp}_${env_name}_seed${seed}_$(date +%m%d-%H%M%S).log"
      echo "[$(date)] start ${env_name} seed ${seed} attempt ${attempt}" | tee -a "stdout_logs/sweep_${exp}.log"
      SEEDS="${seed}" ENVS="${env_name}" timeout --kill-after=5m "${MAX_HOURS_PER_RUN:-3}h" \
        bash scripts/run_humanoid.sh "${exp}" "$@" > "${log}" 2>&1
      if done_run "$env_name" "$seed"; then
        echo "[$(date)] done ${env_name} seed ${seed}" | tee -a "stdout_logs/sweep_${exp}.log"; break
      fi
      echo "[$(date)] FAILED ${env_name} seed ${seed} attempt ${attempt}" | tee -a "stdout_logs/sweep_${exp}.log"
      mv "logs/humanoid/${exp}/${env_name}/seed${seed}_"* failed_runs/ 2>/dev/null
    done
  done
done

kill "${smi_pid}"
{ git rev-parse HEAD; git status --short; uv pip list; nvidia-smi; } > "stdout_logs/provenance_${exp}.txt" 2>&1
tar czf "${exp}_results_$(date +%m%d-%H%M).tar.gz" "logs/humanoid/${exp}" stdout_logs/*"${exp}"* \
    $(ls -d "models/humanoid/${exp}" "runs/humanoid/${exp}" 2>/dev/null)
if [ "${SHUTDOWN:-1}" = 1 ]; then
  sudo shutdown -h now
fi
