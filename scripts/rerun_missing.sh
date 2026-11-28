#!/bin/bash
##################################################################################
# Rerun the tasks of an experiment that have no complete run, after the current queue finishes.
#
# usage: scripts/rerun_missing.sh "<exp_name>:<agent overrides>" ...
#
#   scripts/rerun_missing.sh "baseline_d4:agent.actor_update_period=4" \
#                            "minto_d4:agent.critic_target=minto agent.actor_update_period=4"
#
# Waits until no scripts/run_humanoid.sh is running, then, for each experiment and each of the 4 humanoid tasks
# without a metrics.csv that reaches the final env step, reruns that task (seed 0, up to 2 attempts).
# Reruns disable the on-disk torch.compile cache: the intermittent CUDA-graph error ("live storage data ptrs are in
# the cudagraph pool but not accounted for") has only hit runs that could reuse cached kernels.
##################################################################################

cd "$(dirname "$0")/.." || exit 1
export UV_NO_SYNC=1 OMNI_KIT_ACCEPT_EULA=YES
FINAL_STEP=50000896
ENVS_ALL="Isaac-Velocity-Flat-G1-v0 Isaac-Velocity-Rough-G1-v0 Isaac-Velocity-Flat-H1-v0 Isaac-Velocity-Rough-H1-v0"

# wait for the queue; require 3 idle checks in a row so the gap between two chained runs is not mistaken for the end
idle=0
while [ $idle -lt 3 ]; do
    if pgrep -f "[r]un_humanoid.sh" >/dev/null; then idle=0; else idle=$((idle + 1)); fi
    sleep 20
done

complete() {  # complete <exp> <env>: is there a run of this task that reached the final step?
    local f
    for f in logs/humanoid/"$1"/"$2"/*/metrics.csv; do
        [ -f "$f" ] && [ "$(tail -1 "$f" | cut -d, -f1)" = "$FINAL_STEP" ] && return 0
    done
    return 1
}

for spec in "$@"; do
    exp=${spec%%:*}
    overrides=${spec#*:}
    for env in $ENVS_ALL; do
        for attempt in 1 2; do
            complete "$exp" "$env" && break
            echo "rerun: $exp $env (attempt $attempt)"
            # shellcheck disable=SC2086  # the override string is split into separate arguments on purpose
            SEEDS=0 ENVS=$env TORCHINDUCTOR_FX_GRAPH_CACHE=0 TORCHINDUCTOR_AUTOGRAD_CACHE=0 \
                scripts/run_humanoid.sh "$exp" $overrides 2>&1 | tee -a ~/"${exp}"_rerun.log
        done
        complete "$exp" "$env" || echo "STILL MISSING: $exp $env"
    done
done
echo "RERUNS DONE"
