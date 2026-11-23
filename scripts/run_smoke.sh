#!/bin/bash
##################################################################################
# Smoke tests: each method for 2M env steps on one task and one seed (~8 min each on an L40S).
#
# usage: scripts/run_smoke.sh [exp_name ...]      (default: all four)
#
#   scripts/run_smoke.sh
#   scripts/run_smoke.sh smoke_minto smoke_crossq
#
# Console output of each run is also written to ~/<exp_name>.log.
##################################################################################

cd "$(dirname "$0")/.." || exit 1
export UV_NO_SYNC=1 OMNI_KIT_ACCEPT_EULA=YES SEEDS=0 ENVS=Isaac-Velocity-Flat-G1-v0

declare -A overrides=(
    [smoke_baseline]=""
    [smoke_minto]="agent.critic_target=minto"
    [smoke_crossq]="agent.critic_target=online"
    [smoke_laber]="agent.sampler=laber"
)

exps=("$@")
[ ${#exps[@]} -eq 0 ] && exps=(smoke_baseline smoke_minto smoke_crossq smoke_laber)

for exp in "${exps[@]}"; do
    [ -v "overrides[$exp]" ] || { echo "unknown experiment: $exp"; exit 1; }
    # shellcheck disable=SC2086  # the override string is split into separate arguments on purpose
    scripts/run_humanoid.sh "$exp" ${overrides[$exp]} num_env_steps=2_000_896 2>&1 | tee ~/"$exp".log
done
echo "ALL SMOKE TESTS DONE"
