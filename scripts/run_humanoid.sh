#!/bin/bash
##################################################################################
# Project experiments: IsaacLab humanoid locomotion (G1 / H1, flat / rough)
#
# usage: scripts/run_humanoid.sh <exp_name> [--asym] [--block relu|linear_bottleneck|swiglu] [overrides...]
#
#   --asym     asymmetric critic: the critic also reads a noise-free copy of the policy observations, the actor still
#              reads only the noisy ones (env.privileged_critic=true agent.asymmetric_observation=true)
#   --block B  residual block of the actor and critic (agent.block_type=B): relu (FlashSAC, default),
#              linear_bottleneck or swiglu
#   Flags and overrides can be mixed in any order.
#
#   scripts/run_humanoid.sh uniform_d2
#   scripts/run_humanoid.sh swd_d2      agent.sampler=swd
#   scripts/run_humanoid.sh laber_d2    agent.sampler=laber
#   scripts/run_humanoid.sh window_1to10M agent.sampler=window                      # growing replay window
#   scripts/run_humanoid.sh buffer_1M   agent.buffer_max_length=1_000_000            # its fixed-1M control
#   scripts/run_humanoid.sh uniform_d4  agent.actor_update_period=4 agent.actor_lr_schedule=global_step
#   scripts/run_humanoid.sh uniform_ema agent.actor_delay=ema agent.actor_delay_ema_interval=100 \
#                                       agent.actor_lr_schedule=global_step
#   scripts/run_humanoid.sh crossq      agent.critic_target=online
#   scripts/run_humanoid.sh minto       agent.critic_target=minto
#   scripts/run_humanoid.sh minto_net   agent.critic_target=minto_net
#   scripts/run_humanoid.sh crossq_1q          agent.critic_target=online agent.critic_num_qs=1
#   scripts/run_humanoid.sh crossq_1q_w360     agent.critic_target=online agent.critic_num_qs=1 agent.critic_hidden_dim=360
#   scripts/run_humanoid.sh crossq_sw          agent.critic_target=online agent.critic_surprise_weight_alpha=0.5
#   scripts/run_humanoid.sh crossq_1q_w360_sw  agent.critic_target=online agent.critic_num_qs=1 \
#                                              agent.critic_hidden_dim=360 agent.critic_surprise_weight_alpha=0.5
#   scripts/run_humanoid.sh crossq_asym          --asym agent.critic_target=online
#   scripts/run_humanoid.sh crossq_1q_asym       --asym agent.critic_target=online agent.critic_num_qs=1
#   scripts/run_humanoid.sh crossq_1q_w360_asym  --asym agent.critic_target=online agent.critic_num_qs=1 \
#                                                agent.critic_hidden_dim=360
#   scripts/run_humanoid.sh architecture_swiglu  --block swiglu
#
# SEEDS and ENVS can be set in the environment, e.g. SEEDS="0 1000" ENVS="Isaac-Velocity-Flat-G1-v0".
# Metrics are written to logs/humanoid/<exp_name>/<env_name>/seed<seed>_<time>/metrics.csv
##################################################################################

usage="usage: scripts/run_humanoid.sh <exp_name> [--asym] [--block relu|linear_bottleneck|swiglu] [overrides...]"
exp_name=${1:?$usage}
shift

# Compile every run from scratch. With kernels reused from torch.compile's on-disk cache, runs intermittently crash at
# their first update ("live storage data ptrs are in the cudagraph pool but not accounted for as an output of
# cudagraph trees"); every crash so far started from a warm cache. Costs ~1-2 min of compilation per run.
export TORCHINDUCTOR_FX_GRAPH_CACHE=0 TORCHINDUCTOR_AUTOGRAD_CACHE=0

asym=false
extra_overrides=()
while [ $# -gt 0 ]; do
    case "$1" in
        --asym) asym=true; shift ;;
        --block | --block=*)
            if [ "$1" = --block ]; then block=$2; shift 2; else block=${1#--block=}; shift; fi
            case "$block" in
                relu | linear_bottleneck | swiglu) extra_overrides+=(--overrides "agent.block_type=$block") ;;
                *) echo "--block must be relu, linear_bottleneck or swiglu, got '$block'"; echo "$usage"; exit 1 ;;
            esac ;;
        --*) echo "unknown flag: $1"; echo "$usage"; exit 1 ;;
        *) extra_overrides+=(--overrides "$1"); shift ;;
    esac
done

read -r -a seeds <<< "${SEEDS:-0 1000 2000}"
read -r -a env_names <<< "${ENVS:-Isaac-Velocity-Flat-G1-v0 Isaac-Velocity-Rough-G1-v0 Isaac-Velocity-Flat-H1-v0 Isaac-Velocity-Rough-H1-v0}"

for seed in "${seeds[@]}"; do
    for env_name in "${env_names[@]}"; do
        echo "$exp_name, $env_name, $seed"
        uv run python train.py \
            --config_name flashSAC_base \
            --overrides group_name=humanoid \
            --overrides exp_name=${exp_name} \
            --overrides seed=${seed} \
            `#=== Environment (GPU sim) ===#` \
            --overrides env=isaaclab \
            --overrides env.env_name=${env_name} \
            --overrides num_env_steps=50_000_896 \
            --overrides num_train_envs=1024 \
            --overrides num_eval_envs=null \
            --overrides num_record_envs=null \
            --overrides num_eval_episodes=1024 \
            --overrides num_record_episodes=0 \
            `#=== Agent (GPU sim) ===#` \
            --overrides agent=flashSAC \
            --overrides agent.buffer_max_length=10_000_000 \
            --overrides agent.buffer_min_length=100_000 \
            --overrides agent.buffer_device_type='cuda' \
            --overrides agent.sample_batch_size=2048 \
            --overrides agent.use_amp=true \
            --overrides updates_per_interaction_step=2 \
            `#=== Benchmark default ===#` \
            --overrides agent.asymmetric_observation=${asym} \
            --overrides env.privileged_critic=${asym} \
            --overrides gamma=0.99 \
            --overrides n_step=3 \
            `#=== Experiment ===#` \
            "${extra_overrides[@]}"
    done
done
