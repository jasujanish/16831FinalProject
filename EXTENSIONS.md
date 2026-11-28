# Project extensions (16-831)

Additions to FlashSAC for the course project. With the default config the training behaviour is unchanged.

## Config keys

All keys live in `configs/agent/flashSAC.yaml` and are set with `--overrides agent.<key>=<value>`.

| Key | Values (default first) | Meaning |
|---|---|---|
| `sampler` | `uniform`, `swd`, `laber` | Replay sampling method |
| `swd_decay_steps` | `2000` | SWD decay horizon, in interaction steps |
| `swd_min_weight` | `0.1` | SWD weight floor |
| `laber_multiplier` | `4` | LaBER candidate batch = multiplier × `sample_batch_size` |
| `actor_delay` | `fixed`, `ema` | Actor delay rule |
| `actor_update_period` | `2` | Delay `d` for `actor_delay=fixed` |
| `actor_delay_ema_interval` | `100` | `k`: critic updates between delay recomputations |
| `actor_delay_ema_beta` | `0.9` | EMA decay, applied once per interval |
| `actor_delay_ema_scale` | `2.0` | Delay when the TD error equals its EMA |
| `actor_delay_ema_min` / `_max` | `1.0` / `8.0` | Clip range of the delay |
| `actor_lr_schedule` | `actor_step`, `global_step` | What drives the actor/temperature learning-rate schedule |
| `critic_target` | `target`, `online`, `minto`, `minto_net` | Which networks the TD target is built from |
| `critic_num_qs` | `2` | Critics in the ensemble. `1` = a single critic, no clipped double Q |
| `critic_surprise_weight_alpha` | `0.0` | In-batch surprise weighting of the critic loss (`0` = off) |
| `critic_surprise_weight_min` / `_max` | `0.5` / `3.0` | Clip range of the surprise weights |
| `block_type` | `relu`, `linear_bottleneck`, `swiglu` | Residual block of the actor and critic (Rafay's arch variants): FlashSAC's 4× ReLU block; 2× expansion without the ReLU after the block; gated SiLU with ≈ the same parameters |
| `asymmetric_observation` | `false`, `true` | The critic also reads the env's privileged observations (use with `env.privileged_critic=true`) |

`log_csv` (default `true`, in `configs/flashSAC_base.yaml`) turns the CSV log on or off.

## Methods

**SWD** (`flash_rl/buffers/torch_buffer.py`). WarpSAC's Sample Weight Decay: a transition added `age` interaction
steps ago is sampled with probability ∝ `max(swd_min_weight, 1 - age / swd_decay_steps)`. No importance correction,
as in WarpSAC: both the loss and the BatchNorm statistics follow the recency-biased batch.
This is FlashSAC + SWD, not full WarpSAC (which also changes the critic).

**LaBER** (`FlashSACAgent._select_laber_batch`). Draw `laber_multiplier × B` transitions uniformly, score each with

    surprise_i = mean over the 2 critics of ‖softmax(logits(s_i, a_i)) − target_probs_i‖₂

and keep `B` of them with probability ∝ surprise (with replacement). The importance weights
`w_i = mean(surprise) / surprise_i` correct the critic update for this selection in two places: the loss terms, and the
BatchNorm batch statistics of the critic and target critic (weighted mean and variance). Both corrections are
self-normalised estimates of their uniform-sampling counterparts, so the update is approximately, not exactly,
unbiased. Scoring runs with BatchNorm in eval mode. The actor is updated on a uniform slice of the candidates, not on
the high-surprise selection.

**EMA actor delay** (`flash_rl/agents/flashSAC/actor_delay.py`). Every `interval` critic updates, with `e` the mean
absolute TD error over those updates:

    ema   = beta * ema + (1 - beta) * e
    delay = clip(scale * e / ema, min, max)

The delay is real-valued: each critic update earns `1/delay` credit and an actor update costs 1.

**Actor LR schedule.** In FlashSAC the actor's cosine decay advances once per actor update, so its learning-rate
schedule depends on the delay. `actor_lr_schedule=global_step` ties it to the number of critic updates instead. Use it
for all runs of a delay comparison (it also changes the `d=2` baseline).

**Critic targets** (`_select_target_log_probs` in `flash_rl/agents/flashSAC/update.py`). Every option keeps the
next-state return distribution with the lowest Q (the mean of the distribution) among its candidates, then builds the
categorical TD target from it as before. The online candidates are the next-state half of the critic's own joint
`[s, s']` forward pass, gradient stopped, so they cost no extra forward pass.

| Option | Candidates | Notes |
|---|---|---|
| `target` | the 2 target critics | FlashSAC (clipped double Q) |
| `online` | the 2 online critics | CrossQ-style: no target network. The target critic is neither called nor Polyak-updated; checkpoints save the current critic as `target_critic.pt` |
| `minto` | the 2 target + the 2 online critics | MINTO ([Hendawy et al.](https://arxiv.org/abs/2510.02590)); never above `target` |
| `minto_net` | mean of the 2 target critics vs. mean of the 2 online critics | MINTO without clipped double Q: all pessimism comes from the online-vs-target minimum |

The actor objective is unchanged (`min` over the online critics).

**Single critic** (`critic_num_qs=1`). The critic ensemble has one member instead of two, for the critic and the target
critic alike. The `min` over critics in the TD target and the actor objective is then the critic itself, so there is no
clipped double Q. With the same width the critic has half the parameters of the default pair; `critic_hidden_dim=360`
gives about the same number of parameters as the pair at width 256 (2.18M vs 2.24M on Flat-G1, 2.24M vs 2.34M on
Rough-G1). The default `critic_num_qs=2` reproduces the previous updates exactly.

**In-batch surprise weighting** (`critic_surprise_weight_alpha > 0`, `surprise_weights` in
`flash_rl/agents/flashSAC/update.py`). The batch is still sampled uniformly. Each sample's cross-entropy is weighted by

    w = clip((surprise / mean surprise) ** alpha, min, max), renormalized to mean 1 over the batch

with `surprise` as above, computed from the same forward pass as the loss (no extra network call). Weighting the loss of
uniform samples by `p_i / mean(p)` has the same expected gradient as sampling with probability `p_i` (Fujimoto, Meger &
Precup, "An Equivalence between Loss Functions and Non-Uniform Sampling in Experience Replay", NeurIPS 2020), so this is
prioritized replay without importance correction, with priorities that are fresh on every batch. Unlike LaBER, which
corrects its selection with importance weights and so keeps the uniform expected gradient, this deliberately changes it.
The weights scale only the critic loss: BatchNorm statistics (the batch is uniform) and the actor update are unchanged.
Under LaBER they multiply its importance weights. LaBER scoring always uses the target network, so
`sampler=laber` requires `critic_target=target`. The online pass now runs before the target pass, which does not change
the `target` results (the two networks share no state).

## Running

    scripts/run_humanoid.sh <exp_name> [--asym] [--block relu|linear_bottleneck|swiglu] [overrides...]

runs the four IsaacLab humanoid tasks for each seed; see the examples at the top of the script. `--block B` sets
`agent.block_type=B` for the actor and critic. Flags and overrides can be mixed in any order.

To make runs comparable with ours (machine, setup, exact commands, memory, smoke tests, what to send
back, plotting), follow `STEPS.md`. One check it does not cover: diff the first run's `config.yaml` against the
`results/nish` run of the same method without `_asym`. Only `exp_name`, `seed`, `save_path`, `env.privileged_critic`,
`agent.asymmetric_observation` (and `agent.buffer_device_type`, if set) may differ.

## Privileged critic (IsaacLab)

IsaacLab adds noise to the policy observations (joint velocities ±1.5, base angular velocity ±0.2, height scan ±0.1,
...). `env.privileged_critic=true` adds a `critic` observation group, a copy of the policy terms with the noise turned
off (`add_privileged_critic_group` in `flash_rl/envs/isaaclab.py`). The env then returns `[policy obs, critic obs]`;
with `agent.asymmetric_observation=true` the actor reads only the first (noisy) part and the critic reads both, so the
trained policy takes the same inputs as before.

`--asym` sets both keys (see `STEPS.md` for the commands and the memory each task needs).

- The agent refuses to start with `env.privileged_critic=true` alone, because the actor would then be
  trained on the noise-free observations.
- `play_isaaclab.py` needs the same two overrides to load a checkpoint trained this way.
- Before this change `agent.asymmetric_observation=true` had no effect on IsaacLab tasks: the check for a `critic`
  group was always false, and the combined observation shape was built by concatenating two shape tuples.
- A task that already defines its own `critic` group is not supported yet (`privileged_critic=true` raises).

H1, 3 seeds, 50M steps, final return (mean ± s.e.) against `uniform_d2`: Flat-H1 40.04 ± 0.34 vs. 38.72 ± 0.59,
Rough-H1 31.51 ± 0.26 vs. 28.12 ± 0.63; about 6–10% fewer env steps per second. G1 has not been run.

## Logs

Each run writes `logs/<group_name>/<exp_name>/<env_name>/seed<seed>_<time>/`:

- `config.yaml`: the resolved config of the run.
- `metrics.csv`: one row per logging interval (1% of training). Metrics are averaged over the interval unless noted.
  A cell is empty when the metric was not produced in that interval.

| Columns | Meaning |
|---|---|
| `env_step`, `wall_time` | Row index: environment steps, and seconds since the logger was created |
| `avg_return`, `avg_length`, `avg_success_*` | Evaluation results (10 evaluations per run, plus one at step 0) |
| `eval_env_step`, `eval_train_time`, `eval_num_updates` | Exactly when that evaluation ran. Use these for return curves |
| `time/train_time` | Seconds of training, excluding evaluation and video recording |
| `time/collect_time`, `time/update_time`, `time/eval_time` | Cumulative seconds in env stepping, gradient updates, evaluation |
| `time/env_steps_per_sec`, `time/gpu_max_memory_gb` | Throughput and peak GPU memory |
| `train/num_updates`, `train/num_actor_updates` | Cumulative critic and actor updates |
| `critic/td_abs` | Mean abs. TD error between distribution means (importance-weighted under LaBER). Drives the EMA delay |
| `critic/surprise` | Mean surprise, same weighting |
| `critic/loss`, `critic/q_mean`, `critic/learning_rate` | Critic loss, mean Q of the batch, learning rate |
| `actor/loss`, `actor/entropy`, `actor/learning_rate`, `temperature/value` | Actor and temperature state |
| `replay/age_mean`, `_median`, `_p90`, `_max` | Staleness of the critic's batch, in interaction steps |
| `replay/batch_td_abs`, `replay/batch_surprise` | Unweighted error of the critic's batch |
| `replay/age_surprise_corr` | Pearson correlation between age and surprise in the critic's batch |
| `replay/size` | Transitions in the buffer |
| `critic_target/gap` | Mean \|Q_target − Q_online\| at (s', a') over both critics: how stale the target network is. Not logged for `online` |
| `critic_target/online_frac` | Fraction of the batch whose TD target came from an online critic (`minto`, `minto_net`) |
| `actor_delay/delay`, `td_ratio`, `td_ema` | Current delay, `e / ema`, and the EMA |
| `actor_delay/actor_updated` | Fraction of critic updates that also updated the actor |
| `laber/candidate_*` | Surprise, TD error, age and their correlation over the uniform candidates |
| `laber/effective_sample_frac`, `weight_max`, `unique_frac` | How concentrated the LaBER selection is |
| `critic/surprise_weight_min`, `_max`, `_ess` | Range of the surprise weights and their effective sample fraction (1 = uniform) |
| `critic/surprise_weight_age_corr` | Pearson correlation between sample age and surprise weight |

## Videos

Checkpoints are saved at the end of training to `models/<group_name>/<exp_name>/<env_name>/seed<seed>-<time>/step<N>/`.
To record a trained policy to an mp4 without a display:

    uv run python play_isaaclab.py \
        --checkpoint_path 'models/humanoid/<exp_name>/Isaac-Velocity-Flat-G1-v0/seed0-<time>/step<N>' \
        --video --video_length 500 --follow_robot \
        --overrides env=isaaclab \
        --overrides env.env_name='Isaac-Velocity-Flat-G1-v0' \
        --overrides agent.buffer_max_length=1

The video (`rl-video-step-0.mp4`, 50 steps per second) goes to `<checkpoint_path>/videos` unless `--video_dir` is set.
`--follow_robot` keeps the camera on the robot of environment 0; without it the camera is fixed. Pass any overrides
that change the network (e.g. `agent.critic_hidden_dim`) exactly as in training. Without `--video` the script opens
the Isaac Sim viewport as before.

## Tests

The test command is in `STEPS.md` (step 2). `tests/test_extensions.py` is CPU-only; it covers the SWD sampling
distribution, ages, the delay rules, LaBER weights, the LR schedule, the critic target options and the CSV log. The tests disable `torch.compile`, so run a short GPU job per `critic_target` option
before a sweep.
