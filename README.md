# Extending FlashSAC for Humanoid Locomotion

Course project for CMU 16-831 (Introduction to Robot Learning), Fall 2026.
Authors: Henri Sayag, Jaehoon Choi, Nishchay Jasuja, Rafay Mahmood, Shiyu Chen

We reproduce FlashSAC on four IsaacLab humanoid locomotion tasks, then test changes to its critic aimed at lower wall-clock time and higher return.

## Credit: FlashSAC

This project builds on **FlashSAC** and its code. The paper is the one our project studies, and this repository builds off their official implementation.

- **Paper:** *FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control.* Kim, Lee, et al., arXiv 2026. [arXiv:2604.04539](https://arxiv.org/abs/2604.04539)
- **Code:** [github.com/Holiday-Robot/FlashSAC](https://github.com/Holiday-Robot/FlashSAC)
- **Project page:** [holiday-robot.github.io/FlashSAC](https://holiday-robot.github.io/FlashSAC/)

The training framework, agent, environment wrappers and default configs belong to the FlashSAC authors. Our additions are listed below and in [EXTENSIONS.md](EXTENSIONS.md). Please cite the original paper (see [Citation](#citation)) if you use this code.

## Installation

Tested on Ubuntu 22.04/24.04 with an NVIDIA L40S (AWS g6e). For a fresh AWS GPU instance, `bash scripts/aws_bootstrap.sh && source ~/.bashrc` runs steps 1–3 (it pins Python 3.11.14 from `.python-version` and installs the IsaacLab extra).

1. Install [uv](https://docs.astral.sh/uv/):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   export PATH="$HOME/.local/bin:$PATH"
   ```

2. Pin the Python version:

   | Ubuntu | GPU | Command |
   |---|---|---|
   | 22.04 | RTX 30x0, 40x0 | `uv python pin 3.10.18` |
   | 24.04 | RTX 50x0, Bx00 (Blackwell), L40S | `uv python pin 3.11.14` |

3. Install dependencies. The IsaacLab extra is needed for our experiments, and it must live in its own environment, because it conflicts with `genesis` and `humanoid-bench`:

   ```bash
   uv sync --extra isaaclab
   ```

   Other extras: `mujoco-playground`, `maniskill`, `genesis`, `humanoid-bench`, `myosuite`, `metaworld`, `all` (everything except `isaaclab`).

<details>
<summary>MuJoCo / DMC setup (only needed for the non-IsaacLab tasks)</summary>

```bash
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar xvf mujoco210-linux-x86_64.tar.gz && rm mujoco210-linux-x86_64.tar.gz
mkdir -p ~/.mujoco && mv mujoco210 ~/.mujoco/mujoco210
```

Add to `~/.bashrc`:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/$USER/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_GL="egl"
export MUJOCO_EGL_DEVICE_ID="0"
export MKL_SERVICE_FORCE_INTEL="0"
```

Verify with `uv run python -c "import gymnasium; gymnasium.make('HalfCheetah-v4')"`.

`mujoco-playground` has known issues with JAX > 0.5.2 ([issue #153](https://github.com/google-deepmind/mujoco_playground/issues/153)) and may not work with Python 3.11.

</details>

## Running experiments

### Our humanoid runs

`scripts/run_humanoid.sh <exp_name> [--asym] [overrides...]` trains the four humanoid tasks with the project defaults. `scripts/run_variant.sh` wraps it with retries and per-run timeouts for long sweeps:

```bash
bash scripts/run_humanoid.sh baseline
bash scripts/run_humanoid.sh crossq          agent.critic_target=online
bash scripts/run_humanoid.sh crossq_1q       agent.critic_target=online agent.critic_num_qs=1
bash scripts/run_humanoid.sh crossq_1q_w360  agent.critic_target=online agent.critic_num_qs=1 agent.critic_hidden_dim=360
bash scripts/run_humanoid.sh crossq_1q_asym  --asym agent.critic_target=online agent.critic_num_qs=1

# sweep with retries: pick seeds and tasks with env vars
EXP=baseline_1q SEEDS="0 1000 2000" bash scripts/run_variant.sh agent.critic_num_qs=1
```

The header of each script lists its options. Asymmetric runs on the rough tasks need a larger replay buffer than fits on a 48 GB GPU; see [RUNS.md](RUNS.md) for the CPU-buffer workaround and its time cost.

### General training

```bash
uv run python train.py
uv run python train.py --overrides env=dmc --overrides env.env_name='humanoid-walk'
```

Configs use [Hydra](https://hydra.cc/). The base config is `configs/flashSAC_base.yaml`, with sub-configs in `configs/agent/` and `configs/env/`. Upstream example scripts for each simulator are in `scripts/run_<simulator>.sh`.

### Logging

Set `logger_type` in `configs/flashSAC_base.yaml` to `'wandb'` or `'tensorboard'`. TensorBoard logs go to `runs/` (`tensorboard --logdir runs`). Each run also writes a `metrics.csv` (turn off with `log_csv=false`), which the plotting script reads.

## Results and plots

Pulled runs live in `results/<member>/<method>/<task>/<run>/` (for example `results/nish/crossq_1q/Isaac-Velocity-Rough-H1-v0/...`). Superseded runs are kept under `results/<member>/old_runs/`.

```bash
python scripts/plot_results.py --results_dir results/nish --out_dir results/nish/plots \
    --methods baseline crossq crossq_1q crossq_1q_w360 baseline_1q
```

This writes return vs. steps, return vs. wall-clock time, normalized AUC, update-time and time-breakdown plots, and critic TD-error diagnostics. The docstring at the top of [scripts/plot_results.py](scripts/plot_results.py) explains each figure. `scripts/compare_baseline.py` compares our baseline with the paper's curves (`results/nish/plots/paper_vs_baseline.png`).

## Checkpointing

Checkpoints are saved at the end of training to `models/<group>/<exp>/<env_name>/seed<seed>-<timestamp>/step<N>/`. They contain the actor, critic, target critic, temperature, reward normalizer and agent state. To save at intervals:

```bash
uv run python train.py \
    --overrides save_checkpoint_per_interaction_step=24400 \
    --overrides save_buffer_per_interaction_step=24400
```

To resume, pass `agent_load_path` and optionally `buffer_load_path` (both point to a `step<N>` directory). Optimizer and reward-normalizer states are restored by default (`agent.load_optimizer`, `agent.load_reward_normalizer`).

## Visualization (IsaacLab)

Trained IsaacLab agents can be visualized in the Isaac Sim viewport using `play_isaaclab.py`. It uses the same Hydra config system as training, so pass the same `--overrides` you trained with to make the network architecture match the checkpoint.

```bash
uv run python play_isaaclab.py \
    --checkpoint_path 'models/.../step24400' \
    --num_envs 16 \
    --num_episodes 10 \
    --overrides env=isaaclab \
    --overrides env.env_name='Isaac-Velocity-Flat-G1-v0' \
    --overrides agent=flashSAC \
    --overrides agent.asymmetric_observation=true \
    --overrides agent.buffer_max_length=1
```

| Argument | Description |
|---|---|
| `--checkpoint_path` | Path to the saved checkpoint directory (contains `actor.pt`, etc.) |
| `--num_envs` | Number of parallel environments to visualize (default: 16) |
| `--num_episodes` | Number of episodes to run (default: 10) |
| `--overrides` | Same Hydra overrides used during training |

> [!NOTE]
> `agent.buffer_max_length` can be set to a small value (e.g., 1) since the replay buffer is not used during play.

## Project Structure

```
flash_rl/
  agents/       # Agent implementations (FlashSAC, random)
  buffers/      # Replay buffer implementations
  common/       # Logger (wandb / tensorboard / csv)
  envs/         # Environment wrappers (Gymnasium 1.1 API)
  evaluation.py # Evaluation and video recording
configs/           # Hydra configs (base, agent, env)
scripts/           # Launch, AWS setup and plotting scripts
results/           # Experiment results and plots
models/            # Saved checkpoints
docs/              # Upstream FlashSAC project page
train.py           # Training entry point
play_isaaclab.py   # IsaacLab visualization entry point
EXTENSIONS.md      # Config keys and methods we added
RUNS.md            # Run matrix and status
```

## Development

```bash
uv sync --dev    # install formatters, linter, type checker
./bin/lint       # run Black, Ruff, Mypy
```

## Citation

This project is based on FlashSAC. Please cite the original paper:

```bibtex
@article{kim2026flashsac,
  title={FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control},
  author={Kim, Donghu and Lee, Youngdo and Park, Minho and Kim, Kinam and Nahendra, I Made Aswin and Seno, Takuma and Min, Sehee and Palenicek, Daniel and Vogt, Florian and Kragic, Danica and Peters, Jan and Choo, Jaegul and Lee, Hojoon},
  journal={arXiv preprint arXiv:2604.04539},
  year={2026}
}
```

## License

See [LICENSE](LICENSE) (inherited from FlashSAC).
