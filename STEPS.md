# STEPS.md
Follow these steps and your results can be plotted and compared directly with `results/nish/`. Every step matters: a different machine, launcher or setting makes the comparison meaningless.

## 1. Get the same machine

- **AWS g6e.2xlarge** (NVIDIA L40S GPU, 64 GB RAM), region us-east-2.
- **AMI:** Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu).
- **Disk:** 200 GB. Isaac Sim is large.
- Run **one job per GPU** at a time.

**Exception:** asymmetric-critic runs on the *rough* tasks need more RAM (see step 5).

## 2. Set up the instance (once)

```bash
# uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc

# accept the Isaac Sim license; stop uv from re-syncing on every launch
echo 'export OMNI_KIT_ACCEPT_EULA=YES' >> ~/.bashrc
echo 'export UV_NO_SYNC=1' >> ~/.bashrc
source ~/.bashrc

# system libraries Isaac Sim needs
sudo apt-get update
sudo apt-get install -y libglu1-mesa libxt6t64      # Ubuntu 22.04: libxt6 instead of libxt6t64

# code: the Nish branch, with no local edits
git clone -b Nish https://github.com/jasujanish/16831FinalProject.git
cd 16831FinalProject
uv sync --extra isaaclab

# check the install: 45 tests should pass
uv run python -m pytest tests/ -q
```

## 3. Always launch with `scripts/run_humanoid.sh`

The script fixes every shared setting:
- 50M steps, 1024 parallel envs, batch 2048
- 2 updates per step, 10M replay buffer
- n-step 3, γ = 0.99, mixed precision

It also runs **all 4 tasks × 3 seeds (0, 1000, 2000)** one after another. Don't change the seeds, the tasks or any other setting. Only add the flags for your method.

Start every launch inside **tmux** so it keeps running when you disconnect:

```bash
tmux new -s runs        # detach with Ctrl-b then d; reattach with: tmux attach -t runs
```

## 4. Use these exact commands and names

The name (first argument) is the label the plotting script uses, so copy it exactly.

**Already done**

| Method | Command |
|---|---|
| Baseline | `scripts/run_humanoid.sh baseline` |
| Online | `scripts/run_humanoid.sh crossq agent.critic_target=online` |
| Online + 1 critic | `scripts/run_humanoid.sh crossq_1q agent.critic_target=online agent.critic_num_qs=1` |
| Online + 1 wide critic | `scripts/run_humanoid.sh crossq_1q_w360 agent.critic_target=online agent.critic_num_qs=1 agent.critic_hidden_dim=360` |

**Still to run (asymmetric critic):**

| Method | Command |
|---|---|
| Online + asym. critic | `scripts/run_humanoid.sh crossq_asym --asym agent.critic_target=online` |
| Online + 1 critic + asym. critic | `scripts/run_humanoid.sh crossq_1q_asym --asym agent.critic_target=online agent.critic_num_qs=1` |
| Online + 1 wide critic + asym. critic | `scripts/run_humanoid.sh crossq_1q_w360_asym --asym agent.critic_target=online agent.critic_num_qs=1 agent.critic_hidden_dim=360` |
| Asym. critic | `scripts/run_humanoid.sh baseline_asym --asym` |
| 1 critic + asym. critic | `scripts/run_humanoid.sh baseline_1q_asym --asym agent.critic_num_qs=1` |
| 1 wide critic + asym. critic | `scripts/run_humanoid.sh baseline_1q_w360_asym --asym agent.critic_num_qs=1 agent.critic_hidden_dim=360` |

What the pieces do:
- `--asym` turns on the asymmetric critic. The critic also sees noise-free observations, and the actor still sees only the noisy ones.
- The name doesn't turn anything on by itself. `baseline_asym` without `--asym` would just be the baseline.
- Without `agent.critic_target=online`, the run keeps the target network.

## 5. Asymmetric runs need more memory

`--asym` doubles the replay buffer. Estimated buffer size per task:

| Task | Buffer | Where it fits |
|---|---|---|
| Flat-H1 | ≈ 11 GB | GPU (g6e.2xlarge) |
| Flat-G1 | ≈ 20 GB | GPU (g6e.2xlarge) |
| Rough-H1 | ≈ 41 GB | too big for the GPU next to training |
| Rough-G1 | ≈ 50 GB | too big for the GPU and for a 2xlarge's RAM |

For the **rough** tasks:
- Use a **g6e.4xlarge**. It has the same GPU and 128 GB RAM.
- Add `agent.buffer_device_type=cpu` to the command.
- Learning is unchanged, but wall-clock time is slower, so don't compare those runs' speed with ours.

To run only some tasks, set `ENVS`. For example:

```bash
ENVS="Isaac-Velocity-Rough-G1-v0 Isaac-Velocity-Rough-H1-v0" \
  scripts/run_humanoid.sh crossq_asym --asym agent.critic_target=online agent.buffer_device_type=cpu
```

## 6. Smoke-test before the full runs

Each smoke test is a 2M-step run on Flat-G1, about 8 minutes:

```bash
scripts/run_smoke.sh smoke_crossq_asym smoke_1q_asym smoke_1q_w360_asym
```

The three target-network runs (`baseline_*_asym`) don't have smoke entries yet. One critic *with* a target network has also never run on a GPU. Smoke-test them directly:

```bash
SEEDS=0 ENVS=Isaac-Velocity-Flat-G1-v0 scripts/run_humanoid.sh smoke_baseline_1q_asym --asym agent.critic_num_qs=1 num_env_steps=2_000_896
```

## 7. Check that each run finished

A run is complete when the last line of its `metrics.csv` starts with **50000896**:

```bash
for f in logs/humanoid/*/*/*/metrics.csv; do echo "$(tail -1 $f | cut -d, -f1)  $f"; done
```

Runs occasionally crash at their first update with a `cudagraph` error. Just rerun that task and seed:

```bash
SEEDS="1000" ENVS="Isaac-Velocity-Rough-H1-v0" scripts/run_humanoid.sh <same name and flags as before>
```

Don't keep crashed runs. Delete their folder or set it aside.

## 8. Send back everything

We need the results *and* the trained policies, which are used for the videos. On the instance:

```bash
cd ~/16831FinalProject
{ git log -1; git status --short; nvidia-smi; free -g; } > provenance.txt
tar czf ~/<yourname>_results.tar.gz logs models runs provenance.txt ~/*.log
```

Copy that file to your laptop with `scp`. Unpack it into `results/<yourname>_full/`.

Then copy each complete run's `metrics.csv` and `config.yaml` into the plotting layout:

```
results/<yourname>/<method>/<task>/<run>/metrics.csv
results/<yourname>/<method>/<task>/<run>/config.yaml
# e.g. results/<yourname>/crossq_asym/Isaac-Velocity-Flat-G1-v0/seed0_1128_101500/
```

## 9. Plot

```bash
python scripts/plot_results.py --results_dir results/<yourname> results/nish
```

- **Returns and AUC** compare directly.
- **Speed:** compare `time/update_time` in `metrics.csv`, or runs from the same instance. Two instances of the same type can differ by up to about 10%.
