"""
CPU tests for the project extensions (SWD / LaBER sampling, adaptive actor delay, logging).

    python -m pytest tests/test_extensions.py
"""

import csv
import os

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")  # no compiler needed for the tests

import gymnasium as gym
import hydra
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from flash_rl.agents import create_agent
from flash_rl.agents.flashSAC.actor_delay import ActorDelay
from flash_rl.agents.flashSAC.layer import EnsembleUnitBatchNorm
from flash_rl.buffers.torch_buffer import TorchUniformBuffer
from flash_rl.common.logger import CsvTrainerLogger

OBS_DIM, ACTION_DIM, NUM_ENVS = 5, 2, 8
OBS_SPACE = gym.spaces.Box(-np.inf, np.inf, (NUM_ENVS, OBS_DIM), np.float32)
ACTION_SPACE = gym.spaces.Box(-1.0, 1.0, (NUM_ENVS, ACTION_DIM), np.float32)

if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", lambda s: eval(s))


def make_transition(step: int) -> dict[str, np.ndarray]:
    """A transition whose observation stores the interaction step it was added at."""
    rng = np.random.default_rng(step)
    return {
        "observation": np.full((NUM_ENVS, OBS_DIM), step, dtype=np.float32),
        "action": rng.uniform(-1, 1, (NUM_ENVS, ACTION_DIM)).astype(np.float32),
        "reward": rng.normal(size=NUM_ENVS).astype(np.float32),
        "terminated": rng.random(NUM_ENVS) < 0.05,
        "truncated": np.zeros(NUM_ENVS, dtype=bool),
        "next_observation": rng.normal(size=(NUM_ENVS, OBS_DIM)).astype(np.float32),
    }


def make_buffer(max_length: int, **kwargs) -> TorchUniformBuffer:
    return TorchUniformBuffer(
        observation_space=OBS_SPACE,
        action_space=ACTION_SPACE,
        n_step=1,
        gamma=0.99,
        max_length=max_length,
        min_length=1,
        sample_batch_size=64,
        device_type="cpu",
        **kwargs,
    )


def make_agent(*overrides: str):
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(
            "flashSAC_base",
            overrides=[
                "agent.device_type=cpu",
                "agent.buffer_device_type=cpu",
                "agent.use_compile=false",
                "agent.use_amp=false",
                "agent.buffer_max_length=4000",
                "agent.buffer_min_length=100",
                "agent.sample_batch_size=64",
                *overrides,
            ],
        )
    OmegaConf.resolve(cfg)
    agent = create_agent(OBS_SPACE, ACTION_SPACE, {}, cfg.agent)
    for step in range(100):
        agent.process_transition(make_transition(step))
    return agent


# max_length 100 is not a multiple of NUM_ENVS: the oldest step is partially overwritten after wrap-around
@pytest.mark.parametrize("max_length, num_steps", [(1000, 50), (96, 40), (100, 40)])
def test_buffer_age(max_length: int, num_steps: int) -> None:
    buffer = make_buffer(max_length)
    for step in range(num_steps):
        buffer.add(make_transition(step))
    batch = buffer.sample(batch_size=5000)
    added_at = batch["observation"][:, 0]
    assert torch.equal(batch["age"], (num_steps - 1) - added_at)


@pytest.mark.parametrize("max_length, num_steps", [(1000, 50), (96, 40), (100, 40)])
def test_swd_distribution(max_length: int, num_steps: int) -> None:
    torch.manual_seed(0)
    decay_steps, min_weight = 8, 0.1
    buffer = make_buffer(max_length, sampler="swd", swd_decay_steps=decay_steps, swd_min_weight=min_weight)
    for step in range(num_steps):
        buffer.add(make_transition(step))

    # expected: every stored transition weighted by max(min_weight, 1 - age / decay_steps), newest age = 1
    stored_age = (num_steps - 1) - buffer.get_observations()[:, 0] + 1
    weights = torch.clamp(1 - stored_age / decay_steps, min=min_weight)
    max_age = int(stored_age.max())
    expected = torch.zeros(max_age + 1).index_add_(0, stored_age.long(), weights)
    expected = expected / expected.sum()

    ages = buffer.sample(batch_size=400_000)["age"] + 1
    assert ages.max() <= max_age
    observed = torch.bincount(ages.long(), minlength=max_age + 1) / len(ages)
    assert torch.allclose(observed, expected, atol=3e-3)


def make_window_buffer(max_length: int, start: int, end: int, grow: tuple[int, int]) -> TorchUniformBuffer:
    return make_buffer(
        max_length,
        sampler="window",
        window_start_size=start,
        window_end_size=end,
        window_grow_start=grow[0],
        window_grow_end=grow[1],
    )


# NUM_ENVS = 8 transitions per step. 100 is not a multiple of 8; 60 steps wrap both buffers
@pytest.mark.parametrize("max_length", [100, 160])
@pytest.mark.parametrize("window", [8, 24, 37, 96])
def test_window_distribution(max_length: int, window: int) -> None:
    torch.manual_seed(0)
    buffer = make_window_buffer(max_length, window, window, (0, 0))
    for step in range(60):
        buffer.add(make_transition(step))
    batch = buffer.sample(batch_size=200_000)
    assert torch.equal(batch["age"], 59 - batch["observation"][:, 0])

    # uniform over exactly the newest `window` transitions: full steps of 8, plus a part of the oldest step
    counts = torch.bincount(batch["age"].long()).float() / 200_000
    steps = -(-window // NUM_ENVS)
    expected = torch.tensor([min(NUM_ENVS, window - NUM_ENVS * i) / window for i in range(steps)])
    assert torch.allclose(counts, expected, atol=5e-3)


def test_window_schedule() -> None:
    buffer = make_window_buffer(800, 80, 400, (800, 1600))
    lengths, oldest = [], []
    for step in range(300):
        buffer.add(make_transition(step))
        lengths.append(buffer.window_length)
        oldest.append(buffer._num_added - buffer.window_length)  # index of the oldest transition in the window
    added = [8 * (i + 1) for i in range(300)]
    for n, length in zip(added, lengths):
        if n < 800:
            assert length == min(n, 80)
        elif n >= 1600:
            assert length == 400
        else:
            assert length == int(80 + (n - 800) / 800 * 320)
    # growth rate 0.4 < 1: a transition that has left the window never returns
    assert all(b >= a for a, b in zip(oldest, oldest[1:]))


def test_window_limits() -> None:
    # never longer than what is stored, and other samplers draw from the whole buffer
    buffer = make_window_buffer(800, 400, 800, (0, 0))
    buffer.add(make_transition(0))
    assert buffer.window_length == NUM_ENVS
    uniform = make_buffer(800)
    uniform.add(make_transition(0))
    assert uniform.window_length == len(uniform) == NUM_ENVS
    for start, end, grow in [(0, 80, (0, 0)), (160, 80, (0, 0)), (80, 801, (0, 0)), (80, 160, (10, 5))]:
        with pytest.raises(AssertionError):
            make_window_buffer(800, start, end, grow)


def test_window_save_load(tmp_path) -> None:
    # the schedule counts every transition ever added, not the number stored
    buffer = make_window_buffer(160, 16, 160, (400, 1200))
    for step in range(100):
        buffer.add(make_transition(step))
    path = str(tmp_path / "buffer" / "replay_buffer.pt")
    buffer.save(path)
    loaded = make_window_buffer(160, 16, 160, (400, 1200))
    loaded.load(path)
    assert loaded.window_length == buffer.window_length == int(16 + 0.5 * 144)


def test_fixed_actor_delay_matches_modulo() -> None:
    for period in (1, 2, 4, 8):
        delay = ActorDelay("fixed", period, interval=10, beta=0.9, scale=2.0, min_delay=1.0, max_delay=8.0)
        for step in range(50):
            assert delay.should_update_actor(step) == (step % period == 0)
            delay.observe(1.0)
        assert delay.delay == period


def test_ema_actor_delay() -> None:
    delay = ActorDelay("ema", 2, interval=10, beta=0.9, scale=2.0, min_delay=1.0, max_delay=8.0)

    def run(td_error: float, num_steps: int) -> float:
        updates = 0
        for step in range(num_steps):
            updates += delay.should_update_actor(step)
            delay.observe(td_error)
        return updates / num_steps

    # constant error: ratio 1, so the delay stays at `scale` (FlashSAC's 2)
    assert run(1.0, 1000) == pytest.approx(0.5)
    assert delay.delay == pytest.approx(2.0)
    # error spike: the actor slows down, bounded by max_delay
    run(100.0, 10)
    assert delay.delay == 8.0
    assert run(100.0, 10) <= 0.2
    # error collapse: the actor speeds up, bounded by min_delay
    run(1e-3, 10)
    assert delay.delay == 1.0
    # the state survives a checkpoint
    restored = ActorDelay("ema", 2, interval=10, beta=0.9, scale=2.0, min_delay=1.0, max_delay=8.0)
    restored.load_state_dict(delay.state_dict())
    assert restored.state_dict() == delay.state_dict()


@pytest.mark.parametrize("sampler", ["uniform", "swd", "laber", "window"])
@pytest.mark.parametrize("actor_delay", ["fixed", "ema"])
def test_agent_update(sampler: str, actor_delay: str) -> None:
    agent = make_agent(
        f"agent.sampler={sampler}",
        f"agent.actor_delay={actor_delay}",
        "agent.actor_delay_ema_interval=5",
        "agent.swd_decay_steps=20",
        "agent.window_start_size=200",
        "agent.window_end_size=400",
        "agent.window_grow_start=100",
        "agent.window_grow_end=10_000",
    )
    infos = [agent.update() for _ in range(30)]
    for info in infos:
        assert all(np.isfinite(v) for v in info.values()), info
    info = infos[-1]
    for key in ("critic/td_abs", "critic/surprise", "replay/age_mean", "replay/age_surprise_corr", "actor_delay/delay"):
        assert key in info
    assert ("laber/effective_sample_frac" in info) == (sampler == "laber")
    assert ("actor_delay/td_ema" in info) == (actor_delay == "ema")
    if actor_delay == "fixed":
        assert sum("actor/loss" in i for i in infos) == 15


def test_laber_selection() -> None:
    torch.manual_seed(0)
    agent = make_agent("agent.sampler=laber", "agent.laber_multiplier=8")
    candidates = agent._replay_buffer.sample(batch_size=512)
    candidates["actor_observation"] = candidates["observation"]
    candidates["actor_next_observation"] = candidates["next_observation"]
    batch, info = agent._select_laber_batch(dict(candidates))

    assert batch["observation"].shape == (64, OBS_DIM)
    assert batch["weight"].shape == (64,)
    assert 0.0 < float(info["laber/effective_sample_frac"]) <= 1.0
    assert 0.0 < info["laber/unique_frac"] <= 1.0

    # importance weights have mean 1 in expectation, i.e. the weighted critic loss is unbiased
    mean_weights = [float(agent._select_laber_batch(dict(candidates))[0]["weight"].mean()) for _ in range(300)]
    assert np.mean(mean_weights) == pytest.approx(1.0, abs=0.02)


def test_weighted_batchnorm_recovers_uniform_statistics() -> None:
    torch.manual_seed(0)
    norm = EnsembleUnitBatchNorm(num_ensemble=2, input_dim=3)
    population = torch.randn(2, 20_000, 3).exp()  # skewed, so that prioritizing large values shifts the statistics
    priorities = population[0].norm(dim=-1)
    idxs = torch.multinomial(priorities, 4096, replacement=True)
    batch, weights = population[:, idxs], priorities.mean() / priorities[idxs]

    def normalized_population_mean(sample_weights):
        norm.running_mean.zero_()
        norm.momentum = 1.0  # running_mean becomes the batch mean used for normalization
        norm(batch, training=True, sample_weights=sample_weights)
        return (norm.running_mean - population.mean(dim=1)).abs().max()

    assert normalized_population_mean(None) > 0.5  # selection bias
    assert normalized_population_mean(weights) < 0.1  # corrected
    # uniform weights reproduce the unweighted layer
    out = norm(batch, training=True)
    assert torch.allclose(norm(batch, training=True, sample_weights=torch.ones(4096)), out, atol=1e-4)


def test_buffer_save_load(tmp_path) -> None:
    buffer = make_buffer(1000, sampler="swd")
    for step in range(50):
        buffer.add(make_transition(step))
    path = str(tmp_path / "buffer" / "replay_buffer.pt")
    buffer.save(path)

    loaded = make_buffer(1000, sampler="swd")
    loaded.load(path)
    batch = loaded.sample(batch_size=2000)
    assert torch.equal(batch["age"], 49 - batch["observation"][:, 0])


def test_actor_lr_schedule() -> None:
    used = {}
    for schedule in ("actor_step", "global_step"):
        agent = make_agent(
            f"agent.actor_lr_schedule={schedule}",
            "agent.actor_update_period=4",
            "agent.learning_rate_decay_step=40",
            "agent.learning_rate_warmup_step=0",
        )
        # record the learning rate each optimizer actually steps with
        lrs: dict[str, list[float]] = {"actor": [], "critic": []}
        for name, network in (("actor", agent._actor), ("critic", agent._critic)):
            optimizer = network.optimizer

            def step(*args, _name=name, _optimizer=optimizer, _step=optimizer.step, **kwargs):
                lrs[_name].append(_optimizer.param_groups[0]["lr"])
                return _step(*args, **kwargs)

            optimizer.step = step
        for _ in range(37):
            agent.update()
        used[schedule] = lrs

    # global_step: every actor update uses exactly the critic's learning rate of that update
    assert used["global_step"]["actor"] == pytest.approx(used["global_step"]["critic"][::4], rel=1e-9)
    # actor_step (FlashSAC): the actor's schedule advances 4x slower than the critic's
    assert used["actor_step"]["actor"] == pytest.approx(used["actor_step"]["critic"][:10], rel=1e-9)


def test_csv_logger(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    class DummyLogger:
        def __init__(self) -> None:
            self.metrics: dict[str, float] = {}

        @property
        def average_meter_dict(self):
            return self

        def averages(self) -> dict[str, float]:
            return dict(self.metrics)

        def update_metric(self, **kwargs) -> None:
            self.metrics.update(kwargs)

        def log_metric(self, step: int) -> None:
            pass

        def reset(self) -> None:
            self.metrics = {}

    cfg = OmegaConf.create({"group_name": "g", "exp_name": "e", "seed": 0, "env": {"env_name": "Env"}})
    logger = CsvTrainerLogger(cfg, DummyLogger())
    for step, metrics in enumerate([{"a": 1.0}, {"a": 2.0}, {"a": 3.0, "b": 4.0}, {"b": 5.0}]):
        logger.update_metric(**metrics)
        logger.log_metric(step)
        logger.reset()

    with open(os.path.join(logger.log_dir, "metrics.csv")) as f:
        rows = list(csv.DictReader(f))
    assert [r["env_step"] for r in rows] == ["0", "1", "2", "3"]
    assert [r["a"] for r in rows] == ["1.0", "2.0", "3.0", ""]
    assert [r["b"] for r in rows] == ["", "", "4.0", "5.0"]
    assert os.path.exists(os.path.join(logger.log_dir, "config.yaml"))
