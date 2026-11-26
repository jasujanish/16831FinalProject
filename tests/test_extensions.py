"""
CPU tests for the project extensions (SWD / LaBER sampling, adaptive actor delay, critic targets, single critic,
in-batch surprise weighting, logging).

    python -m pytest tests/test_extensions.py
"""

import csv
import os
from dataclasses import dataclass, field
from types import SimpleNamespace

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
from flash_rl.agents.flashSAC.network import FlashSACDoubleCritic
from flash_rl.agents.flashSAC.update import CRITIC_TARGETS, _select_target_log_probs, surprise_weights
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


def make_agent(*overrides: str, env_info=None):
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
    agent = create_agent(OBS_SPACE, ACTION_SPACE, dict(env_info or {}), cfg.agent)
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


@pytest.mark.parametrize("sampler", ["uniform", "swd", "laber"])
@pytest.mark.parametrize("actor_delay", ["fixed", "ema"])
def test_agent_update(sampler: str, actor_delay: str) -> None:
    agent = make_agent(
        f"agent.sampler={sampler}",
        f"agent.actor_delay={actor_delay}",
        "agent.actor_delay_ema_interval=5",
        "agent.swd_decay_steps=20",
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


def test_select_target_log_probs() -> None:
    torch.manual_seed(0)
    num_bins, batch_size = 11, 512
    bins = torch.linspace(-1, 1, num_bins)

    def random_critics():
        log_probs = torch.log_softmax(3 * torch.randn(2, batch_size, num_bins), dim=-1)
        return (log_probs.exp() * bins).sum(-1), log_probs  # Q = mean of the distribution

    online_qs, online_lp = random_critics()
    target_qs, target_lp = random_critics()

    def q_of(log_probs):
        return (log_probs.exp() * bins).sum(-1)

    selected = {}
    for mode in CRITIC_TARGETS:
        log_probs, online_chosen = _select_target_log_probs(mode, online_qs, online_lp, target_qs, target_lp)
        assert log_probs.shape == (batch_size, num_bins)
        assert torch.allclose(log_probs.exp().sum(-1), torch.ones(batch_size), atol=1e-5)
        assert (online_chosen is not None) == mode.startswith("minto")
        selected[mode] = q_of(log_probs)

    assert torch.allclose(selected["target"], target_qs.min(0).values, atol=1e-5)
    assert torch.allclose(selected["online"], online_qs.min(0).values, atol=1e-5)
    # MINTO: never above the target-network estimate, and lower wherever an online critic is lower
    assert torch.allclose(selected["minto"], torch.minimum(target_qs.min(0).values, online_qs.min(0).values), atol=1e-5)
    assert (selected["minto"] <= selected["target"] + 1e-6).all()
    assert torch.allclose(selected["minto_net"], torch.minimum(target_qs.mean(0), online_qs.mean(0)), atol=1e-5)
    # online-chosen flags match the comparison
    _, online_chosen = _select_target_log_probs("minto_net", online_qs, online_lp, target_qs, target_lp)
    assert torch.equal(online_chosen, online_qs.mean(0) < target_qs.mean(0))
    # online mode does not need the target network
    log_probs, _ = _select_target_log_probs("online", online_qs, online_lp, None, None)
    assert torch.allclose(q_of(log_probs), selected["online"])


def test_critic_targets_agree_when_target_equals_online() -> None:
    # at initialization the target critic is a copy of the critic, so target / online / minto yield the
    # same update. minto_net does not: it averages the 2 critics instead of taking their minimum.
    losses = {}
    for mode in CRITIC_TARGETS:
        torch.manual_seed(0)
        agent = make_agent(f"agent.critic_target={mode}")
        losses[mode] = agent.update()["critic/loss"]
    for mode in ("online", "minto"):
        assert losses[mode] == pytest.approx(losses["target"], rel=1e-6), losses
    assert losses["minto_net"] != pytest.approx(losses["target"], rel=1e-6), losses


@pytest.mark.parametrize("critic_target", CRITIC_TARGETS)
def test_critic_target_update(critic_target: str) -> None:
    agent = make_agent(f"agent.critic_target={critic_target}")
    target_before = [p.detach().clone() for p in agent._target_critic.network.parameters()]
    infos = [agent.update() for _ in range(20)]
    for info in infos:
        assert all(np.isfinite(v) for v in info.values()), info
    info = infos[-1]
    assert ("critic_target/gap" in info) == (critic_target != "online")
    assert ("critic_target/online_frac" in info) == critic_target.startswith("minto")
    if critic_target.startswith("minto"):
        assert 0.0 <= info["critic_target/online_frac"] <= 1.0
    # the target network is frozen (and unused) without Polyak updates, and moves otherwise
    target_after = list(agent._target_critic.network.parameters())
    unchanged = all(torch.equal(a, b) for a, b in zip(target_before, target_after))
    assert unchanged == (critic_target == "online")


def test_online_checkpoint_saves_current_critic_as_target(tmp_path) -> None:
    agent = make_agent("agent.critic_target=online")
    for _ in range(5):
        agent.update()
    agent.save(str(tmp_path))
    saved_critic = torch.load(tmp_path / "critic.pt")["network_state_dict"]
    saved_target = torch.load(tmp_path / "target_critic.pt")["network_state_dict"]
    assert saved_critic.keys() == saved_target.keys()
    assert all(torch.equal(saved_critic[k], saved_target[k]) for k in saved_critic)


def test_laber_requires_target_network() -> None:
    with pytest.raises(AssertionError, match="LaBER"):
        make_agent("agent.sampler=laber", "agent.critic_target=minto")


def num_params(network) -> int:
    return sum(p.numel() for p in network.parameters())


@pytest.mark.parametrize("critic_target", ["online", "target"])
def test_single_critic(critic_target: str) -> None:
    pair = make_agent(f"agent.critic_target={critic_target}")
    single = make_agent(f"agent.critic_target={critic_target}", "agent.critic_num_qs=1")
    assert single._critic.network.num_qs == 1 and single._target_critic.network.num_qs == 1
    # one critic of the same shape has half the parameters of the pair
    assert num_params(single._critic.network) * 2 == num_params(pair._critic.network)
    infos = [single.update() for _ in range(20)]
    for info in infos:
        assert all(np.isfinite(v) for v in info.values()), info
    assert sum("actor/loss" in i for i in infos) == 10


def test_single_wider_critic_matches_pair_size() -> None:
    # Run 2: one critic at width 360 has about as many parameters as the default pair at width 256
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose("flashSAC_base")
    critic_input_dim = 310 + 37  # Rough-G1: observation + action
    pair = FlashSACDoubleCritic(cfg.agent.critic_num_blocks, critic_input_dim, 256, 101, -5.0, 5.0, num_qs=2)
    single = FlashSACDoubleCritic(cfg.agent.critic_num_blocks, critic_input_dim, 360, 101, -5.0, 5.0, num_qs=1)
    assert num_params(single) == pytest.approx(num_params(pair), rel=0.05)


def test_surprise_weights() -> None:
    torch.manual_seed(0)
    surprise = torch.rand(4096).exp()
    w = surprise_weights(surprise, alpha=0.5, w_min=0.5, w_max=3.0)
    assert w.shape == surprise.shape
    assert float(w.mean()) == pytest.approx(1.0, abs=1e-5)
    # monotone in surprise, and close to the clip range (the renormalization shifts it slightly)
    order = surprise.argsort()
    assert (w[order].diff() >= -1e-6).all()
    assert 0.4 < float(w.min()) and float(w.max()) < 3.5
    # alpha = 0 gives uniform weights; alpha = 1 with a wide clip range gives weights proportional to surprise
    assert torch.allclose(surprise_weights(surprise, 0.0, 0.5, 3.0), torch.ones(4096))
    w1 = surprise_weights(surprise, 1.0, 1e-6, 1e6)
    assert torch.allclose(w1, surprise / surprise.mean(), atol=1e-5)


def test_surprise_weighting_only_changes_the_loss() -> None:
    # same seed and batch: identical forward pass and errors, different loss
    infos = {}
    for alpha in (0.0, 0.5):
        torch.manual_seed(0)
        agent = make_agent("agent.critic_target=online", f"agent.critic_surprise_weight_alpha={alpha}")
        torch.manual_seed(1)
        infos[alpha] = agent.update()
    off, on = infos[0.0], infos[0.5]
    assert on["critic/td_abs"] == pytest.approx(off["critic/td_abs"], rel=1e-6)
    assert on["critic/surprise"] == pytest.approx(off["critic/surprise"], rel=1e-6)
    assert on["critic/loss"] != pytest.approx(off["critic/loss"], rel=1e-6)
    assert "critic/surprise_weight_ess" not in off
    assert 0.0 < on["critic/surprise_weight_ess"] <= 1.0
    assert 0.4 < on["critic/surprise_weight_min"] <= 1.0 <= on["critic/surprise_weight_max"] < 3.5


@pytest.mark.parametrize("num_qs, hidden_dim", [(2, 256), (1, 360)])
def test_surprise_weighting_runs(num_qs: int, hidden_dim: int) -> None:
    # Runs 3 and 4: surprise weighting on the default pair and on one wider critic, without a target network
    agent = make_agent(
        "agent.critic_target=online",
        "agent.critic_surprise_weight_alpha=0.5",
        f"agent.critic_num_qs={num_qs}",
        f"agent.critic_hidden_dim={hidden_dim}",
    )
    infos = [agent.update() for _ in range(20)]
    for info in infos:
        assert all(np.isfinite(v) for v in info.values()), info
    assert "critic/surprise_weight_age_corr" in infos[-1]


def test_surprise_weighting_combines_with_laber() -> None:
    agent = make_agent("agent.sampler=laber", "agent.critic_surprise_weight_alpha=0.5")
    infos = [agent.update() for _ in range(10)]
    for info in infos:
        assert all(np.isfinite(v) for v in info.values()), info


def test_invalid_critic_options() -> None:
    with pytest.raises(AssertionError, match="critic_num_qs"):
        make_agent("agent.critic_num_qs=0")
    with pytest.raises(AssertionError):
        make_agent("agent.critic_surprise_weight_min=2.0")


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


def test_privileged_critic_group() -> None:
    """The critic group is a noise-free copy of the policy group; the policy group is left untouched."""
    from flash_rl.envs.isaaclab import add_privileged_critic_group  # imports no IsaacLab module at import time

    @dataclass
    class Term:
        noise: object = "noise"

    @dataclass
    class Group:
        enable_corruption: bool = True
        concatenate_terms: bool = True
        joint_vel: Term = field(default_factory=Term)
        height_scan: object = None  # flat tasks remove this term

    env_cfg = SimpleNamespace(observations=SimpleNamespace(policy=Group()))
    add_privileged_critic_group(env_cfg)
    policy, critic = env_cfg.observations.policy, env_cfg.observations.critic
    assert policy.enable_corruption and not critic.enable_corruption
    assert critic is not policy and critic.joint_vel is not policy.joint_vel
    assert critic.height_scan is None and critic.concatenate_terms
    with pytest.raises(ValueError):
        add_privileged_critic_group(env_cfg)


def test_asymmetric_actor_ignores_critic_observation() -> None:
    """With asymmetric observations the actor uses only the first `actor_observation_size` entries."""
    actor_dim = 3
    agent = make_agent("agent.asymmetric_observation=true", env_info={"actor_observation_size": (actor_dim,)})
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(NUM_ENVS, OBS_DIM)).astype(np.float32)
    other = obs.copy()
    other[:, actor_dim:] = rng.normal(size=(NUM_ENVS, OBS_DIM - actor_dim))
    a1 = agent.sample_actions(0, {"next_observation": obs}, training=False)
    a2 = agent.sample_actions(0, {"next_observation": other}, training=False)
    np.testing.assert_allclose(a1, a2)
    other[:, :actor_dim] += 1.0
    a3 = agent.sample_actions(0, {"next_observation": other}, training=False)
    assert not np.allclose(a1, a3)
    infos = [agent.update() for _ in range(6)]
    assert all(np.isfinite(v) for i in infos for v in i.values())


def test_privileged_env_requires_asymmetric_agent() -> None:
    """Without the asymmetric flag the actor would read the privileged observations: refuse to build the agent."""
    with pytest.raises(ValueError, match="asymmetric_observation"):
        make_agent(env_info={"actor_observation_size": (3,), "asymmetric_obs": True})
