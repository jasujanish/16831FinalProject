import unittest

import torch

from flash_rl.agents.flashSAC.network import FlashSACActor, FlashSACDoubleCritic


class TestArchitectureVariants(unittest.TestCase):
    def test_variants_preserve_network_shapes(self) -> None:
        batch_size = 8
        observation_dim = 5
        action_dim = 3

        for block_type in ("relu", "linear_bottleneck", "swiglu"):
            with self.subTest(block_type=block_type):
                actor = FlashSACActor(
                    num_blocks=1,
                    input_dim=observation_dim,
                    hidden_dim=12,
                    action_dim=action_dim,
                    block_type=block_type,
                )
                observations = torch.randn(batch_size, observation_dim)
                actions, actor_info = actor(observations, training=True)
                self.assertEqual(actions.shape, (batch_size, action_dim))
                self.assertEqual(actor_info["log_prob"].shape, (batch_size,))
                self.assertTrue(torch.isfinite(actions).all().item())

                critic = FlashSACDoubleCritic(
                    num_blocks=1,
                    input_dim=observation_dim + action_dim,
                    hidden_dim=16,
                    num_bins=11,
                    min_v=-5.0,
                    max_v=5.0,
                    block_type=block_type,
                )
                values, critic_info = critic(observations, actions, training=True)
                self.assertEqual(values.shape, (2, batch_size))
                self.assertEqual(critic_info["log_prob"].shape, (2, batch_size, 11))
                self.assertTrue(torch.isfinite(values).all().item())
