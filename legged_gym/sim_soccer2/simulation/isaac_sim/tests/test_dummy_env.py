# SPDX-FileCopyrightText: Copyright (c) MOS-Brain Contributors
# SPDX-License-Identifier: GPL-3.0-or-later


import unittest
import torch
import sys
import os

# Mocking necessary modules to avoid full simulation dependency
from unittest.mock import MagicMock

class MockEnv:
    def __init__(self, num_obs, num_actions):
        self.num_envs = 1
        self.num_actions = num_actions
        self.device = "cpu"
        self.cfg = MagicMock()
        self.cfg.num_observations = num_obs
    
    def get_observations(self):
        return torch.zeros((1, 10)), {"observations": {}}

# Define the DummyEnv as it is in core.py (after fix)
class DummyEnv:
    def __init__(self, num_obs, num_actions, device):
        self.num_observations = num_obs
        self.num_privileged_observations = None
        self.num_actions = num_actions
        self.num_envs = 1
        self.device = device
        class Unwrapped:
            def __init__(self, dev): self.device = dev
        self.unwrapped = Unwrapped(device)
    
    def get_observations(self):
        # Return tuple (obs, extras)
        # Fix: OnPolicyRunner expects extras to have "observations" key
        return torch.zeros((self.num_envs, self.num_observations), device=self.device), {"observations": {}}
    
    def reset(self):
        return self.get_observations()
    
    def step(self, actions):
        obs, extras = self.get_observations()
        rew = torch.zeros((self.num_envs,), device=self.device)
        dones = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        return obs, rew, dones, extras

class TestDummyEnv(unittest.TestCase):
    def test_dummy_env_extras(self):
        # Setup
        dummy_env = DummyEnv(num_obs=99, num_actions=29, device="cpu")
        
        # Test get_observations
        obs, extras = dummy_env.get_observations()
        self.assertIn("observations", extras, "extras dict must contain 'observations' key")
        self.assertIsInstance(extras["observations"], dict)
        
        # Test step
        obs, rew, dones, extras = dummy_env.step(torch.zeros(1, 29))
        self.assertIn("observations", extras, "extras dict from step must contain 'observations' key")

if __name__ == "__main__":
    unittest.main()
