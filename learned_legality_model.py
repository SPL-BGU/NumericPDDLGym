# learned_legality_module_new.py
from typing import Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleConfig
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.utils.typing import TensorType




class LegalityPPOTorchModule(TorchRLModule):
    """
    Outputs:
      - action_dist_inputs: logits for Categorical(policy)
      - vf: value estimates
      - leg_logits: per-action legality logits (used for aux loss)

    Softly discourages predicted-invalid actions by subtracting
      alpha * (1 - sigmoid(leg_logits)) from the policy logits.
    """
    framework: str = "torch"

    def __init__(self, config):
        super().__init__(config)
        C = config
        obs_dim = C.observation_dim
        act_dim = C.action_dim

        # Backbone
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in C.hidden:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        self.backbone = nn.Sequential(*layers)

        # Heads
        self.pi_head   = nn.Linear(in_dim, act_dim)   # policy logits
        self.leg_head  = nn.Linear(in_dim, act_dim)   # legality logits
        self.v_head    = nn.Sequential(nn.Linear(in_dim, 128), nn.Tanh(), nn.Linear(128, 1))

        self.alpha = 2.0  # default penalty scale

    # ------- Helpers to unify the three forward passes -------
    def _forward_impl(self, batch: Mapping[str, TensorType]) -> Dict[str, TensorType]:
        # Expect flat Box observations under Columns.OBS
        x = batch[Columns.OBS].float()
        z = self.backbone(x)

        pi_logits  = self.pi_head(z)                  # [B, A]
        leg_logits = self.leg_head(z)                 # [B, A]
        p_valid    = torch.sigmoid(leg_logits)        # (0,1)

        # Soft penalty (no masks; exploration still possible)
        penalized_logits = pi_logits - self.alpha * (1.0 - p_valid)

        vf = self.v_head(z).squeeze(-1)               # [B]

        return {
            "action_dist_inputs": penalized_logits,
            "vf": vf,
            "leg_logits": leg_logits,                 # for learner aux loss
        }

    # New-stack required entry points:
    def _forward(self, batch: Mapping[str, TensorType], **kwargs) -> Dict[str, TensorType]:
        return self._forward_impl(batch)


