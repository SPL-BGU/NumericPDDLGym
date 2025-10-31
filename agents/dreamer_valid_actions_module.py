from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.algorithms.dreamerv3.torch.dreamerv3_torch_rl_module import DreamerV3TorchRLModule
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType


# ---------- Utilities ----------

def apply_action_mask_to_logits(logits: torch.Tensor, mask: torch.Tensor, eps: float = 1e-45) -> torch.Tensor:
    """Add log(mask) to logits so invalid actions become -inf."""
    mask = mask.clamp_min(eps)
    return logits + mask.log()


# ---------- Masked Dreamer RLModule ----------
# We extend RLlib's DreamerV3 Torch module by adding:
#   1) a small mask head g(h) -> logits over actions
#   2) BCE loss on real sequences vs. env-provided masks
#   3) masking the actor's logits both in REAL and IMAGINATION passes


class MaskedDreamerV3TorchRLModule(DreamerV3TorchRLModule):
    """DreamerV3 module with state-dependent action masking."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        A = self.action_dist_class.param_shape()[0]  # number of actions (Categorical(A))
        # The actor reads a latent "h" (RSSM deterministic state). Get its size from actor MLP input.
        # Actor is an nn.Sequential; first Linear gives us the in_features:
        first_linear: nn.Linear = None
        for m in self.actor.net.modules():
            if isinstance(m, nn.Linear):
                first_linear = m
                break

        latent_dim = first_linear.in_features if first_linear is not None else self.config_model.get("rssm_hidden_dim",
                                                                                                     1024)

        self.mask_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 256), nn.ELU(),
            nn.Linear(256, A),  # mask logits
        )

    # ---- Helpers to get masks in both paths ----
    def _predict_mask_probs(self, h: torch.Tensor) -> torch.Tensor:
        """Predict mask probabilities from latent h (used in imagination)."""
        return torch.sigmoid(self.mask_head(h))

    # ---- Actor calls (REAL + IMAG) ----
    @override(DreamerV3TorchRLModule)
    def _actor_forward_real(
            self,
            h: TensorType,
            obs: Dict[str, TensorType],
            is_exploring: bool,
            **kwargs
    ) -> Dict[str, TensorType]:
        """Actor on REAL data: mask comes from env."""
        out = super()._actor_forward_real(h, obs, is_exploring, **kwargs)

        # out contains "logits" for the Categorical policy (discrete).
        if "action_mask" in obs:
            mask = obs["action_mask"].float()
            out["logits"] = apply_action_mask_to_logits(out["logits"], mask)

        return out

    @override(DreamerV3TorchRLModule)
    def _actor_forward_imagination(
            self,
            h: TensorType,
            is_exploring: bool,
            **kwargs
    ) -> Dict[str, TensorType]:
        """Actor during imagination: mask is predicted from h."""
        out = super()._actor_forward_imagination(h, is_exploring, **kwargs)

        # Predict mask from latent state and apply a straight-through hardening.
        p = self._predict_mask_probs(h)  # [B, A]
        hard = (p > 0.5).float()
        mask = hard.detach() + (p - p.detach())  # straight-through estimator
        out["logits"] = apply_action_mask_to_logits(out["logits"], mask)

        # Expose mask probs for logging/diagnostics.
        out["mask_probs"] = p
        return out

    # ---- Loss: add BCE mask loss on REAL batches ----
    @override(DreamerV3TorchRLModule)
    def loss(self, batch: SampleBatch) -> Dict[str, TensorType]:
        losses = super().loss(batch)

        # Only compute mask loss on REAL sequences (not imagined).
        # RLlib Dreamer puts real data into "model_inputs" (encoder/RSSM posterior) and imagined into rollout path.
        if isinstance(batch.get("obs"), dict) and "action_mask" in batch["obs"]:
            # We need the corresponding latent h used by the actor on real data.
            # RLlib stores deterministic RSSM states under key "h" (as produced in _encoder/_rssm).
            h_real: Optional[torch.Tensor] = batch.get("h")
            if h_real is not None:
                mask_logits = self.mask_head(h_real)  # [B, A]
                target_mask = batch["obs"]["action_mask"].float()  # [B, A]
                # Balance the BCE because illegal actions may be sparse.
                pos_weight = None
                if "mask_pos_weight" in self.config_model:
                    pos_weight = torch.tensor(self.config_model["mask_pos_weight"], device=mask_logits.device)
                loss_mask = F.binary_cross_entropy_with_logits(mask_logits, target_mask, pos_weight=pos_weight)
                losses["loss_mask"] = loss_mask
                losses["total_loss"] = losses["total_loss"] + loss_mask

        return losses
