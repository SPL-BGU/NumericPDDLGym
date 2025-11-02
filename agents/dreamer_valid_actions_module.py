"""RLlib DreamerV3 Torch module with support for action masking."""

from typing import Any, Dict, Tuple

import gymnasium as gym
import torch

from ray.rllib.algorithms.dreamerv3.torch.dreamerv3_torch_rl_module import (
    DreamerV3TorchRLModule,
)
from ray.rllib.core.columns import Columns
from ray.rllib.utils.annotations import override
from ray.rllib.utils.torch_utils import FLOAT_MIN


class ActionMaskingDreamerV3TorchRLModule(DreamerV3TorchRLModule):
    """DreamerV3 Torch RLModule that honours action masks from the environment."""

    @override(DreamerV3TorchRLModule)
    def _forward_inference(
        self, batch: Dict[str, Any], **kwargs: Any
    ) -> Dict[str, Any]:
        action_mask, batch = self._preprocess_batch(batch)
        actions, next_state = self._masked_actor_step(
            batch=batch,
            action_mask=action_mask,
            deterministic=True,
        )
        return self._forward_inference_or_exploration_helper(batch, actions, next_state)

    @override(DreamerV3TorchRLModule)
    def _forward_exploration(
        self, batch: Dict[str, Any], **kwargs: Any
    ) -> Dict[str, Any]:
        action_mask, batch = self._preprocess_batch(batch)
        actions, next_state = self._masked_actor_step(
            batch=batch,
            action_mask=action_mask,
            deterministic=False,
        )
        return self._forward_inference_or_exploration_helper(batch, actions, next_state)

    @override(DreamerV3TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs: Any):
        # Remove the action mask from the batch before calling Dreamer's training
        # routine. Only the raw observation is expected by the default implementation.
        _, batch = self._preprocess_batch(batch)
        return super()._forward_train(batch, **kwargs)

    def _preprocess_batch(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor | None, Dict[str, Any]]:
        """Extract the action mask from the observation dict, if present."""

        observations = batch[Columns.OBS]
        if isinstance(observations, dict):
            action_mask = observations.pop("action_mask")
            batch[Columns.OBS] = observations.pop("observations")
            return action_mask, batch
        return None, batch

    def _masked_actor_step(
        self,
        *,
        batch: Dict[str, Any],
        action_mask: torch.Tensor | None,
        deterministic: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run the Dreamer actor while respecting the provided action mask."""

        with torch.no_grad():
            states = self.dreamer_model.world_model.forward_inference(
                observations=batch[Columns.OBS],
                previous_states=batch[Columns.STATE_IN],
                is_first=batch["is_first"],
            )

            actions, distr_params = self.dreamer_model.actor(
                h=states["h"],
                z=states["z"],
                return_distr_params=True,
            )

            if (
                action_mask is not None
                and isinstance(self.action_space, gym.spaces.Discrete)
            ):
                masked_params = self._apply_action_mask(
                    distr_params=distr_params,
                    action_mask=action_mask,
                )
                distr = self.dreamer_model.actor.get_action_dist_object(masked_params)
                actions = distr.mode if deterministic else distr.sample()
            else:
                distr = self.dreamer_model.actor.get_action_dist_object(distr_params)
                actions = distr.mode if deterministic else actions

            next_state = {"h": states["h"], "z": states["z"], "a": actions}

        return actions, next_state

    def _apply_action_mask(
        self, *, distr_params: torch.Tensor, action_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mask invalid actions by setting their logits to negative infinity."""

        # Align the mask with the distribution parameters (BxT flattened).
        mask = action_mask.reshape(-1, action_mask.shape[-1]).to(distr_params.device)
        mask = mask.to(distr_params.dtype)

        inf_mask = torch.clamp(torch.log(mask), min=FLOAT_MIN)
        return distr_params + inf_mask

