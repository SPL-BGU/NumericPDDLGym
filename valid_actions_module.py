from typing import Dict, Optional, Tuple, Union

import gymnasium as gym
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.utils.typing import TensorType

torch, nn = try_import_torch()


class ActionMaskingRLModule(RLModule):
    """
    A custom RLModule that supports action masking. This module modifies the observation
    space to include an action mask, which is used to restrict the available actions
    during inference, exploration, and training.
    """

    @override(RLModule)
    def __init__(
            self,
            *,
            observation_space: Optional[gym.Space] = None,
            action_space: Optional[gym.Space] = None,
            inference_only: Optional[bool] = None,
            learner_only: bool = False,
            model_config: Optional[Union[dict, DefaultModelConfig]] = None,
            catalog_class=None,
            **kwargs,
    ):
        # While the environment holds an observation space that contains, both,
        # the action mask and the original observation space, the 'RLModule'
        # receives only the `"observation"` element of the space, but not the
        # action mask.
        self.observation_space_with_mask = observation_space
        self.observation_space = observation_space["observations"]

        # Keeps track if observation specs have been checked already.
        self._checked_observations = False

        # The DefaultPPORLModule, in its constructor will build networks for the
        # original observation space (i.e. without the action mask).
        super().__init__(
            observation_space=self.observation_space,
            action_space=action_space,
            inference_only=inference_only,
            learner_only=learner_only,
            model_config=model_config,
            catalog_class=catalog_class,
            **kwargs,
        )


class ActionMaskingTorchRLModule(ActionMaskingRLModule, DefaultPPOTorchRLModule):
    @override(DefaultPPOTorchRLModule)
    def setup(self):
        super().setup()
        # We need to reset here the observation space such that the
        # super`s (`PPOTorchRLModule`) observation space is the
        # original space (i.e. without the action mask) and `self`'s
        # observation space contains the action mask.
        self.observation_space = self.observation_space_with_mask

    @override(DefaultPPOTorchRLModule)
    def _forward_inference(
            self, batch: Dict[str, TensorType], **kwargs
    ) -> Dict[str, TensorType]:
        # Preprocess the original batch to extract the action mask.
        action_mask, batch = self._preprocess_batch(batch)
        # Run the forward pass.
        outs = super()._forward_inference(batch, **kwargs)
        # Mask the action logits and return.
        return self._mask_action_logits(outs, action_mask)

    @override(DefaultPPOTorchRLModule)
    def _forward_exploration(
            self, batch: Dict[str, TensorType], **kwargs
    ) -> Dict[str, TensorType]:
        # Preprocess the original batch to extract the action mask.
        action_mask, batch = self._preprocess_batch(batch)
        # Run the forward pass.
        outs = super()._forward_exploration(batch, **kwargs)
        # Mask the action logits and return.
        return self._mask_action_logits(outs, action_mask)

    @override(DefaultPPOTorchRLModule)
    def _forward_train(
            self, batch: Dict[str, TensorType], **kwargs
    ) -> Dict[str, TensorType]:
        # Run the forward pass.
        outs = super()._forward_train(batch, **kwargs)
        # Mask the action logits and return.
        return self._mask_action_logits(outs, batch["action_mask"])

    @override(ValueFunctionAPI)
    def compute_values(self, batch: Dict[str, TensorType], embeddings=None):
        # Check, if the observations are still in `dict` form.
        if isinstance(batch[Columns.OBS], dict):
            # Preprocess the batch to extract the `observations` to `Columns.OBS`.
            action_mask, batch = self._preprocess_batch(batch)
            # NOTE: Because we manipulate the batch we need to add the `action_mask`
            # to the batch to access them in `_forward_train`.
            batch["action_mask"] = action_mask
        # Call the super's method to compute values for GAE.
        return super().compute_values(batch, embeddings)

    def _preprocess_batch(
            self, batch: Dict[str, TensorType], **kwargs
    ) -> Tuple[TensorType, Dict[str, TensorType]]:
        """Extracts observations and action mask from the batch

        Args:
            batch: A dictionary containing tensors (at least `Columns.OBS`)

        Returns:
            A tuple with the action mask tensor and the modified batch containing
                the original observations.
        """

        # Extract the available actions tensor from the observation.
        action_mask = batch[Columns.OBS].pop("action_mask")

        # Modify the batch for the `DefaultPPORLModule`'s `forward` method, i.e.
        # pass only `"obs"` into the `forward` method.
        batch[Columns.OBS] = batch[Columns.OBS].pop("observations")

        # Return the extracted action mask and the modified batch.
        return action_mask, batch

    def _mask_action_logits(
            self, batch: Dict[str, TensorType], action_mask: TensorType
    ) -> Dict[str, TensorType]:
        """Masks the action logits for the output of `forward` methods

        Args:
            batch: A dictionary containing tensors (at least action logits).
            action_mask: A tensor containing the action mask for the current
                observations.

        Returns:
            A modified batch with masked action logits for the action distribution
            inputs.
        """
        # Convert action mask into an `[0.0][-inf]`-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)

        # Mask the logits.
        batch[Columns.ACTION_DIST_INPUTS] += inf_mask

        # Return the batch with the masked action logits.
        return batch
