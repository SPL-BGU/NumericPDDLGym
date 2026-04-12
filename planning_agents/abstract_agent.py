from abc import ABC, abstractmethod


class AbstractAgent(ABC):
    """Abstract base class for Gym agents."""

    def __init__(self, env):
        self.env = env

    @abstractmethod
    def choose_action(self, state) -> int:
        """Choose an action based on the state."""
        raise NotImplementedError

    def predict(self, observations, state, episode_start, deterministic) -> tuple:
        """Wrapper for gym predict function."""
        return [self.choose_action(observations[0])], None
