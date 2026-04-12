from .abstract_agent import AbstractAgent


class FixedScriptAgent(AbstractAgent):
    """
    Agent that follows a fixed script
    The evnironment should be reset before using this agent
    The script can be loaded from a file or passed as a list
    """

    def __init__(
        self,
        env,
        filename: str = None,
        script: list = [],
    ):
        super().__init__(env)

        if filename:
            file = open(filename, "r")
            self._actions_list = file.read().split("\n")
        else:
            self._actions_list = script

        if len(self._actions_list) == 0:
            raise Exception("Script is empty")

        self._current_action = 0

    # overriding abstract method
    def choose_action(self, state=None) -> int:
        """Ignore the state and return the next action in the script"""
        action = self._next_action()
        return action

    def _next_action(self) -> str:
        """Return the next action in the script."""

        if self._current_action >= len(self._actions_list):
            return "NOP"

        command = self._actions_list[self._current_action]
        self._current_action += 1

        return command

    def reset_script(self):
        """Reset the script to the beginning"""
        self._current_action = 0

    @property
    def length(self) -> int:
        return len(self._actions_list)

    @property
    def actions_list(self) -> list:
        return self._actions_list
