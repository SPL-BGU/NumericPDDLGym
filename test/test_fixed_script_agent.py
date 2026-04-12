import pytest

from planning_agents import FixedScriptAgent


# =========================================================
# Scripts
# =========================================================

COUNTERS_SCRIPT = [
    "(increase_rate c2)",
    "(increase_rate c1)",
    "(increase_rate c2)",
    "(increase_rate c1)",
    "(increase_rate c2)",
    "(increase_rate c2)",
    "(increment c2)",
    "(increment c2)",
    "(increment c2)",
    "(increment c2)",
    "(increase_rate c1)",
    "(increment c1)",
    "(increment c2)",
    "(increment c1)",
]

POGO_SCRIPT = [
    "(tp_to cell21 cell35)",
    "(craft_plank)",
    "(craft_plank)",
    "(tp_to cell35 cell22)",
    "(craft_tree_tap cell22)",
    "(tp_to crafting_table cell22)",
    "(place_tree_tap cell22)",
    "(craft_wooden_pogo cell22)",
]


# =========================================================
# Unit tests (agent only)
# =========================================================


def test_init_with_script_list():
    agent = FixedScriptAgent(env=None, script=COUNTERS_SCRIPT)

    assert agent.length == len(COUNTERS_SCRIPT)
    assert agent.actions_list == COUNTERS_SCRIPT


def test_init_with_file(tmp_path):
    script_file = tmp_path / "script.txt"
    script_file.write_text("\n".join(POGO_SCRIPT))

    agent = FixedScriptAgent(env=None, filename=str(script_file))

    assert agent.length == len(POGO_SCRIPT)
    assert agent.actions_list == POGO_SCRIPT


def test_empty_script_raises():
    with pytest.raises(Exception):
        FixedScriptAgent(env=None, script=[])


def test_choose_action_sequence():
    agent = FixedScriptAgent(env=None, script=POGO_SCRIPT)

    for expected in POGO_SCRIPT:
        assert agent.choose_action() == expected


def test_returns_nop_when_script_finished():
    agent = FixedScriptAgent(env=None, script=["a", "b"])

    assert agent.choose_action() == "a"
    assert agent.choose_action() == "b"
    assert agent.choose_action() == "NOP"
    assert agent.choose_action() == "NOP"


def test_reset_script():
    agent = FixedScriptAgent(env=None, script=["a", "b"])

    agent.choose_action()
    agent.choose_action()
    agent.reset_script()

    assert agent.choose_action() == "a"
    assert agent.choose_action() == "b"


def test_internal_pointer_progression():
    agent = FixedScriptAgent(env=None, script=["a", "b", "c"])

    assert agent._current_action == 0
    agent.choose_action()
    assert agent._current_action == 1
    agent.choose_action()
    assert agent._current_action == 2


def test_realistic_counters_script_execution():
    agent = FixedScriptAgent(env=None, script=COUNTERS_SCRIPT)

    output = [agent.choose_action() for _ in range(len(COUNTERS_SCRIPT))]
    assert output == COUNTERS_SCRIPT


def test_realistic_pogo_script_execution():
    agent = FixedScriptAgent(env=None, script=POGO_SCRIPT)

    output = [agent.choose_action() for _ in range(len(POGO_SCRIPT))]
    assert output == POGO_SCRIPT


# =========================================================
# Integration helper
# =========================================================


class ScriptRunner:
    """
    Runs a FixedScriptAgent inside a real env until termination.
    """

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def run(self, max_steps=200):
        state, info = self.env.reset()

        terminated = False
        truncated = False
        total_reward = 0

        for _ in range(max_steps):
            pddl_action = self.agent.choose_action(state)
            gym_action = self.env.get_action_from_planning(pddl_action)

            state, reward, terminated, truncated, info = self.env.step(gym_action)
            total_reward += reward

            if terminated or truncated:
                break

        return {
            "terminated": terminated,
            "truncated": truncated,
            "reward": total_reward,
            "info": info,
        }


# =========================================================
# Integration tests (deterministic envs)
# =========================================================


def test_counters_script_runs(single_counters_env):
    agent = FixedScriptAgent(single_counters_env, script=COUNTERS_SCRIPT)
    runner = ScriptRunner(single_counters_env, agent)

    result = runner.run()

    assert result["terminated"] is True
    assert result["truncated"] is False
    assert result["reward"] == 1


def test_pogo_script_runs(single_pogo_env):
    agent = FixedScriptAgent(single_pogo_env, script=POGO_SCRIPT)
    runner = ScriptRunner(single_pogo_env, agent)

    result = runner.run()

    assert result["terminated"] is True
    assert result["truncated"] is False
    assert result["reward"] == 1


def test_execution_does_not_crash(single_pogo_env):
    agent = FixedScriptAgent(single_pogo_env, script=POGO_SCRIPT)
    runner = ScriptRunner(single_pogo_env, agent)

    result = runner.run()

    assert result["reward"] is not None
    assert isinstance(result["reward"], (int, float))


def test_deterministic_execution(single_pogo_env):
    agent1 = FixedScriptAgent(single_pogo_env, script=POGO_SCRIPT)
    agent2 = FixedScriptAgent(single_pogo_env, script=POGO_SCRIPT)

    r1 = ScriptRunner(single_pogo_env, agent1).run()
    r2 = ScriptRunner(single_pogo_env, agent2).run()

    assert r1["reward"] == r2["reward"]
    assert r1["terminated"] == r2["terminated"]
