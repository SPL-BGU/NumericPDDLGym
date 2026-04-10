import csv

from rl_agents.logging_callbacks import (
    LogAlgorithmActions,
    STATISTICS_COL_NAMES,
)

# IMPORTANT: import the module itself so we can patch the global
import rl_agents.logging_callbacks as logging_callbacks


# -----------------------
# Mock helpers
# -----------------------


class DummyEpisode:
    def __init__(self):
        self.id_ = 42
        self.rewards = [0, 1, 2.5]

        # Mimic RLlib structure
        self.infos = type("Infos", (), {})()
        self.infos.data = [
            {},  # index 0 (unused)
            {
                "problem_name": "test_problem",
                "domain_name": "test_domain",
                "executing_algorithm": "Random",
                "executed_action": "a1",
                "is_inapplicable": False,
                "previous_state": "(s0)",
                "next_state": "(s1)",
            },
            {
                "executed_action": "a2",
                "is_inapplicable": True,
                "previous_state": "(s1)",
                "next_state": "(s2)",
            },
        ]


class DummyEnv:
    def __init__(self):
        self.single_action_space = type("ActionSpace", (), {"n": 5})()


# -----------------------
# Tests
# -----------------------


def test_logger_writes_csv(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIRECTORY_PATH", str(tmp_path))
    logging_callbacks.OUTPUT_DIRECTORY_PATH = str(tmp_path)

    logger = LogAlgorithmActions(save_traces=False)

    episode = DummyEpisode()
    env = DummyEnv()

    logger.on_episode_start(episode=episode)
    logger.on_episode_end(
        episode=episode,
        env=env,
        env_index=0,
        metrics_logger=None,
    )

    csv_path = tmp_path / "episode_summary.csv"
    assert csv_path.exists()

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    row = rows[0]

    assert set(row.keys()) == set(STATISTICS_COL_NAMES)
    assert row["domain_name"] == "test_domain"
    assert row["trained_instance"] == "test_problem"
    assert row["executing_algorithm"] == "Random"
    assert int(row["num_failed_actions"]) == 1
    assert int(row["num_successful_actions"]) == 1


def test_logger_appends_multiple_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIRECTORY_PATH", str(tmp_path))
    logging_callbacks.OUTPUT_DIRECTORY_PATH = str(tmp_path)

    logger = LogAlgorithmActions(save_traces=False)

    episode = DummyEpisode()
    env = DummyEnv()

    for _ in range(2):
        logger.on_episode_start(episode=episode)
        logger.on_episode_end(
            episode=episode,
            env=env,
            env_index=0,
            metrics_logger=None,
        )

    csv_path = tmp_path / "episode_summary.csv"
    assert csv_path.exists()

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2


def test_logger_saves_traces(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIRECTORY_PATH", str(tmp_path))
    logging_callbacks.OUTPUT_DIRECTORY_PATH = str(tmp_path)

    logger = LogAlgorithmActions(save_traces=True)

    episode = DummyEpisode()
    env = DummyEnv()

    logger.on_episode_start(episode=episode)
    logger.on_episode_end(
        episode=episode,
        env=env,
        env_index=0,
        metrics_logger=None,
    )

    traces_dir = tmp_path / "traces"
    assert traces_dir.exists()

    trace_files = list(traces_dir.glob("*.trajectory"))
    assert len(trace_files) == 1

    content = trace_files[0].read_text()

    assert "(operator:" in content
    assert "(:transition_status" in content


def test_logger_no_traces_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIRECTORY_PATH", str(tmp_path))
    logging_callbacks.OUTPUT_DIRECTORY_PATH = str(tmp_path)

    logger = LogAlgorithmActions(save_traces=False)

    episode = DummyEpisode()
    env = DummyEnv()

    logger.on_episode_start(episode=episode)
    logger.on_episode_end(
        episode=episode,
        env=env,
        env_index=0,
        metrics_logger=None,
    )

    traces_dir = tmp_path / "traces"
    assert not traces_dir.exists()
