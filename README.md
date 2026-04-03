<h1 align="center">Numeric PDDLGym</h1>
<p align="center">
<a href="https://www.python.org/downloads/release/python-31212/"><img alt="Python Version" src="https://img.shields.io/badge/python-3.12-blue"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

A framework for automatically translating Numeric PDDL domains with the [Gymnasium API](https://gymnasium.farama.org/index.html).
This allows Numeric PDDL planning problems to be solved using standard RL algorithms (e.g., PPO via RLlib).

The environment converts states, actions, and goals from Numeric PDDL into fixed-size numeric vectors, enabling direct integration with deep RL libraries.


## ✨ Features

- Supports **Numeric PDDL domains and problems**
- Automatic grounding and vectorization of:
  - Predicates
  - Numeric fluents
  - Goal conditions
- Compatible with the **Gymnasium API**
- Designed for **[RLlib (Ray)](https://docs.ray.io/en/latest/rllib/index.html)** integration
- Works with standard deep RL algorithms (e.g., PPO)

## ⚙️ Configuration

The environment is configured via a dictionary (`env_config`).

| Parameter | Type | Description |
|----------|------|------------|
| `domain_path` | `Path` | Path to the PDDL domain file |
| `problems_list` | `List[Path]` | List of problem files |
| `max_steps` | `int` | Max steps per episode |
| `executing_algorithm` | `str` | Name of RL algorithm (e.g., `"PPO"`) |
| `masking_strategy` | `"pre"` or `"post"` | `"pre"`: Filters invalid actions before execution using PDDL checks.<br>`"post"`: Learns invalid actions after execution from feedback. |
| `map_size` | `int` | (MinecraftEnv only) Grid size, e.g., 6, 10, 15. |

## 🚀 Installation

```
pip install numeric_pddl_gym
```

With RL (RLlib + Torch):

```
pip install numeric_pddl_gym[rl_agents]
```

## ▶️ Run Example

```
python rl_agents/ppo_pddl_rllib_agent.py
```

## ⚠️ Limitations

1. Can't encode complex goal conditions.
2. Agents must be retrained if the problem has a different number of fluents, predicates, or goal conditions.
3. Currently does not support manual interaction with the environment.
4. Designed for fixed-structure problems (no variable-sized domains).
5. Enabling action applicability checking leads to slow runtime in large-scale problems.


## 📚 Citations

Coming soon