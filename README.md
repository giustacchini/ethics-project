# Introduction to Reinforcement Learning Course - Final Project

**Dr. Teddy Lazebnik**  
**1.10.2024**

## Abstract

The final project aims to apply reinforcement learning techniques to a real-world or simulated problem. Students will independently conceive, design, implement, and evaluate a reinforcement learning solution. The project should demonstrate a deep understanding of the course material and the ability to apply it creatively.

## Project Requirements

### Project Selection

Students are expected to propose a unique and challenging reinforcement learning project that aligns with the course content and demonstrates their ability to apply the learned concepts.

### Project Topic Selection

- **Relevance:** The project should be relevant to the field of reinforcement learning and have potential real-world applications.
- **Originality:** Students are encouraged to propose original project ideas or explore novel approaches to existing problems.
- **Feasibility:** The project should be achievable within the given timeframe and resources.

### Problem Formulation

**Clear Definition:** The reinforcement learning problem should be clearly defined, including:

- **States:** The set of all possible states the agent can be in.
- **Actions:** The set of actions the agent can take in each state.
- **Rewards:** The feedback signal that the agent receives for its actions.
- **Environment Dynamics:** The rules governing the transitions between states.

**Objective:** The project should have a well-defined goal or objective for the agent to achieve.

### Data and Environment

- **Data Availability:** If the project involves real-world data, students should ensure its availability and accessibility.
- **Environment Creation:** If a simulated environment is required, students should consider using existing platforms or building their own.
- **Environment Complexity:** The chosen environment should provide sufficient challenges for the reinforcement learning agent.

### Project Scope

The project scope should demonstrate a comprehensive understanding of reinforcement learning concepts and their application to a real-world or simulated problem.

## Depth of Exploration

- **Algorithm Selection:** Students should justify their choice of reinforcement learning algorithm(s) based on the problem characteristics and state-of-the-art research.
- **Hyperparameter Tuning:** A thorough exploration of hyperparameter space is expected to optimize agent performance.
- **Evaluation Metrics:** Students should define appropriate metrics to assess the agent’s performance and compare different approaches.

## Innovation and Creativity

- **Novel Approaches:** Students are encouraged to explore innovative techniques or combinations of methods.
- **Theoretical Contributions:** If applicable, students can contribute to the theoretical understanding of reinforcement learning.
- **Practical Impact:** The project should demonstrate the potential practical impact of the developed solution.

## Project Deliverables

The final project deliverables should comprehensively document the project process, results, and code.

### Written Report

- **Project Overview:** Clear and concise description of the project goals, motivation, and scope.
- **Problem Formulation:** Detailed explanation of the reinforcement learning problem, including state space, action space, reward function, and environment dynamics.
- **Methodology:** Description of the chosen reinforcement learning algorithm(s), implementation details, and hyperparameter tuning process.
- **Results:** Presentation of experimental results, including quantitative and qualitative analysis.
- **Discussion:** Analysis of the results, comparison to existing work, and identification of limitations.
- **Conclusion:** Summary of the project’s findings and contributions.

### Code

- **Clean and Organized Code:** Well-structured, commented, and readable code implementation of the reinforcement learning agent.
- **Reproducibility:** Clear instructions on how to run the code and reproduce the results.
- **Version Control:** Use of version control (e.g., Git) to manage code changes.

### Presentation

- **Clear Communication:** Effective presentation of the project’s key findings and insights - 15 minutes.
- **Visual Aids:** Use of slides, demos, or other visual aids to enhance understanding - your call.

## Evaluation Criteria

The final project will be evaluated based on the following criteria:

- **Problem Formulation:** The clarity and depth of the problem definition, including state space, action space, and reward function.
- **Algorithm Selection:** The appropriateness of the chosen reinforcement learning algorithm(s) for the problem.
- **Implementation Quality:** The correctness, efficiency, and maintainability of the code implementation.

## Additional Comments

Students must adhere to academic integrity guidelines. Plagiarism and collusion will not be tolerated.

If working in groups, all members must contribute to the project. Groups of up to 4 students are allowed. Please note that larger groups are expected to produce “larger” projects and reports.

The reports are expected to be between 8 and 20 pages long (not including the reference list) with 1.5 line spacing, Arial 12 font, one-inch margin.

## Development

Use Python 3.12 and the project virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
pytest
```

`pytest.ini` adds `src` to the import path. Tests require no custom
`PYTHONPATH`, exclusions, or graphical display. Use the virtual environment's
`pytest`; a system-wide installation may use a different Python without the
project dependencies. Coverage is optional: `pytest --cov=src`.

Run the visual demo with `python src/main.py --seed 0`. Run reproducible
headless matches with:

```bash
PYTHONPATH=src python -m soccer.validate_matches --seeds 5 --output /tmp/soccer-validation.json
```

Source organization:

- `src/enums.py`: action and event enums.
- `src/shared/`: domain-independent movement stages, grids, events and rewards.
- `src/soccer/config.py`: match configuration.
- `src/soccer/env.py`: match state, observations and transition orchestration.
- `src/soccer/ball_rules.py`: kicks, possession and dives.
- `src/soccer/discipline.py`: tackles, suspensions and restarts.
- `src/soccer/agent.py`, `ball.py`, `grid.py`: domain objects and rendering.
- `src/soccer/validate_matches.py`: complete-match invariants and replay checks.
- `tests/unit/shared/` and `tests/unit/soccer/`: tests organized by domain.

The ball and discipline classes supply rule methods directly to `SoccerEnv`;
they do not keep separate copies of match state or add delegation wrappers.
`SoccerEnvConfig` remains importable from `soccer.env` for existing callers.
