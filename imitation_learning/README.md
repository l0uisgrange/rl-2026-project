# EE-568 Project 4: Imitation Learning

Compare imitation learning algorithms (IQ-Learn, CSIL, CSIL+SOAR) using expert demonstrations on OpenAI Gym environments.

## Setup

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── sac_agent.py              # SAC implementation (discrete + continuous)
├── collect_trajectories.py   # Expert trajectory collection
├── evaluation.py             # Logging utilities + plot generation
├── models/                   # Trained SAC experts (.pt)
├── expert_data/              # Expert trajectory datasets (.npz)
├── logs/                     # IL algorithm training logs (.csv)
└── plots/                    # Generated evaluation plots
```

## Stage 1: Train Expert Policies

Train SAC agents to optimality on both environments. Use fixed α=0.2 for stable training.

```python
from sac_agent import train_sac

# CartPole (target: 500 reward)
train_sac("CartPole-v1", max_episodes=300, save_path="models/sac_cartpole_expert.pt",
          hidden_dim=128, batch_size=128)

# Pendulum (target: ~-200 reward)
train_sac("Pendulum-v1", max_episodes=200, save_path="models/sac_pendulum_expert.pt",
          hidden_dim=256, batch_size=256)
```

## Stage 2: Collect Expert Trajectories

Generate datasets of K expert trajectories for each environment.

```bash
python collect_trajectories.py --env CartPole-v1 --model models/sac_cartpole_expert.pt --K 1 3 5 10 15
python collect_trajectories.py --env Pendulum-v1 --model models/sac_pendulum_expert.pt --K 1 3 5 10 15
```

Output: `.npz` files in `expert_data/` containing `states`, `actions`, `next_states`, `rewards`, `dones`, and `traj_starts`.

### Loading expert data

```python
data = np.load("expert_data/CartPole-v1_K5.npz")
states, actions, next_states = data["states"], data["actions"], data["next_states"]

# Random mini-batch for training
idx = np.random.choice(len(states), size=256)
batch_s, batch_a = torch.FloatTensor(states[idx]), torch.FloatTensor(actions[idx])

# Iterate individual trajectories
for i in range(len(data["traj_starts"])):
    start = data["traj_starts"][i]
    end = data["traj_starts"][i+1] if i+1 < len(data["traj_starts"]) else len(states)
    traj = states[start:end]
```

## Stage 3: Implement IL Algorithms

Implement IQ-Learn, CSIL, and CSIL+SOAR. The SAC architecture in `sac_agent.py` (`QNetwork`, `GaussianPolicy`, `DiscretePolicy`) can be reused as the backbone for CSIL and SOAR.

### Logging results

Use the standardized logger so all results feed into the same plotting pipeline:

```python
from evaluation import TrainingLogger

logger = TrainingLogger(algorithm="iqlearn", env_name="CartPole-v1", seed=42, K=5)

for step in range(total_steps):
    # ... training ...
    if step % eval_interval == 0:
        logger.log(step, eval_reward)

logger.save("logs/")
```

Run each algorithm for every combination of seeds `[42, 43, 44]`, K values `[1, 3, 5, 10, 15]`, and both environments.

## Stage 4: Generate Evaluation Plots

After all algorithms have logged their results:

```bash
python evaluation.py --logdir logs/ --outdir plots/
```

This produces two plot types (averaged across 3+ seeds):
- **Sample efficiency**: final reward vs K (number of expert trajectories)
- **Learning curves**: reward vs training steps at fixed K=5

To test the pipeline with synthetic data: `python evaluation.py --demo`

## Environment Reference

| Environment  | Obs dim | Action dim | Action type        | Expert target |
|:------------|:--------|:-----------|:-------------------|:-------------|
| CartPole-v1 | 4       | 2          | Discrete           | 500          |
| Pendulum-v1 | 3       | 1          | Continuous [-2, 2] | ≈ -200       |

## Hyperparameters

| Parameter    | CartPole | Pendulum | Notes                              |
|:------------|:---------|:---------|:-----------------------------------|
| hidden_dim  | 128      | 256      | MLP hidden layer size              |
| lr          | 3e-4     | 3e-4     | Adam learning rate                 |
| batch_size  | 128      | 256      | Replay buffer sample size          |
| gamma       | 0.99     | 0.99     | Discount factor                    |
| tau         | 0.005    | 0.005    | Target network Polyak rate         |
| alpha       | 0.2      | 0.2      | Entropy coefficient (fixed)        |
| start_steps | 500      | 1,000    | Random exploration before training |