import argparse
import torch
import gymnasium as gym
from csil import CSILAgent, ENV_CONFIGS
from iq_learn import IQLearnAgent

parser = argparse.ArgumentParser()
parser.add_argument("--env",  default="CartPole", choices=["CartPole", "Pendulum"])
parser.add_argument("--algo", default="csil",     choices=["iqlearn", "csil", "csilsoar"])
parser.add_argument("--K",    type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

env_name   = args.env + "-v1"
model_path = f"../models/{args.algo}_{args.env}_K{args.K}_seed{args.seed}.pt"
cfg        = ENV_CONFIGS[args.env]

env_tmp    = gym.make(env_name)
discrete   = isinstance(env_tmp.action_space, gym.spaces.Discrete)
state_dim  = env_tmp.observation_space.shape[0]
action_dim = env_tmp.action_space.n if discrete else env_tmp.action_space.shape[0]
action_low  = None if discrete else env_tmp.action_space.low
action_high = None if discrete else env_tmp.action_space.high
env_tmp.close()

ckpt = torch.load(model_path, map_location="cpu")

if args.algo == "iqlearn":
    agent = IQLearnAgent(
        state_dim=state_dim, action_dim=action_dim, discrete=discrete,
        action_low=action_low, action_high=action_high,
        hidden_dim=cfg["hidden_dim"], lr=cfg["lr"], alpha=cfg["alpha"],
    )
    agent.critic.load_state_dict(ckpt["critic"])
    agent.critic_target.load_state_dict(ckpt["critic"])
    agent.actor.load_state_dict(ckpt["actor"])
else:
    agent = CSILAgent(
        state_dim=state_dim, action_dim=action_dim, discrete=discrete,
        action_low=action_low, action_high=action_high,
        hidden_dim=cfg["hidden_dim"], lr=cfg["lr"], alpha=cfg["alpha"],
        soar=args.algo == "csilsoar",
    )
    agent.critic.load_state_dict(ckpt["critic"])
    agent.critic_target.load_state_dict(ckpt["critic"])
    agent.actor.load_state_dict(ckpt["actor"])
    agent.bc_policy.load_state_dict(ckpt["bc"])

print(f"Loaded {model_path}")

env = gym.make(env_name, render_mode="human")
ep = 0
try:
    while True:
        state, _ = env.reset()
        done, total_reward, steps = False, 0.0, 0
        while not done:
            action = agent.select_action(state, evaluate=True)
            state, reward, term, trunc, _ = env.step(action)
            total_reward += reward
            done = term or trunc
            steps += 1
        ep += 1
        print(f"ep {ep:3d}  steps {steps:4d}  reward = {total_reward:.1f}")
finally:
    env.close()