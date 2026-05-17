"""
Trajectory Collection Pipeline (Phase 3)
==========================================
Loads trained SAC experts and collects K trajectories for the imitation
learning algorithms (IQ-Learn, CSIL, CSIL+SOAR).

From Lecture 7, Slide 5 (IL setting):
    Given expert demonstrations D_E = {(s_i, a_i)} or trajectories
    → These are what Persons 2 and 3 will consume.

From the project spec (Applied Project 3):
    "Generate the expert dataset generating K trajectories rolling out π_1."

Usage:
    python collect_trajectories.py --env CartPole-v1 --model models/sac_cartpole_expert.pt --K 1 3 5 10 15
    python collect_trajectories.py --env Pendulum-v1 --model models/sac_pendulum_expert.pt --K 1 3 5 10 15

Output format (.npz):
    - states:      (total_steps, state_dim)
    - actions:     (total_steps, action_dim)  
    - next_states: (total_steps, state_dim)
    - rewards:     (total_steps,)
    - dones:       (total_steps,)
    - traj_starts: (K,)  -- index where each trajectory begins
    
    This format lets teammates easily iterate over individual trajectories
    or treat the whole dataset as a flat buffer of (s, a, s') tuples.
"""

import argparse
import os
import numpy as np
import gymnasium as gym
from sac_agent import SACAgent


def collect_trajectories(env_name, model_path, K, seed=42, deterministic=True):
    """
    Roll out a trained SAC policy for K episodes.
    
    Args:
        env_name: Gymnasium environment name
        model_path: Path to saved SAC model (.pt)
        K: Number of trajectories to collect
        seed: Base seed (each trajectory uses seed + k)
        deterministic: If True, use mean action (no sampling)
    
    Returns:
        data: dict with states, actions, next_states, rewards, dones, traj_starts
        episode_rewards: list of total rewards per trajectory
    """
    env = gym.make(env_name)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n if discrete else env.action_space.shape[0]

    # Load the trained agent
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        discrete=discrete,
        action_low=None if discrete else env.action_space.low,
        action_high=None if discrete else env.action_space.high,
        hidden_dim=256,  # will be overridden by loaded weights shape
    )
    
    # Try loading; handle potential hidden_dim mismatch
    try:
        agent.load(model_path)
    except RuntimeError:
        # Retry with smaller hidden dim (CartPole uses 128)
        agent = SACAgent(
            state_dim=state_dim, action_dim=action_dim, discrete=discrete,
            action_low=None if discrete else env.action_space.low,
            action_high=None if discrete else env.action_space.high,
            hidden_dim=128,
        )
        agent.load(model_path)

    # Collect trajectories
    all_states = []
    all_actions = []
    all_next_states = []
    all_rewards = []
    all_dones = []
    traj_starts = []
    episode_rewards = []

    step_idx = 0
    for k in range(K):
        traj_starts.append(step_idx)
        state, _ = env.reset(seed=seed + k)
        done = False
        ep_reward = 0

        while not done:
            action = agent.select_action(state, evaluate=deterministic)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            all_states.append(state)
            if discrete:
                all_actions.append([action])  # (1,) shape for consistency
            else:
                all_actions.append(action)
            all_next_states.append(next_state)
            all_rewards.append(reward)
            all_dones.append(float(done))

            state = next_state
            ep_reward += reward
            step_idx += 1

        episode_rewards.append(ep_reward)

    env.close()

    data = {
        "states": np.array(all_states, dtype=np.float32),
        "actions": np.array(all_actions, dtype=np.float32),
        "next_states": np.array(all_next_states, dtype=np.float32),
        "rewards": np.array(all_rewards, dtype=np.float32),
        "dones": np.array(all_dones, dtype=np.float32),
        "traj_starts": np.array(traj_starts, dtype=np.int64),
    }

    return data, episode_rewards


def save_trajectories(data, episode_rewards, save_path):
    """Save trajectory data as .npz file."""
    np.savez(
        save_path,
        **data,
        episode_rewards=np.array(episode_rewards, dtype=np.float32),
    )
    

def load_trajectories(path):
    """
    Load trajectory data from .npz file.
    
    Usage example for Persons 2 and 3:
        data = load_trajectories("expert_data/CartPole-v1_K5.npz")
        states = data["states"]           # (N, state_dim)
        actions = data["actions"]         # (N, action_dim)  
        next_states = data["next_states"] # (N, state_dim)
        
        # Iterate over individual trajectories:
        traj_starts = data["traj_starts"]
        for i in range(len(traj_starts)):
            start = traj_starts[i]
            end = traj_starts[i+1] if i+1 < len(traj_starts) else len(states)
            traj_states = states[start:end]
            traj_actions = actions[start:end]
    """
    loaded = np.load(path)
    return dict(loaded)


def print_data_summary(data, episode_rewards, env_name, K):
    """Print a summary of the collected data."""
    print(f"\n{'='*55}")
    print(f"  Dataset: {env_name} | K={K} trajectories")
    print(f"{'='*55}")
    print(f"  Total transitions:  {len(data['states'])}")
    print(f"  States shape:       {data['states'].shape}")
    print(f"  Actions shape:      {data['actions'].shape}")
    print(f"  Next states shape:  {data['next_states'].shape}")
    print(f"  Episode rewards:    {[f'{r:.1f}' for r in episode_rewards]}")
    print(f"  Mean reward:        {np.mean(episode_rewards):.1f}")
    print(f"  Std reward:         {np.std(episode_rewards):.1f}")
    print(f"  Traj lengths:       {np.diff(np.append(data['traj_starts'], len(data['states'])))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect expert trajectories")
    parser.add_argument("--env", type=str, required=True, help="Environment name")
    parser.add_argument("--model", type=str, required=True, help="Path to trained SAC model")
    parser.add_argument("--K", type=int, nargs="+", default=[1, 3, 5, 10, 15],
                        help="Number of trajectories to collect (multiple values)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--outdir", type=str, default="expert_data", help="Output directory")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy instead of deterministic")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    for k in args.K:
        print(f"\nCollecting K={k} trajectories from {args.env}...")
        data, rewards = collect_trajectories(
            env_name=args.env,
            model_path=args.model,
            K=k,
            seed=args.seed,
            deterministic=not args.stochastic,
        )

        # Save
        save_path = os.path.join(args.outdir, f"{args.env}_K{k}.npz")
        save_trajectories(data, rewards, save_path)
        print_data_summary(data, rewards, args.env, k)
        print(f"  Saved to: {save_path}")

    print(f"\n{'='*55}")
    print(f"All datasets saved to {args.outdir}/")
    print(f"{'='*55}")
