"""
Evaluation & Plotting Engine (Phase 4)
=======================================
Generates the two required plot types from the project spec:

1. Sample Efficiency: Reward vs K (number of expert trajectories)
2. Learning Curves:   Reward vs training steps (fixed K)

Each plot averages across 3+ random seeds as required by the project.

Usage:
    # After Persons 2 & 3 have logged their results:
    python evaluation.py --logdir logs/

Log format (CSV):
    Each algorithm saves a CSV with columns:
        seed, step, eval_reward, K
    
    Example filename: logs/iqlearn_CartPole-v1.csv
    Example row:      42, 5000, 350.0, 5
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# Logging utilities (for Persons 2 & 3 to use)
# ============================================================

class TrainingLogger:
    """
    Standardized logger for IL algorithms.
    
    Usage:
        logger = TrainingLogger("iqlearn", "CartPole-v1", seed=42, K=5)
        
        for step in range(num_steps):
            # ... training ...
            if step % eval_interval == 0:
                eval_reward = evaluate_policy(agent, env)
                logger.log(step, eval_reward)
        
        logger.save("logs/")
    """
    def __init__(self, algorithm, env_name, seed, K):
        self.algorithm = algorithm
        self.env_name = env_name
        self.seed = seed
        self.K = K
        self.entries = []  # list of (step, eval_reward)
    
    def log(self, step, eval_reward):
        """Log an evaluation result."""
        self.entries.append((step, eval_reward))
    
    def save(self, logdir):
        """Save as CSV."""
        os.makedirs(logdir, exist_ok=True)
        filename = f"{self.algorithm}_{self.env_name}_K{self.K}_seed{self.seed}.csv"
        filepath = os.path.join(logdir, filename)
        
        with open(filepath, "w") as f:
            f.write("seed,step,eval_reward,K\n")
            for step, reward in self.entries:
                f.write(f"{self.seed},{step},{reward},{self.K}\n")
        
        print(f"  Log saved: {filepath}")
        return filepath


def load_logs(logdir, algorithm, env_name, K=None):
    """
    Load all log CSVs for a given algorithm and environment.
    
    Returns:
        data: dict mapping K -> {seed -> [(step, reward), ...]}
    """
    data = {}
    pattern = f"{algorithm}_{env_name}"
    
    for f in sorted(Path(logdir).glob(f"{pattern}*.csv")):
        with open(f) as fh:
            lines = fh.readlines()[1:]  # skip header
        
        for line in lines:
            parts = line.strip().split(",")
            seed = int(parts[0])
            step = int(parts[1])
            reward = float(parts[2])
            k = int(parts[3])
            
            if K is not None and k != K:
                continue
            
            if k not in data:
                data[k] = {}
            if seed not in data[k]:
                data[k][seed] = []
            data[k][seed].append((step, reward))
    
    return data


# ============================================================
# Plot 1: Sample Efficiency (Reward vs K)
# ============================================================

def plot_sample_efficiency(logdir, env_name, algorithms, K_values=None,
                           save_path=None):
    """
    Plot final average reward vs number of expert trajectories K.
    
    This directly answers: "Does SOAR improve sample efficiency over
    base CSIL when expert data is scarce?" (from project plan)
    
    Args:
        logdir: Directory containing CSV logs
        env_name: Environment name
        algorithms: List of algorithm names (e.g., ["iqlearn", "csil", "csil_soar"])
        K_values: List of K values to plot (default: [1, 3, 5, 10, 15])
        save_path: Where to save the figure
    """
    if K_values is None:
        K_values = [1, 3, 5, 10, 15]
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0"]
    
    for idx, algo in enumerate(algorithms):
        data = load_logs(logdir, algo, env_name)
        
        means = []
        stds = []
        valid_Ks = []
        
        for k in K_values:
            if k not in data:
                continue
            # Take the final eval reward from each seed
            final_rewards = []
            for seed, entries in data[k].items():
                if entries:
                    final_rewards.append(entries[-1][1])  # last logged reward
            
            if final_rewards:
                valid_Ks.append(k)
                means.append(np.mean(final_rewards))
                stds.append(np.std(final_rewards))
        
        if valid_Ks:
            means = np.array(means)
            stds = np.array(stds)
            color = colors[idx % len(colors)]
            ax.plot(valid_Ks, means, "o-", label=algo.upper(), color=color, linewidth=2)
            ax.fill_between(valid_Ks, means - stds, means + stds, alpha=0.2, color=color)
    
    ax.set_xlabel("Number of Expert Trajectories (K)", fontsize=12)
    ax.set_ylabel("Final Average Reward", fontsize=12)
    ax.set_title(f"Sample Efficiency — {env_name}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    
    plt.close(fig)
    return fig


# ============================================================
# Plot 2: Learning Curves (Reward vs Training Steps)
# ============================================================

def plot_learning_curves(logdir, env_name, algorithms, K,
                         smooth_window=5, save_path=None):
    """
    Plot learning curves (eval reward vs training steps) for fixed K.
    
    Shows how quickly each algorithm converges.
    
    Args:
        logdir: Directory containing CSV logs
        env_name: Environment name
        algorithms: List of algorithm names
        K: Fixed number of expert trajectories
        smooth_window: Moving average window for smoothing
        save_path: Where to save the figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0"]
    
    for idx, algo in enumerate(algorithms):
        data = load_logs(logdir, algo, env_name, K=K)
        
        if K not in data:
            continue
        
        # Collect all seeds' learning curves
        all_steps = set()
        for seed, entries in data[K].items():
            for step, _ in entries:
                all_steps.add(step)
        all_steps = sorted(all_steps)
        
        if not all_steps:
            continue
        
        # Interpolate each seed to common step grid
        seed_curves = []
        for seed, entries in data[K].items():
            steps = [e[0] for e in entries]
            rewards = [e[1] for e in entries]
            interpolated = np.interp(all_steps, steps, rewards)
            seed_curves.append(interpolated)
        
        seed_curves = np.array(seed_curves)
        mean_curve = np.mean(seed_curves, axis=0)
        std_curve = np.std(seed_curves, axis=0)
        
        # Smooth
        if smooth_window > 1 and len(mean_curve) >= smooth_window:
            kernel = np.ones(smooth_window) / smooth_window
            mean_curve = np.convolve(mean_curve, kernel, mode="valid")
            std_curve = np.convolve(std_curve, kernel, mode="valid")
            all_steps = all_steps[:len(mean_curve)]
        
        color = colors[idx % len(colors)]
        ax.plot(all_steps, mean_curve, label=algo.upper(), color=color, linewidth=2)
        ax.fill_between(all_steps, mean_curve - std_curve, mean_curve + std_curve,
                        alpha=0.2, color=color)
    
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Average Eval Reward", fontsize=12)
    ax.set_title(f"Learning Curves — {env_name} (K={K})", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    
    plt.close(fig)
    return fig


# ============================================================
# Generate all plots
# ============================================================

def generate_all_plots(logdir, outdir="plots"):
    """Generate all required plots for the report."""
    os.makedirs(outdir, exist_ok=True)
    algorithms = ["iqlearn", "csil", "csil_soar"]
    envs = ["CartPole-v1", "Pendulum-v1"]
    
    for env in envs:
        # Plot 1: Sample efficiency
        plot_sample_efficiency(
            logdir, env, algorithms,
            save_path=os.path.join(outdir, f"sample_efficiency_{env}.png"),
        )
        
        # Plot 2: Learning curves (K=5)
        plot_learning_curves(
            logdir, env, algorithms, K=5,
            save_path=os.path.join(outdir, f"learning_curves_{env}_K5.png"),
        )
    
    print(f"\nAll plots saved to {outdir}/")


# ============================================================
# Demo: create example logs to test plotting
# ============================================================

def create_demo_logs(logdir="logs"):
    """Create synthetic demo logs to test the plotting pipeline."""
    os.makedirs(logdir, exist_ok=True)
    
    for algo in ["iqlearn", "csil", "csil_soar"]:
        for env in ["CartPole-v1"]:
            for K in [1, 3, 5, 10, 15]:
                for seed in [42, 43, 44]:
                    logger = TrainingLogger(algo, env, seed, K)
                    
                    # Simulate a learning curve
                    base = {"iqlearn": 200, "csil": 180, "csil_soar": 250}[algo]
                    noise_scale = {"iqlearn": 30, "csil": 40, "csil_soar": 25}[algo]
                    k_bonus = K * 15  # more data = better final performance
                    
                    np.random.seed(seed + K)
                    for step in range(0, 50001, 1000):
                        progress = min(1.0, step / 30000)
                        reward = (base + k_bonus) * progress + np.random.randn() * noise_scale
                        reward = max(0, min(500, reward))
                        logger.log(step, reward)
                    
                    logger.save(logdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluation plots")
    parser.add_argument("--logdir", type=str, default="logs", help="Directory with CSV logs")
    parser.add_argument("--outdir", type=str, default="plots", help="Output directory for plots")
    parser.add_argument("--demo", action="store_true", help="Create demo logs and plots")
    args = parser.parse_args()

    if args.demo:
        print("Creating demo logs...")
        create_demo_logs(args.logdir)
        print("\nGenerating demo plots...")
        generate_all_plots(args.logdir, args.outdir)
    else:
        generate_all_plots(args.logdir, args.outdir)
