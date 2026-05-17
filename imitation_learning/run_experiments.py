"""
IQ-Learn experiment runner.

Smoke mode (default): runs CartPole K=5 seed=42 and Pendulum K=5 seed=42 at
full step count. Used to verify both envs train correctly before launching
the full sweep.

Full mode: runs all 30 configs (2 envs x 5 K-values x 3 seeds).
Each run produces logs/iqlearn_<env>_K<k>_seed<s>.csv.

Usage:
    python run_experiments.py            # smoke test
    python run_experiments.py --full     # full sweep
"""

import argparse
import time

from iq_learn import train_iq_learn, save_log_csv

ENV_CONFIGS = {
    "CartPole-v1": {"total_steps": 20_000, "hidden_dim": 128, "batch_size": 128,
                     "eval_interval": 1000, "lr": 3e-4, "chi2_coef": 0.5,
                     "alpha": 0.2, "auto_alpha": False},
    "Pendulum-v1": {"total_steps": 30_000, "hidden_dim": 256, "batch_size": 256,
                     "eval_interval": 1000, "lr": 1e-4, "chi2_coef": 0.5,
                     "alpha": 0.2, "auto_alpha": True},
}

K_VALUES = [1, 3, 5, 10, 15]
SEEDS = [42, 43, 44]


def run_one(env_name, K, seed, log_dir="logs", verbose=False):
    cfg = ENV_CONFIGS[env_name]
    expert_path = f"expert_data/{env_name}_K{K}.npz"

    print(f"\n[{env_name} | K={K} | seed={seed}] starting ({cfg['total_steps']} steps)")
    t0 = time.time()
    _, log = train_iq_learn(
        env_name=env_name,
        expert_npz_path=expert_path,
        seed=seed,
        total_steps=cfg["total_steps"],
        eval_interval=cfg["eval_interval"],
        eval_episodes=5,
        hidden_dim=cfg["hidden_dim"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        chi2_coef=cfg["chi2_coef"],
        alpha=cfg["alpha"],
        auto_alpha=cfg["auto_alpha"],
        verbose=verbose,
    )
    elapsed = time.time() - t0
    final_reward = log["eval_reward"][-1] if log["eval_reward"] else float("nan")
    max_reward = max(log["eval_reward"]) if log["eval_reward"] else float("nan")

    out_path = f"{log_dir}/iqlearn_{env_name}_K{K}_seed{seed}.csv"
    save_log_csv(log, out_path)

    print(f"  done in {elapsed/60:.1f} min | final={final_reward:.1f} | max={max_reward:.1f} | -> {out_path}")
    return elapsed, final_reward, max_reward


def smoke_test():
    print("=" * 60)
    print("SMOKE TEST: CartPole K=5 + Pendulum K=5, both seed=42")
    print("=" * 60)
    results = []
    for env_name in ["CartPole-v1", "Pendulum-v1"]:
        results.append((env_name, *run_one(env_name, K=5, seed=42, verbose=True)))
    print("\n" + "=" * 60)
    print("Smoke test summary:")
    for env, elapsed, final, mx in results:
        print(f"  {env:14s}  {elapsed/60:5.1f} min  final={final:7.1f}  max={mx:7.1f}")


def full_sweep():
    print("=" * 60)
    print(f"FULL SWEEP: {len(K_VALUES) * len(SEEDS) * len(ENV_CONFIGS)} runs")
    print("=" * 60)
    t_total = time.time()
    for env_name in ENV_CONFIGS:
        for K in K_VALUES:
            for seed in SEEDS:
                run_one(env_name, K, seed)
    print(f"\nAll runs done in {(time.time() - t_total)/60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run full 30-config sweep")
    args = parser.parse_args()

    if args.full:
        full_sweep()
    else:
        smoke_test()
