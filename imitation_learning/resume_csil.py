"""
Resume the CSIL / CSIL+SOAR Pendulum sweep.

Skips any (K, seed, algo) combination whose CSV already exists in logs/.
Runs the missing ones at the new (faster, more stable) hyperparameters from
the fix-csil-pendulum branch:
    - scale_factor = 0.1  (was 1.0 — too aggressive on Pendulum)
    - bc_steps     = 5000 (was 20000 — caused BC overfit on small K)
    - total_steps  = 30000 for Pendulum (was 50000 — peak performance happens
                                          earlier; saves time)

Usage on Colab:
    !python resume_csil.py
"""

import os
import sys
import time
import pathlib

# Make src/ importable
_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE / "src"))
sys.path.insert(0, str(_HERE))

from csil import run_one

K_VALUES = [1, 3, 5, 10, 15]
SEEDS = [42, 43, 44]
LOGS_DIR = _HERE / "logs"


def csv_path(algo, K, seed):
    """Matches the naming used in csil.py: logs/{algo}_{env_key}_K{K}_seed{seed}.csv"""
    return LOGS_DIR / f"{algo}_Pendulum_K{K}_seed{seed}.csv"


def main():
    jobs = []
    for soar in [False, True]:
        algo = "csilsoar" if soar else "csil"
        for K in K_VALUES:
            for seed in SEEDS:
                if csv_path(algo, K, seed).exists():
                    continue
                jobs.append((K, seed, soar))

    if not jobs:
        print("Nothing to do — all Pendulum CSIL/CSILSOAR runs are complete.")
        return

    print(f"Found {len(jobs)} missing Pendulum runs. Starting...")
    print("=" * 60)

    t_total = time.time()
    for i, (K, seed, soar) in enumerate(jobs, 1):
        algo = "csilsoar" if soar else "csil"
        print(f"\n[{i}/{len(jobs)}]  {algo.upper()}  K={K}  seed={seed}")
        run_one("Pendulum", K, seed, soar=soar, verbose=False)

    elapsed = (time.time() - t_total) / 60
    print(f"\n{'=' * 60}\nAll done in {elapsed:.1f} min.")


if __name__ == "__main__":
    main()
