"""Resume the CSIL/CSIL+SOAR Pendulum sweep, skipping configs already in logs/."""

import os
import sys
import time
import pathlib

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE / "src"))
sys.path.insert(0, str(_HERE))

from csil import run_one

K_VALUES = [1, 3, 5, 10, 15]
SEEDS = [42, 43, 44]
LOGS_DIR = _HERE / "logs"


def csv_path(algo, K, seed):
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
        print("All Pendulum CSIL/CSILSOAR runs are complete.")
        return

    print(f"Found {len(jobs)} missing Pendulum runs.")

    t_total = time.time()
    for i, (K, seed, soar) in enumerate(jobs, 1):
        algo = "csilsoar" if soar else "csil"
        print(f"\n[{i}/{len(jobs)}]  {algo.upper()}  K={K}  seed={seed}")
        run_one("Pendulum", K, seed, soar=soar, verbose=False)

    print(f"\nDone in {(time.time() - t_total) / 60:.1f} min.")


if __name__ == "__main__":
    main()
