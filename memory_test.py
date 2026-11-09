"""Check if memory speed limits the program.

Runs several independent copies of the solver at the same time, one process each,
so they do not communicate. Each copy does the same work, so if the copies slow
down, the cause is shared hardware (memory, cache), not MPI.

Usage:
    python memory_test.py
"""

import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchmark import run_solver

HERE = Path(__file__).resolve().parent
REPEATS = 9
COPIES = [1, 2, 4, 6]
# Grids with about as many nodes as one block at P=6: (name, n, iterations).
BLOCKS = [("small", 209, 5000), ("large", 836, 300)]


def run_copies(copies, n, iterations):
    args = ["--n", n, "--eps", 0, "--kmax", iterations]
    with ThreadPoolExecutor(copies) as pool:
        futures = [pool.submit(run_solver, 1, *args) for _ in range(copies)]
        return [future.result()["time"] for future in futures]


def main():
    with (HERE / "results" / "memory_test.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["block", "copies", "repeat", "time"])
        for repeat in range(REPEATS):
            for name, n, iterations in BLOCKS:
                for copies in COPIES:
                    for time in run_copies(copies, n, iterations):
                        writer.writerow([name, copies, repeat, time])
                    file.flush()


if __name__ == "__main__":
    main()
