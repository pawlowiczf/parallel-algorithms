"""Measure jacobi_mpi.py for different P and n. Save raw times to a CSV file.

Each run does a fixed number of iterations (--eps 0), so the work for a given n
is the same for every P.

Usage:
    python benchmark.py
    python benchmark.py --procs 1 2 4 --sizes 256:10000 1024:1500 --repeats 3
"""

import argparse
import csv
import json
import os
import shutil
import site
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TIMEOUT_SECONDS = 900  # mpiexec can hang forever if one process crashes
COLUMNS = ["n", "P", "repeat", "k", "dims", "time", "comm_time"]


def run_solver(procs, *solver_args):
    """Run jacobi_mpi.py on `procs` processes and return its JSON result."""
    env = dict(os.environ)
    # pip installs Intel MPI into the user folder, which may not be on PATH.
    env["PATH"] = (
        str(Path(site.getuserbase()) / "Library" / "bin") + os.pathsep + env["PATH"]
    )
    mpiexec = shutil.which("mpiexec", path=env["PATH"])
    if mpiexec is None:
        raise RuntimeError(
            "mpiexec not found. Install it with: pip install mpi4py impi-rt"
        )

    command = [mpiexec, "-n", str(procs), sys.executable, str(HERE / "jacobi_mpi.py")]
    command += [str(arg) for arg in solver_args]
    result = subprocess.run(
        command, env=env, capture_output=True, text=True, timeout=TIMEOUT_SECONDS
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout.strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--procs", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["128:20000", "512:3000", "2048:200"],
        help="grid sizes as n:iterations",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output", default=str(HERE / "results" / "benchmark_local.csv")
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, COLUMNS)
        writer.writeheader()
        for repeat in range(args.repeats):
            for size in args.sizes:
                n, iterations = size.split(":")
                for procs in args.procs:
                    result = run_solver(
                        procs, "--n", n, "--eps", 0, "--kmax", iterations
                    )
                    writer.writerow(
                        {
                            "n": n,
                            "P": procs,
                            "repeat": repeat,
                            "k": result["k"],
                            "dims": "x".join(map(str, result["dims"])),
                            "time": result["time"],
                            "comm_time": result["comm_time"],
                        }
                    )
                    file.flush()
                    print(
                        f"repeat={repeat} n={n} P={procs} time={result['time']:.3f}s",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
