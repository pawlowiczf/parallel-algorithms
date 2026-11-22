"""Share of the total work held by the heaviest task, for N and depths d.

The speedup of the pool cannot exceed total_work / heaviest_task. Work is
counted in visited tree nodes (deterministic) and in seconds (this machine).
Writes results/task_shares.csv. Sequential, no MPI; takes about a minute.

Run: python task_shares.py [N]
"""

import csv
import sys
import time

import bnb_tsp as b


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    rows = []
    for d in (2, 3, 4):
        nodes, secs = [], []
        for p in b.prefixes(n, d):
            t0 = time.perf_counter()
            r = b.search_subtree((p, n, 1, "minedge", "2opt"))
            secs.append(time.perf_counter() - t0)
            nodes.append(r[2])
        rows.append({"n": n, "d": d, "tasks": len(nodes),
                     "heaviest_share_nodes": max(nodes) / sum(nodes),
                     "max_speedup_nodes": sum(nodes) / max(nodes),
                     "heaviest_share_time": max(secs) / sum(secs),
                     "max_speedup_time": sum(secs) / max(secs)})
        print(rows[-1], flush=True)
    with open("results/task_shares.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
