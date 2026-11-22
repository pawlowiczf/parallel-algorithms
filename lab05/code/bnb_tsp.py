"""Parallel branch and bound for the travelling salesman problem.

Cities lie on a circle at random angles. The master process generates every
path that starts in city 0 (city 1 in the task text) and has `d` cities, and
hands them out with MPIPoolExecutor.map. A worker searches the subtree under
one path with depth-first branch and bound. Pruning is purely local: a task
starts from its own upper bound (--init) and improves it only with its own
results; no bound is shared between tasks.

Run (P workers need P+1 MPI processes, one of them is the master):
    mpiexec -n 5 python -m mpi4py.futures bnb_tsp.py --n 12 --d 3 --init 2opt
Sequential reference without MPI:
    python bnb_tsp.py --n 12 --serial
Options: --bound {none,minedge} lower bound, --init {greedy,2opt} initial
upper bound, --repeats R number of timed repetitions.
"""

import argparse
import itertools
import json
import math
import random
import time
from functools import lru_cache


@lru_cache(maxsize=None)
def distances(n, seed):
    """Distance matrix for n cities at random angles on the unit circle."""
    rng = random.Random(seed)
    angles = [rng.uniform(0, 2 * math.pi) for _ in range(n)]
    pts = [(math.cos(a), math.sin(a)) for a in angles]
    return tuple(
        tuple(math.dist(p, q) for q in pts) for p in pts
    )


def greedy_completion(prefix, dist):
    """Length and tour of the nearest-neighbour completion of `prefix`."""
    n = len(dist)
    tour = list(prefix)
    left = set(range(n)) - set(tour)
    length = sum(dist[a][b] for a, b in zip(tour, tour[1:]))
    while left:
        nxt = min(left, key=dist[tour[-1]].__getitem__)
        length += dist[tour[-1]][nxt]
        tour.append(nxt)
        left.remove(nxt)
    return length + dist[tour[-1]][tour[0]], tour


@lru_cache(maxsize=None)
def instance_data(n, seed):
    """Per-process cache of everything that depends only on the instance.

    Returns (dist, min_out, order):
      min_out[c] - cheapest edge leaving city c. Every city on the remaining
                   part of a tour must still be left once, so the sum of
                   min_out over the unvisited cities plus the last city is a
                   valid lower bound on the length still to be added.
      order[c]   - cities sorted by distance from c (nearest first).
    The cache holds no search results, so tasks still share no pruning bound.
    """
    dist = distances(n, seed)
    min_out = tuple(min(x for j, x in enumerate(row) if j != i)
                    for i, row in enumerate(dist))
    order = tuple(tuple(sorted(range(n), key=row.__getitem__)) for row in dist)
    return dist, min_out, order


@lru_cache(maxsize=None)
def two_opt_tour(n, seed):
    """Nearest-neighbour tour from city 0 improved with 2-opt: (length, tour).

    Used as an initial upper bound. It depends only on the instance, so every
    process computes it once and keeps it; no search information is shared.
    """
    dist = distances(n, seed)
    _, tour = greedy_completion((0,), dist)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = tour[i - 1], tour[i]
                c, e = tour[j], tour[(j + 1) % n]
                if dist[a][c] + dist[b][e] < dist[a][b] + dist[c][e] - 1e-12:
                    tour[i:j + 1] = reversed(tour[i:j + 1])
                    improved = True
    length = sum(dist[tour[k]][tour[(k + 1) % n]] for k in range(n))
    return length, tuple(tour)


def search_subtree(task):
    """Best tour that starts with `prefix`. Returns (length, tour, nodes)."""
    prefix, n, seed, bound, init = task
    dist, min_out, order = instance_data(n, seed)
    use_bound = bound == "minedge"
    # Initial upper bound (best[0]) and the tour that goes with it (best[1]):
    #   greedy - nearest-neighbour completion of this task's prefix
    #   2opt   - 2-opt tour of the whole instance, independent of the prefix
    # The tour is returned if the search finds nothing strictly shorter.
    if init == "2opt":
        length0, tour0 = two_opt_tour(n, seed)
        best = [length0, list(tour0)]
    else:
        best = list(greedy_completion(prefix, dist))
    nodes = [0]
    visited = [False] * n
    for city in prefix:
        visited[city] = True
    length = sum(dist[a][b] for a, b in zip(prefix, prefix[1:]))
    path = list(prefix)

    def dfs(last, length, rest):
        nodes[0] += 1
        # bound: even the optimistic completion is no better than the best tour
        if length + (rest + min_out[last] if use_bound else 0.0) >= best[0]:
            return
        if len(path) == n:
            total = length + dist[last][path[0]]
            if total < best[0]:
                best[0], best[1] = total, path[:]
            return
        row = dist[last]
        # branch: every unvisited city, nearest first
        for city in order[last]:
            if not visited[city]:
                visited[city] = True
                path.append(city)
                dfs(city, length + row[city], rest - min_out[city])
                path.pop()
                visited[city] = False

    rest = sum(min_out[c] for c in range(n) if not visited[c])
    dfs(prefix[-1], length, rest)
    return best[0], best[1], nodes[0]


def prefixes(n, d):
    """All paths of d cities starting in city 0."""
    return [(0,) + p for p in itertools.permutations(range(1, n), d - 1)]


def combine(results):
    """Shortest tour among the task results and the total node count."""
    length, tour, _ = min(results, key=lambda r: r[0])
    return length, tour, sum(r[2] for r in results)


def warm_up(task):
    """Fill this process's caches (instance data, 2-opt tour) before timing.

    The short sleep makes it likely that the warm-up tasks are spread over
    all workers instead of one fast worker taking them all.
    """
    n, seed = task
    instance_data(n, seed)
    two_opt_tour(n, seed)
    time.sleep(0.05)


def run_parallel(executor, n, d, seed, bound, init):
    tasks = [(p, n, seed, bound, init) for p in prefixes(n, d)]
    start = time.perf_counter()
    results = list(executor.map(search_subtree, tasks, chunksize=1))
    elapsed = time.perf_counter() - start
    return elapsed, len(tasks), combine(results)


def run_serial(n, seed, bound, init):
    instance_data(n, seed)  # warm the caches so that every repeat is comparable
    two_opt_tour(n, seed)
    start = time.perf_counter()
    result = search_subtree(((0,), n, seed, bound, init))
    return time.perf_counter() - start, 1, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=12, help="number of cities")
    parser.add_argument("--d", type=int, default=3, help="depth of the task paths")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bound", choices=["none", "minedge"], default="minedge",
                        help="lower bound used for pruning")
    parser.add_argument("--init", choices=["greedy", "2opt"], default="greedy",
                        help="initial upper bound inside every task")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--serial", action="store_true", help="no MPI, one process")
    args = parser.parse_args()
    if not 1 <= args.d <= args.n:
        parser.error("need 1 <= d <= n")

    if args.serial:
        workers = 0
        runs = [run_serial(args.n, args.seed, args.bound, args.init)
                for _ in range(args.repeats)]
    else:
        from mpi4py import MPI
        from mpi4py.futures import MPIPoolExecutor

        workers = MPI.COMM_WORLD.Get_size() - 1
        if workers < 1:
            raise SystemExit("need at least 2 MPI processes (1 master + P workers)")
        with MPIPoolExecutor() as executor:
            # warm-up: start the workers and fill their caches
            list(executor.map(warm_up, [(args.n, args.seed)] * (4 * workers)))
            runs = [
                run_parallel(executor, args.n, args.d, args.seed, args.bound, args.init)
                for _ in range(args.repeats)
            ]

    for elapsed, tasks, (length, tour, nodes) in runs:
        print(json.dumps({
            "n": args.n, "d": 1 if args.serial else args.d, "bound": args.bound, "init": args.init,
            "workers": workers, "seed": args.seed,
            "tasks": tasks, "time": elapsed, "length": length,
            "nodes": nodes, "tour": tour,
        }), flush=True)


if __name__ == "__main__":
    main()
