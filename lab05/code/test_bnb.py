"""Check bnb_tsp against brute force on small instances (no MPI needed).

Run: python test_bnb.py
Covers every combination of --bound, --init and several depths d, and checks
that the returned tour is a valid cycle whose length equals the reported one.
"""

import itertools

import bnb_tsp as b


def tour_length(tour, dist):
    return sum(dist[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))


def main():
    checked = 0
    for seed in range(1, 11):
        for n in (6, 7, 8, 9):
            dist = b.distances(n, seed)
            brute = min(
                tour_length((0,) + p, dist) for p in itertools.permutations(range(1, n))
            )
            for bound in ("none", "minedge"):
                for init in ("greedy", "2opt"):
                    for d in (1, 2, 3, n):
                        tasks = [(p, n, seed, bound, init) for p in b.prefixes(n, d)]
                        length, tour, _ = b.combine(
                            [b.search_subtree(t) for t in tasks]
                        )
                        assert abs(length - brute) < 1e-9, (seed, n, bound, init, d)
                        assert sorted(tour) == list(range(n)), "not a permutation"
                        assert abs(tour_length(tour, dist) - length) < 1e-9, (
                            "tour does not match the reported length"
                        )
                        checked += 1
    print(f"OK: {checked} configurations match brute force")


if __name__ == "__main__":
    main()
