"""Parallel Jacobi method for the membrane Poisson equation (MPI).

This is Algorithm 1 from algorytm_opis.pdf. It solves:
    d2h/dx2 + d2h/dy2 = -p/T  on the square (0,a) x (0,a),  with h = 0 on the edge.

Usage:
    mpiexec -n 4 python jacobi_mpi.py --n 64 --eps 1e-8 --output result.npz
    mpiexec -n 4 python jacobi_mpi.py --n 512 --eps 0 --kmax 2000

With --eps 0 the program always does exactly kmax iterations (used for timing).
Process 0 prints one line of JSON with the results and times.
"""

import argparse
import json

import numpy as np
from mpi4py import MPI


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=64, help="number of grid intervals")
    parser.add_argument("--a", type=float, default=1.0, help="membrane side length")
    parser.add_argument("--p", type=float, default=1.0, help="pressure")
    parser.add_argument("--T", type=float, default=1.0, help="membrane tension")
    parser.add_argument("--eps", type=float, default=1e-8, help="stop threshold")
    parser.add_argument(
        "--kmax", type=int, default=1_000_000, help="maximum iterations"
    )
    parser.add_argument(
        "--m", type=int, default=100, help="check the stop rule every m iterations"
    )
    parser.add_argument(
        "--output", help=".npz file for the full grid (small grids only)"
    )
    return parser.parse_args()


def block_range(total, parts, index):
    """Return (start, size) of one block. Block sizes differ by at most 1."""
    base, extra = divmod(total, parts)
    start = index * base + min(index, extra)
    size = base + (1 if index < extra else 0)
    return start, size


def exchange_halo(cart, neighbors, h, send_col, recv_col):
    """Send block edges to the 4 neighbors and receive their edges into the halo.

    On the membrane edge the neighbor is MPI.PROC_NULL. Sendrecv then does
    nothing, so the halo there stays 0, which is the boundary condition.
    """
    x_low, x_high, y_low, y_high = neighbors

    # Rows are contiguous in memory, so they can be sent directly.
    cart.Sendrecv(h[-2, 1:-1], dest=x_high, recvbuf=h[0, 1:-1], source=x_low)
    cart.Sendrecv(h[1, 1:-1], dest=x_low, recvbuf=h[-1, 1:-1], source=x_high)

    # Columns are not contiguous, so they are copied through buffers.
    send_col[:] = h[1:-1, -2]
    cart.Sendrecv(send_col, dest=y_high, recvbuf=recv_col, source=y_low)
    if y_low != MPI.PROC_NULL:
        h[1:-1, 0] = recv_col

    send_col[:] = h[1:-1, 1]
    cart.Sendrecv(send_col, dest=y_low, recvbuf=recv_col, source=y_high)
    if y_high != MPI.PROC_NULL:
        h[1:-1, -1] = recv_col


def solve(cart, nx, ny, load, eps, kmax, m):
    """Main loop of Algorithm 1. Return h (with halo), k, delta, time and comm time."""
    neighbors = cart.Shift(0, 1) + cart.Shift(1, 1)
    h = np.zeros((nx + 2, ny + 2))
    h_new = np.zeros((nx + 2, ny + 2))
    diff = np.zeros((nx, ny))
    send_col = np.zeros(nx)
    recv_col = np.zeros(nx)
    local_delta = np.zeros(1)
    global_delta = np.zeros(1)
    k = 0
    delta = np.inf
    comm_time = 0.0

    cart.Barrier()
    start_time = MPI.Wtime()
    while True:
        t = MPI.Wtime()
        exchange_halo(cart, neighbors, h, send_col, recv_col)
        comm_time += MPI.Wtime() - t

        # Equation (7). Work in place: allocating temporary arrays in every
        # iteration was several times slower than the computation itself.
        inner = h[1:-1, 1:-1]
        inner_new = h_new[1:-1, 1:-1]
        np.add(h[2:, 1:-1], h[:-2, 1:-1], out=inner_new)
        inner_new += h[1:-1, 2:]
        inner_new += h[1:-1, :-2]
        inner_new += load
        inner_new *= 0.25

        np.subtract(inner_new, inner, out=diff)
        np.abs(diff, out=diff)
        local_delta[0] = diff.max()

        h, h_new = h_new, h
        k += 1

        if k % m == 0:
            t = MPI.Wtime()
            cart.Allreduce(local_delta, global_delta, op=MPI.MAX)
            comm_time += MPI.Wtime() - t
            delta = global_delta[0]

        if delta < eps or k >= kmax:
            break
    elapsed = MPI.Wtime() - start_time

    # If kmax is not a multiple of m, get a current delta for the report.
    if k % m != 0:
        cart.Allreduce(local_delta, global_delta, op=MPI.MAX)
        delta = global_delta[0]
    return h, k, delta, elapsed, comm_time


def gather_grid(cart, h, x_start, y_start, n):
    """Build the full grid on process 0, plus a map of which process owns each node."""
    blocks = cart.gather((x_start, y_start, h[1:-1, 1:-1]), root=0)
    if cart.rank != 0:
        return None, None

    grid = np.zeros((n + 1, n + 1))
    owner = np.full((n + 1, n + 1), -1)
    for rank, (x0, y0, block) in enumerate(blocks):
        nx, ny = block.shape
        grid[x0 + 1 : x0 + 1 + nx, y0 + 1 : y0 + 1 + ny] = block
        owner[x0 + 1 : x0 + 1 + nx, y0 + 1 : y0 + 1 + ny] = rank
    return grid, owner


def main():
    args = parse_args()
    world = MPI.COMM_WORLD

    # Gives sqrt(P) x sqrt(P) when P is a square, as in the design.
    # For other P (such as 2, 3, 6) it gives the closest layout to a square.
    dims = MPI.Compute_dims(world.size, 2)
    inner_nodes = args.n - 1
    if inner_nodes < max(dims):
        raise SystemExit(f"Grid too small: n={args.n} for process grid {dims}")

    cart = world.Create_cart(dims, periods=[False, False], reorder=True)
    cx, cy = cart.Get_coords(cart.rank)
    x_start, nx = block_range(inner_nodes, dims[0], cx)
    y_start, ny = block_range(inner_nodes, dims[1], cy)

    d = args.a / args.n
    load = d * d * args.p / args.T
    h, k, delta, elapsed, comm_time = solve(
        cart, nx, ny, load, args.eps, args.kmax, args.m
    )

    # The slowest process sets the run time.
    max_time = cart.reduce(elapsed, op=MPI.MAX, root=0)
    max_comm_time = cart.reduce(comm_time, op=MPI.MAX, root=0)

    if args.output:
        grid, owner = gather_grid(cart, h, x_start, y_start, args.n)
        if cart.rank == 0:
            np.savez(args.output, h=grid, owner=owner, k=k, delta=delta)

    if cart.rank == 0:
        print(
            json.dumps(
                {
                    "n": args.n,
                    "P": cart.size,
                    "dims": list(dims),
                    "k": k,
                    "delta": float(delta),
                    "time": max_time,
                    "comm_time": max_comm_time,
                }
            )
        )


if __name__ == "__main__":
    main()
