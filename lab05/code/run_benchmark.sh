#!/bin/bash
# Benchmark of bnb_tsp.py on one full Ares node (48 cores).
# Submit with: sbatch run_benchmark.sh
#SBATCH --job-name=bnb_tsp
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --time=02:30:00
#SBATCH --partition=plgrid
#SBATCH --account=plgar2026-cpu
#SBATCH --output=bench_%j.log

module load mpi4py/4.1.0-gompi-2025b
cd "$SLURM_SUBMIT_DIR"
mkdir -p results

# run <N> <init> <workers> <d> <repeats> <outfile>
# P workers need P+1 MPI processes: one master plus P workers.
run() {
    mpiexec -n $(($3 + 1)) python -m mpi4py.futures bnb_tsp.py \
        --n "$1" --init "$2" --d "$4" --repeats "$5" >> "$6"
}

# ---------------------------------------------------------------------------
# Main data set: N = 20 cities, strong local upper bound (2-opt tour).
# ---------------------------------------------------------------------------
N=20; OUT=results/bench_n20_2opt.jsonl; : > "$OUT"
python bnb_tsp.py --n $N --init 2opt --serial --repeats 3 >> "$OUT"

# Experiment 1: fixed d, growing number of workers.
for d in 2 3 4; do
    for P in 1 2 4 8 16 24 32 47; do run $N 2opt $P $d 3 "$OUT"; done
done
# Experiment 2: fixed P, growing d. P = 1 is the baseline for each d.
for d in 1 5; do
    for P in 1 8 32; do run $N 2opt $P $d 3 "$OUT"; done
done
# (d = 2, 3, 4 with P = 1, 8, 32 were already measured in experiment 1.)

# ---------------------------------------------------------------------------
# Contrast: weak local bound (greedy completion of the prefix), N = 16.
# The work grows strongly with d here, so only d = 2 and 3 are measured.
# ---------------------------------------------------------------------------
N=16; OUT=results/bench_n16_greedy.jsonl; : > "$OUT"
python bnb_tsp.py --n $N --init greedy --serial --repeats 3 >> "$OUT"
for d in 2 3; do
    for P in 1 2 4 8 16 32 47; do run $N greedy $P $d 2 "$OUT"; done
done

# Probe: weak bound at N = 20, 8 workers, d = 2 (limited to 15 minutes).
OUT=results/probe_n20_greedy.jsonl; : > "$OUT"
timeout 900 mpiexec -n 9 python -m mpi4py.futures bnb_tsp.py \
    --n 20 --init greedy --d 2 --repeats 1 >> "$OUT" || echo "probe timed out"

# ---------------------------------------------------------------------------
# Ablation: sequential search with and without the lower bound.
# ---------------------------------------------------------------------------
for b in none minedge; do
    OUT=results/ablation_$b.jsonl; : > "$OUT"
    for n in 10 11 12 13 14 15 16; do
        python bnb_tsp.py --n $n --bound $b --init 2opt --serial --repeats 1 >> "$OUT"
    done
done
