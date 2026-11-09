#!/bin/bash
# Optional: run the measurements on the Ares cluster with: sbatch ares_job.sh
# Before running, set your grant below and check the module names in the Quickstart.
#SBATCH --job-name=jacobi
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --time=01:00:00
#SBATCH --partition=plgrid
#SBATCH --account=YOUR_GRANT-cpu
#SBATCH --output=jacobi_%j.log

module load scipy-bundle mpi4py

cd "$SLURM_SUBMIT_DIR"
python benchmark.py --procs 1 2 4 9 16 25 36 48 --repeats 3 \
    --sizes 256:10000 1024:1500 4096:100 --output results/benchmark_ares.csv
