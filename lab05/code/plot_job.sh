#!/bin/bash
# Make the figures on a compute node (not on the login node).
# Submit with: sbatch plot_job.sh
#SBATCH --job-name=bnb_plot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --partition=plgrid-testing
#SBATCH --account=plgar2026-cpu
#SBATCH --output=plot_%j.log

eval "$(conda shell.bash hook)"
conda activate plgpawlowiczfenv      # existing env with pandas and matplotlib
cd "$SLURM_SUBMIT_DIR"
python plot_results.py results/bench_n14.jsonl figures
