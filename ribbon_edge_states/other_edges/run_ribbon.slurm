#!/bin/bash
#SBATCH --partition=mem
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --job-name="ribbon"

cd $SLURM_SUBMIT_DIR

source ~/.bashrc
conda activate pyqmc_dev

export OMP_NUM_THREADS = 1

srun python3 get_ribbon_eigs_e0.0.py

