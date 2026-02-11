#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=long
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=128G
#SBATCH --job-name=MIL_optuna
#SBATCH --output=/projects/e32998/MIL_training/logs/optuna_%j.log
#SBATCH --error=/projects/e32998/MIL_training/logs/optuna_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=parisbozzuti2026@u.northwestern.edu

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

# Install optuna if not already installed
pip install -r requirements.txt

cd /gpfs/projects/e32998/STAT390_Winter2026/Team5/Code/Code2.5_OptunaTraining_10Feb_Paris

python3 optuna_tuning.py \
    --n_trials 50 \
    --n_epochs 15 \
    --load_splits ./data_splits/data_splits_01.npz \
    --per_slice_cap 500 \
    --max_slices_per_stain 5
