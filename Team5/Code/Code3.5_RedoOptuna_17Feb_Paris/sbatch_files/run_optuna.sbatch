#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=64G
#SBATCH --job-name=MIL_optuna
#SBATCH --output=/gpfs/projects/e32998/STAT390_Winter2026/Team5/Code/Code3.5_RedoOptuna_17Feb_Paris/logs/optuna_%j.log
#SBATCH --error=/gpfs/projects/e32998/STAT390_Winter2026/Team5/Code/Code3.5_RedoOptuna_17Feb_Paris/logs/optuna_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=parisbozzuti2026@u.northwestern.edu

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

# Print system info
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Install dependencies
python3 -m pip install --user -r /gpfs/projects/e32998/STAT390_Winter2026/Team5/Code/Code3.5_RedoOptuna_17Feb_Paris/requirements.txt

cd /gpfs/projects/e32998/STAT390_Winter2026/Team5/Code/Code3.5_RedoOptuna_17Feb_Paris/

python3 optuna_tuning_2.py \
    --n_trials 7 \
    --n_epochs 7 \
    --load_splits ./data_splits/data_splits_01.npz \
    --per_slice_cap 500 \
    --max_slices_per_stain 5
