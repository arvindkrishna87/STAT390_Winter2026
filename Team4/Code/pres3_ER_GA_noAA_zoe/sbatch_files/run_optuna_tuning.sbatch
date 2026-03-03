#!/bin/bash
#SBATCH --account=e32998                  # Use your Quest project account (check with 'groups')
#SBATCH --partition=long                  # 'short', 'normal', or 'long' partition
#SBATCH --time=30:00:00                    # max run time
#SBATCH --nodes=1                         # 1 node
#SBATCH --ntasks=16                        # n nores
#SBATCH --mem=128G                          # n GB RAM
#SBATCH --job-name=OPTUNA_Tuning             # Job name (no spaces)
#SBATCH --output=/home/zel9906/Code4_ER_and_GA/logs/%x_%j.out
#SBATCH --error=/home/zel9906/Code4_ER_and_GA/logs/%x_%j.err
#SBATCH --mail-type=ALL                   # Send email on begin, end, and fail
#SBATCH --mail-user=zoelively2026@u.northwestern.edu  # Your NU email

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /home/zel9906/env   # you might want to copy the env and put it in your new project directory, if different from e32998

# Print system info
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Run training
python3 -m pip install -r /home/zel9906/Code4_ER_and_GA/requirements.txt
python3 -m pip install --upgrade certifi
export SSL_CERT_FILE="$(python -m certifi)"


cd /home/zel9906/Code4_ER_and_GA

# without loading checkpoints, trying 5 diff train / test splits

python3 optuna_training.py \
    --embeddings_dir /projects/e32998/patches_pooled4096 \
    --n_trials 10 \
    --n_epochs 8 \
    --per_slice_cap 500 \
    --max_slices_per_stain 5 \
    --load_splits /home/zel9906/Code4_ER_and_GA/data_splits/data_splits_01.npz