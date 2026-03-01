#!/bin/bash
#SBATCH --account=e32998                  # Use your Quest project account (check with 'groups')
#SBATCH --partition=long                  # 'short', 'normal', or 'long' partition
#SBATCH --time=43:00:00                    # max run time
#SBATCH --nodes=1                         # 1 node
#SBATCH --ntasks=16                        # n nores
#SBATCH --mem=128G                          # n GB RAM
#SBATCH --job-name=optuna_tuning_feb27           # Job name (no spaces)
#SBATCH --output=/home/sml7045/presentation3/output/optuna_tuning_feb27.out
#SBATCH --error=/home/sml7045/presentation3/output/optuna_tuning_feb27.err
#SBATCH --mail-type=ALL                   # Send email on begin, end, and fail
#SBATCH --mail-user=sofielangenhuizen2026@u.northwestern.edu  # Your NU email

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /home/sml7045/env   # you might want to copy the env and put it in your new project directory, if different from e32998

# Print system info
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Run training
python3 -m pip install -r /projects/e32998/STAT390_Winter2026/Team3/Code/presentation3/requirements.txt
python3 -m pip install --upgrade certifi
export SSL_CERT_FILE="$(python -m certifi)"


cd /home/sml7045/presentation3

# without loading checkpoints, trying 5 diff train / test splits
#increase epochs to 30 for now

python3 optuna_training.py \
    --n_trials 20 \
    --n_epochs 20 \
    --per_slice_cap 500 \
    --max_slices_per_stain 5 \
    --load_splits /home/sml7045/presentation3/data_splits/data_splits.npz
