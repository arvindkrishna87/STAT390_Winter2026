#!/bin/bash
#SBATCH --account=e32998                  # Use your Quest project account (check with 'groups')
#SBATCH --partition=long                  # 'short', 'normal', or 'long' partition
#SBATCH --time=18:00:00                    # max run time
#SBATCH --nodes=1                         # 1 node
#SBATCH --ntasks=16                        # n nores
#SBATCH --mem=128G                          # n GB RAM
#SBATCH --job-name=MIL_train              # Job name (no spaces)
#SBATCH --output=/projects/e32998/MIL_training/logs/training_logs_%j.log  # Standard output file
#SBATCH --error=/projects/e32998/MIL_training/logs/training_logs_%j.log   # now writing error to the same log file
#SBATCH --mail-type=ALL                   # Send email on begin, end, and fail
#SBATCH --mail-user=trd1531@u.northwestern.edu  # Your NU email

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env   # you might want to copy the env and put it in your new project directory, if different from e32998

# Print system info
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Run training
python3 -m pip install -r /projects/e32998/STAT390_Winter2026/Team6/Code/Code1_ModelTraining_29Jan_Henry/requirements.txt
python3 -m pip install --upgrade certifi
export SSL_CERT_FILE="$(python -m certifi)"


cd /projects/e32998/STAT390_Winter2026/Team6/Code/Code1_ModelTraining_29Jan_Henry

# without loading checkpoints, trying 5 diff train / test splits

python3 main.py \
    --analyze_attention \
    --attention_top_n 8 \
    --per_slice_cap 500 \
    --max_slices_per_stain 5 \
    --load_splits /projects/e32998/STAT390_Winter2026/Team6/Code/Code1_ModelTraining_29Jan_Henry/data_splits/data_splits_05.npz



