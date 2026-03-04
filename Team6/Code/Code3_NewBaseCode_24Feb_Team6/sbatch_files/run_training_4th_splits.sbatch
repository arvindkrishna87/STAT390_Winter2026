#!/bin/bash
#SBATCH --account=e32998                              # Use your Quest project account (check with 'groups')
#SBATCH --partition=long                              # 'short', 'normal', or 'long' partition
#SBATCH --time=12:00:00                               # max run time
#SBATCH --nodes=1                                     # 1 node
#SBATCH --ntasks=16                                   # n cores
#SBATCH --mem=64G                                     # n GB RAM
#SBATCH --job-name=MIL_split_4                        # Job name (no spaces)
#SBATCH --chdir=./                                    # set working directory
#SBATCH --output=./logs/%j_%x.log                     # Standard output file
#SBATCH --error=./logs/%j_%x.log                      # now writing error to the same log file
#SBATCH --mail-type=ALL                               # Send email on begin, end, and fail
#SBATCH --mail-user=henrychau2026@u.northwestern.edu  # Your NU email

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env   # you might want to copy the env and put it in your new project directory, if different from e32998

# Print system info
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Run training
python3 -m pip install -r requirements.txt
python3 -m pip install --upgrade certifi
export SSL_CERT_FILE="$(python -m certifi)"

# Use preset splits
python3 main.py \
    --analyze_attention \
    --attention_top_n 8 \
    --per_slice_cap 500 \
    --load_splits ./data_splits/data_splits_new_04.npz
