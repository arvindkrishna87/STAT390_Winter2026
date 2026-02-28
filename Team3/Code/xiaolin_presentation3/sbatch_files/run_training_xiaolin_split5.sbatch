#!/bin/bash
#SBATCH --account=e32998                  # Use your Quest project account (check with 'groups')
#SBATCH --partition=long                  # 'short', 'normal', or 'long' partition
#SBATCH --time=18:00:00                    # max run time
#SBATCH --nodes=1                         # 1 node
#SBATCH --ntasks=16                        # n nores
#SBATCH --mem=128G                          # n GB RAM
#SBATCH --job-name=train_split_5_confirmation             # Job name (no spaces)
#SBATCH --output=/home/xla6290/presentation3/output/train_split_5_confirmation.out   # Standard output file
#SBATCH --error=/home/xla6290/presentation3/output/train_split_5_confirmation.err   # now writing error to the same log file
#SBATCH --mail-type=ALL                   # Send email on begin, end, and fail
#SBATCH --mail-user=xiaolinliu2026@u.northwestern.edu  # Your NU email


module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /home/xla6290/env   # you might want to copy the env and put it in your new project directory, if different from e32998

# Print system info
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Run training
python3 -m pip install -r /projects/e32998/STAT390_Winter2026/Team3/Code/presentation3/requirements.txt
python3 -m pip install --upgrade certifi
export SSL_CERT_FILE="$(python -m certifi)"


cd /home/xla6290/presentation3

# without loading checkpoints, trying 5 diff train / test splits

python3 main.py \
    --analyze_attention \
    --attention_top_n 8 \
    --per_slice_cap 500 \
    --max_slices_per_stain 5 \
    --load_splits /home/xla6290/presentation3/data_splits/xiaolin_split5.npz



