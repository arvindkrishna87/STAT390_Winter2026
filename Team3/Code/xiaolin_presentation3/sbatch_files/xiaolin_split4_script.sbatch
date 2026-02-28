#!/bin/bash
#SBATCH --account=e32998                  # Use your Quest project account (check with 'groups')
#SBATCH --partition=long                  # 'short', 'normal', or 'long' partition
#SBATCH --time=00:10:00                    # max run time
#SBATCH --nodes=1                         # 1 node
#SBATCH --ntasks=16                        # n nores
#SBATCH --mem=128G                          # n GB RAM
#SBATCH --job-name=split_gen_4              # Job name (no spaces)
#SBATCH --output=/home/xla6290/presentation3/script_gen_output/xiaolin_split4.out
#SBATCH --error=/home/xla6290/presentation3/script_gen_output/xiaolin_split4.err
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

python3 make_splits4.py \
    --labels_csv /home/xla6290/presentation3/extra_benign_case_grade_match.csv \
    --patches_dir /projects/e32998/patches_benign_split \
    --seed 12 \
    --save_dir /home/xla6290/presentation3/data_splits
