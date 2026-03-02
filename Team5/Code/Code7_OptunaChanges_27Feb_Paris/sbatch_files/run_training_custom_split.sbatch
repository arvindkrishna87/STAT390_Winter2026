#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --job-name=MIL_train
#SBATCH --output=/gpfs/projects/e32998/STAT390_Winter2026/Team5/Code/Code7_OptunaChanges_27Feb_Paris/logs/train_%j.log
#SBATCH --error=/gpfs/projects/e32998/STAT390_Winter2026/Team5/Code/Code7_OptunaChanges_27Feb_Paris/logs/train_%j.err
#SBATCH --mail-type=ALL                   # Send email on begin, end, and fail
#SBATCH --mail-user=parisbozzuti2026@u.northwestern.edu  # Your NU email

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env   # you might want to copy the env and put it in your new project directory, if different from e32998

# Print system info
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Run training
python3 -m pip install -r /projects/e32998/STAT390_Winter2026/Team5/Code/Code7_OptunaChanges_27Feb_Paris/requirements.txt
python3 -m pip install --upgrade certifi
export SSL_CERT_FILE="$(python -m certifi)"

cd /projects/e32998/STAT390_Winter2026/Team5/Code/Code7_OptunaChanges_27Feb_Paris

# without loading checkpoints, trying 5 diff train / test splits

python3 main.py \
    --analyze_attention \
    --attention_top_n 8 \
    --per_slice_cap 500 \
    --max_slices_per_stain 5 \
    --load_splits /projects/e32998/STAT390_Winter2026/Team5/Code/Code7_OptunaChanges_27Feb_Paris/data_splits/data_splits.npz
