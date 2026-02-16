#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=long
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=128G
#SBATCH --job-name=MIL_attn_only
#SBATCH --output=/projects/e32998/STAT390_Winter2026/Team1/Code/MIL_trainer_9Dec_JointFinal/logs/attn_only_%j.log
#SBATCH --error=/projects/e32998/STAT390_Winter2026/Team1/Code/MIL_trainer_9Dec_JointFinal/logs/attn_only_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=esz6865@u.northwestern.edu

set -euo pipefail

module purge
module load python-anaconda3/2019.10
module load git
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

cd /projects/e32998/STAT390_Winter2026/Team1/Code/MIL_trainer_9Dec_JointFinal

CKPT="/projects/e32998/MIL_training/final_runs/run_20260215_125840/checkpoints/20260215_134859_HierarchicalAttnMIL_epoch3.pth"
SPLIT="/projects/e32998/STAT390_Winter2026/Team1/Code/MIL_trainer_9Dec_JointFinal/data_splits/data_splits_01.npz"

python3 main.py \
  --resume "$CKPT" \
  --eval_only \
  --analyze_attention \
  --attention_top_n 8 \
  --load_splits "$SPLIT"

echo "End time: $(date)"