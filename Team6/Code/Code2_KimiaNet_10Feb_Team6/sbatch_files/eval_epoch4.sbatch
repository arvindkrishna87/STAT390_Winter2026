#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --job-name=MIL_Eval_7222
#SBATCH --output=./logs/eval_final_%j.log

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

cd /projects/e32998/STAT390_Winter2026/Team6/Code/Code2_KimiaNet_10Feb_Team6
echo "Starting evaluation for Epoch 4 checkpoint..."

python3 main.py \
    --eval_only \
    --resume /projects/e32998/MIL_training/final_runs/run_20260215_111759/checkpoints/20260215_125248_HierarchicalAttnMIL_epoch4.pth \
    --analyze_attention \
    --attention_top_n 8 \
    --load_splits ./data_splits/data_splits_05.npz
