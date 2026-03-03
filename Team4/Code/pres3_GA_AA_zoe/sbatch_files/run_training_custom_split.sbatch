#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=long
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --job-name=MIL_train
#SBATCH --output=/home/zel9906/Code4_GA_withAA/logs/%x_%j.out
#SBATCH --error=/home/zel9906/Code4_GA_withAA/logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zoelively2026@u.northwestern.edu

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /home/zel9906/env

# ---------- paths ----------
PROJECT_DIR="/home/zel9906/Code4_GA_withAA"
SPLITS_NPZ="$PROJECT_DIR/data_splits/data_splits.npz"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR" || exit 1

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running on host: $(hostname)"
echo "Project dir: $PROJECT_DIR"
echo "Using python: $(which python)"
python --version
echo "Using splits: $SPLITS_NPZ"
echo "main.py path: $(readlink -f main.py)"

# ---------- run training ----------
python main.py \
  --run_tag "customsplit" \
  --analyze_attention \
  --attention_top_n 8 \
  --per_slice_cap 500 \
  --max_slices_per_stain 5 \
  --load_splits "$SPLITS_NPZ"

echo "End time: $(date)"