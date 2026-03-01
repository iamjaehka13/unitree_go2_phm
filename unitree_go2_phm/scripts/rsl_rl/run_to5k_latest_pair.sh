#!/usr/bin/env bash
set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate isaaclab

cd /home/iamjaehka13/unitree_go2_phm/unitree_go2_phm/scripts/rsl_rl

echo "[START] $(date '+%F %T') realobs_main_s45: 3000->5000"
python3 train.py \
  --task Unitree-Go2-RealObs-v1 \
  --agent rsl_rl_cfg_entry_point \
  --num_envs 4096 \
  --max_iterations 2000 \
  --seed 45 \
  --resume \
  --load_run 2026-02-20_23-58-46_realobs_main_s45 \
  --checkpoint model_2999.pt \
  --run_name realobs_main_s45_to5k

echo "[DONE ] $(date '+%F %T') realobs_main_s45_to5k"

echo "[START] $(date '+%F %T') baseline_main_s45: 3000->5000"
python3 train.py \
  --task Unitree-Go2-Baseline-v1 \
  --agent rsl_rl_cfg_entry_point \
  --num_envs 4096 \
  --max_iterations 2000 \
  --seed 45 \
  --resume \
  --load_run 2026-02-20_23-58-57_baseline_main_s45 \
  --checkpoint model_2999.pt \
  --run_name baseline_main_s45_to5k

echo "[DONE ] $(date '+%F %T') baseline_main_s45_to5k"
