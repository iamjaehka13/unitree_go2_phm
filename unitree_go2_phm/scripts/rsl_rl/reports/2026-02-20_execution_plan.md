# 2026-02-20 Execution Plan (RealObs Single-Motor, std_safe mainline)

## 0) 오늘 목표
- 2/19에서 가장 좋았던 `std_safe` 세팅을 기준으로 재현성 검증(시드 확장) 진행.
- 경계(1000/2000) 충격 재발 여부 + 후반(2600+) 드리프트를 동일 지표로 비교.
- 주요 체크포인트 영상(1000/2000/2600/2999)까지 확보해서 노션에 바로 반영.

## 1) 기준 세팅(2/19 std_safe와 동일)
- `phm_curriculum_use_performance_gate=False`
- `action_std_stage2_value=0.35`
- `action_std_late_rate_limit_enable=True`
- `action_std_late_rate_limit_start_iter=2500`
- `action_std_late_rate_limit_segment_iters=20`
- `action_std_late_max_up_factor=1.005`
- `action_std_late_max_down_factor=1.02`
- `phm_fault_injection_mode=single_motor_random`
- `max_iterations=3000`, `num_envs=4096`

## 2) 학습 (시드 43, 44)
```bash
cd /home/iamjaehka13/unitree_go2_phm/unitree_go2_phm/scripts/rsl_rl

# seed 43
python3 train.py \
  --task Unitree-Go2-RealObs-v1 \
  --agent rsl_rl_cfg_entry_point \
  --num_envs 4096 \
  --max_iterations 3000 \
  --seed 43 \
  --run_name realobs_single_motor_3k_std_safe_s43_perf \
  --perf_mode \
  --headless \
  env.phm_curriculum_use_performance_gate=False \
  env.phm_fault_injection_mode=single_motor_random \
  agent.action_std_stage2_value=0.35 \
  agent.action_std_late_rate_limit_enable=True \
  agent.action_std_late_rate_limit_start_iter=2500 \
  agent.action_std_late_rate_limit_segment_iters=20 \
  agent.action_std_late_max_up_factor=1.005 \
  agent.action_std_late_max_down_factor=1.02

# seed 44
python3 train.py \
  --task Unitree-Go2-RealObs-v1 \
  --agent rsl_rl_cfg_entry_point \
  --num_envs 4096 \
  --max_iterations 3000 \
  --seed 44 \
  --run_name realobs_single_motor_3k_std_safe_s44_perf \
  --perf_mode \
  --headless \
  env.phm_curriculum_use_performance_gate=False \
  env.phm_fault_injection_mode=single_motor_random \
  agent.action_std_stage2_value=0.35 \
  agent.action_std_late_rate_limit_enable=True \
  agent.action_std_late_rate_limit_start_iter=2500 \
  agent.action_std_late_rate_limit_segment_iters=20 \
  agent.action_std_late_max_up_factor=1.005 \
  agent.action_std_late_max_down_factor=1.02
```

## 3) 정량 평가 (각 런 model_2999)
```bash
cd /home/iamjaehka13/unitree_go2_phm/unitree_go2_phm/scripts/rsl_rl

python3 evaluate.py \
  --task Unitree-Go2-RealObs-v1 \
  --agent rsl_rl_cfg_entry_point \
  --checkpoint logs/rsl_rl/unitree_go2_realobs/<RUN_S43>/model_2999.pt \
  --num_envs 512 \
  --num_episodes 300 \
  --seed 43 \
  --output_dir outputs/eval_2026_02_20/<RUN_S43> \
  --headless

python3 evaluate.py \
  --task Unitree-Go2-RealObs-v1 \
  --agent rsl_rl_cfg_entry_point \
  --checkpoint logs/rsl_rl/unitree_go2_realobs/<RUN_S44>/model_2999.pt \
  --num_envs 512 \
  --num_episodes 300 \
  --seed 44 \
  --output_dir outputs/eval_2026_02_20/<RUN_S44> \
  --headless
```

## 4) 영상 확보 (체크포인트 1000/2000/2600/2999)
```bash
cd /home/iamjaehka13/unitree_go2_phm/unitree_go2_phm/scripts/rsl_rl

# 예시: seed 43 런
for CKPT in 1000 2000 2600 2999; do
  python3 play.py \
    --task Unitree-Go2-RealObs-v1 \
    --agent rsl_rl_cfg_entry_point \
    --checkpoint logs/rsl_rl/unitree_go2_realobs/<RUN_S43>/model_${CKPT}.pt \
    --num_envs 1 \
    --video \
    --video_length 500 \
    --disable_fabric \
    --force_walk_command \
    --play_cmd_lin_x 0.6 \
    --play_cmd_lin_y 0.0 \
    --play_cmd_ang_z 0.0 \
    --force_fault_scenario critical \
    --fault_motor_color_vis \
    --headless
done
```

## 5) 텐서보드 비교 리포트 생성
```bash
cd /home/iamjaehka13/unitree_go2_phm

python3 unitree_go2_phm/scripts/rsl_rl/reports/gen_realobs_run_report.py \
  --root unitree_go2_phm/scripts/rsl_rl/logs/rsl_rl/unitree_go2_realobs \
  --run std_safe_s42=2026-02-19_10-53-22_realobs_single_motor_3k_std_safe_s42_perf \
  --run std_safe_s43=<RUN_S43> \
  --run std_safe_s44=<RUN_S44> \
  --out_dir unitree_go2_phm/scripts/rsl_rl/reports/figures_2026_02_20 \
  --out_md unitree_go2_phm/scripts/rsl_rl/reports/2026-02-20_realobs_growth_journal.md \
  --title "2026-02-20 RealObs Growth Journal"
```

## 6) 성공 기준 (오늘 종료 조건)
- `time_out` 평균: s43/s44가 s42 대비 `-0.03` 이내.
- `bad_orientation` 평균: s43/s44가 s42 대비 `+0.03` 이내.
- 후반 slope(2600~2999):
  - `bad_orientation_slope <= +0.00010/iter`
  - `time_out_slope >= -0.00010/iter`
- 영상에서 `critical + forced walk` 조건에서 완전 정지(standing-only) 패턴이 반복되지 않을 것.

