# 2월 연구 데일리 흐름 정리 (자동 업로드)

## 데일리 스토리 (왜 다음 실험을 했는가)
- 2/16: Baseline 1차에서 후반 안정성 붕괴(time_out 하락, bad_orientation 상승)가 확인되어, 커리큘럼/랜덤화 완화 버전 Run-2를 재실행했다.
- 2/17: 2/16에서 남은 문제(1000/2000 경계 충격, 2600+ 드리프트)를 분리하려고 3개 전략(e1/e2/e3) ablation을 돌려 원인 분해를 진행했다.
- 2/18: e3_ramp_soft가 3k/5k 균형이 가장 좋아 기준선으로 채택했고, 이후 실험은 “후반 drift 억제” 중심으로 전환했다.
- 2/19: 12모터 동시 열화는 학습 난도가 과도해 정책이 생존 편향으로 가는 문제가 커서, single-motor fault로 문제를 재정의하고 base/gate/std_safe를 다시 비교했다.
- 2/20(새벽): std_safe 재현성(s43/s44)을 검증해 경계 충격 억제는 유지됨을 확인했지만, critical 시나리오 생존 0% 한계가 명확해졌다.
- 2/20(추가): 최신 s45 실험에서 ctrl vs critical_exposure를 1k 구간으로 빠르게 비교해, 이후 real-data 연동 전 “critical 대응 개선축”을 어디에 둘지 판단하려고 했다.

## 오늘 결론 요약
- 경계(1000/2000) 급점프 이슈는 구조적으로 완화되었다.
- fresh/used는 걷기 안정성이 유지되지만, critical은 여전히 취약하다.
- 다음 액션은 critical 대응 강화 + real log 상수 보정 결합이다.

## TensorBoard (최신 s45 비교)
- `unitree_go2_phm/scripts/rsl_rl/reports/figures_2026_02_20_latest/tb_full_trends.png`
- `unitree_go2_phm/scripts/rsl_rl/reports/figures_2026_02_20_latest/tb_boundary_zoom.png`
- `unitree_go2_phm/scripts/rsl_rl/reports/figures_2026_02_20_latest/tb_late_phase.png`

## 걷는 영상 (서있는 영상 대체본)
- ctrl_s45 fresh walk: `unitree_go2_phm/scripts/rsl_rl/logs/rsl_rl/unitree_go2_realobs/2026-02-20_12-28-03_realobs_single_motor_4k_ctrl_s45_perf/videos/play/walk_fresh_ckpt999.mp4`
- ctrl_s45 used walk: `unitree_go2_phm/scripts/rsl_rl/logs/rsl_rl/unitree_go2_realobs/2026-02-20_12-28-03_realobs_single_motor_4k_ctrl_s45_perf/videos/play/walk_used_ckpt999.mp4`
- ctrl_s45 critical walk: `unitree_go2_phm/scripts/rsl_rl/logs/rsl_rl/unitree_go2_realobs/2026-02-20_12-28-03_realobs_single_motor_4k_ctrl_s45_perf/videos/play/walk_critical_ckpt999.mp4`
- critical_exposure_s45 fresh walk: `unitree_go2_phm/scripts/rsl_rl/logs/rsl_rl/unitree_go2_realobs/2026-02-20_12-28-10_realobs_single_motor_4k_critical_exposure_s45_perf/videos/play/walk_fresh_ckpt999.mp4`
- critical_exposure_s45 used walk: `unitree_go2_phm/scripts/rsl_rl/logs/rsl_rl/unitree_go2_realobs/2026-02-20_12-28-10_realobs_single_motor_4k_critical_exposure_s45_perf/videos/play/walk_used_ckpt999.mp4`
- critical_exposure_s45 critical walk: `unitree_go2_phm/scripts/rsl_rl/logs/rsl_rl/unitree_go2_realobs/2026-02-20_12-28-10_realobs_single_motor_4k_critical_exposure_s45_perf/videos/play/walk_critical_ckpt999.mp4`

## 상세 리포트
- `unitree_go2_phm/scripts/rsl_rl/reports/2026-02-20_latest_runs_growth_journal.md`
- `unitree_go2_phm/scripts/rsl_rl/reports/2026-02-20_realobs_growth_journal.md`
