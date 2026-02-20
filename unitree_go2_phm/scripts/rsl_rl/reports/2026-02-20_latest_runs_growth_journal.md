# 2026-02-20 최신 RealObs 1k 비교

## 실험 구성
- ctrl_s45: `2026-02-20_12-28-03_realobs_single_motor_4k_ctrl_s45_perf`
- critical_exposure_s45: `2026-02-20_12-28-10_realobs_single_motor_4k_critical_exposure_s45_perf`

## 999 최종 비교
- ctrl_s45
  - reward 19.8131, ep_len 928.41, time_out 0.6747, base_contact 0.0000, bad_orientation 0.3256
  - noise_std 0.3095, err_xy 0.2384, err_yaw 0.3944
- critical_exposure_s45
  - reward 19.8131, ep_len 928.41, time_out 0.6747, base_contact 0.0000, bad_orientation 0.3256
  - noise_std 0.3095, err_xy 0.2384, err_yaw 0.3944

## 후반 드리프트 slope (800~999)
- ctrl_s45: bad_orientation_slope=-0.000199/iter, time_out_slope=+0.000195/iter
- critical_exposure_s45: bad_orientation_slope=-0.000199/iter, time_out_slope=+0.000195/iter

## 경계 변화량(평균, post-pre)
- ctrl_s45
  - iter500: noise_std -0.0036, time_out -0.0718, base_contact +0.0000, bad_orientation +0.0722
  - iter800: noise_std -0.0000, time_out -0.0026, base_contact -0.0000, bad_orientation +0.0031
- critical_exposure_s45
  - iter500: noise_std -0.0036, time_out -0.0718, base_contact +0.0000, bad_orientation +0.0722
  - iter800: noise_std -0.0000, time_out -0.0026, base_contact -0.0000, bad_orientation +0.0031

## 산출물
- summary: `unitree_go2_phm/scripts/rsl_rl/reports/figures_2026_02_20_latest/summary.json`
- figure: `unitree_go2_phm/scripts/rsl_rl/reports/figures_2026_02_20_latest/tb_full_trends.png`
- figure: `unitree_go2_phm/scripts/rsl_rl/reports/figures_2026_02_20_latest/tb_boundary_zoom.png`
- figure: `unitree_go2_phm/scripts/rsl_rl/reports/figures_2026_02_20_latest/tb_late_phase.png`