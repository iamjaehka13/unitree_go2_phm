# 2026-02-20 RealObs Growth Journal (상세판)

## 1) 오늘 실험의 질문
- 2/19에 채택한 `std_safe` 설정이 시드가 바뀌어도 재현되는가?
- 경계(1000/2000) 충격 억제 성질이 유지되는가?
- 후반(2600+) 드리프트가 안정적으로 유지되는가?
- 시나리오 평가(Fresh/Used/Aged/Critical)에서 정책의 강점/약점은 무엇인가?

## 2) 실험 구성
- 비교 런
  - `std_safe_s42`: `2026-02-19_10-53-22_realobs_single_motor_3k_std_safe_s42_perf` (기준)
  - `std_safe_s43`: `2026-02-20_00-31-24_realobs_single_motor_3k_std_safe_s43_perf`
  - `std_safe_s44`: `2026-02-20_00-31-34_realobs_single_motor_3k_std_safe_s44_perf`
- 공통 조건
  - task: `Unitree-Go2-RealObs-v1`
  - single-motor random fault 유지
  - 학습 3000 iter, num_envs 4096
  - `--perf_mode`로 학습 (TF32 ON, deterministic OFF)

## 3) 왜 이 세팅으로 재검증했는가
- 2/19 결론: `std_safe`가 base/gate_on 대비 최종 지표와 slope에서 가장 안정적이었다.
- 따라서 2/20은 “새 전략 탐색”이 아니라 “채택 전략의 재현성 검증”에 집중했다.
- 즉, 목표는 성능 최대치 갱신보다 “같은 설정에서 같은 결론이 반복되는지” 확인이었다.

## 4) 학습 스케줄(핵심)
- 커리큘럼
  - velocity cmd ramp: iter 160~500
  - DR ramp: iter 501~1000
  - push ramp: iter 1001~1600
- 탐색 스케줄 (`std_safe`)
  - action std stage2 target = 0.35 (2000+)
  - late rate limiter ON: iter 2500+, segment 20
  - up_factor 1.005, down_factor 1.02
- 로그 확인 결과(2개 런 모두):
  - iter 1000/2000 경계에서 std 급점프는 없음
  - 2500 이후 std가 0.13대에서 0.15대까지 완만히 증가

## 5) 2999 기준 학습 결과 비교
| run | reward | ep_len | time_out | bad_orientation | base_contact | noise_std | err_xy | err_yaw |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| std_safe_s42 | 14.3652 | 464.80 | 0.8356 | 0.1641 | 0.0000 | 0.1532 | 0.2089 | 0.2265 |
| std_safe_s43 | 14.5652 | 462.17 | 0.8341 | 0.1652 | 0.0000 | 0.1534 | 0.1911 | 0.2049 |
| std_safe_s44 | 15.0436 | 464.64 | 0.8307 | 0.1688 | 0.0000 | 0.1496 | 0.1903 | 0.2142 |

### 해석
- 장점
  - s43/s44 모두 `err_xy`, `err_yaw`가 s42보다 개선 (추종 오차 감소).
  - base_contact는 세 런 모두 0.0 유지.
- 주의점
  - time_out은 s43/s44가 s42 대비 소폭 하락.
  - bad_orientation은 s43/s44가 s42 대비 소폭 증가.
  - 즉, “오차는 줄었지만 안전 지표는 약간 악화”되는 트레이드오프가 생김.

## 6) 후반 드리프트(2600~2999) 비교
| run | bad_orientation slope (/iter) | time_out slope (/iter) |
|---|---:|---:|
| std_safe_s42 | +0.000066 | -0.000063 |
| std_safe_s43 | +0.000181 | -0.000183 |
| std_safe_s44 | +0.000154 | -0.000155 |

### 해석
- 2/20 두 시드 모두 후반 드리프트가 s42보다 가파르다.
- 결론적으로 `std_safe` 전략 자체는 재현됐지만, 2/19 s42는 “유리한 샘플”일 가능성도 있다.
- 따라서 단일 시드 최고치만으로 채택 결론을 고정하면 위험하고, 시드 평균 기준으로 결론을 업데이트해야 한다.

## 7) 경계(1000/2000) 변화량
| run | iter1000 noise_std Δ | iter1000 time_out Δ | iter2000 noise_std Δ | iter2000 time_out Δ |
|---|---:|---:|---:|---:|
| std_safe_s42 | -0.0162 | +0.0159 | -0.0026 | -0.0107 |
| std_safe_s43 | -0.0144 | +0.0056 | -0.0039 | -0.0068 |
| std_safe_s44 | -0.0134 | +0.0105 | -0.0021 | -0.0061 |

### 해석
- 세 시드 모두 1000/2000에서 폭발적 불안정(기존 overwrite 점프 형태)은 재발하지 않았다.
- 즉, 경계 충격 억제라는 1차 목표는 유지된다.

## 8) 시나리오 평가 결과(신규: s43/s44)
### s43
- fresh: survival 100.0%, track 0.0368
- used: survival 100.0%, track 0.0373
- aged: survival 99.6%, track 0.0476
- critical: survival 0.0%, track 0.2663

### s44
- fresh: survival 100.0%, track 0.0349
- used: survival 100.0%, track 0.0377
- aged: survival 99.2%, track 0.0490
- critical: survival 0.0%, track 0.2895

### 시드 평균(2개)
- fresh: survival 100.0%, track 0.0359
- used: survival 100.0%, track 0.0375
- aged: survival 99.41%, track 0.0483
- critical: survival 0.0%, track 0.2779

### 해석
- fresh/used/aged까지는 정책이 안정적.
- critical은 두 시드 모두 0% 생존으로 실패 패턴이 일관적.
- 따라서 현재 모델의 핵심 한계는 “extreme critical 구간 대응 부족”으로 명확하다.

## 9) 오늘 결론(의사결정)
- 유지할 것
  - single-motor 실험 프레임
  - `std_safe` 기반 스케줄(경계 충격 억제 효과는 확인됨)
- 수정/보강할 것
  - 성능 판단 기준을 단일 시드 최고치에서 “시드 평균 + slope”로 변경
  - critical 대응을 별도 축으로 분리해 개선(현재 0% 생존 고정)

## 10) 다음 액션(우선순위)
1. `std_safe`로 seed 45/46 추가해 분산 추정 강화.
2. critical 전용 개선 실험:
   - PHM 후반 구간 길이 확장(예: critical/final 노출 길이 증가)
   - 또는 critical 샘플 비중을 late phase에서 점진 상향.
3. 평가 프로토콜 고정:
   - 보고는 항상 `2999 지표 + 2600~2999 slope + scenario 평가` 3종 세트로 통일.

## 11) 산출물 경로
- 학습 비교 요약
  - `unitree_go2_phm/scripts/rsl_rl/reports/figures_2026_02_20/summary.json`
- 그래프
  - `unitree_go2_phm/scripts/rsl_rl/reports/figures_2026_02_20/tb_full_trends.png`
  - `unitree_go2_phm/scripts/rsl_rl/reports/figures_2026_02_20/tb_boundary_zoom.png`
  - `unitree_go2_phm/scripts/rsl_rl/reports/figures_2026_02_20/tb_late_phase.png`
- 평가 결과
  - `unitree_go2_phm/scripts/rsl_rl/outputs/eval_2026_02_20/2026-02-20_00-31-24_realobs_single_motor_3k_std_safe_s43_perf/eval_20260220_113419.json`
  - `unitree_go2_phm/scripts/rsl_rl/outputs/eval_2026_02_20/2026-02-20_00-31-34_realobs_single_motor_3k_std_safe_s44_perf/eval_20260220_114138.json`

## 12) 플레이 영상 (체크포인트별)
### std_safe_s43 (`2026-02-20_00-31-24_realobs_single_motor_3k_std_safe_s43_perf`)
- `unitree_go2_phm/scripts/rsl_rl/logs/rsl_rl/unitree_go2_realobs/2026-02-20_00-31-24_realobs_single_motor_3k_std_safe_s43_perf/videos/play/rl-video-step-0_ckpt1000.mp4`
- `unitree_go2_phm/scripts/rsl_rl/logs/rsl_rl/unitree_go2_realobs/2026-02-20_00-31-24_realobs_single_motor_3k_std_safe_s43_perf/videos/play/rl-video-step-0_ckpt2000.mp4`
- `unitree_go2_phm/scripts/rsl_rl/logs/rsl_rl/unitree_go2_realobs/2026-02-20_00-31-24_realobs_single_motor_3k_std_safe_s43_perf/videos/play/rl-video-step-0_ckpt2600.mp4`
- `unitree_go2_phm/scripts/rsl_rl/logs/rsl_rl/unitree_go2_realobs/2026-02-20_00-31-24_realobs_single_motor_3k_std_safe_s43_perf/videos/play/rl-video-step-0_ckpt2999.mp4`

### std_safe_s44 (`2026-02-20_00-31-34_realobs_single_motor_3k_std_safe_s44_perf`)
- `unitree_go2_phm/scripts/rsl_rl/logs/rsl_rl/unitree_go2_realobs/2026-02-20_00-31-34_realobs_single_motor_3k_std_safe_s44_perf/videos/play/rl-video-step-0_ckpt1000.mp4`
- `unitree_go2_phm/scripts/rsl_rl/logs/rsl_rl/unitree_go2_realobs/2026-02-20_00-31-34_realobs_single_motor_3k_std_safe_s44_perf/videos/play/rl-video-step-0_ckpt2000.mp4`
- `unitree_go2_phm/scripts/rsl_rl/logs/rsl_rl/unitree_go2_realobs/2026-02-20_00-31-34_realobs_single_motor_3k_std_safe_s44_perf/videos/play/rl-video-step-0_ckpt2600.mp4`
- `unitree_go2_phm/scripts/rsl_rl/logs/rsl_rl/unitree_go2_realobs/2026-02-20_00-31-34_realobs_single_motor_3k_std_safe_s44_perf/videos/play/rl-video-step-0_ckpt2999.mp4`

노션 업로드 시 권장 순서:
1. TensorBoard 3장 (`tb_full_trends`, `tb_boundary_zoom`, `tb_late_phase`)
2. S43 checkpoint 4개 영상
3. S44 checkpoint 4개 영상
