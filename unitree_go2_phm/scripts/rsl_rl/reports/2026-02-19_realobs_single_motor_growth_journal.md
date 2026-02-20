# 2026-02-19 RealObs Single-Motor Growth Journal

## 실험 구성
- realobs_single_motor_3k_base_s42: `2026-02-19_03-41-22_realobs_single_motor_3k_s42_perf`
- realobs_single_motor_3k_gate_on_s42: `2026-02-19_10-53-13_realobs_single_motor_3k_gate_on_s42_perf`
- realobs_single_motor_3k_std_safe_s42: `2026-02-19_10-53-22_realobs_single_motor_3k_std_safe_s42_perf`

## 왜 이 실험을 했는가
- 기존 12모터 동시 열화 설정은 학습 난이도가 너무 높아, 정책이 "의미 있는 fault-tolerant 전략" 대신 생존 편향으로 수렴하는 문제가 있었다.
- 그래서 2/19는 문제를 축소해 single-motor fault로 재정의하고, 다음 질문을 검증했다.
  - Q1: 같은 3k budget에서 어떤 제어전략이 가장 안정적으로 수렴하는가?
  - Q2: 경계(1000/2000)와 후반(2600+)에서 어떤 전략이 덜 무너지는가?

## 전략별 의도
- base: 가장 단순한 기준선. 비교 기준(anchor) 확보 목적.
- gate_on: 난이도 진행을 성능 기반으로 제어해 후반 붕괴를 줄일 수 있는지 확인.
- std_safe: 탐색 분산을 보수적으로 유지해 경계/후반 드리프트를 완화할 수 있는지 확인.

## 2999 최종 비교
- realobs_single_motor_3k_base_s42
  - reward 8.1128, ep_len 451.14, time_out 0.6571, base_contact 0.0000, bad_orientation 0.3346
  - noise_std 0.2937, err_xy 0.2499, err_yaw 0.3192
- realobs_single_motor_3k_gate_on_s42
  - reward 9.2069, ep_len 460.09, time_out 0.7673, base_contact 0.0005, bad_orientation 0.2334
  - noise_std 0.2866, err_xy 0.2667, err_yaw 0.3417
- realobs_single_motor_3k_std_safe_s42
  - reward 14.3652, ep_len 464.80, time_out 0.8356, base_contact 0.0000, bad_orientation 0.1641
  - noise_std 0.1532, err_xy 0.2089, err_yaw 0.2265

## 후반 드리프트 slope (2600~2999)
- realobs_single_motor_3k_base_s42: bad_orientation_slope=+0.000333/iter, time_out_slope=-0.000353/iter
- realobs_single_motor_3k_gate_on_s42: bad_orientation_slope=+0.000211/iter, time_out_slope=-0.000213/iter
- realobs_single_motor_3k_std_safe_s42: bad_orientation_slope=+0.000066/iter, time_out_slope=-0.000063/iter

## 경계 변화량(평균, post-pre)
- realobs_single_motor_3k_base_s42
  - iter1000: noise_std -0.0162, time_out +0.0159, base_contact +0.0000, bad_orientation -0.0160
  - iter2000: noise_std -0.0026, time_out -0.0107, base_contact +0.0000, bad_orientation +0.0115
- realobs_single_motor_3k_gate_on_s42
  - iter1000: noise_std -0.0162, time_out +0.0159, base_contact +0.0000, bad_orientation -0.0160
  - iter2000: noise_std -0.0045, time_out +0.0000, base_contact +0.0000, bad_orientation -0.0000
- realobs_single_motor_3k_std_safe_s42
  - iter1000: noise_std -0.0162, time_out +0.0159, base_contact +0.0000, bad_orientation -0.0160
  - iter2000: noise_std -0.0026, time_out -0.0107, base_contact +0.0000, bad_orientation +0.0115

## 결론 초안
- single-motor 설정에서 3개 전략의 차이를 동일한 3k budget에서 비교 완료.
- 최종 지표 + 2600+ slope + 1000/2000 경계 변화를 함께 보고 다음 학습안(보상/게이트/STD)을 결정.

## 의사결정 결론 (2/19)
- 채택: `std_safe`
  - 근거: 2999 시점과 2600+ slope 모두에서 가장 안정적이며, tracking error도 가장 낮음.
- 보류: `gate_on`
  - 근거: base 대비 개선은 있으나 std_safe 대비 성능/오차 모두 열세.
- 기준선 유지: `base`
  - 근거: 향후 변경 효과를 판단하기 위한 최소 비교군으로는 필요.

## 다음 대처 (구체)
- 실험 축을 single-motor로 유지한 채, 우선 `std_safe` 기반으로 real-log 상수 보정부터 진행.
- 이후 fault 위치 일반화(모터 index 랜덤) 여부를 ablation으로 추가해 "특정 모터 과적합" 여부를 확인.
- 논문 문장도 다음처럼 고정:
  - 2/19는 "문제 난이도 축소를 통한 학습 안정화 단계"
  - 그 다음 단계는 "single-motor에서 검증된 정책을 더 넓은 fault 분포로 확장"
