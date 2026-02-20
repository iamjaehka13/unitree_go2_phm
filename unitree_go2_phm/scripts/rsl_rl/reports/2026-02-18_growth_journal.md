# 2026-02-18 Growth Journal

## 실험 구성
- 3k ablation (seed 42)
  - e1_std_safe
  - e2_gate_on
  - e3_ramp_soft
- 5k reproducibility
  - e3_ramp_soft_5k_s42
  - e3_ramp_soft_5k_s43

## 왜 이 실험을 했는가
- 2/17까지 확인된 핵심 문제:
  - 경계(1000/2000) 급락
  - 2600+ 후반 드리프트
- 2/18 목표:
  - 3k 구간에서 e1/e2/e3 전략 차이를 정량 검증
  - 가장 유망한 e3를 5k + 다중 시드로 재현성 확인

## 설정 차이 (핵심)
- e1 -> e2
  - `phm_curriculum_use_performance_gate: false -> true`
  - action_std stage2 value: `0.35 -> 0.5`
  - late rate limiter: disable
- e2 -> e3
  - gate off
  - `phm_curriculum_critical_end_iter: 2800 -> 2950` (후반 ramp 완화)

## 3k 최종 비교 (iter 2999)
- e1_std_safe_3k_s42
  - reward 7.5450, ep_len 370.40, time_out 0.4547, base_contact 0.0128, bad_orientation 0.5327
- e2_gate_on_3k_s42
  - reward 19.3260, ep_len 763.60, time_out 0.5279, base_contact 0.0138, bad_orientation 0.4583
- e3_ramp_soft_3k_s42
  - reward 17.2905, ep_len 710.50, time_out 0.5372, base_contact 0.0214, bad_orientation 0.4416

해석:
- e1은 안전화 의도였지만 성능이 크게 무너짐.
- e2/e3는 성능 회복. e3는 bad_orientation/time_out 균형이 좋지만 base_contact 증가 기울기 관리가 필요.

## 5k 재현성 (e3 seed42 vs seed43, iter 4999)
- s42: reward 18.0724, ep_len 723.56, time_out 0.5422, base_contact 0.0083, bad_orientation 0.4497
- s43: reward 19.9792, ep_len 690.86, time_out 0.5293, base_contact 0.0188, bad_orientation 0.4521

해석:
- 두 seed 모두 5k에서 수렴 양상은 유사.
- 다만 s43는 base_contact drift가 더 큼 (후반 접촉 리스크).

## 결론 (2/18)
- 3k 기준: e1 탈락, e2/e3 경쟁.
- 5k 기준: e3는 재현성은 확보했지만 seed별 base_contact 편차가 존재.
- 다음 단계: e3 기반으로 base_contact drift 억제(보상/게이트 조건 강화) + bad_orientation/time_out 유지.

## 의사결정 로그 (왜 이렇게 했고, 왜 바꿨는가)
### 문제 정의
- 2/17까지의 핵심 병목은 두 가지였다.
  - 경계(1000/2000)에서 학습 곡선 급락
  - 2600+ 후반 구간에서 drift 누적 (특히 contact/orientation 축)

### 가설 -> 실험 -> 결과
- 가설 A: late std를 안전하게 제한하면 후반 붕괴를 막을 수 있다.
  - 실험: e1_std_safe
  - 결과: 안정화는 되었지만 locomotion 성능 자체가 크게 희생됨 (reward/ep_len 하락).
  - 판단: 과도한 안정화는 목표(강건 보행)와 충돌.
- 가설 B: performance gate로 난이도 진행을 제어하면 성능과 안전을 함께 잡을 수 있다.
  - 실험: e2_gate_on
  - 결과: 3k 단기 지표는 개선됐지만 후반 slope는 악화 신호를 보여 장기 연장 리스크가 큼.
  - 판단: 단기 개선 착시 가능성 존재, 그대로 5k 확장하기엔 위험.
- 가설 C: 난이도 램프 자체를 완화하면 경계 충격/후반 drift를 동시에 줄일 수 있다.
  - 실험: e3_ramp_soft
  - 결과: 3k 지표는 상위권, 5k 재현성도 확보. 다만 seed별 base_contact 편차가 남음.
  - 판단: 현재 best trade-off로 채택.

### 채택/폐기 결정
- 폐기: e1 (성능 손실이 큼)
- 보류: e2 (단기점수는 좋지만 장기 추세 리스크)
- 채택: e3 (3k/5k 균형이 가장 좋음)

## 다음 액션 (실행 단위)
- e3를 기준선으로 고정하고, 목표를 "후반 base_contact 편차 축소"로 재설정.
- 같은 설정에서 다중 seed를 추가로 확보해 분산을 먼저 계량.
- 분산이 큰 경우에만 게이트 조건을 재도입하되, 평가 기준을 slope 중심으로 유지.
