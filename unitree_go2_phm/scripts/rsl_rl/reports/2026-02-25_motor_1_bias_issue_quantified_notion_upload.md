# 모터 1- : Motor ID 편향 이슈 (정량 근거 보강본)

## 문서 목적
- "초기 실험에서 Motor 10이 나빴다"를 문장으로만 적지 않고, 실제 수치로 어디가 얼마나 나빴는지 명확히 제시한다.
- 정책 자체 성능 저하와 motor location 편향을 분리해 해석 가능한 기준을 만든다.

## 데이터 출처
- `unitree_go2_phm/scripts/rsl_rl/eval_results/reval_motorfix_from_env_6996_motor_sweep_standard.json`
- `unitree_go2_phm/scripts/rsl_rl/eval_results/reval_motorfix_forced_walk_6996_motor_sweep_standard.json`
- `unitree_go2_phm/scripts/rsl_rl/eval_results/critical_survival_per_id_forced_walk_vs_yaw0_6996.csv`

## 1) 무엇이 "나빴는지" (핵심 정량 증거)

### 표 1. 가장 문제가 컸던 케이스 (Baseline, from_env, critical, survived)
| 항목 | 값 |
|---|---:|
| mean | 0.4383 |
| std | 0.3829 |
| min | 0.0000 |
| p10 | 0.0000 |
| worst motor | id=10, value=0.0000 |
| id7 value | 1.0000 |
| id10 value | 0.0000 |
| gap(7-10) | 1.0000 |

해석:
- 동일 조건에서 `id7=1.0`, `id10=0.0`은 단순 노이즈라고 보기 어려운 수준의 비대칭이다.
- 평균이 0.4383인데 표준편차가 0.3829로 매우 커서, "모터 위치에 따라 결과가 크게 출렁인다"는 것이 핵심 문제다.

### 표 2. per-id 벡터 (같은 케이스)
`[id0..11] = [0.20, 0.56, 0.36, 0.67, 0.57, 0.91, 0.99, 1.00, 0.00, 0.00, 0.00, 0.00]`

해석:
- "10번만" 나쁜 것이 아니라, 후반 모터 구간(8~11)에서 붕괴가 동반되며 특히 10번이 미러(7번) 대비 최악으로 나타난다.
- 즉 문제는 단일 포인트 오류가 아니라, 위치/분포/명령 결합에 의해 생긴 구조적 편향이다.

## 2) 다른 설정에서도 일관되게 보였는가

### 표 3. 설정별 비교 (요약)
| 설정 | 모델 | 시나리오/지표 | mean | std | worst(id,val) | id7 | id10 | gap(7-10) |
|---|---|---|---:|---:|---|---:|---:|---:|
| from_env | baseline | critical/survived | 0.4383 | 0.3829 | 10, 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| from_env | baseline | critical/survival_walk_only | 0.4697 | 0.3971 | 10, 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| from_env | realobs | critical/walk_ratio | 0.4275 | 0.1392 | 10, 0.1100 | 0.6200 | 0.1100 | 0.5100 |
| forced_walk | baseline | aged/survived | 0.6883 | 0.3533 | 8, 0.0000 | 0.9500 | 0.4900 | 0.4600 |
| forced_walk | realobs | aged/survived | 0.0717 | 0.1002 | 10, 0.0000 | 0.0800 | 0.0000 | 0.0800 |

해석:
- 문제 신호는 여러 설정에서 반복된다. 특히 `from_env critical baseline`에서 가장 명확하다.
- `forced_walk aged`에서도 `gap(7-10)=0.46`가 확인되어, 단일 실험 우연이 아니라 패턴으로 나타난다.

## 3) "왜 문제인지"를 정량으로 설명
- 문제 1: 모터 위치에 따라 성능 분산이 과도함
  - 예: from_env critical baseline `std=0.3829` (mean=0.4383 대비 매우 큼)
- 문제 2: 미러 페어 비대칭이 큼
  - 예: `gap(7-10)=1.0` (동일 thigh 그룹 좌우가 완전히 갈림)
- 문제 3: worst-case가 바닥으로 떨어짐
  - `min=0`, `p10=0`이면 일부 모터에선 사실상 실패 확정 구조

## 4) 원인 가설 (데이터와 연결)
- 원인 A. fixed motor-id 중심 해석의 confound
  - 단일 id 결과만 보면 정책 일반 성능과 위치 취약성이 분리되지 않는다.
- 원인 B. 학습 step 노출량 불균등
  - 빠르게 종료되는 모터는 누적 학습 기여가 줄어들어 편향이 고착될 수 있다.
- 원인 C. 명령 분포/회전 결합
  - from_env와 forced_walk에서 편향 패턴이 다르게 보이는 것은 명령 분포 영향이 있음을 시사한다.

## 5) 우리가 적용한 해결 (문장 아닌 실행 항목)
- 학습 샘플링을 `미러-균등 + step-hold`로 변경
  - 미러 페어 균등 선택 + 좌우 50:50
  - hold 구간 동안 reset되어도 같은 motor 유지
- 평가를 `motor sweep(0..11)` 표준으로 변경
  - mean/std/min/p10/worst id/pair_gap(7-10) 동시 보고
- 프로토콜 분리
  - locomotion 해석과 safety 해석을 분리해 결론 혼선을 줄임

## 6) 해결 성공 판정 기준 (수치 기반)
- `gap(7-10)` 감소 (핵심)
- `std` 감소
- `min/p10` 악화 없이 유지 또는 개선
- 평균만 개선되는 것이 아니라 worst-case도 함께 개선

## 7) 이번 문서의 결론
- "Motor 10이 나빴다"는 표현은 불충분했다.
- 정확한 표현은 다음과 같다:
  - 특정 설정에서 모터 위치 편향이 크게 나타났고,
  - 그 중 `id10`이 미러 대비 최악 신호를 보였으며,
  - 이는 단일 모터 결론이 아니라 분포/노출/명령 결합이 만든 구조적 문제였다.
- 따라서 해결도 단일 튜닝이 아니라 샘플링 구조와 평가 프로토콜을 동시에 바꾸는 방식이 필요했다.
