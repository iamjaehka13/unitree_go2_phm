# 2026-02-16~19 Decision Log (Reason -> Result -> Next)

## 2/16
- 왜:
  - baseline 첫 시도에서 후반 붕괴가 나타났고(orientation/contact 악화), 동일 task를 안정화 재시도해야 했다.
- 무엇을 바꿈:
  - 커리큘럼 타이밍 지연, DR/노이즈/질량 랜덤화 조정, staged std/entropy 스케줄 도입.
- 결과:
  - 생존성은 크게 개선(time_out 상승)했지만 경계(1000/2000)에서 급격한 충격 발생.
- 다음:
  - std 제어를 overwrite 방식에서 완만 제어로 바꾸는 후속 실험 필요.

## 2/17
- 왜:
  - 2/16에서 경계 충격 + 후반 불안정 문제가 동시 존재.
- 무엇을 바꿈:
  - late std limiter를 약/강 두 버전(R2/R3)으로 비교.
- 결과:
  - 약한 limiter는 일부 개선, 강한 limiter는 posture 붕괴를 유발.
- 다음:
  - 공격적 std 복원보다 커리큘럼 램프 완화/게이트 등 구조적 제어로 전환.

## 2/18
- 왜:
  - std만 만지는 접근의 한계를 확인해, e1/e2/e3를 분리된 가설로 비교.
- 무엇을 바꿈:
  - e1: 안전화 중심
  - e2: gate 기반 진행 제어
  - e3: 후반 ramp 완화
- 결과:
  - e1은 성능 손실, e2는 단기 개선 대비 장기 리스크, e3는 3k/5k 균형이 최선.
- 다음:
  - e3를 기준선으로 유지하고, seed별 base_contact 편차 억제를 목표로 수정.

## 2/19
- 왜:
  - 12모터 동시 열화는 난이도가 과도해 정책 비교 해석이 어려웠다.
- 무엇을 바꿈:
  - single-motor fault로 문제를 축소하고 base/gate_on/std_safe를 동일 budget(3k)에서 비교.
- 결과:
  - std_safe가 최종 지표/후반 slope/오차에서 가장 안정적으로 우세.
- 다음:
  - std_safe 기반으로 real-log 상수 보정 -> fault 위치 일반화 ablation 순으로 확장.

## 현재 채택안
- Baseline lineage: 2/18 e3_ramp_soft
- RealObs lineage: 2/19 std_safe

## 남은 리스크
- seed 간 base_contact 편차
- single-motor 설정에서 multi-fault/generalization으로 확장 시 성능 유지 여부

## 즉시 실행 항목
1. std_safe + single-motor 설정으로 추가 seed 러닝
2. 동일 설정에서 fault index randomization ablation
3. real log 기반 상수 보정 및 sim-to-real 리포트 템플릿 고정
