# 2026-02-22 실험 운영 체크포인트 (논문용)

## 문서 목적
- 해보면서 맞추되, 논문 방어 가능한 방식으로 실험을 고정/자동화한다.
- 의사결정을 train 로그가 아닌 12모터 고정 스윕 평가로 통일한다.

## 핵심 운영 원칙
- [ ] Dev(탐색)와 Paper(확정) 실험 분리
- [ ] max_iterations는 넉넉히, 종료는 early-stop
- [ ] 최종 보고는 best checkpoint 기준

## 메인 프로토콜 (변경 금지)
- Task: `Unitree-Go2-RealObs-v1`
- Train fault: `single_motor_random`
- Eval fault: `single_motor_fixed` + id `0..11`
- Scenario: `fresh/used/aged/critical`
- Curriculum: velocity(160~500), DR(501~1000), push(1001~1600), PHM(1601~3000), 이후 final mixture 고정
- PPO: entropy `0.01 -> 0.005 -> 0.003`, action std cap-only + late limiter

## 체크포인트 게이트 (CP0~CP5)
- [ ] CP0 실행 준비: seed/경로/strict 모드 고정
- [ ] CP1 3000 도달: 커리큘럼 종료(수렴 판정 시작점)
- [ ] CP2 4000 도달: 고정 분포 적응 1차 확인
- [ ] CP3 5000 도달: 논문 최소선, 12모터 스윕 + best ckpt 선택
- [ ] CP4 5000+ 연장판단: 개선 있으면 6000~7000, 없으면 early-stop
- [ ] CP5 산출물 확정: seed 통계/heatmap/TB/영상

## 평가/점수/조기중단
- 평가 주기: `eval_interval=100`
- 매 평가: motor id 0..11 고정 스윕
- 지표: `time_out`, `bad_orientation`, `base_contact`, `tracking error`

권장 점수:
- `TO_mean = mean_i(time_out_i)`
- `TO_min = min_i(time_out_i)`
- `BO_mean = mean_i(bad_orientation_i)`
- `Score = 0.6*TO_min + 0.4*TO_mean - 0.8*BO_mean`

조기중단:
- [ ] `W=10` 평가 window
- [ ] Plateau: 최근 W에서 best 개선 < 0.005 and slope ~= 0
- [ ] Regression: TO_min이 best 대비 0.02 이상 하락 2회 연속

## 3000+ 허용 조정 (2개만)
- [ ] LR 안정화: 3000+ 또는 plateau 시 lr 0.5x (필요시 1회 추가)
- [ ] worst-motor 노출: 60~70% uniform + 30~40% recent worst top-k

## 이번 주 실행 TODO
- [ ] 5000+ 장기 run 2개(RealObs/Baseline)
- [ ] eval sweep 자동 루프(100 iter)
- [ ] best checkpoint + early-stop 로그 확정
- [ ] 12모터 heatmap/표 자동 생성
- [ ] 노션에 TB/영상/의사결정 이유까지 업로드

## 원본 문서
- `해야할것.txt`
