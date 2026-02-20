# Go2 PHM 실행/연동 통합 가이드 (KR)

작성일: 2026-02-20  
목적: 기존 `RUNBOOK_1_2_3_EXECUTION_KR.txt`, `REAL_LOG_INTEGRATION_CODE_PLAYBOOK_KR.txt`를 하나로 통합해 실험 실행과 실로봇 로그 연동 절차를 단일 문서로 관리.

## 1) 메인 트랙 정리

1. `Unitree-Go2-Phm-v1`: 상한선/Teacher 용도 (privileged)
2. `Unitree-Go2-Baseline-v1`: 공정 비교군
3. `Unitree-Go2-RealObs-v1`: 실측 채널 기반 메인 정책 (논문 메인)

핵심 비교 축:
- 성능 비교: Baseline vs RealObs
- 학습 이전: Teacher(Phm) -> Student(RealObs) distillation

## 2) 기본 실행 순서

### 2.1 Teacher 학습
```bash
cd /home/iamjaehka13/unitree_go2_phm/unitree_go2_phm/scripts/rsl_rl
python3 train.py --task Unitree-Go2-Phm-v1 --num_envs 4096 --max_iterations 3000 --headless
```

### 2.2 Baseline 학습
```bash
python3 train.py --task Unitree-Go2-Baseline-v1 --num_envs 4096 --max_iterations 3000 --headless
```

### 2.3 Teacher -> RealObs distillation
```bash
python3 distill_teacher_student.py \
  --teacher_task Unitree-Go2-Phm-v1 \
  --student_task Unitree-Go2-RealObs-v1 \
  --teacher_checkpoint <teacher_ckpt> \
  --num_envs 1024 \
  --num_updates 2000 \
  --steps_per_update 24 \
  --headless
```

### 2.4 RealObs 평가
```bash
python3 evaluate.py --task Unitree-Go2-RealObs-v1 --checkpoint <student_or_train_ckpt> --num_envs 512 --num_episodes 300 --headless
```

## 3) 실로봇 로그 연동 파이프라인

실행 위치: `unitree_go2_phm/scripts/real/`

1. SDK 체크
```bash
python3 check_sdk_setup.py
```

2. 실시간 로그 수집/브리지
- `sdk2_bridge/go2_udp_bridge.cpp`
- `run_governor_live_template.py`

3. Raw -> Replay CSV 변환
```bash
python3 log_to_replay_csv.py --input <raw_log> --output <replay_csv>
```

4. 오프라인 governor 평가
```bash
python3 offline_governor_eval_from_log.py --input <raw_or_csv>
```

5. 시뮬 replay 평가
```bash
cd /home/iamjaehka13/unitree_go2_phm/unitree_go2_phm/scripts/rsl_rl
python3 evaluate_replay.py --task Unitree-Go2-RealObs-v1 --checkpoint <ckpt> --replay_csv <csv>
```

## 4) 상수 캘리브레이션 문서

실로봇 기반 상수 보정 항목은 아래 분리 문서에서 유지:
- `docs/REAL_CONSTANTS_CALIBRATION_KR.txt`

우선순위:
1. 열 모델 계수 (`C_THERMAL_*`, `K_*`)
2. 배터리 sag/OCV 관련 계수
3. 실측 임계치(thermal/voltage) 정렬

## 5) 관련 문서

- 실험 전체: `EXPERIMENT_GUIDE.md`
- 논문 청사진: `docs/PAPER_FINAL_BLUEPRINT_KR.txt`
- 발표 통합본: `docs/PRESENTATION_3IN1_MASTER_KR.txt`
- 실로봇 수집 런북: `third_party/GO2_REAL_LOG_RUNBOOK_KR.txt`
