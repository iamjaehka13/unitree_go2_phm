# PHM-Aware Locomotion: 실험 실행 가이드

## 개요

이 문서는 논문용 실험 데이터를 수집하기 위한 **전체 파이프라인**을 설명합니다.

권장 메인 트랙:
- 실로봇 전이/재현성을 목표로 할 경우 `Step 6 (Unitree-Go2-RealObs-v1 + fixed replay)`를 메인 결과로 사용하세요.
- `Step 1~5`의 시나리오 기반 평가는 시뮬레이션 분석용 보조 자료로 두는 것이 안전합니다.

## PPO/DR 범위 근거 (논문에 그대로 기재할 기준)

이 프로젝트의 PPO 기본값과 커리큘럼/DR 범위는 아래 문헌을 기준으로 정했습니다.

1. **Learning to Walk in Minutes (ANYmal, arXiv:2109.11978)**
   - URL: https://arxiv.org/abs/2109.11978
   - 사용 근거: `num_envs=4096`, 짧은 rollout 기반 대규모 병렬 학습, curriculum + DR 철학
2. **legged_gym (ETH 공개 구현, 2109.11978 계열)**
   - URL: https://github.com/leggedrobotics/legged_gym
   - 사용 근거: PPO 실무 기본값(`n_steps=24`, entropy 계열 기본값)과 설정 관행
3. **Curriculum Jumping for Quadrupeds (arXiv:2401.16337)**
   - URL: https://arxiv.org/abs/2401.16337
   - 사용 근거: DR 상한선(질량/지연/마찰/오프셋) 참고. 단, 점프 과제이므로 본 프로젝트 보행에서는 더 보수적으로 축소 적용

### 우리 프로젝트에 실제로 반영한 매핑

| 항목 | 프로젝트 값(기준) | 출처 | 반영 방식 |
|------|--------------------|------|----------|
| `num_envs` | 4096 | 2109.11978, legged_gym | 동일 채택 |
| rollout (`n_steps`) | 24 | legged_gym | 동일 채택 |
| entropy | 시작 0.01, 후반 감소 | 2109.11978/legged_gym 관행 | 안정화 목적 스케줄 적용 |
| friction DR | 대략 0.5~1.3대 | 2109.11978, 2401.16337 | Go2 보행 안정성 기준으로 완화 |
| mass DR | 대략 0.8~1.2대 | 2401.16337 | 점프 과제 대비 축소 |
| latency/cmd transport | 저강도에서 시작해 점증 | 2401.16337 + sim2real 관행 | curriculum으로 단계 적용 |
| disturbance push | 초기 미적용, 후반 도입 | 2109.11978 | 학습 붕괴 방지용 지연 투입 |

### 논문 본문/부록에 넣을 권장 문구

- 본문(Training Details):
  - "PPO hyperparameters follow the commonly used Isaac Gym legged locomotion settings (e.g., 4096 parallel environments and short-horizon rollouts) from prior work [2109.11978] and its public implementation in legged_gym."
- 부록(Reproducibility):
  - "Domain-randomization upper bounds were referenced from prior sim-to-real quadruped studies (e.g., [2401.16337]) and conservatively reduced for stable Go2 walking."

---

## Step 0: 사전 확인

```bash
# Isaac Lab 환경이 활성화되어 있는지 확인
which python  # Isaac Lab Python이어야 함

# 패키지가 설치되어 있는지 확인
python -c "import unitree_go2_phm; print('OK')"
```

---

## Step 1: 학습 (Training) — 약 3~6시간/정책

### 1-A. PHM-Aware 정책 학습 (메인)

```bash
cd /home/iamjaehka13/unitree_go2_phm/unitree_go2_phm/scripts/rsl_rl

python train.py \
    --task Unitree-Go2-Phm-v1 \
    --num_envs 4096 \
    --max_iterations 3000 \
    --headless
```

- **로그 위치**: `logs/rsl_rl/unitree_go2_phm_strategic/<timestamp>/`
- **체크포인트**: 위 디렉토리 안 `model_*.pt` 파일
- **TensorBoard**: `tensorboard --logdir logs/rsl_rl/unitree_go2_phm_strategic`

### 1-B. Baseline 정책 학습 (비교군)

```bash
python train.py \
    --task Unitree-Go2-Baseline-v1 \
    --num_envs 4096 \
    --max_iterations 3000 \
    --headless
```

- **로그 위치**: `logs/rsl_rl/unitree_go2_baseline/<timestamp>/`

### 1-C. (선택) Seed 3회 반복

통계적 유의성을 위해 각 정책을 seed 3개로 학습:

```bash
# PHM-Aware: seed 42, 123, 456
python train.py --task Unitree-Go2-Phm-v1 --num_envs 4096 --max_iterations 3000 --headless --seed 42
python train.py --task Unitree-Go2-Phm-v1 --num_envs 4096 --max_iterations 3000 --headless --seed 123
python train.py --task Unitree-Go2-Phm-v1 --num_envs 4096 --max_iterations 3000 --headless --seed 456

# Baseline: seed 42, 123, 456
python train.py --task Unitree-Go2-Baseline-v1 --num_envs 4096 --max_iterations 3000 --headless --seed 42
python train.py --task Unitree-Go2-Baseline-v1 --num_envs 4096 --max_iterations 3000 --headless --seed 123
python train.py --task Unitree-Go2-Baseline-v1 --num_envs 4096 --max_iterations 3000 --headless --seed 456
```

---

## Step 2: 평가 (Evaluation) — 약 30분/정책

학습이 끝나면 `evaluate.py`로 제어된 열화 시나리오에서 정량 데이터를 수집합니다.

### 2-A. PHM-Aware 정책 평가

```bash
cd /home/iamjaehka13/unitree_go2_phm/unitree_go2_phm/scripts/rsl_rl

python evaluate.py \
    --task Unitree-Go2-Phm-v1 \
    --checkpoint logs/rsl_rl/unitree_go2_phm_strategic/<timestamp>/model_2999.pt \
    --num_envs 512 \
    --num_episodes 100 \
    --output_dir ./eval_results/phm_aware \
    --headless
```

### 2-B. Baseline 정책 평가

**중요**: Baseline은 `Unitree-Go2-Baseline-v1`로 평가합니다.
이 태스크는 PHM-Aware와 동일한 PHM 물리(열화/배터리/고장)를 사용하되 PHM 관측/보상만 제거한 설정이므로 공정 비교가 가능합니다.

```bash
python evaluate.py \
    --task Unitree-Go2-Baseline-v1 \
    --checkpoint logs/rsl_rl/unitree_go2_baseline/<timestamp>/model_2999.pt \
    --num_envs 512 \
    --num_episodes 100 \
    --output_dir ./eval_results/baseline \
    --headless
```

### 2-C. 출력 형식

`evaluate.py`는 **4개 시나리오**(Fresh, Used, Aged, Critical)별로 다음 메트릭을 JSON으로 저장합니다:

| 메트릭 | 의미 | 논문 용도 |
|--------|------|----------|
| `survived` | 에피소드 생존율 (%) | **Table 1 핵심 지표** |
| `mean_tracking_error_xy` | 속도 추적 오차 (m/s) | **Table 1 핵심 지표** |
| `mean_tracking_error_ang` | 각속도 추적 오차 (rad/s) | Table 1 |
| `mean_power` | 평균 전력 소비 (W) | Table 1 효율 비교 |
| `total_energy` | 에피소드 총 에너지 (J) | Table 1 효율 비교 |
| `final_soc` | 에피소드 종료 시 SOC | 배터리 수명 분석 |
| `final_max_temp` | 최대 모터 온도 (°C, legacy key) | 열 안전성 분석 |
| `final_max_temp_coil_hotspot` / `final_max_temp_case_proxy` | task 온도 의미를 명시한 최대 온도 키 | task 간 해석 혼동 방지 |
| `final_max_fatigue` | 최대 피로도 | 기계 수명 분석 |
| `final_min_soh` | 최소 잔여 수명 (SOH) | PHM 핵심 메트릭 |
| `max_saturation` | 토크 포화율 | 액추에이터 부하 |

---

## Step 3: TensorBoard에서 학습 곡선 수집

### 3-A. TensorBoard 실행

```bash
tensorboard --logdir logs/rsl_rl/ --port 6006
```

### 3-B. 수집할 학습 곡선 (Figure 용)

| TensorBoard 태그 | 논문 그래프 | 설명 |
|-------------------|------------|------|
| `Train/mean_reward` | **Fig.2(a)** 학습 곡선 | Baseline vs PHM-Aware 수렴 비교 |
| `Train/mean_episode_length` | **Fig.2(b)** 생존 시간 | 열화 환경에서 에피소드 길이 |
| `Reward/track_lin_vel_xy` | Fig.3 추적 성능 | 속도 추적 보상 |
| `Reward/energy_efficiency` | Fig.3 에너지 효율 | PHM-Aware만 해당 |
| `phm/avg_temp` | Fig.4(a) 온도 추이 | 학습 중 평균 코일 온도 |
| `phm/max_fatigue` | Fig.4(b) 피로도 추이 | 학습 중 최대 피로도 |
| `phm/min_voltage` | Fig.4(c) 전압 추이 | 최소 배터리 전압 |

**TensorBoard → CSV 내보내기**: 각 그래프 우상단 다운로드 버튼으로 CSV 추출 가능.

---

## Step 4: 논문 테이블/그래프 구성

### Table 1: Main Result (핵심 실험)

```
+------------+-------------------+-------------------+
|            |    Baseline       |   PHM-Aware       |
| Scenario   | Surv% | TrackErr | Surv% | TrackErr  |
+------------+-------+----------+-------+-----------+
| Fresh      | 98.0  | 0.15     | 97.5  | 0.16      |  ← 비슷해야 함
| Used       | 85.0  | 0.25     | 92.0  | 0.20      |  ← PHM 우위 시작
| Aged       | 55.0  | 0.45     | 78.0  | 0.30      |  ← 격차 확대
| Critical   | 20.0  | 0.80     | 55.0  | 0.50      |  ← 핵심 contribution
+------------+-------+----------+-------+-----------+
```

**이 테이블이 논문의 핵심 contribution을 증명합니다.**

- Fresh에서는 둘 다 비슷 → "PHM 관측이 건강한 로봇에 해를 끼치지 않음"
- Critical에서 PHM-Aware가 우위 → "열화 상태에서 PHM 관측이 적응적 행동을 유도"

### Table 2: Energy & Thermal Efficiency

```
+------------+-------------------+-------------------+
|            |    Baseline       |   PHM-Aware       |
| Scenario   | Power(W) | MaxT  | Power(W) | MaxT   |
+------------+----------+-------+----------+--------+
| Fresh      |  120     | 35°C  |  115     | 34°C   |
| Aged       |  180     | 72°C  |  140     | 60°C   |
| Critical   |  250     | 88°C  |  160     | 75°C   |
+------------+----------+-------+----------+--------+
```

Note:
- Table 2 values above are illustrative for the privileged PHM track.
- For Step 6 (`Unitree-Go2-RealObs-v1`), report temperature metrics from replay/evaluation outputs with the RealObs safety settings.

### Figure 설계

| Figure | 내용 | 데이터 소스 |
|--------|------|------------|
| **Fig.1** | System Architecture (PHM 시스템 블록 다이어그램) | 직접 그리기 |
| **Fig.2** | 학습 곡선 비교 (reward, episode_length) | TensorBoard CSV |
| **Fig.3** | Bar chart: 시나리오별 생존율 비교 | evaluate.py JSON |
| **Fig.4** | Bar chart: 시나리오별 추적 오차 비교 | evaluate.py JSON |
| **Fig.5** | Line plot: 에피소드 내 온도/SOC 변화 | 추가 로깅 필요 시 확장 |
| **Fig.6** | 시뮬레이션 스크린샷 (Isaac Sim 렌더링) | play.py --video |

---

## Step 5: 영상 수집 (Optional, 발표용)

```bash
python play.py \
    --task Unitree-Go2-Phm-v1 \
    --checkpoint logs/rsl_rl/unitree_go2_phm_strategic/<timestamp>/model_2999.pt \
    --num_envs 16 \
    --video --video_length 500
```

---

## 실행 순서 요약 (체크리스트)

- [ ] Step 1-A: PHM-Aware 학습 (seed 42)
- [ ] Step 1-B: Baseline 학습 (seed 42)
- [ ] Step 2-A: PHM-Aware 평가 (4 scenarios × 100 episodes)
- [ ] Step 2-B: Baseline 평가 (4 scenarios × 100 episodes)
- [ ] Step 3: TensorBoard에서 학습 곡선 CSV 추출
- [ ] Step 4: JSON 결과로 Table 1, 2 작성
- [ ] Step 5: (선택) 영상 촬영

**예상 총 소요 시간**: 학습 6~12시간 + 평가 1시간 + 정리 2시간 ≈ **1~2일**

---

## 추가 실험 (Ablation, 2순위)

시간이 된다면 다음 ablation을 추가하면 논문이 더 강해집니다:

1. **Observation Ablation (Teacher/Privileged)**: thermal_stress만, mech_health만, voltage_budget만 각각 넣은 변형
2. **Observation Ablation (RealObs)**: vibration_level on/off, thermal_stress(=thermal_stress_realobs) on/off, energy_budget(=available_voltage_budget_realobs) on/off
3. **Curriculum Ablation**: Phase 1/2 없이 처음부터 Full Random으로 학습
4. **History Length**: num_observation_history = 1, 3, 5, 10 비교

---

## Step 6: Real-Observable Task & Replay Evaluation (Paper Track)

### 6-A. Teacher policy training (privileged PHM)

Teacher uses privileged observations (including latent PHM channels):

```bash
cd /home/iamjaehka13/unitree_go2_phm/unitree_go2_phm/scripts/rsl_rl

python train.py \
    --task Unitree-Go2-Phm-v1 \
    --num_envs 4096 \
    --max_iterations 3000 \
    --headless
```

### 6-B. Student distillation pretraining (teacher -> real-observable)

Student policy is distilled on `Unitree-Go2-RealObs-v1` while teacher action is used as target.

```bash
python distill_teacher_student.py \
    --teacher_task Unitree-Go2-Phm-v1 \
    --student_task Unitree-Go2-RealObs-v1 \
    --teacher_checkpoint logs/rsl_rl/unitree_go2_phm_strategic/<teacher_run>/model_2999.pt \
    --num_envs 1024 \
    --num_updates 2000 \
    --steps_per_update 24 \
    --fixed_risk_factor 1.0 \
    --align_student_dynamics none \
    --dagger_start_beta 1.0 \
    --dagger_end_beta 0.2 \
    --run_name distill_seed42 \
    --headless
```

Output checkpoint:
- `logs/rsl_rl/unitree_go2_realobs/<distill_run>/student_distill_final.pt`
- Distillation default는 `--align_student_dynamics none`이며 학생 task의 RealObs 안전 의미(case-proxy/센서전압)를 유지합니다.
- 필요 시 `--align_student_dynamics all`로 teacher의 brownout/thermal termination 설정을 학생에 복사할 수 있습니다(반드시 논문에 명시).

### 6-C. Student RL finetuning (real-observable only)

Use the distilled checkpoint as initialization and continue PPO on `Unitree-Go2-RealObs-v1`:

```bash
python train.py \
    --task Unitree-Go2-RealObs-v1 \
    --resume \
    --load_run <distill_run> \
    --checkpoint student_distill_final.pt \
    --num_envs 4096 \
    --max_iterations 2000 \
    --run_name finetune_seed42 \
    --headless
```

권장 비교군:
- `Student-scratch`: `Unitree-Go2-RealObs-v1`를 처음부터 PPO 학습.
- `Student-distilled`: 위 distillation 초기화 후 PPO finetune.

RealObs 채널/안전 기준 (현재 코드):
- `energy_budget`: 측정 전압 채널(`battery_voltage` 우선) 기반 headroom (`cutoff=24.5V`)
- `thermal_stress`: case/housing temperature 우선, 없으면 `coil_temp - 5°C` proxy (`warn/crit=65/70`)
- brownout 판정 채널: `sensor_voltage` (`enter/recover=24.5/25.0V`)
- thermal termination: case-proxy 기준 `72°C`
- thermal safety reward: case/housing margin 기반 (`warn=65°C`, `limit_temp=70°C`, case 미존재 시 `coil-5°C` proxy)

### 6-D. Fixed replay evaluation (governor ON/OFF)

Replay command files are provided in `scripts/rsl_rl/replay_commands/`:
- `s1_thermal_cruise.yaml`
- `s2_voltage_sag_burst.yaml`
- `s3_mixed_maneuver.yaml`

```bash
cd /home/iamjaehka13/unitree_go2_phm/unitree_go2_phm/scripts/rsl_rl

# Governor OFF
python evaluate_replay.py \
    --task Unitree-Go2-RealObs-v1 \
    --checkpoint logs/rsl_rl/unitree_go2_realobs/<finetune_run>/model_1999.pt \
    --command_file replay_commands/s1_thermal_cruise.yaml \
    --num_trials 3 \
    --risk_factor_fixed 1.0 \
    --temp_signal auto --coil_to_case_delta_c 5.0 \
    --output_dir ./replay_results/off \
    --headless

# Governor ON + RL leg fault injection
python evaluate_replay.py \
    --task Unitree-Go2-RealObs-v1 \
    --checkpoint logs/rsl_rl/unitree_go2_realobs/<finetune_run>/model_1999.pt \
    --command_file replay_commands/s1_thermal_cruise.yaml \
    --num_trials 3 \
    --governor \
    --fault_leg RL \
    --fault_kp_scale 0.6 \
    --fault_kd_scale 1.0 \
    --fault_start_s 5.0 \
    --risk_factor_fixed 1.0 \
    --temp_signal auto --coil_to_case_delta_c 5.0 \
    --temp_warn_c 65 --temp_crit_c 70 --temp_stop_c 72 \
    --cell_warn_v 3.20 --cell_stop_v 3.05 --cell_hard_stop_v 3.00 --pack_stop_v 24.5 \
    --output_dir ./replay_results/on_fault \
    --headless
```

`evaluate_replay.py` exports:
- `trial_XX_steps.csv` (step-wise log with command, governor scale, temperature, voltage, power, yaw tracking)
- `summary.json` (per-trial + aggregated metrics)

요약 비교 핵심 지표:
- `time_temp_over_warn_s` (하드코딩 65가 아닌 `--temp_warn_c` 기준)
- `yaw_mae_exec`, `vpack_min_v`, `vcell_min_v`, `mean_scale_lin`, `completed_rate`

Compare ON/OFF summaries:

```bash
python compare_replay_summaries.py \
    --a ./replay_results/off/<run>/summary.json \
    --b ./replay_results/on_fault/<run>/summary.json
```

이 ablation들은 **별도의 env_cfg 파일**을 만들어 관측 채널을 하나씩 제거/추가하면 됩니다.
