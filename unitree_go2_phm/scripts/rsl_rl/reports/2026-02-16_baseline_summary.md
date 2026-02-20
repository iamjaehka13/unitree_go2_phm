# 2026-02-16 Baseline Runs (Why 2 runs, what failed, what changed)

## Scope
- Task: `Unitree-Go2-Baseline-v1`
- Runs compared:
  - `unitree_go2_baseline/2026-02-16_01-35-00_baseline_s42` (Run-1)
  - `unitree_go2_baseline/2026-02-16_18-05-49_baseline_s42` (Run-2)
- Both were trained to iteration `2999`.

## Why there are 2 baseline runs on Feb 16
Run-1 was the first baseline attempt with stronger early disturbance/fault pressure and less structured curriculum.
Run-2 was a stabilization retry to reduce early collapse and improve survival, by changing curriculum timing and randomization policy.

## What changed from Run-1 to Run-2 (key config deltas)
- Observation noise reduced:
  - gyro noise roughly `[-0.2, 0.2] -> [-0.12, 0.12]`
  - gravity noise roughly `[-0.05, 0.05] -> [-0.03, 0.03]`
- Friction randomization narrowed and shifted:
  - `0.3~1.2 -> 0.6~1.25`
- Mass randomization changed:
  - startup additive mass `(-1.0, +3.0)` -> per-reset scaling `(0.9, 1.1)`
- PHM schedule delayed/softened:
  - `used_start_iter: 1000 -> 1601`
  - `critical_end_iter: 2400 -> 2800`
- Push schedule delayed:
  - `push_start: 501 -> 1001`
- DR curriculum added (Run-2 only):
  - starts at iter `501`, ramps friction/mass/delay progressively
- Entropy/action-std stage schedule added (Run-2 only):
  - entropy: `0.01 -> 0.005 -> 0.003` at 1000/2000 boundaries
  - action std target: `1.0 -> 0.7 -> 0.5` at 1000/2000 boundaries

## End-of-run metrics (iter 2999)
| metric | Run-1 | Run-2 | delta (Run-2 - Run-1) |
|---|---:|---:|---:|
| Train/mean_reward | 11.1618 | 14.4327 | +3.2708 |
| Train/mean_episode_length | 554.47 | 699.91 | +145.44 |
| Episode_Termination/time_out | 0.3328 | 0.5769 | +0.2441 |
| Episode_Termination/base_contact | 0.0456 | 0.0020 | -0.0436 |
| Episode_Termination/bad_orientation | 0.6219 | 0.4212 | -0.2007 |
| Metrics/base_velocity/error_vel_xy | 0.1501 | 0.2328 | +0.0827 |
| Metrics/base_velocity/error_vel_yaw | 0.1758 | 0.2240 | +0.0482 |
| Policy/mean_noise_std | 0.3253 | 0.1812 | -0.1441 |

Interpretation:
- Run-2 clearly improved survival/safety (`time_out` up, `bad_orientation` and `base_contact` down).
- Tracking errors got worse, indicating a stability-first policy tendency.

## Main failure/problem observed in Run-1
- Progressive degradation after mid-late phase:
  - `time_out` dropped (e.g., around iter 1000: ~0.67, iter 2999: ~0.33)
  - `bad_orientation` increased (iter 1000: ~0.32, iter 2999: ~0.62)
  - `base_contact` rose in late phase (near 2600 around ~0.058)
- This indicates early/hard disturbances and weaker scheduling made long-horizon stability poor.

## New problem introduced in Run-2
Boundary shock at iteration transitions due staged std overwrite:
- At iter `999 -> 1000`:
  - `Policy/mean_noise_std: 0.3103 -> 0.6961`
  - `Train/mean_reward: 19.06 -> -0.30`
  - `Train/mean_episode_length: 939.52 -> 15.07`
- At iter `1999 -> 2000`:
  - `Policy/mean_noise_std: 0.2070 -> 0.4956`
  - `Train/mean_reward: 26.39 -> -0.09`
  - `Train/mean_episode_length: 932.95 -> 14.40`

Interpretation:
- Run-2 solved Run-1 style chronic collapse, but introduced sharp boundary transients.
- This is the origin of later discussions around `overwrite -> cap-only` and smoother std control.

## Videos generated for Feb 16 report
- Run-1 video:
  - `logs/rsl_rl/unitree_go2_baseline/2026-02-16_01-35-00_baseline_s42/videos/play/rl-video-step-0.mp4`
- Run-2 video:
  - `logs/rsl_rl/unitree_go2_baseline/2026-02-16_18-05-49_baseline_s42/videos/play/rl-video-step-0.mp4`

## One-line summary for Notion
- Feb 16 had two baseline runs because Run-1 showed late-phase stability collapse; Run-2 improved survival by curriculum/randomization redesign, but created hard transition shocks at iter 1000/2000 due staged action-std jumps.
