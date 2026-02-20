# 2026-02-17 Growth Journal (3 baseline runs)

## Runs
- R1: `2026-02-17_03-18-48_baseline_s42` (baseline reference)
- R2: `2026-02-17_13-23-42_baseline_s42` (late std limiter: up=1.01, down=1.1, start=2200)
- R3: `2026-02-17_20-09-53_baseline_s42` (late std limiter: up=1.02, down=1.0, start=2200)

## Why 3 runs on Feb 17
- Background problem from Feb 16: iteration boundary shocks near 1000/2000 and late-phase instability.
- Purpose of Feb 17:
  1. Keep baseline as reference (R1)
  2. Test safer late-phase std control (R2)
  3. Test stronger late-phase std restore (R3)

## Config changes between runs
- R1 -> R2:
  - `action_std_late_rate_limit_enable: true`
  - `action_std_late_rate_limit_start_iter: 2200`
  - `action_std_late_rate_limit_segment_iters: 20`
  - `action_std_late_max_up_factor: 1.01`
  - `action_std_late_max_down_factor: 1.1`
- R2 -> R3:
  - `action_std_late_max_up_factor: 1.02` (more aggressive)
  - `action_std_late_max_down_factor: 1.0` (down clamp)

## Final metrics (iter 2999)
| metric | R1 Base | R2 mild limiter | R3 aggressive limiter |
|---|---:|---:|---:|
| mean_reward | 17.6073 | 9.0518 | 4.5951 |
| mean_episode_length | 724.28 | 402.50 | 364.81 |
| time_out | 0.5513 | 0.5301 | 0.2995 |
| base_contact | 0.0263 | 0.0165 | 0.0064 |
| bad_orientation | 0.4231 | 0.4534 | 0.6941 |
| error_vel_xy | 0.2248 | 0.2505 | 0.2155 |
| error_vel_yaw | 0.2321 | 0.2595 | 0.2691 |
| mean_noise_std | 0.1848 | 0.1693 | 0.3405 |

## What happened / issue analysis
- Common inherited issue: all 3 runs still show sharp collapse around 1000 and 2000 boundaries (episode length/reward drops), consistent with staged schedule side effects.
- R2 (mild limiter):
  - Pros: lower base_contact than R1.
  - Cons: reward/episode length/time_out decreased; bad_orientation increased vs R1.
  - Interpretation: safer contact but overall locomotion quality regressed.
- R3 (aggressive limiter):
  - Noise std climbs strongly after 2200.
  - base_contact stays low, but bad_orientation explodes and time_out collapses.
  - Interpretation: over-aggressive late exploration destabilized posture.

## Late-phase drift slope (2600 -> 2999)
- R1: base_contact `+0.000044/iter`, bad_orientation `+0.000195/iter`, time_out `-0.000238/iter`
- R2: base_contact `+0.000030/iter`, bad_orientation `+0.000167/iter`, time_out `-0.000199/iter`
- R3: base_contact `+0.000002/iter`, bad_orientation `+0.000593/iter`, time_out `-0.000596/iter`

## Growth-journal conclusion for Feb 17
- Direction learned:
  - "base_contact only" reduction is not enough; must jointly control bad_orientation + time_out.
  - late std restoration has a narrow safe range; aggressive restore can break posture.
- Next-day action logic:
  - keep mild schedule/ramp style first,
  - then tune for posture stability before pushing exploration.
