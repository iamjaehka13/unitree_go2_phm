#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path('unitree_go2_phm/scripts/rsl_rl/logs/rsl_rl/unitree_go2_realobs')
OUT_DIR = Path('unitree_go2_phm/scripts/rsl_rl/reports/figures_2026_02_19')
OUT_MD = Path('unitree_go2_phm/scripts/rsl_rl/reports/2026-02-19_realobs_single_motor_growth_journal.md')
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [
    ('realobs_single_motor_3k_base_s42', ROOT / '2026-02-19_03-41-22_realobs_single_motor_3k_s42_perf'),
    ('realobs_single_motor_3k_gate_on_s42', ROOT / '2026-02-19_10-53-13_realobs_single_motor_3k_gate_on_s42_perf'),
    ('realobs_single_motor_3k_std_safe_s42', ROOT / '2026-02-19_10-53-22_realobs_single_motor_3k_std_safe_s42_perf'),
]

TAGS = {
    'reward': 'Train/mean_reward',
    'ep_len': 'Train/mean_episode_length',
    'time_out': 'Episode_Termination/time_out',
    'base_contact': 'Episode_Termination/base_contact',
    'bad_orientation': 'Episode_Termination/bad_orientation',
    'noise_std': 'Policy/mean_noise_std',
    'err_xy': 'Metrics/base_velocity/error_vel_xy',
    'err_yaw': 'Metrics/base_velocity/error_vel_yaw',
}


def read_scalars(event_file: Path):
    acc = EventAccumulator(str(event_file), size_guidance={'scalars': 0})
    acc.Reload()
    out = {}
    for k, tag in TAGS.items():
        if tag not in acc.Tags().get('scalars', []):
            continue
        evs = acc.Scalars(tag)
        steps = np.array([e.step for e in evs], dtype=np.int64)
        vals = np.array([e.value for e in evs], dtype=np.float64)
        # keep last value for duplicated steps
        uniq = {}
        for s, v in zip(steps, vals):
            uniq[int(s)] = float(v)
        xs = np.array(sorted(uniq.keys()), dtype=np.int64)
        ys = np.array([uniq[int(x)] for x in xs], dtype=np.float64)
        out[k] = (xs, ys)
    return out


def val_at(series, step):
    xs, ys = series
    if len(xs) == 0:
        return float('nan')
    idx = np.where(xs <= step)[0]
    if len(idx) == 0:
        return float(ys[0])
    return float(ys[idx[-1]])


def window_mean(series, lo, hi):
    xs, ys = series
    m = (xs >= lo) & (xs <= hi)
    if not np.any(m):
        return float('nan')
    return float(np.mean(ys[m]))


def late_slope(series, lo=2600, hi=2999):
    xs, ys = series
    m = (xs >= lo) & (xs <= hi)
    if np.sum(m) < 2:
        return float('nan')
    x = xs[m].astype(np.float64)
    y = ys[m].astype(np.float64)
    a, _ = np.polyfit(x, y, 1)
    return float(a)


all_data = {}
summary = {'runs': {}, 'notes': {}}

for label, run_dir in RUNS:
    event_files = list(run_dir.glob('events.out.tfevents.*'))
    if not event_files:
        raise FileNotFoundError(f'No event file in {run_dir}')
    data = read_scalars(event_files[0])
    all_data[label] = data

    s = {}
    for k in ['reward', 'ep_len', 'time_out', 'base_contact', 'bad_orientation', 'noise_std', 'err_xy', 'err_yaw']:
        if k in data:
            s[k] = val_at(data[k], 2999)

    if 'time_out' in data:
        s['time_out_slope_2600_2999'] = late_slope(data['time_out'])
    if 'bad_orientation' in data:
        s['bad_orientation_slope_2600_2999'] = late_slope(data['bad_orientation'])

    for b in [1000, 2000]:
        for k in ['noise_std', 'time_out', 'base_contact', 'bad_orientation']:
            if k in data:
                pre = window_mean(data[k], b - 20, b - 1)
                post = window_mean(data[k], b, b + 20)
                s[f'{k}_delta_{b}'] = post - pre

    summary['runs'][label] = s

# Plot 1: full trends
fig, axs = plt.subplots(5, 1, figsize=(12, 18), sharex=True)
plot_keys = ['reward', 'ep_len', 'time_out', 'base_contact', 'bad_orientation']
plot_titles = ['Train/mean_reward', 'Train/mean_episode_length', 'Episode_Termination/time_out',
               'Episode_Termination/base_contact', 'Episode_Termination/bad_orientation']
for ax, k, title in zip(axs, plot_keys, plot_titles):
    for label, _ in RUNS:
        if k in all_data[label]:
            x, y = all_data[label][k]
            ax.plot(x, y, label=label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axvline(1000, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(2000, color='gray', linestyle='--', alpha=0.5)
axs[-1].set_xlabel('iteration')
axs[0].legend(loc='best', fontsize=9)
fig.tight_layout()
fig.savefig(OUT_DIR / 'tb_3runs_full_trends.png', dpi=160)
plt.close(fig)

# Plot 2: boundary zoom for std + terminations
fig, axs = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
windows = [(950, 1050), (1950, 2050)]
for j, (lo, hi) in enumerate(windows):
    ax = axs[0, j]
    for label, _ in RUNS:
        if 'noise_std' in all_data[label]:
            x, y = all_data[label]['noise_std']
            m = (x >= lo) & (x <= hi)
            ax.plot(x[m], y[m], label=label)
    ax.set_title(f'Policy/mean_noise_std ({lo}-{hi})')
    ax.grid(True, alpha=0.3)

    ax2 = axs[1, j]
    for label, _ in RUNS:
        if 'time_out' in all_data[label]:
            x, y = all_data[label]['time_out']
            m = (x >= lo) & (x <= hi)
            ax2.plot(x[m], y[m], label=f'{label}:time_out')
        if 'bad_orientation' in all_data[label]:
            x, y = all_data[label]['bad_orientation']
            m = (x >= lo) & (x <= hi)
            ax2.plot(x[m], y[m], linestyle='--', alpha=0.8, label=f'{label}:bad_orient')
    ax2.set_title(f'time_out / bad_orientation ({lo}-{hi})')
    ax2.grid(True, alpha=0.3)
axs[0, 0].legend(fontsize=8)
axs[1, 0].legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(OUT_DIR / 'tb_boundary_zoom_1000_2000.png', dpi=160)
plt.close(fig)

# Plot 3: late phase 2600+
fig, axs = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
late_keys = ['reward', 'time_out', 'base_contact', 'bad_orientation']
for ax, k in zip(axs.ravel(), late_keys):
    for label, _ in RUNS:
        if k in all_data[label]:
            x, y = all_data[label][k]
            m = x >= 2600
            ax.plot(x[m], y[m], label=label)
    ax.set_title(f'{TAGS[k]} (2600+)')
    ax.grid(True, alpha=0.3)
axs[0, 0].legend(fontsize=8)
axs[-1, 0].set_xlabel('iteration')
axs[-1, 1].set_xlabel('iteration')
fig.tight_layout()
fig.savefig(OUT_DIR / 'tb_late_phase_2600plus.png', dpi=160)
plt.close(fig)

with open(OUT_DIR / 'summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# markdown summary
r = summary['runs']
md = []
md.append('# 2026-02-19 RealObs Single-Motor Growth Journal')
md.append('')
md.append('## 실험 구성')
for label, run_dir in RUNS:
    md.append(f'- {label}: `{run_dir.name}`')
md.append('')
md.append('## 2999 최종 비교')
for label in r:
    s = r[label]
    md.append(f'- {label}')
    md.append(f"  - reward {s.get('reward', float('nan')):.4f}, ep_len {s.get('ep_len', float('nan')):.2f}, time_out {s.get('time_out', float('nan')):.4f}, base_contact {s.get('base_contact', float('nan')):.4f}, bad_orientation {s.get('bad_orientation', float('nan')):.4f}")
    md.append(f"  - noise_std {s.get('noise_std', float('nan')):.4f}, err_xy {s.get('err_xy', float('nan')):.4f}, err_yaw {s.get('err_yaw', float('nan')):.4f}")
md.append('')
md.append('## 후반 드리프트 slope (2600~2999)')
for label in r:
    s = r[label]
    md.append(f"- {label}: bad_orientation_slope={s.get('bad_orientation_slope_2600_2999', float('nan')):+.6f}/iter, time_out_slope={s.get('time_out_slope_2600_2999', float('nan')):+.6f}/iter")
md.append('')
md.append('## 경계 변화량(평균, post-pre)')
for label in r:
    s = r[label]
    md.append(f'- {label}')
    md.append(f"  - iter1000: noise_std {s.get('noise_std_delta_1000', float('nan')):+.4f}, time_out {s.get('time_out_delta_1000', float('nan')):+.4f}, base_contact {s.get('base_contact_delta_1000', float('nan')):+.4f}, bad_orientation {s.get('bad_orientation_delta_1000', float('nan')):+.4f}")
    md.append(f"  - iter2000: noise_std {s.get('noise_std_delta_2000', float('nan')):+.4f}, time_out {s.get('time_out_delta_2000', float('nan')):+.4f}, base_contact {s.get('base_contact_delta_2000', float('nan')):+.4f}, bad_orientation {s.get('bad_orientation_delta_2000', float('nan')):+.4f}")
md.append('')
md.append('## 결론 초안')
md.append('- single-motor 설정에서 3개 전략의 차이를 동일한 3k budget에서 비교 완료.')
md.append('- 최종 지표 + 2600+ slope + 1000/2000 경계 변화를 함께 보고 다음 학습안(보상/게이트/STD)을 결정.')

OUT_MD.write_text('\n'.join(md), encoding='utf-8')
print(f'[OK] wrote {OUT_MD}')
print(f'[OK] wrote {OUT_DIR / "summary.json"}')
print(f'[OK] figures -> {OUT_DIR}')
