"""
Section 6.4 — Rule layer SPC validation (Figure 15 & Figure 16)
Device CZ-01, 40.2-hour full CZ crystal growth cycle

Outputs:
  figure_fdc_spc.png    → Figure 15: SPC control charts for 6 key parameters
  figure_fdc_pareto.png → Figure 16: Pareto chart of OOC events

Usage:
  python cz01_spc_analysis.py

Data:
  Requires the Device CZ-01 FDC dataset. The raw data file is proprietary
  and cannot be distributed publicly (confidentiality agreement).
  Contact the corresponding author for access arrangements.
  Update DATA_PATH below to point to your local copy.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = 'device_cz01_fdc.txt'   # ← update to your local path

df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
df['DATETIME'] = pd.to_datetime(df['DATETIME'], format='mixed', errors='coerce')
df = df.dropna(subset=['DATETIME']).reset_index(drop=True)
df = df.sort_values('DATETIME').reset_index(drop=True)

print(f"Rows loaded : {len(df)}")
print(f"Time range  : {df['DATETIME'].min()} → {df['DATETIME'].max()}")
print(f"Duration    : {(df['DATETIME'].max()-df['DATETIME'].min()).total_seconds()/3600:.1f} hours")

params = {
    'Heater Output Power':     '加热器输出功率',
    'Melt Surface Temp (°C)':  '液面温度',
    'Crystal Diameter (mm)':   '晶棒直径',
    'Seed Rotation (rpm)':     '籽晶回转速度',
    'Crucible Rotation (rpm)': '坩埚回转速度',
    'Heater Output Current':   '加热器输出电流',
}

t_hours = (df['DATETIME'] - df['DATETIME'].min()).dt.total_seconds() / 3600

spc_results = {}
for label, col in params.items():
    series    = df[col].values.astype(float)
    mean      = np.mean(series)
    std       = np.std(series)
    ucl       = mean + 3 * std
    lcl       = mean - 3 * std
    ooc_mask  = (series > ucl) | (series < lcl)
    ooc_count = int(ooc_mask.sum())
    ooc_rate  = ooc_count / len(series)
    above_mean = (series > mean).astype(int)
    run9_count = 0
    for i in range(8, len(above_mean)):
        w = above_mean[i-8:i+1]
        if w.sum() == 9 or w.sum() == 0:
            run9_count += 1
    spc_results[label] = dict(
        mean=mean, std=std, UCL=ucl, LCL=lcl,
        ooc_count=ooc_count, ooc_rate=ooc_rate,
        run9_count=run9_count, ooc_mask=ooc_mask, series=series,
    )

print("\n" + "="*65)
print("SPC Detection Summary (I-MR 3σ rule)")
print("="*65)
print(f"{'Parameter':<26} {'Mean':>8} {'Std':>8} {'OOC':>6} {'Rate':>7} {'Run-9':>7}")
print("-"*65)
for label, r in spc_results.items():
    print(f"{label:<26} {r['mean']:>8.2f} {r['std']:>8.2f} "
          f"{r['ooc_count']:>6d} {r['ooc_rate']:>6.1%} {r['run9_count']:>7d}")
print("-"*65)
print(f"Total OOC events: {sum(r['ooc_count'] for r in spc_results.values())}")

fig, axes = plt.subplots(3, 2, figsize=(14, 10))
axes = axes.flatten()
color_line = '#1565C0'
color_ooc  = '#D32F2F'
color_ctl  = '#388E3C'

for idx, (label, col) in enumerate(params.items()):
    ax = axes[idx]
    r  = spc_results[label]
    s  = r['series']
    ax.plot(t_hours, s, color=color_line, linewidth=0.8, alpha=0.85, label=label)
    ax.axhline(r['UCL'],  color=color_ctl, linestyle='--', linewidth=1.2,
               label=f"UCL={r['UCL']:.1f}")
    ax.axhline(r['mean'], color='gray',    linestyle='-',  linewidth=0.8, alpha=0.6,
               label=f"Mean={r['mean']:.1f}")
    ax.axhline(r['LCL'],  color=color_ctl, linestyle='--', linewidth=1.2,
               label=f"LCL={r['LCL']:.1f}")
    ooc_t = t_hours[r['ooc_mask']]
    ooc_v = s[r['ooc_mask']]
    ax.scatter(ooc_t, ooc_v, color=color_ooc, s=18, zorder=5,
               label=f"OOC ({r['ooc_count']}pts)")
    ax.set_title(label, fontsize=11)
    ax.set_xlabel('Time (hours)', fontsize=9)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figure_fdc_spc.png', dpi=150, bbox_inches='tight', facecolor='white')
print("\nSaved: figure_fdc_spc.png")

fig2, ax2 = plt.subplots(figsize=(9, 5))
labels_list   = list(spc_results.keys())
ooc_counts    = [spc_results[l]['ooc_count'] for l in labels_list]
ooc_rates     = [spc_results[l]['ooc_rate'] * 100 for l in labels_list]
sorted_idx    = np.argsort(ooc_counts)[::-1]
sorted_labels = [labels_list[i] for i in sorted_idx]
sorted_counts = [ooc_counts[i]  for i in sorted_idx]
sorted_rates  = [ooc_rates[i]   for i in sorted_idx]
bars = ax2.bar(sorted_labels, sorted_counts,
               color='#1565C0', alpha=0.85, edgecolor='white')
ax2_r = ax2.twinx()
cumulative = np.cumsum(sorted_counts) / max(sum(sorted_counts), 1) * 100
ax2_r.plot(sorted_labels, cumulative, 'ro-',
           linewidth=1.5, markersize=5, label='Cumulative %')
ax2_r.set_ylabel('Cumulative OOC (%)', color='red')
ax2_r.set_ylim(0, 110)
max_count = max(sorted_counts) if max(sorted_counts) > 0 else 1
ax2.set_ylim(0, max_count * 1.25)
for bar, count, rate in zip(bars, sorted_counts, sorted_rates):
    h = bar.get_height()
    if count > 0:
        ax2.text(bar.get_x() + bar.get_width() / 2, h * 0.5,
                 f'{count}\n({rate:.1f}%)',
                 ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    else:
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.15,
                 f'{count}\n({rate:.1f}%)',
                 ha='center', va='bottom', fontsize=9, color='black')
ax2.set_ylabel('OOC Event Count')
ax2.grid(axis='y', alpha=0.3)
ax2_r.legend(loc='center right')
ax2.set_xticklabels(sorted_labels, rotation=20, ha='right', fontsize=9)
plt.tight_layout()
plt.savefig('figure_fdc_pareto.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: figure_fdc_pareto.png")
print("\nSection 6.4 analysis complete.")