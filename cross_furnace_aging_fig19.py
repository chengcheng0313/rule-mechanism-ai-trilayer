"""
Figure 19 — Per-furnace v/G trajectory and error compression
论文6.5节配图：4炉次代表性轨迹对比（左：v/G轨迹，右：误差压缩）

用法：python Figure19_script.py
依赖：data.xlsx 与脚本在同一目录，或修改下方 DATA_PATH
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = 'data.xlsx'   # ← 修改为实际路径

df = pd.read_excel(DATA_PATH, sheet_name='拉晶数据')
body = df[df.阶段=='等径'].copy().sort_values(['炉次号','时间']).reset_index(drop=True)
G0 = body[body.炉次号=='B001']['温度梯度'].mean()
#vg_crit = body[body.炉次号=='B001']['v/G比值'].mean()
vg_crit = 0.075   # Voronkov临界值，物理常数
delta = vg_crit * 0.05

vg_traj = {}
for fid in sorted(body.炉次号.unique()):
    seg = body[body.炉次号==fid].copy().reset_index(drop=True)
    v = seg['拉速'].values
    G_true = seg['温度梯度'].values
    vg_true = seg['v/G比值'].values
    G_k = np.full(len(seg), G0)
    for i in range(1, len(seg)):
        if i >= 30:
            w = max(0, i-30)
            G_obs = np.mean(v[w:i] / vg_true[w:i])
            G_k[i] = G_k[i-1] + 0.3*(G_obs - G_k[i-1])
        else:
            G_k[i] = G_k[i-1]
    vg_traj[fid] = {'true': vg_true, 'C': v/G0, 'D': v/G_k,
                    'err_C': np.abs(v/G0-vg_true), 'err_D': np.abs(v/G_k-vg_true)}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8.5,
    'axes.titlesize': 8.8, 'axes.labelsize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150, 'axes.grid': True,
    'grid.alpha': 0.25, 'grid.linestyle': '--',
})

selected   = ['B001', 'B003', 'B006', 'B008']
row_labels = [
    'B001  |  Age Factor = 1.00  |  New furnace (baseline)',
    'B003  |  Age Factor = 0.95  |  Slight aging',
    'B006  |  Age Factor = 0.82  |  Moderate aging',
    'B008  |  Age Factor = 0.72  |  Severe aging',
]
color_true='#222222'; color_C='#d73027'; color_D='#2166ac'
color_err_C='#f4a582'; color_err_D='#92c5de'

fig, axes = plt.subplots(4, 2, figsize=(14, 15),
                          gridspec_kw={'wspace': 0.35, 'hspace': 0.72})
fig.subplots_adjust(left=0.07, right=0.97, top=0.97, bottom=0.04)

for row, (fid, rlabel) in enumerate(zip(selected, row_labels)):
    d = vg_traj[fid]
    t = np.arange(len(d['true']))
    mae_C = np.mean(d['err_C']); mae_D = np.mean(d['err_D'])
    comp  = (mae_C-mae_D)/mae_C*100

    ax = axes[row, 0]
    ax.plot(t, d['true'], color=color_true, lw=1.5, zorder=4, label='Ground Truth v/G')
    ax.plot(t, d['C'],    color=color_C,    lw=1.1, ls='--', zorder=3, label='Strategy C (fixed G₀)')
    ax.plot(t, d['D'],    color=color_D,    lw=1.1, ls='-',  zorder=3, label='Strategy D (Kalman)')
    ax.axhline(vg_crit,         color='#7b2d8b', lw=1.0, ls=':', alpha=0.8, label=f'v/G crit ({vg_crit:.4f})')
    ax.axhline(vg_crit+delta,   color='#7b2d8b', lw=0.6, ls=':', alpha=0.4)
    ax.axhline(vg_crit-delta,   color='#7b2d8b', lw=0.6, ls=':', alpha=0.4)
    ax.fill_between(t, vg_crit-delta, vg_crit+delta, alpha=0.07, color='#7b2d8b')
    comp_str = f'Compression = {comp:.1f}%' if comp > 0 else f'Compression = {comp:.1f}% (no drift)'
    ax.set_title(f'v/G Trajectory  —  {rlabel}\n'
                 f'MAE_C = {mae_C*1000:.3f}×10⁻³    MAE_D = {mae_D*1000:.3f}×10⁻³    {comp_str}')
    ax.set_xlabel('Time Step (body growth phase, 80 samples)')
    ax.set_ylabel('v/G Ratio')
    if row == 0:
        ax.legend(fontsize=7.2, framealpha=0.8, ncol=2, loc='upper right', handlelength=1.8)

    ax2 = axes[row, 1]
    ax2.fill_between(t, d['err_C']*1000, alpha=0.30, color=color_err_C, label='|Error| Strategy C')
    ax2.fill_between(t, d['err_D']*1000, alpha=0.55, color=color_err_D, label='|Error| Strategy D')
    ax2.plot(t, d['err_C']*1000, color=color_C, lw=0.8, alpha=0.8)
    ax2.plot(t, d['err_D']*1000, color=color_D, lw=0.8, alpha=0.9)
    ax2.axvline(30, color='#666666', lw=1.0, ls=':', alpha=0.7)
    err_max = max(d['err_C'].max(), d['err_D'].max())*1000
    ax2.set_ylim(-err_max*0.05, err_max*1.28)
    ax2.text(32, err_max*1.18, 'Kalman\nactive →', fontsize=7, color='#555555', va='top', ha='left')
    if fid in ('B006', 'B008'):
        ax2.text(0.98, 0.96, f'Aging phase\nCompression {comp:.1f}%',
                 transform=ax2.transAxes, ha='right', va='top', fontsize=7.5,
                 color='#d73027', style='italic',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor='#d73027', alpha=0.85))
    ax2.set_title(f'Absolute Prediction Error  —  {fid}')
    ax2.set_xlabel('Time Step (body growth phase, 80 samples)')
    ax2.set_ylabel('|v/G Error| (×10⁻³)')
    if row == 0:
        ax2.legend(fontsize=7.2, framealpha=0.8, loc='upper right')

plt.savefig('Figure19_vG_trajectory_per_furnace.png', dpi=200, bbox_inches='tight', facecolor='white')
print('Saved: Figure19_vG_trajectory_per_furnace.png')
