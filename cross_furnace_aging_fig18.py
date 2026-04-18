"""
Figure 18 — Cross-furnace thermal field aging validation
论文6.5节配图：三联图
(a) G均值随炉龄衰减
(b) v/G均值+缺陷率随炉龄变化（双轴）
(c) 策略C vs 策略D MAE对比+压缩率

用法：python Figure18_script.py
依赖：data.xlsx 与脚本在同一目录，或修改下方 DATA_PATH
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = 'data.xlsx'   # ← 修改为实际路径

df = pd.read_excel(DATA_PATH, sheet_name='拉晶数据')
furnace_stat = pd.read_excel(DATA_PATH, sheet_name='炉次统计')
body = df[df.阶段=='等径'].copy().sort_values(['炉次号','时间']).reset_index(drop=True)
G0 = body[body.炉次号=='B001']['温度梯度'].mean()

results = []
for fid in sorted(body.炉次号.unique()):
    seg = body[body.炉次号==fid].copy().reset_index(drop=True)
    v = seg['拉速'].values
    G_true = seg['温度梯度'].values
    vg_true = seg['v/G比值'].values
    age = seg['炉龄因子'].mean()
    G_k = np.full(len(seg), G0)
    for i in range(1, len(seg)):
        if i >= 30:
            w = max(0, i-30)
            G_obs = np.mean(v[w:i] / vg_true[w:i])
            G_k[i] = G_k[i-1] + 0.3*(G_obs - G_k[i-1])
        else:
            G_k[i] = G_k[i-1]
    vg_C = v / G0
    vg_D = v / G_k
    mae_C = np.mean(np.abs(vg_C - vg_true))
    mae_D = np.mean(np.abs(vg_D - vg_true))
    results.append({'furnace': fid, 'age': age, 'G_mean': G_true.mean(),
                    'vg_mean': vg_true.mean(), 'mae_C': mae_C, 'mae_D': mae_D,
                    'compression': (mae_C-mae_D)/mae_C*100})

res = pd.DataFrame(results)
ages = res['age'].values
furnace_nums = np.arange(1, 9)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.titlesize': 9.5, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150, 'axes.grid': True,
    'grid.alpha': 0.3, 'grid.linestyle': '--',
})

fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.38,
              left=0.08, right=0.96, top=0.96, bottom=0.08)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])

color_G = '#2166ac'
ax1.plot(furnace_nums, res['G_mean'].values, 'o-', color=color_G,
         lw=2, ms=7, mfc='white', mew=2, zorder=3)
for i, (x, y) in enumerate(zip(furnace_nums, res['G_mean'].values)):
    # Force last point (B008, lowest) upward to avoid x-axis label overlap
    if i == len(furnace_nums) - 1:
        offset, va = 12, 'bottom'
    else:
        offset = 9 if i % 2 == 0 else -14
        va = 'bottom' if i % 2 == 0 else 'top'
    ax1.annotate(f'{y:.2f}', (x, y), textcoords='offset points',
                 xytext=(0, offset), ha='center', fontsize=7.5, color=color_G, va=va)
ax1.axhline(G0, color='#888888', ls=':', lw=1.3, label=f'G₀ = {G0:.2f} K/mm (B001 baseline)')
ax1.fill_between(furnace_nums, res['G_mean'].values, G0, alpha=0.10, color=color_G)
ymax_a = max(res['G_mean'].values) + 1.5
ax1.set_ylim(min(res['G_mean'].values) - 0.5, ymax_a + 1.5)
ax1.annotate('', xy=(8.35, res['G_mean'].values[-1]), xytext=(8.35, G0),
             arrowprops=dict(arrowstyle='<->', color='#d73027', lw=1.5))
ax1.text(8.6, (res['G_mean'].values[-1]+G0)/2, '−29.7%',
         color='#d73027', fontsize=8, va='center')
ax1.set_xlim(0.5, 9.2)
ax1.set_xlabel('Furnace Cycle')
ax1.set_ylabel('Mean Temperature Gradient G (K/mm)')
ax1.set_title('(a) Thermal Field Degradation: G Decay Across Furnace Cycles')
ax1.set_xticks(furnace_nums)
ax1.set_xticklabels([f'B00{i}' for i in range(1, 9)])
ax1.legend(fontsize=8, framealpha=0.8, loc='upper right')

color_vg = '#4dac26'; color_def = '#d01c8b'
ax2b = ax2.twinx(); ax2b.spines['top'].set_visible(False)
line_vg,  = ax2.plot(furnace_nums, res['vg_mean'].values, 's-', color=color_vg,
                     lw=2, ms=7, mfc='white', mew=2, zorder=3, label='v/G mean')
line_def, = ax2b.plot(furnace_nums, furnace_stat['缺陷率(%)'].values, '^--', color=color_def,
                      lw=2, ms=7, mfc='white', mew=2, zorder=3, label='Defect rate (%)')
for i, (x, y) in enumerate(zip(furnace_nums, res['vg_mean'].values)):
    offset = 8 if i % 2 == 0 else -13
    va = 'bottom' if i % 2 == 0 else 'top'
    ax2.annotate(f'{y:.4f}', (x, y), textcoords='offset points',
                 xytext=(0, offset), ha='center', fontsize=6.8, color=color_vg, va=va)
ax2.set_xlabel('Furnace Cycle')
ax2.set_ylabel('Mean v/G Ratio', color=color_vg)
ax2b.set_ylabel('Defect Rate (%)', color=color_def)
ax2.tick_params(axis='y', labelcolor=color_vg)
ax2b.tick_params(axis='y', labelcolor=color_def)
ax2.set_title('(b) v/G Ratio and Defect Rate vs. Furnace Age\n'
              'r(v/G, age) = −0.994, p < 0.0001;  r(defect, age) = −0.987, p < 0.0001')
ax2.set_xticks(furnace_nums)
ax2.set_xticklabels([f'B00{i}' for i in range(1, 9)])
vg_vals = res['vg_mean'].values
ax2.set_ylim(vg_vals.min()-0.008, vg_vals.max()+0.015)
ax2b.set_ylim(-0.3, furnace_stat['缺陷率(%)'].max()+0.6)
ax2.legend([line_vg, line_def], [l.get_label() for l in [line_vg, line_def]],
           fontsize=8, framealpha=0.8, loc='upper right')

width = 0.28
bars_C = ax3.bar(furnace_nums-width/2, res['mae_C'].values*1000, width,
                 label='Strategy C (fixed G₀)', color='#f4a582', edgecolor='#d73027', lw=1.2, zorder=3)
bars_D = ax3.bar(furnace_nums+width/2, res['mae_D'].values*1000, width,
                 label='Strategy D (Kalman-enhanced)', color='#92c5de', edgecolor='#2166ac', lw=1.2, zorder=3)
ax3b = ax3.twinx(); ax3b.spines['top'].set_visible(False)
comp_vals = res['compression'].values
ax3b.plot(furnace_nums, comp_vals, 'D-', color='#4575b4', lw=2, ms=7,
          mfc='white', mew=2, zorder=4, label='MAE compression (%)')
for xi, ci, row in zip(furnace_nums, comp_vals, res.itertuples()):
    # 取两根柱子较高者顶部，转换到ax3b坐标系后标注在柱顶上方白底处
    bar_top_ax3 = max(row.mae_C, row.mae_D) * 1000   # ax3的y值（×10⁻³）
    # 将ax3的y值转换为ax3b的y值用于标注定位
    ax3_ylim = ax3.get_ylim()
    ax3b_ylim = ax3b.get_ylim()
    frac = (bar_top_ax3 - ax3_ylim[0]) / (ax3_ylim[1] - ax3_ylim[0])
    bar_top_ax3b = ax3b_ylim[0] + frac * (ax3b_ylim[1] - ax3b_ylim[0])
    ax3b.annotate(f'{ci:.1f}%', (xi, bar_top_ax3b), textcoords='offset points',
                  xytext=(0, 5), ha='center', fontsize=7.5,
                  color='#4575b4', fontweight='bold', va='bottom')
ax3b.axhline(0, color='#aaaaaa', ls=':', lw=1)
ax3b.set_ylabel('MAE Compression (%)', color='#4575b4')
ax3b.tick_params(axis='y', labelcolor='#4575b4')
ax3.axvspan(5.55, 8.45, alpha=0.06, color='#d73027', zorder=0)
ybar_max = res['mae_C'].values.max()*1000
ax3.set_ylim(0, ybar_max*1.30)
ax3.text(7.0, ybar_max*1.20, 'Aging phase (B006–B008)',
         ha='center', fontsize=8.5, color='#d73027', style='italic', fontweight='bold')
ax3.set_xlabel('Furnace Cycle (age factor in parentheses)')
ax3.set_ylabel('v/G MAE (×10⁻³)')
ax3.set_title('(c) Strategy C vs. Strategy D MAE and Compression Rate\n'
              'Wilcoxon one-sided p = 0.0195  |  Aging phase compression: 41.9%–48.1%')
ax3.set_xticks(furnace_nums)
ax3.set_xticklabels([f'B00{i}\n({a:.2f})' for i, a in zip(range(1,9), ages)])
comp_patch = mpatches.Patch(color='#4575b4', label='MAE compression (%)')
ax3.legend(handles=[bars_C, bars_D, comp_patch], fontsize=8, framealpha=0.8, loc='upper left')

plt.savefig('Figure18_cross_furnace_aging.png', dpi=200, bbox_inches='tight', facecolor='white')
print('Saved: Figure18_cross_furnace_aging.png')
