"""
Device CZ-01 Offline Comparison Analysis — Figure 18
Paper: Rule-Mechanism-AI Tri-layer Architecture (CCPE submission)
Section 6.4
Declaration: Offline replay analysis, not live deployment.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Column name mapping: Chinese data columns -> English display labels
CN_TO_EN = {
    '坩埚回转速度':  'Crucible Rotation Speed (rpm)',
    '加热器输出功率': 'Heater Output Power (%)',
    '液面温度':      'Melt Surface Temperature (deg C)',
    '晶棒直径':      'Crystal Diameter (mm)',
    '籽晶回转速度':  'Seed Rotation Speed (rpm)',
    '加热器输出电流': 'Heater Output Current (A)',
}

def en(cn): return CN_TO_EN.get(cn, cn)

# ── Load data ────────────────────────────────────────────────
def load_data(filepath='P74-00318.txt'):
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    df['DATETIME'] = pd.to_datetime(df['DATETIME'], format='mixed', errors='coerce')
    df = df.dropna(subset=['DATETIME']).reset_index(drop=True)
    df = df.sort_values('DATETIME').reset_index(drop=True)
    t_hours = (df['DATETIME'] - df['DATETIME'].min()).dt.total_seconds() / 3600
    print(f"Rows: {len(df)}, Duration: {t_hours.max():.1f} h")
    return df, t_hours

# ── Phase assignment ─────────────────────────────────────────
def define_phases(t_hours):
    n = len(t_hours)
    phases = np.full(n, 'stable', dtype=object)
    phases[t_hours <= 1.0]  = 'startup'
    phases[t_hours >= 25.0] = 'tailing'
    print(f"Startup (t<=1h): {(phases=='startup').sum()} pts")
    print(f"Stable (1h<t<25h): {(phases=='stable').sum()} pts")
    print(f"Tailing (t>=25h): {(phases=='tailing').sum()} pts")
    return phases

# ── Static SPC ───────────────────────────────────────────────
def static_spc(values):
    m, s = np.mean(values), np.std(values)
    ucl, lcl = m + 3*s, m - 3*s
    ooc = (values > ucl) | (values < lcl)
    return dict(mean=m, UCL=ucl, LCL=lcl, ooc=ooc,
                n_ooc=ooc.sum(), rate=ooc.mean())

# ── Phase-aware SPC ──────────────────────────────────────────
def phase_aware_spc(values, phases):
    ooc = np.zeros(len(values), dtype=bool)
    stats = {}
    for phase in ['startup', 'stable', 'tailing']:
        mask = phases == phase
        if not mask.any(): continue
        if phase == 'tailing':
            stats[phase] = dict(UCL=np.inf, LCL=-np.inf, n_ooc=0, note='exempt')
            continue
        v = values[mask]
        m, s = np.mean(v), np.std(v)
        ucl, lcl = m + 3*s, m - 3*s
        phase_ooc = (v > ucl) | (v < lcl)
        ooc[mask] = phase_ooc
        stats[phase] = dict(mean=m, UCL=ucl, LCL=lcl,
                            n_ooc=phase_ooc.sum(), n_pts=mask.sum())
    return dict(ooc=ooc, n_ooc=ooc.sum(), rate=ooc.mean(), stats=stats)

# ── Main analysis ────────────────────────────────────────────
def run_analysis(df, t_hours, phases):
    col = '坩埚回转速度'
    values = df[col].values.astype(float)

    s_res = static_spc(values)
    a_res = phase_aware_spc(values, phases)

    startup_mask = phases == 'startup'
    s_su = s_res['ooc'][startup_mask].sum()
    a_su = a_res['ooc'][startup_mask].sum()
    red_pct = (s_su - a_su) / s_su * 100 if s_su > 0 else 0

    print(f"\n--- Static SPC ---")
    print(f"  UCL={s_res['UCL']:.3f}, LCL={s_res['LCL']:.3f}")
    print(f"  Total OOC: {s_res['n_ooc']} ({s_res['rate']:.1%})")
    print(f"  Startup OOC: {s_su}")

    print(f"\n--- Phase-aware SPC ---")
    for ph, st in a_res['stats'].items():
        if 'note' in st:
            print(f"  [{ph}] {st['note']}")
        else:
            print(f"  [{ph}] UCL={st['UCL']:.3f}, n_ooc={st['n_ooc']}/{st['n_pts']}")
    print(f"  Total OOC: {a_res['n_ooc']}")
    print(f"  Startup OOC: {a_su}")

    print(f"\n*** Startup false alarm reduction: {s_su} -> {a_su} ({red_pct:.1f}%) ***")

    return dict(col=col, values=values,
                static=s_res, aware=a_res,
                startup_mask=startup_mask,
                s_su=s_su, a_su=a_su, red_pct=red_pct)

# ── Plot Figure 18 ───────────────────────────────────────────
def plot_figure18(t_hours, phases, res):
    col    = res['col']
    values = res['values']
    s_res  = res['static']
    a_res  = res['aware']
    disp   = en(col)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                             gridspec_kw={'height_ratios': [2, 2, 1]})

    startup_mask = res['startup_mask']

    # ── Panel A: Static SPC ──────────────────────────────────
    ax1 = axes[0]
    ax1.plot(t_hours, values, color='#1565C0', lw=0.8, alpha=0.85, label=disp)
    ax1.axhline(s_res['UCL'],  color='#388E3C', ls='--', lw=1.2,
                label=f"UCL={s_res['UCL']:.2f}")
    ax1.axhline(s_res['mean'], color='gray', ls='-', lw=0.8, alpha=0.5)
    ax1.axhline(s_res['LCL'],  color='#388E3C', ls='--', lw=1.2,
                label=f"LCL={s_res['LCL']:.2f}")
    ax1.scatter(t_hours[s_res['ooc']], values[s_res['ooc']],
                color='#D32F2F', s=20, zorder=5,
                label=f"OOC ({s_res['n_ooc']} pts, {s_res['rate']:.1%})")
    ax1.axvspan(0, 1.0, alpha=0.1, color='red', label='Startup phase (t<=1h)')
    ax1.set_ylabel(disp, fontsize=9)
    ax1.set_title('Method 1: Static SPC (no SOP state machine)', fontsize=10)
    ax1.legend(fontsize=7, loc='upper right')
    ax1.grid(alpha=0.3); ax1.set_xlim(0, t_hours.max())

    # ── Panel B: Phase-aware SPC ─────────────────────────────
    ax2 = axes[1]
    ax2.plot(t_hours, values, color='#1565C0', lw=0.8, alpha=0.85, label=disp)

    colors = {'startup': '#F57F17', 'stable': '#388E3C'}
    for phase, color in colors.items():
        if phase not in a_res['stats']: continue
        st = a_res['stats'][phase]
        if st.get('UCL', np.inf) == np.inf: continue
        mask = phases == phase
        t0, t1 = t_hours[mask].min(), t_hours[mask].max()
        ax2.hlines(st['UCL'], t0, t1, colors=color, ls='--', lw=1.5,
                   label=f"{phase.capitalize()} UCL={st['UCL']:.2f}")
        ax2.hlines(st['LCL'], t0, t1, colors=color, ls='--', lw=1.5,
                   label=f"{phase.capitalize()} LCL={st['LCL']:.2f}")
        ax2.hlines(st['mean'], t0, t1, colors=color, ls='-', lw=0.8, alpha=0.5)

    ax2.axvline(1.0,  color='orange', ls=':', lw=1.5, label='Phase boundary (t=1h)')
    ax2.axvline(25.0, color='purple', ls=':', lw=1.5, label='Phase boundary (t=25h)')
    ax2.scatter(t_hours[a_res['ooc']], values[a_res['ooc']],
                color='#D32F2F', s=20, zorder=5,
                label=f"OOC ({a_res['n_ooc']} pts, {a_res['rate']:.1%})")
    ax2.axvspan(0, 1.0, alpha=0.1, color='orange', label='Startup phase (t<=1h)')
    ax2.set_ylabel(disp, fontsize=9)
    ax2.set_title('Method 2: Phase-aware SPC (SOP state machine)', fontsize=10)
    ax2.legend(fontsize=7, loc='upper right')
    ax2.grid(alpha=0.3); ax2.set_xlim(0, t_hours.max())

    # ── Panel C: Startup zoom-in ─────────────────────────────
    ax3 = axes[2]
    ax3.plot(t_hours[startup_mask], values[startup_mask],
             color='#1565C0', lw=1.0, alpha=0.9, label='Crucible Rotation')
    ax3.axhline(s_res['UCL'], color='#D32F2F', ls='--', lw=1.5,
                label=f"Static UCL={s_res['UCL']:.2f} (causes false alarms)")
    ax3.axhline(s_res['LCL'], color='#D32F2F', ls='--', lw=1.5)

    if 'startup' in a_res['stats']:
        st = a_res['stats']['startup']
        if st.get('UCL', np.inf) != np.inf:
            ax3.axhline(st['UCL'], color='#388E3C', ls='--', lw=1.5,
                        label=f"Phase-aware UCL={st['UCL']:.2f} (startup-specific)")
            ax3.axhline(st['LCL'], color='#388E3C', ls='--', lw=1.5)

    false_alarm_mask = s_res['ooc'] & startup_mask
    ax3.scatter(t_hours[false_alarm_mask], values[false_alarm_mask],
                color='#D32F2F', s=40, zorder=5, marker='x',
                label=f"Static SPC false alarms: {false_alarm_mask.sum()} pts")

    ax3.set_title(
        f"Startup Phase Zoom-in (t<=1h): "
        f"Static SPC OOC={res['s_su']} pts -> "
        f"Phase-aware SPC OOC={res['a_su']} pts "
        f"(False alarm reduction: {res['red_pct']:.1f}%)",
        fontsize=9, fontweight='bold')
    ax3.set_xlabel('Time (hours)', fontsize=9)
    ax3.set_ylabel(disp, fontsize=9)
    ax3.legend(fontsize=7, loc='upper right')
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('figure18_spc_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: figure18_spc_comparison.png")
    plt.close()

# ── Entry ────────────────────────────────────────────────────
if __name__ == '__main__':
    df, t_hours = load_data('P74-00318.txt')
    phases = define_phases(t_hours)
    res = run_analysis(df, t_hours, phases)
    plot_figure18(t_hours, phases, res)
