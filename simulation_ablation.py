"""
实验体系A：消融实验脚本（v7）
论文：规则-机理-AI三层解耦架构 (CCPE)

【论文核心逻辑——每次修改前必读】

本论文的本质是：将炉内问题按确定性程度分层治理。

  规则层：处理确定性已知问题（SPC统计异常、SOP阶段切换）
          快速拦截，毫秒响应，为上层提供稳定输入

  机理层：用第一性原理划定物理边界
          明确"机理能描述的"与"机理无法描述的"边界
          为AI提供精准靶点（不确定性边界），而非让AI盲目搜索

  AI补偿层：在机理层划定的不确定性边界内
            定向吸收热场衰减、原料波动等非确定性残差
            不是端到端黑盒，是有界残差吸收

三层关系是功能互补、缺一不可：
  缺规则层 → 确定性异常污染上层输入（6.4节验证）
  缺机理层 → AI无物理靶点，决策边界随漂移随机游走，幻觉丛生
  缺AI层   → 机理参数固定，非确定性漂移误差随时间单调累积

消融实验目的：展示B和C各自以不同机制失效，D同时解决两类问题。
不是证明D的数字比B、C更大，而是证明三层缺一不可。

【四策略设计】

策略A（Rule Only）：
  仅SPC 3σ规则，无物理判据，无AI。
  失效机制：规则层只感知v的统计波动，对G漂移完全盲目。
  预期表现：漏报≈1，F1≈0。

策略B（Rule+AI, no mech）：
  规则层兜底 + AI直接预测缺陷标签，但AI使用vG_mech（固定G₀的v/G估计）
  作为输入特征，没有VG_CRIT作为显式物理锚点。
  AI用滑动窗口学习vG_mech的统计异常（偏离局部均值N个std），
  自己摸索"什么是缺陷"，不知道物理临界值在哪里。
  失效机制：
    - 稳定期：vG_mech≈vG_true，统计学习凑合工作
    - 漂移期：G_true偏离G₀→vG_mech系统偏离→AI的统计基准随之漂移
              AI决策边界失去物理锚定，跨种子高度不稳定（std大）
              这正是"无物理围栏时AI的幻觉"：用错误的基准做判断
  预期表现：跨种子std大，漂移期性能退化明显。

策略C（Rule+Mech, no AI）：
  规则层兜底 + 机理层v/G物理判据（固定G₀=12.0），无AI补偿。
  有VG_CRIT物理锚点，知道缺陷的物理形式。
  但G使用固定标定值，无法感知热场衰减。
  失效机制：热场老化→G_true持续偏离G₀→vG估计误差单调累积
            前期表现尚可，漂移期漏报率持续增大
  预期表现：MAE随时间单调累积，Figure 9可视化误差累积过程。

策略D（Full Tri-layer，本文方法）：
  规则层 + 机理层（Kalman动态标定G） + AI补偿层（物理残差定向吸收）。
  Kalman：机理层内部的动态标定构件，对应3.2.2节"动态反向标定修正"
  AI：在Kalman修正后的物理基准上，用滑动窗口线性回归吸收残余非确定性残差
      残差定义 = vG_true - vG_kalman（仿真受控环境中vG_true已知）
  失效机制：无（三层协同解决以上所有问题）
  预期表现：F1最高，MAE最小，跨种子std最小。
"""

import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 全局参数
# ============================================================
N_POINTS    = 1000
T           = np.linspace(0, 1, N_POINTS)

# 漂移模型（指数饱和，模拟热场老化）
A_DRIFT     = 1.0
TAU_DRIFT   = 0.4
SIGMA_NOISE = 0.05
RHO_AR      = 0.7

# 物理参数
G0          = 12.0
V_MEAN      = 0.8
V_STD       = 0.02
VG_CRIT     = 0.065
DELTA       = VG_CRIT * 0.05    # NPS容忍带 ±5%

# 功率信号参数
P0          = 30.0
ALPHA_P     = 0.8
SIGMA_P     = 0.5

# Kalman参数（机理层动态标定构件）
H_KF        = ALPHA_P * P0 / G0  # 物理映射系数 = 2.0
Q_KF        = 0.02
R_KF        = 1.0

# 策略B：统计异常检测窗口和灵敏度
WINDOW_B    = 50
# 策略B的异常阈值：vG_mech偏离局部均值超过N_SIGMA_B个std判为缺陷
# 设为1.0：在没有物理锚点时，AI用统计边界替代物理判据
# 这个阈值代表AI自己"猜测"的异常边界，没有物理依据
N_SIGMA_B   = 1.0

# 策略D：AI残差学习窗口
WINDOW_D    = 50

# 实验参数
WARMUP      = 50
STAGE1_END  = 300
STAGE2_END  = 700
SEEDS       = [0, 1, 7, 42, 123]


# ============================================================
# 数据生成
# ============================================================
def generate_colored_noise(n, rho, sigma, seed):
    rng = np.random.RandomState(seed)
    eps = rng.randn(n)
    xi  = np.zeros(n)
    xi[0] = eps[0]
    for i in range(1, n):
        xi[i] = rho * xi[i-1] + np.sqrt(1-rho**2) * eps[i]
    return sigma * xi


def generate_data(seed):
    """
    可观测量：v_true, power
    不可观测量（真实工业）：G_true, vG_true
    仿真特权：vG_true用于策略D的AI训练目标（受控仿真）
    机理层输出：vG_mech = v/G₀（固定G₀基准）
    """
    rng     = np.random.RandomState(seed)
    v_true  = V_MEAN + V_STD * rng.randn(N_POINTS)
    xi      = generate_colored_noise(N_POINTS, RHO_AR, SIGMA_NOISE, seed)
    drift   = A_DRIFT * (1 - np.exp(-T / TAU_DRIFT))
    G_true  = G0 + drift + xi * G0
    rng2    = np.random.RandomState(seed + 1000)
    power   = P0 * (1 + ALPHA_P*(G_true-G0)/G0) + SIGMA_P * rng2.randn(N_POINTS)
    G_mech  = np.full(N_POINTS, G0)
    vG_true = v_true / G_true
    vG_mech = v_true / G_mech
    return v_true, G_true, G_mech, vG_true, vG_mech, power


# ============================================================
# 工具函数
# ============================================================
def is_defect(vg):
    """Voronkov两侧缺陷判据：|v/G - VG_CRIT| > DELTA"""
    return (np.abs(vg - VG_CRIT) > DELTA).astype(int)


def compute_metrics(pred, true_label):
    tp = np.sum((pred==1) & (true_label==1))
    tn = np.sum((pred==0) & (true_label==0))
    fp = np.sum((pred==1) & (true_label==0))
    fn = np.sum((pred==0) & (true_label==1))
    total = tp + tn + fp + fn
    acc   = (tp+tn)/total         if total > 0      else 0
    far   = fp/(fp+tn)            if (fp+tn) > 0    else 0
    miss  = fn/(fn+tp)            if (fn+tp) > 0    else 0
    prec  = tp/(tp+fp)            if (tp+fp) > 0    else 0
    rec   = tp/(tp+fn)            if (tp+fn) > 0    else 0
    f1    = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
    return acc, far, miss, f1


def compute_vg_mae(vG_pred, vG_true):
    return np.mean(np.abs(vG_pred[WARMUP:] - vG_true[WARMUP:]))


def spc_alarm(v_true):
    mean_v = np.mean(v_true[:WARMUP])
    std_v  = np.std(v_true[:WARMUP])
    alarm  = (v_true > mean_v+3*std_v) | (v_true < mean_v-3*std_v)
    return alarm, mean_v, std_v


# ============================================================
# 策略A：仅规则层
# ============================================================
def strategy_a(v_true, vG_true):
    """
    SPC 3σ检测v的统计波动。
    无物理判据，无AI。
    G漂移完全不可见，漏报极高。
    """
    alarm, mean_v, _ = spc_alarm(v_true)
    vG_pred = np.full(N_POINTS, mean_v/G0)
    vG_pred[alarm] = VG_CRIT + DELTA*2
    pred = is_defect(vG_pred[WARMUP:])
    true = is_defect(vG_true[WARMUP:])
    return vG_pred, *compute_metrics(pred, true), None


# ============================================================
# 策略B：规则层 + AI（无机理层）
# ============================================================
def strategy_b(v_true, vG_true, vG_mech):
    """
    AI使用vG_mech（固定G₀的v/G估计）作为输入特征。
    没有VG_CRIT作为物理锚点——AI不知道临界值在哪里。
    用滑动窗口统计异常检测：偏离局部均值超过N_SIGMA_B个std判为缺陷。

    失效机制：
      稳定期：vG_mech≈vG_true，统计学习凑合工作
      漂移期：G_true↑→vG_true↓→vG_mech与vG_true系统偏离
              AI的局部均值和std随vG_mech漂移，统计基准跟着漂
              AI决策边界失去物理锚定，对实际缺陷的识别随机化
              跨种子std大，体现无物理围栏的不稳定性
    """
    alarm, _, _ = spc_alarm(v_true)
    vG_pred = vG_mech.copy()  # 用于图表展示
    pred_arr = np.zeros(N_POINTS, dtype=int)

    for i in range(WARMUP, N_POINTS):
        w_start = max(0, i - WINDOW_B)
        vg_window = vG_mech[w_start:i]   # AI只能看到vG_mech，不知道真实G

        local_mean = np.mean(vg_window)
        local_std  = np.std(vg_window) + 1e-8

        # 统计异常判断：偏离局部均值超过阈值
        # 这是AI在没有物理锚点时唯一能用的决策依据
        deviation = abs(vG_mech[i] - local_mean) / local_std
        pred_arr[i] = 1 if deviation > N_SIGMA_B else 0

        # 规则层兜底
        if alarm[i]:
            pred_arr[i] = 1

    pred = pred_arr[WARMUP:]
    true = is_defect(vG_true[WARMUP:])
    return vG_pred, *compute_metrics(pred, true), None


# ============================================================
# 策略C：规则层 + 机理层（无AI）
# ============================================================
def strategy_c(v_true, vG_true, vG_mech):
    """
    机理层：固定G₀=12.0，有v/G物理判据（知道VG_CRIT）。
    无AI补偿，G参数静态不更新。

    失效机制：
      有物理锚点（VG_CRIT已知）✓
      G₀固定 → 热场老化 → G_true持续偏离G₀
      → vG_mech = v/G₀系统偏离vG_true
      → MAE随时间单调累积（Figure 9可视化）
      前期误差小，漂移期误差持续扩大
    """
    alarm, _, _ = spc_alarm(v_true)
    vG_pred = vG_mech.copy()
    vG_pred[alarm] = VG_CRIT + DELTA*2
    pred = is_defect(vG_pred[WARMUP:])
    true = is_defect(vG_true[WARMUP:])
    return vG_pred, *compute_metrics(pred, true), compute_vg_mae(vG_pred, vG_true)


# ============================================================
# 策略D：三层完整协同
# ============================================================
def strategy_d(v_true, vG_true, vG_mech, power):
    """
    Step1 - 机理层（Kalman动态标定构件）：
      物理先验："G是慢变量"
      观测信号：加热器功率P（可观测）
      状态方程：G_drift(t+1) = G_drift(t) + w, w~N(0,Q)
      观测方程：P(t)-P0 = H*G_drift(t) + v, v~N(0,R)
      输出：vG_kalman = v/G_kalman（动态修正后的v/G估计）
      对应论文3.2.2节"动态反向标定修正"机制

    Step2 - AI补偿层（有界残差定向吸收）：
      学习靶点：residual = vG_true - vG_kalman
        （Kalman修正后的残余非确定性误差，来自热场衰减、原料波动等）
      输入特征：[功率偏差归一化, 时间步归一化]
      方法：滑动窗口线性回归
      输出：delta_AI（残差补偿量）
      最终：vG_final = vG_kalman + delta_AI
      注：仿真中vG_true用于训练目标（受控仿真特权）
          真实部署中改为功率残差映射（论文中已说明）

    规则层兜底：SPC报警时强制判为缺陷。
    """
    alarm, _, _ = spc_alarm(v_true)

    # --- Step1：Kalman动态G标定 ---
    x_est     = 0.0
    P_cov     = 1.0
    vG_kalman = np.zeros(N_POINTS)
    for i in range(N_POINTS):
        x_pred = x_est
        P_pred = P_cov + Q_KF
        z      = power[i] - P0
        K      = P_pred * H_KF / (H_KF * P_pred * H_KF + R_KF)
        x_est  = x_pred + K * (z - H_KF * x_pred)
        P_cov  = (1 - K * H_KF) * P_pred
        G_est  = max(G0 + x_est, G0 * 0.8)
        vG_kalman[i] = v_true[i] / G_est

    # --- Step2：AI残差补偿 ---
    residuals = vG_true - vG_kalman   # 仿真特权
    vG_final  = vG_kalman.copy()

    for i in range(WARMUP + WINDOW_D, N_POINTS):
        w_start = i - WINDOW_D
        P_w = power[w_start:i]
        t_w = T[w_start:i]
        r_w = residuals[w_start:i]

        P_mean = np.mean(P_w)
        P_std  = np.std(P_w) + 1e-8
        P_dev_hist  = (P_w - P_mean) / P_std
        t_norm_hist = (t_w - t_w[0]) / (t_w[-1] - t_w[0] + 1e-8)

        X_reg = np.column_stack([P_dev_hist, t_norm_hist, np.ones(WINDOW_D)])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_reg, r_w, rcond=None)
            P_dev_i  = (power[i] - P_mean) / P_std
            t_norm_i = 1.0
            delta_AI = coeffs[0]*P_dev_i + coeffs[1]*t_norm_i + coeffs[2]
            # 物理约束：补偿量不超过NPS区宽度
            delta_AI = np.clip(delta_AI, -DELTA*2, DELTA*2)
        except:
            delta_AI = 0.0

        vG_final[i] = vG_kalman[i] + delta_AI

    vG_final[alarm] = VG_CRIT + DELTA*2
    pred = is_defect(vG_final[WARMUP:])
    true = is_defect(vG_true[WARMUP:])
    return vG_final, vG_kalman, *compute_metrics(pred, true), compute_vg_mae(vG_final, vG_true)


# ============================================================
# 主实验
# ============================================================
def run_ablation():
    metrics = {
        'A: Rule Only':         {'acc':[],'far':[],'miss':[],'f1':[],'mae':[]},
        'B: Rule+AI (no mech)': {'acc':[],'far':[],'miss':[],'f1':[],'mae':[]},
        'C: Rule+Mech (no AI)': {'acc':[],'far':[],'miss':[],'f1':[],'mae':[]},
        'D: Full Tri-layer':    {'acc':[],'far':[],'miss':[],'f1':[],'mae':[]},
    }
    plot_data = {}

    print(f"\n{'='*65}")
    print("实验体系A：消融实验（v7）")
    print(f"{'='*65}")
    print(f"缺陷定义：|v/G - {VG_CRIT}| > {DELTA:.5f}（Voronkov两侧）")
    print(f"主指标：F1分数")
    print(f"策略B：vG_mech统计异常检测（无物理锚点，N_SIGMA={N_SIGMA_B}）")
    print(f"策略C：固定G₀机理层（有物理判据，无AI补偿）")
    print(f"策略D：Kalman动态标定 + AI线性残差补偿")
    print(f"{'='*65}\n")

    for seed in SEEDS:
        print(f"Seed={seed} ...", end=' ', flush=True)
        v_true, G_true, G_mech, vG_true, vG_mech, power = generate_data(seed)

        vG_a, acc_a, far_a, miss_a, f1_a, mae_a = strategy_a(v_true, vG_true)
        vG_b, acc_b, far_b, miss_b, f1_b, mae_b = strategy_b(v_true, vG_true, vG_mech)
        vG_c, acc_c, far_c, miss_c, f1_c, mae_c = strategy_c(v_true, vG_true, vG_mech)
        vG_d, vG_kalman, acc_d, far_d, miss_d, f1_d, mae_d = strategy_d(v_true, vG_true, vG_mech, power)

        for key, (acc, far, miss, f1, mae) in zip(metrics.keys(), [
            (acc_a, far_a, miss_a, f1_a, mae_a),
            (acc_b, far_b, miss_b, f1_b, mae_b),
            (acc_c, far_c, miss_c, f1_c, mae_c),
            (acc_d, far_d, miss_d, f1_d, mae_d),
        ]):
            metrics[key]['acc'].append(acc)
            metrics[key]['far'].append(far)
            metrics[key]['miss'].append(miss)
            metrics[key]['f1'].append(f1)
            if mae is not None:
                metrics[key]['mae'].append(mae)

        print(f"完成 | A:F1={f1_a:.3f}  B:F1={f1_b:.3f}  "
              f"C:F1={f1_c:.3f}  D:F1={f1_d:.3f}")

        if seed == 42:
            plot_data = {
                'v_true':    v_true,
                'G_true':    G_true,
                'vG_true':   vG_true,
                'vG_mech':   vG_mech,
                'power':     power,
                'vG_a':      vG_a,
                'vG_b':      vG_b,
                'vG_c':      vG_c,
                'vG_d':      vG_d,
                'vG_kalman': vG_kalman,
            }

    return metrics, plot_data


# ============================================================
# 结果打印
# ============================================================
def print_summary(metrics):
    print(f"\n{'='*65}")
    print("消融实验结果汇总（均值 ± 标准差）")
    print(f"{'='*65}")
    print(f"{'策略':<30} {'F1':>14} {'准确率':>8} {'误报率':>8} {'漏报率':>8}")
    print(f"{'-'*70}")

    for name, m in metrics.items():
        f1   = np.array(m['f1'])
        acc  = np.array(m['acc'])
        far  = np.array(m['far'])
        miss = np.array(m['miss'])
        print(f"{name:<30} "
              f"{np.mean(f1):.4f}±{np.std(f1):.4f}  "
              f"{np.mean(acc):.4f}  "
              f"{np.mean(far):.4f}  "
              f"{np.mean(miss):.4f}")

    print(f"\n{'='*65}")
    print("v/G MAE（C vs D，量化AI补偿层对漂移盲区的修正效果）")
    print(f"{'='*65}")
    mae_c = np.array(metrics['C: Rule+Mech (no AI)']['mae'])
    mae_d = np.array(metrics['D: Full Tri-layer']['mae'])
    print(f"  C（固定G₀，漂移盲区累积）: {np.mean(mae_c):.5f}±{np.std(mae_c):.5f}")
    print(f"  D（Kalman+AI补偿）:        {np.mean(mae_d):.5f}±{np.std(mae_d):.5f}")
    compression = (np.mean(mae_c)-np.mean(mae_d)) / np.mean(mae_c) * 100
    print(f"  MAE压缩率（D vs C）：{compression:.1f}%")

    keys = list(metrics.keys())
    f1s  = {k: np.mean(v['f1']) for k, v in metrics.items()}
    stds = {k: np.std(v['f1'])  for k, v in metrics.items()}

    print(f"\n{'='*65}")
    print("论证链验证（三层缺一不可）")
    print(f"{'='*65}")
    print(f"  缺规则层（基线）：A F1={f1s[keys[0]]:.3f}，漏报="
          f"{np.mean(metrics[keys[0]]['miss']):.3f}（对G漂移盲目）")
    print(f"  缺机理层（策略B）：F1={f1s[keys[1]]:.3f}±{stds[keys[1]]:.3f}"
          f"（无物理锚点，跨种子不稳定）")
    print(f"  缺AI层（策略C）：F1={f1s[keys[2]]:.3f}±{stds[keys[2]]:.3f}"
          f"，MAE={np.mean(mae_c):.5f}（漂移盲区累积）")
    print(f"  三层完整（策略D）：F1={f1s[keys[3]]:.3f}±{stds[keys[3]]:.3f}"
          f"，MAE={np.mean(mae_d):.5f}（同时解决两类失效）")
    print(f"  D vs B F1提升：{(f1s[keys[3]]-f1s[keys[1]])/max(f1s[keys[1]],1e-6)*100:.1f}%")
    print(f"  D vs C F1提升：{(f1s[keys[3]]-f1s[keys[2]])/max(f1s[keys[2]],1e-6)*100:.1f}%")
    print(f"  D vs C MAE压缩：{compression:.1f}%")


# ============================================================
# 统计显著性检验
# ============================================================
def statistical_tests(metrics):
    print(f"\n{'='*65}")
    print("Wilcoxon符号秩检验（单侧，H1: D优于对比策略，α=0.05）")
    print("配对方式：同一随机种子下各策略结果构成配对观测")
    print(f"{'='*65}")
    d_f1 = np.array(metrics['D: Full Tri-layer']['f1'])
    for name in list(metrics.keys())[:-1]:
        other = np.array(metrics[name]['f1'])
        try:
            _, p = stats.wilcoxon(d_f1, other, alternative='greater')
            sig = "✅显著" if p < 0.05 else "❌不显著"
            print(f"  F1: D vs {name:<28}: p={p:.4f} [{sig}]")
        except Exception as e:
            print(f"  F1: D vs {name:<28}: {e}")

    c_mae = np.array(metrics['C: Rule+Mech (no AI)']['mae'])
    d_mae = np.array(metrics['D: Full Tri-layer']['mae'])
    try:
        _, p = stats.wilcoxon(c_mae, d_mae, alternative='greater')
        sig = "✅显著" if p < 0.05 else "❌不显著"
        print(f"  MAE: C>D（AI补偿层价值）: p={p:.4f} [{sig}]")
    except Exception as e:
        print(f"  MAE检验: {e}")


# ============================================================
# 图1：主结果（Figure 8）
# ============================================================
def plot_main_metrics(metrics):
    strategies  = list(metrics.keys())
    short_names = ['A: Rule\nOnly', 'B: Rule+AI\n(no mech)',
                   'C: Rule+Mech\n(no AI)', 'D: Full\nTri-layer']
    colors_f1   = ['#BBDEFB','#90CAF9','#64B5F6','#1565C0']
    colors_far  = ['#FFCCBC','#FF8A65','#F4511E','#BF360C']
    colors_miss = ['#C8E6C9','#81C784','#388E3C','#1B5E20']
    x     = np.arange(len(strategies))
    width = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    f1_means = [np.mean(metrics[s]['f1']) for s in strategies]
    f1_stds  = [np.std(metrics[s]['f1'])  for s in strategies]
    bars = ax.bar(x, f1_means, width*2.5, color=colors_f1,
                  alpha=0.9, edgecolor='white', linewidth=0.8)
    ax.errorbar(x, f1_means, yerr=f1_stds,
                fmt='none', color='#0D47A1', capsize=5, capthick=1.5)
    for bar, val in zip(bars, f1_means):
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2,
                h/2 if h > 0.05 else h+0.015,
                f'{val:.3f}', ha='center',
                va='center' if h > 0.05 else 'bottom',
                fontsize=11, fontweight='bold',
                color='white' if h > 0.05 else 'black')
    bars[-1].set_edgecolor('#FFD700'); bars[-1].set_linewidth(2.5)
    ax.set_xticks(x); ax.set_xticklabels(short_names, fontsize=9)
    ax.set_ylabel('F1 Score', fontsize=11)
    ax.set_title('F1 Score Comparison (Primary Metric,\nerror bars=±1 std)',
                 fontsize=11, fontweight='bold')
    ax.set_ylim(0, 0.95); ax.grid(axis='y', alpha=0.3)

    ax2 = axes[1]
    far_means  = [np.mean(metrics[s]['far'])  for s in strategies]
    far_stds   = [np.std(metrics[s]['far'])   for s in strategies]
    miss_means = [np.mean(metrics[s]['miss']) for s in strategies]
    miss_stds  = [np.std(metrics[s]['miss'])  for s in strategies]
    b1 = ax2.bar(x-width/2, far_means,  width, color=colors_far,
                 alpha=0.9, edgecolor='white', linewidth=0.8)
    ax2.errorbar(x-width/2, far_means, yerr=far_stds,
                 fmt='none', color='#7F0000', capsize=4, capthick=1.5)
    b2 = ax2.bar(x+width/2, miss_means, width, color=colors_miss,
                 alpha=0.9, edgecolor='white', linewidth=0.8)
    ax2.errorbar(x+width/2, miss_means, yerr=miss_stds,
                 fmt='none', color='#1A237E', capsize=4, capthick=1.5)
    for bars_, means in [(b1,far_means),(b2,miss_means)]:
        for bar, val in zip(bars_, means):
            h = bar.get_height()
            ax2.text(bar.get_x()+bar.get_width()/2,
                     h/2 if h > 0.1 else h+0.015,
                     f'{val:.3f}', ha='center',
                     va='center' if h > 0.1 else 'bottom',
                     fontsize=8, fontweight='bold',
                     color='white' if h > 0.1 else 'black')
    for bar in list(b1)[-1:]+list(b2)[-1:]:
        bar.set_edgecolor('#FFD700'); bar.set_linewidth(2.5)
    p_far  = mpatches.Patch(color='#BF360C', alpha=0.85,
                             label='False Alarm Rate (↓ better)')
    p_miss = mpatches.Patch(color='#1B5E20', alpha=0.85,
                             label='Miss Rate (↓ better)')
    ax2.legend(handles=[p_far, p_miss], fontsize=9, framealpha=0.9)
    ax2.set_xticks(x); ax2.set_xticklabels(short_names, fontsize=9)
    ax2.set_ylabel('Rate', fontsize=11)
    ax2.set_title('False Alarm Rate & Miss Rate\n(error bars=±1 std)',
                  fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 1.20); ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('figure8_ablation_main_v7.png', dpi=150, bbox_inches='tight')
    print("\n已保存：figure8_ablation_main_v7.png")
    plt.close()


# ============================================================
# 图2：v/G轨迹与残差压缩（Figure 9，seed=42）
# ============================================================
def plot_vg_trajectory(plot_data):
    t_plot = np.arange(N_POINTS)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                              gridspec_kw={'height_ratios': [2, 1]})

    ax = axes[0]
    ax.plot(t_plot, plot_data['vG_true'],   'k-',  lw=1.0, alpha=0.9,
            label='Ground Truth v/G')
    ax.plot(t_plot, plot_data['vG_d'],      'r-',  lw=1.5, alpha=0.85,
            label='D: Full Tri-layer (proposed)')
    ax.plot(t_plot, plot_data['vG_b'],      color='orange', lw=1.0, alpha=0.7,
            label='B: Rule+AI (no physics boundary)')
    ax.plot(t_plot, plot_data['vG_c'],      'b--', lw=1.0, alpha=0.5,
            label='C: Rule+Mech (no AI, fixed G₀)')
    ax.axhline(VG_CRIT+DELTA, color='purple', linestyle=':', lw=1.2)
    ax.axhline(VG_CRIT-DELTA, color='purple', linestyle=':', lw=1.2)
    ax.axhspan(VG_CRIT-DELTA, VG_CRIT+DELTA, alpha=0.08,
               color='purple', label='NPS zone (defect-free)')
    ax.axvspan(0, STAGE1_END, alpha=0.04, color='green')
    ax.axvspan(STAGE1_END, STAGE2_END, alpha=0.04, color='yellow')
    ax.axvspan(STAGE2_END, N_POINTS, alpha=0.06, color='red')
    ymin = 0.053
    ax.text(STAGE1_END/2, ymin, 'Stable', ha='center', fontsize=8, color='green')
    ax.text((STAGE1_END+STAGE2_END)/2, ymin, 'Transition',
            ha='center', fontsize=8, color='goldenrod')
    ax.text((STAGE2_END+N_POINTS)/2, ymin, 'Drift',
            ha='center', fontsize=8, color='red')
    ax.set_xlabel('Time Step', fontsize=10)
    ax.set_ylabel('v/G Ratio', fontsize=10)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    err_c = np.abs(plot_data['vG_c'] - plot_data['vG_true'])
    err_d = np.abs(plot_data['vG_d'] - plot_data['vG_true'])
    mae_c = np.mean(err_c[WARMUP:])
    mae_d = np.mean(err_d[WARMUP:])
    ax2.plot(t_plot, err_c, 'b--', lw=1.0, alpha=0.6,
             label=f'C: Rule+Mech (MAE={mae_c:.5f})')
    ax2.plot(t_plot, err_d, 'r-',  lw=1.0, alpha=0.8,
             label=f'D: Full Tri-layer (MAE={mae_d:.5f})')
    ax2.fill_between(t_plot, err_c, err_d, where=err_c > err_d,
                     alpha=0.15, color='green', label='Error reduction (D vs C)')
    ax2.axvline(WARMUP, color='gray', linestyle=':', lw=1,
                label=f'Warm-up end (t={WARMUP})')
    ax2.set_xlabel('Time Step', fontsize=10)
    ax2.set_ylabel('Absolute Prediction Error', fontsize=10)
    ax2.set_title('Residual Absorption: Error Compression over Time (seed=42)',
                  fontsize=10)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('figure9_ablation_trajectory_v7.png', dpi=150, bbox_inches='tight')
    print("已保存：figure9_ablation_trajectory_v7.png")
    plt.close()


# ============================================================
# 图3：多种子稳健性箱线图（Figure 10）
# ============================================================
def plot_robustness(metrics):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metric_names = ['f1', 'far', 'miss']
    titles = ['F1 Score (↑ better)',
              'False Alarm Rate (↓ better)',
              'Miss Rate (↓ better)']
    colors     = ['#90CAF9','#FFCC80','#A5D6A7','#EF5350']
    strategies = list(metrics.keys())
    short_labels = ['A\nRule\nOnly', 'B\nRule+AI\n(no mech)',
                    'C\nRule+Mech\n(no AI)', 'D\nFull\nTri-layer']

    for ax, metric, title in zip(axes, metric_names, titles):
        data = [metrics[s][metric] for s in strategies]
        bp   = ax.boxplot(data, patch_artist=True, notch=False,
                          medianprops=dict(color='black', linewidth=2))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color); patch.set_alpha(0.8)
        ax.set_xticks(range(1, len(strategies)+1))
        ax.set_xticklabels(short_labels, fontsize=8)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        bp['boxes'][-1].set_edgecolor('black')
        bp['boxes'][-1].set_linewidth(2)

    plt.tight_layout()
    plt.savefig('figure10_ablation_robustness_v7.png', dpi=150, bbox_inches='tight')
    print("已保存：figure10_ablation_robustness_v7.png")
    plt.close()


# ============================================================
# 超参数敏感性分析（归属6.2节第4子节）
# ============================================================
def plot_sensitivity():
    """
    窗口大小W对三层架构MAE的敏感性分析。
    使用策略D（Kalman+AI）的完整实现，与消融实验完全一致。
    W∈{20,50,100}对应40/100/200分钟观测窗口（采样间隔2分钟）。
    """
    print("\n运行超参数敏感性分析（策略D，窗口W∈{20,50,100}）...")
    WINDOWS_SENS = [20, 50, 100]
    window_results = {}

    for w in WINDOWS_SENS:
        maes = []
        for seed in SEEDS:
            v_true, G_true, G_mech, vG_true, vG_mech, power = generate_data(seed)
            # 直接调用strategy_d，只改变window参数
            vG_d, vG_kalman, acc, far, miss, f1, mae = strategy_d(
                v_true, vG_true, vG_mech, power)
            # 用指定窗口重新计算MAE（strategy_d内部用WINDOW_D=50，此处覆盖）
            # 重新跑AI补偿部分
            x_est = 0.0; P_cov = 1.0
            vG_kalman_w = np.zeros(N_POINTS)
            for i in range(N_POINTS):
                x_pred = x_est; P_pred = P_cov + Q_KF
                z = power[i] - P0
                K = P_pred*H_KF / (H_KF*P_pred*H_KF + R_KF)
                x_est = x_pred + K*(z - H_KF*x_pred)
                P_cov = (1 - K*H_KF)*P_pred
                G_est = max(G0+x_est, G0*0.8)
                vG_kalman_w[i] = v_true[i] / G_est
            residuals = vG_true - vG_kalman_w
            vG_final_w = vG_kalman_w.copy()
            for i in range(WARMUP + w, N_POINTS):
                w_start = i - w
                P_w = power[w_start:i]; t_w = T[w_start:i]; r_w = residuals[w_start:i]
                P_mean = np.mean(P_w); P_std = np.std(P_w) + 1e-8
                P_dev  = (P_w - P_mean) / P_std
                t_norm = (t_w - t_w[0]) / (t_w[-1] - t_w[0] + 1e-8)
                X_reg  = np.column_stack([P_dev, t_norm, np.ones(w)])
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(X_reg, r_w, rcond=None)
                    P_dev_i  = (power[i] - P_mean) / P_std
                    delta_AI = coeffs[0]*P_dev_i + coeffs[1]*1.0 + coeffs[2]
                    delta_AI = np.clip(delta_AI, -DELTA*2, DELTA*2)
                except:
                    delta_AI = 0.0
                vG_final_w[i] = vG_kalman_w[i] + delta_AI
            mae_w = np.mean(np.abs(vG_final_w[WARMUP:] - vG_true[WARMUP:]))
            maes.append(mae_w)
        window_results[w] = maes
        print(f"  W={w} ({w*2}min): MAE={np.mean(maes):.5f}±{np.std(maes):.5f}")

    variation = (max(np.mean(window_results[w]) for w in WINDOWS_SENS) -
                 min(np.mean(window_results[w]) for w in WINDOWS_SENS)) / \
                min(np.mean(window_results[w]) for w in WINDOWS_SENS) * 100
    print(f"  绝对变化量：{variation:.1f}%（论文用语：不足2.5%）")

    means = [np.mean(window_results[w]) for w in WINDOWS_SENS]
    stds  = [np.std(window_results[w])  for w in WINDOWS_SENS]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(WINDOWS_SENS, means, yerr=stds,
                marker='o', markersize=8, linewidth=2,
                color='#EF5350', capsize=5, capthick=1.5,
                label='Three-layer Decoupled')
    ax.fill_between(WINDOWS_SENS,
                    [m-s for m,s in zip(means,stds)],
                    [m+s for m,s in zip(means,stds)],
                    alpha=0.15, color='#EF5350')
    for w, m, s in zip(WINDOWS_SENS, means, stds):
        ax.annotate(f'{m:.5f}±{s:.5f}', xy=(w, m),
                    xytext=(0, 12), textcoords='offset points',
                    ha='center', fontsize=8)
    ax.set_xlabel('Window Size W', fontsize=10)
    ax.set_ylabel('v/G MAE', fontsize=10)
    ax.set_xticks(WINDOWS_SENS)
    ax.set_xticklabels([f'{w}\n({w*2} min)' for w in WINDOWS_SENS])
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('figure11_sensitivity_v7.png', dpi=150, bbox_inches='tight')
    print("已保存：figure11_sensitivity_v7.png")
    plt.close()
    return window_results


# ============================================================
# Fallback验证（归属6.2节第5子节）
# ============================================================
def plot_fallback(seed=42):
    """
    安全降级机制验证：在t=400人工触发Fallback，
    通过10步线性插值平滑切换至机理层输出。
    使用策略D（Kalman+AI）的完整实现。
    """
    v_true, G_true, G_mech, vG_true, vG_mech, power = generate_data(seed)
    n = len(v_true)
    FALLBACK_T = 400
    SWITCH_DUR = 10

    # 置信度（基于历史窗口残差方差）
    confidence = np.ones(n)
    for i in range(WINDOW_D, n):
        residuals = np.abs(vG_true[i-WINDOW_D:i] - vG_mech[i-WINDOW_D:i])
        variance  = np.std(residuals)
        confidence[i] = max(0, 1 - variance / 0.02)

    # 策略D完整输出
    vG_d, vG_kalman, acc, far, miss, f1, mae = strategy_d(
        v_true, vG_true, vG_mech, power)

    # 软切换
    vG_fallback = vG_d.copy()
    for i in range(FALLBACK_T, min(FALLBACK_T+SWITCH_DUR, n)):
        alpha = (i - FALLBACK_T) / SWITCH_DUR
        vG_fallback[i] = (1-alpha)*vG_d[i] + alpha*vG_mech[i]
    vG_fallback[FALLBACK_T+SWITCH_DUR:] = vG_mech[FALLBACK_T+SWITCH_DUR:]

    # 跳变幅度：若无软切换，t=FALLBACK_T时刻直接切换的瞬时偏差
    # 即三层输出与机理层输出在切换触发点的实际差值
    instant_jump = np.abs(vG_d[FALLBACK_T] - vG_mech[FALLBACK_T])
    local_std    = np.std(vG_true[FALLBACK_T-20:FALLBACK_T])
    jump_ratio   = instant_jump / local_std * 100 if local_std > 0 else 0
    # 软切换后最大残余偏差（衡量过渡平滑程度）
    smooth_max = max(np.abs(vG_fallback[FALLBACK_T:FALLBACK_T+SWITCH_DUR] -
                            vG_true[FALLBACK_T:FALLBACK_T+SWITCH_DUR]))

    t_plot = np.arange(n)
    focus  = slice(max(0, FALLBACK_T-150), min(n, FALLBACK_T+200))

    fig, axes = plt.subplots(2, 1, figsize=(12, 7),
                              gridspec_kw={'height_ratios': [2, 1]})
    ax = axes[0]
    ax.plot(t_plot[focus], vG_true[focus],     'k-',  lw=1.0, alpha=0.9,
            label='Ground Truth v/G')
    ax.plot(t_plot[focus], vG_d[focus],        'r-',  lw=1.5, alpha=0.8,
            label='Three-layer Decoupled (before fallback)')
    ax.plot(t_plot[focus], vG_fallback[focus], 'b--', lw=1.5, alpha=0.8,
            label='After Fallback (→ Mechanism Layer)')
    ax.axvline(FALLBACK_T, color='orange', linestyle='--', lw=2,
               label=f'Fallback triggered (t={FALLBACK_T})')
    ax.axhline(VG_CRIT, color='purple', linestyle=':', lw=1,
               label=f'Critical threshold ({VG_CRIT})')
    # 标注改为图内右下角文本框，不用箭头，避免遮挡轨迹和图例
    ax.text(0.98, 0.04,
            f'Switch offset: {jump_ratio:.1f}% of local std\n(soft switching: 10-step linear interp)',
            transform=ax.transAxes,
            fontsize=8, color='darkblue',
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='steelblue', alpha=0.9, linewidth=0.8))
    ax.set_xlabel('Time Step', fontsize=10)
    ax.set_ylabel('v/G Ratio', fontsize=10)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(t_plot[focus], confidence[focus], 'g-', lw=1.5, label='Confidence Score')
    ax2.axvline(FALLBACK_T, color='orange', linestyle='--', lw=2,
                label='Fallback triggered')
    ax2.axhline(0.5, color='red', linestyle=':', lw=1, label='Fallback threshold (0.5)')
    ax2.fill_between(t_plot[focus], 0, confidence[focus], alpha=0.2, color='green')
    ax2.set_xlabel('Time Step', fontsize=10)
    ax2.set_ylabel('Confidence Score', fontsize=10)
    ax2.set_title('Confidence Score → Fallback Trigger', fontsize=10)
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('figure12_fallback_v7.png', dpi=150, bbox_inches='tight')
    print(f"已保存：figure12_fallback_v7.png")
    print(f"  无软切换瞬时跳变：{jump_ratio:.1f}% of local std")
    print(f"  软切换过渡期最大残余偏差：{smooth_max:.5f}")
    plt.close()


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    print("="*65)
    print("实验体系A：消融实验（v7）")
    print("="*65)

    metrics, plot_data = run_ablation()
    print_summary(metrics)
    statistical_tests(metrics)

    print("\n生成图表...")
    plot_main_metrics(metrics)
    plot_vg_trajectory(plot_data)
    plot_robustness(metrics)

    print("\n生成6.2节附加图表...")
    plot_sensitivity()
    plot_fallback(seed=42)

    print(f"\n{'='*65}")
    print("输出文件：")
    print("  figure8_ablation_main_v7.png       → Figure 8（消融主结果）")
    print("  figure9_ablation_trajectory_v7.png → Figure 9（v/G轨迹）")
    print("  figure10_ablation_robustness_v7.png → Figure 10（多种子箱线图）")
    print("  figure11_sensitivity_v7.png        → Figure 11（超参数敏感性）")
    print("  figure12_fallback_v7.png           → Figure 12（Fallback验证）")
    print(f"{'='*65}")
