"""
修复8：三层架构原型系统延迟benchmark
测试环境请记录：CPU型号、内存大小、是否GPU
"""
import time
import numpy as np

# ── 参数（与实验一致）──────────────────────────────
WINDOW = 50
N_REPEAT = 1000  # 重复次数，取均值
G0 = 12.0
VG_CRIT = 0.065

# ── 模拟输入数据 ────────────────────────────────────
np.random.seed(42)
v_stream = np.random.normal(0.8, 0.02, N_REPEAT + WINDOW)
G_history = np.random.normal(G0, 0.3, N_REPEAT + WINDOW)
vG_history = v_stream / G_history

# ── 层1：规则层（SPC I-MR 3σ）──────────────────────
def rule_layer_single(v, v_history):
    mean_v = np.mean(v_history)
    std_v = np.std(v_history)
    ucl = mean_v + 3 * std_v
    lcl = mean_v - 3 * std_v
    return v > ucl or v < lcl

t0 = time.perf_counter()
for i in range(WINDOW, WINDOW + N_REPEAT):
    rule_layer_single(v_stream[i], v_stream[i-WINDOW:i])
t1 = time.perf_counter()
rule_latency_ms = (t1 - t0) / N_REPEAT * 1000

# ── 层2：机理层（v/G计算）──────────────────────────
def mech_layer_single(v, G0):
    vG = v / G0
    return vG, vG < VG_CRIT

t0 = time.perf_counter()
for i in range(WINDOW, WINDOW + N_REPEAT):
    mech_layer_single(v_stream[i], G0)
t1 = time.perf_counter()
mech_latency_ms = (t1 - t0) / N_REPEAT * 1000

# ── 层3：AI补偿层（滑动窗口线性回归残差估计）────────
from sklearn.linear_model import LinearRegression

def ai_layer_single(vG_window):
    X = np.arange(len(vG_window)).reshape(-1, 1)
    y = vG_window
    model = LinearRegression().fit(X, y)
    residual = y[-1] - model.predict([[len(vG_window)]])[0]
    return residual

t0 = time.perf_counter()
for i in range(WINDOW, WINDOW + N_REPEAT):
    ai_layer_single(vG_history[i-WINDOW:i])
t1 = time.perf_counter()
ai_latency_ms = (t1 - t0) / N_REPEAT * 1000

# ── 端到端三层总延迟 ───────────────────────────────
def tri_layer_single(v, v_history, vG_history, G0):
    # Rule
    mean_v = np.mean(v_history)
    std_v = np.std(v_history)
    ooc = v > mean_v + 3*std_v or v < mean_v - 3*std_v
    # Mech
    vG_mech = v / G0
    # AI
    X = np.arange(len(vG_history)).reshape(-1, 1)
    model = LinearRegression().fit(X, vG_history)
    correction = model.predict([[len(vG_history)]])[0]
    vG_final = vG_mech + correction
    return ooc, vG_mech, vG_final

t0 = time.perf_counter()
for i in range(WINDOW, WINDOW + N_REPEAT):
    tri_layer_single(
        v_stream[i],
        v_stream[i-WINDOW:i],
        vG_history[i-WINDOW:i],
        G0
    )
t1 = time.perf_counter()
e2e_latency_ms = (t1 - t0) / N_REPEAT * 1000

# ── 输出结果 ───────────────────────────────────────
print("=" * 55)
print("三层架构原型系统延迟Benchmark")
print(f"重复次数：{N_REPEAT}次，取均值")
print("=" * 55)
print(f"规则层单次判断延迟：  {rule_latency_ms*1000:.2f} μs")
print(f"机理层单次计算延迟：  {mech_latency_ms*1000:.2f} μs")
print(f"AI层单次推理延迟：    {ai_latency_ms*1000:.2f} μs")
print(f"三层端到端延迟：      {e2e_latency_ms*1000:.2f} μs")
print("=" * 55)
print("请记录您的测试硬件：")
print("  CPU：")
print("  内存：")
print("  OS：")
print("  Python版本：")