import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (f1_score, roc_auc_score,
                             precision_score, recall_score, roc_curve)
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
import xgboost as xgb
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# v4 修改说明（相对于v3）：
# 1. 纯XGBoost基线同样加入训练集最优阈值搜索，确保对比公平
# 2. 输出每折逐折原始指标，用于Wilcoxon统计显著性检验
# 3. 其余逻辑与v3完全一致，保证实验可复现性
# ============================================================

print("读取数据...")
X = pd.read_csv('secom_features_clean.csv')
y = pd.read_csv('secom_labels_clean.csv').values.ravel()
print(f"特征矩阵: {X.shape}, 标签: {dict(pd.Series(y).value_counts())}")


def evaluate(y_true, y_pred, y_prob=None):
    res = {
        'F1':        round(f1_score(y_true, y_pred, zero_division=0), 4),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall':    round(recall_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_prob is not None:
        res['AUC'] = round(roc_auc_score(y_true, y_prob), 4)
    return res


def find_best_threshold(y_true, y_prob):
    """在训练集上搜索最优F1阈值（统一函数，两个方法共用）"""
    best_thresh, best_f1 = 0.5, 0
    for t in np.arange(0.1, 0.7, 0.05):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_thresh


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {
    'Baseline_SPC':     [],
    'Baseline_XGBoost': [],
    'Proposed_3Layer':  [],
}

# 逐折原始数据（用于统计检验）
fold_records = []
last_fold = {}

print("\n开始5折交叉验证（v4：公平阈值对比）...")

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"  Fold {fold+1}/5 ...", end=' ')

    X_train = X.iloc[train_idx].values
    X_test  = X.iloc[test_idx].values
    y_train = y[train_idx]
    y_test  = y[test_idx]

    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    # ----------------------------------------------------------
    # Baseline 1: SPC（逻辑与v3相同）
    # ----------------------------------------------------------
    means = X_train.mean(axis=0)
    stds  = X_train.std(axis=0)
    stds[stds == 0] = 1e-6
    z = np.abs((X_test - means) / stds)
    spc_pred = (z > 3).any(axis=1).astype(int)
    spc_res = evaluate(y_test, spc_pred)
    results['Baseline_SPC'].append(spc_res)

    # ----------------------------------------------------------
    # Baseline 2: 纯XGBoost
    # [v4修改] 加入与三层方法相同的训练集最优阈值搜索
    # ----------------------------------------------------------
    xgb_b = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=scale_pos, subsample=0.8,
        colsample_bytree=0.8, eval_metric='logloss',
        random_state=42, verbosity=0
    )
    xgb_b.fit(X_train, y_train)
    prob_b = xgb_b.predict_proba(X_test)[:, 1]

    # [v4新增] 在训练集上搜索最优阈值
    prob_b_train = xgb_b.predict_proba(X_train)[:, 1]
    best_thresh_b = find_best_threshold(y_train, prob_b_train)
    pred_b = (prob_b >= best_thresh_b).astype(int)

    xgb_res = evaluate(y_test, pred_b, prob_b)
    results['Baseline_XGBoost'].append(xgb_res)

    # ----------------------------------------------------------
    # 本文方法：三层协同（逻辑与v3完全相同）
    # ----------------------------------------------------------
    pca = PCA(n_components=30, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca  = pca.transform(X_test)

    lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
    lof.fit(X_train)
    uncertainty_train = -lof.score_samples(X_train)
    uncertainty_test  = -lof.score_samples(X_test)

    u_min = uncertainty_train.min()
    u_max = uncertainty_train.max()
    uncertainty_train_norm = (uncertainty_train - u_min) / (u_max - u_min + 1e-6)
    uncertainty_test_norm  = (uncertainty_test  - u_min) / (u_max - u_min + 1e-6)

    X_train_aug = np.column_stack([X_train, X_train_pca, uncertainty_train_norm])
    X_test_aug  = np.column_stack([X_test,  X_test_pca,  uncertainty_test_norm])

    xgb_3l = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=scale_pos, subsample=0.8,
        colsample_bytree=0.8, eval_metric='logloss',
        random_state=42, verbosity=0
    )
    xgb_3l.fit(X_train_aug, y_train)
    prob_3l = xgb_3l.predict_proba(X_test_aug)[:, 1]

    # 熔断机制（与v3相同）
    fallback_threshold = np.percentile(uncertainty_test_norm, 85)
    fallback_mask = uncertainty_test_norm > fallback_threshold

    prob_final = prob_3l.copy()
    spc_prob = spc_pred.astype(float) * 0.8
    for i in range(len(prob_final)):
        if fallback_mask[i]:
            w = min(uncertainty_test_norm[i], 1.0)
            prob_final[i] = (1 - w) * prob_3l[i] + w * spc_prob[i]

    # 训练集最优阈值（与v3相同）
    prob_train_3l = xgb_3l.predict_proba(X_train_aug)[:, 1]
    best_thresh_3l = find_best_threshold(y_train, prob_train_3l)
    pred_final = (prob_final >= best_thresh_3l).astype(int)

    proposed_res = evaluate(y_test, pred_final, prob_final)
    results['Proposed_3Layer'].append(proposed_res)

    # 记录逐折数据（用于统计检验）
    fold_record = {
        'fold': fold + 1,
        'xgb_F1':       xgb_res['F1'],
        'xgb_Recall':   xgb_res['Recall'],
        'xgb_AUC':      xgb_res.get('AUC', np.nan),
        'prop_F1':      proposed_res['F1'],
        'prop_Recall':  proposed_res['Recall'],
        'prop_AUC':     proposed_res.get('AUC', np.nan),
        'thresh_xgb':   best_thresh_b,
        'thresh_prop':  best_thresh_3l,
    }
    fold_records.append(fold_record)

    if fold == 4:
        last_fold = {
            'y_test':     y_test,
            'prob_b':     prob_b,
            'prob_final': prob_final,
        }

    print(f"完成 (阈值_XGB={best_thresh_b:.2f}, 阈值_3L={best_thresh_3l:.2f})")


# ============================================================
# 汇总结果
# ============================================================
print("\n" + "="*65)
print("实验结果汇总（5折均值 ± 标准差）[v4：公平阈值对比]")
print("="*65)

summary = {}
for method, fold_results in results.items():
    df_r = pd.DataFrame(fold_results)
    means = df_r.mean().round(4)
    stds  = df_r.std().round(4)
    summary[method] = means
    print(f"\n{method}:")
    for col in df_r.columns:
        print(f"  {col}: {means[col]:.4f} ± {stds[col]:.4f}")

# ============================================================
# 逐折原始数据输出（用于Wilcoxon检验）
# ============================================================
print("\n" + "="*65)
print("逐折原始数据（用于统计显著性检验）")
print("="*65)
fold_df = pd.DataFrame(fold_records)
print(fold_df.to_string(index=False))
fold_df.to_csv('fold_level_results_v4.csv', index=False)
print("\n已保存: fold_level_results_v4.csv")

# ============================================================
# Wilcoxon符号秩检验（F1 和 Recall）
# ============================================================
from scipy import stats

print("\n" + "="*65)
print("Wilcoxon符号秩检验（提出方法 vs 纯XGBoost）")
print("="*65)

for metric in ['F1', 'Recall', 'AUC']:
    col_xgb  = f'xgb_{metric}'
    col_prop = f'prop_{metric}'
    if col_xgb in fold_df.columns and col_prop in fold_df.columns:
        xgb_vals  = fold_df[col_xgb].values
        prop_vals = fold_df[col_prop].values
        if not np.isnan(xgb_vals).any() and not np.isnan(prop_vals).any():
            stat, p = stats.wilcoxon(prop_vals, xgb_vals, alternative='greater')
            print(f"  {metric}: W={stat:.1f}, p={p:.4f} "
                  f"({'显著 p<0.05' if p < 0.05 else '不显著'})")

# ============================================================
# 论文用对比表格
# ============================================================
print("\n" + "="*65)
print("论文用对比表格（Table 2 v4版本）")
print("="*65)
summary_df = pd.DataFrame(summary).T
print(summary_df.to_string())
summary_df.to_csv('experiment_results_v4.csv')
print("\n已保存: experiment_results_v4.csv")

# F1相对提升
f1_xgb  = summary['Baseline_XGBoost']['F1']
f1_prop = summary['Proposed_3Layer']['F1']
print(f"\nF1相对提升（vs 纯XGBoost）: {(f1_prop - f1_xgb) / f1_xgb * 100:.1f}%")

# ============================================================
# ROC曲线（Fold 5）
# ============================================================
plt.figure(figsize=(7, 5))
fpr1, tpr1, _ = roc_curve(last_fold['y_test'], last_fold['prob_b'])
auc1 = roc_auc_score(last_fold['y_test'], last_fold['prob_b'])
plt.plot(fpr1, tpr1, 'b--', label=f'Pure XGBoost (Fold 5 AUC={auc1:.3f})')

fpr2, tpr2, _ = roc_curve(last_fold['y_test'], last_fold['prob_final'])
auc2 = roc_auc_score(last_fold['y_test'], last_fold['prob_final'])
plt.plot(fpr2, tpr2, 'r-', linewidth=2,
         label=f'Proposed 3-Layer (Fold 5 AUC={auc2:.3f})')

plt.plot([0, 1], [0, 1], 'k:', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison on SECOM Dataset')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_comparison_v4.png', dpi=150)
print("已保存: roc_comparison_v4.png")
print("\n全部完成！[v4]")
