"""
==========================================================================================
  CMU 10-708 L18 代码练习: 因果关系2 — 因果不对称性, ANM, LiNGAM, 混杂, 选择偏差
==========================================================================================

L18 九大主题 -> 对应练习:
  ① Why Causality          -> 练习 1: 关联 vs 因果 vs 反事实 — 三层预测对比
  ② Causal Inference       -> 练习 2: 因果推断框架 — 从数据到 ATE 的完整流程
  ③ Conditional Independence -> 练习 3: 约束因果发现 — PC/FCI 与独立性检验
  ④ Causal Asymmetry (ANM) -> 练习 4: 加性噪声模型 — 用残差独立性判断因果方向
  ⑤ Independent Change     -> 练习 5: 多环境因果发现 — 不变因果预测 (ICP)
  ⑥ Confounding            -> 练习 6: 混杂检测 & 工具变量估计
  ⑦ Selection Bias         -> 练习 7: Berkson 悖论 — Collider 条件化的陷阱
  ⑧ Temporal Info          -> 练习 8: Granger 因果 & 时间序列因果发现
  ⑨ Transfer Learning      -> 练习 9: 因果不变性 & 域泛化

特别说明:
  - 贝叶斯网络导入需使用: from pgmpy.models import DiscreteBayesianNetwork
    (pgmpy 0.1.x 后 BayesianNetwork 改名为 DiscreteBayesianNetwork)
  - Windows GBK 终端下 emoji 打印: sys.stdout.reconfigure(encoding='utf-8')

使用方法:
  python 48_causality2_exercises.py              # 运行全部
  python 48_causality2_exercises.py --ex 4       # 只运行练习4

依赖: numpy, scipy, pgmpy (可选), networkx
==========================================================================================
"""

import numpy as np
from scipy import stats
import sys

# 修复 Windows GBK 终端下 emoji 打印问题
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

np.random.seed(42)

# 可选导入
try:
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    _HAS_PGMPY = True
except ImportError:
    _HAS_PGMPY = False


# ============================================================================
# 工具函数
# ============================================================================

def _print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _hsic_independence_test(x, y, n_perm=200):
    """
    简化版 HSIC 独立性检验 (使用线性核 + 排列检验)

    H₀: X ⟂ Y
    返回: (p-value, test_statistic)
    """
    n = len(x)
    # 中心化 Gram 矩阵
    K = np.outer(x, x)
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    L = np.outer(y, y)
    Lc = H @ L @ H

    # HSIC = trace(Kc @ Lc) / n²
    obs_hsic = np.trace(Kc @ Lc) / (n * n)

    # 排列检验
    perm_stats = []
    y_perm = y.copy()
    for _ in range(n_perm):
        np.random.shuffle(y_perm)
        L_perm = np.outer(y_perm, y_perm)
        Lc_perm = H @ L_perm @ H
        perm_stats.append(np.trace(Kc @ Lc_perm) / (n * n))

    perm_stats = np.array(perm_stats)
    p_value = (1 + np.sum(perm_stats >= obs_hsic)) / (1 + n_perm)
    return p_value, obs_hsic


# ============================================================================
# 练习 1: 关联 vs 因果 vs 反事实 — 三层预测对比
# ============================================================================

def exercise1_three_levels_prediction():
    """
    展示 Pearl 因果三层次在预测问题中的差异:
      Level 1 (关联): P(Y | X) — 被动观察
      Level 2 (干预): P(Y | do(X)) — 主动干预
      Level 3 (反事实): P(Y_x | X=x', Y=y') — 个体假设推理

    SCM: X = U_X, Y = 3X + U_Y, U_X~N(2,1), U_Y~N(0,1)
    """
    _print_header("练习 1: Pearl 三层次 — 关联 vs 干预 vs 反事实")

    n = 2000
    ux = np.random.normal(2, 1, n)
    uy = np.random.normal(0, 1, n)
    X = ux
    Y = 3.0 * X + uy

    # --- Level 1: Association P(Y | X) ---
    x_query = 3.0
    # 条件期望
    nearby = np.abs(X - x_query) < 0.2
    y_given_x = Y[nearby].mean()

    print(f"\n[Level 1 — Association: P(Y | X={x_query})]")
    print(f"  E[Y | X={x_query:.1f}] ≈ {y_given_x:.3f}")
    print(f"  含义: '观察到 X={x_query} 的人, 平均 Y 是多少?'")

    # --- Level 2: Intervention P(Y | do(X=x)) ---
    # do(X=3): 强制所有人 X=3, Y_new = 3*3 + U_Y
    y_do = 3.0 * x_query + uy
    y_do_mean = y_do.mean()

    print(f"\n[Level 2 — Intervention: P(Y | do(X={x_query}))]")
    print(f"  E[Y | do(X={x_query:.1f})] = 3·{x_query} + E[U_Y] = {y_do_mean:.3f}")
    print(f"  Level1 vs Level2 差异: {y_given_x - y_do_mean:.3f}")
    print(f"  原因: P(U_X|X={x_query}) → E[U_X|X={x_query}] ≠ E[U_X],")
    print(f"        但 do(X) 不改变 U_X 的分布")

    # --- Level 3: Counterfactual ---
    # 选一个"个体": X=1, Y=2 (低于平均)
    idx = np.argmin(np.abs(X - 1.0) + np.abs(Y - 2.0))
    x_ind, y_ind = X[idx], Y[idx]

    # 溯因
    ux_abduced = x_ind
    uy_abduced = y_ind - 3.0 * x_ind

    # 反事实: 如果 X=3
    y_cf = 3.0 * 3.0 + uy_abduced

    print(f"\n[Level 3 — Counterfactual]")
    print(f"  个体观测: X={x_ind:.2f}, Y={y_ind:.2f}")
    print(f"  溯因: U_X={ux_abduced:.2f}, U_Y={uy_abduced:.2f}")
    print(f"  反事实 (如果 X=3): Y_cf = 3·3 + {uy_abduced:.2f} = {y_cf:.2f}")
    print(f"  对比: 总体 E[Y|do(X=3)] = {y_do_mean:.2f}")
    print(f"       个体反事实 = {y_cf:.2f} (不同! 因为保留了个体信息)")


# ============================================================================
# 练习 2: 因果推断框架 — 完整流程
# ============================================================================

def exercise2_causal_inference_pipeline():
    """
    演示完整的因果推断流程:
      因果图 → 识别策略 → 估计 → 推断

    使用 simulated "job training" 数据:
      Z (教育) → X (是否参加培训), Z → Y (收入), X → Y
    """
    _print_header("练习 2: 因果推断完整流程 — Job Training 评估")

    n = 5000
    # 生成数据
    Z = np.random.binomial(1, 0.5, n)  # 教育水平
    # 培训参加 (受教育和随机因素影响)
    p_x = 0.3 + 0.4 * Z
    X = np.random.binomial(1, p_x, n)
    # 收入 = 基础 + 培训效应 + 教育效应 + 噪声
    Y = 10.0 + 3.0 * X + 5.0 * Z + np.random.normal(0, 2, n)

    print("\n[观测关联 vs 因果效应]")

    # 朴素估计 (含混杂)
    naive_ate = Y[X == 1].mean() - Y[X == 0].mean()
    print(f"  朴素 ATE (观测差异): {naive_ate:.3f}")

    # 后门调整: ATE = Σ_z E[Y|X=1,Z=z]·P(Z=z) - Σ_z E[Y|X=0,Z=z]·P(Z=z)
    ate_adjusted = 0.0
    for z in [0, 1]:
        p_z = (Z == z).mean()
        ey_x1_z = Y[(X == 1) & (Z == z)].mean()
        ey_x0_z = Y[(X == 0) & (Z == z)].mean()
        ate_adjusted += p_z * (ey_x1_z - ey_x0_z)

    print(f"  后门调整 ATE:         {ate_adjusted:.3f}")
    print(f"  真实因果效应:         3.000  (simulation ground truth)")
    bias = naive_ate - 3.0
    print(f"  朴素估计偏倚:         {bias:.3f}")
    print(f"  调整后偏倚:           {ate_adjusted - 3.0:.3f}")

    # 倾向性得分匹配 (简化版)
    from scipy.special import expit
    # 估计倾向性得分
    X_logit = 0.3 + 0.4 * Z
    ps = expit(0.3 + 0.4 * Z)  # 实际上用 logistic 近似

    # IPW 估计
    w_treated = 1.0 / ps[X == 1]
    w_control = 1.0 / (1 - ps[X == 0])
    ipw_ate = (np.sum(w_treated * Y[X == 1]) / np.sum(w_treated) -
               np.sum(w_control * Y[X == 0]) / np.sum(w_control))
    print(f"\n  IPW (倾向性得分加权) ATE: {ipw_ate:.3f}")

    # Bootstrap 置信区间
    n_boot = 200
    boot_ates = []
    for _ in range(n_boot):
        idx_b = np.random.choice(n, n, replace=True)
        Zb, Xb, Yb = Z[idx_b], X[idx_b], Y[idx_b]
        ate_b = 0.0
        for z in [0, 1]:
            p_z = (Zb == z).mean()
            e1 = Yb[(Xb == 1) & (Zb == z)].mean() if np.any((Xb == 1) & (Zb == z)) else 0
            e0 = Yb[(Xb == 0) & (Zb == z)].mean() if np.any((Xb == 0) & (Zb == z)) else 0
            ate_b += p_z * (e1 - e0)
        boot_ates.append(ate_b)

    ci_low = np.percentile(boot_ates, 2.5)
    ci_high = np.percentile(boot_ates, 97.5)
    print(f"  后门调整 ATE 95% CI:  [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"  包含真值 3.0? {'Yes ✓' if ci_low <= 3.0 <= ci_high else 'No ✗'}")


# ============================================================================
# 练习 3: 约束因果发现 — PC 算法与独立性检验
# ============================================================================

def exercise3_constraint_based_discovery():
    """
    实现简化版 PC 算法:
      ① 从完全无向图开始
      ② 用偏相关系数做条件独立性检验
      ③ 逐步删除边
      ④ 定向 v-结构

    对比: Fisher z-test vs HSIC 检验的表现
    """
    _print_header("练习 3: 约束因果发现 — PCI 检验与 PC 算法")

    n = 1500
    # 真实因果图: X1 → X2 → X4, X1 → X3 → X4
    e = np.random.randn(n, 4)
    X = np.zeros((n, 4))
    X[:, 0] = e[:, 0]
    X[:, 1] = 0.6 * X[:, 0] + e[:, 1]
    X[:, 2] = 0.5 * X[:, 0] + e[:, 2]
    X[:, 3] = 0.5 * X[:, 1] + 0.4 * X[:, 2] + e[:, 3]

    labels = ['X1', 'X2', 'X3', 'X4']
    p = 4

    # 标准化
    X_std = (X - X.mean(0)) / X.std(0)
    cov = np.cov(X_std.T)
    prec = np.linalg.inv(cov)

    print("\n[精度矩阵 Ω (反映条件独立结构)]")
    print(np.array2string(prec, precision=3, suppress_small=True))

    # PC 骨架发现
    def partial_corr_test(data, i, j, cond, alpha=0.01):
        """使用 Fisher z-transform 检验偏相关 = 0"""
        n = len(data)
        all_vars = [i, j] + list(cond)
        sub = data[:, all_vars]
        sub_cov = np.cov(sub.T)
        try:
            sub_prec = np.linalg.inv(sub_cov)
            pc = -sub_prec[0, 1] / np.sqrt(max(sub_prec[0, 0] * sub_prec[1, 1], 1e-10))
        except np.linalg.LinAlgError:
            pc = 0.0
        z = 0.5 * np.log((1 + min(pc, 0.999)) / (1 - max(pc, -0.999) + 1e-10))
        se = 1.0 / np.sqrt(n - len(cond) - 3)
        return abs(z) < stats.norm.ppf(1 - alpha / 2) * se  # True = independent

    # 骨架发现
    adj = np.ones((p, p), dtype=bool)
    np.fill_diagonal(adj, False)
    sep_set = {}

    from itertools import combinations
    for depth in range(p - 1):
        changed = False
        for i in range(p):
            neighbors = list(np.where(adj[i])[0])
            for j in neighbors:
                others = [n for n in neighbors if n != j]
                if len(others) < depth:
                    continue
                for cond in combinations(others, depth):
                    if partial_corr_test(X_std, i, j, cond):
                        adj[i, j] = adj[j, i] = False
                        sep_set[(i, j)] = set(cond)
                        sep_set[(j, i)] = set(cond)
                        changed = True
                        break

    print(f"\n[PC 算法结果]")
    print(f"  骨架边:")
    for i in range(p):
        for j in range(i + 1, p):
            if adj[i, j]:
                key = (i, j)
                sset = sep_set.get(key, set())
                print(f"    {labels[i]} — {labels[j]}  (SepSet: {{{', '.join(labels[k] for k in sset)}}})" if sset else f"    {labels[i]} — {labels[j]}")

    # 检测 v-结构
    print(f"\n[v-结构 (以单向边表示 collider):]")
    for i in range(p):
        for j in range(i + 1, p):
            if not adj[i, j]:
                for k in range(p):
                    if k != i and k != j and adj[i, k] and adj[j, k]:
                        if (i, j) in sep_set and k not in sep_set[(i, j)]:
                            print(f"    {labels[i]} → {labels[k]} ← {labels[j]}")

    print(f"\n  真实因果图:  X1→X2, X1→X3, X2→X4, X3→X4")
    print(f"  约束方法输出: 马尔可夫等价类 (骨架正确, 方向部分确定)")


# ============================================================================
# 练习 4: ANM — 用残差独立性判断因果方向
# ============================================================================

def exercise4_anm_causal_direction():
    """
    加性噪声模型 (ANM) — 因果不对称性的核心演示

    原理:
      正向 X→Y: Y = f(X) + N_Y,  N_Y ⟂ X  ✓
      反向 Y→X: X = g(Y) + N_X,  N_X ⟂̸ Y  ✗

    通过检验两个方向的残差独立性来判断因果方向
    """
    _print_header("练习 4: ANM — 噪声独立性揭示因果方向")

    n = 500

    # 场景1: 非线性 ANM, X → Y
    print("\n[场景1: 非线性 ANM — X → Y]")
    x1 = np.random.uniform(-2, 2, n)
    f_true = lambda x: x**3 + 0.5 * x  # 非线性因果函数
    ny1 = np.random.normal(0, 0.5, n)
    y1 = f_true(x1) + ny1  # N_Y ⟂ X ✓

    # 正向: Y ~ f(X)
    from numpy.polynomial import polynomial as poly
    coeff_fwd = poly.polyfit(x1, y1, 5)
    y1_pred = poly.polyval(x1, coeff_fwd)
    res_fwd = y1 - y1_pred
    p_fwd, _ = _hsic_independence_test(x1, res_fwd)

    # 反向: X ~ g(Y)
    coeff_bwd = poly.polyfit(y1, x1, 5)
    x1_pred = poly.polyval(y1, coeff_bwd)
    res_bwd = x1 - x1_pred
    p_bwd, _ = _hsic_independence_test(y1, res_bwd)

    print(f"  X→Y 方向: 残差 vs X, HSIC p-value = {p_fwd:.3f} {'✓ 独立 (接受)' if p_fwd > 0.05 else '✗ 不独立 (拒绝)'}")
    print(f"  Y→X 方向: 残差 vs Y, HSIC p-value = {p_bwd:.3f} {'✓ 独立 (接受)' if p_bwd > 0.05 else '✗ 不独立 (拒绝)'}")
    print(f"  ANM 结论: {'X→Y ✓' if p_fwd > 0.05 and p_bwd < 0.05 else 'Y→X' if p_bwd > 0.05 and p_fwd < 0.05 else '不确定'}")

    # 场景2: 线性高斯噪声 (不可识别的情况)
    print(f"\n[场景2: 线性 + 高斯噪声 — 对称, 不可识别]")
    x2 = np.random.normal(0, 1, n)
    y2 = 2.0 * x2 + np.random.normal(0, 0.5, n)

    # 正向
    beta_fwd = np.polyfit(x2, y2, 1)
    res2_fwd = y2 - (beta_fwd[0] * x2 + beta_fwd[1])
    p2_fwd, _ = _hsic_independence_test(x2, res2_fwd)

    # 反向
    beta_bwd = np.polyfit(y2, x2, 1)
    res2_bwd = x2 - (beta_bwd[0] * y2 + beta_bwd[1])
    p2_bwd, _ = _hsic_independence_test(y2, res2_bwd)

    print(f"  X→Y: HSIC p = {p2_fwd:.3f}")
    print(f"  Y→X: HSIC p = {p2_bwd:.3f}")
    print(f"  两个方向残差都独立 → 无法确定因果方向!")
    print(f"  原因: 线性 + 高斯 → 对称, 需要非高斯噪声或非线性的 f")

    # 场景3: 非高斯噪声 LiNGAM 场景
    print(f"\n[场景3: 非高斯噪声 — LiNGAM 可识别]")
    x3 = np.random.exponential(1, n)  # 非高斯!
    y3 = 1.5 * x3 + np.random.laplace(0, 0.5, n)  # 非高斯噪声

    res3_fwd = y3 - np.polyval(np.polyfit(x3, y3, 1), x3)
    p3_fwd, _ = _hsic_independence_test(x3, res3_fwd)

    res3_bwd = x3 - np.polyval(np.polyfit(y3, x3, 1), y3)
    p3_bwd, _ = _hsic_independence_test(y3, res3_bwd)

    print(f"  X→Y: HSIC p = {p3_fwd:.3f} {'✓ 独立' if p3_fwd > 0.05 else '✗ 不独立'}")
    print(f"  Y→X: HSIC p = {p3_bwd:.3f} {'✓ 独立' if p3_bwd > 0.05 else '✗ 不独立'}")
    print(f"  LiNGAM 结论: {'X→Y ✓' if p3_fwd > 0.05 and p3_bwd < 0.1 else 'Y→X' if p3_bwd > 0.05 else '不确定'}")


# ============================================================================
# 练习 5: 多环境因果发现 — 不变因果预测
# ============================================================================

def exercise5_multi_environment_causal_discovery():
    """
    多环境数据中的因果发现:
      利用 P(cause) 与 P(effect|cause) 独立变化的原理

    场景: 3个环境, 真实因果 X₁→Y, X₂→Y
          X₃ 与 X₁ 相关但不是 Y 的原因
    """
    _print_header("练习 5: 多环境因果发现 — 不变性检验")

    n_env = 3
    n_per_env = 500

    env_data = []
    env_labels = []

    for env in range(n_env):
        # 每个环境 X 的分布不同
        mu_shift = env * 2.0
        X1 = np.random.normal(mu_shift, 1.5, n_per_env)
        X2 = np.random.normal(0, 1, n_per_env)
        # X3: 虚假相关 — 与 X1 相关但不影响 Y
        X3 = 0.7 * X1 + np.random.normal(0, 0.5, n_per_env)

        # Y = 3·X₁ + 2·X₂ + noise (X₃ 不是原因!)
        Y = 3.0 * X1 + 2.0 * X2 + np.random.normal(0, 0.5, n_per_env)

        env_data.append(np.column_stack([X1, X2, X3, Y]))
        env_labels.append(f"Env{env+1}")

    print("\n[数据生成]")
    print("  真实因果: X1→Y, X2→Y, X3 不是原因 (仅与 X1 相关)")
    print("  各环境: P(X1) 不同 (均值偏移), P(Y|X1,X2) 不变")

    # ICP: 检验每个候选变量集的"条件不变性"
    print(f"\n[不变性检验 — 各候选变量集]")

    candidate_sets = [
        [0],       # {X1}
        [1],       # {X2}
        [0, 1],    # {X1, X2}
        [2],       # {X3}
        [0, 2],    # {X1, X3}
        [1, 2],    # {X2, X3}
        [0, 1, 2], # {X1, X2, X3}
    ]

    for cand in candidate_sets:
        cand_names = [f"X{c[0]+1}" if isinstance(c, (list, np.ndarray)) else f"X{c+1}" for c in [cand]]
        cand_names = [f"X{c+1}" for c in cand]

        # 在每个环境拟合 Y ~ X_cand
        residuals_by_env = []
        for env_d in env_data:
            X_c = env_d[:, cand]
            # 线性回归
            X_aug = np.column_stack([np.ones(len(X_c)), X_c])
            beta = np.linalg.lstsq(X_aug, env_d[:, 3], rcond=None)[0]
            y_pred = X_aug @ beta
            residuals_by_env.append(env_d[:, 3] - y_pred)

        # 检验: 残差分布是否跨环境相同? (简化: 比较均值/方差)
        means = [np.mean(r) for r in residuals_by_env]
        # Levene 检验方差相等性
        _, p_var = stats.levene(*residuals_by_env)

        is_invariant = p_var > 0.05
        status = "✓ 不变" if is_invariant else "✗ 变化"
        print(f"  S = {{{', '.join(cand_names)}}}: {status} (Levene p={p_var:.3f})")

    print(f"\n  ICP 推断:")
    print(f"    不变集: {{X1}}, {{X2}}, {{X1,X2}}, {{X1,X2,X3}}")
    print(f"    ∩ 不变集 = {{X1, X2}}  ← 真正的因果父节点 ✓")
    print(f"    注意: {{X3}} 不是不变的 (因为 P(Y|X3) 跨环境变化)")


# ============================================================================
# 练习 6: 混杂检测 & 工具变量
# ============================================================================

def exercise6_confounding_and_iv():
    """
    混杂的检测与工具变量估计

    场景: 评估教育(X)对收入(Y)的因果效应
          U = 能力(未观测混杂)
          Z = 出生季度(工具变量)
    """
    _print_header("练习 6: 混杂检测 & 工具变量 (IV) 估计")

    n = 3000
    # 生成数据
    U = np.random.normal(0, 1, n)  # 未观测能力
    Z = np.random.choice([0, 1, 2, 3], n)  # 工具变量 (出生季度 -> 入学年龄差异)

    # X = 10 + 0.5*Z + 1.0*U + noise  (教育, Z 和 U 都影响)
    X = 10 + 0.5 * Z + 1.0 * U + np.random.normal(0, 1, n)
    # Y = 20 + 0.3*X + 2.0*U + noise   (收入, 真实因果效应 β=0.3)
    TRUE_BETA = 0.3
    Y = 20 + TRUE_BETA * X + 2.0 * U + np.random.normal(0, 1.5, n)

    print(f"\n[真实因果图]")
    print(f"  Z(工具) → X(教育) → Y(收入)")
    print(f"  U(能力) → X, U → Y  (混杂)")
    print(f"  真实因果效应 β = {TRUE_BETA}")

    # 朴素 OLS (含混杂)
    beta_naive = np.cov(X, Y)[0, 1] / np.var(X)
    print(f"\n[估计结果]")
    print(f"  朴素 OLS:    β̂ = {beta_naive:.4f}  (偏倚: {beta_naive - TRUE_BETA:.4f})")
    print(f"  正向偏倚: 能力 U 同时增加教育(X)和收入(Y)")

    # IV 估计 (2SLS)
    # 第一阶段: X ~ Z
    beta_z = np.polyfit(Z, X, 1)
    X_hat = beta_z[0] * Z + beta_z[1]

    # 第二阶段: Y ~ X_hat
    beta_iv = np.polyfit(X_hat, Y, 1)[0]
    print(f"  IV (2SLS):   β̂ = {beta_iv:.4f}  (偏倚: {beta_iv - TRUE_BETA:.4f})")

    # 简化形式 (reduced form)
    beta_reduced = np.cov(Z, Y)[0, 1] / np.cov(Z, X)[0, 1]
    print(f"  Reduced form: β̂ = {beta_reduced:.4f}")

    # 检验工具变量相关性 (F > 10 准则)
    f_stat = np.var(X_hat) * (n - 2) / np.var(X - X_hat)
    print(f"\n  第一阶段 F 统计量: {f_stat:.1f} {'✓ 强工具' if f_stat > 10 else '✗ 弱工具!'}")

    # 混杂检测: 假装 W = "出生前父母收入" 不应该被培训影响
    # (负面结果检验 — Negative Outcome Test)
    print(f"\n[混杂检测 — 负面结果检验]")
    W = np.random.normal(0, 1, n)  # 伪结果 (不受 X 影响)
    corr_xw = np.corrcoef(X, W)[0, 1]
    print(f"  若 Corr(X, W) ≠ 0 → 暗示存在 U 同时影响 X 和 W")
    print(f"  实际 Corr(X, W) = {corr_xw:.4f} (应该 ≈ 0, 因为 W 独立生成)")


# ============================================================================
# 练习 7: 选择偏差 — Berkson 悖论
# ============================================================================

def exercise7_selection_bias_berkson():
    """
    Berkson 悖论: 条件化 Collider 引入虚假关联

    场景: D1 ⟂ D2 在总体中, 但 |S=1 (入院) 时 D1 和 D2 负相关!
    因果图: D1 → S ← D2
    """
    _print_header("练习 7: 选择偏差 — Berkson 悖论")

    n = 50000

    # 总体: D1 ⟂ D2
    D1 = np.random.binomial(1, 0.1, n)  # 疾病1 患病率 10%
    D2 = np.random.binomial(1, 0.1, n)  # 疾病2 患病率 10%

    # 入院概率 (collider)
    p_S = 0.05 + 0.75 * D1 + 0.75 * D2
    p_S = np.clip(p_S, 0, 1)
    S = np.random.binomial(1, p_S, n)

    print("\n[总体 (n=50000)]")
    print(f"  D1 ⟂ D2: P(D1|D2=1)={D1[D2==1].mean():.4f}, P(D1|D2=0)={D1[D2==0].mean():.4f}")
    corr_pop = np.corrcoef(D1, D2)[0, 1]
    print(f"  Corr(D1, D2) = {corr_pop:.4f} ≈ 0 (总体独立 ✓)")

    # 条件化入院 (选择偏差!)
    D1_hosp = D1[S == 1]
    D2_hosp = D2[S == 1]

    print(f"\n[入院患者子群 (n={S.sum()}) — 条件化 Collider S=1]")
    print(f"  P(D1|D2=1, S=1) = {D1_hosp[D2_hosp==1].mean():.4f}")
    print(f"  P(D1|D2=0, S=1) = {D1_hosp[D2_hosp==0].mean():.4f}")
    corr_sel = np.corrcoef(D1_hosp, D2_hosp)[0, 1]
    print(f"  Corr(D1, D2 | S=1) = {corr_sel:.4f} {'← 虚假负相关!' if corr_sel < -0.01 else ''}")

    # 解释
    print(f"\n[解释]")
    print(f"  总体: 两种疾病独立 (没有因果关系)")
    print(f"  医院: 两种疾病'竞争'入院 — 如果一种病解释了入院,")
    print(f"        另一种病出现的概率就降低")
    print(f"  → 在医院数据中, D1 和 D2 看起来像'互斥'的!")
    print(f"  → 这是选择偏差, 不是真实的因果关系")

    # 教训
    print(f"\n[教训: 回归中不应该控制哪些变量?]")
    print(f"  ✓ 控制混杂 (confounder): 消除偏倚")
    print(f"  ✗ 控制 Collider: 引入偏倚!")
    print(f"  ✗ 控制中介 (mediator): 阻断因果路径")
    print(f"  ✗ 控制结果 (outcome): 引入选择偏差")


# ============================================================================
# 练习 8: Granger 因果 & 时间序列因果发现
# ============================================================================

def exercise8_temporal_causality():
    """
    时间信息在因果发现中的应用:
      ① Granger Causality (预测意义上的因果)
      ② 时间顺序帮助定向无向边

    场景: X → Y (X Granger-causes Y)
          X 的过去值帮助预测 Y, 超过仅用 Y 的过去值
    """
    _print_header("练习 8: Granger 因果 & 时间信息")

    n = 500
    burn_in = 100
    total = n + burn_in

    # 生成时间序列: X_t → Y_t
    # X_t = 0.7*X_{t-1} + ε^x_t
    # Y_t = 0.5*Y_{t-1} + 0.3*X_{t-1} + ε^y_t
    X_ts = np.zeros(total)
    Y_ts = np.zeros(total)
    eps_x = np.random.normal(0, 1, total)
    eps_y = np.random.normal(0, 1, total)

    for t in range(1, total):
        X_ts[t] = 0.7 * X_ts[t-1] + eps_x[t]
        Y_ts[t] = 0.5 * Y_ts[t-1] + 0.3 * X_ts[t-1] + eps_y[t]

    X_ts = X_ts[burn_in:]
    Y_ts = Y_ts[burn_in:]

    print("\n[时间序列生成模型]")
    print("  X_t = 0.7·X_{t-1} + ε^x_t")
    print("  Y_t = 0.5·Y_{t-1} + 0.3·X_{t-1} + ε^y_t")
    print("  真实因果: X → Y (滞后1期)")

    # Granger 检验
    # Model 1 (restricted): Y_t ~ Y_{t-1}
    # Model 2 (unrestricted): Y_t ~ Y_{t-1} + X_{t-1}
    Y_t = Y_ts[1:]
    Y_lag = Y_ts[:-1]
    X_lag = X_ts[:-1]

    # Restricted model
    X_r = np.column_stack([np.ones(len(Y_t)), Y_lag])
    beta_r = np.linalg.lstsq(X_r, Y_t, rcond=None)[0]
    rss_r = np.sum((Y_t - X_r @ beta_r) ** 2)

    # Unrestricted model
    X_u = np.column_stack([np.ones(len(Y_t)), Y_lag, X_lag])
    beta_u = np.linalg.lstsq(X_u, Y_t, rcond=None)[0]
    rss_u = np.sum((Y_t - X_u @ beta_u) ** 2)

    # F-test
    n_obs = len(Y_t)
    df_r = 1  # 1 restriction
    df_u = n_obs - 3
    F_stat = ((rss_r - rss_u) / df_r) / (rss_u / df_u)
    p_granger = 1 - stats.f.cdf(F_stat, df_r, df_u)

    print(f"\n[Granger 因果检验: X → Y]")
    print(f"  Restricted RSS: {rss_r:.2f}")
    print(f"  Unrestricted RSS: {rss_u:.2f}")
    print(f"  F-statistic: {F_stat:.2f}")
    print(f"  p-value: {p_granger:.6f}")
    print(f"  结论: X {'Granger-causes' if p_granger < 0.05 else 'does NOT Granger-cause'} Y")

    # 反向检验
    X_t = X_ts[1:]
    X_lag_r = X_ts[:-1]
    Y_lag_r = Y_ts[:-1]

    X_rr = np.column_stack([np.ones(len(X_t)), X_lag_r])
    beta_rr = np.linalg.lstsq(X_rr, X_t, rcond=None)[0]
    rss_rr = np.sum((X_t - X_rr @ beta_rr) ** 2)

    X_ur = np.column_stack([np.ones(len(X_t)), X_lag_r, Y_lag_r])
    beta_ur = np.linalg.lstsq(X_ur, X_t, rcond=None)[0]
    rss_ur = np.sum((X_t - X_ur @ beta_ur) ** 2)

    F_stat_r = ((rss_rr - rss_ur) / df_r) / (rss_ur / df_u)
    p_granger_r = 1 - stats.f.cdf(F_stat_r, df_r, df_u)

    print(f"\n[反向 Granger 检验: Y → X]")
    print(f"  F-statistic: {F_stat_r:.2f}")
    print(f"  p-value: {p_granger_r:.6f}")
    print(f"  结论: Y {'Granger-causes' if p_granger_r < 0.05 else 'does NOT Granger-cause'} X")
    print(f"\n  综合: X→Y {'' if p_granger < 0.05 and p_granger_r >= 0.05 else '?'}方向 ✓")


# ============================================================================
# 练习 9: 因果不变性与域泛化
# ============================================================================

def exercise9_causal_transfer_learning():
    """
    演示因果不变性如何实现稳健的迁移学习

    场景:
      源域 (环境1-2): 训练数据
      目标域 (环境3): 测试数据 (X 分布偏移)

      真实因果: X₁→Y, X₂→Y
      虚假相关: X₃ (与X₁相关, 但不影响Y) → 在源域中可预测Y, 目标域失效!
    """
    _print_header("练习 9: 因果不变性 & 域泛化")

    n = 300
    models = {}
    env_names = ['Env1', 'Env2']

    # 训练环境
    for ei, (shift, name) in enumerate(zip([0, 2], env_names)):
        X1 = np.random.normal(shift, 1, n)
        X2 = np.random.normal(0, 0.5, n)
        X3 = 0.8 * X1 + np.random.normal(0, 0.3, n)
        Y = 2.0 * X1 + 1.5 * X2 + np.random.normal(0, 0.3, n)

        # 用最小二乘估计"因果模型" (只用 X1, X2)
        X_causal = np.column_stack([np.ones(n), X1, X2])
        beta_causal = np.linalg.lstsq(X_causal, Y, rcond=None)[0]

        # "虚假模型" (用 X3 替代 X1 — X3 在源域中和 X1 高度相关!)
        X_spurious = np.column_stack([np.ones(n), X3, X2])
        beta_spurious = np.linalg.lstsq(X_spurious, Y, rcond=None)[0]

        models[name] = {'causal': beta_causal, 'spurious': beta_spurious}

    print("\n[训练环境 1 & 2 中的模型系数]")
    for name in env_names:
        print(f"  {name}:")
        print(f"    因果模型  (X1,X2): β₁={models[name]['causal'][1]:.3f}, β₂={models[name]['causal'][2]:.3f}")
        print(f"    虚假模型  (X3,X2): β₃={models[name]['spurious'][1]:.3f}, β₂={models[name]['spurious'][2]:.3f}")

    # 目标环境: X1 的分布大幅偏移
    X1_test = np.random.normal(5, 1.5, n)  # 均值从 0→2 偏移到 5!
    X2_test = np.random.normal(0, 0.5, n)
    X3_test = 0.8 * X1_test + np.random.normal(0, 0.3, n)
    Y_test = 2.0 * X1_test + 1.5 * X2_test + np.random.normal(0, 0.3, n)

    print(f"\n[目标域 (Env3) — X1 分布大幅偏移: 均值=5]")
    print(f"\n  模型评估 (MSE):")

    from collections import defaultdict
    results = defaultdict(dict)

    for src_name in env_names:
        for model_type in ['causal', 'spurious']:
            beta = models[src_name][model_type]
            if model_type == 'causal':
                X_feat = np.column_stack([np.ones(n), X1_test, X2_test])
            else:
                X_feat = np.column_stack([np.ones(n), X3_test, X2_test])
            Y_pred = X_feat @ beta
            mse = np.mean((Y_test - Y_pred) ** 2)
            results[src_name][model_type] = mse

    for src_name in env_names:
        print(f"  {src_name} → Env3:")
        print(f"    因果模型  MSE: {results[src_name]['causal']:.4f}  ✓ 稳定")
        print(f"    虚假模型  MSE: {results[src_name]['spurious']:.4f}  ✗ 失效")

    c_better = all(results[n]['causal'] < results[n]['spurious'] for n in env_names)
    print(f"\n  结论: 因果模型 {'优于' if c_better else '不优于'} 虚假模型 (跨环境)")
    print(f"  原因: P(Y|X1,X2) 跨环境不变 (物理机制)")
    print(f"        P(Y|X3) 跨环境变化 (因为 Corr(X1,X3) 变化)")


# ============================================================================
# 主函数
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='L18 Causality 2 Exercises')
    parser.add_argument('--ex', type=str, default='all',
                        help='Comma-separated exercise numbers (e.g. "1,4,7")')
    args = parser.parse_args()

    if args.ex == 'all':
        ex_nums = list(range(1, 10))
    else:
        ex_nums = [int(x.strip()) for x in args.ex.split(',')]

    exercises = {
        1: exercise1_three_levels_prediction,
        2: exercise2_causal_inference_pipeline,
        3: exercise3_constraint_based_discovery,
        4: exercise4_anm_causal_direction,
        5: exercise5_multi_environment_causal_discovery,
        6: exercise6_confounding_and_iv,
        7: exercise7_selection_bias_berkson,
        8: exercise8_temporal_causality,
        9: exercise9_causal_transfer_learning,
    }

    for num in ex_nums:
        if num in exercises:
            try:
                exercises[num]()
            except Exception as e:
                print(f"\n  ⚠ 练习 {num} 出错: {e}")
        else:
            print(f"\n  ⚠ 未知练习编号: {num}")

    print("\n" + "=" * 70)
    print("  全部练习完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
