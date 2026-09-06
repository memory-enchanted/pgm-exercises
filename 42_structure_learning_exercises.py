"""
==========================================================================================
  CMU 10-708 L16 代码练习: 结构学习 — GGM, Graphical Lasso, Neighbor Selection
==========================================================================================

L16 五大主题 -> 对应练习:
  ① 高斯图模型 (GGM)        -> 练习 1: 精度矩阵 vs 图结构 — Σ vs Ω 对比
  ② Graphical Lasso          -> 练习 2: 坐标下降从零实现 — 正则化路径
  ③ Neighbor Selection       -> 练习 3: 逐节点 Lasso — AND/OR 规则
  ④ 偏相关系数 & 边检验       -> 练习 4: 偏相关计算与显著性测试
  ⑤ 时变 Graphical Lasso     -> 练习 5: 滑动窗口 + 时间平滑结构学习
  ⑥ PGM 视角                 -> 练习 6: 约束 vs 分数 vs 正则化 三种范式

特别说明:
  - 贝叶斯网络导入需使用: from pgmpy.models import DiscreteBayesianNetwork
    (pgmpy 0.1.x 后 BayesianNetwork 改名为 DiscreteBayesianNetwork)
  - Windows GBK 终端下 emoji 打印: sys.stdout.reconfigure(encoding='utf-8')

使用方法:
  python 42_structure_learning_exercises.py           # 运行全部
  python 42_structure_learning_exercises.py --ex 1    # 只运行练习1

依赖: numpy
==========================================================================================
"""

import numpy as np
import sys

# 修复 Windows GBK 终端下 emoji 打印问题
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

np.random.seed(42)


# ============================================================================
# 练习 1: GGM — 精度矩阵 vs 图结构
# ============================================================================

def exercise1_ggm_precision_graph():
    """
    构建一个已知图结构的高斯图模型, 验证精度矩阵 Ω 和协方差 Σ 的关系:
    - Ω_ij = 0 ⇔ X_i ⟂ X_j | rest
    - 展示 Σ（全关联） vs Ω（净关联）

    图结构: 1—2—3—4  (链式, 4个节点)
    即: 边 (1,2), (2,3), (3,4)
    """
    print("=" * 70)
    print("练习 1: GGM — 精度矩阵与图结构的对应关系")
    print("=" * 70)

    p = 4  # 节点数

    # 从图结构构造精度矩阵 Ω (稀疏!)
    # 链式图: 对角=1, 相邻边=-0.5
    Omega_true = np.array([
        [1.0, -0.5,  0.0,  0.0],
        [-0.5, 1.0, -0.5,  0.0],
        [0.0, -0.5,  1.0, -0.5],
        [0.0,  0.0, -0.5,  1.0],
    ])

    # 验证 Ω 是正定的 (所有特征值 > 0)
    eigvals = np.linalg.eigvalsh(Omega_true)
    print(f"\n  Ω (精度矩阵) 特征值: {[f'{v:.3f}' for v in eigvals]}")
    print(f"  正定: {np.all(eigvals > 0)}")

    # 协方差 Σ = Ω^{-1}
    Sigma_true = np.linalg.inv(Omega_true)

    print(f"\n  -- 精度矩阵 Ω (稀疏, 编码条件独立) --")
    print(f"  {'':>6s}", end="")
    for j in range(p):
        print(f"  X{j+1:>6d}", end="")
    print()
    for i in range(p):
        print(f"  X{i+1:>4d}", end="")
        for j in range(p):
            mark = "*" if i != j and abs(Omega_true[i, j]) > 1e-6 else " "
            print(f" {Omega_true[i,j]:>6.3f}", end="")
        print()
    print(f"  * = 非零非对角元 → 有边!")

    print(f"\n  -- 协方差矩阵 Σ (稠密, 包含间接关联) --")
    for i in range(p):
        print(f"  X{i+1}:", end="")
        for j in range(p):
            print(f" {Sigma_true[i,j]:>7.4f}", end="")
        print()

    # 关键对比: 间接效应
    print(f"\n  -- 间接效应分析 --")
    print(f"  图结构: 1—2—3—4  (链式)")
    print(f"  Ω_14 = {Omega_true[0,3]:.3f}  → X₁ ⟂ X₄ | X₂,X₃  (给定中间变量, 首尾独立!)")
    print(f"  Σ_14 = {Sigma_true[0,3]:.4f} → 但 X₁ 和 X₄ 协方差不等于 0!")
    print(f"  原因: 关联通过 X₁→X₂→X₃→X₄ 间接传递")
    print(f"  Σ 捕获了总效应 (直接+间接), Ω 捕获了净效应 (仅直接)")

    # ==== 模拟数据验证条件独立性 ====
    n = 5000
    L = np.linalg.cholesky(Sigma_true)  # Cholesky: Σ = L·L^T
    X = (L @ np.random.randn(p, n)).T    # X ~ N(0, Σ)

    # 经验协方差和精度矩阵
    S_emp = np.cov(X, rowvar=False)
    Omega_emp = np.linalg.inv(S_emp)

    print(f"\n  -- 从样本估计 (n={n}) --")
    print(f"  Ω̂_14 = {Omega_emp[0,3]:.5f}  (理论值 = 0)")
    print(f"  Σ̂_14 = {S_emp[0,3]:.5f}   (理论值 = {Sigma_true[0,3]:.4f})")

    # 偏相关系数验证
    def partial_corr(Omega, i, j):
        return -Omega[i, j] / np.sqrt(Omega[i, i] * Omega[j, j])

    print(f"\n  -- 偏相关系数 (Partial Correlation) --")
    for i in range(p):
        for j in range(i + 1, p):
            rho = partial_corr(Omega_true, i, j)
            edge = "有边" if abs(rho) > 1e-6 else "无边"
            print(f"    ρ(X{i+1}, X{j+1} | rest) = {rho:+.3f} → {edge}")

    print(f"\n  洞察:")
    print(f"    GGM 的核心: 图结构 = Ω 的支撑集")
    print(f"    Ω_ij = 0 ⇔ ρ_{{ij|rest}} = 0 ⇔ X_i ⟂ X_j | rest")
    print(f"    结构学习 = 从数据中发现哪些 Ω_ij = 0!")


# ============================================================================
# 练习 2: Graphical Lasso — 坐标下降从零实现
# ============================================================================

def exercise2_graphical_lasso():
    """
    从零实现 Graphical Lasso 的坐标下降算法 (Friedman et al., 2008),
    并在模拟数据上展示正则化路径。

    算法核心: 分块坐标下降, 每次更新 Ω 的一行/列,
            内层是一个 Lasso 问题。

    数据结构: p=20, n=100, 真实图 = 链式 + 随机边
    """
    print("=" * 70)
    print("练习 2: Graphical Lasso — 坐标下降从零实现")
    print("=" * 70)

    p = 20
    n = 100

    # ==== 生成真实稀疏精度矩阵 ====
    # 链式结构 (相邻连接) + 少量随机边
    Omega_star = np.eye(p) * 1.5
    for i in range(p - 1):
        Omega_star[i, i + 1] = -0.4
        Omega_star[i + 1, i] = -0.4
    # 添加 3 条随机边
    for _ in range(3):
        i, j = np.random.choice(p, 2, replace=False)
        if abs(i - j) > 1:  # 避免和已有的链边重复
            Omega_star[i, j] = -0.3
            Omega_star[j, i] = -0.3

    # 确保正定: 对角线主导
    for i in range(p):
        row_sum = np.sum(np.abs(Omega_star[i])) - abs(Omega_star[i, i])
        if Omega_star[i, i] <= row_sum:
            Omega_star[i, i] = row_sum + 0.5

    Sigma_star = np.linalg.inv(Omega_star)

    # 生成样本
    L = np.linalg.cholesky(Sigma_star)
    X = (L @ np.random.randn(p, n)).T
    S = np.cov(X, rowvar=False)  # 经验协方差

    n_edges_true = np.sum(np.abs(Omega_star) > 1e-6) - p  # 非对角非零元
    n_edges_true = n_edges_true // 2
    print(f"\n  真实图: p={p}, edges={n_edges_true}")
    print(f"  样本: n={n}")

    # ==== Graphical Lasso 实现 ====
    def soft_threshold(x, lam):
        """软阈值算子"""
        return np.sign(x) * np.maximum(np.abs(x) - lam, 0)

    def graphical_lasso(S, lam, max_iter=200, tol=1e-4):
        """
        坐标下降 Graphical Lasso (Friedman et al., 2008)

        参数:
          S: 经验协方差矩阵 (p×p)
          lam: ℓ1 正则化参数
        返回:
          Omega: 估计的精度矩阵
          W: 估计的协方差矩阵
        """
        p = S.shape[0]
        W = S + lam * np.eye(p)  # 初始: 加对角线确保正定
        Omega = np.linalg.inv(W)

        for it in range(max_iter):
            max_change = 0
            for j in range(p):
                # 将 W 分块: 把第 j 行/列移到最后
                idx = list(range(p))
                idx.remove(j)
                idx.append(j)  # [0,1,...,j-1, j+1,...,p-1, j]

                # 重新排列
                W_perm = W[np.ix_(idx, idx)]
                S_perm = S[np.ix_(idx, idx)]

                # 块: W = [[W_11, w_12], [w_21, w_22]]
                W_11 = W_perm[:p-1, :p-1]
                s_12 = S_perm[:p-1, p-1]

                # 解 Lasso: min_β ½β^T W_11 β - s_12^T β + λ||β||₁
                # 坐标下降解这个 Lasso
                beta = np.zeros(p - 1)
                if p > 2:
                    # 用坐标下降解内层 Lasso
                    for inner_it in range(50):
                        beta_old = beta.copy()
                        for k in range(p - 1):
                            # 部分残差
                            residual = s_12[k] - W_11[k] @ beta + W_11[k, k] * beta[k]
                            beta[k] = soft_threshold(residual, lam) / W_11[k, k]
                        if np.max(np.abs(beta - beta_old)) < 1e-5:
                            break
                else:
                    beta[0] = soft_threshold(s_12[0], lam) / W_11[0, 0]

                # 更新 w_12 = W_11 @ beta
                w_12 = W_11 @ beta

                # 更新 W
                old_w = W[:, j].copy()
                W[np.ix_(idx[:p-1], [j])] = w_12.reshape(-1, 1)
                W[j, idx[:p-1]] = w_12

                max_change = max(max_change, np.max(np.abs(old_w[idx[:p-1]] - w_12)))

            if max_change < tol:
                break

        # 从 W 恢复 Ω
        Omega = np.linalg.inv(W)
        return Omega, W

    # ==== 测试不同的 λ ====
    lam_values = [0.01, 0.05, 0.1, 0.2, 0.5]
    print(f"\n  -- Graphical Lasso 正则化路径 --")
    print(f"  {'λ':>8s}  {'边数':>6s}  {'log-lik':>10s}  {'TPR':>8s}  {'FPR':>8s}")

    for lam in lam_values:
        Omega_hat, W_hat = graphical_lasso(S, lam)

        # 统计边数
        adj_hat = np.abs(Omega_hat) > 1e-4
        np.fill_diagonal(adj_hat, False)
        n_edges_hat = adj_hat.sum() // 2

        # 对数似然
        log_lik = np.linalg.slogdet(Omega_hat)[1] - np.trace(S @ Omega_hat)

        # 比较: TPR = 正确找到的边 / 真实边数
        adj_true = np.abs(Omega_star) > 1e-6
        np.fill_diagonal(adj_true, False)
        tp = np.sum(adj_hat & adj_true) // 2
        fp = np.sum(adj_hat & ~adj_true) // 2
        fn = np.sum(~adj_hat & adj_true) // 2
        tn = np.sum(~adj_hat & ~adj_true) // 2

        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)

        print(f"  {lam:8.3f}  {n_edges_hat:>6d}  {log_lik:>10.2f}  {tpr:>8.3f}  {fpr:>8.3f}")

    # ==== 对比: 无正则化 (直接 S⁻¹) ====
    print(f"\n  -- 无正则化的效果 --")
    try:
        Omega_mle = np.linalg.inv(S)
        adj_mle = np.abs(Omega_mle) > 1e-6
        np.fill_diagonal(adj_mle, False)
        n_edges_mle = adj_mle.sum() // 2
        tp_mle = np.sum(adj_mle & adj_true) // 2
        fp_mle = np.sum(adj_mle & ~adj_true) // 2
        fn_mle = np.sum(~adj_mle & adj_true) // 2
        tpr_mle = tp_mle / max(tp_mle + fn_mle, 1)
        fpr_mle = fp_mle / max(fp_mle + ~adj_mle.sum() // 2, 1)
        print(f"  MLE (S⁻¹): 边数={n_edges_mle}, TPR={tpr_mle:.3f}")
        print(f"  注意: 没有 ℓ1 惩罚 → 几乎完全图 (p(p-1)/2 = {p*(p-1)//2})")
        print(f"        edges = {n_edges_mle}/{p*(p-1)//2} — 稠密!")
    except np.linalg.LinAlgError:
        print(f"  S 奇异! (p={p} ≈ n={n}) → 无法直接求逆")
        print(f"  这正是 Graphical Lasso 存在的原因!")

    print(f"\n  洞察:")
    print(f"    Graphical Lasso = ℓ1 正则化的精度矩阵估计")
    print(f"    λ 小 → 图稠密 (保留许多边)")
    print(f"    λ 大 → 图稀疏 (只保留最强的边)")
    print(f"    坐标下降 = 逐行/列解 Lasso → 保证收敛到全局最优!")


# ============================================================================
# 练习 3: Neighbor Selection — 逐节点 Lasso
# ============================================================================

def exercise3_neighbor_selection():
    """
    实现 Meinshausen-Bühlmann Neighbor Selection:
    对每个节点 j, 用 Lasso 回归 X_j ~ X_{-j}
    非零系数 → 该变量是 j 的邻居

    与 Graphical Lasso 对比: 计算效率 & 边检测准确性
    """
    print("=" * 70)
    print("练习 3: Neighbor Selection — 逐节点 Lasso 回归")
    print("=" * 70)

    p = 15
    n = 100

    # 生成真实图: 5 条孤立边 + 孤立节点 (无间接路径 → Lasso 完美区分)
    # 边: (1,2), (3,4), (5,6), (7,8), (9,10), 节点 11-15 孤立
    Omega_star = np.eye(p)
    edges_true_list = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    for i, j in edges_true_list:
        Omega_star[i, j] = -0.6
        Omega_star[j, i] = -0.6

    # 确保正定 (对角线主导)
    for i in range(p):
        row_sum = np.sum(np.abs(Omega_star[i])) - abs(Omega_star[i, i])
        Omega_star[i, i] = row_sum + 0.5

    Sigma_star = np.linalg.inv(Omega_star)
    L = np.linalg.cholesky(Sigma_star)
    X = (L @ np.random.randn(p, n)).T

    # 真实邻居
    adj_true = np.abs(Omega_star) > 1e-6
    np.fill_diagonal(adj_true, False)

    # ==== Neighbor Selection ====
    def soft_threshold(x, lam):
        return np.sign(x) * np.maximum(np.abs(x) - lam, 0)

    def lasso_single(X, y, lam, max_iter=500):
        """单变量 Lasso (坐标下降), λ 在目标函数尺度: (1/2n)||y-Xβ||² + λ||β||₁"""
        n_samples, n_features = X.shape
        beta = np.zeros(n_features)
        X_norms = np.sum(X**2, axis=0)

        for it in range(max_iter):
            beta_old = beta.copy()
            for k in range(n_features):
                if X_norms[k] < 1e-12:
                    continue
                residual = y - X @ beta + X[:, k] * beta[k]
                rho = X[:, k] @ residual
                # soft(rho, n*lam) / X_norms — 正确缩放
                beta[k] = soft_threshold(rho, n_samples * lam) / X_norms[k]

            if np.max(np.abs(beta - beta_old)) < 1e-5:
                break
        return beta

    # 标准化
    X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

    lam_ns = 0.15
    neighbors_or = set()
    neighbors_and = set()
    all_betas = []

    print(f"\n  -- 逐节点 Lasso 回归 (λ={lam_ns}) --")
    for j in range(p):
        # 构建 X_{-j}
        X_others = np.delete(X_std, j, axis=1)
        y_j = X_std[:, j]

        beta_j = lasso_single(X_others, y_j, lam_ns)
        all_betas.append(beta_j)

        # 非零系数 → 邻居
        selected = np.where(np.abs(beta_j) > 1e-5)[0]
        # 映射回原索引
        selected_orig = [k if k < j else k + 1 for k in selected]

        print(f"    Node {j+1:>2d}: neighbors = {[s+1 for s in selected_orig]}")

        for k in selected_orig:
            neighbors_or.add((min(j, k), max(j, k)))

    # AND rule: 双向确认
    for (i, j) in neighbors_or:
        # 检查 j 是否也在 i 的邻居中
        beta_i = all_betas[i]
        j_in_i = j if j < i else j - 1
        if j_in_i < len(beta_i) and abs(beta_i[j_in_i]) > 1e-5:
            neighbors_and.add((i, j))

    # 统计
    true_edges = set()
    for i in range(p):
        for j in range(i + 1, p):
            if adj_true[i, j]:
                true_edges.add((i, j))

    tp_or = len(neighbors_or & true_edges)
    fp_or = len(neighbors_or - true_edges)
    fn_or = len(true_edges - neighbors_or)

    tp_and = len(neighbors_and & true_edges)
    fp_and = len(neighbors_and - true_edges)
    fn_and = len(true_edges - neighbors_and)

    print(f"\n  -- AND vs OR 规则对比 --")
    print(f"  {'':<15s} {'OR rule':>12s} {'AND rule':>12s}")
    print(f"  {'-'*15} {'-'*12} {'-'*12}")
    print(f"  {'发现边':<15s} {len(neighbors_or):>12d} {len(neighbors_and):>12d}")
    print(f"  {'TP (正确)':<15s} {tp_or:>12d} {tp_and:>12d}")
    print(f"  {'FP (假阳)':<15s} {fp_or:>12d} {fp_and:>12d}")
    print(f"  {'FN (假阴)':<15s} {fn_or:>12d} {fn_and:>12d}")
    print(f"  {'Precision':<15s} {tp_or/max(tp_or+fp_or,1):>12.3f} {tp_and/max(tp_and+fp_and,1):>12.3f}")
    print(f"  {'Recall':<15s} {tp_or/max(tp_or+fn_or,1):>12.3f} {tp_and/max(tp_and+fn_and,1):>12.3f}")

    print(f"\n  -- 计算效率对比 --")
    print(f"  Neighbor Selection: p={p} 个独立 Lasso → 可完全并行!")
    print(f"  Graphical Lasso:    需要 O(p³) 的正定约束操作")
    print(f"  当 p >> n 时, NS 可以处理 p≈10000+, GL 限于 p≈1000")

    # 不同的 λ 的影响
    print(f"\n  -- λ 对边数的影响 (OR rule) --")
    for lam_test in [0.05, 0.1, 0.15, 0.25, 0.4]:
        edges_lam = 0
        for j in range(p):
            X_others = np.delete(X_std, j, axis=1)
            y_j = X_std[:, j]
            beta_j = lasso_single(X_others, y_j, lam_test)
            edges_lam += np.sum(np.abs(beta_j) > 1e-5)
        edges_lam //= 2
        print(f"    λ={lam_test:.2f}: ~{edges_lam} edges (true={len(true_edges)})")

    print(f"\n  洞察:")
    print(f"    Neighbor Selection = 把 1 个 p×p 问题拆成 p 个 1×(p-1) 问题")
    print(f"    AND rule: 更保守 (两边都确认才有边) → FP 少但可能丢边")
    print(f"    OR rule:  更灵敏 (单边确认即可) → Recall 高但可能有假阳")
    print(f"    实践中常用 AND rule + 交叉验证选 λ")


# ============================================================================
# 练习 4: 偏相关系数与边检验
# ============================================================================

def exercise4_partial_correlation():
    """
    计算偏相关系数并进行显著性检验。

    偏相关 ρ_{ij|rest} = -Ω_ij / √(Ω_ii · Ω_jj)
    在原假设 H0: ρ = 0 下, t = ρ·√(n-p) / √(1-ρ²) ~ t_{n-p}
    """
    print("=" * 70)
    print("练习 4: 偏相关系数计算与边显著性检验")
    print("=" * 70)

    p = 8
    n = 200

    # 生成真实图: 环 1-2-3-4-5-6-7-8-1
    Omega_true = np.eye(p)
    for i in range(p):
        j = (i + 1) % p
        Omega_true[i, j] = -0.45
        Omega_true[j, i] = -0.45
    # 添加一条"弱边" (1, 4): 弱偏相关
    Omega_true[0, 3] = -0.15
    Omega_true[3, 0] = -0.15
    # 对角线
    for i in range(p):
        Omega_true[i, i] = np.sum(np.abs(Omega_true[i])) - abs(Omega_true[i, i]) + 0.5

    Sigma_true = np.linalg.inv(Omega_true)
    L = np.linalg.cholesky(Sigma_true)
    X = (L @ np.random.randn(p, n)).T

    # 经验精度矩阵
    S = np.cov(X, rowvar=False)
    Omega_hat = np.linalg.inv(S)

    # ==== 偏相关系数 ====
    print(f"\n  -- 偏相关系数矩阵 (ρ_{ij|rest}) --")
    P_corr = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            if i != j:
                P_corr[i, j] = -Omega_hat[i, j] / np.sqrt(Omega_hat[i, i] * Omega_hat[j, j])

    print(f"  {'':>6s}", end="")
    for j in range(p):
        print(f"  X{j+1:>6d}", end="")
    print()
    for i in range(p):
        print(f"  X{i+1:>4d}", end="")
        for j in range(p):
            if i == j:
                print(f"  {'—':>6s}", end="")
            else:
                rho = P_corr[i, j]
                mark = "*" if abs(rho) > 0.1 else " "
                print(f" {rho:>6.3f}{mark}", end="")
        print()
    print(f"  * = |ρ| > 0.1 (可能的边)")

    # ==== 显著性检验 ====
    print(f"\n  -- 边显著性检验 (t-test, α=0.01) --")
    alpha = 0.01
    df = n - p
    from scipy import stats as sp_stats

    t_critical = sp_stats.t.ppf(1 - alpha / 2, df)

    edges_found = []
    for i in range(p):
        for j in range(i + 1, p):
            rho = P_corr[i, j]
            t_stat = rho * np.sqrt(df) / np.sqrt(1 - rho**2 + 1e-12)
            p_val = 2 * (1 - sp_stats.t.cdf(abs(t_stat), df))
            sig = "***" if p_val < alpha else ""
            if sig:
                edges_found.append((i + 1, j + 1, rho, t_stat, p_val))

    edges_found.sort(key=lambda x: abs(x[2]), reverse=True)
    for i, j, rho, t_stat, p_val in edges_found:
        is_true = abs(Omega_true[i-1, j-1]) > 1e-6
        status = "✓ 真实边" if is_true else "✗ 假阳性"
        print(f"  ({i},{j}): ρ={rho:+.4f}  t={t_stat:+.3f}  p={p_val:.4f}  {status}")

    # ==== 边的强弱分析 ====
    print(f"\n  -- 真实边 vs 检测到的边 --")
    print(f"  {'边':>8s}  {'真实 Ω_ij':>12s}  {'估计 ρ':>12s}  {'检测到':>8s}")
    for i in range(p):
        for j in range(i + 1, p):
            if abs(Omega_true[i, j]) > 1e-6 or abs(P_corr[i, j]) > 0.15:
                true_val = Omega_true[i, j]
                rho = P_corr[i, j]
                detected = "✓" if abs(rho) > t_critical / np.sqrt(df + t_critical**2) else "✗"
                print(f"  ({i+1},{j+1})  {true_val:>+12.4f}  {rho:>+12.4f}  {detected:>8s}")

    print(f"\n  洞察:")
    print(f"    偏相关 ρ = -Ω_ij/√(Ω_ii·Ω_jj) → 标准化后的'净关联'")
    print(f"    t-test 给出统计显著性 — 但需校正多重比较 (Bonferroni / FDR)")
    print(f"    弱边 (|Ω| 小) 需要更大的 n 才能检测到")
    print(f"    边检测本质上是: 对每对 (i,j) 做条件独立性检验")


# ============================================================================
# 练习 5: 时变 Graphical Lasso
# ============================================================================

def exercise5_time_varying_gl():
    """
    模拟时变图结构, 展示滑动窗口 + Graphical Lasso 估计时变网络。

    场景: T=10 个时间点, 图结构在第 5 个时间点发生"断点"变化
      阶段 1 (t=1-5): 链式结构  1-2-3-4-5
      阶段 2 (t=6-10): 模块化结构 {1,2}-{3,4}-5
    """
    print("=" * 70)
    print("练习 5: 时变 Graphical Lasso — 滑动窗口 + 结构断点检测")
    print("=" * 70)

    p = 6
    n_per_time = 80
    T = 10  # 时间点

    # ==== 生成时变数据 ====
    print(f"\n  -- 时变结构设定 (T={T} 个时间点) --")
    print(f"  阶段 1 (t=1-5): 链式  1—2—3—4—5—6")
    print(f"  阶段 2 (t=6-10): 两个团 {1,2,3}—{4,5,6} (断点!)")

    # 阶段 1 精度矩阵 (链式)
    Omega_phase1 = np.eye(p)
    for i in range(p - 1):
        Omega_phase1[i, i + 1] = -0.4
        Omega_phase1[i + 1, i] = -0.4
    for i in range(p):
        Omega_phase1[i, i] = np.sum(np.abs(Omega_phase1[i])) - abs(Omega_phase1[i, i]) + 0.5

    # 阶段 2 精度矩阵 (模块化)
    Omega_phase2 = np.eye(p)
    # 团 1: {1,2,3} 全连接
    for i in range(3):
        for j in range(i + 1, 3):
            Omega_phase2[i, j] = -0.4
            Omega_phase2[j, i] = -0.4
    # 团 2: {4,5,6} 全连接
    for i in range(3, 6):
        for j in range(i + 1, 6):
            Omega_phase2[i, j] = -0.4
            Omega_phase2[j, i] = -0.4
    # 团间连接: 2—5
    Omega_phase2[1, 4] = -0.25
    Omega_phase2[4, 1] = -0.25
    for i in range(p):
        Omega_phase2[i, i] = np.sum(np.abs(Omega_phase2[i])) - abs(Omega_phase2[i, i]) + 0.5

    # 验证正定
    assert np.all(np.linalg.eigvalsh(Omega_phase1) > 0)
    assert np.all(np.linalg.eigvalsh(Omega_phase2) > 0)

    Sigma1 = np.linalg.inv(Omega_phase1)
    Sigma2 = np.linalg.inv(Omega_phase2)

    # 生成每个时间点的数据
    L1 = np.linalg.cholesky(Sigma1)
    L2 = np.linalg.cholesky(Sigma2)

    data_times = []
    for t in range(T):
        if t < 5:
            X_t = (L1 @ np.random.randn(p, n_per_time)).T
        else:
            X_t = (L2 @ np.random.randn(p, n_per_time)).T
        data_times.append(X_t)

    # ==== 滑动窗口 Graphical Lasso ====
    window = 4  # 窗口半宽: 共 2*window+1 个时间点 (当足够时)
    lam = 0.12

    def graphical_lasso_simple(S, lam, max_iter=100, tol=1e-4):
        """简化版 Graphical Lasso (与练习2相同)"""
        p_s = S.shape[0]
        W = S + lam * np.eye(p_s)

        for it in range(max_iter):
            max_change = 0
            for j in range(p_s):
                idx = list(range(p_s))
                idx.remove(j)
                idx.append(j)

                W_perm = W[np.ix_(idx, idx)]
                S_perm = S[np.ix_(idx, idx)]

                W_11 = W_perm[:p_s-1, :p_s-1]
                s_12 = S_perm[:p_s-1, p_s-1]

                beta = np.zeros(p_s - 1)
                if p_s > 2:
                    for inner_it in range(30):
                        beta_old = beta.copy()
                        for k in range(p_s - 1):
                            residual = s_12[k] - W_11[k] @ beta + W_11[k, k] * beta[k]
                            beta[k] = np.sign(residual) * max(abs(residual) - lam, 0) / W_11[k, k]
                        if np.max(np.abs(beta - beta_old)) < 1e-5:
                            break
                else:
                    beta[0] = np.sign(s_12[0]) * max(abs(s_12[0]) - lam, 0) / W_11[0, 0]

                w_12 = W_11 @ beta
                old_w = W[:, j].copy()
                W[np.ix_(idx[:p_s-1], [j])] = w_12.reshape(-1, 1)
                W[j, idx[:p_s-1]] = w_12
                max_change = max(max_change, np.max(np.abs(old_w[idx[:p_s-1]] - w_12)))

            if max_change < tol:
                break

        Omega = np.linalg.inv(W)
        return Omega

    print(f"\n  -- 滑动窗口估计 (窗口半宽={window}, λ={lam}) --")
    print(f"  {'t':>4s}  {'边数':>6s}  {'阶段':>6s}")

    Omega_hats = []
    for t in range(T):
        # 滑动窗口: [max(0, t-window), min(T-1, t+window)]
        t_start = max(0, t - window)
        t_end = min(T - 1, t + window)
        X_window = np.vstack([data_times[s] for s in range(t_start, t_end + 1)])
        S_window = np.cov(X_window, rowvar=False)

        Omega_t = graphical_lasso_simple(S_window, lam)
        Omega_hats.append(Omega_t)

        adj_t = np.abs(Omega_t) > 1e-3
        np.fill_diagonal(adj_t, False)
        n_edges = adj_t.sum() // 2
        phase = "P1" if t < 5 else "P2"
        print(f"  {t+1:>4d}  {n_edges:>6d}  {phase:>6s}")

    # ==== 结构断点检测 ====
    print(f"\n  -- 相邻图的结构差异 (可能指示断点) --")
    for t in range(T - 1):
        diff = np.sum(np.abs(Omega_hats[t] - Omega_hats[t + 1]))
        marker = " ← 断点!" if t == 4 else ""
        bar = "█" * int(diff * 3) if diff > 0 else ""
        print(f"  Δ(t={t+1}→{t+2}): {diff:.4f} {bar}{marker}")

    # ==== 展示恢复的图 ====
    print(f"\n  -- 恢复的图结构 (ASCII) --")
    for t in range(T):
        adj_t = np.abs(Omega_hats[t]) > 1e-3
        np.fill_diagonal(adj_t, False)
        edges_t = []
        for i in range(p):
            for j in range(i + 1, p):
                if adj_t[i, j]:
                    edges_t.append(f"{i+1}-{j+1}")
        phase_label = "[P1 链式]" if t < 5 else "[P2 模块]"
        print(f"  t={t+1:>2d} {phase_label}: {', '.join(edges_t)}")

    print(f"\n  洞察:")
    print(f"    滑动窗口 = 用局部时间邻域的数据估计当前图")
    print(f"    窗口太大 → 平滑掉结构变化 (敏感度低)")
    print(f"    窗口太小 → 估计噪声大 (方差高)")
    print(f"    结构差异 Δ(t) 的峰值 → 可能的结构断点!")
    print(f"    时变 Graphical Lasso 在神经科学 (fMRI 脑连接) 中广泛应用")


# ============================================================================
# 练习 6: PGM 视角 — 结构学习的三种范式
# ============================================================================

def exercise6_structure_learning_paradigms():
    """
    对比结构学习的三种范式:
    1. 基于约束 (Constraint-based): PC 算法, CI 测试
    2. 基于分数 (Score-based): Hill-climbing + BIC
    3. 基于正则化 (Regularization): Graphical Lasso

    在同一个数据集上, 用三种方法学习图结构并对比结果。

    注意: pgmpy 0.1.x 后 BayesianNetwork 改名为 DiscreteBayesianNetwork
    """
    print("=" * 70)
    print("练习 6: PGM 视角 — 结构学习的三种范式对比")
    print("=" * 70)

    try:
        from pgmpy.models import DiscreteBayesianNetwork
        from pgmpy.estimators import HillClimbSearch, BicScore, BayesianEstimator

        print("\n  ✅ pgmpy 导入成功 (DiscreteBayesianNetwork)")

        # 生成离散数据 (真实 DAG: 1→2, 1→3, 2→4, 3→4)
        np.random.seed(123)
        n_samples = 500

        # 手动根据真实 DAG 生成数据
        # X1 ~ Bernoulli(0.5)
        X1 = np.random.binomial(1, 0.5, n_samples)
        # X2 | X1
        X2 = np.array([np.random.binomial(1, 0.8 if x1 == 1 else 0.2) for x1 in X1])
        # X3 | X1
        X3 = np.array([np.random.binomial(1, 0.7 if x1 == 1 else 0.3) for x1 in X1])
        # X4 | X2, X3
        X4 = np.array([np.random.binomial(1, 0.9 if (x2 + x3) >= 1 else 0.1)
                       for x2, x3 in zip(X2, X3)])

        import pandas as pd
        data = pd.DataFrame({
            'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4
        })

        print(f"\n  真实 DAG: 1→2, 1→3, 2→4, 3→4  (v-结构在 4)")
        print(f"  样本数: {n_samples}")

        # ==== 范式 1: 基于分数的结构学习 (Hill-climbing + BIC) ====
        print(f"\n  -- 范式 1: 基于分数 (Hill-climbing + BIC) --")
        hc = HillClimbSearch(data)
        best_model = hc.estimate(scoring_method=BicScore(data))
        print(f"  学习到的边: {list(best_model.edges())}")

        # ==== 范式 2: 基于正则化 (GGM on discretized data) ====
        print(f"\n  -- 范式 2: 基于正则化 (Graphical Lasso on correlation) --")
        # 对离散数据, 可以用相关矩阵的 Graphical Lasso
        corr = data.corr().values
        # 简化的 GLasso
        lam_ggm = 0.1
        W = corr + lam_ggm * np.eye(4)
        Omega = np.linalg.inv(W)
        adj_gl = np.abs(Omega) > 1e-3
        np.fill_diagonal(adj_gl, False)
        edges_gl = []
        for i in range(4):
            for j in range(i + 1, 4):
                if adj_gl[i, j]:
                    edges_gl.append((f'X{i+1}', f'X{j+1}'))
        print(f"  相关矩阵 → GL → 学习到的边: {edges_gl}")
        print(f"  注意: 离散数据的 GGM 需要更严谨的方法 (如 Ising model)")

        # ==== 范式 3: 互信息 + 阈值 (简化约束方法) ====
        print(f"\n  -- 范式 3: 基于约束 (互信息 + 阈值) --")
        from sklearn.metrics import mutual_info_score

        edges_mi = []
        for i in range(4):
            for j in range(i + 1, 4):
                mi = mutual_info_score(data.iloc[:, i], data.iloc[:, j])
                if mi > 0.02:  # 阈值
                    edges_mi.append((f'X{i+1}', f'X{j+1}'))
        print(f"  互信息 > 0.02 → 边: {edges_mi}")
        print(f"  注意: 真正的约束方法需要条件互信息 (如 PC 算法)")

        # ==== 对比总结 ====
        print(f"\n  -- 三种范式对比 --")
        print(f"  {'':<20s} {'基于分数':<20s} {'基于正则化':<20s} {'基于约束':<20s}")
        print(f"  {'-'*20} {'-'*20} {'-'*20} {'-'*20}")
        print(f"  {'代表方法':<20s} {'Hill-climb+BIC':<20s} {'Graphical Lasso':<20s} {'PC Algorithm':<20s}")
        print(f"  {'图类型':<20s} {'DAG':<20s} {'无向图':<20s} {'DAG (等价类)':<20s}")
        print(f"  {'可扩展性':<20s} {'p<30':<20s} {'p>>n 可行':<20s} {'p<50 (CI测试)':<20s}")
        print(f"  {'统计保证':<20s} {'BIC一致性':<20s} {'Oracle性质':<20s} {'渐近一致':<20s}")
        print(f"  {'计算复杂度':<20s} {'NP-hard (启发式)':<20s} {'O(p³)':<20s} {'O(p^k)最坏':<20s}")

        print(f"\n  洞察:")
        print(f"    三种范式对应不同的统计哲学:")
        print(f"    - 基于分数: 寻找'最匹配数据'的 DAG (优化问题)")
        print(f"    - 基于约束: 通过条件独立性测试推断结构 (假设检验)")
        print(f"    - 基于正则化: 用凸优化 + 稀疏惩罚估计精度矩阵 (机器学习)")
        print(f"    对于高斯数据, 基于正则化的方法最具可扩展性和理论保证")

    except ImportError as e:
        print(f"\n  ⚠ pgmpy 导入失败: {e}")
        print(f"  请安装: pip install pgmpy scikit-learn pandas")
        print(f"\n  手动对比三种范式:")
        print(f"  ┌───────────────┬──────────────────┬──────────────────┐")
        print(f"  │ 基于分数       │ 基于约束          │ 基于正则化        │")
        print(f"  ├───────────────┼──────────────────┼──────────────────┤")
        print(f"  │ 优化 BIC/BDe   │ CI 测试 + 方向    │ ℓ1 凸优化        │")
        print(f"  │ Hill-climbing  │ PC / GS 算法      │ Graphical Lasso  │")
        print(f"  │ DAG 搜索       │ 等价类 (CPDAG)    │ 无向图 (GGM)      │")
        print(f"  │ p < ~30        │ p < ~50           │ p >> n 可行       │")
        print(f"  └───────────────┴──────────────────┴──────────────────┘")


# ============================================================================
# 综合测试: 结构学习全流程
# ============================================================================

def exercise_bonus_full_pipeline():
    """
    完整结构学习流程演示:
    原始数据 → 中心化/标准化 → 经验协方差
    → Graphical Lasso (多 λ) → 模型选择 (BIC)
    → 最终图结构 → 与真实结构对比
    """
    print("=" * 70)
    print("综合测试: 结构学习全流程 — 从数据到图")
    print("=" * 70)

    p = 10
    n = 150

    # Step 1: 生成真实结构 (随机稀疏图)
    np.random.seed(789)
    Omega_true = np.eye(p)
    # 随机添加边 (约 15% 的边)
    for i in range(p):
        for j in range(i + 1, p):
            if np.random.random() < 0.15:
                val = -0.3 - np.random.random() * 0.2
                Omega_true[i, j] = val
                Omega_true[j, i] = val

    # 确保正定
    for i in range(p):
        Omega_true[i, i] = np.sum(np.abs(Omega_true[i])) - abs(Omega_true[i, i]) + 0.3

    true_edges = set()
    for i in range(p):
        for j in range(i + 1, p):
            if abs(Omega_true[i, j]) > 1e-6:
                true_edges.add((i + 1, j + 1))

    # Step 2: 生成数据
    Sigma_true = np.linalg.inv(Omega_true)
    L = np.linalg.cholesky(Sigma_true)
    X = (L @ np.random.randn(p, n)).T
    S = np.cov(X, rowvar=False)

    print(f"\n  Step 1-2: 数据生成完成 (p={p}, n={n})")
    print(f"    真实边数: {len(true_edges)} / {p*(p-1)//2}")

    # Step 3: Graphical Lasso (多 λ)
    def graphical_lasso_simple(S, lam, max_iter=100):
        p_s = S.shape[0]
        W = S + lam * np.eye(p_s)
        for it in range(max_iter):
            max_change = 0
            for j in range(p_s):
                idx = list(range(p_s))
                idx.remove(j)
                idx.append(j)
                W_perm = W[np.ix_(idx, idx)]
                S_perm = S[np.ix_(idx, idx)]
                W_11 = W_perm[:p_s-1, :p_s-1]
                s_12 = S_perm[:p_s-1, p_s-1]
                beta = np.zeros(p_s - 1)
                if p_s > 2:
                    for _ in range(30):
                        beta_old = beta.copy()
                        for k in range(p_s - 1):
                            res = s_12[k] - W_11[k] @ beta + W_11[k, k] * beta[k]
                            beta[k] = np.sign(res) * max(abs(res) - lam, 0) / W_11[k, k]
                        if np.max(np.abs(beta - beta_old)) < 1e-5:
                            break
                else:
                    beta[0] = np.sign(s_12[0]) * max(abs(s_12[0]) - lam, 0) / W_11[0, 0]
                w_12 = W_11 @ beta
                old_w = W[:, j].copy()
                W[np.ix_(idx[:p_s-1], [j])] = w_12.reshape(-1, 1)
                W[j, idx[:p_s-1]] = w_12
                max_change = max(max_change, np.max(np.abs(old_w[idx[:p_s-1]] - w_12)))
            if max_change < 1e-4:
                break
        return np.linalg.inv(W)

    print(f"\n  Step 3: Graphical Lasso 路径扫描")
    lam_list = np.logspace(-2, 0, 10)

    results = []
    for lam in lam_list:
        Omega_hat = graphical_lasso_simple(S, lam)
        adj_hat = np.abs(Omega_hat) > 1e-3
        np.fill_diagonal(adj_hat, False)
        n_edges_hat = adj_hat.sum() // 2

        # BIC
        log_lik = n * (np.linalg.slogdet(Omega_hat)[1] - np.trace(S @ Omega_hat))
        k = n_edges_hat + p  # 非零边 + 对角线
        bic = -2 * log_lik + k * np.log(n)

        results.append((lam, n_edges_hat, bic))

    # 选 BIC 最优的 λ
    best_lam, best_edges, best_bic = min(results, key=lambda x: x[2])

    print(f"  {'λ':>10s}  {'边数':>6s}  {'BIC':>10s}")
    for lam, edges, bic in results:
        marker = " ← best" if lam == best_lam else ""
        print(f"  {lam:>10.4f}  {edges:>6d}  {bic:>10.1f}{marker}")

    # Step 4: 最终图结构
    Omega_final = graphical_lasso_simple(S, best_lam)
    adj_final = np.abs(Omega_final) > 1e-3
    np.fill_diagonal(adj_final, False)
    found_edges = set()
    for i in range(p):
        for j in range(i + 1, p):
            if adj_final[i, j]:
                found_edges.add((i + 1, j + 1))

    tp = len(found_edges & true_edges)
    fp = len(found_edges - true_edges)
    fn = len(true_edges - found_edges)

    print(f"\n  Step 4: 最终图 (BIC 选 λ={best_lam:.4f})")
    print(f"    学习到的边: {sorted(found_edges)}")
    print(f"    TP={tp}, FP={fp}, FN={fn}")
    print(f"    Precision = {tp/max(tp+fp,1):.3f}")
    print(f"    Recall    = {tp/max(tp+fn,1):.3f}")

    print(f"\n  -- 全流程总结 --")
    print(f"  ① 从数据估计协方差 S")
    print(f"  ② Graphical Lasso 在多个 λ 下估计 Ω̂_λ")
    print(f"  ③ 用 BIC 选择最优 λ")
    print(f"  ④ 输出最终的图结构 (Ω̂ 的非零模式)")
    print(f"\n  这就是现代高维图结构学习的标准流程!")
    print(f"  扩展到 fMRI 脑连接、基因调控网络、金融网络等")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    run_all = '--ex' not in sys.argv

    exercises = [
        ('1', exercise1_ggm_precision_graph),
        ('2', exercise2_graphical_lasso),
        ('3', exercise3_neighbor_selection),
        ('4', exercise4_partial_correlation),
        ('5', exercise5_time_varying_gl),
        ('6', exercise6_structure_learning_paradigms),
        ('bonus', exercise_bonus_full_pipeline),
    ]

    for ex_id, ex_func in exercises:
        if not run_all and ex_id not in sys.argv:
            continue
        try:
            ex_func()
        except Exception as e:
            print(f"\n  [!] 练习{ex_id}执行出错: {e}")
            import traceback
            traceback.print_exc()
