"""
=============================================================================
  CMU 10-708 L5 代码练习: 参数估计 (Parameter Estimation)
=============================================================================

本文件包含 5 个代码练习:

  练习 1: MLE 手算 — 一元 & 多元高斯分布的 MLE
  练习 2: IRLS — 迭代重加权最小二乘拟合逻辑回归
  练习 3: K-Means — 手写硬聚类算法
  练习 4: EM for GMM — 用 EM 拟合高斯混合模型
  练习 5: K-Means vs EM — 对比硬分配和软分配的差异

使用方法:
  python 09_bp_exercises.py           # 运行全部练习
  python 09_bp_exercises.py --ex 1    # 只运行练习1

依赖: numpy, scipy, matplotlib, scikit-learn
=============================================================================
"""

import numpy as np
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


# ============================================================================
# 练习 1: MLE 手算 — 高斯分布
# ============================================================================

def exercise1_mle_gaussian():
    """
    手算一元和多元高斯分布的 MLE。
    验证: MLE 解 = 样本均值 & 样本协方差。
    """
    print("=" * 70)
    print("练习 1: MLE 手算 — 高斯分布")
    print("=" * 70)

    # --- 一元高斯 ---
    np.random.seed(42)
    true_mu, true_sigma = 5.0, 2.0
    N = 1000
    data_1d = np.random.normal(true_mu, true_sigma, N)

    # MLE 公式
    mu_mle = np.mean(data_1d)
    sigma2_mle = np.mean((data_1d - mu_mle) ** 2)  # 分母 N (有偏MLE)

    print(f"\n  ── 一元高斯 MLE (N={N}) ──")
    print(f"  真实 μ  = {true_mu},    MLE μ*  = {mu_mle:.4f}")
    print(f"  真实 σ² = {true_sigma**2}, MLE σ²* = {sigma2_mle:.4f}")
    print(f"  无偏估计 σ² = {np.var(data_1d, ddof=1):.4f} (分母 N-1)")

    # 证明 MLE 是充分统计量的函数
    print(f"\n  充分统计量: Σx={np.sum(data_1d):.1f}, Σx²={np.sum(data_1d**2):.1f}")
    print(f"  MLE 只用这两个数就可以算出 — 不需要原始数据!")

    # --- 多元高斯 ---
    d = 3
    true_mu_vec = np.array([1.0, -2.0, 3.0])
    true_Sigma = np.array([[2.0, 0.5, 0.3],
                            [0.5, 1.0, 0.2],
                            [0.3, 0.2, 1.5]])
    data_mv = np.random.multivariate_normal(true_mu_vec, true_Sigma, N)

    mu_mv_mle = np.mean(data_mv, axis=0)
    centered = data_mv - mu_mv_mle
    Sigma_mle = (centered.T @ centered) / N  # (d,N) @ (N,d) = (d,d)

    print(f"\n  ── 多元高斯 MLE (d={d}, N={N}) ──")
    print(f"  真实 μ:  {true_mu_vec}")
    print(f"  MLE  μ*: {mu_mv_mle}")
    print(f"\n  真实 Σ:\n{true_Sigma}")
    print(f"\n  MLE  Σ*:\n{Sigma_mle}")
    print(f"\n  Σ 的参数数量: {d*(d+1)//2} (d={d} → {d*(d+1)//2} 个)")

    # 演示: N < d 时 Σ 不可逆
    print(f"\n  ── 演示: N < d 时 Σ 不可逆 ──")
    N_small = 2
    data_small = np.random.multivariate_normal(true_mu_vec, true_Sigma, N_small)
    Sigma_small = np.cov(data_small.T, ddof=1) if N_small > 1 else np.eye(d)
    rank = np.linalg.matrix_rank(Sigma_small)
    print(f"  N={N_small}, d={d} → Σ 的秩={rank} (满秩需要 N≥d={d})")
    print(f"  结论: 高维问题 (d>N) 需要正则化 (Ledoit-Wolf, 稀疏估计等)")

    print("\n  🎯 洞察:")
    print("    高斯 MLE = 最直观的结果: 均值算平均, 协方差算外积平均")
    print("    MLE 方差是 SAMPLE variance (分母N), 不是无偏估计 (分母N-1)")
    print("    高维 (d>N): Σ* 奇异不可逆 → 需要正则化或降维")


# ============================================================================
# 练习 2: IRLS — 迭代重加权最小二乘
# ============================================================================

def exercise2_irls_logistic():
    """
    手写 IRLS 拟合逻辑回归, 追踪每次迭代的权重变化。
    """
    print("=" * 70)
    print("练习 2: IRLS — 逻辑回归的参数估计")
    print("=" * 70)

    np.random.seed(42)
    N = 200
    d = 2  # 特征维度 (不含截距)

    # 生成数据
    X = np.random.randn(N, d)
    true_theta = np.array([2.0, -1.0])
    intercept = 0.5
    logits = X @ true_theta + intercept
    probs = 1 / (1 + np.exp(-logits))
    y = np.random.binomial(1, probs)

    # 加截距列
    X_aug = np.column_stack([np.ones(N), X])  # (N, 3)
    theta = np.zeros(d + 1)  # [intercept, theta_1, theta_2]

    print(f"\n  ── 逻辑回归 IRLS (N={N}, d={d+1}含截距) ──")
    print(f"  真实参数: intercept={intercept}, θ={true_theta}")

    print(f"\n  {'迭代':>5s}  {'log-lik':>12s}  {'||g||':>10s}  {'θ₀':>8s}  {'θ₁':>8s}  {'θ₂':>8s}")
    print(f"  {'─'*5}  {'─'*12}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}")

    for iteration in range(20):
        # 当前预测
        eta = X_aug @ theta
        mu = 1 / (1 + np.exp(-eta))

        # 计算对数似然
        eps = 1e-15
        log_lik = np.sum(y * np.log(mu + eps) + (1 - y) * np.log(1 - mu + eps))

        # 梯度
        g = X_aug.T @ (y - mu)

        # 权重矩阵 W = diag(μ(1-μ))
        W_diag = mu * (1 - mu)
        W = np.diag(W_diag)

        # 工作响应 z = Xθ + W⁻¹(y - μ)
        z = eta + (y - mu) / np.maximum(W_diag, 1e-10)

        # IRLS 更新
        XtWX = X_aug.T @ W @ X_aug
        XtWz = X_aug.T @ W @ z

        try:
            theta_new = np.linalg.solve(XtWX, XtWz)
        except np.linalg.LinAlgError:
            theta_new = np.linalg.lstsq(XtWX, XtWz, rcond=None)[0]

        delta = np.linalg.norm(theta_new - theta)
        theta = theta_new

        if iteration < 10 or iteration % 5 == 0:
            print(f"  {iteration:>5d}  {log_lik:>12.2f}  {np.linalg.norm(g):>10.4f}  "
                  f"{theta[0]:>8.4f}  {theta[1]:>8.4f}  {theta[2]:>8.4f}")

        if delta < 1e-8:
            print(f"\n  IRLS 在 {iteration+1} 次迭代后收敛!")
            break

    # 对比 sklearn
    try:
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(penalty=None, fit_intercept=True, max_iter=1000)
        lr.fit(X, y)
        print(f"\n  ── 对比 sklearn ──")
        print(f"  IRLS (手写): θ = {theta}")
        print(f"  sklearn:     θ = [{lr.intercept_[0]:.4f}, {lr.coef_[0][0]:.4f}, {lr.coef_[0][1]:.4f}]")
    except ImportError:
        pass

    print("\n  🎯 洞察:")
    print("    IRLS = 每步求解加权最小二乘, 权重 = μ(1-μ)")
    print("    预测不确定的点 (μ≈0.5) 权重最大 → 对决策边界贡献最多")
    print("    与牛顿法等价, 但利用了 Hessian 的期望形式 (Fisher Scoring)")


# ============================================================================
# 练习 3: K-Means — 手写硬聚类
# ============================================================================

def exercise3_kmeans():
    """
    手写 K-Means, 可视化收敛过程。
    """
    print("=" * 70)
    print("练习 3: K-Means — 硬分配聚类")
    print("=" * 70)

    np.random.seed(42)
    N = 300
    K = 3

    # 生成 3 个高斯混合的数据
    true_centers = np.array([[0, 0], [5, 5], [0, 8]])
    data = np.vstack([
        np.random.randn(100, 2) * 1.0 + true_centers[0],
        np.random.randn(100, 2) * 1.2 + true_centers[1],
        np.random.randn(100, 2) * 0.8 + true_centers[2],
    ])
    np.random.shuffle(data)

    # 随机初始化中心
    rng = np.random.RandomState(123)
    centers = data[rng.choice(N, K, replace=False)]

    print(f"\n  ── K-Means 聚类 (N={N}, K={K}) ──")
    print(f"  初始中心 (随机选):")
    for k in range(K):
        print(f"    μ_{k} = [{centers[k][0]:.2f}, {centers[k][1]:.2f}]")

    losses = []
    for iteration in range(30):
        # Step 1: 分配 (Assignment) — E-Step 的硬版本
        # 对每个点, 找最近的中心
        dists = np.zeros((N, K))
        for k in range(K):
            dists[:, k] = np.sum((data - centers[k]) ** 2, axis=1)
        assignments = np.argmin(dists, axis=1)

        # 计算损失 (类内平方和)
        loss = 0
        for i in range(N):
            loss += np.sum((data[i] - centers[assignments[i]]) ** 2)
        losses.append(loss)

        # Step 2: 更新 (Update) — M-Step 的硬版本
        new_centers = np.zeros_like(centers)
        for k in range(K):
            mask = assignments == k
            if mask.sum() > 0:
                new_centers[k] = data[mask].mean(axis=0)
            else:
                new_centers[k] = centers[k]  # 空类, 保持不变

        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers

        print(f"  迭代 {iteration+1:>2d}: 损失={loss:.1f}, 中心移动={shift:.4f}")

        if shift < 1e-6:
            print(f"\n  K-Means 在 {iteration+1} 次迭代后收敛!")
            break

    print(f"\n  ── 最终结果 ──")
    for k in range(K):
        n_k = (assignments == k).sum()
        print(f"  类 {k}: 中心=({centers[k][0]:.2f}, {centers[k][1]:.2f}), "
              f"点数={n_k}")

    print(f"\n  损失函数变化: {losses[0]:.0f} → {losses[-1]:.0f}")
    print(f"  损失下降: {losses[0] - losses[-1]:.0f}")

    print("\n  🎯 洞察:")
    print("    K-Means = 两步交替:")
    print("      (1) 分配: 每个点 → 最近中心 (硬, 0或1)")
    print("      (2) 更新: 每个类 → 类内均值")
    print("    损失(类内平方和)单调下降 — 类似 EM 的单调性")


# ============================================================================
# 练习 4: EM for GMM — 高斯混合模型
# ============================================================================

def exercise4_em_gmm():
    """
    手写 EM 算法拟合高斯混合模型 (GMM)。
    展示 E 步 (软分配) 和 M 步 (加权 MLE) 的区别。
    """
    print("=" * 70)
    print("练习 4: EM for GMM — 高斯混合模型")
    print("=" * 70)

    np.random.seed(42)
    N = 300
    K = 3

    # 生成数据 (与 K-Means 练习相同的分布)
    true_centers = np.array([[0, 0], [5, 5], [0, 8]])
    data = np.vstack([
        np.random.randn(100, 2) * 1.0 + true_centers[0],
        np.random.randn(100, 2) * 1.2 + true_centers[1],
        np.random.randn(100, 2) * 0.8 + true_centers[2],
    ])
    np.random.shuffle(data)
    d = 2

    # 初始化参数
    rng = np.random.RandomState(123)
    # 用 K-Means 初始化 (好习惯!)
    from scipy.spatial.distance import cdist
    init_centers = data[rng.choice(N, K, replace=False)]
    for _ in range(10):
        dists = cdist(data, init_centers)
        labels = np.argmin(dists, axis=1)
        for k in range(K):
            if (labels == k).sum() > 0:
                init_centers[k] = data[labels == k].mean(axis=0)

    mu = init_centers.copy()
    Sigma = np.array([np.eye(d) for _ in range(K)])
    pi = np.ones(K) / K  # 混合系数

    print(f"\n  ── EM for GMM (N={N}, K={K}, d={d}) ──")
    print(f"  初始化: π={pi}, 各 μ 用 K-Means 初始化, Σ=I")

    log_liks = []
    for iteration in range(50):
        # ===== E-Step: 计算后验概率 (responsibilities) =====
        # γ_{nk} = P(zⁿ=k | xⁿ) ∝ π_k × N(xⁿ | μ_k, Σ_k)

        gamma = np.zeros((N, K))  # responsibility matrix

        for k in range(K):
            try:
                from scipy.stats import multivariate_normal
                gamma[:, k] = pi[k] * multivariate_normal.pdf(data, mean=mu[k], cov=Sigma[k])
            except np.linalg.LinAlgError:
                # Σ 奇异时的 fallback
                Sigma[k] += 1e-6 * np.eye(d)
                from scipy.stats import multivariate_normal
                gamma[:, k] = pi[k] * multivariate_normal.pdf(data, mean=mu[k], cov=Sigma[k])

        # 归一化 (每行 sum = 1)
        gamma_sum = gamma.sum(axis=1, keepdims=True)
        gamma_sum = np.maximum(gamma_sum, 1e-15)
        gamma /= gamma_sum

        # 计算对数似然
        log_lik = np.sum(np.log(gamma_sum))
        log_liks.append(log_lik)

        # ===== M-Step: 用加权数据做 MLE =====
        N_k = gamma.sum(axis=0)  # 每个类的"有效样本数"

        # 更新 π
        pi_new = N_k / N

        # 更新 μ
        mu_new = np.zeros_like(mu)
        for k in range(K):
            mu_new[k] = (gamma[:, k].reshape(-1, 1) * data).sum(axis=0) / max(N_k[k], 1e-10)

        # 更新 Σ
        Sigma_new = np.zeros_like(Sigma)
        for k in range(K):
            if N_k[k] > 1e-10:
                centered = data - mu_new[k]
                # 加权协方差
                Sigma_new[k] = (centered.T @ (gamma[:, k].reshape(-1, 1) * centered)) / N_k[k]
                # 正则化微扰
                Sigma_new[k] += 1e-6 * np.eye(d)

        # 检查收敛
        mu_shift = np.linalg.norm(mu_new - mu)
        mu, Sigma, pi = mu_new, Sigma_new, pi_new

        if iteration < 5 or iteration % 5 == 0:
            print(f"  迭代 {iteration+1:>2d}: log-lik={log_lik:>10.2f}, "
                  f"N_k={N_k.round(0)}, μ移动={mu_shift:.4f}")

        if mu_shift < 1e-5:
            print(f"\n  EM 在 {iteration+1} 次迭代后收敛!")
            break

    # 最终结果
    print(f"\n  ── 最终结果 ──")
    for k in range(K):
        print(f"  分量 {k}: π={pi[k]:.3f}, μ=({mu[k][0]:.2f}, {mu[k][1]:.2f})")
        print(f"         Σ={Sigma[k].diagonal()} (对角元素)")

    print(f"\n  ── log-lik 变化 ──")
    print(f"  初始 → 最终: {log_liks[0]:.2f} → {log_liks[-1]:.2f}")
    print(f"  增量: {log_liks[-1] - log_liks[0]:.2f}")
    print(f"  单调性: {'✅ 始终递增' if all(np.diff(log_liks) >= -1e-10
                    for _ in [1] if all(np.diff(log_liks) >= -1e-10)) else '⚠️ 有波动'}")

    print("\n  🎯 洞察:")
    print("    E-Step: γ_{nk} = P(zⁿ=k|xⁿ) — 软分配 (概率, 不是0/1)")
    print("    M-Step: 用 γ 作为权重做加权 MLE — 每个点对每个类都有贡献")
    print("    软分配让 EM 能表达不确定性 — 边界上的点可以同时属于多个类")


# ============================================================================
# 练习 5: K-Means vs EM — 对比
# ============================================================================

def exercise5_kmeans_vs_em():
    """
    同一数据上对比 K-Means 和 EM for GMM。
    展示硬分配 vs 软分配的根本区别。
    """
    print("=" * 70)
    print("练习 5: K-Means vs EM — 硬 vs 软分配")
    print("=" * 70)

    np.random.seed(42)

    # 生成非球形、不同大小的类 — K-Means 会挣扎!
    N = 300
    # 类1: 细长的椭圆 (K-Means 不喜欢)
    cov1 = np.array([[3.0, 2.5], [2.5, 3.0]])
    cluster1 = np.random.multivariate_normal([0, 0], cov1, 100)

    # 类2: 小圆 (K-Means 可能忽略)
    cluster2 = np.random.multivariate_normal([6, 0], 0.3 * np.eye(2), 50)

    # 类3: 中等圆形
    cluster3 = np.random.multivariate_normal([3, 5], np.eye(2), 150)

    data = np.vstack([cluster1, cluster2, cluster3])
    np.random.shuffle(data)
    K = 3

    # --- K-Means ---
    from scipy.spatial.distance import cdist
    rng = np.random.RandomState(123)
    km_centers = data[rng.choice(len(data), K, replace=False)]
    for _ in range(100):
        dists = cdist(data, km_centers)
        labels = np.argmin(dists, axis=1)
        new_centers = np.array([data[labels == k].mean(axis=0) for k in range(K)])
        if np.linalg.norm(new_centers - km_centers) < 1e-6:
            break
        km_centers = new_centers
    km_labels = labels

    print("\n  ── K-Means 结果 ──")
    for k in range(K):
        n_k = (km_labels == k).sum()
        print(f"  类 {k}: 中心=({km_centers[k][0]:.2f}, {km_centers[k][1]:.2f}), N={n_k}")

    # --- EM for GMM ---
    from scipy.stats import multivariate_normal
    # 用 K-Means 结果初始化
    mu = km_centers.copy()
    Sigma = np.array([np.eye(2) for _ in range(K)])
    pi = np.array([(km_labels == k).sum() / len(data) for k in range(K)])

    for _ in range(50):
        # E-Step
        gamma = np.zeros((len(data), K))
        for k in range(K):
            try:
                gamma[:, k] = pi[k] * multivariate_normal.pdf(data, mean=mu[k], cov=Sigma[k])
            except np.linalg.LinAlgError:
                Sigma[k] += 1e-6 * np.eye(2)
                gamma[:, k] = pi[k] * multivariate_normal.pdf(data, mean=mu[k], cov=Sigma[k])
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-15

        # M-Step
        N_k = gamma.sum(axis=0)
        mu_new = np.array([(gamma[:, k:k+1] * data).sum(axis=0) / max(N_k[k], 1e-10)
                           for k in range(K)])
        Sigma_new = np.array([
            ((data - mu_new[k]).T @ (gamma[:, k:k+1] * (data - mu_new[k])) / max(N_k[k], 1e-10))
            + 1e-6 * np.eye(2)
            for k in range(K)
        ])
        pi_new = N_k / len(data)
        if np.linalg.norm(mu_new - mu) < 1e-6:
            mu, Sigma, pi = mu_new, Sigma_new, pi_new
            break
        mu, Sigma, pi = mu_new, Sigma_new, pi_new

    em_soft_labels = gamma.argmax(axis=1)

    print(f"\n  ── EM for GMM 结果 ──")
    for k in range(K):
        n_k = gamma[:, k].sum()
        print(f"  分量 {k}: π={pi[k]:.3f}, μ=({mu[k][0]:.2f}, {mu[k][1]:.2f})")
        print(f"         Σ=[[{Sigma[k][0,0]:.2f}, {Sigma[k][0,1]:.2f}],"
              f" [{Sigma[k][1,0]:.2f}, {Sigma[k][1,1]:.2f}]]")
        print(f"         有效 N={n_k:.1f}")

    # 对比
    print(f"\n  ── 核心区别 ──")
    print(f"  {'':20s} {'K-Means':>15s} {'EM-GMM':>15s}")
    print(f"  {'─'*20} {'─'*15} {'─'*15}")
    print(f"  {'分配方式':20s} {'硬 (0或1)':>15s} {'软 (概率)':>15s}")
    print(f"  {'协方差':20s} {'隐含 Σ=I':>15s} {'学习完整 Σ':>15s}")
    print(f"  {'类大小':20s} {'隐含等大':>15s} {'学习 π_k':>15s}")
    print(f"  {'类形状':20s} {'只能是球':>15s} {'任意椭圆':>15s}")
    print(f"  {'ε-敏感度':20s} {'高 (σ²→0 极限)':>15s} {'低 (完整模型)':>15s}")

    print("\n  🎯 一句话总结:")
    print("    K-Means = GMM EM 当 Σ_k=σ²I 且 σ²→0 时的极限情况")
    print("    → 理解了 EM, 就把 K-Means 为什么'只爱球形'也理解了")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    run_all = '--ex' not in sys.argv

    exercises = [
        ('1', exercise1_mle_gaussian),
        ('2', exercise2_irls_logistic),
        ('3', exercise3_kmeans),
        ('4', exercise4_em_gmm),
        ('5', exercise5_kmeans_vs_em),
    ]

    for ex_id, ex_func in exercises:
        if not run_all and ex_id not in sys.argv:
            continue
        try:
            ex_func()
            print()
        except Exception as e:
            print(f"\n  ⚠️ 练习{ex_id}执行出错: {e}")
            import traceback
            traceback.print_exc()
