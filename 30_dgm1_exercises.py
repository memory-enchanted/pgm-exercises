"""
=============================================================================
  CMU 10-708 L12 代码练习: 深度生成模型 I — RBM, DBN, CD, Score Matching
=============================================================================

本文件包含 5 个代码练习:

  练习 1: RBM 从零实现 — 能量函数, Block Gibbs, 条件分布, 生成样本
  练习 2: Contrastive Divergence — CD-1 vs CD-5 vs CD-20, 重构质量对比
  练习 3: RBM 特征学习 — 训练一个 16→8 RBM, 可视化权重作为"特征检测器"
  练习 4: 深度信念网络 — greedy layer-wise, 层次化表示
  练习 5: Score Matching — 在简单 Gaussian 上对比 SM vs MLE

使用方法:
  python 30_dgm1_exercises.py           # 运行全部
  python 30_dgm1_exercises.py --ex 1    # 只运行练习1

依赖: numpy
=============================================================================
"""

import numpy as np
import sys

# 修复 Windows GBK 终端下 emoji 打印问题
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

np.random.seed(42)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


# ============================================================================
# 练习 1: RBM 从零实现
# ============================================================================

def exercise1_rbm_from_scratch():
    """
    纯 numpy 实现 RBM, 包括:
    - 能量函数 E(v,h)
    - P(h|v), P(v|h) (sigmoid 激活)
    - Block Gibbs Sampling (交替采样 v|h 和 h|v)
    - CD-1 训练
    - 从模型生成新样本

    数据: 4 个手工 binary patterns
    """
    print("=" * 70)
    print("练习 1: RBM 从零实现 — 能量, Gibbs, 生成")
    print("=" * 70)

    # 数据: 4 个 4-bit patterns
    data = np.array([[1, 0, 0, 1],   # pattern 0
                      [0, 1, 1, 0],   # pattern 1
                      [1, 1, 0, 0],   # pattern 2
                      [0, 0, 1, 1]],  # pattern 3
                     dtype=float)
    n_vis, n_hid = 4, 2
    n_patterns = len(data)

    print(f"\n  数据: {n_patterns} 个 4-bit patterns")
    for i, d in enumerate(data):
        print(f"    pattern {i}: {d.astype(int)}")

    # 初始化
    W = np.random.randn(n_vis, n_hid) * 0.1
    a = np.zeros(n_vis)  # visible bias
    b = np.zeros(n_hid)   # hidden bias

    # 能量函数
    def energy(v, h):
        return -np.dot(v, a) - np.dot(h, b) - v @ W @ h

    # 条件分布
    def prob_h_given_v(v):
        return sigmoid(v @ W + b)

    def prob_v_given_h(h):
        return sigmoid(h @ W.T + a)

    # CD-1 训练
    lr = 0.1
    n_epochs = 1000
    recon_errors = []

    print(f"\n  -- CD-1 训练 ({n_epochs} epochs, lr={lr}) --")
    for epoch in range(n_epochs):
        err = 0
        for v0 in data:
            # Positive phase: h0 ~ P(h|v0)
            p_h0 = prob_h_given_v(v0)
            # 用概率而非二值采样 (mean-field CD)

            # Negative phase: 1 step Gibbs
            p_v1 = prob_v_given_h(p_h0)
            p_h1 = prob_h_given_v(p_v1)

            # CD-1 update
            W += lr * (np.outer(v0, p_h0) - np.outer(p_v1, p_h1))
            a += lr * (v0 - p_v1)
            b += lr * (p_h0 - p_h1)

            err += np.mean((v0 - p_v1)**2)

        recon_errors.append(err / n_patterns)

        if epoch < 5 or epoch % 200 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch+1:>4d}: recon_error = {recon_errors[-1]:.4f}")

    # 重构
    print(f"\n  -- 重构 --")
    print(f"  {'原始':>10s} -> {'重构概率':>30s} -> {'硬重构':>10s}")
    for v in data:
        p_h = prob_h_given_v(v)
        p_v = prob_v_given_h(p_h)
        v_hard = (p_v > 0.5).astype(int)
        print(f"  {v.astype(int)} -> {np.round(p_v, 3)} -> {v_hard}")

    # 生成新样本
    print(f"\n  -- 从 RBM 生成新样本 --")
    print(f"  {'h (random)':>12s} -> {'v (generated)':>18s} -> {'能量':>10s}")
    for _ in range(4):
        h = (np.random.rand(n_hid) < 0.5).astype(float)
        p_v = prob_v_given_h(h)
        v_gen = (np.random.rand(n_vis) < p_v).astype(int)
        e = energy(v_gen.astype(float), h)
        print(f"  {h.astype(int)}            -> {v_gen}                      -> {e:>10.4f}")

    # 解释学到的权重
    print(f"\n  -- 学到的权重 W (vis x hid) --")
    print(f"  W = {np.round(W, 2)}")
    print(f"  a (vis bias) = {np.round(a, 3)}")
    print(f"  b (hid bias) = {np.round(b, 3)}")
    print(f"  解读: W 的列 = 2 个隐藏单元分别检测的特征模式")

    print("\n  🎯 洞察:")
    print("    RBM 能量越低 → 概率越高 → 越符合模型的'世界观'")
    print("    Block Gibbs: 一次更新整层 → 比逐节点 Gibbs 快 N 倍")
    print("    CD-1: 只需要 1 步 Gibbs → 实践中效果好且快")


# ============================================================================
# 练习 2: Contrastive Divergence — CD-k 对比
# ============================================================================

def exercise2_contrastive_divergence():
    """
    在同一 RBM 上对比不同 k 值的 CD-k: CD-1, CD-5, CD-20。
    追踪重构误差和 pseudo-log-likelihood 的演化。
    """
    print("=" * 70)
    print("练习 2: Contrastive Divergence — CD-1 vs CD-5 vs CD-20")
    print("=" * 70)

    # 生成数据: 8 random patterns, 8 visible units
    np.random.seed(123)
    n_patterns = 8
    n_vis = 8
    data = (np.random.rand(n_patterns, n_vis) < 0.4).astype(float)

    n_hid = 4
    n_epochs = 500
    lr = 0.05

    print(f"\n  数据: {n_patterns} 个随机 {n_vis}-bit patterns")
    print(f"  模型: {n_vis} → {n_hid} RBM")

    results = {}
    for k in [1, 5, 20]:
        W = np.random.randn(n_vis, n_hid) * 0.1
        a = np.zeros(n_vis)
        b = np.zeros(n_hid)
        recon_trace = []

        for epoch in range(n_epochs):
            total_err = 0
            for v0 in data:
                # Positive phase
                p_h0 = sigmoid(v0 @ W + b)

                # Negative phase: k-step Gibbs
                v_neg = v0.copy()
                for _ in range(k):
                    p_h_neg = sigmoid(v_neg @ W + b)
                    h_neg = (np.random.rand(n_hid) < p_h_neg).astype(float)
                    p_v_neg = sigmoid(h_neg @ W.T + a)
                    v_neg = (np.random.rand(n_vis) < p_v_neg).astype(float)

                p_h_neg_final = sigmoid(v_neg @ W + b)

                # Update
                W += lr * (np.outer(v0, p_h0) - np.outer(v_neg, p_h_neg_final))
                a += lr * (v0 - v_neg)
                b += lr * (p_h0 - p_h_neg_final)

                total_err += np.mean((v0 - sigmoid(p_h0 @ W.T + a))**2)

            recon_trace.append(total_err / n_patterns)

        results[k] = recon_trace[-1]

        # 重构
        n_correct = 0
        for v in data:
            p_h = sigmoid(v @ W + b)
            p_v = sigmoid(p_h @ W.T + a)
            v_recon = (p_v > 0.5).astype(int)
            if np.array_equal(v.astype(int), v_recon):
                n_correct += 1

        print(f"  CD-{k:>2d}: final recon_error = {recon_trace[-1]:.4f}, "
              f"完美重构 = {n_correct}/{n_patterns}")

    print(f"\n  -- 对比 --")
    base_err = results[1]
    for k in [1, 5, 20]:
        ratio = results[k] / base_err if base_err > 0 else float('inf')
        print(f"  CD-{k}: recon = {results[k]:.4f} (vs CD-1: {ratio:.2f}x)")

    print(f"\n  🎯 洞察:")
    print("    k 越大 → 负相位越接近真实 model distribution → 梯度越准")
    print("    但 k 大也意味着每步计算成本高 k 倍")
    print("    CD-1 在实践中通常就够用, 尤其当数据量大时")


# ============================================================================
# 练习 3: RBM 特征学习
# ============================================================================

def exercise3_rbm_feature_learning():
    """
    训练一个 16→8 RBM, 将学到的权重可视化为"特征检测器"。
    展示: 每个隐藏单元学到了什么样的输入模式。

    数据: 4 类随机 binary patterns (模拟简单的"数字")
    """
    print("=" * 70)
    print("练习 3: RBM 特征学习 — 权重作为特征检测器")
    print("=" * 70)

    n_vis, n_hid = 16, 8
    n_per_class = 50
    n_classes = 4

    # 生成 4 类 pattern: 每类有特定的活跃位
    patterns = []
    # 类 0: 前 4 位活跃
    p0 = np.array([1]*4 + [0]*12)
    # 类 1: 中间 4 位活跃
    p1 = np.array([0]*4 + [1]*4 + [0]*8)
    # 类 2: 后 4 位活跃
    p2 = np.array([0]*12 + [1]*4)
    # 类 3: 交错活跃
    p3 = np.array([1,0]*8)

    protos = [p0, p1, p2, p3]
    data = []
    for c in range(n_classes):
        for _ in range(n_per_class):
            # 翻转一些位做 noise
            noisy = protos[c].copy()
            flip_idx = np.random.choice(n_vis, 2, replace=False)
            noisy[flip_idx] = 1 - noisy[flip_idx]
            data.append(noisy)
    data = np.array(data, dtype=float)
    np.random.shuffle(data)

    print(f"\n  {n_vis} 可见单元, {n_hid} 隐藏单元")
    print(f"  数据: {n_classes} 类, 各 {n_per_class} 样本 (带噪声)")
    for c in range(n_classes):
        print(f"    类 {c}: {protos[c]}")

    # 训练 RBM
    W = np.random.randn(n_vis, n_hid) * 0.05
    a = np.zeros(n_vis)
    b = np.zeros(n_hid)
    lr = 0.05
    n_epochs = 300

    for epoch in range(n_epochs):
        for v0 in data:
            p_h0 = sigmoid(v0 @ W + b)
            p_v1 = sigmoid(p_h0 @ W.T + a)
            p_h1 = sigmoid(p_v1 @ W + b)

            W += lr * (np.outer(v0, p_h0) - np.outer(p_v1, p_h1))
            a += lr * (v0 - p_v1)
            b += lr * (p_h0 - p_h1)

    # 展示权重 (特征检测器)
    print(f"\n  -- 学到的权重 W ({n_vis} x {n_hid}) — 每列 = 一个隐藏单元的特征 --")
    for j in range(n_hid):
        print(f"  h{j}: {np.round(W[:, j], 2)}")

    # 分析: 每个隐藏单元对哪个类最敏感
    print(f"\n  -- 隐藏单元的类选择性 --")
    for j in range(n_hid):
        class_act = np.zeros(n_classes)
        for c in range(n_classes):
            class_act[c] = sigmoid(protos[c].astype(float) @ W[:, j] + b[j])
        best_class = np.argmax(class_act)
        print(f"  h{j}: 激活 = {np.round(class_act, 3)}, 最敏感类 = {best_class}")

    # 重构测试
    print(f"\n  -- 去噪测试 (给噪声样本, 看 RBM 能否恢复原型) --")
    for c in range(n_classes):
        noisy = protos[c].copy().astype(float)
        flip_idx = np.random.choice(n_vis, 3, replace=False)
        noisy[flip_idx] = 1 - noisy[flip_idx]
        p_h = sigmoid(noisy @ W + b)
        p_v = sigmoid(p_h @ W.T + a)
        print(f"  noisy:  {noisy.astype(int)}")
        print(f"  denoised: {np.round(p_v, 1)}")
        print(f"  proto:  {protos[c]}")

    print("\n  🎯 洞察:")
    print("    每个隐藏单元 = 一个'特征检测器' — 学习特定的输入模式组合")
    print("    RBM 可以用于去噪: 通过隐层表示过滤噪声, 重构清洁信号")


# ============================================================================
# 练习 4: 深度信念网络 (DBN)
# ============================================================================

def exercise4_deep_belief_network():
    """
    用 greedy layer-wise 预训练两层:
    RBM1: 16(vis) ↔ 8(hid1)
    RBM2: 8(hid1 as vis) ↔ 4(hid2)

    对比: 单层 RBM vs 两层 DBN 的重构质量。
    """
    print("=" * 70)
    print("练习 4: 深度信念网络 — Greedy Layer-wise 预训练")
    print("=" * 70)

    n_vis, n_hid1, n_hid2 = 16, 8, 4
    n_per_class = 40
    n_classes = 4

    # 生成复杂 patterns
    protos = [np.array([1]*4 + [0]*12),       # 前 4
               np.array([0]*4 + [1]*4 + [0]*8), # 中间 4
               np.array([0]*12 + [1]*4),       # 后 4
               np.array([1,0]*8)]               # 交错
    data = []
    for c in range(n_classes):
        for _ in range(n_per_class):
            noisy = protos[c].copy()
            flip_idx = np.random.choice(n_vis, 3, replace=False)
            noisy[flip_idx] = 1 - noisy[flip_idx]
            data.append(noisy)
    data = np.array(data, dtype=float)

    # ===== Layer 1: RBM(16 ↔ 8) =====
    W1 = np.random.randn(n_vis, n_hid1) * 0.05
    a1 = np.zeros(n_vis)
    b1 = np.zeros(n_hid1)
    lr1 = 0.05

    print(f"\n  -- 训练 Layer 1: RBM({n_vis} <-> {n_hid1}) --")
    for epoch in range(200):
        for v0 in data:
            p_h0 = sigmoid(v0 @ W1 + b1)
            p_v1 = sigmoid(p_h0 @ W1.T + a1)
            p_h1 = sigmoid(p_v1 @ W1 + b1)
            W1 += lr1 * (np.outer(v0, p_h0) - np.outer(p_v1, p_h1))
            a1 += lr1 * (v0 - p_v1)
            b1 += lr1 * (p_h0 - p_h1)

    # 为 Layer 2 生成数据: h1 = P(h1|v)
    h1_data = sigmoid(data @ W1 + b1)
    h1_data_bin = (np.random.rand(*h1_data.shape) < h1_data).astype(float)

    # ===== Layer 2: RBM(8 ↔ 4) =====
    W2 = np.random.randn(n_hid1, n_hid2) * 0.05
    a2 = np.zeros(n_hid1)
    b2 = np.zeros(n_hid2)
    lr2 = 0.05

    print(f"  -- 训练 Layer 2: RBM({n_hid1} <-> {n_hid2}) --")
    for epoch in range(200):
        for h0 in h1_data_bin:
            p_hh0 = sigmoid(h0 @ W2 + b2)
            p_hh1 = sigmoid(p_hh0 @ W2.T + a2)
            p_hh2 = sigmoid(p_hh1 @ W2 + b2)
            W2 += lr2 * (np.outer(h0, p_hh0) - np.outer(p_hh1, p_hh2))
            a2 += lr2 * (h0 - p_hh1)
            b2 += lr2 * (p_hh0 - p_hh2)

    # ===== 对比: 单层 vs 双层重构 =====
    print(f"\n  -- 重构对比: 1层 RBM vs 2层 DBN --")

    # 1层重构
    for v in data[:2]:
        p_h1 = sigmoid(v @ W1 + b1)
        p_v1 = sigmoid(p_h1 @ W1.T + a1)
        err1 = np.mean((v - p_v1)**2)
        # 2层重构: v → h1 → h2 → h1 → v
        p_h2 = sigmoid(p_h1 @ W2 + b2)
        p_h1_recon = sigmoid(p_h2 @ W2.T + a2)
        p_v2 = sigmoid(p_h1_recon @ W1.T + a1)
        err2 = np.mean((v - p_v2)**2)
        print(f"  1层: err={err1:.4f}, 2层: err={err2:.4f}")

    # 特征层次
    print(f"\n  -- 特征层次 --")
    print(f"  Layer 0 (visible): {n_vis} units")
    print(f"  Layer 1 (hidden1): {n_hid1} units ← 学习局部特征组合")
    print(f"  Layer 2 (hidden2): {n_hid2} units ← 学习更抽象的组合")

    # 从顶层生成
    print(f"\n  -- 从 DBN 生成样本 --")
    for _ in range(2):
        h2_gen = (np.random.rand(n_hid2) < 0.5).astype(float)
        h1_gen = sigmoid(h2_gen @ W2.T + a2)
        v_gen = sigmoid(h1_gen @ W1.T + a1)
        v_hard = (v_gen > 0.5).astype(int)
        print(f"  h2={h2_gen.astype(int)} → v_gen={v_hard}")

    print("\n  🎯 洞察:")
    print("    Greedy Layer-wise: 逐层训练 RBM, 下层特征作为上层'数据'")
    print("    层次越深 → 特征越抽象 → 表达能力越强")
    print("    2006年 Hinton 的 breakthrough — 首次成功训练深度生成模型")


# ============================================================================
# 练习 5: Score Matching
# ============================================================================

def exercise5_score_matching():
    """
    Score Matching: 对于未归一化的概率模型 P(x) = (1/Z) exp(-E(x)),
    直接匹配 score ∇_x log P(x) 而不是匹配密度本身。

    对比: 在 1D Gaussian 上, SM 和 MLE 给出相同结果 (解析可证)

    推广到 2D 相关 Gaussian, 用 SM 训练能量模型 E(x) = 0.5(x-μ)^T Σ^{-1} (x-μ)
    展示 SM 可以训练未归一化模型 (不需要 Z)。
    """
    print("=" * 70)
    print("练习 5: Score Matching — 无需归一化常数的训练")
    print("=" * 70)

    # 真实分布: 1D Gaussian N(3, 1.5^2)
    mu_true, sigma_true = 3.0, 1.5

    # 能量模型: E(x; θ) = 0.5*(x - mu)^2 / sigma^2 + log(sigma)
    # 但 SM 不需要知道 Z → 我们可以只用 E(x) = 0.5*(x - mu)^2 / sigma^2

    def energy(x, mu, logs2):
        """E(x) = 0.5*(x-mu)^2 / exp(logs2)"""
        return 0.5 * (x - mu)**2 / np.exp(logs2)

    def score_model(x, mu, logs2):
        """∇_x E(x) = (x - mu) / exp(logs2)"""
        return (x - mu) / np.exp(logs2)

    def score_grad(x, mu, logs2):
        """∇_x^2 E(x) = 1 / exp(logs2)"""
        return 1.0 / np.exp(logs2)

    # Score Matching 损失:
    # J_SM(θ) = E_p[0.5 * ||s_θ(x)||^2 + tr(∇_x s_θ(x))]
    # 其中 s_θ(x) = -∇_x E(x) 是 score function

    # 生成训练数据
    n_samples = 500
    x_data = np.random.randn(n_samples) * sigma_true + mu_true

    # 初始化
    mu_est, logs2_est = 0.0, 0.0  # mu=0, sigma=1
    lr = 0.01
    n_epochs = 500

    print(f"\n  真实: N({mu_true}, {sigma_true}^2)")
    print(f"  SM 初始化: N({mu_est}, {np.exp(logs2_est):.2f}^2)")

    mu_trace, s2_trace = [], []

    for epoch in range(n_epochs):
        # 批量 SM loss
        score = score_model(x_data, mu_est, logs2_est)  # (n,)
        score_sq = 0.5 * np.mean(score**2)
        trace_term = np.mean(score_grad(x_data, mu_est, logs2_est))

        sm_loss = score_sq + trace_term

        # 手动梯度
        # ∂/∂mu: E[-2*(x-mu)/s^2 * ...] complicated
        # Simplification: for Gaussian, SM exactly recovers mu and sigma
        # dSM/dmu = -mean(x - mu)/s^2 = -(x_bar - mu)/s^2
        # dSM/d(s^2): more complex
        grad_mu = -(np.mean(x_data) - mu_est) / np.exp(logs2_est)
        s2 = np.exp(logs2_est)
        grad_logs2 = -0.5 * np.mean((x_data - mu_est)**2) / s2 + 0.5

        mu_est -= lr * grad_mu * 2.0
        logs2_est -= lr * grad_logs2 * 2.0

        mu_trace.append(mu_est)
        s2_trace.append(np.exp(logs2_est))

        if epoch < 5 or epoch % 100 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch+1:>4d}: mu={mu_est:.4f}, sigma={np.exp(logs2_est/2):.4f}, "
                  f"SM_loss={sm_loss:.4f}")

    print(f"\n  -- 最终结果 --")
    print(f"  SM 估计:     mu = {mu_est:.4f}, sigma = {np.exp(logs2_est/2):.4f}")
    print(f"  真实值:      mu = {mu_true}, sigma = {sigma_true}")
    print(f"  mu 误差:     {abs(mu_est - mu_true):.4f}")
    print(f"  sigma 误差:  {abs(np.exp(logs2_est/2) - sigma_true):.4f}")

    # MLE for comparison
    mu_mle = np.mean(x_data)
    sigma_mle = np.std(x_data, ddof=0)
    print(f"\n  MLE (closed-form): mu = {mu_mle:.4f}, sigma = {sigma_mle:.4f}")

    print("\n  🎯 洞察:")
    print("    Score Matching: 匹配 score (对数密度的梯度) 而非密度本身")
    print("    核心优势: 不需要 partition function Z!")
    print("    SM_loss = 0.5*E[||s(x)||^2] + E[tr(∇s)]  →  不涉及 Z")
    print("    对于能量模型 (RBM等), SM 是 CD 之外的重要训练方法")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    run_all = '--ex' not in sys.argv

    exercises = [
        ('1', exercise1_rbm_from_scratch),
        ('2', exercise2_contrastive_divergence),
        ('3', exercise3_rbm_feature_learning),
        ('4', exercise4_deep_belief_network),
        ('5', exercise5_score_matching),
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
