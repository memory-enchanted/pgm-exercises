"""
=============================================================================
  CMU 10-708 L8 代码练习: 变分推断 II — SVI, BBVI, Wake-Sleep, VAE
=============================================================================

本文件包含 5 个代码练习:

  练习 1: Wake-Sleep 算法 — Bernoulli 混合模型上的交替训练
  练习 2: SVI 随机变分推断 — mini-batch 自然梯度 vs 全量 CAVI
  练习 3: Reparameterization Trick — 方差对比实验
  练习 4: BBVI + Control Variate — 降低 score function 梯度方差
  练习 5: 简易 VAE — 纯 numpy 实现的编码器-解码器

使用方法:
  python 18_vi2_exercises.py           # 运行全部练习
  python 18_vi2_exercises.py --ex 1    # 只运行练习1

依赖: numpy, scipy (可选, 仅练习5用于 log-likelihood)
=============================================================================
"""

import numpy as np
import sys

# 修复 Windows GBK 终端下 emoji 打印问题
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# 全局随机种子
np.random.seed(42)


# ============================================================================
# 练习 1: Wake-Sleep 算法
# ============================================================================

def exercise1_wake_sleep():
    """
    在 Bernoulli 混合模型上实现 Wake-Sleep 算法:
    - 生成模型 P(X|Z): Z 是 K 维 one-hot, X 是 D 维 Bernoulli
    - 识别网络 Q(Z|X): 输入 X, 输出 softmax 概率

    Wake 阶段: 用真实数据训练 P(X|Z)
    Sleep 阶段: 用生成数据训练 Q(Z|X)
    """
    print("=" * 70)
    print("练习 1: Wake-Sleep 算法 — Bernoulli 混合模型")
    print("=" * 70)

    # --- 模型设置 ---
    D = 10   # 数据维度
    K = 3    # 隐类别数

    # 真实参数（未知给算法）
    true_mu = np.array([[0.9, 0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.1, 0.9, 0.1],  # 类别0: 偏爱特征0,3,4,8
                         [0.1, 0.9, 0.9, 0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.9],  # 类别1: 偏爱特征1,2,5,6,9
                         [0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.1, 0.1, 0.5, 0.5]]) # 类别2: 混合

    true_pi = np.array([0.3, 0.4, 0.3])  # 先验 P(Z)

    # 生成数据
    N = 500
    true_z = np.zeros((N, K))
    X = np.zeros((N, D))
    for n in range(N):
        z = np.random.choice(K, p=true_pi)
        true_z[n, z] = 1
        X[n] = np.random.binomial(1, true_mu[z])

    print(f"\n  数据: N={N}, D={D}, K={K}")
    print(f"  真实 π = {true_pi}")
    print(f"  各类别活跃特征: 0=[0,3,4,8], 1=[1,2,5,6,9], 2=混合")

    # --- 初始化: 识别网络 Q(Z|X) = softmax(X·W + b) ---
    W = np.random.randn(D, K) * 0.1
    b = np.zeros(K)

    # 生成模型 P(X|Z): mu[k, d] = P(X_d=1 | Z=k)
    mu = np.random.uniform(0.3, 0.7, (K, D))
    pi = np.ones(K) / K

    n_epochs = 30
    print(f"\n  -- 训练 {n_epochs} 轮 --")
    print(f"  {'epoch':>5s}  {'wake_loss':>10s}  {'sleep_loss':>10s}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}")

    for epoch in range(n_epochs):
        # ===== Wake 阶段: 用真实数据训练生成模型 P(X|Z) =====
        # 1. 用识别网络推断 Z | X
        logits = X @ W + b
        q_z = np.exp(logits - logits.max(axis=1, keepdims=True))
        q_z /= q_z.sum(axis=1, keepdims=True)

        # 2. 用 Q(Z|X) 更新 mu (最大似然)
        # E-step 用 Q 的软分配
        q_z_safe = np.maximum(q_z, 1e-10)
        for k in range(K):
            mu[k] = np.sum(q_z_safe[:, k:k+1] * X, axis=0) / np.sum(q_z_safe[:, k])

        wake_loss = -np.sum(q_z * np.sum(X[:, None, :] * np.log(np.maximum(mu, 1e-10)) +
                                         (1 - X[:, None, :]) * np.log(np.maximum(1 - mu, 1e-10)),
                                         axis=2))

        # ===== Sleep 阶段: 用生成数据训练识别网络 Q(Z|X) =====
        # 1. 从先验采样 Z, 生成 X_fake
        z_sleep = np.random.choice(K, size=N, p=pi)
        X_fake = np.random.binomial(1, mu[z_sleep])
        z_sleep_oh = np.eye(K)[z_sleep]

        # 2. 用 (Z, X_fake) 训练识别网络 (交叉熵)
        logits_sleep = X_fake @ W + b
        q_sleep = np.exp(logits_sleep - logits_sleep.max(axis=1, keepdims=True))
        q_sleep /= q_sleep.sum(axis=1, keepdims=True)

        sleep_loss = -np.sum(z_sleep_oh * np.log(np.maximum(q_sleep, 1e-10)))

        # 3. 梯度更新 W, b (简单 SGD)
        grad_W = X_fake.T @ (q_sleep - z_sleep_oh) / N
        grad_b = np.mean(q_sleep - z_sleep_oh, axis=0)
        lr = 0.1
        W -= lr * grad_W
        b -= lr * grad_b

        # 更新 pi
        pi = q_z.mean(axis=0)

        if epoch < 5 or epoch == n_epochs - 1:
            print(f"  {epoch+1:>5d}  {wake_loss:>10.2f}  {sleep_loss:>10.2f}")

    # 评估
    print(f"\n  -- 最终结果 --")
    print(f"  学习到的 mu (vs 真实):")
    for k in range(K):
        # 找最匹配的真实类别
        best_j = np.argmax([np.sum(np.abs(mu[k] - true_mu[j])) for j in range(K)])
        print(f"    mu[{k}] (匹配 真实类别{best_j}): {np.round(mu[k], 2)}")
        print(f"    真实类别{best_j}:                  {true_mu[best_j]}")

    print(f"\n  学习到的 pi = {np.round(pi, 3)}, 真实 pi = {true_pi}")

    print("\n  🎯 洞察:")
    print("    Wake: 用'识别网络'指导'生成模型'参数更新")
    print("    Sleep: 用'生成模型'创造数据训练'识别网络'")
    print("    缺点: 两个阶段优化不同目标 — 不保证收敛到全局最优")


# ============================================================================
# 练习 2: SVI — 随机变分推断
# ============================================================================

def exercise2_svi_vs_cavi():
    """
    SVI (Stochastic VI) vs 全量 CAVI:
    在 Gaussian 均值估计问题上对比:
    - CAVI: 每轮用全部数据
    - SVI: 每轮用 mini-batch, 自然梯度 + Robbins-Monro 步长
    """
    print("=" * 70)
    print("练习 2: SVI vs CAVI — mini-batch 自然梯度")
    print("=" * 70)

    # --- 模型: P(X|mu) = N(mu, 1), P(mu) = N(0, 1) ---
    # 变分后验: Q(mu) = N(lambda_1, exp(lambda_2))
    # ELBO = E_Q[Σ log P(x_n|mu)] - KL(Q||P)

    # 真实参数
    true_mu = 2.5
    N = 10000  # 大数据集
    X = np.random.randn(N) + true_mu  # N(2.5, 1)

    # 初始化变分参数: Q = N(m, s^2), s = exp(log_s)
    m_cavi = 0.0
    log_s_cavi = 0.0  # s = 1

    m_svi = 0.0
    log_s_svi = 0.0

    batch_size = 100
    n_batches_per_epoch = N // batch_size  # 100
    n_epochs = 10
    total_batches = n_epochs * n_batches_per_epoch

    # 精确后验: P(mu|X) = N( (N*X_bar)/(N+1), 1/(N+1) )
    X_bar = X.mean()
    exact_m = N * X_bar / (N + 1)
    exact_s = np.sqrt(1.0 / (N + 1))

    print(f"\n  N={N}, true_mu={true_mu}, 精确后验: mu|X ~ N({exact_m:.4f}, {exact_s:.4f})")
    print(f"\n  -- CAVI (全量) vs SVI (mini-batch={batch_size}) --")
    print(f"  {'epoch':>5s}  {'CAVI_m':>10s}  {'CAVI_s':>10s}  {'SVI_m':>10s}  {'SVI_s':>10s}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    # 记录 SVI 结果用于绘图
    svi_m_trace = []

    for epoch in range(n_epochs):
        # --- CAVI: 全量更新 (共轭模型, 1步精确) ---
        # P(mu|X) = N(N*X_bar/(N+1), 1/(N+1))
        s2_cavi = 1.0 / (N + 1.0)
        s_cavi = np.sqrt(s2_cavi)
        m_cavi = s2_cavi * N * X_bar
        log_s_cavi = np.log(s_cavi)

        # --- SVI: mini-batch 自然梯度 + Robbins-Monro ---
        for b in range(n_batches_per_epoch):
            batch = X[b * batch_size: (b+1) * batch_size]
            M = len(batch)
            batch_bar = batch.mean()

            # Robbins-Monro step size
            t = epoch * n_batches_per_epoch + b + 1
            rho = (t + 10.0) ** (-0.7)

            # 自然梯度更新 (指数族, 带 scaling N/M)
            # eta_1 = m/s^2: prior=0, per-point contrib=x_i
            # eta_2 = -1/(2s^2): prior=-1/2, per-point contrib=-1/2
            # Mini-batch 估计 (缩放到全量):
            eta_1_batch = (N / M) * M * batch_bar   # = N * batch_bar
            eta_2_batch = -0.5 + (N / M) * M * (-0.5)  # = -(N+1)/2

            # 当前自然参数
            s2_cur = np.exp(2 * log_s_svi)
            eta_1_cur = m_svi / max(s2_cur, 1e-10)
            eta_2_cur = -0.5 / max(s2_cur, 1e-10)

            # RM 平均
            eta_1_new = (1 - rho) * eta_1_cur + rho * eta_1_batch
            eta_2_new = (1 - rho) * eta_2_cur + rho * eta_2_batch

            # 转回 moment 参数
            s2_new = -0.5 / eta_2_new
            m_svi = eta_1_new * s2_new
            log_s_svi = 0.5 * np.log(max(s2_new, 1e-10))

        svi_m_trace.append(m_svi)
        s_svi = np.exp(log_s_svi)

        if epoch < 5 or epoch == n_epochs - 1:
            print(f"  {epoch+1:>5d}  {m_cavi:>10.4f}  {s_cavi:>10.4f}  {m_svi:>10.4f}  {s_svi:>10.4f}")

    # 最终结果对比
    print(f"\n  -- 最终对比 --")
    print(f"  精确后验:   m = {exact_m:.4f}, s = {exact_s:.4f}")
    print(f"  CAVI (全量): m = {m_cavi:.4f}, s = {s_cavi:.4f}")
    print(f"  SVI (随机):  m = {m_svi:.4f}, s = {np.exp(log_s_svi):.4f}")
    c_err = abs(m_cavi - exact_m)
    s_err = abs(m_svi - exact_m)
    print(f"  CAVI 误差: {c_err:.6f}, SVI 误差: {s_err:.6f}")
    print(f"  SVI 只用 {batch_size}/{N} = {100*batch_size/N:.1f}% 数据/批次!")

    print("\n  🎯 洞察:")
    print("    CAVI 每轮需要全部数据 (N=10000) → O(N) per iteration")
    print("    SVI 每轮只用 mini-batch (100) → O(batch) per iteration")
    print("    自然梯度 + Robbins-Monro → SVI 收敛到正确附近!")


# ============================================================================
# 练习 3: Reparameterization Trick — 方差对比
# ============================================================================

def exercise3_reparameterization_trick():
    """
    对比 Score Function (REINFORCE) 和 Reparameterization 两种梯度估计器。

    任务: 估计 d/dμ E_{z~N(μ,σ²)}[z²]
    真实梯度 = 2μ

    比较两种估计器的方差。
    """
    print("=" * 70)
    print("练习 3: Reparameterization Trick — 梯度方差对比")
    print("=" * 70)

    mu_true = 2.0
    sigma_true = 0.5
    n_samples = 1000
    n_trials = 100  # 重复估计次数, 计算方差

    # 真实梯度: d/dμ E[z²] = 2μ = 4.0
    true_grad_mu = 2 * mu_true
    # 真实梯度: d/dσ E[z²] = 2σ = 1.0
    true_grad_sigma = 2 * sigma_true

    sf_grads_mu = np.zeros(n_trials)
    sf_grads_sigma = np.zeros(n_trials)
    rp_grads_mu = np.zeros(n_trials)
    rp_grads_sigma = np.zeros(n_trials)

    for trial in range(n_trials):
        # --- Score Function (REINFORCE) ---
        eps_sf = np.random.randn(n_samples)
        z_sf = mu_true + sigma_true * eps_sf
        f_z = z_sf**2

        # Score: ∇log N(z|μ,σ)
        score_mu = (z_sf - mu_true) / sigma_true**2
        score_sigma = ((z_sf - mu_true)**2 - sigma_true**2) / sigma_true**3

        sf_grads_mu[trial] = np.mean(f_z * score_mu)
        sf_grads_sigma[trial] = np.mean(f_z * score_sigma)

        # --- Reparameterization ---
        eps_rp = np.random.randn(n_samples)
        z_rp = mu_true + sigma_true * eps_rp

        # ∇_μ E[f(z)] = E[∇_z f · ∇_μ z] = E[2z · 1] = 2·E[z]
        rp_grads_mu[trial] = np.mean(2 * z_rp)

        # ∇_σ E[f(z)] = E[∇_z f · ∇_σ z] = E[2z · ε] = 2·E[z·ε]
        rp_grads_sigma[trial] = np.mean(2 * z_rp * eps_rp)

    print(f"\n  f(z) = z², z ~ N(μ={mu_true}, σ={sigma_true})")
    print(f"  真实梯度 d/dμ E[f] = {true_grad_mu}")
    print(f"  真实梯度 d/dσ E[f] = {true_grad_sigma}")

    print(f"\n  -- 梯度估计 (n_samples={n_samples}, n_trials={n_trials}) --")
    print(f"  {'方法':>25s}  {'均值':>12s}  {'偏差':>12s}  {'方差':>12s}  {'方差比':>10s}")
    print(f"  {'-'*25}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*10}")

    for name, grads, true_val in [
        ("Score Func (d/dμ)", sf_grads_mu, true_grad_mu),
        ("Reparam     (d/dμ)", rp_grads_mu, true_grad_mu),
        ("Score Func (d/dσ)", sf_grads_sigma, true_grad_sigma),
        ("Reparam     (d/dσ)", rp_grads_sigma, true_grad_sigma),
    ]:
        mean_est = np.mean(grads)
        bias = abs(mean_est - true_val)
        var = np.var(grads)
        ratio = ""
        if 'd/dμ' in name and 'Reparam' in name:
            ratio = f"{np.var(sf_grads_mu)/var:.0f}x"
        elif 'd/dσ' in name and 'Reparam' in name:
            ratio = f"{np.var(sf_grads_sigma)/var:.0f}x"
        print(f"  {name:>25s}  {mean_est:>12.4f}  {bias:>12.6f}  {var:>12.4f}  {ratio:>10s}")

    sf_var_mu = np.var(sf_grads_mu)
    rp_var_mu = np.var(rp_grads_mu)
    sf_var_sigma = np.var(sf_grads_sigma)
    rp_var_sigma = np.var(rp_grads_sigma)

    print(f"\n  📊 方差缩减:")
    print(f"    d/dμ: SF方差={sf_var_mu:.2f}, RP方差={rp_var_mu:.4f}, "
          f"缩减 {sf_var_mu/rp_var_mu:.0f}x")
    print(f"    d/dσ: SF方差={sf_var_sigma:.2f}, RP方差={rp_var_sigma:.4f}, "
          f"缩减 {sf_var_sigma/rp_var_sigma:.0f}x")

    print("\n  🎯 洞察:")
    print("    Score Function: f(z)*∇log Q — f(z) 的规模直接影响方差")
    print("    Reparameterization: ∇f * ∇g — 只依赖 f 的局部梯度, 方差极低")
    print("    这就是 VAE 能用 SGD 训练的原因 — Reparam 提供低方差梯度!")


# ============================================================================
# 练习 4: BBVI + Control Variate
# ============================================================================

def exercise4_bbvi_control_variate():
    """
    实现 BBVI 的 Score Function 梯度 + Control Variate 方差缩减。

    模型: P(X|Z) = N(Z, 1), Z ~ N(0, 1)
    变分后验: Q(Z) = N(lambda, 1), 估计 d/dλ ELBO

    BBVI:
      g_SF = (log P(X,Z) - log Q(Z)) * ∇_λ log Q(Z)

    Control Variate:
      g_CV = (log P(X,Z) - log Q(Z) - c) * ∇_λ log Q(Z)
      where c* = Cov(A, score) / Var(score)
    """
    print("=" * 70)
    print("练习 4: BBVI + Control Variate — 方差缩减")
    print("=" * 70)

    # 参数
    x_obs = 3.0  # 观测
    true_post_mu = x_obs / 2.0  # 精确后验: N(x/2, 1/2)

    # 变分参数: Q = N(lambda, 1)
    lam = 0.0

    n_iters = 200
    n_mc = 50  # Monte Carlo 样本数
    lr = 0.01

    # 用于计算 control variate 的 rolling statistics
    running_mean_sq_score = 0.0

    sf_trace = [lam]
    cv_trace = [lam]

    # 先跑 SF baseline
    lam_sf = 0.0
    for it in range(n_iters):
        z = lam_sf + np.random.randn(n_mc)  # Q(Z), sigma=1
        log_p = -0.5 * (x_obs - z)**2       # log N(x|z,1) 忽略常数
        log_q = -0.5 * (z - lam_sf)**2       # log Q(z) 忽略常数
        score = z - lam_sf                    # ∇_λ log Q = (z-λ)/1²

        # Score function gradient
        A = log_p - log_q
        grad_sf = np.mean(A * score)
        lam_sf -= lr * grad_sf
        sf_trace.append(lam_sf)

    # 再跑 CV 版本
    lam_cv = 0.0
    for it in range(n_iters):
        z = lam_cv + np.random.randn(n_mc)
        log_p = -0.5 * (x_obs - z)**2
        log_q = -0.5 * (z - lam_cv)**2
        score = z - lam_cv
        A = log_p - log_q

        # 最优 c* = Cov(A, score) / Var(score)
        var_score = np.var(score)
        cov_a_score = np.mean((A - np.mean(A)) * (score - np.mean(score)))
        c_star = cov_a_score / (var_score + 1e-10)

        grad_cv = np.mean((A - c_star) * score)
        lam_cv -= lr * grad_cv
        cv_trace.append(lam_cv)

    sf_final = sf_trace[-1]
    cv_final = cv_trace[-1]

    print(f"\n  观测 X = {x_obs}, 精确后验均值 = {true_post_mu:.4f}")
    print(f"\n  -- SF vs SF+CV 收敛对比 --")
    print(f"  SF (baseline) 最终 λ = {sf_final:.4f}, 误差 = {abs(sf_final - true_post_mu):.4f}")
    print(f"  SF + CV        最终 λ = {cv_final:.4f}, 误差 = {abs(cv_final - true_post_mu):.4f}")

    # 用最后 50 步的方差衡量收敛稳定性
    sf_last_var = np.var(sf_trace[-50:])
    cv_last_var = np.var(cv_trace[-50:])

    print(f"\n  最后 50 步的轨迹方差:")
    print(f"    SF baseline: {sf_last_var:.6f}")
    print(f"    SF + CV:     {cv_last_var:.6f}")
    if cv_last_var < sf_last_var:
        print(f"    CV 缩减方差 {(sf_last_var/cv_last_var):.1f}x ✅")
    else:
        print(f"    CV 未见缩减 (可能已充分收敛)")

    print("\n  🎯 洞察:")
    print("    Control Variate: 用 A - c* 替代 A, c* 吸收部分方差")
    print("    最优 c* = Cov(A, score)/Var(score) — 可用 MC 样本估计")
    print("    效果: 方差降低 → 收敛更稳定 → 可用更少 MC 样本")


# ============================================================================
# 练习 5: 简易 VAE — 纯 numpy
# ============================================================================

def exercise5_simple_vae():
    """
    用纯 numpy 实现一个极简 VAE, 在 1D 数据上演示:
    - Encoder: x → mu(x), log_sigma(x)  [MLP]
    - Reparameterization: z = mu + sigma * eps
    - Decoder: z → x_hat  [MLP]
    - ELBO = reconstruction - KL divergence

    数据: 来自两个高斯混合的真实分布
    """
    print("=" * 70)
    print("练习 5: 简易 VAE — 纯 numpy 编码器-解码器")
    print("=" * 70)

    # --- 生成 1D 数据 (两个模式) ---
    N = 2000
    # 模式1: N(-2, 0.5), 模式2: N(2, 0.5)
    z_mode = np.random.binomial(1, 0.5, N)
    X_raw = np.where(z_mode, np.random.randn(N) * 0.5 + 2.0,
                     np.random.randn(N) * 0.5 - 2.0)
    # 归一化到 [-1, 1] 区间
    X = X_raw / 3.0
    X = X.reshape(-1, 1)

    print(f"\n  数据: N={N}, 两个模式 (N(-2,0.5) 和 N(2,0.5))")
    print(f"  输入维度: 1, 隐变量维度: 2")

    # --- VAE 参数 ---
    D_in = 1
    D_hidden = 16
    D_latent = 2  # 隐变量维度

    # Encoder: 1 → 16 → (2+2) [mu and log_var]
    W_enc1 = np.random.randn(D_in, D_hidden) * np.sqrt(2.0 / D_in)
    b_enc1 = np.zeros(D_hidden)
    W_enc_mu = np.random.randn(D_hidden, D_latent) * np.sqrt(2.0 / D_hidden)
    b_enc_mu = np.zeros(D_latent)
    W_enc_logvar = np.random.randn(D_hidden, D_latent) * np.sqrt(2.0 / D_hidden)
    b_enc_logvar = np.zeros(D_latent)

    # Decoder: 2 → 16 → 1
    W_dec1 = np.random.randn(D_latent, D_hidden) * np.sqrt(2.0 / D_latent)
    b_dec1 = np.zeros(D_hidden)
    W_dec_out = np.random.randn(D_hidden, D_in) * np.sqrt(2.0 / D_hidden)
    b_dec_out = np.zeros(D_in)

    # --- 训练 ---
    n_epochs = 100
    batch_size = 128
    n_batches = N // batch_size
    lr = 0.001

    def relu(x):
        return np.maximum(x, 0)

    print(f"\n  -- 训练 {n_epochs} epochs --")
    print(f"  {'epoch':>6s}  {'ELBO':>10s}  {'recon':>10s}  {'KL':>10s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}")

    for epoch in range(n_epochs):
        # Shuffle
        idx = np.random.permutation(N)
        X_shuf = X[idx]

        total_elbo = 0
        total_recon = 0
        total_kl = 0

        for b in range(n_batches):
            x_batch = X_shuf[b * batch_size: (b+1) * batch_size]
            M = x_batch.shape[0]

            # --- Encoder ---
            h_enc = relu(x_batch @ W_enc1 + b_enc1)
            mu = h_enc @ W_enc_mu + b_enc_mu
            log_var = h_enc @ W_enc_logvar + b_enc_logvar
            sigma = np.exp(0.5 * log_var)

            # --- Reparameterization ---
            eps = np.random.randn(M, D_latent)
            z = mu + sigma * eps

            # --- Decoder ---
            h_dec = relu(z @ W_dec1 + b_dec1)
            x_recon = h_dec @ W_dec_out + b_dec_out

            # --- Loss ---
            # Reconstruction: Gaussian log-likelihood (MSE)
            recon_loss = 0.5 * np.mean(np.sum((x_batch - x_recon)**2, axis=1))

            # KL: KL(N(mu,sigma²) || N(0,I)) = 0.5 * Σ(sigma² + mu² - 1 - log(sigma²))
            kl_loss = 0.5 * np.mean(np.sum(sigma**2 + mu**2 - 1 - log_var, axis=1))

            elbo = -(recon_loss + kl_loss)
            loss = -elbo

            total_elbo += elbo
            total_recon += recon_loss
            total_kl += kl_loss

            # --- Backprop (手动) ---
            # dLoss/dRecon
            d_recon = (x_recon - x_batch) / M

            # dLoss/dKL
            d_mu_kl = mu / M
            d_logvar_kl = 0.5 * (np.exp(log_var) - 1) / M

            # Gradient through decoder
            d_h_dec = d_recon @ W_dec_out.T
            d_h_dec[h_dec <= 0] = 0
            d_W_dec_out = h_dec.T @ d_recon
            d_b_dec_out = np.sum(d_recon, axis=0)
            d_W_dec1 = z.T @ d_h_dec
            d_b_dec1 = np.sum(d_h_dec, axis=0)
            d_z = d_h_dec @ W_dec1.T

            # Gradient through reparameterization
            d_mu_recon = d_z
            d_logvar_recon = d_z * eps * 0.5 * np.exp(0.5 * log_var)

            d_mu = d_mu_recon + d_mu_kl
            d_logvar = d_logvar_recon + d_logvar_kl

            # Gradient through encoder
            d_h_enc_mu = d_mu @ W_enc_mu.T
            d_h_enc_logvar = d_logvar @ W_enc_logvar.T
            d_h_enc = d_h_enc_mu + d_h_enc_logvar
            d_h_enc[h_enc <= 0] = 0

            d_W_enc1 = x_batch.T @ d_h_enc
            d_b_enc1 = np.sum(d_h_enc, axis=0)
            d_W_enc_mu = h_enc.T @ d_mu
            d_b_enc_mu = np.sum(d_mu, axis=0)
            d_W_enc_logvar = h_enc.T @ d_logvar
            d_b_enc_logvar = np.sum(d_logvar, axis=0)

            # SGD update
            for param, grad in [
                (W_enc1, d_W_enc1), (b_enc1, d_b_enc1),
                (W_enc_mu, d_W_enc_mu), (b_enc_mu, d_b_enc_mu),
                (W_enc_logvar, d_W_enc_logvar), (b_enc_logvar, d_b_enc_logvar),
                (W_dec1, d_W_dec1), (b_dec1, d_b_dec1),
                (W_dec_out, d_W_dec_out), (b_dec_out, d_b_dec_out),
            ]:
                # gradient clipping
                grad_clipped = np.clip(grad, -5, 5)
                param[...] = param - lr * grad_clipped

        avg_elbo = total_elbo / n_batches
        avg_recon = total_recon / n_batches
        avg_kl = total_kl / n_batches

        if epoch < 5 or epoch % 20 == 0 or epoch == n_epochs - 1:
            print(f"  {epoch+1:>6d}  {avg_elbo:>10.4f}  {avg_recon:>10.4f}  {avg_kl:>10.4f}")

    # --- 生成新样本 ---
    print(f"\n  -- 从 VAE 生成新样本 --")
    n_gen = 1000
    z_gen = np.random.randn(n_gen, D_latent)
    h_gen = relu(z_gen @ W_dec1 + b_dec1)
    x_gen = (h_gen @ W_dec_out + b_dec_out).flatten() * 3.0  # 反归一化

    print(f"  生成样本均值: {np.mean(x_gen):.4f} (真实≈0)")
    print(f"  生成样本标准差: {np.std(x_gen):.4f} (真实≈2.1)")
    print(f"  真实数据两模式: 约在 -2 和 +2")

    # 检查是否生成了两个模式
    mode1_mask = x_gen < 0
    mode2_mask = x_gen > 0
    n_mode1 = np.sum(mode1_mask)
    n_mode2 = np.sum(mode2_mask)
    if n_mode1 > 100 and n_mode2 > 100:
        mean1 = np.mean(x_gen[mode1_mask])
        mean2 = np.mean(x_gen[mode2_mask])
        print(f"  模式1 (负侧): n={n_mode1}, mean={mean1:.3f}")
        print(f"  模式2 (正侧): n={n_mode2}, mean={mean2:.3f}")
        print(f"  ✅ VAE 成功捕获了两个数据模式!")
    else:
        print(f"  ⚠️ 模式分离不明显 (正常 — 简单模型/少迭代)")

    print("\n  🎯 洞察:")
    print("    Encoder (amortized inference): x → μ(x), σ(x) — 学推断!")
    print("    Reparameterization: z = μ + σ·ε — 使梯度可传播")
    print("    Decoder: z → x̂ — 学生成")
    print("    ELBO = -E[||x-x̂||²] - KL(Q||P(z)) — 重建 + 正则")
    print("    这就是 VAE 的全部核心 — 用 numpy 也能实现!")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    run_all = '--ex' not in sys.argv

    exercises = [
        ('1', exercise1_wake_sleep, False),
        ('2', exercise2_svi_vs_cavi, False),
        ('3', exercise3_reparameterization_trick, False),
        ('4', exercise4_bbvi_control_variate, False),
        ('5', exercise5_simple_vae, False),
    ]

    for ex_id, ex_func, _ in exercises:
        if not run_all and ex_id not in sys.argv:
            continue
        try:
            ex_func()
        except Exception as e:
            print(f"\n  [!] 练习{ex_id}执行出错: {e}")
            import traceback
            traceback.print_exc()
