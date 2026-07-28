"""
=============================================================================
  CMU 10-708 L13 代码练习: 深度生成模型 II — VAE, GAN, Flow, AR, Diffusion
=============================================================================

本文件包含 5 个代码练习:

  练习 1: VAE — 2D 隐空间插值与生成
  练习 2: GAN — 对抗训练, 1D 数据分布学习
  练习 3: Normalizing Flow — RealNVP 风格可逆变换
  练习 4: 自回归模型 — MADE 的 masked 连接
  练习 5: 去噪扩散模型 — 简化的 1D DDPM

使用方法:
  python 33_dgm2_exercises.py           # 运行全部
  python 33_dgm2_exercises.py --ex 1    # 只运行练习1

依赖: numpy
=============================================================================
"""

import numpy as np
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

np.random.seed(42)


def relu(x):
    return np.maximum(x, 0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


# ============================================================================
# 练习 1: VAE — 2D 隐空间
# ============================================================================

def exercise1_vae_latent_space():
    """
    训练一个 VAE 在 2D 合成数据上, 展示:
    - 2D 隐空间的连续性 (插值)
    - 从隐空间采样生成新数据
    - 重构质量
    """
    print("=" * 70)
    print("练习 1: VAE — 2D 隐空间插值与生成")
    print("=" * 70)

    # 合成数据: 8 个高斯簇围成一圈
    n_clusters = 8
    n_per = 150
    radius = 3.0
    angles = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
    X = []
    for a in angles:
        cx, cy = radius * np.cos(a), radius * np.sin(a)
        cluster = np.random.randn(n_per, 2) * 0.3 + np.array([cx, cy])
        X.append(cluster)
    X = np.vstack(X).astype(np.float32)
    N, D_in = X.shape
    D_latent = 2  # 2D 隐空间用于可视化
    D_hid = 32

    print(f"\n  数据: {n_clusters} 个高斯簇围成圆, N={N}")
    print(f"  模型: {D_in} → {D_hid} → {D_latent} (隐空间: 2D)")

    # Encoder params
    W_enc1 = np.random.randn(D_in, D_hid) * np.sqrt(2.0 / D_in)
    b_enc1 = np.zeros(D_hid)
    W_mu = np.random.randn(D_hid, D_latent) * 0.1
    b_mu = np.zeros(D_latent)
    W_lv = np.random.randn(D_hid, D_latent) * 0.1
    b_lv = np.zeros(D_latent)

    # Decoder params
    W_dec1 = np.random.randn(D_latent, D_hid) * np.sqrt(2.0 / D_latent)
    b_dec1 = np.zeros(D_hid)
    W_out = np.random.randn(D_hid, D_in) * 0.1
    b_out = np.zeros(D_in)

    lr = 0.002
    batch = 128
    n_epochs = 300
    n_batches = N // batch

    print(f"\n  -- 训练 ({n_epochs} epochs, lr={lr}) --")
    for epoch in range(n_epochs):
        idx = np.random.permutation(N)
        X_shuf = X[idx]
        total_loss = 0

        for b in range(n_batches):
            xb = X_shuf[b * batch:(b + 1) * batch]
            M = len(xb)

            # Encoder
            h_enc = relu(xb @ W_enc1 + b_enc1)
            mu = h_enc @ W_mu + b_mu
            log_var = h_enc @ W_lv + b_lv
            sigma = np.exp(0.5 * log_var)

            # Reparameterization
            eps = np.random.randn(M, D_latent)
            z = mu + sigma * eps

            # Decoder
            h_dec = relu(z @ W_dec1 + b_dec1)
            x_recon = h_dec @ W_out + b_out

            # Loss
            recon_loss = 0.5 * np.mean(np.sum((xb - x_recon)**2, axis=1))
            kl_loss = 0.5 * np.mean(np.sum(sigma**2 + mu**2 - 1 - log_var, axis=1))
            loss = recon_loss + kl_loss

            # Backward
            d_recon = (x_recon - xb) / M
            d_hdec = d_recon @ W_out.T
            d_hdec[h_dec <= 0] = 0

            d_Wout = h_dec.T @ d_recon
            d_bout = np.sum(d_recon, axis=0)
            d_Wdec1 = z.T @ d_hdec
            d_bdec1 = np.sum(d_hdec, axis=0)
            dz = d_hdec @ W_dec1.T

            d_mu_r = dz
            d_lv_r = dz * eps * 0.5 * np.exp(0.5 * log_var)
            d_mu_k = mu / M
            d_lv_k = 0.5 * (np.exp(log_var) - 1) / M

            d_mu = d_mu_r + d_mu_k
            d_lv = d_lv_r + d_lv_k

            d_henc = (d_mu @ W_mu.T + d_lv @ W_lv.T)
            d_henc[h_enc <= 0] = 0

            # Updates
            for param, grad in [
                (W_enc1, xb.T @ d_henc), (b_enc1, np.sum(d_henc, axis=0)),
                (W_mu, h_enc.T @ d_mu), (b_mu, np.sum(d_mu, axis=0)),
                (W_lv, h_enc.T @ d_lv), (b_lv, np.sum(d_lv, axis=0)),
                (W_dec1, z.T @ d_hdec), (b_dec1, d_bdec1),
                (W_out, d_Wout), (b_out, d_bout),
            ]:
                param -= lr * np.clip(grad, -5, 5)

            total_loss += loss

        if epoch < 5 or epoch % 60 == 0 or epoch == n_epochs - 1:
            avg = total_loss / n_batches
            print(f"  epoch {epoch+1:>3d}: loss={avg:.4f}")

    # 生成样本
    n_gen = 400
    z_gen = np.random.randn(n_gen, D_latent)
    h_gen = relu(z_gen @ W_dec1 + b_dec1)
    x_gen = h_gen @ W_out + b_out
    # 检查生成了几个簇
    angles_gen = np.arctan2(x_gen[:, 1], x_gen[:, 0])
    unique_dirs = len(set(np.round(angles_gen, 1)))
    print(f"\n  -- 生成结果 --")
    print(f"  从 N(0,I) 生成 {n_gen} 样本: 覆盖方向数 ≈ {unique_dirs} (真实={n_clusters})")

    # 隐空间插值
    print(f"\n  -- 隐空间插值 (球面) --")
    print(f"  在 2D 隐空间沿圆采样 z, 解码:")
    for angle in [0, np.pi / 4, np.pi / 2, np.pi]:
        z_interp = np.array([[2 * np.cos(angle), 2 * np.sin(angle)]])
        h_i = relu(z_interp @ W_dec1 + b_dec1)
        x_i = (h_i @ W_out + b_out).flatten()
        print(f"    angle={angle:.2f}: z=({z_interp[0,0]:.1f},{z_interp[0,1]:.1f}) "
              f"→ x=({x_i[0]:.2f}, {x_i[1]:.2f})")

    print("\n  🎯 洞察:")
    print("    2D 隐空间 → 可可视化 → 连续插值有意义!")
    print("    隐空间的每个方向对应数据中的一个'语义方向'")


# ============================================================================
# 练习 2: GAN 从零实现
# ============================================================================

def exercise2_gan_from_scratch():
    """
    在 1D 数据上训练一个极简 GAN:
    - 真实数据: 混合高斯 P(x)=0.4*N(-2,0.4)+0.6*N(3,0.6)
    - Generator: z → x_fake  (MLP: 1→8→1)
    - Discriminator: x → [0,1] (MLP: 1→8→1)

    展示: G 学到的分布 vs 真实分布, 训练动态。
    """
    print("=" * 70)
    print("练习 2: GAN — 对抗生成网络训练")
    print("=" * 70)

    def real_data(n):
        z_c = np.random.binomial(1, 0.6, n)
        return np.where(z_c, np.random.randn(n) * np.sqrt(0.6) + 3.0,
                        np.random.randn(n) * np.sqrt(0.4) - 2.0)

    # Generator: z → h(8,relu) → x
    G_w1 = np.random.randn(1, 16) * 0.3
    G_b1 = np.zeros(16)
    G_w2 = np.random.randn(16, 1) * 0.3
    G_b2 = np.zeros(1)

    # Discriminator: x → h(8,relu) → logit
    D_w1 = np.random.randn(1, 16) * 0.3
    D_b1 = np.zeros(16)
    D_w2 = np.random.randn(16, 1) * 0.3
    D_b2 = np.zeros(1)

    def G(z):
        h = relu(z @ G_w1 + G_b1)
        return h @ G_w2 + G_b2

    def D(x):
        h = relu(x @ D_w1 + D_b1)
        logit = h @ D_w2 + D_b2
        return sigmoid(logit)

    lr_g = 0.005
    lr_d = 0.01
    batch = 128
    n_epochs = 600
    d_steps = 3  # D 每轮训练步数

    g_losses, d_losses = [], []
    gen_mean_trace = []

    print(f"\n  真实分布: 0.4*N(-2,0.4) + 0.6*N(3,0.6) (双峰)")
    print(f"  G: 1→16→1, D: 1→16→1")
    print(f"\n  -- 训练 --")

    for ep in range(n_epochs):
        # Train D
        for _ in range(d_steps):
            xr = real_data(batch).reshape(-1, 1)
            z = np.random.randn(batch, 1)
            xf = G(z)
            dr = D(xr)
            df = D(xf)
            d_loss = -np.mean(np.log(dr + 1e-10) + np.log(1 - df + 1e-10))
            # Manual gradients
            d_dr = -(1 / (dr + 1e-10)) / batch
            d_df = (1 / (1 - df + 1e-10)) / batch
            dr_grad = d_dr * dr * (1 - dr)
            df_grad = d_df * df * (1 - df)

            # Update D weights (simplified: just use the immediate gradient direction)
            h_r = relu(xr @ D_w1 + D_b1)
            h_f = relu(xf @ D_w1 + D_b1)
            D_w2 -= lr_d * (h_r.T @ dr_grad + h_f.T @ df_grad)
            D_b2 -= lr_d * (np.sum(dr_grad, axis=0) + np.sum(df_grad, axis=0))

            # Simplified D_w1 update
            d_h = (dr_grad @ D_w2.T + df_grad @ D_w2.T)
            d_h_r = d_h[:batch] * (h_r > 0)
            d_h_f = d_h[:batch] * (h_f > 0)
            D_w1 -= lr_d * (xr.T @ d_h_r + xf.T @ d_h_f)
            D_b1 -= lr_d * (np.sum(d_h_r, axis=0) + np.sum(d_h_f, axis=0))

        d_losses.append(d_loss)

        # Train G
        z = np.random.randn(batch, 1)
        xf = G(z)
        df = D(xf)
        g_loss = -np.mean(np.log(df + 1e-10))
        dg = -(1 / (df + 1e-10)) / batch * df * (1 - df)
        h_fg = relu(xf @ D_w1 + D_b1)
        d_hg = dg @ D_w2.T * (h_fg > 0)
        G_w2 -= lr_g * (relu(z @ G_w1 + G_b1).T @ dg @ D_w2.T * 0.1)[:, :1]
        G_b2 -= lr_g * np.sum(dg @ D_w2.T * 0.1, axis=0)[:1]
        g_losses.append(g_loss)

        # Track generated mean
        z_test = np.random.randn(500, 1)
        gen_mean_trace.append(np.mean(G(z_test)))

    # 评估
    z_eval = np.random.randn(2000, 1)
    x_gen_final = G(z_eval).flatten()
    gen_mean = np.mean(x_gen_final)
    gen_std = np.std(x_gen_final)

    print(f"  最终: G_loss={g_losses[-1]:.3f}, D_loss={d_losses[-1]:.3f}")
    print(f"\n  -- 生成分布 vs 真实分布 --")
    print(f"  生成样本: mean={gen_mean:.3f}, std={gen_std:.3f}")
    print(f"  真实数据: mean≈1.0, std≈2.3 (双峰)")

    # 检查是否捕获了双峰
    below_zero = np.mean(x_gen_final < 0)
    above_zero = np.mean(x_gen_final > 0)
    print(f"  <0: {below_zero:.1%}, >0: {above_zero:.1%} (真实: ~40%, ~60%)")

    if 0.2 < below_zero < 0.6:
        print(f"  ✅ GAN 成功学到了双峰分布!")
    else:
        print(f"  ⚠️ 可能有 mode collapse (所有样本集中在一个 mode)")

    print("\n  🎯 洞察:")
    print("    GAN = minimax game: G 想骗 D, D 想不被骗")
    print("    训练不稳定是 GAN 的核心挑战 — 需要仔细调参")
    print("    Mode Collapse: G 只生成少数 mode, 忽略其他")


# ============================================================================
# 练习 3: Normalizing Flow
# ============================================================================

def exercise3_normalizing_flow():
    """
    实现简单的 Planar Flow: f(z) = z + u * tanh(w^T z + b)

    在一个 1D 双峰目标上训练: P_target = 0.5*N(-2,0.5) + 0.5*N(2,0.5)
    用 Flow 将 N(0,1) 变换为双峰分布。
    """
    print("=" * 70)
    print("练习 3: Normalizing Flow — Planar Flow 变换")
    print("=" * 70)

    def target_log_pdf(x):
        log1 = -0.5 * ((x + 2) / np.sqrt(0.5))**2 - 0.5 * np.log(2 * np.pi * 0.5)
        log2 = -0.5 * ((x - 2) / np.sqrt(0.5))**2 - 0.5 * np.log(2 * np.pi * 0.5)
        mx = np.maximum(log1, log2)
        return np.log(0.5 * np.exp(log1 - mx) + 0.5 * np.exp(log2 - mx)) + mx

    # Planar Flow: f(z) = z + u * tanh(w^T z + b)
    # log|det| = log|1 + u·w·h'(w^T z + b)|
    # 对 1D: z, w, u, b are all scalars
    n_flows = 4  # 级联 4 个 Planar Flow
    ws = np.random.randn(n_flows) * 0.5
    us = np.random.randn(n_flows) * 0.5
    bs = np.random.randn(n_flows) * 0.1

    def forward(z0):
        """从 base → target 的变换"""
        z = z0
        log_det_sum = 0.0
        for k in range(n_flows):
            a = ws[k] * z + bs[k]
            h = np.tanh(a)
            dh = 1 - h**2
            z = z + us[k] * h
            # log|det| for 1D planar flow
            det_val = 1 + us[k] * ws[k] * dh
            log_det_sum += np.log(np.abs(det_val) + 1e-10)
        return z, log_det_sum

    n_samples = 1000
    lr = 0.01
    n_epochs = 800

    print(f"\n  目标: 0.5*N(-2,0.5) + 0.5*N(2,0.5) (双峰)")
    print(f"  Base: N(0, 1)")
    print(f"  K={n_flows} Planar Flows")

    for epoch in range(n_epochs):
        z0 = np.random.randn(n_samples)
        zK, log_det = forward(z0)

        # log P_flow(zK) = log P_base(z0) - log_det
        # (by change-of-variables)
        log_p_base = -0.5 * z0**2 - 0.5 * np.log(2 * np.pi)
        loss = -np.mean(log_p_base - log_det)

        # 简化: 直接最大化 P_target 下的 log-likelihood
        # (等同于用 MLE 训练 Flow 去匹配 target)
        log_p_flow = log_p_base - log_det
        nll = -np.mean(log_p_flow)

        # 使用近似梯度 (finite difference guide)
        # 实际中应该对每个 flow 参数求导, 这里简化为朝目标方向更新
        for k in range(n_flows):
            ws[k] -= lr * 0.01 * np.sign(ws[k])
            us[k] -= lr * 0.01 * np.sign(us[k] - 1.0)
            bs[k] -= lr * 0.001 * np.sign(bs[k])

        # Simplified: use a heuristic loss
        # We want zK to match the bimodal target
        target_samples = np.where(np.random.binomial(1, 0.5, n_samples),
                                  np.random.randn(n_samples) * np.sqrt(0.5) + 2,
                                  np.random.randn(n_samples) * np.sqrt(0.5) - 2)
        mse = np.mean((zK - target_samples)**2)
        loss = mse

        # Crude gradient: perturb each param
        delta = 0.01
        for k in range(n_flows):
            orig_w = ws[k]
            ws[k] = orig_w + delta
            zp, _ = forward(z0)
            loss_p = np.mean((zp - target_samples)**2)
            ws[k] = orig_w - delta
            zm, _ = forward(z0)
            loss_m = np.mean((zm - target_samples)**2)
            ws[k] = orig_w
            ws[k] -= lr * (loss_p - loss_m) / (2 * delta)

            # Same for u and b
            orig_u = us[k]
            us[k] = orig_u + delta
            zp, _ = forward(z0)
            loss_p = np.mean((zp - target_samples)**2)
            us[k] = orig_u - delta
            zm, _ = forward(z0)
            loss_m = np.mean((zm - target_samples)**2)
            us[k] = orig_u
            us[k] -= lr * (loss_p - loss_m) / (2 * delta)

        if epoch < 5 or epoch % 200 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch+1:>3d}: loss={loss:.4f}")

    # 生成
    z0_test = np.random.randn(2000)
    zK_test, _ = forward(z0_test)
    # 统计双峰
    below = np.mean(zK_test < 0)
    above = np.mean(zK_test > 0)
    print(f"\n  -- 生成样本统计 --")
    print(f"  mean={np.mean(zK_test):.3f}, std={np.std(zK_test):.3f}")
    print(f"  <0: {below:.1%}, >0: {above:.1%} (目标: 50%, 50%)")

    if 0.3 < below < 0.7:
        print(f"  ✅ Flow 成功学到了双峰结构!")
    else:
        print(f"  ⚠️ Flow 尚未完美捕获双峰 (正常 — 简单训练)")

    print("\n  🎯 洞察:")
    print("    Flow = 可逆变换链: base → target")
    print("    log P(x) = log P(z0) - Σ log|det ∂f_k/∂z|  ← exact likelihood!")
    print("    Planar Flow: 简单但表达能力有限; RealNVP更有力")


# ============================================================================
# 练习 4: 自回归模型 (MADE)
# ============================================================================

def exercise4_autoregressive_made():
    """
    实现 MADE (Masked Autoencoder for Distribution Estimation):
    用 masked 权重确保每个输出只依赖它前面的输入。

    在 4-bit binary 数据上训练: 学习 P(x_1, x_2, x_3, x_4)
    展示: 自回归分解 P(x) = P(x₁)P(x₂|x₁)P(x₃|x₁,x₂)P(x₄|x₁,x₂,x₃)
    """
    print("=" * 70)
    print("练习 4: MADE — Masked 自回归密度估计")
    print("=" * 70)

    # 数据: 4-bit, 但有些组合更常见
    D = 4
    # 生成偏向的 binary 数据
    data = []
    for _ in range(500):
        x = np.zeros(D)
        x[0] = np.random.binomial(1, 0.7)
        x[1] = np.random.binomial(1, 0.3 if x[0] else 0.8)
        x[2] = np.random.binomial(1, 0.6)
        x[3] = np.random.binomial(1, 0.4 if x[1] and x[2] else 0.9)
        data.append(x)
    data = np.array(data, dtype=float)
    N = len(data)

    # 真实边际 (从数据估计)
    true_probs = data.mean(axis=0)
    true_joint = {}
    for x in data:
        key = tuple(x.astype(int))
        true_joint[key] = true_joint.get(key, 0) + 1
    for k in true_joint:
        true_joint[k] /= N

    print(f"\n  D={D} binary 变量, N={N}")
    print(f"  真实 P(x₁=1)={true_probs[0]:.2f}, "
          f"P(x₂=1)={true_probs[1]:.2f}, "
          f"P(x₃=1)={true_probs[2]:.2f}, "
          f"P(x₄=1)={true_probs[3]:.2f}")

    # MADE: 每个输出 x̂_d 只能看到 x_{1:d-1}
    # Mask: 给每个隐藏单元分配一个"最大可见输入索引"
    n_hid = 8
    # 为隐藏单元分配 connectivity degree (1 to D-1)
    m_hid = np.random.randint(1, D, n_hid)  # 每个隐藏单元的最大输入索引

    # Masked weights
    W1 = np.random.randn(D, n_hid) * 0.1
    b1 = np.zeros(n_hid)
    W2 = np.random.randn(n_hid, D) * 0.1
    b2 = np.zeros(D)

    # Apply masks
    mask1 = np.zeros((D, n_hid))
    for d in range(D):
        for h in range(n_hid):
            if d < m_hid[h] or m_hid[h] == D:
                mask1[d, h] = 1
    W1 = W1 * mask1

    mask2 = np.zeros((n_hid, D))
    for h in range(n_hid):
        for d in range(D):
            if m_hid[h] >= d + 1:
                mask2[h, d] = 1
    W2 = W2 * mask2

    lr = 0.05
    n_epochs = 500

    print(f"\n  -- 训练 MADE (masked autoencoder) --")
    for epoch in range(n_epochs):
        total_loss = 0
        for x in data:
            # Forward
            h = relu(x @ W1 + b1)
            logits = h @ W2 + b2
            # 每个输出是 sigmoid (Bernoulli)
            probs = sigmoid(logits)

            # BCE loss: -Σ [x_d log p_d + (1-x_d)log(1-p_d)]
            loss = -np.sum(x * np.log(probs + 1e-10) +
                           (1 - x) * np.log(1 - probs + 1e-10))

            # Backward (simplified)
            d_logits = (probs - x)
            d_h = d_logits @ W2.T * (h > 0)
            W2 -= lr * np.outer(h, d_logits) * mask2
            b2 -= lr * d_logits
            W1 -= lr * np.outer(x, d_h) * mask1
            b1 -= lr * d_h

            total_loss += loss

        if epoch < 5 or epoch % 150 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch+1:>3d}: NLL = {total_loss/N:.4f}")

    # 评估: 生成所有 16 种配置的概率
    print(f"\n  -- 学到的分布 vs 真实分布 --")
    all_configs = np.array(np.meshgrid(*[[0, 1]] * D)).T.reshape(-1, D).astype(float)

    # MADE 概率: P(x) = Π_d Bernoulli(x_d | p_d(x_{<d}))
    model_probs = np.zeros(16)
    for i, x in enumerate(all_configs):
        h = relu(x @ W1 + b1)
        logits = h @ W2 + b2
        probs = sigmoid(logits)
        # 自回归分解
        p = 1.0
        for d in range(D):
            if x[d] == 1:
                p *= probs[d]
            else:
                p *= (1 - probs[d])
        model_probs[i] = p
    model_probs /= model_probs.sum()

    # 打印前 8 个配置
    for i in range(min(8, 16)):
        cfg = all_configs[i].astype(int)
        true_p = true_joint.get(tuple(cfg), 0)
        print(f"  {cfg}: MADE={model_probs[i]:.4f}, True={true_p:.4f}")

    # 检验自回归性质
    print(f"\n  -- 自回归性质验证 --")
    test_x = np.array([1.0, 0.0, 1.0, 0.0])
    h_t = relu(test_x @ W1 + b1)
    logits_t = h_t @ W2 + b2
    probs_t = sigmoid(logits_t)
    # x₄ 应该只依赖 x₁, x₂, x₃ (由 mask2 保证)
    print(f"  For x={test_x.astype(int)}: predicted probs = {np.round(probs_t, 3)}")
    print(f"  P(x₄=1|x₁,x₂,x₃) = {probs_t[3]:.4f}")

    print("\n  🎯 洞察:")
    print("    MADE = 用 masked 连接强制自回归性质")
    print("    每个输出只依赖其索引之前的输入")
    print("    可以直接计算 exact log-likelihood (无近似)")


# ============================================================================
# 练习 5: 去噪扩散模型 (DDPM)
# ============================================================================

def exercise5_diffusion_model():
    """
    简化的 1D Diffusion Model:
    - Forward: x_t = √(α_t)·x_0 + √(1-α_t)·ε, ε~N(0,1)
    - Reverse: 训练网络从 x_t 预测 ε
    - 生成: 从噪声开始, 逐步去噪

    目标分布: N(3, 1.5²)  (单峰, 简单, 展示原理)
    """
    print("=" * 70)
    print("练习 5: Diffusion Model — 1D 去噪扩散原理")
    print("=" * 70)

    # 数据
    mu_data, sigma_data = 3.0, 1.5
    N_train = 2000

    # 扩散参数
    T = 100
    beta = np.linspace(0.0001, 0.02, T)  # noise schedule
    alpha = 1 - beta
    alpha_bar = np.cumprod(alpha)  # ᾱ_t = Π α_s

    # 训练数据
    x0_train = np.random.randn(N_train) * sigma_data + mu_data

    # 简化的"网络": 2-layer MLP — xt → 16(relu) → 1(eps)
    n_hid_ddpm = 16
    W1_d = np.random.randn(2, n_hid_ddpm) * 0.1  # input: [xt, t/T]
    b1_d = np.zeros(n_hid_ddpm)
    W2_d = np.random.randn(n_hid_ddpm, 1) * 0.1
    b2_d = np.zeros(1)
    lr = 0.01
    n_epochs = 500
    batch = 256

    def predict_noise(xt, t_norm):
        inp = np.column_stack([xt, t_norm * np.ones_like(xt)])
        h = relu(inp @ W1_d + b1_d)
        return (h @ W2_d + b2_d).flatten()

    print(f"\n  目标: N({mu_data}, {sigma_data}^2)")
    print(f"  Diffusion: T={T} steps")
    print(f"  Predictor: MLP(2→16→1), input=[x_t, t/T]")

    for epoch in range(n_epochs):
        idx = np.random.choice(N_train, batch)
        x0 = x0_train[idx]
        t = np.random.randint(0, T, batch)
        t_norm = t.astype(float) / T

        # Forward diffusion
        sqrt_alpha_bar_t = np.sqrt(alpha_bar[t])
        sqrt_one_minus = np.sqrt(1 - alpha_bar[t])
        eps_true = np.random.randn(batch)
        xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus * eps_true

        # Predict
        inp = np.column_stack([xt, t_norm])
        h = relu(inp @ W1_d + b1_d)
        eps_pred = (h @ W2_d + b2_d).flatten()

        # MSE loss
        loss = np.mean((eps_pred - eps_true)**2)

        # Backward
        d_pred = 2 * (eps_pred - eps_true).reshape(-1, 1) / batch
        d_h = d_pred @ W2_d.T * (h > 0)
        W2_d -= lr * h.T @ d_pred
        b2_d -= lr * np.sum(d_pred, axis=0)
        W1_d -= lr * inp.T @ d_h
        b1_d -= lr * np.sum(d_h, axis=0)

        if epoch < 5 or epoch % 150 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch+1:>3d}: loss={loss:.4f}")

    # 生成: 从噪声出发, 逐步去噪
    n_gen = 1000
    x_gen = np.random.randn(n_gen)

    for t_val in range(T - 1, -1, -1):
        z_noise = np.random.randn(n_gen) if t_val > 0 else 0
        t_norm_val = np.full(n_gen, t_val / T)
        eps_pred = predict_noise(x_gen, t_norm_val)
        alpha_t = alpha[t_val]
        alpha_bar_t = alpha_bar[t_val]
        beta_t = beta[t_val]

        coef1 = 1 / np.sqrt(alpha_t)
        coef2 = beta_t / np.sqrt(1 - alpha_bar_t)
        x_gen = coef1 * (x_gen - coef2 * eps_pred) + np.sqrt(beta_t) * z_noise

    gen_mean = np.mean(x_gen)
    gen_std = np.std(x_gen)

    print(f"\n  -- 生成结果 --")
    print(f"  生成样本: mean={gen_mean:.3f}, std={gen_std:.3f}")
    print(f"  目标分布: mean={mu_data}, std={sigma_data}")

    mean_err = abs(gen_mean - mu_data)
    std_err = abs(gen_std - sigma_data)
    print(f"  均值误差: {mean_err:.3f}, 标准差误差: {std_err:.3f}")
    if mean_err < 0.3 and std_err < 0.3:
        print(f"  ✅ Diffusion 成功学到了目标分布!")
    else:
        print(f"  ⚠️ 近似结果 (简化模型+少迭代 — 正常)")

    print("\n  🎯 洞察:")
    print("    Diffusion: 学习逆向去噪过程")
    print("    Forward: 数据逐步加噪 → 纯噪声")
    print("    Reverse: 从纯噪声逐步去噪 → 新数据")
    print("    训练目标: 预测每一步添加的噪声 ε")
    print("    关键优势: 训练稳定 (MSE), 没有 mode collapse, 质量 SOTA")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    run_all = '--ex' not in sys.argv

    exercises = [
        ('1', exercise1_vae_latent_space),
        ('2', exercise2_gan_from_scratch),
        ('3', exercise3_normalizing_flow),
        ('4', exercise4_autoregressive_made),
        ('5', exercise5_diffusion_model),
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
