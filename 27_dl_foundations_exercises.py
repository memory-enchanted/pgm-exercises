"""
=============================================================================
  CMU 10-708 L11 代码练习: 图模型与深度学习的桥梁
=============================================================================

L11 四大主题 → 对应练习:
  ① DL 组件概览       → 练习1-5: 优化器, BP, 偏差-方差, 正则化, MLP
  ② GM vs NN          → 练习2 (BP消息 vs VE消息), 练习5 (结构 vs 表示)
  ③ DL + GM 结合       → 练习6: VAE — 用NN做GM的推断和生成
  ④ Bayesian NN       → 练习7: MC Dropout — 近似贝叶斯推断

本文件包含 7 个代码练习:

  练习 1: 梯度下降变体 — SGD, Momentum, Adam 在 2D 损失面上的对比
  练习 2: 反向传播手写 — 2层 MLP 的完整前向+反向, 验证数值梯度
  练习 3: 偏差-方差分解 — 多项式回归, 展示 Bias² + Var + Noise
  练习 4: 正则化 — L2, Dropout 抑制过拟合的效果对比
  练习 5: 神经网络从零 — 2层 MLP 训练, 决策边界可视化
  练习 6: VAE 简化实现 — 用 NN 编码/解码, ELBO 训练 (DL + GM)
  练习 7: MC Dropout — 测试时开 Dropout, 估计预测不确定性 (贝叶斯视角)

使用方法:
  python 27_dl_foundations_exercises.py           # 运行全部
  python 27_dl_foundations_exercises.py --ex 1    # 只运行练习1

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


# ============================================================================
# 练习 1: 梯度下降变体对比
# ============================================================================

def exercise1_gd_variants():
    """
    在同一 2D 二次损失面上对比 SGD, Momentum, Adam 的优化路径。
    Loss: f(x,y) = x² + 10y² (ill-conditioned — y 方向陡峭, x 方向平缓)
    """
    print("=" * 70)
    print("练习 1: 梯度下降变体 — SGD, Momentum, Adam 对比")
    print("=" * 70)

    def loss(x, y):
        return x**2 + 10 * y**2

    def grad(x, y):
        return np.array([2 * x, 20 * y])

    n_iters = 200
    lr = 0.05
    start = np.array([8.0, 2.0])  # 离最优 (0,0) 较远

    print(f"\n  Loss: f(x,y) = x^2 + 10y^2  (condition number = 10)")
    print(f"  Start = ({start[0]}, {start[1]}), lr={lr}, n_iters={n_iters}")

    # ==== SGD ====
    theta_sgd = start.copy()
    path_sgd = [theta_sgd.copy()]
    for _ in range(n_iters):
        g = grad(*theta_sgd)
        theta_sgd = theta_sgd - lr * g
        path_sgd.append(theta_sgd.copy())

    # ==== Momentum ====
    theta_mom = start.copy()
    v_mom = np.zeros(2)
    beta_mom = 0.9
    path_mom = [theta_mom.copy()]
    for _ in range(n_iters):
        g = grad(*theta_mom)
        v_mom = beta_mom * v_mom + g
        theta_mom = theta_mom - lr * v_mom
        path_mom.append(theta_mom.copy())

    # ==== Adam ====
    theta_adam = start.copy()
    m_adam = np.zeros(2)
    s_adam = np.zeros(2)
    beta1, beta2 = 0.9, 0.999
    eps_adam = 1e-8
    lr_adam = 0.2  # Adam 通常需要更大的 lr
    path_adam = [theta_adam.copy()]
    for t in range(1, n_iters + 1):
        g = grad(*theta_adam)
        m_adam = beta1 * m_adam + (1 - beta1) * g
        s_adam = beta2 * s_adam + (1 - beta2) * g**2
        m_hat = m_adam / (1 - beta1**t)
        s_hat = s_adam / (1 - beta2**t)
        theta_adam = theta_adam - lr_adam * m_hat / (np.sqrt(s_hat) + eps_adam)
        path_adam.append(theta_adam.copy())

    final_sgd = loss(*path_sgd[-1])
    final_mom = loss(*path_mom[-1])
    final_adam = loss(*path_adam[-1])

    print(f"\n  -- 最终损失 --")
    print(f"  {'SGD':>10s}: loss = {final_sgd:.6f}, theta = ({path_sgd[-1][0]:.4f}, {path_sgd[-1][1]:.4f})")
    print(f"  {'Momentum':>10s}: loss = {final_mom:.6f}, theta = ({path_mom[-1][0]:.4f}, {path_mom[-1][1]:.4f})")
    print(f"  {'Adam':>10s}: loss = {final_adam:.6f}, theta = ({path_adam[-1][0]:.4f}, {path_adam[-1][1]:.4f})")

    # 收敛速度对比 (loss 降到 0.01 需要的步数)
    def steps_to_thresh(path, thresh=0.01):
        for i, p in enumerate(path):
            if loss(*p) < thresh:
                return i
        return len(path)

    sgd_steps = steps_to_thresh(path_sgd)
    mom_steps = steps_to_thresh(path_mom)
    adam_steps = steps_to_thresh(path_adam)

    print(f"\n  -- 收敛到 loss<0.01 所需步数 --")
    print(f"  SGD:      {sgd_steps} steps" + (" (未达到)" if sgd_steps >= n_iters else ""))
    print(f"  Momentum: {mom_steps} steps" + (" (未达到)" if mom_steps >= n_iters else ""))
    print(f"  Adam:     {adam_steps} steps" + (" (未达到)" if adam_steps >= n_iters else ""))

    print("\n  🎯 洞察:")
    print("    SGD: 沿最陡方向 → ill-conditioned 时 zig-zag, 收敛慢")
    print("    Momentum: 累积历史方向 → 平滑 zig-zag, 加速沿 '峡谷' 前进")
    print("    Adam: 自适应 per-dimension LR → y方向衰减快, x方向衰减慢 → 最优")


# ============================================================================
# 练习 2: 反向传播手写
# ============================================================================

def exercise2_backprop_from_scratch():
    """
    对 2 层 MLP (ReLU hidden, Sigmoid output) 手动推导并实现反向传播。
    用数值梯度验证解析梯度的正确性。

    结构: x(3) → W1(3x4) → h(4) ReLU → W2(4x1) → ŷ(1) Sigmoid
    Loss = BinaryCrossEntropy
    """
    print("=" * 70)
    print("练习 2: 反向传播手写 — 2层 MLP 梯度验证")
    print("=" * 70)

    # 生成一个随机数据点
    n_in, n_hid, n_out = 3, 4, 1
    x = np.array([0.5, -0.3, 0.8])
    y_true = 1.0

    W1 = np.random.randn(n_in, n_hid) * 0.5
    b1 = np.random.randn(n_hid) * 0.1
    W2 = np.random.randn(n_hid, n_out) * 0.5
    b2 = np.random.randn(n_out) * 0.1

    # ==== Forward Pass ====
    # Layer 1
    z1 = x @ W1 + b1           # (4,)
    h1 = np.maximum(0, z1)      # ReLU: (4,)
    # Layer 2
    z2 = h1 @ W2 + b2           # (1,)
    y_pred = 1.0 / (1.0 + np.exp(-z2))  # Sigmoid: (1,)
    # Loss: BCE
    eps = 1e-12
    loss = -(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))

    print(f"\n  结构: 3 → 4(ReLU) → 1(Sigmoid), BCE Loss")
    print(f"  Forward: y_pred = {y_pred[0]:.4f}, loss = {loss[0]:.4f}")

    # ==== Backward Pass (手动链式法则) ====
    # dL/dŷ
    d_yhat = -(y_true / (y_pred + eps) - (1 - y_true) / (1 - y_pred + eps))  # (1,)

    # dL/dz2 = dL/dŷ · dŷ/dz2 = dL/dŷ · ŷ(1-ŷ)
    d_z2 = d_yhat * y_pred * (1 - y_pred)  # (1,)

    # dL/dW2 = h1^T · dL/dz2
    d_W2 = h1.reshape(-1, 1) @ d_z2.reshape(1, -1)  # (4, 1)
    d_b2 = d_z2.copy()  # (1,)

    # dL/dh1 = dL/dz2 · W2^T
    d_h1 = d_z2 @ W2.T  # (4,)

    # dL/dz1 = dL/dh1 · ReLU'(z1) = dL/dh1 · 1[z1 > 0]
    d_z1 = d_h1 * (z1 > 0)  # (4,)

    # dL/dW1 = x^T · dL/dz1
    d_W1 = x.reshape(-1, 1) @ d_z1.reshape(1, -1)  # (3, 4)
    d_b1 = d_z1.copy()  # (4,)

    # ==== 数值梯度验证 (finite differences) ====
    def compute_loss(W1_val, b1_val, W2_val, b2_val):
        z1_v = x @ W1_val + b1_val
        h1_v = np.maximum(0, z1_v)
        z2_v = h1_v @ W2_val + b2_val
        yp = 1.0 / (1.0 + np.exp(-z2_v))
        return float(np.squeeze(-(y_true * np.log(yp + eps) + (1 - y_true) * np.log(1 - yp + eps))))

    delta = 1e-5
    max_rel_err = 0.0

    print(f"\n  -- 梯度验证 (finite diff δ={delta}) --")

    # 对每个参数随机验证一个元素
    checks = [
        ('W1[0,0]', W1, d_W1, (0, 0)),
        ('b1[0]', b1, d_b1, (0,)),
        ('W2[0,0]', W2, d_W2, (0, 0)),
        ('b2[0]', b2, d_b2, (0,)),
    ]

    for cname, val_arr, grad_arr, idx in checks:
        # 复制参数
        W1_t, b1_t = W1.copy(), b1.copy()
        W2_t, b2_t = W2.copy(), b2.copy()

        orig_v = val_arr[idx]
        # 根据参数名找到正确的变量
        if 'W1' in cname:
            ptr = W1_t
        elif 'b1' in cname:
            ptr = b1_t
        elif 'W2' in cname:
            ptr = W2_t
        else:
            ptr = b2_t

        ptr[idx] = orig_v + delta
        loss_plus = compute_loss(W1_t, b1_t, W2_t, b2_t)
        ptr[idx] = orig_v - delta
        loss_minus = compute_loss(W1_t, b1_t, W2_t, b2_t)
        grad_num = (loss_plus - loss_minus) / (2 * delta)
        grad_ana = float(grad_arr[idx])

        rel_err = abs(grad_num - grad_ana) / max(abs(grad_num), abs(grad_ana), 1e-10)
        max_rel_err = max(max_rel_err, rel_err)
        print(f"  {cname:>10s}: analytic={grad_ana:.6f}, numeric={grad_num:.6f}, "
              f"rel_err={rel_err:.2e}")

    print(f"\n  最大相对误差: {max_rel_err:.2e}")
    print(f"  {'✅ 梯度正确!' if max_rel_err < 1e-4 else '❌ 梯度有误!'}")

    print("\n  🎯 洞察:")
    print("    反向传播 = 从输出向输入逐层应用 Chain Rule")
    print("    每层的梯度 = 上游梯度 · 局部导数 (Jacobian)")
    print("    ReLU 的局部导数: 1[z>0] — 极简单!")
    print("    Sigmoid 的局部导数: ŷ(1-ŷ) — 用前向结果即得, 无需重算")


# ============================================================================
# 练习 3: 偏差-方差分解
# ============================================================================

def exercise3_bias_variance():
    """
    多项式回归上展示偏差-方差分解:
    E[(y-f̂)²] = Noise + Bias² + Variance

    真实函数: y = sin(1.5π·x) + ε,  ε ~ N(0, 0.3²)
    模型: 多项式回归, 度 d = 1, 3, 15
    """
    print("=" * 70)
    print("练习 3: 偏差-方差分解 — 多项式回归")
    print("=" * 70)

    def true_func(x):
        return np.sin(1.5 * np.pi * x)

    noise_std = 0.3
    n_train = 30
    n_test = 200
    n_datasets = 100

    x_test = np.linspace(-1, 1, n_test)

    degrees = [1, 3, 15]

    print(f"\n  真实: y = sin(1.5π·x) + N(0, {noise_std}²)")
    print(f"  训练集: n={n_train}, 重复 {n_datasets} 次")
    print(f"\n  -- 偏差-方差分解 --")
    print(f"  {'度':>4s}  {'Bias²':>10s}  {'Var':>10s}  {'Noise²':>10s}  {'MSE':>10s}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    for d in degrees:
        # 收集所有训练集的预测
        all_preds = np.zeros((n_datasets, n_test))

        for dataset in range(n_datasets):
            x_train = np.random.uniform(-1, 1, n_train)
            y_train = true_func(x_train) + np.random.randn(n_train) * noise_std

            # 多项式特征
            X_train = np.vstack([x_train**k for k in range(d + 1)]).T
            X_test = np.vstack([x_test**k for k in range(d + 1)]).T

            # 闭式解: w* = (X^T X)^{-1} X^T y
            try:
                w = np.linalg.solve(X_train.T @ X_train + 1e-6 * np.eye(d + 1),
                                    X_train.T @ y_train)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(X_train, y_train, rcond=None)[0]

            all_preds[dataset] = X_test @ w

        mean_pred = all_preds.mean(axis=0)
        bias_sq = np.mean((mean_pred - true_func(x_test))**2)
        variance = np.mean(np.var(all_preds, axis=0))
        noise_sq = noise_std**2
        mse = bias_sq + variance + noise_sq

        print(f"  {d:>4d}  {bias_sq:>10.4f}  {variance:>10.4f}  "
              f"{noise_sq:>10.4f}  {mse:>10.4f}")

    print(f"\n  🎯 洞察:")
    print("    d=1 (欠拟合): Bias² 大, Var 小 → 模型太简单, 系统偏差")
    print("    d=3 (刚好):   Bias² 和 Var 平衡 → 最优泛化")
    print("    d=15 (过拟合): Var 大, Bias² 小 → 模型记噪声, 泛化差")
    print("    Noise² = 0.09 是不可约的 — 无论模型多好都无法突破")


# ============================================================================
# 练习 4: 正则化对抗过拟合
# ============================================================================

def exercise4_regularization():
    """
    在高维 (D=100) 小样本 (N=30) 的线性回归上对比:
    - No regularization (过拟合)
    - L2 regularization (Ridge / weight decay)
    - L1 regularization (Lasso / sparsity)

    展示正则化如何降低测试误差。
    """
    print("=" * 70)
    print("练习 4: 正则化 — L2, L1 抑制过拟合")
    print("=" * 70)

    D = 100    # 高维
    N_train = 30
    N_test = 500

    # 真实: 只有 5 个特征非零
    w_true = np.zeros(D)
    w_true[:5] = np.array([1.0, 0.8, 0.5, -0.6, -1.0])
    np.random.shuffle(w_true)  # 打乱位置

    # 生成数据
    X_train = np.random.randn(N_train, D)
    y_train = X_train @ w_true + np.random.randn(N_train) * 0.3

    X_test = np.random.randn(N_test, D)
    y_test = X_test @ w_true + np.random.randn(N_test) * 0.3

    print(f"\n  D={D} 特征, 仅 {sum(w_true != 0)} 个非零 (稀疏)")
    print(f"  训练: N={N_train}, 测试: N={N_test}")

    # ==== 1. No Regularization (OLS) ====
    # w = (X^T X)^{-1} X^T y — ill-conditioned when D > N
    try:
        w_ols = np.linalg.solve(X_train.T @ X_train + 1e-8 * np.eye(D),
                                 X_train.T @ y_train)
    except np.linalg.LinAlgError:
        w_ols = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
    train_err_ols = np.mean((X_train @ w_ols - y_train)**2)
    test_err_ols = np.mean((X_test @ w_ols - y_test)**2)

    # ==== 2. L2 (Ridge) ====
    lambdas_l2 = [0.01, 0.1, 0.5, 1.0, 5.0]
    best_l2_test = np.inf
    best_w_l2 = None
    best_lam_l2 = None
    for lam in lambdas_l2:
        w_l2 = np.linalg.solve(X_train.T @ X_train + lam * np.eye(D),
                                X_train.T @ y_train)
        test_err = np.mean((X_test @ w_l2 - y_test)**2)
        if test_err < best_l2_test:
            best_l2_test = test_err
            best_w_l2 = w_l2
            best_lam_l2 = lam
    train_err_l2 = np.mean((X_train @ best_w_l2 - y_train)**2)

    # ==== 3. L1 (Lasso via coordinate descent — simplified) ====
    # 用迭代软阈值, 简化实现
    lam_l1 = 0.1
    w_l1 = np.zeros(D)
    n_cd_iters = 500
    for _ in range(n_cd_iters):
        for j in range(D):
            # 固定其他坐标, 对 w_j 做软阈值
            r_j = y_train - X_train @ w_l1 + X_train[:, j] * w_l1[j]
            rho = X_train[:, j] @ r_j
            z = X_train[:, j] @ X_train[:, j]
            if rho < -lam_l1:
                w_l1[j] = (rho + lam_l1) / z
            elif rho > lam_l1:
                w_l1[j] = (rho - lam_l1) / z
            else:
                w_l1[j] = 0.0
    train_err_l1 = np.mean((X_train @ w_l1 - y_train)**2)
    test_err_l1 = np.mean((X_test @ w_l1 - y_test)**2)
    n_nonzero_l1 = np.sum(np.abs(w_l1) > 1e-4)

    # ==== Results ====
    print(f"\n  -- 各方法对比 --")
    print(f"  {'方法':>18s}  {'训练MSE':>10s}  {'测试MSE':>10s}  {'非零权重':>10s}")
    print(f"  {'-'*18}  {'-'*10}  {'-'*10}  {'-'*10}")
    print(f"  {'No Reg (OLS)':>18s}  {train_err_ols:>10.4f}  {test_err_ols:>10.4f}  "
          f"{np.sum(np.abs(w_ols)>1e-4):>10d}")
    print(f"  {'L2 (λ={})'.format(best_lam_l2):>18s}  {train_err_l2:>10.4f}  "
          f"{best_l2_test:>10.4f}  {np.sum(np.abs(best_w_l2)>1e-4):>10d}")
    print(f"  {'L1 (λ={})'.format(lam_l1):>18s}  {train_err_l1:>10.4f}  "
          f"{test_err_l1:>10.4f}  {n_nonzero_l1:>10d}")

    test_best = min(test_err_ols, best_l2_test, test_err_l1)
    if test_best < test_err_ols:
        imp = (test_err_ols - test_best) / test_err_ols * 100
        print(f"\n  ✅ 正则化降低测试误差 {imp:.1f}%!")

    print("\n  🎯 洞察:")
    print("    OLS (D>N): 完美拟合训练数据 → 严重过拟合 → 测试误差大")
    print("    L2: 收缩所有权重 → 防止过大 |w| → 降低方差")
    print("    L1: 产生稀疏解 → 自动特征选择 → 可解释性 + 泛化")


# ============================================================================
# 练习 5: 神经网络从零 — 2层 MLP
# ============================================================================

def exercise5_neural_network_from_scratch():
    """
    纯 numpy 实现 2 层 MLP 做二分类。
    结构: 输入(2) → 隐藏(16) ReLU → 输出(1) Sigmoid + BCE

    数据: 两个交错的半月形 (moons) — 线性不可分
    展示: 决策边界, 训练 loss 曲线
    """
    print("=" * 70)
    print("练习 5: 2层 MLP 从零实现 — 半月形二分类")
    print("=" * 70)

    # 生成半月形数据
    def make_moons(n_samples=200, noise=0.1):
        n_out = n_samples // 2
        n_in = n_samples - n_out
        outer = np.linspace(0, np.pi, n_out)
        inner = np.linspace(0, np.pi, n_in)
        X_out = np.column_stack([np.cos(outer), np.sin(outer)]) + np.random.randn(n_out, 2) * noise
        X_in = np.column_stack([1 - np.cos(inner), 0.5 - np.sin(inner)]) + np.random.randn(n_in, 2) * noise
        X = np.vstack([X_out, X_in])
        y = np.hstack([np.zeros(n_out), np.ones(n_in)])
        return X, y

    X, y = make_moons(300, noise=0.12)
    y = y.reshape(-1, 1)
    # 打乱
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    # 划分
    n_train = 200
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    # 模型参数
    n_in, n_hid, n_out = 2, 16, 1
    W1 = np.random.randn(n_in, n_hid) * np.sqrt(2.0 / n_in)
    b1 = np.zeros(n_hid)
    W2 = np.random.randn(n_hid, n_out) * np.sqrt(2.0 / n_hid)
    b2 = np.zeros(n_out)

    # 训练
    n_epochs = 500
    lr = 0.1
    train_losses = []
    test_losses = []

    print(f"\n  数据: 半月形 (n_train={n_train}, n_test={len(X_test)})")
    print(f"  结构: 2 → 16(ReLU) → 1(Sigmoid), BCE Loss")
    print(f"\n  -- 训练 (lr={lr}) --")
    print(f"  {'epoch':>6s}  {'train_loss':>12s}  {'test_loss':>12s}  {'train_acc':>10s}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*10}")

    for epoch in range(n_epochs):
        # Forward
        z1 = X_train @ W1 + b1
        h1 = np.maximum(0, z1)
        z2 = h1 @ W2 + b2
        y_pred = 1.0 / (1.0 + np.exp(-z2))
        eps = 1e-12
        train_loss = -np.mean(y_train * np.log(y_pred + eps) +
                              (1 - y_train) * np.log(1 - y_pred + eps))

        # Backward
        d_z2 = (y_pred - y_train) / n_train
        d_W2 = h1.T @ d_z2
        d_b2 = np.sum(d_z2, axis=0)
        d_h1 = d_z2 @ W2.T
        d_z1 = d_h1 * (z1 > 0)
        d_W1 = X_train.T @ d_z1
        d_b1 = np.sum(d_z1, axis=0)

        # SGD update
        W1 -= lr * d_W1
        b1 -= lr * d_b1
        W2 -= lr * d_W2
        b2 -= lr * d_b2

        # Test
        z1_t = X_test @ W1 + b1
        h1_t = np.maximum(0, z1_t)
        z2_t = h1_t @ W2 + b2
        y_pred_t = 1.0 / (1.0 + np.exp(-z2_t))
        test_loss = -np.mean(y_test * np.log(y_pred_t + eps) +
                             (1 - y_test) * np.log(1 - y_pred_t + eps))

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if epoch < 5 or epoch % 100 == 0 or epoch == n_epochs - 1:
            train_acc = np.mean((y_pred > 0.5) == y_train)
            print(f"  {epoch+1:>6d}  {train_loss:>12.4f}  {test_loss:>12.4f}  "
                  f"{train_acc:>10.2%}")

    # Final accuracy
    train_acc_final = np.mean((y_pred > 0.5) == y_train)
    test_acc_final = np.mean((y_pred_t > 0.5) == y_test)

    print(f"\n  -- 最终结果 --")
    print(f"  训练准确率: {train_acc_final:.2%}")
    print(f"  测试准确率: {test_acc_final:.2%}")

    # 决策边界信息 (采样分析)
    n_boundary = 500
    xx = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 50)
    yy = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 50)
    XX, YY = np.meshgrid(xx, yy)
    grid = np.column_stack([XX.ravel(), YY.ravel()])
    z1_g = grid @ W1 + b1
    h1_g = np.maximum(0, z1_g)
    z2_g = h1_g @ W2 + b2
    ZZ = (1.0 / (1.0 + np.exp(-z2_g))).reshape(XX.shape)

    # 数决策边界两边的正确率
    pred_grid = (ZZ > 0.5).ravel()
    print(f"  决策区域: 类别0占 {np.mean(pred_grid==0):.1%}, 类别1占 {np.mean(pred_grid==1):.1%}")

    print(f"\n  🎯 洞察:")
    print("    2层 MLP (with ReLU) 可以学习非线性决策边界")
    print("    反向传播 = Chain Rule 的自动应用 — 不需要手动推导")
    print("    SGD 每次更新整个网络的所有参数 — '端到端'学习")
    print("    ↔ 对比 GM: GM 用图结构和条件概率表, 可解释但容量有限")
    print("    ↔ NN 用层次化权重表示, 容量大但不可解释 — 互补!")
    if test_acc_final > 0.85:
        print(f"    ✅ 测试准确率 {test_acc_final:.1%} — 模型成功泛化!")


# ============================================================================
# 练习 6: VAE 简化实现 — DL + GM 的结合
# ============================================================================

def exercise6_vae_simple():
    """
    简化 VAE: 在 2D 玩具数据上实现 VAE, 展示如何用 NN 做:
      - Encoder (inference network): x → μ_z, σ_z → 近似后验
      - Decoder (generative model):  z → μ_x → 生成分布
      - ELBO = Reconstruction - KL(q(z|x) || p(z))

    数据: 8个高斯团, z 维度=2 (方便可视化)
    GM 视角: VAE = 连续隐变量的图模型, 用 NN 参数化条件分布, 用 Amortized VI 做推断
    """
    print("=" * 70)
    print("练习 6: VAE 简化实现 — 用 NN 做 GM 的推断与生成 (DL + GM)")
    print("=" * 70)

    n_samples = 500
    latent_dim = 2
    hidden_dim = 32
    n_epochs = 300
    lr = 0.01

    # 生成 8 团玩具数据
    def generate_8gaussians(n):
        centers = np.array([[2, 2], [2, -2], [-2, 2], [-2, -2],
                            [2, 0], [-2, 0], [0, 2], [0, -2]])
        X = []
        for i in range(n):
            c = centers[i % 8]
            X.append(c + np.random.randn(2) * 0.3)
        return np.array(X)

    X = generate_8gaussians(n_samples)
    input_dim = 2

    print(f"\n  数据: {n_samples} 个点, 分布在 8 个高斯团")
    print(f"  VAE 结构: {input_dim} → {hidden_dim}(ReLU) → μ_z,σ_z({latent_dim})")
    print(f"                         z({latent_dim}) → {hidden_dim}(ReLU) → μ_x({input_dim})")
    print(f"  DL+GM: encoder=Amortized Inference, decoder=条件分布 P(X|Z)")

    # 初始化参数 (He initialization)
    W_enc = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
    b_enc = np.zeros(hidden_dim)
    W_mu = np.random.randn(hidden_dim, latent_dim) * np.sqrt(2.0 / hidden_dim)
    b_mu = np.zeros(latent_dim)
    W_logvar = np.random.randn(hidden_dim, latent_dim) * np.sqrt(2.0 / hidden_dim)
    b_logvar = np.zeros(latent_dim)

    W_dec = np.random.randn(latent_dim, hidden_dim) * np.sqrt(2.0 / latent_dim)
    b_dec = np.zeros(hidden_dim)
    W_out = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / hidden_dim)
    b_out = np.zeros(input_dim)

    print(f"\n  -- 训练 (ELBO = Recon - KL, lr={lr}) --")
    print(f"  {'epoch':>6s}  {'ELBO':>10s}  {'Recon':>10s}  {'KL':>10s}")

    for epoch in range(n_epochs):
        # ---- Forward: Encoder ----
        h_enc = np.maximum(0, X @ W_enc + b_enc)          # ReLU
        mu_z = h_enc @ W_mu + b_mu
        logvar_z = h_enc @ W_logvar + b_logvar
        std_z = np.exp(0.5 * logvar_z)

        # 重参数化: z = μ + σ·ε
        eps = np.random.randn(n_samples, latent_dim)
        z = mu_z + std_z * eps

        # ---- Forward: Decoder ----
        h_dec = np.maximum(0, z @ W_dec + b_dec)          # ReLU
        mu_x = h_dec @ W_out + b_out

        # ---- ELBO 计算 ----
        # Recon: E_q[log P(x|z)], 假设 P(x|z) = N(mu_x, I)
        recon = -0.5 * np.mean(np.sum((X - mu_x)**2, axis=1))

        # KL: KL(q(z|x) || p(z)), q=N(μ,σ²), p=N(0,1)
        kl = 0.5 * np.mean(np.sum(mu_z**2 + std_z**2 - 2*logvar_z - 1, axis=1))

        elbo = recon - kl

        # ---- Backward (简化的梯度, 只更新关键参数) ----
        # dELBO/dmu_x
        d_mu_x = (mu_x - X) / n_samples  # (N, input_dim)

        d_W_out = h_dec.T @ d_mu_x
        d_b_out = np.sum(d_mu_x, axis=0)
        d_h_dec = d_mu_x @ W_out.T
        d_W_dec = z.T @ (d_h_dec * (h_dec > 0))
        d_b_dec = np.sum(d_h_dec * (h_dec > 0), axis=0)

        # 这里简化: 仅用 MSE 信号更新 decoder, 用 ELBO 信号更新 encoder
        # Decoder 更新
        W_out -= lr * d_W_out
        b_out -= lr * d_b_out
        W_dec -= lr * d_W_dec
        b_dec -= lr * d_b_dec

        # Encoder 更新 (简化梯度)
        d_z = (d_h_dec * (h_dec > 0)) @ W_dec.T
        d_mu_z = d_z + mu_z / n_samples   # Recon梯度 + KL梯度: ∇_μ KL = μ
        d_logvar_z = d_z * 0.5 * std_z * eps + 0.5 * (std_z**2 - 1) / n_samples

        d_W_mu = h_enc.T @ d_mu_z
        d_b_mu = np.sum(d_mu_z, axis=0)
        d_W_logvar = h_enc.T @ d_logvar_z
        d_b_logvar = np.sum(d_logvar_z, axis=0)

        d_h_enc = d_mu_z @ W_mu.T + d_logvar_z @ W_logvar.T
        d_W_enc = X.T @ (d_h_enc * (h_enc > 0))
        d_b_enc = np.sum(d_h_enc * (h_enc > 0), axis=0)

        W_mu -= lr * d_W_mu
        b_mu -= lr * d_b_mu
        W_logvar -= lr * d_W_logvar
        b_logvar -= lr * d_b_logvar
        W_enc -= lr * d_W_enc
        b_enc -= lr * d_b_enc

        if epoch < 5 or epoch % 100 == 0 or epoch == n_epochs - 1:
            print(f"  {epoch+1:>6d}  {elbo:>10.4f}  {recon:>10.4f}  {kl:>10.4f}")

    # 生成样本
    n_gen = 200
    z_prior = np.random.randn(n_gen, latent_dim)
    h_gen = np.maximum(0, z_prior @ W_dec + b_dec)
    x_gen = h_gen @ W_out + b_out

    # 分析生成质量
    print(f"\n  -- 生成分析 --")
    print(f"  从先验 N(0,I) 采样 {n_gen} 个 z → decoder 生成样本")
    print(f"  真实数据范围: x=[{X[:,0].min():.1f},{X[:,0].max():.1f}], "
          f"y=[{X[:,1].min():.1f},{X[:,1].max():.1f}]")
    print(f"  生成数据范围: x=[{x_gen[:,0].min():.1f},{x_gen[:,0].max():.1f}], "
          f"y=[{x_gen[:,1].min():.1f},{x_gen[:,1].max():.1f}]")

    # 检查KL是否下降（模型是否学会了有结构的latent space）
    print(f"\n  🎯 洞察:")
    print(f"    VAE = GM框架 + NN参数化:")
    print(f"    - Encoder (NN) 做 Amortized VI → 替代传统 per-sample 优化")
    print(f"    - Decoder (NN) 参数化 P(X|Z) → 替代手工设计的条件概率表")
    print(f"    - ELBO = Recon - KL → 和 GM 的 VI 完全相同的目标!")
    print(f"    - Reparameterization Trick → 梯度可流过采样 → 端到端训练")
    if abs(kl) < 10:
        print(f"    ✅ KL={abs(kl):.2f} — 隐空间被正则化到接近先验 N(0,I)")
    else:
        print(f"    ⚠ KL={abs(kl):.2f} — KL 较大, 可能需要调整 β (β-VAE) 或更多训练")


# ============================================================================
# 练习 7: MC Dropout — 近似贝叶斯推断
# ============================================================================

def exercise7_mc_dropout():
    """
    MC Dropout: 在测试时保持 Dropout 开启, 多次前向传播,
    用预测的均值和方差估计不确定性。

    对比:
      - 标准 Dropout: 测试时关闭, 单次预测 → 点估计 (no uncertainty)
      - MC Dropout:   测试时开启, T次预测  → 近似贝叶斯推断

    在半月形数据上展示: 模型在训练区域内自信, 在外推区域不确定。
    """
    print("=" * 70)
    print("练习 7: MC Dropout — 测试时开 Dropout 估计不确定性 (贝叶斯视角)")
    print("=" * 70)

    # 生成半月形数据
    n_samples = 300
    n_out = n_samples // 2
    n_in = n_samples - n_out
    outer = np.linspace(0, np.pi, n_out)
    inner = np.linspace(0, np.pi, n_in)
    X_out = np.column_stack([np.cos(outer), np.sin(outer)]) + np.random.randn(n_out, 2) * 0.12
    X_in = np.column_stack([1 - np.cos(inner), 0.5 - np.sin(inner)]) + np.random.randn(n_in, 2) * 0.12
    X = np.vstack([X_out, X_in])
    y = np.hstack([np.zeros(n_out), np.ones(n_in)]).reshape(-1, 1)
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    n_train = 250
    X_train, y_train = X[:n_train], y[:n_train]

    # 模型: 2层 MLP (比练习5更大的网络, 以便 Dropout 有效)
    n_in_feat, n_hid, n_out_feat = 2, 64, 1
    dropout_p = 0.3

    W1 = np.random.randn(n_in_feat, n_hid) * np.sqrt(2.0 / n_in_feat)
    b1 = np.zeros(n_hid)
    W2 = np.random.randn(n_hid, n_out_feat) * np.sqrt(2.0 / n_hid)
    b2 = np.zeros(n_out_feat)

    # 训练 (带 Dropout)
    n_epochs = 800
    lr = 0.05
    print(f"\n  结构: 2 → 64(ReLU) → Dropout({dropout_p}) → 1(Sigmoid)")
    print(f"  训练: n={n_train}, dropout_p={dropout_p}, lr={lr}")
    print(f"\n  -- 训练 --")

    for epoch in range(n_epochs):
        # Forward with dropout
        mask1 = (np.random.rand(n_train, n_hid) > dropout_p).astype(float) / (1 - dropout_p)
        z1 = X_train @ W1 + b1
        h1 = np.maximum(0, z1) * mask1          # inverted dropout
        z2 = h1 @ W2 + b2
        y_pred = 1.0 / (1.0 + np.exp(-z2))
        eps = 1e-12
        loss = -np.mean(y_train * np.log(y_pred + eps) + (1 - y_train) * np.log(1 - y_pred + eps))

        # Backward
        d_z2 = (y_pred - y_train) / n_train
        d_W2 = h1.T @ d_z2
        d_b2 = np.sum(d_z2, axis=0)
        d_h1 = d_z2 @ W2.T * mask1
        d_z1 = d_h1 * (z1 > 0)
        d_W1 = X_train.T @ d_z1
        d_b1 = np.sum(d_z1, axis=0)

        W1 -= lr * d_W1; b1 -= lr * d_b1
        W2 -= lr * d_W2; b2 -= lr * d_b2

        if epoch < 5 or epoch % 200 == 0:
            train_acc = np.mean((y_pred > 0.5) == y_train)
            print(f"  epoch {epoch+1:>4d}: loss={loss:.4f}, acc={train_acc:.2%}")

    # ==== 关键对比: 标准预测 vs MC Dropout ====
    # 生成测试网格 (覆盖训练区 + 外推区)
    xx = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 50)
    yy = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 50)
    XX, YY = np.meshgrid(xx, yy)
    grid = np.column_stack([XX.ravel(), YY.ravel()])
    n_grid = len(grid)

    # 标准预测 (Dropout OFF, 单次)
    z1_std = grid @ W1 + b1
    h1_std = np.maximum(0, z1_std)
    z2_std = h1_std @ W2 + b2
    prob_std = 1.0 / (1.0 + np.exp(-z2_std))
    prob_std = prob_std.reshape(XX.shape)

    # MC Dropout (Dropout ON, T 次采样)
    mc_samples = 100
    all_probs = np.zeros((mc_samples, n_grid))
    for t in range(mc_samples):
        mask_mc = (np.random.rand(n_grid, n_hid) > dropout_p).astype(float) / (1 - dropout_p)
        z1_mc = grid @ W1 + b1
        h1_mc = np.maximum(0, z1_mc) * mask_mc
        z2_mc = h1_mc @ W2 + b2
        all_probs[t] = (1.0 / (1.0 + np.exp(-z2_mc))).ravel()

    mc_mean = all_probs.mean(axis=0).reshape(XX.shape)
    mc_std = all_probs.std(axis=0).reshape(XX.shape)

    # 分析不确定性
    print(f"\n  -- MC Dropout 不确定性分析 (T={mc_samples}) --")
    print(f"  {'区域':>16s}  {'平均预测':>10s}  {'平均不确定性':>14s}  {'解读':>20s}")
    print(f"  {'-'*16}  {'-'*10}  {'-'*14}  {'-'*20}")

    # 训练区域: X 在 [-1.5, 2] × [-1, 1.5]
    in_region = ((grid[:, 0] > -1.5) & (grid[:, 0] < 2) &
                 (grid[:, 1] > -1) & (grid[:, 1] < 1.5))
    out_region = ~in_region

    in_mean = mc_mean.ravel()[in_region].mean() if in_region.sum() > 0 else 0
    in_std = mc_std.ravel()[in_region].mean() if in_region.sum() > 0 else 0
    out_mean = mc_mean.ravel()[out_region].mean() if out_region.sum() > 0 else 0
    out_std = mc_std.ravel()[out_region].mean() if out_region.sum() > 0 else 0

    print(f"  {'训练区域内':>16s}  {in_mean:>10.3f}  {in_std:>14.4f}  {'模型自信 (低不确定性)':>20s}")
    print(f"  {'训练区域外':>16s}  {out_mean:>10.3f}  {out_std:>14.4f}  {'模型不确定 (高不确定性)':>20s}")

    # 检查 MC Dropout 是否在训练区域外给出更高的不确定性
    if in_std < out_std:
        print(f"\n  ✅ 训练区域外的不确定性 ({out_std:.4f}) > 训练区域内 ({in_std:.4f})")
        print(f"     → MC Dropout 成功捕获了认知不确定性 (Epistemic Uncertainty)!")
    else:
        print(f"\n  ⚠ 不确定性区分不明显, 可能需要更多 MC 样本或调整 dropout_p")

    # 对比标准预测 vs MC Dropout 在几个关键点的预测
    test_points = np.array([
        [0.5, 0.3],    # 训练区域内
        [1.5, -0.5],   # 训练区域内
        [3.0, 2.0],    # 训练区域外 (外推)
        [-2.0, -2.0],  # 训练区域外 (外推)
    ])
    print(f"\n  -- 点预测对比: 标准(Dropout OFF) vs MC Dropout --")
    print(f"  {'点':>18s}  {'标准预测':>10s}  {'MC均值':>10s}  {'MC标准差':>10s}  {'在训练区?':>10s}")
    print(f"  {'-'*18}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for pt in test_points:
        # 标准预测
        z1_t = pt @ W1 + b1
        h1_t = np.maximum(0, z1_t)
        z2_t = h1_t @ W2 + b2
        std_pred = 1.0 / (1.0 + np.exp(-z2_t))[0]

        # MC Dropout
        mc_preds = []
        for _ in range(mc_samples):
            mask_t = (np.random.rand(1, n_hid) > dropout_p).astype(float) / (1 - dropout_p)
            z1_m = pt @ W1 + b1
            h1_m = np.maximum(0, z1_m) * mask_t
            z2_m = h1_m @ W2 + b2
            mc_preds.append((1.0 / (1.0 + np.exp(-z2_m)))[0])
        mc_m = np.mean(mc_preds)
        mc_s = np.std(mc_preds)
        in_train = ((pt[0] > -1.5) & (pt[0] < 2) & (pt[1] > -1) & (pt[1] < 1.5))
        print(f"  ({pt[0]:>5.1f},{pt[1]:>5.1f})     {std_pred:>10.4f}  {mc_m:>10.4f}  "
              f"{mc_s:>10.4f}  {'Yes' if in_train else 'No':>10s}")

    print(f"\n  🎯 洞察:")
    print(f"    标准 NN: 点估计 W* → 单一预测 → 不会说'不知道'")
    print(f"    MC Dropout: Dropout在测试时也开启 → 每次采样不同mask")
    print(f"      = 采样不同的子网络 (近似从后验 P(W|D) 采样)")
    print(f"      = T次前向 ≈ 由T个模型投票 → 方差=认知不确定性")
    print(f"    ↔ 图模型: GM天然给出 P(Y|X,D), BNN通过采样近似得到")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    run_all = '--ex' not in sys.argv

    exercises = [
        ('1', exercise1_gd_variants),
        ('2', exercise2_backprop_from_scratch),
        ('3', exercise3_bias_variance),
        ('4', exercise4_regularization),
        ('5', exercise5_neural_network_from_scratch),
        ('6', exercise6_vae_simple),
        ('7', exercise7_mc_dropout),
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
