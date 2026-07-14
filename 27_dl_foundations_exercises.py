"""
=============================================================================
  CMU 10-708 L11 代码练习: 深度学习的统计与算法基础
=============================================================================

本文件包含 5 个代码练习:

  练习 1: 梯度下降变体 — SGD, Momentum, Adam 在 2D 损失面上的对比
  练习 2: 反向传播手写 — 2层 MLP 的完整前向+反向, 验证数值梯度
  练习 3: 偏差-方差分解 — 多项式回归, 展示 Bias² + Var + Noise
  练习 4: 正则化 — L2, Dropout 抑制过拟合的效果对比
  练习 5: 神经网络从零 — 2层 MLP 训练, 决策边界可视化

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
    if test_acc_final > 0.85:
        print(f"    ✅ 测试准确率 {test_acc_final:.1%} — 模型成功泛化!")


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
