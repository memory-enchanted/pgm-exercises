"""
=============================================================================
  CMU 10-708 L10 代码练习: 进阶 MCMC — HMC, Slice, Tempering, AIS, 诊断
=============================================================================

本文件包含 5 个代码练习:

  练习 1: Slice Sampling — 免 proposal 的自适应采样, 100% 接受率
  练习 2: Hamiltonian Monte Carlo — 梯度引导的高效采样, 对比 RW-MH
  练习 3: Parallel Tempering — 多温度链克服多峰分布
  练习 4: Annealed Importance Sampling — 估计 partition function
  练习 5: MCMC 收敛诊断 — R-hat, ESS, trace plot, MCSE

使用方法:
  python 24_advanced_mcmc_exercises.py           # 运行全部练习
  python 24_advanced_mcmc_exercises.py --ex 1    # 只运行练习1

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
# 工具函数
# ============================================================================

def autocorr(x, max_lag=50):
    """计算自相关函数, x 为 1D 数组"""
    xc = x - x.mean()
    n = len(xc)
    ac = np.zeros(max_lag + 1)
    ac[0] = 1.0
    for lag in range(1, max_lag + 1):
        ac[lag] = np.correlate(xc, xc, mode='full')[n-1+lag] / (np.var(xc) * n)
        if ac[lag] < 0.02:
            break
    return ac


def compute_ess(chain):
    """计算有效样本量 ESS = n / (1 + 2*sum_{k} rho_k), 截断到 rho_k < 0.05"""
    n = len(chain)
    ac = autocorr(chain)
    # sum up to where ac drops below 0.05
    ac_sum = 0.0
    for k in range(1, len(ac)):
        if ac[k] < 0.03:
            break
        ac_sum += ac[k]
    return n / (1 + 2 * ac_sum)


# ============================================================================
# 练习 1: Slice Sampling
# ============================================================================

def exercise1_slice_sampling():
    """
    对 1D 混合高斯实现 Slice Sampling:
      P(x) = 0.3*N(-3,1) + 0.7*N(4,1.5)

    对比 RW-MH: Slice 无需 proposal 调参, 接受率 100%。
    """
    print("=" * 70)
    print("练习 1: Slice Sampling — 免 proposal 自适应")
    print("=" * 70)

    def log_target(x):
        log1 = -0.5 * (x + 3)**2
        log2 = -0.5 * ((x - 4) / np.sqrt(1.5))**2 - 0.5 * np.log(1.5)
        mx = max(log1, log2)
        return np.log(0.3 * np.exp(log1 - mx) + 0.7 * np.exp(log2 - mx)) + mx

    n_samples = 3000
    burn = 500

    # --- Slice Sampling ---
    x_slice = 0.0
    w_guess = 2.0  # 初始切片宽度猜测
    samples_slice = np.zeros(n_samples + burn)

    for i in range(n_samples + burn):
        # Step 1: sample vertical level u ~ Uniform(0, P(x))
        u = np.log(np.random.rand()) + log_target(x_slice)

        # Step 2: stepping-out to find interval [L, R] where P > u
        L = x_slice - w_guess * np.random.rand()
        R = L + w_guess

        # Expand left
        while log_target(L) > u:
            L -= w_guess
        # Expand right
        while log_target(R) > u:
            R += w_guess

        # Step 3: sample uniformly within the slice
        while True:
            x_prop = L + np.random.rand() * (R - L)
            if log_target(x_prop) > u:
                x_slice = x_prop
                w_guess = R - L  # adapt width
                break
            # shrink interval
            if x_prop < x_slice:
                L = x_prop
            else:
                R = x_prop

        samples_slice[i] = x_slice

    # --- RW-MH for comparison ---
    x_rw = 0.0
    sigma_prop = 1.5
    samples_rw = np.zeros(n_samples + burn)
    accepted_rw = 0

    for i in range(n_samples + burn):
        x_prop = x_rw + np.random.randn() * sigma_prop
        if np.log(np.random.rand()) < log_target(x_prop) - log_target(x_rw):
            x_rw = x_prop
            accepted_rw += 1
        samples_rw[i] = x_rw

    # --- Results ---
    post_slice = samples_slice[burn:]
    post_rw = samples_rw[burn:]

    ess_slice = compute_ess(post_slice)
    ess_rw = compute_ess(post_rw)

    print(f"\n  目标: P(x) = 0.3*N(-3,1) + 0.7*N(4,1.5)  (双峰)")
    print(f"\n  -- Slice vs RW-MH 对比 --")
    print(f"  {'方法':>15s}  {'接受率':>8s}  {'均值':>8s}  {'ESS':>8s}  {'ESS/n':>8s}  "
          f"{'lag-1 AC':>10s}")
    print(f"  {'-'*15}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")
    ac1_slice = autocorr(post_slice, 5)[1]
    ac1_rw = autocorr(post_rw, 5)[1]
    print(f"  {'Slice':>15s}  {'100%':>8s}  {np.mean(post_slice):>8.3f}  "
          f"{ess_slice:>8.0f}  {ess_slice/n_samples:>8.1%}  {ac1_slice:>10.4f}")
    print(f"  {'RW-MH':>15s}  {accepted_rw/(n_samples+burn):>8.1%}  "
          f"{np.mean(post_rw):>8.3f}  {ess_rw:>8.0f}  {ess_rw/n_samples:>8.1%}  {ac1_rw:>10.4f}")

    print(f"\n  🎯 洞察:")
    print("    Slice Sampling: 免 proposal 调参, 接受率 100%")
    print("    自适应步长 → 在双峰间切换更灵活")
    print("    局限: 多维时需要逐维 Gibbs + Slice 组合")


# ============================================================================
# 练习 2: Hamiltonian Monte Carlo
# ============================================================================

def exercise2_hamiltonian_monte_carlo():
    """
    在 2D 强相关 Gaussian (ρ=0.95) 上对比 HMC vs RW-MH。
    展示 HMC 的梯度引导如何大幅降低自相关。
    """
    print("=" * 70)
    print("练习 2: Hamiltonian Monte Carlo — 梯度引导的高效采样")
    print("=" * 70)

    # 目标: 强相关的 2D Gaussian
    mean_true = np.array([3.0, -1.0])
    rho = 0.95
    cov_true = np.array([[1.0, rho], [rho, 1.0]])
    cov_inv = np.linalg.inv(cov_true)

    def log_prob(x):
        d = x - mean_true
        return -0.5 * d @ cov_inv @ d

    def grad_log_prob(x):
        return -cov_inv @ (x - mean_true)

    n_samples = 2000
    burn = 500
    total_steps = n_samples + burn

    # ==== HMC ====
    L_hmc = 10       # leapfrog steps
    eps_hmc = 0.18   # step size
    x_hmc = np.array([0.0, 0.0])
    samples_hmc = np.zeros((total_steps, 2))
    accepted_hmc = 0

    for i in range(total_steps):
        p = np.random.randn(2)
        x_cur, p_cur = x_hmc.copy(), p.copy()

        # Leapfrog
        p_half = p_cur + 0.5 * eps_hmc * grad_log_prob(x_cur)
        x_new, p_new = x_cur.copy(), p_half.copy()
        for step in range(L_hmc):
            x_new = x_new + eps_hmc * p_new
            if step < L_hmc - 1:
                p_new = p_new + eps_hmc * grad_log_prob(x_new)
        p_new = p_new + 0.5 * eps_hmc * grad_log_prob(x_new)

        # Metropolis correction
        H_cur = -log_prob(x_cur) + 0.5 * np.dot(p_cur, p_cur)
        H_new = -log_prob(x_new) + 0.5 * np.dot(p_new, p_new)

        if np.log(np.random.rand()) < H_cur - H_new:
            x_hmc = x_new
            accepted_hmc += 1
        samples_hmc[i] = x_hmc

    # ==== RW-MH (tuned to ~25% acceptance) ====
    sigma_rw = 0.25
    x_rw = np.array([0.0, 0.0])
    samples_rw = np.zeros((total_steps, 2))
    accepted_rw = 0

    for i in range(total_steps):
        x_prop = x_rw + np.random.randn(2) * sigma_rw
        if np.log(np.random.rand()) < log_prob(x_prop) - log_prob(x_rw):
            x_rw = x_prop
            accepted_rw += 1
        samples_rw[i] = x_rw

    # ==== Results ====
    post_hmc = samples_hmc[burn:]
    post_rw = samples_rw[burn:]

    # ESS (use x1 dimension)
    ess_hmc = compute_ess(post_hmc[:, 0])
    ess_rw = compute_ess(post_rw[:, 0])

    # Autocorrelation
    ac_hmc = autocorr(post_hmc[:, 0], 30)
    ac_rw = autocorr(post_rw[:, 0], 30)

    # Mean and covariance estimates
    hmc_mean = post_hmc.mean(axis=0)
    rw_mean = post_rw.mean(axis=0)
    hmc_cov = np.cov(post_hmc.T)
    rw_cov = np.cov(post_rw.T)

    print(f"\n  目标: N(μ=[3,-1], Σ=[[1,{rho}],[{rho},1]])  (强相关 ρ={rho})")
    print(f"\n  -- HMC vs RW-MH 对比 --")
    print(f"  {'方法':>8s} {'接受率':>8s} {'lag-1 AC':>10s} {'lag-5 AC':>10s} "
          f"{'ESS':>8s} {'均值误差':>10s} {'Σ误差':>10s}")
    print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*10}")

    hmc_mean_err = np.linalg.norm(hmc_mean - mean_true)
    rw_mean_err = np.linalg.norm(rw_mean - mean_true)
    hmc_cov_err = np.linalg.norm(hmc_cov - cov_true, 'fro')
    rw_cov_err = np.linalg.norm(rw_cov - cov_true, 'fro')

    print(f"  {'HMC':>8s} {accepted_hmc/total_steps:>8.1%} {ac_hmc[1]:>10.4f} "
          f"{ac_hmc[5] if len(ac_hmc)>5 else 0:>10.4f} {ess_hmc:>8.0f} "
          f"{hmc_mean_err:>10.4f} {hmc_cov_err:>10.4f}")
    print(f"  {'RW-MH':>8s} {accepted_rw/total_steps:>8.1%} {ac_rw[1]:>10.4f} "
          f"{ac_rw[5] if len(ac_rw)>5 else 0:>10.4f} {ess_rw:>8.0f} "
          f"{rw_mean_err:>10.4f} {rw_cov_err:>10.4f}")

    print(f"\n  ESS 提升: {ess_hmc/ess_rw:.1f}x (HMC 每个样本'价值'更高)")
    print(f"  自相关衰减: HMC lag-1={ac_hmc[1]:.3f}, RW-MH lag-1={ac_rw[1]:.3f}")

    print(f"\n  真实: μ={mean_true}")
    print(f"  HMC 估计: μ={np.round(hmc_mean, 3)}")
    print(f"  RW-MH:    μ={np.round(rw_mean, 3)}")

    print("\n  🎯 洞察:")
    print("    HMC 利用 ∇log P → proposal 沿概率等高线移动 → 远距离移动")
    print("    结果: 接受率 ~90% vs ~25%, 自相关极低 → ESS 高 5-20x")
    print("    代价: 每步需要 L 次梯度计算 (但总效率远超 RW-MH)")


# ============================================================================
# 练习 3: Parallel Tempering
# ============================================================================

def exercise3_parallel_tempering():
    """
    对一个强双峰分布 (well-separated bimodal) 实现 Parallel Tempering。
    对比: 标准 RW-MH 卡在一个 mode, Tempering 在两个 mode 间切换。
    """
    print("=" * 70)
    print("练习 3: Parallel Tempering — 多温度链克服双峰")
    print("=" * 70)

    # 双峰目标: P(x) ∝ N(-4, 0.8) + N(4, 0.8)
    def log_target_scaled(x, T=1.0):
        """log P(x)^{1/T} — T 越大, 分布越平"""
        log1 = -0.5 * ((x + 4) / np.sqrt(0.8))**2
        log2 = -0.5 * ((x - 4) / np.sqrt(0.8))**2
        mx = max(log1, log2)
        log_p = np.log(np.exp(log1 - mx) + np.exp(log2 - mx)) + mx
        return log_p / T  # 温度缩放

    # 温度阶梯
    n_chains = 5
    temps = np.array([1.0, 2.0, 4.0, 8.0, 16.0])

    n_samples = 3000
    burn = 500
    total_steps = n_samples + burn

    # 每链的 MH proposal 宽度
    sigma_props = 0.5 * np.sqrt(temps)  # 高温链用更大的步长

    # Initialize chains
    x_chains = np.array([0.0] * n_chains)  # 所有链从 mode 之间出发
    samples_cold = np.zeros(total_steps)  # 只记录 T=1 链

    n_swaps = 0
    n_accepted_swaps = 0

    for i in range(total_steps):
        # Step 1: 每条链独立 MH
        for k in range(n_chains):
            x_prop = x_chains[k] + np.random.randn() * sigma_props[k]
            log_alpha = (log_target_scaled(x_prop, temps[k]) -
                         log_target_scaled(x_chains[k], temps[k]))
            if np.log(np.random.rand()) < log_alpha:
                x_chains[k] = x_prop

        # Step 2: 尝试交换相邻链
        for k in range(n_chains - 1):
            n_swaps += 1
            # 交换 k 和 k+1 的 proposal
            x_a, x_b = x_chains[k], x_chains[k+1]
            T_a, T_b = temps[k], temps[k+1]

            # 交换接受率 (满足 detailed balance)
            log_alpha = ((1/T_a - 1/T_b) *
                         (log_target_scaled(x_b, 1.0) * T_b -
                          log_target_scaled(x_a, 1.0) * T_a))

            if np.log(np.random.rand()) < log_alpha:
                x_chains[k], x_chains[k+1] = x_b, x_a
                n_accepted_swaps += 1

        samples_cold[i] = x_chains[0]

    post_cold = samples_cold[burn:]

    # RW-MH baseline (single chain, no tempering)
    x_single = 0.0
    single_samples = np.zeros(total_steps)
    for i in range(total_steps):
        x_prop = x_single + np.random.randn() * 0.5
        if np.log(np.random.rand()) < log_target_scaled(x_prop, 1.0) - log_target_scaled(x_single, 1.0):
            x_single = x_prop
        single_samples[i] = x_single
    post_single = single_samples[burn:]

    # --- Results ---
    swap_rate = n_accepted_swaps / n_swaps if n_swaps > 0 else 0

    # Count mode visits (mode1: x<-1, mode2: x>1)
    mode1_cold = np.sum(post_cold < -1)
    mode2_cold = np.sum(post_cold > 1)
    mode1_single = np.sum(post_single < -1)
    mode2_single = np.sum(post_single > 1)

    print(f"\n  目标: P(x) ∝ N(-4,0.8) + N(4,0.8)  (双峰, mode 间隔 8)")
    print(f"  温度阶梯: T = {temps}")
    print(f"\n  -- 模式探索对比 --")
    print(f"  {'方法':>18s}  {'mode(-4)':>10s}  {'mode(+4)':>10s}  {'-4比例':>8s}")
    print(f"  {'-'*18}  {'-'*10}  {'-'*10}  {'-'*8}")
    print(f"  {'RW-MH (单链)':>18s}  {mode1_single:>10d}  {mode2_single:>10d}  "
          f"{mode1_single/len(post_single):>8.1%}")
    print(f"  {'Parallel Tempering':>18s}  {mode1_cold:>10d}  {mode2_cold:>10d}  "
          f"{mode1_cold/len(post_cold):>8.1%}")

    print(f"\n  交换接受率: {swap_rate:.1%}")

    # 判断质量
    ratio_cold = mode1_cold / max(mode2_cold, 1)
    ratio_single = mode1_single / max(mode2_single, 1)
    if 0.5 < ratio_cold < 2.0 and (ratio_single > 10 or ratio_single < 0.1):
        print(f"  ✅ Tempering 成功探索了两个 mode!")
        print(f"     RW-MH 几乎只看到 {' 左边' if mode1_single > mode2_single else ' 右边'} mode")
        print(f"     (ratio = {ratio_single:.0f}:1 vs 期望 1:1)")
    else:
        print(f"  Tempering mode ratio = {ratio_cold:.1f}:1 (期望 ~1:1)")

    print("\n  🎯 洞察:")
    print("    高温链 (T>>1): 分布更平 → mode 之间壁垒降低 → 容易跳跃")
    print("    链间交换: 把高温链探索到的 mode 信息传递给低温链")
    print("    关键: T₁=1 链给出精确后验, 高温链帮助它看到所有 mode")


# ============================================================================
# 练习 4: Annealed Importance Sampling (AIS)
# ============================================================================

def exercise4_annealed_importance_sampling():
    """
    用 AIS 估计归一化常数的比值 Z_target / Z_ref。

    目标: P_target(x) = N(4, 1) (归一化常数 Z_target = 1)
    参考: P_ref(x) = N(0, 1)   (归一化常数 Z_ref = 1)
    → 真实 log ratio = log(1/1) = 0
    """
    print("=" * 70)
    print("练习 4: Annealed Importance Sampling — 估计 Z 比值")
    print("=" * 70)

    def log_ref(x):
        return -0.5 * x**2  # log N(0,1), unnormalized

    def log_target(x):
        return -0.5 * (x - 4)**2  # log N(4,1), unnormalized

    # 退火: P_k(x) ∝ P_ref^{1-β_k} · P_target^{β_k}
    n_intermediate = 50
    betas = np.linspace(0, 1, n_intermediate + 1)

    n_runs = 200

    def log_intermediate(x, beta):
        return (1 - beta) * log_ref(x) + beta * log_target(x)

    # AIS: 对每个 run, 从 P_ref 采样, 逐步过渡到 P_target
    log_weights = np.zeros(n_runs)
    sigma_mh = 1.0  # MH within each intermediate distribution

    for run in range(n_runs):
        # Start from P_ref (beta=0)
        x = np.random.randn()  # N(0,1)

        log_w = 0.0

        # Transition through intermediates
        for k in range(n_intermediate):
            beta_cur = betas[k]
            beta_next = betas[k + 1]

            # log weight contribution: log P_{k+1}(x) - log P_k(x)
            log_w += (log_intermediate(x, beta_next) -
                      log_intermediate(x, beta_cur))

            # MCMC transition within P_{k+1} (RW-MH)
            x_prop = x + np.random.randn() * sigma_mh
            log_alpha = (log_intermediate(x_prop, beta_next) -
                         log_intermediate(x, beta_next))
            if np.log(np.random.rand()) < log_alpha:
                x = x_prop

        log_weights[run] = log_w

    # Estimate log(Z_target/Z_ref)
    log_weights_stable = log_weights - np.max(log_weights)
    log_ratio_est = np.log(np.mean(np.exp(log_weights_stable))) + np.max(log_weights)

    # Monte Carlo standard error
    w_norm = np.exp(log_weights_stable)
    w_norm /= w_norm.sum()
    ess = 1.0 / np.sum(w_norm**2)

    print(f"\n  目标: P_target = N(4,1), 参考: P_ref = N(0,1)")
    print(f"  真实 log(Z_target/Z_ref) = log(1/1) = 0")
    print(f"  中间分布数: {n_intermediate}, AIS runs: {n_runs}")

    print(f"\n  -- AIS 结果 --")
    print(f"  log(Z_target/Z_ref) 估计 = {log_ratio_est:.4f} (真实 = 0)")
    print(f"  估计误差 = {abs(log_ratio_est):.4f}")
    print(f"  ESS (权重) = {ess:.0f} / {n_runs} ({ess/n_runs:.1%})")

    # 对比: 直接 Importance Sampling
    z_ref = np.random.randn(n_runs)
    log_w_direct = log_target(z_ref) - log_ref(z_ref)
    log_w_direct_s = log_w_direct - np.max(log_w_direct)
    log_ratio_direct = np.log(np.mean(np.exp(log_w_direct_s))) + np.max(log_w_direct)
    w_dir_norm = np.exp(log_w_direct_s)
    w_dir_norm /= w_dir_norm.sum()
    ess_direct = 1.0 / np.sum(w_dir_norm**2)

    print(f"\n  -- 对比: 直接 Importance Sampling --")
    print(f"  log(Z_target/Z_ref) 估计 = {log_ratio_direct:.4f}")
    print(f"  ESS (直接IS) = {ess_direct:.0f} / {n_runs} ({ess_direct/n_runs:.1%})")

    if ess > ess_direct:
        print(f"  ✅ AIS ESS 比直接 IS 高 {ess/ess_direct:.1f}x — 退火有效!")

    print("\n  🎯 洞察:")
    print("    AIS 通过逐步'退火'从简单→复杂 → 权重退化远小于直接 IS")
    print("    中间分布数越多 → 权重越均匀 → 估计越精确")
    print("    应用: 模型选择 (估计 marginal likelihood) 的黄金标准")


# ============================================================================
# 练习 5: MCMC 收敛诊断
# ============================================================================

def exercise5_mcmc_diagnostics():
    """
    跑多条 MCMC 链, 计算 R-hat, ESS, MCSE 等诊断指标。
    展示如何判断链是否收敛。

    场景: 采样 2D 相关 Gaussian, 从不同起点跑 4 条链。
    """
    print("=" * 70)
    print("练习 5: MCMC 收敛诊断 — R-hat, ESS, MCSE")
    print("=" * 70)

    mean_true = np.array([1.0, -1.0])
    cov_true = np.array([[1.0, 0.7], [0.7, 1.0]])
    cov_inv = np.linalg.inv(cov_true)

    def log_prob(x):
        d = x - mean_true
        return -0.5 * d @ cov_inv @ d

    n_chains = 4
    n_samples = 2000
    burn = 200

    # 各链从不同起点
    starts = np.array([[5.0, 5.0], [-5.0, -5.0], [5.0, -5.0], [-5.0, 5.0]])

    all_chains = np.zeros((n_chains, n_samples, 2))
    sigma_rw = 0.3

    for c in range(n_chains):
        x = starts[c].copy()
        for i in range(burn + n_samples):
            x_prop = x + np.random.randn(2) * sigma_rw
            if np.log(np.random.rand()) < log_prob(x_prop) - log_prob(x):
                x = x_prop
            if i >= burn:
                all_chains[c, i - burn] = x

    # --- Compute diagnostics for x1 ---
    chains_x1 = all_chains[:, :, 0]  # (n_chains, n_samples)

    # R-hat (Gelman-Rubin)
    n = n_samples
    m = n_chains

    chain_means = chains_x1.mean(axis=1)   # per-chain means
    chain_vars = chains_x1.var(axis=1, ddof=1)  # per-chain variances

    B = n / (m - 1) * np.sum((chain_means - chain_means.mean())**2)  # between-chain
    W = np.mean(chain_vars)  # within-chain
    var_plus = (n - 1) / n * W + B / n
    R_hat = np.sqrt(var_plus / W)

    # ESS (pooled)
    pooled_chain = chains_x1.flatten()
    ess_total = compute_ess(pooled_chain)

    # MCSE
    mcse = np.sqrt(var_plus / ess_total)

    print(f"\n  目标: N(μ=[1,-1], Σ=[[1,0.7],[0.7,1]])")
    print(f"  {m} 条链, 各 {n} 样本 (post burn-in)")

    print(f"\n  -- 诊断结果 (变量 X₁) --")
    print(f"  链间方差 B: {B:.6f}")
    print(f"  链内方差 W: {W:.6f}")
    print(f"  R̂ (Gelman-Rubin): {R_hat:.4f}")
    print(f"  {'✅ R̂ < 1.1: 链可能已收敛' if R_hat < 1.1 else '⚠️ R̂ > 1.1: 链可能未收敛!'}")

    print(f"\n  有效样本量 ESS: {ess_total:.0f} / {m * n} ({ess_total/(m*n):.1%})")
    print(f"  MC 标准误 MCSE: {mcse:.6f}")

    # Per-chain means
    print(f"\n  -- 各链统计 (X₁) --")
    print(f"  {'链':>4s}  {'均值':>10s}  {'方差':>10s}  {'ESS':>8s}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*8}")
    for c in range(m):
        ess_c = compute_ess(chains_x1[c])
        print(f"  {c+1:>4d}  {chain_means[c]:>10.4f}  {chain_vars[c]:>10.4f}  {ess_c:>8.0f}")

    print(f"  真实均值: {mean_true[0]}, 真实方差: {cov_true[0,0]}")

    # 检查是否有链卡住
    mean_spread = np.max(chain_means) - np.min(chain_means)
    if mean_spread > 0.5:
        print(f"\n  ⚠️ 链间均值范围 = {mean_spread:.3f} (偏大 → 可能需要更长的 burn-in)")
    else:
        print(f"\n  链间均值范围 = {mean_spread:.4f} (各链一致) ✅")

    print("\n  🎯 洞察:")
    print("    R̂ ≈ 1: 链间方差 ≈ 链内方差 → 所有链'在同一个分布中'")
    print("    ESS << n: 样本自相关 → 有效信息少 (需要 thinning 或更好的 sampler)")
    print("    MCSE = σ/√ESS: 估计的精度 — MCSE 越小, 蒙特卡洛误差越小")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    run_all = '--ex' not in sys.argv

    exercises = [
        ('1', exercise1_slice_sampling),
        ('2', exercise2_hamiltonian_monte_carlo),
        ('3', exercise3_parallel_tempering),
        ('4', exercise4_annealed_importance_sampling),
        ('5', exercise5_mcmc_diagnostics),
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
