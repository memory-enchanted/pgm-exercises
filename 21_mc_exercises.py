"""
=============================================================================
  CMU 10-708 L9 代码练习: 蒙特卡洛方法 — Rejection, Importance, MCMC
=============================================================================

本文件包含 5 个代码练习:

  练习 1: Rejection Sampling — 从混合高斯中采样, 观察接受率
  练习 2: Importance Sampling — 加权样本, 对比不同 proposal 的 ESS
  练习 3: Metropolis-Hastings — 2D 相关高斯, trace plot, 收敛分析
  练习 4: Gibbs Sampling — 二元高斯, 精确 full conditional, 对比 MH
  练习 5: MCMC for Bayesian Network — pgmpy GibbsSampling vs VE

使用方法:
  python 21_mc_exercises.py           # 运行全部练习
  python 21_mc_exercises.py --ex 1    # 只运行练习1

依赖: numpy, pgmpy (仅练习5)
=============================================================================
"""

import numpy as np
import pandas as pd
import sys

# 修复 Windows GBK 终端下 emoji 打印问题
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

np.random.seed(42)


# ============================================================================
# 练习 1: Rejection Sampling
# ============================================================================

def exercise1_rejection_sampling():
    """
    从混合高斯分布 P(x) = 0.3*N(-2,0.5) + 0.7*N(2,0.8) 中采样,
    使用建议分布 Q(x) = N(0, 3^2)。
    展示接受率和采样效率。
    """
    print("=" * 70)
    print("练习 1: Rejection Sampling — 混合高斯")
    print("=" * 70)

    # 目标分布 P(x) (未归一化也可以 — rejection 只需要比例)
    def target_log_pdf(x):
        """log P(x) = log(0.3*N(-2,0.5) + 0.7*N(2,0.8))"""
        log_p1 = -0.5 * ((x + 2) / np.sqrt(0.5))**2 - 0.5 * np.log(2 * np.pi * 0.5)
        log_p2 = -0.5 * ((x - 2) / np.sqrt(0.8))**2 - 0.5 * np.log(2 * np.pi * 0.8)
        return np.log(0.3 * np.exp(log_p1) + 0.7 * np.exp(log_p2))

    # 建议分布 Q(x) = N(0, 3^2)
    def proposal_log_pdf(x):
        return -0.5 * (x / 3.0)**2 - 0.5 * np.log(2 * np.pi * 9)

    def sample_proposal(n):
        return np.random.randn(n) * 3.0

    # 找到 M 使得 M*Q(x) >= P(x) for all x
    # 网格搜索
    x_grid = np.linspace(-10, 10, 2000)
    log_p = target_log_pdf(x_grid)
    log_q = proposal_log_pdf(x_grid)
    log_ratio = log_p - log_q
    log_M = np.max(log_ratio)
    M = np.exp(log_M)

    print(f"\n  目标: P(x) = 0.3*N(-2,0.5) + 0.7*N(2,0.8)")
    print(f"  建议: Q(x) = N(0, 3^2)")
    print(f"  包络常数 M = exp(max(log P - log Q)) = {M:.4f}")
    print(f"  理论接受率 = 1/M = {1/M:.4f} ({1/M*100:.1f}%)")

    # Rejection Sampling
    n_try = 5000
    x_proposals = sample_proposal(n_try)
    log_p_vals = target_log_pdf(x_proposals)
    log_q_vals = proposal_log_pdf(x_proposals)
    log_u = np.log(np.random.rand(n_try))

    # accept if log_u < log_p - log_q - log_M
    accepted_mask = log_u < (log_p_vals - log_q_vals - log_M)
    accepted_samples = x_proposals[accepted_mask]
    n_accepted = np.sum(accepted_mask)
    acc_rate = n_accepted / n_try

    print(f"\n  实际接受率: {acc_rate:.4f} ({acc_rate*100:.1f}%)")
    print(f"  获得 {n_accepted} 个有效样本 (总提案 {n_try} 次)")

    # 对比: 直接从混合分布采样 (ground truth)
    n_gt = 2000
    z_gt = np.random.binomial(1, 0.7, n_gt)  # 0->comp1, 1->comp2
    samples_gt = np.where(z_gt,
                          np.random.randn(n_gt) * np.sqrt(0.8) + 2.0,
                          np.random.randn(n_gt) * np.sqrt(0.5) - 2.0)

    print(f"\n  -- 样本统计对比 --")
    print(f"  {'':>15s}  {'均值':>10s}  {'标准差':>10s}")
    print(f"  {'-'*15}  {'-'*10}  {'-'*10}")
    print(f"  {'Ground Truth':>15s}  {np.mean(samples_gt):>10.4f}  {np.std(samples_gt):>10.4f}")
    print(f"  {'Rejection':>15s}  {np.mean(accepted_samples):>10.4f}  {np.std(accepted_samples):>10.4f}")

    # 效率分析
    rejection_waste = n_try - n_accepted
    print(f"\n  丢弃了 {rejection_waste} 个提案 ({rejection_waste/n_try*100:.1f}%)")
    print(f"  效率: 每 {1/acc_rate:.1f} 个提案得到 1 个有效样本")

    print("\n  🎯 洞察:")
    print("    Rejection Sampling = 简单但低效 (尤其高维)")
    print("    关键: 找到紧的包络 M (M 越小越好)")
    print("    维度灾难: D 维中, 接受率 ~ (1/M)^D → 指数衰减!")


# ============================================================================
# 练习 2: Importance Sampling
# ============================================================================

def exercise2_importance_sampling():
    """
    用 Importance Sampling 估计 E[x^2] where x ~ Mixture of Gaussians.
    对比两个 proposal:
      - Good: Q = N(2, 2^2) (接近 P 的质量)
      - Bad:  Q = N(-6, 0.5^2) (远离 P)

    展示 ESS 的差异。
    """
    print("=" * 70)
    print("练习 2: Importance Sampling — 权重退化 & ESS")
    print("=" * 70)

    # 目标: P(x) = 0.3*N(-2,0.5) + 0.7*N(2,0.8)
    def target_log_pdf(x):
        log_p1 = -0.5 * ((x + 2) / np.sqrt(0.5))**2
        log_p2 = -0.5 * ((x - 2) / np.sqrt(0.8))**2
        max_log = np.maximum(log_p1, log_p2)
        return np.log(0.3 * np.exp(log_p1 - max_log) + 0.7 * np.exp(log_p2 - max_log)) + max_log

    # 真实 E[x^2] (用大量样本估计)
    n_ref = 100000
    z_ref = np.random.binomial(1, 0.7, n_ref)
    x_ref = np.where(z_ref, np.random.randn(n_ref)*np.sqrt(0.8)+2,
                     np.random.randn(n_ref)*np.sqrt(0.5)-2)
    true_ex2 = np.mean(x_ref**2)
    print(f"\n  目标: P(x) = 0.3*N(-2,0.5) + 0.7*N(2,0.8)")
    print(f"  真实 E[x^2] ≈ {true_ex2:.4f} (MC 参考, n={n_ref})")

    n_samples = 2000

    # --- Good Proposal: Q1 = N(2, 2^2) ---
    mu1, sig1 = 2.0, 2.0
    z1 = np.random.randn(n_samples) * sig1 + mu1
    log_p1 = target_log_pdf(z1)
    log_q1 = -0.5 * ((z1 - mu1)/sig1)**2 - np.log(sig1 * np.sqrt(2*np.pi))
    log_w1 = log_p1 - log_q1
    w1 = np.exp(log_w1 - np.max(log_w1))
    w1_norm = w1 / w1.sum()
    est1 = np.sum(w1_norm * z1**2)
    ess1 = 1.0 / np.sum(w1_norm**2)

    # --- Bad Proposal: Q2 = N(-6, 0.5^2) ---
    mu2, sig2 = -6.0, 0.5
    z2 = np.random.randn(n_samples) * sig2 + mu2
    log_p2 = target_log_pdf(z2)
    log_q2 = -0.5 * ((z2 - mu2)/sig2)**2 - np.log(sig2 * np.sqrt(2*np.pi))
    log_w2 = log_p2 - log_q2
    w2 = np.exp(log_w2 - np.max(log_w2))
    w2_norm = w2 / w2.sum()
    est2 = np.sum(w2_norm * z2**2)
    ess2 = 1.0 / np.sum(w2_norm**2)

    print(f"\n  -- 结果 (n={n_samples}) --")
    print(f"  {'Proposal':>20s}  {'E[x^2] est':>12s}  {'ESS':>8s}  {'ESS/n':>8s}")
    print(f"  {'-'*20}  {'-'*12}  {'-'*8}  {'-'*8}")
    print(f"  {'Good: N(2,2^2)':>20s}  {est1:>12.4f}  {ess1:>8.1f}  {ess1/n_samples:>8.1%}")
    print(f"  {'Bad: N(-6,0.5^2)':>20s}  {est2:>12.4f}  {ess2:>8.1f}  {ess2/n_samples:>8.1%}")

    # 展示权重集中度
    print(f"\n  -- 权重集中度 --")
    top5_ratio_good = np.sum(np.sort(w1_norm)[-5:])
    top5_ratio_bad = np.sum(np.sort(w2_norm)[-5:])
    print(f"  Good proposal: 前5大权重占比 = {top5_ratio_good:.1%}")
    print(f"  Bad proposal:  前5大权重占比 = {top5_ratio_bad:.1%}")
    if top5_ratio_bad > 0.9:
        print(f"  ⚠️ Bad proposal 几乎所有权重集中在极少样本 → 估计不可靠")

    print("\n  🎯 洞察:")
    print("    Importance Sampling = 用权重修正 proposal-target 不匹配")
    print("    ESS 衡量'等效独立样本数' — ESS << n 意味着估计不可靠")
    print("    好 proposal: 覆盖 P 的高概率区域 → 权重均匀 → ESS 大")


# ============================================================================
# 练习 3: Metropolis-Hastings
# ============================================================================

def exercise3_metropolis_hastings():
    """
    在 2D 相关 Gaussian 上运行 Random-Walk MH。
    展示: trace plot, 接受率, 收敛, proposal 宽度的影响。
    """
    print("=" * 70)
    print("练习 3: Metropolis-Hastings — 2D 相关高斯")
    print("=" * 70)

    # 目标: 2D 相关 Gaussian
    mean_true = np.array([2.0, 3.0])
    cov_true = np.array([[1.0, 0.8], [0.8, 1.0]])
    cov_inv = np.linalg.inv(cov_true)

    def log_target(x):
        d = x - mean_true
        return -0.5 * d @ cov_inv @ d

    # 运行三条不同 proposal 宽度的链
    sigmas = [0.1, 0.5, 2.0]
    n_steps = 3000
    burn_in = 1000

    print(f"\n  目标: N(μ={mean_true}, Σ=[[1,0.8],[0.8,1]])")
    print(f"  链长: {n_steps}, burn-in: {burn_in}")
    print(f"\n  -- 不同 proposal 宽度的表现 --")
    print(f"  {'σ_prop':>8s}  {'接受率':>8s}  {'均值估计':>16s}  {'协方差(1,2)':>12s}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*16}  {'-'*12}")

    final_samples = {}

    for sigma in sigmas:
        x = np.array([0.0, 0.0])  # 初始点
        samples = np.zeros((n_steps, 2))
        accepted = 0

        for i in range(n_steps):
            # Random walk proposal
            x_prop = x + np.random.randn(2) * sigma
            log_alpha = log_target(x_prop) - log_target(x)

            if np.log(np.random.rand()) < log_alpha:
                x = x_prop.copy()
                accepted += 1

            samples[i] = x

        acc_rate = accepted / n_steps
        post_burn = samples[burn_in:]
        est_mean = post_burn.mean(axis=0)
        est_cov = np.cov(post_burn.T)

        final_samples[sigma] = post_burn

        print(f"  {sigma:>8.2f}  {acc_rate:>8.1%}  "
              f"({est_mean[0]:.3f},{est_mean[1]:.3f})  "
              f"{est_cov[0,1]:>12.4f}")

    print(f"\n  真实: μ=({mean_true[0]},{mean_true[1]}), Σ[0,1]={cov_true[0,1]}")

    # 最佳 sigma 的分析
    best_sigma = 0.5
    best_samples = final_samples[best_sigma]
    post_mean = best_samples.mean(axis=0)
    post_cov = np.cov(best_samples.T)

    print(f"\n  -- σ={best_sigma} 链的诊断 --")
    print(f"  后验均值: ({post_mean[0]:.4f}, {post_mean[1]:.4f})")
    print(f"  真实均值: ({mean_true[0]}, {mean_true[1]})")
    print(f"  后验协方差: [[{post_cov[0,0]:.3f},{post_cov[0,1]:.3f}],")
    print(f"               [{post_cov[1,0]:.3f},{post_cov[1,1]:.3f}]]")

    # 自相关分析
    chain_x1 = best_samples[:, 0]
    # 简单自相关
    def autocorr(x, lag):
        xc = x - x.mean()
        return np.correlate(xc, xc, mode='full')[len(xc)-1+lag] / (np.var(xc) * len(xc))

    ac_lag1 = autocorr(chain_x1, 1)
    ac_lag5 = autocorr(chain_x1, 5)
    # ESS approx: n / (1 + 2*sum ac)
    ac_sum = 0
    for lag in range(1, 20):
        ac = autocorr(chain_x1, lag)
        if ac < 0.05:
            break
        ac_sum += ac
    ess_est = len(chain_x1) / (1 + 2 * ac_sum)

    print(f"\n  自相关 (lag 1): {ac_lag1:.4f}")
    print(f"  自相关 (lag 5): {ac_lag5:.4f}")
    print(f"  近似 ESS: {ess_est:.0f} (共 {len(chain_x1)} 样本)")

    print("\n  🎯 洞察:")
    print("    σ 太小 → 接受率高但混合慢 (高自相关)")
    print("    σ 太大 → 接受率低, 链卡住")
    print("    最优 σ ≈ 后验标准差的 0.2-0.5 倍 (产生 ~23% 接受率)")
    print("    MCMC 样本相关 → ESS < 实际样本数")


# ============================================================================
# 练习 4: Gibbs Sampling
# ============================================================================

def exercise4_gibbs_sampling():
    """
    在 2D 相关 Gaussian 上实现 Gibbs 采样, 与 MH 对比。

    Gibbs: 轮流从 P(x1|x2) 和 P(x2|x1) 中采样
    接受率 = 100% (但每步只移动一个维度)

    对比: MH 可以沿任意方向移动, 但有拒绝
    """
    print("=" * 70)
    print("练习 4: Gibbs Sampling — 二元高斯 (full conditionals)")
    print("=" * 70)

    # 目标: 2D 相关 Gaussian
    mu = np.array([2.0, 3.0])
    cov = np.array([[1.0, 0.8], [0.8, 1.0]])

    # 精确的 full conditionals:
    # x1 | x2 ~ N(mu1 + cov12/cov22*(x2-mu2), cov11 - cov12^2/cov22)
    # x2 | x1 ~ N(mu2 + cov12/cov11*(x1-mu1), cov22 - cov12^2/cov11)
    cond1_var = cov[0,0] - cov[0,1]**2 / cov[1,1]  # 1 - 0.64/1 = 0.36
    cond1_slope = cov[0,1] / cov[1,1]  # 0.8 / 1 = 0.8

    cond2_var = cov[1,1] - cov[0,1]**2 / cov[0,0]  # 1 - 0.64/1 = 0.36
    cond2_slope = cov[0,1] / cov[0,0]  # 0.8 / 1 = 0.8

    n_steps = 3000
    burn_in = 500

    x1, x2 = 0.0, 0.0
    samples = np.zeros((n_steps, 2))

    for i in range(n_steps):
        # Gibbs step: sample x1 | x2
        mean1 = mu[0] + cond1_slope * (x2 - mu[1])
        x1 = np.random.randn() * np.sqrt(cond1_var) + mean1

        # Gibbs step: sample x2 | x1
        mean2 = mu[1] + cond2_slope * (x1 - mu[0])
        x2 = np.random.randn() * np.sqrt(cond2_var) + mean2

        samples[i] = [x1, x2]

    post_burn = samples[burn_in:]
    est_mean = post_burn.mean(axis=0)
    est_cov = np.cov(post_burn.T)

    print(f"\n  目标: N(μ=[2,3], Σ=[[1,0.8],[0.8,1]])")
    print(f"  Full conditionals:")
    print(f"    X1|X2 ~ N(2 + 0.8*(X2-3), 0.36)")
    print(f"    X2|X1 ~ N(3 + 0.8*(X1-2), 0.36)")

    print(f"\n  -- Gibbs 结果 (n={n_steps}, burn={burn_in}) --")
    print(f"  后验均值: ({est_mean[0]:.4f}, {est_mean[1]:.4f})")
    print(f"  真实均值: ({mu[0]}, {mu[1]})")
    print(f"  后验 Σ: [[{est_cov[0,0]:.4f},{est_cov[0,1]:.4f}],")
    print(f"          [{est_cov[1,0]:.4f},{est_cov[1,1]:.4f}]]")

    # 自相关
    def autocorr(x, lag):
        xc = x - x.mean()
        return np.correlate(xc, xc, mode='full')[len(xc)-1+lag] / (np.var(xc) * len(xc))

    ac_gibbs = autocorr(post_burn[:,0], 1)
    print(f"\n  Gibbs lag-1 自相关: {ac_gibbs:.4f}")

    # 对比: 用 σ=0.5 的 MH (练习3 已验证)
    x_mh = np.array([0.0, 0.0])
    samples_mh = np.zeros((n_steps, 2))
    cov_inv = np.linalg.inv(cov)
    def log_target(x):
        d = x - mu
        return -0.5 * d @ cov_inv @ d
    for i in range(n_steps):
        x_prop = x_mh + np.random.randn(2) * 0.5
        if np.log(np.random.rand()) < log_target(x_prop) - log_target(x_mh):
            x_mh = x_prop.copy()
        samples_mh[i] = x_mh

    post_burn_mh = samples_mh[burn_in:]
    ac_mh = autocorr(post_burn_mh[:,0], 1)

    # 每步移动距离
    step_dist_gibbs = np.mean(np.sqrt(np.sum(np.diff(post_burn, axis=0)**2, axis=1)))
    step_dist_mh = np.mean(np.sqrt(np.sum(np.diff(post_burn_mh, axis=0)**2, axis=1)))

    print(f"  MH   lag-1 自相关: {ac_mh:.4f}")
    print(f"\n  -- 移动效率对比 --")
    print(f"  Gibbs 平均每步移动: {step_dist_gibbs:.4f}")
    print(f"  MH    平均每步移动: {step_dist_mh:.4f}")
    print(f"  注意: Gibbs 每步变 2 个变量, MH 每步变 2 个变量 — 公平对比")

    print("\n  🎯 洞察:")
    print("    Gibbs = 100% 接受率, 但只能沿坐标轴移动")
    print("    相关性强时: Gibbs → zig-zag → 混合慢 (高自相关)")
    print("    MH 可以沿任意方向 → 但需要调 proposal")
    print("    两者互补: Gibbs 不需调参, MH 更灵活")


# ============================================================================
# 练习 5: MCMC for Bayesian Network
# ============================================================================

def exercise5_mcmc_bayesian_network():
    """
    使用 pgmpy 的 GibbsSampling 在一个小 Bayesian Network 上采样,
    对比采样的边际与 VE 的精确边际。
    """
    print("=" * 70)
    print("练习 5: MCMC for Bayesian Network — GibbsSampling vs VE")
    print("=" * 70)

    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.sampling import GibbsSampling
    from pgmpy.inference import VariableElimination

    # 构建学生网络
    model = DiscreteBayesianNetwork([
        ('Difficulty', 'Grade'),
        ('Intelligence', 'Grade'),
        ('Intelligence', 'SAT'),
        ('Grade', 'Letter'),
    ])

    model.add_cpds(
        TabularCPD('Difficulty', 2, [[0.6], [0.4]]),
        TabularCPD('Intelligence', 2, [[0.7], [0.3]]),
        TabularCPD('Grade', 3,
                   [[0.3, 0.05, 0.9, 0.5],
                    [0.4, 0.25, 0.08, 0.3],
                    [0.3, 0.7, 0.02, 0.2]],
                   evidence=['Difficulty', 'Intelligence'],
                   evidence_card=[2, 2]),
        TabularCPD('SAT', 2,
                   [[0.95, 0.2], [0.05, 0.8]],
                   evidence=['Intelligence'], evidence_card=[2]),
        TabularCPD('Letter', 2,
                   [[0.1, 0.4, 0.7],
                    [0.9, 0.6, 0.3]],
                   evidence=['Grade'], evidence_card=[3]),
    )
    model.check_model()

    # 精确推断 (VE)
    ve = VariableElimination(model)

    print(f"\n  学生网络: D→G←I, I→S, G→L")
    print(f"  变量: D(2), I(2), G(3), S(2), L(2)")

    # --- Gibbs 采样 (无证据) ---
    print(f"\n  -- Gibbs 采样 (无证据, n=5000, burn=500) --")
    gibbs = GibbsSampling(model)
    samples_df = gibbs.sample(size=5000)

    # pgmpy returns a pd.DataFrame; convert to records
    if isinstance(samples_df, pd.DataFrame):
        samples_list = samples_df.to_dict('records')
    else:
        # generator case
        samples_list = list(samples_df)

    all_vars = list(samples_list[0].keys())
    samples_dict = {v: np.array([s[v] for s in samples_list]) for v in all_vars}
    burn = 500
    post_burn = {v: samples_dict[v][burn:] for v in all_vars}

    print(f"  -- 边际分布对比 --")
    print(f"  {'变量':>12s}  {'VE (精确)':>20s}  {'MCMC (Gibbs)':>20s}  {'匹配':>6s}")
    print(f"  {'-'*12}  {'-'*20}  {'-'*20}  {'-'*6}")

    for var in ['Difficulty', 'Intelligence', 'SAT', 'Letter']:
        ve_result = ve.query([var], show_progress=False)
        ve_vals = ve_result.values

        # MCMC 估计
        n_states = ve_result.cardinality[0]
        mc_counts = np.zeros(int(n_states))
        for val in post_burn[var]:
            mc_counts[int(val)] += 1
        mc_vals = mc_counts / mc_counts.sum()

        match = np.allclose(ve_vals, mc_vals, atol=0.03)
        ve_str = np.array2string(ve_vals, precision=3, suppress_small=True)
        mc_str = np.array2string(mc_vals, precision=3, suppress_small=True)
        print(f"  {var:>12s}  {ve_str:>20s}  {mc_str:>20s}  {'✅' if match else '⚠️':>6s}")

    # --- Gibbs 采样 (有证据) ---
    print(f"\n  -- Gibbs 采样 (有证据: Letter=0, n=5000) --")
    # Rejection-like: sample from joint, filter
    samples_with_ev = []
    n_need = 5000
    n_try = 0
    ev_samples_df = gibbs.sample(size=20000)
    if isinstance(ev_samples_df, pd.DataFrame):
        ev_records = ev_samples_df.to_dict('records')
    else:
        ev_records = list(ev_samples_df)
    for s in ev_records:
        n_try += 1
        if s['Letter'] == 0:
            samples_with_ev.append(s)
        if len(samples_with_ev) >= n_need:
            break

    ev_dict = {v: np.array([s[v] for s in samples_with_ev]) for v in all_vars}
    ev_burn = {v: ev_dict[v][500:] for v in all_vars}

    print(f"  需要 {n_try} 个样本来获得 {n_need} 个 Letter=0 的样本")
    print(f"  实际 P(Letter=0) = {ve.query(['Letter'], show_progress=False).values[0]:.3f}")

    print(f"\n  -- 有证据时边际对比 --")
    print(f"  {'变量':>12s}  {'VE (精确)':>20s}  {'MCMC':>20s}  {'匹配':>6s}")
    print(f"  {'-'*12}  {'-'*20}  {'-'*20}  {'-'*6}")

    for var in ['Difficulty', 'Intelligence', 'Grade']:
        ve_result = ve.query([var], evidence={'Letter': 0}, show_progress=False)
        ve_vals = ve_result.values

        n_states = ve_result.cardinality[0]
        mc_counts = np.zeros(int(n_states))
        for val in ev_burn[var]:
            mc_counts[int(val)] += 1
        mc_vals = mc_counts / mc_counts.sum()

        match = np.allclose(ve_vals, mc_vals, atol=0.05)
        ve_str = np.array2string(ve_vals, precision=3, suppress_small=True)
        mc_str = np.array2string(mc_vals, precision=3, suppress_small=True)
        print(f"  {var:>12s}  {ve_str:>20s}  {mc_str:>20s}  {'✅' if match else '⚠️':>6s}")

    print("\n  🎯 洞察:")
    print("    Gibbs 在离散 Bayesian Network 上天然适用 — full conditionals 易得")
    print("    有证据时: 需要大量采样因为 Gibbs 可能要穿越低概率区域")
    print("    MCMC ≈ 渐近精确 — 样本足够多时误差 → 0")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    run_all = '--ex' not in sys.argv

    exercises = [
        ('1', exercise1_rejection_sampling, False),
        ('2', exercise2_importance_sampling, False),
        ('3', exercise3_metropolis_hastings, False),
        ('4', exercise4_gibbs_sampling, False),
        ('5', exercise5_mcmc_bayesian_network, True),
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
