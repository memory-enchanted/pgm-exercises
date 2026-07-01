# CMU 10-708 Lecture 9 课后练习 & 答案

> 配套教材: Murphy Ch.23-24, Bishop Ch.11
>
> 题目分为三级: 🟢 基础 | 🟡 进阶 | 🔴 挑战

---

## 🟢 基础题 (必须掌握)

### Q1. MC 估计的收敛速度

用 Monte Carlo 估计 E[f(x)] = ∫ f(x) P(x) dx。

**(A)** 由中心极限定理, MC 估计的误差以什么速率衰减？

**(B)** 若 f(x) 的方差为 σ², 需要多少个样本来使估计的 95% 置信区间宽度 < 0.01?

**(C)** 为什么 MC 不受维度灾难影响? (对比: 网格积分需要 O(1/h^D) 个点)

<details>
<summary>点击查看答案</summary>

**(A) 衰减速率:**

由 CLT: 样本均值 X̄_S = (1/S) Σ f(z_s) 满足:
```
√S (X̄_S - μ) → N(0, σ²)

误差 ~ O(σ/√S) = O(1/√S)
```

与维度 D 无关!

**(B) 所需样本:**

95% CI: X̄_S ± 1.96 × σ/√S

要求: 2 × 1.96 × σ/√S < 0.01 → √S > 392 σ → S > 153664 σ²

若 σ=1: S > ~154,000 个样本。

**(C) 维度无关性:**

Monte Carlo: 在概率集中区域采样 — 高维中 P 的大多数质量集中在典型集 (typical set) 中 → 采样效率不由空间体积决定, 而由概率集中度决定。

网格积分: 需要在 D 维空间中均匀铺点 → 点数 = (1/h)^D → 指数增长!

例如 D=100: 网格需要 10^100 个点 (不可能), MC 可能只需要 10^4-10^6 个样本。

</details>

---

### Q2. Rejection Sampling 的接受率

目标 P(x) = N(0, 1), 建议 Q(x) = N(0, 4)。

**(A)** 求包络常数 M, 使 M·Q(x) ≥ P(x) 对所有 x 成立。

**(B)** 理论接受率是多少？

**(C)** 如果改用 Q(x) = Cauchy(0, 1) (重尾分布), M 是增大还是减小?

<details>
<summary>点击查看答案</summary>

**(A) M 的求解:**

```
M = max_x P(x)/Q(x)

P(x) = (1/√(2π)) exp(-x²/2)
Q(x) = (1/(2√(2π))) exp(-x²/8)

P/Q = 2 exp(-x²/2 + x²/8) = 2 exp(-3x²/8)

最大值在 x=0: M = 2
```

**(B) 接受率:**

接受率 = 1/M = 1/2 = 50%

(实际接受率 ≈ 50%, 因为 Q 的宽度是 P 的 2 倍)

**(C) Cauchy(0,1) vs N(0,4):**

Cauchy PDF ~ 1/(π(1+x²)), 对于大 x, Cauchy ∝ 1/x², 而 Gaussian ∝ exp(-x²)。

Gaussian 尾部衰减比 Cauchy **快得多** → Cauchy 可以包络 Gaussian 但 M 需要很大 (因为 Cauchy 在中心附近密度低)。

实际上: 用 Cauchy 包络 Gaussian → M 会很大 (因中心附近 P/Q 比值大) → 接受率极低。

M 取决于"两个分布在哪儿最不成比例", 而不是尾巴的厚度。好的 Q 应该**形状相似** 于 P。

</details>

---

### Q3. Importance Sampling 的 ESS

用 Q 做 Importance Sampling 估计 E_P[f]。权重 w_s = P(z_s)/Q(z_s)。

**(A)** 写出有效样本量 ESS 的公式。

**(B)** 给定 5 个归一化权重: [0.5, 0.3, 0.1, 0.07, 0.03], 计算 ESS。

**(C)** 什么时候 ESS 会很小? 怎么预防?

<details>
<summary>点击查看答案</summary>

**(A) ESS 公式:**

```
ESS = (Σ w_s)² / Σ w_s² = 1 / Σ (w_s/Σ w)² = 1 / Σ w̃_s²

其中 w̃_s 是归一化权重。
```

**(B) 计算:**

```
ESS = 1 / (0.5² + 0.3² + 0.1² + 0.07² + 0.03²)
    = 1 / (0.25 + 0.09 + 0.01 + 0.0049 + 0.0009)
    = 1 / 0.3558
    = 2.81

共 5 样本, ESS=2.81 → 等效于 ~3 个独立样本
```

**(C) ESS 小的情况:**

当 Q 和 P 严重不匹配时 — 少数样本的 P(z)/Q(z) 比值极大, 霸占了全部权重。

**预防**:
1. 选择 proposal Q 使其覆盖 P 的高概率区域 (尝试 Q 的尾部比 P 重)
2. 使用 resampling (SIR — Sampling Importance Resampling)
3. 在高维中, Importance Sampling → 几乎必定失败; 改用 MCMC

</details>

---

### Q4. Metropolis-Hastings 的接受率推导

**(A)** 写出 MH 的接受概率 α(z→z*) 的公式。

**(B)** 证明 MH 满足 Detailed Balance: P(z)·Q(z*|z)·α(z→z*) = P(z*)·Q(z|z*)·α(z*→z)。

<details>
<summary>点击查看答案</summary>

**(A) MH 接受概率:**

```
α(z → z*) = min(1, [P(z*) · Q(z | z*)] / [P(z) · Q(z* | z)])
```

如果 Q 是对称的 (Q(z*|z) = Q(z|z*), 如 Random Walk):
```
α = min(1, P(z*)/P(z))  ← Metropolis (原始版本)
```

**(B) Detailed Balance 证明:**

```
P(z) · Q(z*|z) · α(z→z*)

代入 α:
= P(z) · Q(z*|z) · min(1, P(z*)·Q(z|z*) / (P(z)·Q(z*|z)))

Case 1: P(z*)·Q(z|z*) ≥ P(z)·Q(z*|z) → min=1
  = P(z) · Q(z*|z) · 1

Case 2: P(z*)·Q(z|z*) < P(z)·Q(z*|z) → min = ratio
  = P(z) · Q(z*|z) · P(z*)·Q(z|z*) / (P(z)·Q(z*|z))
  = P(z*) · Q(z|z*)

两种情况都等于 min(P(z)·Q(z*|z), P(z*)·Q(z|z*))

这个表达式对 z 和 z* 是对称的:
  = P(z*) · Q(z|z*) · α(z*→z)  ✓

Detailed Balance 成立!
```

</details>

---

## 🟡 进阶题

### Q5. Random Walk MH 的 Proposal 宽度

目标 P(x) = N(0, 1)。用 Random Walk MH: Q(x*|x) = N(x, σ²)。

**(A)** 写出 α(x → x*) 的表达式 (Q 对称, 所以简化)。

**(B)** 当 σ → 0 时, 接受率 → ? 链的行为如何?

**(C)** 当 σ → ∞ 时, 接受率 → ? 链的行为如何?

**(D)** 最优接受率的近似值是多少 (1D)?

<details>
<summary>点击查看答案</summary>

**(A) α(x→x*):**

Q 对称 → α = min(1, P(x*)/P(x))

```
α(x→x*) = min(1, exp(-(x*² - x²)/2))
        = min(1, exp(-(x* - x)(x* + x)/2))
```

**(B) σ → 0:**

x* 非常接近 x → P(x*)/P(x) ≈ 1 → α ≈ 1 (接受率 ~100%)

但链每次只移动极小的距离 → 需要大量步数才能探索整个空间 → **混合极慢** → ESS 极低。

**(C) σ → ∞:**

x* 几乎总是远离 x (在 P 概率极低的区域) → P(x*)/P(x) ≈ 0 → α ≈ 0

链几乎总被拒绝 → 卡在一个点不动 → **样本没有多样性**。

**(D) 最优接受率:**

1D Random Walk MH: 最优接受率 ≈ **44%**
高维 (D > 5) Random Walk MH: 最优接受率 ≈ **23.4%** (理论结果, Roberts et al. 1997)

实践中: 调整 σ 使接受率落在 20%–50% 之间通常效果不错。

</details>

---

### Q6. Gibbs 采样与 MH 的关系

**(A)** 证明 Gibbs 采样是 MH 的特例: 将 Gibbs 的 proposal Q_i(x*|x) = P(x_i* | x_{-i}) 代入 MH 接受率公式, 验证 α = 1。

**(B)** 用 Gibbs 采样一个 2D Gaussian: x ~ N(0, Σ), Σ = [[1, ρ], [ρ, 1]]。写出 Gibbs 每步的条件分布 P(x₁|x₂) 和 P(x₂|x₁)。

**(C)** 当 ρ → 1 (强相关) 时, Gibbs 的混合效率如何? 画草图解释 "zig-zag" 行为。

<details>
<summary>点击查看答案</summary>

**(A) 证明 α=1:**

Gibbs proposal: Q(x*|x) = P(x_i* | x_{-i}) × δ(x_{-i}* = x_{-i})

MH 接受率:
```
α = min(1, P(x*)·Q(x|x*) / (P(x)·Q(x*|x)))

注意: P(x*) = P(x_i*|x_{-i}*)·P(x_{-i}*)
由于 x_{-i}* = x_{-i} (Gibbs 不改动其他变量):
P(x*) = P(x_i*|x_{-i})·P(x_{-i})

P(x*)·Q(x|x*) = P(x_i*|x_{-i})·P(x_{-i}) · P(x_i|x_{-i})×1
P(x)·Q(x*|x) = P(x_i|x_{-i})·P(x_{-i}) · P(x_i*|x_{-i})×1

两者完全相等! → α = min(1, 1) = 1 ✓
```

**(B) 条件分布:**

对于二元 Gaussian N(0, Σ):
```
x₁|x₂ ~ N(ρ·x₂, 1-ρ²)
x₂|x₁ ~ N(ρ·x₁, 1-ρ²)
```

Gibbs 每步: 从上述条件分布中采样。

**(C) ρ→1 时的 zig-zag:**

当 ρ 接近 1: 变量几乎完全共线 (x₁ ≈ x₂)。

Gibbs 的 proposal 只能沿坐标轴移动 → 每一步只能改变一个维度 → 链沿等高线的窄"峡谷"走 zig-zag → 移动极其缓慢 → 需要极多步数才能遍历整个后验。

```
         x₂
         ↑
         |    / (ridge ρ≈1)
         |   /
         |  / ← Gibbs steps (tiny zig-zags)
         | /
         |/────────→ x₁
```

**解决方案**: 用 MH with 适当 proposal, 或重新参数化 (reparameterization) 解耦变量。

</details>

---

### Q7. 重要性采样的退化

用 Q = N(0, I_D) 做 Importance Sampling 估计 P = N(μ, I_D) 下的期望。

**(A)** 证明 log 权重 w(z) 的方差随 D 增大而增大。

**(B)** 估算 D 多大时 ESS ≈ 1 (即完全退化)。

<details>
<summary>点击查看答案</summary>

**(A) 方差分析:**

```
log w(z) = log P(z) - log Q(z)
         = -½||z-μ||² + ½||z||²
         = zᵀμ - ½||μ||²

z ~ Q → zᵀμ ~ N(0, ||μ||²)

Var[log w] = Var[zᵀμ] = ||μ||² = Σ μ_d²

对于固定 ||μ||² ∝ D → Var[log w] ∝ D → log 权重的标准差 ∝ √D
```

权重的方差随 D 线性增长 → 在高维中, 几乎所有权重集中在极少数样本上。

**(B) ESS ≈ 1 的临界 D:**

```
ESS = (Σ w_s)² / Σ w_s²

当 Var[log w] 很大时: ESS ≈ S · exp(-Var[log w]/2)

临界条件: Var[log w] ≈ 2 log S

对于 S=1000, ||μ||² = D·δ² (δ = 每维偏移):

D_crit ≈ 2 log(1000) / δ² ≈ 14 / δ²

若每维偏 1 个标准差 (δ=1): D_crit ≈ 14
若每维偏 0.1 个标准差 (δ=0.1): D_crit ≈ 1400
```

**启示**: 即使 Q 和 P 看起来"很接近", 在高维中 IS 也会退化。这就是为什么 MCMC 更有用。

</details>

---

## 🔴 挑战题

### Q8. Detailed Balance → Stationary Distribution

**(A)** 证明: 如果转移核 T 满足 detailed balance w.r.t. π, 则 π 是 T 的 stationary distribution。

**(B)** 满足 detailed balance 是 stationary 的**充分但不必要**条件。举一个 stationary but not detailed balanced 的转移核例子。

<details>
<summary>点击查看答案</summary>

**(A) 证明:**

```
(πT)(z') = ∫ π(z) T(z'|z) dz

由 detailed balance: π(z)·T(z'|z) = π(z')·T(z|z')
→ ∫ π(z) T(z'|z) dz = ∫ π(z') T(z|z') dz
                     = π(z') ∫ T(z|z') dz
                     = π(z') · 1 = π(z')

所以 (πT)(z') = π(z') → π 是 T 的不动点 → stationary distribution.
```

**(B) 非 reversible 的 stationary 例子:**

一个 3 状态 Markov 链:
```
T = [[0,   1,   0  ],
     [0,   0,   1  ],
     [1,   0,   0  ]]
```

Stationary distribution: π = [1/3, 1/3, 1/3]

验证: πT = [1/3, 1/3, 1/3] ✓

但 detailed balance 不满足:
π(1)·T(2|1) = (1/3)·1 = 1/3
π(2)·T(1|2) = (1/3)·0 = 0
≠ → detailed balance 违反!

这个链是一个"确定性循环": 1→2→3→1→... 不断旋转。它有时间上的对称性 (stationary), 但没有"可逆性"(reversibility)。

</details>

---

### Q9. 自适应 MCMC 与收敛诊断

**(A)** 解释 Gelman-Rubin R-hat 统计量的原理: 为什么比较"链内方差"和"链间方差"能诊断收敛?

**(B)** 为什么自适应 MCMC (在 burn-in 期间调整 proposal) 需要小心? 什么条件下自适应 MCMC 仍然保证收敛?

<details>
<summary>点击查看答案</summary>

**(A) R-hat 原理:**

跑 M 条独立的链, 每条链 N 步 (post burn-in)。

```
链内方差 W = (1/M) Σ s_j²  (每条链内部方差的平均)

链间方差 B = (N/(M-1)) Σ (θ̄_j - θ̄)²

边际后验方差估计: V̂ = (1-1/N)·W + (1/N)·B

R̂ = √(V̂ / W)
```

解释:
- 收敛前: 链在不同的区域 → B >> W → R̂ >> 1
- 收敛后: 链混合覆盖整个后验 → B ≈ W → R̂ ≈ 1

经验规则: R̂ < 1.1 → 可能已收敛。

**(B) 自适应 MCMC 的陷阱:**

自适应 = 用过去样本调整 proposal (如调整 σ), 违反了 Markov 性。

问题: 如果一直调整 → 链不再满足 detailed balance at each step → 稳态可能偏离目标分布。

**保证收敛的条件** (Roberts & Rosenthal, 2007):
1. **Diminishing Adaptation**: 调整量随迭代趋于 0
2. **Containment**: 自适应参数保持在一个紧集中

实践中: 只在 burn-in 阶段自适应, 之后固定 proposal → 安全有效。

</details>

---

### Q10. Hamiltonian Monte Carlo (HMC) 简介

HMC 是 MCMC 的进阶变体, 利用梯度信息 (模拟物理中的 Hamiltonian 动力学) 来做远距离 proposal。

**(A)** HMC 与 Random Walk MH 的核心区别是什么? 为什么 HMC 在高维中更有效?

**(B)** 解释为什么 HMC 在采样高维 Gaussian 时几乎不需要 burn-in。

<details>
<summary>点击查看答案</summary>

**(A) 核心区别:**

| | Random Walk MH | HMC |
|---|---------------|-----|
| Proposal | z* = z + ε, ε ~ N(0,σ²) | 沿梯度方向模拟 L 步物理动力学 |
| 信息利用 | 无梯度信息 | 利用 ∇log P(z) 引导移动方向 |
| 移动距离 | O(σ) — 非常局部 | O(L·step_size) — 可以很远 |
| 接受率 | 随 σ 增大急速下降 | 在理想条件下 ≈ 100% |
| 高维性能 | 差 (RW 在高维中退化为扩散) | 好 (利用梯度克服维度) |

HMC 的关键优势: 利用 ∇log P 的梯度 → proposal 偏向 P 的高概率方向 → 即使在 D=1000 的高维中也有效。

**(B) 高维 Gaussian 中几乎不需要 burn-in:**

HMC 对 N(μ, Σ) 的采样: 如果知道 Σ (或近似), 可以将 Gaussian 变换为标准 Gaussian, 直接 i.i.d. 采样。

即使不知道 Σ, HMC 的梯度引导使得 proposal 沿着"概率等高线"的切线方向移动 — 这正是 posterior 的相关结构方向 → 链非常快就到达稳态 → burn-in 极短。

这就是为什么 HMC (及其变体 NUTS, 用于 Stan) 成为现代贝叶斯推断的主力采样器。

</details>

---

## 📊 综合自测评分

每题 10 分，共 100 分。

| 得分 | 评价 |
|------|------|
| 90-100 | L9 完全掌握, 已理解 MC/MCMC 的核心原理和局限性 |
| 70-89  | 主干扎实, 建议亲自动手跑一个 MCMC 实验 |
| 50-69  | 概念清晰, 回去推一遍 MH 的 detailed balance 证明 |
| < 50   | 先吃透 Q1-Q4, 确保理解 Rejection/Importance/MH/Gibbs |

---

> L9 完成了 Monte Carlo 方法的学习。L10 可能会涉及更进阶的 MCMC (HMC, NUTS) 或进入 PGM 的第三大模块: 学习 (Learning)。
