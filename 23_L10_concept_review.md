# CMU 10-708 Lecture 10 概念体系梳理 — 进阶 MCMC 方法

> 整理自 Eric Xing 教授 CMU 10-708 Lecture 10: Advanced MCMC Methods
>
> 核心教材: Murphy Ch.24, Neal (2011) HMC Handbook, Gelman et al. (2013) BDA3

---

## 📐 全局定位：L9 → L10 — 从基础的到进阶的 MCMC

```
L9: 基础 MCMC                          L10: 进阶 MCMC
─────────────────────                 ─────────────────────
Metropolis-Hastings (RW)              Hamiltonian MC (HMC)
Gibbs Sampling                         Slice Sampling
Rejection / Importance                 Parallel Tempering
Basic convergence checks               AIS (Annealed IS)
                                       Advanced diagnostics

L9 回答: "怎么做 MCMC?"               L10 回答: "怎么做高效的 MCMC?
                                       多峰怎么办? 维数高了怎么办?"
```

**一句话概括 L10**: L9 的 Random-Walk MH 在高维、强相关、多峰分布中效率极差。L10 介绍一系列进阶技术 — HMC (梯度引导), Slice Sampling (自适应), Parallel Tempering (多峰) — 来大幅提升 MCMC 的效率和鲁棒性。

---

## 概念 1：为什么 Random Walk MH 不够用？

### RW-MH 的三大失败场景

```
┌──────────────────────────────────────────────────────────────┐
│                                                               │
│  1. 高维 (D > 50):                                           │
│     RW proposal 的接受率随 D 指数衰减                         │
│     即使接受, 每步移动距离 ~ σ/√D → 极小                      │
│                                                               │
│  2. 强相关 (ρ → 1):                                          │
│     后验沿窄"峡谷"分布 → RW 几乎总是"撞墙"被拒               │
│     Gibbs 也难 (zig-zag)                                      │
│                                                               │
│  3. 多峰 (multi-modal):                                      │
│     RW 卡在一个 mode 中, 极难跳到另一个 mode                  │
│     → 链无法探索完整的后验 → 估计严重偏差                     │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## 概念 2：Hamiltonian Monte Carlo (HMC) (🔑🔑🔑)

### 核心思想

把采样问题变成**物理模拟**: 引入辅助"动量"变量 p, 构造 Hamiltonian 系统:

```
H(x, p) = U(x) + K(p)
         = -log P(x) + ½pᵀM^{-1}p

其中:
  U(x) = -log P(x)   ← "势能" (目标分布的负对数)
  K(p) = ½pᵀp        ← "动能" (动量, 通常设 M=I)
```

### HMC 的 Proposal

从当前 x 出发:
1. 随机采样动量 p ~ N(0, M)
2. 用 **Leapfrog 积分** 模拟 L 步 Hamiltonian 动力学
3. 达到新点 (x*, p*)
4. Metropolis 接受/拒绝

**关键**: 因为 H 守恒 (理想情况), 接受率接近 100%! 同时 leapfrog 可以移动很远。

### Leapfrog Integrator

```
┌──────────────────────────────────────────────────────────────┐
│                                                               │
│  p(t + ε/2) = p(t) + (ε/2) · ∇log P(x(t))    (半步步长)     │
│  x(t + ε)   = x(t) + ε · p(t + ε/2)           (全步步长)     │
│  p(t + ε)   = p(t + ε/2) + (ε/2) · ∇log P(x(t+ε)) (半步)    │
│                                                               │
│  重复 L 次                                                    │
│                                                               │
│  性质:                                                        │
│    - 可逆 (reversible): 时间反演对称                          │
│    - 保体积 (volume-preserving): Jacobian = 1                 │
│    - 近似能量守恒: O(ε²) error per step                       │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### HMC vs RW-MH

| | RW-MH | HMC |
|---|-------|-----|
| Proposal | x*=x+ε, 随机方向 | 沿梯度引导方向, L 步远 |
| 梯度信息 | 不使用 | **关键依赖** ∇log P |
| 高维接受率 | ~0% (D>100) | ~60-90% |
| 自相关 | 极高 (ρ>0.9) | 低 (ρ~0.1-0.3) |
| 计算成本/步 | O(1) | O(L·D) (需要 L 次梯度) |
| 参数调优 | 1个 (σ) | 2个 (ε, L) — 更敏感 |

### HMC 的"No U-Turn"优化 (NUTS)

HMC 的问题: L 太小 → 混合不好; L 太大 → 链可能"转回来" (U-turn), 浪费计算。

NUTS (Hoffman & Gelman, 2014): 自适应选择 L — 当链开始往回走时自动停止。

这就是 Stan 中使用的算法 — 免去了手动调 L 的麻烦。

---

## 概念 3：Slice Sampling — 免调参的采样

### 核心思想

不需要 proposal 分布! 通过"切片"目标分布来自动确定步长。

```
┌──────────────────────────────────────────────────────────────┐
│                                                               │
│  Slice Sampling (1D):                                         │
│                                                               │
│  1. 在高度方向采样: u ~ Uniform(0, P(x_current))             │
│  2. 在水平方向找"切片": {x: P(x) ≥ u}                       │
│  3. 在切片内均匀采样新点 x_new                               │
│                                                               │
│  这个算法满足 detailed balance w.r.t. P(x) — 无需调参!       │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### 优点

- **无参数**: 不需要 proposal 宽度 σ
- **自适应**: 步长自动适应目标分布的局部形状
- **接受率**: 100% (无拒绝步! 但 step 2 需要步出法 stepping-out)

### 局限

- 在多维中使用"逐维 Gibbs + 每维 Slice"组合
- 强相关时与 Gibbs 有相同的 zig-zag 问题

---

## 概念 4：Parallel Tempering — 多峰分布的采样 (🔑🔑)

### 问题

标准 MCMC 卡在局部 mode 中 → 无法探索多峰后验的全部。

### 核心思想

同时跑 K 条**不同温度**的链: 高温链更容易穿越 mode 之间的壁垒 → 通过链间交换, 将信息传递回低温链 (目标链)。

```
┌──────────────────────────────────────────────────────────────┐
│                                                               │
│  Parallel Tempering:                                          │
│                                                               │
│  定义温度阶梯: T₁=1 < T₂ < ... < T_K                         │
│  链 k 的目标: P_k(x) ∝ P(x)^{1/T_k}                         │
│                                                               │
│  For each iteration:                                          │
│    1. 每条链各自跑一步 MH (在自己的温度上)                    │
│    2. 随机选相邻一对 (k, k+1), 尝试交换状态:                 │
│       α = min(1, [P_k(x_{k+1})·P_{k+1}(x_k)] /              │
│                    [P_k(x_k)·P_{k+1}(x_{k+1})])               │
│                                                               │
│  高温链: 分布更"平" → 容易穿越壁垒                           │
│  低温链 (T₁=1): 精确后验 → 通过交换获得高温链探索的信息       │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### 直观理解

```
      P(x) at T=1              P(x) at T=10
         /\                        ___
        /  \                      /   \
    ___/    \___              ___/     \___
   mode1  mode2              mode1   mode2
    ↑壁垒高                   ↑壁垒低 → 容易跨越!

链 1 (T=1):   卡在 mode1 ────────────→ 精确后验
链 2 (T=10):  在 mode1↔mode2 自由穿梭 → 告诉链1 "mode2 也存在!"
```

---

## 概念 5：退火重要性采样 (AIS)

### 动机

Importance Sampling 在 proposal 和 target 差别大时退化。AIS: 通过一系列中间分布逐步"退火"。

```
从简单分布 P_0 (易采样) 出发, 经过一系列中间分布:
  P_0 → P_1 → P_2 → ... → P_K = P_target

每步用 MCMC 过渡 → 累积重要性权重 → 估计 partition function 比值
```

### AIS 权重

```
w = P₁/P₀ · P₂/P₁ · ... · P_K/P_{K-1}
  = P_target(z_0)/P_0(z_0) · Π (transition ratios)

log Z_target ≈ log Σ w_s - log S  (需要 reference Z_0 known)
```

---

## 概念 6：MCMC 诊断进阶

### 有效样本量 (ESS) 的精确计算

```
ESS = S / (1 + 2 Σ_{k=1}^∞ ρ_k)

其中 ρ_k = lag-k 自相关

实践中: 截断到第一个 ρ_k < 0.05 的 lag
```

### Gelman-Rubin R̂ 的改进版

```
R̂ = √((N-1)/N + B/(N·W)) · df/(df-2)

改进: 使用 Student-t 近似 (原版用 Gaussian 低估尾部)
      folded into rank-normalized R̂ (Vehtari et al. 2021)
```

### 其他诊断

| 诊断 | 检测什么 |
|------|---------|
| **Trace plot** | 视觉检查: 漂移, 趋势, 周期 |
| **Geweke 诊断** | 前段 vs 后段均值的 z 检验 |
| **MCSE** | MC 标准误: σ/√ESS (衡量估计精度) |
| **Divergences** (HMC) | Leapfrog 发散 → 步长太大, 需调小 ε |

---

## 📋 全部概念一张表

| 概念 | 一句话 |
|------|--------|
| **HMC** | 梯度引导 + 物理模拟 → 高维中极高效的采样 |
| **Leapfrog** | HMC 的积分器: 可逆, 保体积, O(ε²) 精度 |
| **NUTS** | 自动调 L: 检测 U-turn 时停止 → 免去手动调参 |
| **Slice Sampling** | 免 proposal: 在 P 的"水平切片"内均匀采样 |
| **Parallel Tempering** | 多温度链: 高温探索 + 交换 → 克服多峰 |
| **AIS** | 退火 + MCMC → 估计 partition function |
| **R̂ (Gelman-Rubin)** | 链内/链间方差比 → 收敛诊断 |
| **ESS** | 考虑自相关的等效独立样本数 |

---

## 🔗 概念关系图

```
          L9: 基础 MCMC (MH, Gibbs)
                    │
          问题: 高维, 强相关, 多峰
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
  HMC          Slice Sampler   Parallel Tempering
 梯度引导       免 proposal      多温度链
    │               │               │
    ▼               ▼               ▼
 高维高效      自适应步长      克服多峰
    │                               │
    ▼                               ▼
  NUTS                          AIS (退火IS)
 自动L选择                      Z(X) 估计
    │
    └─────── 进阶诊断 ───────┘
         R̂, ESS, MCSE, Trace
```

---

## 🎯 核心洞见一句话

| # | 洞见 |
|---|------|
| 1 | **HMC = RW-MH + 梯度** — 利用 ∇log P 做远距离、高接受率的 proposal |
| 2 | **Leapfrog 的三个性质** — 可逆, 保体积, O(ε²) → HMC 的数学保证 |
| 3 | **Slice = 无参数自适应** — 不需要 proposal, 自动调整步长 |
| 4 | **Tempering = 多链协作** — 高温探索, 低温精确, 交换信息 |
| 5 | **ESS = 样本质量的最终衡量** — 不是样本数量, 而是"等效独立样本数" |

---

## 🧪 自测清单

- [ ] HMC 的 Hamiltonian H(x,p) 由哪两部分组成? 各自对应什么物理量?
- [ ] Leapfrog 积分器为什么是可逆和保体积的?
- [ ] HMC 的 ε 太小/太大 分别会导致什么问题?
- [ ] Slice Sampling 为什么不需要 proposal 分布?
- [ ] Parallel Tempering 中温度 T 的作用是什么? T₁ 为什么必须等于 1?
- [ ] R̂ 统计量中, 链内方差 W 和链间方差 B 分别衡量什么?
- [ ] ESS 为什么比样本数量更能反映 MCMC 的"有效信息量"?

---

> L10 完成了 MCMC 方法的进阶 — 从基础的 RW-MH/Gibbs 到现代贝叶斯推断的主力工具 HMC (Stan) 和并行/退火技术。掌握这些后, 你已拥有处理真实世界贝叶斯模型的完整工具箱。
