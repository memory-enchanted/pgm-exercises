# CMU 10-708 Lecture 10 课后练习 & 答案

> 配套教材: Murphy Ch.24, Neal (2011), Gelman et al. BDA3 Ch.11-12
>
> 题目分为三级: 🟢 基础 | 🟡 进阶 | 🔴 挑战

---

## 🟢 基础题 (必须掌握)

### Q1. HMC 的 Hamiltonian

HMC 引入辅助动量 p, 构造 Hamiltonian: H(x, p) = U(x) + K(p)。

**(A)** U(x) 和 K(p) 分别对应什么? 写出它们的标准形式。

**(B)** HMC 的 joint distribution over (x, p) 是什么? 为什么采样 (x, p) 再丢掉 p 正好给出 P(x)?

<details>
<summary>点击查看答案</summary>

**(A) U 和 K:**

```
U(x) = -log P(x)           ← "势能" = 目标分布的负对数
K(p) = ½ pᵀ M^{-1} p       ← "动能" = 动量, M 为质量矩阵 (通常 M=I)
```

H(x, p) = -log P(x) + ½ pᵀ p  (当 M=I)

**(B) Joint distribution:**

```
P(x, p) ∝ exp(-H(x, p))
        ∝ exp(-U(x)) · exp(-K(p))
        ∝ P(x) · N(p | 0, M)
```

x 和 p 在 joint 下是**独立的**: x 的边际恰好是目标 P(x), p 的边际是 N(0, M)。

采样 (x, p) from joint → x 的边际 = P(x)。丢掉 p → 得到 P(x) 的样本。

核心洞察: HMC 的 proposal 机制保证了 joint 的 stationary distribution 是 exp(-H) → x 的边际自动是 P(x)。

</details>

---

### Q2. Leapfrog 积分器的性质

**(A)** 写出一次 Leapfrog 步 (ε) 的更新公式。

**(B)** Lensky 的三个关键性质: 可逆性 (reversible), 保体积 (volume-preserving), 辛性 (symplectic)。解释为什么保体积对 MCMC 很重要。

<details>
<summary>点击查看答案</summary>

**(A) Leapfrog (半步-全步-半步):**

```
p(t + ε/2) = p(t) + (ε/2) ∇log P(x(t))
x(t + ε)   = x(t) + ε · p(t + ε/2)
p(t + ε)   = p(t + ε/2) + (ε/2) ∇log P(x(t + ε))
```

**(B) 保体积的重要性:**

保体积 = 变换的 Jacobian 行列式 = 1。

在 MH 接受率中: α = min(1, P(x*)/P(x) · |J|)

如果 |J| = 1 (保体积), α = min(1, P(x*)/P(x)) — 与 RW-MH 相同形式!

如果 |J| ≠ 1: 需要在 α 中额外乘 |J| → 接受率可能降低 → 效率下降。

Leapfrog 恰好保体积 (因为它是 shear 变换的组合), 所以 HMC 的 MH 校正简洁高效。

</details>

---

### Q3. Slice Sampling 的工作机制

**(A)** 描述 Slice Sampling 的两步: (1) 选高度 u, (2) 找切片并采样。

**(B)** 为什么 Slice Sampling 的接受率是 100%? 它是否仍然满足 detailed balance?

<details>
<summary>点击查看答案</summary>

**(A) 两步:**

**Step 1**: u ~ Uniform(0, P(x_current))
→ 在目标分布概率密度下方随机选一个"高度"

**Step 2**: 找到区间 {x: P(x) ≥ u} ("切片"), 在区间内均匀采样 x_new
→ 新点必然满足 P(x_new) ≥ u (因为从切片内采的)
→ 且均匀采样保证 detailed balance

**(B) 为什么接受率 100%:**

Slice Sampling 没有显式的"接受/拒绝"步 — 它直接在切片内均匀采样, 而切片内的任何点都满足被采样的条件。

**Detailed Balance?** 是的! 证明:

Slice Sampling 可看作在扩充空间 (x, u) 上做 Gibbs 采样:
- P(x, u) ∝ 1[0 ≤ u ≤ P(x)]

Sampling u|x ← Uniform(0, P(x))
Sampling x|u ← Uniform({x: P(x) ≥ u})

这是标准的 Gibbs 步 → 满足 detailed balance → 接受率 100%。

</details>

---

### Q4. Parallel Tempering 的温度

**(A)** 温度为 T 的分布 P_T(x) ∝ P(x)^(1/T)。当 T=1, T→∞, T→0 时, 分布分别变成什么样?

**(B)** 为什么交换操作 (swap) 需要 Metropolis 接受/拒绝? 写出一对相邻链 (k, k+1) 交换的接受概率。

<details>
<summary>点击查看答案</summary>

**(A) 温度效应:**

```
T = 1:   P_T(x) = P(x)               ← 精确目标后验
T → ∞:  P_T(x) → 常数 (均匀分布)     ← 所有 mode 等概率, 极易探索
T → 0:  P_T(x) → δ(x - x_MAP)        ← 坍缩到 MAP 点, 无探索能力
```

高温 → 分布"扁平化": mode 之间壁垒降低, 链容易跳跃。

**(B) 交换接受率:**

链 k (温度 T_k) 和链 k+1 (温度 T_{k+1}) 尝试交换状态 x_k ↔ x_{k+1}:

```
α = min(1, P_{T_k}(x_{k+1})·P_{T_{k+1}}(x_k) / (P_{T_k}(x_k)·P_{T_{k+1}}(x_{k+1})))
```

展开: α = min(1, exp( (1/T_k - 1/T_{k+1}) · (log P(x_k) - log P(x_{k+1})) ))

当 T_k 和 T_{k+1} 接近时, 接受率较高 → 需要足够的温度阶梯。

</details>

---

## 🟡 进阶题

### Q5. HMC 的参数调优

**(A)** ε (步长) 太大或太小分别会导致什么问题?

**(B)** L (步数) 太大或太小分别会导致什么问题? NUTS 如何解决 L 的选择?

**(C)** 质量矩阵 M 的作用是什么? 什么时候 M ≠ I 很重要?

<details>
<summary>点击查看答案</summary>

**(A) ε 的影响:**

- **ε 太小**: Leapfrog 几乎不移动 → 退化为 RW → 自相关高, 计算浪费
- **ε 太大**: Leapfrog 数值误差大 → H 不守恒 → 接受率暴跌 → 链卡住
- **最优 ε**: 使接受率在 60-90% 之间, 且无 divergences

**(B) L 的影响:**

- **L 太小**: 每步移动距离短 → 自相关仍然高 (接近 RW)
- **L 太大**: 链可能做 U-turn (沿等高线绕回来) → 白费计算, 还可能走回起点附近
- **NUTS**: 用"No U-Turn Sampler" — 自动检测 U-turn 并停止 leapfrog, 不需要手动指定 L

**(C) 质量矩阵 M:**

标准 HMC 用 M=I — 假设各维度尺度相同。

当各维度的尺度差异很大时 (如 x1 的方差是 1, x2 的方差是 100):
- M=I: 需要小的 ε 来适应快变化的维度 → 慢维度探索极慢
- M 匹配后验协方差的逆: 各维度在动量空间中"均衡" → 效率大幅提升

实践中: M 设为后验方差的估计 (从 preliminary run 中获得) — 这是 Stan 的默认做法。

</details>

---

### Q6. AIS 的退火路径

**(A)** AIS 用 P_k ∝ P_ref^(1-β_k) × P_target^(β_k) 作为中间分布。β_k 的路径如何影响 AIS 的效率?

**(B)** 如果只有 2 个中间分布 (K=2), AIS 和直接 IS 有什么区别? K → ∞ 时, AIS 的 bias → ?

<details>
<summary>点击查看答案</summary>

**(A) β 路径的影响:**

β 从 0 → 1 的路径决定中间分布的"间距":

- **均匀路径** β_k = k/K: 简单但不最优
- **在"相变点"密集**: 如果 P_ref 和 P_target 在某个 β 处发生剧烈变化, 应该在该区域多放中间分布
- **路径太短** (K 太小): 相邻中间分布差别大 → 权重退化 → 高方差估计
- **路径太长** (K 太大): 计算浪费, 收益递减

实践中: K 通常需要 O(√D) 到 O(D) (D 是维度), 取决于 P_ref 和 P_target 的距离。

**(B) K=2 vs K→∞:**

K=2: 退化为直接 IS — 只有起始和目标, 没有中间过渡。如果两者差别大 → 权重退化严重。

K→∞: 相邻分布无限接近 → MCMC 在各中间分布上充分混合 → 权重近乎均匀 → 估计无偏且方差 → 0。

但实际中 K 有限 → AIS 是**有偏**的 (下偏: E[log Ẑ_AIS] < log Z_true), 偏倚随 K 增大而减小。

</details>

---

### Q7. MCMC 的 ESS 与效率

**(A)** 证明 ESS = n / (1 + 2 Σ ρ_k)。提示: Var(X̄_n) = (σ²/n) · (1 + 2 Σ (1 - k/n) ρ_k)。

**(B)** ESS/n 衡量什么? 如果 ESS/n ≈ 1%, 这意味着什么? 有什么改进策略?

<details>
<summary>点击查看答案</summary>

**(A) 证明:**

对相关的序列 X_1, ..., X_n:
```
Var(X̄_n) = (σ²/n) · (1 + 2 Σ_{k=1}^{n-1} (1 - k/n) ρ_k)

对于大 n, (1 - k/n) → 1:
Var(X̄_n) ≈ (σ²/n) · (1 + 2 Σ_{k=1}^∞ ρ_k)

ESS 定义为: ESS = n · σ² / (n · Var(X̄_n))
                = n / (1 + 2 Σ_{k=1}^∞ ρ_k)
```

直观: 如果样本独立 (ρ_k = 0): ESS = n。如果强相关 (ρ_k ≈ 1): ESS << n。

**(B) ESS/n 的含义:**

ESS/n = 1% 意味着: 2000 个 MCMC 样本只给出了 ~20 个独立样本的信息量。

每个样本高度重复前一个 → 链在"浪费时间"原地踏步。

**改进策略**:
1. 换更好的 sampler (HMC 替代 RW-MH)
2. 重新参数化 (去相关)
3. 增大 proposal 步长 (但注意不要破坏接受率)
4. Thinning (只治标不治本 — 减少存储但不增加信息)

</details>

---

## 🔴 挑战题

### Q8. HMC 的最优接受率

Roberts & Rosenthal (1998) 证明: 在高维中, RW-MH 的最优接受率为 **23.4%**。Beskos et al. (2013) 证明 HMC 的最优接受率为 **65.1%**。

**(A)** 为什么 HMC 的最优接受率比 RW-MH 高这么多?

**(B)** 解释: 在什么条件下, HMC 的 ESS-per-gradient (总效率) 优于 RW-MH?

<details>
<summary>点击查看答案</summary>

**(A) 接受率差异的原因:**

RW-MH: proposal 是随机的, 高接受率意味着每一步只移动很小距离 (σ 太小) → 需要在"移动远"和"接受率高"之间折中 → 最优 23.4%。

HMC: proposal 是**定向**的 (沿梯度方向), 即使是远距离移动 (L 步, 每步 ε), Hamiltonian 近似守恒 → 接受率可以保持在 ~65% 同时移动很远。

换句话说: HMC 的接受率不是"距离的代价", 而是"积分精度的代价"。

**(B) ESS-per-gradient 对比:**

HMC 每步需要 L 次梯度计算。

ESSPG_HMC = ESS_HMC / (n_steps × L)

RW-MH 每步只需要 1 次 log P 计算。

ESSPG_RW = ESS_RW / (n_steps)

当 D > ~10 时, HMC 的 ESSPG 几乎总是 >> RW-MH — 因为 ESS_HMC/ESS_RW 随 D 指数增长, 而 L 只需要随 √D 增长。

**结论**: 对 D > 10 的模型, HMC 总是更优。现代贝叶斯软件 (Stan, PyMC) 默认使用 HMC/NUTS 而非 RW-MH 正是这个原因。

</details>

---

### Q9. 自适应提案的危险与补救

**(A)** 解释: 为什么在运行中使用自适应 proposal (如根据已采样点调整 σ) 可能破坏 MCMC 的收敛保证?

**(B)** Roberts & Rosenthal (2007) 给出了自适应 MCMC 的两个安全条件: Diminishing Adaptation 和 Containment。解释它们的含义。

<details>
<summary>点击查看答案</summary>

**(A) 自适应破坏收敛:**

MCMC 的收敛依赖于马尔可夫链的**同质性** (homogeneity) — 转移核 T 不能随时间变化。

如果在 t 时刻根据过去样本调整 proposal → T_t 依赖于过去 → 链不再是 homogeneous Markov chain → standard ergodic theorem 失效 → 可能收敛到错误的分布。

**(B) 两个安全条件:**

**Diminishing Adaptation**: 调整量随 t → ∞ 趋于 0。
```
sup_x ||T_{t+1}(·|x) - T_t(·|x)|| → 0 as t → ∞
```
保证最终 T_t 收敛到一个固定的 T_∞, asymptotic 行为可控。

**Containment**: 自适应参数保持在一个紧集中 (bounded)。
→ 防止参数发散到非法值 (如 σ → 0 或 σ → ∞)。

满足这两个条件的自适应 MCMC 仍然保证收敛到目标分布。

**实践中**: 只在 burn-in 期间自适应, burn-in 后固定 proposal → 简单且安全。

</details>

---

### Q10. 从 MCMC 到现代概率编程

**(A)** Stan 使用 HMC/NUTS 作为默认推断引擎。列出 Stan 相比手写 MCMC 的至少 3 个优势。

**(B)** 解释: 概率编程语言 (PPL) 如何让"模型构建"和"推断"解耦 — 为什么这对贝叶斯建模的革命性如此重要?

<details>
<summary>点击查看答案</summary>

**(A) Stan 的优势:**

1. **自动调参**: NUTS 自动选择 L, dual averaging 自动调 ε → 用户不需要手动调 MCMC 参数
2. **自动诊断**: 内置 R̂, ESS, divergences 检查 → 告知用户采样是否可靠
3. **高效实现**: C++ 后端 + 自动微分 → 比手写 Python/R MCMC 快 10-100x
4. **声明式建模**: 用户只需写模型 (log P), Stan 自动推导梯度 → 不需要手写 ∇log P
5. **丰富的后处理**: posterior predictive checks, LOO-CV, Bayes factors

**(B) 模型与推断解耦:**

**传统贝叶斯建模**: 模型 + 推断不可分 — 每个新模型需要重新推导、实现 sampler → 门槛极高, 容易出错。

**PPL 范式**:
```
用户: 写 P(X, Z) = ...   ← 声明式, 只关心建模
PPL:  自动推导 ∇log P → HMC/NUTS → 后验样本
```

这种解耦的革命性:
- 统计学家可以快速迭代模型 (不用每次重写 sampler)
- 推断质量的提升"免费"获得 (HMC 替代手写 Gibbs)
- 贝叶斯方法从"专家工具"走向"大众可用" — 和深度学习框架的 democratization 类似

Stan 的成功证明了这种范式的力量 — 它已成为贝叶斯统计的事实标准。

</details>

---

## 📊 综合自测评分

每题 10 分，共 100 分。

| 得分 | 评价 |
|------|------|
| 90-100 | L10 完全掌握, 已理解 HMC/NUTS/Slice/Tempering 的原理和优势 |
| 70-89  | 主干扎实, 建议用 Stan 跑一个真实模型体验 HMC 的效率 |
| 50-69  | 概念清晰, 回去推一遍 HMC 的 leapfrog 和 detailed balance |
| < 50   | 先吃透 Q1-Q4, 确保理解 HMC/Tempering/Slice 的基本机制 |

---

> L10 完成了进阶 MCMC 的学习。从 RW-MH (L9) 到 HMC/NUTS (L10) 到现代 PPL, 你已经掌握了从基础到前沿的贝叶斯推断工具链。
