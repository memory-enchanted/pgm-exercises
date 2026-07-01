# CMU 10-708 Lecture 7 课后练习 & 答案

> 配套教材: Bishop Ch.10, Murphy Ch.21, Blei et al. (2017)
>
> 题目分为三级: 🟢 基础 | 🟡 进阶 | 🔴 挑战

---

## 🟢 基础题 (必须掌握)

### Q1. KL 散度的手算

给定两个离散分布:

```
P = [0.2, 0.5, 0.3]
Q = [0.5, 0.3, 0.2]
```

**(A)** 计算 KL(P || Q)。

**(B)** 计算 KL(Q || P)。它们相等吗？为什么？

<details>
<summary>点击查看答案</summary>

**(A) KL(P || Q) = Σ_i P(i) × log(P(i)/Q(i))**

```
KL(P||Q) = 0.2 × log(0.2/0.5)
         + 0.5 × log(0.5/0.3)
         + 0.3 × log(0.3/0.2)

= 0.2 × (-0.9163) + 0.5 × (0.5108) + 0.3 × (0.4055)
= -0.1833 + 0.2554 + 0.1216
= 0.1938
```

**(B) KL(Q || P) = Σ_i Q(i) × log(Q(i)/P(i))**

```
KL(Q||P) = 0.5 × log(0.5/0.2)
         + 0.3 × log(0.3/0.5)
         + 0.2 × log(0.2/0.3)

= 0.5 × (0.9163) + 0.3 × (-0.5108) + 0.2 × (-0.4055)
= 0.4581 - 0.1533 - 0.0811
= 0.2238
```

KL(P||Q) = 0.194 ≠ KL(Q||P) = 0.224 — **不对称!**

**为什么**: KL 散度不是真正的"距离"度量。KL(P||Q) 衡量用 Q 近似 P 时丢失的信息; KL(Q||P) 衡量用 P 近似 Q 时丢失的信息。两个方向信息损失不同（因为 log-ratio 用不同分布加权）。

**VI 为什么选用 KL(Q||P)?**
- KL(Q||P) 是"mode-seeking": Q 只在 P 概率高的地方放置概率，避免在 P≈0 处放置概率 → 不会发散
- KL(P||Q) 是"mass-covering": Q 必须在 P 非零的所有地方都有概率 → 在 P 低概率区域 Q 也必须覆盖

</details>

---

### Q2. ELBO 分解验证

对一个隐变量模型 P(X, Z)，已知:

```
P(Z=0) = 0.6,  P(Z=1) = 0.4
P(X=1 | Z=0) = 0.1
P(X=1 | Z=1) = 0.8
观测: X = 1
```

**(A)** 计算后验 P(Z | X=1)。

**(B)** 取 Q(Z) = [0.5, 0.5]，计算 ELBO(Q) 和 KL(Q || P(Z|X))，验证 ELBO + KL = log P(X)。

**(C)** 取 Q(Z) = P(Z|X=1)，验证此时 KL=0, ELBO=log P(X)。

<details>
<summary>点击查看答案</summary>

**(A) 后验:**

```
联合: P(Z=0, X=1) = 0.6 × 0.1 = 0.06
      P(Z=1, X=1) = 0.4 × 0.8 = 0.32

P(X=1) = 0.06 + 0.32 = 0.38
log P(X=1) = ln(0.38) = -0.9676

后验:
P(Z=0 | X=1) = 0.06/0.38 = 0.1579
P(Z=1 | X=1) = 0.32/0.38 = 0.8421
```

**(B) Q = [0.5, 0.5]:**

```
ELBO = E_Q[log P(X,Z)] - E_Q[log Q]
     = [0.5 × ln(0.06) + 0.5 × ln(0.32)]
       - [0.5 × ln(0.5) + 0.5 × ln(0.5)]
     = 0.5 × (-2.8134) + 0.5 × (-1.1394) - 2 × 0.5 × (-0.6931)
     = -1.4067 - 0.5697 + 0.6931
     = -1.2833

KL(Q||P) = 0.5 × ln(0.5/0.1579) + 0.5 × ln(0.5/0.8421)
         = 0.5 × (1.1527) + 0.5 × (-0.5213)
         = 0.5764 - 0.2607
         = 0.3157

验证: ELBO + KL = -1.2833 + 0.3157 = -0.9676 = log P(X) ✓
```

**(C) Q = P(Z|X) = [0.1579, 0.8421]:**

```
ELBO = 0.1579 × ln(0.06) + 0.8421 × ln(0.32)
       - [0.1579 × ln(0.1579) + 0.8421 × ln(0.8421)]
     = 0.1579 × (-2.8134) + 0.8421 × (-1.1394)
       - [0.1579 × (-1.8458) + 0.8421 × (-0.1719)]
     = -0.4443 - 0.9595 + 0.2915 + 0.1447
     = -0.9676 = log P(X)

KL = 0 (Q 就是后验, 无近似误差)
```

</details>

---

### Q3. CAVI 更新公式的手推导

对模型 P(Z₁, Z₂, X), 在 Mean-Field 假设 Q(Z₁, Z₂) = Q₁(Z₁) × Q₂(Z₂) 下。

**(A)** 写出 ELBO 关于 Q₁ 的表达式（把 Q₂ 视为固定）。

**(B)** 用变分法（或拉格朗日乘子法）推导 Q₁* 的最优条件:
      log Q₁*(Z₁) = E_{Q₂}[log P(Z₁, Z₂, X)] + const

<details>
<summary>点击查看答案</summary>

**(A) ELBO 表达式:**

```
ELBO = E_Q[log P(Z₁, Z₂, X)] - E_Q[log Q(Z₁, Z₂)]

第一项: E_Q[log P] = Σ_{z₁,z₂} Q₁(z₁)Q₂(z₂) · log P(z₁, z₂, X)
                   = Σ_{z₁} Q₁(z₁) · (Σ_{z₂} Q₂(z₂) · log P(z₁, z₂, X))

第二项: E_Q[log Q] = Σ_{z₁} Q₁(z₁)·log Q₁(z₁) + Σ_{z₂} Q₂(z₂)·log Q₂(z₂)
                   = -H(Q₁) - H(Q₂)
```

所以:
```
ELBO = Σ_{z₁} Q₁(z₁) · E_{Q₂}[log P(z₁, Z₂, X)]
       + H(Q₁) + H(Q₂)
```

注意 H(Q₂) 与 Q₁ 无关, 在优化 Q₁ 时可视为常数。

**(B) 变分求导:**

定义 f(Q₁) = Σ Q₁(z₁)·E_{Q₂}[log P] + H(Q₁) + λ(Σ Q₁(z₁) - 1)

对每个 Q₁(z₁) 求导:
```
∂f/∂Q₁(z₁) = E_{Q₂}[log P(z₁, Z₂, X)] - (log Q₁(z₁) + 1) + λ = 0

→ log Q₁(z₁) = E_{Q₂}[log P(z₁, Z₂, X)] + λ - 1
             = E_{Q₂}[log P(z₁, Z₂, X)] + const
```

归一化: Q₁(z₁) ∝ exp(E_{Q₂}[log P(z₁, Z₂, X)])

等价于: Q₁*(Z₁) ∝ exp(E_{Q_{-1}}[log P(Z, X)])

</details>

---

### Q4. Mean-Field 更新的物理直觉

对 Ising 模型 P(Z) ∝ exp(Σ_{i~j} w_{ij} Z_i Z_j), 其中 Z_i ∈ {+1, -1}。

**(A)** 写出 Mean-Field 下 Q_i(Z_i) 的更新公式。

**(B)** 解释为什么更新公式中 Z_i 只感受到来自邻居的"平均"影响。

<details>
<summary>点击查看答案</summary>

**(A) 更新公式:**

log P(Z) = Σ_{i~j} w_{ij} Z_i Z_j + const (忽略 partition function)

对 Q_i:
```
log Q_i(Z_i) = E_{Q_{-i}}[Σ_{i~j} w_{ij} Z_i Z_j] + const
             = Z_i · Σ_{j∈N(i)} w_{ij} · E_{Q_j}[Z_j] + const
```

令 m_j = E_{Q_j}[Z_j] = Q_j(+1)×1 + Q_j(-1)×(-1) = 2·Q_j(+1) - 1

这是节点 j 在 Q_j 下的"平均磁矩"(magnetization)。

```
log Q_i(Z_i) = Z_i · Σ_{j∈N(i)} w_{ij} · m_j + const

→ Q_i(Z_i=+1) = σ(2 · Σ w_{ij}·m_j)
  Q_i(Z_i=-1) = 1 - Q_i(+1)

其中 σ(x) = 1/(1+e^{-2x}) (sigmoid)
```

**(B) "平均场"的直觉:**

真实 Ising 模型中, Z_i 受到每个邻居 Z_j 的**具体取值**影响。

Mean-Field VI 中: Z_i 的邻居 Z_j 被"平均化"了 — Z_i 不再看到 Z_j 的具体值, 只看到邻居的"平均倾向" m_j = E[Z_j]。

```
真实:    Z_i 感受 Z_j=+1 还是 Z_j=-1 (具体的+1/-1)
Mean-Field: Z_i 感受 m_j = 0.3 (模糊的"偏+1多一点")
```

这就是"平均场"(mean field)名字的由来 — 每个自旋只感受邻居的**平均场**, 而非具体配置。

</details>

---

## 🟡 进阶题

### Q5. ELBO 单调上升的证明

证明 CAVI 的每一步更新后 ELBO 不会下降。

<details>
<summary>点击查看答案</summary>

**证明思路:**

CAVI 按坐标轮流更新 Q_i。更新 Q_i 时:

1. 对固定的 Q_{-i}, ELBO(Q_i, Q_{-i}) 作为 Q_i 的泛函, 最优解就是 CAVI 更新公式给出的 Q_i*。

2. 更新后: ELBO(Q_i*, Q_{-i}) ≥ ELBO(Q_i^old, Q_{-i})

3. 然后轮到 Q_{i+1} 更新: ELBO 再次不降。

4. 每个坐标更新都是"在当前其他坐标固定下, 精确最大化 ELBO"。

因此 ELBO 在整个过程中单调不降。

**形式化**:

对于固定的 Q_{-i}, 最大化 ELBO 等价于最小化 KL(Q_i·Q_{-i} || P)。

```
ELBO(Q_i, Q_{-i}) = -KL(Q_i·Q_{-i} || P) + log P(X)

最优 Q_i* = argmin_{Q_i} KL(Q_i·Q_{-i} || P)
          → log Q_i* = E_{Q_{-i}}[log P] + const
```

这正是 CAVI 的更新公式。每一步都确切找到了该坐标上的全局最优 → ELBO 不降。

**注意**: CAVI 只能保证收敛到**局部最优**, 不一定是全局最优。因为 Q 的因子化形式限制了可达的分布空间 (非凸优化)。

</details>

---

### Q6. Mean-Field 近似的"低估方差"现象

设真实后验 P(Z₁, Z₂) 是强负相关的二元分布:

```
P(Z₁=0, Z₂=0) = 0.05
P(Z₁=0, Z₂=1) = 0.45
P(Z₁=1, Z₂=0) = 0.45
P(Z₁=1, Z₂=1) = 0.05
```

**(A)** 求 Mean-Field 近似 Q*(Z₁, Z₂)。（提示: Q₁ 和 Q₂ 是对称的）

**(B)** 对比 Q* 和 P 给出的方差 Var(Z₁) 和协方差 Cov(Z₁, Z₂)。Mean-Field 如何扭曲了不确定性？

<details>
<summary>点击查看答案</summary>

**(A) Mean-Field 解:**

由对称性, Q₁ = Q₂ = [q₀, 1-q₀]。

CAVI 更新 (对 Q₁):
```
log Q₁(0) = E_{Q₂}[log P(0, Z₂)]
          = q₀·log(0.05) + (1-q₀)·log(0.45)

log Q₁(1) = E_{Q₂}[log P(1, Z₂)]
          = q₀·log(0.45) + (1-q₀)·log(0.05)
```

令差值: log Q₁(0) - log Q₁(1) = q₀·log(0.05/0.45) + (1-q₀)·log(0.45/0.05)

定点条件: q₀/(1-q₀) = exp(上述差值) 且 q₀ = 0.5 (由对称性)。

验证: 当 q₀ = 0.5 时,
```
log Q₁(0) - log Q₁(1) = 0.5 × (-2.197) + 0.5 × (2.197) = 0
→ Q₁(0) = Q₁(1) → q₀ = 0.5 ✓
```

所以 Q*(Z₁, Z₂) = [0.25, 0.25, 0.25, 0.25] — 完全均匀!

**(B) 对比:**

真实后验 P:
```
E[Z₁] = 0×0.5 + 1×0.5 = 0.5
Var(Z₁) = 0.5² = 0.25
Cov(Z₁, Z₂) = E[Z₁·Z₂] - 0.5²
             = (1×0×0.45 + 0×1×0.45 + 1×1×0.05) - 0.25
             = 0.05 - 0.25 = -0.20  (强负相关!)
```

Mean-Field Q*:
```
E_{Q}[Z₁] = 0.5
Var_Q(Z₁) = 0.25  (方差恰好一致 — 这次是巧合)
Cov_Q(Z₁, Z₂) = 0  (Mean-Field 强制独立 → 协方差 = 0!)
```

**关键扭曲**: Mean-Field 丢失了 Z₁ 和 Z₂ 之间的强负相关性 (Cov = -0.20 → 0)

后果:
- Q*(Z₁=0, Z₂=0) = 0.25 远高于真实值 0.05 → 过度估计了"一致"的概率
- Q*(Z₁=1, Z₂=1) = 0.25 远高于真实值 0.05 → 同上
- 整体上, Q* 低估了反相关性 → 过度平滑 → 不确定性被扭曲

</details>

---

### Q7. 为什么 VI 不能用在离散有环图上得到精确解?

**(A)** 把 Ising 模型 P(Z) ∝ exp(Σ_{i~j} w_{ij} Z_i Z_j) 的精确 log partition function log Z 写为对所有 2^N 种配置求和。

**(B)** 解释为什么 Mean-Field VI 的 ELBO 是这个 log Z 的下界, 以及为什么这个下界永远不紧 (有环时)。

<details>
<summary>点击查看答案</summary>

**(A) Partition function:**

```
Z = Σ_{z₁=±1} Σ_{z₂=±1} ... Σ_{z_N=±1} exp(Σ_{i~j} w_{ij} z_i z_j)
```

精确计算需要枚举所有 2^N 种配置 — 指数复杂度。

**(B) ELBO 作为下界:**

```
ELBO(Q) = E_Q[Σ w_{ij} Z_i Z_j] + H(Q)
        = Σ_{i~j} w_{ij} · E_{Q_i}[Z_i] · E_{Q_j}[Z_j] + H(Q)

注意: E_Q[Z_i Z_j] = E_{Q_i}[Z_i] · E_{Q_j}[Z_j] (因为 Mean-Field 假设 Z_i ⟂ Z_j)
```

而真实的 E[Z_i Z_j] ≠ E[Z_i]·E[Z_j] (有环时邻居是相关的)。

所以 ELBO 中的成对项用**独立近似**替换了真实相关性 → 下界不紧。

只有当图是树, 且使用 Bethe 近似 (而非 Mean-Field) 时, 下界才是紧的。

</details>

---

## 🔴 挑战题

### Q8. VI 与 BP 的变分解释

**(A)** 写出 Bethe 变分族的形式并解释它与 Mean-Field 族的区别。

**(B)** 证明在树状图上, Bethe VI 的定点方程恰好等于 BP 的消息更新方程。

<details>
<summary>点击查看答案</summary>

**(A) Bethe 变分族:**

Bethe 近似假设:
```
Q(Z) ∝ ∏_{i~j} ψ_{ij}(Z_i, Z_j) / ∏_i [ψ_i(Z_i)]^{d_i - 1}
```

其中 d_i 是节点 i 的度, ψ_{ij} 是边上的"伪边际", ψ_i 是节点边际。

与 Mean-Field 的区别:
```
Mean-Field: Q(Z) = ∏_i Q_i(Z_i)
            → 只保留节点边际, 丢失所有成对相关

Bethe:      Q(Z) ∝ ∏_{边} τ_{ij} / ∏_{节点} τ_i^{d_i-1}
            → 保留了成对边际 τ_{ij}(Z_i, Z_j)
            → 在树上, 这正是真实联合的因子分解形式!
```

**(B) 树上的等价性:**

在树上, 真实联合 P(Z) 可因子分解为:
```
P(Z) = ∏_i P(Z_i|Z_{pa(i)}) = ∏_{边} [P(Z_i, Z_j) / (P(Z_i)·P(Z_j))] × ∏_i P(Z_i)
     = ∏_{边} τ_{ij} / ∏_i P(Z_i)^{d_i-1}
```

这正是 Bethe 形式! 所以在树上, Bethe 变分族的表达式恰好与真实分布一致 → 约束是紧的 → Bethe VI 可恢复精确解。

对 Bethe 自由能做变分求导, 得到的定点方程:
```
τ_{ij}(Z_i, Z_j) ∝ ψ_{ij}(Z_i, Z_j) × (来自 i 的其他邻居的消息) × (来自 j 的其他邻居的消息)
```

这恰好是 BP 中"边信念 = 边势函数 × 两侧入边消息"的形式。

**总结**: BP = Bethe 变分族上的 VI, 在树上精确, 在有环图上 = Loopy BP。

</details>

---

### Q9. 从数据的角度理解 ELBO

**(A)** 把 ELBO 重写为: ELBO = E_Q[log P(X|Z)] - KL(Q(Z) || P(Z))

**(B)** 解释第一项 E_Q[log P(X|Z)] 和第二项 KL(Q||P(Z)) 分别对应机器学习的什么概念。

<details>
<summary>点击查看答案</summary>

**(A) 重写 ELBO:**

```
ELBO = E_Q[log P(X, Z)] - E_Q[log Q(Z)]
     = E_Q[log P(X|Z) + log P(Z)] - E_Q[log Q(Z)]
     = E_Q[log P(X|Z)] + E_Q[log P(Z)] - E_Q[log Q(Z)]
     = E_Q[log P(X|Z)] - [E_Q[log Q(Z)] - E_Q[log P(Z)]]
     = E_Q[log P(X|Z)] - KL(Q(Z) || P(Z))
```

**(B) 对应关系:**

```
ELBO = 重建项 (Reconstruction) - 正则化项 (Regularization)
     = E_Q[log P(X|Z)] - KL(Q(Z) || P(Z))
```

| 项 | 含义 | ML 对应 |
|----|------|---------|
| E_Q[log P(X\|Z)] | Q 期望下的对数似然: 隐变量 Z 能多好地"解释"数据 X | 重建误差 (reconstruction loss) |
| KL(Q(Z)\|\|P(Z)) | 变分后验 Q 与先验 P(Z) 的偏离: Q 不能离先验太远 | 正则化 (regularization) |

**这就是 VAE (Variational Autoencoder) 的核心损失函数!**

VAE 中:
- Q(Z|X) 是编码器 (encoder) — 给定 X, 输出隐变量分布
- P(X|Z) 是解码器 (decoder) — 给定 Z, 重建 X
- ELBO = 重建损失 - KL 正则 → 训练 VAE 的损失函数

VI 不仅是推断算法, 更是现代深度生成模型的基础!

</details>

---

### Q10. VI 的局限性与其扩展

列出 Mean-Field VI 的至少 3 个局限性, 并简述各自的改进方向。

<details>
<summary>点击查看答案</summary>

| 局限 | 描述 | 改进方向 |
|------|------|---------|
| **1. 独立性假设过强** | Q = ∏ Q_i 丢失所有变量间相关性 | 结构化 VI (Structured VI): 保留部分变量间的相关性 (如 Bethe 近似, 树状 VI) |
| **2. 局部最优** | CAVI 只保证收敛到局部最优, 结果依赖初始化 | 多次随机初始化; 确定性退火 (deterministic annealing); 更好的初始化策略 |
| **3. 不适用于大数据** | CAVI 每步需要遍历全部数据 | 随机变分推断 (SVI): 用小批量 (mini-batch) 梯度更新, 适用于大规模数据 |
| **4. 模型限制** | 要求条件共轭 (conditionally conjugate) 才有闭式解 | 黑盒变分推断 (BBVI): 用 score function 或 reparameterization gradient 估计梯度 |
| **5. ELBO 与真实后验的 gap 未知** | 无法知道 Q 离真实后验有多远 (KL 不可直接算) | 重要性加权自编码器 (IWAE): 用重要性采样给出更紧的下界 |

**现代 VI 全景** (L8 将涉及):
```
CAVI (L7) → SVI (大数据)
          → BBVI (非共轭模型)  
          → VAE (深度生成模型)
          → NVL (normalizing flows, 更丰富的 Q)
```

</details>

---

## 📊 综合自测评分

每题 10 分，共 100 分。

| 得分 | 评价 |
|------|------|
| 90-100 | L7 完全掌握，已理解 VI 的核心思想和实现细节 |
| 70-89  | 主干扎实，建议动手实现一个 CAVI |
| 50-69  | 概念框架有了，回去重看 ELBO 和 CAVI 推导 |
| < 50   | 先吃透 Q1-Q3，弄清 KL、ELBO、CAVI 更新的含义 |

---

> L7 是推断方法的重大转折 — 从精确到近似, 从组合计算到优化。理解 VI 后, L8 将介绍 MCMC 采样方法 — 另一条通往近似推断的道路。
