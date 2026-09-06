# CMU 10-708 Lecture 18 课后练习 & 答案 — 因果关系2

> 配套教材: Peters et al. (2017) *Elements of Causal Inference*, Pearl (2009) *Causality*, Janzing et al. (2012), Shimizu et al. (2006)
>
> 题目覆盖 L18 九大主题, 三级: 🟢 基础 | 🟡 进阶 | 🔴 挑战

---

# 第一部分: Why Causality & Causal Inference

---

## 🟢 Q1. 因果思维的必要性

一个电商平台的数据科学家发现: "加购物车的用户购买概率是 80%，不加购物车的只有 5%"。于是她建议: "我们应该随机给一些用户自动添加商品到购物车，这样购买率会大幅提升!"

**(A)** 指出这个推理中的因果谬误。

**(B)** 画出因果图。自动加购物车 (A) 对购买 (P) 的因果效应与"用户自己加购物车"有什么不同？

**(C)** 如果要真正评估 A 对 P 的因果效应，最可靠的方法是什么？

<details>
<summary>点击查看答案</summary>

**(A) 因果谬误:**

这个推理混淆了关联和因果:
- 观测数据: 用户**自己选择**加购物车 → 80% 购买率 (反映用户意图)
- 干预: 系统**强制**给用户加购物车 → 这些用户没有购买意图!
- P(P | A=自己加) ≠ P(P | do(A=强制加))

**(B) 因果图:**

```
观测 (用户自己加):
  U (购买意图) → A (加购物车)
  U (购买意图) → P (购买)
  A → P  (可能有也可能没有因果效应)

干预 (系统强制加):
  do(A=1): 切断 U→A
  A → P   (纯因果效应, 无意图混杂)
```

**(C)** RCT (随机对照试验): 随机分配用户到"自动加购物车"组和对照组，比较两组购买率。这是消除混杂的金标准。
</details>

---

## 🟢 Q2. 因果推断框架

给定观测数据，要估计 ATE = E[Y|do(X=1)] - E[Y|do(X=0)]。

**(A)** 列出实现因果效应估计的 4 种必要假设。

**(B)** 对于因果图 `Z → X, Z → Y, X → Y`，写出用观测分布表示 ATE 的公式。需要调节什么变量？

<details>
<summary>点击查看答案</summary>

**(A) 4 种必要假设:**

1. **因果马尔可夫 (Causal Markov):** 给定父节点，变量条件独立于非后代
2. **因果忠实性 (Causal Faithfulness):** 数据中的 CI 关系精确反映图结构
3. **因果充分性 (Causal Sufficiency):** 图中没有遗漏的公共原因
4. **正定性 (Positivity):** $P(X=x \mid Z=z) > 0$ 对所有 $x, z$（每个 Z 层中都有 treated 和 control）

**(B)** 后门准则: 调节 Z 即可。

$$ATE = \sum_z \left(E[Y \mid X=1, Z=z] - E[Y \mid X=0, Z=z]\right) \cdot P(Z=z)$$

Z 满足后门准则: Z 不是 X 的后代，且阻断后门路径 X←Z→Y。
</details>

---

# 第二部分: Conditional Independence — 约束方法

---

## 🟡 Q3. FCI 算法与隐混杂检测

观测到以下条件独立关系（来自 4 变量 $X_1, X_2, X_3, X_4$ 的样本）:

- $X_1 \perp X_4 \mid \{X_2, X_3\}$
- $X_2 \perp X_4 \mid \{X_1, X_3\}$
- $X_1 \perp X_2$ (无条件!)
- $X_1 \not\perp X_3$ (无条件依赖)
- $X_1 \not\perp X_3 \mid X_2$ (给定 X₂ 后仍然依赖!)

**(A)** 画出与这些 CI 一致的 DAG（假设因果充分性）。

**(B)** 如果 $X_1$ 和 $X_3$ 在所有可观测条件集下都不能 d-separated，FCI 算法会如何表示 $X_1$ 和 $X_3$ 的关系？画出可能的 PAG。

<details>
<summary>点击查看答案</summary>

**(A)** 分析 CI 关系:

- $X_1 \perp X_2$: 两者无条件独立 → 无边
- $X_1 \not\perp X_3$: 有依赖 → 有路径
- $X_1 \not\perp X_3 \mid X_2$: 给定 X₂ 后仍然依赖 → X₂ 不阻断 X₁-X₃ 之间的所有路径

可能的 DAG（不包含隐混杂）:
- X₁→X₃→X₄, X₂ 与 X₁ 独立, X₂→X₄

但 $X_2 \perp X_4 \mid \{X_1, X_3\}$ 在各种图结构下需要验证...

一个合理的满足因果充分性的 DAG:
```
X₁ → X₃ → X₄
      ↗
    X₂
```
即 X₂→X₃, X₁→X₃, X₃→X₄。检验:
- $X_1 \perp X_2$: ✓ (两者无共同原因, 无直接边)
- $X_1 \not\perp X_3$: ✓ (X₁→X₃)
- $X_1 \not\perp X_3 \mid X_2$: ✓ (给定 X₂ 不阻断 X₁→X₃)
- $X_1 \perp X_4 \mid \{X_2, X_3\}$: ✓ (X₁→X₃→X₄, 给定 X₃ 阻断)

**(B)** 如果 $X_1$ 和 $X_3$ 在所有可观测条件下都不能 d-separated:

FCI 会在 $X_1$ 和 $X_3$ 之间画 `↔` 边，表示存在未观测的混杂变量：

```
PAG:
  X₁ ↔ X₃ → X₄
        ↗
      X₂

或:
  X₁ ∘→ X₃ ∘→ X₄
        ↗
      X₂
```

$\circ\!\rightarrow$ 表示: "→ 或 ↔ 但方向不确定"。$X_1 \leftrightarrow X_3$ 表示两者有未观测的公共原因。
</details>

---

# 第三部分: Causal Asymmetry from Noise

---

## 🟡 Q4. ANM 残差独立性 — 手算验证

考虑以下数据生成过程（你已知真相是 X→Y）:

$$X \sim \text{Uniform}(-1, 1)$$
$$Y = X^3 + N_Y, \quad N_Y \sim N(0, 0.25)$$

**(A)** 写出 $P(Y \mid X)$ 和 $P(X \mid Y)$ 的表达式（不需要完整密度，说明其形式即可）。

**(B)** 如果我们错误地假设 Y→X 并拟合 $X = g(Y) + N_X$，$N_X$ 会独立于 Y 吗？为什么？

**(C)** 如果 f 是线性函数 $f(X) = 2X$，且 $N_Y$ 是高斯噪声。两个方向的残差会是什么情况？因果方向可识别吗？

<details>
<summary>点击查看答案</summary>

**(A)**

正向 ($X \to Y$):
$$P(Y \mid X) = \phi\left(\frac{Y - X^3}{0.5}\right) \cdot \frac{1}{0.5}$$
其中 $\phi$ 是标准正态密度。$Y - X^3$ 与 $X$ 独立。

反向 ($Y \to X$):
$$P(X \mid Y) = \frac{P(Y \mid X) \cdot P(X)}{P(Y)}$$
$$= \frac{\phi((Y - X^3)/0.5) \cdot \mathbb{1}_{[-1,1]}(X)/2}{\int_{-1}^{1} \phi((Y - t^3)/0.5) \cdot 1/2 \; dt}$$

这个分布**不是**"X 的函数 + 独立噪声"的形式。给定 Y，X 的分布集中在 $Y^{1/3}$ 附近但形状复杂。

**(B)** 残差不会独立于 Y:

拟合 $X = g(Y) + N_X$:
- $X$ 由 $Y = X^3 + N_Y$ 决定
- 逆关系: $X = (Y - N_Y)^{1/3}$
- $N_X = X - g(Y)$ 是 X 无法被 g(Y) 解释的部分

给定 Y，X 大约等于 $Y^{1/3}$，但被 $N_Y$ 扰动:
$X \approx Y^{1/3} \cdot (1 - \frac{N_Y}{3Y} + ...)$（泰勒展开）

$N_X$ 的分布**依赖于** Y（通过 $Y^{1/3}$ 的缩放因子），所以 $N_X \not\perp Y$ ✗

**(C)** 线性 + 高斯 = 对称:

正向: $Y = 2X + N_Y$
反向回归: $X = \frac{1}{2}Y - \frac{1}{2}N_Y$

定义 $N_X = X - \hat{X} = X - \frac{1}{2}Y$
$N_X = X - \frac{1}{2}(2X + N_Y) = X - X - \frac{1}{2}N_Y = -\frac{1}{2}N_Y$

$N_X \perp Y$? 
$N_X = -\frac{1}{2}N_Y$，而 $Y = 2X + N_Y$
由于 $X \perp N_Y$ 且 $N_X$ 只是 $N_Y$ 的缩放:
$N_X \perp Y$? 不! $Y$ 含 $N_Y$ → $N_X$ 与 $Y$ 相关!

等等——在线性高斯的对称性下，如果 X 是高斯，Y 是高斯，则两个方向的残差都独立：
$X \sim N(0, \sigma^2_X)$, $Y = 2X + N_Y$ with $N_Y \sim N(0, \sigma^2_N)$
则 $Y \sim N(0, 4\sigma^2_X + \sigma^2_N)$

反向: $X = \frac{2\sigma^2_X}{4\sigma^2_X + \sigma^2_N}Y + N'_X$
$N'_X \sim N\left(0, \frac{\sigma^2_X \sigma^2_N}{4\sigma^2_X + \sigma^2_N}\right)$

且 $N'_X \perp Y$ (因为联合高斯中线性回归残差总独立于回归量)!

→ **两个方向的残差都独立! 因果方向不可识别!**

这就是为什么 LiNGAM 需要**非高斯噪声**。
</details>

---

## 🔴 Q5. LiNGAM — 非高斯性辨识因果方向

双变量 LiNGAM 模型:

$$X = \varepsilon_X$$
$$Y = \beta X + \varepsilon_Y$$

其中 $\varepsilon_X \sim \text{Exponential}(1)$, $\varepsilon_Y \sim \text{Laplace}(0, 1)$, $\varepsilon_X \perp \varepsilon_Y$, $\beta = 0.8$。

**(A)** 计算 $P(Y)$ 的前四阶矩（均值和方差到峰度）。

**(B)** 解释为什么非高斯性允许辨识 $\beta$ 和因果方向。

**(C)** 如果 $\varepsilon_X$ 和 $\varepsilon_Y$ 都是高斯 → 两个方向的 β 估计都有效 → 不可识别。用数学说明（从协方差矩阵的分解推导）。

<details>
<summary>点击查看答案</summary>

**(A)**

已知: $\varepsilon_X \sim \text{Exp}(1)$, $E[\varepsilon_X] = 1$, $\text{Var}(\varepsilon_X) = 1$
$\varepsilon_Y \sim \text{Laplace}(0,1)$, $E[\varepsilon_Y] = 0$, $\text{Var}(\varepsilon_Y) = 2$

$$\begin{aligned}
E[X] &= E[\varepsilon_X] = 1 \\
\text{Var}(X) &= 1 \\
E[Y] &= \beta \cdot 1 + 0 = 0.8 \\
\text{Var}(Y) &= \beta^2 \cdot 1 + 2 = 0.64 + 2 = 2.64
\end{aligned}$$

高阶矩（由独立非高斯分量决定）:
$$\begin{aligned}
E[X^3] &= \text{skewness of Exp(1)} \cdot (\text{Var})^{3/2} + ... \\
\text{skew}(X) &= 2 \text{ (Exp(1))}
\end{aligned}$$

**(B)** 非高斯性允许辨识:

在联合高斯下，协方差矩阵 $\Sigma$ 可分解为:
$$\Sigma = \begin{pmatrix} 1 & \beta \\ \beta & \beta^2+2 \end{pmatrix}$$

这个 $\Sigma$ 可以写成:
- $A A^T$ 其中 $A = \begin{pmatrix} 1 & 0 \\ \beta & \sqrt{2} \end{pmatrix}$ (X→Y)
- 也可以写成 $B B^T$ 其中 $B$ 对应 Y→X

在非高斯情况下，ICA 可以唯一确定混合矩阵（除了排列和缩放）：
从 $X = A \cdot E$ 中恢复 $E$ → 得到因果顺序。

加上 acyclicity 约束 → 排列确定 → 因果方向唯一！

**(C)** 高斯情况:

如果 $\varepsilon_X \sim N(0,1)$, $\varepsilon_Y \sim N(0, \sigma^2)$:

协方差:
$$\Sigma = \begin{pmatrix} 1 & \beta \\ \beta & \beta^2+\sigma^2 \end{pmatrix}$$

这个 $\Sigma$ 可以有多种 Cholesky 分解（对应不同的因果方向）:
- 下三角: $\begin{pmatrix} l_{11} & 0 \\ l_{21} & l_{22} \end{pmatrix}$ → $X \to Y$
- 上三角也有对应分解 → $Y \to X$

两种分解都给出同样的高斯分布 → 仅从数据无法区分 → 不可识别。
</details>

---

# 第四部分: Independent Change & Confounding

---

## 🟡 Q6. 多环境因果不变性

两个环境的数据:

| 环境 | E[X₁] | E[X₂] | E[Y] | Corr(X₁, X₂) |
|------|-------|-------|------|-------------|
| Env1 | 2.0   | 0.0   | 10.0 | 0.0          |
| Env2 | 5.0   | 0.0   | 19.0 | 0.0          |

已知 $X_1 \perp X_2$ 在所有环境中。

**(A)** 如果因果方向是 $X_1 \to Y$，利用跨环境一致性估计因果效应 $\beta_1$。$\beta_2$ 怎么估计？

**(B)** 如果 $X_2$ 不是 Y 的原因，但在 Env1 中 Corr(X₂, Y) ≈ 0.3（通过 X₁ 的间接关联），它在 Env2 中会怎么变化？这说明什么？

<details>
<summary>点击查看答案</summary>

**(A)**

假设模型: $Y = \beta_1 X_1 + \beta_2 X_2 + \varepsilon$, $\varepsilon \perp (X_1, X_2)$

$$\begin{aligned}
E[Y \mid \text{Env1}] &= \beta_1 \cdot 2.0 + \beta_2 \cdot 0.0 = 10.0 \\
E[Y \mid \text{Env2}] &= \beta_1 \cdot 5.0 + \beta_2 \cdot 0.0 = 19.0
\end{aligned}$$

$$\beta_1 = \frac{19.0 - 10.0}{5.0 - 2.0} = \frac{9.0}{3.0} = 3.0$$

$\beta_2$ 无法从这组矩中估计（需要 $X_2$ 也在某个环境中改变）。可以用:
$$\beta_2 = \text{Cov}(X_2, Y - \beta_1 X_1) / \text{Var}(X_2)$$
（在任一环境中，从残差回归）。

**(B)**

在 Env1 中:
$$\text{Cov}(X_2, Y) = \text{Cov}(X_2, \beta_1 X_1 + \beta_2 X_2 + \varepsilon) = \beta_1 \text{Cov}(X_2, X_1) + \beta_2 \text{Var}(X_2)$$

如果 $X_1 \perp X_2$，则 $\text{Cov}(X_2, X_1) = 0$:
$$\text{Cov}(X_2, Y) = \beta_2 \text{Var}(X_2)$$

但题目说 Corr(X₂, Y) ≈ 0.3 且 $X_2$ 不是原因 ($\beta_2 = 0$) → 这意味着 $X_2$ 和 $X_1$ 不独立？或是模型设定错误！

如果 $X_2$ 确实不是 Y 的原因: Cov(X₂, Y) 完全来自 Cov(X₂, X₁)·β₁。如果 $X_1$ 的分布改变但 $X_1$ 和 $X_2$ 的**关系**也改变 → Cov(X₂, Y) 也会改变。

→ **$X_2$ 与 Y 的条件关联不稳定** → 跨环境变化 → 排除 $X_2$ 作为原因 ✓

这就是 ICP (不变因果预测) 的核心逻辑!
</details>

---

## 🔴 Q7. 工具变量 — 数学推导

结构方程:

$$\begin{aligned}
X &= \alpha Z + \delta U + \varepsilon_X \\
Y &= \beta X + \gamma U + \varepsilon_Y
\end{aligned}$$

其中 $Z \perp U$, $Z \perp \varepsilon_X$, $Z \perp \varepsilon_Y$, $U \perp (\varepsilon_X, \varepsilon_Y)$。

**(A)** 证明 $\beta = \text{Cov}(Z, Y) / \text{Cov}(Z, X)$ (Wald estimator)。

**(B)** 推导 IV 估计的渐近方差（提示: 使用 Delta 方法）。

**(C)** 如果工具变量 Z 弱相关 ($\alpha$ 很小)，IV 估计会有什么问题？

<details>
<summary>点击查看答案</summary>

**(A)** Wald estimator 推导:

$$\begin{aligned}
\text{Cov}(Z, Y) &= \text{Cov}(Z, \beta X + \gamma U + \varepsilon_Y) \\
&= \beta \text{Cov}(Z, X) + \gamma \text{Cov}(Z, U) + \text{Cov}(Z, \varepsilon_Y) \\
&= \beta \text{Cov}(Z, X) + 0 + 0 \\
&= \beta \text{Cov}(Z, X)
\end{aligned}$$

$$\boxed{\beta = \frac{\text{Cov}(Z, Y)}{\text{Cov}(Z, X)}}$$

关键: $\text{Cov}(Z, U) = 0$ (外生性) 和 $\text{Cov}(Z, \varepsilon_Y) = 0$ (排斥性)。

**(B)** 渐近方差 (Delta 方法):

令 $S_{ZY} = \text{Cov}(Z, Y)$, $S_{ZX} = \text{Cov}(Z, X)$

$$\hat{\beta}_{IV} = \frac{\hat{S}_{ZY}}{\hat{S}_{ZX}} \equiv g(\hat{S}_{ZY}, \hat{S}_{ZX})$$

$$\nabla g = \left(\frac{1}{S_{ZX}}, -\frac{S_{ZY}}{S_{ZX}^2}\right)$$

$$\text{Var}(\hat{\beta}_{IV}) \approx \frac{1}{n} \cdot \frac{\text{Var}(Z \cdot (Y - \beta X))}{[\text{Cov}(Z, X)]^2}$$

$$\boxed{\text{Var}(\hat{\beta}_{IV}) \approx \frac{\sigma^2_{\varepsilon}}{n \cdot \alpha^2 \cdot \text{Var}(Z)}}$$

**(C)** 弱工具变量问题:

如果 $\alpha$ 很小 (Z 对 X 的效应弱):

1. **方差膨胀:** $\text{Var}(\hat{\beta}_{IV}) \propto 1/\alpha^2$ → 弱工具 = 巨大方差
2. **有限样本偏倚:** 即使在大样本下一致，有限样本中 IV 估计向 OLS 偏倚
3. **敏感度:** 对排斥性假设的微小违反极度敏感

F > 10 经验准则: 第一阶段的 F 统计量 > 10 → 避免弱工具问题。
</details>

---

# 第五部分: Selection Bias

---

## 🟡 Q8. Berkson 悖论 — 数值计算

总体中: $D_1 \perp D_2$, P(D₁=1) = 0.15, P(D₂=1) = 0.20。

入院概率: P(S=1 | D₁, D₂) = 0.10 + 0.80·D₁ + 0.80·D₂ (上限为 1)。

**(A)** 计算总体中 P(D₁=1 | D₂=1) 和 P(D₁=1)。

**(B)** 计算入院患者中 P(D₁=1 | D₂=1, S=1) 和 P(D₁=1 | D₂=0, S=1)。入院患者中 D₁ 和 D₂ 是正相关还是负相关？

**(C)** 这个悖论对医学研究有什么警示？

<details>
<summary>点击查看答案</summary>

**(A)** 总体:

$D_1 \perp D_2$ → P(D₁=1 | D₂=1) = P(D₁=1) = 0.15 ✓

**(B)** 入院患者:

先计算入院总概率:

| D₁ | D₂ | P(D₁,D₂) | P(S=1\|D₁,D₂) | 联合 P |
|----|----|-----------|-----------------|--------|
| 0  | 0  | 0.85×0.80=0.68 | 0.10 | 0.0680 |
| 1  | 0  | 0.15×0.80=0.12 | 0.90 | 0.1080 |
| 0  | 1  | 0.85×0.20=0.17 | 0.90 | 0.1530 |
| 1  | 1  | 0.15×0.20=0.03 | 1.00 | 0.0300 |

P(S=1) = 0.0680 + 0.1080 + 0.1530 + 0.0300 = 0.3590

入院患者中:
$$\begin{aligned}
P(D_1=1 \mid D_2=1, S=1) &= \frac{0.0300}{0.1530+0.0300} = \frac{0.0300}{0.1830} = 0.1639 \\
P(D_1=1 \mid D_2=0, S=1) &= \frac{0.1080}{0.0680+0.1080} = \frac{0.1080}{0.1760} = 0.6136
\end{aligned}$$

$$\text{Odds ratio} = \frac{0.1639/(1-0.1639)}{0.6136/(1-0.6136)} = \frac{0.1961}{1.587} = 0.1236$$

→ 负相关! OR < 1，入院患者中 D₁ 和 D₂ "互斥"!

**(C)** 警示:

1. **医院数据 ≠ 总体数据:** 医院中的疾病关联不代表总体中的关联
2. **不要条件化结果:** 选择样本 (=条件化入院状态) 会引入虚假关联
3. **识别研究设计的 collider:** 病例对照研究, 志愿者研究, 高流量用户分析都可能存在
4. **外部有效性:** 研究人群 ≠ 目标人群时，效应估计可能有偏
</details>

---

# 第六部分: Temporal Info & Transfer Learning

---

## 🟡 Q9. Granger 因果 vs 真实因果

时间序列 $(X_t, Y_t, Z_t)$ 由以下模型生成:

$$\begin{aligned}
Z_t &= 0.5Z_{t-1} + \varepsilon_{Z,t} \\
X_t &= 0.4Z_{t-1} + 0.3X_{t-1} + \varepsilon_{X,t} \\
Y_t &= 0.6Y_{t-1} + 0.2X_{t-1} + \varepsilon_{Y,t}
\end{aligned}$$

**(A)** X Granger-causes Y 吗？写检验假设。

**(B)** Z 和 Y 之间有任何 Granger 因果关系吗？

**(C)** 真实因果图是什么？Granger 因果和真实因果的关系是什么？

<details>
<summary>点击查看答案</summary>

**(A)** X Granger-causes Y:

检验 $H_0$: $Y_t$ 的预测中，$X_{t-1}$ 的系数 = 0（控制了 $Y_{t-1}$ 后）

因为真实系数 = 0.2 ≠ 0 → **X Granger-causes Y** ✓

这里 Granger 因果 = 真实因果（由 SCM 的结构直接得出）。

**(B)** Z 和 Y:

Z 不直接影响 Y（Y 的方程中没有 Z）。但:
$Z_t$ → $X_{t+1}$ → $Y_{t+2}$ (间接, 通过 X)

检验 Granger 因果时，通常检验**直接**滞后效应:
$Y_t = \beta_1 Y_{t-1} + \beta_2 X_{t-1} + \beta_3 Z_{t-1} + \varepsilon_t$

这里 $\beta_3 = 0$ → Z 不直接 Granger-cause Y。

但如果只检验 Z 和 Y (不控制 X):
$Y_t = \beta_1 Y_{t-1} + \beta_3 Z_{t-1} + \varepsilon_t$
$\beta_3$ 可能显著 ≠ 0（因为 Z 通过 X 间接影响 Y）。

→ **Granger 因果依赖于控制变量集的选择!**

**(C)** 真实因果图:

```
Z_{t-1} → X_t
Z_{t-1} → Z_t (自回归)
X_{t-1} → X_t (自回归)
Y_{t-1} → Y_t (自回归)
X_{t-1} → Y_t
```

Granger 因果 vs 真实因果:
- Granger: 预测意义上的（样本内预测改进）
- 真实因果: 干预意义上的（do(X) 改变 Y 的分布）
- 当存在瞬时因果或隐混杂时，两者会偏离
</details>

---

## 🟢 Q10. 迁移学习 — 紧凑变化描述

跨域迁移学习中的"紧凑变化描述"是什么意思？用因果模块性概念解释。

<details>
<summary>点击查看答案</summary>

**紧凑变化描述 (Compact Description of Changes):**

当数据分布从源域变到目标域时:
- 不是所有 $P(X_i \mid pa(X_i))$ 都改变
- 通常只有少数因果模块改变
- 这种稀疏变化的描述就是"紧凑"的

因果模块性:
- 每个 $P(X_i \mid pa(X_i))$ 是一个独立模块
- 模块间的变化是独立的（或至少稀疏的）

迁移含义:
- 在源域学习不变的模块（跨域稳定的 $P(X_i \mid pa(X_i))$）
- 在目标域只需重新学习少数改变的模块
- v.s. $P(Y \mid X)$ — 联合分布的一个切片
  → 当 $P(X)$ 改变时，$P(Y \mid X)$ 作为整体改变
  → 需要完全重新训练
  
因果迁移 ≈ 找出哪些部件变了，只更新那些部件！

例子: 自动驾驶
- 因果模块: P(障碍物|传感器), P(碰撞|速度,障碍物)
- 从美国 → 英国: P(靠右行驶) 变为 P(靠左行驶)，但 P(碰撞|...) 不变
- 只需更新一个模块!
</details>

---

## 参考

- Peters, J., Janzing, D., & Schölkopf, B. (2017). *Elements of Causal Inference*. MIT Press.
- Pearl, J. (2009). *Causality* (2nd ed.). Cambridge.
- Shimizu, S., Hoyer, P. O., Hyvärinen, A., & Kerminen, A. (2006). A linear non-gaussian acyclic model for causal discovery. *JMLR*.
- Janzing, D., Mooij, J., Zhang, K., et al. (2012). Information-geometric approach to inferring causal directions. *Artificial Intelligence*.
- Peters, J., Bühlmann, P., & Meinshausen, N. (2016). Causal inference by using invariant prediction. *JRSS-B*.
