# CMU 10-708 Lecture 8 课后练习 & 答案

> 配套教材: Bishop Ch.10, Murphy Ch.21, Blei et al. (2017), Kingma & Welling (2014)
>
> 题目分为三级: 🟢 基础 | 🟡 进阶 | 🔴 挑战

---

## 🟢 基础题 (必须掌握)

### Q1. Wake-Sleep 的两阶段

Wake-Sleep 算法有两个交替的阶段。对下面的模型:
- 生成模型: Z ~ Bernoulli(0.3), X|Z ~ N(μ_z, 1)
- 识别网络: Q(Z|X) = sigmoid(w·X + b)

**(A)** Wake 阶段更新哪个部分的参数？写出更新 μ_0, μ_1 的公式。

**(B)** Sleep 阶段更新哪个部分的参数？写出更新 w, b 的目标函数。

<details>
<summary>点击查看答案</summary>

**(A) Wake 阶段: 更新生成模型参数 μ_0, μ_1**

给定真实数据 X_real:
1. 用识别网络 Q(Z|X_real) 得到每个数据点的软分配 q_n = P(Z=1|X_n)
2. 最大化 E_{Q(Z|X)}[log P(X|Z)]:

```
μ_0 = Σ_n (1-q_n)·X_n / Σ_n (1-q_n)
μ_1 = Σ_n q_n·X_n / Σ_n q_n
```

即: 用 Q 给出的"责任"(responsibility) 做加权平均。

**(B) Sleep 阶段: 更新识别网络参数 w, b**

给定生成的数据 X_fake:
1. 从先验采样 Z ~ Bernoulli(0.3)
2. 用生成模型产生 X_fake ~ N(μ_z, 1)
3. 最大化 E_{P(Z,X)}[log Q(Z|X_fake)]:

```
L(w,b) = Σ_n [Z_n·log Q(Z_n=1|X_n) + (1-Z_n)·log(1-Q(Z_n=1|X_n))]
```

这是标准的二分类交叉熵损失 — 用生成的数据训练识别网络!

</details>

---

### Q2. SVI 的 Robbins-Monro 条件

SVI 用步长 ρ_t 做随机梯度更新: λ_t = λ_{t-1} + ρ_t · g_t

**(A)** 写出 Robbins-Monro 条件: Σρ_t = ∞ 和 Σρ_t² < ∞。分别解释它们保证了什么。

**(B)** 判断以下步长序列是否满足 RM 条件:
   (i) ρ_t = 0.01  
   (ii) ρ_t = 1/t  
   (iii) ρ_t = 1/√t

<details>
<summary>点击查看答案</summary>

**(A) RM 条件:**

```
Σ_{t=1}^∞ ρ_t = ∞  →  "能走到无穷远"
  保证算法不会被卡住, 有足够的步长总量到达最优点

Σ_{t=1}^∞ ρ_t² < ∞  →  "步长衰减够快, 噪声最终消失"
  保证随机梯度的噪声在极限下被平均掉, 最终收敛
```

**(B) 判断:**

**(i)** ρ_t = 0.01:
- Σ 0.01 = ∞ ✓
- Σ 0.0001 = ∞ ✗ → **不满足!** 步长不衰减 → 永远在最优附近震荡, 不收敛

**(ii)** ρ_t = 1/t:
- Σ 1/t = ∞ ✓ (调和级数)
- Σ 1/t² = π²/6 < ∞ ✓ (巴塞尔问题)
- → **满足!** 这是经典的 Robbins-Monro 步长

**(iii)** ρ_t = 1/√t:
- Σ 1/√t = ∞ ✓ (p=0.5 的 p-级数)
- Σ 1/t = ∞ ✗ → **不满足!** 衰减太慢
- → 步长衰减不够快, 噪声累积

实践中常用: ρ_t = (t + τ)^{-κ}, 其中 0.5 < κ ≤ 1

</details>

---

### Q3. Score Function vs Reparameterization

对分布 z ~ N(μ, σ²) 和函数 f(z) = z³, 写出:

**(A)** Score function 梯度估计器 (for d/dμ)。

**(B)** Reparameterization 梯度估计器 (for d/dμ)。

**(C)** 解释为什么 (B) 的方差通常比 (A) 低很多。

<details>
<summary>点击查看答案</summary>

**(A) Score Function (REINFORCE):**

```
g_SF = f(z) · ∇_μ log q(z|μ)
     = z³ · (z - μ)/σ²

其中 z ~ N(μ, σ²)
E[g_SF] = d/dμ E[z³] = d/dμ (μ³ + 3μσ²) = 3μ² + 3σ²
```

**(B) Reparameterization:**

```
z = μ + σ·ε,  ε ~ N(0, 1)

g_RP = ∇_z f(z) · ∇_μ z
     = 3z² · 1
     = 3(μ + σ·ε)²

E[g_RP] = 3E[(μ+σε)²] = 3(μ² + σ²) = 3μ² + 3σ² ✓
```

**(C) 方差对比:**

SF 方差取决于 f(z) 的绝对值: Var(f(z)·score) ≈ E[f(z)²·score²]

当 f(z) = z³ 时, z 的 6 次方贡献巨大方差。

RP 方差取决于 f 的局部导数: Var(3z²) — 只涉及 4 次方, 方差显著更低。

更一般地:
- SF: 方差 ∝ Var(f) × Var(score) — f 的规模直接放大方差
- RP: 方差 ∝ Var(∇f) — 只依赖局部变化, 与 f 的绝对值无关

这就是为什么 VAE 必须用 Reparameterization。

</details>

---

### Q4. VAE 的 ELBO 分解

VAE 的 ELBO: L = E_{Q(z|x)}[log P(x|z)] - KL(Q(z|x) || P(z))

**(A)** 如果 encoder Q(z|x) 总是输出 N(0, I)（即 μ=0, σ=1），KL 项是多少？重建项会怎样？

**(B)** 如果 encoder Q(z|x) 输出 σ → 0（确定性的），KL 项会怎样？潜在的问题是什么？

<details>
<summary>点击查看答案</summary>

**(A) Q(z|x) = N(0, I):**

KL(N(0,I) || N(0,I)) = 0

但 μ=0, σ=1 意味着 encoder 完全忽略了输入 x → z 不包含任何关于 x 的信息 → 重建 log P(x|z) 会很差 → ELBO 很低。

这是"后验坍塌"(posterior collapse) 的极端情况 — encoder 不工作。

**(B) Q(z|x) = N(μ(x), σ→0):**

KL(N(μ, ε) || N(0, I)) = 0.5(ε + μ² - 1 - log ε) → ∞ (当 ε→0 时, -log ε → ∞)

即使 σ 很小但不为零:
- KL 项惩罚 σ 太小 (因为 -log σ² 项)
- 如果 μ 偏离 0, KL 进一步惩罚

潜在问题: encoder 过于确定性 → 丧失随机性 → VAE 退化为普通自编码器 → 失去了生成能力 (采样时只能从一个点生成, 而非从连续隐空间中采样)。

**平衡**: KL 项充当正则器, 防止 Q(z|x) 坍缩为点估计, 保证隐空间的连续性和生成能力。

</details>

---

## 🟡 进阶题

### Q5. 自然梯度的几何直觉

**(A)** 对分布族 Q_λ(z) = N(λ_1, exp(λ_2)), 计算 Fisher 信息矩阵 F。

**(B)** 计算从 λ=(0, 0) (即 N(0,1)) 出发, 沿自然梯度方向走一步 (步长=1) 到达的分布参数。

**(C)** 普通梯度下降在同样的起点和步长下会走向哪里？两者有何不同？

<details>
<summary>点击查看答案</summary>

**(A) Fisher 信息矩阵:**

log Q_λ = -0.5 log(2π) - 0.5 λ_2 - 0.5 (z-λ_1)² exp(-λ_2)

F_{ij} = E_Q[∂_i log Q · ∂_j log Q]

```
F = [[exp(-λ_2),     0    ],
     [    0    , 0.5        ]]
```

当 λ=(0,0): F = [[1, 0], [0, 0.5]]

**(B) 自然梯度:**

自然梯度 = F^{-1} · ∇_λ ELBO

在起点 λ=(0,0): F^{-1} = [[1, 0], [0, 2]]

如果 ELBO 关于 λ 的梯度是 g, 自然梯度方向是 F^{-1}·g。

自然梯度对 σ 相关参数 (λ_2) 加速了 2 倍 — 因为 σ 在 N(0,1) 附近变化比 μ 更"敏感"。

**(C) 区别:**

普通梯度: 依赖参数化 — λ 的不同选择 (如用 λ_2 还是 exp(λ_2) 还是 σ) 会产生完全不同的更新方向和大小。

自然梯度: **不依赖参数化** — 无论怎么参数化分布, 在分布空间中的更新方向相同。这是 Riemann 几何中"最陡下降"的正确推广。

**直觉**: 自然梯度是"在 KL 几何下的最陡方向", 普通梯度是"在欧氏几何下的最陡方向"。分布空间是弯曲的 (非欧), 用 KL 度量更合理。

</details>

---

### Q6. Control Variate 的最优系数

以 BBVI 为例, score function 梯度为 g = (f(z) - c) · ∇log Q(z)。

**(A)** 证明 E[(f(z) - c)·∇log Q] = E[f(z)·∇log Q] 对任意常数 c 成立（c 不依赖于 z）。

**(B)** 求最优 c* = argmin_c Var[(f-c)·∇log Q]。

<details>
<summary>点击查看答案</summary>

**(A) 无偏性:**

```
E[(f - c)·∇log Q] = E[f·∇log Q] - c·E[∇log Q]
                  = E[f·∇log Q] - c·0
                  = E[f·∇log Q]
```

因为 E_Q[∇log Q] = ∫ Q·(∇Q/Q) = ∇∫ Q = ∇1 = 0。

**(B) 最优 c*:**

Var[(f-c)·s] = E[(f-c)²·s²] - E[(f-c)·s]²

其中 s = ∇log Q。E[(f-c)·s] = E[f·s] (由A), 这是常数。

令 ∂/∂c Var = 0:
```
∂/∂c E[(f-c)²·s²] = E[-2(f-c)·s²] = 0
→ c·E[s²] = E[f·s²]
→ c* = E[f·s²] / E[s²]
```

**多维情况**: c* = E[f·sᵀs]^{-1} E[f·s] (向量形式)

实践中, 用 Monte Carlo 样本估计这些期望即可得到近最优的 c*。

</details>

---

### Q7. Amortized Inference 的优势

传统 Mean-Field VI: 对每个数据点 x_n 都需要从头运行 CAVI 得到 Q_n(Z)。

VAE: 训练一个 encoder 网络 f_φ(x) → (μ(x), σ(x)), 对所有数据点共享。

**(A)** 从计算复杂度角度, 比较两种方法对 N=10^6 个数据点做推断的开销。

**(B)** Amortized inference 有什么潜在的局限？

<details>
<summary>点击查看答案</summary>

**(A) 计算复杂度:**

传统 Mean-Field:
- 每个数据点运行一次 CAVI: O(K·T) per data point
  (K=变分参数维度, T=CAVI 迭代次数)
- 总开销: O(N·K·T) — 与 N 线性, 但常数大

VAE (Amortized):
- 训练 encoder 一次: O(N·E·D) 
  (E=epochs, D=网络参数量)
- 对**新数据**做推断: O(D) — 仅一次前向传播!
- 总开销: O(N·E·D) + O(N_new·D)

当 N 很大且有很多新数据时, amortized 开销远小于 per-data-point VI。

**(B) 潜在局限:**

1. **Amortization Gap**: Encoder 的输出是对所有数据点的"平均"近似, 对个别数据点可能不如专属的逐点 CAVI 精确。

2. **网络容量限制**: 如果 encoder 网络太小, 可能无法表达复杂的后验结构。

3. **训练-测试分布偏移**: Encoder 在训练数据上训练, 如果测试数据分布变化, encoder 的推断质量会下降 — 而 CAVI 不受此影响 (每次独立优化)。

</details>

---

## 🔴 挑战题

### Q8. IWAE — 更紧的变分下界

标准 VAE 的 ELBO 用单个 MC 样本估计期望。重要性加权自编码器 (IWAE) 用 S 个样本的加权平均:

```
L_S = E_{z_1,...,z_S ~ Q} [log (1/S Σ_{s=1}^S P(x,z_s)/Q(z_s|x))]
```

**(A)** 证明 L_S ≥ L_1 (ELBO), 且 L_S → log P(x) 当 S → ∞。

**(B)** 解释为什么 IWAE 的下界更紧, 但梯度估计方差更大。

<details>
<summary>点击查看答案</summary>

**(A) 证明:**

令 R_s = P(x, z_s) / Q(z_s|x)。需要证明:

E[log(1/S Σ R_s)] ≥ E[log R_1]

由 Jensen 不等式 (log 是凹函数):

E[log(1/S Σ R_s)] ≥ log E[1/S Σ R_s] = log(P(x))

实际上我们需要证明 L_{S+1} ≥ L_S。

```
L_{S+1} = E[log(1/(S+1) Σ_{s=1}^{S+1} R_s)]
        = E[log(1/(S+1) (Σ_{s=1}^S R_s + R_{S+1}))]
        
由对称性, 这等价于:
= E[log(1/S Σ_{s=1}^S E[R_{S+1}|R_1..R_S] average)]
  ≥ E[log(1/S Σ R_s)]  (Jensen again)
= L_S
```

且 L_S → log P(x) 当 S → ∞ (大数定律 + log 连续性)。

**(B) 更紧但方差更大:**

更紧: 使用更多 MC 样本 → 更好地估计 P(x) → 下界更高 → 后验近似更好。

方差更大: IWAE 的梯度包含重要性权重:
```
∇L_S = Σ_s w_s · ∇log(P(x,z_s)/Q(z_s)),  w_s = R_s / Σ_t R_t
```

当 S 较小时, w_s 的行为类似"赢家通吃" → 有效样本数少 → 方差大。

实践中: S=5 或 S=10 在紧度和方差之间取得平衡。

</details>

---

### Q9. Normalizing Flows — 超越高斯 Q

标准 VAE 的 Q(z|x) 是高斯分布 — 表达能力有限。Normalizing Flow 通过一系列可逆变换增强 Q:

```
z_0 ~ N(0, I)
z_K = f_K ∘ f_{K-1} ∘ ... ∘ f_1(z_0)

log Q(z_K) = log N(z_0;0,I) - Σ_k log |det ∂f_k/∂z_{k-1}|
```

**(A)** 为什么 Normalizing Flow 能表达比高斯更复杂的后验？

**(B)** 什么条件保证了变换后的 Q(z_K) 仍是合法分布且可以高效采样和计算密度？

<details>
<summary>点击查看答案</summary>

**(A) 表达能力:**

高斯 Q: 单峰、对称、薄尾 → 无法表达多峰或偏斜的后验。

Normalizing Flow: K 步可逆非线性变换 → 可以把简单的高斯"弯曲"成几乎任意形状的分布:
- 可以产生多峰 (通过"折叠"空间)
- 可以改变尾部厚度
- 可以产生相关性

类似于: 高斯 = 橡皮泥球, Normalizing Flows = 可以捏成任意形状。

**(B) 合法分布的条件:**

1. **可逆性 (Invertibility)**: 每个 f_k 必须是双射 (bijection) → 保证从 z_K 能唯一回到 z_0

2. **可微性**: f_k 和 f_k^{-1} 必须是可微的 (diffeomorphism)

3. **Jacobian 行列式易算**: |det ∂f_k/∂z_{k-1}| 必须能在 O(D) 时间内计算 (不能直接 O(D³) 算行列式)

常见满足条件的变换:
- **Planar Flow**: f(z) = z + u·h(w^T z + b), det 可用 matrix determinant lemma O(D)
- **Real NVP**: 仿射耦合层 (affine coupling), Jacobian 是三角阵 → det 是 diagonal 乘积 O(D)
- **Inverse Autoregressive Flow (IAF)**: 同样利用三角 Jacobian

这些变换保持计算可行性同时大幅增强 Q 的灵活性。

</details>

---

### Q10. VAE 中的 "后验坍塌" 问题

在训练 VAE (尤其是用强 decoder, 如 PixelCNN) 时, 常出现 KL(Q||P) → 0 的现象。

**(A)** 解释这个现象的成因。

**(B)** 提出至少两种缓解策略。

<details>
<summary>点击查看答案</summary>

**(A) 成因:**

当 decoder P_θ(x|z) 非常强大 (能仅从噪声 z ~ N(0,I) 生成好结果) 时:

ELBO = E_Q[log P(x|z)] - KL(Q||P)

Decoder 发现: 与其依赖 encoder 提供有用的 z, 不如直接忽略 z (或只用噪声), 自己学会生成 x。

此时:
- Encoder 被迫输出 Q(z|x) ≈ P(z) = N(0, I) → KL → 0
- Decoder 学会了 P(x|z) ≈ P(x) (忽略 z)
- z 和 x 之间没有互信息 → VAE 退化为普通生成模型

这是"懒惰 encoder"问题 — decoder 太强, encoder 不需要工作。

**(B) 缓解策略:**

**1. KL Annealing (KL 退火)**:
训练初期给 KL 项乘一个小权重 β → 1:
```
L = E_Q[log P(x|z)] - β_t · KL(Q||P)
β_t: 0 → 1 (如线性从 0 增到 1 在最初几轮)
```
让 encoder 先"学会编码有用信息", 再逐步引入 KL 正则。

**2. Free Bits**:
限制 KL 每个维度不能低于某个阈值 λ:
```
KL_modified = Σ_d max(KL_d, λ)
```
确保每个隐维度至少保留 λ nats 的信息 → encoder 被迫使用所有隐维度。

**3. Weak Decoder**:
故意限制 decoder 的能力 (如更小的网络、更简单的结构) → decoder 需要依赖 encoder 的 z。

**4. Aggressive Training of Encoder**:
给 encoder 更大的学习率, 或先用 reconstruction-only loss 预训练 encoder。

</details>

---

## 📊 综合自测评分

每题 10 分，共 100 分。

| 得分 | 评价 |
|------|------|
| 90-100 | L8 完全掌握, 理解了 VI 的现代扩展和深度学习关联 |
| 70-89  | 主干扎实, 建议亲自动手做一个 mini-VAE |
| 50-69  | 概念清晰但需加强推导, 重看 Reparameterization 和 SVI 部分 |
| < 50   | 先吃透 Q1-Q4, 确保理解 Wake-Sleep, SVI, Reparam, VAE ELBO |

---

> L8 完成了 VI 的进阶之旅。从 CAVI 到 SVI 到 VAE 到 Normalizing Flows, VI 从一个小众的推断方法发展为深度学习时代生成模型的核心技术。
