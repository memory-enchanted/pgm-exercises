# CMU 10-708 Lecture 8 概念体系梳理 — 变分推断 II (Advanced VI)

> 整理自 Eric Xing 教授 CMU 10-708 Lecture 8: Variational Inference II — SVI, BBVI, Wake-Sleep, VAE
>
> 核心教材: Bishop Ch.10, Murphy Ch.21, Blei et al. (2017), Kingma & Welling (2014, VAE)

---

## 📐 全局定位：L7 → L8 的进阶

```
L7: 变分推断基础 (VI I)                L8: 变分推断进阶 (VI II)
─────────────────────────             ─────────────────────────────
Mean-Field 假设                        Wake-Sleep (无监督学习)
CAVI (坐标上升)                        SVI (随机梯度 → 大数据)
ELBO 单调收敛                          BBVI (黑盒 → 非共轭模型)
                                       Reparameterization (低方差梯度)
精确推断的替代方案                      VAE (深度生成模型)

L7 回答: "怎么用优化做推断?"           L8 回答: "怎么让 VI 实际可用?
                                        大数据? 非共轭? 深度学习?"
```

**一句话概括 L8**：L7 的 CAVI 在实际中有诸多限制（需要全量数据、需要条件共轭、无法扩展到深度模型）。L8 介绍一系列技术（SVI, BBVI, Wake-Sleep, Reparameterization）来突破这些限制，最终引出 Variational Autoencoder (VAE) — 将 VI 思想与深度学习融合的标志性成果。

---

## 概念 1：Wake-Sleep 算法 — 无监督学习的 VI 视角

### 动机

CAVI 需要知道完整的条件分布 P(Z_i | Z_{-i}, X) — 即"给定其他变量和数据, Z_i 的后验"。但很多时候这个条件分布很复杂。

**Wake-Sleep 的核心思想**: 用一个**识别网络 (recognition network)** Q(Z|X) 来近似后验, 并交替更新识别网络和生成模型。

### 两阶段训练

```
┌──────────────────────────────────────────────────────────────┐
│                                                               │
│  Wake 阶段 (醒着): 用真实数据训练生成模型                     │
│                                                               │
│    1. 给定数据 X_real, 用识别网络 Q(Z|X_real) 采样 Z          │
│    2. 用采样到的 (Z, X_real) 更新生成模型 P(X|Z)              │
│       → 最大化 E_{Q(Z|X)}[log P(X_real | Z)]                 │
│                                                               │
│  Sleep 阶段 (睡着): 用生成的数据训练识别网络                  │
│                                                               │
│    1. 从先验 P(Z) 采样 Z, 用生成模型 P(X|Z) 生成 X_fake       │
│    2. 用配对数据 (Z, X_fake) 更新识别网络 Q(Z|X_fake)         │
│       → 最大化 E_{P(Z,X)}[log Q(Z|X_fake)]                   │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### 与 EM 算法的联系

```
EM 的 E 步 = 计算精确后验 P(Z|X, θ^old)
Wake-Sleep = 用一个网络 Q 去近似后验 → 不需要精确后验!
```

Wake-Sleep 是**近似 EM** — 牺牲精确性换来可扩展性。

### 局限

Wake-Sleep 优化的不是同一个目标函数:
- Wake 阶段: 优化 log P(X|Z)
- Sleep 阶段: 优化 log Q(Z|X)

两个阶段优化不同目标 → 不保证收敛 → 这是后续 VAE 要解决的问题。

---

## 概念 2：随机变分推断 (SVI) — 大数据上的 VI

### CAVI 的瓶颈

```
CAVI 每轮迭代:
  for i in 1..N:
    log Q_i ← E_{Q_{-i}}[log P(X, Z)]  ← 需要全部数据 X!

数据量 N = 1,000,000 → 每轮太慢!
```

### SVI 的核心思想

把 ELBO 写成数据点的和, 用**随机梯度**更新。

```
ELBO(Q) = Σ_{n=1}^N E_Q[log P(x_n | Z)] - KL(Q(Z) || P(Z))
         = N × (平均每个数据点的 ELBO)

SVI:
  1. 采样一个小批量 (mini-batch) 数据 S ⊂ {1..N}
  2. 计算有噪声但无偏的梯度 g ≈ ∇ELBO
  3. 用自然梯度 (natural gradient) 更新变分参数:
     λ ← λ + ρ_t · g
```

### 自然梯度 vs 普通梯度

```
普通梯度: ∇_λ ELBO     ← 依赖参数化
自然梯度: F^{-1} ∇_λ   ← F 是 Fisher 信息矩阵, 不依赖参数化

自然梯度 = 在分布空间中的最陡下降方向 (KL 几何下的梯度)

对于指数族: 自然梯度 = E_{Q}[充分统计量] - λ  ← 极其简洁!
```

### Robbins-Monro 步长条件

```
ρ_t 满足:
  Σ_t ρ_t = ∞       (走无限远 → 能到达最优点)
  Σ_t ρ_t^2 < ∞     (步长衰减 → 最终收敛)

典型选择: ρ_t = (t + τ)^{-κ},  0.5 < κ ≤ 1
```

---

## 概念 3：黑盒变分推断 (BBVI) — 非共轭模型

### 问题

CAVI 要求: log P(Z_i | Z_{-i}, X) 属于某个指数族, 且与 Q_i 共轭 → 才有闭式更新。

如果模型是"非共轭"的 (如包含神经网络, 或任意似然函数), CAVI 无法直接使用。

### BBVI 方案：Score Function Gradient (REINFORCE)

```
∇_λ ELBO = ∇_λ E_{Q_λ(Z)}[log P(X, Z) - log Q_λ(Z)]

用 score function identity: ∇_λ Q_λ = Q_λ · ∇_λ log Q_λ

→ ∇_λ ELBO = E_{Q_λ}[ (log P(X,Z) - log Q_λ(Z)) · ∇_λ log Q_λ(Z) ]
              └────────────┬────────────┘   └──────┬──────┘
                     "优势函数" A(Z)            score function

Monte Carlo 估计:
  g = (1/S) Σ_{s=1}^S A(z_s) · ∇_λ log Q_λ(z_s),  z_s ~ Q_λ
```

### 核心问题：高方差

Score function gradient 的方差通常很大（尤其在 Z 维度高时）→ 收敛极慢。

### 解决方案

| 方法 | 原理 |
|------|------|
| **Rao-Blackwellization** | 用条件期望减少方差: E[A·∇log Q] = E[E[A\|subset]·∇log Q_subset] |
| **Control Variates** | g_cv = g - c·(∇log Q) + E[c·∇log Q]; 选 c 使 Var[g_cv] 最小 |
| **Reparameterization** | 改变采样方式, 避免 score function (见概念4) |

---

## 概念 4：Reparameterization Trick — 低方差梯度的关键 (🔑🔑🔑)

### 核心思想

如果 z ~ Q_λ(z) 可以写为:
```
z = g(ε, λ),   ε ~ p(ε) (不依赖 λ 的简单噪声)
```

那么:
```
∇_λ E_{Q_λ(z)}[f(z)] = ∇_λ E_{p(ε)}[f(g(ε, λ))]
                      = E_{p(ε)}[∇_λ f(g(ε, λ))]
                      = E_{p(ε)}[∇_z f(g(ε, λ)) · ∇_λ g(ε, λ)]
```

### 常见 Reparameterization

```
分布 Q_λ         重参数化
─────────────────────────────────────
N(μ, σ²)       z = μ + σ·ε,  ε~N(0,1)
Gamma(α, β)    z ~ Gamma(α, 1)/β  (不完全 reparameterizable)
Bernoulli(p)   连续松弛: Gumbel-Softmax / Concrete
Dirichlet(α)   同上
```

### 为什么 Reparam 梯度方差低？

```
Score function: g_SF = f(z) · ∇_λ log Q(z)
  → f(z) 和 ∇log Q 都可能很大, 乘积方差更大

Reparameterization: g_RP = ∇_z f · ∇_λ g
  → 只依赖 f 的局部梯度, 不依赖 f 的绝对值
  → 方差通常比 SF 低 1-3 个数量级!
```

**这就是 VAE 可行的数学基础。**

---

## 概念 5：变分自编码器 (VAE) (🔑🔑🔑)

### VAE = 神经网络 + VI

```
VAE 的 ELBO:

  L(θ, φ; x) = E_{Q_φ(z|x)}[log P_θ(x|z)] - KL(Q_φ(z|x) || P(z))
               └────────┬────────────┘   └──────────┬──────────┘
                  重建损失 (decoder)         正则项 (encoder 别跑太远)

其中:
  Q_φ(z|x) = N(z | μ_φ(x), σ²_φ(x))   ← 编码器 (encoder, 识别网络)
  P_θ(x|z) = 某种分布 (Bernoulli/高斯) ← 解码器 (decoder, 生成网络)
  P(z) = N(0, I)                       ← 先验
```

### VAE 结构

```
        μ_φ(x) + σ_φ(x)·ε
  x ───→ [Encoder NN] ───→ z ───→ [Decoder NN] ───→ x̂
           ↑                          ↑
       参数 φ                      参数 θ
       
  训练: 最大化 ELBO via reparameterized gradient + SGD
  生成: 从 N(0,I) 采样 z → 通过 Decoder → 生成新样本
```

### VAE vs 传统 VI

| 维度 | 传统 CAVI | VAE |
|------|----------|-----|
| 变分参数 | 每个数据点有独立的 Q_i | Encoder 网络共享参数 (amortized) |
| 优化 | 闭式坐标上升 | SGD + reparam gradient |
| 模型 | 指数族共轭 | 任意神经网络 |
| 数据量 | 小/中等 | 大规模 (mini-batch) |
| 推断 | 逐个数据点推断 | 一次性前向传播 → 快速 |

### Amortized Inference (摊销推断)

传统 VI: 对每个数据点 x_n, 都需要运行一轮 CAVI 来得到 Q(z|x_n)

VAE: 训练一个 Encoder 网络 → 输入 x → 直接输出 Q(z|x) 的参数

**"摊销"的含义**: 训练 Encoder 的一次性开销被所有数据点共享 → 新数据点只需一次前向传播。

---

## 📋 全部概念一张表

| 概念 | 一句话 |
|------|--------|
| **Wake-Sleep** | 交替训练生成模型 (wake) 和识别网络 (sleep) |
| **SVI** | 用 mini-batch 随机梯度 + 自然梯度做 VI, 扩展到大数据 |
| **自然梯度** | 在分布空间 (KL 几何) 下, 不依赖参数化的最陡方向 |
| **BBVI** | Score function gradient, 适用于非共轭模型的"万能"VI |
| **Score Function** | ∇log Q — 衡量参数变化对分布的影响, 但梯度方差大 |
| **Control Variate** | 减方差技术: 用已知期望的项抵消 score function 的噪声 |
| **Reparameterization** | z = g(ε,λ) → ∇E[f] = E[∇f·∇g], 方差极低 |
| **VAE** | 编码器+解码器+Reparam → 深度生成模型, VI 的现代应用 |
| **Amortized Inference** | 用网络学推断 → 一次前向传播替代逐点 CAVI |

---

## 🔗 概念关系图

```
              L7: VI I (CAVI, Mean-Field, ELBO)
                         │
              局限性: 全量数据, 共轭模型, 浅层
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   Wake-Sleep          SVI              BBVI
   近似EM          随机梯度+自然梯度     Score Func
        │                │                │
        │          大数据VI             方差太大!
        │                │                │
        └────────────────┼────────────────┘
                         │
                         ▼
               Reparameterization Trick
                (低方差梯度估计)
                         │
                         ▼
                  VAE (VAriational Autoencoder)
                  编码器 + 解码器 + Reparam
                  深度生成模型 | Amortized VI
                         │
           ┌─────────────┼─────────────┐
           ▼             ▼             ▼
        IWAE        Normalizing     β-VAE
    (更紧的下界)      Flows        (解耦表示)
```

---

## 🎯 核心洞见一句话

| # | 洞见 |
|---|------|
| 1 | **Wake-Sleep = 近似 EM** — 用识别网络 Q 替代精确后验, 但优化目标分裂 |
| 2 | **SVI = CAVI + SGD** — 自然梯度 + mini-batch → VI 扩展到百万级数据 |
| 3 | **BBVI = 万能但高方差** — Score function gradient 通用于非共轭, 但需要方差缩减 |
| 4 | **Reparameterization = VI 的魔法** — 把不可微的采样变成可微的变换, 方差降低 100x |
| 5 | **VAE = Neural VI** — Amortized 推断 + Reparam + 神经网络 → 深度生成模型的天作之合 |

---

## 🧪 自测清单（看 L8 前带着这些问题）

- [ ] Wake-Sleep 和 EM 的区别是什么？为什么 Wake-Sleep 不保证收敛？
- [ ] SVI 的"自然梯度"和普通梯度有什么不同？为什么用自然梯度更新？
- [ ] Robbins-Monro 条件 Σρ_t=∞, Σρ_t²<∞ 分别保证了什么？
- [ ] Score function gradient 的方差为什么大？怎样降低？
- [ ] Reparameterization trick 适用的条件是什么？哪些分布可以 reparameterize？
- [ ] VAE 的 encoder 为什么叫"amortized inference"？
- [ ] VAE 的 ELBO 中 KL 项 (KL(Q(z\|x) || P(z))) 起到了什么正则作用？

---

> L8 完成了 VI 的进阶之旅 — 从 L7 的基础 CAVI 到能驱动深度生成模型的 VAE。L9 将进入 MCMC 采样方法 — 另一套完全不同的近似推断哲学。
