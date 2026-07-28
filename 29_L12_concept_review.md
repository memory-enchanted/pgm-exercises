# CMU 10-708 Lecture 12 概念体系梳理 — 深度生成模型 I

> 整理自 Eric Xing 教授 CMU 10-708 Lecture 12: Deep Generative Models I — RBM, DBN, DBM
>
> 核心教材: Goodfellow et al. (2016) Deep Learning Ch.20, Hinton (2002, 2006), Salakhutdinov (2015)

---

## 📐 全局定位：PGM + DL → 深度生成模型

```
L1-L10: PGM 推断                          L11: DL 统计/算法基础
    │                                          │
    └──────────────┬────────────────────────────┘
                   │
                   ▼
       L12: 深度生成模型 I (DGM)
       Restricted Boltzmann Machines
       Deep Belief Networks
       Deep Boltzmann Machines
       
     = PGM 结构 + 深度学习优化技巧
     = 把 PGM 的生成建模能力 "深" 出来!
```

**一句话概括 L12**: 深度生成模型把 PGM 的"图结构"引入深度网络 — 用 RBM 作为构建块, 层叠成 Deep Belief Network / Deep Boltzmann Machine, 实现层次化的特征学习和生成。

---

## 概念 1：生成模型 vs 判别模型

```
生成模型 (Generative):           判别模型 (Discriminative):
    P(X, Z) 联合建模                 P(Y|X) 直接建模条件
    → 可以从头生成数据                → 只能分类/回归
    → 可处理缺失数据                  → 更简单, 通常更准
    → 可做异常检测                   → 不需要建模 X 的分布
    
PGM: 天然适合生成建模               L11: 大多是判别式
L12: 深度化的生成模型                L13: 将扩展到生成式
```

---

## 概念 2：Restricted Boltzmann Machine (RBM) (🔑🔑🔑)

### 结构

```
        h₁    h₂    ...    h_M     ← 隐层 (binary)
         ╱╲   ╱╲         ╱╲
        ╱  ╲ ╱  ╲ ...  ╱  ╲      W (权重矩阵)
       v₁   v₂   ...   v_N        ← 可见层 (binary)
```

RBM 是一个**无向二部图**: 可见层 ↔ 隐层, 无层内连接。

### 能量函数与联合分布

```
┌──────────────────────────────────────────────────────────────┐
│                                                               │
│  Energy: E(v, h) = -vᵀ W h - aᵀ v - bᵀ h                   │
│                                                               │
│  Joint:  P(v, h) = (1/Z) · exp(-E(v, h))                    │
│                                                               │
│  Marginal: P(v) = (1/Z) Σ_h exp(-E(v, h))                   │
│                                                               │
│  Z = Σ_{v,h} exp(-E(v, h))  ← Partition Function (难!)       │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### 条件独立性 (二部图的关键性质!)

由于无层内连接, 给定可见层, 隐层节点条件独立:

```
P(h_j = 1 | v) = σ(b_j + Σ_i W_{ij} v_i)

P(v_i = 1 | h) = σ(a_i + Σ_j W_{ij} h_j)

其中 σ(x) = 1/(1 + e^{-x})  (sigmoid)
```

**这个条件独立性使得 Gibbs 采样在 RBM 上极高效!** (Block Gibbs: 交替采样 v|h 和 h|v)

---

## 概念 3：对比散度 (Contrastive Divergence, CD) (🔑🔑🔑)

### 问题

RBM 的 log-likelihood 梯度:

```
∂ log P(v) / ∂W_{ij} = E_{data}[v_i h_j] - E_{model}[v_i h_j]
                       └── "正相位" ──┘   └── "负相位" ──┘
                       期望在 P(h|v) 下     期望在 P(v,h) 下
                       → 容易计算           → 需要从模型采样 (难!)
```

### CD-k 算法

```
┌──────────────────────────────────────────────────────────────┐
│                                                               │
│  CD-k (Contrastive Divergence with k Gibbs steps):            │
│                                                               │
│  对每个训练样本 v^(0):                                        │
│                                                               │
│  Positive Phase:                                              │
│    h^(0) ~ P(h | v^(0))      ← 从数据出发, 条件采样           │
│                                                               │
│  Negative Phase (k 步 Gibbs):                                 │
│    For t = 1..k:                                              │
│      v^(t) ~ P(v | h^(t-1))  ← 交替 Gibbs                    │
│      h^(t) ~ P(h | v^(t))                                    │
│                                                               │
│  Update:                                                      │
│    ΔW = lr · (v^(0)·h^(0)^T - v^(k)·h^(k)^T)                 │
│                                                               │
│  CD-1 (k=1): 最快, 在实践中效果就很好!                        │
│  CD-∞: 等价于精确 MLE (但太慢)                                │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### CD 为什么有效?

精确 MLE 需要: 从 P(v,h) 采样 (稳态 → 需要无限步 Gibbs)
CD 近似: 从数据 v^(0) 出发, 只跑 k 步 → 用"受扰的 data"替代"model sample"

**CD 不是 MLE 的无偏估计**, 但在实践中效果出奇好。

---

## 概念 4：深度信念网络 (Deep Belief Network, DBN)

### 结构

```
    h₃    h₃    ...          ← 顶层 (RBM, undirected)
     │╲    │╲
    h₂    h₂    ...          ← 隐层 2
     │╲    │╲
    h₁    h₁    ...          ← 隐层 1
     │╲    │╲
    v₁    v₂    ...          ← 可见层

顶层是 RBM (无向), 下面各层是 Sigmoid Belief Net (有向, top-down)
```

### Greedy Layer-wise Pretraining

```
┌──────────────────────────────────────────────────────────────┐
│                                                               │
│  Step 1: 训练 RBM(v, h1) → 得到 W₁                           │
│                                                               │
│  Step 2: 用 h1 ~ P(h1|v) 的样本作为 "数据",                  │
│          训练 RBM(h1, h2) → 得到 W₂                           │
│                                                               │
│  Step 3: 同上, 训练 RBM(h2, h3) → 得到 W₃                    │
│                                                               │
│  Step 4 (可选): 用 Wake-Sleep 或 BP 对整个 DBN fine-tune      │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

**为什么有效**: 每层学习"更好的数据表示" — 下层学到 edges, 中层学到 parts, 顶层学到 objects。

---

## 概念 5：深度玻尔兹曼机 (DBM)

### DBM vs DBN

```
DBN: 顶层无向 + 下面有向 (Sigmoid Belief Net)
DBM: 全部无向! (每一层都是无向二部图)

DBM 更"一致" (纯能量模型), 但训练更难
```

### DBM 训练

- 不能简单 greedy layer-wise (因为所有边都是无向的)
- 需要更复杂的近似方法: mean-field inference + 联合训练
- 或使用 Salakhutdinov & Hinton (2009) 的两阶段方法

---

## 概念 6：Wake-Sleep / VAE / GAN — 三代训练范式的对比 (🔑🔑)

### 三者都解决"生成模型怎么训练?"

```
              ┌──────────────────┬───────────────────┬──────────────────┐
              │   Wake-Sleep     │       VAE         │       GAN        │
              │   (Hinton 1995)  │ (Kingma 2013)     │ (Goodfellow 2014)│
├──────────────┼──────────────────┼───────────────────┼──────────────────┤
│ 核心思想     │ 交替训练识别网络  │ Encoder+Decoder   │ Generator vs     │
│              │ 和生成网络       │ + Reparam + ELBO  │ Discriminator    │
├──────────────┼──────────────────┼───────────────────┼──────────────────┤
│ 优化目标     │ Wake: log P(X|Z)  │ 最大化 ELBO       │ min_G max_D      │
│              │ Sleep:log Q(Z|X)  │ = E[logP]-KL     │ V(D,G)           │
├──────────────┼──────────────────┼───────────────────┼──────────────────┤
│ 目标一致性   │ ❌ 不同!          │ ✅ 同一个 ELBO    │ ✅ 同一个博弈     │
│              │ Wake和Sleep       │ Encoder和Decoder  │ G和D共享目标      │
│              │ 优化不同loss       │ 共享ELBO          │                  │
├──────────────┼──────────────────┼───────────────────┼──────────────────┤
│ 收敛保证     │ ❌ 不保证         │ ✅ ELBO单调上升   │ ❌ 不保证         │
│              │ (目标不一致导致    │ (VI的标准性质)    │ (博弈可能震荡)    │
│              │  可能发散)        │                   │                  │
├──────────────┼──────────────────┼───────────────────┼──────────────────┤
│ 推断方式     │ 识别网络Q(Z|X)    │ Encoder Q(Z|X)    │ 不需要推断!       │
│              │ (近似后验)        │ (Amortized VI)    │ (G直接从z生成x)   │
├──────────────┼──────────────────┼───────────────────┼──────────────────┤
│ 生成方式     │ P(X|Z) 从上到下   │ Decoder P(X|Z)    │ Generator G(z)    │
│              │ 生成              │ 从z采样生成       │ 从噪声生成        │
├──────────────┼──────────────────┼───────────────────┼──────────────────┤
│ 可评价likelihood│ ❌ 只能算bound  │ ❌ ELBO是下界     │ ❌ 完全无法算     │
│              │ (Wake-Sleep不     │ (VAE的ELBO)       │ (没有P(x)的表达式) │
│              │  直接优化bound)   │                   │                  │
├──────────────┼──────────────────┼───────────────────┼──────────────────┤
│ 主要问题     │ 目标分裂 → 不收敛 │ 生成图像模糊      │ Mode Collapse     │
│              │                  │ (pixel-wise loss) │ 训练不稳定        │
└──────────────┴──────────────────┴───────────────────┴──────────────────┘
```

### 三者的历史关系

```
Wake-Sleep (1995) ──→ 启发了 VAE (2013)
    │                      │
    │  "用网络近似后验"      │  "用一个ELBO统一优化"
    │  "交替训练两个网络"    │  "Reparam trick实现低方差梯度"
    │                      │
    └────── 启发了 GAN (2014) ──┘
           "对抗博弈替代likelihood"
           "不需要推断隐变量!"
```

**Wake-Sleep → VAE 的进化**: 
- Wake-Sleep 有两个不同的优化目标 → VAE 用一个 ELBO 统一了 encoder 和 decoder 的训练
- Wake-Sleep 用采样的离散 z → VAE 用 Reparameterization 实现可微的 z
- VAE 有收敛保证 (ELBO 单调), Wake-Sleep 没有

**VAE → GAN 的进化**:
- VAE 的生成质量受限于 pixel-wise reconstruction loss → 模糊
- GAN 完全抛弃了 likelihood — 用 Discriminator 判断"真假"比算 pixel MSE 更接近人类感知
- GAN 不需要显式推断 z → 更简单, 但训练更难

---

## 📋 全部概念一张表

| 概念 | 一句话 |
|------|--------|
| **RBM** | 无向二部图: E(v,h) = -vWh - av - bh, P(v,h) ∝ e^{-E} |
| **条件独立性** | P(h\|v) = ∏ P(h_j\|v), P(v\|h) = ∏ P(v_i\|h) — 无层内连接! |
| **Block Gibbs** | 交替采样 v\|h 和 h\|v — 极高效 |
| **CD-k** | 从数据出发 k 步 Gibbs → 近似负相位梯度 |
| **DBN** | 堆叠 RBM + 顶层无向 + 下层有向 → greedy layer-wise 预训练 |
| **DBM** | 全无向的深层模型 — 更优雅但更难训练 |
| **Partition Function** | Z = Σ e^{-E} — RBM 的归一化常数, 难计算 (CD 绕过了它) |

---

## 🔗 概念关系图

```
            Ising / MRF (L2-L3)
           能量模型 E(x) = -θᵀφ(x)
                   │
                   ▼
              RBM (二部图, 无向)
              E(v,h) = -vᵀWh - aᵀv - bᵀh
                   │
          ┌────────┼────────┐
          ▼        ▼        ▼
        CD-k     PCD    Score Matching
       (训练)   (训练)   (训练)
          │
          ├──→ DBN (RBM积木 + 有向层)
          │     greedy layer-wise
          │
          └──→ DBM (全无向深层)
                mean-field + joint training
```

---

## 🎯 核心洞见一句话

| # | 洞见 |
|---|------|
| 1 | **RBM = 二部无向图** — 无层内连接 → 条件独立 → Block Gibbs 高效 |
| 2 | **CD-k = 近似的 MLE** — 从数据出发替代从模型采样, k=1 就够用 |
| 3 | **能量函数 → 生成模型** — P(v) ∝ Σ_h e^{-E(v,h)} : 低能量配置 = 高概率 |
| 4 | **Layer-wise = 层次化特征** — 每层学更深层次的数据表示 |
| 5 | **RBM → DBN → 深度学习革命** — 2006年 Hinton 的 breakthrough |
| 6 | **架构即先验** — RBM的二部图结构本身就是一种"领域知识"的编码：层内无连接 = 条件独立 = Block Gibbs可行 |

---

## 🧪 自测清单

- [ ] RBM 的能量函数 E(v,h) 由哪三项组成? 对应的参数是什么?
- [ ] 为什么 RBM 的 P(h\|v) 可以因子分解为 ∏ P(h_j\|v)? 什么结构性质保证了这一点?
- [ ] CD-k 中的 k 越小/越大 分别意味着什么? CD-1 为什么在实践中最常用?
- [ ] DBN 的 greedy layer-wise pretraining 依次训练了哪些 RBM?
- [ ] DBM 和 DBN 在结构上最核心的区别是什么?

---

> L12 是深度生成模型的"第一代" — RBM 和 DBN。L13 将进入"第二代"深度生成模型: VAE (回顾 L8) 和 GAN。
