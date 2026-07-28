# CMU 10-708 Lecture 13 概念体系梳理 — 深度生成模型 II

> 整理自 Eric Xing 教授 CMU 10-708 Lecture 13: Deep Generative Models II — VAE, GAN, Flows, Diffusion
>
> 核心教材: Goodfellow et al. (2016) Ch.20, Kingma & Welling (2014, 2019), Goodfellow et al. (2014), Ho et al. (2020)

---

## 📐 全局定位：DGM I → DGM II 的进化

```
L12 (DGM I): 第一代深度生成模型            L13 (DGM II): 第二代深度生成模型
────────────────────────────            ──────────────────────────────
RBM (能量模型, CD训练)                   VAE (变分推断 + 神经网络)
DBN (逐层预训练)                         GAN (对抗训练, minimax博弈)
DBM (全无向深层)                         Normalizing Flow (可逆变换)
                                         Autoregressive (逐像素生成)
                                         Diffusion Model (去噪扩散)

L12: 能量函数是核心                      L13: 不需要能量函数/partition function!
```

**一句话概括 L13**: 2014年后的深度生成模型抛弃了能量函数和 partition function, 转而用神经网络直接参数化生成过程 — VAE 用 ELBO, GAN 用对抗博弈, Flow 用可逆变换, Diffusion 用去噪过程。

---

## 概念 1：四类深度生成模型的统一视角

```
┌──────────────────────────────────────────────────────────────┐
│                                                               │
│  生成模型 = 如何从简单噪声 z 到复杂数据 x?                    │
│                                                               │
│  VAE:       x = decoder(z),  z ~ Q(z|x) [Encoder]            │
│             训练: 最大化 ELBO                                  │
│                                                               │
│  GAN:       x = generator(z), z ~ p(z)                       │
│             训练: minimax 博弈 (G 骗 D, D 区分真假)            │
│                                                               │
│  Flow:      x = f_K ∘ ... ∘ f_1(z), z ~ N(0,I)              │
│             训练: 最大化 exact log-likelihood                  │
│                                                               │
│  Diffusion: x_T → x_{T-1} → ... → x_0 (逐步去噪)             │
│             训练: 匹配去噪过程的 score function               │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## 概念 2：GAN — 对抗生成网络 (🔑🔑🔑)

### 核心思想

```
Generator G: z → x_fake    "造假者"  想骗过 Discriminator
Discriminator D: x → [0,1] "鉴别者"  想区分真假

Minimax Game:
  min_G max_D  E_{x~P_data}[log D(x)] + E_{z~P_z}[log(1 - D(G(z)))]
```

### 训练过程

```
For each iteration:
  # 1. 训练 D (k 步)
  x_real ~ P_data, x_fake = G(z), z ~ P_z
  D_loss = -[log D(x_real) + log(1 - D(x_fake))]
  更新 D 以增大 D_loss

  # 2. 训练 G (1 步)
  z ~ P_z, x_fake = G(z)
  G_loss = -log D(G(z))    ← "骗 D"的损失
  更新 G 以减小 G_loss
```

### GAN 的核心挑战

| 挑战 | 描述 |
|------|------|
| **Mode Collapse** | G 只生成少数几个 mode, 忽略其他 → 多样性差 |
| **Training Instability** | G 和 D 的博弈可能不收敛 → 震荡, 梯度消失 |
| **Evaluation** | 没有 likehood → 难量化评估 (用 IS, FID) |
| **Vanishing Gradient** | D 太强 → D(G(z))≈0 → G 的梯度消失 |

---

## 概念 3：Normalizing Flows — 可逆生成模型 (🔑🔑)

### 核心思想

```
z_0 ~ N(0, I)  (简单分布)
  ↓ f_1 (可逆变换)
z_1
  ↓ f_2
 ...
  ↓ f_K
x = z_K  (复杂分布)

log P(x) = log N(z_0; 0, I) + Σ_k log |det ∂f_k/∂z_{k-1}|
```

**关键**: 如果 f_k 可逆且 Jacobian 好算 → exact log-likelihood!

### 两种经典 Flow

```
Planar Flow:
  f(z) = z + u·h(w^T z + b)
  log|det| = log|1 + u^T h'(w^T z + b) w|  ← O(D)

Real NVP (Affine Coupling):
  分两半: z_{1:d} (不变), z_{d+1:D} = z_{d+1:D} ⊙ exp(s(z_{1:d})) + t(z_{1:d})
  Jacobian 是三角阵 → log|det| = Σ s_i  ← O(D)
```

---

## 概念 4：自回归模型 — 逐维生成

### MADE / PixelCNN

```
P(x) = P(x_1) · P(x_2|x_1) · P(x_3|x_1,x_2) · ... · P(x_D|x_{1:D-1})

每个维度条件依赖于前面的维度:
  x̂_d = f(x_{1:d-1}; θ)

训练: 最大化 Σ log P(x_d | x_{1:d-1}) (teacher forcing)
生成: 逐维采样 (慢! O(D) steps)
```

---

## 概念 5：扩散模型 — 去噪生成 (🔑)

### DDPM (Denoising Diffusion Probabilistic Models)

```
Forward (加噪): x_0 → x_1 → ... → x_T (逐步加高斯噪声 → 纯噪声)
Reverse (去噪): x_T → x_{T-1} → ... → x_0 (学一个网络去噪)

训练: 预测每个时间步添加的噪声 ε
  Loss = E[|| ε_θ(x_t, t) - ε ||²]

生成: 从 x_T ~ N(0,I) 开始, 逐步去噪 → x_0
```

### 为什么 Diffusion 成为 SOTA?

- 训练稳定 (简单的 MSE loss, 无对抗, 无可逆约束)
- 质量极高 (DALL-E, Stable Diffusion, Midjourney 都用它!)
- 理论优雅 (与 score-based modeling 的深层联系)

---

## 概念 6：散度家族 — KL, JS, Wasserstein (🔑🔑)

### 为什么需要不同散度?

训练生成模型 = 让 P_model 逼近 P_data。用什么度量"逼近"?

```
┌──────────────────────────────────────────────────────────────┐
│                                                               │
│  KL(P || Q) = Σ P(x) log(P(x)/Q(x))                          │
│    - 非对称: KL(P||Q) ≠ KL(Q||P)                             │
│    - Mode-seeking: Q 集中在 P 的一个 mode 上 (VI 用它)        │
│    - 问题: Q(x)→0 且 P(x)>0 → KL→∞ (惩罚"没覆盖")           │
│                                                               │
│  JS(P || Q) = ½KL(P||M) + ½KL(Q||M), M=(P+Q)/2              │
│    - 对称! JS(P||Q) = JS(Q||P)                               │
│    - 有界: JS ∈ [0, log 2]                                   │
│    - Vanilla GAN 用 JS (通过 optimal D)                       │
│    - 问题: P和Q不重叠 → JS=log2常数 → 梯度消失!              │
│                                                               │
│  Wasserstein-1 (Earth Mover's Distance):                      │
│    W(P, Q) = inf_{γ} E_{(x,y)~γ}[||x - y||]                  │
│    - 对称, 满足三角不等式                                     │
│    - 即使P和Q不重叠, W仍给出有意义的距离!                     │
│    - WGAN: min_G max_{||f||_L≤1} E_P[f] - E_Q[f]            │
│    - 提供平滑的梯度 → 训练更稳定                              │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Vanilla GAN 的 JS 为什么容易梯度消失?

```
在高维空间中, P_data 和 P_G 的支撑集几乎必定不重叠:
  → JS(P_data || P_G) ≈ log 2 (常数!)
  → D 可以完美区分 (D→1 on real, D→0 on fake)
  → G 的梯度 ∇_G log(1-D(G(z))) → 0
  → 训练停滞!
```

### WGAN 的关键改进

```
WGAN 把 Discriminator 换成 Critic:
  - Critic f_w(x) 输出实数 (不是概率), 必须 1-Lipschitz
  - W-loss = E_{real}[f_w] - E_{fake}[f_w]
  - Critic 最大化 W-loss, Generator 最小化 -E_{fake}[f_w]
  
Lipschitz 约束的实现:
  - WGAN (原始): weight clipping [-c, c] → 简单但粗暴
  - WGAN-GP: gradient penalty → 更好 (Gulrajani et al. 2017)
```

---

## 概念 7：GAN 家族的演变 (🔑)

### Vanilla GAN → WGAN → Progressive GAN → BigGAN

```
┌──────────────────────────────────────────────────────────────┐
│                                                               │
│  Vanilla GAN (Goodfellow 2014):                               │
│    全连接 G 和 D, JS散度, 极不稳定, 只生成28x28              │
│                                                               │
│  DCGAN (Radford 2015):                                        │
│    卷积 G 和 D, BatchNorm, 训练稳定很多                      │
│    首次生成"看起来像真的"64x64图像                            │
│                                                               │
│  WGAN / WGAN-GP (Arjovsky 2017):                              │
│    Wasserstein距离替代JS → 梯度永不消失                      │
│    Gradient Penalty替代weight clipping → 训练极稳定          │
│    损失曲线与生成质量相关 → 可做early stopping!              │
│                                                               │
│  Progressive GAN (Karras 2018):                               │
│    渐进式训练: 4x4 → 8x8 → 16x16 → ... → 1024x1024          │
│    先学粗结构, 再逐步加细 → 高质量高分辨率                   │
│    关键技巧: 平滑的fade-in, minibatch stddev                 │
│                                                               │
│  StyleGAN / StyleGAN2 (Karras 2019-2020):                     │
│    风格调制: 分离"粗/中/细"层次的控制                       │
│    Mapping network + AdaIN → 可控生成                        │
│     Wasserstein + Gradient Penalty + R1 regularization        │
│                                                               │
│  BigGAN (Brock 2019):                                         │
│    大规模: batch=2048, 参数翻倍, TPU训练                     │
│    "截断trick": 采样时截断z的尾部 → 质量↑多样性↓            │
│    ImageNet 128x128: IS=166, FID=7.4 (当时SOTA)             │
│    证明: GAN + 足够大 = 惊人质量                              │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### 关键对比表

| 方法 | 年份 | 核心散度 | 训练稳定性 | 分辨率 | 关键创新 |
|------|------|---------|-----------|--------|---------|
| Vanilla GAN | 2014 | JS | 差 | 28x28 | 对抗训练框架 |
| DCGAN | 2015 | JS | 一般 | 64x64 | 卷积架构 |
| WGAN-GP | 2017 | Wasserstein | **好** | 64x64 | Gradient Penalty |
| Progressive GAN | 2018 | Wasserstein | 好 | 1024x1024 | 渐进式训练 |
| BigGAN | 2019 | Hinge | 一般(需调) | 512x512 | 大规模+截断 |
| StyleGAN2 | 2020 | W + R1 | 好 | 1024x1024 | 风格调制 |

### 为什么 Wasserstein > JS?

```
两个 1D 例子:
  P_real = δ(0)  (所有数据在 0)
  P_fake = δ(θ)  (生成器输出 θ)

JS(P_real, P_fake):
  θ=0:   JS = 0
  θ=0.1: JS ≈ log 2  (跳跃! 不连续!)
  θ≠0:   JS = log 2  (常数, 梯度=0)
  → G 收不到任何信号, 无法学习!

W(P_real, P_fake):
  θ=0:   W = 0
  θ=0.1: W = 0.1
  θ=1:   W = 1
  → W = |θ|, 梯度=sign(θ) → G 始终知道该往哪走!
```

---

## 概念 8：VAE vs GAN — 生成质量与训练稳定性的博弈

### 深度对比

```
┌──────────────────────────────────────────────────────────────┐
│              VAE                          GAN                │
├──────────────────────────────────────────────────────────────┤
│ 原理    Encoder Q(z|x)                Generator G(z)→x       │
│         Decoder P(x|z)                Discriminator D(x)     │
│         优化 ELBO                      优化 minimax           │
├──────────────────────────────────────────────────────────────┤
│ 输出    有隐空间z → 可插值            隐空间z → 可插值       │
│         有likelihood下界               无likelihood           │
├──────────────────────────────────────────────────────────────┤
│ 训练    稳定 (ELBO单调)                不稳定 (博弈震荡)      │
│         Reparam trick                  Mode Collapse         │
├──────────────────────────────────────────────────────────────┤
│ 生成    模糊 (pixel-wise loss)         锐利 (Discriminator   │
│         "平均化"所有可能               判定"真假"更贴近感知) │
├──────────────────────────────────────────────────────────────┤
│ 应用    表示学习, 异常检测              图像生成, style       │
│         压缩, 插值                     transfer, 超分辨率    │
└──────────────────────────────────────────────────────────────┘
```

### VAE-GAN hybrid (结合两者优势)

VAE的Decoder = GAN的Generator → Encoder提供隐空间结构, GAN的D提供高质量重建:

```
x → Encoder → z → Decoder/G → x̂
                                ↓
                         Discriminator → real/fake

Loss = VAE的ELBO + GAN的adversarial loss
```

这样既有VAE的稳定训练和隐空间, 又有GAN的锐利生成!

---

## 概念 9：将领域知识融入深度学习 (🔑🔑)

### PGM → DL 的核心启示

PGM 的最大优势：**图结构 = 领域知识的形式化编码**。深度学习如何吸取这个优势？

```
PGM 范式:                           DL 范式:
  专家设计图结构 (因果/依赖)          自动学习特征
  + 数据估计参数                     + 海量数据
  = 可解释, 数据效率高                = 黑盒, 需要大数据

最佳实践: 用领域知识设计架构, 用数据学习参数!
```

### 六大融合策略

```
┌──────────────────────────────────────────────────────────────┐
│                                                               │
│  1. Architecture as Prior (架构即先验)                        │
│     CNN:  平移不变性 (translation equivariance)               │
│     RNN:  时序依赖 (temporal structure)                       │
│     GNN:  图结构 (relational structure)                       │
│     Transformer: 全对全注意 (no built-in structure)           │
│                                                               │
│  2. Physics-Informed Neural Networks (PINNs)                  │
│     损失 = 数据拟合 + PDE残差                                 │
│     Loss = MSE_data + λ·||PDE_residual||²                   │
│     → 网络学会服从物理定律                                    │
│                                                               │
│  3. Structured Latent Space (结构化隐空间)                    │
│     VAE with graph-structured prior                           │
│     Tree-structured VAE for syntax trees                      │
│     Disentangled VAE (β-VAE): 各维度对应独立因子             │
│                                                               │
│  4. Knowledge Distillation & Rule Regularization              │
│     大模型 → 小模型 (知识蒸馏)                                │
│     Logical rules → soft constraints in loss                  │
│     → 把符号知识"编译"进神经网络                              │
│                                                               │
│  5. Equivariance & Invariance (等变与不变)                    │
│     Group CNN: 旋转/反射等变 → 数据增强的"硬编码"版          │
│     SE(3)-Transformer: 3D分子/蛋白质的物理对称性              │
│     → 不是为了"更好", 是为了"必须对"                         │
│                                                               │
│  6. Neuro-Symbolic (神经符号)                                 │
│     Neural perception + Symbolic reasoning                    │
│     如: CNN提取物体 → 逻辑规则推理关系 → 最终决策             │
│     → 结合DL的感知能力和符号AI的推理能力                      │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### 具体例子

**例1: CNN的平移等变性 = 编码了"空间平移"这个领域知识**

```
标准全连接: 输入pixel(0,0)和pixel(0,1)的权重完全独立
           → 需要分别学习"猫在左边"和"猫在右边"
           → 数据效率极低!

CNN: 同一filter扫过所有位置 → 自动保证平移等变性
     → 看到一个位置的猫 = 学会所有位置的猫
     → 数据效率提升了 H×W 倍!
```

**例2: 图神经网络 (GNN) = 编码了"关系结构"**

```
传统DL: 输入 = 固定大小的向量
GNN:    输入 = (节点特征, 邻接矩阵)
        → 天然处理分子结构、社交网络、知识图谱
        → Message Passing = 图上做BP!
        h_v^(l+1) = UPDATE(h_v^(l), AGGREGATE({h_u^(l): u∈N(v)}))
        
这恰好是 PGM 的消息传递 + 深度学习的参数化!
```

**例3: PINN — 让神经网络"懂物理"**

```
任务: 预测热传导的温度场 u(x,t)

普通NN Loss: MSE(u_pred, u_true) → 需要大量标注数据

PINN Loss: MSE_data + λ·MSE(∂u/∂t - α·∂²u/∂x²)
           └─数据项─┘   └─────── 物理约束 (热方程) ───────┘

→ 即使数据极少, 物理约束确保预测物理合理
→ 自动微分计算 ∂u/∂t 和 ∂²u/∂x² → 无需标注导数
```

### 与 PGM 的对应关系

| PGM 概念 | DL 对应 | 例子 |
|---------|--------|------|
| 图结构 (DAG/UG) | 网络架构 | CNN grid, GNN graph, Transformer dense |
| 条件独立性 | 参数共享 / Mask | CNN的局部连接, Attention mask |
| 因子分解 P(X)=∏ψ_C | 模块化设计 | Encoder-Decoder, multi-head attention |
| 领域知识 (专家设计) | 架构先验 + 损失约束 | PINN, equivariant networks |
| 消息传递 (BP) | 层间信息流 | GNN message passing, Transformer attention |

### 核心权衡

```
纯数据驱动 (DL extreme):           纯知识驱动 (PGM extreme):
  需要海量数据                        需要精细建模
  自动发现模式                        可解释, 可信
  可能学到虚假相关                    建模偏差

最佳实践 = DL架构 + PGM先验:
  → 用领域知识约束模型空间 (减少方差)
  → 用数据学习剩余参数 (减少偏差)
  → "让模型知道它不应该学到什么"
```

---

## 📋 全部概念一张表

| 概念 | 一句话 | 优势 | 劣势 |
| **VAE** | Encoder+Decoder, ELBO训练 | 有隐空间 | 生成模糊 |
| **GAN** | G vs D 对抗博弈 | 生成质量最高(曾) | 训练不稳, mode collapse |
| **Flow** | 可逆变换, exact likelihood | 精确似然 | 计算量大, 维度限制 |
| **AR** | 逐维生成 P(x_d\|x_{<d}) | 似然好, 稳定 | 生成慢 O(D) |
| **Diffusion** | 逐步去噪 | 质量SOTA, 训练稳 | 生成慢 (多步) |
| **Domain Knowledge** | 架构=先验, 物理约束, GNN=BP | 数据效率高, 可解释 | 需要专家知识 |

---

## 🔗 概念关系图

```
             PGM 生成模型 (L12: RBM)
                    │
                    ▼
    ┌───────────────┼───────────────┬──────────────┬──────────────┐
    ▼               ▼               ▼              ▼              ▼
  VAE             GAN            Flow          Diffusion    Domain Knowledge
 ELBO优化      对抗博弈       可逆变换        去噪过程     架构=先验
 (L7-L8 VI)   (minimax)   (exact LL)    (score matching)  PINN/GNN
    │               │               │              │              │
    ▼               ▼               ▼              ▼              ▼
 隐空间连续     生成高质      似然可算       质量SOTA      数据效率高
 生成模糊      训练不稳      维度受限       生成慢        可解释性强
```

---

## 🎯 核心洞见一句话

| # | 洞见 |
|---|------|
| 1 | **DGM II = 不需要能量函数** — 用网络直接参数化生成过程 |
| 2 | **GAN = 博弈论** — G 和 D 的 minimax 让你不用算 likelihood |
| 3 | **Flow = 代数** — 可逆变换 + Jacobian → exact LL |
| 4 | **Diffusion = 去噪** — 学习逆向去噪过程, 训练稳, 质量SOTA |
| 5 | **没有免费的午餐** — 每种方法都有 trade-off (质量 vs 速度 vs 稳定性) |
| 6 | **领域知识 = 数据效率** — CNN的平移等变, PINN的物理约束, GNN的图结构 → 用知识换数据 |

---

## 🧪 自测清单

- [ ] GAN 的 minimax objective 是什么? G 和 D 分别优化什么?
- [ ] Mode Collapse 是什么现象? 为什么 GAN 容易出现?
- [ ] Normalizing Flow 的 log-likelihood 公式中 Jacobian 的作用是什么?
- [ ] 自回归模型的生成为什么是 O(D) 步? 有什么加速方法?
- [ ] Diffusion 的 forward process 和 reverse process 分别做什么?
- [ ] 为什么 CNN 的平移等变性可以理解为"领域知识的编码"?
- [ ] PINN 的 PDE 残差项从 MAP 角度等价于什么先验?
- [ ] GNN 的消息传递和 BP 有什么结构对应? 为什么这很重要?

---

> L13 是深度生成模型的"第二代"。从 VAE 到 GAN 到 Diffusion, 生成模型在过去十年经历了巨大的革新 — 而这一切都根植于 L1-L10 的 PGM 统计基础。
