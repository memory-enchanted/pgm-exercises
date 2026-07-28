# CMU 10-708 Lecture 13 课后练习 & 答案

> 配套教材: Goodfellow et al. (2016) Ch.20, Kingma & Welling (2014, 2019), Goodfellow et al. (2014), Ho et al. (2020)
>
> 题目分为三级: 🟢 基础 | 🟡 进阶 | 🔴 挑战

---

## 🟢 基础题 (必须掌握)

### Q1. GAN 的 Minimax Objective

**(A)** 写出 GAN 的 minimax 目标函数: min_G max_D V(D, G)。

**(B)** 证明: 对固定的 G, 最优 D*(x) = P_data(x) / (P_data(x) + P_G(x))。

**(C)** 将 D* 代入 V, 证明此时 V(G, D*) = -log 4 + 2·JSD(P_data || P_G)。

<details>
<summary>点击查看答案</summary>

**(A) GAN objective:**

```
V(D, G) = E_{x~P_data}[log D(x)] + E_{z~P_z}[log(1 - D(G(z)))]

min_G max_D V(D, G)
```

**(B) 最优 D:**

```
V = ∫ P_data(x) log D(x) + P_G(x) log(1 - D(x)) dx

对每个 x, 对 D 求导:
∂/∂D [P_data log D + P_G log(1-D)] = P_data/D - P_G/(1-D) = 0
→ D* = P_data / (P_data + P_G)
```

**(C) 代入:**

```
V(D*, G) = E_{P_data}[log(P_data/(P_data+P_G))]
          + E_{P_G}[log(P_G/(P_data+P_G))]
        = KL(P_data || M) + KL(P_G || M)  - 2·log 2
          (where M = (P_data + P_G)/2)
        = 2·JSD(P_data || P_G) - log 4
```

其中 JSD 是 Jensen-Shannon Divergence。当 P_G = P_data 时, JSD=0, V = -log 4。

</details>

---

### Q2. Mode Collapse

**(A)** 描述 GAN 中的 Mode Collapse 现象。

**(B)** 为什么 GAN 容易出现 Mode Collapse? 从优化角度解释。

<details>
<summary>点击查看答案</summary>

**(A) Mode Collapse:**

Generator 只生成数据分布中少数几个(甚至一个) mode, 而忽略了其他 mode。

例如: 数据是 8 个高斯簇, GAN 的 Generator 只学会了生成其中 1-2 个簇的样本 → 生成的"多样性"极差。

**(B) 原因:**

从 G 的视角: G_loss = -E[log D(G(z))]

如果 G 发现某个特定的输出 x* 能很好地骗过 D (D(x*) ≈ 1), G 的最优策略是: **对所有 z 都输出 x*** — 因为这样所有假样本都被判为"真"。

G 没有动力去覆盖多个 mode — 精确复制一个 mode 就能拿高分!

这是 minimax formulation 的内在缺陷: G 的损失不惩罚"缺乏多样性"。

**缓解方法**: 
- Unrolled GAN: 用 D 的未来几步更新来算 G 的梯度
- Minibatch Discrimination: 让 D 能看到一个 batch 内的样本差异
- WGAN: 用 Wasserstein 距离替代 JS 散度

</details>

---

### Q3. Normalizing Flow 的 Change-of-Variables

**(A)** 写出 change-of-variables 公式: 若 z ~ p_z, x = f(z), f 可逆, 则 p_x(x) = ?

**(B)** 对 K 步级联 Flow x = f_K ∘ ... ∘ f_1(z₀), 写出 log p(x) 的完整公式。

<details>
<summary>点击查看答案</summary>

**(A) Change-of-Variables:**

```
p_x(x) = p_z(f^{-1}(x)) · |det ∂f^{-1}/∂x|
       = p_z(z) · |det ∂f/∂z|^{-1}
```

其中 Jacobian 行列式保证概率守恒: ∫ p_z(z) dz = ∫ p_x(x) dx = 1。

**(B) K 步级联:**

令 z_k = f_k(z_{k-1}), z_0 ~ p_0, x = z_K:

```
log p(x) = log p_0(z_0) - Σ_{k=1}^K log |det ∂f_k/∂z_{k-1}|
```

每步的 Jacobian 行列式累加 → 可以算 exact log-likelihood。

这就是 Flow 的核心优势 — 不需要近似, 不需要采样, 直接算 LL。

</details>

---

### Q4. 自回归模型的训练 vs 生成

**(A)** 自回归模型 P(x) = Π_d P(x_d | x_{<d}) 在训练时为什么是 O(D)?

**(B)** 在生成时为什么也是 O(D)? 有什么加速思路?

<details>
<summary>点击查看答案</summary>

**(A) 训练: O(D)**

训练时使用 teacher forcing: 所有 x_{<d} 从真实数据给出 → 每个条件 P(x_d|x_{<d}) 可以**并行**计算 → O(1) (用 masked convolutions/attention)。

但实践中, 一个 forward pass 计算所有 D 个条件分布 = O(D) 参数计算量。

**(B) 生成: O(D)**

生成时必须**逐维采样**: x_1 → x_2 → x_3 → ... → x_D — 因为 x_d 依赖于已生成的 x_{<d} → 无法并行 → O(D) sequential steps。

对于 D=784 (MNIST): 还行 (~1 秒)。对于 D=65536 (256×256 图像): 极慢!

**加速方法**:
- WaveNet: 用 dilated convolutions 缓存中间状态
- Parallel Wavenet: 用 Flow 蒸馏 AR 模型 → 生成变快
- PixelCNN++: 共享中间表示
- Transformer AR: KV-cache 加速自回归生成

</details>

---

## 🟡 进阶题

### Q5. VAE vs GAN 的生成质量

**(A)** VAE 生成的图像通常比 GAN 模糊。从优化目标和模型假设两个角度解释。

**(B)** 有什么方法可以提升 VAE 的生成质量?

<details>
<summary>点击查看答案</summary>

**(A) VAE 模糊的原因:**

**优化目标**: VAE 用 ELBO = E[log P(x|z)] - KL(Q||P)。重建项通常是 pixel-wise MSE 或 BCE → 倾向于生成"平均"图像 → 模糊。

**模型假设**: VAE 假设 P(x|z) 是 Gaussian (或 Bernoulli) → 每个像素独立给定 z → 忽略了像素之间的相关性 → 丧失高频细节。

而 GAN 的 Discriminator 可以直接判断图像是否"真实"(sharp, realistic) → 不需要显式建模像素依赖 → G 被强迫生成锐利图像。

**(B) 改进 VAE:**

- **更好的 decoder**: PixelCNN decoder (VLAE), 自回归 decoder → 像素之间不再条件独立
- **更好的 loss**: 加上 perceptual loss, adversarial loss (VAE-GAN hybrid)
- **更强的 prior**: VampPrior, hierarchical prior
- **更大的模型**: VQ-VAE (离散隐变量 + autoregressive prior)
- **Normalizing Flow prior**: 用 Flow 替代 Gaussian prior

</details>

---

### Q6. Diffusion Model 的 Forward Process

**(A)** DDPM 的 forward process: x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε, ε~N(0,I)。证明: 当 T 足够大时, x_T ~ N(0, I)。

**(B)** 为什么 Diffusion 的训练和生成是两套"时间方向"? 这个设计有什么优势?

<details>
<summary>点击查看答案</summary>

**(A) 证明:**

x_t | x_0 ~ N(√ᾱ_t x_0, (1-ᾱ_t) I)

当 T 大且 noise schedule 合理 (如 linear β_t):
α_t = 1 - β_t, ᾱ_T = Π α_t → 0 (当 T → ∞)

所以 x_T | x_0 ~ N(0, I) (与 x_0 无关)。

对任何 x_0, 经过足够多步加噪后, 分布趋近标准高斯 → forward process "抹掉了"所有数据结构 → reverse process 从纯噪声出发就有意义。

**(B) 两套时间方向:**

Forward (加噪): x_0 → x_1 → ... → x_T — 固定的, 不需要学习, 就是加噪声。
Reverse (去噪): x_T → x_{T-1} → ... → x_0 — 需要学习, 用网络预测噪声。

**优势**:
- Forward 是闭式解 (不需要模拟 T 步, 直接 jump 到 x_t)
- 训练目标只是简单的 MSE (预测噪声)
- 不需要对抗训练 (vs GAN)
- 不需要可逆约束 (vs Flow)
- 生成过程虽慢但稳定, 质量极高

</details>

---

## 🔴 挑战题

### Q7. WGAN — 用 Wasserstein 距离替代 JS

**(A)** GAN 的 JS 散度在两个分布"不重叠"时会怎样? 这为什么导致梯度消失?

**(B)** WGAN 用 Earth Mover's Distance (Wasserstein-1): W(P_r, P_g) = sup_{||f||_L≤1} E_{P_r}[f] - E_{P_g}[f]。解释: 为什么 W-distance 即使在分布不重叠时仍能提供有意义的梯度?

<details>
<summary>点击查看答案</summary>

**(A) JS 散度的问题:**

当 P_r 和 P_g 的"支撑集"不重叠时 (在低维流形上几乎总是):
- JS(P_r || P_g) = log 2 (常数!)
- ∇_θ JS = 0 → Generator 的梯度消失 → 训练停止

在图像生成中, P_r 和 P_g 通常是高维空间中的低维流形 → 几乎必然不重叠 → JS 梯度为 0。

**(B) W-distance 的优势:**

Wasserstein 距离衡量将一个分布"搬运"到另一个的最小代价。

即使两个分布的支撑集不重叠:
- W(P_r, P_g) 仍能反映它们之间的"距离"
- 例如: P_r = δ(0), P_g = δ(θ): JS = log 2 (常数), W = |θ| (随 θ 变化!)
- ∇_θ W = sign(θ) → 持续提供非零梯度 → G 总能朝正确方向更新

WGAN 实现: 用 weight clipping 或 gradient penalty 强制 Discriminator (Critic) 满足 1-Lipschitz 条件。

</details>

---

### Q8. Score-based Generative Modeling

**(A)** Score function s(x) = ∇_x log P(x)。为什么从 score 可以生成样本?

**(B)** Langevin Dynamics: x_{t+1} = x_t + (ε/2) ∇_x log P(x_t) + √ε z_t, z_t~N(0,I)。解释它和 Diffusion 的关系。

<details>
<summary>点击查看答案</summary>

**(A) 从 score 生成:**

Score s(x) 指向对数概率密度增长最快的方向 → 沿着 s(x) 走可以"爬"到 P 的高概率区域。

如果在不同噪声水平上都有 score estimate → 可以:
1. 从纯噪声开始 (高噪声水平的 score 引导到大致的数据区域)
2. 逐步降低噪声水平 → 精细调整 → 最终到达数据分布的高概率区域

**(B) Langevin 与 Diffusion:**

Langevin Dynamics = 用 score 指导 MCMC:
```
x_{t+1} = x_t + η·s(x_t) + √(2η)·z_t
         └── 梯度方向 ──┘ └── 噪声探索 ──┘
```

Diffusion Model 的 reverse process:
```
x_{t-1} = (1/√α_t)·(x_t - (1-α_t)/√(1-ᾱ_t)·ε_θ(x_t, t)) + σ_t·z
```

可以重写为 score-based 形式:
```
x_{t-1} = x_t + (1-α_t)/√(1-ᾱ_t)·s_θ(x_t, t) + σ_t·z
```

**本质相同!** Diffusion = 用不同噪声水平下的 score, 做 annealed Langevin dynamics!

这就是 Song & Ermon (2019) 和 Ho et al. (2020) 的深刻联系 — Diffusion 和 Score-based 模型本质上是同一类方法。

</details>

---

### Q9. 评估生成模型

**(A)** 为什么评估生成模型比评估分类器难?

**(B)** 列出 3 种生成模型的评估指标并解释各自衡量什么。

<details>
<summary>点击查看答案</summary>

**(A) 为什么难:**

分类器: 准确率、F1 — 直接, 客观。

生成模型: 没有"正确答案"! 生成的每张图都是新的。
- P(x) 通常不可直接算 (除了 Flow 和 AR)
- 需要评估两个维度: **质量**(单张好吗?) 和 **多样性**(覆盖所有 mode 了吗?)
- Mode collapse 无法从单张样本看出

**(B) 三种指标:**

| 指标 | 全称 | 衡量 |
|------|------|------|
| **IS** | Inception Score | 用预训练分类器看生成图像的类别分布: 一张图类别明确 (高 confidence) + 多张图类别多样 → IS 高 |
| **FID** | Frechet Inception Distance | 比较 real 和 fake 在 Inception 特征空间中的分布差异 (用 Gaussian 假设) → FID 低 = 分布接近 |
| **NLL** | Negative Log-Likelihood | 在测试集上的平均 -log P(x) → 直接衡量密度估计质量 (仅 Flow/AR 可用) |

IS 的问题: 不检测 mode collapse (只看边缘多样性)。
FID 更好: 捕捉了真实和生成分布的"距离"。
NLL 最好但受限: 只对 tractable density 模型可用。

</details>

---

### Q10. DGM 的发展时间线

按时间排序以下里程碑: GAN, VAE, Diffusion (DDPM), RBM/DBN, Normalizing Flow, StyleGAN。

**(A)** 给出正确的年份顺序。

**(B)** 简要说明每一代方法的"核心创新"和"主要局限"。

<details>
<summary>点击查看答案</summary>

**(A) 时间线:**

```
2006: RBM / DBN (Hinton)      ← 第一代: 能量模型+逐层预训练
2013: VAE (Kingma & Welling)  ← 第二代: 变分推断+神经网络
2014: GAN (Goodfellow et al.) ← 第二代: 对抗训练
2015: Normalizing Flow        ← 第二代: 可逆变换
2018: StyleGAN (Karras et al.)← GAN的顶峰: 高质量可控生成
2020: DDPM (Ho et al.)        ← 第三代: 去噪扩散, SOTA质量
```

**(B) 创新与局限:**

| 年份 | 方法 | 核心创新 | 主要局限 |
|------|------|---------|---------|
| 2006 | RBM/DBN | 能量模型+CD训练, 首次成功训练深度生成模型 | 训练慢, 质量被后来者超越 |
| 2013 | VAE | Encoder+Decoder, ELBO优化, Amortized VI | 生成模糊, 质量不如GAN |
| 2014 | GAN | 对抗博弈, 生成质量极高(当时) | 训练不稳, mode collapse |
| 2015 | Flow | 可逆变换, exact likelihood | 维度限制, 计算量大 |
| 2018 | StyleGAN | 风格调制, 高质量可控 | 仍是GAN系列, 训练复杂 |
| 2020 | DDPM | 去噪Diffusion, 训练稳, 质量SOTA | 生成慢(多步) |

**历史轨迹**: 从 struggle to train → 终于能训练 → 追求质量 → 追求可控 → 质量+稳定兼得。

</details>

---

### Q11. Wake-Sleep / VAE / GAN 三者综合对比

**(A)** 从优化目标、是否需要推断隐变量、训练稳定性三个维度对比 Wake-Sleep, VAE, GAN。

**(B)** 解释：为什么说 VAE 是 Wake-Sleep 的"修正版"？解决了什么关键问题？

**(C)** 为什么 GAN 完全不需要推断 z？这是优势还是劣势？

<details>
<summary>点击查看答案</summary>

**(A) 三维对比:**

| | Wake-Sleep | VAE | GAN |
|---|-----------|-----|-----|
| **优化目标** | Wake: log P(X\|Z); Sleep: log Q(Z\|X) — 不一致! | 统一的 ELBO = E[logP] - KL(Q\|\|P) | Minimax: min_G max_D V(D,G) |
| **推断隐变量** | 需要! Q(Z\|X) 是识别网络 | 需要! Encoder Q(Z\|X) | **不需要!** G直接从z生成 |
| **训练稳定性** | 差 (目标分裂, 不收敛) | **好** (ELBO单调) | 差 (博弈震荡, mode collapse) |

**(B) VAE = Wake-Sleep的修正版:**

Wake-Sleep的致命缺陷: Wake和Sleep优化**两个不同的目标** → 不保证收敛。

VAE的修正:
- ELBO = E_Q[log P(X\|Z)] - KL(Q\|\|P(Z)) — **一个目标函数!**
- Encoder和Decoder共享同一个ELBO → 提升Decoder的同时自动提升Encoder
- Reparameterization trick → 梯度可以通过采样z反向传播
- 理论保证: 每一步ELBO不降 → 收敛到局部最优

**(C) GAN不需要推断z:**

GAN中z ~ p(z) (如N(0,I))是随机噪声, 不是"后验推断"的结果。

**优势**:
- 不需要Encoder → 模型更简单 → 参数量少一半
- 不需要推理隐变量 → 对于"生成"这个任务, 推断不是必需的
- G专注于生成, D专注于鉴别 → 分工明确

**劣势**:
- 无法做推断 (没有 Q(z\|x)) → 不能做 representation learning, anomaly detection
- 无法在隐空间插值 (除非用 BiGAN/ALI 等方法加Encoder)
- 缺少一个"理解数据"的视角

**互补性**: VAE好于理解, GAN好于生成。VAE-GAN hybrid (如VAE/GAN)结合两者优势。

</details>

---

### Q12. 领域知识融入深度学习 — 架构即先验

**(A)** 解释为什么 CNN 的卷积结构可以被理解为一种"领域知识"的编码。这种编码带来了什么统计收益（用偏差-方差语言）？

**(B)** 物理信息神经网络 (PINN) 在标准 MSE loss 上额外加 PDE 残差项。从 PGM 的 MAP 估计角度解释这等价于什么先验。

**(C)** 图神经网络 (GNN) 的消息传递公式和 L5 的 Belief Propagation 有什么结构上的相似性？这种相似性说明了什么？

<details>
<summary>点击查看答案</summary>

**(A) CNN = 编码了"平移等变性"这个领域知识:**

**编码方式**: 
- 同一个 filter 在所有位置共享 → 自动保证 f(T(x)) = T(f(x)) (平移等变)
- 局部连接 → 只让邻近像素交互

**统计收益 (偏差-方差)**:
- **降低方差**: 参数共享 → 有效参数量从 H×W×C 降到 K²×C → 大幅降低方差
- **降低偏差**: 如果任务确实需要平移等变性, CNN 的 inductive bias 正好匹配 → 偏差也低
- 对比全连接: 需要海量数据来"发现"平移不变性 → 数据不够时方差爆炸

**本质**: CNN 的架构 = "视觉世界是平移不变的"这个领域知识的硬编码。没有这个先验, 深度学习在图像上根本不可行。

**(B) PINN 的 PDE 残差 = Gaussian 先验:**

```
标准 NN Loss: MSE_data = (1/N) Σ (u_pred_i - u_true_i)²
              等价于 MLE with Gaussian noise

PINN Loss: MSE_data + λ·MSE_pde
          = -log P(data|u) - log P(u)  ← MAP!

其中 P(u) ∝ exp(-λ·(∂u/∂t - α∂²u/∂x²)²)
→ "偏离热方程的解"被惩罚
→ 等价于: u 的先验是"越满足 PDE, 概率越高"
```

**(C) GNN vs BP:**

GNN Message Passing:
```
h_v^(l+1) = UPDATE(h_v^(l), Σ_{u∈N(v)} MESSAGE(h_u^(l), e_{uv}))
```

BP Message:
```
m_{i→j}(X_j) = Σ_{X_i} ψ_{ij}(X_i,X_j) · Π_{k∈N(i)\j} m_{k→i}(X_i)
BEL(X_i) ∝ Π_{k∈N(i)} m_{k→i}(X_i)
```

**结构相似性**:
- GNN 的 AGGREGATE(Σ MESSAGE) ↔ BP 的消息乘积 (Π m_{k→i})
- GNN 的 UPDATE ↔ BP 的信念计算 (BEL)
- 两者都沿图结构传播信息, 每层/每轮整合越来越远的邻居

**这个相似性揭示了**:
- BP 可以看作"可微的消息传递" — GNN 正是用神经网络参数化了这个消息传递过程!
- GNN = BP + 深度学习的参数化和端到端训练
- 这也是为什么 PGM 和 GNN 之间有深层联系 — 它们是同一种图算法的不同实现

</details>

---

### Q13. 如何选择"知识融入"的粒度?

在面对一个新问题时, 你有以下选择:
(a) 纯黑盒 DL (如 Transformer)
(b) 架构先验 (如 CNN for 图像, GNN for 图)
(c) 强物理约束 (如 PINN)
(d) 纯 PGM (如手建 Bayesian Network)

**(A)** 给出每种方法的数据需求量排序 (少→多)。

**(B)** 给出每种方法的"模型偏差"排序 (高→低)。

**(C)** 在什么场景下你会从 (a) 转向 (d)? 什么场景下从 (d) 转向 (a)?

<details>
<summary>点击查看答案</summary>

**(A) 数据需求量 (少→多):**

```
(d) 纯 PGM < (c) PINN < (b) 架构先验 < (a) 黑盒 DL
```

- PGM: 结构由专家给定 → 只需估计参数 → 几十到几百样本可能就够
- PINN: 物理方程提供强约束 → 少量数据 + 物理规律 → 几百到几千样本
- 架构先验: 提供归纳偏置但不如物理方程精确 → 几千到几万样本
- 黑盒 DL: 一切从数据学 → 需要海量数据 (百万+)

**(B) 模型偏差 (高→低):**

```
(d) 纯 PGM > (c) PINN > (b) 架构先验 > (a) 黑盒 DL
```

偏差高 = 模型空间小 (strong assumptions) → 数据需求少但可能建模不足。
偏差低 = 模型空间大 (weak assumptions) → 数据需求多但可能更灵活。

**(C) 什么时候转向:**

**数据少 + 领域知识丰富 → 偏向 (d) PGM 或 (c) PINN**:
- 医疗诊断: 医生有因果知识, 数据稀缺, 可解释性必须 → PGM
- 物理模拟: PDE 已知, 实验数据少 → PINN

**数据多 + 领域知识模糊 → 偏向 (a) 黑盒 DL**:
- 自然语言: 语法规则复杂且例外多 → Transformer
- 推荐系统: 用户行为难用简单规则描述 → DL

**最佳实践 = 混合**: 用 PGM 的图结构作为 DL 架构的"骨架", 用数据学习参数。
```

</details>

---

## 📊 综合自测评分

每题 10 分，共 130 分 (Q11-Q13 为附加综合题)。

| 得分 | 评价 |
|------|------|
| 120-130 | L13 完全掌握, 已理解四类 DGMs、散度家族、GAN演变、领域知识融合 |
| 90-119  | 主干扎实, 建议亲手跑一个 VAE/GAN/Diffusion |
| 70-89   | 概念清晰, 回去推一遍 GAN optimal D 和 Flow change-of-variables |
| < 70    | 先吃透 Q1-Q4, 确保理解 GAN/VAE/Flow/AR 的基本机制 |

---

> L13 完成了深度生成模型的"第二代"。从 RBM (L12) 到 GAN 到 Diffusion, 你已见证了生成模型十年的演变 — 从能量函数到对抗博弈, 从 VI 到 score matching。
