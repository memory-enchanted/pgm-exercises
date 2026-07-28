# CMU 10-708 Lecture 11 概念体系梳理 — 图模型与深度学习的桥梁

> 整理自 Eric Xing 教授 CMU 10-708 Lecture 11: Graphical Models & Deep Learning
>
> 核心教材: Goodfellow et al. (2016) Deep Learning, Murphy Ch.13, Ch.28, Bishop Ch.5

---

## 📐 全局定位：L11 的四层递进

```
L1-L10: 概率图模型 (PGM)               L11: PGM ↔ DL 桥梁
──────────────────────────            ─────────────────────────────
显式建模条件独立性                    ① DL组件概览: 理解神经网络的积木
精确/近似推断                         ② GM vs NN: 两种范式的深层对比
参数: 条件概率表/势函数                ③ 结合DL与GM: 取各自之长
可解释性强                            ④ 贝叶斯学习NN: 用概率视角看NN

L11 = 把图模型的概率思维带入深度学习，再从深度学习带回工具增强图模型。
```

**一句话概括 L11**: 深度学习用神经网络做函数近似，图模型用概率做结构化推理——L11 先讲 DL 组件，再对比两种范式，最后展示二者的融合：深度生成模型与贝叶斯神经网络。

---

# 第一部分：An Overview of DL Components

---

## 概念 1.1：神经网络的基本积木 (🔑)

### 从感知机到深度网络

```
单层 (感知机):  y = σ(Wx + b)
              └── 仿射变换 + 非线性激活

多层 (MLP):     h₁ = σ₁(W₁x + b₁)
                h₂ = σ₂(W₂h₁ + b₂)
                y  = σ₃(W₃h₂ + b₃)
              └── 层层堆叠 → 学习层次化表示 (hierarchical representations)
```

### 关键组件一览

| 组件 | 角色 | 例子 |
|------|------|------|
| **线性层** | 仿射变换: z = Wx + b | Linear / Dense / Conv |
| **激活函数** | 引入非线性 | ReLU, GELU, Sigmoid, Tanh |
| **损失函数** | 衡量预测与真实的差距 | MSE, CrossEntropy, NLL |
| **优化器** | 更新参数以最小化损失 | SGD, Adam, AdamW |
| **正则化** | 防止过拟合 | Dropout, BatchNorm, Weight Decay |
| **初始化** | 设定参数初值 | Xavier, Kaiming (He) |

---

## 概念 1.2：激活函数 — 为什么需要非线性 (🔑)

```
没有激活函数: 多层线性层 = 一个线性层
    f(x) = W₃(W₂(W₁x + b₁) + b₂) + b₃
         = (W₃W₂W₁)x + (W₃W₂b₁ + W₃b₂ + b₃)
         = W'x + b'    ← 仍然是线性的!

加上激活函数: 多层 → 万能近似器 (Universal Approximator)
    f(x) = σ₃(W₃ σ₂(W₂ σ₁(W₁x + b₁) + b₂) + b₃)
                        ← 真正的非线性层次表示!
```

### 常用激活函数对比

| 激活函数 | 公式 | 输出范围 | 特点 |
|---------|------|---------|------|
| **Sigmoid** | σ(x) = 1/(1+e⁻ˣ) | (0, 1) | 平滑, 饱和 → 梯度消失 |
| **Tanh** | tanh(x) | (-1, 1) | 零中心, 饱和 → 梯度消失 |
| **ReLU** | max(0, x) | [0, ∞) | 不饱和, 简单高效, 死神经元 |
| **Leaky ReLU** | max(αx, x) | (-∞, ∞) | 解决死神经元问题 |
| **GELU** | x·Φ(x) | (-∞, ∞) | Transformer 标配, 平滑版 ReLU |
| **Softmax** | eˣⁱ/Σeˣʲ | (0, 1) | 多分类输出层, 输出为概率分布 |

```
ReLU 为什么统治了 DL?
  - 梯度 = 1 (x>0) 或 0 (x<0) → 不饱和 → 缓解深层网络的梯度消失
  - 计算极简单: max(0,x) → 比 sigmoid/tanh 快很多
  - 天然产生稀疏激活 → 类似 L1 正则化的效果
```

---

## 概念 1.3：损失函数 — 训练信号的设计 (🔑)

### 从 MLE 视角理解损失函数

```
神经网络训练 = MLE (或 MAP) + 梯度下降

P(y|x, θ) = 分布(f_θ(x))
  → MLE: θ* = argmin_θ -Σ log P(y_i | x_i, θ)
           = argmin_θ Σ Loss(y_i, f_θ(x_i))

损失函数 = 负对数似然 (Negative Log-Likelihood)
```

### 常见损失与统计假设

| 损失函数 | 统计假设 | 使用场景 |
|---------|---------|---------|
| **MSE** | y\|x ~ N(f(x), σ²) | 回归 |
| **MAE (L1)** | y\|x ~ Laplace(f(x), b) | 回归 (对异常值鲁棒) |
| **Binary Cross-Entropy** | y\|x ~ Bernoulli(p) | 二分类 |
| **Categorical Cross-Entropy** | y\|x ~ Categorical(π) | 多分类 |
| **NLL (Negative Log-Likelihood)** | 任意分布族 | 通用 |
| **KL Divergence** | 两个分布的差异 | VAE, 知识蒸馏 |
| **Contrastive Loss** | 正负样本对的相似度 | 表示学习, SimCLR |

---

## 概念 1.4：优化器 — 如何高效找到最优参数 (🔑🔑)

### 梯度下降的统计解释

```
Full Batch GD:  g = ∇_θ (1/N) Σᵢ Loss(x_i, θ)
                → 精确梯度, 但每步要过全部数据 (昂贵)

SGD:           ĝ = ∇_θ (1/B) Σ_{i∈batch} Loss(x_i, θ)
                → 梯度的无偏但含噪估计
                → Var[ĝ] ∝ σ²/B  (batch越大, 噪声越小)
```

### 关键优化器演进

| 优化器 | 核心思想 | 更新规则 (简化) |
|--------|---------|----------------|
| **SGD** | 沿负梯度方向 | θ ← θ - η·g |
| **Momentum** | 累积历史方向 → 加速 | v ← βv + g; θ ← θ - η·v |
| **AdaGrad** | Per-dimension 自适应LR | s ← s + g²; θ ← θ - η·g/√(s+ε) |
| **RMSprop** | 指数移动平均的 AdaGrad | s ← β₂s + (1-β₂)g² |
| **Adam** | Momentum + RMSprop | m ← β₁m + (1-β₁)g; s ← β₂s + (1-β₂)g² |
| **AdamW** | Adam + 解耦 Weight Decay | 权重衰减与自适应LR分离 |

### Adam 的三个关键组件

```
1. 一阶矩 (动量):  m_t = β₁m_{t-1} + (1-β₁)g_t
   → 平滑梯度方向, 加速收敛, 逃离局部最优

2. 二阶矩 (自适应LR):  s_t = β₂s_{t-1} + (1-β₂)g_t²
   → 每个参数有独立的学习率, 高频参数LR小, 低频参数LR大

3. 偏差校正:  m̂ = m_t/(1-β₁ᵗ),  ŝ = s_t/(1-β₂ᵗ)
   → 修正初始化带来的零偏, 训练初期尤其重要

更新: θ_{t+1} = θ_t - η·m̂/(√ŝ + ε)
      └── m̂/√ŝ 是无量纲的 → 步长与参数尺度无关
```

---

## 概念 1.5：计算图与反向传播 (🔑🔑🔑)

### 计算图 = 深度学习执行的底层表示

```
前向传播: 沿计算图从左→右计算, 得到 Loss
反向传播: 沿计算图从右→左传递梯度 (Chain Rule)

     x → [W₁] → h₁ → [ReLU] → a₁ → [W₂] → h₂ → [Softmax] → L
     ←────────────────── ∂L/∂x ←─────────────────────── 1
```

### 链式法则 = 图上的消息传递

```
对于复合函数 L = f(g(h(x))):

∂L/∂x = f'(g)·g'(h)·h'(x)
        └── 每个节点的局部导数 (Jacobian) 乘以上游梯度 ──┘

类比 L5 的 BP (Belief Propagation):
  BP 消息:  收集邻居的"概率因子" → 乘积求和 → 发给下一个
  反向传播: 收集子节点的"梯度"　 → 链式法则 → 发给父节点

两者都是 "在图结构上传递信息" 的图算法!
```

---

## 概念 1.6：正则化 — 约束模型复杂度 (🔑)

### 从 PGM 的 Prior 到 DL 的技巧

| DL 正则化 | PGM 对应 | 机制 |
|-----------|---------|------|
| **L2 (Weight Decay)** | Gaussian Prior p(w) ∝ exp(-λ‖w‖²) | 约束权重不要太大 |
| **L1** | Laplace Prior p(w) ∝ exp(-λ\|w\|) | 诱导稀疏权重 |
| **Dropout** | 隐式 Ensemble / 贝叶斯模型平均 | 训练时随机丢弃神经元 |
| **Batch Normalization** | 控制层间分布漂移 | 稳定每层输入分布 |
| **Early Stopping** | 限制有效参数空间 | 在验证误差最小时停止 |
| **Data Augmentation** | 编码不变性先验 | 人工扩充数据 |
| **Label Smoothing** | Soft prior on labels | 防止过度自信的预测 |

---

## 概念 1.7：典型架构一览

```
Feedforward (MLP):      适合表格数据, 万能近似
      x → [Linear→ReLU]×N → Output

CNN:                    适合网格数据 (图像, 音频)
      x → [Conv→ReLU→Pool]×N → FC → Output

RNN / LSTM / GRU:       适合序列数据 (文本, 时间序列)
      x₁→h₁→x₂→h₂→...→h_T → Output

Transformer:            适合序列和集合 (NLP, CV, 多模态)
      x → [Self-Attention→FFN→LayerNorm]×N → Output

GNN:                    适合图数据 (分子, 社交网络)
      Graph → [MessagePassing→Update]×N → Output
```

---

# 第二部分：Similarities and Differences between GMs and NNs

---

## 概念 2.1：两种建模范式的根本对比 (🔑🔑)

```
图模型 (GM):                          神经网络 (NN):
─────────────────                    ─────────────────
显式建模 P(X) 或 P(Y|X)              隐式学习 f: X→Y
变量之间的关系 → 图结构               表示 → 层次化特征
参数: 条件概率表 / 势函数              参数: 权重矩阵 / 偏置
推断: VE, BP, MCMC                    推断: 前向传播 (一步)
学习: MLE, EM, 贝叶斯推断              学习: 反向传播 + SGD
可解释: ✅ 图结构 = 领域知识           可解释: ❌ 黑盒
数据效率: 小样本可工作                  数据效率: 需要大量数据
```

### 核心分歧：表示 vs 推断

```
GM:  "我显式地说出变量之间的依赖关系"
      └→ 图结构 = 条件独立性的声明
      └→ 推断 = 在图上传概率消息

NN:  "我学会一个从输入到输出的函数"
      └→ 权重 = 分布式表示中的特征
      └→ 推断 = 一次确定性的前向计算
```

---

## 概念 2.2：共通之处 — 概率视角下的统一 (🔑)

### 1. 目标函数：都是 MLE/MAP

```
GM 的 MLE:   θ* = argmax Πᵢ P(x_i | θ)
             = argmax Σᵢ log P(x_i | θ)

NN 的 MLE:   W* = argmin Σᵢ Loss(y_i, f_W(x_i))
             = argmax Σᵢ log P(y_i | x_i, W)

完全相同的形式! 区别只在于:
  - GM: P 由条件概率表/势函数定义, 可归一化
  - NN: P 由神经网络输出参数化, 自动归一化 (softmax)
```

### 2. 消息传递：图算法的共同基因

```
PGM 的 Belief Propagation:   NN 的反向传播:
─────────────────────────    ────────────────
节点发送"消息"给邻居          节点发送"梯度"给上游
消息 = 概率值的乘积求和       梯度 = 局部导数的乘积求和
Sum-Product 半环              Chain Rule (自动微分)

共同模式: Collect → Aggregate → Propagate
```

| | BP (Belief Propagation) | BProp (Backpropagation) |
|---|---|---|
| **图结构** | 因子图 (factor graph) | 计算图 (computation graph) |
| **消息内容** | 概率向量 (边际) | 梯度 (偏导数) |
| **聚合操作** | Σ/Π (Sum-Product) | Σ (Chain Rule) |
| **传递方向** | 双向 (forward + backward) | 单向 (从 Loss 往回) |
| **输出** | 边际概率 P(X_i) | 参数梯度 ∂L/∂θ |

### 3. 变量消除 ↔ 反向传播的深层对应

```
VE (变量消除): 按序消除变量, 传递中间因子 → 计算 P(X_q|X_e)
反向传播:      按序消除中间节点, 传递梯度 → 计算 ∂L/∂θ

VE 中: 消除 X 时, 把涉及 X 的所有因子乘起来, 再对 X 求和
反向传播中: 梯度流经节点时, 收集所有来自下游的梯度, 乘上该节点的局部 Jacobian

两者都遵循: 拓扑序 + 局部操作 + 传递中间结果
```

---

## 概念 2.3：关键差异 — 互补的优缺点 (🔑)

### 差异 1: 可解释性

```
GM:  图结构直接编码了领域知识
     → 医生可以看"症状-疾病-检查结果"的图结构
     → 每条边都有语义含义

NN:  权重矩阵是黑盒
     → 虽然可以可视化特征图, 但没有显式的变量关系
     → 事后解释 (LIME, SHAP, Grad-CAM) 只是近似
```

### 差异 2: 推断方式

```
GM:  推断可能很昂贵
     → 精确推断 (VE, JT): 对树结构 O(N), 一般图 #P-hard
     → 近似推断 (VI, MCMC): 每次推断都要迭代

NN:  推断极快
     → 一次前向传播 = O(#参数) = 确定性的
     → 但无法给出不确定性估计 (predictive uncertainty)
```

### 差异 3: 数据效率与泛化

```
GM:  小数据友好
     → 先验知识 (图结构) 大幅约束了参数空间
     → 几百个样本即可学习可解释的模型

NN:  大数据依赖
     → 没有显式的结构先验 → 需要大量数据来"发现"规律
     → 但数据充足时, 可以学习极其复杂的模式
```

### 差异 4: 处理缺失数据

```
GM:  自然处理缺失值
     → 推断缺失变量的后验: P(X_miss | X_obs)
     → 生成式模型天然支持

NN:  难以处理缺失值
     → 需要特殊技巧: 插补 (imputation), mask, 或专门的架构
     → 判别式模型直接学 P(Y|X), 不学 P(X)
```

### 差异 5: 生成 vs 判别

```
GM (生成式): P(X, Y) → 可生成数据, 可做各种条件查询
     P(Y|X) = P(X,Y) / Σ_Y P(X,Y)

NN (判别式): P(Y|X) → 只做预测, 不需要建模 X 的分布
     更简单, 更直接, 但不会"理解"数据是怎么生成的
```

---

## 概念 2.4：一张表总结异同

| 维度 | 图模型 (GM) | 神经网络 (NN) |
|------|-----------|-------------|
| **表示** | 图 + 条件概率 | 层次化权重 |
| **推断** | VE / BP / MCMC / VI | 前向传播 |
| **学习** | MLE / EM / 贝叶斯 | SGD + 反向传播 |
| **可解释性** | 高 (结构=知识) | 低 (黑盒) |
| **数据效率** | 高 (强先验) | 低 (需要大数据) |
| **表达能力** | 受限于图结构和参数化 | 极强 (万能近似) |
| **不确定性** | 天然 (后验分布) | 需要额外努力 (BNN, Ensemble) |
| **缺失数据** | 天然支持 | 困难 |
| **生成能力** | 天然 (生成式模型) | 需特殊架构 (VAE, GAN) |
| **计算效率** | 推断可能很贵 | 推断极快 |
| **共性** | 都做 MLE/MAP, 都用图算法 (消息传递), 都处理概率 | |

---

# 第三部分：Combining DL Methods and GMs

---

## 概念 3.1：为什么要把 DL 和 GM 结合起来？ (🔑)

```
DL 的优势:                    GM 的优势:
─────────────────           ─────────────────
强大的函数近似能力           结构化的概率建模
端到端学习                   天然的不确定性估计
处理高维原始数据              处理缺失数据和稀疏数据
层次化表示学习                可解释的变量依赖关系

结合目标: 取各自之长, 补各自之短
    → 用 NN 参数化 GM 中复杂的条件概率
    → 用 GM 给 NN 加上结构化和概率化的约束
    → 结果: 既有表达力, 又有概率语义
```

---

## 概念 3.2：深度生成模型 — 用 NN 构建 GM (🔑🔑🔑)

### VAE (Variational Autoencoder)

```
传统 GM: P(X) = Σ_Z P(Z)P(X|Z), Z 是离散隐变量
VAE:     P(X) = ∫ P(Z)P_θ(X|Z) dZ, Z 是连续隐变量

P_θ(X|Z) = N(X | μ_θ(Z), σ²I)    ← 用 NN (decoder) 参数化!
Q_φ(Z|X) = N(Z | μ_φ(X), σ²_φ(X)) ← 用 NN (encoder) 做推断!

训练: ELBO = E_Q[log P_θ(X|Z)] - KL(Q_φ(Z|X) || P(Z))
      └── 重构项 (autoencoder) ─┘   └── 正则项 (GM先验) ─┘

这就是: GM 的概率框架 + NN 的强大参数化 = VAE!
```

### GAN (Generative Adversarial Network)

```
GAN 的隐式生成:
  不需要显式地定义 P(X), 只需要从 P(X) 中采样!

  Generator G: z ~ N(0,I) → G(z) = 生成的样本
  Discriminator D: 判断 x 是真实的还是生成的

  min_G max_D E[log D(x)] + E[log(1-D(G(z)))]

  这 = 隐式的概率模型 → 没有显式的 likelihood, 但能生成极高质量的样本
```

### Normalizing Flow

```
Flow: X = f_θ(Z), Z ~ N(0,I), f 是可逆的 NN

P_X(x) = P_Z(f⁻¹(x)) · |det ∂f⁻¹(x)/∂x|
         └─ 简单分布 ─┘   └─ Jacobian 行列式 (体积变化) ─┘

优势: 精确的 likelihood 计算 + 高效的采样
```

### 三种 DGM 对比

| | VAE | GAN | Normalizing Flow |
|---|---|---|---|
| **Likelihood** | ELBO (下界) | 无 (隐式) | 精确 |
| **采样质量** | 中等 (模糊) | 极好 | 中等-好 |
| **隐空间** | 平滑, 连续, 可插值 | 通常无 encoder | 必须同维度 |
| **推断** | Amortized VI | 无 | 精确反函数 |
| **训练稳定性** | 好 | 困难 (mode collapse) | 好 |

---

## 概念 3.3：用 NN 增强 GM 的推断 — Amortized Inference (🔑)

### 传统 VI vs Amortized VI

```
传统 VI (如 L7-L8):
  对每个观测 x_i, 优化 q_i*(z) = argmin KL(q(z) || P(z|x_i))
  → 每个数据点都要跑一次优化 → 慢!

Amortized VI (VAE encoder):
  训练一个 NN: q_φ(z|x) ≈ argmin KL(q(z) || P(z|x))
  → 推断 = 一次前向传播 → 极快!
  → "摊销"了推断成本 → amortized inference
```

### NN 参数化势函数

```
传统 MRF (如 L4): P(X) ∝ Π_c ψ_c(X_c), ψ_c 由表格定义

神经 MRF: ψ_c(X_c) = exp(NN_c(X_c))
  → 用 NN 学习势函数, 而不是手工设计
  → NN 可以处理高维连续的因子
```

---

## 概念 3.4：结构化预测 — NN 特征 + GM 结构 (🔑)

### 条件随机场 (CRF) + NN = 深度结构化模型

```
传统 CRF (如 L6):
  P(Y|X) ∝ Π_c ψ_c(Y_c, X)
  特征 ψ_c 需要手工设计 (HMM, 词性标注特征...)

深度 CRF (神经网络 CRF):
  P(Y|X) ∝ Π_c exp(NN_φ(X)_c · Y_c)
  
  步骤:
    1. NN_φ 从原始 X 提取特征 (如 CNN 处理图像 → 特征图)
    2. CRF 层基于这些特征做结构化预测
    3. 整个模型端到端训练!

应用: 语义分割 (DeepLab), 序列标注 (BiLSTM-CRF), 姿态估计
```

---

## 概念 3.5：DGMs 的历史脉络

```
1980s: 玻尔兹曼机 (Boltzmann Machine) — 最早的神经+概率模型
       ↓
2006: Deep Belief Networks (DBN) — 逐层预训练, 深度学习复兴的标志
       ↓
2013: VAE (Kingma & Welling) — Amortized VI + NN, 可扩展的深度生成模型
       ↓
2014: GAN (Goodfellow et al.) — 对抗训练, 隐式生成模型
       ↓
2015: Normalizing Flows — 可逆NN, 精确likelihood
       ↓
2019+: Diffusion Models — 渐进式去噪, 当前SOTA生成模型
       ↓
2020+: 大规模生成模型 (GPT, DALL-E, Stable Diffusion)
```

---

# 第四部分：Bayesian Learning of NNs

---

## 概念 4.1：为什么需要贝叶斯神经网络？ (🔑🔑)

### 标准 NN 的问题

```
标准 NN: 给出点估计 ŷ = f_W(x)
  → 仅一个预测值, 不知道有多"确定"
  → 对分布外 (OOD) 数据: 可能给出高置信度的错误预测
  → 无法说"我不知道"

贝叶斯 NN: 给出预测分布 P(y|x, D)
  → 预测 + 不确定性 (Epistemic Uncertainty)
  → OOD 数据: 后验预测分布宽 → 模型"知道自己的无知"
  → 在医疗、自动驾驶等安全关键的场景至关重要
```

### BNN 的核心思想

```
标准 NN:  W* = argmin Loss(D, W)    → 单一最优权重

贝叶斯 NN: P(W|D) ∝ P(D|W) · P(W)  → 权重的后验分布!
           └── 所有可能的权重, 按"合理度"加权 ──┘

预测: P(y|x, D) = ∫ P(y|x, W) · P(W|D) dW
                  └── 对所有可能的权重做模型平均 ──┘
      = 无穷多个 NN 的加权集成!
```

### 两种不确定性

```
偶然不确定性 (Aleatoric): 数据本身的噪声
  → 即使知道真实函数, 预测仍有误差
  → 例: 掷硬币 — 知道概率是 0.5, 但每次结果不确定
  → 不可约, 通过建模输出分布来捕获

认知不确定性 (Epistemic): 模型知识的不足
  → 知道更多数据/更好的模型可以减少
  → 例: 没见过这类数据 → "我不确定"
  → BNN 通过后验分布自然捕获!
```

---

## 概念 4.2：BNN 的核心挑战 — 后验推断 (🔑🔑)

### 为什么精确后验不可行

```
P(W|D) = P(D|W)P(W) / P(D)
        = P(D|W)P(W) / ∫ P(D|W)P(W) dW

问题: 积分 ∫ P(D|W)P(W) dW 是高维 (百万~亿维) 的!
      → 精确计算不可能
      → 需要近似推断 ← PGM 的 VI/MCMC 工具再次登场!
```

### BNN 推断方法一览

| 方法 | 思想 | 计算代价 | 质量 |
|------|------|---------|------|
| **Laplace 近似** | 在 MAP 处做高斯近似 | 低 (后处理) | 近似粗糙 |
| **Variational Inference** | 优化 q(W) ≈ P(W\|D) | 中 (训练时优化) | 中等 |
| **MC Dropout** | Dropout 在测试时也开着 | 极低 (几乎免费) | 合理近似 |
| **SWAG** | SGD 轨迹的高斯拟合 | 低 | 较好 |
| **Deep Ensembles** | 独立训练 M 个 NN | 高 (M 倍训练) | 极好 |
| **HMC / SGLD** | MCMC 采样权重 | 极高 | 接近精确 |

---

## 概念 4.3：关键方法详解 (🔑)

### 1. MC Dropout (Gal & Ghahramani, 2016)

```
核心洞见: Dropout 训练的 NN ≈ 对深度高斯过程做 VI!

做法:
  训练时: 正常用 Dropout (如 p=0.5)
  测试时: 也开 Dropout! 前向传播 T 次 (如 100 次)
          每次采样不同的 dropout mask → T 个不同的预测
  
  P(y|x) ≈ (1/T) Σₜ f_{W, mask_t}(x)     ← 预测均值
  Var ≈ (1/T) Σₜ (fₜ - mean)²            ← 认知不确定性

优点: 几乎免费! 只需在测试时多跑几次前向
缺点: 不确定性估计的校准不如真正的贝叶斯方法
```

### 2. Bayes by Backprop (Blundell et al., 2015)

```
核心: 用反向传播优化变分分布 q(W|θ) 的参数 θ

假设: q(W) = Πᵢ N(w_i | μ_i, σ_i²)  ← 高斯 Mean-Field VI

重参数化: w = μ + σ·ε, ε ~ N(0,1)
  → 梯度可以流过 μ 和 σ!

目标: ELBO = E_q[log P(D|W)] - KL(q(W) || P(W))
            └── 似然 ──┘   └── 正则化 (向先验靠拢) ──┘

训练: 参数数量翻倍 (μ 和 σ 各一组), 用 SGD/Adam 优化 ELBO
```

### 3. Laplace 近似

```
三步:
  1. 训练标准 NN → 找到 MAP 估计 W_MAP
  2. 计算 Hessian H = ∇²(-log P(D,W))|W_MAP
  3. 后验近似: P(W|D) ≈ N(W_MAP, H⁻¹)

优点: 在已有训练好的模型上做后处理
缺点: 高斯 + 单峰近似, Hessian 计算昂贵 (需近似)
```

### 4. SWAG (Stochastic Weight Averaging Gaussian)

```
核心: 利用 SGD 的轨迹来估计后验的协方差

  1. 训练标准 NN (带恒定 LR 或余弦衰减)
  2. 收集 SGD 迭代后期的权重快照: W₁, W₂, ..., W_K
  3. 拟合高斯: μ = mean(W_k), Σ = diag(Σ_diag) + low-rank(Σ_lr)

预测: 从 N(μ, Σ) 采样权重 → 前向传播 → 模型平均

优点: 实现简单, 不改变训练过程, 质量好
```

---

## 概念 4.4：BNN 的实践考量

### 先验选择

```
先验 P(W) 的选择影响很大:

常用:
  - N(0, σ²I): 最常用, 等价于 L2 正则化
  - N(0, diag(σ²)): per-parameter 方差
  - Scale mixture (Spike-and-Slab): 部分权重→0, 部分活跃
  - 基于架构的先验 (如 Conv 层的结构化先验)

启发: 标准 NN 的 Weight Decay ≈ BNN 的 Gaussian 先验
```

### 计算 vs 收益

```
方法              额外计算    不确定性质量    适用场景
────────────────  ────────    ───────────    ────────
MC Dropout        ~T 次前向     ★★☆☆☆        快速原型, 已有模型
Deep Ensembles    M 倍训练     ★★★★★        最高质量, 有资源
Bayes by Backprop 2x 参数      ★★★☆☆        科研, 探索
Laplace           1次Hessian   ★★☆☆☆        后处理, 轻量
SWAG              存储K个快照  ★★★★☆        实用, 性价比高
```

---

## 📋 全部概念一张表

| # | 主题 | 核心概念 |
|---|------|---------|
| **1** | **DL 组件** | 层, 激活 (ReLU/GELU), 损失=MLE, 优化器 (SGD→Adam), 反向传播, 正则化 |
| **2** | **GM vs NN** | 显式 vs 隐式建模, 推断方式不同, 统一在 MLE/MAP 下, 消息传递是公共基因 |
| **3** | **DL + GM** | VAE=GM框架+NN参数化, Amortized Inference, 深度CRF, Diffusion |
| **4** | **Bayesian NN** | P(W\|D) 替代 W*, MC Dropout, VI, Laplace, SWAG, Ensemble |

---

## 🔗 概念关系图

```
                    ┌──────────────────┐
                    │ ① DL 组件概览     │
                    │ 层,激活,优化,BP   │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ ② GM vs NN │  │③ DL + GM│  │④ BNN     │
        │ 对比两种   │  │ 融合两种  │  │ 概率视角  │
        │ 范式       │  │ 范式      │  │ 看 NN     │
        └──────────┘  └──────────┘  └──────────┘
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                    ┌──────────────────┐
                    │ 统一视角:         │
                    │ 概率建模 + 深度表示│
                    │ + 结构化先验      │
                    └──────────────────┘
```

---

## 🎯 核心洞见一句话

| # | 洞见 |
|---|------|
| 1 | **DL = 层次化表示 + 梯度优化** — 损失函数就是负对数似然, 反向传播就是计算图上的消息传递 |
| 2 | **GM 和 NN 不是对立的** — 都做 MLE/MAP, 都用图消息传递, GM 重结构, NN 重表示 |
| 3 | **VAE = GM + NN 的最佳案例** — 用 NN 做推断 (encoder) 和生成 (decoder), 用 ELBO 做目标 |
| 4 | **BNN = 给 NN 加上不确定性** — P(W\|D) 替代点估计, 预测 = 所有可能权重的加权平均 |
| 5 | **PGM 工具箱在 DL 中无处不在** — VI, MCMC, ELBO, Prior, Posterior 全是 L1-L10 的概念! |

---

## 🧪 自测清单

- [ ] DL 组件: 激活函数为什么需要非线性? ReLU 比起 Sigmoid 好在哪里?
- [ ] DL 组件: CrossEntropy Loss 对应什么统计假设? Adam 维护了哪两个 EMA?
- [ ] GM vs NN: 图模型的"消息传递"和反向传播有什么结构相似性?
- [ ] GM vs NN: 什么场景用 GM 比 NN 更合适? (小数据? 可解释? 缺失数据?)
- [ ] DL+GM: VAE 的 encoder 和 decoder 分别对应 GM 中的什么?
- [ ] DL+GM: 什么是 Amortized Inference, 它比传统 VI 好在哪里?
- [ ] BNN: 为什么点估计的 NN 无法给出"我不知道"?
- [ ] BNN: MC Dropout 为什么是近似的贝叶斯推断? 测试时和标准 Dropout 有什么区别?
- [ ] BNN: 比较 MC Dropout, Bayes by Backprop, Laplace, SWAG 的代价与质量

---

> L11 完成了从 PGM 到 DL 的全面桥梁。你已经具备了用概率视角理解深度学习、用深度学习工具增强图模型的双向能力。L12-L13 将深入深度生成模型的具体架构 (VAE, GAN, Diffusion)。
