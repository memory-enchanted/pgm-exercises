# CMU 10-708 Lecture 11 课后练习 & 答案

> 配套教材: Goodfellow et al. (2016) Deep Learning Ch.6-8, Murphy Ch.13, Ch.28, Bishop Ch.5
>
> 题目覆盖 L11 四大主题, 三级: 🟢 基础 | 🟡 进阶 | 🔴 挑战

---

# 第一部分: An Overview of DL Components

---

## 🟢 Q1. 损失函数与统计假设

**(A)** MSE Loss: L = (1/N) Σ (y_i - f(x_i))² 对应什么统计假设?

**(B)** Binary Cross-Entropy Loss: L = -Σ [y_i log p_i + (1-y_i) log(1-p_i)] 对应什么统计假设?

**(C)** 给 MSE 加上 L2 正则: L + λ‖w‖²。从 MAP 角度解释这等价于对权重施加了什么先验?

<details>
<summary>点击查看答案</summary>

**(A) MSE = MLE with Gaussian noise:**

```
假设 y = f(x) + ε, ε ~ N(0, σ²)

P(y|x) = N(y | f(x), σ²) ∝ exp(-(y - f(x))² / (2σ²))

log P(y_1,...,y_N | X) = -N log σ - N/2 log(2π) - (1/2σ²) Σ (y_i - f(x_i))²

最大化 log likelihood ⟺ 最小化 Σ (y_i - f(x_i))² = MSE
```

**(B) Cross-Entropy = MLE with Bernoulli output:**

```
y|x ~ Bernoulli(p), p = sigmoid(f(x))

P(y|x) = pʸ (1-p)^{1-y}

log P = y log p + (1-y) log(1-p)

最大化 ⟺ 最小化 -[y log p + (1-y) log(1-p)] = BCE
```

**(C) L2 Regularization = Gaussian Prior (MAP):**

```
w ~ N(0, (1/2λ) · I)  →  P(w) ∝ exp(-λ‖w‖²)

MAP: max_w log P(y|X,w) + log P(w)
    = max_w -MSE - λ‖w‖²
    = min_w MSE + λ‖w‖²
```

</details>

---

## 🟢 Q2. 激活函数 — 为什么需要非线性?

**(A)** 证明: 如果没有激活函数, 任意多层的全连接网络等价于一个线性层。写出合并后的权重矩阵。

**(B)** ReLU(x) = max(0, x)。分析 ReLU 的优缺点各一个。

**(C)** 深层网络中, 为什么 Sigmoid 容易导致"梯度消失"?

<details>
<summary>点击查看答案</summary>

**(A) 多层线性 = 一层线性:**

```
无激活的 3 层:
  h₁ = W₁x + b₁
  h₂ = W₂h₁ + b₂
  h₃ = W₃h₂ + b₃
    = W₃(W₂(W₁x + b₁) + b₂) + b₃
    = (W₃W₂W₁)x + (W₃W₂b₁ + W₃b₂ + b₃)
    = W'x + b'
```

任意多层的线性堆叠始终是仿射变换 → 无法学习非线性函数。激活函数打破这个线性级联, 赋予网络万能近似能力。

**(B) ReLU:**

- 优点: 梯度 = 1 (x>0), 不饱和, 极大缓解深层网络的梯度消失
- 缺点: 死神经元 — 如果某个神经元输出始终 <0, 梯度永远为 0, 该神经元永不更新 (Dying ReLU)

**(C) Sigmoid 梯度消失:**

```
σ'(x) = σ(x)(1 - σ(x))  → 最大值 = 0.25 (在 x=0 处)

|x| 大时: σ'(x) → 0 (饱和)

L 层网络: ∂L/∂W₁ = ∂L/∂h_L · σ'(z_L) · W_L · σ'(z_{L-1}) · ... · σ'(z₁)
                 └─ 每个 ≤ 0.25 ─┘

乘积: (0.25)^L → 指数衰减! 对于 L=20, (0.25)²⁰ ≈ 10⁻¹²
```

</details>

---

## 🟢 Q3. 反向传播的手算

对计算图: L = (w·x + b - y)², 已知 w=2, x=3, b=1, y=10。

**(A)** 画计算图, 标注中间变量和值。

**(B)** 从 L 出发, 反向计算 ∂L/∂w, ∂L/∂b, ∂L/∂x。验证解析梯度公式: ∂L/∂w = 2(wx+b-y)·x。

<details>
<summary>点击查看答案</summary>

**(A) 计算图:**

```
x=3 ─┐
      ├→ [×] → z₁=6 ─┐
w=2 ─┘                ├→ [+] → z₂=7 ─→ [-] → r=-3 → [²] → L=9
b=1 ──────────────────┘             y=10 ─┘
```

**(B) 反向计算 (Chain Rule):**

```
前向: z₁ = w·x = 6,  z₂ = z₁ + b = 7,  r = z₂ - y = -3,  L = r² = 9

反向:
  ∂L/∂r = 2r = -6
  ∂L/∂z₂ = ∂L/∂r · ∂r/∂z₂ = -6 · 1 = -6
  ∂L/∂z₁ = ∂L/∂z₂ · ∂z₂/∂z₁ = -6 · 1 = -6
  ∂L/∂w  = ∂L/∂z₁ · ∂z₁/∂w  = -6 · 3 = -18
  ∂L/∂b  = ∂L/∂z₂ · ∂z₂/∂b  = -6 · 1 = -6
  ∂L/∂x  = ∂L/∂z₁ · ∂z₁/∂x  = -6 · 2 = -12

验证: dL/dw = 2(wx+b-y)·x = 2·(-3)·3 = -18 ✓
      dL/db = 2(wx+b-y)·1 = 2·(-3)·1 = -6  ✓
```

</details>

---

## 🟢 Q4. 优化器对比

**(A)** Momentum 的更新规则: v ← βv + g; θ ← θ - η·v。展开 v_t, 说明它是过去梯度的什么形式。

**(B)** Adam 维护了哪两个指数移动平均? 为什么需要偏差校正 (bias correction)?

**(C)** 简述: 为什么 Adam 的步长 m̂/√ŝ 是"无量纲"的? 这个特性有什么好处?

<details>
<summary>点击查看答案</summary>

**(A) Momentum 展开:**

```
v_t = β v_{t-1} + g_t
    = g_t + β g_{t-1} + β² g_{t-2} + ... + β^t g_0

这是过去所有梯度的指数衰减加权和 (Exponential Moving Average)
→ 最新梯度权重最大, 历史梯度权重按 β^k 衰减
→ β=0.9: 有效窗口 ≈ 1/(1-β) = 10 步
```

**(B) Adam 的两个 EMA:**

```
一阶矩: m_t = β₁ m_{t-1} + (1-β₁) g_t     (动量, β₁=0.9)
二阶矩: s_t = β₂ s_{t-1} + (1-β₂) g_t²    (自适应LR, β₂=0.999)

偏差校正:
  初始化 m₀=s₀=0, 且 β≈1 → 早期 m_t 和 s_t 严重偏向 0
  E[m_t] = (1-β₁ᵗ)·E[g_t] → 除以 (1-β₁ᵗ) 校正

  m̂_t = m_t / (1 - β₁ᵗ)
  ŝ_t = s_t / (1 - β₂ᵗ)

当 t 大时 β₁ᵗ → 0, 校正因子 → 1, 自然退化为无校正版本。
```

**(C) 无量纲:**

```
g 的单位 = [loss] / [param]
m 的单位 = [loss] / [param]
s 的单位 = [loss²] / [param²] → √s 的单位 = [loss] / [param]

m̂/√ŝ → [loss/param] / [loss/param] = 无量纲

好处: 步长的量级与参数本身的尺度无关 → 每参数自适应学习率
→ 天生适合各层参数梯度尺度差异大的场景 (如 NLP、Transformer)
```

</details>

---

# 第二部分: Similarities and Differences between GMs and NNs

---

## 🟡 Q5. 反向传播与 Belief Propagation 的结构对应

**(A)** L5 BP 的消息公式和反向传播中的梯度传递有什么结构上的相似性? 写出二者的"聚合-传递"模式。

**(B)** 计算图 (computation graph) 和因子图 (factor graph) 的节点和边分别对应什么?

**(C)** 核心区别: BP 的消息内容是概率值, 反向传播的消息内容是梯度。这导致了什么本质差异?

<details>
<summary>点击查看答案</summary>

**(A) 聚合-传递模式:**

```
BP 消息:
  m_{i→j}(X_j) = Σ_{X_i} ψ_{ij} · Π_{k∈N(i)\j} m_{k→i}(X_i)
  收集来自其他邻居的消息 → 乘上局部因子 → 对 X_i 求和 → 发给 j

反向传播:
  ∂L/∂a_i = Σ_{j∈children(i)} ∂L/∂a_j × ∂a_j/∂a_i
  收集来自子节点的梯度 → 乘上局部 Jacobian → 求和 → 发给父节点

共同结构:
  Collect (收集) → Multiply (乘局部信息) → Aggregate (求和) → Send (发送)
```

**(B) 图节点对应:**

| 因子图 | 计算图 |
|--------|--------|
| 变量节点 (circle) | 数据/参数节点 |
| 因子节点 (square) | 操作节点 (+, ×, ReLU, MatMul) |
| 边: 变量参与因子 | 边: 数据流入/流出操作 |
| 消息: 概率因子 (向量) | 消息: 梯度 (同维度向量/矩阵) |

**(C) 本质差异:**

```
BP (概率):  消息传递后得到边际概率 → 理解"每个变量是啥"
BProp (梯度): 消息传递后得到参数梯度 → 知道"往哪调参"

BP 做 Sum-Product (离散 & 概率)
BProp 做 Chain Rule (连续 & 自动微分)

这也解释了为什么:
- GM 能给出 P(X_i|evidence) → 不确定性估计天然
- NN 只能给出 ŷ = f(x) → 需要特殊方法 (如 BNN) 才能得到不确定性
```

</details>

---

## 🟡 Q6. GM 和 NN 的互补性

**(A)** 写出三种 GM 比 NN 更适合的场景, 以及各自的理由。

**(B)** 写出三种 NN 比 GM 更适合的场景。

**(C)** 为什么说 GM 和 NN 并不是对立的, 而是互补的? 用一句话概括。

<details>
<summary>点击查看答案</summary>

**(A) GM 更适合的场景:**

| 场景 | 理由 |
|------|------|
| 小样本数据 | GM 的图结构 = 强先验 → 约束参数空间 → 不需要大数据 |
| 需要可解释性 | 图结构直接编码领域知识, 每条边有语义 (如医疗诊断) |
| 缺失数据 | GM 天然能做 P(X_miss\|X_obs) 的推断 |
| 不确定性量化 | GM 给出后验分布, 天然支持"我不知道"的判断 |
| 结构化预测 | 变量之间有已知的约束关系 (如 HMM, CRF) |

**(B) NN 更适合的场景:**

| 场景 | 理由 |
|------|------|
| 大数据 + 高维原始输入 | NN 从数据中学习表示, 不需要手工设计特征 |
| 图像、语音、文本 | CNN/Transformer 的归纳偏置 (inductive bias) 极适合 |
| 快速推断 | 一次前向 = O(参数), 不迭代 |
| 端到端学习 | 从原始输入直接到输出, 不需要中间的结构化建模 |
| 极强表达能力 | 万能近似定理 → 只要有足够数据和算力 |

**(C) 一句话:**

```
GM 和 NN 互补: GM 提供结构化概率框架 (推理+不确定),
NN 提供强大的函数近似能力 (表示学习+端到端优化),
现代最好的方法 (VAE, Diffusion, Graph Neural Nets) 都是二者的融合。
```

</details>

---

# 第三部分: Combining DL Methods and GMs

---

## 🟡 Q7. VAE — GM 框架 + NN 参数化

**(A)** VAE 的 encoder q_φ(z|x) 和 decoder p_θ(x|z) 分别对应 GM 中的什么概念?

**(B)** 写出 VAE 的 ELBO 目标, 并解释为什么它分为重构项和 KL 正则项。

**(C)** 什么是重参数化技巧 (reparameterization trick)? 为什么 VAE 需要它?

<details>
<summary>点击查看答案</summary>

**(A) GM 对应:**

```
Encoder q_φ(z|x): 对应 GM 中的推断过程 (E-step / VI)
  → 给定观测 x, 推断隐变量 z 的后验分布
  → 用 NN 做 Amortized Inference → 一次前向 = 推断完毕!

Decoder p_θ(x|z): 对应 GM 中的生成模型 (条件概率表/势函数)
  → 给定隐变量 z, 生成 x 的分布
  → 用 NN 参数化 → 可以建模极复杂的 P(X|Z)
```

**(B) ELBO:**

```
ELBO = E_{q_φ(z|x)}[log p_θ(x|z)] - KL(q_φ(z|x) || p(z))
       └── 重构项 ──┘           └── KL 正则项 ──┘

重构项: 让 decoder 从 z 中重建 x ← 数据拟合 (MLE)
KL 正则: 让 encoder 的后验 q(z|x) 不要偏离先验 p(z) 太远 ← 正则化

两者平衡 = GM 中 VI 的标准配方!
```

**(C) 重参数化技巧:**

```
问题: z ~ N(μ_φ(x), σ_φ(x)²)
     采样操作不可导! → 梯度无法流回 φ!

解决: z = μ + σ·ε, ε ~ N(0,I)
     ε 的采样和参数无关 → 梯度可以流过 μ 和 σ!

这使整个 VAE 可以端到端用反向传播训练。
```

</details>

---

## 🟡 Q8. 深度结构化模型

**(A)** 传统 CRF 和 Neural CRF 的核心区别是什么?

**(B)** 为什么说"NN 做特征提取 + CRF 做结构化预测"是 DL 和 GM 结合的典范?

**(C)** Amortized Inference 比传统 VI 有什么优势? 有什么代价?

<details>
<summary>点击查看答案</summary>

**(A) 核心区别:**

```
传统 CRF:
  特征 ψ_c(Y_c, X) 由人工设计
  → 需要领域专家, 表达能力有限

Neural CRF:
  特征由 NN_φ(X) 自动学习
  → 端到端训练, 特征和推断联合优化
  → 可处理原始高维数据 (像素, 字符)
```

**(B) 分工合作:**

```
NN (特征提取):     从原始数据中提取有意义的特征
                  卷积 → 边缘/纹理; 自注意力 → 上下文

GM/CRF (结构化预测): 基于特征做结构化推理
                   确保输出满足约束 (如标签平滑、一致性)

端到端训练:        梯度从 CRF 的损失流回 NN ↔ 两者协同优化!

应用: BiLSTM-CRF (命名实体识别), DeepLab (语义分割)
```

**(C) Amortized VI vs 传统 VI:**

| | Amortized VI (NN encoder) | 传统 VI (per-sample) |
|---|---|---|
| **推断速度** | 一次前向传播 → 极快 | 每个样本需迭代优化 → 慢 |
| **能否共享** | 不同样本共享 NN 参数 | 每个样本独立优化 |
| **精度** | 近似, 受 NN 表达能力限制 | 可精确到局部最优 |
| **泛化** | 可泛化到未见过的数据 | 只对当前样本有效 |
| **代价** | 可能有 Amortization Gap | 推断更精确但更昂贵 |

</details>

---

# 第四部分: Bayesian Learning of NNs

---

## 🔴 Q9. 贝叶斯神经网络的核心思想

**(A)** 标准 NN 训练得到点估计 W*, 贝叶斯 NN 得到的是什么? 写出贝叶斯预测的公式。

**(B)** 解释偶然不确定性 (Aleatoric) 和认知不确定性 (Epistemic) 的区别。为什么标准 NN 只能捕获前者?

**(C)** 为什么 BNN 的后验推断在高维中不可行? 写出后验形式并指出困难所在。

<details>
<summary>点击查看答案</summary>

**(A) BNN 预测:**

```
标准 NN:  ŷ = f_{W*}(x)                              (点估计)

贝叶斯 NN: P(y|x, D) = ∫ P(y|x, W) · P(W|D) dW
           后验预测 = 所有可能权重的预测的加权平均!

W 不再是单一点, 而是一个分布 → 预测天然带有不确定性。
```

**(B) 两种不确定性:**

```
偶然不确定性 (Aleatoric):
  → 数据本身的噪声, 无法通过更多数据消除
  → 例: 掷硬币, 即使知道 P(heads)=0.5, 每次结果仍不确定
  → 通过在输出层建模方差来捕获 (如异方差回归)
  → 标准 NN 也可以建模这个

认知不确定性 (Epistemic):
  → 模型知识不足, 可通过更多数据减少
  → 例: 从没见过猫的模型看到猫 → "我不确定这是什么"
  → 只有通过权重的后验分布才能捕获
  → 标准 NN 没有 → 可能对陌生输入给出高置信度的错误预测!

标准 NN: 输出 = f_{W*}(x), 对任何 x 都强制给出一个答案
BNN:     输出 = 分布, 对训练分布外的 x → 方差大 → "我不知道"
```

**(C) 后验推断的困难:**

```
P(W|D) = P(D|W) P(W) / P(D)
        = P(D|W) P(W) / ∫ P(D|W) P(W) dW

问题: W 维度 = 百万~亿!
积分 ∫ P(D|W) P(W) dW 在高维空间中无法精确计算 (维数灾难)

需要近似推断:
  - VI: 找 q(W) ≈ P(W|D)
  - MCMC: 采样 W ~ P(W|D)
  - Laplace: 在 W_MAP 处做高斯近似
  - MC Dropout: 隐式的近似贝叶斯推断

这正是 PGM 工具 (VI, MCMC) 在 DL 中的直接应用!
```

</details>

---

## 🔴 Q10. MC Dropout 与贝叶斯近似

**(A)** MC Dropout 和标准 Dropout 在测试时有什么区别? 为什么这个区别赋予了它贝叶斯解释?

**(B)** 用 MC Dropout 做 T 次前向传播后, 如何计算:
  - 预测均值
  - 预测总方差
  - 认知不确定性 (Epistemic Uncertainty)

**(C)** MC Dropout 作为贝叶斯方法的优缺点是什么?

<details>
<summary>点击查看答案</summary>

**(A) 测试时区别:**

```
标准 Dropout:
  训练: Dropout ON
  测试: Dropout OFF, 权重乘 (1-p)
  → 单一确定性预测 ŷ

MC Dropout:
  训练: Dropout ON (和标准一样!)
  测试: Dropout 仍然 ON!
  → T 次前向, 每次 mask 不同
  → T 个预测 → 分布 → 不确定性!

贝叶斯解释 (Gal & Ghahramani 2016):
  带 Dropout 的 NN ≈ 深度高斯过程的 VI
  每次 dropout mask = 从近似后验 q(W) 采样不同的子网络
  T 次采样 ≈ 蒙特卡洛近似贝叶斯预测
```

**(B) 不确定性计算:**

```
预测均值:  μ_pred = (1/T) Σₜ f_{W_t}(x)

预测总方差: Var_total = (1/T) Σₜ f_{W_t}(x)² - μ_pred²

若同时建模偶然不确定性 (如输出 σ²):
  Aleatoric:  E[σ²] = (1/T) Σₜ σ²_{W_t}(x)
  Epistemic:  Var_pred = (1/T) Σₜ (f_{W_t}(x) - μ_pred)²

总方差 = Aleatoric + Epistemic
```

**(C) 优缺点:**

优点:
  - 几乎零额外成本: 不需要修改训练, 只需测试时多跑几次
  - 简单易用: 任何带 Dropout 的模型都能用
  - 实践中效果不错: 很多场景下不确定性质量可接受

缺点:
  - 不确定性校准不完美: 不是真正的贝叶斯后验
  - dropout rate p 变成需要调的超参数 (且其值和"先验"的关系不透明)
  - 对 ReLU + BN 组合的模型: MC Dropout 可能不太适合
  - 不确定性不随数据量增大而消失 (asymptotically inconsistent)
```

</details>

---

## 🔴 Q11. 贝叶斯 NN 方法比较

**(A)** 简述 Bayes by Backprop, Laplace 近似, 和 SWAG 的核心思想。

**(B)** 从计算代价、不确定性质量、实现难度三个维度比较: MC Dropout, Deep Ensembles, Bayes by Backprop, SWAG。

<details>
<summary>点击查看答案</summary>

**(A) 三种方法:**

```
Bayes by Backprop (BBB, Blundell+ 2015):
  → 假设 q(W) = Π N(w_i | μ_i, σ_i²) (Mean-Field Gaussian VI)
  → 用重参数化采样: w = μ + σ·ε
  → 优化 ELBO = E_q[log P(D|W)] - KL(q||prior)
  → 参数翻倍 (每个权重有 μ 和 σ)

Laplace 近似:
  → 训练标准 NN 得到 W_MAP
  → 计算 Hessian: H = ∇²(-log P) |_{W_MAP}
  → P(W|D) ≈ N(W_MAP, H⁻¹)
  → 后处理, 不改变训练

SWAG (Maddox+ 2019):
  → 利用 SGD 轨迹: 收集训练后期的 K 个权重快照
  → 拟合高斯: μ = mean(W_k)
              Σ = diag(Σ_diag) + low-rank(Σ_lr)
  → 从拟合的高斯中采样做预测
```

**(B) 综合比较:**

| 方法 | 计算代价 | 不确定性质量 | 实现难度 | 适用场景 |
|------|---------|------------|---------|---------|
| **MC Dropout** | 极低 (T次前向) | ★★☆☆☆ | 极低 | 快速原型, 已有模型 |
| **Deep Ensembles** | 高 (M倍训练) | ★★★★★ | 低 | 最高质量, 有GPU资源 |
| **BBB** | 中 (2x参数) | ★★★☆☆ | 中 | 科研, 探索 |
| **SWAG** | 低 (存K快照) | ★★★★☆ | 低-中 | 实用性价比高 |
| **Laplace** | 低 (1次Hessian) | ★★☆☆☆ | 中 | 后处理, 轻量 |
| **HMC/SGLD** | 极高 | ★★★★★ | 高 | 需要接近精确后验 |

选择建议:
  - 需要最佳不确定性 → Deep Ensembles
  - 快速尝试 → MC Dropout
  - 性价比 → SWAG
  - 纯后处理不重训 → Laplace

</details>

---

# 🔴 挑战题

---

## Q12. 统一视角: PGM 工具在 DL 中的应用

**(A)** L1-L10 学到的 PGM 工具 (VI, MCMC, ELBO, Prior/Posterior, Message Passing) 在 DL 中分别对应什么? 填表。

**(B)** 自选一个结合了 DL 和 GM 的现代模型 (如 VAE, Diffusion, Neural Process, Graph Neural Network), 分析它如何"继承"了 GM 的哪些思想和 DL 的哪些能力。

<details>
<summary>点击查看答案</summary>

**(A) PGM 工具箱在 DL 中的映射:**

| PGM 概念 | DL 对应 | 出现位置 |
|---------|---------|---------|
| **MLE** | 负对数似然 = 损失函数 | 所有 NN 训练 |
| **MAP** | MLE + L2/L1 正则化 | Weight Decay |
| **Prior** | 权重先验 P(W), 架构先验 | BNN, 正则化, 初始化 |
| **VI** | Bayes by Backprop, VAE encoder | BNN, DGM |
| **ELBO** | VAE 的训练目标 | VAE, Diffusion (变体) |
| **MCMC** | SGLD, HMC for BNN | 贝叶斯推断 |
| **Message Passing** | 反向传播, GNN 消息传递 | 所有 NN, GNN |
| **Latent Variables** | VAE 的 z, GAN 的 z, Diffusion 的 x_t | DGM |
| **因子图** | 计算图 | 自动微分框架 |
| **条件独立性** | 架构的归纳偏置 (CNN的局部性, Transformer的注意力) | 架构设计 |

**(B) 以 VAE 为例的分析:**

```
VAE = GM 的思想 + DL 的能力

继承 GM:
  - 生成式模型: P(X) = ∫ P(Z)P(X|Z) dZ (隐变量模型)
  - VI 框架: 优化 ELBO 而非精确的 log P(X)
  - KL 正则: 约束后验不要离先验太远
  - 概率语义: 可以计算 (近似) likelihood, 可以做推断

继承 DL:
  - NN 参数化: encoder 和 decoder 都是 NN → 强大的表达能力
  - 反向传播 + SGD: 端到端训练
  - Amortized Inference: 推断 = 一次前向, 而非每样本优化
  - 表示学习: Z 空间学出有语义的结构 (如 MNIST → 数字属性)

效果: 比传统 GM 表达能力更强 (能生成逼真图像),
      比纯 NN 更有概率语义 (有隐空间, 能做插值和采样)。
```

</details>

---

## 📊 综合自测评分

每题 10 分 (Q1-Q10), Q11-Q12 各 10 分加分, 共 120 分。

| 得分 | 评价 |
|------|------|
| 100-120 | L11 完全掌握, 已能在 PGM ↔ DL 之间自如迁移概念 |
| 80-99  | 主干扎实, 建议动手跑一遍 27 的代码练习 |
| 60-79  | 概念清晰, 重新推一遍 ELBO 和 MC Dropout 的原理 |
| < 60   | 先吃透 Q1-Q6 (组件+异同), 再看 VAE 和 BNN |

---

> L11 建立了 PGM → DL 的全面桥梁。你已具备:
> 1. 用 DL 组件搭建模型的能力 (Part 1)
> 2. 在 GM 和 NN 之间自如比较和选择的判断力 (Part 2)
> 3. 将二者融合使用的视野 (Part 3)
> 4. 用贝叶斯视角审视和增强神经网络的能力 (Part 4)
>
> 这正是现代机器学习的核心素养 — 既懂概率推理的结构化之美, 又懂深度学习的表示学习之力。
