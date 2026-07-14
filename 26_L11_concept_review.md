# CMU 10-708 Lecture 11 概念体系梳理 — 深度学习的统计与算法基础

> 整理自 Eric Xing 教授 CMU 10-708 Lecture 11: Statistical & Algorithmic Foundations of Deep Learning
>
> 核心教材: Goodfellow et al. (2016) Deep Learning Ch.6-8, Murphy Ch.13, Hastie et al. ESL Ch.11

---

## 📐 全局定位：PGM → Deep Learning 的桥梁

```
L1-L10: 概率图模型 (PGM)               L11: DL 的统计/算法基础
──────────────────────────            ──────────────────────────
显式建模条件独立性                    隐式学习表示 (representations)
精确/近似推断                         梯度下降 + 反向传播
参数: 条件概率表/势函数                参数: 权重矩阵 + 偏置
可解释性强                            黑盒但表达能力极强

L11 = "PGM 的统计原理 + 深度学习"的接口
      MLE, 梯度优化, 正则化 — PGM 的概念在 DL 中无处不在!
```

**一句话概括 L11**: 深度学习是"用梯度下降在大规模神经网络上做 MLE/MAP"。理解其统计基础 (MLE, 偏差-方差, 正则化) 和算法基础 (反向传播, SGD, Adam) 是把 PGM 知识迁移到 DL 的关键。

---

## 概念 1：MLE/MAP → 神经网络训练的损失函数 (🔑)

### 从 PGM 视角看神经网络

```
PGM:  log P(X | θ) = Σ_i log P(x_i | θ)
      → MLE: θ* = argmax Σ log P(x_i | θ)

神经网络 (分类):
      P(y|x, W) = softmax(f_W(x))
      → MLE: W* = argmin Σ CrossEntropy(y_i, f_W(x_i))
              = argmin -Σ log P(y_i | x_i, W)

完全相同的形式! NN 的训练 = 用梯度下降做 MLE!
```

### 常见损失函数的统计对应

| 损失函数 | 统计对应 | 假设 |
|---------|---------|------|
| **MSE** | MLE with Gaussian output | y\|x ~ N(f(x), σ²) |
| **Cross-Entropy** | MLE with Categorical output | y\|x ~ Categorical(softmax(f(x))) |
| **L1 Loss (MAE)** | MLE with Laplace output | y\|x ~ Laplace(f(x), b) |
| **Hinge Loss** | SVM / Max-Margin | — |
| **L2 Regularization** | Gaussian prior on W (MAP) | W ~ N(0, 1/λ) |
| **L1 Regularization** | Laplace prior on W (MAP) | W ~ Laplace(0, 1/λ) |

---

## 概念 2：反向传播 (Backpropagation) — 计算图上的 VE (🔑🔑🔑)

### 核心洞见

```
反向传播 = 在计算图上做自动微分 (Chain Rule)

计算图: 节点 = 操作, 边 = 数据流
前向: x → f₁ → f₂ → ... → f_L → Loss
反向: ∂L/∂x ← ∂L/∂f₁ ← ... ← ∂L/∂f_L ← 1 (从 Loss 往回传)
```

### 从 PGM 视角看反向传播

```
VE (变量消除): 在因子图上传递消息 → 计算边际
反向传播:      在计算图上传递梯度 → 计算 ∂Loss/∂θ

两者都是 "在图结构上传递信息"!
  VE:    信息 = 概率消息 (因子)
  BP:    信息 = 梯度 (chain rule)
```

### 链式法则 = 消息传递

```
对于复合函数 L = f(g(h(x))):

∂L/∂x = f'(g(h(x))) · g'(h(x)) · h'(x)
        └── 从输出向输入逐层传递 ──┘

     x → [h] → [g] → [f] → L
     ←─── ∂h ←── ∂g ←── ∂f ←─ 1
     
这就像 L5 的 BP 消息 — 只是消息内容是梯度而非概率!
```

---

## 概念 3：梯度下降和变体

### SGD 的统计解释

```
Full Gradient: g = ∇_θ (1/N) Σ_{i=1}^N Loss(x_i, θ)
             = 后验的精确梯度 (需要全部数据)

SGD: ĝ = ∇_θ (1/B) Σ_{i∈batch} Loss(x_i, θ)
    = 梯度的有噪但无偏估计

Var[ĝ] ∝ σ²/B  (batch size 越大, 噪声越小)
```

### 关键优化器对比

| 优化器 | 更新规则 | 关键创新 |
|--------|---------|---------|
| **SGD** | θ ← θ - η·g | 基础 |
| **Momentum** | v ← βv + g; θ ← θ - η·v | 累积历史梯度方向 → 加速收敛 |
| **RMSprop** | s ← β₂s + (1-β₂)g²; θ ← θ - η·g/√(s+ε) | 自适应学习率 per-dimension |
| **Adam** | v ← β₁v + (1-β₁)g; s ← β₂s + (1-β₂)g² | Momentum + RMSprop (王者) |

---

## 概念 4：偏差-方差权衡 — DL 的统计核心 (🔑)

### 经典分解

```
对于回归模型 f̂(x) 在点 x₀:

E[(y - f̂)²] = σ² + Bias[f̂]² + Var[f̂]
              └┬┘   └──┬──┘   └─┬─┘
            不可约  偏差²    方差
            噪声    (欠拟合)  (过拟合)

偏差: 模型平均预测与真实的差距 → 模型太简单 → 欠拟合
方差: 模型在不同训练集上的预测差异 → 模型太复杂 → 过拟合
```

### 深度学习中的偏差-方差

```
欠拟合: 增大模型, 更多层, 更宽 → 降低偏差
过拟合: 更多数据, 正则化, Dropout → 降低方差

DL 的反直觉现象 (double descent):
  参数数 >> 数据数 时, 测试误差可能又下降了!
  传统偏差-方差理论在此失效... → L12-L13 会深入
```

---

## 概念 5：正则化 — 从 PGM 的 Prior 到 DL 的技巧

```
PGM 视角:              DL 对应:
─────────────────     ─────────────
L2 正则化              Weight Decay = Gaussian prior
L1 正则化              Sparse weights = Laplace prior
Early Stopping         防止过拟合 (限制有效迭代数)
Dropout                "随机扔掉神经元" = 隐式的贝叶斯模型平均
Batch Normalization    稳定训练, 允许更大学习率
Data Augmentation      "免费"的额外数据 = 更强的先验
```

### Dropout 的贝叶斯解释

```
Dropout: 每次前向随机 mask 掉 p% 的神经元

等价于: 训练 2^N 个共享参数的子网络 (ensemble!)

在测试时: 所有神经元激活, 乘以 (1-p) → 
          近似模型平均 (≈ 贝叶斯模型平均!)
```

---

## 📋 全部概念一张表

| 概念 | 一句话 |
|------|--------|
| **MLE in DL** | 损失函数 = 负对数似然, 训练 = MLE/MAP |
| **反向传播** | 计算图上的 chain rule = 梯度消息传递 |
| **SGD** | 有噪梯度 → 收敛到 MLE (非凸情况: 局部最优) |
| **Momentum** | 累积梯度方向 → 加速, 平滑 |
| **Adam** | Momentum + 自适应学习率 → DL 的默认选择 |
| **偏差-方差** | 模型太简单→欠拟合, 太复杂→过拟合 |
| **L2/L1 Reg** | Gaussian/Laplace Prior → 权重约束 |
| **Dropout** | 随机子网络 → 隐式的 Ensemble/贝叶斯平均 |

---

## 🔗 概念关系图

```
        PGM 统计基础 (MLE, MAP, Prior, 边际)
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
    损失函数    正则化      偏差-方差
   (MLE/MAP)   (Prior)    (泛化理论)
        │           │           │
        └───────────┼───────────┘
                    │
                    ▼
            优化算法 (SGD → Adam)
                    │
                    ▼
            反向传播 (计算图 + Chain Rule)
                    │
                    ▼
              深度学习训练
```

---

## 🎯 核心洞见一句话

| # | 洞见 |
|---|------|
| 1 | **NN 训练 = MLE/MAP + 梯度下降** — 损失函数就是负对数似然 |
| 2 | **反向传播 = 计算图上的消息传递** — 和 L5 BP 共享相同的图算法哲学 |
| 3 | **SGD = 随机近似的梯度** — B 个样本给梯度的无偏但 noisy 估计 |
| 4 | **Adam = Momentum + RMSprop** — 一阶矩 (方向) + 二阶矩 (自适应步长) |
| 5 | **正则化 = Prior** — L2=Gaussian, L1=Laplace, Dropout=隐式集成 |

---

## 🧪 自测清单

- [ ] Cross-Entropy loss 对应什么统计假设?
- [ ] 反向传播和 L5 BP 在"图算法"层面有什么共同点?
- [ ] SGD, Momentum, Adam 分别更新了哪些"状态"变量?
- [ ] 偏差-方差分解中, 哪一项对应过拟合? 如何降低它?
- [ ] Dropout 在训练和测试时分别怎么工作? 为什么它有效?

---

> L11 完成了 PGM → DL 的基础桥梁。理解这些统计和算法基础后, L12-L13 将深入深度学习的核心架构 (CNN, RNN, Transformer)。
