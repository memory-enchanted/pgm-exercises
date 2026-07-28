# CMU 10-708 Lecture 12 课后练习 & 答案

> 配套教材: Goodfellow et al. (2016) Ch.20, Hinton (2002, 2006), Salakhutdinov (2015)
>
> 题目分为三级: 🟢 基础 | 🟡 进阶 | 🔴 挑战

---

## 🟢 基础题 (必须掌握)

### Q1. RBM 的能量与条件分布

对一个 3 可见、2 隐藏的 RBM: E(v, h) = -v^T W h - a^T v - b^T h。

给定 W = [[2, -1], [1, 2], [-1, 3]], a = [0.1, -0.2, 0.0], b = [0.5, -0.3],

**(A)** 计算配置 v=(1,0,1), h=(1,1) 的能量 E(v,h)。

**(B)** 计算 P(h₁=1 | v=(1,0,1)) 和 P(v₂=1 | h=(1,1))。

<details>
<summary>点击查看答案</summary>

**(A) 能量:**

```
E(v,h) = -v^T W h - a^T v - b^T h

v^T W = [1,0,1] · [[2,-1],[1,2],[-1,3]] = [2-1, -1+3] = [1, 2]

v^T W h = [1,2] · [1,1] = 3

a^T v = 0.1×1 + (-0.2)×0 + 0.0×1 = 0.1
b^T h = 0.5×1 + (-0.3)×1 = 0.2

E = -3 - 0.1 - 0.2 = -3.3
```

**(B) 条件概率:**

P(h₁=1|v) = σ(b₁ + Σ_i W_{i1} v_i):
```
b₁ + W_{:,1}^T v = 0.5 + 2×1 + 1×0 + (-1)×1 = 0.5 + 2 - 1 = 1.5
P(h₁=1|v) = σ(1.5) = 1/(1+e^{-1.5}) = 0.8176
```

P(v₂=1|h) = σ(a₂ + Σ_j W_{2j} h_j):
```
a₂ + W_{2,:} h = -0.2 + 1×1 + 2×1 = -0.2 + 3 = 2.8
P(v₂=1|h) = σ(2.8) = 1/(1+e^{-2.8}) = 0.9427
```

</details>

---

### Q2. CD-k 的梯度公式

**(A)** 写出 RBM 的 ∂ log P(v)/∂W_{ij} 的精确公式 (涉及 model expectation)。

**(B)** 写出 CD-1 的近似梯度。为什么 CD-1 用 P(h_j=1|v) 而非采样?

<details>
<summary>点击查看答案</summary>

**(A) 精确梯度:**

```
∂ log P(v) / ∂W_{ij} = E_{P(h|v)}[v_i h_j] - E_{P(v',h')}[v'_i h'_j]
                      = v_i · P(h_j=1|v)  -  Σ_{v',h'} v'_i h'_j P(v',h')
```

第一项 (正相位): 在给定数据 v 下, 条件期望 — 闭式可得。
第二项 (负相位): 在模型联合分布下的期望 — 需要从 P(v,h) 采样 (无限步 Gibbs)。

**(B) CD-1 近似:**

负相位: 从数据 v^(0) 出发, 做 1 步 Gibbs, 用结果 (v^(1), h^(1)) 近似 model expectation:

```
ΔW_{ij} ≈ v_i^(0)·P(h_j=1|v^(0)) - v_i^(1)·P(h_j=1|v^(1))
```

**用概率 P(h|v) 而非采样**: 这是"mean-field"近似 → 降低方差 (不引入采样噪声)。实践中常这样做, 但严格来说会引入小的偏差。

</details>

---

### Q3. Block Gibbs 的效率

**(A)** 为什么 RBM 可以用"Block Gibbs"(一次更新整层) 而不是逐个节点 Gibbs?

**(B)** Block Gibbs 在 RBM 上每步的采样复杂度是多少?

<details>
<summary>点击查看答案</summary>

**(A) 条件独立性:**

RBM 是二部图:
- 给定 v, h 之间没有边 → h_j ⟂ h_k | v → P(h|v) = ∏_j P(h_j|v)
- 给定 h, v 之间没有边 → v_i ⟂ v_k | h → P(v|h) = ∏_i P(v_i|h)

因此可以同时(并行)采样整层的所有节点 → Block Gibbs。

**(B) 复杂度:**

```
P(h|v): O(N_vis × N_hid)  ← 矩阵乘法 v @ W
P(v|h): O(N_hid × N_vis)  ← 矩阵乘法 h @ W^T

总计: O(N_vis × N_hid) per Gibbs step
```

非常高效! 复杂度只与网络大小线性相关。

</details>

---

### Q4. DBN 的 Layer-wise Pretraining

**(A)** 列出 DBN greedy layer-wise pretraining 的步骤。

**(B)** 这种训练方式的直觉是什么? 为什么"逐层训练"有意义?

<details>
<summary>点击查看答案</summary>

**(A) 步骤:**

```
Step 1: RBM1(v, h1) 在原始数据上训练 → 得到 W1
Step 2: 用 h1 = sigmoid(v @ W1 + b1) 作为"数据" → 训练 RBM2(h1, h2) → 得到 W2
Step 3: 重复, 训练 RBM3(h2, h3)...
Step 4 (可选): 联合 fine-tune (wake-sleep, BP)

最终: DBN 的 bottom-up 权重 = 各 RBM 的 W_k (或转置)
```

**(B) 直觉:**

每层 RBM 学习"当前输入的最佳表示":
- Layer 1: 学习像素 → edges 的映射
- Layer 2: 学习 edges → object parts 的映射
- Layer 3: 学习 parts → objects 的映射

逐层训练的意义: 直接从像素到 objects 太难 (高度非线性), 分步学更容易。

类比: 先学加法, 再学乘法, 最后学微积分 → 逐步建立抽象。

</details>

---

## 🟡 进阶题

### Q5. CD 的偏差来源

**(A)** CD-k 不是 MLE 的无偏估计。偏差来自哪里?

**(B)** 为什么实践中 CD-1 就常常足够? 在什么情况下 CD-1 会失败?

<details>
<summary>点击查看答案</summary>

**(A) 偏差来源:**

精确 MLE: 负相位用 P_model(v,h) 下的期望 → 需要从稳态分布采样。

CD-k: 从数据 v^(0) 出发, 跑 k 步 Gibbs → v^(k) 的分布**不是** P_model, 而是数据分布的"受扰版本"。

偏差 = CD 用"data distribution after k Gibbs updates"替代了真正的"model stationary distribution"。

**(B) CD-1 有效/失败的条件:**

**有效**: 当数据分布 P_data 与模型 P_model 接近时 → 1 步 Gibbs 就能近似 model distribution。训练初期 (随机 W) P_model ≈ uniform → 远离数据 → CD-1 可能差, 但随着训练, P_model 逐步趋近数据 → CD-1 越来越好。

**失败**:
- 数据分布多峰且 mode 间距大: 1 步 Gibbs 无法从一个 mode 穿越到另一个
- 模型和数据极不匹配时
- 高维数据: 1 步 Gibbs 混合不充分

缓解: 使用 PCD (Persistent CD) — 维护 persistent chains, 每次从上一轮的 v 出发 (而非数据), 等效于用更多 Gibbs 步。

</details>

---

### Q6. Partition Function 为什么难? 怎么绕开?

**(A)** 对于有 N 个二值可见单元的 RBM, Z 的精确计算需要多少项求和?

**(B)** CD 和 Score Matching 分别如何绕开 Z 的计算?

<details>
<summary>点击查看答案</summary>

**(A) Z 的计算:**

```
Z = Σ_v Σ_h exp(-E(v,h))

v 有 2^N 种配置, h 有 2^M 种配置 → 总共 2^{N+M} 项!

当 N=784 (MNIST), M=500: 2^1284 ≈ 10^386 → 宇宙中的原子数 ~10^80
```

完全不可行。

**(B) 绕开 Z 的方法:**

**CD**: 不直接计算 Z, 而是用采样近似梯度。梯度依赖于 P(h|v) 和 P(v|h) — 这些条件分布不涉及 Z!

```
P(h|v) = P(v,h)/P(v) ∝ exp(-E(v,h))  ← Z 约掉了!
```

**Score Matching**: 匹配的是 score = ∇_x log P(x), 而 ∇_x log P(x) = -∇_x E(x) — 也不涉及 Z!

```
log P(x) = -E(x) - log Z
∇_x log P(x) = -∇_x E(x)  ← Z 消失了! (log Z 对 x 的导数为 0)
```

两种方法都巧妙地避开了 Z 的计算。

</details>

---

### Q7. RBM → DBN 的权重绑定

在 DBN 中, top-down 生成权重和 bottom-up 识别权重可以**绑定** (tied weights): W_topdown = W_bottomup^T。

**(A)** 解释这种绑定在 PGM 视角下是什么意思。

**(B)** 为什么它有助于正则化?

<details>
<summary>点击查看答案</summary>

**(A) PGM 视角:**

在 DBN 的顶层 RBM 中, W 同时用于:
- 正向 (bottom-up): P(h|v) = σ(vW + b)
- 反向 (top-down): P(v|h) = σ(hW^T + a)

绑定意味着: 如果没有绑定, RBM 可以有不同的 W_in 和 W_out → 但这违背了 RBM 的无向定义 (能量模型中权重是对称的)。

在 RBM 中, W 必须是同一个矩阵 — 因为 E(v,h) = -v^T W h 要求 W 同时出现在两个方向。

**(B) 正则化效果:**

绑定 = 参数减半 → 降低过拟合风险。

此外, 绑定确保了"识别"和"生成"使用相同的特征:
- 识别: v → h 用 W
- 生成: h → v 用 W^T

保证了表征的一致性 → "识别到的特征" 和 "生成时用的特征"是同一套。

</details>

---

## 🔴 挑战题

### Q8. Persistent CD (PCD)

**(A)** PCD 与 CD-k 有什么区别? 画出两种方法的负相位采样过程。

**(B)** PCD 在什么情况下优于 CD? 什么时候 CD 反而更好?

<details>
<summary>点击查看答案</summary>

**(A) 区别:**

**CD-k**: 每次从数据 v^(0) 出发 → k 步 Gibbs → v^(k) 用于负相位。

**PCD (Persistent CD)**: 维护一组"持久链" (persistent chains)。每次训练:
- 从上一次负相位的 v_neg 出发 (而非数据)
- 跑 k 步 Gibbs → 新的 v_neg 用于负相位
- 保存 v_neg 供下一次使用

```
CD-k:           PCD:
v_data → ...    v_persist (上次的) → ...
   ↓k steps          ↓k steps
v_neg (丢弃)    v_neg → 保存为下次起点
```

**(B) 优劣势:**

**PCD 更好**:
- 当模型分布多峰: persistent chains 在 mode 间缓慢移动 → 更好地探索整个 model distribution
- 对高维数据: CD-1 的 1 步 Gibbs 混合极不充分, PCD 累积了多步混合

**CD 更好**:
- 当数据分布快速变化 (如 online learning): persistent chains 可能滞后于模型变化
- 初始化简单: CD 总是从数据出发, 不担心 persistent chains 的"陈旧"问题

实践中: PCD (或 Fast PCD) 通常优于 CD, 但需要小心调参。

</details>

---

### Q9. RBM 的 Free Energy 与异常检测

**(A)** RBM 的 Free Energy: F(v) = -log Σ_h exp(-E(v,h))。写出它的计算复杂度。

**(B)** 为什么 F(v) 可以用于异常检测? 低 F(v) 意味着什么?

<details>
<summary>点击查看答案</summary>

**(A) Free Energy:**

```
F(v) = -a^T v - Σ_j softplus(b_j + Σ_i W_{ij} v_i)

其中 softplus(x) = log(1 + e^x)

计算复杂度: O(N_vis × N_hid) — 无需对 h 求和!
```

关键: 由于条件独立 P(h|v) = ∏ P(h_j|v), 对 h 的求和可以因子分解:

```
Σ_h exp(Σ_j (b_j + Σ_i W_{ij} v_i) h_j)
= Π_j Σ_{h_j∈{0,1}} exp((b_j + Σ_i W_{ij} v_i) h_j)
= Π_j (1 + exp(b_j + Σ_i W_{ij} v_i))
```

取 log 得 softplus。这个因子分解使本来 O(2^M) 的求和变成 O(M)!

**(B) 异常检测:**

F(v) = -log P(v) + const (忽略 log Z)

低 F(v) ↔ 高 P(v) ↔ 数据"正常" (模型认为概率高)
高 F(v) ↔ 低 P(v) ↔ 数据"异常" (模型认为概率低)

实际操作:
1. 在正常数据上训练 RBM
2. 对测试数据计算 F(v)
3. 设定阈值: F(v) > threshold → 异常

应用: 欺诈检测, 工业异常检测, 网络安全。

</details>

---

### Q10. RBM 在协同过滤中的应用 (Netflix Prize)

**(A)** 如何把 RBM 用于推荐系统中的协同过滤 (user × item matrix)?

**(B)** 解释: 为什么 RBM 在这个任务上有效? (Hint: 与矩阵分解的对比)

<details>
<summary>点击查看答案</summary>

**(A) RBM for Collaborative Filtering:**

- 每个用户是一个 RBM 实例: 可见单元 = 各物品的评分 (1-5 分, softmax 单元)
- 隐藏单元学习"用户偏好类型"(喜欢动作片? 喜欢文艺片?)
- 所有权重在所有用户间**共享**

训练: 对每个用户 u, 用该用户评过的物品训练 RBM (用 CD)

预测: 对用户 u 的未评物品, 固定已评物品为观测, 用 Gibbs 采样预测评分分布。

**(B) 为什么有效:**

矩阵分解 (SVD): 线性, 只能捕捉用户的"一个偏好方向"

RBM: 非线性, 可以学习用户的**多种**偏好组合 — 每个隐藏单元可以独立激活 (不像 SVD 的 embedding 是互斥的):
- h₁=1: "喜欢动作+不喜欢文艺"
- h₂=1: "喜欢新片+不喜欢老片"
- h₁=1, h₂=1: "喜欢动作新片, 不喜欢文艺老片" (AND combination)

这就是 RBM 对比 SVD 的核心优势 — 隐藏单元的**组合**可以表达指数级多的偏好模式。

Netflix Prize: RBM-based 方法曾是 top performer (Salakhutdinov et al. 2007)。

</details>

---

## 📊 综合自测评分

每题 10 分，共 100 分。

| 得分 | 评价 |
|------|------|
| 90-100 | L12 完全掌握, 已理解 RBM/DBN 的原理和训练 |
| 70-89  | 主干扎实, 建议亲手实现一个 RBM + CD |
| 50-69  | 概念清晰, 回去推一遍能量函数和条件分布的推导 |
| < 50   | 先吃透 Q1-Q4, 确保理解 RBM 结构和 CD 原理 |

---

> L12 是"第一代"深度生成模型。掌握 RBM 后, L13 将进入"第二代": VAE (回顾 L8) 和 GAN — 两者都不需要 energy-based 建模。
