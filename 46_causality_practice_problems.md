# CMU 10-708 Lecture 17 课后练习 & 答案 — 因果关系1

> 配套教材: Pearl (2009) *Causality*, Pearl et al. (2016) *Causal Inference in Statistics: A Primer*, Peters et al. (2017) *Elements of Causal Inference*
>
> 题目覆盖 L17 七大主题, 三级: 🟢 基础 | 🟡 进阶 | 🔴 挑战

---

# 第一部分: Causality 基础 & 辛普森悖论

---

## 🟢 Q1. 区分关联与因果

一个数据分析师发现: "带打火机的人患肺癌的概率是普通人的 3 倍"。于是他得出结论: "打火机导致肺癌"。

**(A)** 指出这个推理中的问题。画出可能的因果图。

**(B)** 如果真正的因果图是: 吸烟 → 打火机、吸烟 → 肺癌，打火机对肺癌的总因果效应是什么？

**(C)** 这个例子说明了关联和因果的什么区别？

<details>
<summary>点击查看答案</summary>

**(A) 问题:** 打火机和肺癌之间的关联是**虚假关联 (spurious correlation)**，两者都由一个共同的未观测原因——吸烟——引起。

**因果图:**
```
吸烟(S) → 打火机(L)
吸烟(S) → 肺癌(C)
打火机(L) → 肺癌(C): 无直接边!
```

即: `L ← S → C` — 经典的 Fork 结构。

**(B) 总因果效应:**

打火机 L 对肺癌 C 没有因果边 → 总因果效应 = 0。

$P(C | do(L=1)) = P(C | do(L=0)) = P(C)$

带打火机不会增加肺癌风险——前提是 smoking 不变！

**(C) 区别:**

- 关联 $P(C | L) \neq P(C)$: 带打火机的人更可能吸烟，吸烟者更可能患肺癌 → 关联 ≠ 0
- 因果 $P(C | do(L)) = P(C)$: 强制所有人带打火机不会增加肺癌率

**核心教训:** 当存在公共原因 (common cause) 时，关联 ≠ 因果。
</details>

---

## 🟡 Q2. 辛普森悖论 — 手算

某诊所评估两种肾结石治疗方案 A 和 B。数据如下:

| | 小结石 | | 大结石 | | 总计 | |
|---|---|---|---|---|---|---|
| | 成功 | 失败 | 成功 | 失败 | 成功 | 失败 |
| 方案A | 81 | 6 | 192 | 71 | 273 | 77 |
| 方案B | 234 | 36 | 55 | 25 | 289 | 61 |

**(A)** 计算总体成功率: P(成功|A) 和 P(成功|B)。哪个方案看起来更好？

**(B)** 分别计算小结石和大结石子组中的成功率。哪个方案在每组中更好？

**(C)** 解释发生反转的原因——为什么总体和分组结论不同？

**(D)** 使用后门调整公式计算 P(成功 | do(方案A)) 和 P(成功 | do(方案B))。正确的因果结论是什么？

<details>
<summary>点击查看答案</summary>

**(A) 总体成功率:**

- 方案A: 273 / (273 + 77) = 273/350 = 0.780 = 78.0%
- 方案B: 289 / (289 + 61) = 289/350 = 0.826 = 82.6%

→ 朴素结论: 方案B更好！

**(B) 分组成功率:**

- 小结石 (A): 81 / (81 + 6) = 81/87 = 0.931 = 93.1%
- 小结石 (B): 234 / (234 + 36) = 234/270 = 0.867 = 86.7%
- 大结石 (A): 192 / (192 + 71) = 192/263 = 0.730 = 73.0%
- 大结石 (B): 55 / (55 + 25) = 55/80 = 0.688 = 68.8%

→ 方案A在两种结石类型中都更好！辛普森悖论！

**(C) 反转原因:**

结石大小(S) 同时影响:
- 治疗方案(T) 的选择: 方案B 更多用于小结石 (270/350 = 77% vs 方案A的 87/350 = 25%)
- 成功率(R): 小结石天然成功率更高

$$T \leftarrow S \rightarrow R$$

S 是混杂变量。方案B 看上去好，仅仅因为更多用于"容易治"的小结石。

**(D) 后门调整:**

$$P(R | do(T=A)) = \sum_{s \in \{小,大\}} P(R | T=A, S=s) \cdot P(S=s)$$

```
P(S=小) = (87 + 270) / 700 = 357/700 = 0.51
P(S=大) = (263 + 80) / 700 = 343/700 = 0.49
```

$$P(R | do(T=A)) = 0.931 \times 0.51 + 0.730 \times 0.49$$
$$= 0.475 + 0.358 = 0.833 = 83.3\% $$

$$P(R | do(T=B)) = 0.867 \times 0.51 + 0.688 \times 0.49$$
$$= 0.442 + 0.337 = 0.779 = 77.9\% $$

**正确结论:** 方案A 更好！(ATE = +5.4%)
</details>

---

# 第二部分: Intervention & do-算子

---

## 🟢 Q3. 截断因子分解

考虑因果图:

```
X → Y → Z
↓
W
```

即边集: X→Y, Y→Z, X→W。

**(A)** 写出观测分布 P(X, Y, Z, W) 的因子分解。

**(B)** 写出干预分布 P(Y, Z, W | do(X=x)) 的截断因子分解。

**(C)** 现在考虑 do(Y=y)。写出 P(X, Z, W | do(Y=y)) 的截断因子分解。注意哪些变量保持了原始分布？

<details>
<summary>点击查看答案</summary>

**(A) 观测因子分解:**

$$P(X, Y, Z, W) = P(X) \cdot P(Y | X) \cdot P(Z | Y) \cdot P(W | X)$$

**(B) do(X=x) 截断因子分解:**

切断所有指向 X 的边（此处 X 没有父节点，无需切断），强制 X=x:

$$P(Y, Z, W | do(X=x)) = P(Y | X=x) \cdot P(Z | Y) \cdot P(W | X=x)$$

注意：P(X) 项被去掉！X 被强制设值，不再是随机变量。

**(C) do(Y=y) 截断因子分解:**

切断指向 Y 的边（即 X→Y），强制 Y=y:

$$P(X, Z, W | do(Y=y)) = P(X) \cdot P(Z | Y=y) \cdot P(W | X)$$

关键观察:
- P(X) 保持不变 — 因为切断了 Y←X，X 不再影响 Y，但 X 的边际分布不变
- P(Z|Y=y) — Z 仍依赖 Y (因果路径 Y→Z)
- P(W|X) — 不变，因为 X 和 W 不受 do(Y) 影响
</details>

---

## 🟡 Q4. 后门路径识别

给定以下因果图，列出 X 和 Y 之间所有后门路径，并给出满足后门准则的最小调节集。

**(A)**
```
Z₁ → X → Y
↑         ↑
Z₂ → Z₃ ↗
```
即: Z₁→X, Z₁→Y, Z₁←Z₂→Z₃, Z₃→Y

**(B)**
```
X ← W₁ ← U → W₂ → Y
X → M → Y
```
即: U→W₁, W₁→X, U→W₂, W₂→Y, X→M, M→Y

**(C)**
```
A → X
B → X
B → Y
C → A
C → Y
X → Y
```
即: A→X, B→X, B→Y, C→A, C→Y, X→Y

<details>
<summary>点击查看答案</summary>

**(A)**

后门路径（以指向 X 的边为起点）:
1. X ← Z₁ → Y  ✓ 后门路径 (Z₁→X 是指向 X 的边)
2. X ← Z₁ ← Z₂ → Z₃ → Y  ✓ 后门路径

注意: X → Y 不是后门路径（不始于指向 X 的边）。

最小调整集: {Z₁} 或 {Z₂, Z₃}

- {Z₁}: 阻断路径1 (条件化Z₁阻断 Fork), 阻断路径2 (条件化Z₁阻断链 Z₂→Z₁)
- {Z₂, Z₃}: 阻断路径2 (条件化Z₂阻断链 Z₂→Z₁..., 条件化Z₃阻断 Z₃→Y)

**(B)**

后门路径:
```
1. X ← W₁
          ← U → W₂ → Y  ← 错误的!
```
修改: 路径必须是连续的。正确:
1. X ← W₁ ← U → W₂ → Y  (以指向X的边 X←W₁ 开头)

最小调整集: 只需 {W₁}！

- 条件化 W₁ 阻断了 U→W₁→X 的路径 (链)
- 注意: {W₁} 阻断了整个后门路径，因为 U 和 X 在给定 W₁ 下 d-separated
- 更保守的选择: {U}, {W₂} 也有效，但 U 可能未观测
- 如果 {W₁, U, W₂} 中只有 W₁ 或 W₂ 可观测 → {W₁} 或 {W₂} 均可

**(C)**

后门路径:
1. X ← A ← C → Y  (以指向X的 X←A 开头)
2. X ← B → Y  (以指向X的 X←B 开头)
3. X ← A ← C → ? ...  B→X, B→Y 构成路径2

最小调整集: {A, B} 或 {C, B}

验证 {A, B}:
- 路径1: X←A←C→Y → 给定A阻断 (链 X←A←C...)
- 路径2: X←B→Y → 给定B阻断 (Fork)

验证 {C, B}:
- 路径1: X←A←C→Y → 给定C阻断 (Fork, 且链)
- 路径2: X←B→Y → 给定B阻断 (Fork)
</details>

---

# 第三部分: 后门准则 & 前门准则

---

## 🟡 Q5. 后门调整 — 手算数值

考虑因果图 `Z → X, Z → Y, X → Y`。已知:

- P(Z=0) = 0.4, P(Z=1) = 0.6
- P(X=1 | Z=0) = 0.3, P(X=1 | Z=1) = 0.8
- P(Y=1 | X=0, Z=0) = 0.1, P(Y=1 | X=1, Z=0) = 0.6
- P(Y=1 | X=0, Z=1) = 0.2, P(Y=1 | X=1, Z=1) = 0.9

**(A)** 计算观测关联 P(Y=1 | X=1) 和 P(Y=1 | X=0)。

**(B)** 使用后门调整计算 P(Y=1 | do(X=1)) 和 P(Y=1 | do(X=0))。

**(C)** 计算 ATE = P(Y=1 | do(X=1)) - P(Y=1 | do(X=0))。混杂偏倚是多少？

<details>
<summary>点击查看答案</summary>

**(A) 观测关联 P(Y=1 | X=x):**

由贝叶斯公式:
$$P(Y | X) = \sum_{z} P(Y | X, Z=z) \cdot P(Z=z | X)$$

先算 P(Z | X):

$$\begin{aligned}
P(X=0) &= P(X=0|Z=0)P(Z=0) + P(X=0|Z=1)P(Z=1) \\
       &= 0.7 \times 0.4 + 0.2 \times 0.6 = 0.28 + 0.12 = 0.40 \\
P(X=1) &= 0.3 \times 0.4 + 0.8 \times 0.6 = 0.12 + 0.48 = 0.60
\end{aligned}$$

$$\begin{aligned}
P(Z=0 | X=0) &= \frac{P(X=0|Z=0)P(Z=0)}{P(X=0)} = \frac{0.7 \times 0.4}{0.40} = \frac{0.28}{0.40} = 0.70 \\
P(Z=1 | X=0) &= \frac{0.2 \times 0.6}{0.40} = 0.30 \\
P(Z=0 | X=1) &= \frac{0.3 \times 0.4}{0.60} = \frac{0.12}{0.60} = 0.20 \\
P(Z=1 | X=1) &= \frac{0.8 \times 0.6}{0.60} = 0.80
\end{aligned}$$

现在:
$$\begin{aligned}
P(Y=1 | X=0) &= 0.1 \times 0.70 + 0.2 \times 0.30 = 0.07 + 0.06 = 0.13 \\
P(Y=1 | X=1) &= 0.6 \times 0.20 + 0.9 \times 0.80 = 0.12 + 0.72 = 0.84
\end{aligned}$$

观测差异 = 0.84 - 0.13 = 0.71

**(B) 后门调整:**

$$P(Y=1 | do(X=0)) = \sum_{z} P(Y=1 | X=0, Z=z) \cdot P(Z=z)$$
$$= 0.1 \times 0.4 + 0.2 \times 0.6 = 0.04 + 0.12 = 0.16$$

$$P(Y=1 | do(X=1)) = \sum_{z} P(Y=1 | X=1, Z=z) \cdot P(Z=z)$$
$$= 0.6 \times 0.4 + 0.9 \times 0.6 = 0.24 + 0.54 = 0.78$$

注意关键区别: 后门调整用 $P(Z=z)$ 而非 $P(Z=z|X=x)$！

**(C) ATE & 混杂偏倚:**

$$\begin{aligned}
\text{ATE} &= P(Y=1 | do(X=1)) - P(Y=1 | do(X=0)) \\
           &= 0.78 - 0.16 = 0.62
\end{aligned}$$

$$\begin{aligned}
\text{混杂偏倚} &= \text{观测关联} - \text{ATE} \\
               &= (0.84 - 0.13) - 0.62 = 0.71 - 0.62 = 0.09
\end{aligned}$$

Z 构成正向混杂: 观测关联夸大了因果效应。
</details>

---

## 🔴 Q6. 前门调整 — 手算数值

考虑因果图: `U → X, U → Y, X → M → Y`，其中 U 未观测。

已知以下可观测分布:
- P(X=1) = 0.5
- P(M=1 | X=0) = 0.2, P(M=1 | X=1) = 0.7
- P(Y=1 | X=0, M=0) = 0.1, P(Y=1 | X=0, M=1) = 0.4
- P(Y=1 | X=1, M=0) = 0.3, P(Y=1 | X=1, M=1) = 0.8

**(A)** 验证前门准则适用于此图（中介变量 M 满足条件）。

**(B)** 使用前门调整公式计算 P(Y=1 | do(X=1))。分步展示计算。

**(C)** 为什么后门准则在此场景下不适用？为什么前门准则有效？

<details>
<summary>点击查看答案</summary>

**(A) 前门准则验证:**

M 应满足:
1. ✓ M 截断了所有 X→Y 的因果路径 (X→M→Y)
2. ✓ X→M 没有未阻断的后门路径:
   - X ← U → M? 没有! U 不指向 M (因为 M 在 X 和 Y 之间)
   - 可能的路径: U→X→M? 这是因果路径 (front door), 不是后门
   - 所以 X→M 干净, 不调节任何变量时 d-separated
3. ✓ 所有 M→Y 的后门路径都被 X 阻断:
   - M ← X ← U → Y: 包含指向 M 的后门 (M ← X), 但给定 X 即阻断!

**(B) 前门调整计算:**

$$\begin{aligned}
P(Y=1 | do(X=1)) &= \sum_{m} P(M=m | X=1) \cdot \sum_{x'} P(Y=1 | X=x', M=m) \cdot P(X=x')
\end{aligned}$$

**Step 1:** 计算 $\sum_{x'} P(Y=1 | X=x', M=0) \cdot P(X=x')$:

$$= 0.1 \times 0.5 + 0.3 \times 0.5 = 0.05 + 0.15 = 0.20$$

$$\sum_{x'} P(Y=1 | X=x', M=1) \cdot P(X=x') = 0.4 \times 0.5 + 0.8 \times 0.5 = 0.20 + 0.40 = 0.60$$

**Step 2:** 组合:

$$\begin{aligned}
P(Y=1 | do(X=1)) &= P(M=0 | X=1) \times 0.20 + P(M=1 | X=1) \times 0.60 \\
                 &= 0.3 \times 0.20 + 0.7 \times 0.60 \\
                 &= 0.06 + 0.42 = 0.48
\end{aligned}$$

类似地, $P(Y=1 | do(X=0))$:
$$\begin{aligned}
&= 0.8 \times 0.20 + 0.2 \times 0.60 \\
&= 0.16 + 0.12 = 0.28
\end{aligned}$$

ATE = 0.48 - 0.28 = 0.20

**(C) 为什么后门不适用，前门适用?**

后门准则: 需要调节 U 来阻断 X ← U → Y，但 U 未观测！

前门准则: 不直接调节 U，而是:
- 通过 P(M|X) 利用 M  捕捉 X 对 M 的因果效应（此步没有后门）
- 通过 P(Y|X,M)·P(X) 识别 M 对 Y 的因果效应（用 X 阻断 M←X←U→Y）

**核心洞见:** 即使 U 未观测，前门调整通过中介 M 间接识别了因果效应——因为 M "位于" U 的影响范围之外（U 没有指向 M 的边）。
</details>

---

# 第四部分: do-Calculus

---

## 🔴 Q7. do-Calculus 规则应用

给定以下因果图:

```
Z₁ → X → M → Y
Z₁ → Z₂ → Y
     Z₂ → M
```

即边集: Z₁→X, Z₁→Z₂, Z₂→M, Z₂→Y, X→M, M→Y

使用 do-Calculus 推导 P(Y | do(X)) 的表达式，仅用可观测分布表示。

**(A)** 写出观测分布的因子分解。

**(B)** 要识别 P(Y | do(X))，需消除哪些后门路径？调节集选什么？

**(C)** 最终给出 P(Y | do(X=x)) 的表达式。

<details>
<summary>点击查看答案</summary>

**(A) 观测因子分解:**

$$P(Z_1, Z_2, X, M, Y) = P(Z_1) \cdot P(Z_2 | Z_1) \cdot P(X | Z_1) \cdot P(M | Z_2, X) \cdot P(Y | Z_2, M)$$

**(B) 后门路径分析:**

X 和 Y 之间的后门路径（始于指向 X 的边）:
1. X ← Z₁ → Z₂ → M → Y
2. X ← Z₁ → Z₂ → Y

两条路径都经过 Z₁。调节 Z₁ 阻断两条路径。

Z₁ 满足后门准则:
- ✓ 不是 X 的后代
- ✓ d-separates 所有后门路径

**(C) 推导:**

**Step 1:** 使用后门准则（do-Calculus 规则 2 的特例）:
$$P(Y | do(X=x)) = \sum_{z_1} P(Y | X=x, Z_1=z_1) \cdot P(Z_1=z_1)$$

但这仍然包含 $P(Y|X=x, Z_1=z_1)$，需要进一步展开。

**Step 2:** 展开 Y 的分布（利用因子分解）:

$$\begin{aligned}
P(Y | X=x, Z_1=z_1) &= \sum_{z_2, m} P(Y | Z_2=z_2, M=m) \cdot P(M=m | Z_2=z_2, X=x) \\
&\quad \cdot P(Z_2=z_2 | Z_1=z_1)
\end{aligned}$$

**Step 3:** 代入:

$$\boxed{P(Y | do(X=x)) = \sum_{z_1} P(Z_1=z_1) \sum_{z_2} P(Z_2=z_2 | Z_1=z_1) \sum_{m} P(M=m | Z_2=z_2, X=x) \cdot P(Y | Z_2=z_2, M=m)}$$

等价于:
$$\boxed{P(Y | do(X=x)) = \sum_{z_1, z_2, m} P(Z_1=z_1) \cdot P(Z_2=z_2 | Z_1=z_1) \cdot P(M=m | Z_2=z_2, X=x) \cdot P(Y | Z_2=z_2, M=m)}$$

验证: 所有项都可从 P(V) 计算 → 因果效应可识别 ✓

注意，更简洁的写法（直接用后门准则）:
$$P(y | do(x)) = \sum_{z_1} P(y | x, z_1) \cdot P(z_1)$$

其中:
$$P(y | x, z_1) = \sum_{z_2, m} P(z_2 | z_1) \cdot P(m | z_2, x) \cdot P(y | z_2, m)$$
</details>

---

# 第五部分: 反事实推理

---

## 🟡 Q8. 线性 SCM 反事实

考虑线性 SCM:

$$\begin{aligned}
X &= U_X \\
Y &= 2X + U_Y
\end{aligned}$$

其中 $U_X \sim N(0, 1)$, $U_Y \sim N(0, 1)$, $U_X \perp U_Y$。

**(A)** 观测到个体 Alice: X=1, Y=3。对Alice, 计算反事实 $E[Y_{X=0} | X=1, Y=3]$ — "如果当时 X=0, Y 会是多少？"

**(B)** 同样对 Alice，计算 $E[Y_{X=-1} | X=1, Y=3]$。

**(C)** 在总体层面，计算 $E[Y_{X=0}]$ (即 ATE)。为什么这个和 Alice 的个体反事实不同？

**(D)** 假设我们不知道 Alice 的 U_Y，只知道她 X=1。此时 $E[Y_{X=0} | X=1]$ 是多少？

<details>
<summary>点击查看答案</summary>

**(A) Alice 的反事实:**

**Step 1 — Abduction (溯因):**

$$\begin{aligned}
U_X &= X = 1 \\
U_Y &= Y - 2X = 3 - 2 \times 1 = 1
\end{aligned}$$

**Step 2 — Action (行动):**

新模型: $X = 0$ (强制), $Y = 2 \times 0 + U_Y = 0 + U_Y$

**Step 3 — Prediction (预测):**

$$\begin{aligned}
E[Y_{X=0} | X=1, Y=3] &= 2 \times 0 + E[U_Y | X=1, Y=3] \\
&= 0 + 1 = 1
\end{aligned}$$

Alice 如果 X=0，Y 会是 1（而非观测到的 3）。

**(B)** 同样:

$$\begin{aligned}
E[Y_{X=-1} | X=1, Y=3] &= 2 \times (-1) + 1 = -1
\end{aligned}$$

**(C)** 总体 ATE:

$$\begin{aligned}
E[Y_{X=0}] &= E[2 \times 0 + U_Y] = E[U_Y] = 0
\end{aligned}$$

$$\begin{aligned}
\text{ATE} &= E[Y_{X=1}] - E[Y_{X=0}] = (2 \times 1 + 0) - 0 = 2
\end{aligned}$$

为什么不同？
- Alice 的 $E[Y_{X=0} | X=1, Y=3] = 1$: 利用了她的个体信息(U_Y=1)
- 总体 $E[Y_{X=0}] = 0$: 对全体的平均 (E[U_Y]=0)
- Alice 的 U_Y=1 > 平均水平 → 她的反事实结果高于总体均值

**(D)** 只知道 X=1:

$$\begin{aligned}
E[Y_{X=0} | X=1] &= 2 \times 0 + E[U_Y | X=1] \\
&= 0 + E[U_Y] \quad (\text{因为 } U_Y \perp X \text{ — 但注意 } X=U_X, U_X \perp U_Y)\\
&= 0
\end{aligned}$$

因为 X 和 U_Y 独立（没有混杂），知道 X=1 不提供关于 U_Y 的信息。所以反事实预期 = 总体平均。
</details>

---

## 🔴 Q9. 归因概率 — PN, PS, PNS

一个医学研究评估药物(T)对康复(R)的效果。SCM 为:

$$\begin{aligned}
T &= \mathbb{1}[U_T > 0.5], \quad U_T \sim \text{Uniform}(0, 1) \\
R &= \mathbb{1}[3T + U_R > 0.3], \quad U_R \sim N(0, 2), \quad U_T \perp U_R
\end{aligned}$$

**(A)** 计算总体概率 P(R=1) 和 P(R=1 | T=1), P(R=1 | T=0)。

**(B)** 计算 P(R=1 | do(T=1)) 和 P(R=1 | do(T=0))。ATE 是多少？

**(C)** 计算 Probability of Necessity (PN):
$$PN = P(R_{T=0} = 0 \mid T=1, R=1)$$
即：在服药且康复的个体中，如果不服药，有多少概率不会康复？

<details>
<summary>点击查看答案</summary>

**(A) 总体概率:**

$$\begin{aligned}
P(T=1) &= P(U_T > 0.5) = 0.5 \\
P(T=0) &= 0.5
\end{aligned}$$

$$\begin{aligned}
P(R=1 | T=1) &= P(3 \times 1 + U_R > 0.3) = P(U_R > -2.7) \\
&= \Phi(2.7/2) = \Phi(1.35) \approx 0.9115
\end{aligned}$$

$$\begin{aligned}
P(R=1 | T=0) &= P(3 \times 0 + U_R > 0.3) = P(U_R > 0.3) \\
&= 1 - \Phi(0.3/2) = 1 - \Phi(0.15) \approx 1 - 0.5596 = 0.4404
\end{aligned}$$

$$\begin{aligned}
P(R=1) &= P(R=1|T=1)P(T=1) + P(R=1|T=0)P(T=0) \\
&= 0.9115 \times 0.5 + 0.4404 \times 0.5 \approx 0.6760
\end{aligned}$$

**(B) 因果效应:**

由于 $U_T \perp U_R$ （没有混杂——T 的外生变量独立于 R 的外生变量），观测关联 = 因果效应。

验证:
$$P(R=1 | do(T=1)) = P(3 + U_R > 0.3) = P(U_R > -2.7) = 0.9115$$
$$P(R=1 | do(T=0)) = P(U_R > 0.3) = 0.4404$$
$$\text{ATE} = 0.9115 - 0.4404 = 0.4711$$

**(C) PN:**

PN 指: 在 T=1, R=1 的个体中，如果 T 被设为 0，R 变成 0 的概率。

$$\begin{aligned}
PN &= P(R_{T=0} = 0 \mid T=1, R=1) \\
   &= 1 - P(R_{T=0} = 1 \mid T=1, R=1)
\end{aligned}$$

我们需要:
$$P(R_{T=0} = 1 \mid T=1, R=1) = \frac{P(T=1, R=1, R_{T=0}=1)}{P(T=1, R=1)}$$

由于 $U_T \perp U_R$:
$$\begin{aligned}
P(T=1, R=1, R_{T=0}=1) &= P(U_T > 0.5, 3+U_R > 0.3, U_R > 0.3) \\
&= P(U_T > 0.5) \cdot P(U_R > 0.3) \quad (\text{因为 } U_R > 0.3 \Rightarrow 3+U_R > 0.3)\\
&= 0.5 \times 0.4404 = 0.2202
\end{aligned}$$

$$\begin{aligned}
P(T=1, R=1) &= P(U_T > 0.5, 3+U_R > 0.3) \\
&= P(U_T > 0.5) \cdot P(U_R > -2.7) \\
&= 0.5 \times 0.9115 = 0.4558
\end{aligned}$$

$$\begin{aligned}
PN &= 1 - \frac{0.2202}{0.4558} = 1 - 0.4831 = 0.5169
\end{aligned}$$

**解释:** 在服药且康复的个体中，约 51.7% 的人如果不服药就不会康复——这证明了药物的必要性。
</details>

---

# 第六部分: 因果发现

---

## 🟡 Q10. PC 算法手算

给定以下来自某未知 DAG 的条件独立性：

| 条件独立关系 |
|---|
| $X_1 \perp X_3 \mid X_2$ |
| $X_1 \perp X_4 \mid \{X_2, X_3\}$ |
| $X_2 \perp X_4 \mid X_3$ |
| （所有其他 CI 都来自这些的闭合） |

**(A)** 应用 PC 算法步骤，从完全无向图开始，逐步删除边。记录每个 CI 如何删除对应边。

**(B)** 给出骨架（无向图）。识别所有 v-结构。

**(C)** 画出 CPDAG。

<details>
<summary>点击查看答案</summary>

**(A) 边删除过程:**

**初始化:** 4个节点，完全图（6条边）。

**d=0 (条件集大小 0):**

| 节点对 | 无条件独立? | 动作 |
|--------|------------|------|
| (1,2) | 无CI说 | 保留 |
| (1,3) | $X_1 \perp X_3$ 无条件? 没有说。需要条件化 $X_2$ | 保留 (在d=1测试) |
| (1,4) | 无CI说 | 保留 |
| (2,3) | 无CI说 | 保留 |
| (2,4) | 无CI说 | 保留 |
| (3,4) | 无CI说 | 保留 |

**d=1 (条件集大小 1):**

| 节点对 | 条件独立? | SepSet | 动作 |
|--------|----------|--------|------|
| (1,2) | $X_1 \perp X_2$? 未给定 | — | 保留 |
| (1,3) | $X_1 \perp X_3 \mid X_2$ ✓ | {2} | **删除 (1,3)** |
| (1,4) | $X_1 \perp X_4 \mid X_2$? 未给定 | — | 保留 |
| (2,3) | $X_2 \perp X_3$? 未给定 | — | 保留 |
| (2,4) | $X_2 \perp X_4 \mid X_3$ ✓ | {3} | **删除 (2,4)** |
| (3,4) | $X_3 \perp X_4$? 未给定 | — | 保留 |

现在剩下边: (1,2), (1,4), (2,3), (3,4)

**d=2 (条件集大小 2):**

| 节点对 | 条件独立? | SepSet | 动作 |
|--------|----------|--------|------|
| (1,4) | $X_1 \perp X_4 \mid \{X_2, X_3\}$ ✓ | {2,3} | **删除 (1,4)** |

剩余边: (1,2), (2,3), (3,4) ✓

**(B) 骨架:**

```
X₁ — X₂ — X₃ — X₄  (链式结构!)
```

这是一个 4-节点链。

**v-结构识别:**

检查所有未连接的三元组:
- (X₁, X₃)不连接, 共享邻居 X₂。
  SepSet(1,3) = {X₂}。X₂ ∈ SepSet → **不是** v-结构。
  定向: X₁ — X₂ — X₃ (不能定向为 collider)

- (X₁, X₄)不连接，共享邻居 X₂, X₃。
  SepSet(1,4) = {X₂, X₃}。X₂ ∈ SepSet 且 X₃ ∈ SepSet。
  但这不直接对应单个 v-结构（因为 X₁ 和 X₄ 有两个中间节点）。

- (X₂, X₄)不连接，共享邻居 X₃。
  SepSet(2,4) = {X₃}。X₃ ∈ SepSet → **不是** v-结构。

**结论:** 没有 v-结构！骨架完全无向。

**(C) CPDAG:**

$$
X_1 — X_2 — X_3 — X_4
$$

全部无向边 —— 这是一个马尔可夫等价类，包含所有链式 DAG（如 X₁→X₂→X₃→X₄, X₁←X₂←X₃←X₄, X₁←X₂→X₃→X₄ 等）。

等价类中的 DAG: 所有 3 条边的方向都可以取任何值，只要不形成 v-结构或 cycle。这给出了该链式骨架的 8 种 DAG——但其中一些可能违反 acyclicity，实际有 $2^3 - 2 = 6$ 种（只有 2 种形成 cycle: 全部顺时针和全部逆时针）。

更准确地说，链式骨架等价类中的 CPDAG:
- 所有边保持无向（因为没有 v-结构可定向）

这在给定 CI 约束下是正确的：仅观测数据无法确定链式结构中边的方向！
</details>

---

# 第七部分: 因果对 ML 的启示

---

## 🟢 Q11. 半监督学习的因果分析

考虑两个数据集，生成过程不同：

**数据集 A (因果方向):** 图片 → 标签
$$Label = f(Image) + noise$$

**数据集 B (反因果方向):** 标签 → 图片
$$Image = g(Label) + noise$$

**(A)** 在半监督学习中（大量无标签图片，少量有标签），哪个数据集的 SSL 会更有帮助？为什么？（从因果角度分析）

**(B)** 在数据集 A 中，如果训练集和测试集的 P(Image) 不同（如训练时是室内照片，测试时是室外照片），哪个模型更可能泛化：学 P(Label | Image) 的，还是学 P(Image | Label) 的？

<details>
<summary>点击查看答案</summary>

**(A) SSL 有效性:**

- **数据集 A (因果方向 Image → Label):**
  机制: $P(Label | Image, \theta_{L|I})$ 和 $P(Image)$ 的"参数独立" (modularity)。
  无标签数据 $P(Image)$ → 不提供关于 $P(Label | Image)$ 的信息。
  → SSL 通常**无帮助**！

- **数据集 B (反因果方向 Label → Image):**
  机制: $P(Image | Label)$ 和 $P(Label)$ 耦合。
  无标签数据 $P(Image)$ = $\sum_{L} P(Image | L) \cdot P(L)$ 包含关于 $P(L)$ 的信息。
  而 $P(L | Image) \propto P(Image | L) \cdot P(L)$。
  → SSL 可能**有帮助**！

这就是为什么图像分类 (反因果: 类别→像素) 中 SSL 通常有效，
但在因果预测任务中 SSL 可能无效。

**(B) OOD 泛化:**

- **P(Label | Image) — 因果模型:**
  因果方向机制在不同环境下**不变**。
  即使 P(Image) 从室内变为室外，$f(Image)$ 这个函数不变。
  → 因果模型泛化更好 ✓

- **P(Image | Label) — 反因果模型:**
  反因果方向机制依赖 P(Label)，在不同环境下可能改变。
  → 泛化能力差 ✗

**结论:** 当面临分布偏移时，学习因果方向的模型（P(effect | cause)）比反因果方向的模型更稳健。这解释了为什么在实践中，直接预测 $P(Y | X)$ 的分类器在 i.i.d. 假设下工作，但在 OOD 场景下可能失败——如果数据生成是反因果的话，需要学习因果表示。
</details>

---

## 参考

- Pearl, J. (2009). *Causality* (2nd ed.). Cambridge.
- Pearl, J., Glymour, M., & Jewell, N. P. (2016). *Causal Inference in Statistics: A Primer*. Wiley.
- Peters, J., Janzing, D., & Schölkopf, B. (2017). *Elements of Causal Inference*. MIT Press.
