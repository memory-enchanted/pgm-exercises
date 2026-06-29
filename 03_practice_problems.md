# CMU 10-708 L1-L3: 课后练习题集 (附答案)

> 配套视频: [BV1tX4y1371G](https://www.bilibili.com/video/BV1tX4y1371G/) CMU 10-708 概率图模型
>
> 覆盖范围: L1 (导论), L2 (有向图模型/贝叶斯网络), L3 (无向图模型/马尔可夫随机场)
>
> 建议: **先自己做, 再看答案。** 每题后面标注的 ⭐ 表示难度。

---

## 第 1 部分: 基础概念 (L1-L2)

### Q1: 图模型的定义 ⭐

以下哪些是关于概率图模型的正确描述？(多选)

A. 图模型用图的节点表示随机变量
B. 图模型用边表示变量之间的概率依赖关系
C. 每个图模型都唯一对应一个概率分布
D. 概率图模型的核心思想是用图结构来编码条件独立性
E. 有向图模型一定比无向图模型表达能力更强

<details>
<summary>点击查看答案</summary>

**正确答案: A, B, D**

- A ✅ — 节点 = 随机变量, 这是图模型的基本设定
- B ✅ — 边编码了变量之间的依赖关系
- C ❌ — 一个图结构可以对应无穷多个概率分布 (只要分布满足图编码的独立性即可)
- D ✅ — 这正是 PGM 的核心思想: 用图这种直观的方式编码 CI 关系
- E ❌ — 有向图和无向图各有能/不能编码的独立性集合, 不存在一方完全"更强"
</details>

---

### Q2: 因子分解 ⭐⭐

给定以下贝叶斯网络结构:

```
    A ──→ C ←── B
          │
          ↓
          D
```

写出 P(A, B, C, D) 的因子分解形式。

<details>
<summary>点击查看答案</summary>

根据贝叶斯网络的因子分解规则:

```
P(A, B, C, D) = P(A) × P(B) × P(C | A, B) × P(D | C)
```

规则: 每个变量的条件概率只以它的父节点为条件:
- Pa(A) = ∅ → P(A)
- Pa(B) = ∅ → P(B)
- Pa(C) = {A, B} → P(C | A, B)
- Pa(D) = {C} → P(D | C)

</details>

---

### Q3: 因子分解的参数数量 ⭐⭐

一个包含 n 个二值随机变量的完全连接 DAG (每对节点之间都有边, 且按拓扑序排列), 其联合分布需要多少个参数？

(A) 2^n 个 &emsp; (B) 2^n - 1 个 &emsp; (C) n·2^n 个 &emsp; (D) 2^{n+1} 个

<details>
<summary>点击查看答案</summary>

**正确答案: B (2^n - 1)**

分析: 完全连接 DAG 的因子分解不会减少任何参数。n 个二值变量的联合分布有 2^n 个条目, 但概率之和为 1, 所以独立参数为 2^n - 1。

如果没有图结构 (即完全连接 DAG), 参数数量随 n 指数增长 → 这就是为什么需要稀疏的图结构!
</details>

---

## 第 2 部分: d-分离 (L2 重点)

### Q4: 单路径 d-分离判断 ⭐⭐

判断以下每条路径在给定 Z 时是活跃 (active) 还是阻塞 (blocked):

| 路径 | Z | 活跃/阻塞? |
|------|---|-----------|
| (a) X → M → Y | {M} | ? |
| (b) X ← M → Y | {M} | ? |
| (c) X → M ← Y | {M} | ? |
| (d) X → M ← Y | ∅ | ? |
| (e) X → M ← Y | {N} (N 是 M 的子节点) | ? |
| (f) X → M → N → Y | {M} | ? |
| (g) X → M ← W → Y | {M} | ? |

<details>
<summary>点击查看答案</summary>

| 路径 | Z | 活跃/阻塞 | 解释 |
|------|---|----------|------|
| (a) X → M → Y | {M} | **阻塞** | 链式, 中间节点被观测 |
| (b) X ← M → Y | {M} | **阻塞** | 分叉, 共同原因被观测 |
| (c) X → M ← Y | {M} | **活跃** 🔥 | Collider 被观测, 路径激活! |
| (d) X → M ← Y | ∅ | **阻塞** | Collider 未观测, 默认阻塞 |
| (e) X → M ← Y | {N} | **活跃** 🔥 | Collider 的后代被观测! |
| (f) X → M → N → Y | {M} | **阻塞** | 链式中第一个中间节点就阻塞了 |
| (g) X → M ← W → Y | {M} | **活跃** 🔥 | Collider M 被观测激活; W→Y 那半边的分叉结构也被激活 |

</details>

---

### Q5: 学生网络 d-分离 ⭐⭐⭐

回到经典的学生网络:

```
        Difficulty ──→ Grade ←── Intelligence
                            │           │
                            ↓           ↓
                           Letter      SAT
```

判断以下陈述的真假:

(a) Difficulty ⟂ Intelligence  (无条件)
(b) Difficulty ⟂ Intelligence | Grade
(c) Difficulty ⟂ SAT
(d) Difficulty ⟂ SAT | Grade
(e) Intelligence ⟂ Letter | Grade
(f) SAT ⟂ Letter
(g) SAT ⟂ Letter | Intelligence

<details>
<summary>点击查看答案</summary>

(a) **真** ✅ — D→G←I, collider G 未观测 → 路径阻塞 → D ⟂ I

(b) **假** ❌ — 观测 collider G → D→G←I 被激活! 这是经典的"解释消除" (explaining away)

(c) **真** ✅ — 唯一路径: D→G←I→SAT。G 是 collider 且未观测 → 阻塞 → D ⟂ SAT

(d) **假** ❌ — 观测 G 后 collider 被激活, 整条路径 D→G←I→SAT 变活跃

(e) **真** ✅ — 唯一路径: I→G→L。G 被观测 → 链式阻塞 → I ⟂ L | G

(f) **假** ❌ — 唯一路径: S←I→G→L。无观测 → 所有节点都是链式/分叉且未阻塞 → 活跃

(g) **真** ✅ — S←I 和 I→G→L, I 被观测 → S←I→G→L 在 I 处被阻断

> 💡 核心经验: Collider 是 d-分离判断中最容易出错的地方。看到 V 型结构 (→ ←) 就警惕!

</details>

---

### Q6: d-分离的"所有路径"原则 ⭐⭐⭐

考虑以下图:

```
    A → B → C
        ↑
        D → E
```

(即: A→B, B→C, D→B, D→E)

判断: A ⟂ E | B 是否成立?

<details>
<summary>点击查看答案</summary>

**不成立! ❌**

分析所有从 A 到 E 的无向路径:

- **路径 1**: A → B ← D → E
  - B 是 collider (A→B←D)
  - B 在 Z={B} 中 → collider 被观测 → **路径活跃!**

因为有一条路径活跃, A 和 E 在给定 B 时就是 d-连通的 → 不独立。

> ⚠️ 陷阱: 很多人只看 A→B→C 方向, 忘了 B 同时也是 D→B←A 的 collider。
> 一个节点可以在不同路径中扮演不同角色!
</details>

---

### Q7: 复杂 d-分离 ⭐⭐⭐⭐

![复杂DAG](想象以下图结构)

```
    X₁ → X₂ → X₃ → X₄
          ↓         ↑
          X₅ → X₆ ─┘
```

(即: X₁→X₂, X₂→X₃, X₃→X₄, X₂→X₅, X₅→X₆, X₆→X₄)

判断:

(a) X₁ ⟂ X₆ | {X₂}
(b) X₁ ⟂ X₄ | {X₂, X₃}
(c) X₁ ⟂ X₄ | {X₃}
(d) X₁ ⟂ X₆ | {X₂, X₄}

<details>
<summary>点击查看答案</summary>

(a) **真** ✅ — 所有路径: X₁→X₂→X₅→X₆ 和 X₁→X₂→X₃→X₄←X₆。路径1在X₂处阻塞(链式); 路径2中 X₄ 是 collider 且未观测 → 阻塞。所有路径阻塞 → d-分离

(b) **真** ✅ — X₁→X₂→X₃→X₄ 被 X₂ 和 X₃ 双重阻塞; X₁→X₂→X₅→X₆→X₄ 被 X₂ 阻塞。X₄ 是 collider 但所有路径在到达它之前就已经被阻塞了。

(c) **假** ❌ — X₁→X₂→X₃→X₄, 只给 X₃, X₂ 未被阻塞 → 路径1活跃!

(d) **假** ❌ — 路径: X₁→X₂→X₅→X₆→X₄。X₂ 阻塞了这条...但还有另一条: X₁→X₂→X₃→X₄。X₄ 被观测 → 考虑 X₄ 作为 collider 的角色...实际上这里 X₄ 同时被 X₃→X₄ 和 X₆→X₄ 指向, 是 collider。观测 X₄ 激活了 X₁→X₂→X₃→X₄←X₆←X₅←X₂ 这条路径! (X₂ 同时出现在路径两端但这是无向路径层面, 在 collider 被激活后整条路径通)

</details>

---

## 第 3 部分: 局部马尔可夫性质 (L2)

### Q8: 识别父节点和非后代 ⭐

在图 `X → Y → Z → W` 中:

(a) Y 的父节点是什么?
(b) Y 的非后代节点有哪些?
(c) Y 的局部马尔可夫性质要求什么条件独立?

<details>
<summary>点击查看答案</summary>

(a) Pa(Y) = {X}

(b) 后代: Y 的后代 = {Z, W} (沿箭头能到达的)
   非后代: 所有节点 - {Y} - {Z, W} = {X}
   所以 NonDesc(Y) = {X}

(c) 局部马尔可夫: Y ⟂ NonDesc(Y) | Pa(Y)
   即: Y ⟂ X | X

   这 trivially 成立! (给定 X, Y 和 X 当然"独立")

   实际上这个例子中 NonDesc(Y) = Pa(Y), 所以局部马尔可夫性质自动满足。
   更有趣的例子是 NonDesc 包含"旁系"节点的情况。
</details>

---

### Q9: 局部马尔可夫 vs 全局马尔可夫 ⭐⭐⭐

考虑以下图:

```
    A → B
    ↓   ↓
    C   D
    ↓  ↗
    E
```

(即 A→B, A→C, B→D, C→E, D→E)

(a) 写出 P(A,B,C,D,E) 的因子分解
(b) 写出节点 E 的局部马尔可夫性质 (即: 列出 E ⟂ ? | Pa(E))
(c) 使用 d-分离 (全局马尔可夫) 判断: E ⟂ A | {C, D}?
(d) 比较 (b) 和 (c) 的答案: 局部马尔可夫性质给出的独立性和全局马尔可夫给出的有什么不同?

<details>
<summary>点击查看答案</summary>

(a) **因子分解:**
```
P(A,B,C,D,E) = P(A) × P(B|A) × P(C|A) × P(D|B) × P(E|C,D)
```

(b) **E 的局部马尔可夫性质:**
- Pa(E) = {C, D}
- 后代: E 没有子节点 → Descendants(E) = ∅
- 非后代: {A, B, C, D} (除了E的所有节点)
- **E ⟂ {A, B} | {C, D}**

(c) **d-分离判断: E ⟂ A | {C, D}?**
- 路径1: A → C → E, C∈Z → 阻塞 ✓
- 路径2: A → B → D → E, D∈Z → 阻塞 ✓
- 所有路径阻塞 → E ⟂ A | {C, D} ✅

(d) **比较:**
- 局部马尔可夫: E ⟂ {A, B} | {C, D}
- 全局马尔可夫: E ⟂ A | {C, D} 只是局部马尔可夫的一个子集
- 全局马尔可夫能给出更多的条件独立性声明 (如 A ⟂ B | ∅? 等等)
- 在 DAG 中两者等价 → 能从局部推出全局

</details>

---

## 第 4 部分: I-Map 与马尔可夫毯 (L2)

### Q10: I-Map 概念 ⭐⭐

解释以下概念:
(a) I(G) 是什么?
(b) I(P) 是什么?
(c) "G 是 P 的 I-map" 是什么意思?
(d) "G 是 P 的 minimal I-map" 和 "P 是 G 的 perfect map" 有什么区别?

<details>
<summary>点击查看答案</summary>

(a) **I(G)** = 图 G 编码的所有条件独立性 (CI) 声明的集合。
   即: 所有能被 d-分离推导出的 (X ⟂ Y | Z) 三元组。

(b) **I(P)** = 概率分布 P 中实际成立的所有条件独立性集合。
   即: P(X,Y|Z) = P(X|Z)P(Y|Z) 的所有三元组。

(c) **G 是 P 的 I-map** ⟺ I(G) ⊆ I(P)
   图编码的每个 CI 声明在 P 中确实成立。图是分布独立性的"安全近似"
   (图可能遗漏某些 P 中成立的独立性, 但不会声称错误的独立性)。

(d) **Minimal I-map**: G 是 P 的 I-map, 且删去任何一条边都会使 G 不再是 I-map
   → "图的边刚好够用, 没有多余的边"

   **Perfect map (P-map)**: I(G) = I(P), 图完美地捕捉了分布中的所有独立性
   → I-map 的超集: 不仅不少说 (sound), 还不少漏 (complete)

   并非所有分布都有 DAG perfect map! (例如某些包含对称独立性的分布)
</details>

---

### Q11: 马尔可夫毯 ⭐⭐⭐

对于学生网络, 写出节点 Intelligence 的马尔可夫毯。

<details>
<summary>点击查看答案</summary>

**Intelligence 的马尔可夫毯 MB(Intelligence):**

1. **父节点**: Pa(I) = ∅ (Intelligence 没有父节点)
2. **子节点**: Children(I) = {Grade, SAT}
3. **配偶节点** (子节点的其他父节点):
   - Grade 的父节点: {Difficulty, Intelligence} → "配偶" = {Difficulty}
   - SAT 的父节点: {Intelligence} → "配偶" = ∅

**MB(Intelligence) = {Grade, SAT, Difficulty}**

验证性质: I ⟂ (不在 MB 中的节点) | MB(I)
→ I ⟂ Letter | {Grade, SAT, Difficulty}

直觉: 你想预测一个人的智商, 只需要知道他的成绩、SAT分数、和课业难度。
推荐信内容不能提供额外信息 (给定前三者后)。
</details>

---

## 第 5 部分: 无向图/L3

### Q12: 有向图 vs 无向图的表达能力 ⭐⭐⭐⭐

以下哪个条件独立性集合可以用 DAG 表示, 但不能用无向图表示?
(提示: 想想 collider!)

A. X ⟂ Y | Z
B. X ⟂ Y (无条件独立)
C. X ⟂ Y | Z  且  X 和 Z 边缘相关, Y 和 Z 边缘相关
D. X ⟂ Y (无条件独立), 但 X 和 Y 在给定 Z 时变得相关

<details>
<summary>点击查看答案</summary>

**正确答案: D**

分析:
- **DAG**: X → Z ← Y 可以表示 D — X 和 Y 无条件独立 (collider 未观测),
  但给定 Z 后变得相关 (collider 被激活)。
  这就是 V-structure / explaining away!

- **无向图**: 无法表示这种"给定更多变量反而产生依赖"的模式。
  在无向图中, 添加观测变量只会破坏依赖、不会创造依赖。
  无向图的 CI 集合满足"单调性": 更多条件 ⇒ 更多独立。

这正好是有向图的核心优势 — 能表示 explaining away (解释消除) 效应!

</details>

---

### Q13: 局部马尔可夫性质 (无向图版本) ⭐⭐

在图 `X₁ - X₂ - X₃ - X₄` (一条链状 MRF) 中:

写出 X₂ 的局部马尔可夫性质 (无向图版本)。

<details>
<summary>点击查看答案</summary>

在无向图中, 局部马尔可夫性质定义:
**X ⟂ (其余所有节点 - X的闭包) | X的邻居**

- X₂ 的邻居 (马尔可夫毯) bd(X₂) = {X₁, X₃}
- X₂ 的闭包 cl(X₂) = {X₂} ∪ {X₁, X₃} = {X₁, X₂, X₃}
- 其余节点: {X₄}

**局部马尔可夫性质: X₂ ⟂ X₄ | {X₁, X₃}**

直觉: 给定 X₂ 的两个邻居后, X₂ 和图的其余部分 (X₄) 独立。

对比 DAG 版本 (如果这个链是 X₁→X₂→X₃→X₄):
- Pa(X₂) = {X₁}, NonDesc(X₂) = {X₃, X₄}
- DAG 局部马尔可夫: X₂ ⟂ {X₃, X₄} | {X₁}

注意差异! 无向图中邻居包括两边的 X₁ 和 X₃,
而 DAG 中只包含父节点 X₁。
</details>

---

### Q14: 马尔可夫毯的对称性 ⭐⭐⭐

无向图的马尔可夫毯和有向图的马尔可夫毯之间有一个关键差异 — 对称性。请解释这个差异。

<details>
<summary>点击查看答案</summary>

**无向图的马尔可夫毯是对称的:**
X ∈ MB(Y)  ⟺  Y ∈ MB(X)

因为 MB(X) 就是 X 的邻居集合, 邻居关系在无向图中是对称的。

**有向图的马尔可夫毯不对称:**
例如在学生网络中:
- MB(Grade) = {Difficulty, Intelligence, Letter}
- MB(Letter) = {Grade}

Grade 在 Letter 的 MB 中, 但 Letter 也在 Grade 的 MB 中 ✓
- MB(Difficulty) = {Intelligence, Grade} (Intelligence 是 Difficulty 的"配偶"!)
- MB(Intelligence) = {Difficulty, Grade, SAT}

Difficulty 在 Intelligence 的 MB 中吗? MB(Intelligence) = {Difficulty, Grade, SAT} → Yes!
Intelligence 在 Difficulty 的 MB 中吗? MB(Difficulty) = {Intelligence, Grade} → Yes!

这个例子中碰巧是对称的, 但在更复杂的 DAG 中, MB 可以不对称。
(例如当 X 是 Y 的配偶, 但 Y 不是 X 的配偶时)

对称性是 MRF 马尔可夫毯的一个优雅属性, 源于无向边缺乏方向性。
</details>

---

## 第 6 部分: 综合应用题

### Q15: 从领域知识到图结构 ⭐⭐⭐⭐

一位医生想对 COVID-19 的诊断过程建模。她认为:
- 年龄影响基础健康状况
- 基础健康状况和年龄都影响感染 COVID 的风险
- COVID 感染导致发烧和咳嗽
- 发烧和咳嗽都可以导致检测呈阳性
- 基础健康状况也影响发烧的严重程度

请画出对应的贝叶斯网络, 并写出因子分解。

<details>
<summary>点击查看答案</summary>

**贝叶斯网络结构:**

```
    年龄 ──→ 基础健康
      ↘        ↓
        ↘     ↙
          感染COVID
          ↓       ↓
        发烧     咳嗽
          ↓       ↓
          ↘     ↙
          检测阳性
```

(以及: 基础健康 → 发烧)

**因子分解:**

```
P(年龄, 基础健康, 感染, 发烧, 咳嗽, 检测阳性)
= P(年龄)
× P(基础健康 | 年龄)
× P(感染 | 年龄, 基础健康)
× P(发烧 | 感染, 基础健康)
× P(咳嗽 | 感染)
× P(检测阳性 | 发烧, 咳嗽)
```

</details>

---

### Q16: d-分离实战 — 分析因果效应 ⭐⭐⭐⭐⭐

使用 Q15 的 COVID 网络, 回答:

(a) 如果我们想研究"年龄对检测阳性率的总因果效应", 应该对哪些变量进行"不观测"?
    (即: 不应该 condition on 哪些变量, 以免引入选择偏差)

(b) 给定"检测阳性", "年龄"和"基础健康"是否独立? 为什么?

(c) 如果我们想用"咳嗽"的信息来帮助预测某人是否感染 COVID,
   但已知该人的"年龄"和"基础健康"信息, "咳嗽"还有额外信息量吗?

<details>
<summary>点击查看答案</summary>

(a) **不应该 condition on 感染、发烧、咳嗽中的任何一个。**

原因: 这三者都是年龄到检测阳性的因果路径上的中间节点 (mediator)。
如果观测它们, 会阻塞因果路径, 导致我们低估年龄的总效应。
更危险的是, condition on 检测阳性 (=condition on collider 的后代/结果)
可能引入 collider bias, 产生虚假的负相关。

→ 要做因果效应估计, 保持中间路径开放, 只需用后门准则调整混杂因子
(这里是: 控制"基础健康"也不一定对, 需要仔细分析...实际上控制基础健康
可能会阻塞部分年龄的效应。正确的做法取决于想要直接效应还是总效应)

(b) **不独立!** 路径: 年龄 → 基础健康 (活跃) 和 年龄 → 感染 ← 基础健康。
检测阳性是 collider 感染的后代 → 相当于观测了 collider 的后代 →
激活了 年龄 → 感染 ← 基础健康 这条路径! 这是经典的 Berkson's paradox。

> 这解释了为什么在医院 (condition on "生病") 看到的相关性不能推广到
> 全人群 — 选择偏差。

(c) **仍然有!**

路径: 感染 → 咳嗽。这条路径是链式, 没有阻塞。
即使给定了年龄和基础健康, 咳嗽仍然提供关于感染的额外信息,
因为咳嗽是感染的直接结果, 不受年龄/基础健康的完全屏蔽。

用 d-分离验证: 感染 ⟂ 咳嗽 | {年龄, 基础健康}?
路径 感染→咳嗽, 无中间节点, 无条件阻塞 → 不独立!
(注意: 感染不是咳嗽和年龄的 collider, 这条路径没有中间阻塞节点)

</details>

---

## 答案速查表

| 题目 | 答案 |
|------|------|
| Q1 | A, B, D |
| Q2 | P(A)P(B)P(C\|A,B)P(D\|C) |
| Q3 | B (2^n - 1) |
| Q4 | 见详细答案 |
| Q5 | (a)✅ (b)❌ (c)✅ (d)❌ (e)✅ (f)❌ (g)✅ |
| Q6 | 不独立 ❌ |
| Q7 | (a)✅ (b)✅ (c)❌ (d)❌ |
| Q8 | Pa={X}, NonDesc={X} |
| Q9 | 见详细答案 |
| Q10 | 见详细答案 |
| Q11 | {Grade, SAT, Difficulty} |
| Q12 | D |
| Q13 | X₂ ⟂ X₄ \| {X₁, X₃} |
| Q14 | MRF的MB对称, DAG的MB不对称 |
| Q15 | 见详细答案 |
| Q16 | 见详细答案 |

---

> 📖 **参考资源**
> - Koller & Friedman, *Probabilistic Graphical Models*, Ch.3
> - Bishop, *Pattern Recognition and Machine Learning*, Ch.8
> - [Eric Xing 课程讲义](http://www.cs.cmu.edu/~epxing/Class/10708/lecture.html)
> - [CMU 10-708 2019 课程主页](https://sailinglab.github.io/pgm-spring-2019/)
