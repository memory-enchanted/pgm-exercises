# CMU 10-708 Lecture 17 概念体系梳理 — 因果关系1

> 整理自 Eric Xing 教授 CMU 10-708 Lecture 17: Causality 1 — Causality, Intervention, Causal Graph Model, Identification, Counterfactual, Causal Discovery
>
> 核心教材: Pearl (2009) *Causality*, Pearl et al. (2016) *Causal Inference in Statistics: A Primer*, Peters et al. (2017) *Elements of Causal Inference*

---

## 📐 全局定位：从概率到因果

```
L1-L10: 图模型 & 推断 (已知图, 做概率查询)
   L5: 参数估计 (已知图, 估参数)
   L16: 结构学习 (从数据学图结构)
    │
    ▼
  L17: 因果关系 (Causality 1)
  
  核心问题: 关联 ≠ 因果 — 如何从数据中推断因果关系?
  
  七大主题:
  ① Causality — 因果 vs 关联, 辛普森悖论, Pearl 因果层次
  ② Intervention — do-算子, 图手术, 截断因子分解
  ③ Causal Graph Model — 因果贝叶斯网络, d-Separation, SCM
  ④ Identification — 后门准则, 前门准则, do-Calculus
  ⑤ Counterfactual Reasoning — 反事实推理, 溯因-行动-预测
  ⑥ Causal Discovery — PC 算法, 马尔可夫等价类, LiNGAM
  ⑦ Implications in ML — 分布外泛化, 公平性, 可解释性
```

**一句话概括 L17**: 因果推断超越概率推断, 通过 do-算子和反事实推理回答 "如果……会怎样" 的问题 — 关联是观察世界的影子, 因果是改变世界的杠杆。

---

## 概念 1：Causality 基础 — 关联 ≠ 因果 (🔑🔑🔑)

---

### 1.1 为什么概率图模型不够? (🔑🔑)

```
贝叶斯网络 (L1-L4):
  P(X₁, ..., Xₙ) = ∏ P(Xᵢ | pa(Xᵢ))
  
  解释: "观察到 X₁=x₁, 则 X₂ 的分布是……"
  局限: 只能回答观察层面的问题, 无法回答干预层面的问题

因果图模型:
  P(X₁, ..., Xₙ | do(Xₖ=xₖ)) = ∏_{i≠k} P(Xᵢ | pa(Xᵢ)) · 𝟙[Xₖ=xₖ]
  
  解释: "强制将 Xₖ 设为 xₖ 后, 系统的分布是……"
  能力: 可以预测干预效果, 回答反事实问题

关键差异:
  P(Y | X=x):     看后视镜 — 在人群中筛选 X=x 的人, 观察 Y
  P(Y | do(X=x)): 把控方向盘 — 强制所有人 X=x, 测量 Y 的变化
```

### 1.2 辛普森悖论 — 因果推断的"Hello World" (🔑🔑🔑)

```
场景: 评估某种药物是否有效

              总体        男性        女性
            康复 未康复   康复 未康复   康复 未康复
  服药组     20   20      18   12       2    8
  对照组     16   24       7    3       9   21

  总体: P(康复|服药) = 20/40 = 50%
        P(康复|对照) = 16/40 = 40%  → 药物有效? ✓

  分组: 男性  P(康复|服药) = 18/30 = 60%  >  P(康复|对照) = 7/10 = 70%  ✗
        女性  P(康复|服药) = 2/10  = 20%  <  P(康复|对照) = 9/30 = 30%  ✗

为什么方向反转?
  性别(G) 同时影响 服药(T) 和 康复(R):
    男性更可能服药 (30/40 vs 10/40)
    而男性天然康复率更高 (不管服不服药)
  → T ← G → R: G 是混杂变量!

正确的因果效应:
  P(R | do(T=服)) = Σ_g P(R | T=服, G=g) · P(G=g)
                  = P(R|T=服,G=男)·P(G=男) + P(R|T=服,G=女)·P(G=女)
                  = 0.60·0.50 + 0.20·0.50 = 0.30 + 0.10 = 0.40

  P(R | do(T=对)) = 0.70·0.50 + 0.30·0.50 = 0.35 + 0.15 = 0.50

  因果效应 = P(R|do(T=服)) - P(R|do(T=对)) = 0.36 - 0.46 = -0.10
  → 药物实际上降低了康复率! (正确结论)
```

### 1.3 Pearl 因果三层次 (🔑🔑)

```
Level 1 — Association (关联): P(Y | X)
  "看到乌云 → 下雨概率?"
  工具: 条件概率, 相关性, 回归
  图操作: 无 (纯观察)
  
Level 2 — Intervention (干预): P(Y | do(X))
  "人工降雨后 → 雨量?"
  工具: do-算子, 后门调整, IV, RCT
  图操作: 删除指向 X 的边

Level 3 — Counterfactual (反事实): P(Y_x | X=x', Y=y')
  "昨天没人工降雨, 如果降了会怎样?"
  工具: SCM, 溯因-行动-预测, 概率 of Necessity/Sufficiency
  图操作: 完整 SCM 修改 + 外生变量更新

           Level 3 (反事实)
          /             \
    Level 2 (干预)        ← 可归约?
          \
        Level 1 (关联)

原则: 高层问题不能仅靠低层信息回答!
      要回答 Level 3, 必须要有 SCM (或等价信息)
      要回答 Level 2, 必须要有因果图 (或有 RCT 数据)
```

---

## 概念 2：Intervention — do-算子与图手术 (🔑🔑🔑)

---

### 2.1 do-算子的图操作 — "图手术" (🔑🔑)

```
do(X=x) 的图手术 (Graph Surgery):
  
  原始图 G:          干预后图 G_do(X):
    Z                   Z
   ↗ ↘                ↗
  X → Y              X(=x) → Y
  ↑                  (固定)
  W                   W

操作:
  ① 删除所有指向 X 的边: Z→X 和 W→X 被删除
  ② 将 X 固定为 x (不再是随机变量)
  ③ 保持 X→Y 的边不变

截断因子分解公式 (Truncated Factorization):
  P(v₁, ..., vₙ | do(Xᵢ=x)) = 
    ∏_{j: Vⱼ∉X} P(Vⱼ | pa(Vⱼ))  ·  𝟙[Xᵢ=x]
    
直观: 去掉 P(X|pa(X)) 这一项 — 因为 X 不再由 pa(X) 生成, 而是被强制设置
```

### 2.2 后门路径 (Backdoor Path) (🔑🔑)

```
后门路径定义: X 和 Y 之间任何以指向 X 的边为起点的无向路径

例子:
  Z₁ ← W → Z₂ → X → Y        ← 没有指向 X 的边, 不是后门路径
  X ← Z₁ ← W → Y             ← 有! X ← Z₁ 是指向 X 的, 这是后门路径!
  X ← Z₁ → Z₂ → Y            ← 有! 同样以指向 X 的边开始

关键识别: 后门路径 = X "倒退"出去, 然后走到 Y

为什么叫"后门"?
  X → ... → Y  是因果路径 (前门 — front door)
  X ← ... → Y  是非因果路径 (后门 — back door)
  
  后门路径携带着混杂关联 (confounding association)
  如果不阻断后门路径, 观测到的"X和Y的关联"中会混入非因果成分
```

### 2.3 干预 vs 条件化 — 关键对比 (🔑🔑🔑)

```
例子: 图结构 Z → X → Y, 且 Z → Y (Z 是混杂变量)

条件化 P(Y | X=x):
  只看 X=x 的子群体
  子群体中, X 和 Z 仍然关联 (因为 Z→X)
  所以 X 和 Y 之间的关联 = 直接因果 + Z 的混杂
  公式: P(Y | X=x) = Σ_z P(Y | x,z) P(z | x)
                                             ^^^^^^^^
                                          受 X 影响, 扭曲了 Z 的权重

干预 P(Y | do(X=x)):
  强制所有人 X=x, 切断 Z→X
  人群中 X 和 Z 独立
  所以 X 和 Y 之间的关联 = 纯直接因果
  公式: P(Y | do(X=x)) = Σ_z P(Y | x,z) P(z)
                                            ^^^^^
                                        不受 X 影响, 保持自然权重
                                        
这就是后门调整公式！
```

---

## 概念 3：Causal Graph Model — 因果图模型 (🔑🔑)

---

### 3.1 因果贝叶斯网络 vs 概率贝叶斯网络 (🔑)

```
概率贝叶斯网络:
  - 边表示概率依赖: X→Y 只是表示 P 中 Y 依赖 X
  - 多个等价的贝叶斯网络可能表示同一个 P
    (例如 X→Y 和 Y→X 的完全图都编码 P(X,Y) = P(X)P(Y|X))
  - 只是一个"概率的分解", 不承诺因果方向

因果贝叶斯网络:
  - 边表示直接因果关系: X→Y 表示 X 是 Y 的直接原因
  - 附加假设:
    ① 因果马尔可夫: 给定直接原因, 变量独立于其他非后代
    ② 因果忠实性: P 中的条件独立精确反映图结构
    ③ 因果充分性: 图中没有遗漏的公共原因 (无隐混杂)
  - 因果方向是确定的 — 不同方向编码不同物理机制
```

### 3.2 结构因果模型 (SCM) — 因果的形式化 (🔑🔑)

```
SCM 是一个四元组 M = ⟨U, V, F, P(U)⟩:

  U = {U₁, ..., Uₙ}:  外生变量 (exogenous)
    — "外来的噪声", 由模型外部的原因决定
    — 彼此独立 (因果充分性假设)
  
  V = {V₁, ..., Vₙ}:  内生变量 (endogenous)
    — 由模型内部的变量决定
  
  F = {f₁, ..., fₙ}:  结构方程 (structural equations)
    — Vᵢ = fᵢ(pa(Vᵢ), Uᵢ)
    — pa(Vᵢ) ⊆ V \ {Vᵢ}
    — 方程定义的是因果关系, 不是可逆的!
  
  P(U): 外生变量的概率分布

例子: 药物 vs 康复
  U = {U_T, U_R}     — 未建模的个体因素
  V = {T, R}          — 服药(T), 康复(R)
  
  F:
    T = 𝟙[U_T > 0.5]            — 是否服药取决于个体倾向
    R = 𝟙[α·T + U_R > 0.3]     — 康复方程 (α 是因果效应)
  
关键性质:
  ① F 中的等号 = 赋值, 非代数等号!
     T = f(U_T) 和 U_T = f⁻¹(T) 含义完全不同
     第一个是因果, 第二个是诊断 (逆因果)
  
  ② 外生变量独立 ≠ 内生变量独立
     U_T ⟂ U_R  (by assumption)
     T   ⟂̸ R   (because T→R!)
  
  ③ SCM 蕴含了因果图和观测分布
     因果图: T→R (从结构方程直接可得)
     观测分布: P(T,R) = ∫ P(R|T,U_R)P(T|U_T)P(U_T)P(U_R) dU
```

### 3.3 d-Separation 与因果 (🔑🔑)

```
回顾三种连接在因果语义下的含义:

  链 Chain:  X → Z → Y
    因果含义: X 通过 Z 间接影响 Y
    条件化 Z: 路径阻断 (d-separated)
    直觉: 给定中间原因, 结果不再反映初始原因

  叉 Fork:  X ← Z → Y
    因果含义: Z 是 X 和 Y 的公共原因 (混杂)
    条件化 Z: 路径阻断 (消除混杂)
    直觉: 控制公共原因, 去除非因果关联 → 后门准则!

  对撞 Collider:  X → Z ← Y
    因果含义: Z 是 X 和 Y 的共同结果
    条件化 Z: 路径 OPEN! (选择偏差)
    直觉: 条件化结果, 在其原因间引入虚假关联
          例: 只分析入院患者 (Z=入院), 会发现 X 和 Y 负相关
               (即使 X 和 Y 在总体中独立! 这就是 Berkson's Paradox)
```

---

## 概念 4：Identification of Causal Effect — 因果效应的识别 (🔑🔑🔑)

---

### 4.1 可识别性 (Identifiability) — 核心问题 (🔑🔑)

```
问题: 给定因果图 G 和观测分布 P(V), P(Y|do(X)) 是否可以唯一确定?

可识别 (identifiable): P(Y|do(X)) 可以表示为仅含 P(V) 的表达式 → 不需要 RCT!
不可识别 (non-identifiable): 存在两个不同的 SCM, 产生相同的 P(V),
                             但给出不同的 P(Y|do(X))

核心结果: do-Calculus 是完备的!
  — 任何可识别的因果效应都能通过 do-Calculus 的三条规则推导出来
  — 任何不可识别的因果效应, do-Calculus 会"卡住" (无法继续化简)
```

### 4.2 后门准则 (Backdoor Criterion) — 消除混杂 (🔑🔑🔑)

```
直觉: 阻断所有后门路径, 留下的就是因果关系

定义: Z 满足关于 (X, Y) 的后门准则, 如果:
  ① Z 中没有 X 的后代 (不阻断因果路径)
  ② Z d-separates X 和 Y 在 G_backdoor 中
     (G_backdoor = 删除 X→... 所有从前门出去的边后的图)

后门调整公式 (Backdoor Adjustment):
  P(y | do(x)) = Σ_z P(y | x, z) P(z)
  
  离散: P(y|do(x)) = Σ_z P(y|x,z) P(z)
  连续: P(y|do(x)) = ∫ P(y|x,z) dP(z) = E_{Z}[P(y|x,Z)]

为什么需要条件① (Z 不能是 X 的后代)?
  X → M → Y, 调节 M 会阻断因果路径:
    P(y|do(x)) = Σ_m P(m|x) P(y|m)  ✓  (中介分析)
    P(y|x) ≠ Σ_m P(y|x,m) P(m|x)  ✗  (如果只条件化 M 会怎样)
    
后门准则的变体 — 倾向性得分 (Propensity Score):
  令 e(z) = P(X=1 | Z=z) 为倾向性得分
  → P(y | do(X=1)) = E[ 𝟙[X=1]·Y / e(Z) ]
  只需调节一维的 e(z), 而非高维的 z!
```

### 4.3 前门准则 (Frontdoor Criterion) — 绕过未观测混杂 (🔑🔑)

```
问题: 存在未观测的混杂变量 U (后门准则不适用!)

  结构: U → X → M → Y
        U → Y

  U 未观测 → 无法直接调整 U

前门准则要求: M 满足
  ① M 截断了所有 X→Y 的因果路径 (X→M→Y)
  ② X→M 没有未阻断的后门路径 (没有 U' 同时影响 X 和 M)
  ③ 所有 M→Y 的后门路径都被 X 阻断 (X 条件化即可)

前门调整公式 (Frontdoor Adjustment):
  P(y | do(x)) = Σ_m P(m | x) Σ_{x'} P(y | x', m) P(x')

直观解释 (分两步):
  Step 1: P(m|do(x)) = P(m|x)  ← 因为 X→M 没有后门路径
  Step 2: P(y|do(m)) = Σ_{x'} P(y|x',m) P(x')  ← 用 X 阻断 M→Y 的后门路径 (M←X←U→Y)
  Step 3: P(y|do(x)) = Σ_m P(m|do(x)) P(y|do(m))
                     = Σ_m P(m|x) Σ_{x'} P(y|x',m) P(x') ✓
  
经典例子: 吸烟 → 焦油 → 肺癌
  
  吸烟基因(U) → 吸烟(X) → 焦油(M) → 肺癌(Y)
  吸烟基因(U) → 肺癌(Y)
  
  U 未观测, 后门准则不可用
  但 M=焦油 满足前门准则!
  → P(肺癌|do(吸烟)) 可识别!
```

### 4.4 do-Calculus 三条规则 — 完备的因果演算 (🔑)

```
规则1 (观察的插入/删除):
  P(y | do(x), z, w) = P(y | do(x), w)
  if Y ⟂_G_{X̄} Z | X, W
  
  含义: 如果 Z 在干预X后仅提供无关信息, 则可删除
  
规则2 (干预与观察的交换):
  P(y | do(x), do(z), w) = P(y | do(x), z, w)
  if Y ⟂_G_{X̄,Z̲} Z | X, W
  
  含义: 当所有后门路径被阻断时, do(Z) 等价于 seeing Z
  
  关键! 这就是后门准则的推广
  
规则3 (干预的插入/删除):
  P(y | do(x), do(z), w) = P(y | do(x), w)
  if Y ⟂_G_{X̄,Z̅(W)} Z | X, W
  
  含义: 如果 Z 不影响 Y (在做X后), 删除它的干预

图符号说明:
  G_{X̄}:  删除所有指向 X 的边
  G_{Z̲}:  删除所有从 Z 出发的边
  G_{Z̄(W)}: 删除所有指向 Z 且 Z 不在 W 中的边
```

---

## 概念 5：Counterfactual Reasoning — 反事实推理 (🔑🔑)

---

### 5.1 反事实 vs 干预 — 关注个体 (🔑🔑)

```
干预: P(Y | do(X=x)) — 群体层面
  "如果对群体施加干预, 平均效果是多少?"
  
反事实: Y_x(u) — 个体层面
  "对于具体的个体 u, 如果当时做了不同的选择, 结果会怎样?"

例子:
  干预: "如果所有人都上大学, 平均收入是多少?"
  反事实: "你高中毕业直接工作了, 如果你当时上了大学, 你的收入会是多少?"
```

### 5.2 反事实计算三步法 (🔑🔑🔑)

```
给定: SCM M, 观测到 (X=x, Y=y)
求: P(Y_{x'} | X=x, Y=y)  — "如果 X 当时是 x', Y 会怎样?"

Step 1 — Abduction (溯因):
  利用观测 X=x, Y=y 更新外生变量 U 的分布
  → 计算 P(U | X=x, Y=y)
  → 贝叶斯更新: P(u|obs) ∝ P(obs|u) P(u)

Step 2 — Action (行动):
  修改 SCM: 将 X 的结构方程替换为 X = x'
  → do(X=x') 操作, 形成修改后的模型 M_{x'}

Step 3 — Prediction (预测):
  在修改后的模型 M_{x'} 中
  用更新后的 P(U|obs) 作 U 的分布
  → 计算 Y 的分布

例子 — 线性 SCM:
  模型:
    X = U_X                         (U_X ~ N(μ_X, σ²_X))
    Y = β·X + U_Y                   (U_Y ~ N(μ_Y, σ²_Y), U_X ⟂ U_Y)
  
  观测: X=x, Y=y
  求: E[Y_{x'} | X=x, Y=y]  — "如果 X 当时是 x', Y 的期望是多少?"
  
  Step 1 (Abduction):
    U_X = x  (确定性的)
    U_Y = y - βx  (确定性的, 因为 Y = βx + U_Y ⇒ U_Y = y - βx)
    
  Step 2 (Action):
    新模型: X = x' (固定)
             Y = β·x' + U_Y
    
  Step 3 (Prediction):
    E[Y_{x'} | x, y] = β·x' + E[U_Y | x, y]
                     = β·x' + (y - βx)
                     = β·x' + y - βx
                     = y + β·(x' - x)
  
  解释: 如果 X 从 x 变为 x', Y 会变化 β(x'-x)
        这个结果依赖于个体 u 的 U_Y = y - βx
        不同个体的反事实结果不同!
```

### 5.3 归因的概率 (Probabilities of Causation) (🔑)

```
三个重要的反事实概率:

① Probability of Necessity (PN) — "X 是必要的吗?"
  PN = P(Y_{x'} = false | X=x, Y=true)
  含义: 在 X=x 且 Y=true 的个体中, 如果 X 被设为 x',
        有多少个体的 Y 会变成 false?

② Probability of Sufficiency (PS) — "X 是充分的吗?"
  PS = P(Y_x = true | X=x', Y=false)
  含义: 在 X=x' 且 Y=false 的个体中, 如果 X 被设为 x,
        有多少个体的 Y 会变成 true?

③ Probability of Necessity and Sufficiency (PNS) — 两者兼具
  PNS = P(Y_x = true, Y_{x'} = false)
  含义: 总体中有多少个体对 X 的两种取值给出不同的 Y?

界限关系 (无需完整 SCM 即可计算):
  max{0, P(Y=true|X=x) - P(Y=true|X=x')} ≤ PNS ≤ 
  min{P(Y=true|X=x), 1 - P(Y=true|X=x')}
  
  这是可观测的概率, 给出了 PNS 的紧界!
```

---

## 概念 6：Causal Discovery — 因果发现 (🔑🔑)

---

### 6.1 问题定义与假设 (🔑)

```
因果发现: 仅从观测数据 (i.i.d. 样本) 推断因果结构

输入: D = {v^(1), ..., v^(n)} ~ P(V) i.i.d.
输出: 因果图 G (或等价类)

关键假设:
  ① 因果马尔可夫: G 中编码的 d-separation 蕴含 P 中的条件独立
  ② 因果忠实性: P 中的条件独立全部由 G 的 d-separation 蕴含
  ③ 因果充分性: 没有隐混杂变量 (所有变量的公共原因都在 V 中)

仅有假设①②③ → 最多学到马尔可夫等价类 → CPDAG
要确定所有边的方向 → 需要额外假设或干预数据
```

### 6.2 PC 算法 — 基于约束的方法 (🔑🔑🔑)

```
PC 算法 (Peter & Clark, Spirtes & Glymour 1991):

阶段一: 学习骨架 (Skeleton)
  从完全无向图开始
  for depth d = 0, 1, 2, ..., |V|-2:
    for each 相邻节点对 (X, Y):
      for each 条件集 S ⊆ Adj(X)\{Y} with |S|=d:
        if X ⟂ Y | S (条件独立检验):
          删除边 X—Y
          SepSet(X,Y) = S
          break (不需要测试更大的 S)

阶段二: 定向 v-结构 (Collider Orientation)
  for each 三元组 X—Z—Y with X—Y 无直接边:
    if Z ∉ SepSet(X,Y):
      定向为 X → Z ← Y (collider!)
    
  直觉: Z 不在分离集中 → Z 不是被条件化的变量
        → X 和 Y 是通过 Z 依赖的 (非条件化时)
        → Z 是对撞节点

阶段三: 传播方向 (Meek Rules, 1995)
  规则 R1: X → Z — Y → X → Z → Y  (避免新 collider)
  规则 R2: X → Z → Y, X—Y → X → Y  (避免 cycle)
  规则 R3: X → Z — Y, X → W ← Y, Z—W → Z → Y
  规则 R4: X → Z → W, X—W → X → W  (避免 cycle)

输出: CPDAG (Completed Partially Directed Acyclic Graph)
  = 马尔可夫等价类的图形表示

复杂度: 最坏 O(p^d) 其中 d 是图的最大度
        但在稀疏图中 (d 很小) — 实际可扩展到数百个变量
```

### 6.3 马尔可夫等价类 — 什么可以从数据学到? (🔑🔑)

```
定理: 两个 DAG G₁ 和 G₂ 马尔可夫等价 ⟺
  ① 相同骨架 (same skeleton) — 忽略方向的边集相同
  ② 相同 v-结构 (same v-structures) — collider 模式相同

为什么无法区分等价类中的 DAG?

  例1: X → Y vs Y → X
    两种 DAG 编码相同的条件独立集: {X ⟂̸ Y, 无其他CI}
    P(X,Y) = P(X)P(Y|X) = P(Y)P(X|Y)
    → 仅从 P 无法区分方向!

  例2: X → Z → Y vs X ← Z ← Y vs X ← Z → Y
    三种 DAG 编码相同 CI: {X ⟂ Y | Z, 无其他}
    → 仅从 P 无法区分!

CPDAG 中的边类型:
  →  有向边 (directed): 在所有等价 DAG 中方向相同
  — 无向边 (undirected): 在不同等价 DAG 中方向可能不同
  一个 CPDAG 可能包含这两种边

突破口 — 需要什么才能确定方向?
  ① 干预数据 (RCT 或自然实验)
  ② 非高斯假设 (LiNGAM)
  ③ 加性噪声模型 (ANM)
  ④ 时间顺序 (先因后果)
```

### 6.4 超越等价类 — 确定因果方向的额外方法 (🔑)

```
方法1: LiNGAM (Linear Non-Gaussian Acyclic Model)
  模型: X = B·X + E, 其中 E 的各分量独立且非高斯
  关键: 利用 ICA (独立成分分析) 确定方向
  直觉: 正确方向下, 残差独立于原因
        错误方向下, 残差不独立于原因
  
  例: 正确 X→Y: Y = βX + ε, ε ⟂ X
        
      错误 Y→X: X = γY + ε' = γ(βX + ε) + ε' = γβX + γε + ε'
                ε' = X(1-γβ)/γ - ε
                Cov(X, ε') = Cov(X, -ε) ≠ 0 一般情况
                → 残差与"原因"相关 → 拒绝!

方法2: ANM (Additive Noise Model)
  模型: Y = f(X) + N_Y, N_Y ⟂ X
  方法: 回归 Y on X → 残差, 检验残差 ⟂ X?
        回归 X on Y → 残差, 检验残差 ⟂ Y?
        哪个方向的残差独立, 哪个就是因果方向

方法3: IGCI (Information-Geometric Causal Inference)
  原理: 在因果方向 P(cause) 和 P(effect|cause) 的信息几何
        是"独立"的; 在反方向则依赖
```

---

## 概念 7：Implications in Machine Learning — 对机器学习的启示 (🔑)

---

### 7.1 分布外泛化 (OOD Generalization) (🔑🔑🔑)

```
挑战: ML 模型在训练分布上表现好, 但在新环境/新数据上表现差

因果视角的洞见:
  因果关系 = 世界的"不变机制" (invariant mechanism)
  — P(effect | direct causes) 在不同环境下可能保持稳定
  — P(cause | effect) (反因果方向) 在不同环境下会改变

例子: 图片分类
  区分"骆驼"和"牛"
  
  传统 ML: 学 P(label | image)
    可能依赖背景颜色 (沙漠 vs 草原) 作捷径
    → 新环境 (动物园的骆驼) 失败!
  
  因果 ML: 学 P(label | causal_features)
    causal_features = 形状, 驼峰, 角等 (label → causal_features)
    背景不是原因, 而是混杂 (label → background, weakly)
    → 不同环境都稳定

方法: 不变风险最小化 (IRM, Arjovsky et al. 2019)
  学习一个特征表示 Φ(x), 使得在 Φ(x) 下:
    P(y | Φ(x)) 在所有训练环境中都一致
  等价于: 学习"因果表示" (causal representation)
```

### 7.2 公平性 (Fairness) (🔑🔑)

```
问题: ML 模型不应歧视敏感属性 A (性别, 种族, ...)

反事实公平性 (Counterfactual Fairness, Kusner et al. 2017):
  定义: 预测器 Ŷ 是反事实公平的, 如果
    P(Ŷ_{A=a} | X=x, A=a) = P(Ŷ_{A=a'} | X=x, A=a)
    
  含义: 对于具体的个体, 如果他们的敏感属性不同 (其他条件不变),
        预测结果不应该改变

为什么反事实公平 > 统计公平?
  统计公平 (如 Demographic Parity): Ŷ ⟂ A
    简单去除敏感属性 → 可能通过代理变量 (如邮编) 间接歧视
    → "禁止使用性别" 但模型可能从"购买商品类型"推断出性别
    
  反事实公平: 对每个个体的敏感属性做"思想实验"
    → 区分因果效应 (应消除) vs 统计关联 (可能不是歧视)
    → 更精细的公平性概念

实现: 需要因果图 → 确定哪些变量应调节, 哪些不应调节
```

### 7.3 强化学习与因果 (🔑)

```
RL 中的动作 = 干预 (do-operator!):
  转移概率 P(S_{t+1} | S_t, A_t) 
    = 在 RL 中 = P(S_{t+1} | do(A_t), S_t)
    
  为什么? 因为 agent 主动选择动作 (intervention)
  而不是被动观察 (observation)
  
  → RL 天然地在做因果推断!

探索 vs 利用 = 干预 vs 观察:
  探索 = 尝试新动作 (做干预) → 学习因果效应 P(S'|do(A))
  利用 = 基于已有知识选择最佳动作

Contextual Bandit = Causal Effect Estimation:
  奖励 R 在动作 A=a 下的期望 = E[R | do(A=a), context]
  → 需要处理混杂 (context 同时影响 A 和 R)
```

### 7.4 半监督学习的因果失败 (🔑🔑)

```
半监督学习 (SSL) 什么时候有效?

因果分析 (Schölkopf et al. 2012):
  
  情况1: Y 是 X 的部分原因 (Y → X₁, ..., Xₖ — 反因果方向)
    P(X|Y) 和 P(X) 以非平凡方式耦合
    → 无标签数据 P(X) 提供了关于 P(Y|X) 的信息
    → SSL 可能有效! ✓
    
  情况2: X 是 Y 的原因 (X₁, ..., Xₖ → Y — 因果方向)
    P(Y|X) 和 P(X) 参数独立 (modularity of mechanisms)
    → 无标签数据 P(X) 不提供关于 P(Y|X) 的信息
    → SSL 不帮助 (除非有特殊先验)! ✗

  结论: SSL 的效果取决于数据生成过程的因果方向!
        这个洞见来自因果推断, 纯统计框架无法解释
```

---

## 关键公式速查

| 概念 | 公式 |
|------|------|
| 截断因子分解 | $P(v \mid do(x)) = \prod_{v_i \notin X} P(v_i \mid pa(v_i))\big|_{X=x}$ |
| 后门调整 | $P(y \mid do(x)) = \sum_z P(y \mid x, z)P(z)$ |
| 倾向性得分加权 | $P(y \mid do(X=1)) = E\left[\frac{\mathbf{1}[X=1] \cdot Y}{e(Z)}\right]$ |
| 前门调整 | $P(y \mid do(x)) = \sum_m P(m \mid x) \sum_{x'} P(y \mid x', m)P(x')$ |
| do-Calculus 规则2 | $P(y \mid do(x), do(z), w) = P(y \mid do(x), z, w)$ if $Y \perp_{\mathcal{G}_{\bar{X}\underline{Z}}} Z \mid X, W$ |
| 反事实 (线性SCM) | $E[Y_{x'} \mid x, y] = y + \beta(x' - x)$ |
| PNS 下界 | $\max\{0, P(y \mid x) - P(y \mid x')\}$ |
| PC 算法条件检验 | $X \perp Y \mid S$ for $S \subseteq Adj(X) \setminus \{Y\}$ |
| 因果马尔可夫 | $X \perp_{\mathcal{G}} Y \mid Z \Rightarrow X \perp_P Y \mid Z$ |

---

## 参考资料

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
- Pearl, J., Glymour, M., & Jewell, N. P. (2016). *Causal Inference in Statistics: A Primer*. Wiley.
- Peters, J., Janzing, D., & Schölkopf, B. (2017). *Elements of Causal Inference*. MIT Press.
- Spirtes, P., Glymour, C., & Scheines, R. (2000). *Causation, Prediction, and Search* (2nd ed.). MIT Press.
- Arjovsky, M., Bottou, L., Gulrajani, I., & Lopez-Paz, D. (2019). Invariant Risk Minimization. *arXiv:1907.02893*.
- Kusner, M. J., Loftus, J., Russell, C., & Silva, R. (2017). Counterfactual Fairness. *NeurIPS*.
- Schölkopf, B., et al. (2012). On causal and anticausal learning. *ICML*.
