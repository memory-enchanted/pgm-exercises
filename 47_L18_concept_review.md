# CMU 10-708 Lecture 18 概念体系梳理 — 因果关系2

> 整理自 Eric Xing 教授 CMU 10-708 Lecture 18: Causality 2 — Causal Asymmetry, Confounding, Selection Bias, Temporal Info, Transfer Learning
>
> 核心教材: Pearl (2009) *Causality*, Peters et al. (2017) *Elements of Causal Inference*, Schölkopf et al. (2012), Janzing et al. (2012)

---

## 📐 全局定位：从因果推断到因果发现

```
L17: 因果关系1 (Causality 1)
  已知因果图 → do-算子, 后门/前门调整, 反事实推理
  核心工具: 图手术, do-Calculus, SCM
    │
    ▼
  L18: 因果关系2 (Causality 2)
  
  不假设已知因果图! 更深层的问题:
  ① 为什么需要因果? — 决策, 迁移, 稳健预测
  ② 如何发现因果? — 因果不对称性 (asymmetry)
  ③ 什么会扭曲因果? — 混杂, 选择偏差
  ④ 因果如何帮助ML? — 迁移学习, 分布外泛化
  
  九大主题:
  ① Why Causality — 因果思维的必要性
  ② Causal Inference — 因果推断框架总览
  ③ Conditional Independence — 约束方法 (PC, FCI)
  ④ Causal Asymmetry from Noise — 噪声独立性 → 因果方向
  ⑤ Independent Change — P(cause) 与 P(effect|cause) 独立变化
  ⑥ Confounding — 混杂的深层分析
  ⑦ Selection Bias — 选择偏差 & Berkson 悖论
  ⑧ Temporal Info — 时间信息在因果发现中的应用
  ⑨ Transfer Learning — 因果不变性 & 紧凑变化描述
```

**一句话概括 L18**: L17 教我们"给定因果图怎么用"，L18 教我们"因果图从哪来"——利用因果方向的不对称性（噪声独立、机制独立、变化独立性），从数据中发现因果结构，并利用因果不变性实现稳健迁移学习。

---

## 概念 1：Why Causality — 为什么需要因果思维 (🔑🔑)

---

### 1.1 超越预测 — 因果的三个核心动机 (🔑🔑)

```
动机一: 决策与干预 (Decision Making)
  预测模型: "这个病人的30天再入院风险是 23%"
  因果模型: "如果给这个病人安排家庭护理, 再入院风险会降到多少?"
  
  预测回答 "what will happen"
  因果回答 "what if we act"
  
  例子: 推荐系统
    关联: "买尿布的人也买啤酒" → 推荐啤酒
    因果: "如果推荐啤酒, 用户会买吗?" → 提升GMV的干预
    
动机二: 稳健性与迁移 (Robustness & Transfer)
  关联模型: 依赖 P(Y|X), 在训练分布上学到的模式
            → 环境改变, 模式失效 (协变量偏移, covariate shift)
  因果模型: 依赖不变的因果机制 P(effect|cause)
            → 即使 P(X) 改变, 机制仍不变
            → 这就是"迁移学习"的因果基础!
  
  例子: 图像分类
    牛 vs 骆驼: 沙漠背景 vs 草原背景
    关联模型: 学到 "棕色背景 → 骆驼" (虚假相关!)
    因果模型: 学到 "驼峰 → 骆驼" (因果特征)
    迁移: 动物园里的骆驼 — 绿色背景 → 因果模型仍有效!
    
动机三: 公平性与可解释性 (Fairness & Explainability)
  统计公平: Y_pred ⟂ A (敏感属性) — 简单粗暴
  因果公平: 区分直接歧视 (A→Y) vs 间接效应 (A→E→Y)
            → 允许通过正当渠道的差异, 消除不正当的歧视
```

### 1.2 因果推断的"不可能三角" (🔑🔑)

```
没有免费的午餐 — 因果推断总要在三者间权衡:

        观测数据 (Observational)
             /\
            /  \
           /    \
          /      \
         /___因果___\
  无假设             RCT
  (Impossible)      (Expensive/Unethical)

三条路:
  ① RCT (随机对照试验): 消除所有混杂, 但昂贵/不道德
  ② 强假设 + 观测数据: 因果图 + 后门准则 + do-Calculus
  ③ 因果发现: 从数据学因果图 (L18 的重点!)
  
  L18 的核心问题: 如何从观测数据中"发现"因果关系
                 (不只是"使用"已知的因果图)
```

---

## 概念 2：Causal Inference — 因果推断框架总览 (🔑🔑)

---

### 2.1 因果推断的两大类问题 (🔑)

```
类型一: 因果效应估计 (Causal Effect Estimation)
  已知: 因果图 G (或部分已知)
  目标: 估计 P(Y|do(X)) 或 ATE
  方法: 后门调整, 前门调整, do-Calculus, IV, 倾向性得分
  对应: L17 主要内容

类型二: 因果发现 (Causal Discovery)
  已知: 来自 P(V) 的 i.i.d. 样本
  目标: 推断因果图 G 的结构
  方法:
    ① 基于约束 (Constraint-based): PC, FCI
    ② 基于得分 (Score-based): GES
    ③ 基于FCM (FCM-based): LiNGAM, ANM, IGCI
    ④ 混合方法
  对应: L18 主要内容
```

### 2.2 因果发现的挑战 (🔑🔑)

```
核心挑战: 马尔可夫等价类
  — 仅从条件独立性, 无法唯一确定因果图
  
  例: X → Y 和 Y → X 编码相同的 CI 集 (都是空集)
      → CI-based 方法无法区分!
  
  例: X → Z → Y, X ← Z ← Y, X ← Z → Y
      都编码 {X ⟂ Y | Z} → 三种不同的因果结构!
      
所以 L18 的核心贡献: 超越条件独立性, 利用其他不对称性!

不对称性来源:
  ① 噪声与原因的独立性 (ANM, LiNGAM)
  ② P(cause) 与 P(effect|cause) 的独立变化
  ③ 时间先后顺序
  ④ 干预/分布偏移中的不变性
```

---

## 概念 3：Conditional Independence — 约束方法 (🔑🔑)

---

### 3.1 PC 算法回顾与局限 (🔑)

```
PC 算法 (从 L17 回顾):
  ① 从完全无向图开始
  ② 条件独立性检验 → 删除边
  ③ 识别 v-结构 → 定向 collider
  ④ Meek 规则 → 传播方向
  输出: CPDAG (马尔可夫等价类)

局限:
  ✗ 只能学到等价类, 不是唯一 DAG
  ✗ 假设因果充分性 (无隐混杂)
  ✗ 条件独立性检验在有限样本下可能出错
  ✗ 计算复杂度: 最坏 O(p^d), d = 最大度
```

### 3.2 FCI 算法 — 处理隐混杂 (🔑🔑)

```
FCI (Fast Causal Inference):
  动机: PC 假设无隐混杂 — 现实中几乎总是不成立!
  
  FCI 允许存在隐混杂变量
  
  与 PC 的区别:
    ① 更复杂的条件独立性检验 (不仅用相邻节点)
    ② 引入新的边类型来表示隐混杂:
       X ↔ Y    X 和 Y 有未观测的公共原因 (confounded)
       X ∘→ Y   X→Y 或 X↔Y (不确定)
       X ∘—∘ Y  X→Y 或 X←Y 或 X↔Y (完全不确定)
       
  输出: PAG (Partial Ancestral Graph)
    — 比 CPDAG 包含更多不确定性 (因为有隐混杂)

核心理念:
  如果观测到以下 CI 模式, 就能推断隐混杂:
    X ⟂̸ Y | ∅ (边际依赖)
    X ⟂ Y | Z, W (条件独立)
    但不存在任何观测变量的子集能完全解释 X 和 Y 的依赖
  → 必定存在隐混杂!
```

### 3.3 独立性检验方法 (🔑)

```
条件独立性检验是约束方法的基石. 常用方法:

离散变量:
  G² 检验 (似然比检验):
    G² = 2 · Σ_{ijk} n_{ijk} · log(n_{ijk}/ê_{ijk})
    其中 ê_{ijk} 是 H₀: X ⟂ Y | Z 下的期望频数
    G² ~ χ²(df) under H₀

连续变量 — 高斯数据:
  偏相关系数检验 (Fisher z-transform):
    z(ρ̂) = ½ · ln((1+ρ̂)/(1-ρ̂))
    √(n-|Z|-3) · z(ρ̂) → N(0, 1) under H₀: ρ = 0

连续变量 — 非高斯数据:
  Hilbert-Schmidt Independence Criterion (HSIC):
    基于核的方法, 不需要参数假设
    检验 X ⟂ Y (无条件独立)
    条件独立: 使用核条件独立性检验 (KCI-test)
```

---

## 概念 4：Causal Asymmetry from Noise — 噪声中的因果不对称 (🔑🔑🔑)

---

### 4.1 核心直觉 — 为什么噪声能揭示因果方向? (🔑🔑🔑)

```
考虑因果模型 X → Y (加性噪声模型 ANM):

  正向 (因果):  Y = f(X) + N_Y,   其中 N_Y ⟂ X
  
  反向 (反因果): X = g(Y) + N_X,   一般 N_X ⟂̸ Y!

为什么?
  在正向: N_Y 是"与X无关的随机扰动", 机制独立 → N_Y ⟂ X ✓
  在反向:
    X = g(Y) + N_X = g(f(X) + N_Y) + N_X
    → N_X = X - g(f(X) + N_Y)
    → N_X 通过 f(X) 依赖于 X
    → N_X ⟂̸ Y! ✗

这就是因果不对称性的根源:
  在正确方向, 残差与原因独立
  在错误方向, 残差与结果不独立

算法: 对 X→Y 和 Y→X 各拟合回归,
      检验两个方向的残差独立性,
      选择残差独立的方向!
```

### 4.2 LiNGAM — 线性非高斯无环模型 (🔑🔑🔑)

```
LiNGAM (Linear Non-Gaussian Acyclic Model):

模型: X = B·X + E
  其中: B 是严格下三角矩阵 (对应 DAG)
        E 的各分量独立且非高斯 (关键假设!)

为什么非高斯?
  如果 E 是联合高斯的:
    协方差矩阵对称 → 图中多个 DAG 编码相同的 P
    → 无法确定方向

  如果 E 非高斯 (至少一个分量):
    可以利用 ICA (独立成分分析) 辨识 B!
    → 方向可识别!

LiNGAM 算法:
  ① 用 ICA 估计 W: 使得 W·X 的分量尽可能独立
     X = BX + E → (I-B)·X = E → X = A·E, A = (I-B)⁻¹
     ICA 给出 W = A⁻¹ = I-B (但行列顺序不确定)
  
  ② 排列与缩放: 将 W 排列为下三角矩阵
     利用 B 的 acyclicity (严格下三角)
     寻找排列 P 使得 P·W·P^T 尽可能下三角
  
  ③ 剪枝: 对小系数做显著性检验, 去除不显著的边

关键性质:
  LiNGAM 能识别完整的因果 DAG (不仅等价类!)
  但假设限制: 线性 + 非高斯噪声 + 无环 + 无隐混杂

实际扩展: DirectLiNGAM (Shimizu et al. 2011)
  — 无需 ICA, 直接用回归+独立性检验
  — 更稳定, 更易扩展到更多变量
```

### 4.3 加性噪声模型 (ANM) — 非线性扩展 (🔑🔑)

```
ANM (Additive Noise Model, Hoyer et al. 2009):

模型 (双变量):
  正向: Y = f(X) + N_Y,  N_Y ⟂ X
  其中 f 是任意函数 (不仅线性!)

因果推断方法:
  ① 回归 Y on X → f̂, 残差 N̂_Y = Y - f̂(X)
     检验 N̂_Y ⟂ X? (使用 HSIC 或类似方法)
  
  ② 回归 X on Y → ĝ, 残差 N̂_X = X - ĝ(Y)
     检验 N̂_X ⟂ Y?
  
  ③ 决策:
     如果 N̂_Y ⟂ X 但 N̂_X ⟂̸ Y → X → Y
     如果 N̂_X ⟂ Y 但 N̂_Y ⟂̸ X → Y → X
     如果两者都独立 → 无法确定 (可能是高斯噪声/对称分布)
     如果两者都不独立 → 模型错误 (X 和 Y 可能不是直接的因果)

为什么 ANM 优于 LiNGAM?
  ✓ 允许非线性 f
  ✓ 不需要非高斯假设
  ✗ 但需要选择合适的回归方法 (GP, 神经网络等)
  ✗ 独立性检验在高维更困难

可识别性定理:
  在温和条件下, ANM 方向是可识别的:
  如果 f 不是"平凡的"(非常数且非线性程度足够),
  且噪声不是高斯的,
  则只有一个方向的残差是独立的
```

### 4.4 IGCI — 信息几何因果推断 (🔑)

```
IGCI (Information-Geometric Causal Inference, Janzing et al. 2012):

核心思想: P(cause) 与 P(effect|cause) 是"独立"的机制

  在因果方向:
    P_X (cause 的分布)
    P_{Y|X} (因果机制)
    → 这两个应该"独立"(没有任何信息重叠)
    → 即: P_X 不应包含关于 f 的信息, f 不应包含关于 P_X 的信息

数学形式 (熵版本):
  C_{X→Y} = H(X) + E[log |f'(X)|] - H(Y)  (与因果方向有关)

直觉:
  在因果方向, log P'(X) 和 log f'(X) 独立 → 协方差为 0
  在反因果方向, log P'(Y) 和 log g'(Y) 相关 (因为 P_Y 取决于 f 和 P_X)

决策规则 (简化版):
  如果 ∫ p(x) · log|f'(x)| dx < ∫ p(y) · log|g'(y)| dy:
    → X → Y
  否则:
    → Y → X

优势:
  ✓ 确定性关系也适用 (无噪声!)
  ✓ 不需要噪声独立的假设
  ✗ 假设 P_X 和 f 独立(在特定信息几何意义下)
  ✗ 密度估计在有限样本下困难
```

---

## 概念 5：Independent Change — P(cause) 与 P(effect|cause) 独立变化 (🔑🔑🔑)

---

### 5.1 因果模块性 (Causal Modularity) (🔑🔑)

```
因果模块性假设:
  自然界的因果机制是模块化的 (modular)
  
  含义:
    ① 每个因果机制 P(child | parents) 是一个独立的"模块"
    ② 改变一个模块不影响其他模块
    ③ P(cause) 的改变独立于 P(effect|cause) 的机制
    
  这是因果推断的核心假设之一!

例子: 
  日出时间 (cause) → 温度 (effect)
  
  P(日出时间): 由地球自转决定 (物理定律)
  P(温度 | 日出时间): 由辐射传热决定 (物理定律)
  
  这两个机制完全不同, 是独立的模块!
  
  如果地球自转变慢 (P(cause) 改变):
    日出时间分布改变
    但 P(温度 | 日出时间) 不变! (物理定律不受影响)
    
  如果大气成分改变 (P(effect|cause) 改变):
    给定日出时间的温度分布改变
    但日出时间不变!
```

### 5.2 多环境数据中的因果发现 (🔑🔑🔑)

```
利用分布偏移发现因果方向:

假设有 K 个环境, 在每个环境中观测 (X, Y):

如果 X → Y:
  P_e(X) 在不同环境可能不同 (cause 的分布变化)
  P(Y|X) 在不同环境保持不变 (机制是稳定的!)
  
  因为: 模块性 → P(X) 和 P(Y|X) 独立变化
        → 它们不太可能同时变化
        → 环境变化通常只影响 P(X) 或 P(Y|X), 不是两者

如果 Y → X:
  同理, P_e(Y) 在不同环境可能不同
  P(X|Y) 在不同环境保持不变

利用这个不对称性:
  ① 对每个候选因果方向, 检验"条件分布"在不同环境是否不变
  ② 选择条件分布更稳定的方向

具体检验 (双变量):
  方向 X→Y:
    H₀: P_e(Y|X) 在所有环境 e 中相同
    备择: 不同环境有不同的 P_e(Y|X)
    
  方向 Y→X:
    H₀: P_e(X|Y) 在所有环境 e 中相同
    
如果只有 X→Y 的 H₀ 成立 → 证据支持 X→Y
如果只有 Y→X 的 H₀ 成立 → 证据支持 Y→X

这就是因果不变性 (Causal Invariance) 思想!
```

### 5.3 不变因果预测 (ICP) (🔑🔑)

```
不变因果预测 (Invariant Causal Prediction, Peters et al. 2016):

问题: 给定多环境数据, 找出 Y 的直接原因

  Y = f(X_{S*}) + noise
  
  其中 S* 是 Y 的真正原因集 (因果父节点)

核心思想:
  S* 是满足以下条件的(最小)集合:
    对 S* 中变量的任何子集,
    P(Y | X_{S*}) 在所有环境中不变

ICP 算法:
  ① 对每个候选原因集 S:
     检验: P_e(Y | X_S) 在不同环境 e 中是否相同?
  
  ② 收集所有"不变集": I = {S: P_e(Y|X_S) 跨环境不变}
  
  ③ 输出: Ŝ = ∩_{S∈I} S  (所有不变集的交集)
  
  在温和条件下, Ŝ ⊆ S* 以高概率成立
  (即: 选出的变量确实是真正的原因, 但可能不完整)

为什么 ICP 不选非原因变量?
  假设 Z 不是 Y 的原因, 但与某个原因 X 相关:
    Z 可能在环境1中关联 Y (因为 Z 和 X 的关联)
    但在环境2中, Z 和 X 的关联改变 → Z 和 Y 的关联也改变!
    → P(Y|Z) 在不同环境中不稳定
    → Z 被排除

关键洞察: 只有真正的因果父节点, 其条件分布才具有跨环境不变性!
```

---

## 概念 6：Confounding — 混杂的深层分析 (🔑🔑🔑)

---

### 6.1 混杂的三种图结构 (🔑🔑)

```
混杂 (Confounding): X 和 Y 之间存在非因果的统计关联,
                    由共同原因引起

三种经典的混杂图结构:

① 经典混杂 (Classic Confounding):
    Z → X
    Z → Y
    X → Y  (可能存在)
    
    问题: X 和 Y 之间的关联 = 因果效应 + Z 的混杂
    解决: 后门调整 P(Y|do(X)) = Σ_z P(Y|X,z) P(z)

② M-结构 (M-Structure):
    Z₁ → X
    Z₂ → Y
    Z₁ ← U → Z₂  (U 是隐变量)
    
    问题: 如果调节 Z₁ 或 Z₂ ← 会打开路径
    正确: 不需要调节! (默认 X 和 Y 无混杂)
    但: 如果错误地调节 Z₁ → 反而引入虚假关联

③ 蝴蝶结构 (Butterfly Structure):
    U₁ → X    U₂ → X
    U₁ → Y    U₂ → Y
    
    X 和 Y 有多个共同原因 → 需要找到最小调节集
    最小调节集可能不是唯一的!
```

### 6.2 隐混杂与工具变量 (🔑🔑🔑)

```
问题: 当后门准则需要的变量未观测时, 怎么办?

工具变量 (Instrumental Variable, IV):
  
  因果图: U → X, U → Y, X → Y, Z → X
           (Z 不影响 Y, Z 与 U 独立)
  
  Z 称为工具变量, 满足:
    ① Z 与 X 相关 (相关性, relevance): Z → X
    ② Z 只通过 X 影响 Y (排斥性, exclusion): Z → X → Y, 无 Z→Y
    ③ Z 与混杂变量 U 独立 (外生性, exogeneity): Z ⟂ U
  
  IV 估计:
    在线性模型 Y = βX + U 中:
      Cov(Z, Y) = Cov(Z, βX + U) = β Cov(Z, X) + Cov(Z, U)
                = β Cov(Z, X) + 0
      → β = Cov(Z, Y) / Cov(Z, X)
    
    这就是著名的 Wald estimator!
  
  例子: 评估教育(X)对收入(Y)的效应 (有能力的混杂 U)
    工具变量 Z: 出生季度 (Angrist & Krueger, 1991)
    假设: 出生季度 ≈ 随机 (Z ⟂ U)
         出生季度 → 入学年龄 → 受教育年限 (Z → X)
         出生季度 → 收入? 只通过教育 (exclusion)
```

### 6.3 混杂的检测 (🔑)

```
如何从数据中检测混杂?

方法一: 负面结果检验 (Negative Outcome Test)
  找一个已知 X 不应影响的变量 W
  
  如果检测到 X 和 W 的关联:
    → 这种关联不是因果的 → 必然来自混杂
    → 推断存在 U 同时影响 X 和 W
  
  例子: X=药物, W=去年的事故次数 (不应被药物影响)
         如果检测到药物"降低"了去年的事故次数:
           → 荒谬! 必然有混杂(如: 健康意识)

方法二: 添加伪结果 (Pseudo-Outcome)
  类似思路, 使用已知不受X影响的结果

方法三: 使用 FCI 算法
  如前面所述, FCI 可以检测存在隐混杂的变量对 (↔ 边)
```

---

## 概念 7：Selection Bias — 选择偏差 (🔑🔑🔑)

---

### 7.1 Berkson 悖论 — 选择偏差的经典案例 (🔑🔑🔑)

```
Berkson 悖论 (1946):

场景: 医院研究
  只研究入院患者 (S=1)
  疾病1 (D1) 和疾病2 (D2) 在总体中独立!

  因果图: D1 → S ← D2
          (两种疾病都增加入院概率)
          D1 ⟂ D2 in population

  条件化 S=1 (只看入院患者):
    D1 ⟂̸ D2 | S=1!
    
  为什么?
    如果入院患者有 D1, 那么:
      要么 D1 是入院原因 → D2 概率低于平均
      (因为 D1 已经在驱动入院, D2 "不需要"了)
    在医院人群中, D1 和 D2 呈负相关!
    
  但这完全是选择偏差 — 不是因为 D1 真的预防 D2

数值例子:
  总体:
    P(D1=1) = 0.1, P(D2=1) = 0.1, D1 ⟂ D2
    P(S=1 | D1=0, D2=0) = 0.05  (健康的很少入院)
    P(S=1 | D1=1, D2=0) = 0.80  (疾病1 + 高入院率)
    P(S=1 | D1=0, D2=1) = 0.80  (疾病2 + 高入院率)
    P(S=1 | D1=1, D2=1) = 0.95  (两种病都有的几乎必然入院)
    
  总体中: P(D1=1 | D2=1) = P(D1=1) = 0.1  (独立!)
  
  入院患者中 (S=1):
    入院总概率: P(S=1) = 0.05×0.81 + 0.8×0.09 + 0.8×0.09 + 0.95×0.01
                        = 0.0405 + 0.072 + 0.072 + 0.0095 = 0.194
    
    P(D1=1, D2=1, S=1) = 0.01 × 0.95 = 0.0095
    P(D2=1, S=1) = P(D1=0,D2=1,S=1) + P(D1=1,D2=1,S=1)
                  = 0.09×0.8 + 0.0095 = 0.072 + 0.0095 = 0.0815
    
    P(D1=1 | D2=1, S=1) = 0.0095 / 0.0815 ≈ 0.117
    
    P(D1=1, D2=0, S=1) = 0.09 × 0.8 = 0.072
    P(D2=0, S=1) = 0.81×0.05 + 0.072 = 0.0405 + 0.072 = 0.1125
    
    P(D1=1 | D2=0, S=1) = 0.072 / 0.1125 ≈ 0.640
    
  在入院患者中:
    P(D1|D2=1) = 0.117 << P(D1|D2=0) = 0.640
    → D1 和 D2 显著负相关!
    
  但这是完全虚假的 — 由选择偏差造成!
```

### 7.2 选择偏差的图识别 (🔑🔑)

```
选择偏差 = 条件化 collider (或其子孙) 导致的偏差

图识别规则:
  如果分析中条件化了 collider → 可能引入选择偏差
  
  常见的 collider 条件化:
    ① 条件化 collider 本身 (如 S=1)
    ② 条件化 collider 的子孙
    ③ 数据收集过程隐含条件化 collider
       (如: 只收集住院患者数据 → 隐式条件化 S=1)

恢复方法:
  ① 如果有关于选择机制的额外信息 → 逆概率加权 (IPW)
  ② 如果知道未选择总体的某些矩 → bounds 分析
  ③ 敏感度分析: 假设不同程度的选择偏差, 看结论是否稳健

避免选择偏差:
  ① 不要在回归中控制 collider
  ② 不要基于结果变量选择样本
  ③ 如果有 collider 被隐式条件化, 使用 FCI 或选择图
```

### 7.3 选择偏差 vs 混杂 (🔑)

```
              混杂 (Confounding)        选择偏差 (Selection Bias)
图结构:       X ← U → Y                X → S ← Y (S被条件化)
             (公共原因)                (公共结果被选择)

解决:         调节 U (后门准则)         不要条件化 S 或使用 IPW

条件化效应:   打开后门路径(错误方向!)   打开 collider 路径(错误方向!)

CI 视角:      调节 U → 阻断路径        条件化 S → 打开路径!
              (好!)                    (坏!)

记忆口诀:
  混杂 (Fork):     条件化 → 好 (阻断虚假关联)
  选择偏差 (Collider): 条件化 → 坏 (产生虚假关联!)
  
  这就是为什么:
    回归中加控制变量 → 消除混杂 ✓
    回归中加中介/结果 → 引入选择偏差 ✗
```

---

## 概念 8：Temporal Info — 时间信息在因果发现中的应用 (🔑🔑)

---

### 8.1 时间优先原则 (🔑)

```
最基础的因果线索: 原因在时间上先于结果!

Temporal Precedence:
  如果 X 发生在 Y 之前 → Y 不可能是 X 的原因
  
  但反过来: X 在 Y 之前 → X 可能是也可能不是 Y 的原因
            (X 和 Y 可能都是 Z 的结果, 只是 X 出现更早)

Granger Causality (Granger, 1969):
  时间序列 {X_t, Y_t} 中:
    X "Granger-causes" Y if:
      用过去的 X 和 Y 预测 Y_t 
      > 仅用过去的 Y 预测 Y_t (在统计上显著更好)
  
  注意: Granger Causality ≠ 真实因果关系!
    — 只是预测意义上的 (predictive causality)
    — 如果有混杂 U_t → X_t, U_{t+1} → Y_{t+1}
      → X 可能"Granger-cause" Y 但并非真实原因

利用时间差的因果发现:
  如果有时间序列或面板数据:
    — 滞后变量 (lagged variables) 提供自然的工具变量
    — 时间顺序帮助定向无向边
    — 结构 VAR (SVAR) → 结合时间信息 + 因果约束
```

### 8.2 时间序列中的因果发现 (🔑🔑)

```
线性结构 VAR (SVAR):

  模型: X_t = Σ_{k=1}^{p} B_k · X_{t-k} + ε_t
  
  其中 B_k 是 (n×n) 矩阵, ε_t 是独立噪声
  
  瞬时因果 (contemporaneous):
    B_0 · X_t = ε_t  (B_0 的对角为1)
    (B_0)_{ij} ≠ 0 → X_j 瞬时 cause X_i
  
  滞后因果 (lagged):
    (B_k)_{ij} ≠ 0 → X_j 在 k 期前 cause X_i (在当期)
  
  与静态 LiNGAM 的关系:
    — 带滞后变量的 SVAR ≈ LiNGAM with temporal ordering
    — 时间信息 + LiNGAM → 更好的因果发现

  挑战:
    — 等间距观测 vs 真实因果滞后
    — 瞬时因果的方向仍然需要非高斯假设
    — 隐混杂在时间序列中同样存在
```

---

## 概念 9：Transfer Learning — 因果不变性与迁移 (🔑🔑🔑)

---

### 9.1 迁移学习的因果视角 (🔑🔑🔑)

```
传统迁移学习:
  源域 (source) → 目标域 (target)
  假设: 两者共享某些结构, 但 P(X,Y) 不同
  
  问题: 什么结构在不同域之间是不变的?
  
因果回答: 因果机制 P(effect | direct causes) 是不变的!

紧凑变化描述 (Compact Description of Changes):
  分布的跨域变化可分解为:
    ① P(cause) 的改变 (独立模块)
    ② P(effect|cause) 的改变 (独立模块)
  
  由于模块性, 域变化通常只涉及少量模块的改变
  → "紧凑" (compact) — 不是所有东西都变!

应用: 域适应 (Domain Adaptation)
  假设:
    — 因果图相同
    — P(effect|cause) 机制不变
    — 只有 P(cause) 变化
  
  策略:
    ① 在源域学 P(Y|X_S) (使用不变机制)
    ② 在目标域调整 P(X_S) (cause 的分布)
    ③ 预测: P_target(Y) = ∫ P(Y|X_S) · P_target(X_S) dX_S

例子: 跨人群的医学诊断
  因果图: 基因 → 疾病
  P(疾病|基因) 在所有人种中不变 (生物学机制)
  P(基因) 在不同人种中不同 (群体遗传学)
  
  迁移策略:
    在人群A学习 P(疾病|基因)
    在人群B使用 P(疾病|基因) + 人群B的 P(基因) 分布
    → 无需重新训练!
```

### 9.2 域泛化的因果原理 (🔑🔑)

```
域泛化 (Domain Generalization):
  多个源域 → 未知目标域
  目标: 学一个在所有域都好的预测器

因果方法: 不变风险最小化 (IRM, Arjovsky et al. 2019)

  IRM 目标:
    min Σ_e R_e(Φ)  +  λ · Σ_e ||∇_{w|w=1} R_e(w·Φ)||²
    经验风险        不变性惩罚

  直觉:
    学一个特征表示 Φ(x), 使得:
      最优线性分类器在 Φ 上对所有环境都一样
    → Φ 捕捉了"因果特征"(不变机制)
    → 忽略"虚假特征"(环境依赖)

为什么有效?
  因果特征: P(y | causal_feature) 跨环境不变
  虚假特征: P(y | spurious_feature) 跨环境变化
  
  IRM 隐式地寻找因果特征（因为只有它们是不变的）

局限:
  — 需要多个训练环境（每个有不同的 P(X)）
  — 需要环境标签
  — 如果因果特征本身也可变 → 可能失败
```

### 9.3 因果表征学习 (🔑)

```
因果表征学习 (Causal Representation Learning):

目标: 学习数据的因果表示 — 将观测 X 分解为独立的因果变量

核心思想:
  X (高维观测, 如像素) = g(S₁, ..., Sₙ)  (观测由独立因果变量生成)
  
  其中 S₁, ..., Sₙ 是因果变量, 具有稀疏的因果图结构
  
挑战:
  ① 不可识别性: 许多 S 的表示可以同样好地重建 X
  ② 需要额外的归纳偏置 (inductive bias) 来辨识因果变量

方法:
  ① 利用干预: 不同干预下的数据 → 因果变量独立变化
  ② 利用稀疏性: 因果图中每个变量只有少量父节点
  ③ 利用时间: 时间序列中的慢特征 (slow features)
  ④ 利用独立机制: 因果变量的变化是稀疏的 (少数变量同时改变)

关键论文:
  — Schölkopf et al. (2021): Toward Causal Representation Learning
  — Locatello et al. (2019): Challenging Common Assumptions in Unsupervised Learning of Disentangled Representations
    (无监督解耦 = 不可能; 需要因果归纳偏置!)
```

---

## 关键公式速查

| 概念 | 公式 |
|------|------|
| ANM 正向 | $Y = f(X) + N_Y, \quad N_Y \perp X$ |
| ANM 反向 | $X = g(Y) + N_X, \quad N_X \not\perp Y$ (一般) |
| LiNGAM | $X = BX + E, \quad E_i \text{ 非高斯且独立}$ |
| Wald Estimator (IV) | $\beta = \text{Cov}(Z, Y) / \text{Cov}(Z, X)$ |
| 后门调整 | $P(y \mid do(x)) = \sum_z P(y \mid x, z)P(z)$ |
| IGCI 熵版本 | $C_{X\to Y} = H(X) + E[\log \mid f'(X)\mid] - H(Y)$ |
| ICP 不变检验 | $H_0: P_e(Y \mid X_S) \text{ 在所有环境 } e \text{ 相同}$ |
| Granger Causality | $Y_t = \alpha + \sum_k \beta_k Y_{t-k} + \sum_k \gamma_k X_{t-k} + \varepsilon_t$ |

---

## 参考资料

- Peters, J., Janzing, D., & Schölkopf, B. (2017). *Elements of Causal Inference*. MIT Press.
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
- Shimizu, S., Hoyer, P. O., Hyvärinen, A., & Kerminen, A. (2006). A linear non-gaussian acyclic model for causal discovery. *JMLR*.
- Hoyer, P. O., Janzing, D., Mooij, J. M., Peters, J., & Schölkopf, B. (2009). Nonlinear causal discovery with additive noise models. *NeurIPS*.
- Janzing, D., Mooij, J., Zhang, K., et al. (2012). Information-geometric approach to inferring causal directions. *Artificial Intelligence*.
- Peters, J., Bühlmann, P., & Meinshausen, N. (2016). Causal inference by using invariant prediction. *JRSS-B*.
- Arjovsky, M., Bottou, L., Gulrajani, I., & Lopez-Paz, D. (2019). Invariant Risk Minimization. *arXiv:1907.02893*.
- Schölkopf, B., Locatello, F., Bauer, S., et al. (2021). Toward Causal Representation Learning. *Proceedings of the IEEE*.
