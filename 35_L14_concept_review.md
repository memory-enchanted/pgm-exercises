# CMU 10-708 Lecture 14 概念体系梳理 — 深度序列模型

> 整理自 Eric Xing 教授 CMU 10-708 Lecture 14: Deep Sequence Models
>
> 核心教材: Goodfellow et al. (2016) Ch.10 (Sequence Modeling), Vaswani et al. (2017), Graves (2013)

---

## 📐 全局定位：从静态到序列的进化

```
L1-L10: 概率图模型 (静态变量依赖)     L11: DL 统计/算法基础
    │                                      │
    └──────────────┬───────────────────────┘
                   │
    ┌──────────────┼──────────────┐
    ▼              ▼              ▼
  L12-L13:       L14:           L15+:
  DGM (生成)     Sequence       ...
                 Models

L14 的核心使命: 让神经网络理解"顺序" — 时间、位置、因果
```

**一句话概括 L14**: 序列模型从 CNN (局部感受野) → RNN (循环状态) → Attention (全局加权) → Transformer (纯注意力), 逐步解决了长距离依赖和并行化两大核心难题。

---

# 概念 1：CNN for Sequences — 一维卷积的序列建模

---

## 1.1 从图像卷积到时序卷积 (🔑)

```
图像 CNN (2D):                    时序 CNN (1D):
─────────────────                 ─────────────────
Input: H × W × C                  Input: T × D  (时间步 × 特征)
Kernel: K_h × K_w × C × F         Kernel: K × D × F  (窗口大小 × 特征 × 滤波器)
Slide: 空间轴上滑动               Slide: 时间轴上滑动
Output: H' × W' × F               Output: T' × F

核心不变: 卷积 = 局部加权求和 → 提取局部模式
```

## 1.2 时序卷积的三个核心概念

```
1. 感受野 (Receptive Field):
   每层卷积看 K 个时间步 → L 层 = K^L 的感受野
   空洞卷积 (Dilated Conv): 跳跃采样 → 指数级扩大感受野

2. 因果卷积 (Causal Convolution):
   时刻 t 的输出只能依赖 ≤ t 的输入 (不用未来信息)
   实现: 在序列左侧 zero-pad K-1 个位置

3. 残差连接 (Residual Connection):
   y = Conv(x) + x  (当维度匹配时)
   → 让梯度直接流过 → 可堆叠很深 (WaveNet: 30+ 层)
```

## 1.3 1D 卷积详细计算

```
输入序列:  x = [x₁, x₂, x₃, x₄, x₅],  D=1 (一维信号)
Kernel:    w = [w₀, w₁, w₂],  K=3
Stride=1, No padding

输出: y_t = Σ_{k=0}^{K-1} w_k · x_{t+k}

y₁ = w₀·x₁ + w₁·x₂ + w₂·x₃
y₂ = w₀·x₂ + w₁·x₃ + w₂·x₄
y₃ = w₀·x₃ + w₁·x₄ + w₂·x₅

输入长度 T=5, Kernel K=3, Stride=1, No Pad → 输出长度 = T - K + 1 = 3
```

### 因果关系对比

```
普通卷积:           因果卷积 (causal):
t=1: x₁,x₂,x₃      t=1: [pad],[pad],x₁  ← 只看过去!
t=2: x₂,x₃,x₄      t=2: [pad],x₁,x₂
t=3: x₃,x₄,x₅      t=3: x₁,x₂,x₃
```

## 1.4 CNN 序列模型的优缺点

```
✅ 优点:
  - 并行计算: 各时间步卷积独立 → 训练极快
  - 局部模式敏感: 擅长捕获 n-gram / 局部周期性
  - 梯度稳定: 无循环结构 → 无梯度消失/爆炸 (深层时仍有)
  - 灵活的感受野: 通过空洞卷积可以指数级扩大

❌ 缺点:
  - 长距离依赖弱: 需要很多层才能让远距离 token 交互
  - 位置敏感: 对绝对位置不敏感 → 需要位置编码
  - 固定感受野: 不像 Attention 可以动态选择关注哪些位置
```

---

# 概念 2：RNN — 循环状态机

---

## 2.1 Vanilla RNN 的核心公式 (🔑)

```
时刻 t 的 RNN:
  h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b_h)    ← 状态更新
  y_t = W_hy · h_t + b_y                            ← 输出

展开 (Unrolling):
  x₁ → [RNN] → h₁ → [RNN] → h₂ → [RNN] → h₃ → ... → h_T
                ↓          ↓          ↓
               y₁         y₂         y₃

关键: 所有时刻共享同一套参数 (W_hh, W_xh, W_hy)!
```

## 2.2 反向传播通过时间 (BPTT) (🔑🔑)

```
BPTT = 在展开的计算图上做反向传播

Forward:  h₁ = σ(W·h₀ + U·x₁)
          h₂ = σ(W·h₁ + U·x₂)
          ...
          h_T = σ(W·h_{T-1} + U·x_T)
          L = Σ_t Loss(y_t, target_t)

Backward: ∂L/∂W = Σ_t ∂L/∂h_t · ∂h_t/∂W
          但 ∂h_t/∂W 依赖 h_{t-1} → 链式法则展开 T 步!

          ∂L/∂h₁ = ∂L/∂h_T · Π_{i=2}^{T} ∂h_i/∂h_{i-1}
                         └──── 梯度连乘 T-1 次 ────┘
```

### 梯度消失 vs 梯度爆炸

```
∂h_t/∂h_{t-1} = W_hh · diag(σ'(W_hh·h_{t-1} + W_xh·x_t))

连乘 T 次:
  |λ_max| < 1 → 梯度指数衰减 → 梯度消失 → 学不到长距离依赖
  |λ_max| > 1 → 梯度指数增长 → 梯度爆炸 → 训练不稳定

梯度爆炸解法: Gradient Clipping
梯度消失解法: LSTM / GRU (门控机制!)
```

## 2.3 LSTM — 长短期记忆 (🔑🔑🔑)

```
LSTM 的三个门 + 一个细胞状态:

遗忘门: f_t = σ(W_f·[h_{t-1}, x_t] + b_f)     ∈ [0,1]
输入门: i_t = σ(W_i·[h_{t-1}, x_t] + b_i)     ∈ [0,1]
输出门: o_t = σ(W_o·[h_{t-1}, x_t] + b_o)     ∈ [0,1]

候选记忆: g_t = tanh(W_g·[h_{t-1}, x_t] + b_g)   ∈ [-1,1]

细胞状态更新: C_t = f_t ⊙ C_{t-1} + i_t ⊙ g_t
                    └─ 忘记旧记忆 ─┘  └─ 写新记忆 ─┘

隐藏状态: h_t = o_t ⊙ tanh(C_t)
                └─ 输出门控制暴露多少细胞状态 ─┘

⊙ = element-wise product (逐元素乘法)
```

### LSTM 如何解决梯度消失?

```
关键: 细胞状态 C_t 的更新路径

C_t = f_t·C_{t-1} + i_t·g_t
∂C_t/∂C_{t-1} = f_t  (不经过 tanh' !)

当 f_t ≈ 1 (遗忘门开着): 梯度几乎无损地反向传播!
当 f_t ≈ 0: 故意遗忘不重要的信息

对比 RNN: ∂h_t/∂h_{t-1} 每次经过 tanh'(·) ≤ 1 → 指数衰减
```

## 2.4 GRU — 门控循环单元 (🔑)

```
GRU = LSTM 的简化版 (合并 C_t 和 h_t, 合并遗忘门和输入门):

重置门: r_t = σ(W_r·[h_{t-1}, x_t] + b_r)     ∈ [0,1]
更新门: z_t = σ(W_z·[h_{t-1}, x_t] + b_z)     ∈ [0,1]

候选:   h̃_t = tanh(W_h·[r_t ⊙ h_{t-1}, x_t] + b_h)

新状态: h_t = (1-z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
              └── 保留旧状态 ──┘  └── 写新状态 ──┘

参数更少 (少一个门), 效果与 LSTM 相近。
```

## 2.5 RNN 家族对比

| | Vanilla RNN | LSTM | GRU |
|---|---|---|---|
| **状态变量** | h_t | h_t, C_t | h_t |
| **门数量** | 0 | 3 (f, i, o) | 2 (r, z) |
| **梯度传播** | 差 (tanh' 连乘) | 好 (f_t 直接通路) | 好 (z_t 控制) |
| **参数量** | 2·Dh·(Dh+Dx) | 4·Dh·(Dh+Dx) | 3·Dh·(Dh+Dx) |
| **长序列** | ❌ | ✅ | ✅ |
| **计算量** | 最低 | 最高 | 中等 |

---

# 概念 3：Attention Mechanisms — 选择性聚焦

---

## 3.1 为什么需要 Attention? (🔑🔑)

```
RNN 的问题:
  h_T 需要编码整个输入序列的信息 → 信息瓶颈 (bottleneck)!
  长序列: 最后的 h_T 很难记住开头的内容
  "the cat, which ate the fish, was happy" → was ↔ cat (远距离依赖!)

Attention 的核心洞见:
  解码时, 不要只看一个固定向量 h_T, 而是动态地"回顾"所有 encoder 状态!
  → 让 decoder 自己学会在每一步该"看"哪些输入位置
```

## 3.2 三种 Attention 机制 (🔑🔑)

### (A) Bahdanau Attention (Additive)

```
Encoder: h₁, h₂, ..., h_T  (双向 RNN 的 hidden states)

Decoder 在时刻 t:
  e_{t,j} = v^T · tanh(W_a·s_{t-1} + U_a·h_j)     ← 对齐分数 (alignment score)
  α_{t,j} = softmax(e_t)_j                          ← 注意力权重 (归一化)
  c_t     = Σ_j α_{t,j} · h_j                       ← 上下文向量 (加权和)

  s_t = RNN_decoder(s_{t-1}, [y_{t-1}, c_t])        ← 解码器状态更新
  y_t ~ softmax(W_y·[s_t, c_t])                     ← 预测

专有名词:
  Query  = s_{t-1} (decoder 状态: "我要找什么?")
  Key    = h_j     (encoder 状态: "我包含什么?")
  Value  = h_j     (encoder 状态: "我提供什么信息?")
```

### (B) Luong Attention (Multiplicative)

```
三种分数计算方式:

1. Dot:       e_{t,j} = s_t^T · h_j                  ← 最简单
2. General:   e_{t,j} = s_t^T · W_a · h_j            ← 最通用
3. Concat:    e_{t,j} = v^T·tanh(W_a·[s_t; h_j])     ← Bahdanau 式

注意: Luong 用 s_t (当前状态) 而非 s_{t-1} → 先算状态, 再 attention
```

### (C) Scaled Dot-Product Attention (Transformer 的基石)

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V

Q (Query):  [T_q, d_k]    ← "我正在找什么?"
K (Key):    [T_k, d_k]    ← "每个位置包含什么?"
V (Value):  [T_k, d_v]    ← "每个位置提供什么?"

计算步骤:
  1. Scores = Q·K^T           → [T_q, T_k]   (每对 query-key 的相似度)
  2. Scaled = Scores / √d_k   → 防止 d_k 大时 softmax 进入饱和区
  3. Weights = softmax(Scaled) → [T_q, T_k]  (行归一化)
  4. Output = Weights · V      → [T_q, d_v]  (对 Value 加权求和)
```

### 为什么除以 √d_k?

```
假设 q 和 k 的每个元素独立 N(0,1):
  q·k = Σ_{i=1}^{d_k} q_i·k_i → 期望=0, 方差=d_k
  → d_k 很大时, 点积值很大 → softmax 梯度 → 0 (饱和)

除以 √d_k: Var(q·k/√d_k) = d_k/d_k = 1
  → 保持方差为 1 → softmax 在合理区间 → 梯度正常流动
```

## 3.3 Attention 的三种形式

| 形式 | Query | Key | Value | 应用 |
|------|-------|-----|-------|------|
| **Encoder-Decoder** | Decoder 状态 | Encoder 状态 | Encoder 状态 | Seq2Seq (翻译) |
| **Self-Attention** | 序列自身 | 序列自身 | 序列自身 | Transformer |
| **Cross-Attention** | 一个序列 | 另一个序列 | 另一个序列 | 多模态, 条件生成 |

---

# 概念 4：Transformer — 纯注意力架构

---

## 4.1 整体架构 (🔑🔑🔑)

```
Transformer = Encoder × N + Decoder × N

┌───── Encoder ─────┐    ┌──────── Decoder ─────────┐
│                    │    │                          │
│  Input Embedding   │    │  Output Embedding        │
│  + Pos Encoding    │    │  + Pos Encoding          │
│       ↓            │    │       ↓                  │
│  ┌──────────────┐  │    │  ┌──────────────────┐   │
│  │ Multi-Head   │  │    │  │ Masked Multi-Head│   │
│  │ Self-Attn    │  │    │  │ Self-Attn (因果)  │   │
│  └──────────────┘  │    │  └──────────────────┘   │
│       ↓            │    │       ↓                  │
│  ┌──────────────┐  │    │  ┌──────────────────┐   │
│  │ Add & Norm   │  │    │  │ Cross-Attention  │←──┼── Encoder Output
│  └──────────────┘  │    │  │ (Q=dec, K,V=enc) │   │
│       ↓            │    │  └──────────────────┘   │
│  ┌──────────────┐  │    │       ↓                  │
│  │ Feed-Forward │  │    │  ┌──────────────────┐   │
│  │ Network      │  │    │  │ Add & Norm       │   │
│  └──────────────┘  │    │  └──────────────────┘   │
│       ↓            │    │       ↓                  │
│  ┌──────────────┐  │    │  ┌──────────────────┐   │
│  │ Add & Norm   │  │    │  │ Feed-Forward     │   │
│  └──────────────┘  │    │  │ Network          │   │
│       ↓            │    │  └──────────────────┘   │
│  × N 次            │    │       ↓                  │
│                    │    │  ┌──────────────────┐   │
│                    │    │  │ Add & Norm       │   │
│                    │    │  └──────────────────┘   │
│                    │    │       ↓                  │
│                    │    │  × N 次                  │
│                    │    │       ↓                  │
│                    │    │  Linear + Softmax        │
└────────────────────┘    └──────────────────────────┘
```

## 4.2 位置编码 (Positional Encoding) (🔑🔑)

```
为什么需要: Attention 对位置不敏感 → "A B" 和 "B A" 有相同的 attention weights!
           → 必须注入位置信息

Sinusoidal 编码 (原论文):
  PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
  PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})

  pos: 位置索引 (0, 1, 2, ...)
  i:   维度索引 (0, 1, ..., d_model/2 - 1)

特性:
  - 每个位置有唯一编码 (sin/cos 组合)
  - PE_{pos+k} 可由 PE_{pos} 线性表示 → 模型可学到相对位置!
  - 不需要学习参数 → 可直接用于任意长序列
```

## 4.3 Feed-Forward Network (FFN)

```
FFN(x) = ReLU(x·W₁ + b₁)·W₂ + b₂
       = GELU(x·W₁)·W₂  (现代实现)

每个位置独立处理: 同层同参数, 但不同位置的 token 各自计算
  → 这 = "每个 token 独立做特征变换"
  → 和 Attention 互补: Attention 负责 token 间通信, FFN 负责 token 内变换

典型维度: d_model → d_ff=4·d_model → d_model (先扩后缩)
```

## 4.4 Add & Norm (残差连接 + 层归一化)

```
Pre-LN (现代做法):                Post-LN (原论文):
  x = x + Attn(LayerNorm(x))       x = LayerNorm(x + Attn(x))
  x = x + FFN(LayerNorm(x))        x = LayerNorm(x + FFN(x))

Pre-LN 训练更稳定 (梯度不经过 LN), 是现代 Transformer 的默认选择。
```

---

# 概念 5：Multi-Head Attention — 并行多视角注意力

---

## 5.1 核心思想 (🔑🔑🔑)

```
单头 Attention: 只有一种"关注方式"
  → "the cat sat on the mat" → 可能只关注"主语-谓语"关系

Multi-Head: h 个注意力头并行计算, 各自有不同的 QKV 投影
  → Head 1: 关注"句法依赖" (主谓宾)
  → Head 2: 关注"共指消解" (it → cat)
  → Head 3: 关注"局部短语" (相邻词)
  → ...

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O

head_i = Attention(Q·W^Q_i, K·W^K_i, V·W^V_i)

W^Q_i ∈ R^{d_model × d_k}
W^K_i ∈ R^{d_model × d_k}
W^V_i ∈ R^{d_model × d_v}
W^O   ∈ R^{h·d_v × d_model}

通常: d_k = d_v = d_model / h
```

## 5.2 维度流转详解

```
输入: X ∈ R^{T × d_model}

对每个头 i:
  Q_i = X @ W^Q_i  →  [T, d_k]
  K_i = X @ W^K_i  →  [T, d_k]
  V_i = X @ W^V_i  →  [T, d_v]

  head_i = softmax(Q_i @ K_i^T / √d_k) @ V_i  →  [T, d_v]

Concat: [head_1 | head_2 | ... | head_h]  →  [T, h·d_v]

Output: Concat @ W_O  →  [T, d_model]

计算复杂度: O(T²·d_model) — 受序列长度的平方限制!
```

## 5.3 Masked Self-Attention (Decoder)

```
Decoder 自注意力: 不能"偷看"未来 token!

做法: 在 softmax 之前, 给未来位置加上 -∞ mask

Attention Matrix (T=4):
         K₁  K₂  K₃  K₄
      ┌─────────────────┐
  Q₁  │ s₁₁  -∞   -∞   -∞ │  → Q₁ 只能 attend K₁
  Q₂  │ s₂₁ s₂₂  -∞   -∞ │  → Q₂ 只能 attend K₁,K₂
  Q₃  │ s₃₁ s₃₂ s₃₃  -∞ │  → Q₃ 只能 attend K₁,K₂,K₃
  Q₄  │ s₄₁ s₄₂ s₄₃ s₄₄ │  → Q₄ 可以 attend 全部
      └─────────────────┘

softmax(-∞) = 0 → 未来位置的权重被消除
```

## 5.4 Self-Attention vs Cross-Attention vs Masked Self-Attention

| | Q 来源 | K,V 来源 | Mask? | 用途 |
|---|---|---|---|---|
| **Self-Attn (Encoder)** | 输入序列 | 输入序列 | 无 | 双向上下文编码 |
| **Masked Self-Attn (Decoder)** | 输出序列 | 输出序列 | 因果mask | 自回归生成 |
| **Cross-Attn (Decoder)** | Decoder | Encoder输出 | 无 | 对齐输入-输出 |

---

## 📋 演化路径一张表

| 模型 | 关键创新 | 长距离依赖 | 并行化 | 计算复杂度 |
|------|---------|-----------|--------|-----------|
| **CNN** | 局部卷积 + 空洞 | 需深层 | ✅ 完全并行 | O(T·K·D²) |
| **RNN** | 循环状态 | ❌ (梯度消失) | ❌ 顺序 | O(T·D²) |
| **LSTM/GRU** | 门控机制 | ✅ | ❌ 顺序 | O(T·D²) |
| **Seq2Seq+Attn** | 动态对齐 | ✅ | ❌ Decoder串行 | O(T_s·T_t·D) |
| **Transformer** | 纯 Self-Attention | ✅ | ✅ 完全并行 | O(T²·D) |
| **Linformer/Performer** | 线性注意力 | ✅ | ✅ | O(T·D²) |

---

## 🔗 概念关系图

```
              词嵌入 + 位置编码
                    │
     ┌──────────────┼──────────────┐
     ▼              ▼              ▼
  1D CNN         RNN/LSTM      Self-Attention
  (局部)        (循环状态)     (全局交互)
     │              │              │
     └──────────────┼──────────────┘
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
    Scaled Dot-Product    Multi-Head
       Attention          Attention
          │                   │
          └─────────┬─────────┘
                    ▼
              Transformer
           (Encoder-Decoder)
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
      BERT         GPT        T5
  (Encoder-only)(Decoder-only)(Encoder-Decoder)
```

---

## 🎯 核心洞见一句话

| # | 洞见 |
|---|------|
| 1 | **CNN on sequences = 局部模式检测器** — 并行高效, 适合捕获 n-gram, 但长距离依赖需很多层 |
| 2 | **RNN = 循环状态机** — 天然适合序列, 但串行慢且梯度消失导致难以学习长距离依赖 |
| 3 | **LSTM = RNN + 门控高速公路** — f_t 直接通路让梯度无损传播, 解决了长距离依赖 |
| 4 | **Attention = 可微分的查表** — Query 去匹配 Key, 按相似度对 Value 加权平均 |
| 5 | **Transformer = Attention + FFN + Residuals** — 抛弃循环, 全并行, Attention 负责通信, FFN 负责计算 |
| 6 | **Multi-Head = 多视角 Attention** — 不同头关注不同类型的依赖关系, 拼接后获得丰富表示 |
| 7 | **位置编码 = 序列的灵魂** — 没有它, Transformer 只是"词袋模型" |

---

## 🧪 自测清单

- [ ] 1D 因果卷积 vs 普通卷积: 区别在哪? 为什么需要因果?
- [ ] RNN 的梯度为什么会消失? BPTT 中 ∂h_T/∂h_1 展开后是什么?
- [ ] LSTM 的 f_t, i_t, o_t 三个门各自的功能是什么? C_t 和 h_t 有什么区别?
- [ ] Bahdanau Attention 和 Luong Attention 的区别? (先算状态还是先算 attention?)
- [ ] Scaled Dot-Product Attention 为什么除以 √d_k?
- [ ] Multi-Head Attention 中 d_k = d_v = d_model / h 的原因?
- [ ] Self-Attention 和 Cross-Attention 有什么区别? Decoder 的 mask 怎么实现?
- [ ] Transformer 的 Encoder 和 Decoder 各包含哪些子层? 各自的输入输出是什么?
- [ ] Sinusoidal 位置编码为什么用 sin/cos 交替? PE_{pos+k} 和 PE_pos 有什么关系?

---

> L14 建立了从 CNN → RNN → Attention → Transformer 的完整序列建模视角。你已经理解了现代大语言模型 (GPT, BERT) 的核心积木 — Multi-Head Self-Attention 和 Transformer 架构。下一步将学习这些架构的大规模预训练方法。
