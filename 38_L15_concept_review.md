# CMU 10-708 Lecture 15 概念体系梳理 — 深度生成模型案例：文本生成

> 整理自 Eric Xing 教授 CMU 10-708 Lecture 15: Case Study of Deep Generative Models — Text Generation
>
> 核心教材: Goodfellow et al. (2016) Ch.10, Bengio et al. (2003), Sutskever et al. (2014), Vaswani et al. (2017)

---

## 📐 全局定位：从生成模型到文本生成

```
L12-L13: 深度生成模型 (通用框架)        L14: 序列模型 (架构积木)
    │                                       │
    └───────────────┬───────────────────────┘
                    │
                    ▼
      L15: 案例研究 — 文本生成 (Text Generation)
      
      = DGM 的生成能力 + 序列模型的结构
      = 把 VAE / GAN / Diffusion / AR 的思想落地到离散文本
      
核心问题: 如何让机器生成流畅、连贯、有意义的自然语言?
```

**一句话概括 L15**: 文本生成 = 自回归分解 P(x₁,...,x_T) = Π_t P(x_t | x_{<t}) + 解码策略 + 全局控制（VAE/GAN/Diffusion），核心挑战是离散 token 空间使梯度传播困难。

---

## 概念 1：自回归语言模型 (Autoregressive LM) — 文本生成的基础范式

---

### 1.1 自回归分解 (🔑🔑🔑)

```
语言建模 = 对序列的联合概率建模:

  P(x₁, x₂, ..., x_T) = P(x₁) · P(x₂|x₁) · P(x₃|x₁,x₂) · ... · P(x_T|x₁,...,x_{T-1})
                       = Π_{t=1}^T P(x_t | x_{<t})

x_t ∈ V (词汇表, |V| ≈ 10⁴ ~ 10⁵)
x_{<t} = (x₁, ..., x_{t-1}) — "上文" / 前缀

生成过程 (逐 token):
  x₁ ~ P(·)
  x₂ ~ P(·|x₁)
  x₃ ~ P(·|x₁, x₂)
  ...

本质: 序列生成 → 重复做 |V|-way 分类!
```

### 1.2 n-gram 语言模型 (统计方法)

```
Markov 假设: 当前词只依赖前 n-1 个词

  Bigram (n=2):  P(x_t | x_{<t}) ≈ P(x_t | x_{t-1})
  Trigram (n=3): P(x_t | x_{<t}) ≈ P(x_t | x_{t-2}, x_{t-1})

MLE 估计 (计数):
  P(x_t | x_{t-1}) = count(x_{t-1}, x_t) / count(x_{t-1})

问题:
  - 数据稀疏: 大量 n-gram 从未出现 → 概率为 0
  - 平滑技术: Add-1, Kneser-Ney, Good-Turing
  - 上下文窗口固定: n=3 只能看前 2 个词 → 长距离依赖丢失!
```

### 1.3 神经语言模型 (Neural LM) (🔑🔑)

```
核心洞见: 用神经网络替代计数表, 学习词的连续表示!

Bengio et al. (2003) — 第一个神经语言模型:
  Input:  x_{t-n+1}, ..., x_{t-1}  (前 n-1 个词)
  Embed:  每个词 → e ∈ R^d  (词向量, 分布式表示)
  Hidden: h = tanh(W·[e_{t-n+1}; ...; e_{t-1}] + b)
  Output: P(x_t = v | context) = softmax(W_out·h)[v]

优势:
  ✅ 词向量捕获语义相似性 → "cat" 和 "kitty" 共享统计强度
  ✅ 可处理更长上下文 (RNN / Transformer)
  ✅ 泛化到未见过的 n-gram (通过词向量组合)

代价:
  ❌ |V| 很大时 softmax 极贵 → 分层 softmax, NCE, sampled softmax
```

### 1.4 GPT 式自回归生成 (🔑🔑🔑)

```
GPT = Generative Pre-trained Transformer

结构: Decoder-only Transformer (Masked Self-Attention)
训练: 给定上文, 预测下一个 token (Language Modeling)
  L = -Σ_{t=1}^{T-1} log P(x_{t+1} | x₁, ..., x_t)

生成 (Inference):
  prompt = "The cat sat on the"
  1. 输入 prompt, 得到 P(x_next | prompt)
  2. 采样/选择 x_next
  3. prompt = prompt + x_next, 重复

关键设计:
  - Masked Self-Attention: 时刻 t 只能看 ≤ t 的位置 (因果!)
  - Positional Encoding: 注入位置信息
  - 每次预测只看过去, 不能看未来
```

---

## 概念 2：Seq2Seq + Attention — 条件文本生成

---

### 2.1 Seq2Seq 架构 (🔑🔑)

```
条件文本生成: 给定 source X, 生成 target Y
  - 机器翻译:  X="Hello" → Y="Bonjour"
  - 摘要:      X=长文章 → Y=短摘要
  - 对话:      X=用户消息 → Y=回复

Seq2Seq (Sutskever et al., 2014):
  ┌─────────────────┐     ┌─────────────────┐
  │ Encoder (RNN)   │     │ Decoder (RNN)   │
  │                 │     │                 │
  │ "The cat sat"   │ --> │ h_T (context)   │ --> "Le chat"
  │  ↓  ↓  ↓  ↓    │     │        ↓        │
  │ h₁ h₂ h₃ h₄    │     │  s₁ → s₂ → s₃  │
  └─────────────────┘     └─────────────────┘

问题: 所有源信息被压缩到一个向量 h_T → 信息瓶颈!
```

### 2.2 Seq2Seq + Attention (🔑🔑🔑)

```
Decoder 每个时刻动态回顾所有 Encoder 状态:

  e_{t,j} = score(s_{t-1}, h_j)        ← 对齐分数
  α_{t,j} = softmax(e_t)_j             ← 注意力权重 (Σ_j α = 1)
  c_t     = Σ_j α_{t,j} · h_j          ← 上下文向量 (加权和)

  s_t = RNN(s_{t-1}, [y_{t-1}, c_t])
  y_t ~ softmax(W·[s_t, c_t])

直观理解:
  生成 "chat" 时, Decoder 重点关注 Encoder 的 "sat" 位置!
  → 解决了信息瓶颈, 实现了软对齐
```

---

## 概念 3：解码策略 (Decoding Strategies) — 从概率到文本

---

### 3.1 贪心解码 vs 集束搜索 (🔑🔑)

```
Greedy Decoding: 每步选概率最大的 token
  y_t = argmax P(y | y_{<t}, X)
  
  问题: 局部最优 ≠ 全局最优!
  例: y₁="a" 概率高, 但后续根本无法生成好句子
      y₁="the" 概率略低, 但后续生成空间广阔!

Beam Search: 每步保留 B 个最优候选序列
  B = beam width

  时刻 t, 维护 B 个部分序列 {y^{(1)}, ..., y^{(B)}}
  对每个序列的 B×|V| 个扩展, 保留最高分的 B 个

  例 (B=2, |V|=5):
    t=1: (the, -0.3), (a, -0.5)       ← 保留2个
    t=2: 各扩展 5 个 = 10 个候选
         保留: (the cat, -0.8), (a dog, -1.1)
    ...
```

### 3.2 Beam Search 详解

```
算法:
  beams = [([], score=0)]   # (序列, log概率和)

  for t = 1 to T:
      candidates = []
      for (seq, score) in beams:
          probs = model.get_probs(seq)   # P(x_t | seq), shape (|V|,)
          for v in vocab:
              candidates.append((seq + [v], score + log probs[v]))
      
      # 按 score 排序, 保留前 B 个
      candidates.sort(key=score, reverse=True)
      beams = candidates[:B]

  返回最高分的完整序列

分数是 log P(y) = Σ_t log P(y_t | y_{<t})
  → 由于每个 log P ≤ 0, 长序列得分更低 (趋于负无穷)
  → 实践: 用平均 log 概率或 length penalty 归一化
```

### 3.3 温度采样 (Temperature Sampling) (🔑)

```
Temperature τ: 控制输出的"创造性"程度

  P_τ(x_t | x_{<t}) = softmax(logits / τ)

  τ → 0:  分布趋近 one-hot → 确定性 (等价于 greedy)
  τ = 1:  原始分布
  τ → ∞:  分布趋近均匀 → 完全随机

效果:
  τ = 0.3: "The cat sat on the mat."     ← 保守, 高概率词
  τ = 0.8: "The cat sat on the mat."     ← 平衡
  τ = 1.5: "A feline rest upon carpet."   ← 多样化
  τ = 3.0: "Quantum potato democracy..."  ← 胡言乱语!
```

### 3.4 Top-k & Top-p (Nucleus) 采样 (🔑🔑)

```
问题: 纯温度采样中, 长尾的极低概率词 (= 噪声) 也会被采样

Top-k Sampling:
  每步只从概率最高的 k 个 token 中采样 (k ≈ 40)
  → 截断长尾, 保持输出可控

Top-p (Nucleus) Sampling:
  每步累加概率直到 ≥ p, 只在这些 token 中采样 (p ≈ 0.9)
  → 自适应截断: 分布集中时保留少, 分散时保留多

对比:
  Greedy:     确定性, 容易陷入重复 → "I am a cat. I am a cat. I am..."
  Temperature: 有随机性, 但长尾噪声 → 偶尔出奇怪词
  Top-k:      固定截断, 分布尖锐时仍保留无关词
  Top-p:      动态截断, 自动适应分布形状 → 生成质量最高!
```

---

## 概念 4：VAE for Text — 隐变量控制生成

---

### 4.1 为什么需要 VAE? (🔑)

```
标准自回归 LM 的问题:
  - 没有全局控制: 无法指定"风格"/"主题"/"情感"
  - 生成 = 逐 token 采样 → 局部决策, 无全局规划

VAE for Text 的想法:
  引入全局隐变量 z 控制生成:
    z ~ P(z)            ← 隐变量 (编码主题/风格)
    x ~ P(x | z)        ← 自回归解码器

生成:
  z ~ N(0, I)           → 采样一个"主题"
  x ~ AR_Decoder(z)     → 根据主题逐 token 生成
  
插值: z_α = z_1·(1-α) + z_2·α → 生成"介于两者之间"的句子
```

### 4.2 文本 VAE 的核心困难 (🔑🔑)

```
问题: 后验崩塌 (Posterior Collapse)!

  标准 VAE: L = E_q(z|x)[log P(x|z)] - KL(q(z|x) || P(z))
            └─── 重构损失 ───┘   └──── KL 正则 ────┘

  对于文本:
    解码器 (AR LM) 极其强大 → 可以"忽视 z" 直接生成!
    → q(z|x) 趋近 P(z) = N(0,I) → KL → 0
    → z 不包含任何信息 → VAE 退化为普通 LM

解法:
  1. KL Annealing: 训练初期降低 KL 权重, 后期逐渐增大
  2. Word Dropout: 随机丢弃 decoder 输入 → 强制依赖 z
  3. Free Bits: 保证 KL ≥ λ (每个维度至少 λ 的信息量)
  4. δ-VAE / β-VAE: 调整 KL 权重
```

### 4.3 离散隐变量的替代方案

```
离散 z (如类别变量) 天然适合文本:
  - 每个类别 = 一个"写作风格"/"主题"
  - 不强制连续性 (不像高斯)

训练: Gumbel-Softmax trick (可微的离散采样)
  z = softmax((log π + g) / τ)   ← τ→0 退化为 one-hot
  g ~ Gumbel(0,1)

或直接用 Vector Quantized VAE (VQ-VAE):
  z = argmin ||z_e(x) - e_k||   ← 离散码本
  decoder 用离散 z 的条件序列生成
```

---

## 概念 5：GAN for Text — 对抗训练生成文本

---

### 5.1 核心挑战 (🔑🔑)

```
GAN 在图像上成功:
  Generator: z → image (连续空间, 可微)
  Discriminator: image → real/fake (可微)

文本的困境:
  Generator: z → token序列 (离散空间, 不可微!)
  
  argmax/采样 操作没有梯度 → D 的梯度无法回传 G!

类比: 如果图像 GAN 中每个像素只能选 [0,1...,255],
       且选择操作不可微 — 同样的困境!
```

### 5.2 SeqGAN — 用策略梯度解决 (🔑🔑🔑)

```
核心洞见: 把文本生成当成强化学习问题!

  State:     已生成的部分序列 y_{<t}
  Action:    选择下一个 token y_t
  Policy:    G(y_t | y_{<t}) — 生成器
  Reward:    D(y_{1:T}) — 判别器对完整序列的评分

用 Policy Gradient (REINFORCE) 训练 G:

  ∇J ≈ E[ Σ_t ∇log G(y_t | y_{<t}) · R_t ]

  R_t = D(y_{1:T}) 或 Monte Carlo rollout 估计的 Q 值

技巧:
  - 用 MLE 预训练 G (保证初始生成不全是垃圾)
  - 交替训练 D 和 G (同 GAN)
  - D 用 CNN/RNN 对序列做二分类 (real/fake)
```

### 5.3 GAN vs VAE vs AR 在文本生成中的对比

| | AR (GPT) | VAE | GAN (SeqGAN) |
|---|---|---|---|
| **训练目标** | MLE (交叉熵) | ELBO | Minimax |
| **隐变量** | 无 | 连续 z | 连续 z (Generator 输入) |
| **生成质量** | 最高 (流畅) | 中等 (后验崩塌) | 较低 (训练不稳定) |
| **多样性** | 中等 (mode collapse 轻) | 高 (z 随机) | 低 (mode collapse 重) |
| **可控性** | 低 (prompt only) | 高 (z 插值) | 中 |
| **训练难度** | 低 | 中 (后验崩塌) | 高 (不可微+不稳定) |

---

## 概念 6：评估指标 — 如何衡量生成文本的质量?

---

### 6.1 Perplexity (困惑度) (🔑🔑)

```
定义:
  PPL = exp( -1/T · Σ_{t=1}^T log P(x_t | x_{<t}) )
      = exp(CrossEntropy)

直觉: 模型在每个位置, 平均需要在多少个等可能的选项中"困惑"地选择

例: 完美模型 P(x_t|x_{<t}) = 1 → PPL = 1 (不困惑)
    均匀分布 P(x_t) = 1/|V| = 1/10000 → PPL = 10000 (极度困惑)

PPL 越低越好 (越不困惑)

注意:
  ✅ 客观, 不依赖参考文本
  ❌ 不直接衡量语义质量 (高PPL ≠ 不通顺, 低PPL ≠ 有意义)
  ❌ 不能跨词汇表大小比较
```

### 6.2 BLEU (Bilingual Evaluation Understudy) (🔑🔑)

```
BLEU = 机器翻译的自动评估指标 (也广泛用于文本生成)

核心思想: 生成文本和参考文本的 n-gram 重叠度

Step 1: n-gram Precision
  P_n = Σ_{n-gram} min(count_gen(n-gram), count_ref(n-gram))
        ─────────────────────────────────────────────────
                    Σ_{n-gram} count_gen(n-gram)
  
  (截断计数: 每个 n-gram 的匹配数不超过参考中的出现次数)

Step 2: Brevity Penalty (过短惩罚)
  BP = 1                          if c ≥ r
  BP = exp(1 - r/c)               if c < r
  (c=生成长度, r=参考长度)

Step 3: 最终 BLEU
  BLEU = BP · exp( Σ_{n=1}^4 w_n · log P_n )
  
  通常 w_n = 1/4 (均匀加权 unigram~4-gram)

BLEU ∈ [0, 1], 越高越好

优点: 快速, 自动, 多 n-gram 覆盖
缺点: 不同表达 (同义) 得 0 分, 不捕捉语义等价
```

### 6.3 其他指标速览

| 指标 | 全称 | 主要用途 | 核心思想 |
|------|------|---------|---------|
| **Perplexity** | — | 语言模型 | 下一个词的平均困惑度 |
| **BLEU** | Bilingual Evaluation Understudy | 翻译 | n-gram 精确匹配 |
| **ROUGE** | Recall-Oriented Understudy for Gisting Evaluation | 摘要 | n-gram 召回率 |
| **METEOR** | Metric for Evaluation of Translation with Explicit ORdering | 翻译 | 同义词匹配 + 词序惩罚 |
| **CIDEr** | Consensus-based Image Description Evaluation | 图像描述 | TF-IDF 加权 n-gram |
| **BERTScore** | — | 通用 | BERT 上下文嵌入的余弦相似度 |

---

## 📋 文本生成方法一张表

| 方法 | 核心思想 | 优势 | 劣势 | 代表模型 |
|------|---------|------|------|---------|
| **n-gram LM** | 计数 + Markov 假设 | 简单, 可解释 | 数据稀疏, 短上下文 | KenLM |
| **RNN LM** | 循环状态编码历史 | 长上下文, 词向量 | 串行慢, 梯度问题 | LSTM-LM |
| **Transformer LM (GPT)** | Masked Self-Attention | 并行, 长距, 强 | O(T²), 单向 | GPT-2/3/4 |
| **Seq2Seq+Attn** | Encoder-Decoder + 动态对齐 | 条件生成, 对齐可解释 | 推理串行 | GNMT, Transformer |
| **VAE for Text** | 隐变量 z → 解码 | 可控生成, 插值 | 后验崩塌 | Optimus, CTRL |
| **SeqGAN** | RL 训练生成器 | 对抗性训练 | 不稳定, 梯度方差大 | SeqGAN, LeakGAN |
| **Diffusion for Text** | 逐步去噪离散 token | 生成的渐进性 | 离散扩散仍在探索 | D3PM, Diffusion-LM |

---

## 🔗 概念关系图

```
                 文本生成 (L15)
                      │
        ┌─────────────┼─────────────┐
        ▼              ▼              ▼
   无条件生成      条件生成       可控生成
   (LM)          (Seq2Seq)      (VAE/GAN)
        │              │              │
        ▼              ▼              ▼
   自回归分解    Encoder-Decoder  隐变量 z
   P(x_t|x_{<t})  + Attention     + 解码器
        │              │              │
        └──────────────┼──────────────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
          解码策略           评估指标
    Greedy / Beam /       PPL / BLEU
    Temp / Top-k / Top-p  ROUGE / BERTScore
```

---

## 🎯 核心洞见一句话

| # | 洞见 |
|---|------|
| 1 | **文本生成 = 自回归分解 + 解码策略** — 任何文本都可分解为逐 token 的条件概率连乘 |
| 2 | **n-gram → Neural LM = 从计数到表示** — 词向量让模型理解了"cat ≈ kitty"而不是"cat ≠ kitty" |
| 3 | **Beam Search 不等于最优** — B 个局部最优候选 ≠ 全局最优序列; 且 B 越大, 生成越短越"安全" |
| 4 | **Temperature 是创造性的旋钮** — τ 小=保守流畅, τ 大=多样但可能胡言乱语 |
| 5 | **Top-p 比 Top-k 更聪明** — 动态截断自动适应分布形状, 分布尖锐时截少, 平坦时保留多 |
| 6 | **文本 VAE 的敌人是后验崩塌** — 强 AR decoder 会忽视 z, 需要 KL annealing / word dropout 对抗 |
| 7 | **文本 GAN 的核心挑战是不可微** — argmax → 策略梯度, 但方差大、不稳定 |
| 8 | **Perplexity 低 ≠ 文本好** — PPL 衡量的是"模型不困惑", 但人类判断的是"文本有意义" |
| 9 | **BLEU 只看表面匹配** — "The cat sat" vs "A feline rested" → BLEU=0, 但语义等价! |

---

## 🧪 自测清单

- [ ] 自回归分解: P(x₁,...,x_T) 如何分解为条件概率的乘积?
- [ ] n-gram LM 的 Markov 假设: 为什么 n 不能太大?
- [ ] 神经 LM 相比 n-gram LM 的核心优势是什么? (词向量, 长上下文)
- [ ] GPT 的 Masked Self-Attention 如何保证"不能偷看未来"?
- [ ] Seq2Seq: Encoder 和 Decoder 各自的输入输出是什么? Attention 解决了什么问题?
- [ ] Beam Search (B=3) 在 |V|=10 时, 每步评估多少个候选?
- [ ] Temperature τ=0.1 vs τ=2.0: 分别产生什么样的生成效果?
- [ ] Top-p 和 Top-k 的区别是什么? 为什么 Top-p 通常更好?
- [ ] 文本 VAE 的后验崩塌 (posterior collapse) 是什么? 如何缓解?
- [ ] SeqGAN 为什么需要策略梯度? argmax 的梯度问题如何解决?
- [ ] BLEU 的 brevity penalty 是为了解决什么问题?
- [ ] Perplexity=50 意味着什么? PPL=1 又意味着什么?

---

> L15 将 L12-L13 的深度生成模型和 L14 的序列模型融合到文本生成这一具体应用场景,
> 展示了从统计 n-gram 到神经 AR 模型、从无条件生成到可控生成 (VAE/GAN) 的完整技术脉络。
> 理解了自回归分解、解码策略和评估指标, 你就理解了现代大语言模型 (GPT 系列) 的核心运作原理。
