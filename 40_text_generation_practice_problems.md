# CMU 10-708 Lecture 15 课后练习 & 答案 — 文本生成

> 配套教材: Goodfellow et al. (2016) Ch.10, Bengio et al. (2003), Sutskever et al. (2014), Vaswani et al. (2017)
>
> 题目覆盖 L15 六大主题, 三级: 🟢 基础 | 🟡 进阶 | 🔴 挑战

---

# 第一部分: 自回归语言模型

---

## 🟢 Q1. 自回归分解

一个长度为 T=4 的序列 "the cat sat mat", 词汇表大小 V=10000。

**(A)** 写出自回归分解 P("the cat sat mat") 的完整表达式。

**(B)** 如果每个条件概率是均匀的 (P(x_t | x_{<t}) = 1/V), 这个序列的 log 概率是多少?

**(C)** 为什么自回归分解总是合法的概率分解? (提示: 链式法则)

<details>
<summary>点击查看答案</summary>

**(A) 自回归分解:**

```
P(the, cat, sat, mat)
  = P(the) · P(cat | the) · P(sat | the, cat) · P(mat | the, cat, sat)
  = Π_{t=1}^{4} P(x_t | x_{<t})
```

其中 P(x₁ | x_{<1}) = P(x₁)（第一个词只依赖 <SOS> 或无条件）。

**(B) 均匀分布:**

```
P(x_t | x_{<t}) = 1/10000 = 10⁻⁴

log P = Σ_t log(10⁻⁴) = 4 × (-4 × log 10)
      = 4 × (-4 × 2.3026)
      = 4 × (-9.2103)
      = -36.8414
```

**(C) 为什么合法:**

自回归分解是概率链式法则的直接应用:

```
P(A, B, C) = P(A)·P(B|A)·P(C|A,B)

这是概率论基本定理, 对任意 N 个变量恒成立。
不需要任何独立性假设!

关键: 每一项 P(x_t | x_{<t}) 是有效的条件概率分布 (≥0, Σ=1)
     → 乘积自然是有效联合分布。

这就是为什么"自回归"是文本生成的通用框架:
  你只需要学会预测"下一个词", 就能生成任意长的文本!
```

</details>

---

## 🟢 Q2. Bigram LM 概率计算

给定以下训练语料: "the cat sat the cat ran the dog sat"

**(A)** 构建词汇表, 计算 bigram 计数矩阵。

**(B)** 用 MLE (无平滑) 计算 P(sat | the) 和 P(ran | cat)。

**(C)** 对于测试序列 "the cat ran", 用 MLE bigram 计算 log P。这个问题是什么? 如何解决?

<details>
<summary>点击查看答案</summary>

**(A) 词汇表 & 计数:**

```
语料: the cat sat the cat ran the dog sat
词汇表 (V=6): {the, cat, sat, ran, dog}

Bigram 对:
  (the, cat): 1
  (cat, sat): 1
  (sat, the): 1
  (the, cat): 1   ← 第 2 次
  (cat, ran): 1
  (ran, the): 1
  (the, dog): 1
  (dog, sat): 1

合并计数:
  the → cat: 2, dog: 1,   total(the→*) = 3
  cat → sat: 1, ran: 1,   total(cat→*) = 2
  sat → the: 1,           total(sat→*) = 1
  ran → the: 1,           total(ran→*) = 1
  dog → sat: 1,           total(dog→*) = 1
```

**(B) MLE 概率:**

```
P(sat | the) = count(the, sat) / count(the) = 0/3 = 0  ← 没见过!
P(ran | cat) = count(cat, ran) / count(cat) = 1/2 = 0.5
```

**(C) 测试序列 "the cat ran":**

```
log P = log P(the) + log P(cat|the) + log P(ran|cat)
      = log(3/8) + log(2/3) + log(1/2)      ← 3个the/8个词, etc.
      = -0.981 + (-0.405) + (-0.693)
      = -2.079

但是! 如果测试是 "the cat sat":
  P(sat|cat) = 1/2 → OK
但如果 "the dog ran":
  P(ran|dog) = 0/1 = 0 → log = -∞  ← 零概率问题!

解决方案:
  - Add-1 (Laplace) 平滑: P(w_j|w_i) = (count+1)/(total+V)
  - Add-k 平滑
  - Kneser-Ney 平滑 (最先进的 n-gram 平滑)
  - Backoff: 回退到 unigram
  - 或直接用神经 LM — 词向量天然平滑!
```

</details>

---

# 第二部分: 解码策略

---

## 🟡 Q3. Beam Search 手算

给定以下简化的语言模型 (V=4, 词汇表: a, b, c, d), Beam width B=2, 生成长度 T=3 (不包含 prompt "a")。

条件概率表:

| 上文 | P(a) | P(b) | P(c) | P(d) |
|------|------|------|------|------|
| a (prompt) | 0.1 | 0.4 | 0.3 | 0.2 |
| b | 0.1 | 0.1 | 0.5 | 0.3 |
| c | 0.3 | 0.2 | 0.1 | 0.4 |
| d | 0.2 | 0.4 | 0.3 | 0.1 |

**(A)** 写出 t=1, t=2, t=3 时 beams 的状态 (每个候选的序列和 log 概率)。

**(B)** 最终选出的最优序列是什么? 它真的是全局最优吗?

**(C)** 如果 B=1 (退化为 greedy), 会输出什么? 和 B=2 的结果有何不同?

<details>
<summary>点击查看答案</summary>

**(A) Beam Search 过程:**

```
初始:
  beams = [([a], 0.0)]

t=1 (生成第1个新token):
  从 "a" 出发, 4 个候选:
    [a, a]: log 0.1 = -2.303
    [a, b]: log 0.4 = -0.916
    [a, c]: log 0.3 = -1.204
    [a, d]: log 0.2 = -1.609

  保留 B=2 个最高分:
    beams = [([a,b], -0.916), ([a,c], -1.204)]

t=2 (生成第2个新token):
  从 "a,b" 出发:
    [a,b,a]: -0.916 + log 0.1 = -0.916 + (-2.303) = -3.219
    [a,b,b]: -0.916 + log 0.1 = -3.219
    [a,b,c]: -0.916 + log 0.5 = -0.916 + (-0.693) = -1.609
    [a,b,d]: -0.916 + log 0.3 = -0.916 + (-1.204) = -2.120

  从 "a,c" 出发:
    [a,c,a]: -1.204 + log 0.3 = -1.204 + (-1.204) = -2.408
    [a,c,b]: -1.204 + log 0.2 = -1.204 + (-1.609) = -2.813
    [a,c,c]: -1.204 + log 0.1 = -1.204 + (-2.303) = -3.507
    [a,c,d]: -1.204 + log 0.4 = -1.204 + (-0.916) = -2.120

  8 个候选中保留 B=2:
    beams = [([a,b,c], -1.609), ([a,b,d], -2.120)]
    (注意: [a,c,d] 也是 -2.120, 并列取第一个)

t=3 (生成第3个新token):
  从 "a,b,c" 出发:
    [a,b,c,a]: -1.609 + log 0.3 = -1.609 + (-1.204) = -2.813
    [a,b,c,b]: -1.609 + log 0.2 = -1.609 + (-1.609) = -3.218
    [a,b,c,c]: -1.609 + log 0.1 = -1.609 + (-2.303) = -3.912
    [a,b,c,d]: -1.609 + log 0.4 = -1.609 + (-0.916) = -2.525

  从 "a,b,d" 出发:
    [a,b,d,a]: -2.120 + log 0.2 = -2.120 + (-1.609) = -3.729
    [a,b,d,b]: -2.120 + log 0.4 = -2.120 + (-0.916) = -3.036
    [a,b,d,c]: -2.120 + log 0.3 = -2.120 + (-1.204) = -3.324
    [a,b,d,d]: -2.120 + log 0.1 = -2.120 + (-2.303) = -4.423

  保留 B=2:
    beams = [([a,b,c,a], -2.813), ([a,b,c,d], -2.525)]

  等等, 我重新排序: -2.525 > -2.813 > -3.036 > ...
    beams = [([a,b,c,d], -2.525), ([a,b,c,a], -2.813)]
```

**(B) 最优序列:**

```
[a, b, c, d] (logP = -2.525) = "a b c d"

这是 Beam Search 找到的最优序列 — 但它是全局最优吗?
不一定! B=2 时我们丢弃了很多路径, 可能有更好的:
  例: [a, d, b, c] 在 t=1 就被丢弃了 (因为 [a,d] 分数低)
  但也许后续 P(c|a,d,b) 极高, 导致全局更优!

Beam Search 不能保证全局最优 (那是 NP-hard),
但实践中 B≈5-10 已经能找到很好的近似解。
```

**(C) B=1 (Greedy):**

```
t=1: 选 b (max prob=0.4)             → [a, b]
t=2: 从 b, 选 c (max prob=0.5)       → [a, b, c]
t=3: 从 c, 选 d (max prob=0.4)       → [a, b, c, d]

输出: "a b c d" (logP = -2.525)

巧合: 本例中 Greedy 和 B=2 Beam Search 结果相同。
但这不总是成立 — Greedy 每步只看眼前, 可能陷入"短视"。
```

</details>

---

## 🟡 Q4. Temperature 的效果

一个 3 分类问题 (V=3, 词汇: A, B, C), 当前 logits = [2.0, 0.5, -1.0]。

**(A)** 计算 τ = 1.0, 0.5, 2.0 时的概率分布。

**(B)** τ → 0 和 τ → ∞ 时, 分布分别趋向什么?

**(C)** 为什么低 τ "更确定" 但容易"重复"? (联系 GPT 的退化现象)

<details>
<summary>点击查看答案</summary>

**(A) 各温度下的概率:**

```
原始 logits: [2.0, 0.5, -1.0]

τ = 1.0 (标准):
  直接 softmax(logits):
    exp(2.0)=7.389, exp(0.5)=1.649, exp(-1.0)=0.368
    sum = 9.406
    P = [7.389/9.406, 1.649/9.406, 0.368/9.406]
      = [0.786, 0.175, 0.039]

τ = 0.5 (低温, 更"确定"):
  scaled = [2.0/0.5, 0.5/0.5, -1.0/0.5] = [4.0, 1.0, -2.0]
  exp(4.0)=54.598, exp(1.0)=2.718, exp(-2.0)=0.135
  sum = 57.451
  P = [54.598/57.451, 2.718/57.451, 0.135/57.451]
    = [0.950, 0.047, 0.003]  ← A 几乎独占!

τ = 2.0 (高温, 更"随机"):
  scaled = [2.0/2.0, 0.5/2.0, -1.0/2.0] = [1.0, 0.25, -0.5]
  exp(1.0)=2.718, exp(0.25)=1.284, exp(-0.5)=0.607
  sum = 4.609
  P = [2.718/4.609, 1.284/4.609, 0.607/4.609]
    = [0.590, 0.279, 0.132]  ← 分布更均匀
```

**(B) 极限行为:**

```
τ → 0:  所有概率集中到 argmax
         P → [1, 0, 0] (等价于 greedy)
         
         产生"确定性退化": 每次选同一个 → "I am a cat. I am a cat. I am..."

τ → ∞:  所有概率趋于均匀
         P → [1/3, 1/3, 1/3]
         
         完全随机 → 胡言乱语: "Quantum potato democratically..."
```

**(C) 低 τ 的退化问题:**

```
当 τ 很低时, 模型倾向于选择"最安全"的词 (概率最高的)。

在自回归生成中:
  - 第 1 步选 "I" (概率最高)
  - 第 2 步选 "am" (在 "I" 条件下概率最高)
  - 第 3 步选 "a"   ...等等

问题是: "最安全"的序列往往会形成循环!
  "I am a cat. I am a cat. I am..." ← GPT-2 早期版本的典型退化

原因:
  - 重复序列 (如 "I am a cat.") 在训练语料中频繁出现
  - 条件概率 P("I am a cat." | "I am a cat.") 其实不低!
  - Greedy 被困在这个"高概率循环"中

解决: 用 Top-p sampling 引入可控随机性, 打破循环!
```

</details>

---

# 第三部分: 评估指标

---

## 🟡 Q5. Perplexity 计算

测试集有 3 个序列, LM 给出的条件概率如下:

| 序列 | P(w₁) | P(w₂\|w₁) | P(w₃\|w₁,w₂) | P(w₄\|w₁,w₂,w₃) |
|------|-------|-----------|---------------|-------------------|
| S1 (T=3): "the cat sat" | 0.3 | 0.5 | 0.4 | — |
| S2 (T=4): "the dog ran fast" | 0.3 | 0.3 | 0.2 | 0.5 |
| S3 (T=3): "a mat on" | 0.2 | 0.1 | 0.05 | — |

**(A)** 计算每个序列的 log 概率和 perplexity。

**(B)** 计算整个测试集的 perplexity (词级别)。

**(C)** S3 的 PPL 最高, 为什么? 有哪些可能原因?

<details>
<summary>点击查看答案</summary>

**(A) 每个序列:**

```
S1: "the cat sat"
  log P = log(0.3) + log(0.5) + log(0.4)
        = -1.204 + (-0.693) + (-0.916)
        = -2.813
  NLL = 2.813 / 3 = 0.938
  PPL = exp(0.938) = 2.56

S2: "the dog ran fast"
  log P = log(0.3) + log(0.3) + log(0.2) + log(0.5)
        = -1.204 + (-1.204) + (-1.609) + (-0.693)
        = -4.710
  NLL = 4.710 / 4 = 1.178
  PPL = exp(1.178) = 3.25

S3: "a mat on"
  log P = log(0.2) + log(0.1) + log(0.05)
        = -1.609 + (-2.303) + (-2.996)
        = -6.908
  NLL = 6.908 / 3 = 2.303
  PPL = exp(2.303) = 10.0
```

**(B) 测试集 PPL (词级):**

```
总词数: 3 + 4 + 3 = 10
总 log P: -2.813 + (-4.710) + (-6.908) = -14.431
NLL = 14.431 / 10 = 1.443
PPL = exp(1.443) = 4.23
```

**(C) S3 PPL 高的原因:**

```
PPL=10 意味着模型平均在 10 个词中"困惑"地选择。

可能原因:
  1. "a mat on" 在训练语料中罕见 (a mat on what?)
  2. 它是语法不完整的片段 (on 后面通常还有词)
  3. 模型没见过 "a mat" 这样的组合

注意: PPL=10 不意味着这句话"错"了 —
       只是说模型认为这个序列"出乎意料"。
```

</details>

---

## 🟡 Q6. BLEU 手算

候选: "the cat sat on the mat"
参考: "the cat is on the mat"

**(A)** 计算 1-gram precision (P₁)。列出匹配的 unigram。

**(B)** 计算 2-gram precision (P₂)。列出匹配的 bigram。

**(C)** 计算 3-gram 和 4-gram precision, 然后计算 BLEU (w_n=1/4)。

**(D)** 为什么 BLEU=1 也不一定意味着翻译完美?

<details>
<summary>点击查看答案</summary>

**(A) P₁ (unigram):**

```
候选 unigrams: the(N=2), cat(1), sat(1), on(1), mat(1)
参考 unigrams: the(N=2), cat(1), is(1), on(1), mat(1)

逐个匹配 (clipped by reference count):
  the: count_cand=2, count_ref=2 → clipped=2
  cat: count_cand=1, count_ref=1 → clipped=1
  sat: count_cand=1, count_ref=0 → clipped=0
  on:  count_cand=1, count_ref=1 → clipped=1
  mat: count_cand=1, count_ref=1 → clipped=1

clipped_sum = 2+1+0+1+1 = 5
total = 6 (候选总词数)

P₁ = 5/6 = 0.833
```

**(B) P₂ (bigram):**

```
候选 bigrams:
  (the,cat):1, (cat,sat):1, (sat,on):1, (on,the):1, (the,mat):1

参考 bigrams:
  (the,cat):1, (cat,is):1, (is,on):1, (on,the):1, (the,mat):1

匹配:
  (the,cat): cand=1, ref=1 → 1
  (cat,sat): cand=1, ref=0 → 0
  (sat,on):  cand=1, ref=0 → 0
  (on,the):  cand=1, ref=1 → 1
  (the,mat): cand=1, ref=1 → 1

clipped_sum = 3
total = 5

P₂ = 3/5 = 0.600
```

**(C) P₃, P₄ & BLEU:**

```
候选 3-grams:
  (the,cat,sat):1, (cat,sat,on):1, (sat,on,the):1, (on,the,mat):1

参考 3-grams:
  (the,cat,is):1, (cat,is,on):1, (is,on,the):1, (on,the,mat):1

匹配: (on,the,mat) 匹配 → 1
其他都不匹配

P₃ = 1/4 = 0.250

候选 4-grams:
  (the,cat,sat,on):1, (cat,sat,on,the):1, (sat,on,the,mat):1

参考 4-grams:
  (the,cat,is,on):1, (cat,is,on,the):1, (is,on,the,mat):1

匹配: 无! → 0

P₄ = 0/3 = 0  ← 注意: log(0) = -∞!

实际 BLEU 实现中, 如果 P_n = 0, 通常用一个小值或跳过该 n。

Brevity Penalty:
  c = 6, r = 6 → BP = 1.0 (等长, 无惩罚)

如果忽略 P₄=0 的问题, 只算 unigram~trigram:
  log P₁ = log 0.833 = -0.182
  log P₂ = log 0.600 = -0.511
  log P₃ = log 0.250 = -1.386

  avg log = (-0.182 + (-0.511) + (-1.386)) / 3 = -0.693
  BLEU₃ = 1.0 × exp(-0.693) = 0.500
```

**(D) BLEU=1 也不完美:**

```
BLEU=1 仅在所有 n-gram 精确匹配参考时发生。

问题:
  1. 只衡量 n-gram 重叠, 不衡量语义: "the feline rested" vs "the cat sat" → BLEU≈0
  2. 不评价格式/流畅度: "cat the mat on sat the" 修改词序仍有相同 unigram P₁
     (这也是为什么需要 P₂, P₃, P₄! 高阶 n-gram 部分捕捉词序)
  3. 不衡量事实准确性: "the cat sat on the sun" vs 参考 "the cat sat on the mat"
     BLEU 会因 "the cat sat on the" 的高阶匹配而给出不低的分数!
  4. 对同义表达零容忍
```

</details>

---

# 第四部分: Seq2Seq + Attention

---

## 🔴 Q7. Bahdanau Attention 维度推导

一个 Seq2Seq 模型:
- Source 序列长度 S=7
- Target 序列长度 T=5
- Encoder hidden_dim = 256 (双向 → 512)
- Decoder hidden_dim = 512
- 使用 Bahdanau (Additive) Attention

**(A)** 写出 Attention score e_{t,j} 的计算公式和涉及的参数矩阵维度。

**(B)** 写出上下文向量 c_t 的计算过程 (从 e_{t,j} 到 c_t)。

**(C)** 计算 Attention 部分的总参数量。

<details>
<summary>点击查看答案</summary>

**(A) Bahdanau Attention:**

```
e_{t,j} = v^T · tanh(W_a · [s_{t-1}; h_j] + b_a)

其中:
  s_{t-1}: decoder 前一时刻状态, 维度 (512,)
  h_j:    encoder 第 j 个状态, 维度 (512,)  [双向拼接后]
  [s_{t-1}; h_j]: 拼接, 维度 (1024,)

  W_a: 权重矩阵, 维度 (hidden_attn, 1024), 通常 hidden_attn = decoder_dim
       此处 W_a ∈ R^{512 × 1024}

  b_a: 偏置, 维度 (512,)
  v:   投影向量, 维度 (512,)
  e_{t,j}: 标量!  (一个实数)

注意: W_a, b_a, v 在所有时间步和源位置间共享!
```

**(B) c_t 计算:**

```
1. 对每个源位置 j=1..S, 计算 e_{t,j}  (S 个标量)

2. Attention weights (softmax):
   α_{t,j} = exp(e_{t,j}) / Σ_{k=1}^{S} exp(e_{t,k})

   α_t 是一个概率分布: α_{t,j} ≥ 0, Σ_j α_{t,j} = 1

3. 上下文向量 = 加权和:
   c_t = Σ_{j=1}^{S} α_{t,j} · h_j

   c_t 维度: (512,) — 和 encoder hidden state 相同

直观: α_{t,j} 是 decoder 在 t 时刻对源位置 j 的"关注度",
      c_t 是源信息的加权融合。
```

**(C) 参数量:**

```
W_a: 512 × 1024 = 524,288
b_a: 512
v:   512
──────────────────
Total: 525,312

≈ 0.5M 参数 — 相比 Transformer 的 QKV 投影矩阵 (百万级) 非常小!
这也是 Attention 的美妙之处: 用很少的参数实现强大的动态聚焦。
```

</details>

---

# 第五部分: VAE & GAN for Text

---

## 🔴 Q8. 文本 VAE 的后验崩塌

**(A)** VAE 的损失函数 L = -E_q[log P(x|z)] + KL(q(z|x) || P(z))。解释两项各自的含义。

**(B)** 为什么强大的 AR decoder 会导致 KL → 0 (后验崩塌)?

**(C)** 列举至少 3 种缓解后验崩塌的策略, 并简要解释原理。

<details>
<summary>点击查看答案</summary>

**(A) VAE 损失分解:**

```
1. 重构损失 (Reconstruction):
   -E_q(z|x)[log P(x|z)]

   含义: 给定从 posterior q(z|x) 采样的 z, decoder 重构 x 的能力。
   越小 → decoder 越能准确还原输入。

2. KL 正则 (Regularization):
   KL(q(z|x) || P(z)) = KL(N(μ, σ²) || N(0, I))
                       = -½ Σ(1 + log σ² - μ² - σ²)

   含义: posterior q(z|x) 和 prior P(z) 的接近程度。
   越小 → z 越接近标准正态分布 (更"规范")。

目标: 找一个紧凑的隐空间, 既能重构 (rec小), 又规范 (KL小)。
```

**(B) 后验崩塌机制:**

```
AR Decoder 极强:
  - 给定前文 x_{<t}, 甚至不需要 z, 就能很好地预测 x_t
  - 例: P("mat" | "the cat sat on the") 已经非常高, z 提供不了额外信息

Decoder 学会"忽视" z:
  - 梯度: ∂rec/∂z ≈ 0  (z 对重构没什么帮助)
  - Encoder 发现: 无论输出什么 z, decoder 都给出几乎相同的 rec loss
  - 最优策略: 让 q(z|x) = P(z) = N(0,I) → KL = 0
  - → z 不编码任何信息, VAE 退化为普通 LM!

数学上:
  ELBO = E_q[log P(x|z)] - KL

  当 decoder 满足 P(x|z) ≈ P(x) (z被忽略):
    第一项 ≈ E_q[log P(x)]  (不依赖 z)
    为最小化整体: 让 KL → 0 (让 q(z|x) ≈ P(z))
```

**(C) 缓解策略:**

```
1. KL Annealing (KL 退火):
   训练初期: KL 权重很小 (如 0.01)
   训练后期: 逐渐增加到 1.0

   原理: 先让 z 学会编码有用信息 (弱 KL), 
         再逐渐规范隐空间 (强 KL)。
         类比: 先"鼓励"z 去帮助重构, 再"要求"z 规范化。

2. Word Dropout / Input Dropout:
   随机丢弃 decoder 的一部分输入 token (如 30%)
   
   原理: 削弱 decoder 的上下文预测能力 → 
         迫使它依赖 z 来"填补"缺失信息
         类似于降噪自编码器 (Denoising AE)

3. Free Bits:
   修改目标: KL = Σ_j max(λ, KL_j)
   保证每个隐维度至少有 λ 的信息量

   原理: 如果某个维度 KL_j < λ, 损失对它无梯度 →
         鼓励 encoder 使用所有维度

4. β-VAE (增大 KL 权重):
   L = rec + β·KL, β > 1
   
   原理: 更强的 KL 约束 → 更"解耦"的表示
         但 β 太大会导致重构质量严重下降

5. δ-VAE:
   逐步增加 decoder 容量 (从小模型开始)
   原理: 先让弱 decoder 学会依赖 z, 再增强 decoder
```

</details>

---

## 🔴 Q9. SeqGAN 的策略梯度

**(A)** 写出 SeqGAN 中 Generator G 的 REINFORCE 梯度估计。

**(B)** reward R 通常用什么信号? 为什么需要 Monte Carlo rollout?

**(C)** 为什么 SeqGAN 的训练比图像 GAN 更不稳定?

<details>
<summary>点击查看答案</summary>

**(A) REINFORCE 梯度:**

```
目标: 最大化 E_{Y ~ G}[R(Y)]  (生成高 reward 的序列)

策略梯度 (REINFORCE):
  ∇J(θ) ≈ 1/N · Σ_{n=1}^{N} Σ_{t=1}^{T} ∇log G_θ(y_t^{(n)} | y_{<t}^{(n)}) · R_t^{(n)}

其中:
  - y_t^{(n)}: 第 n 个样本的第 t 个 token
  - G_θ: Generator 的参数化策略
  - R_t: 从时刻 t 开始的累积 reward

完整的 REINFORCE (with baseline):
  ∇J ≈ E[ Σ_t ∇log G(y_t|y_{<t}) · (R_t - b_t) ]
  
  baseline b_t 减小方差 (如 critic 网络估计的 V 值)
```

**(B) Reward 信号:**

```
1. 完整序列的判别器打分:
   R(Y) = D(Y)  ∈ [0, 1]  (判别器认为 Y 是"真实"的概率)

2. 中间步骤的 reward: Monte Carlo Rollout
   问题: 生成到一半 (y_{<t}), 还没有完整序列 → D 无法打分!
   
   MC Rollout:
     从当前部分序列 y_{<t} 出发, 用 G 随机生成 K 个完整序列
     对每个完整序列用 D 打分, 取平均
     
     R_t ≈ 1/K · Σ_{k=1}^{K} D(y_{<t} ⊕ rollout_k)

   这极大增加了计算量 (每次梯度更新需要 B × T × K 次生成!)
```

**(C) 为什么不稳定:**

```
1. 离散空间的不可微性:
   图像: pixel ∈ R (连续) → 梯度可直传
   文本: token ∈ {1..V} (离散) → argmax 无梯度

   必须用策略梯度 → 高方差梯度估计

2. Reward 稀疏:
   只有完整序列才有 D 打分 → 中段 token 的 credit assignment 困难
   MC Rollout 缓解了这个问题但代价昂贵

3. Mode Collapse 加剧:
   GAN 本身容易 mode collapse (生成器只产生少数几个样本)
   文本的离散空间使 collapse 更严重 — G 一旦学会生成某几个序列,
   D 很难将 G "推"到新的序列

4. 训练信号矛盾:
   D 的梯度通过策略梯度传给 G → 噪声极大
   图像 GAN: D 梯度直接传给 G (通过生成图像)

5. MLE 预训练的不稳定性:
   SeqGAN 通常用 MLE 预训练 G, 但切换到 RL 后,
   G 可能"忘记" MLE 阶段学到的语言流畅性 (catastrophic forgetting)
```

</details>

---

## 🔴 Q10. 自注意力的计算复杂度与长文本生成

**(A)** 标准 Transformer 自注意力的 FLOPs 如何随序列长度 T 增长? T=1000 时, attention 的部分大约多少 FLOPs?

**(B)** 为什么长文本生成是 Transformer 的瓶颈? 和训练时有什么区别?

**(C)** 列举至少 3 种降低自注意力复杂度的方法, 简述核心思想。

<details>
<summary>点击查看答案</summary>

**(A) 复杂度分析:**

```
标准 Self-Attention:
  Scores = Q @ K^T:  (T, d) × (d, T) = O(T²·d)
  Softmax: O(T²)
  Output = Attn @ V: (T, T) × (T, d) = O(T²·d)

总 FLOPs ≈ 2·T²·d (假设 multi-head 中 d_k = d/h)

T=1000, d=512:
  FLOPs ≈ 2 × 1000² × 512 = 1.024 × 10⁹ ≈ 1 GFLOP

对比 FFN: O(T·d·d_ff) = O(T·d²·4) = 1000 × 512 × 2048 ≈ 1 GFLOP

当 T ≈ d_ff 时, Attention 和 FFN 相当。
当 T >> d_ff 时, Attention 主导 → 平方增长!
```

**(B) 长文本生成的瓶颈:**

```
训练 vs 推理的关键差异:

训练:
  - 一次性输入整个序列, 并行计算 attention
  - 瓶颈: O(T²) 的内存 (存储 attention matrix)
  - GPT-3: T=2048, attention matrix = 2048² × 8头 ≈ 33M floats → 可接受

推理 (生成):
  - 逐 token 生成, 每步都要重新计算整个 attention!
  - 生成第 t 个 token 的 FLOPs ≈ O(t²·d)
  - 生成 T 个 token 总 FLOPs ≈ Σ_{t=1}^T O(t²·d) ≈ O(T³·d)!

  例: T=4096 → 总 FLOPs ≈ O(4096³) → 非常昂贵!

  KV Cache 优化:
    缓存已计算的 K,V 值
    生成第 t 步: 只需计算新 token 的 Q, 和所有缓存的 K,V 做 attention
    FLOPs 降为 Σ O(t·d) ≈ O(T²·d) (还是二次, 但好多了)

  即使有 KV Cache, T=100K 时 T² = 10¹⁰ → 不可接受!
```

**(C) 降低复杂度的方法:**

```
1. Sparse Attention (局部+稀疏全局):
   - 每个 token 只关注 O(log T) 或 O(√T) 个其他 token
   - 模式: 局部窗口 + 空洞 + 少量全局 token
   - 例: Sparse Transformer, Longformer, BigBird
   - 复杂度: O(T·log T) 或 O(T·√T)

2. Low-Rank Approximation (Linformer):
   - 将 T×T attention matrix 低秩分解为 T×k (k << T)
   - 用线性投影压缩序列长度维度
   - 复杂度: O(T·k) where k ≈ 256

3. Kernel-based Attention (Performer):
   - softmax(QK^T) ≈ φ(Q)·φ(K)^T  (随机特征近似)
   - 改变计算顺序: (φ(Q)·φ(K)^T)·V = φ(Q)·(φ(K)^T·V)
   - 先算 φ(K)^T·V (d×d 而非 T×T!)
   - 复杂度: O(T·d²) (与 T 线性!)

4. State Space Models (Mamba / S4):
   - 完全抛弃 attention, 用结构化状态空间模型
   - 复杂度: O(T·d) 严格线性!
   - Mamba 在长序列 (T>10K) 上匹配甚至超越 Transformer

5. Sliding Window + Recurrence (Transformer-XL):
   - 缓存前一段的 hidden states
   - Attention = 当前段 (局部) + 缓存 (历史)
   - 复杂度: O(T_seg² + T_seg·T_cache)
```

</details>

---

## 📊 综合自测评分

每题 10 分 (Q1-Q10), 共 100 分。

| 得分 | 评价 |
|------|------|
| 90-100 | L15 完全掌握, 文本生成的从统计到深度、从训练到解码已贯通 |
| 75-89  | 主干扎实, 建议动手跑 39 的代码练习加深理解 |
| 60-74  | 概念基本清晰, 重点回顾 Q3 (Beam Search) 和 Q6 (BLEU) |
| < 60   | 先把 Q1-Q4 (AR + Decoding基础) 手算一遍, 建立直觉 |

---

> L15 将深度生成模型落地到文本生成这一核心应用:
> 自回归分解 (链式法则) → 解码策略 (Greedy/Beam/Temp/Top-p) → 评估 (PPL/BLEU)
> → 可控生成 (VAE/GAN) → 挑战 (离散梯度/后验崩塌/长序列)
>
> 理解了这些概念, 你就理解了 ChatGPT 在"下一词预测"背后所有的技术基础 —
> 从语言模型训练到 RLHF 中的策略梯度, 从贪婪解码到 Top-p 采样的创造性控制。
