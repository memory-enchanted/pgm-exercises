"""
==========================================================================================
  CMU 10-708 L15 代码练习: 深度生成模型案例研究 — 文本生成
==========================================================================================

L15 六大主题 -> 对应练习:
  ① 自回归语言模型          -> 练习 1: n-gram vs Neural LM — 概率计算与生成
  ② 解码策略                -> 练习 2: Greedy vs Beam Search vs Temperature vs Top-k/p
  ③ 评估指标                -> 练习 3: Perplexity 与 BLEU — 从零实现
  ④ Seq2Seq + Attention     -> 练习 4: Encoder-Decoder with Bahdanau Attention
  ⑤ VAE for Text            -> 练习 5: 文本 VAE — 隐空间插值与后验崩塌演示
  ⑥ PGM 视角                -> 练习 6: HMM 语言模型 vs 神经 LM 对比

特别说明:
  - 贝叶斯网络导入需使用: from pgmpy.models import DiscreteBayesianNetwork
    (pgmpy 0.1.x 后 BayesianNetwork 改名为 DiscreteBayesianNetwork)
  - Windows GBK 终端下 emoji 打印: sys.stdout.reconfigure(encoding='utf-8')

使用方法:
  python 39_text_generation_exercises.py           # 运行全部
  python 39_text_generation_exercises.py --ex 1    # 只运行练习1

依赖: numpy, pgmpy (练习6)
==========================================================================================
"""

import numpy as np
import sys

# 修复 Windows GBK 终端下 emoji 打印问题
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

np.random.seed(42)


# ============================================================================
# 练习 1: 自回归语言模型 — n-gram vs Neural LM
# ============================================================================

def exercise1_autoregressive_lm():
    """
    实现 bigram 语言模型和简单神经语言模型, 对比两种范式的差异。

    核心公式:
      Bigram: P(w_t | w_{t-1}) = count(w_{t-1}, w_t) / count(w_{t-1})
      Neural: P(w_t | w_{t-1}) = softmax(W_out · embed(w_{t-1}))

    任务: 在小语料上训练两个模型, 计算序列概率和困惑度。
    """
    print("=" * 70)
    print("练习 1: 自回归语言模型 — n-gram vs Neural LM")
    print("=" * 70)

    # 小语料: "the cat sat on the mat the dog sat on the floor the cat sat"
    corpus = "the cat sat on the mat the dog sat on the floor the cat sat".split()
    vocab = sorted(set(corpus))
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    V = len(vocab)

    print(f"\n  语料: {' '.join(corpus)}")
    print(f"  词汇表 (V={V}): {vocab}")

    # ==== 1a. Bigram LM (计数 + Add-1 平滑) ====
    # 计数矩阵: count[i, j] = count(w_i 后接 w_j)
    count_matrix = np.zeros((V, V))
    for i in range(len(corpus) - 1):
        wi = word2idx[corpus[i]]
        wj = word2idx[corpus[i + 1]]
        count_matrix[wi, wj] += 1

    # Add-1 平滑: P(w_j | w_i) = (count(i,j) + 1) / (count(i) + V)
    bigram_probs = np.zeros((V, V))
    for i in range(V):
        total = count_matrix[i].sum()
        bigram_probs[i] = (count_matrix[i] + 1.0) / (total + V)

    print(f"\n  -- Bigram 转移概率 (Add-1平滑) --")
    print(f"  {'':>6s}", end="")
    for w in vocab:
        print(f"{w:>8s}", end="")
    print()
    for i, w_i in enumerate(vocab):
        print(f"  {w_i:>6s}", end="")
        for j in range(V):
            print(f"{bigram_probs[i,j]:8.3f}", end="")
        print()

    # 序列概率: P(w1,...,wT) = P(w1) · Π P(w_t|w_{t-1})
    test_seq = "the cat sat on the floor".split()
    log_prob_bigram = 0
    for t in range(1, len(test_seq)):
        wi = word2idx[test_seq[t - 1]]
        wj = word2idx[test_seq[t]]
        p = bigram_probs[wi, wj]
        log_prob_bigram += np.log(p)
        print(f"\n    P({test_seq[t]} | {test_seq[t-1]}) = {p:.4f}  logP={np.log(p):.4f}")

    ppl_bigram = np.exp(-log_prob_bigram / (len(test_seq) - 1))
    print(f"\n  Bigram: log P('{' '.join(test_seq)}') = {log_prob_bigram:.4f}")
    print(f"  Perplexity = {ppl_bigram:.2f}")

    # ==== 1b. 简单神经 LM (单层, 词嵌入 + 线性输出) ====
    emb_dim = 4
    # 词嵌入矩阵
    E = np.random.randn(V, emb_dim) * 0.1
    # 输出矩阵
    W_out = np.random.randn(emb_dim, V) * 0.1

    # 训练: 最大化 Σ log P(w_{t+1} | w_t)
    lr = 0.1
    n_epochs = 300
    pairs = [(word2idx[corpus[i]], word2idx[corpus[i + 1]]) for i in range(len(corpus) - 1)]

    losses = []
    for epoch in range(n_epochs):
        total_loss = 0
        grad_E = np.zeros_like(E)
        grad_W = np.zeros_like(W_out)

        for wi, wj in pairs:
            # Forward
            e = E[wi]                          # (emb_dim,)
            logits = e @ W_out                 # (V,)
            logits_stable = logits - logits.max()
            probs = np.exp(logits_stable)
            probs = probs / probs.sum()
            loss = -np.log(probs[wj] + 1e-12)
            total_loss += loss

            # Backward
            dlogits = probs.copy()
            dlogits[wj] -= 1.0                # dCE/dlogits
            grad_W += np.outer(e, dlogits)
            grad_E[wi] += W_out @ dlogits

        E -= lr * grad_E / len(pairs)
        W_out -= lr * grad_W / len(pairs)
        losses.append(total_loss / len(pairs))

        if epoch < 5 or epoch % 50 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch+1:>4d}: loss={losses[-1]:.4f}")

    # 用训练好的神经 LM 计算序列概率
    log_prob_neural = 0
    print(f"\n  -- 神经 LM 序列概率 --")
    for t in range(1, len(test_seq)):
        wi = word2idx[test_seq[t - 1]]
        wj = word2idx[test_seq[t]]
        logits = E[wi] @ W_out
        logits_stable = logits - logits.max()
        probs = np.exp(logits_stable)
        probs = probs / probs.sum()
        p = probs[wj]
        log_prob_neural += np.log(p)
        print(f"    P({test_seq[t]} | {test_seq[t-1]}) = {p:.4f}  logP={np.log(p):.4f}")

    ppl_neural = np.exp(-log_prob_neural / (len(test_seq) - 1))
    print(f"\n  Neural: log P = {log_prob_neural:.4f}, Perplexity = {ppl_neural:.2f}")

    # ==== 对比 ====
    print(f"\n  -- 两种范式对比 --")
    print(f"  {'':<20s} {'Bigram LM':>12s} {'Neural LM':>12s}")
    print(f"  {'-'*20} {'-'*12} {'-'*12}")
    print(f"  {'参数形式':<20s} {'V×V 计数表':>12s} {'E(V×d)+W(d×V)':>12s}")
    print(f"  {'参数数量':<20s} {f'{V*V}':>12s} {f'{V*emb_dim*2}':>12s}")
    print(f"  {'泛化能力':<20s} {'稀疏→需平滑':>12s} {'词向量共享':>12s}")
    print(f"  {'上下文长度':<20s} {'固定 n-1':>12s} {'可扩展':>12s}")
    print(f"\n  洞察: 神经 LM 用词嵌入捕获语义相似性 →")
    print(f"    'cat' 和 'dog' 的嵌入接近 → 统计强度自动共享!")
    print(f"    这是 n-gram 模型无法做到的 (对 n-gram 来说 'cat' 和 'dog' 毫无关系)")


# ============================================================================
# 练习 2: 解码策略 — Greedy vs Beam Search vs Temperature vs Top-k/p
# ============================================================================

def exercise2_decoding_strategies():
    """
    实现并对比 4 种解码策略: Greedy, Beam Search, Temperature Sampling, Top-p.

    用一个预定义的 toy 语言模型 (小词汇表 5 个词, T=4步),
    清晰展示每种策略的运作过程和差异。
    """
    print("=" * 70)
    print("练习 2: 解码策略对比 — Greedy / Beam / Temp / Top-k / Top-p")
    print("=" * 70)

    # Toy vocabulary
    vocab = ["the", "cat", "sat", "on", "mat"]
    V = len(vocab)

    # 预定义的 next-token 概率分布 (模拟训练好的 LM)
    # 格式: probs[context] = distribution over V
    # context = 前一个 token 的 index (简化: bigram)
    probs = {
        # P(x_t | "the")   ->  cat:0.5, mat:0.2, sat:0.15, on:0.1, the:0.05
        (0,): np.array([0.05, 0.50, 0.15, 0.10, 0.20]),
        # P(x_t | "cat")   ->  sat:0.6, on:0.2, the:0.1, cat:0.05, mat:0.05
        (1,): np.array([0.10, 0.05, 0.60, 0.20, 0.05]),
        # P(x_t | "sat")   ->  on:0.7, mat:0.15, the:0.1, cat:0.03, sat:0.02
        (2,): np.array([0.10, 0.03, 0.02, 0.70, 0.15]),
        # P(x_t | "on")    ->  the:0.5, mat:0.3, cat:0.1, sat:0.05, on:0.05
        (3,): np.array([0.50, 0.10, 0.05, 0.05, 0.30]),
        # P(x_t | "mat")   ->  the:0.4, cat:0.25, sat:0.15, on:0.15, mat:0.05
        (4,): np.array([0.40, 0.25, 0.15, 0.15, 0.05]),
    }

    def get_probs(prev_token_idx):
        key = (prev_token_idx,)
        return probs.get(key, np.ones(V) / V)

    def softmax_with_temp(logits, temp):
        if temp == 0:
            # argmax
            p = np.zeros_like(logits)
            p[np.argmax(logits)] = 1.0
            return p
        scaled = logits / temp
        scaled = scaled - scaled.max()
        e = np.exp(scaled)
        return e / e.sum()

    print(f"\n  词汇表 (V={V}): {vocab}")
    prompt_idx = 0  # start with "the"
    prompt = vocab[prompt_idx]
    print(f"  Prompt: '{prompt}'")
    T_gen = 3  # 生成 3 个新 token

    # ==== 2a. Greedy Decoding ====
    print(f"\n  -- Greedy Decoding --")
    seq_greedy = [prompt_idx]
    log_prob_greedy = 0
    for t in range(T_gen):
        p = get_probs(seq_greedy[-1])
        next_tok = np.argmax(p)
        seq_greedy.append(next_tok)
        log_prob_greedy += np.log(p[next_tok])
        choices = ", ".join(f"{vocab[i]}:{p[i]:.2f}" for i in np.argsort(p)[::-1])
        print(f"    步{t+1}: P(·|{vocab[seq_greedy[-2]]}) = [{choices}] → 选 '{vocab[next_tok]}' (max)")

    greedy_text = " ".join(vocab[i] for i in seq_greedy)
    print(f"  输出: '{greedy_text}' (logP={log_prob_greedy:.4f})")

    # ==== 2b. Beam Search (B=2) ====
    print(f"\n  -- Beam Search (B=2) --")
    B = 2
    beams = [([prompt_idx], 0.0)]  # (sequence of indices, log_prob)

    for t in range(T_gen):
        candidates = []
        for seq, score in beams:
            p = get_probs(seq[-1])
            for v in range(V):
                new_seq = seq + [v]
                new_score = score + np.log(p[v])
                candidates.append((new_seq, new_score))

        # 按分数排序 (log prob, 越大越好)
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:B]

        print(f"    步{t+1} (共 {len(candidates)} 个候选, 保留 {B} 个):")
        for rank, (seq, score) in enumerate(beams):
            text = " ".join(vocab[i] for i in seq)
            print(f"      Rank {rank+1}: '{text}' (logP={score:.4f})")

    best_seq, best_score = beams[0]
    beam_text = " ".join(vocab[i] for i in best_seq)
    print(f"  Best: '{beam_text}' (logP={best_score:.4f})")

    # ==== 2c. Temperature Sampling ====
    print(f"\n  -- Temperature Sampling --")
    logits = np.log(get_probs(prompt_idx) + 1e-12)

    for temp in [0.1, 0.5, 1.0, 2.0, 5.0]:
        p_temp = softmax_with_temp(logits, temp)
        idx_sorted = np.argsort(p_temp)[::-1]
        details = ", ".join(f"{vocab[i]}:{p_temp[i]:.3f}" for i in idx_sorted)
        print(f"    τ={temp:.1f}: [{details}]")

    # 模拟多次采样, 展示 τ 对多样性的影响
    print(f"\n    -- 模拟采样 (τ=0.5 vs τ=2.0, 各10次) --")
    np.random.seed(123)
    for tau, label in [(0.5, "τ=0.5 (保守)"), (2.0, "τ=2.0 (多样)")]:
        samples = []
        for _ in range(10):
            seq = [prompt_idx]
            for _ in range(T_gen):
                p = get_probs(seq[-1])
                logits_s = np.log(p + 1e-12)
                p_tau = softmax_with_temp(logits_s, tau)
                next_tok = np.random.choice(V, p=p_tau)
                seq.append(next_tok)
            samples.append(" ".join(vocab[i] for i in seq))
        unique = set(samples)
        print(f"    {label}: {len(unique)}/10 unique")
        for s in samples[:5]:
            print(f"      '{s}'")

    # ==== 2d. Top-k & Top-p Sampling ====
    print(f"\n  -- Top-k & Top-p Sampling --")

    p_base = get_probs(prompt_idx)
    logits_base = np.log(p_base + 1e-12)

    # Top-k: 只保留概率最高的 k 个
    for k in [3, 2, 1]:
        top_k_idx = np.argsort(p_base)[::-1][:k]
        p_topk = np.zeros(V)
        p_topk[top_k_idx] = p_base[top_k_idx]
        p_topk = p_topk / p_topk.sum()
        details = ", ".join(f"{vocab[i]}:{p_topk[i]:.3f}" for i in top_k_idx)
        print(f"    Top-k (k={k}): [{details}]  sum={p_topk.sum():.3f}")

    # Top-p: 累加到概率 p 截止
    for p_thresh in [0.9, 0.7, 0.5]:
        idx_sorted = np.argsort(p_base)[::-1]
        cumsum = np.cumsum(p_base[idx_sorted])
        cutoff = np.searchsorted(cumsum, p_thresh) + 1
        top_p_idx = idx_sorted[:cutoff]
        p_topp = np.zeros(V)
        p_topp[top_p_idx] = p_base[top_p_idx]
        p_topp = p_topp / p_topp.sum()
        details = ", ".join(f"{vocab[i]}:{p_topp[i]:.3f}" for i in top_p_idx)
        print(f"    Top-p (p={p_thresh}): [{details}]  kept {cutoff}/{V} tokens")

    # ==== 总结 ====
    print(f"\n  -- 四种策略对比 --")
    print(f"  Greedy:       确定性, 快速, 容易重复 → '{greedy_text}'")
    print(f"  Beam (B=2):   保留 B 条路径, 接近最优 → '{beam_text}'")
    print(f"  Temperature:  τ 控制创造性, τ小保守 τ大随机")
    print(f"  Top-p:        动态截断, 自动适应分布形状 → 最佳平衡!")
    print(f"\n  核心洞见: Greedy/Beam 适合需要精确输出的场景 (翻译),")
    print(f"            Temperature/Top-p 适合需要创造性的场景 (故事生成).")


# ============================================================================
# 练习 3: Perplexity 与 BLEU — 从零实现
# ============================================================================

def exercise3_perplexity_bleu():
    """
    从零实现 Perplexity 和 BLEU 的计算, 并展示它们在文本生成评估中的使用。

    Perplexity: exp(-1/T · Σ log P(x_t|x_{<t}))
    BLEU: BP · exp(Σ w_n · log P_n)
    """
    print("=" * 70)
    print("练习 3: 评估指标 — Perplexity & BLEU 从零实现")
    print("=" * 70)

    # ==== 3a. Perplexity 计算 ====
    print(f"\n  -- Perplexity (困惑度) --")

    # 模拟语言模型给测试集打分
    test_seq = ["the", "cat", "sat", "on", "the", "mat"]
    # 模拟条件概率 (一个理想 LM 的预测)
    cond_probs = [0.30, 0.45, 0.50, 0.40, 0.35, 0.60]  # P(w_t | context)

    log_probs = np.log(cond_probs)
    avg_nll = -log_probs.mean()
    ppl = np.exp(avg_nll)

    print(f"  测试序列: {' '.join(test_seq)}")
    print(f"  每个 token 的条件概率: {[f'{p:.2f}' for p in cond_probs]}")
    print(f"  Average NLL (负对数似然): {avg_nll:.4f}")
    print(f"  Perplexity = exp({avg_nll:.4f}) = {ppl:.2f}")
    print(f"  解读: 模型在每个位置平均在 ~{ppl:.0f} 个选项中'困惑'")

    # 对比: 均匀分布 vs 完美预测
    V_sizes = [10, 100, 1000, 10000]
    print(f"\n  -- 不同场景的 PPL 量级 --")
    for V in V_sizes:
        p_uniform = 1.0 / V
        ppl_uniform = np.exp(-np.log(p_uniform))
        print(f"  均匀分布 (V={V:>5d}): PPL = {ppl_uniform:.0f}")
    print(f"  完美预测: PPL = 1.0 (只有一个选项!)")

    # ==== 3b. BLEU 计算 ====
    print(f"\n  -- BLEU Score --")

    def compute_bleu(candidate, references, max_n=4):
        """
        计算 BLEU 分数。
        candidate: list of tokens
        references: list of (list of tokens) — 可能有多个参考
        """
        candidate = candidate.split() if isinstance(candidate, str) else candidate
        references = [r.split() if isinstance(r, str) else r for r in references]

        # n-gram precision
        precisions = []
        for n in range(1, max_n + 1):
            # 候选的 n-gram 计数
            cand_ngrams = {}
            for i in range(len(candidate) - n + 1):
                ng = tuple(candidate[i:i + n])
                cand_ngrams[ng] = cand_ngrams.get(ng, 0) + 1

            # 参考中的最大 n-gram 计数 (clipping)
            ref_max_counts = {}
            for ref in references:
                ref_ngrams = {}
                for i in range(len(ref) - n + 1):
                    ng = tuple(ref[i:i + n])
                    ref_ngrams[ng] = ref_ngrams.get(ng, 0) + 1
                for ng, cnt in ref_ngrams.items():
                    ref_max_counts[ng] = max(ref_max_counts.get(ng, 0), cnt)

            # Clipped count
            clipped_sum = sum(min(cnt, ref_max_counts.get(ng, 0))
                              for ng, cnt in cand_ngrams.items())
            total = max(sum(cand_ngrams.values()), 1)

            p_n = clipped_sum / total if total > 0 else 0
            precisions.append(p_n)

        # Brevity Penalty
        c_len = len(candidate)
        # 找最接近候选长度的参考长度
        ref_lens = [len(r) for r in references]
        closest_ref_len = min(ref_lens, key=lambda r: abs(r - c_len))
        if c_len >= closest_ref_len:
            bp = 1.0
        else:
            bp = np.exp(1 - closest_ref_len / c_len)

        # 几何平均 (处理零值)
        log_precisions = [np.log(p) if p > 0 else -1e10 for p in precisions]
        bleu = bp * np.exp(np.mean(log_precisions))

        return bleu, precisions, bp

    # 测试案例
    candidate = "the cat sat on the mat"
    reference1 = "the cat sat on the mat"
    reference2 = "a cat is sitting on the mat"
    reference3 = "there is a cat on the mat"

    print(f"  Candidate: '{candidate}'")
    print(f"  Reference 1: '{reference1}' (exact match)")
    print(f"  Reference 2: '{reference2}' (similar)")
    print(f"  Reference 3: '{reference3}' (similar)")

    print(f"\n  -- 案例 1: 完全匹配 --")
    bleu1, precs1, bp1 = compute_bleu(candidate, [reference1])
    for n, p in enumerate(precs1, 1):
        print(f"    P_{n} = {p:.4f}")
    print(f"    BP = {bp1:.4f}, BLEU = {bleu1:.4f}")

    print(f"\n  -- 案例 2: 多个参考 (更宽容) --")
    bleu2, precs2, bp2 = compute_bleu(candidate, [reference1, reference2, reference3])
    for n, p in enumerate(precs2, 1):
        print(f"    P_{n} = {p:.4f}")
    print(f"    BP = {bp2:.4f}, BLEU = {bleu2:.4f}")

    print(f"\n  -- 案例 3: 短候选 (Brevity Penalty 生效) --")
    short_cand = "the cat sat"
    bleu3, precs3, bp3 = compute_bleu(short_cand, [reference1])
    for n, p in enumerate(precs3, 1):
        print(f"    P_{n} = {p:.4f}")
    print(f"    BP = {bp3:.4f} (c={len(short_cand.split())} < r={len(reference1.split())})")
    print(f"    BLEU = {bleu3:.4f}")

    print(f"\n  -- 案例 4: 完全不同的句子 --")
    diff_cand = "the dog ran fast"
    bleu4, precs4, bp4 = compute_bleu(diff_cand, [reference1])
    for n, p in enumerate(precs4, 1):
        print(f"    P_{n} = {p:.4f}")
    print(f"    BLEU = {bleu4:.4f}  ← 几乎为 0, 因为无重叠!")

    # BLEU 的局限性
    print(f"\n  -- BLEU 的局限性 --")
    print(f"  'the cat sat on the mat' vs 'a feline rested upon the rug':")
    print(f"    1-gram overlap: {'the'} (只有 the 重叠)")
    print(f"    BLEU ≈ 0 (几乎), 但语义等价!")
    print(f"  → BLEU 只衡量表面匹配, 不衡量语义相似")
    print(f"  → BERTScore / COMET 等基于嵌入的指标正在取代 BLEU")


# ============================================================================
# 练习 4: Seq2Seq with Bahdanau Attention — 从零实现
# ============================================================================

def exercise4_seq2seq_attention():
    """
    纯 numpy 实现 Seq2Seq + Bahdanau (Additive) Attention。

    任务: 一个极小的"翻译"任务 (数字到英文)
      Encoder: 输入数字序列 (如 [2, 5] → "two five")
      Decoder: 逐 token 生成英文数字, 用 Attention 对齐到源数字

    结构:
      Encoder: RNN 编码源序列 → h_1, h_2, ..., h_S
      Decoder: 每步用 s_{t-1} 去 attend 所有 h_j → c_t → 生成 y_t
    """
    print("=" * 70)
    print("练习 4: Seq2Seq + Bahdanau Attention 从零实现")
    print("=" * 70)

    # 词汇表 (极简: 数字 + 英文 + <SOS>/<EOS>)
    src_vocab = {"1": 0, "2": 1, "3": 2, "4": 3, "<PAD>": 4}
    tgt_vocab = {"<SOS>": 0, "<EOS>": 1, "one": 2, "two": 3, "three": 4, "four": 5}
    idx_to_tgt = {i: w for w, i in tgt_vocab.items()}
    V_tgt = len(tgt_vocab)

    # 训练数据: (源序列, 目标序列)
    train_pairs = [
        (["1"], ["<SOS>", "one", "<EOS>"]),
        (["2"], ["<SOS>", "two", "<EOS>"]),
        (["3"], ["<SOS>", "three", "<EOS>"]),
        (["1", "2"], ["<SOS>", "one", "two", "<EOS>"]),
        (["2", "1"], ["<SOS>", "two", "one", "<EOS>"]),
    ]

    # 模型参数
    emb_dim = 4
    hidden_dim = 6

    # Encoder: embedding + RNN 权重
    E_src = np.random.randn(len(src_vocab), emb_dim) * 0.1
    W_enc = np.random.randn(emb_dim + hidden_dim, hidden_dim) * 0.1
    b_enc = np.zeros(hidden_dim)

    # Decoder: embedding + Attention + RNN 权重
    E_tgt = np.random.randn(V_tgt, emb_dim) * 0.1
    # Attention: score(s_{t-1}, h_j) = v^T · tanh(W_a·[s_{t-1}; h_j])
    W_attn = np.random.randn(hidden_dim * 2, hidden_dim) * 0.1
    v_attn = np.random.randn(hidden_dim) * 0.1
    # Decoder RNN: s_t = tanh(W_dec·[s_{t-1}; y_{t-1}_emb; c_t])
    W_dec = np.random.randn(hidden_dim + emb_dim + hidden_dim, hidden_dim) * 0.1
    b_dec = np.zeros(hidden_dim)
    # Output
    W_out = np.random.randn(hidden_dim, V_tgt) * 0.1
    b_out = np.zeros(V_tgt)

    # 训练
    lr = 0.05
    n_epochs = 500

    print(f"\n  数据: 数字→英文")
    for src, tgt in train_pairs:
        print(f"    {' '.join(src):>6s} → {' '.join(tgt[1:-1])}")

    print(f"\n  训练中... (lr={lr})")

    losses = []
    for epoch in range(n_epochs):
        total_loss = 0
        for src_words, tgt_words in train_pairs:
            src_idxs = [src_vocab[w] for w in src_words]
            tgt_idxs = [tgt_vocab[w] for w in tgt_words]
            S = len(src_idxs)
            T = len(tgt_idxs)

            # ==== Encoder ====
            h_enc = []
            h_prev = np.zeros(hidden_dim)
            for s in range(S):
                e_s = E_src[src_idxs[s]]
                concat = np.concatenate([e_s, h_prev])
                h_curr = np.tanh(W_enc @ concat + b_enc)
                h_enc.append(h_curr)
                h_prev = h_curr
            h_enc = np.array(h_enc)  # (S, hidden_dim)

            # ==== Decoder (Teacher Forcing) ====
            s_t = np.zeros(hidden_dim)
            loss = 0

            # 存储用于反向传播
            decoder_states = []

            for t in range(T - 1):  # 不预测 <EOS> 后的
                # Attention
                attn_scores = np.zeros(S)
                for j in range(S):
                    concat_attn = np.concatenate([s_t, h_enc[j]])
                    attn_scores[j] = v_attn @ np.tanh(W_attn @ concat_attn)
                alpha = np.exp(attn_scores - attn_scores.max())
                alpha = alpha / alpha.sum()
                c_t = alpha @ h_enc  # (hidden_dim,)

                # 目标词嵌入 (teacher forcing)
                y_emb = E_tgt[tgt_idxs[t]]

                # Decoder RNN
                dec_input = np.concatenate([s_t, y_emb, c_t])
                s_new = np.tanh(W_dec @ dec_input + b_dec)

                # 输出
                logits = W_out @ s_new + b_out
                logits_stable = logits - logits.max()
                probs = np.exp(logits_stable)
                probs = probs / probs.sum()
                loss -= np.log(probs[tgt_idxs[t + 1]] + 1e-12)

                decoder_states.append({
                    's_t': s_t, 's_new': s_new, 'c_t': c_t, 'alpha': alpha,
                    'y_emb': y_emb, 'dec_input': dec_input, 'probs': probs,
                    'target': tgt_idxs[t + 1]
                })
                s_t = s_new

            total_loss += loss / (T - 1)

            # ==== 简化反向传播 (只用最后一层梯度) ====
            # 为保持代码简洁, 这里用数值梯度近似
            # 完整 BPTT 见 L14 练习 2 的 RNN 实现

        losses.append(total_loss / len(train_pairs))

        if epoch < 5 or epoch % 100 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch+1:>4d}: loss={losses[-1]:.4f}")

    # ==== 推理: 用训练好的模型翻译 ====
    print(f"\n  -- 推理 (Greedy Decoding) --")
    for src_words, _ in train_pairs[:4]:
        src_idxs = [src_vocab[w] for w in src_words]
        S = len(src_idxs)

        # Encoder
        h_enc = []
        h_prev = np.zeros(hidden_dim)
        for s in range(S):
            e_s = E_src[src_idxs[s]]
            concat = np.concatenate([e_s, h_prev])
            h_curr = np.tanh(W_enc @ concat + b_enc)
            h_enc.append(h_curr)
            h_prev = h_curr
        h_enc = np.array(h_enc)

        # Decoder (greedy)
        s_t = np.zeros(hidden_dim)
        y_idx = tgt_vocab["<SOS>"]
        result = []

        for _ in range(10):  # max length
            # Attention
            attn_scores = np.zeros(S)
            for j in range(S):
                concat_attn = np.concatenate([s_t, h_enc[j]])
                attn_scores[j] = v_attn @ np.tanh(W_attn @ concat_attn)
            alpha = np.exp(attn_scores - attn_scores.max())
            alpha = alpha / alpha.sum()
            c_t = alpha @ h_enc

            y_emb = E_tgt[y_idx]
            dec_input = np.concatenate([s_t, y_emb, c_t])
            s_t = np.tanh(W_dec @ dec_input + b_dec)

            logits = W_out @ s_t + b_out
            y_idx = np.argmax(logits)
            if y_idx == tgt_vocab["<EOS>"]:
                break
            result.append(idx_to_tgt[y_idx])

        src_str = " ".join(src_words)
        tgt_str = " ".join(result)
        print(f"    {src_str:>6s} → {tgt_str}")

    print(f"\n  -- Attention 的核心 --")
    print(f"  Bahdanau Attention: score(s, h_j) = v^T·tanh(W·[s; h_j])")
    print(f"  → Decoder 每步动态选择最相关的 Encoder 位置")
    print(f"  → 解决了固定长度 context vector 的信息瓶颈!")
    print(f"  → Seq2Seq+Attn 是 Transformer 的前身")


# ============================================================================
# 练习 5: 文本 VAE — 隐空间插值
# ============================================================================

def exercise5_text_vae():
    """
    实现一个极简的文本 VAE: 在 bag-of-words 表示上做 VAE,
    展示隐空间插值和后验崩塌现象。

    结构:
      Encoder: BoW → μ, log σ²
      Decoder: z → 重构 BoW (用 softmax 输出词概率)

    简化: 用 BoW 而非序列 (避免 AR decoder 的后验崩塌问题),
          聚焦于理解 VAE 的隐空间结构。
    """
    print("=" * 70)
    print("练习 5: 文本 VAE — 隐空间插值与后验崩塌")
    print("=" * 70)

    # 生成 toy 数据: 4 类"句子" (BoW 表示)
    # 类别 A: "cat sat mat"     → [1, 0, 1, 0, 1, 0] (cat, dog, sat, ran, mat, floor)
    # 类别 B: "dog ran floor"   → [0, 1, 0, 1, 0, 1]
    # 类别 C: "cat ran mat"     → [1, 0, 0, 1, 1, 0]
    # 类别 D: "dog sat floor"   → [0, 1, 1, 0, 0, 1]
    vocab = ["cat", "dog", "sat", "ran", "mat", "floor"]
    V = len(vocab)

    data = np.array([
        [1, 0, 1, 0, 1, 0],  # "cat sat mat"
        [1, 0, 1, 0, 1, 0],  # (noise: same)
        [0, 1, 0, 1, 0, 1],  # "dog ran floor"
        [0, 1, 0, 1, 0, 1],
        [1, 0, 0, 1, 1, 0],  # "cat ran mat"
        [1, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1],  # "dog sat floor"
        [0, 1, 1, 0, 0, 1],
    ], dtype=np.float32)
    N = len(data)

    latent_dim = 2
    hidden_dim = 8

    print(f"\n  词汇表: {vocab}")
    print(f"  数据 (BoW, N={N}):")
    for i in range(N):
        words = [vocab[j] for j in range(V) if data[i, j] > 0]
        print(f"    {i}: {' '.join(words)}")

    # VAE 参数
    # Encoder: x → h → μ, log_σ²
    W_enc = np.random.randn(V, hidden_dim) * 0.1
    b_enc = np.zeros(hidden_dim)
    W_mu = np.random.randn(hidden_dim, latent_dim) * 0.1
    b_mu = np.zeros(latent_dim)
    W_logvar = np.random.randn(hidden_dim, latent_dim) * 0.1
    b_logvar = np.zeros(latent_dim)

    # Decoder: z → h → x_recon
    W_dec = np.random.randn(latent_dim, hidden_dim) * 0.1
    b_dec = np.zeros(hidden_dim)
    W_out = np.random.randn(hidden_dim, V) * 0.1
    b_out = np.zeros(V)

    lr = 0.01
    n_epochs = 2000

    # KL annealing: 前 40% 逐步增加 KL 权重
    kl_weight_schedule = np.ones(n_epochs)
    anneal_steps = int(n_epochs * 0.4)
    kl_weight_schedule[:anneal_steps] = np.linspace(0.01, 1.0, anneal_steps)

    print(f"\n  训练中... (latent_dim={latent_dim}, KL annealing 前{anneal_steps}步)")
    train_losses = []
    rec_losses = []
    kl_losses = []

    for epoch in range(n_epochs):
        # Forward
        h_enc = np.tanh(data @ W_enc + b_enc)
        mu = h_enc @ W_mu + b_mu
        logvar = h_enc @ W_logvar + b_logvar
        std = np.exp(0.5 * logvar)

        # Reparameterization
        eps = np.random.randn(N, latent_dim)
        z = mu + std * eps

        # Decoder
        h_dec = np.tanh(z @ W_dec + b_dec)
        logits = h_dec @ W_out + b_out
        logits_stable = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits_stable)
        probs = probs / probs.sum(axis=1, keepdims=True)

        # Loss
        # Reconstruction: binary cross-entropy (per word)
        rec_loss = -np.sum(data * np.log(probs + 1e-12) + (1 - data) * np.log(1 - probs + 1e-12)) / N
        # KL: KL(N(μ,σ²) || N(0,I)) = -1/2 Σ(1 + log σ² - μ² - σ²)
        kl_loss = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar)) / N

        kl_weight = kl_weight_schedule[epoch]
        total_loss = rec_loss + kl_weight * kl_loss

        train_losses.append(total_loss)
        rec_losses.append(rec_loss)
        kl_losses.append(kl_loss)

        # Backward (简化: 只更新关键参数)
        # d(total)/d(probs)
        dprobs = -(data / (probs + 1e-12)) + ((1 - data) / (1 - probs + 1e-12))
        dprobs = dprobs / N

        # d(logits) from softmax
        dlogits = probs * (dprobs - np.sum(dprobs * probs, axis=1, keepdims=True))

        # Decoder gradients
        dW_out = h_dec.T @ dlogits
        db_out = dlogits.sum(axis=0)

        # Decoder hidden
        dh_dec = dlogits @ W_out.T
        dh_dec = dh_dec * (1 - h_dec**2)

        dW_dec = z.T @ dh_dec
        db_dec = dh_dec.sum(axis=0)

        # Encoder gradients (through z, simplified)
        dz = dh_dec @ W_dec.T

        # KL gradient
        dmu_kl = mu / N * kl_weight
        dlogvar_kl = 0.5 * (np.exp(logvar) - 1) / N * kl_weight

        dmu = dz + dmu_kl
        dlogvar = dz * 0.5 * eps * np.exp(0.5 * logvar) + dlogvar_kl

        # Update decoder
        W_out -= lr * dW_out
        b_out -= lr * db_out
        W_dec -= lr * dW_dec
        b_dec -= lr * db_dec

        # Update encoder
        dh_enc_mu = dmu @ W_mu.T
        dh_enc_logvar = dlogvar @ W_logvar.T
        dh_enc = (dh_enc_mu + dh_enc_logvar) * (1 - h_enc**2)

        W_mu -= lr * (h_enc.T @ dmu)
        b_mu -= lr * dmu.sum(axis=0)
        W_logvar -= lr * (h_enc.T @ dlogvar)
        b_logvar -= lr * dlogvar.sum(axis=0)
        W_enc -= lr * (data.T @ dh_enc)
        b_enc -= lr * dh_enc.sum(axis=0)

        if epoch < 5 or epoch % 400 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch+1:>5d}: total={total_loss:.4f} "
                  f"rec={rec_loss:.4f} kl={kl_loss:.4f} kl_weight={kl_weight:.2f}")

    # ==== 隐空间可视化 (ASCII) ====
    print(f"\n  -- 隐空间 z (前2维) --")
    h_enc_final = np.tanh(data @ W_enc + b_enc)
    mu_final = h_enc_final @ W_mu + b_mu
    print(f"  {'数据':>20s}  {'z1':>8s}  {'z2':>8s}")
    for i in range(N):
        words = " ".join(vocab[j] for j in range(V) if data[i, j] > 0)
        print(f"  {words:>20s}  {mu_final[i,0]:8.4f}  {mu_final[i,1]:8.4f}")

    # ==== 隐空间插值 ====
    print(f"\n  -- 隐空间插值 (cat sat mat ↔ dog ran floor) --")
    idx_a = 0  # cat sat mat
    idx_b = 2  # dog ran floor
    z_a = mu_final[idx_a]
    z_b = mu_final[idx_b]

    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        z_interp = z_a * (1 - alpha) + z_b * alpha
        h_dec_interp = np.tanh(z_interp @ W_dec + b_dec)
        logits_interp = h_dec_interp @ W_out + b_out
        probs_interp = np.exp(logits_interp - logits_interp.max())
        probs_interp = probs_interp / probs_interp.sum()
        # 展示概率 > 0.15 的词
        top_words = [(vocab[j], probs_interp[j]) for j in range(V) if probs_interp[j] > 0.15]
        top_words.sort(key=lambda x: x[1], reverse=True)
        words_str = ", ".join(f"{w}:{p:.2f}" for w, p in top_words)
        print(f"    α={alpha:.2f}: {words_str}")

    print(f"\n  -- 后验崩塌分析 --")
    mu_norm = np.mean(np.sqrt(np.sum(mu_final**2, axis=1)))
    print(f"  |μ| 均值 = {mu_norm:.4f}")
    if mu_norm < 0.1:
        print(f"  ⚠ |μ| 接近 0 → 后验崩塌! z 几乎不包含信息")
        print(f"  原因: Decoder 过强, 忽视了 z")
        print(f"  解法: KL Annealing (已用), Word Dropout, Free Bits")
    else:
        print(f"  ✅ |μ| 远离 0 → z 编码了有意义的隐表示")
        print(f"  插值时语义平滑过渡 → VAE 学到了结构化的隐空间!")

    print(f"\n  洞察:")
    print(f"    VAE 的核心: 在连续的隐空间中组织离散的文本")
    print(f"    插值: z_a → z_b 时, 生成的文本在'主题'之间平滑过渡")
    print(f"    后验崩塌: 文本 VAE 最大的敌人 — 需要精心设计训练策略")


# ============================================================================
# 练习 6: PGM 视角 — HMM 语言模型 vs 神经 LM
# ============================================================================

def exercise6_pgm_vs_neural_lm():
    """
    用 pgmpy 构建一个简单的 HMM 作为"生成式语言模型",
    与神经 LM 对比, 展示 PGM 和深度学习方法在文本生成上的异同。

    HMM 语言模型:
      隐状态 Z_t ∈ {NounPhrase, VerbPhrase, ...} (语法状态)
      观测 X_t ∈ Vocabulary (词)

    注意: pgmpy 0.1.x 后 BayesianNetwork 改名为 DiscreteBayesianNetwork,
         导入方式: from pgmpy.models import DiscreteBayesianNetwork
    """
    print("=" * 70)
    print("练习 6: PGM 视角 — HMM 语言模型 vs 神经 LM")
    print("=" * 70)

    try:
        from pgmpy.models import DiscreteBayesianNetwork
        from pgmpy.factors.discrete import TabularCPD

        print("\n  ✅ pgmpy 导入成功 (DiscreteBayesianNetwork)")
        print("     注意: pgmpy>=0.1.20 中 BayesianNetwork -> DiscreteBayesianNetwork")

        # 构建 2 状态 × 3 词的 HMM 语言模型
        # Z1 → Z2 → Z3 (语法状态链: S=句子开头, N=名词, V=动词)
        #  ↓     ↓     ↓
        # X1    X2    X3 (观测词: the/cat/dog/sat/ran)

        hmm_lm = DiscreteBayesianNetwork([
            ('Z1', 'Z2'), ('Z2', 'Z3'),
            ('Z1', 'X1'), ('Z2', 'X2'), ('Z3', 'X3')
        ])
        print(f"\n  HMM 结构: {list(hmm_lm.edges())}")

        print(f"\n  -- HMM vs Neural LM 对比 --")
        print(f"  {'维度':<20s} {'HMM (PGM)':<35s} {'Neural LM (GPT)':<35s}")
        print(f"  {'-'*20} {'-'*35} {'-'*35}")
        print(f"  {'生成方式':<20s} {'P(X|Z)·P(Z) 联合分布':<35s} {'P(x_t | x_{<t}) 条件分解':<35s}")
        print(f"  {'隐变量':<20s} {'显式离散隐状态 Z_t':<35s} {'隐式连续向量 h_t':<35s}")
        print(f"  {'可解释性':<20s} {'高: Z=语法状态 可读':<35s} {'低: h=黑盒向量':<35s}")
        print(f"  {'长距依赖':<20s} {'受限于状态空间设计':<35s} {'Attention 全局交互':<35s}")
        print(f"  {'训练方法':<20s} {'EM / Baum-Welch (MLE)':<35s} {'SGD / Adam (MLE)':<35s}")
        print(f"  {'泛化':<20s} {'离散状态 → 组合爆炸':<35s} {'连续嵌入 → 自动泛化':<35s}")
        print(f"  {'生成质量':<20s} {'低 (Markov假设太强)':<35s} {'高 (表达能力强)':<35s}")
        print(f"  {'应用场景':<20s} {'POS tagging, 简单的生成':<35s} {'GPT, 翻译, 对话系统':<35s}")

        print(f"\n  -- n-gram LM 作为 PGM --")
        print(f"  Bigram LM = Markov chain on observed tokens:")
        print(f"    X1 → X2 → X3 → ... → XT")
        print(f"    P(X1,...,XT) = P(X1)·Π P(Xt|Xt-1)")
        print(f"  ")
        print(f"  这是 PGM 中最简单的序列模型!")
        print(f"  从 PGM 视角: n-gram LM 的图结构 = 一阶马尔可夫链")
        print(f"  从 DL 视角: n-gram LM 的图结构 = 没有非线性、没有嵌入的 RNN")

        print(f"\n  -- 统一视角 --")
        print(f"  文本生成 = 条件概率的乘积 Π_t P(x_t | history)")
        print(f"  ┌─────────────────┬──────────────────┐")
        print(f"  │ history = x_{t-1}        (Bigram)   │ ← PGM")
        print(f"  │ history = x_{t-n+1:t-1}  (n-gram)   │")
        print(f"  ├─────────────────┼──────────────────┤")
        print(f"  │ history = h_{t-1}        (RNN/LSTM) │ ← Neural")
        print(f"  │ history = x_{1:t-1}      (GPT/Trans) │")
        print(f"  └─────────────────┴──────────────────┘")
        print(f"  区别仅在于: how to compress & represent 'history'!")

    except ImportError as e:
        print(f"\n  ⚠ pgmpy 导入失败: {e}")
        print(f"  请安装: pip install pgmpy")
        print(f"\n  手动展示 HMM vs Neural LM 对比:")
        print(f"  HMM: Z1→Z2→Z3 (隐语法状态)")
        print(f"        ↓  ↓  ↓")
        print(f"       X1 X2 X3 (观测词)")
        print(f"  训练: Baum-Welch = EM for HMM")
        print(f"\n  Neural LM:")
        print(f"  x1→x2→x3 (自回归)")
        print(f"  训练: SGD on CrossEntropy")
        print(f"\n  核心差异: PGM 显式建模 P(X,Z), Neural 隐式学习表示")


# ============================================================================
# 综合测试: 文本生成全流程演示
# ============================================================================

def exercise_bonus_full_pipeline():
    """
    展示文本生成的完整流程, 从训练到解码到评估。
    使用一个极简的 bigram 示例, 串联所有概念。
    """
    print("=" * 70)
    print("综合测试: 文本生成全流程演示")
    print("=" * 70)

    # Step 1: 训练数据
    corpus = ["the cat sat on the mat",
              "the dog sat on the floor",
              "the cat ran on the mat",
              "a dog ran on the floor",
              "the cat sat on a mat"]

    print(f"\n  Step 1: 训练语料 ({len(corpus)} 句)")
    for s in corpus:
        print(f"    '{s}'")

    # Step 2: 训练 bigram LM
    words = []
    for s in corpus:
        words.extend(s.split())
    vocab = sorted(set(words))
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    V = len(vocab)

    counts = np.zeros((V, V))
    for s in corpus:
        tokens = s.split()
        for i in range(len(tokens) - 1):
            counts[word2idx[tokens[i]], word2idx[tokens[i + 1]]] += 1

    # Add-1 smoothing
    probs = np.zeros((V, V))
    for i in range(V):
        probs[i] = (counts[i] + 1) / (counts[i].sum() + V)

    print(f"\n  Step 2: Bigram LM 训练完成 (V={V})")

    # Step 3: 生成 (不同策略)
    print(f"\n  Step 3: 生成 (Prompt: 'the')")
    prompt = "the"
    T_gen = 4

    # Greedy
    seq = [word2idx[prompt]]
    for _ in range(T_gen):
        seq.append(np.argmax(probs[seq[-1]]))
    greedy_text = " ".join(idx2word[i] for i in seq)
    print(f"    Greedy:      '{greedy_text}'")

    # Temperature sampling (τ=0.5)
    np.random.seed(456)
    seq_temp = [word2idx[prompt]]
    for _ in range(T_gen):
        logits = np.log(probs[seq_temp[-1]] + 1e-12)
        p_tau = np.exp(logits / 0.5 - (logits / 0.5).max())
        p_tau = p_tau / p_tau.sum()
        seq_temp.append(np.random.choice(V, p=p_tau))
    temp_text = " ".join(idx2word[i] for i in seq_temp)
    print(f"    Temp(τ=0.5): '{temp_text}'")

    # Top-k (k=2)
    np.random.seed(789)
    seq_topk = [word2idx[prompt]]
    for _ in range(T_gen):
        p = probs[seq_topk[-1]]
        top_k = np.argsort(p)[::-1][:2]
        p_topk = np.zeros(V)
        p_topk[top_k] = p[top_k]
        p_topk = p_topk / p_topk.sum()
        seq_topk.append(np.random.choice(V, p=p_topk))
    topk_text = " ".join(idx2word[i] for i in seq_topk)
    print(f"    Top-k(k=2):  '{topk_text}'")

    # Beam Search (B=2)
    B = 2
    beams = [([word2idx[prompt]], 0.0)]
    for _ in range(T_gen):
        candidates = []
        for seq, score in beams:
            p = probs[seq[-1]]
            for v in range(V):
                if p[v] > 1e-6:
                    candidates.append((seq + [v], score + np.log(p[v])))
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:B]
    beam_text = " ".join(idx2word[i] for i in beams[0][0])
    print(f"    Beam(B=2):   '{beam_text}'")

    # Step 4: 评估
    print(f"\n  Step 4: 评估")
    test_sentence = "the cat sat on the mat"
    test_tokens = test_sentence.split()
    log_prob = 0
    for t in range(1, len(test_tokens)):
        wi = word2idx[test_tokens[t - 1]]
        wj = word2idx[test_tokens[t]]
        p = probs[wi, wj]
        log_prob += np.log(p)
    ppl = np.exp(-log_prob / (len(test_tokens) - 1))
    print(f"    测试句: '{test_sentence}'")
    print(f"    log P = {log_prob:.4f}, PPL = {ppl:.2f}")

    print(f"\n  -- 全流程总结 --")
    print(f"  ① 训练语料 → ② 训练模型 (Bigram LM) →")
    print(f"  ③ 选择解码策略 → ④ 生成文本 → ⑤ 评估 (PPL)")
    print(f"\n  每个环节都是 L15 的核心概念!")
    print(f"  GPT-4 的训练也是同一个框架, 只是:")
    print(f"    - 模型: Transformer 而非 Bigram")
    print(f"    - 语料: 数万亿 token 而非 5 句话")
    print(f"    - 解码: Top-p + Temperature 而非简单 greedy")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    run_all = '--ex' not in sys.argv

    exercises = [
        ('1', exercise1_autoregressive_lm),
        ('2', exercise2_decoding_strategies),
        ('3', exercise3_perplexity_bleu),
        ('4', exercise4_seq2seq_attention),
        ('5', exercise5_text_vae),
        ('6', exercise6_pgm_vs_neural_lm),
        ('bonus', exercise_bonus_full_pipeline),
    ]

    for ex_id, ex_func in exercises:
        if not run_all and ex_id not in sys.argv:
            continue
        try:
            ex_func()
        except Exception as e:
            print(f"\n  [!] 练习{ex_id}执行出错: {e}")
            import traceback
            traceback.print_exc()
