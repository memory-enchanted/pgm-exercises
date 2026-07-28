"""
==========================================================================================
  CMU 10-708 L14 代码练习: 深度序列模型
==========================================================================================

L14 五大主题 -> 对应练习:
  ① CNN for Sequences     -> 练习 1: 1D 因果卷积 — 时序预测
  ② RNN                   -> 练习 2: Vanilla RNN 从零 — 字符级语言模型
  ③ LSTM / GRU            -> 练习 3: LSTM 从零 — 门控机制手写
  ④ Attention Mechanisms  -> 练习 4: Scaled Dot-Product Attention — 手算验证
  ⑤ Multi-Head Attention  -> 练习 5: Multi-Head Self-Attention 实现
  ⑥ Transformer           -> 练习 6: Transformer Encoder Block

特别说明:
  - 贝叶斯网络导入需使用: from pgmpy.models import DiscreteBayesianNetwork
    (pgmpy 0.1.x 后 BayesianNetwork 改名为 DiscreteBayesianNetwork)
  - Windows GBK 终端下 emoji 打印: sys.stdout.reconfigure(encoding='utf-8')

使用方法:
  python 36_sequence_models_exercises.py           # 运行全部
  python 36_sequence_models_exercises.py --ex 1    # 只运行练习1

依赖: numpy, pgmpy
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
# 练习 1: 1D 因果卷积 — 时序预测
# ============================================================================

def exercise1_cnn_1d():
    """
    实现一维因果卷积 (Causal Conv1D), 并对比"普通卷积 vs 因果卷积"的区别。

    任务: 给定信号 x = [x0, x1, x2, x3, x4], kernel w = [w0, w1, w2] (K=3)
          分别计算普通卷积和因果卷积的输出。
    """
    print("=" * 70)
    print("练习 1: 1D 因果卷积 — 时序预测 (CNN for Sequences)")
    print("=" * 70)

    # 输入信号
    x = np.array([0.5, 1.0, -0.5, 0.3, 0.8])
    # 卷积核
    w = np.array([0.5, -0.3, 0.2])

    T = len(x)
    K = len(w)

    print(f"\n  输入信号 x({T}步): {x}")
    print(f"  卷积核 w({K}): {w}")

    # ==== 1a. 普通卷积 (Standard Conv1D) ====
    out_len_std = T - K + 1
    y_std = np.zeros(out_len_std)
    for t in range(out_len_std):
        y_std[t] = np.sum(x[t:t+K] * w)

    # ==== 1b. 因果卷积 (Causal Conv1D, 只允许看过去) ====
    # 左侧 pad K-1 个零, 保证时刻 t 只看 [t-K+1, t]
    x_padded = np.pad(x, (K-1, 0), mode='constant')  # [0, 0, x0, x1, x2, x3, x4]
    y_causal = np.zeros(T)
    for t in range(T):
        y_causal[t] = np.sum(x_padded[t:t+K] * w)

    print(f"\n  -- 普通卷积 (no padding, stride=1) --")
    for t in range(out_len_std):
        window = x[t:t+K]
        print(f"  y_std[{t+1}]  = {w[0]:+.1f}*{window[0]:+.1f} + "
              f"{w[1]:+.1f}*{window[1]:+.1f} + {w[2]:+.1f}*{window[2]:+.1f} = {y_std[t]:+.4f}")

    print(f"\n  -- 因果卷积 (causal, left-pad={K-1}) --")
    for t in range(T):
        window = x_padded[t:t+K]
        labels = [f"x_{t-K+1+i}" if t-K+1+i >= 0 else "pad" for i in range(K)]
        print(f"  y_causal[{t+1}] = {w[0]:+.1f}*{window[0]:+.1f}[{labels[0]}] + "
              f"{w[1]:+.1f}*{window[1]:+.1f}[{labels[1]}] + "
              f"{w[2]:+.1f}*{window[2]:+.1f}[{labels[2]}] = {y_causal[t]:+.4f}")

    # 验证: t=0 时刻, 因果卷积只看 [0, 0, x0] — 即只看当前位置+过去
    print(f"\n  -- 关键验证 --")
    print(f"  因果卷积 t=0: 只有 x_0 可见, x_1,x_2 被 mask → 适合时序预测!")
    print(f"  普通卷积 t=0: 看到了 x_0,x_1,x_2 → 有信息泄露 (偷看未来)!")

    # ==== 1c. 空洞卷积 (Dilated Conv) 示例 ====
    dilation = 2
    y_dilated = np.zeros(T - (K-1)*dilation)
    for t in range(len(y_dilated)):
        idx = t + np.arange(K) * dilation
        y_dilated[t] = np.sum(x[idx] * w)

    print(f"\n  -- 空洞卷积 (dilation={dilation}) --")
    print(f"  有效感受野: (K-1)*dilation + 1 = {(K-1)*dilation + 1}")
    print(f"  输出长度: {len(y_dilated)} (vs 普通卷积 {out_len_std})")
    for t in range(min(3, len(y_dilated))):
        print(f"  y_dilated[{t+1}] = {w[0]:+.1f}*x_{t} + "
              f"{w[1]:+.1f}*x_{t+dilation} + {w[2]:+.1f}*x_{t+2*dilation} = {y_dilated[t]:+.4f}")

    print(f"\n  洞察:")
    print(f"    普通卷积: 固定窗口, 有信息泄露 (对时序预测不适用)")
    print(f"    因果卷积: 只看过去, 可用于自回归生成 (WaveNet, GPT)")
    print(f"    空洞卷积: 跳跃采样, 指数级扩大感受野, 同时保持因果性")


# ============================================================================
# 练习 2: Vanilla RNN 从零 — 字符级序列预测
# ============================================================================

def exercise2_rnn_from_scratch():
    """
    纯 numpy 实现 vanilla RNN 做字符级序列预测。

    任务: 学习序列 "hello" -> "elloh" (每个字符后移一位)
    结构: input(emb_dim) -> RNN(hidden_dim) -> output(vocab_size)
    """
    print("=" * 70)
    print("练习 2: Vanilla RNN 从零 — 字符级语言模型 (RNN)")
    print("=" * 70)

    # 数据: "hello" -> 预测下一个字符
    chars = ['h', 'e', 'l', 'l', 'o']
    vocab = {c: i for i, c in enumerate(sorted(set(chars)))}
    idx_to_char = {i: c for c, i in vocab.items()}
    vocab_size = len(vocab)

    # One-hot 编码
    def one_hot(idx, size):
        v = np.zeros(size)
        v[idx] = 1.0
        return v

    # 序列: 输入 x[:-1], 目标 x[1:] (预测下一个字符)
    x_seq = [one_hot(vocab[c], vocab_size) for c in chars[:-1]]  # h,e,l,l
    y_target = [vocab[c] for c in chars[1:]]                       # e,l,l,o

    print(f"\n  词汇表: {vocab}")
    print(f"  输入序列: {[chars[i] for i in range(len(chars)-1)]}")
    print(f"  目标序列: {[idx_to_char[t] for t in y_target]}")

    # RNN 参数
    hidden_dim = 8
    input_dim = vocab_size
    output_dim = vocab_size

    # 初始化 (Kaiming/He)
    W_xh = np.random.randn(input_dim, hidden_dim) * 0.1
    W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.1
    b_h = np.zeros(hidden_dim)
    W_hy = np.random.randn(hidden_dim, output_dim) * 0.1
    b_y = np.zeros(output_dim)

    n_epochs = 500
    lr = 0.1
    seq_len = len(x_seq)

    print(f"\n  结构: {input_dim}(one-hot) → {hidden_dim}(RNN) → {output_dim}(softmax)")
    print(f"  训练中... (lr={lr})")

    train_losses = []

    for epoch in range(n_epochs):
        h = np.zeros(hidden_dim)  # 初始状态
        h_states = []  # 保存所有时间步的状态 (用于 BPTT)
        z_states = []  # 保存激活前的值 (用于反向传播)
        y_preds = []

        # ==== Forward Pass (Unrolling) ====
        total_loss = 0
        for t in range(seq_len):
            # RNN 单元
            z = x_seq[t] @ W_xh + h @ W_hh + b_h     # tanh 激活前
            h_new = np.tanh(z)
            # 输出
            y_logit = h_new @ W_hy + b_y
            y_prob = np.exp(y_logit - np.max(y_logit))  # 稳定 softmax
            y_prob = y_prob / np.sum(y_prob)

            z_states.append(z)
            h_states.append(h)      # 注意: 存的是之前的 h!
            y_preds.append(y_prob)
            h = h_new

            total_loss -= np.log(y_prob[y_target[t]] + 1e-12)

        avg_loss = total_loss / seq_len
        train_losses.append(avg_loss)

        # ==== Backward Pass (BPTT) ====
        dW_xh = np.zeros_like(W_xh)
        dW_hh = np.zeros_like(W_hh)
        db_h = np.zeros_like(b_h)
        dW_hy = np.zeros_like(W_hy)
        db_y = np.zeros_like(b_y)
        dh_next = np.zeros(hidden_dim)

        for t in reversed(range(seq_len)):
            # 输出层梯度
            dy = y_preds[t].copy()
            dy[y_target[t]] -= 1  # dCE/dŷ
            dW_hy += np.outer(h_states[t+1] if t+1 < len(h_states) else h, dy)
            db_y += dy
            dh = dy @ W_hy.T + dh_next

            # tanh 的梯度
            dz = dh * (1 - np.tanh(z_states[t])**2)
            dW_xh += np.outer(x_seq[t], dz)
            dW_hh += np.outer(h_states[t], dz)
            db_h += dz

            dh_next = dz @ W_hh.T

        # Gradient Clipping (防止梯度爆炸)
        for grad in [dW_xh, dW_hh, db_h, dW_hy, db_y]:
            norm = np.linalg.norm(grad)
            if norm > 5.0:
                grad *= 5.0 / norm

        # SGD 更新
        W_xh -= lr * dW_xh
        W_hh -= lr * dW_hh
        b_h -= lr * db_h
        W_hy -= lr * dW_hy
        b_y -= lr * db_y

        if epoch < 5 or epoch % 100 == 0 or epoch == n_epochs - 1:
            preds_str = ''.join([idx_to_char[np.argmax(p)] for p in y_preds])
            print(f"  epoch {epoch+1:>4d}: loss={avg_loss:.4f}, preds='{preds_str}'")

    # 测试: 用 "hell" 预测下一个字符
    print(f"\n  -- 最终测试 --")
    h_test = np.zeros(hidden_dim)
    preds_test = []
    for t in range(seq_len):
        z_test = x_seq[t] @ W_xh + h_test @ W_hh + b_h
        h_test = np.tanh(z_test)
        y_logit = h_test @ W_hy + b_y
        y_prob = np.exp(y_logit - np.max(y_logit))
        y_prob = y_prob / np.sum(y_prob)
        preds_test.append(idx_to_char[np.argmax(y_prob)])

    input_str = ''.join(chars[:-1])
    pred_str = ''.join(preds_test)
    target_str = ''.join([idx_to_char[t] for t in y_target])
    print(f"  输入: '{input_str}' -> 预测: '{pred_str}' (目标: '{target_str}')")

    print(f"\n  洞察:")
    print(f"    RNN = 循环状态机: h_t = tanh(W_xh·x_t + W_hh·h_{{t-1}} + b_h)")
    print(f"    BPTT = 在展开的计算图上反向传播 (时间维度上链式法则)")
    print(f"    梯度爆炸用 Gradient Clipping: ||g|| > threshold -> scale down")


# ============================================================================
# 练习 3: LSTM 从零 — 门控机制手写
# ============================================================================

def exercise3_lstm_from_scratch():
    """
    纯 numpy 实现 LSTM: 三个门 + 细胞状态 + 隐藏状态。

    任务: 给定一个短序列, 逐步计算 LSTM 在每个时刻的内部状态,
          并与 vanilla RNN 对比关键差异 (细胞状态 C_t 的梯度通路)。

    序列: [x1, x2, x3], 维恩 = 2D 输入, hidden_dim = 3
    """
    print("=" * 70)
    print("练习 3: LSTM 从零 — 门控机制详解 (LSTM)")
    print("=" * 70)

    # 小规模显式参数, 方便手算验证
    input_dim, hidden_dim = 2, 3

    # 输入序列 (3 个时间步)
    X = np.array([
        [1.0, 0.0],   # t=1
        [0.0, 1.0],   # t=2
        [0.5, 0.5],   # t=3
    ])

    # 手动设定权重 (小值, 便于观察)
    # 四个门 (f, i, o, g) 共用输入: [h_{t-1}, x_t] -> dim = hidden + input = 5
    concat_dim = hidden_dim + input_dim  # 5

    W_f = np.array([[ 0.2,  0.1, -0.1,  0.3,  0.0],
                     [-0.1,  0.2,  0.1,  0.0,  0.2],
                     [ 0.0, -0.1,  0.2,  0.1,  0.1]])
    W_i = np.array([[ 0.1,  0.0,  0.2, -0.1,  0.1],
                     [ 0.2,  0.1,  0.0,  0.2,  0.0],
                     [-0.1,  0.2,  0.1,  0.0,  0.2]])
    W_o = np.array([[ 0.0,  0.2,  0.1,  0.1, -0.1],
                     [ 0.1,  0.0,  0.2,  0.0,  0.1],
                     [ 0.2,  0.1,  0.0,  0.2,  0.0]])
    W_g = np.array([[ 0.3, -0.1,  0.2,  0.1,  0.0],
                     [ 0.0,  0.2, -0.1,  0.3,  0.1],
                     [ 0.1,  0.0,  0.3, -0.1,  0.2]])

    b_f = np.array([0.0, 0.0, 0.0])
    b_i = np.array([0.0, 0.0, 0.0])
    b_o = np.array([0.0, 0.0, 0.0])
    b_g = np.array([0.0, 0.0, 0.0])

    # 初始状态
    h = np.zeros(hidden_dim)
    C = np.zeros(hidden_dim)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    print(f"\n  LSTM: input_dim={input_dim}, hidden_dim={hidden_dim}")
    print(f"  输入序列 X: {X.tolist()}")
    print(f"\n  -- 逐步计算 --")

    for t in range(len(X)):
        x_t = X[t]
        concat = np.concatenate([h, x_t])  # [h_{t-1}, x_t], shape = (5,)

        # 四个门
        f_gate = sigmoid(W_f @ concat + b_f)    # 遗忘门
        i_gate = sigmoid(W_i @ concat + b_i)    # 输入门
        o_gate = sigmoid(W_o @ concat + b_o)    # 输出门
        g_cand = np.tanh(W_g @ concat + b_g)    # 候选记忆

        # 细胞状态 & 隐藏状态更新
        C_new = f_gate * C + i_gate * g_cand
        h_new = o_gate * np.tanh(C_new)

        print(f"\n  t={t+1}: x={x_t.tolist()}")
        print(f"    concat(h_{{t-1}}, x_t) = {[f'{v:.3f}' for v in concat]}")
        print(f"    f_gate (遗忘): {[f'{v:.3f}' for v in f_gate]}")
        print(f"    i_gate (输入): {[f'{v:.3f}' for v in i_gate]}")
        print(f"    o_gate (输出): {[f'{v:.3f}' for v in o_gate]}")
        print(f"    g_cand (候选): {[f'{v:.3f}' for v in g_cand]}")
        print(f"    C: {[f'{v:.3f}' for v in C]} -> {[f'{v:.3f}' for v in C_new]}")
        print(f"    h: {[f'{v:.3f}' for v in h]} -> {[f'{v:.3f}' for v in h_new]}")

        C, h = C_new, h_new

    # 对比: 用同样的输入跑 vanilla RNN
    print(f"\n  -- LSTM vs Vanilla RNN 对比 --")
    print(f"  LSTM 关键差异:")
    print(f"    1. 细胞状态 C_t: 梯度通过 f_t 直接传播 → ∂C_t/∂C_{{t-1}} = f_t (无 tanh')")
    print(f"    2. 遗忘门 f_t ∈ [0,1]: 自适应控制信息的丢弃/保留")
    print(f"    3. RNN:  ∂h_t/∂h_{{t-1}} 每次经过 tanh'() ≤ 1 → 连乘 T 次 → 梯度消失!")
    print(f"    4. LSTM: 当 f_t ≈ 1 时, 梯度可无损传播数百步")

    # 展示梯度通路
    print(f"\n  -- 梯度通路对比 --")
    print(f"  最后时刻的 f_gate = {[f'{v:.3f}' for v in f_gate]}")
    print(f"  如果 f ≈ 1 → ∂C_T/∂C_1 ≈ 1^T = 1 → 梯度不会消失!")
    print(f"  如果 f ≈ 0 → 该维度的历史被主动遗忘 → 聚焦于近期信息")


# ============================================================================
# 练习 4: Scaled Dot-Product Attention — 手算验证
# ============================================================================

def exercise4_scaled_dot_product_attention():
    """
    实现 Scaled Dot-Product Attention, 并逐步手算验证。

    Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V

    用极小的矩阵 (T=3, d_k=2) 展示每个中间步骤。
    """
    print("=" * 70)
    print("练习 4: Scaled Dot-Product Attention — 手算验证 (Attention)")
    print("=" * 70)

    # 小矩阵: T=3个token, d_k=2维
    Q = np.array([[1.0, 0.0],
                   [0.0, 1.0],
                   [1.0, 1.0]])   # (3, 2) — Query

    K = np.array([[1.0, 0.5],
                   [0.5, 1.0],
                   [0.0, 1.0]])   # (3, 2) — Key

    V = np.array([[1.0, 0.0],
                   [0.0, 1.0],
                   [0.5, 0.5]])   # (3, 2) — Value

    d_k = Q.shape[1]  # 2

    print(f"\n  Q (Query, T=3, d_k={d_k}):")
    print(f"    Q1={Q[0].tolist()}")
    print(f"    Q2={Q[1].tolist()}")
    print(f"    Q3={Q[2].tolist()}")
    print(f"\n  K (Key):")
    for i, k in enumerate(K):
        print(f"    K{i+1}={k.tolist()}")
    print(f"\n  V (Value):")
    for i, v in enumerate(V):
        print(f"    V{i+1}={v.tolist()}")

    # Step 1: Scores = Q @ K^T
    scores = Q @ K.T
    print(f"\n  -- Step 1: Scores = Q @ K^T --")
    print(f"  Scores matrix (3x3):")
    for i in range(3):
        row_str = "  ".join(f"{scores[i,j]:.3f}" for j in range(3))
        print(f"    Q{i+1}: [{row_str}]")

    # Step 2: Scale by 1/sqrt(d_k)
    scaled = scores / np.sqrt(d_k)
    print(f"\n  -- Step 2: Scale by 1/sqrt(d_k) = 1/{np.sqrt(d_k):.3f} --")
    print(f"  (防止点积过大 -> softmax梯度饱和)")
    for i in range(3):
        row_str = "  ".join(f"{scaled[i,j]:.3f}" for j in range(3))
        print(f"    Q{i+1}: [{row_str}]")

    # Step 3: Softmax (行归一化)
    def softmax(vec):
        e = np.exp(vec - np.max(vec))
        return e / np.sum(e)

    attn_weights = np.array([softmax(scaled[i]) for i in range(3)])
    print(f"\n  -- Step 3: Softmax (行归一化) --")
    print(f"  Attention Weights (3x3) — 每行求和=1:")
    for i in range(3):
        row_str = "  ".join(f"{attn_weights[i,j]:.3f}" for j in range(3))
        row_sum = attn_weights[i].sum()
        print(f"    Q{i+1}: [{row_str}] sum={row_sum:.3f}")

    # Step 4: Output = Attention @ V
    output = attn_weights @ V
    print(f"\n  -- Step 4: Output = Attention Weights @ V --")
    for i in range(3):
        print(f"    Out{i+1} = {[f'{v:.3f}' for v in output[i]]}")

    # 解释
    print(f"\n  -- 解读 --")
    for i in range(3):
        top_k = np.argmax(attn_weights[i])
        print(f"  Q{i+1} 最关注 K{top_k+1} (权重={attn_weights[i,top_k]:.3f})")

    # 验证除以 sqrt(d_k) 的必要性
    print(f"\n  -- 验证 scale 的必要性 --")
    dk_large = 64
    qk_prod = np.sum(np.random.randn(dk_large) * np.random.randn(dk_large))
    qk_prod_scaled = qk_prod / np.sqrt(dk_large)
    print(f"  假设 d_k={dk_large}, 随机 q·k = {qk_prod:.2f} (Var≈{dk_large})")
    print(f"  q·k/√d_k = {qk_prod_scaled:.2f} (Var≈1) ← softmax 不饱和!")
    print(f"  不 scale: softmax 的大值->接近1, 小值->接近0 -> 梯度消失")


# ============================================================================
# 练习 5: Multi-Head Self-Attention
# ============================================================================

def exercise5_multi_head_attention():
    """
    实现 Multi-Head Self-Attention (MHA).

    MHA(Q,K,V) = Concat(head_1, ..., head_h) @ W_O
    head_i = Attention(Q @ W^Q_i, K @ W^K_i, V @ W^V_i)

    任务: T=4个token, d_model=6, h=2个头, d_k=d_v=3
    """
    print("=" * 70)
    print("练习 5: Multi-Head Self-Attention 实现 (Transformer)")
    print("=" * 70)

    T = 4        # 序列长度
    d_model = 6  # 模型维度
    h = 2        # 头数
    d_k = 3      # 每头的 QK 维度
    d_v = 3      # 每头的 V 维度

    # 输入 (模拟 4 个 token 的嵌入)
    X = np.array([
        [0.5, 0.1, 0.3, 0.8, 0.2, 0.4],   # token 1
        [0.3, 0.7, 0.2, 0.1, 0.9, 0.5],   # token 2
        [0.1, 0.4, 0.6, 0.3, 0.2, 0.8],   # token 3
        [0.8, 0.2, 0.1, 0.5, 0.3, 0.7],   # token 4
    ])

    print(f"\n  输入 X: T={T}, d_model={d_model}")
    print(f"  Multi-Head: h={h} 个头, d_k={d_k}, d_v={d_v}")

    # 初始化投影矩阵 (He initialization)
    W_Q = np.random.randn(h, d_model, d_k) * np.sqrt(2.0 / d_model)
    W_K = np.random.randn(h, d_model, d_k) * np.sqrt(2.0 / d_model)
    W_V = np.random.randn(h, d_model, d_v) * np.sqrt(2.0 / d_model)
    W_O = np.random.randn(h * d_v, d_model) * np.sqrt(2.0 / (h * d_v))

    # Multi-Head Attention 实现
    def multi_head_attention(X, W_Q, W_K, W_V, W_O, h, d_k, d_v):
        T, d_model = X.shape
        heads = []

        for i in range(h):
            # 投影
            Q_i = X @ W_Q[i]   # (T, d_k)
            K_i = X @ W_K[i]   # (T, d_k)
            V_i = X @ W_V[i]   # (T, d_v)

            # Scaled Dot-Product Attention
            scores = Q_i @ K_i.T / np.sqrt(d_k)          # (T, T)
            attn = np.exp(scores - scores.max(axis=1, keepdims=True))
            attn = attn / attn.sum(axis=1, keepdims=True)  # softmax
            head_i = attn @ V_i                            # (T, d_v)
            heads.append(head_i)

        # Concat & Project
        concat = np.concatenate(heads, axis=1)  # (T, h*d_v)
        output = concat @ W_O                    # (T, d_model)
        return output

    output = multi_head_attention(X, W_Q, W_K, W_V, W_O, h, d_k, d_v)

    print(f"\n  -- Multi-Head Attention 输出 --")
    print(f"  输入 X: ({T}, {d_model})")
    print(f"  每个头: X -> Q_i({d_k}), K_i({d_k}), V_i({d_v})")
    print(f"          head_i = Attention(Q_i, K_i, V_i) -> ({T}, {d_v})")
    print(f"  Concat: [{h} × ({T}, {d_v})] -> ({T}, {h*d_v}={h*d_v})")
    print(f"  Output: ({T}, {h*d_v}) @ W_O({h*d_v}, {d_model}) -> ({T}, {d_model})")

    # 展示每个头的 attention patterns
    print(f"\n  -- 各头 Attention Pattern (以 Token 1 为例) --")
    for i in range(h):
        Q_i = X @ W_Q[i]
        K_i = X @ W_K[i]
        scores_i = Q_i[0] @ K_i.T / np.sqrt(d_k)
        attn_i = np.exp(scores_i - scores_i.max())
        attn_i = attn_i / attn_i.sum()
        pat = "  ".join(f"T{j+1}:{attn_i[j]:.3f}" for j in range(T))
        print(f"  Head {i+1}: [{pat}]")

    print(f"\n  -- 参数分析 --")
    n_params_qkv = h * 3 * d_model * d_k  # 3个投影矩阵, h个头
    n_params_o = h * d_v * d_model         # 输出投影
    total = n_params_qkv + n_params_o
    print(f"  QKV 投影参数: {n_params_qkv} (h×3×d_model×d_k = {h}×3×{d_model}×{d_k})")
    print(f"  W_O 投影参数: {n_params_o} (h×d_v×d_model = {h}×{d_v}×{d_model})")
    print(f"  总参数: {total}")

    # 验证 d_k = d_model / h 的设计
    print(f"\n  -- 设计原则 --")
    print(f"  1. d_k = d_v = d_model/h = {d_model}/{h} = {d_model//h}")
    print(f"     -> 保证 MHA 和单头 Attention 的计算量相近")
    print(f"  2. 每个头学习不同的 QKV 投影 -> 捕捉不同类型的依赖关系")
    print(f"  3. Concat + W_O -> 融合多头信息回到 d_model 空间")

    print(f"\n  洞察:")
    print(f"    单头: 只有一种'关注模式' -> 信息单一")
    print(f"    Multi-Head: h 个并行视角 -> 同时关注句法、语义、位置等!")


# ============================================================================
# 练习 6: Transformer Encoder Block
# ============================================================================

def exercise6_transformer_encoder():
    """
    组装 Transformer Encoder Block:
      x = x + MultiHeadSelfAttn(x)   (或 Pre-LN: x = x + Attn(LN(x)))
      x = x + FFN(x)

    FFN(x) = GELU(x·W1 + b1)·W2 + b2

    并展示 Positional Encoding 的生成。
    """
    print("=" * 70)
    print("练习 6: Transformer Encoder Block 组装 (Transformer)")
    print("=" * 70)

    T = 6        # 序列长度
    d_model = 8  # 模型维度
    d_ff = 16    # FFN 中间维度
    h = 2        # 头数
    d_k = d_model // h

    # 随机输入
    X = np.random.randn(T, d_model) * 0.5

    print(f"\n  Encoder Block: T={T}, d_model={d_model}, d_ff={d_ff}, h={h}")

    # ==== Positional Encoding ====
    print(f"\n  -- Positional Encoding (Sinusoidal) --")
    pos_enc = np.zeros((T, d_model))
    for pos in range(T):
        for i in range(0, d_model, 2):
            pos_enc[pos, i]   = np.sin(pos / (10000 ** (i / d_model)))
            pos_enc[pos, i+1] = np.cos(pos / (10000 ** (i / d_model)))

    print(f"  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))")
    print(f"  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))")
    print(f"\n  Positional Encoding Matrix ({T} x {d_model}):")
    print(f"  {'pos':>4s} " + "".join(f"  dim{i:<2d}" for i in range(d_model)))
    for pos in range(T):
        vals = "".join(f"  {pos_enc[pos, i]:+.2f}" for i in range(d_model))
        print(f"  {pos:>4d} {vals}")

    X_pe = X + pos_enc  # 将位置编码加到输入上

    # ==== Layer Normalization ====
    def layer_norm(x, eps=1e-6):
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return (x - mean) / (std + eps)

    # ==== FFN ====
    W1_ffn = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
    b1_ffn = np.zeros(d_ff)
    W2_ffn = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
    b2_ffn = np.zeros(d_model)

    def gelu(x):
        """Gaussian Error Linear Unit"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

    def ffn(x):
        hidden = gelu(x @ W1_ffn + b1_ffn)
        return hidden @ W2_ffn + b2_ffn

    # ==== Multi-Head Self-Attention ====
    W_Q = np.random.randn(h, d_model, d_k) * np.sqrt(2.0 / d_model)
    W_K = np.random.randn(h, d_model, d_k) * np.sqrt(2.0 / d_model)
    W_V = np.random.randn(h, d_model, d_k) * np.sqrt(2.0 / d_model)
    W_O = np.random.randn(h * d_k, d_model) * np.sqrt(2.0 / (h * d_k))

    def multi_head_self_attn(x):
        heads = []
        for i in range(h):
            Q_i = x @ W_Q[i]
            K_i = x @ W_K[i]
            V_i = x @ W_V[i]
            scores = Q_i @ K_i.T / np.sqrt(d_k)
            attn = np.exp(scores - scores.max(axis=1, keepdims=True))
            attn = attn / attn.sum(axis=1, keepdims=True)
            heads.append(attn @ V_i)
        concat = np.concatenate(heads, axis=1)
        return concat @ W_O

    # ==== Full Encoder Block (Post-LN style) ====
    # Sub-layer 1: Self-Attention + Residual + LN
    attn_out = multi_head_self_attn(X_pe)
    x1 = X_pe + attn_out           # Residual
    x1_ln = layer_norm(x1)         # LayerNorm

    # Sub-layer 2: FFN + Residual + LN
    ffn_out = ffn(x1_ln)
    x2 = x1_ln + ffn_out           # Residual
    encoder_out = layer_norm(x2)   # LayerNorm

    print(f"\n  -- Encoder Block 数据流 --")
    print(f"  Input X:        ({T}, {d_model})")
    print(f"  + Pos Encoding: ({T}, {d_model})")
    print(f"  -> Self-Attn:   ({T}, {d_model})")
    print(f"  -> + Residual:  ({T}, {d_model})  ← 梯度高速路!")
    print(f"  -> + LayerNorm: ({T}, {d_model})")
    print(f"  -> FFN:         ({T}, {d_ff}) -> ({T}, {d_model})  ← 先扩后缩")
    print(f"  -> + Residual:  ({T}, {d_model})")
    print(f"  -> + LayerNorm: ({T}, {d_model})")
    print(f"  Output:         ({T}, {d_model})")

    print(f"\n  -- 残差连接的作用 --")
    print(f"  Without Residual: 梯度需穿过 Attention + FFN (可能消失)")
    print(f"  With Residual:    梯度可走'高速路'直接回流 → 可堆叠 N 层")

    print(f"\n  -- FFN 的'先扩后缩' --")
    print(f"  d_model={d_model} -> d_ff={d_ff} ({d_ff//d_model}x) -> d_model={d_model}")
    print(f"  直觉: 给每个 token 更多'思考空间', 然后压缩回原维度")


# ============================================================================
# 练习 7: PGM 视角 — HMM 和 RNN 的对比
# ============================================================================

def exercise7_pgm_vs_rnn():
    """
    用 pgmpy 构建一个简单的 HMM (作为概率图序列模型),
    并与 RNN/LSTM 对比, 展示两种序列建模范式的异同。

    这是 L14 的核心洞见: PGM 序列模型(HMM) vs 神经序列模型(RNN/LSTM)

    注意: pgmpy 0.1.x 后 BayesianNetwork 改名为 DiscreteBayesianNetwork,
         导入方式: from pgmpy.models import DiscreteBayesianNetwork
    """
    print("=" * 70)
    print("练习 7: PGM 视角 — HMM vs RNN 序列建模对比")
    print("=" * 70)

    try:
        from pgmpy.models import DiscreteBayesianNetwork
        from pgmpy.factors.discrete import TabularCPD

        print("\n  ✅ pgmpy 导入成功 (DiscreteBayesianNetwork)")
        print("     注意: pgmpy>=0.1.20 中 BayesianNetwork -> DiscreteBayesianNetwork")

        # 构建 3 状态 × 2 观测的 HMM
        #  Z1 -> Z2 -> Z3 (隐状态链)
        #   |     |     |
        #  X1    X2    X3 (观测)
        hmm = DiscreteBayesianNetwork([
            ('Z1', 'Z2'), ('Z2', 'Z3'),
            ('Z1', 'X1'), ('Z2', 'X2'), ('Z3', 'X3')
        ])
        print(f"\n  HMM 结构: {list(hmm.edges())}")

        print(f"\n  -- HMM vs RNN 对比 --")
        print(f"  {'维度':<20s} {'HMM (PGM)':<30s} {'RNN/LSTM (NN)':<30s}")
        print(f"  {'-'*20} {'-'*30} {'-'*30}")
        print(f"  {'状态表示':<20s} {'离散隐变量 Z_t ∈ {1..K}':<30s} {'连续向量 h_t ∈ R^d':<30s}")
        print(f"  {'转移/更新':<20s} {'P(Z_t | Z_{{t-1}}) 条件概率表':<30s} {'h_t = tanh(W·h_{{t-1}}+U·x_t)':<30s}")
        print(f"  {'发射/输出':<20s} {'P(X_t | Z_t) 条件概率表':<30s} {'y_t = softmax(W·h_t)':<30s}")
        print(f"  {'推断':<20s} {'Forward-Backward (精确边际)':<30s} {'前向传播 (点估计)':<30s}")
        print(f"  {'学习':<20s} {'EM / Baum-Welch (MLE)':<30s} {'BPTT + SGD (MLE)':<30s}")
        print(f"  {'不确定性':<20s} {'天然的全概率分布':<30s} {'需特殊技巧 (BNN, MC Dropout)':<30s}")
        print(f"  {'可解释性':<20s} {'高 — 转移/发射矩阵可读':<30s} {'低 — 权重矩阵黑盒':<30s}")
        print(f"  {'长序列':<20s} {'受限于状态空间大小':<30s} {'LSTM门控→可处理数百步':<30s}")

        print(f"\n  -- 核心洞见 --")
        print(f"  HMM:  Z1→Z2→Z3 (马尔可夫链)")
        print(f"  RNN:  h1→h2→h3 (确定性循环)")
        print(f"  共同点: 都是'前一时刻影响后一时刻'的序列模型")
        print(f"  不同点: HMM用概率转移, RNN用确定性函数+非线性")
        print(f"  LSTM 的 f·C_{{t-1}} + i·g ≈ 概率化的信息保留!")

    except ImportError as e:
        print(f"\n  ⚠ pgmpy 导入失败: {e}")
        print(f"  请安装: pip install pgmpy")
        print(f"\n  手动展示 HMM vs RNN 对比:")
        print(f"  HMM: 离散隐变量 + 转移概率表 + EM学习")
        print(f"  RNN: 连续隐状态 + 确定性更新 + SGD学习")
        print(f"  LSTM: 门控 = 可微的'软'概率转移!")


# ============================================================================
# 综合测试: 自注意力矩阵可视化
# ============================================================================

def exercise_bonus_attention_visualization():
    """
    生成一个 attention pattern 示例并展示不同位置的关系。
    不需要 matplotlib, 直接在控制台用 ASCII 展示。
    """
    print("=" * 70)
    print("综合测试: 自注意力矩阵可视化 (ASCII)")
    print("=" * 70)

    T = 8
    d_model = 4

    # 模拟序列
    tokens = ["The", "cat", "sat", "on", "the", "mat", ".", "<EOS>"]

    X = np.random.randn(T, d_model) * 0.5

    # 单头 Self-Attention
    d_k = d_model
    W_Q = np.random.randn(d_model, d_k) * 0.3
    W_K = np.random.randn(d_model, d_k) * 0.3
    W_V = np.random.randn(d_model, d_model) * 0.3

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    scores = Q @ K.T / np.sqrt(d_k)
    attn = np.exp(scores - scores.max(axis=1, keepdims=True))
    attn = attn / attn.sum(axis=1, keepdims=True)

    print(f"\n  Attention Pattern (T={T}) — 行=Query, 列=Key:")
    print(f"  {'':>6s}", end="")
    for tok in tokens:
        print(f"{tok:>7s}", end="")
    print()

    for i, tok_q in enumerate(tokens):
        print(f"  {tok_q:>6s}", end="")
        for j in range(T):
            val = attn[i, j]
            # 用符号表示注意力强度
            if val > 0.3:
                bar = "██"
            elif val > 0.15:
                bar = "▓▓"
            elif val > 0.05:
                bar = "░░"
            else:
                bar = "  "
            print(f"  {bar:<4s}", end="")
        print()

    print(f"\n  图例: ██=强(>.3)  ▓▓=中(>.15)  ░░=弱(>.05)  =极弱(<.05)")
    print(f"\n  解读:")
    print(f"  - 对角线通常最强 (每个token关注自己)")
    print(f"  - 语义相关的token会产生跨位置的强attention")
    print(f"  - 这就是Transformer能捕获长距离依赖的秘密!")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    run_all = '--ex' not in sys.argv

    exercises = [
        ('1', exercise1_cnn_1d),
        ('2', exercise2_rnn_from_scratch),
        ('3', exercise3_lstm_from_scratch),
        ('4', exercise4_scaled_dot_product_attention),
        ('5', exercise5_multi_head_attention),
        ('6', exercise6_transformer_encoder),
        ('7', exercise7_pgm_vs_rnn),
        ('bonus', exercise_bonus_attention_visualization),
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
