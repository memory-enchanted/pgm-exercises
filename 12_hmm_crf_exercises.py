"""
=============================================================================
  CMU 10-708 L6 代码练习: HMM & CRF — 序列模型的推断
=============================================================================

本文件包含 5 个代码练习:

  练习 1: Forward 算法手写 — 逐步追踪 α 消息 (纯 numpy, 不用 pgmpy)
  练习 2: Viterbi 算法 — 解码最可能状态序列 + 回溯
  练习 3: HMM 作为 BayesianNetwork — 用 pgmpy BP/VE 推断
  练习 4: Forward-Backward = Sum-Product BP — 连接 L5 概念
  练习 5: CRF 势函数 — 对比 HMM 的局部归一化 vs CRF 的全局归一化

使用方法:
  python 12_hmm_crf_exercises.py           # 运行全部练习
  python 12_hmm_crf_exercises.py --ex 1    # 只运行练习1

依赖: numpy, pgmpy (仅练习3-4)
=============================================================================
"""

import numpy as np
from collections import defaultdict
import sys

# 修复 Windows GBK 终端下 emoji 打印问题
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


# ============================================================================
# 练习 1: Forward 算法 — 手算 α 消息
# ============================================================================

def exercise1_forward_algorithm():
    """
    在 HMM 上手写 Forward 算法, 逐步追踪每条 α 消息,
    验证: P(X) = Σ_j α_T(j)
    并将 α 解释为 L4 VE 的中间结果。
    """
    print("=" * 70)
    print("练习 1: Forward 算法 — 手算 α 消息 (L4 VE 视角)")
    print("=" * 70)

    # --- HMM 模型定义 ---
    # 两个隐藏状态: 0=Rainy, 1=Sunny  (K=2)
    # 三个观测值: 0=Walk, 1=Shop, 2=Clean  (|X|=3)

    K = 2  # 隐状态数
    M = 3  # 观测数
    T = 4  # 序列长度

    # π = P(Z₁)
    pi = np.array([0.6, 0.4])  # P(Z₁=Rainy)=0.6

    # A = P(Z_t | Z_{t-1}), 行=Z_t, 列=Z_{t-1}
    A = np.array([[0.7, 0.4],   # P(Rain→Rain)=0.7, P(Sun→Rain)=0.4
                   [0.3, 0.6]])  # P(Rain→Sun)=0.3,  P(Sun→Sun)=0.6

    # B = P(X_t | Z_t), 行=X_t, 列=Z_t
    B = np.array([[0.1, 0.6],   # P(Walk|Rain)=0.1,  P(Walk|Sun)=0.6
                   [0.4, 0.3],   # P(Shop|Rain)=0.4,  P(Shop|Sun)=0.3
                   [0.5, 0.1]])  # P(Clean|Rain)=0.5, P(Clean|Sun)=0.1

    # 观测序列
    obs_seq = [0, 1, 2, 0]  # Walk → Shop → Clean → Walk
    obs_names = ['Walk', 'Shop', 'Clean', 'Walk']
    state_names = ['Rainy', 'Sunny']

    print(f"\n  HMM: K={K} 状态 ({'/'.join(state_names)}), "
          f"|X|={M} 观测 ({'/'.join(['Walk','Shop','Clean'])}), T={T}")
    print(f"  观测序列: {' → '.join(obs_names)}")

    # --- 打印模型参数 ---
    print(f"\n  ── 模型参数 ──")
    print(f"  π = P(Z₁): {pi}")
    print("  A = P(Z_t|Z_{t-1}):")
    print(f"      {A[0,:]}")
    print(f"      {A[1,:]}")
    print("  B = P(X_t|Z_t) — 每列对应一个状态:")
    print(f"      Walk : {B[0,:]}")
    print(f"      Shop : {B[1,:]}")
    print(f"      Clean: {B[2,:]}")

    # --- Forward 算法 ---
    print(f"\n  ── Forward 算法 (逐步追踪 α 消息) ──\n")

    # 存储所有 α_t
    alpha = np.zeros((T, K))

    # Step 1: α₁(j) = π_j × B_j(X₁)
    print("  Step 1: 初始化 α₁")
    for j in range(K):
        alpha[0, j] = pi[j] * B[obs_seq[0], j]
        print(f"    α₁({state_names[j]}) = π_{j} × B_{j}({obs_names[0]}) "
              f"= {pi[j]:.1f} × {B[obs_seq[0], j]:.1f} = {alpha[0, j]:.4f}")
    print(f"    α₁ = {alpha[0]}")

    # Steps 2..T: α_t(j) = [Σ_i α_{t-1}(i) × A_{i→j}] × B_j(X_t)
    for t in range(1, T):
        print(f"\n  Step {t+1}: α_{t+1} = [Σ α_t × A] × B(X_{t+1})")
        for j in range(K):
            # 来自所有前一个状态的总和
            sum_from_prev = 0
            for i in range(K):
                contrib = alpha[t-1, i] * A[j, i]
                sum_from_prev += contrib
                print(f"    α_{t}({state_names[i]}) × A({state_names[i]}→{state_names[j]}) "
                      f"= {alpha[t-1,i]:.4f} × {A[j,i]:.1f} = {contrib:.4f}")
            alpha[t, j] = sum_from_prev * B[obs_seq[t], j]
            print(f"    → α_{t+1}({state_names[j]}) = "
                  f"{sum_from_prev:.4f} × {B[obs_seq[t],j]:.1f} = {alpha[t,j]:.4f}")

    # 最终: P(X) = Σ_j α_T(j)
    px = alpha[T-1].sum()
    print(f"\n  ── 最终结果 ──")
    print(f"  P(X) = Σ_j α_T(j) = {alpha[T-1,0]:.6f} + {alpha[T-1,1]:.6f} = {px:.6f}")

    # --- 解释: α_t 就是 VE 的中间结果 ---
    print(f"\n  ── L4 VE 视角重新解释 ──\n")
    print("  Forward 的 α_t(Z_t) 等于 VE 中消去 Z₁,...,Z_{t-1} 后的中间因子 τ(Z_t):")
    for t in range(T):
        print(f"    α_{t} = τ(Z_{t+1}) ← 消去 Z₁...Z_t 后, 剩下 scope={{Z_{t+1}}}")

    print("\n  🎯 洞察:")
    print("    Forward 算法 = 链上 VE, 消除顺序从左到右")
    print("    每个 α_t = 一条'正向消息' = '截至t的所有路径的概率'")
    print("    最终 P(X) = Σ α_T  — 所有路径概率之和")


# ============================================================================
# 练习 2: Viterbi 算法 — MAP 解码
# ============================================================================

def exercise2_viterbi_algorithm():
    """
    用 Viterbi 算法找最可能的隐状态序列。
    关键: 把 Σ 换成 max, 加回溯指针。
    """
    print("=" * 70)
    print("练习 2: Viterbi 算法 — MAP 解码 (Max-Product on chain)")
    print("=" * 70)

    # 使用与练习 1 相同的 HMM 参数
    K = 2
    M = 3
    T = 4
    pi = np.array([0.6, 0.4])
    A = np.array([[0.7, 0.4], [0.3, 0.6]])
    B = np.array([[0.1, 0.6], [0.4, 0.3], [0.5, 0.1]])
    obs_seq = [0, 1, 2, 0]
    state_names = ['Rainy', 'Sunny']

    print(f"\n  HMM 参数与练习 1 相同")
    print(f"  观测: {' → '.join(['Walk','Shop','Clean','Walk'])}")

    # --- Viterbi 算法 ---
    print(f"\n  ── Viterbi 算法 (Max-Product + 回溯) ──\n")

    delta = np.zeros((T, K))   # δ_t(j) = 到状态 j 的最优路径概率
    psi = np.zeros((T, K), dtype=int)  # ψ_t(j) = 回溯指针

    # Step 1: δ₁(j) = π_j × B_j(X₁)
    for j in range(K):
        delta[0, j] = pi[j] * B[obs_seq[0], j]
        psi[0, j] = 0  # 无前驱
    print(f"  δ₁ = {delta[0]}")

    # Steps 2..T: δ_t(j) = [max_i δ_{t-1}(i) × A_{i→j}] × B_j(X_t)
    for t in range(1, T):
        print(f"\n  Step {t+1}:")
        for j in range(K):
            candidates = np.zeros(K)
            for i in range(K):
                candidates[i] = delta[t-1, i] * A[j, i]
            best_i = np.argmax(candidates)
            delta[t, j] = candidates[best_i] * B[obs_seq[t], j]
            psi[t, j] = best_i
            print(f"    δ_{t+1}({state_names[j]}): max 前驱 = {state_names[best_i]} "
                  f"(值={candidates[best_i]:.4f}), × B = {delta[t,j]:.6f}")

    # 最终: 找最优末态
    best_last = np.argmax(delta[T-1])
    best_prob = delta[T-1, best_last]

    # 回溯
    best_path = [0] * T
    best_path[T-1] = best_last
    for t in range(T-2, -1, -1):
        best_path[t] = psi[t+1, best_path[t+1]]

    print(f"\n  ── 最优路径 (回溯) ──")
    path_str = ' → '.join([f"{state_names[s]}" for s in best_path])
    print(f"  Z* = {best_path} 即: {path_str}")
    print(f"  P(Z*, X) = {best_prob:.6f}")

    # --- 对比: Sum-Product (边际) vs Max-Product (MAP) ---
    print(f"\n  ── 对比: 边际最大值 vs Viterbi 路径 ──")

    # 用 Forward 算 α (用于平滑)
    alpha = np.zeros((T, K))
    alpha[0] = pi * B[obs_seq[0]]
    for t in range(1, T):
        for j in range(K):
            alpha[t, j] = np.dot(alpha[t-1], A[j, :]) * B[obs_seq[t], j]

    # 用 Backward 算 β
    beta = np.zeros((T, K))
    beta[T-1] = 1.0
    for t in range(T-2, -1, -1):
        for i in range(K):
            beta[t, i] = np.dot(A[:, i], B[obs_seq[t+1]] * beta[t+1])

    # 平滑边际 γ_t(j) ∝ α_t(j) × β_t(j)
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)

    for t in range(T):
        best_marginal = np.argmax(gamma[t])
        print(f"    t={t+1}: 边际最大 = {state_names[best_marginal]} "
              f"(γ={gamma[t]}), Viterbi = {state_names[best_path[t]]}")

    print("\n  🎯 洞察:")
    print("    Viterbi (Max-Product) = 找'单条最优路径'")
    print("    边际解码 (Sum-Product) = 每个位置独立选最大")
    print("    两者可能不同! 因为独立最大不一定构成合法路径")


# ============================================================================
# 练习 3: HMM 作为 BayesianNetwork — pgmpy 推断
# ============================================================================

def exercise3_hmm_as_bayesian_network():
    """
    把 HMM 表示为 DiscreteBayesianNetwork, 用 pgmpy 的 VE 和 BP 做推断。
    验证 Forward (=VE) 和 Forward-Backward (=BP) 的结果一致。
    """
    print("=" * 70)
    print("练习 3: HMM 作为 BayesianNetwork — pgmpy 推断")
    print("=" * 70)

    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination, BeliefPropagation

    # 构建 T=3 的 HMM:
    #   Z1 → Z2 → Z3
    #   ↓     ↓     ↓
    #   X1    X2    X3
    T = 3
    K = 2
    M = 2  # 观测二值: 0=Cold, 1=Hot

    model = DiscreteBayesianNetwork([
        ('Z1', 'Z2'), ('Z2', 'Z3'),
        ('Z1', 'X1'), ('Z2', 'X2'), ('Z3', 'X3'),
    ])

    # P(Z1)
    model.add_cpds(TabularCPD('Z1', K, [[0.6], [0.4]]))

    # P(Z2|Z1), P(Z3|Z2)
    model.add_cpds(TabularCPD('Z2', K, [[0.7, 0.3], [0.3, 0.7]],
                               evidence=['Z1'], evidence_card=[K]))
    model.add_cpds(TabularCPD('Z3', K, [[0.7, 0.3], [0.3, 0.7]],
                               evidence=['Z2'], evidence_card=[K]))

    # P(X1|Z1), P(X2|Z2), P(X3|Z3)
    model.add_cpds(TabularCPD('X1', M, [[0.9, 0.2], [0.1, 0.8]],
                               evidence=['Z1'], evidence_card=[K]))
    model.add_cpds(TabularCPD('X2', M, [[0.9, 0.2], [0.1, 0.8]],
                               evidence=['Z2'], evidence_card=[K]))
    model.add_cpds(TabularCPD('X3', M, [[0.9, 0.2], [0.1, 0.8]],
                               evidence=['Z3'], evidence_card=[K]))

    model.check_model()

    print(f"\n  HMM 结构: Z₁→Z₂→Z₃, Z_t→X_t")
    print(f"  参数: P(Z₁=0)=0.6, A=[[0.7,0.3],[0.3,0.7]], B=[[0.9,0.2],[0.1,0.8]]")

    # --- 任务 1: 用 VE 做滤波 P(Z_t | X₁...X_t) ---
    print(f"\n  ── 滤波 (Filtering): P(Z_t | X₁,...,X_t) ──")
    ve = VariableElimination(model)

    print("\n  P(Z1 | X1=Hot):")
    r = ve.query(['Z1'], evidence={'X1': 1}, show_progress=False)
    print(f"    {r.values}")

    print("\n  P(Z2 | X1=Hot, X2=Cold):")
    r = ve.query(['Z2'], evidence={'X1': 1, 'X2': 0}, show_progress=False)
    print(f"    {r.values}")

    print("\n  P(Z3 | X1=Hot, X2=Cold, X3=Hot):")
    r = ve.query(['Z3'], evidence={'X1': 1, 'X2': 0, 'X3': 1}, show_progress=False)
    print(f"    {r.values}")

    # --- 任务 2: 用 VE 做平滑 P(Z_t | X₁...X_T) ---
    print(f"\n  ── 平滑 (Smoothing): P(Z_t | 全部观测) ──")
    for t in [1, 2, 3]:
        r = ve.query([f'Z{t}'], evidence={'X1': 1, 'X2': 0, 'X3': 1},
                     show_progress=False)
        print(f"    P(Z{t} | X₁=Hot, X₂=Cold, X₃=Hot) = {r.values}")

    # --- 任务 3: Viterbi / MAP 解码 ---
    print(f"\n  ── MAP 解码 (Viterbi): 最可能的隐状态序列 ──")
    map_result = ve.map_query(variables=['Z1', 'Z2', 'Z3'],
                              evidence={'X1': 1, 'X2': 0, 'X3': 1},
                              show_progress=False)
    print(f"    Z* = {map_result}")

    # --- 任务 4: BP 验证 ---
    print(f"\n  ── BP 验证: 用 BeliefPropagation 计算所有边际 ──")
    bp = BeliefPropagation(model)
    bp.calibrate()

    # BP 不能直接用 evidence, 但可以比较无证据的边际
    for node in ['Z1', 'Z2', 'Z3', 'X1', 'X2', 'X3']:
        ve_r = ve.query([node], show_progress=False)
        bp_r = bp.query([node])
        match = np.allclose(ve_r.values, bp_r.values, atol=1e-6)
        print(f"    P({node}): VE={ve_r.values}, BP={bp_r.values}, {'✅' if match else '❌'}")

    print("\n  🎯 洞察:")
    print("    HMM 就是特殊结构的 BayesianNetwork")
    print("    VE/BP 的通用算法直接适用 — 不需要特殊处理!")
    print("    滤波 = VE with partial evidence, 平滑 = VE with full evidence")


# ============================================================================
# 练习 4: Forward-Backward = Sum-Product BP (连接 L5)
# ============================================================================

def exercise4_forward_backward_as_bp():
    """
    手写 Forward-Backward, 然后把 α 和 β 重新解释为 BP 消息。
    证明: Forward-Backward = Sum-Product on chain。
    """
    print("=" * 70)
    print("练习 4: Forward-Backward = Sum-Product BP (连接 L5)")
    print("=" * 70)

    # 使用简化的 HMM: K=3 状态, T=4
    K = 3
    T = 4
    state_names = ['S0', 'S1', 'S2']
    obs_names = ['Obs0', 'Obs1']

    # 参数
    pi = np.array([0.5, 0.3, 0.2])  # P(Z₁)
    A = np.array([[0.6, 0.2, 0.1],    # P(Z_t|Z_{t-1})
                   [0.3, 0.5, 0.3],
                   [0.1, 0.3, 0.6]])
    B = np.array([[0.8, 0.3, 0.2],    # P(X_t|Z_t)
                   [0.2, 0.7, 0.8]])
    obs = [0, 1, 0, 1]  # 观测序列

    # Step 1: Forward pass (α 消息)
    alpha = np.zeros((T, K))
    alpha[0] = pi * B[obs[0]]

    print("\n  ── Forward α 消息 (L5: 从左→右的 BP 消息) ──")
    print(f"  α₁ = π × B(X₁) = {np.round(alpha[0], 4)}")
    for t in range(1, T):
        for j in range(K):
            alpha[t, j] = np.dot(alpha[t-1], A[j, :]) * B[obs[t], j]
        print(f"  α_{t+1} = [Σ α_{t}(i)·A_{{i→j}}] × B(X_{t+1}) = {np.round(alpha[t], 4)}")

    # Step 2: Backward pass (β 消息)
    beta = np.zeros((T, K))
    beta[T-1] = 1.0

    print(f"\n  ── Backward β 消息 (L5: 从右→左的 BP 消息) ──")
    print(f"  β_T = [1, 1, 1]  (叶子无信息)")
    for t in range(T-2, -1, -1):
        for i in range(K):
            beta[t, i] = np.dot(A[:, i], B[obs[t+1]] * beta[t+1])
        print(f"  β_{t+1} = Σ_j A_{{i→j}}·B(X_{t+1})·β_{t+2}(j) = {np.round(beta[t], 4)}")

    # Step 3: γ = α × β (产品给出节点边际)
    gamma = alpha * beta
    gamma_norm = gamma / gamma.sum(axis=1, keepdims=True)

    print(f"\n  ── γ_t = α_t × β_t (L5: P(Z_t|X) = 入边消息乘积) ──")
    for t in range(T):
        print(f"  γ_{t+1}(Z) = α_{t+1} ⊙ β_{t+1} = {np.round(gamma[t], 4)}")
        print(f"    归一化: P(Z_{t+1}|X) = {np.round(gamma_norm[t], 4)}")

    # Step 4: L5 BP 视角
    print(f"\n  ── L5 Sum-Product BP 视角 ──")
    print("""
    在 HMM 链 Z₁—Z₂—Z₃—Z₄ 上:
      (X₁,X₂,X₃,X₄ 为观测, 已嵌入边因子)

    边因子: ψ(Z_t, Z_{t+1}) = P(Z_{t+1}|Z_t) × P(X_{t+1}|Z_{t+1})

    Forward 消息:
      m_{t→t+1}(Z_{t+1}) = Σ_{Z_t} ψ(Z_t, Z_{t+1}) × m_{t-1→t}(Z_t)
                         = α_{t+1}  ← Forward 的 α 就是 BP 消息!

    Backward 消息:
      m_{t+1→t}(Z_t) = Σ_{Z_{t+1}} ψ(Z_t, Z_{t+1}) × m_{t+2→t+1}(Z_{t+1})
                      = β_t  ← Backward 的 β 就是反向 BP 消息!

    节点边际:
      P(Z_t | X) ∝ m_{t-1→t}(Z_t) × m_{t+1→t}(Z_t)
                = α_t × β_t
                ∝ γ_t  ← 入边消息的逐元素乘积!

    完全对应 L5 公式:
      P(X_i) ∝ ∏_{k∈N(i)} m_{k→i}(X_i)
    """)

    # Step 5: 用暴力枚举验证 P(Z₁|X)
    print("  ── 暴力枚举验证 P(Z₁ | X₁₋₄) ──")
    # 枚举所有 3^4=81 种 Z₁...Z₄ 组合
    exact_gamma1 = np.zeros(K)
    total_prob = 0
    for z1 in range(K):
        for z2 in range(K):
            for z3 in range(K):
                for z4 in range(K):
                    p = (pi[z1] * B[obs[0], z1] *
                         A[z2, z1] * B[obs[1], z2] *
                         A[z3, z2] * B[obs[2], z3] *
                         A[z4, z3] * B[obs[3], z4])
                    exact_gamma1[z1] += p
                    total_prob += p
    exact_gamma1 /= total_prob

    print(f"  Forward-Backward: {np.round(gamma_norm[0], 4)}")
    print(f"  暴力枚举:         {np.round(exact_gamma1, 4)}")
    print(f"  匹配: {'✅ 完全一致' if np.allclose(gamma_norm[0], exact_gamma1, atol=1e-6) else '❌ 不匹配'}")

    print("\n  🎯 洞察:")
    print("    L5 的 Sum-Product BP = 树状图上的通用消息传递算法")
    print("    L6 的 Forward-Backward = 链状图上的 Sum-Product BP 特例")
    print("    α 和 β 不过是链上两个方向的消息 — 没有新算法, 只有新名字!")


# ============================================================================
# 练习 5: CRF vs HMM — 局部归一化 vs 全局归一化
# ============================================================================

def exercise5_crf_potentials():
    """
    对比 HMM 的局部归一化 (A 的每列 sum=1) 和 CRF 的全局归一化。
    演示 Label Bias 问题: 局部归一化如何导致观测信息被淹没。
    """
    print("=" * 70)
    print("练习 5: CRF 势函数 — Label Bias 演示")
    print("=" * 70)

    K = 3  # 状态数
    T = 3  # 序列长度

    state_names = ['S0', 'S1', 'S2']

    # --- 场景: 状态 S0 只有一条出边 — 强制到 S1 ---
    print("\n  ── 场景: 状态 S0 只能转移到 S1 ──")

    # HMM 参数
    pi = np.array([1.0, 0.0, 0.0])  # 必然从 S0 开始

    # 转移矩阵: S0 只能到 S1
    A_hmm = np.array([[0.0, 0.5, 0.2],    # P(S0|prev): S0→S0=0, S1→S0=0.5, S2→S0=0.2
                       [1.0, 0.3, 0.3],    # P(S1|prev): 唯一从S0出去的边!
                       [0.0, 0.2, 0.5]])   # P(S2|prev)

    # 发射矩阵: S1 强烈倾向于 Obs=0
    B_hmm = np.array([[0.9, 0.1, 0.5],
                       [0.1, 0.9, 0.5]])
    obs_hmm = [1, 0, 0]  # Obs1, Obs0, Obs0

    print(f"\n  HMM 转移矩阵 A (列归一化):")
    for i in range(K):
        print(f"    P(Z_{{t}}|{state_names[i]}): {A_hmm[:, i]}, sum={A_hmm[:, i].sum():.1f}")

    # Forward (HMM)
    alpha_hmm = np.zeros((T, K))
    alpha_hmm[0] = pi * B_hmm[obs_hmm[0]]
    for t in range(1, T):
        for j in range(K):
            alpha_hmm[t, j] = np.dot(alpha_hmm[t-1], A_hmm[j, :]) * B_hmm[obs_hmm[t], j]

    gamma_hmm = alpha_hmm[T-1] / alpha_hmm[T-1].sum()

    print(f"\n  HMM 平滑结果 P(Z_T | X):")
    for i in range(K):
        print(f"    P(Z_T={state_names[i]} | X) = {gamma_hmm[i]:.4f}")

    print("    因为 P(Z_t=S1|Z_{t-1}=S0)=1.0, 一旦模型进入 S0,")
    print("    无论观测是什么, 下一步都必然是 S1!")
    print("    观测 [Obs=1, Obs=0, Obs=0] 的第二位 Obs=0 强烈暗示 S0")
    print("    但 HMM 局部归一化强制 S0->S1 概率为 1, 信息丢失!")

    # --- CRF 风格: 用未归一化的势函数 ---
    print(f"\n  ── CRF 势函数 (未归一化, 全局归一化) ──")

    # CRF 的转移势函数: 不需要列归一化!
    psi_trans = np.array([[0.01, 0.5, 0.2],   # 势可以任意大小!
                           [3.0,  0.3, 0.3],   # S0→S1 的势 > 0
                           [0.01, 0.2, 0.5]])

    psi_emit = np.array([[0.9, 0.1, 0.5],
                          [0.1, 0.9, 0.5]])

    print("  CRF 转移势函数 ψ(Z_t, Z_{t-1}) — 不需要列归一化:")
    for i in range(K):
        for j in range(K):
            print(f"    ψ({state_names[i]}→{state_names[j]}) = {psi_trans[j, i]:.2f}")

    # CRF Forward (结构同 HMM, 但势函数不归一化)
    alpha_crf = np.zeros((T, K))
    alpha_crf[0] = pi * psi_emit[obs_hmm[0]]
    for t in range(1, T):
        for j in range(K):
            alpha_crf[t, j] = np.dot(alpha_crf[t-1], psi_trans[j, :]) * psi_emit[obs_hmm[t], j]

    gamma_crf = alpha_crf[T-1] / alpha_crf[T-1].sum()

    print(f"\n  CRF 结果 P(Z_T | X) (全局归一化):")
    for i in range(K):
        print(f"    P(Z_T={state_names[i]} | X) = {gamma_crf[i]:.4f}")

    print(f"\n  ✅ CRF 优势:")
    print(f"    转移势 ψ(S0→S1)=3.0 仅表示'偏好', 不是强制!")
    print(f"    观测的 emit 势可以和转移势竞争 — 信息不会丢失")
    print(f"    全局归一化 Z(X) = Σ_Z ∏ ψ 吸收了一切")

    # 对比表
    print(f"\n  ── HMM vs CRF 对比 ──")
    print(f"  {' ':20s} {'HMM':>15s} {'CRF':>15s}")
    print(f"  {'─'*20} {'─'*15} {'─'*15}")
    print(f"  {'转移参数形式':20s} {'概率 (列归一化)':>15s} {'势函数 (不归一化)':>15s}")
    print(f"  {'归一化方式':20s} {'局部 (每列 sum=1)':>15s} {'全局 (一个 Z(X))':>15s}")
    print(f"  {'Label Bias':20s} {'有':>15s} {'无':>15s}")
    print(f"  {'推断算法':20s} {'Forward/Viterbi':>15s} {'Forward/Viterbi':>15s}")

    print("\n  🎯 洞察:")
    print("    HMM 和 CRF 的推断算法完全相同!")
    print("    区别仅在于: 势函数是否局部归一化")
    print("    局部归一化 → Label Bias → CRF 用全局归一化解决")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    run_all = '--ex' not in sys.argv

    exercises = [
        ('1', exercise1_forward_algorithm, False),
        ('2', exercise2_viterbi_algorithm, False),
        ('3', exercise3_hmm_as_bayesian_network, True),
        ('4', exercise4_forward_backward_as_bp, False),
        ('5', exercise5_crf_potentials, False),
    ]

    for ex_id, ex_func, needs_pgmpy in exercises:
        if not run_all and ex_id not in sys.argv:
            continue
        try:
            if needs_pgmpy:
                import pgmpy
            ex_func()
        except ImportError:
            print(f"\n  [!] 练习{ex_id}需要 pgmpy。请先: pip install pgmpy")
        except Exception as e:
            print(f"\n  [!] 练习{ex_id}执行出错: {e}")
            import traceback
            traceback.print_exc()
