"""
=============================================================================
  CMU 10-708 L7 代码练习: 变分推断 I — Mean-Field & CAVI
=============================================================================

本文件包含 5 个代码练习:

  练习 1: KL 散度手写 — 两个离散分布间的 KL, 验证不对称性
  练习 2: ELBO 分解验证 — log P(X) = ELBO + KL, 逐个验证
  练习 3: Mean-Field CAVI — 2变量模型, 坐标上升从零实现
  练习 4: Mean-Field VI for Bayesian Network — 与 VE 精确解对比
  练习 5: VI 定点方程 = BP 消息 — 树上的等价性

使用方法:
  python 15_vi_exercises.py           # 运行全部练习
  python 15_vi_exercises.py --ex 1    # 只运行练习1

依赖: numpy, scipy (可选, 仅用于 logsumexp), pgmpy (仅练习4-5)
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
# 工具函数
# ============================================================================

def logsumexp(x, axis=None):
    """数值稳定的 log-sum-exp"""
    x_max = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True)))


def ensure_normalized(arr, axis=-1):
    """确保分布归一化"""
    s = arr.sum(axis=axis, keepdims=True)
    return arr / s


# ============================================================================
# 练习 1: KL 散度手写
# ============================================================================

def exercise1_kl_divergence():
    """
    计算两个离散分布之间的 KL 散度, 验证不对称性:
    KL(P||Q) != KL(Q||P)
    """
    print("=" * 70)
    print("练习 1: KL 散度 — 分布间距离的度量")
    print("=" * 70)

    # 定义两个简单的 3 值分布
    P = np.array([0.1, 0.2, 0.7])
    Q = np.array([0.4, 0.4, 0.2])

    K = len(P)

    print(f"\n  P = {P}")
    print(f"  Q = {Q}")

    # KL(P || Q) = Σ_i P(i) * log(P(i) / Q(i))
    kl_pq = 0.0
    for i in range(K):
        if P[i] > 0 and Q[i] > 0:
            kl_pq += P[i] * np.log(P[i] / Q[i])

    # KL(Q || P) = Σ_i Q(i) * log(Q(i) / P(i))
    kl_qp = 0.0
    for i in range(K):
        if Q[i] > 0 and P[i] > 0:
            kl_qp += Q[i] * np.log(Q[i] / P[i])

    print(f"\n  KL(P||Q) = {kl_pq:.4f}")
    print(f"  KL(Q||P) = {kl_qp:.4f}")
    print(f"  不对称: {'✅ 确实不对称' if abs(kl_pq - kl_qp) > 1e-6 else '❌ 意外对称'}")

    # 可视化两个 KL 的含义
    print(f"\n  KL(P||Q) (mode-seeking): P 集中处 Q 不能太小")
    print(f"    P(2)=0.7, Q(2)=0.2 → KL 中该项贡献 P(2)*log(0.7/0.2)={P[2]*np.log(P[2]/Q[2]):.4f}")
    print(f"    (如果 Q(2) 很小, log 项爆炸 → KL 很大 → 惩罚 Q)")
    print(f"\n  KL(Q||P) (mass-covering): Q 集中处 P 不能太小")
    print(f"    Q(0)=0.4, P(0)=0.1 → KL 中该项贡献 Q(0)*log(0.4/0.1)={Q[0]*np.log(Q[0]/P[0]):.4f}")

    # 验证 KLD 的非负性
    print(f"\n  非负性检验:")
    print(f"    KL(P||Q) = {kl_pq:.4f} >= 0 ✅")
    print(f"    KL(Q||P) = {kl_qp:.4f} >= 0 ✅")
    print(f"    KL(P||P) = {np.sum(P * np.log(P / P)):.4f} = 0 ✅")

    print("\n  🎯 洞察:")
    print("    KL(P||Q) != KL(Q||P) — 不是对称的距离度量")
    print("    VI 用 KL(Q||P): mode-seeking — Q 避开 P 概率低的区域")
    print("    EP 用 KL(P||Q): mass-covering — Q 必须覆盖 P 的所有可能值")


# ============================================================================
# 练习 2: ELBO 分解验证
# ============================================================================

def exercise2_elbo_decomposition():
    """
    在一个简单的 2 变量模型上验证:
    log P(X) = ELBO(Q) + KL(Q(Z) || P(Z|X))

    其中:
    ELBO(Q) = E_Q[log P(X, Z)] - E_Q[log Q(Z)]
    """
    print("=" * 70)
    print("练习 2: ELBO 分解 — log P(X) = ELBO + KL")
    print("=" * 70)

    # --- 定义真实联合分布 P(X, Z) ---
    # Z ∈ {0, 1, 2}, X ∈ {0, 1}
    K = 3  # |Z|
    M = 2  # |X|

    # P(Z): 先验
    p_z = np.array([0.3, 0.5, 0.2])

    # P(X|Z): 似然
    p_x_given_z = np.array([[0.9, 0.4, 0.1],   # P(X=0|Z)
                             [0.1, 0.6, 0.9]])  # P(X=1|Z)

    # 假设观测 X=1
    x_obs = 1

    # 计算后验 P(Z|X=1)
    joint = p_z * p_x_given_z[x_obs]
    p_x = joint.sum()
    posterior = joint / p_x

    print(f"\n  P(Z): {p_z}")
    print(f"  P(X=1|Z): {p_x_given_z[x_obs]}")
    print(f"  P(X=1) = Σ_Z P(Z)·P(X=1|Z) = {p_x:.4f}")
    print(f"  P(Z|X=1) = {posterior}")

    # --- 定义变分分布 Q(Z) ---
    # 随便选一个初始 Q（不等于后验）
    q_z_init = np.array([0.5, 0.3, 0.2])

    print(f"\n  ── 验证 1: 初始 Q = {q_z_init} ──")

    # ELBO = Σ_Z Q(Z) * [log P(X, Z) - log Q(Z)]
    elbo_init = 0.0
    for i in range(K):
        if q_z_init[i] > 0:
            elbo_init += q_z_init[i] * (np.log(joint[i]) - np.log(q_z_init[i]))

    # KL(Q || P(Z|X))
    kl_init = 0.0
    for i in range(K):
        if q_z_init[i] > 0:
            kl_init += q_z_init[i] * np.log(q_z_init[i] / posterior[i])

    print(f"    ELBO(Q) = {elbo_init:.4f}")
    print(f"    KL(Q||P) = {kl_init:.4f}")
    print(f"    ELBO + KL = {elbo_init + kl_init:.4f}")
    print(f"    log P(X=1) = {np.log(p_x):.4f}")
    print(f"    匹配: {'✅' if abs(elbo_init + kl_init - np.log(p_x)) < 1e-6 else '❌'}")

    # --- 验证 2: Q = 真实后验时 ---
    q_z_best = posterior.copy()
    print(f"\n  ── 验证 2: Q = P(Z|X) = {posterior} ──")

    elbo_best = 0.0
    for i in range(K):
        if q_z_best[i] > 0:
            elbo_best += q_z_best[i] * (np.log(joint[i]) - np.log(q_z_best[i]))

    kl_best = 0.0
    for i in range(K):
        if q_z_best[i] > 0:
            kl_best += q_z_best[i] * np.log(q_z_best[i] / posterior[i])

    print(f"    ELBO(Q) = {elbo_best:.4f}")
    print(f"    KL(Q||P) = {kl_best:.10f}")
    print(f"    ELBO + KL = {elbo_best + kl_best:.4f}")
    print(f"    当 Q=P(Z|X) 时: KL=0, ELBO=log P(X) ✅")

    # --- 展示 ELBO 随 Q 的变化 ---
    print(f"\n  ── ELBO 随 Q 的变化 (网格搜索) ──")
    # 枚举 Q=(q0, q1, 1-q0-q1) 在单纯形上的点
    best_elbo = -np.inf
    best_q = None
    for i in range(11):
        q0 = i / 10.0
        for j in range(11 - i):
            q1 = j / 10.0
            q2 = 1.0 - q0 - q1
            if q2 < 0:
                continue
            q = np.array([q0, q1, q2])
            elbo = 0.0
            for k in range(K):
                if q[k] > 0:
                    elbo += q[k] * (np.log(joint[k]) - np.log(q[k]))
            if elbo > best_elbo:
                best_elbo = elbo
                best_q = q.copy()

    print(f"    网格搜索最优 Q = {best_q}")
    print(f"    真实后验 P(Z|X) = {posterior}")
    print(f"    匹配: {'✅ VI 找到正确后验' if np.allclose(best_q, posterior, atol=1e-1) else '❌'}")

    print("\n  🎯 洞察:")
    print("    ELBO + KL(Q||P) = log P(X) — 恒等式!")
    print("    最大化 ELBO ⟺ 最小化 KL(Q||P)")
    print("    当 Q = 真实后验时: KL=0, ELBO 达到最大值 log P(X)")


# ============================================================================
# 练习 3: Mean-Field CAVI — 2 变量模型从零实现
# ============================================================================

def exercise3_mean_field_cavi():
    """
    对 2 变量模型 p(Z₁, Z₂, X) 实现 Mean-Field CAVI:
    Q(Z₁, Z₂) = Q₁(Z₁) × Q₂(Z₂)

    跟踪每次迭代的 ELBO, 展示单调上升。
    """
    print("=" * 70)
    print("练习 3: Mean-Field CAVI — 2 变量模型")
    print("=" * 70)

    # --- 模型: P(Z₁, Z₂, X) ---
    # Z₁ ∈ {0, 1}, Z₂ ∈ {0, 1}, 观测 X=1
    K1, K2 = 2, 2

    # 联合概率表 P(Z₁, Z₂, X=1):
    # 格式: joint[z1, z2] = P(Z₁=z1, Z₂=z2, X=1) (未归一化)
    # 选择非因子化的 log_joint — Z₁和Z₂ 有真实相关性
    log_joint = np.array([[-2.0, -0.5],    # Z₁=0: Z₂=0→低, Z₂=1→高 (反相关!)
                           [-0.5, -2.0]])   # Z₁=1: Z₂=0→高, Z₂=1→低 (反相关!)

    joint = np.exp(log_joint)
    log_px = logsumexp(log_joint)
    px = np.exp(log_px)

    print(f"\n  log P(Z₁, Z₂, X=1):")
    print(f"      Z₂=0  Z₂=1")
    for z1 in range(K1):
        print(f"  Z₁={z1} {log_joint[z1,0]:6.2f} {log_joint[z1,1]:6.2f}")
    print(f"  log P(X=1) = {log_px:.4f}")

    # 精确后验
    exact_posterior = joint / px
    print(f"\n  精确后验 P(Z₁,Z₂|X=1):")
    for z1 in range(K1):
        for z2 in range(K2):
            print(f"    P(Z₁={z1}, Z₂={z2}|X) = {exact_posterior[z1, z2]:.4f}")

    # --- Mean-Field CAVI ---
    print(f"\n  ── Mean-Field CAVI 迭代 ──\n")

    # 初始化 Q₁, Q₂ (随机, 非均匀)
    q1 = np.array([0.5, 0.5])
    q2 = np.array([0.5, 0.5])

    max_iters = 20
    elbo_history = []

    for iteration in range(max_iters):
        # --- 更新 Q₁(Z₁) ---
        # log Q₁*(z₁) = E_{Q₂}[log P(Z₁, Z₂, X)] + const
        log_q1_new = np.zeros(K1)
        for z1 in range(K1):
            log_q1_new[z1] = np.dot(q2, log_joint[z1, :])  # E_{Q₂}[log P]
        # 归一化
        log_q1_new -= logsumexp(log_q1_new)
        q1 = np.exp(log_q1_new)

        # --- 更新 Q₂(Z₂) ---
        # log Q₂*(z₂) = E_{Q₁}[log P(Z₁, Z₂, X)] + const
        log_q2_new = np.zeros(K2)
        for z2 in range(K2):
            log_q2_new[z2] = np.dot(q1, log_joint[:, z2])  # E_{Q₁}[log P]
        log_q2_new -= logsumexp(log_q2_new)
        q2 = np.exp(log_q2_new)

        # --- 计算 ELBO ---
        # ELBO = E_Q[log P] - E_Q[log Q]
        #      = Σ_{z1,z2} q1(z1)·q2(z2)·log P(z1,z2,X) - Σ q1·log q1 - Σ q2·log q2
        elbo = 0.0
        for z1 in range(K1):
            for z2 in range(K2):
                elbo += q1[z1] * q2[z2] * log_joint[z1, z2]

        # 熵项
        for z1 in range(K1):
            if q1[z1] > 0:
                elbo -= q1[z1] * np.log(q1[z1])
        for z2 in range(K2):
            if q2[z2] > 0:
                elbo -= q2[z2] * np.log(q2[z2])

        elbo_history.append(elbo)

        if iteration < 5 or iteration == max_iters - 1:
            kl_q_p = 0.0
            for z1 in range(K1):
                for z2 in range(K2):
                    q_joint = q1[z1] * q2[z2]
                    if q_joint > 0 and exact_posterior[z1, z2] > 0:
                        kl_q_p += q_joint * np.log(q_joint / exact_posterior[z1, z2])

            print(f"  iter {iteration+1:2d}: Q₁={np.round(q1,4)}, Q₂={np.round(q2,4)}, "
                  f"ELBO={elbo:.4f}, KL={kl_q_p:.4f}")

    # 验证 ELBO 单调上升
    increasing = all(elbo_history[i] <= elbo_history[i+1] + 1e-10
                     for i in range(len(elbo_history)-1))
    print(f"\n  ELBO 单调上升: {'✅' if increasing else '❌'}")
    print(f"    ELBO 起点: {elbo_history[0]:.4f}")
    print(f"    ELBO 终点: {elbo_history[-1]:.4f}")
    print(f"    log P(X):  {log_px:.4f} (上界)")
    print(f"    gap = log P(X) - ELBO = KL(Q||P) = {log_px - elbo_history[-1]:.6f}")

    # 展示 VI 近似的后验
    print(f"\n  ── VI 近似的后验 Q(Z₁,Z₂) = Q₁·Q₂ᵀ ──")
    q_joint = np.outer(q1, q2)
    for z1 in range(K1):
        for z2 in range(K2):
            print(f"    Q(Z₁={z1}, Z₂={z2}) = {q_joint[z1, z2]:.4f}  "
                  f"(真实: {exact_posterior[z1, z2]:.4f})")

    # --- 解释: Mean-Field 近似误差的来源 ---
    print(f"\n  ── 为什么有近似误差? ──")
    # 真实后验的互信息
    exact_p1 = exact_posterior.sum(axis=1)  # Z₁ 边际
    exact_p2 = exact_posterior.sum(axis=0)  # Z₂ 边际
    print(f"    真实后验 P(Z₁,Z₂|X):")
    print(f"      P(Z₁=0)=1-P(Z₁=1)={exact_p1[0]:.3f}, P(Z₂=0)=1-P(Z₂=1)={exact_p2[0]:.3f}")
    print(f"    如果 Z₁⟂Z₂ (独立): P(Z₁=0,Z₂=0) = {exact_p1[0]*exact_p2[0]:.4f}")
    print(f"    但真实值: P(Z₁=0,Z₂=0) = {exact_posterior[0,0]:.4f}")
    if abs(exact_posterior[0,0] - exact_p1[0]*exact_p2[0]) > 1e-4:
        print(f"    差值 = {abs(exact_posterior[0,0] - exact_p1[0]*exact_p2[0]):.4f} → Z₁和Z₂不独立!")
        print(f"    Mean-Field 强制 Q(Z₁,Z₂)=Q₁·Q₂ → 无法表达这种相关性 → 近似误差不可消除")
    else:
        print(f"    Z₁和Z₂恰好独立 → Mean-Field = 精确 (这种情况很罕见!)")


# ============================================================================
# 练习 4: Mean-Field VI for Bayesian Network
# ============================================================================

def exercise4_vi_for_bayesian_network():
    """
    对一个简单的 Diamond DAG 实现 Mean-Field VI,
    与 VE 的精确结果对比, 观察近似的质量。

    图结构: A → B, A → C, B → D, C → D
    所有变量仅有 1 个父节点 (D 除外), 简化索引处理。
    """
    print("=" * 70)
    print("练习 4: Mean-Field VI for Bayesian Network — 与精确解对比")
    print("=" * 70)

    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination

    # 钻石 DAG: A→B, A→C, B→D, C→D
    # Moral graph 中 B-C 因共同子节点 D 形成填充边 → 4-循环!
    model = DiscreteBayesianNetwork([
        ('A', 'B'), ('A', 'C'),
        ('B', 'D'), ('C', 'D'),
    ])

    np.random.seed(42)
    for node in ['A', 'B', 'C', 'D']:
        card = 2
        parents = list(model.get_parents(node))
        if not parents:
            cpd = TabularCPD(node, card, [[0.6], [0.4]])
        elif len(parents) == 1:
            cpd = TabularCPD(node, card,
                           [[0.8, 0.3], [0.2, 0.7]],
                           evidence=parents, evidence_card=[card])
        else:
            # D has 2 parents: B, C
            cpd = TabularCPD(node, card,
                           [[0.95, 0.4, 0.3, 0.05],
                            [0.05, 0.6, 0.7, 0.95]],
                           evidence=parents, evidence_card=[card, card])
        model.add_cpds(cpd)

    model.check_model()
    ve = VariableElimination(model)

    print(f"\n  ── 图结构: 钻石 DAG (A→B,C; B,C→D) ──")
    print(f"  Moral graph 加边 B-C → 4-循环 → treewidth=2 > 1")

    # 精确边际
    print(f"\n  ── 精确 VE 边际 (无证据) ──")
    exact_marginals = {}
    for node in ['A', 'B', 'C', 'D']:
        r = ve.query([node], show_progress=False)
        exact_marginals[node] = r.values
        print(f"    P({node}) = {r.values}")

    # --- Mean-Field VI ---
    # Q(A,B,C,D) = Q_a(A)·Q_b(B)·Q_c(C)·Q_d(D)
    # 手动提取参数 (只适用于这个特定结构)
    print(f"\n  ── Mean-Field VI (坐标上升) ──")

    # 从 CPD 中提取参数
    p_a = np.array([0.6, 0.4])
    p_b_given_a = np.array([[0.8, 0.3], [0.2, 0.7]])   # 行=B, 列=A
    p_c_given_a = np.array([[0.8, 0.3], [0.2, 0.7]])   # 行=C, 列=A
    p_d_given_bc = np.array([[[0.95, 0.4], [0.3, 0.05]],  # D=0: [B,C]
                              [[0.05, 0.6], [0.7, 0.95]]]) # D=1: [B,C]

    # 初始化 Q
    q = {v: np.ones(2)/2 for v in ['A', 'B', 'C', 'D']}

    n_iters = 30
    for it in range(n_iters):
        # 更新 Q(A): log Q(A) = log P(A) + E_Q(B)[log P(B|A)] + E_Q(C)[log P(C|A)] + const
        log_qa = np.zeros(2)
        for a in range(2):
            log_qa[a] = (np.log(p_a[a]) +
                         q['B'][0]*np.log(p_b_given_a[0,a]) + q['B'][1]*np.log(p_b_given_a[1,a]) +
                         q['C'][0]*np.log(p_c_given_a[0,a]) + q['C'][1]*np.log(p_c_given_a[1,a]))
        log_qa -= logsumexp(log_qa)
        q['A'] = np.exp(log_qa)

        # 更新 Q(B): log Q(B) = E_Q(A)[log P(B|A)] + E_Q(C,D)[log P(D|B,C)] + const
        log_qb = np.zeros(2)
        for b in range(2):
            log_qb[b] = (q['A'][0]*np.log(p_b_given_a[b,0]) + q['A'][1]*np.log(p_b_given_a[b,1]))
            # 加上来自子节点 D 的项: E_Q(C,D)[log P(D|B=b,C)]
            for c in range(2):
                for d in range(2):
                    log_qb[b] += q['C'][c] * q['D'][d] * np.log(max(p_d_given_bc[d,b,c], 1e-12))
        log_qb -= logsumexp(log_qb)
        q['B'] = np.exp(log_qb)

        # 更新 Q(C): 对称于 Q(B)
        log_qc = np.zeros(2)
        for c in range(2):
            log_qc[c] = (q['A'][0]*np.log(p_c_given_a[c,0]) + q['A'][1]*np.log(p_c_given_a[c,1]))
            for b in range(2):
                for d in range(2):
                    log_qc[c] += q['B'][b] * q['D'][d] * np.log(max(p_d_given_bc[d,b,c], 1e-12))
        log_qc -= logsumexp(log_qc)
        q['C'] = np.exp(log_qc)

        # 更新 Q(D): log Q(D) = E_Q(B,C)[log P(D|B,C)] + const
        log_qd = np.zeros(2)
        for d in range(2):
            for b in range(2):
                for c in range(2):
                    log_qd[d] += q['B'][b] * q['C'][c] * np.log(max(p_d_given_bc[d,b,c], 1e-12))
        log_qd -= logsumexp(log_qd)
        q['D'] = np.exp(log_qd)

        if it < 5 or it == n_iters - 1:
            print(f"  iter {it+1:2d}: Q(A)={np.round(q['A'], 3)}, "
                  f"Q(B)={np.round(q['B'], 3)}, "
                  f"Q(C)={np.round(q['C'], 3)}, "
                  f"Q(D)={np.round(q['D'], 3)}")

    # 对比
    print(f"\n  ── VI 近似 vs 精确 VE ──")
    all_ok = True
    for node in ['A', 'B', 'C', 'D']:
        vi_val = q[node]
        ex_val = exact_marginals[node]
        vi_argmax = np.argmax(vi_val)
        ex_argmax = np.argmax(ex_val)
        ok = vi_argmax == ex_argmax
        if not ok:
            all_ok = False
        print(f"    P({node}): VI={np.round(vi_val,4)}, Exact={ex_val}, "
              f"argmax {'✅' if ok else '⚠️'}")

    if all_ok:
        print(f"\n  ✅ 所有变量的 argmax 与精确解一致 — VI 抓住了正确的边际趋势")
    else:
        print(f"\n  ⚠️ 部分变量 argmax 不一致 — Mean-Field 近似的局限性")

    # 展示 KL 近似误差
    print(f"\n  ── 近似误差来源 ──")
    print(f"    Mean-Field 假设 Q=∏Q_i → B 和 C 在 Q 下独立")
    print(f"    但真实后验中, B ⟂̸ C | ... (它们通过 D 和 A 相关)")
    print(f"    → VI 低估了不确定性, 可能过度自信")

    print("\n  🎯 洞察:")
    print("    Mean-Field VI 用独立 Q 近似相关后验")
    print("    边际趋势通常正确 (argmax 一致), 但低估了不确定性")
    print("    对于有环图, VI 复杂度 << 精确 VE 的指数复杂度")


# ============================================================================
# 练习 5: VI 定点方程 = BP 消息 (树上的等价性)
# ============================================================================

def exercise5_vi_equals_bp_on_tree():
    """
    在树上展示: Mean-Field VI 的定点方程 和 BP 的更新公式 之间的对应关系。

    关键: 对于树结构, 如果把 BP 消息"重写"为节点边际,
    VI 的更新公式恰好给出和 BP 一样的结果。
    """
    print("=" * 70)
    print("练习 5: VI 定点方程 = BP 消息 (树上的等价性)")
    print("=" * 70)

    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import BeliefPropagation, VariableElimination

    # 构建一个简单的树: A → B → C → D (链)
    model = DiscreteBayesianNetwork([
        ('A', 'B'), ('B', 'C'), ('C', 'D'),
    ])

    np.random.seed(123)
    for node in ['A', 'B', 'C', 'D']:
        parents = model.get_parents(node)
        card = 2
        if not parents:
            cpd = TabularCPD(node, card, [[0.5], [0.5]])
        else:
            cpd = TabularCPD(node, card,
                           [[0.8, 0.3], [0.2, 0.7]],
                           evidence=parents, evidence_card=[card])
        model.add_cpds(cpd)

    model.check_model()
    bp = BeliefPropagation(model)
    bp.calibrate()
    ve = VariableElimination(model)

    # BP 边际 (精确, 因为链是树)
    print(f"\n  ── BP 精确边际 (树 = 精确) ──")
    bp_marginals = {}
    for node in ['A', 'B', 'C', 'D']:
        bp_r = bp.query([node])
        bp_marginals[node] = bp_r.values
        print(f"    P({node}) = {bp_r.values}")

    # --- 手动实现 Mean-Field VI (在树上) ---
    # Q(A,B,C,D) = Q_a(A)·Q_b(B)·Q_c(C)·Q_d(D)
    # 对每个节点, log Q_i ∝ E_{Q_{-i}}[log P]
    #
    # 链的 log P: log P(A) + log P(B|A) + log P(C|B) + log P(D|C)

    # 参数
    p_a = np.array([0.5, 0.5])
    p_b_given_a = np.array([[0.8, 0.3], [0.2, 0.7]])  # 行=B, 列=A
    p_c_given_b = np.array([[0.8, 0.3], [0.2, 0.7]])  # 行=C, 列=B
    p_d_given_c = np.array([[0.8, 0.3], [0.2, 0.7]])  # 行=D, 列=C

    # 初始化 Q
    q = {var: np.ones(2) / 2 for var in ['A', 'B', 'C', 'D']}

    n_iters = 30
    for it in range(n_iters):
        # 更新 Q(A)
        log_qa = np.zeros(2)
        for a in range(2):
            log_qa[a] = np.log(p_a[a]) + np.dot(q['B'], np.log(p_b_given_a[:, a]))
        log_qa -= logsumexp(log_qa)
        q['A'] = np.exp(log_qa)

        # 更新 Q(B)
        log_qb = np.zeros(2)
        for b in range(2):
            log_qb[b] = (np.dot(q['A'], np.log(p_b_given_a[b, :])) +
                         np.dot(q['C'], np.log(p_c_given_b[:, b])))
        log_qb -= logsumexp(log_qb)
        q['B'] = np.exp(log_qb)

        # 更新 Q(C)
        log_qc = np.zeros(2)
        for c in range(2):
            log_qc[c] = (np.dot(q['B'], np.log(p_c_given_b[c, :])) +
                         np.dot(q['D'], np.log(p_d_given_c[:, c])))
        log_qc -= logsumexp(log_qc)
        q['C'] = np.exp(log_qc)

        # 更新 Q(D)
        log_qd = np.zeros(2)
        for d in range(2):
            log_qd[d] = np.dot(q['C'], np.log(p_d_given_c[d, :]))
        log_qd -= logsumexp(log_qd)
        q['D'] = np.exp(log_qd)

    print(f"\n  ── Mean-Field VI 结果 (30 轮迭代) ──")
    all_match = True
    for node in ['A', 'B', 'C', 'D']:
        vi_val = q[node]
        bp_val = bp_marginals[node]
        match = np.allclose(vi_val, bp_val, atol=1e-4)
        if not match:
            all_match = False
        print(f"    Q({node}) = {np.round(vi_val, 4)}, BP = {bp_val}, "
              f"{'✅' if match else '⚠️'}")

    if all_match:
        print(f"\n  ✅ 罕见! 此模型上 Mean-Field VI = BP")
    else:
        print(f"\n  ⚠️ Mean-Field VI != BP — 这是正常的!")
        print(f"    Mean-Field 强制 Q=∏Q_i → 丢失了链上相邻节点的相关性")
        print(f"    BP 在树上精确 → 它内部用的是 Bethe 近似 (保留成对相关)")
        print(f"    只有 Bethe 变分族在树上才等于精确 BP")

    # --- 解释对应关系 ---
    print(f"\n  ── Mean-Field VI vs BP: 为什么不同? ──")
    print("""
    Mean-Field 假设: Q(A,B,C,D) = Q_a(A) · Q_b(B) · Q_c(C) · Q_d(D)
    → 强制所有变量独立 → 丢失了 A-B, B-C, C-D 之间的相关性
    → 即使图是树, Mean-Field VI 也只是 BP 的近似!

    BP (Belief Propagation): 在树上精确!
    → BP 内部使用的变分族是 Bethe 近似:
      Q_Bethe(A,B,C,D) ∝ ∏_i P(Z_i|Z_{pa(i)}) / ∏_i P(Z_i)^{d_i-1}
    → Bethe 族保留了成对 (pairwise) 相关性
    → 树上 Bethe VI = BP = 精确!

    所以:
      Mean-Field VI ≈ 粗糙近似, 丢失所有相关性, 但 O(N·K) 极快
      Bethe VI      = Loopy BP (有环图) / BP (树), 保留成对相关

    VI 的实际威力: 当 treewidth 太大导致精确 BP/VE 不可行时,
                    Mean-Field VI 仍然可运行 — O(N×K) per iteration!
    """)

    print("\n  🎯 洞察:")
    print("    Mean-Field VI: Q=∏Q_i → 粗糙但 O(N·K), 任何图都可用")
    print("    Bethe VI: 保留成对相关 → Loopy BP 是其特例")
    print("    核心权衡: 变分族越丰富 → 近似越好, 但计算越贵")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    run_all = '--ex' not in sys.argv

    exercises = [
        ('1', exercise1_kl_divergence, False),
        ('2', exercise2_elbo_decomposition, False),
        ('3', exercise3_mean_field_cavi, False),
        ('4', exercise4_vi_for_bayesian_network, True),
        ('5', exercise5_vi_equals_bp_on_tree, True),
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
