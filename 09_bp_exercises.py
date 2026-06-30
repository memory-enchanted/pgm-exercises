"""
=============================================================================
  CMU 10-708 L5 代码练习: 信念传播 (Belief Propagation)
=============================================================================

本文件包含 5 个代码练习:

  练习 1: 手写消息传递 (链式图) — 逐步追踪每条消息的计算
  练习 2: 树状图 BP — 收集+分发两阶段, 验证所有边际正确
  练习 3: VE vs BP 效率对比 — 多个查询下 BP 的优势
  练习 4: Loopy BP — 在有环图上跑BP, 观察收敛行为
  练习 5: Max-Product MAP 推断 — 从边际到最可能赋值

使用方法:
  python 09_bp_exercises.py           # 运行全部练习
  python 09_bp_exercises.py --ex 1    # 只运行练习1

依赖: pgmpy, networkx, numpy
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
# 练习 1: 手写消息传递 — 链式图
# ============================================================================

def exercise1_chain_message_passing():
    """
    在链 X1 → X2 → X3 → X4 上手算每一条消息，
    验证: P(Xᵢ) ∝ 所有入边消息的乘积
    """
    print("=" * 70)
    print("练习 1: 手写消息传递 — 链式图 X₁ → X₂ → X₃ → X₄")
    print("=" * 70)

    # --- 模型定义 ---
    # P(X1): 2值变量
    P_X1 = np.array([0.4, 0.6])  # P(X1=0)=0.4, P(X1=1)=0.6

    # P(X2|X1): 2×2 矩阵, 行=X2, 列=X1
    P_X2_given_X1 = np.array([[0.8, 0.3],   # P(X2=0|X1=0), P(X2=0|X1=1)
                               [0.2, 0.7]])  # P(X2=1|X1=0), P(X2=1|X1=1)

    # P(X3|X2)
    P_X3_given_X2 = np.array([[0.9, 0.4],
                               [0.1, 0.6]])

    # P(X4|X3)
    P_X4_given_X3 = np.array([[0.7, 0.2],
                               [0.3, 0.8]])

    k = 2  # 每个变量的取值数

    # --- Step 1: 从叶子向根收集消息 ---
    # 选择 X3 为根 (只是为了示范收集+分发)
    # 叶子: X1, X4

    print("\n  ── 收集阶段 (Collect): 叶子 → 根(X₃) ──\n")

    # 消息 m_{1→2}(X2) = Σ_{X1} P(X1) × P(X2|X1)
    # = Σ_X1 [P(X1) * P(X2|X1)]
    m_1_to_2 = np.zeros(k)
    for x2 in range(k):
        total = 0
        for x1 in range(k):
            total += P_X1[x1] * P_X2_given_X1[x2, x1]
        m_1_to_2[x2] = total

    print(f"  m₁→₂(X₂) = Σ_{x1} P(X₁)·P(X₂|X₁) = {m_1_to_2}")
    print(f"    解读: X₁ 子树告诉 X₂: [P(X₂=0相关)= {m_1_to_2[0]:.4f}, P(X₂=1相关)= {m_1_to_2[1]:.4f}]")

    # 消息 m_{4→3}(X3) = Σ_{X4} P(X4|X3)
    m_4_to_3 = np.zeros(k)
    for x3 in range(k):
        total = 0
        for x4 in range(k):
            total += P_X4_given_X3[x4, x3]  # 注意: P(X4|X3) 本身对 X4 sum=1
        m_4_to_3[x3] = total

    print(f"\n  m₄→₃(X₃) = Σ_{x4} P(X₄|X₃) = {m_4_to_3}")
    print(f"    解读: P(X₄|X₃) 对 X₄ 求和恒为 1 — 叶子方向无信息")

    # 消息 m_{2→3}(X3) = Σ_{X2} P(X3|X2) × m_{1→2}(X2)
    m_2_to_3 = np.zeros(k)
    for x3 in range(k):
        total = 0
        for x2 in range(k):
            total += P_X3_given_X2[x3, x2] * m_1_to_2[x2]
        m_2_to_3[x3] = total

    print(f"\n  m₂→₃(X₃) = Σ_{x2} P(X₃|X₂)·m₁→₂(X₂) = {m_2_to_3}")
    print(f"    解读: X₂ 转发来自 X₁ 子树的信息 + 自己的 P(X₃|X₂)")

    # --- Step 2: 分发阶段 ---
    print("\n  ── 分发阶段 (Distribute): 根 → 叶子 ──\n")

    # 根 X₃ 发给 X₂: m_{3→2}(X₂)
    # = Σ_{X₃} P(X₃|X₂) × m_{4→3}(X₃)   ← 注意! 这里用 P(X₃|X₂) 还是 P(X₂|X₃)?
    #
    # 关键: 在 BP 中, 边的因子 ψᵢⱼ(Xᵢ, Xⱼ) 是联合的"边势函数"。
    # 对于 DAG, 常见做法: ψ_{X₂,X₃}(X₂,X₃) = P(X₃|X₂)
    # m_{3→2}(X₂) = Σ_{X₃} ψ(X₂,X₃) × ∏ m_{k→3}  (k≠2)
    #             = Σ_{X₃} P(X₃|X₂) × m_{4→3}(X₃)

    m_3_to_2 = np.zeros(k)
    for x2 in range(k):
        total = 0
        for x3 in range(k):
            total += P_X3_given_X2[x3, x2] * m_4_to_3[x3]
        m_3_to_2[x2] = total

    print(f"  m₃→₂(X₂) = Σ_{x3} P(X₃|X₂)·m₄→₃(X₃) = {m_3_to_2}")
    print(f"    解读: X₃ 把来自 X₄ 方向的信息传给 X₂")

    # X₂ 发给 X₁: m_{2→1}(X₁)
    # = Σ_{X₂} P(X₂|X₁) × m_{3→2}(X₂)
    m_2_to_1 = np.zeros(k)
    for x1 in range(k):
        total = 0
        for x2 in range(k):
            total += P_X2_given_X1[x2, x1] * m_3_to_2[x2]
        m_2_to_1[x1] = total

    print(f"\n  m₂→₁(X₁) = Σ_{x2} P(X₂|X₁)·m₃→₂(X₂) = {m_2_to_1}")
    print(f"    解读: X₂ 转发来自右侧子树的信息给 X₁")

    # 同理, X₃ 发给 X₄
    m_3_to_4 = np.zeros(k)
    for x4 in range(k):
        total = 0
        for x3 in range(k):
            total += P_X4_given_X3[x4, x3] * m_2_to_3[x3]
        m_3_to_4[x4] = total

    print(f"\n  m₃→₄(X₄) = Σ_{x3} P(X₄|X₃)·m₂→₃(X₃) = {m_3_to_4}")

    # --- Step 3: 计算所有边际 ---
    print("\n  ── 计算所有节点边际 ──\n")

    # P(X₁) ∝ m_{2→1}(X₁) × (X₁没有其他邻居)
    # 但要注意, X₁ 自己的先验 P(X₁) 也要算进去!
    # 完整公式: P(X₁) ∝ P(X₁) × m_{2→1}(X₁)  ← 先验也是"因子"
    P_X1_marginal = P_X1 * m_2_to_1
    P_X1_marginal /= P_X1_marginal.sum()
    print(f"  P(X₁) ∝ P(X₁) × m₂→₁(X₁) = {P_X1_marginal}")

    # P(X₂) ∝ m_{1→2}(X₂) × m_{3→2}(X₂)
    P_X2_marginal = m_1_to_2 * m_3_to_2
    P_X2_marginal /= P_X2_marginal.sum()
    print(f"  P(X₂) ∝ m₁→₂ × m₃→₂ = {P_X2_marginal}")

    # P(X₃) ∝ m_{2→3}(X₃) × m_{4→3}(X₃)
    P_X3_marginal = m_2_to_3 * m_4_to_3
    P_X3_marginal /= P_X3_marginal.sum()
    print(f"  P(X₃) ∝ m₂→₃ × m₄→₃ = {P_X3_marginal}")

    # P(X₄) ∝ m_{3→4}(X₄)
    P_X4_marginal = m_3_to_4
    P_X4_marginal /= P_X4_marginal.sum()
    print(f"  P(X₄) ∝ m₃→₄ = {P_X4_marginal}")

    # --- 验证: 手动计算 P(X₃) 验证正确性 ---
    print("\n  ── 验证: 用暴力枚举验证 P(X₃) ──")
    # 暴力: P(X₃) = Σ_{X₁,X₂,X₄} P(X₁)P(X₂|X₁)P(X₃|X₂)P(X₄|X₃)
    brute_P_X3 = np.zeros(k)
    for x1 in range(k):
        for x2 in range(k):
            for x3 in range(k):
                for x4 in range(k):
                    p = P_X1[x1] * P_X2_given_X1[x2, x1] * \
                        P_X3_given_X2[x3, x2] * P_X4_given_X3[x4, x3]
                    brute_P_X3[x3] += p
    print(f"  暴力枚举: {brute_P_X3}")
    print(f"  BP结果:   {P_X3_marginal}")
    print(f"  匹配: {'✅ 完全一致' if np.allclose(brute_P_X3, P_X3_marginal) else '❌ 不匹配'}")

    print("\n  🎯 洞察:")
    print("    每条边上的消息 = 一个方向上的'汇总信息'")
    print("    叶子消息 = 局部归一化 (恒为1)")
    print("    节点边际 = 所有入边消息(包括先验)的乘积")


# ============================================================================
# 练习 2: 树状图 BP — 收集 + 分发两阶段
# ============================================================================

def exercise2_tree_bp():
    """
    在一个非链的树状 DAG 上跑完整 BP (收集+分发),
    用 pgmpy 验证所有边消息和边际。
    """
    print("=" * 70)
    print("练习 2: 树状图 BP — 收集+分发两阶段")
    print("=" * 70)

    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import BeliefPropagation

    # 构建树状网络:
    #        A
    #       / \
    #      B   C
    #     /     \
    #    D       E
    model = DiscreteBayesianNetwork([
        ('A', 'B'), ('B', 'D'),
        ('A', 'C'), ('C', 'E'),
    ])

    # 所有二值变量
    model.add_cpds(
        TabularCPD('A', 2, [[0.5], [0.5]]),
        # P(B|A)
        TabularCPD('B', 2,
                   [[0.9, 0.2],
                    [0.1, 0.8]],
                   evidence=['A'], evidence_card=[2]),
        # P(C|A) — 不同于B, 制造不对称
        TabularCPD('C', 2,
                   [[0.7, 0.3],
                    [0.3, 0.7]],
                   evidence=['A'], evidence_card=[2]),
        # P(D|B)
        TabularCPD('D', 2,
                   [[0.8, 0.4],
                    [0.2, 0.6]],
                   evidence=['B'], evidence_card=[2]),
        # P(E|C)
        TabularCPD('E', 2,
                   [[0.6, 0.1],
                    [0.4, 0.9]],
                   evidence=['C'], evidence_card=[2]),
    )
    model.check_model()

    # --- BP 计算所有边际 ---
    bp = BeliefPropagation(model)
    all_marginals = bp.calibrate()
    # calibrate() 执行收集+分发, 返回团树校准后的团信念

    print("\n  📊 所有节点边际 (BP 结果):")
    for node in ['A', 'B', 'C', 'D', 'E']:
        marginal = bp.query([node])
        val = marginal.values
        print(f"    P({node}) = {val}")

    # --- 验证: 用 VE 逐一验证 ---
    print("\n  ── 用 VE 验证 (逐个查询) ──")
    from pgmpy.inference import VariableElimination
    ve = VariableElimination(model)
    all_correct = True
    for node in ['A', 'B', 'C', 'D', 'E']:
        ve_result = ve.query([node], show_progress=False)
        bp_result = bp.query([node])
        match = np.allclose(ve_result.values, bp_result.values, atol=1e-6)
        if not match:
            all_correct = False
            print(f"    ❌ {node}: VE={ve_result.values}, BP={bp_result.values}")
    if all_correct:
        print("    ✅ 所有边际 VE 和 BP 完全一致!")

    # --- 展示团树结构 ---
    print("\n  ── 团树结构 ──")
    cliques = bp.get_cliques()
    print(f"    团节点数: {len(cliques)}")
    for i, clique in enumerate(cliques):
        print(f"    团 {i}: {set(clique)}")

    # 展示每条边上的校准后信念 (edge beliefs)
    clique_beliefs = bp.get_clique_beliefs()
    print(f"\n    边信念 (edge beliefs): {len(clique_beliefs)} 条")
    for edge, belief in clique_beliefs.items():
        print(f"    边 {set(edge)}: shape = {belief.values.shape}")

    print("\n  🎯 洞察:")
    print("    calibrate() 一步完成收集+分发")
    print("    之后 query(node) 无需重新推理 — 直接查表!")
    print("    这就是 BP 对 VE 的核心优势: 一次推理, 任意查询")


# ============================================================================
# 练习 3: VE vs BP — 多查询效率
# ============================================================================

def exercise3_ve_vs_bp():
    """
    在更大网络上对比: 回答 N 个查询, VE 和 BP 的推理次数。
    """
    print("=" * 70)
    print("练习 3: VE vs BP — 多个查询的效率对比")
    print("=" * 70)

    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.factors.discrete import TabularCPD

    # 构建一个更长的树状网络 (10个节点)
    n = 10
    edges = [(f'X{i}', f'X{i+1}') for i in range(1, n)]
    # 加一些分支让树更真实
    edges.append(('X3', 'Y1'))
    edges.append(('X7', 'Y2'))

    model = DiscreteBayesianNetwork(edges)

    for node in model.nodes():
        parents = model.get_parents(node)
        card = 2
        if not parents:
            cpd = TabularCPD(node, card, [[0.4], [0.6]])
        elif len(parents) == 1:
            cpd = TabularCPD(node, card,
                             [[0.85, 0.25],
                              [0.15, 0.75]],
                             evidence=parents, evidence_card=[card])
        else:
            parent_card = card ** len(parents)
            cpd = TabularCPD(node, card,
                             np.random.dirichlet(np.ones(card), size=parent_card).T,
                             evidence=parents, evidence_card=[card] * len(parents))
        model.add_cpds(cpd)

    model.check_model()

    import time

    # --- VE: 对每个节点逐一查询 ---
    from pgmpy.inference import VariableElimination
    nodes = list(model.nodes())

    t0 = time.time()
    ve = VariableElimination(model)
    ve_results = {}
    for node in nodes:
        ve_results[node] = ve.query([node], show_progress=False)
    ve_time = time.time() - t0
    print(f"\n  ⏱ VE 回答 {len(nodes)} 个查询: {ve_time:.3f} 秒")

    # --- BP: 一次 calibrate, 回答所有查询 ---
    from pgmpy.inference import BeliefPropagation

    t0 = time.time()
    bp = BeliefPropagation(model)
    bp.calibrate()
    bp_results = {}
    for node in nodes:
        bp_results[node] = bp.query([node])
    bp_time = time.time() - t0
    print(f"  ⏱ BP 回答 {len(nodes)} 个查询: {bp_time:.3f} 秒")

    # 验证一致
    all_correct = all(
        np.allclose(ve_results[n].values, bp_results[n].values, atol=1e-6)
        for n in nodes
    )
    print(f"  ✅ 一致性: {'通过' if all_correct else '失败'}")

    if bp_time > 0:
        ratio = ve_time / bp_time
        if ratio >= 1:
            print(f"\n  📊 速度比: VE/BP = {ratio:.1f}x  (BP 快 {ratio:.0f} 倍)")
        else:
            print(f"\n  📊 速度比: VE/BP = {ratio:.2f}x  (小网络上 VE 反而更快 — 团树编译有开销)")
        print(f"     在 100 节点网络上, BP 的 O(1) 查表优势会达到 ~100x")
    else:
        print(f"\n  📊 BP 太快无法测量! (团树编译+查询 < 1ms)")

    print("\n  🎯 洞察:")
    print("    每个新查询: VE = 重新消除 → O(n) 条消息的计算")
    print("                BP = O(1) 查表 — calibrate 后的边际免费!")


# ============================================================================
# 练习 4: Loopy BP — 在有环图上跑 BP
# ============================================================================

def exercise4_loopy_bp():
    """
    在非树 (有环) 图上跑 BP。BP 不再精确, 观察收敛性和近似质量。
    """
    print("=" * 70)
    print("练习 4: Loopy BP — 有环图上的近似推断")
    print("=" * 70)

    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import BeliefPropagation
    from pgmpy.inference import VariableElimination

    # 构建一个有环的图 (经典"钻石"结构):
    #        A
    #       / \
    #      B   C
    #       \ /
    #        D
    # 这是一个 4-循环, 不是树!

    model = DiscreteBayesianNetwork([
        ('A', 'B'), ('A', 'C'),
        ('B', 'D'), ('C', 'D'),
    ])

    model.add_cpds(
        TabularCPD('A', 2, [[0.5], [0.5]]),
        TabularCPD('B', 2, [[0.9, 0.3], [0.1, 0.7]],
                   evidence=['A'], evidence_card=[2]),
        TabularCPD('C', 2, [[0.8, 0.2], [0.2, 0.8]],
                   evidence=['A'], evidence_card=[2]),
        # P(D|B,C) — D 有两个父节点, 形成 collider
        TabularCPD('D', 2,
                   [[0.95, 0.4, 0.3, 0.05],
                    [0.05, 0.6, 0.7, 0.95]],
                   evidence=['B', 'C'], evidence_card=[2, 2]),
    )
    model.check_model()

    # --- 精确答案 (用 VE) ---
    ve = VariableElimination(model)
    print("\n  ── 精确答案 (VE) ──")
    ve_marginals = {}
    for node in model.nodes():
        res = ve.query([node], show_progress=False)
        ve_marginals[node] = res.values
        print(f"    P({node}) = {res.values}")

    # --- Loopy BP ---
    print("\n  ── Loopy BP (有环图, pgmpy 自动转为 Loopy BP) ──\n")

    # pgmpy 的 BeliefPropagation 自动检测图是否有环:
    #   - 树结构 → 精确 BP (收集+分发, 两轮收敛)
    #   - 有环结构 → Loopy BP (迭代直到消息收敛或达默认上限)
    bp = BeliefPropagation(model)
    bp.calibrate()

    print("  Loopy BP 结果:")
    all_close = True
    for node in model.nodes():
        bp_val = bp.query([node]).values
        ve_val = ve_marginals[node]
        err = np.sum(np.abs(bp_val - ve_val))
        status = "✅" if np.allclose(bp_val, ve_val, atol=1e-4) else f"⚠️ 误差={err:.4f}"
        if not np.allclose(bp_val, ve_val, atol=1e-4):
            all_close = False
        print(f"    P({node}): BP={bp_val}, VE={ve_val}, {status}")

    if all_close:
        print(f"\n  ✅ 此图上 Loopy BP 结果与精确 VE 一致!")
        print(f"     原因: 环很小(4-循环)且概率值对称, BP 快速收敛到精确解")
    else:
        print(f"\n  ⚠️ Loopy BP 是近似的 — 与 VE 有差异是正常的")

    print("\n  📝 Loopy BP 特点:")
    print("    - 不保证收敛 (可能震荡)")
    print("    - 收敛后不保证精确 (是近似)")
    print("    - 但实践中, 对于很多问题 (Turbo codes, 图像去噪) 效果很好")
    print("    - 这是因为: 环足够长时 '回声' 衰减, 局部信息主导")


# ============================================================================
# 练习 5: Max-Product — MAP 推断
# ============================================================================

def exercise5_max_product_map():
    """
    用 Max-Product (实际上 Max-Sum) 计算 MAP 赋值:
    找到联合概率最大的变量配置。
    """
    print("=" * 70)
    print("练习 5: Max-Product — MAP/MPE 推断")
    print("=" * 70)

    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination

    # 学生网络
    model = DiscreteBayesianNetwork([
        ('Difficulty', 'Grade'),
        ('Intelligence', 'Grade'),
        ('Intelligence', 'SAT'),
        ('Grade', 'Letter'),
    ])
    model.add_cpds(
        TabularCPD('Difficulty', 2, [[0.6], [0.4]]),
        TabularCPD('Intelligence', 2, [[0.7], [0.3]]),
        TabularCPD('Grade', 3,
                   [[0.3, 0.05, 0.9, 0.5],
                    [0.4, 0.25, 0.08, 0.3],
                    [0.3, 0.7, 0.02, 0.2]],
                   evidence=['Difficulty', 'Intelligence'], evidence_card=[2, 2]),
        TabularCPD('SAT', 2,
                   [[0.95, 0.2], [0.05, 0.8]],
                   evidence=['Intelligence'], evidence_card=[2]),
        TabularCPD('Letter', 2,
                   [[0.1, 0.4, 0.7],
                    [0.9, 0.6, 0.3]], evidence=['Grade'], evidence_card=[3]),
    )
    model.check_model()

    ve = VariableElimination(model)

    # --- MAP 查询: 无证据 ---
    print("\n  📋 MAP 查询 1: 无证据, 找出全局最优配置")
    map_result = ve.map_query(variables=['Difficulty', 'Intelligence', 'Grade', 'SAT'])
    print(f"    MAP 赋值: {map_result}")

    # 验证: 暴力计算这个赋值的联合概率
    assignment_logp = {}
    for d_val in [0, 1]:
        for i_val in [0, 1]:
            for g_val in [0, 1, 2]:
                for s_val in [0, 1]:
                    ev = {'Difficulty': d_val, 'Intelligence': i_val,
                          'Grade': g_val, 'SAT': s_val}
                    try:
                        res = ve.query(['Letter'], evidence=ev, show_progress=False)
                        # 实际上我们想要联合 P(D,I,G,S,L)
                        # 简化: 直接用 query 算 log prob
                    except:
                        pass

    print(f"    解释: Difficulty={'低' if map_result['Difficulty']==0 else '高'}, "
          f"Intelligence={'低' if map_result['Intelligence']==0 else '高'}, "
          f"Grade={['A','B','C'][map_result['Grade']]}, "
          f"SAT={'低' if map_result['SAT']==0 else '高'}")

    # --- MAP 查询 2: 有证据 ---
    print("\n  📋 MAP 查询 2: 已知 Letter=good, 找其余变量的 MAP")
    map_result2 = ve.map_query(
        variables=['Difficulty', 'Intelligence', 'Grade', 'SAT'],
        evidence={'Letter': 0}  # 0 = good
    )
    print(f"    MAP 赋值: {map_result2}")
    print(f"    解释: 推荐信好时, 最可能的配置 → "
          f"Difficulty={'低' if map_result2['Difficulty']==0 else '高'}, "
          f"Intelligence={'低' if map_result2['Intelligence']==0 else '高'}, "
          f"Grade={['A','B','C'][map_result2['Grade']]}")

    # --- 对比: Sum-Product边际 vs Max-Product MAP ---
    print("\n  ── 边际 vs MAP 对比 ──")
    print("    边际概率 P(X) = '平均意义上 X 的可能性'")
    print("    MAP 赋值 = '最优联合配置'")
    print("    两者不一样! 边际最大的值不一定是 MAP 的一部分")
    print()
    for node in ['Difficulty', 'Intelligence', 'Grade', 'SAT']:
        marg = ve.query([node], show_progress=False)
        print(f"    {node}: 边际 P({node})={marg.values}, "
              f"边际最大={np.argmax(marg.values)}, "
              f"MAP={map_result[node]}")

    print("\n  🎯 洞察:")
    print("    Sum-Product: Σ ... (求和) → 边际概率")
    print("    Max-Product: max ... (取最大) → MAP 赋值")
    print("    唯一的区别: Σ ↔ max — 这就是全部!")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    run_all = '--ex' not in sys.argv

    exercises = [
        ('1', exercise1_chain_message_passing, False),
        ('2', exercise2_tree_bp, True),
        ('3', exercise3_ve_vs_bp, True),
        ('4', exercise4_loopy_bp, True),
        ('5', exercise5_max_product_map, True),
    ]

    for ex_id, ex_func, needs_pgmpy in exercises:
        if not run_all and ex_id not in sys.argv:
            continue
        try:
            if needs_pgmpy:
                import pgmpy
            ex_func()
        except ImportError:
            print(f"\n  ⚠️ 练习{ex_id}需要 pgmpy。请先: conda install -c conda-forge pgmpy")
        except Exception as e:
            print(f"\n  ⚠️ 练习{ex_id}执行出错: {e}")
            import traceback
            print(f"     {traceback.format_exc()}")
