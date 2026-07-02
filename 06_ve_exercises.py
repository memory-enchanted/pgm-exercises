"""
=============================================================================
  CMU 10-708 L4 代码练习: 变量消除算法 (Variable Elimination)
=============================================================================

本文件包含 5 个代码练习, 逐步深入理解 L4 的核心概念:

  练习 1: 因子基础 — 手写因子乘积和边缘化
  练习 2: VE 步步跟踪 — 可视化每一步消除
  练习 3: 消除顺序对决 — 好顺序 vs 坏顺序, 对比计算量
  练习 4: 诱导图 & 填充边 — 理解复杂度根源
  练习 5: VE = 消息传递 — 看到消元结果就是消息

使用方法:
  python 06_ve_exercises.py           # 运行全部练习
  python 06_ve_exercises.py --ex 1    # 只运行练习1

依赖: pgmpy, networkx, numpy, matplotlib
=============================================================================
"""

import numpy as np
import itertools
from collections import defaultdict
import sys

# ============================================================================
# 工具函数
# ============================================================================

def pretty_print_factor(scope, values_dict, name=""):
    """格式化打印一个因子"""
    scope = list(scope)
    print(f"\n  Factor {name}" if name else f"\n  Factor")
    print(f"  Scope: {scope}")
    print(f"  Size: {' × '.join([str(len(set(k[i] for k in values_dict.keys())))
                                  for i in range(len(scope))])}")

    # 打印表头
    header = "  " + "  ".join([f"{v:^8s}" for v in scope]) + "  |  value"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for assignment, val in sorted(values_dict.items()):
        row = "  " + "  ".join([f"{str(a):^8s}" for a in assignment])
        print(f"{row}  |  {val:.4f}")


def cartesian_product(domains):
    """生成变量取值的笛卡尔积"""
    keys = list(domains.keys())
    values = [domains[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


# ============================================================================
# 练习 1: 因子基础 — 手写乘积和边缘化
# ============================================================================

def exercise1_factor_basics():
    """
    用纯 Python 实现因子的两个基本操作: 乘积和边缘化。
    这让你彻底理解 VE 底层到底在做什么。
    """
    print("=" * 70)
    print("练习 1: 因子基础 — 乘积 & 边缘化")
    print("=" * 70)

    # --- 定义两个简单因子 ---
    # φ₁(Grade, Intelligence)
    G_vals = ['A', 'B', 'C']
    I_vals = ['low', 'high']

    phi1 = {}  # P(G|I)
    for g in G_vals:
        for i in I_vals:
            phi1[(g, i)] = np.random.random()  # 简化起见用随机值
    # 归一化: 对每个 I, sum_G = 1
    for i in I_vals:
        s = sum(phi1[(g, i)] for g in G_vals)
        for g in G_vals:
            phi1[(g, i)] /= s

    # φ₂(Intelligence, SAT)
    S_vals = [0, 1]
    phi2 = {}
    for i in I_vals:
        for s in S_vals:
            phi2[(i, s)] = np.random.random()
    for i in I_vals:
        norm = sum(phi2[(i, s)] for s in S_vals)
        for s in S_vals:
            phi2[(i, s)] /= norm

    print("\n  ── 原始因子 ──\n")
    # 打印 φ₁
    print("  φ₁(Grade, Intelligence): P(G|I)")
    header = f"  {'Grade':^8s} {'I':^8s} |  P(G|I)"
    print(header)
    print("  " + "-" * len(header))
    for (g, i), v in sorted(phi1.items()):
        print(f"  {g:^8s} {i:^8s} |  {v:.4f}")

    print("\n  φ₂(Intelligence, SAT): P(S|I)")
    header = f"  {'I':^8s} {'SAT':^8s} |  P(S|I)"
    print(header)
    print("  " + "-" * len(header))
    for (i, s), v in sorted(phi2.items()):
        print(f"  {i:^8s} {s:^8d} |  {v:.4f}")

    # --- 操作 1: 因子乘积 ---
    print("\n  ── 操作 A: φ₁ × φ₂ → φ₁₂(G, I, S) ──\n")

    phi12 = {}
    for (g, i), v1 in phi1.items():
        for s in S_vals:
            key = (g, i, s)
            v2 = phi2[(i, s)]
            phi12[key] = v1 * v2

    print("  φ₁₂(Grade, Intelligence, SAT):")
    header = f"  {'Grade':^8s} {'I':^8s} {'SAT':^8s} |  value"
    print(header)
    print("  " + "-" * len(header))
    for (g, i, s), v in sorted(phi12.items()):
        print(f"  {g:^8s} {i:^8s} {s:^8d} |  {v:.4f}")

    print(f"\n    φ₁ 大小: {len(phi1)}  →  φ₁₂ 大小: {len(phi12)}")
    print("    → 乘积使 scope 扩大 (并集), 表格维度增加")

    # --- 操作 2: 因子边缘化 ---
    print("\n  ── 操作 B: Σ_I φ₁₂(G, I, S) → τ(G, S) ──\n")

    tau = {}
    for g in G_vals:
        for s in S_vals:
            tau[(g, s)] = sum(phi12[(g, i, s)] for i in I_vals)

    print("  τ(Grade, SAT) = Σ_Intelligence φ₁₂:")
    header = f"  {'Grade':^8s} {'SAT':^8s} |  τ(G,S)"
    print(header)
    print("  " + "-" * len(header))
    for (g, s), v in sorted(tau.items()):
        print(f"  {g:^8s} {s:^8d} |  {v:.4f}")

    # 验证: sum over G,S 不一定为1, 但 sum over G 对每个 S 也不一定为1
    print(f"\n    φ₁₂ 大小: {len(phi12)}  →  τ 大小: {len(tau)}")
    print("    → 边缘化使 scope 缩小, 维度降低")

    # --- 洞察 ---
    print("\n  🎯 洞察:")
    print("    VE 的全部计算 = 反复做 '乘积 + 边缘化' 直到只剩下查询变量")
    print("    乘积: 合并信息, 扩大表格")
    print("    边缘化: 消去变量, 缩小表格")
    print("    → 乘积变大, 求和缩小 — 像呼吸一样交替进行\n")


# ============================================================================
# 练习 2: VE 步步跟踪 — 可视化消除过程
# ============================================================================

def exercise2_ve_step_by_step():
    """
    在学生网络上用 pgmpy 做 VE，打印每一步的中间因子，
    让你看到因子如何一步步被消去、合并、缩小。
    """
    print("=" * 70)
    print("练习 2: VE 步步跟踪 — 可视化每一步")
    print("=" * 70)

    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination

    # 构建学生网络
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
                   evidence=['Difficulty', 'Intelligence'],
                   evidence_card=[2, 2]),
        TabularCPD('SAT', 2,
                   [[0.95, 0.2],
                    [0.05, 0.8]],
                   evidence=['Intelligence'], evidence_card=[2]),
        TabularCPD('Letter', 2,
                   [[0.1, 0.4, 0.7],
                    [0.9, 0.6, 0.3]],
                   evidence=['Grade'], evidence_card=[3]),
    )
    model.check_model()
    infer = VariableElimination(model)

    # 问题: P(Intelligence | Letter='good')
    print("\n  📋 查询: P(Intelligence | Letter=good)")
    print("  📋 消除顺序: [Difficulty, SAT, Grade]")

    # 展示每个 CPD
    print("\n  ── 初始因子 (CPD 集合) ──")
    for cpd in model.get_cpds():
        node = cpd.variable
        deps = cpd.variables
        size = np.prod(cpd.cardinality)
        print(f"    P({node}|{', '.join(deps[1:])})" if len(deps) > 1
              else f"    P({node})")
        print(f"    scope: {list(deps)}, 表大小: {size}")

    # 分析每一步的计算量
    print("\n  ── 按消除顺序分析每一步 ──\n")

    # Step 1: 消除 Difficulty
    print("  Step 1: 消除 Difficulty")
    print("    涉及的因子: P(D), P(G|D,I)")
    print("    乘积后 scope: {D, G, I} (大小: 2×3×2 = 12)")
    print("    消去 D 后 scope: {G, I} (大小: 3×2 = 6)")
    print("    计算量: 12 (乘积) + 12 (求和) = 24")

    # Step 2: 消除 SAT
    print("\n  Step 2: 消除 SAT")
    print("    涉及的因子: P(S|I)")
    print("    乘积 scope: {S, I} (大小: 2×2 = 4)")
    print("    消去 S 后 scope: {I} (大小: 2)")
    print("    计算量: 4 + 4 = 8")
    print("    *注意: 只有一个因子时, 乘积=自身, 直接求和")

    # Step 3: 消除 Grade
    print("\n  Step 3: 消除 Grade")
    print("    涉及的因子: τ₁(G,I) (消D的结果) + P(L=good|G)")
    print("    乘积 scope: {G, I} (大小: 3×2 = 6)")
    print("    消去 G 后 scope: {I} (大小: 2)")
    print("    计算量: 6 + 6 = 12")

    print(f"\n    📊 总计算量: 24 + 8 + 12 = 44")
    print(f"    📊 暴力枚举: 2×2×3×2×2 = 48")
    print(f"    📊 节省: {(1-44/48)*100:.0f}% (小网络中不明显, 大网络中是天文数字)")

    # 实际用 pgmpy 计算
    print("\n  ── pgmpy 实际计算结果 ──")
    result = infer.query(['Intelligence'], evidence={'Letter': 0})
    print(result)

    print("\n  🎯 洞察:")
    print("    VE 的本质: 用'动态规划'避免重复计算")
    print("    每次消除 = 把含X的因子全乘起来 → 对X求和 → 结果替代原因子")
    print("    '乘积'这一步决定了中间因子的最大宽度 (复杂度瓶颈)")


# ============================================================================
# 练习 3: 消除顺序对决 — 好坏顺序对比
# ============================================================================

def exercise3_elimination_order_comparison():
    """
    用 pgmpy 对比不同消除顺序的计算代价。
    好顺序 = 小中间因子，坏顺序 = 大中间因子。
    """
    print("=" * 70)
    print("练习 3: 消除顺序对决")
    print("=" * 70)

    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination

    # 构建一个稍复杂的网络 (8个节点的树状网络)
    # 结构: A → C → E → G
    #        ↓   ↑   ↓   ↑
    #        B   D   F   H
    model = DiscreteBayesianNetwork([
        ('A', 'B'), ('A', 'C'),
        ('C', 'D'), ('C', 'E'),
        ('E', 'F'), ('E', 'G'),
        ('G', 'H'),
    ])

    for node, card in [('A', 2), ('B', 2), ('C', 2), ('D', 2),
                        ('E', 2), ('F', 2), ('G', 2), ('H', 2)]:
        parents = model.get_parents(node)
        if not parents:
            cpd = TabularCPD(node, card, [[0.5]] * card)
        else:
            parent_card = np.prod([model.get_cpds(p).variable_card for p in parents])
            # 随机参数
            vals = np.random.dirichlet(np.ones(card), size=int(parent_card)).T
            cpd = TabularCPD(node, card, vals,
                           evidence=parents,
                           evidence_card=[model.get_cpds(p).variable_card for p in parents])
        model.add_cpds(cpd)

    infer = VariableElimination(model)

    # 好顺序: 从叶子开始消 (B → D → F → C → E → G, 不消查询变量H和证据A)
    print("\n  🔵 好顺序 (从叶子向根消除):  B → D → F → C → E → G")
    print("     每次消叶子: 因子 scope 始终 ≤ 2")
    print("     复杂度: O(n · k²) — 线性!")

    # 坏顺序: 从根开始消 (G → E → C → B → D → F, 不消查询变量H和证据A)
    print("\n  🔴 坏顺序 (从根向叶子消除):  G → E → C → B → D → F")
    print("     消G时要把 E→G, G→H 的因子和G的CPD全乘起来!")
    print("     中间因子快速增长, 复杂度接近指数")

    # 验证两个顺序都正确, 但复杂度不同
    try:
        # 消除顺序中不能包含查询变量(H)和证据变量(A)
        r1 = infer.query(['H'], evidence={'A': 0},
                        elimination_order=['B', 'D', 'F', 'C', 'E', 'G'])
        # 注意: 坏顺序可能因为中间因子太大而变慢
        r2 = infer.query(['H'], evidence={'A': 0},
                        elimination_order=['G', 'E', 'C', 'B', 'D', 'F'])
        print(f"\n    ✅ 好顺序结果: {r1.values}")
        print(f"    ✅ 坏顺序结果: {r2.values}")
        print("    → 结果相同, 但中间计算量差距巨大!")
    except Exception as e:
        print(f"    ⚠️ 执行出错: {e}")

    print("\n  🎯 洞察:")
    print("    好顺序的核心原则: 先消'叶子'或'度最小的节点'")
    print("    实际启发式算法: min-degree, min-fill, 贪心搜索")
    print("    最优顺序是 NP-hard (Arnborg et al., 1987)")


# ============================================================================
# 练习 4: 诱导图 & 填充边 — 理解复杂度根源
# ============================================================================

def exercise4_induced_graph():
    """
    手动画出诱导图, 展示不同消除顺序如何产生不同的填充边。

    我们用一个小图:  A — B — C — D (链)
    演示顺序1: A→B→C→D (完美, 0填充边)
    演示顺序2: B→A→C→D (产生填充边 A—C)
    """
    print("=" * 70)
    print("练习 4: 诱导图 & 填充边")
    print("=" * 70)

    import networkx as nx

    G = nx.Graph()
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D')])

    def simulate_elimination(G, order):
        """模拟消除过程, 返回填充边列表和每步的团大小"""
        G = G.copy()
        fill_ins = []
        max_clique_sizes = []

        for node in order:
            neighbors = list(G.neighbors(node))

            # 记录当前节点参与的团大小 (邻居数+1)
            max_clique_sizes.append(len(neighbors) + 1)

            # 填充边: 把所有邻居两两相连
            for i, u in enumerate(neighbors):
                for v in neighbors[i + 1:]:
                    if not G.has_edge(u, v):
                        G.add_edge(u, v)
                        fill_ins.append((u, v))

            # 删除节点
            G.remove_node(node)

        return fill_ins, max_clique_sizes

    # 顺序 1: 从叶子开始
    order1 = ['A', 'B', 'C', 'D']
    fills1, cliques1 = simulate_elimination(G, order1)

    print(f"\n  🔵 顺序 1: 从叶子消起 → {' → '.join(order1)}")
    print(f"     填充边: {fills1 if fills1 else '无! ✨'}")
    print(f"     每步最大团大小: {cliques1}")
    print(f"     树宽度: {max(cliques1) - 1}")

    # 顺序 2: 从中间开始
    order2 = ['B', 'A', 'C', 'D']
    fills2, cliques2 = simulate_elimination(G, order2)

    print(f"\n  🔴 顺序 2: 从中间消起 → {' → '.join(order2)}")
    print(f"     填充边: {fills2}")
    print(f"     每步最大团大小: {cliques2}")
    print(f"     树宽度: {max(cliques2) - 1}")

    print(f"\n  📊 对比:")
    print(f"     顺序1 最大中间因子: k^{max(cliques1)} = k² (小)")
    print(f"     顺序2 最大中间因子: k^{max(cliques2)} = k³ (大)")
    print(f"     在 k=10 时差距: 100 vs 1000, 差10倍!")
    print(f"     在 k=100 时差距: 10,000 vs 1,000,000, 差100倍!")

    print("\n  🎯 洞察:")
    print("    填充边 = 代价。每加一条填充边, 中间因子就多一个维度")
    print("    '度小优先'原则: 每次消除邻居最少的节点, 填充边最少")
    print("    这就是为什么树的 VE 是线性的: 叶子(度=1)被消除时不产生任何填充边")


# ============================================================================
# 练习 5: VE = 消息传递 — 给 L5 搭桥
# ============================================================================

def exercise5_ve_as_message_passing():
    """
    把 VE 的中间结果解释为消息, 建立 L4→L5 的桥梁。

    选一个简单的树状图: X₁ → X₂ → X₃ (马尔可夫链)
    展示 VE 结果 = 沿链传递的消息
    """
    print("=" * 70)
    print("练习 5: VE = 消息传递 (通往 L5 的桥梁)")
    print("=" * 70)

    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination

    # X1 → X2 → X3
    model = DiscreteBayesianNetwork([('X1', 'X2'), ('X2', 'X3')])

    model.add_cpds(
        TabularCPD('X1', 2, [[0.4], [0.6]]),
        TabularCPD('X2', 2,
                   [[0.8, 0.3],   # P(X2=0|X1=0), P(X2=0|X1=1)
                    [0.2, 0.7]],  # P(X2=1|X1=0), P(X2=1|X1=1)
                   evidence=['X1'], evidence_card=[2]),
        TabularCPD('X3', 2,
                   [[0.9, 0.4],   # P(X3=0|X2=0), P(X3=0|X2=1)
                    [0.1, 0.6]],  # P(X3=1|X2=0), P(X3=1|X2=1)
                   evidence=['X2'], evidence_card=[2]),
    )

    infer = VariableElimination(model)

    # 查询 P(X3)
    print("\n  📋 查询: P(X3) — 链末端的边际概率")

    print("\n  ── VE 过程 (消除 X1, 然后消除 X2) ──\n")

    print("  消除 X1:")
    print("    涉及因子: P(X1), P(X2|X1)")
    print("    乘积 + 消去 X1:")
    print("      τ₁(X2=0) = P(X1=0)·P(X2=0|X1=0) + P(X1=1)·P(X2=0|X1=1)")
    print(f"               = 0.4×0.8 + 0.6×0.3 = 0.32 + 0.18 = 0.50")
    print("      τ₁(X2=1) = 0.4×0.2 + 0.6×0.7 = 0.08 + 0.42 = 0.50")
    print()
    print("    这就是消息 m_{1→2}(X2) = τ₁(X2)")
    print("    → 从 X1 到 X2 的消息: [0.50, 0.50]")

    print("\n  消除 X2:")
    print("    涉及因子: τ₁(X2), P(X3|X2)")
    print("    乘积 + 消去 X2:")
    print("      τ₂(X3=0) = τ₁(X2=0)·P(X3=0|X2=0) + τ₁(X2=1)·P(X3=0|X2=1)")
    print(f"               = 0.50×0.9 + 0.50×0.4 = 0.45 + 0.20 = 0.65")
    print("      τ₂(X3=1) = 0.50×0.1 + 0.50×0.6 = 0.05 + 0.30 = 0.35")
    print()
    print("    这就是消息 m_{2→3}(X3) = τ₂(X3)")
    print("    → 从 X2 到 X3 的消息: [0.65, 0.35]")

    # 验证
    result = infer.query(['X3'])
    print(f"\n  ✅ pgmpy 验证: P(X3) = {result.values}")

    print("\n  ── 消息传递视角 ──")
    print("""
       P(X1)           P(X2|X1)          P(X3|X2)
      ┌─────┐         ┌────────┐         ┌────────┐
      │  X1 │──m₁→₂──→│   X2   │──m₂→₃──→│   X3   │
      └─────┘         └────────┘         └────────┘

      m₁→₂(X2) = Σ_X1 P(X1)·P(X2|X1)          ← 消去 X1 的结果
      m₂→₃(X3) = Σ_X2 m₁→₂(X2)·P(X3|X2)       ← 消去 X2 的结果

    核心观察: 每次 VE 消除的结果 = 沿边传递的消息!
    """)

    print("\n  🎯 洞察:")
    print("    L4 VE 的 τ₁, τ₂, ... 就是 Sum-Product BP 中的消息 m_{i→j}")
    print("    L4 视角: 按顺序'消除'变量")
    print("    BP 视角: 在图上'传递'消息")
    print("    同一个算法, 两个视角 — BP 的优势是可以复用消息!")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    run_all = '--ex' not in sys.argv

    if run_all or '1' in sys.argv:
        try:
            exercise1_factor_basics()
        except Exception as e:
            print(f"\n  ⚠️ 练习1执行出错: {e}")

    if run_all or '2' in sys.argv:
        try:
            exercise2_ve_step_by_step()
        except ImportError:
            print("\n  ⚠️ 练习2需要 pgmpy。请先运行: conda install -c conda-forge pgmpy")
        except Exception as e:
            print(f"\n  ⚠️ 练习2执行出错: {e}")

    if run_all or '3' in sys.argv:
        try:
            exercise3_elimination_order_comparison()
        except ImportError:
            print("\n  ⚠️ 练习3需要 pgmpy。请先运行: conda install -c conda-forge pgmpy")
        except Exception as e:
            print(f"\n  ⚠️ 练习3执行出错: {e}")

    if run_all or '4' in sys.argv:
        try:
            exercise4_induced_graph()
        except Exception as e:
            print(f"\n  ⚠️ 练习4执行出错: {e}")

    if run_all or '5' in sys.argv:
        try:
            exercise5_ve_as_message_passing()
        except ImportError:
            print("\n  ⚠️ 练习5需要 pgmpy。请先运行: conda install -c conda-forge pgmpy")
        except Exception as e:
            print(f"\n  ⚠️ 练习5执行出错: {e}")
