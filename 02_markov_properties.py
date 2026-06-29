"""
=============================================================================
  CMU 10-708 L1-L3 核心概念练习 (2): 局部 & 全局马尔可夫性质
=============================================================================

L2-L3 中反复出现的三个马尔可夫性质, 是理解 PGM "表示论" 的核心。
它们层层递进, 描述了图的哪些部分和概率分布的哪些条件独立性有关系。

=============================================================================
一、理论框架: 三种马尔可夫性质的层级关系
=============================================================================

设 G 是一个 DAG, P 是一个在 G 上因子化的概率分布:

  P(X₁,...,Xₙ) = ∏ᵢ P(Xᵢ | Pa(Xᵢ))

                                     ┌─────────────────────┐
                                     │   分解 (Factorization) │
                                     │  P = ∏ P(Xᵢ|Pa(Xᵢ))  │
                                     └──────────┬──────────┘
                                                │
                                    ┌───────────┴───────────┐
                                    ▼                       ▼
                         ┌──────────────────┐    ┌──────────────────────┐
                         │ 局部马尔可夫性质  │    │ 全局马尔可夫性质 (d-分离)│
                         │ (Local Markov)   │    │ (Global Markov)       │
                         │                  │    │                       │
                         │ X ⟂ NonDesc(X)  │    │ d-sep(X,Y|Z) ⇒ X⟂Y|Z │
                         │      | Pa(X)     │    │                       │
                         └────────┬─────────┘    └───────────┬───────────┘
                                  │                          │
                                  └──────────┬───────────────┘
                                             ▼
                                  ┌──────────────────────┐
                                  │  成对马尔可夫性质      │
                                  │  (Pairwise Markov)   │
                                  │                      │
                                  │  非邻接 + 非条件给定   │
                                  │  父节点 ⇒ 条件独立     │
                                  └──────────────────────┘

对于 DAG:  分解 ⟺ 局部马尔可夫 ⟺ 全局马尔可夫 (三者等价!)
          成对马尔可夫最弱, 但三者互相蕴含

对于无向图 (MRF): 三者不再等价, 只在正的分布下全局↔局部↔成对

=============================================================================
二、定义速查
=============================================================================

【局部马尔可夫性质 (Local Markov Property)】
────────────────────────────────────────────
每个节点 X 在给定其父节点 Pa(X) 的条件下, 与其所有非后代节点独立:

    X ⟂ NonDescendants(X) | Pa(X)

"非后代" = 图中所有不是 X 后代的节点 (不包括 X 自己)
"父节点" = 直接指向 X 的节点

直觉: 一旦知道了"直接原因" (父节点), X 就和图中所有其他"非结果"
      节点无关了 — 父节点完全"屏蔽"了外界对 X 的影响。

例子 (学生网络):
    Grade ⟂ {Difficulty, Intelligence, SAT} 的无交集部分?
    实际上 Grade 的非后代 = {Difficulty, Intelligence} (SAT 是 Grade...不,
    SAT 不是 Grade 的后代, 它们没有边相连; 但 SAT 是 Intelligence 的后代,
    不是 Grade 的非后代...等等)

    先明确: 后代 = 从该节点出发沿箭头方向能到达的所有节点
    Grade 的后代 = {Letter}
    Grade 的非后代 = {Difficulty, Intelligence, SAT}  (图中除了 Grade 和它的后代)
    Grade 的父节点 = {Difficulty, Intelligence}

    所以局部马尔可夫性质要求: Grade ⟂ SAT | {Difficulty, Intelligence}
    即: 知道成绩(DIfficulty+Intelligence)后, SAT分数不能提供关于Grade的额外信息

【全局马尔可夫性质 (Global Markov Property)】
────────────────────────────────────────────
如果 X 和 Y 在图中被 Z d-分离, 则在概率分布 P 中有:

    X ⟂ Y | Z

即: d-separation in G  ⇒  conditional independence in P

这是最强、最常用的性质, 也是练习1中反复使用的规则。

【成对马尔可夫性质 (Pairwise Markov Property)】
────────────────────────────────────────────
对于图中没有边直接相连的两个节点 X 和 Y, 给定所有其他节点:

    X ⟂ Y | (所有其他节点)

这在无向图中更常用, 在 DAG 中可由局部/全局性质推出。

=============================================================================
三、关键洞见: 为什么这些性质重要?
=============================================================================

1. 它们建立了"图结构"和"概率分布"之间的桥梁
   → 看图就能判断哪些变量独立
   → 不用做任何概率计算!

2. 它们是模型设计的指导原则
   → 如果领域知识说 X 和 Y 在给定 Z 时独立,
     那么图结构中 X 和 Y 应该被 Z d-分离

3. 它们是推理算法正确性的理论基础
   → 变量消除法、信念传播等算法的正确性都依赖于这些性质

=============================================================================
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import itertools
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 设置中文字体避免乱码（与 01_d_separation.py 相同的可靠方案）
fm.fontManager.addfont('C:/Windows/Fonts/simhei.ttf')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei'] + plt.rcParams['font.sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def build_student_network():
    """构建学生网络并返回"""
    model = BayesianNetwork([
        ('Difficulty', 'Grade'),
        ('Intelligence', 'Grade'),
        ('Intelligence', 'SAT'),
        ('Grade', 'Letter'),
    ])
    model.add_cpds(TabularCPD('Difficulty', 2, [[0.6], [0.4]]))
    model.add_cpds(TabularCPD('Intelligence', 2, [[0.7], [0.3]]))
    model.add_cpds(TabularCPD('Grade', 3,
                              [[0.3, 0.05, 0.9, 0.5],
                               [0.4, 0.25, 0.08, 0.3],
                               [0.3, 0.7, 0.02, 0.2]],
                              evidence=['Difficulty', 'Intelligence'],
                              evidence_card=[2, 2]))
    model.add_cpds(TabularCPD('SAT', 2,
                              [[0.95, 0.2],
                               [0.05, 0.8]],
                              evidence=['Intelligence'], evidence_card=[2]))
    model.add_cpds(TabularCPD('Letter', 2,
                              [[0.1, 0.4, 0.7],
                               [0.9, 0.6, 0.3]],
                              evidence=['Grade'], evidence_card=[3]))
    return model


# ============================================================================
# 练习 1: 验证局部马尔可夫性质
# ============================================================================

def exercise_local_markov():
    """
    对每个节点, 验证局部马尔可夫性质:

        X ⟂ NonDescendants(X) | Pa(X)

    方法:
    1. 找到 X 的父节点 Pa(X)
    2. 找到 X 的非后代节点 NonDesc(X)
    3. 用 pgmpy 验证: X 是否和每个 NonDesc 节点在给定 Pa(X) 时 d-分离
    """
    print("=" * 70)
    print("练习 1: 验证局部马尔可夫性质 (Local Markov Property)")
    print("=" * 70)

    model = build_student_network()

    # 获取所有非后代节点
    descendants = {}
    non_descendants = {}
    parents = {}

    for node in model.nodes():
        descendants[node] = nx.descendants(model, node)
        parents[node] = set(model.get_parents(node))
        # 非后代 = 所有节点 - 该节点本身 - 该节点的后代
        all_others = set(model.nodes()) - {node}
        non_descendants[node] = all_others - descendants[node]

        print(f"\n  📌 节点: {node}")
        print(f"     父节点 Pa({node}) = {parents[node] or '∅'}")
        print(f"     后代 = {descendants[node] or '∅'}")
        print(f"     非后代 NonDesc({node}) = {non_descendants[node] or '∅'}")

    # 对每个节点, 验证局部马尔可夫性质
    print("\n  ── 验证局部马尔可夫性质 ──\n")

    for node in model.nodes():
        pa = parents[node]
        nd = non_descendants[node]

        if not nd:
            print(f"  {node}: 无非后代节点, 无需验证")
            continue

        print(f"  {node} ⟂ NonDesc | Pa({node}):")

        for nd_node in sorted(nd):
            # 检查: node ⟂ nd_node | Pa(node)
            is_separated = not model.is_dconnected(node, nd_node, observed=list(pa))
            status = "✅ 独立" if is_separated else "❌ 依赖"
            pa_str = ', '.join(pa) if pa else '∅'
            print(f"    {node} ⟂ {nd_node} | {{{pa_str}}}  → {status}")

    print()
    print("  🎯 局部马尔可夫性质的核心直觉:")
    print("     每个节点的父节点就像一道防火墙, 屏蔽了来自")
    print("     上游和侧向的所有信息 — 给定直接原因,")
    print("     其他非结果变量就没用了。")

    return model


# ============================================================================
# 练习 2: 全局马尔可夫性质 — 遍历验证
# ============================================================================

def exercise_global_markov():
    """
    全局马尔可夫性质: d-sep(X, Y | Z) in G  ⇒  X ⟂ Y | Z in P

    练习: 对于学生网络, 找出所有通过 {Grade} d-分离的变量对。
          这在实际应用中相当于: "我知道成绩后, 哪些变量对就无关了?"
    """
    print("\n" + "=" * 70)
    print("练习 2: 全局马尔可夫性质 — 系统性探索")
    print("=" * 70)

    model = build_student_network()
    all_nodes = list(model.nodes())

    print("\n  找出所有被 Grade d-分离的变量对:")
    print("  ──────────────────────────────────")

    observed = ['Grade']
    separated_pairs = []

    for x, y in itertools.combinations(all_nodes, 2):
        if x == y or x in observed or y in observed:
            continue
        if not model.is_dconnected(x, y, observed=observed):
            separated_pairs.append((x, y))

    print(f"  观测变量 Z = {observed}")
    print(f"  被 d-分离的变量对 (共 {len(separated_pairs)} 对):")
    for x, y in separated_pairs:
        print(f"    ✅ {x} ⟂ {y} | Grade")

    print()
    print("  根据全局马尔可夫性质, 这些变量对在真实分布中都是条件独立的。")
    print("  这意味着: 一旦你知道了一个学生的成绩 (Grade), ")
    print("  你就不能从 SAT 分数推断出课程难度, 反之亦然。")

    # 再测试一个: 无观测变量时
    print("\n  找出无条件独立的变量对 (Z = ∅):")
    print("  ──────────────────────────────────")

    unconditional_sep = []
    for x, y in itertools.combinations(all_nodes, 2):
        if not model.is_dconnected(x, y, observed=[]):
            unconditional_sep.append((x, y))

    print(f"  无条件 d-分离的变量对 (共 {len(unconditional_sep)} 对):")
    for x, y in unconditional_sep:
        print(f"    ✅ {x} ⟂ {y}")

    if not unconditional_sep:
        print("    (没有! 这很正常 — 大部分节点在无观测时都有相关性)")
        print("    例如 Difficulty 和 Intelligence 唯一路径经过 collider Grade")
        print("    而 Grade 未观测, 所以它们是边缘独立的! 这是唯一的一对...")
        print()
        print("    💡 等等, 让我重新检查: Difficulty ←?→ Intelligence")
        print("       D → G ← I, G 是 collider 且未观测")
        print("       → D ⟂ I (无条件独立)")

    return model


# ============================================================================
# 练习 3: 从分解推导局部马尔可夫性质 (手动)
# ============================================================================

def exercise_factorization_to_local_markov():
    """
    这是一个 "纸笔练习" — 用数学推导展示:
        P(X) = ∏ P(Xᵢ|Pa(Xᵢ))  ⇒  局部马尔可夫性质

    以 4 个节点的简单链为例:  X₁ → X₂ → X₃ → X₄

    分解: P(X₁,X₂,X₃,X₄) = P(X₁) · P(X₂|X₁) · P(X₃|X₂) · P(X₄|X₃)

    验证局部马尔可夫性质对 X₃ 成立:
        X₃ ⟂ NonDesc(X₃) | Pa(X₃)
    →   X₃ ⟂ {X₁} | {X₂}    (X₄ 是后代, 不在非后代中)
    """
    print("\n" + "=" * 70)
    print("练习 3: 从因子分解推导局部马尔可夫性质 (纸笔推导)")
    print("=" * 70)

    print("""
    图结构: X₁ → X₂ → X₃ → X₄

    因子分解:
        P(X₁,X₂,X₃,X₄) = P(X₁) × P(X₂|X₁) × P(X₃|X₂) × P(X₄|X₃)

    ═══════════════════════════════════════════════════════════
    验证 X₃ 的局部马尔可夫性质: X₃ ⟂ X₁ | X₂
    ═══════════════════════════════════════════════════════════

    步骤 1: 写出 X₁, X₂, X₃ 的联合分布
    ────────────────────────────────────
        P(X₁, X₂, X₃) = Σ_{X₄} P(X₁) P(X₂|X₁) P(X₃|X₂) P(X₄|X₃)

        Σ_{X₄} P(X₄|X₃) = 1  (概率求和 = 1)

        → P(X₁, X₂, X₃) = P(X₁) P(X₂|X₁) P(X₃|X₂)

    步骤 2: 计算条件分布 P(X₃ | X₁, X₂)
    ────────────────────────────────────
        P(X₃ | X₁, X₂) = P(X₁, X₂, X₃) / P(X₁, X₂)

        先算 P(X₁, X₂) = Σ_{X₃} P(X₁) P(X₂|X₁) P(X₃|X₂)
                        = P(X₁) P(X₂|X₁) Σ_{X₃} P(X₃|X₂)
                        = P(X₁) P(X₂|X₁) × 1
                        = P(X₁) P(X₂|X₁)

        P(X₃ | X₁, X₂) = [P(X₁) P(X₂|X₁) P(X₃|X₂)] / [P(X₁) P(X₂|X₁)]
                        = P(X₃|X₂)

    结论: P(X₃ | X₁, X₂) = P(X₃ | X₂)
    ────────────────────────────────────
    给定 X₂ 后, X₃ 的分布不依赖于 X₁ !
    即 X₃ ⟂ X₁ | X₂  ✅ 验证了局部马尔可夫性质

    ═══════════════════════════════════════════════════════════
    推广: 对任意 DAG, 上述推导可以推广到所有节点
    ═══════════════════════════════════════════════════════════

    核心思路:
    - P(X₁,...,Xₙ) = ∏ P(Xᵢ|Pa(Xᵢ))
    - 对任意节点 Xₖ, 联合分布可以分解为:
        (只涉及 Xₖ 的祖先和 Xₖ 的项) × (涉及 Xₖ 后代的项)
    - 边缘化掉后代后, 前半部分只保留到 P(Xₖ|Pa(Xₖ))
    - 所有非后代节点的项不会包含 Xₖ, 因而在计算 P(Xₖ|Pa(Xₖ), ...) 时被消去
    - 最终得到 P(Xₖ | Pa(Xₖ), 其他非后代) = P(Xₖ | Pa(Xₖ))
    """)

    # 用 pgmpy 数值验证
    print("    pgmpy 数值验证:\n")

    chain = BayesianNetwork([('X1', 'X2'), ('X2', 'X3'), ('X3', 'X4')])
    chain.add_cpds(TabularCPD('X1', 2, [[0.5], [0.5]]))
    chain.add_cpds(TabularCPD('X2', 2,
                              [[0.7, 0.3],
                               [0.3, 0.7]],
                              evidence=['X1'], evidence_card=[2]))
    chain.add_cpds(TabularCPD('X3', 2,
                              [[0.8, 0.2],
                               [0.2, 0.8]],
                              evidence=['X2'], evidence_card=[2]))
    chain.add_cpds(TabularCPD('X4', 2,
                              [[0.6, 0.4],
                               [0.4, 0.6]],
                              evidence=['X3'], evidence_card=[2]))

    print(f"    d_separated(X3, X1, Z=['X2']) = {not chain.is_dconnected('X3', 'X1', ['X2'])}")
    print(f"    → X₃ ⟂ X₁ | X₂  ✅")

    # 通过推理验证: P(X₃|X₁=0, X₂=0) 应该等于 P(X₃|X₂=0)
    infer = VariableElimination(chain)
    p_x3_given_x1x2 = infer.query(['X3'], evidence={'X1': 0, 'X2': 0})
    p_x3_given_x2 = infer.query(['X3'], evidence={'X2': 0})

    print(f"\n    P(X₃ | X₁=0, X₂=0) = {p_x3_given_x1x2.values}")
    print(f"    P(X₃ | X₂=0)       = {p_x3_given_x2.values}")
    print(f"    → 两者相同! 验证了局部马尔可夫性质 ✅")
    print()


# ============================================================================
# 练习 4: 无向图的马尔可夫性质对比
# ============================================================================

def exercise_undirected_markov():
    r"""
    对于无向图 (马尔可夫随机场/MRF), 三种马尔可夫性质不再等价!

    这是 L3 的重点内容。我们用一个简单的 4 节点网格图来理解。

    无向图的三种定义:
    ┌─────────────────────────────────────────────────────────┐
    │ 成对马尔可夫:  X ⟂ Y | V \ {X, Y}  对所有不相邻的 X,Y │
    │ 局部马尔可夫:  X ⟂ V \ (cl(X) ∪ {X}) | bd(X)           │
    │ 全局马尔可夫:  d-sep(X,Y|Z) in UG ⇒ X ⟂ Y | Z           │
    │                                                          │
    │ cl(X) = X 的闭包 (X + X 的邻居)                          │
    │ bd(X) = X 的马尔可夫毯 (X 的邻居)                        │
    └─────────────────────────────────────────────────────────┘

    关键: 对于正的分布 (所有概率 > 0):
        全局 ⟺ 局部 ⟺ 成对  (三者等价, Hammersley-Clifford 定理)

    但对于非正分布, 可能成对成立但局部/全局不成立!
    """
    print("\n" + "=" * 70)
    print("练习 4: 无向图的马尔可夫性质 (MRF)")
    print("=" * 70)

    print("""
    考虑一个 2×2 网格图:

        X₁ ── X₂
        │      │
        │      │
        X₃ ── X₄

    成对马尔可夫性质:
      X₁ ⟂ X₄ | {X₂, X₃}  (不相邻的节点, 给定所有其他节点后独立)

    局部马尔可夫性质 (以 X₁ 为例):
      bd(X₁) = {X₂, X₃}  (马尔可夫毯 = 邻居)
      X₁ ⟂ {所有非闭包节点} | {X₂, X₃}

    ⚠️ 注意和 DAG 的区别:
      - DAG 中是"父节点"屏蔽, 无向图中是"邻居"屏蔽
      - DAG 中三种性质等价, 无向图中仅在正分布下等价
      - 无向图中"马尔可夫毯"的概念比 DAG 的"父节点"更对称
    """)

    # 用 pgmpy 构建无向图
    from pgmpy.models import DiscreteMarkovNetwork as MarkovNetwork

    mrf = MarkovNetwork()
    mrf.add_edges_from([('X1', 'X2'), ('X1', 'X3'),
                        ('X2', 'X4'), ('X3', 'X4')])

    print("  MRF 边缘: X1-X2, X1-X3, X2-X4, X3-X4")
    print(f"  X₁ 的邻居 (马尔可夫毯): {list(mrf.neighbors('X1'))}")
    print(f"  X₁ 的非邻居: {set(mrf.nodes()) - set(mrf.neighbors('X1')) - {'X1'}}")
    print()

    # 在无向图中, d-分离的定义更简单
    # X 和 Y 被 Z 分离 ⇔ 所有路径都必须经过 Z 中的至少一个节点
    print("  无向图中的分离 (UG separation):")
    print("    X₁ 和 X₄ 被 {X₂, X₃} 分离吗? → 是 (每条路径都要经过 X₂ 或 X₃)")
    print("    X₁ 和 X₄ 被 {X₂} 分离吗?    → 否 (路径 X₁-X₃-X₄ 不经过 X₂)")
    print()


# ============================================================================
# 练习 5: 马尔可夫毯 (Markov Blanket) 的实践意义
# ============================================================================

def exercise_markov_blanket():
    """
    马尔可夫毯是 DAG 中一个非常实用的概念。

    DAG 中节点 X 的马尔可夫毯 MB(X) 包含:
        1. X 的父节点
        2. X 的子节点
        3. X 的子节点的其他父节点 (X 的"配偶"节点)

    性质: X ⟂ (所有不在 MB(X) 中的节点) | MB(X)

    即: 马尔可夫毯 contains all the information needed to predict X.
    这在特征选择中非常重要!
    """
    print("\n" + "=" * 70)
    print("练习 5: 马尔可夫毯 (Markov Blanket)")
    print("=" * 70)

    model = build_student_network()

    print("\n  学生网络中每个节点的马尔可夫毯:\n")

    for node in model.nodes():
        mb = set()
        # 父节点
        mb.update(model.get_parents(node))
        # 子节点
        children = set(model.get_children(node))
        mb.update(children)
        # 配偶节点 (子节点的其他父节点)
        for child in children:
            mb.update(model.get_parents(child))
        mb.discard(node)  # 移除自己

        print(f"  MB({node}) = {mb}")

    # 验证: Grade 的马尔可夫毯
    print("\n  ── 验证 Grade 的马尔可夫毯 ──")
    print("  MB(Grade) = 父节点{D,I} ∪ 子节点{L} ∪ 配偶(=子节点的其他父节点, L无其他父)")
    print("            = {Difficulty, Intelligence, Letter}")
    print()
    print("  验证: Grade 是否和 SAT (不在 MB 中) 独立?")
    mb_grade = {'Difficulty', 'Intelligence', 'Letter'}
    is_sep = not model.is_dconnected('Grade', 'SAT', observed=list(mb_grade))
    print(f"    d_separated(Grade, SAT, Z=MB(Grade)) = {is_sep}")
    print(f"    → 给定 MB 后, Grade 和 SAT 独立 ✅")
    print()
    print("  🎯 直觉: 要预测一个学生的成绩 (Grade), 你只需要知道:")
    print("     • 课有多难 (Difficulty) — 父节点")
    print("     • 学生有多聪明 (Intelligence) — 父节点")
    print("     • 推荐信怎么样 (Letter) — 子节点")
    print("     • SAT 分数? 不需要! — SAT 在给定 MB 后与 Grade 无关")


# ============================================================================
# 综合练习: 概念对照表
# ============================================================================

def summary_table():
    """打印所有核心概念的总结对照表"""
    print("\n" + "=" * 70)
    print("📋 综合总结: PGM 核心概念对照表")
    print("=" * 70)

    print("""
    ┌────────────────────┬──────────────────────────────────────────────┐
    │       概念          │                  一句话解释                    │
    ├────────────────────┼──────────────────────────────────────────────┤
    │ 因子分解            │ P(X) = ∏ P(Xᵢ|Pa(Xᵢ)), 将联合分布拆成局部条件 │
    │ (Factorization)    │ 概率的乘积, 每项只依赖该节点的父节点              │
    ├────────────────────┼──────────────────────────────────────────────┤
    │ 局部马尔可夫性质    │ X ⟂ NonDesc(X) | Pa(X)                        │
    │ (Local Markov)     │ "给定直接原因后, 与其他非结果变量无关"            │
    ├────────────────────┼──────────────────────────────────────────────┤
    │ 全局马尔可夫性质    │ d-sep(X,Y|Z) in G  ⇒  X ⟂ Y | Z in P          │
    │ (Global Markov)    │ "图中分离 ⇒ 概率独立"                           │
    ├────────────────────┼──────────────────────────────────────────────┤
    │ d-分离              │ 图论判据: 每条路径都被 Z 阻塞                  │
    │ (d-separation)     │ 阻塞条件: 链式/分叉观测中间 → 阻塞               │
    │                    │          汇聚观测中间/后代 → 激活!              │
    ├────────────────────┼──────────────────────────────────────────────┤
    │ I-map              │ G 的 d-分离都是 P 中的条件独立                    │
    │ (独立性映射)        │ 即全局马尔可夫性质成立                            │
    ├────────────────────┼──────────────────────────────────────────────┤
    │ 马尔可夫毯          │ 父 + 子 + 配偶 = 预测 X 所需的最小信息集          │
    │ (Markov Blanket)   │ X ⟂ 外部世界 | MB(X)                          │
    ├────────────────────┼──────────────────────────────────────────────┤
    │ 有向图 vs 无向图    │ DAG: 三种性质等价; MRF: 仅正分布下等价             │
    │                    │ DAG用父节点定义, MRF用邻居(马尔可夫毯)定义          │
    └────────────────────┴──────────────────────────────────────────────┘
    """)


# ============================================================================
# 运行
# ============================================================================

if __name__ == '__main__':
    print("🎓 CMU 10-708: 马尔可夫性质 交互式练习")
    print("=" * 70)

    exercise_local_markov()
    exercise_global_markov()
    exercise_factorization_to_local_markov()
    exercise_undirected_markov()
    exercise_markov_blanket()
    summary_table()

    print("✅ 所有练习完成!")
    print()
    print("💡 建议下一步:")
    print("   1. 回到视频 L2-L3, 对照这些练习重新理解课件")
    print("   2. 用纸笔对更复杂的图手动写出局部/全局马尔可夫性质")
    print("   3. 思考: 如果图中存在环, 三种性质还等价吗?")
