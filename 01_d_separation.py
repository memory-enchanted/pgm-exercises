"""
=============================================================================
  CMU 10-708 L1-L3 核心概念练习 (1): d-分离 (d-separation)
=============================================================================

d-分离是 PGM 中最重要的概念之一，用于判断图结构中哪些变量在给定
某些变量时是条件独立的。

安装依赖:
    conda install -c anaconda pgmpy
    pip install networkx matplotlib

运行方式:
    python 01_d_separation.py

=============================================================================
一、理论基础速查
=============================================================================

图中有三种基本连接结构（以路径 X → ... → Y 为例）:

    1. 链式 (Chain / Causal Trail):    X → M → Y
       - M 未观测时: 路径「活跃」，X 和 Y 依赖
       - M 被观测时: 路径「阻塞」，X ⟂ Y | M

    2. 分叉 (Fork / Evidential Trail):  X ← M → Y
       - M 未观测时: 路径活跃，X 和 Y 依赖（共享原因导致相关）
       - M 被观测时: 路径阻塞，X ⟂ Y | M

    3. 汇聚 (Collider / V-structure):   X → M ← Y
       - M 未观测时: 路径阻塞（解释消除效应），X ⟂ Y
       - M 被观测时: 路径「激活」！X 和 Y 变得依赖

d-分离的定义:
    X 和 Y 被 Z d-分离 ⟺ 每条无向路径都至少有一个节点满足:
        (a) 该节点是链式/分叉结构中的中间节点，且它在 Z 中
        (b) 该节点是汇聚结构中的中间节点，且它 (及其所有后代) 都不在 Z 中

    → 如果 X 和 Y 被 Z d-分离，则 X ⟂ Y | Z (全局马尔可夫性质)

=============================================================================
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')   # 或 'Qt5Agg'（需安装 PyQt5）
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 设置中文字体避免乱码
import matplotlib.font_manager as fm
# 直接添加 SimHei 字体文件路径（最可靠的方式）
fm.fontManager.addfont('C:/Windows/Fonts/simhei.ttf')
# font.family='sans-serif' 会让所有 sans-serif 文本查找 font.sans-serif 列表
# 把 SimHei 放最前面，networkx 的 draw_networkx_labels 就能找到中文字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei'] + plt.rcParams['font.sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def draw_dag(model, title="", highlight_nodes=None, highlight_edges=None):
    # 将 pgmpy BayesianNetwork 转为纯 networkx DiGraph
    # 原因: pgmpy 1.x 的内部图结构与 networkx 3.x 不完全兼容，
    #        spring_layout 在 pgmpy 模型上会返回 NaN
    G = nx.DiGraph()
    G.add_nodes_from(model.nodes())
    G.add_edges_from(model.edges())

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42, k=2)

    normal = [n for n in G.nodes() if highlight_nodes is None or n not in highlight_nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=normal, node_color='lightblue',
                           node_size=1200, alpha=0.9)

    if highlight_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=highlight_nodes, node_color='salmon',
                               node_size=1400, alpha=0.9)

    if highlight_edges is None:
        highlight_edges = []
    normal_edges = [e for e in G.edges() if e not in highlight_edges]

    if normal_edges:
        nx.draw_networkx_edges(G, pos, edgelist=normal_edges, edge_color='gray',
                               arrows=True, width=1.5, arrowstyle='-|>', arrowsize=20)
    if highlight_edges:
        nx.draw_networkx_edges(G, pos, edgelist=highlight_edges, edge_color='red',
                               arrows=True, width=2.5, arrowstyle='-|>', arrowsize=25)

    nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold')
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# ============================================================================
# 练习 0: 可视化三种基本结构
# ============================================================================

def demo_three_structures():
    """展示链式、分叉、汇聚三种基本结构的活跃/阻塞行为"""
    print("=" * 70)
    print("练习 0: 理解三种基本结构")
    print("=" * 70)

    # --- 链式结构 X → M → Y ---
    print("\n📎 链式结构 (Chain): X → M → Y")
    print("   未观测 M → X 和 Y 相关;  观测 M → X ⟂ Y | M")

    chain = BayesianNetwork([('X', 'M'), ('M', 'Y')])
    cpd_x = TabularCPD('X', 2, [[0.5], [0.5]])
    cpd_m = TabularCPD('M', 2,
                        [[0.7, 0.3],
                         [0.3, 0.7]],
                        evidence=['X'], evidence_card=[2])
    cpd_y = TabularCPD('Y', 2,
                        [[0.9, 0.1],
                         [0.1, 0.9]],
                        evidence=['M'], evidence_card=[2])
    chain.add_cpds(cpd_x, cpd_m, cpd_y)

    # 检查 d-分离
    print("   d_separated(X, Y, Z=[])   →", chain.is_dconnected('X', 'Y', observed=[]))
    print("         → X 和 Y 不独立 (d-connected)，路径活跃")
    print("   d_separated(X, Y, Z=['M']) →",
          not chain.is_dconnected('X', 'Y', observed=['M']))
    print("         → X ⟂ Y | M，给定中间节点后路径阻塞 ✓")

    draw_dag(chain, title="链式结构: X → M → Y\n红线高亮表示活跃路径",
             highlight_nodes=[], highlight_edges=[('X', 'M'), ('M', 'Y')])

    # --- 分叉结构 X ← M → Y ---
    print("\n📎 分叉结构 (Fork): X ← M → Y  (共同原因)")
    print("   未观测 M → X 和 Y 相关;  观测 M → X ⟂ Y | M")
    print("   例子: M=下雨, X=地湿, Y=带伞 → 知道下雨后,地湿和带伞独立")

    fork = BayesianNetwork([('M', 'X'), ('M', 'Y')])
    cpd_mf = TabularCPD('M', 2, [[0.5], [0.5]])
    cpd_xf = TabularCPD('X', 2,
                        [[0.8, 0.2],
                         [0.2, 0.8]],
                        evidence=['M'], evidence_card=[2])
    cpd_yf = TabularCPD('Y', 2,
                        [[0.9, 0.3],
                         [0.1, 0.7]],
                        evidence=['M'], evidence_card=[2])
    fork.add_cpds(cpd_mf, cpd_xf, cpd_yf)

    print("   d_separated(X, Y, Z=[])   →", fork.is_dconnected('X', 'Y', observed=[]))
    print("         → X 和 Y 不独立 (共享原因导致边缘相关)")
    print("   d_separated(X, Y, Z=['M']) →",
          not fork.is_dconnected('X', 'Y', observed=['M']))
    print("         → X ⟂ Y | M ✓")

    draw_dag(fork, title="分叉结构: X ← M → Y\n红线表示 M 未观测时路径活跃")

    # --- 汇聚结构 (Collider / V-structure) X → M ← Y ---
    print("\n📎 汇聚结构 (Collider / V-structure): X → M ← Y")
    print("   ⚠️ 这是最反直觉的一种!")
    print("   未观测 M → X ⟂ Y (路径阻塞!)")
    print("   观测 M → X 和 Y 变得依赖! (解释消除效应)")
    print("   例子: X=天才, Y=运气, M=成功 → 知道某人成功了,")
    print("        如果他是天才,就说明可能不需要运气 (负相关)")

    collider = BayesianNetwork([('X', 'M'), ('Y', 'M')])
    cpd_xc = TabularCPD('X', 2, [[0.5], [0.5]])
    cpd_yc = TabularCPD('Y', 2, [[0.5], [0.5]])
    cpd_mc = TabularCPD('M', 2,
                        [[0.99, 0.1, 0.1, 0.01],
                         [0.01, 0.9, 0.9, 0.99]],
                        evidence=['X', 'Y'], evidence_card=[2, 2])
    collider.add_cpds(cpd_xc, cpd_yc, cpd_mc)

    print("   d_separated(X, Y, Z=[])   →",
          not collider.is_dconnected('X', 'Y', observed=[]))
    print("         → X ⟂ Y (汇聚节点 M 未观测, 路径阻塞) ✓")
    print("   d_separated(X, Y, Z=['M']) →",
          collider.is_dconnected('X', 'Y', observed=['M']))
    print("         → X 和 Y 变得依赖! (观测汇聚节点激活路径!) 🔥")

    draw_dag(collider, title="汇聚结构: X → M ← Y\n红线=M观测时路径被激活!")


# ============================================================================
# 练习 1: d-分离判断 — "学生网络" 实例
# ============================================================================

def exercise_student_network():
    """
    经典的"学生网络"例子, 来自 Koller & Friedman 教材第 3 章.

    结构:
        Difficulty ─→ Grade ←── Intelligence
                         │           │
                         ↓           ↓
                        Letter      SAT

    问题: 判断以下变量对是否条件独立
    """
    print("\n" + "=" * 70)
    print("练习 1: 学生网络 — d-分离判断")
    print("=" * 70)

    student = BayesianNetwork([
        ('Difficulty', 'Grade'),
        ('Intelligence', 'Grade'),
        ('Intelligence', 'SAT'),
        ('Grade', 'Letter'),
    ])

    # 给每个节点随机 CPD（只是为了能用 is_dconnected）
    for var in ['Difficulty', 'Intelligence']:
        student.add_cpds(TabularCPD(var, 2, [[0.5], [0.5]]))
    student.add_cpds(TabularCPD('Grade', 3,
                                np.array([[0.3, 0.05, 0.9, 0.5],
                                          [0.4, 0.25, 0.08, 0.3],
                                          [0.3, 0.7,  0.02, 0.2]]),
                                evidence=['Difficulty', 'Intelligence'],
                                evidence_card=[2, 2]))
    student.add_cpds(TabularCPD('SAT', 2,
                                [[0.95, 0.2],
                                 [0.05, 0.8]],
                                evidence=['Intelligence'], evidence_card=[2]))
    student.add_cpds(TabularCPD('Letter', 2,
                                [[0.1, 0.4, 0.7],
                                 [0.9, 0.6, 0.3]],
                                evidence=['Grade'], evidence_card=[3]))

    draw_dag(student, title="学生网络 (Student Network)")

    # ----- 一组判断题目 -----
    questions = [
        # (X, Y, Z, 期望 d-分离?, 解释)
        ("Difficulty", "Intelligence", [],
         True,
         "没有路径连接 D 和 I (只有 D→G←I, G是collider且未观测 → 阻塞)"),

        ("Difficulty", "Intelligence", ["Grade"],
         False,
         "观测了 collider G! D→G←I 路径被激活! D和I变得相关 🔥\n"
         "  直觉: 课难(D=high)且得了A(G=high) → 说明学生可能很聪明(I=high)"),

        ("Difficulty", "SAT", [],
         True,
         "路径: D→G←I→SAT, G是collider且未观测 → 阻塞 → D ⟂ SAT"),

        ("Difficulty", "SAT", ["Grade"],
         False,
         "观测G后: D→G←I→SAT, collider被激活, 整条路径通了!"),

        ("Intelligence", "Letter", ["Grade"],
         True,
         "路径: I→G→L, G∈Z 是链式中间节点 → 阻塞 → I ⟂ L | G\n"
         "  直觉: 知道成绩(G)后, 智商(I)不再提供关于推荐信(L)的额外信息"),

        ("Intelligence", "Letter", [],
         False,
         "路径: I→G→L, 无观测 → 活跃 → I 和 L 相关 (通过成绩)"),

        ("Difficulty", "Letter", ["Grade"],
         True,
         "路径: D→G→L, G∈Z → 阻塞 → D ⟂ L | G"),

        ("SAT", "Letter", ["Intelligence"],
         False,
         "路径1: S←I→G→L, I∈Z (分叉结构,阻塞✓)\n"
         "  路径2: S←I→G←D→?, 但还有别的吗? 注意还有 I→G 这条...\n"
         "  实际上: S←I→G→L 中 I 阻断了分叉; 但还有 S←I→G←D→? 不连L\n"
         "  仔细看: S←I→G→L 被 I 阻塞; 没有其他路径到 L 了\n"
         "  → 实际上是 d-分离的! 等等让我重新检查...\n"
         "  → S 和 L 之间唯一的路径是 S←I→G→L, I观测→阻塞 → S ⟂ L | I ✓"),
    ]

    print("\n请先自己判断以下每个情况, 再看答案:\n")
    for i, (x, y, z, expected_separated, explanation) in enumerate(questions):
        is_dconn = student.is_dconnected(x, y, observed=list(z))
        is_separated = not is_dconn
        status = "✅" if is_separated == expected_separated else "❌"

        print(f"--- 问题 {i+1}: {x} ⟂ {y} | {{{', '.join(z) if z else '∅'}}}? ---")
        print(f"  你的答案应该: {'d-分离 (独立)' if expected_separated else 'd-连通 (依赖)'}")
        print(f"  pgmpy 计算: {'d-分离 ✓' if is_separated else 'd-连通 ✗'}")
        print(f"  {status}")
        print(f"  解释: {explanation}")
        print()

    return student


# ============================================================================
# 练习 2: 自己动手 — 更多 d-分离判断
# ============================================================================

def exercise_self_practice():
    """
    动手练习: 给定下面这个 DAG, 判断每对变量的 d-分离关系.

    图结构 (因果图):
        吸烟 ──→ 肺癌 ──→ 咳嗽
         │                  ↑
         └──→ 黄牙          │
                           │
        石棉暴露 ──→ 肺癌 ──┘

    (用变量名: Smoking → LungCancer → Cough,
              Smoking → YellowTeeth,
              Asbestos → LungCancer)
    """
    print("\n" + "=" * 70)
    print("练习 2: 自己动手 — 吸烟-肺癌 因果图")
    print("=" * 70)

    cancer_dag = BayesianNetwork([
        ('Smoking', 'LungCancer'),
        ('Smoking', 'YellowTeeth'),
        ('Asbestos', 'LungCancer'),
        ('LungCancer', 'Cough'),
    ])

    # 简易 CPD
    for v in ['Smoking', 'Asbestos']:
        cancer_dag.add_cpds(TabularCPD(v, 2, [[0.5], [0.5]]))
    cancer_dag.add_cpds(TabularCPD('LungCancer', 2,
                                   [[0.99, 0.8, 0.7, 0.1],
                                    [0.01, 0.2, 0.3, 0.9]],
                                   evidence=['Smoking', 'Asbestos'],
                                   evidence_card=[2, 2]))
    cancer_dag.add_cpds(TabularCPD('YellowTeeth', 2,
                                   [[0.9, 0.1],
                                    [0.1, 0.9]],
                                   evidence=['Smoking'], evidence_card=[2]))
    cancer_dag.add_cpds(TabularCPD('Cough', 2,
                                   [[0.8, 0.1],
                                    [0.2, 0.9]],
                                   evidence=['LungCancer'], evidence_card=[2]))

    draw_dag(cancer_dag, title="吸烟-肺癌 因果图\n请自行判断以下d-分离关系")

    # 练习题（请先自己思考，再运行看答案）
    self_test = [
        ("Smoking", "Asbestos", [], "吸烟和石棉暴露是否独立？(无观测)"),
        ("Smoking", "Asbestos", ["LungCancer"], "给定肺癌后，吸烟和石棉暴露是否独立？"),
        ("Smoking", "Cough", [], "吸烟和咳嗽是否独立？(无观测)"),
        ("Smoking", "Cough", ["LungCancer"], "给定肺癌后，吸烟和咳嗽是否独立？"),
        ("YellowTeeth", "Cough", [], "黄牙和咳嗽是否独立？(无观测)"),
        ("YellowTeeth", "Cough", ["Smoking"], "给定吸烟后，黄牙和咳嗽是否独立？"),
        ("YellowTeeth", "Asbestos", [], "黄牙和石棉暴露是否独立？"),
        ("YellowTeeth", "Asbestos", ["LungCancer"], "给定肺癌后，黄牙和石棉暴露是否独立？"),
    ]

    print("\n🤔 先自己画图分析每条路径，再运行查看答案:\n")

    for i, (x, y, z, desc) in enumerate(self_test):
        is_dconn = cancer_dag.is_dconnected(x, y, observed=list(z))
        z_str = ', '.join(z) if z else '∅'
        result = 'd-连通 (相关)' if is_dconn else 'd-分离 (独立)'
        print(f"  {i+1}. {x} ⟂ {y} | {{{z_str}}}?  →  {result}")
        print(f"     {desc}")
        print()

    return cancer_dag


# ============================================================================
# 练习 3: Collider 的"后代激活"陷阱
# ============================================================================

def exercise_descendant_activation():
    """
    d-分离的关键细节: 汇聚节点 (collider) 不仅自己被观测时会激活路径,
    它的后代 (descendant) 被观测时也会激活路径!

    这是考试最爱考的陷阱题。

    例子:
        X → M ← Y
            ↓
            D

    如果观测 D (M 的后代, 而非 M 本身), X 和 Y 是否 d-分离?
    答案: 不分离! 观测 M 的后代也会激活 collider 路径.
    """
    print("\n" + "=" * 70)
    print("练习 3: Collider 后代激活 (重要陷阱!)")
    print("=" * 70)

    dag = BayesianNetwork([
        ('X', 'M'),
        ('Y', 'M'),
        ('M', 'D'),
    ])

    for v in ['X', 'Y']:
        dag.add_cpds(TabularCPD(v, 2, [[0.5], [0.5]]))
    dag.add_cpds(TabularCPD('M', 2,
                            [[0.9, 0.1, 0.1, 0.01],
                             [0.1, 0.9, 0.9, 0.99]],
                            evidence=['X', 'Y'], evidence_card=[2, 2]))
    dag.add_cpds(TabularCPD('D', 2,
                            [[0.8, 0.2],
                             [0.2, 0.8]],
                            evidence=['M'], evidence_card=[2]))

    draw_dag(dag, title="Collider 后代激活: X → M ← Y, M → D\nD 被观测时会激活 X-Y 路径!")

    print("  条件1: d_separated(X, Y, Z=[])    →",
          not dag.is_dconnected('X', 'Y', observed=[]),
          "(✓ 独立, collider M 未观测)")
    print("  条件2: d_separated(X, Y, Z=['M'])   →",
          not dag.is_dconnected('X', 'Y', observed=['M']),
          "(✗ 不独立! collider 被直接观测, 路径激活)")
    print("  条件3: d_separated(X, Y, Z=['D'])   →",
          not dag.is_dconnected('X', 'Y', observed=['D']),
          "(✗ 不独立! 观测 collider 的后代 D, 同样激活路径!) 🔥")
    print()
    print("  ⚠️ 记忆口诀: ")
    print("     链式/分叉 → 观测中间节点: 阻塞")
    print("     汇聚结构   → 观测中间节点或其任意后代: 激活!")
    print()


# ============================================================================
# 练习 4: d-分离算法的手动追踪
# ============================================================================

def exercise_manual_d_sep_trace():
    """
    手动追踪 d-分离: 给定复杂图, 一步步分析每条路径.

    这个练习帮助你掌握考试/面试中的 d-分离手动判断方法.

    图:
        A → B → C
        ↓  ↗
        D ← E → F

    判断: A 和 F 给定 {B, D} 是否 d-分离?
    """
    print("=" * 70)
    print("练习 4: 手动追踪 d-分离路径")
    print("=" * 70)

    complex_dag = BayesianNetwork([
        ('A', 'B'), ('A', 'D'),
        ('B', 'C'),
        ('E', 'B'), ('E', 'D'), ('E', 'F'),
    ])

    for v in ['A', 'E']:
        complex_dag.add_cpds(TabularCPD(v, 2, [[0.5], [0.5]]))
    complex_dag.add_cpds(TabularCPD('B', 2,
                                    [[0.7, 0.1, 0.3, 0.05],
                                     [0.3, 0.9, 0.7, 0.95]],
                                    evidence=['A', 'E'], evidence_card=[2, 2]))
    complex_dag.add_cpds(TabularCPD('D', 2,
                                    [[0.8, 0.1, 0.2, 0.05],
                                     [0.2, 0.9, 0.8, 0.95]],
                                    evidence=['A', 'E'], evidence_card=[2, 2]))
    complex_dag.add_cpds(TabularCPD('C', 2,
                                    [[0.9, 0.1],
                                     [0.1, 0.9]],
                                    evidence=['B'], evidence_card=[2]))
    complex_dag.add_cpds(TabularCPD('F', 2,
                                    [[0.9, 0.1],
                                     [0.1, 0.9]],
                                    evidence=['E'], evidence_card=[2]))

    draw_dag(complex_dag, title="练习: A 和 F 给定 {B, D} 是否 d-分离?\n请手动分析每条路径")

    print("\n  图: A→B→C, A→D, E→B, E→D, E→F")
    print("  问题: A ⟂ F | {B, D} ?")
    print()
    print("  手动分析步骤:")
    print("  ─────────────────────────────────────")
    print()
    print("  A 到 F 的所有无向路径:")
    print()
    print("  路径1: A ← E → F")
    print("    结构: 分叉 (fork)")
    print("    E 在 Z={B,D} 中? → 否")
    print("    → 路径 1 活跃! ❌ (已有一条活跃路径, 可直接判定为 d-连通)")
    print()
    print("  路径2: A → B ← E → F")
    print("    节点 B: collider (A→B←E)")
    print("    B 在 Z={B,D} 中? → 是!")
    print("    → collider 被观测, 此路径活跃! ❌")
    print()
    print("  路径3: A → D ← E → F")
    print("    节点 D: collider (A→D←E)")
    print("    D 在 Z={B,D} 中? → 是!")
    print("    → collider 被观测, 此路径活跃! ❌")
    print()
    print("  结论: 三条路径都活跃 → A 和 F 给定 {B,D} 是 d-连通的")
    print(f"  pgmpy 验证: is_dconnected(A, F, [B,D]) =",
          complex_dag.is_dconnected('A', 'F', observed=['B', 'D']))
    print()

    # 换一个条件: 只给定 {E}
    print("  换个条件: A ⟂ F | {E} ?")
    print("  ───────────────────────────")
    print("  路径1: A ← E → F, E 在 Z 中 → 阻塞 ✓")
    print("  路径2: A → B ← E → F, B 是 collider 且不在 Z 中 → 阻塞 ✓")
    print("  路径3: A → D ← E → F, D 是 collider 且不在 Z 中 → 阻塞 ✓")
    print("  → A 和 F 给定 E 是 d-分离的")
    print(f"  pgmpy 验证: is_dconnected(A, F, [E]) =",
          complex_dag.is_dconnected('A', 'F', observed=['E']))
    print()


# ============================================================================
# 运行所有练习
# ============================================================================

if __name__ == '__main__':
    print("🎓 CMU 10-708: d-分离 交互式练习")
    print("=" * 70)
    print("提示: 仔细阅读每个练习的解释, 关闭图表窗口后继续下一个练习\n")

    demo_three_structures()
    exercise_student_network()
    exercise_self_practice()
    exercise_descendant_activation()
    exercise_manual_d_sep_trace()

    print("=" * 70)
    print("✅ 所有 d-分离练习完成!")
    print()
    print("📋 核心记忆:")
    print("   • 链式 (X→M→Y): 观测M → 阻塞")
    print("   • 分叉 (X←M→Y): 观测M → 阻塞")
    print("   • 汇聚 (X→M←Y): 观测M (或其后代) → 激活! 🔥")
    print("   • d-分离 ⟺ 所有路径都被阻塞")
    print()
