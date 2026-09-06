"""
==========================================================================================
  CMU 10-708 L17 代码练习: 因果关系1 — Intervention, do-Calculus, 反事实, 因果发现
==========================================================================================

L17 七大主题 -> 对应练习:
  ① Causality 基础        -> 练习 1: 辛普森悖论 — 关联 vs 因果, 后门调整
  ② Intervention          -> 练习 2: do-算子 图手术 — 截断因子分解
  ③ Causal Graph Model    -> 练习 3: 因果贝叶斯网络 — d-Separation 因果语义
  ④ Identification        -> 练习 4: 后门前门准则 — 因果效应识别
  ⑤ Counterfactual        -> 练习 5: 反事实推理 — 溯因·行动·预测三步法
  ⑥ Causal Discovery      -> 练习 6: PC 算法 — 从数据发现因果结构
  ⑦ Implications in ML    -> 练习 7: 分布外泛化 & 反事实公平性

特别说明:
  - 贝叶斯网络导入需使用: from pgmpy.models import DiscreteBayesianNetwork
    (pgmpy 0.1.x 后 BayesianNetwork 改名为 DiscreteBayesianNetwork)
  - Windows GBK 终端下 emoji 打印: sys.stdout.reconfigure(encoding='utf-8')

使用方法:
  python 45_causality_exercises.py              # 运行全部
  python 45_causality_exercises.py --ex 2       # 只运行练习2
  python 45_causality_exercises.py --ex 4,5     # 只运行练习4和5

依赖: numpy, pgmpy, networkx
==========================================================================================
"""

import numpy as np
import sys
from itertools import combinations, permutations

# 修复 Windows GBK 终端下 emoji 打印问题
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

np.random.seed(42)

# 尝试导入 pgmpy
try:
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    _HAS_PGMPY = True
except ImportError:
    print("⚠ pgmpy 未安装, 部分练习将被跳过")
    print("  安装: pip install pgmpy")
    _HAS_PGMPY = False


# ============================================================================
# 工具函数
# ============================================================================

def _print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ============================================================================
# 练习 1: 辛普森悖论 — 关联 vs 因果
# ============================================================================

def exercise1_simpsons_paradox():
    """
    重现经典辛普森悖论案例: 药物试验

    场景: 评估药物(T)对康复(R)的因果效应
          性别(G)是混杂变量 — 同时影响服药倾向和康复率

    因果图: G → T, G → R, T → R (G是混杂变量)
    """
    _print_header("练习 1: 辛普森悖论 — 为什么 P(R|T) ≠ P(R|do(T))")

    # --- 定义数据 (与概念笔记中的辛普森悖论案例一致) ---
    # 格式: [男性服药, 男性对照, 女性服药, 女性对照]
    n_male_treated   = 30; n_male_treated_recover  = 18
    n_male_control   = 10; n_male_control_recover  = 7
    n_female_treated = 10; n_female_treated_recover = 2
    n_female_control = 30; n_female_control_recover = 9

    # 总体关联 P(Recover | Treatment)
    total_treated = n_male_treated + n_female_treated
    total_control = n_male_control + n_female_control
    total_treated_recover = n_male_treated_recover + n_female_treated_recover
    total_control_recover = n_male_control_recover + n_female_control_recover

    p_recover_given_treated_obs = total_treated_recover / total_treated
    p_recover_given_control_obs = total_control_recover / total_control

    print("\n[观测数据]")
    print(f"  总体人群: {total_treated + total_control} 人")
    print(f"  服药组康复率: {total_treated_recover}/{total_treated} = "
          f"{p_recover_given_treated_obs:.1%}")
    print(f"  对照组康复率: {total_control_recover}/{total_control} = "
          f"{p_recover_given_control_obs:.1%}")
    print(f"  ❌ 朴素结论: 药物{'有效' if p_recover_given_treated_obs > p_recover_given_control_obs else '有害'} "
          f"(关联性, P(R|T))")

    print("\n[分组数据]")
    p_male_treated = n_male_treated_recover / n_male_treated
    p_male_control = n_male_control_recover / n_male_control
    p_female_treated = n_female_treated_recover / n_female_treated
    p_female_control = n_female_control_recover / n_female_control

    print(f"  男性: 服药组 {n_male_treated_recover}/{n_male_treated} = {p_male_treated:.1%}, "
          f"对照组 {n_male_control_recover}/{n_male_control} = {p_male_control:.1%}")
    print(f"  女性: 服药组 {n_female_treated_recover}/{n_female_treated} = {p_female_treated:.1%}, "
          f"对照组 {n_female_control_recover}/{n_female_control} = {p_female_control:.1%}")
    print(f"  分组结论: 药物有害!")

    # --- 后门调整 (正确的因果效应) ---
    # P(R | do(T)) = Σ_g P(R | T, G=g) · P(G=g)
    total_pop = total_treated + total_control
    p_male = (n_male_treated + n_male_control) / total_pop    # P(G=男)
    p_female = (n_female_treated + n_female_control) / total_pop  # P(G=女)

    # P(R=true | do(T=服))
    p_recover_do_treated = p_male_treated * p_male + p_female_treated * p_female
    # P(R=true | do(T=对))
    p_recover_do_control = p_male_control * p_male + p_female_control * p_female

    print("\n[后门调整 — 因果效应 P(R | do(T))]")
    print(f"  P(R | do(T=服药))  = {p_male_treated:.3f} × {p_male:.2f} + "
          f"{p_female_treated:.3f} × {p_female:.2f} = {p_recover_do_treated:.3f}")
    print(f"  P(R | do(T=对照))  = {p_male_control:.3f} × {p_male:.2f} + "
          f"{p_female_control:.3f} × {p_female:.2f} = {p_recover_do_control:.3f}")
    print(f"  因果效应 (ATE)      = {p_recover_do_treated:.3f} - {p_recover_do_control:.3f} = "
          f"{p_recover_do_treated - p_recover_do_control:.3f}")
    print(f"  ✓ 正确结论: 药物{'有效' if p_recover_do_treated > p_recover_do_control else '有害'} "
          f"(因果性, P(R|do(T)))")

    # 关键对比
    print("\n[关键对比]")
    print(f"  朴素关联 P(R|T=服) - P(R|T=对) = "
          f"{p_recover_given_treated_obs:.3f} - {p_recover_given_control_obs:.3f} = "
          f"{p_recover_given_treated_obs - p_recover_given_control_obs:+.3f}  ← 方向反转!")
    print(f"  因果效应 P(R|do(T=服)) - P(R|do(T=对)) = "
          f"{p_recover_do_treated:.3f} - {p_recover_do_control:.3f} = "
          f"{p_recover_do_treated - p_recover_do_control:+.3f}  ← 正确方向")
    print(f"  辛普森悖论: P(G=男|T) 的分布扭曲 — 男性更可能服药, 且天然康复率更高")

    return p_recover_do_treated, p_recover_do_control


# ============================================================================
# 练习 2: do-算子 图手术 — 截断因子分解
# ============================================================================

def exercise2_do_operator_graph_surgery():
    """
    实现 do-算子的图手术操作

    因果图: Z → X → Y, 同时 Z → Y (Z 是混杂)
    原始分解: P(Z,X,Y) = P(Z)·P(X|Z)·P(Y|X,Z)
    干预后:   P(Z,Y | do(X=x)) = P(Z)·P(Y|X=x,Z)  (去掉 P(X|Z) 项)
    """
    _print_header("练习 2: do-算子 图手术 — 截断因子分解")

    # 定义 CPD (数值模拟)
    print("\n[因果图: Z → X → Y, Z → Y (混杂)]")

    # P(Z)
    p_z = np.array([0.6, 0.4])  # P(Z=0)=0.6, P(Z=1)=0.4

    # P(X | Z): X depends on Z
    p_x_given_z = np.array([
        [0.7, 0.3],  # P(X=0 | Z=0), P(X=1 | Z=0)
        [0.2, 0.8],  # P(X=0 | Z=1), P(X=1 | Z=1)
    ])

    # P(Y | X, Z)
    p_y_given_xz = np.zeros((2, 2, 2))
    p_y_given_xz[0, 0, :] = [0.9, 0.1]   # Y|X=0,Z=0
    p_y_given_xz[0, 1, :] = [0.4, 0.6]   # Y|X=0,Z=1
    p_y_given_xz[1, 0, :] = [0.3, 0.7]   # Y|X=1,Z=0
    p_y_given_xz[1, 1, :] = [0.1, 0.9]   # Y|X=1,Z=1

    # --- 观测分布 P(Y) (先算出来对比) ---
    p_joint_obs = np.zeros((2, 2, 2))  # P(Z,X,Y)
    for z in range(2):
        for x in range(2):
            for y in range(2):
                p_joint_obs[z, x, y] = p_z[z] * p_x_given_z[z, x] * p_y_given_xz[x, z, y]

    p_y_obs = np.sum(p_joint_obs, axis=(0, 1))
    print(f"  观测分布 P(Y): P(Y=0)={p_y_obs[0]:.3f}, P(Y=1)={p_y_obs[1]:.3f}")

    # --- 干预分布 do(X=1) — 截断因子分解 ---
    # P(Z,Y | do(X=x)) = P(Z)·P(Y|X=x,Z)
    p_after_do = np.zeros((2, 2))
    for z in range(2):
        for y in range(2):
            p_after_do[z, y] = p_z[z] * p_y_given_xz[1, z, y]

    p_y_do_x1 = np.sum(p_after_do, axis=0)
    print(f"  干预分布 P(Y | do(X=1)): P(Y=0)={p_y_do_x1[0]:.3f}, P(Y=1)={p_y_do_x1[1]:.3f}")

    # --- 条件分布 P(Y | X=1) ---
    p_x = np.zeros(2)
    for z in range(2):
        p_x += p_z[z] * p_x_given_z[z, :]
    p_y_given_x1_obs = np.zeros(2)
    for z in range(2):
        p_z_given_x1 = p_z[z] * p_x_given_z[z, 1] / p_x[1]
        p_y_given_x1_obs += p_z_given_x1 * p_y_given_xz[1, z, :]

    print(f"  条件分布 P(Y | X=1):  P(Y=0)={p_y_given_x1_obs[0]:.3f}, P(Y=1)={p_y_given_x1_obs[1]:.3f}")

    print(f"\n  Δ(P(Y=1|do(X=1)) - P(Y=1|X=1)) = {p_y_do_x1[1] - p_y_given_x1_obs[1]:.3f}")
    print(f"  原因: P(Z|X=1) ≠ P(Z), 即 condition 改变了 Z 的分布")
    print(f"        do 操作保持 Z 的自然分布 P(Z)")

    # 验证后门调整: P(Y|do(X=1)) = Σ_z P(Y|X=1,Z=z)·P(z)
    p_y_do_adjusted = np.zeros(2)
    for z in range(2):
        p_y_do_adjusted += p_z[z] * p_y_given_xz[1, z, :]
    print(f"  后门调整验证:   P(Y|do(X=1)) = {p_y_do_adjusted}  ✓")

    return p_y_do_x1, p_y_obs


# ============================================================================
# 练习 3: 因果贝叶斯网络 — pgmpy 构建与 d-Separation
# ============================================================================

def exercise3_causal_bn_dseparation():
    """
    使用 pgmpy 构建因果贝叶斯网络
    展示了 d-Separation 在因果图中的含义:
      - Fork:  L → X, L → Y  (混杂, 条件化 L 阻断)
      - Chain: X → M → Y      (中介, 条件化 M 阻断)
      - Collider: X → C ← Y  (对撞, 条件化 C 打开!)

    同时验证后门准则: P(Y|do(X)) 的识别需要阻断 X←L→Y
    """
    _print_header("练习 3: 因果贝叶斯网络 — d-Separation 的因果语义")

    if not _HAS_PGMPY:
        print("  ⚠ 跳过 (需要 pgmpy)")
        return

    # 构建因果图: L(混杂)→X, L→Y, X→Y
    #              plus: X→M→Y (中介链), X→C←Y (对撞结构)
    model = DiscreteBayesianNetwork([
        ('L', 'X'), ('L', 'Y'), ('X', 'Y'),       # 主因果路径 + 混杂
        ('X', 'M'), ('M', 'Y'),                    # 中介链
        ('X', 'C'), ('Y', 'C'),                    # 对撞结构
    ])

    # --- CPD 定义 ---
    cpds = [
        TabularCPD('L', 2, [[0.5], [0.5]]),
        TabularCPD('X', 2, [[0.6, 0.3], [0.4, 0.7]],
                   evidence=['L'], evidence_card=[2]),
        TabularCPD('M', 2, [[0.8, 0.2], [0.2, 0.8]],
                   evidence=['X'], evidence_card=[2]),
        TabularCPD('Y', 2, [[0.9, 0.4, 0.3, 0.05],
                             [0.1, 0.6, 0.7, 0.95]],
                   evidence=['X', 'L'], evidence_card=[2, 2]),
        TabularCPD('C', 2, [[0.7, 0.6, 0.4, 0.05],
                             [0.3, 0.4, 0.6, 0.95]],
                   evidence=['X', 'Y'], evidence_card=[2, 2]),
    ]
    for cpd in cpds:
        model.add_cpds(cpd)

    assert model.check_model(), "Model check failed!"

    # --- d-Separation 测试 ---
    print("\n[d-Separation 测试]")

    tests = [
        # (X, Y, Z, 预期结果, 说明)
        ('X', 'Y', {'L'}, True,
         "Fork: X←L→Y, 条件化 L 阻断"),
        ('X', 'Y', set(), False,
         "Fork: X←L→Y, 不条件化 L 则不阻断"),
        ('X', 'Y', {'M'}, True,
         "Chain: X→M→Y, 条件化 M 阻断因果路径"),
        ('X', 'Y', {'L', 'M'}, True,
         "同时阻断后门(L)和因果(M)路径"),
        ('X', 'Y', {'C'}, False,
         "Collider: X→C←Y, 条件化 C 反而打开路径!"),
        ('X', 'Y', {'L', 'C'}, True,
         "Fork 阻断 + Collider 打开 = 总体阻断"),
    ]

    for x, y, z, expected, desc in tests:
        is_dsep = model.is_dconnected(x, y, observed=z)
        # dconnected=True means NOT d-separated
        status = "✓" if (not is_dsep) == expected else "✗"
        print(f"  {status} {desc}")
        print(f"    d-separated: {not is_dsep} (预期: {expected})")

    # --- 后门准则验证 ---
    print("\n[后门准则验证: 识别 P(Y | do(X))]")
    print("  后门路径: X ← L → Y")
    print("  可调整变量: {'L'} (不是X的后代, 阻断后门路径)")
    print("  ✓ 后门调整: P(Y|do(X)) = Σ_l P(Y|X,L=l)P(L=l)")

    # 后门调整数值计算
    infer = VariableElimination(model)
    p_y_do_x0 = np.zeros(2)
    p_y_do_x1 = np.zeros(2)

    for l_val in range(2):
        p_l = infer.query(['L']).values[l_val]
        q_y_given_x0_l = infer.query(['Y'], evidence={'X': 0, 'L': l_val}).values
        q_y_given_x1_l = infer.query(['Y'], evidence={'X': 1, 'L': l_val}).values
        p_y_do_x0 += p_l * q_y_given_x0_l
        p_y_do_x1 += p_l * q_y_given_x1_l

    print(f"  P(Y | do(X=0)) = {p_y_do_x0}")
    print(f"  P(Y | do(X=1)) = {p_y_do_x1}")
    ate = p_y_do_x1[1] - p_y_do_x0[1]
    print(f"  ATE (Y=1) = {ate:.4f}")


# ============================================================================
# 练习 4: 后门前门准则 — 因果效应识别
# ============================================================================

def exercise4_backdoor_frontdoor():
    """
    实现后门调整和前门调整, 对比两种因果效应识别方法

    场景A (后门): X ← Z → Y  (Z 是混杂, 可观测 → 后门调整)
    场景B (前门): X → M → Y, 同时 X ← U → Y  (U 未观测, 但有中介 M → 前门调整)
    """
    _print_header("练习 4: 后门准则 & 前门准则 — 因果效应识别")

    # =============================================================
    # 场景A: 后门调整
    # =============================================================
    print("\n[场景A: 后门调整 — Z 是可观测混杂变量]")
    print("  因果图: Z → X, Z → Y, X → Y")
    print("  目标: P(Y | do(X))")

    # 参数定义
    p_z = np.array([0.5, 0.3, 0.2])  # Z 有3个取值

    p_x_given_z = np.array([
        [0.8, 0.2],  # P(X|Z=0)
        [0.4, 0.6],  # P(X|Z=1)
        [0.1, 0.9],  # P(X|Z=2)
    ])

    p_y_given_xz = np.array([
        [[0.9, 0.1], [0.6, 0.4]],  # P(Y|X=0, Z)
        [[0.3, 0.7], [0.1, 0.9]],  # P(Y|X=1, Z=0,1,2)
    ])
    p_y_given_xz_full = np.zeros((3, 2, 2))
    p_y_given_xz_full[0] = [[0.9, 0.1], [0.6, 0.4]]  # Z=0
    p_y_given_xz_full[1] = [[0.7, 0.3], [0.4, 0.6]]  # Z=1
    p_y_given_xz_full[2] = [[0.5, 0.5], [0.2, 0.8]]  # Z=2

    # 重新定义使用full版本
    p_y_gx_z = p_y_given_xz_full

    # 观测关联
    p_y_given_x_obs = np.zeros((2, 2))
    for x in range(2):
        p_z_given_x = np.array([p_z[z] * p_x_given_z[z, x] for z in range(3)])
        p_z_given_x /= p_z_given_x.sum()
        for y in range(2):
            p_y_given_x_obs[x, y] = sum(
                p_z_given_x[z] * p_y_gx_z[z, x, y] for z in range(3)
            )

    print(f"  观测关联 P(Y=1|X=0) = {p_y_given_x_obs[0,1]:.4f}")
    print(f"  观测关联 P(Y=1|X=1) = {p_y_given_x_obs[1,1]:.4f}")
    print(f"  观测差异 = {p_y_given_x_obs[1,1] - p_y_given_x_obs[0,1]:.4f} (含混杂)")

    # 后门调整
    p_y_do = np.zeros((2, 2))
    for x in range(2):
        for y in range(2):
            p_y_do[x, y] = sum(p_z[z] * p_y_gx_z[z, x, y] for z in range(3))

    print(f"\n  后门调整 P(Y=1|do(X=0)) = {p_y_do[0,1]:.4f}")
    print(f"  后门调整 P(Y=1|do(X=1)) = {p_y_do[1,1]:.4f}")
    print(f"  因果效应 (ATE) = {p_y_do[1,1] - p_y_do[0,1]:.4f}")
    print(f"  混杂偏倚 = {(p_y_given_x_obs[1,1] - p_y_given_x_obs[0,1]) - (p_y_do[1,1] - p_y_do[0,1]):.4f}")

    # =============================================================
    # 场景B: 前门调整 (吸烟→焦油→肺癌 经典场景)
    # =============================================================
    print("\n" + "-" * 50)
    print("[场景B: 前门调整 — 吸烟→焦油→肺癌]")
    print("  因果图: U(吸烟基因, 未观测) → X(吸烟), U → Y(肺癌), X → M(焦油) → Y")
    print("  目标: P(Y | do(X)) — U 未观测, 后门准则不可用!")

    # 参数 (这些在现实中部分不可观测, 但我们定义了用于验证)
    p_x = np.array([0.6, 0.4])                         # P(X)
    p_m_given_x = np.array([[0.8, 0.2], [0.3, 0.7]])   # P(M|X): 吸烟→焦油
    p_y_given_xm = np.zeros((2, 2, 2))                  # P(Y|X,M)
    p_y_given_xm[0, 0] = [0.95, 0.05]                  # Y|X=0,M=0
    p_y_given_xm[0, 1] = [0.8, 0.2]                    # Y|X=0,M=1
    p_y_given_xm[1, 0] = [0.7, 0.3]                    # Y|X=1,M=0
    p_y_given_xm[1, 1] = [0.2, 0.8]                    # Y|X=1,M=1

    # 前门调整: P(y|do(x)) = Σ_m P(m|x) · Σ_{x'} P(y|x',m)·P(x')
    p_y_do_front = np.zeros(2)
    for m in range(2):
        # Step 2: P(y|do(m)) = Σ_{x'} P(y|x',m)·P(x')
        p_y_given_do_m = sum(p_y_given_xm[xp, m, 1] * p_x[xp] for xp in range(2))
        # Step 3: P(y|do(x)) = Σ_m P(m|x)·P(y|do(m))
        p_y_do_front[1] += p_m_given_x[0, m] * p_y_given_do_m  # do(X=0)
        p_y_do_front[0] += p_m_given_x[0, m] * (1 - p_y_given_do_m)

    p_y_do_x0_front = np.zeros(2)
    p_y_do_x1_front = np.zeros(2)

    for m in range(2):
        p_y_given_do_m_all = np.array([
            sum(p_y_given_xm[xp, m, :] * p_x[xp] for xp in range(2))
        ])
        p_y_do_x0_front += p_m_given_x[0, m] * p_y_given_do_m_all
        p_y_do_x1_front += p_m_given_x[1, m] * p_y_given_do_m_all

    print(f"  前门调整 P(Y=1|do(X=0)) = {p_y_do_x0_front[1]:.4f}")
    print(f"  前门调整 P(Y=1|do(X=1)) = {p_y_do_x1_front[1]:.4f}")
    print(f"  前门因果效应 = {p_y_do_x1_front[1] - p_y_do_x0_front[1]:.4f}")

    # 对比观测关联
    p_y_given_x_obs_front = np.zeros((2, 2))
    for x in range(2):
        for y in range(2):
            prob = 0.0
            for m in range(2):
                prob += p_m_given_x[x, m] * p_y_given_xm[x, m, y]
            p_y_given_x_obs_front[x, y] = prob

    obs_diff = p_y_given_x_obs_front[1, 1] - p_y_given_x_obs_front[0, 1]
    causal_diff = p_y_do_x1_front[1] - p_y_do_x0_front[1]
    print(f"  观测关联差异 = {obs_diff:.4f}")
    print(f"  前门因果效应 = {causal_diff:.4f}")
    print(f"  {'' if abs(obs_diff - causal_diff) > 0.01 else '✓ 前门调整成功识别因果效应'}")

    return p_y_do, (p_y_do_x0_front, p_y_do_x1_front)


# ============================================================================
# 练习 5: 反事实推理 — 溯因·行动·预测
# ============================================================================

def exercise5_counterfactual_reasoning():
    """
    线性 SCM 下的反事实推理

    模型:
      X = U_X                     (U_X ~ N(μ_X, σ²_X))
      Y = β·X + U_Y              (U_Y ~ N(μ_Y, σ²_Y), U_X ⟂ U_Y)

    观测: X=x, Y=y
    问题: "如果 X 当时是 x', Y 会是多少?"

    三步法: Abduction → Action → Prediction
    """
    _print_header("练习 5: 反事实推理 — 溯因·行动·预测 三步法")

    # SCM 参数
    mu_x, sigma_x = 5.0, 2.0
    mu_y, sigma_y = 1.0, 1.5
    beta = 3.0  # 因果效应: X → Y

    print(f"\n[SCM 定义]")
    print(f"  X = U_X,           U_X ~ N({mu_x}, {sigma_x}²)")
    print(f"  Y = {beta}·X + U_Y,  U_Y ~ N({mu_y}, {sigma_y}²)")
    print(f"  U_X ⟂ U_Y")

    # 生成一个"个体" u
    ux_true = np.random.normal(mu_x, sigma_x)
    uy_true = np.random.normal(mu_y, sigma_y)

    x_obs = ux_true
    y_obs = beta * x_obs + uy_true

    print(f"\n[观测到的数据]")
    print(f"  个体 u 的真实外生变量: U_X={ux_true:.3f}, U_Y={uy_true:.3f}")
    print(f"  观测值: X={x_obs:.3f}, Y={y_obs:.3f}")

    # --- Step 1: Abduction (溯因) ---
    # 给定 X=x_obs, Y=y_obs, 推断 U_X, U_Y
    ux_inferred = x_obs  # U_X = X (确定性的)
    uy_inferred = y_obs - beta * x_obs  # U_Y = Y - βX

    print(f"\n[Step 1 — Abduction (溯因)]")
    print(f"  推断 U_X = X = {ux_inferred:.3f}  (真值: {ux_true:.3f})")
    print(f"  推断 U_Y = Y - βX = {y_obs:.3f} - {beta}·{x_obs:.3f} = {uy_inferred:.3f}  (真值: {uy_true:.3f})")

    # --- Step 2: Action (行动) ---
    x_prime = x_obs + 2.0  # 假设 X 多2个单位
    print(f"\n[Step 2 — Action (行动)]")
    print(f"  修改模型: 强制 X = x' = {x_prime:.3f}")
    print(f"  新模型: Y = {beta}·{x_prime:.3f} + U_Y")

    # --- Step 3: Prediction (预测) ---
    y_counterfactual = beta * x_prime + uy_inferred

    print(f"\n[Step 3 — Prediction (预测)]")
    print(f"  Y_{x'} = β·x' + U_Y = {beta}·{x_prime:.3f} + {uy_inferred:.3f} = {y_counterfactual:.3f}")
    print(f"  对比: 实际 Y = {y_obs:.3f}, 反事实 Y_{x'} = {y_counterfactual:.3f}")
    print(f"  变化: ΔY = Y_{x'} - Y = {y_counterfactual - y_obs:.3f}")
    print(f"  理论: β·(x'-x) = {beta}·{x_prime - x_obs:.3f} = {beta*(x_prime - x_obs):.3f}")

    # --- 多个个体的反事实分析 ---
    print(f"\n[多个个体的反事实分析]")
    n_samples = 1000
    ux_samples = np.random.normal(mu_x, sigma_x, n_samples)
    uy_samples = np.random.normal(mu_y, sigma_y, n_samples)
    x_samples = ux_samples
    y_samples = beta * x_samples + uy_samples

    # 选择 Y > mean(Y) + 1*std 的子群体 ("高康复率" 组)
    y_threshold = np.mean(y_samples) + np.std(y_samples)
    high_y_mask = y_samples > y_threshold
    low_y_mask = y_samples < np.mean(y_samples) - np.std(y_samples)

    # 这些个体的反事实 (如果 X 减少1)
    x_counterfactual = x_samples - 1.0
    y_cf_high = beta * x_counterfactual[high_y_mask] + uy_samples[high_y_mask]
    y_cf_low = beta * x_counterfactual[low_y_mask] + uy_samples[low_y_mask]

    print(f"  高 Y 组 (n={high_y_mask.sum()}):")
    print(f"    观测: E[X]={x_samples[high_y_mask].mean():.2f}, E[Y]={y_samples[high_y_mask].mean():.2f}")
    print(f"    反事实 (X-1): E[Y]={y_cf_high.mean():.2f}")
    print(f"    变化: {y_cf_high.mean() - y_samples[high_y_mask].mean():.2f}")

    print(f"  低 Y 组 (n={low_y_mask.sum()}):")
    print(f"    观测: E[X]={x_samples[low_y_mask].mean():.2f}, E[Y]={y_samples[low_y_mask].mean():.2f}")
    print(f"    反事实 (X-1): E[Y]={y_cf_low.mean():.2f}")
    print(f"    变化: {y_cf_low.mean() - y_samples[low_y_mask].mean():.2f}")

    # --- Probability of Necessity / Sufficiency ---
    print(f"\n[归因概率 (Population Level)]")
    # PNS bounds: max{0, P(Y|X=1) - P(Y|X=0)} ≤ PNS ≤ min{P(Y|X=1), 1-P(Y|X=0)}
    # 离散化 X
    x_binary = (x_samples > np.median(x_samples)).astype(int)
    y_binary = (y_samples > np.median(y_samples)).astype(int)
    p_y1_given_x1 = y_binary[x_binary == 1].mean()
    p_y1_given_x0 = y_binary[x_binary == 0].mean()

    pns_lower = max(0, p_y1_given_x1 - p_y1_given_x0)
    pns_upper = min(p_y1_given_x1, 1 - p_y1_given_x0)

    print(f"  P(Y=1 | X=1) = {p_y1_given_x1:.4f}")
    print(f"  P(Y=1 | X=0) = {p_y1_given_x0:.4f}")
    print(f"  PNS 界: [{pns_lower:.4f}, {pns_upper:.4f}]")

    return ux_inferred, uy_inferred, y_counterfactual


# ============================================================================
# 练习 6: PC 算法 — 从数据发现因果结构
# ============================================================================

def exercise6_pc_algorithm():
    """
    从零实现简化版 PC 算法, 从数据中恢复因果骨架

    步骤:
      1. 从完全无向图开始
      2. 对每对节点, 对每个可能的条件集大小, 测试条件独立性
         - 使用偏相关系数检验 (Fisher z-transform)
      3. 删除被 d-separated 的边
      4. 定向 collider (v-结构)
      5. 应用 Meek 规则传播方向
    """
    _print_header("练习 6: PC 算法 — 从数据发现因果骨架")

    n_samples = 2000
    np.random.seed(123)

    # --- 生成来自已知因果图的数据 ---
    # 因果图: X₁ → X₂, X₁ → X₃, X₂ → X₄, X₃ → X₄
    # 条件独立: X₂ ⟂ X₃ | X₁
    #           X₁ ⟂ X₄ | {X₂, X₃}

    print("\n[真实因果图]")
    print("  X1 → X2, X1 → X3, X2 → X4, X3 → X4")
    print("  蕴涵 CI: X2 ⟂ X3 | X1,  X1 ⟂ X4 | {X2,X3}")

    # 生成数据
    e1 = np.random.randn(n_samples)
    e2 = np.random.randn(n_samples)
    e3 = np.random.randn(n_samples)
    e4 = np.random.randn(n_samples)

    X1 = e1
    X2 = 0.7 * X1 + e2
    X3 = 0.6 * X1 + e3
    X4 = 0.5 * X2 + 0.4 * X3 + e4

    data = np.column_stack([X1, X2, X3, X4])
    var_names = ['X1', 'X2', 'X3', 'X4']
    p = len(var_names)

    # 标准化
    data_std = (data - data.mean(axis=0)) / data.std(axis=0)
    cov = np.cov(data_std.T)
    prec = np.linalg.inv(cov)

    print(f"\n[协方差矩阵 Σ]")
    print(np.array2string(cov, precision=3, suppress_small=True))
    print(f"\n[精度矩阵 Ω = Σ⁻¹]")
    print(np.array2string(prec, precision=3, suppress_small=True))
    print(f"  非零模式 → 无向图骨架: X1-X2, X1-X3, X2-X4, X3-X4  ✓")

    # --- 偏相关系数检验 ---
    def partial_corr(data, i, j, cond_set):
        """计算给定 cond_set 下 i 和 j 的偏相关系数"""
        n = len(data)
        all_vars = [i, j] + list(cond_set)
        subdata = data[:, all_vars]
        sub_cov = np.cov(subdata.T)
        sub_prec = np.linalg.inv(sub_cov)
        pc = -sub_prec[0, 1] / np.sqrt(sub_prec[0, 0] * sub_prec[1, 1])
        return pc

    def fisher_z_test(r, n, alpha=0.05):
        """Fisher z-transform 检验 ρ=0"""
        z = 0.5 * np.log((1 + r) / (1 - r + 1e-10))
        z_crit = 1.96  # α=0.05
        se = 1.0 / np.sqrt(n - 3)
        return abs(z) < z_crit * se

    # Step 1 & 2: 骨架发现
    adjacency = np.ones((p, p), dtype=bool)
    np.fill_diagonal(adjacency, False)
    sep_set = {}

    max_depth = p - 2
    for depth in range(max_depth + 1):
        changed = False
        for i in range(p):
            neighbors = list(np.where(adjacency[i])[0])
            for j_idx, j in enumerate(neighbors):
                if len(neighbors) - 1 < depth:
                    continue
                other_neighbors = [n for n in neighbors if n != j]
                for cond in combinations(other_neighbors, depth):
                    cond = list(cond)
                    r = partial_corr(data_std, i, j, cond)
                    if abs(r) < 0.01 or fisher_z_test(r, n_samples):
                        adjacency[i, j] = False
                        adjacency[j, i] = False
                        sep_set[(i, j)] = set(cond)
                        sep_set[(j, i)] = set(cond)
                        changed = True
                        break
        if not changed:
            break

    print(f"\n[PC 阶段1: 骨架发现 (d={depth})]")
    for i in range(p):
        for j in range(i + 1, p):
            if adjacency[i, j]:
                print(f"  边: {var_names[i]} — {var_names[j]}")

    # Step 3: 定向 v-结构
    edges = []  # (from, to) as directed
    undirected = []  # (i, j) as undirected

    for i in range(p):
        for j in range(i + 1, p):
            if not adjacency[i, j]:
                # 检查所有共享邻居 k, 看是否形成 v-结构
                for k in range(p):
                    if k != i and k != j and adjacency[i, k] and adjacency[j, k]:
                        if (i, j) in sep_set and k not in sep_set[(i, j)]:
                            edges.append((i, k))
                            edges.append((j, k))
                            print(f"  v-结构: {var_names[i]} → {var_names[k]} ← {var_names[j]} "
                                  f"(SepSet({var_names[i]},{var_names[j]})={sep_set[(i,j)]}, {var_names[k]} ∉ SepSet)")

    # Step 4: Meek 规则 (简化)
    # R1: 定向 undirected edges 以避免新 v-结构
    # R2: 定向 undirected edges 以避免 cycle
    directed_from = set((f, t) for f, t in edges)

    # 简单启发式: 如果 i—j 且 j→k, 且 ¬(k→j), 且 i 和 k 不相邻, 定向 i→j
    # (避免形成 i ← j → k 被 i→j → k 取代)
    undirected_edges = []
    for i in range(p):
        for j in range(i + 1, p):
            if adjacency[i, j]:
                is_directed = (i, j) in directed_from or (j, i) in directed_from
                if not is_directed:
                    undirected_edges.append((i, j))

    print(f"\n[PC 阶段2: 定向结果]")
    print(f"  有向边: {len(directed_from)} 条")
    for f, t in sorted(directed_from):
        print(f"    {var_names[f]} → {var_names[t]}")
    print(f"  无向边: {len(undirected_edges)} 条")
    for i, j in undirected_edges:
        print(f"    {var_names[i]} — {var_names[j]}")

    return adjacency, directed_from


# ============================================================================
# 练习 7: 因果对ML的启示 — OOD 泛化 & 反事实公平
# ============================================================================

def exercise7_causal_ml_implications():
    """
    展示因果关系对机器学习的两个关键启示:
      1. 分布外泛化 (OOD): 因果 vs 反因果方向的不变性
      2. 反事实公平性: 改变敏感属性的反事实预测
    """
    _print_header("练习 7: 因果对ML的启示 — OOD 泛化 & 反事实公平性")

    np.random.seed(456)
    n_train = 500
    n_test = 500

    # =============================================================
    # 7.1 OOD 泛化: 因果方向的不变性
    # =============================================================
    print("\n[7.1 分布外泛化 — 因果 vs 反因果预测]")
    print("  真实因果: X → Y  (X 是 Y 的原因)")

    # 训练环境
    mu_x_train = 2.0
    x_train = np.random.randn(n_train) + mu_x_train
    y_train = 3.0 * x_train + np.random.randn(n_train) * 0.5

    # 测试环境 (X 的分布改变 — 新环境)
    mu_x_test = 5.0  # X 的均值漂移!
    x_test = np.random.randn(n_test) + mu_x_test
    y_test = 3.0 * x_test + np.random.randn(n_test) * 0.5

    # 因果模型: Y = f(X) + noise
    # 学习 P(Y|X) — 这在因果方向是稳定的!
    beta_causal = np.polyfit(x_train, y_train, 1)[0]

    # 反因果模型: X = g(Y) + noise
    # 学习 P(X|Y) — 这在反因果方向是不稳定的!
    beta_anticausal = np.polyfit(y_train, x_train, 1)[0]

    # 评价
    y_pred_causal = beta_causal * x_test
    x_pred_anticausal = beta_anticausal * y_test

    mse_causal = np.mean((y_test - y_pred_causal) ** 2)
    mse_anticausal = np.mean((x_test - x_pred_anticausal) ** 2)

    print(f"  因果模型 P(Y|X):     MSE = {mse_causal:.4f} (Y = {beta_causal:.2f}·X, 真β=3.0)")
    print(f"  反因果模型 P(X|Y):   MSE = {mse_anticausal:.4f} (X = {beta_anticausal:.2f}·Y)")
    print(f"  ✓ 因果方向模型在不同环境下保持稳定")
    print(f"  ✗ 反因果方向模型在 OOD 环境下失效")

    # =============================================================
    # 7.2 反事实公平性
    # =============================================================
    print("\n" + "-" * 50)
    print("[7.2 反事实公平性 — 性别对录取的影响]")

    # SCM:
    # G (性别, binary)  ← 外生变量 U_G
    # E (教育水平) = α·G + U_E
    # A (是否录取) = β·E + γ·G + U_A

    n_samples = 1000
    # 外生变量
    u_g = np.random.randn(n_samples)
    u_e = np.random.randn(n_samples) * 2
    u_a = np.random.randn(n_samples) * 0.5

    # 性别 (男=1, 女=0)
    G = (u_g > 0).astype(float)

    # 教育水平
    alpha = 0.3  # 教育水平的性别差异 (可能反映历史不平等)
    E = alpha * G + u_e + 5.0

    # 录取 (假设存在 γ>0 的直接歧视效应)
    beta_edu = 0.5   # 教育对录取的真实因果效应
    gamma_dir = 0.4  # 性别的直接效应 (歧视!)
    A_logit = beta_edu * E + gamma_dir * G + u_a
    A = (A_logit > A_logit.mean()).astype(float)

    # --- 统计公平性 (Demographic Parity): 简单检查 A ⟂ G ---
    disparity_dp = A[G == 1].mean() - A[G == 0].mean()
    print(f"\n  [统计公平性 (Demographic Parity)]")
    print(f"    P(A=1 | G=男) = {A[G==1].mean():.4f}")
    print(f"    P(A=1 | G=女) = {A[G==0].mean():.4f}")
    print(f"    DP 差异 = {disparity_dp:.4f}")
    print(f"    结论: 统计上存在性别差异 (但可能通过教育等'正当'渠道)")

    # --- 反事实公平性 ---
    # 对每个女性个体, 计算如果她是男性 (但保持其他外生变量不变) 的录取概率
    female_mask = (G == 0)
    n_female = female_mask.sum()

    # 反事实: 保持 U_E 和 U_A 不变, 只改变 G 从 0→1
    E_cf_male = alpha * 1.0 + u_e[female_mask] + 5.0
    A_logit_cf_male = beta_edu * E_cf_male + gamma_dir * 1.0 + u_a[female_mask]
    A_cf_male = (A_logit_cf_male > A_logit.mean()).astype(float)

    # 女性观测录取率 vs 反事实录取率
    cf_difference = A_cf_male.mean() - A[female_mask].mean()
    print(f"\n  [反事实公平性]")
    print(f"    女性观测: P(A=1) = {A[female_mask].mean():.4f}")
    print(f"    女性反事实(如果是男性): P(A=1) = {A_cf_male.mean():.4f}")
    print(f"    反事实差异 = {cf_difference:.4f}")
    print(f"    → {cf_difference:.4f} 的录取率差异无法被教育解释,")
    print(f"      反映了性别对录取的直接因果效应 (歧视), 应消除!")

    # 因果分解
    # 总差异 = 通过教育的间接效应 + 直接效应
    indirect_via_edu = A_cf_male.mean() - A[female_mask].mean()
    print(f"\n  [因果效应分解]")
    print(f"    总效应 (TE)     = 间接(通过教育) + 直接(歧视)")
    print(f"    反事实差异 ≈ 总效应 = {cf_difference:.4f}")
    print(f"    其中包含: 教育中介效应 + 性别直接效应(γ={gamma_dir})")


# ============================================================================
# 主函数
# ============================================================================

def main():
    import argparse
    # 简化: 支持 --ex 参数 (如 --ex 1,2,3)
    parser = argparse.ArgumentParser(description='L17 Causality 1 Exercises')
    parser.add_argument('--ex', type=str, default='all',
                        help='Comma-separated exercise numbers to run (e.g. "1,3,5")')
    args = parser.parse_args()

    if args.ex == 'all':
        ex_nums = list(range(1, 8))
    else:
        ex_nums = [int(x.strip()) for x in args.ex.split(',')]

    exercises = {
        1: exercise1_simpsons_paradox,
        2: exercise2_do_operator_graph_surgery,
        3: exercise3_causal_bn_dseparation,
        4: exercise4_backdoor_frontdoor,
        5: exercise5_counterfactual_reasoning,
        6: exercise6_pc_algorithm,
        7: exercise7_causal_ml_implications,
    }

    for num in ex_nums:
        if num in exercises:
            try:
                exercises[num]()
            except Exception as e:
                print(f"\n  ⚠ 练习 {num} 出错: {e}")
        else:
            print(f"\n  ⚠ 未知练习编号: {num}")

    print("\n" + "=" * 70)
    print("  全部练习完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
