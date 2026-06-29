# CMU 10-708 概率图模型 (PGM) 课后练习

基于 **CMU 10-708 Probabilistic Graphical Models**（Eric Xing 教授）L1-L4 课程的练习集，覆盖两大模块：

- **表示 (Representation)** — L1-L3：图模型如何编码联合分布
- **精确推理 (Exact Inference)** — L4：变量消除算法

> 配套视频：[BV1tX4y1371G](https://www.bilibili.com/video/BV1tX4y1371G/)
>
> 核心教材：Koller & Friedman, *Probabilistic Graphical Models: Principles and Techniques*

---

## 📁 文件结构

| 文件 | 内容 | 对应课程 |
|------|------|----------|
| `04_L1_L3_concept_review.md` | L1-L3 完整概念体系梳理 | L1-L3 |
| `01_d_separation.ipynb` / `.py` | d-分离理论与代码练习 | L1-L3 |
| `02_markov_properties.ipynb` / `.py` | 局部 & 全局马尔可夫性质练习 | L1-L3 |
| `03_practice_problems.md` | L1-L3 课后选择题 & 简答题（附答案） | L1-L3 |
| `05_L4_concept_review.md` | L4 变量消除概念体系梳理 | L4 |
| `06_ve_exercises.ipynb` / `.py` | 变量消除算法 5 步代码练习 | L4 |
| `07_ve_practice_problems.md` | L4 课后练习题（附答案） | L4 |
| `convert_to_ipynb.py` | `.py` → `.ipynb` 转换工具 | — |
| `requirements.txt` | Python 依赖清单 | — |

---

## 🚀 快速开始

### 环境依赖

**方式一：一键安装（推荐）**

```bash
pip install -r requirements.txt
```

**方式二：手动安装**

```bash
conda install -c anaconda pgmpy
pip install networkx matplotlib numpy
```

### 运行方式

**Jupyter Notebook（推荐）**：直接在 IDE 或 Jupyter Lab 中打开 `.ipynb` 文件，按顺序执行每个 cell。

**纯 Python 脚本**：

```bash
python 01_d_separation.py
python 02_markov_properties.py
python 06_ve_exercises.py              # 运行全部练习
python 06_ve_exercises.py --ex 1       # 只运行指定练习 (1-5)
```

`.py` 文件包含完整注释与交互式练习，所有代码块顺序排列，可直接运行。

### 转换工具

如需修改 `.py` 文件后重新生成 `.ipynb`：

```bash
python convert_to_ipynb.py
```

---

## 📖 学习路线

```
模块一：表示 (Representation)
───────────────────────────────
04_L1_L3_concept_review.md   ← 先看：建立整体概念框架
          │
          ▼
01_d_separation.ipynb        ← 核心技能：d-分离判断条件独立性
          │
          ▼
02_markov_properties.ipynb   ← 进阶：三层马尔可夫性质的等价关系
          │
          ▼
03_practice_problems.md      ← 检验：综合练习 + 答案自测

模块二：精确推理 (Exact Inference)
───────────────────────────────────
05_L4_concept_review.md      ← 先看：理解 VE 在 PGM 中的定位
          │
          ▼
06_ve_exercises.ipynb        ← 5 个练习逐步深入变量消除
          │
          ▼
07_ve_practice_problems.md   ← 检验：手算因子乘积与消元顺序
```

---

## 📚 内容概要

### 模块一：表示 (L1-L3)

**01 — d-分离 (d-separation)**
- 三种基本图结构：链式 (Chain)、分叉 (Fork)、汇聚 (Collider)
- 路径活跃性判断规则
- 使用 pgmpy 进行 d-分离查询
- **核心结论**：d-sep(X, Y | Z) ⇒ X ⟂ Y | Z

**02 — 马尔可夫性质 (Markov Properties)**
- 分解 → 局部 / 全局马尔可夫性质 → 成对马尔可夫性质
- 三种性质的层级关系与等价条件
- 从图结构推导条件独立性的完整流程

**03 — 课后练习**
- 图模型定义、因子分解、I-map 等基础概念
- 有向图与无向图的对比
- 每题标注难度 ⭐，附详细解析

**04 — 概念体系梳理**
- PGM 三大支柱：表示 → 推理 → 学习
- L1-L3 知识点索引表，适合考前复习或快速查阅

### 模块二：精确推理 (L4)

**05 — L4 概念体系梳理**
- 变量消除 (Variable Elimination) 在推理中的位置
- 因子乘积、边缘化、消元顺序、诱导图等核心概念
- 精确推理 vs 近似推理的取舍

**06 — 变量消除代码练习（5 个练习）**
1. 因子基础 — 手写因子乘积与边缘化
2. VE 步步跟踪 — 可视化每一步消除
3. 消元顺序对决 — 好顺序 vs 坏顺序对比计算量
4. 诱导图 & 填充边 — 理解复杂度根源
5. VE = 消息传递 — 消元结果就是消息

**07 — L4 课后练习**
- 🟢 基础：因子乘积手算、消元过程 Walk-through
- 🟡 进阶：消元顺序对比、诱导图宽度分析
- 🔴 挑战：最小填充推理

---

## 🔗 参考资料

- [CMU 10-708 课程主页](https://www.cs.cmu.edu/~epxing/Class/10708-20/)
- [Koller & Friedman 教材](https://mitpress.mit.edu/9780262013192/probabilistic-graphical-models/)
- [pgmpy 文档](https://pgmpy.org/)

---

## 📄 License

MIT
