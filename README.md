# CMU 10-708 概率图模型 (PGM) 课后练习

基于 **CMU 10-708 Probabilistic Graphical Models**（Eric Xing 教授）L1-L3 课程的练习题集，覆盖概率图模型的**表示 (Representation)** 理论基础。

> 配套视频：[BV1tX4y1371G](https://www.bilibili.com/video/BV1tX4y1371G/)
>
> 核心教材：Koller & Friedman, *Probabilistic Graphical Models: Principles and Techniques*, Ch.1-4

---

## 📁 文件结构

| 文件 | 内容 | 类型 |
|------|------|------|
| `01_d_separation.ipynb` / `.py` | d-分离概念与图结构练习 | Notebook / 脚本 |
| `02_markov_properties.ipynb` / `.py` | 局部 & 全局马尔可夫性质练习 | Notebook / 脚本 |
| `03_practice_problems.md` | L1-L3 课后选择题 & 简答题（附答案） | Markdown |
| `04_L1_L3_concept_review.md` | L1-L3 完整概念体系梳理 | Markdown |
| `convert_to_ipynb.py` | `.py` → `.ipynb` 转换工具 | Python |

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
```

`.py` 文件包含完整注释与交互式练习，所有代码块在文件中顺序排列，可直接运行。

### 转换工具

如需修改 `.py` 文件后重新生成 `.ipynb`：

```bash
python convert_to_ipynb.py
```

---

## 📖 学习路线

```
04_L1_L3_concept_review.md   ← 先看：建立整体概念框架
          │
          ▼
01_d_separation.ipynb        ← 核心技能：判断条件独立性
          │
          ▼
02_markov_properties.ipynb   ← 进阶：理解三层马尔可夫性质
          │
          ▼
03_practice_problems.md      ← 检验：综合练习 + 答案自测
```

---

## 📚 内容概要

### 01 — d-分离 (d-separation)

- 三种基本图结构：链式 (Chain)、分叉 (Fork)、汇聚 (Collider)
- 路径活跃性判断规则
- 使用 pgmpy 库进行 d-分离查询
- **核心结论**：d-sep(X, Y | Z) ⇒ X ⟂ Y | Z（全局马尔可夫性质）

### 02 — 马尔可夫性质 (Markov Properties)

- 分解 (Factorization) → 局部 / 全局马尔可夫性质 → 成对马尔可夫性质
- 三种性质的层级关系与等价条件
- 从图结构推导条件独立性的完整流程

### 03 — 课后练习题集

- 基础概念（图模型定义、因子分解、I-map）
- 有向图与无向图的区别
- 每题标注难度 ⭐，附详细答案解析

### 04 — 概念体系梳理

- PGM 三大支柱：表示 → 推理 → 学习
- L1-L3 知识点索引表
- 适用于考前复习或快速查阅

---

## 🔗 参考资料

- [CMU 10-708 课程主页](https://www.cs.cmu.edu/~epxing/Class/10708-20/)
- [Koller & Friedman 教材](https://mitpress.mit.edu/9780262013192/probabilistic-graphical-models/)
- [pgmpy 文档](https://pgmpy.org/)

---

## 📄 License

MIT
