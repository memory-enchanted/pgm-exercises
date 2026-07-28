# CMU 10-708 概率图模型 (PGM) 课后练习

基于 **CMU 10-708 Probabilistic Graphical Models**（Eric Xing 教授）L1-L14 课程的练习集，覆盖十二大模块：

- **表示 (Representation)** — L1-L3：图模型如何编码联合分布
- **精确推理：变量消除 (VE)** — L4：一次回答一个查询
- **参数估计 (Parameter Estimation)** — L5：MLE、GLM、EM、K-Means
- **序列模型：HMM & CRF** — L6：Forward-Backward / Viterbi 算法的 VE/BP 视角
- **近似推断：变分推断 I (VI)** — L7：Mean-Field & CAVI
- **近似推断：变分推断 II (VI)** — L8：SVI, BBVI, Wake-Sleep, VAE
- **近似推断：蒙特卡洛 (MCMC)** — L9：Rejection, Importance, MH, Gibbs
- **进阶 MCMC** — L10：HMC, Slice Sampling, Parallel Tempering, AIS
- **深度学习基础** — L11：梯度下降、反向传播、偏差-方差、正则化
- **深度生成模型 I (DGM I)** — L12：RBM、DBN、DBM、Contrastive Divergence
- **深度生成模型 II (DGM II)** — L13：VAE、GAN、Normalizing Flow、Diffusion
- **深度序列模型** — L14：CNN、RNN、LSTM/GRU、Attention、Transformer

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
| `08_L5_concept_review.md` | L5 参数估计概念体系梳理 | L5 |
| `09_pe_exercises.ipynb` / `.py` | 参数估计 5 步代码练习 (MLE/IRLS/K-Means/EM) | L5 |
| `10_pe_practice_problems.md` | L5 课后练习题（附答案） | L5 |
| `11_L6_concept_review.md` | L6 HMM & CRF 概念体系梳理 | L6 |
| `12_hmm_crf_exercises.ipynb` / `.py` | HMM & CRF 5 步代码练习 | L6 |
| `13_hmm_crf_practice_problem.md` | L6 课后练习题（附答案） | L6 |
| `14_L7_concept_review.md` | L7 变分推断 I 概念体系梳理 | L7 |
| `15_vi_exercises.ipynb` / `.py` | 变分推断 I 5 步代码练习 | L7 |
| `16_vi_practice_problems.md` | L7 课后练习题（附答案） | L7 |
| `17_L8_concept_review.md` | L8 变分推断 II 概念体系梳理 | L8 |
| `18_vi2_exercises.ipynb` / `.py` | 变分推断 II 5 步代码练习 | L8 |
| `19_vi2_practice_problems.md` | L8 课后练习题（附答案） | L8 |
| `20_L9_concept_review.md` | L9 蒙特卡洛方法概念体系梳理 | L9 |
| `21_mc_exercises.ipynb` / `.py` | 蒙特卡洛方法 5 步代码练习 | L9 |
| `22_mc_practice_problems.md` | L9 课后练习题（附答案） | L9 |
| `23_L10_concept_review.md` | L10 进阶 MCMC 概念体系梳理 | L10 |
| `24_advanced_mcmc_exercises.ipynb` / `.py` | 进阶 MCMC 5 步代码练习 | L10 |
| `25_advanced_mcmc_practice_problems.md` | L10 课后练习题（附答案） | L10 |
| `26_L11_concept_review.md` | L11 深度学习基础概念体系梳理 | L11 |
| `27_dl_foundations_exercises.ipynb` / `.py` | 深度学习基础 5 步代码练习 | L11 |
| `28_dl_foundations_practice_problems.md` | L11 课后练习题（附答案） | L11 |
| `29_L12_concept_review.md` | L12 深度生成模型 I 概念体系梳理 | L12 |
| `30_dgm1_exercises.ipynb` / `.py` | 深度生成模型 I 5 步代码练习 (RBM/CD/DBN/Score Matching) | L12 |
| `31_dgm1_practice_problems.md` | L12 课后练习题（附答案） | L12 |
| `32_L13_concept_review.md` | L13 深度生成模型 II 概念体系梳理 | L13 |
| `33_dgm2_exercises.ipynb` / `.py` | 深度生成模型 II 5 步代码练习 (VAE/GAN/Flow/Diffusion) | L13 |
| `34_dgm2_practice_problems.md` | L13 课后练习题（附答案） | L13 |
| `35_L14_concept_review.md` | L14 深度序列模型概念体系梳理 | L14 |
| `36_sequence_models_exercises.ipynb` / `.py` | 深度序列模型 6 步代码练习 (CNN/RNN/LSTM/Attn/Transformer) | L14 |
| `37_sequence_models_practice_problems.md` | L14 课后练习题（附答案） | L14 |
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
pip install networkx matplotlib numpy pandas scipy scikit-learn
```

### 运行方式

**Jupyter Notebook（推荐）**：直接在 IDE 或 Jupyter Lab 中打开 `.ipynb` 文件，按顺序执行每个 cell。

**纯 Python 脚本**：

```bash
# L1-L3: 表示
python 01_d_separation.py
python 02_markov_properties.py

# L4: 变量消除
python 06_ve_exercises.py              # 全部练习
python 06_ve_exercises.py --ex 1       # 指定练习 (1-5)

# L5: 参数估计
python 09_pe_exercises.py
python 09_pe_exercises.py --ex 1

# L6: HMM & CRF
python 12_hmm_crf_exercises.py
python 12_hmm_crf_exercises.py --ex 1

# L7: 变分推断 I
python 15_vi_exercises.py
python 15_vi_exercises.py --ex 1

# L8: 变分推断 II
python 18_vi2_exercises.py
python 18_vi2_exercises.py --ex 1

# L9: 蒙特卡洛方法
python 21_mc_exercises.py
python 21_mc_exercises.py --ex 1

# L10: 进阶 MCMC
python 24_advanced_mcmc_exercises.py
python 24_advanced_mcmc_exercises.py --ex 1

# L11: 深度学习基础
python 27_dl_foundations_exercises.py
python 27_dl_foundations_exercises.py --ex 1

# L12: 深度生成模型 I
python 30_dgm1_exercises.py
python 30_dgm1_exercises.py --ex 1

# L13: 深度生成模型 II
python 33_dgm2_exercises.py
python 33_dgm2_exercises.py --ex 1

# L14: 深度序列模型
python 36_sequence_models_exercises.py
python 36_sequence_models_exercises.py --ex 1
```

### 转换工具

修改 `.py` 后重新生成 `.ipynb`：

```bash
python convert_to_ipynb.py
```

---

## 📖 学习路线

```
模块一：表示 (Representation)
───────────────────────────────
04_L1_L3_concept_review.md   ← 先看概念框架
          │
          ▼
01_d_separation.ipynb        ← d-分离（条件独立性的核心判据）
          │
          ▼
02_markov_properties.ipynb   ← 三层马尔可夫性质的等价关系
          │
          ▼
03_practice_problems.md      ← 综合练习 + 答案自测

模块二：精确推理 — 变量消除 (L4)
─────────────────────────────────
05_L4_concept_review.md      ← VE 在推理中的定位
          │
          ▼
06_ve_exercises.ipynb        ← 5 个练习逐步深入 VE
          │
          ▼
07_ve_practice_problems.md   ← 手算因子乘积与消元顺序

模块三：参数估计 (L5)
────────────────────────
08_L5_concept_review.md      ← MLE → 指数族 → GLM → EM 的主线
          │
          ▼
09_pe_exercises.ipynb        ← 5 个练习：MLE → IRLS → K-Means → EM → 对比
          │
          ▼
10_pe_practice_problems.md   ← 课后练习 + 答案自测

模块四：序列模型 — HMM & CRF (L6)
─────────────────────────────────
11_L6_concept_review.md      ← HMM/CRF 概念 (Forward/Backward/Viterbi 的 VE/BP 视角)
          │
          ▼
12_hmm_crf_exercises.ipynb   ← 5 个练习: Forward → Viterbi → HMM-as-BN → FB=BP → CRF
          │
          ▼
13_hmm_crf_practice_problem.md ← 课后练习 + 答案自测

模块五：近似推断 — 变分推断 I (L7)
───────────────────────────────────
14_L7_concept_review.md      ← 精确 → 近似的范式转换
          │
          ▼
15_vi_exercises.ipynb        ← 5 个练习: KL → ELBO → CAVI → vs VE → VI=BP
          │
          ▼
16_vi_practice_problems.md   ← 课后练习 + 答案自测

模块六：近似推断 — 变分推断 II (L8)
────────────────────────────────────
17_L8_concept_review.md      ← VI 进阶：大数据、非共轭、深度学习
          │
          ▼
18_vi2_exercises.ipynb       ← 5 个练习: Wake-Sleep → SVI → Reparam → BBVI → VAE
          │
          ▼
19_vi2_practice_problems.md  ← 课后练习 + 答案自测

模块七：近似推断 — 蒙特卡洛方法 (L9)
─────────────────────────────────────
20_L9_concept_review.md      ← VI vs MCMC：两条近似路线的对比
          │
          ▼
21_mc_exercises.ipynb        ← 5 个练习: Rejection → Importance → MH → Gibbs → pgmpy
          │
          ▼
22_mc_practice_problems.md   ← 课后练习 + 答案自测

模块八：进阶 MCMC (L10)
─────────────────────────
23_L10_concept_review.md     ← 基础 MCMC 的局限 → 进阶方法的动机
          │
          ▼
24_advanced_mcmc_exercises.ipynb ← 5 个练习: Slice → HMC → Tempering → AIS → 诊断
          │
          ▼
25_advanced_mcmc_practice_problems.md ← 课后练习 + 答案自测

模块九：深度学习基础 (L11)
───────────────────────────
26_L11_concept_review.md     ← PGM → DL 的桥梁：MLE, 梯度, 正则化
          │
          ▼
27_dl_foundations_exercises.ipynb ← 5 个练习: SGD → 反向传播 → Bias-Var → 正则化 → MLP
          │
          ▼
28_dl_foundations_practice_problems.md ← 课后练习 + 答案自测

模块十：深度生成模型 I (L12)
─────────────────────────────
29_L12_concept_review.md     ← RBM → DBN → DBM 的层次化生成路线
          │
          ▼
30_dgm1_exercises.ipynb      ← 5 个练习: RBM → CD → 特征学习 → DBN → Score Matching
          │
          ▼
31_dgm1_practice_problems.md ← 课后练习 + 答案自测

模块十一：深度生成模型 II (L13)
───────────────────────────────
32_L13_concept_review.md     ← VAE, GAN, Flow, Diffusion 统一视角
          │
          ▼
33_dgm2_exercises.ipynb      ← 5 个练习: VAE → GAN → Flow → AR → Diffusion
          │
          ▼
34_dgm2_practice_problems.md ← 课后练习 + 答案自测

模块十二：深度序列模型 (L14)
─────────────────────────────
35_L14_concept_review.md     ← CNN → RNN → LSTM → Attention → Transformer
          │
          ▼
36_sequence_models_exercises.ipynb ← 6 个练习: 1D CNN → RNN → LSTM → Attn → Multi-Head → Transformer
          │
          ▼
37_sequence_models_practice_problems.md ← 课后练习 + 答案自测
```

---

## 📚 内容概要

### 模块一：表示 (L1-L3)

**01 — d-分离 (d-separation)**
- 三种基本图结构：链式 (Chain)、分叉 (Fork)、汇聚 (Collider)
- 路径活跃性判断规则，使用 pgmpy 进行 d-分离查询
- **核心结论**：d-sep(X, Y | Z) ⇒ X ⟂ Y | Z

**02 — 马尔可夫性质 (Markov Properties)**
- 分解 → 局部 / 全局马尔可夫性质 → 成对马尔可夫性质
- 三种性质的层级关系与等价条件

**03 — 课后练习**
- 图模型定义、因子分解、I-map 等基础概念，每题标注难度 ⭐，附详细解析

**04 — 概念体系梳理**
- PGM 三大支柱：表示 → 推理 → 学习，适合考前复习

### 模块二：精确推理 — 变量消除 (L4)

**05 — L4 概念体系梳理**
- VE 在推理中的位置；因子乘积、边缘化、消元顺序、诱导图

**06 — 变量消除代码练习（5 个练习）**
1. 因子基础 — 手写因子乘积与边缘化
2. VE 步步跟踪 — 可视化每一步消除
3. 消元顺序对决 — 好顺序 vs 坏顺序
4. 诱导图 & 填充边 — 理解复杂度根源
5. VE = 消息传递 — 消元结果就是消息

**07 — L4 课后练习**
- 🟢 基础：因子乘积手算、消元过程 Walk-through
- 🟡 进阶：消元顺序对比、诱导图宽度分析
- 🔴 挑战：最小填充推理

### 模块三：参数估计 (L5)

**08 — L5 概念体系梳理**
- MLE → 指数族 → GLM → EM 的主线
- 多元高斯 MLE、IRLS 迭代加权最小二乘
- K-Means = EM 的特例 (硬分配 + 球形协方差的极限)

**09 — 参数估计代码练习（5 个练习）**
1. MLE 手算 — 一元 & 多元高斯，验证充分统计量
2. IRLS — 迭代重加权最小二乘拟合逻辑回归
3. K-Means — 手写硬聚类，追踪损失下降
4. EM for GMM — 高斯混合模型，E步(软分配) + M步(加权MLE)
5. K-Means vs EM — 硬分配 vs 软分配，理解 K-Means 的隐含假设

**10 — L5 课后练习**
- 🟢 基础：Bernoulli/Gaussian MLE、GLM 组件识别、EM 性质
- 🟡 进阶：IRLS 权重分析、EM 的 ELBO 推导、GMM 完整推导
- 🔴 挑战：截断高斯 EM、IRLS vs 梯度下降对比

### 模块四：序列模型 — HMM & CRF (L6)

**11 — L6 概念体系梳理**
- HMM 定义、Forward/Backward/Viterbi 算法
- 从 VE/BP 视角重新理解：Forward = VE on chain, Forward-Backward = Sum-Product BP
- CRF 的全局归一化 vs HMM 的局部归一化、Label Bias 问题

**12 — HMM & CRF 代码练习（5 个练习）**
1. Forward 算法手写 — 逐步追踪 α 消息（纯 numpy）
2. Viterbi 算法 — MAP 解码 + 回溯，对比边际解码
3. HMM 作为 BayesianNetwork — 用 pgmpy VE/BP 做滤波、平滑、MAP 解码
4. Forward-Backward = Sum-Product BP — α 和 β 就是链上的 BP 消息
5. CRF 势函数 — 演示 Label Bias，对比局部 vs 全局归一化

**13 — L6 课后练习**
- 🟢 基础：HMM 三要素、Forward α 手算、Backward β 手算
- 🟡 进阶：Forward-Backward = Sum-Product BP 证明、Viterbi 手算与回溯
- 🔴 挑战：Label Bias 深入分析、二阶 HMM treewidth 推导

### 模块五：近似推断 — 变分推断 I (L7)

**14 — L7 概念体系梳理**
- 精确 → 近似的范式转换：当 treewidth 过大时，把推断变为优化问题
- Mean-Field 假设、ELBO 分解、CAVI 坐标上升
- VI 定点方程 = BP 消息（树上的等价性）

**15 — 变分推断 I 代码练习（5 个练习）**
1. KL 散度手写 — 两个离散分布间的 KL，验证不对称性
2. ELBO 分解验证 — log P(X) = ELBO + KL，逐个验证
3. Mean-Field CAVI — 2 变量模型，坐标上升从零实现
4. Mean-Field VI for Bayesian Network — 与 VE 精确解对比
5. VI 定点方程 = BP 消息 — 树上的等价性

**16 — L7 课后练习**
- 🟢 基础：KL 散度手算、ELBO 分解、Mean-Field CAVI 推导
- 🟡 进阶：CAVI 更新公式证明、与 Gibbs Sampling 的关系
- 🔴 挑战：非共轭模型的 VI、结构化变分族

### 模块六：近似推断 — 变分推断 II (L8)

**17 — L8 概念体系梳理**
- CAVI 的限制：需要全量数据、条件共轭、无法扩展到深度模型
- SVI（随机变分推断）、BBVI（黑盒变分推断）、Wake-Sleep 算法
- Reparameterization Trick → VAE：VI 与深度学习融合的标志性成果

**18 — 变分推断 II 代码练习（5 个练习）**
1. Wake-Sleep 算法 — Bernoulli 混合模型上的交替训练
2. SVI 随机变分推断 — mini-batch 自然梯度 vs 全量 CAVI
3. Reparameterization Trick — 方差对比实验
4. BBVI + Control Variate — 降低 score function 梯度方差
5. 简易 VAE — 纯 numpy 实现的编码器-解码器

**19 — L8 课后练习**
- 🟢 基础：SVI vs CAVI 对比、Reparameterization Trick 原理
- 🟡 进阶：ELBO 梯度推导、VAE 损失函数
- 🔴 挑战：IWAE、β-VAE、Control Variate 设计

### 模块七：近似推断 — 蒙特卡洛方法 (L9)

**20 — L9 概念体系梳理**
- VI vs MCMC：优化 vs 采样，永远有偏 vs 渐近精确
- Rejection Sampling、Importance Sampling、MCMC 基本原理
- Metropolis-Hastings、Gibbs Sampling、收敛诊断

**21 — 蒙特卡洛代码练习（5 个练习）**
1. Rejection Sampling — 从混合高斯中采样，观察接受率
2. Importance Sampling — 加权样本，对比不同 proposal 的 ESS
3. Metropolis-Hastings — 2D 相关高斯，trace plot，收敛分析
4. Gibbs Sampling — 二元高斯，精确 full conditional，对比 MH
5. MCMC for Bayesian Network — pgmpy GibbsSampling vs VE

**22 — L9 课后练习**
- 🟢 基础：Rejection Sampling 接受率、Importance Weight 归一化
- 🟡 进阶：MH proposal 对比、Gibbs 条件分布推导
- 🔴 挑战：MCMC 收敛诊断（R-hat）、Hamiltonian Monte Carlo 原理

### 模块八：进阶 MCMC (L10)

**23 — L10 概念体系梳理**
- 基础 MCMC → 进阶 MCMC 的演进：RW-MH 在高维、强相关、多峰分布中效率极差
- HMC（哈密尔顿蒙特卡洛）：梯度引导的高效采样
- Slice Sampling：免 proposal 的自适应采样
- Parallel Tempering：多温度链克服多峰分布
- AIS（退火重要性采样）：估计 partition function

**24 — 进阶 MCMC 代码练习（5 个练习）**
1. Slice Sampling — 免 proposal 的自适应采样，100% 接受率
2. Hamiltonian Monte Carlo — 梯度引导的高效采样，对比 RW-MH
3. Parallel Tempering — 多温度链克服多峰分布
4. Annealed Importance Sampling — 估计 partition function
5. MCMC 收敛诊断 — R-hat, ESS, trace plot, MCSE

**25 — L10 课后练习**
- 🟢 基础：HMC Hamiltonian 构造、Slice 原理
- 🟡 进阶：HMC vs RW-MH 效率分析、Parallel Tempering 温度选择
- 🔴 挑战：NUTS 原理、AIS 权重推导

### 模块九：深度学习基础 (L11)

**26 — L11 概念体系梳理**
- PGM → DL 的桥梁：深度学习 = 用梯度下降在大规模神经网络上做 MLE/MAP
- 反向传播、SGD 及其变体（Momentum、Adam）
- 偏差-方差分解、正则化（L2、Dropout）
- PGM 的统计概念在 DL 中无处不在

**27 — 深度学习基础代码练习（5 个练习）**
1. 梯度下降变体 — SGD, Momentum, Adam 在 2D 损失面上的对比
2. 反向传播手写 — 2 层 MLP 的完整前向 + 反向，验证数值梯度
3. 偏差-方差分解 — 多项式回归，展示 Bias² + Var + Noise
4. 正则化 — L2, Dropout 抑制过拟合的效果对比
5. 神经网络从零 — 2 层 MLP 训练，决策边界可视化

**28 — L11 课后练习**
- 🟢 基础：损失函数的统计对应（MSE ↔ Gaussian, CE ↔ Categorical）
- 🟡 进阶：反向传播手动推导、偏差-方差权衡分析
- 🔴 挑战：Adam 的偏差修正、Dropout 的集成解释

### 模块十：深度生成模型 I (L12)

**29 — L12 概念体系梳理**
- PGM + DL 融合的第一代成果：RBM → DBN → DBM
- 能量函数、Block Gibbs Sampling、Contrastive Divergence
- Greedy layer-wise pretraining 的动机与局限

**30 — 深度生成模型 I 代码练习（5 个练习）**
1. RBM 从零实现 — 能量函数、Block Gibbs、条件分布、生成样本
2. Contrastive Divergence — CD-1 vs CD-5 vs CD-20，重构质量对比
3. RBM 特征学习 — 16→8 RBM，可视化权重作为"特征检测器"
4. 深度信念网络 — greedy layer-wise 逐层预训练，层次化表示
5. Score Matching — 在简单 Gaussian 上对比 SM vs MLE

**31 — L12 课后练习**
- 🟢 基础：RBM 能量与条件分布、Gibbs 采样步骤
- 🟡 进阶：CD-k 的梯度推导、DBN 逐层训练原理
- 🔴 挑战：Score Matching 与 CD 的理论联系、DBM 的联合训练

### 模块十一：深度生成模型 II (L13)

**32 — L13 概念体系梳理**
- 第二代深度生成模型的统一视角：不需要能量函数/partition function
- VAE：ELBO + Reparameterization Trick
- GAN：Minimax 博弈、最优判别器、JSD 解释
- Normalizing Flow：可逆变换 + Change of Variables
- Diffusion：前向加噪 + 反向去噪 = 随机微分方程

**33 — 深度生成模型 II 代码练习（5 个练习）**
1. VAE — 2D 隐空间插值与生成
2. GAN — 对抗训练，1D 数据分布学习
3. Normalizing Flow — RealNVP 风格可逆变换
4. 自回归模型 — MADE 的 masked 连接
5. 去噪扩散模型 — 简化的 1D DDPM

**34 — L13 课后练习**
- 🟢 基础：GAN minimax objective、最优 D 推导、VAE ELBO
- 🟡 进阶：Flow 的 Jacobian 计算、Diffusion 的 noise schedule
- 🔴 挑战：Wasserstein GAN、DDPM 的简化损失推导

### 模块十二：深度序列模型 (L14)

**35 — L14 概念体系梳理**
- 序列建模的演进：CNN (局部感受野) → RNN (循环状态) → Attention (全局加权) → Transformer (纯注意力)
- 长距离依赖和并行化两大核心难题的逐步解决
- Self-Attention、Multi-Head Attention、Positional Encoding

**36 — 深度序列模型代码练习（6 个练习）**
1. 1D 因果卷积 — 时序预测
2. Vanilla RNN 从零 — 字符级语言模型
3. LSTM 从零 — 门控机制手写（遗忘门、输入门、输出门）
4. Scaled Dot-Product Attention — 手算验证
5. Multi-Head Self-Attention 实现
6. Transformer Encoder Block — 组装完整编码器块

**37 — L14 课后练习**
- 🟢 基础：1D 卷积手算、RNN 隐藏状态更新
- 🟡 进阶：LSTM 门控公式、Attention 权重计算
- 🔴 挑战：Multi-Head 的维度分析、Positional Encoding 的设计空间

---

## 🔗 参考资料

- [CMU 10-708 课程主页](https://www.cs.cmu.edu/~epxing/Class/10708-20/)
- [Koller & Friedman 教材](https://mitpress.mit.edu/9780262013192/probabilistic-graphical-models/)
- [pgmpy 文档](https://pgmpy.org/)

---

## 📄 License

MIT
