# Reinforcement Learning — Code Implementation (Chapters 4–7)

本项目系统复现强化学习课程（赵世钰 · 西湖大学）Chapter 4–7 的核心算法（DP、MC、SA、TD）。
后续将继续扩展至 Chapter 8–10（函数逼近、策略梯度、Actor‑Critic 等）。

项目特点：

- 自定义 GridWorld 环境
- 完整的 各章算法实现（VI/PI/MC/SA/TD/Q‑learning）
- 统一 TD 框架（SARSA / Expected SARSA / n‑step SARSA / Q‑learning）
- 日志系统（Python logging + TensorBoard + Timing）
- 简洁模块化设计，易于扩展
- 支持策略图与值函数可视化

## ⚙️ How to Run
安装依赖
```shell
pip install -r requirements.txt
```
运行某一个章节的实验：
```shell
python main/chapter_4_1_value_iteration.py
python main/chapter_5_3_mc_epsilon_greedy.py
python main/chapter_7_1_sarsa.py
```

---

## 📂 Project Structure

```
reinforcement_learning/
│
├── logs/                                  # 运行日志 / TensorBoard 输出 / 时间统计
│
├── main/                                  # 各章节可运行脚本（入口）
│   ├── chapter_4_1_value_iteration.py
│   ├── chapter_4_2_policy_iteration.py
│   ├── chapter_4_3_truncated_policy_iteration.py
│   ├── chapter_5_1_mc_basic.py
│   ├── chapter_5_2_mc_exploring_starts.py
│   ├── chapter_5_3_mc_epsilon_greedy.py
│   ├── chapter_6_1_robbins_monro.py
│   ├── chapter_6_2_gd_variants.py
│   ├── chapter_7_1_sarsa.py
│   ├── chapter_7_2_expected_sarsa.py
│   ├── chapter_7_3_nstep_sarsa.py
│   ├── chapter_7_4_q_learning_on_policy.py
│   └── chapter_7_5_q_learning_off_policy.py
│
├── source/
│   ├── algorithms/
│   │   ├── vp_planner.py                  # V & P：VI / PI / Truncated PI
│   │   ├── mc_planner.py                  # MC：Basic / ES / ε-greedy
│   │   ├── sa_planner.py                  # SA：RM / GD / SGD / BGD / MBGD
│   │   └── td_planner.py                  # TD：SARSA / Expected SARSA / n-Step SARSA / (on & off) Q-learning
│   │
│   ├── domain_object/
│   │   ├── action.py                      # Action 枚举（UP/DOWN/LEFT/RIGHT/STAY）
│   │   └── transition.py                  # Transition 数据结构
│   │
│   ├── utils/
│   │   ├── mdp_ops.py                     # V & P 用 Q/V/backup 工具
│   │   ├── policy_ops.py                  # MC 通用策略函数
│   │   ├── sa_schedules.py                # SA 相关函数
│   │   ├── logger_manager.py              # 日志管理（logging + TensorBoard）
│   │   ├── timing.py                      # 代码执行时间统计装饰器
│   │   └── render.py                      # 网格策略可视化
│   │
│   ├── grid_word.py                       # 网格世界
│   └── sa_expression.py                   # SA表达式
│   
└── test/                                  # 测试目录
```
---



## 🧱 GridWorld Environment

`grid_world.py` 定义了可配置的 MDP 网格环境：

- 任意尺寸 `(height, width)`
- forbidden states（奖励 -10，可设为吸收态）
- target state（奖励 +1，吸收态）
- 五种动作：上 / 下 / 左 / 右 / 原地
- 支持：
  - `step()` 用于 MC 采样
  - `get_P()` 生成 Gym-style 转移矩阵，用于 VI/PI

奖励模型、转移概率均可自定义。

---

## 🔷 Chapter 4 — Value & Policy Iteration 

V & P 相关算法位于：

`source/algorithms/vp_planner.py`

提供三种经典 V & P  方法：

### **1. Value Iteration**
入口：`main/chapter_4_1_value_iteration.py`

- 基于 Bellman Optimality  
- 迭代计算 $V(s)$，每次使用 $\max_a q_k(s,a)$  
- 输出最优策略 $π^* $ 与值函数 $V^*$

### **2. Policy Iteration**
入口：`main/chapter_4_2_policy_iteration.py`

流程：

1. Policy Evaluation（完整求解 $V_π$）
2. Policy Improvement（贪心改进）
3. 直到策略稳定

### **3. Truncated Policy Iteration**
入口：`main/chapter_4_3_truncated_policy_iteration.py`

- 仅执行 **有限次评估 sweep**
- 更适用于大规模 MDP
- 介于 PI 与 VI 之间的折衷算法

---

## 🔶 Chapter 5 — Monte‑Carlo (Model-Free)

MC 相关算法位于：
`source/algorithms/mc_planner.py`

支持 MC 三件套：

### **1. MC Basic**
入口：`main/chapter_5_1_mc_basic.py`

- 对每个 `(s, a)` 重复采样 episode  
- 平均回报估计 $q(s,a)$  
- 再执行贪心策略改进

### **2. MC Exploring Starts (ES)**
入口：`main/chapter_5_2_mc_exploring_starts.py`

- 每个 episode 随机选择起始 `(s0, a0)`
- 尽量保证所有 `(s,a)` 都能被探索
- 比 MC Basic 收敛快，效率高

### **3. MC ε‑Greedy（On‑policy）**
入口：`main/chapter_5_3_mc_epsilon_greedy.py`

- 不需要保证所有 `(s,a)` 都能被探索
- 行为策略 = 目标策略 = $ε$‑greedy(Q)
- 可配置 $ε$ decay  


---

## 🟣 Chapter 6 — Stochastic Approximation & SGD

SA 相关算法位于：`source/algorithms/sa_planner.py`  
支持两大范式：**Robbins–Monro 随机逼近**（求根）与 **SGD 及其变体**（优化）。

核心抽象（`source/sa_expression.py`）：

- **RootFindingOracle**：封装带噪声的观测函数 $ĝ(w,η) = g(w) + η$，用于 RM 算法求根 $g(w)=0 $ 
- **MinimizationOracle**：封装随机目标 $f(w,X)$ 与梯度估计，用于 SGD 类算法最小化 $J(w)=E[f(w,X)]$

步长调度（`source/utils/sa_schedules.py`）：

- **Robbins–Monro 步长**：$a_k = a_0 / k^β$，满足收敛充分条件 $（0.5 &lt; β ≤ 1 时 Σa_k=∞ 且 Σa_k²&lt;∞）$
- **固定学习率**：用于 GD 对比实验
- **级数诊断**：自动验证步长条件是否满足

### **1. Robbins–Monro 算法**
入口：`main/chapter_6_1_robbins_monro.py`

求解方程 g(w)=0 的随机迭代算法：$w_{k+1} = w_k - a_k · ĝ(w_k)$


- 支持多维/标量根查找
- 可配置高斯噪声 $η \sim N(0, σ²)$ 模拟真实观测噪声
- 可选投影操作保证迭代有界
- 示例：求解 $tanh(w-1)=0$，真根 $w^*=1$

收敛条件（充分条件）：
- $g(w)$ 满足 Lipschitz 条件且梯度有界$（0 < c1 ≤ ||∇g|| ≤ c2）$
- 步长满足 $Σa_k=∞, Σa_k²<∞$（默认 $β=1.0$）
- 噪声零均值、二阶矩有界

### **2. GD 算法（BGD / MBGD / SGD）**
入口：`main/chapter_6_2_gd_variants.py`

针对最小化问题 $J(w)=E[f(w,X)]$，实现四种梯度方法：

| 方法       | 批次大小                  | 步长策略  | 特点              |
|----------|-----------------------|-------|-----------------|
| **GD**   | 全量数据                  | 固定 lr | 确定性梯度，仅作对比基准    |
| **BGD**  | 全量数据                  | RM 步长 | 每轮遍历全数据集，理论保证收敛 |
| **MBGD** | batch_size (e.g., 64) | RM 步长 | 小批量随机梯度，实用折中    |
| **SGD**  | 1                     | RM 步长 | 单样本更新，方差大但计算快   |

**统一收敛框架**：
- 除 GD 使用固定学习率外，BGD/MBGD/SGD 均采用 RM 步长（$a_k = a_0/k^β$）
- 在满足标准假设（凸性/平滑性/步长条件）时，$w_k$ 几乎必然收敛到 $∇J(w)=0$ 的根
- 支持投影操作处理约束优化

**示例任务**：均值估计  
最小化 $J(w) = E[\frac{1}{2}||w-X||²]$，其中 $X \sim N(μ,σ²)$，最优解 $w^*=μ$。通过不同 batch size 对比收敛速度与稳定性。


---
## 🟦 Chapter 7 — Temporal‑Difference (TD) Learning
TD 学习是强化学习中最重要的在线估计方法。
本项目提供 统一 TD 控制框架（TDPlanner），支持 4 种典型 TD 算法：

- SARSA（One‑step, on‑policy）
- Expected SARSA（On‑policy）
- n‑step SARSA（前视 n 步，on‑policy）
- Q‑learning（one‑step, off‑/on‑policy）

所有代码统一位于：`source/algorithms/td_planner.py`，以下为各子模块说明。

### 7.1 ⭐ SARSA（On‑policy）
入口脚本：`main/chapter_7_1_sarsa.py`
 

行为策略 = 目标策略 = $ε$‑greedy(Q)

每一步更新：
$Q(s_t,a_t)←Q(s_t,a_t)+α[r_{t+1}+\gamma·Q(s_{t+1},a_{t+1})−Q(s_t,a_t)]$ 

- 每次更新后对 当前状态 立即进行一次 $ε$‑greedy 政策改进
- 支持 $ε$ 衰减（exploration annealing）
- 支持 TensorBoard 记录：
  - episode return
  - |TD‑error|
  - Q(s) 最大值

### 7.2 ⭐ Expected SARSA（期望 SARSA）

入口脚本：`main/chapter_7_2_expected_sarsa.py`

Expected SARSA 目标为：
$G_t = r_{t+1} + \gamma · \mathbb{E}_{a'\sim \pi} [Q(s_{t+1}, a')]$

优点：
- variance 更低
- 收敛更稳定

本代码中 Expected SARSA 完全遵循该公式实现。

### 7.3 ⭐ n‑step SARSA（前视 n 步）
入口脚本：`main/chapter_7_3_nstep_sarsa.py`

本项目实现了 教材标准的“前视 n 步 + 逐片段（segment）更新”方法：

1. 对当前状态 $s_t$ 生成一个 最多 n 步的 SARSA 片段（终止就提前截断）
2. 计算前视目标：

   - 若未终止且 $k=n$，则 $G_t^{(n)} = \sum_{i=1}^k\gamma^{i-1}·r_{t+i} + \gamma^n Q(s_{t+n}, q_{t+n})$
   - 若在中途终止，则  $G_t^{(n)} = \sum_{i=1}^k\gamma^{i-1}·r_{t+i}$

3. 用该 $G_t^{(n)}$ 更新 $Q(s_t, a_t)$
4. 对 $s_t$ 做一次 $ε$‑greedy 政策更新
5. 环境向前推进一格，从下一个状态继续滚动生成下一片段

注：SARSA是 $n=1$ 的特例

### 7.4 ⭐ Q‑learning（On‑policy 版本）
入口脚本：`main/chapter_7_4_q_learning_on_policy.py`

On‑policy 形式的 Q‑learning 仍采用 $ε$‑greedy 行为策略，但目标项为：

  $G_t = r_{t+1} + \gamma · \max_{a'} Q(s_{t+1}, a')$

更新只影响当前 $(s_t,a_t)$，并立即对状态 $s_t$ 做一次 ε 政策改进（保持 on‑policy）。

### 7.5 ⭐ Q‑learning（Off‑policy 版本）

入口脚本：`main/chapter_7_5_q_learning_off_policy.py`

行为策略（exploring policy）使用较大的 ε： behavior_epsilon = 0.3

目标策略（评估策略）使用较小的 ε： target_epsilon = 0.05

并保持：
- 行为策略用于采样
- 目标策略用于 $\max_aQ$ 的更新

### 7.6 📌 统一 TD 框架（TDPlanner）

TDPlanner 提供以下统一能力：

- 环境交互（env.step）
- 采样一集（episode）或若干步
- ε‑greedy / greedy 策略改进
- 全章节统一日志接口（TensorBoard）
- n‑步与一‑步 SARSA/Q‑learning 共用数据结构

可以直接在任何 GridWorld 上调用：
```python
  planner = TDPlanner(env, TDConfig(...))
  Q, pi = planner.sarsa(...)
  Q, pi = planner.expected_sarsa(...)
  Q, pi = planner.n_step_sarsa(n=3, ...)
  Q, pi = planner.q_learning_on_policy(...)
  Q, pi = planner.q_learning_off_policy(...)
```
---
## 📊 Logging / TensorBoard / Timing

项目提供完整日志支持：

### ✔ Python Logging  
日志输出到：`logs/run.log`，由：`utils/logger_manager.py`统一管理。

---

### ✔ TensorBoard 可视化

可视化内容包括：

- episode returns  
- epsilon 衰减  
- 最大 Q 变化  
- MC/V&P/RM/GD 收敛趋势  

运行： 
```shell
tensorboard --logdir logs/
```

---

### ✔ 时间统计（Timing）

`timing.py` 提供：`@record_time_decorator("task_name")` 自动记录每一段代码的运行时间至log


---

## ⏳ To Be Continued (Chapters 8–10)
本仓库仍在持续开发，未来将加入 Chapter 8-10 的部分 （⏳ TODO）
- Chapter 8：Value Function Approximation（Linear / NN）
- Chapter 9：Policy Gradient Methods（REINFORCE / Baseline）
- Chapter 10：Actor–Critic（A2C / n‑step AC 等）

## 🙌 Acknowledgement
本项目由 Zhiying Chen 主导开发， 算法与代码设计由 M365 Copilot 协助完善。