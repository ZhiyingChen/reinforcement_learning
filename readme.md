# Reinforcement Learning — Code Implementation (Chapters 4–5)

本项目旨在系统复现强化学习课程（赵世钰 · 西湖大学）中第 4–5 章的主要算法。  
后续将继续扩展至 Chapter 6–10（TD、Q-learning、Function Approximation、Policy Gradient 等）。

项目特点：

- 自定义 **GridWorld 环境**
- 完整的 **DP + MC 两大类算法框架**
- **日志系统（Python logging + TensorBoard + Timing）**
- 清晰的模块化代码结构，便于扩展
- 支持策略可视化、状态价值可视化

## ⚙️ How to Run
安装依赖
```shell
pip install -r requirements.txt
```
运行某一个章节的实验：
```shell
python main/chapter_4_1_value_iteration.py
python main/chapter_5_3_mc_epsilon_greedy.py
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
│   └── chapter_5_3_mc_epsilon_greedy.py
│
├── source/
│   ├── algorithms/
│   │   ├── dp_planner.py                  # DP：VI / PI / Truncated PI
│   │   └── mc_planner.py                  # MC：Basic / ES / ε-greedy
│   │
│   ├── domain_object/
│   │   ├── action.py                      # Action 枚举（UP/DOWN/LEFT/RIGHT/STAY）
│   │   └── transition.py                  # Transition 数据结构
│   │
│   ├── utils/
│   │   ├── grid_world.py                  # GridWorld 环境
│   │   ├── mdp_ops.py                     # DP 用 Q/V/backup 工具
│   │   ├── policy_ops.py                  # MC/DP 通用策略函数
│   │   ├── logger_manager.py              # 日志管理（logging + TensorBoard）
│   │   ├── timing.py                      # 代码执行时间统计装饰器
│   │   └── render.py                      # 网格策略可视化
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
  - `get_P()` 生成 Gym-style MDP 动力学，用于 DP（VI/PI）

奖励模型、转移概率均可自定义。

---

## 🔷 Chapter 4 — Value & Policy Iteration 

V & P 相关算法位于：

`source/algorithms/dp_planner.py`

提供三种经典 V & P  方法：

### **1. Value Iteration**
入口：`main/chapter_4_1_value_iteration.py`

- 基于 Bellman Optimality  
- 迭代计算 V(s)，每次使用 `max_a q_k(s,a)`  
- 输出最优策略 π\* 与值函数 V\*

### **2. Policy Iteration**
入口：`main/chapter_4_2_policy_iteration.py`

流程：

1. Policy Evaluation（完整求解 V^π）
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

- 对每个 (s, a) 重复采样 episode  
- 平均回报估计 q(s,a)  
- 再执行贪心策略改进

### **2. MC Exploring Starts (ES)**
入口：`main/chapter_5_2_mc_exploring_starts.py`

- 每个 episode 随机选择起始 `(s0, a0)`
- 尽量保证所有 (s,a) 都能被探索
- 比 MC Basic 收敛快，效率高

### **3. MC ε‑Greedy（On‑policy）**
入口：`main/chapter_5_3_mc_epsilon_greedy.py`

- 不需要保证所有 (s,a) 都能被探索
- 行为策略 = 目标策略 = ε‑greedy(Q)
- 可配置 epsilon decay  


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
- MC/DP 收敛趋势  

运行： 
```shell
tensorboard --logdir logs/
```

---

### ✔ 时间统计（Timing）

`timing.py` 提供：`@record_time_decorator("task_name")` 自动记录每一段代码的运行时间至log


---

## 🚀 To Be Continued (Chapters 6–10)
本仓库仍在持续开发，未来将加入 Chapter 6-10 的部分 （⏳ TODO）

## 🙌 Acknowledgement
本项目由 Zhiying Chen 主导开发， 算法与代码设计由 M365 Copilot 协助完善。