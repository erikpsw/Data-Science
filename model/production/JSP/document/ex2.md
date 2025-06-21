# 柔性作业车间调度问题 (FJSP) 数学建模与理论分析

## 1. 问题定义

### 1.1 基本符号定义

**集合定义：**

* $J = \{J_1, J_2, ..., J_n\}$：工件集合
* $M = \{M_1, M_2, ..., M_m\}$：机器集合
* $O = \{O_{i,j} | i \in \{1,\dots,n\}, j \in \{1,\dots,n_i\}\}$：工序集合
* $P = \{P_1, P_2, ..., P_k\}$：产品类型集合

**参数说明：**

* $n$：工件总数
* $m$：机器总数
* $n_i$：工件 $J_i$ 的工序数量
* $O_{i,j}$：工件 $J_i$ 的第 $j$ 个工序
* $M_{i,j} \subseteq M$：工序 $O_{i,j}$ 的可选机器集合
* $p_{i,j,k}$：工序 $O_{i,j}$ 在机器 $M_k$ 上的加工时间
* $\tau_i \in P$：工件 $J_i$ 的产品类型

### 1.2 决策变量

**机器分配变量：**

$$
x_{i,j,k} =
\begin{cases}
1, & \text{若 } O_{i,j} \text{ 在机器 } M_k \text{ 上加工} \\
0, & \text{否则}
\end{cases}
$$

**工序排序变量：**

$$
y_{i,j,i',j'} =
\begin{cases}
1, & \text{若 } O_{i,j} \text{ 在 } O_{i',j'} \text{ 之前加工} \\
0, & \text{否则}
\end{cases}
$$

**时间变量：**

* $S_{i,j}$：开始时间
* $C_{i,j}$：完成时间
* $C_{\max}$：最大完工时间（Makespan）

---

## 2. 数学模型

### 2.1 目标函数

$$
\min C_{\max} = \min \max_{i,j} C_{i,j}
$$

### 2.2 约束条件

**机器分配约束：**

$$
\sum_{k \in M_{i,j}} x_{i,j,k} = 1, \quad \forall i,j
$$

**工件内部顺序约束：**

$$
C_{i,j} \leq S_{i,j+1}, \quad \forall i, j = 1,\dots,n_i - 1
$$

**机器容量约束：**

$$
S_{i',j'} \geq C_{i,j} \cdot \sum_{k} x_{i,j,k} \cdot x_{i',j',k} \cdot (1 - y_{i,j,i',j'})
$$

对于所有 $(i,j) \ne (i',j')$ 且存在公共机器 $k \in M_{i,j} \cap M_{i',j'}$

**工序完成时间计算：**

$$
C_{i,j} = S_{i,j} + \sum_k p_{i,j,k} \cdot x_{i,j,k} \cdot \alpha_{i,j}
$$

---

## 3. 同产品连续加工优化模型

### 3.1 连续加工判定函数

$$
\beta_{i,j,i',j',k} =
\begin{cases}
1, & \text{若满足连续加工条件} \\
0, & \text{否则}
\end{cases}
$$

条件成立当且仅当：

$$
x_{i,j,k} = x_{i',j',k} = 1,\quad \tau_i = \tau_{i'},\quad C_{i,j} = S_{i',j'},\quad y_{i,j,i',j'} = 1
$$

### 3.2 优化系数

$$
\alpha_{i,j} =
\begin{cases}
0.75, & \exists (i',j',k) \text{ 使 } \beta_{i',j',i,j,k} = 1 \\
1.0, & \text{否则}
\end{cases}
$$

### 3.3 优化效果量化

**节省时间总量：**

$$
T_{\text{saved}} = \sum_{i,j} \sum_k 0.25 \cdot p_{i,j,k} \cdot x_{i,j,k} \cdot (1 - \alpha_{i,j})
$$

**优化率：**

$$
\eta = \frac{T_{\text{saved}}}{\sum_{i,j} \sum_k p_{i,j,k} \cdot x_{i,j,k}} \times 100\%
$$

---

## 4. 遗传算法建模

### 4.1 染色体编码

$$
I = (X, Y)
$$

* $X = \{x_{i,j,k}\}$：机器分配矩阵
* $Y = \{y_{i,j,i',j'}\}$：工序顺序矩阵

### 4.2 适应度函数

$$
f(I) = \frac{1}{C_{\max}(I) + 1}
$$

### 4.3 选择概率（轮盘赌）

$$
P_s(I_i) = \frac{f(I_i)}{\sum_{j=1}^{|P|} f(I_j)}
$$

### 4.4 交叉操作

$$
X_{\text{child}}[i,j] =
\begin{cases}
X_1[i,j], & \text{if } rand() < 0.5 \\
X_2[i,j], & \text{otherwise}
\end{cases}
$$

$$
Y_{\text{child}} = \Phi(X_{\text{child}})
$$

### 4.5 变异操作

选中 $O_{i,j}$ 后：

$$
x'_{i,j,k} =
\begin{cases}
1, & k = k_{\text{new}} \\
0, & \text{otherwise}
\end{cases}
$$

---

## 5. 复杂度分析

### 5.1 时间复杂度

* Makespan计算：

$$
O\left(\sum_{i=1}^{n} n_i \cdot \log \left(\sum_{i=1}^{n} n_i \right)\right)
$$

* 遗传算法：

$$
O(G \cdot |P| \cdot \sum_{i=1}^{n} n_i \cdot m)
$$

### 5.2 空间复杂度

$$
O\left(|P| \cdot \left(\sum_{i=1}^{n} n_i \cdot m + \left(\sum_{i=1}^{n} n_i\right)^2\right)\right)
$$

---

## 6. 理论性质

### 6.1 可行性条件

若存在可行解，则：

$$
\sum_{k \in M_{i,j}} x_{i,j,k} = 1,\quad
C_{i,j} \geq S_{i,j} + \min_{k \in M_{i,j}} p_{i,j,k}
$$

### 6.2 调度上下界

**下界：**

$$
LB = \max \left\{
\max_i \sum_j \min_k p_{i,j,k},
\max_k \frac{\sum_{i,j: k \in M_{i,j}} p_{i,j,k}}{|M_{i,j}|}
\right\}
$$

**上界（考虑优化）：**

$$
UB = LB \cdot (1 + \epsilon) \cdot (1 - 0.25 \cdot \rho)
$$

### 6.3 收敛性分析

若满足：

1. 精英保留；
2. $p_m > 0$；
3. 保持种群多样性；

则：

$$
\lim_{t \to \infty} P(f_t = f^*) = 1
$$

---

## 7. 优化策略分析

### 7.1 聚类度量

$$
\gamma = \frac{\sum_{k} \sum_{t=1}^{T_k - 1} \delta(\tau(O_k^t), \tau(O_k^{t+1}))}{|\{(k,t) : t < T_k\}|}
$$

其中 $\delta(a,b) = 1$ 当 $a = b$，否则为 0。

### 7.2 优化潜力

$$
T_{\text{max\_saved}} = 0.25 \cdot \sum_{i,j} \min_k p_{i,j,k} \cdot \gamma
$$

$$
\eta_{\text{actual}} = \frac{T_{\text{saved}}}{T_{\text{max\_saved}}} \times 100\%
$$

---

## 8. 扩展模型

### 8.1 多目标优化

$$
\min \mathbf{F} =
\begin{pmatrix}
C_{\max} \\
\sum_{k=1}^{m} |U_k - \bar{U}| \\
\sum_{i=1}^{n} w_i \cdot C_i
\end{pmatrix}
$$

### 8.2 动态调度

引入时间窗口 $[t, t + \Delta t]$，调度变量变为时间函数：

$$
x_{i,j,k}(t),\quad y_{i,j,i',j'}(t),\quad S_{i,j}(t)
$$

---

## 9. 性能指标

### 9.1 调度性能

* Makespan效率：

$$
\eta_{\text{makespan}} = \frac{LB}{C_{\max}} \times 100\%
$$

* 平均机器利用率：

$$
U_{\text{avg}} = \frac{\sum_{k=1}^{m} \sum_{i,j} p_{i,j,k} \cdot x_{i,j,k}}{m \cdot C_{\max}}
$$

### 9.2 算法性能

* 收敛速度：

$$
v_{\text{conv}} = \frac{f_0 - f_{\infty}}{G_{\text{conv}}}
$$

* 解的稳定性：

$$
\sigma_{\text{stability}} = \sqrt{\frac{1}{R} \sum_{r=1}^{R} (C_{\max}^{(r)} - \bar{C}_{\max})^2}
$$

---

如需：

* 将此 Markdown 导出为 PDF、HTML、Word
* 增加图表（如 Gantt 图、收敛图）
* 附加代码实现（如遗传算法伪代码或 Python 实现）

欢迎随时告诉我。是否需要我直接导出为 `.md` 文件供你下载？
