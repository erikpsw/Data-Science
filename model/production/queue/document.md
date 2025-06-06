https://www.cnblogs.com/haohai9309/p/18416440

## 二、排队系统绩效指标

### 主要指标定义

| 指标 | 符号 | 定义 |
|------|------|------|
| 系统空闲概率 | $P_0$ | 系统中没有顾客的概率，即所有服务台都空闲 |
| 系统中有n个顾客的概率 | $P_n$ | 系统中恰好有n个顾客的概率，反映系统在不同状态下的可能性 |
| 平均队列长度 | $L_q$ | 排队系统中平均排队的顾客数，不包括正在接受服务的顾客 |
| 系统中的平均顾客数 | $L_s$ | 系统中包括排队和正在接受服务的顾客的平均数量 |
| 顾客的平均等待时间 | $W_q$ | 顾客在系统中排队的平均等待时间，不包括服务时间 |
| 顾客的平均系统时间 | $W_s$ | 顾客在系统中的总时间，包括等待时间和服务时间 |

### M/M/1 与 M/M/c 系统对比

| 指标 | M/M/1 系统 | M/M/c 系统 |
|------|------------|------------|
| **到达过程** | 顾客到达服从泊松分布，平均到达率为 $\lambda$ | 顾客到达服从泊松分布，平均到达率为 $\lambda$ |
| **服务过程** | 服务时间服从指数分布，服务率为 $\mu$ | 服务时间服从指数分布，服务率为 $\mu$ |
| **服务台数量** | 1个服务台 | $c$ 个服务台 |
| **系统空闲概率** $P_0$ | $P_0 = 1 - \rho$，其中 $\rho = \frac{\lambda}{\mu}$ | $P_0 = \left[ \sum_{n=0}^{c-1} \frac{(\lambda/\mu)^n}{n!} + \frac{(\lambda/\mu)^c}{c! (1 - \rho)} \right]^{-1}$，其中 $\rho = \frac{\lambda}{c\mu}$ |
| **系统中有n个顾客的概率** $P_n$ | $P_n = (1 - \rho) \rho^n$ | 对于 $n < c$：$P_n = \frac{(\lambda/\mu)^n}{n!} P_0$<br>对于 $n \geq c$：$P_n = \frac{(\lambda/\mu)^n}{c^{n-c} \cdot c!} P_0 \rho^{n-c}$ |
| **平均队列长度** $L_q$ | $L_q = \frac{\rho^2}{1 - \rho}$ | $L_q = P_0 \frac{(\lambda/\mu)^c \rho}{c! (1 - \rho)^2}$ |
| **系统中的平均顾客数** $L_s$ | $L_s = \frac{\rho}{1 - \rho}$ | $L_s = L_q + \frac{\lambda}{\mu}$ |
| **顾客的平均等待时间** $W_q$ | $W_q = \frac{\rho}{\mu (1 - \rho)}$ | $W_q = \frac{L_q}{\lambda}$ |
| **顾客的平均系统时间** $W_s$ | $W_s = \frac{1}{\mu - \lambda}$ | $W_s = W_q + \frac{1}{\mu}$ |

通过这些指标，可以有效评估排队系统的服务效率、顾客的等待情况以及系统的稳定性。

## 三、M/M/c 队列完整公式集

### 参数定义

设定：
- $\lambda$：顾客到达率（人/分钟）
- $\mu$：每个服务台的服务率（人/分钟）
- $c$：服务台数量
- $\rho = \frac{\lambda}{c \mu}$：系统利用率（必须 $\rho < 1$ 才稳定）
- $R = \frac{\lambda}{\mu}$：总服务需求

### 核心公式

#### 1. 系统空闲概率 $P_0$

$$P_0 = \left[ \sum_{n=0}^{c-1} \frac{R^n}{n!} + \frac{R^c}{c!} \cdot \frac{1}{1 - \rho} \right]^{-1}$$

#### 2. 系统中有n个顾客的概率 $P_n$

对于 $n \leq c$：
$$P_n = \frac{R^n}{n!} P_0$$

对于 $n > c$：
$$P_n = \frac{R^c}{c!} \rho^{n-c} P_0$$

#### 3. 平均队列长度 $L_q$

$$L_q = P_0 \frac{R^c \rho}{c! (1 - \rho)^2}$$

#### 4. 系统中的平均顾客数 $L_s$

$$L_s = L_q + R$$

#### 5. 顾客的平均等待时间 $W_q$

$$W_q = \frac{L_q}{\lambda}$$

#### 6. 顾客的平均系统时间 $W_s$

$$W_s = W_q + \frac{1}{\mu}$$

### Little定律验证

这些公式满足Little定律：
- $L_s = \lambda W_s$
- $L_q = \lambda W_q$

### 手动计算步骤

1. **验证稳定性**：确保 $\rho = \frac{\lambda}{c\mu} < 1$
2. **计算 $R$ 和 $\rho$**：$R = \frac{\lambda}{\mu}$，$\rho = \frac{\lambda}{c\mu}$
3. **计算 $P_0$**：使用求和公式
4. **计算 $L_q$**：平均队列长度
5. **计算其他指标**：依次计算 $L_s$、$W_q$、$W_s$

这些公式可用于与SimPy仿真结果进行对比验证。