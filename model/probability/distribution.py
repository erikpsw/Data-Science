import pandas as pd

df = pd.read_csv('distribution.csv', sep=' ')

# Process 1: the weight of a bottle leaving a filling process
# Process 2: the time between failure of a machine
# Process 3: the check-in time at an airport
# Process 4: the number of orders for a product received at a warehouse from a retail outlet

# Convert columns to numeric
for col in ['Process1', 'Process2', 'Process3', 'Process4']:
    df[col] = pd.to_numeric(df[col])

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

# Descriptive statistics
print(df[['Process1', 'Process2', 'Process3', 'Process4']].describe())

# Shapiro-Wilk test for each process
processes = ['Process1', 'Process2', 'Process3', 'Process4']
for process in processes:
    statistic, p_value = stats.shapiro(df[process])
    print(f'\nShapiro-Wilk test for {process}:')
    print(f'Statistic: {statistic:.4f}')
    print(f'p-value: {p_value:.4f}')

# Visualize the distribution of each process
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
sns.histplot(df['Process1'], kde=True)
plt.title('Process 1 Distribution')

plt.subplot(2, 2, 2)
sns.histplot(df['Process2'], kde=True)
plt.title('Process 2 Distribution')

plt.subplot(2, 2, 3)
sns.histplot(df['Process3'], kde=True)
plt.title('Process 3 Distribution')

plt.subplot(2, 2, 4)
sns.histplot(df['Process4'], kde=True)
plt.title('Process 4 Distribution')

plt.tight_layout()
plt.show()

# Distribution fitting for each process
plt.figure(figsize=(12, 10))

# Process 1: Normal distribution (weight of bottles)
plt.subplot(2, 2, 1)
sns.histplot(df['Process1'], kde=True, stat='density')
mu, sigma = stats.norm.fit(df['Process1'])
x = np.linspace(df['Process1'].min(), df['Process1'].max(), 100)
y = stats.norm.pdf(x, mu, sigma)
plt.plot(x, y, 'r-', label=f'Normal (μ={mu:.2f}, σ={sigma:.2f})')
plt.title('Process 1: Bottle Weight Distribution')
plt.legend()

# Process 2: Exponential distribution (time between failures)
plt.subplot(2, 2, 2)
sns.histplot(df['Process2'], kde=True, stat='density')
# Log-transform for fitting
param = stats.expon.fit(df['Process2'])
x = np.linspace(df['Process2'].min(), df['Process2'].max(), 100)
y = stats.expon.pdf(x, *param)
plt.plot(x, y, 'r-', label=f'Exponential (λ={1/param[1]:.4f})')
plt.title('Process 2: Time Between Failures')
plt.legend()

# Process 3: Gamma distribution (check-in time)
plt.subplot(2, 2, 3)
sns.histplot(df['Process3'], kde=True, stat='density')
shape, loc, scale = stats.gamma.fit(df['Process3'])
x = np.linspace(df['Process3'].min(), df['Process3'].max(), 100)
y = stats.gamma.pdf(x, shape, loc=loc, scale=scale)
plt.plot(x, y, 'r-', label=f'Gamma (α={shape:.2f}, β={1/scale:.2f})')
plt.title('Process 3: Check-in Time Distribution')
plt.legend()

# Process 4: Poisson distribution (number of orders)
plt.subplot(2, 2, 4)
sns.histplot(df['Process4'], kde=True, stat='density', discrete=True)
lambda_param = np.mean(df['Process4'])
x = np.arange(df['Process4'].min(), df['Process4'].max()+1)
y = stats.poisson.pmf(x, lambda_param)
plt.plot(x, y, 'ro-', label=f'Poisson (λ={lambda_param:.2f})')
plt.title('Process 4: Number of Orders Distribution')
plt.legend()

plt.tight_layout()
plt.show()
