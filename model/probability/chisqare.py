import pandas as pd

df = pd.read_csv('distribution.csv', sep=' ')

# Convert columns to numeric
for col in ['Process1', 'Process2', 'Process3', 'Process4']:
    df[col] = pd.to_numeric(df[col])

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

def create_intervals(data, n_intervals=8):
    """Create n equal intervals based on percentiles"""
    percentiles = np.linspace(0, 100, n_intervals + 1)
    intervals = []
    for i in range(len(percentiles)-1):
        lower = data.quantile(percentiles[i]/100)
        upper = data.quantile(percentiles[i+1]/100)
        if i == 0:
            lower = -float('inf')
        if i == len(percentiles)-2:
            upper = float('inf')
        intervals.append((lower, upper))
    return intervals

# Descriptive statistics
print(df[['Process1', 'Process2', 'Process3', 'Process4']].describe())

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

# Calculate intervals for each process
intervals_process1 = create_intervals(df['Process1'])
intervals_process2 = create_intervals(df['Process2'])
intervals_process3 = create_intervals(df['Process3'])
intervals_process4 = create_intervals(df['Process4'], n_intervals=4)

print("Process1 intervals:", intervals_process1)
print("Process2 intervals:", intervals_process2)
print("Process3 intervals:", intervals_process3)
print("Process4 intervals:", intervals_process4)

results = {}

for process in ['Process1', 'Process2', 'Process3', 'Process4']:
    observed_frequencies = []
    expected_frequencies = []

    if process == 'Process1':
        intervals = intervals_process1
        mu, sigma = stats.norm.fit(df['Process1'])
        fitted_distribution = stats.norm(loc=mu, scale=sigma)
    elif process == 'Process2':
        intervals = intervals_process2
        param = stats.expon.fit(df['Process2'])
        fitted_distribution = stats.expon(loc=param[0], scale=param[1])
    elif process == 'Process3':
        intervals = intervals_process3
        shape, loc, scale = stats.gamma.fit(df['Process3'])
        fitted_distribution = stats.gamma(a=shape, loc=loc, scale=scale)
    elif process == 'Process4':
        intervals = intervals_process4
        lambda_param = np.mean(df['Process4'])
        fitted_distribution = stats.poisson(mu=lambda_param)

    for interval in intervals:
        observed_count = len(df[(df[process] >= interval[0]) & (df[process] < interval[1])])
        observed_frequencies.append(observed_count)

        if process == 'Process4':
            if interval[0] == -float('inf'):
                lower_bound = df[process].min()
            else:
                lower_bound = interval[0]

            if interval[1] == float('inf'):
                upper_bound = df[process].max() + 1
            else:
                upper_bound = interval[1]

            expected_prob = fitted_distribution.pmf(np.arange(lower_bound, upper_bound))
            expected_prob = np.sum(expected_prob)
        else:
            if interval[0] == -float('inf'):
                expected_prob = fitted_distribution.cdf(interval[1])
            elif interval[1] == float('inf'):
                expected_prob = 1 - fitted_distribution.cdf(interval[0])
            else:
                expected_prob = fitted_distribution.cdf(interval[1]) - fitted_distribution.cdf(interval[0])
        expected_frequencies.append(expected_prob * len(df))

    total_observed = sum(observed_frequencies)
    total_expected = sum(expected_frequencies)
    if total_expected != 0:
      adjustment_factor = total_observed / total_expected
      expected_frequencies = [freq * adjustment_factor for freq in expected_frequencies]

    chi2_statistic, p_value = stats.chisquare(observed_frequencies, expected_frequencies)
    results[process] = {'chi2_statistic': chi2_statistic, 'p_value': p_value}

print(results)