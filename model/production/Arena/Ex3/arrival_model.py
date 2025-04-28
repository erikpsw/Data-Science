import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from scipy.stats import gamma

# Read the arrival time data
data = pd.read_csv('arival.csv', header=None, sep=' ').values.flatten()
data = data[~np.isnan(data)]

# Basic statistical analysis
mean_time = np.mean(data)
std_time = np.std(data)
min_time = np.min(data)
max_time = np.max(data)

print("\nBasic Statistics:")
print(f"Mean arrival time: {mean_time:.2f} minutes")
print(f"Standard deviation: {std_time:.2f} minutes")
print(f"Min time: {min_time:.2f} minutes")
print(f"Max time: {max_time:.2f} minutes")

# Fit Gamma distribution
shape, loc, scale = gamma.fit(data)
print(f"\nGamma Distribution Parameters:")
print(f"Shape (α): {shape:.4f}")
print(f"Location: {loc:.4f}")
print(f"Scale (β): {scale:.4f}")

# Create histogram with Gamma fit
plt.figure(figsize=(10, 6))
sns.histplot(data, bins=15, stat='density', alpha=0.5)

# Plot fitted Gamma distribution
x = np.linspace(min_time, max_time, 100)
pdf = gamma.pdf(x, shape, loc, scale)
plt.plot(x, pdf, label='Gamma Distribution Fit', color='red', alpha=0.7)

plt.title('Histogram of Arrival Times with Gamma Distribution Fit')
plt.xlabel('Time (minutes)')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)

# Perform Kolmogorov-Smirnov test
statistic, p_value = stats.kstest(data, 'gamma', args=(shape, loc, scale))
print(f"\nGoodness-of-fit Test (Kolmogorov-Smirnov):")
print(f"KS statistic: {statistic:.4f}")
print(f"p-value: {p_value:.4f}")

plt.show()

# Q-Q plot
plt.figure(figsize=(8, 6))
stats.probplot(data, dist='gamma', sparams=(shape,), plot=plt)
plt.title('Gamma Q-Q Plot')
plt.grid(True)
plt.show()

# Calculate percentiles for simulation input
percentiles = [25, 50, 75, 90]
empirical_percentiles = np.percentile(data, percentiles)
theoretical_percentiles = gamma.ppf(np.array(percentiles)/100, shape, loc, scale)

print("\nPercentile Analysis:")
for i, p in enumerate(percentiles):
    print(f"{p}th Percentile:")
    print(f"  Empirical: {empirical_percentiles[i]:.2f} minutes")
    print(f"  Theoretical: {theoretical_percentiles[i]:.2f} minutes")