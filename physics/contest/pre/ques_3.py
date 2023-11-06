import numpy as np
from scipy.stats import rv_continuous

G = 6.67e-11
M_sun = 1.989e30
c = 3e8
year_in_seconds = 3.1536e7  
thresholds = [10**i for i in range(9, 0, -1)]  
N = 1000000  
np.random.seed(0)  

class PDF_e0(rv_continuous):
    def _pdf(self, e0):
        return (2/np.sqrt(np.pi)) * e0**(-3/2) * (1-e0)**(-1/4)

class PDF_a0(rv_continuous):
    def _pdf(self, a0, eta):
        return (15/(8*np.sqrt(np.pi))) * eta**(-18/37) * a0**(-7/4)

pdf_e0 = PDF_e0(a=0, b=1)  
pdf_a0 = PDF_a0(a=0, b=np.inf)  
e0_samples = pdf_e0.rvs(size=N)
eta = np.random.uniform(0, 1, N)  
a0_samples = pdf_a0.rvs(eta, size=N)

t = (5 * c**5 * (1 - e0_samples**2)**(7/2) * a0_samples**4) / (
    512 * G**3 * M_sun**3 * (1 + (73/24) * e0_samples**2 + (37/96) * e0_samples**4)
)
t_years = t / year_in_seconds  

probabilities = [(t_years < threshold).mean() for threshold in thresholds]

for threshold, prob in zip(thresholds, probabilities):
    print(f"合并时间小于 {threshold} 年的概率为：{prob:.6f}")