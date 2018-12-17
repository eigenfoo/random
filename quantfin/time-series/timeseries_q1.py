
# coding: utf-8

# # Time Series Problem Set: Question 1

# In[1]:


import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import levinson_durbin
from scipy.stats import kurtosis, kstat
from scipy.special import comb


# In[2]:


# Read in data
YEAR_NUMBER = 2000
df = pd.read_csv(f'../portfolio-analysis/{YEAR_NUMBER}_data.csv', index_col=0)

# Cut data to 250 days
sp500 = df.SP500[:250]
assert len(sp500) == 250


# ## Part (a)

# In[3]:


M = 10
lagged = np.vstack([sp500[i:240+i] for i in range(M + 1)]).T
cov = np.cov(lagged.T)
eigvals = np.linalg.eigvalsh(cov)  # Eigenvalues of a symmetric matrix
msg = 'R is positive definite.' if (eigvals > 0).all() else 'R is NOT positive definite!'
print(msg)


# ## Part (b)
# 
# If all reflection coefficients $k_m$ had magnitude less than 1, then the corresponding polynomial is stable. If all reflection coefficients above a certain order are all 0, then the corresponding system is exactly $AR$. 

# In[4]:


# The Levinson-Durbin and least squares coefficients generally agree,
# except for the first coefficient, which represents \delta_{p0}.

_, a_lv, _, sigma, _ = levinson_durbin(s=sp500, nlags=10)
ar_coeff_lv = np.hstack([1, a_lv])
ar_coeff_ls, _, _, _ = np.linalg.lstsq(np.hstack([np.ones([lagged.shape[0], 1]), lagged[:, 1:]]),
                                       lagged[:, 0], rcond=None)

print(f'Levinson-Durbin:\n{ar_coeff_lv}\n')
print(f'Least Squares:\n{ar_coeff_ls}')


# ## Part (c)

# In[5]:


aic = (2/250)*np.log(sigma[1:]) + [2*i/250 for i in range(10)]
optimal_lag = np.argmin(aic) + 1
print(f'Optimal lag value: {optimal_lag}')


# ## Part (d)

# In[6]:


sp500_diff = sp500.diff()[1:]


# In[7]:


# Part (a)
M = 10
lagged = np.vstack([sp500_diff[i:240+i] for i in range(M)]).T
cov = np.cov(lagged.T)
eigvals = np.linalg.eigvalsh(cov)  # Eigenvalues of a symmetric matrix
msg = 'R is positive definite.' if (eigvals > 0).all() else 'R is NOT positive definite!'
print(msg)

print(20*'-')

# Part (b)
_, a_lv, _, sigma, _ = levinson_durbin(s=sp500_diff, nlags=10)
ar_coeff_lv = np.hstack([1, a_lv])
ar_coeff_ls, _, _, _ = np.linalg.lstsq(np.hstack([np.ones([lagged.shape[0], 1]), lagged[:, 1:]]),
                                       lagged[:, 0], rcond=None)

print(f'Levinson-Durbin:\n{ar_coeff_lv}\n')
print(f'Least Squares:\n{ar_coeff_ls}')

print(20*'-')

# Part (c)
aic = (2/250)*np.log(sigma[1:]) + [2*i/250 for i in range(10)]
optimal_lag = np.argmin(aic) + 1
print(f'Optimal lag value: {optimal_lag}')


# ## Part (e)

# In[8]:


# For the direct model, M = 1

M = 1
lagged = np.vstack([sp500[i:250-M-1+i] for i in range(M + 1)]).T
x = lagged[:, 1:]
y = lagged[:, 0]
_, ar_coeff, _, sigma, _ = levinson_durbin(s=sp500_diff, nlags=M)
resid = pd.Series(y - x @ ar_coeff)

reflection_coeff = ar_coeff[-1]
cov = np.array([resid.autocorr(lag=i) for i in range(1, 11)])

print(f'Reflection coefficient: {reflection_coeff}')
print(f'Covariance coefficients: {cov}')


# In[9]:


# For the direct model, M = 10

M = 10
lagged = np.vstack([sp500[i:250-M-1+i] for i in range(M + 1)]).T
x = lagged[:, 1:]
y = lagged[:, 0]
_, ar_coeff, _, sigma, _ = levinson_durbin(s=sp500_diff, nlags=M)
resid = pd.Series(y - x @ ar_coeff)

reflection_coeff = ar_coeff[-1]
cov = np.array([resid.autocorr(lag=i) for i in range(1, 11)])

print(f'Reflection coefficient: {reflection_coeff}')
print(f'Covariance coefficients: {cov}')


# In[10]:


# For the first difference model, M = 1

M = 1
lagged = np.vstack([sp500_diff[i:250-M-1+i] for i in range(M + 1)]).T
x = lagged[:, 1:]
y = lagged[:, 0]
_, ar_coeff, _, sigma, _ = levinson_durbin(s=sp500_diff, nlags=M)
resid = pd.Series(y - x @ ar_coeff)

reflection_coeff = ar_coeff[-1]
cov = np.array([resid.autocorr(lag=i) for i in range(1, 11)])

print(f'Reflection coefficient: {reflection_coeff}')
print(f'Covariance coefficients: {cov}')


# In[11]:


# For the first difference model, M = 10

M = 10
lagged = np.vstack([sp500_diff[i:250-M-1+i] for i in range(M + 1)]).T
x = lagged[:, 1:]
y = lagged[:, 0]
_, ar_coeff, _, sigma, _ = levinson_durbin(s=sp500_diff, nlags=M)
resid = pd.Series(y - x @ ar_coeff)

reflection_coeff = ar_coeff[-1]
cov = np.array([resid.autocorr(lag=i) for i in range(1, 11)])

print(f'Reflection coefficient: {reflection_coeff}')
print(f'Covariance coefficients: {cov}')


# ## Part (f)

# In[12]:


# Taken from https://www.statsmodels.org/dev/_modules/statsmodels/stats/moment_helpers.html
def mnc2cum(mnc):
    '''convert non-central moments to cumulants
    recursive formula produces as many cumulants as moments

    http://en.wikipedia.org/wiki/Cumulant#Cumulants_and_moments
    '''
    mnc = [1] + list(mnc)
    kappa = [1]
    for nn,m in enumerate(mnc[1:]):
        n = nn+1
        kappa.append(m)
        for k in range(1,n):
            kappa[n] -= comb(n-1,k-1,exact=1) * kappa[k]*mnc[n-k]

    return kappa[1:]


# In[13]:


# Kurtosis is not close to 3, which would be expected for a Gaussian variable.
# It looks like the residuals are not Gaussian!
kurt = kurtosis(resid)
non_central_moments = [np.mean(resid**k) for k in range(3, 7)]
cumul = mnc2cum(non_central_moments)

print(f'Kurtosis: {kurt}')
print(f'Cumulants: {cumul}')

