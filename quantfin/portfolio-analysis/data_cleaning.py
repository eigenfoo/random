
# coding: utf-8

# # Data Cleaning
# 
# George Ho

# In[1]:


import numpy as np
import pandas as pd


# ## Fama-French Industrial Portfolio Data

# In[2]:


# Only read from the 24278th line onward, skipping the last row
# (a.k.a. only the data for the equal-weighted portfolios)
ff = pd.read_csv('48_Industry_Portfolios_daily.csv',
                 skiprows=24278, skipfooter=1, index_col=0,
                 parse_dates=True, engine='python')

# From the years 2000 to 2016. Had to skip some weekends.
ff = ff.loc['2000-01-03':'2016-12-30']

# Check for missing data
ff = ff.replace(-99.99, np.nan)
ff = ff.replace(-999, np.nan)
msg = ('Data missing' if ff.isna().any().any()
       else 'No missing data!')
print(msg)


# In[3]:


ff.head()


# ## LIBOR Data
# 
# Data downloaded from https://fred.stlouisfed.org/series/USD3MTD156N, from the relevant time period.

# In[4]:


libor = (pd.read_csv('libor.csv', index_col=0, parse_dates=True)
           .squeeze())

# Coerce data to numeric values. Non-numeric data becomes NaNs.
libor = pd.to_numeric(libor, errors='coerce')

# Forward-fill missing values
libor = libor.fillna(method='ffill')

# Check for missing data
msg = ('Data missing!' if libor.isna().any()
       else 'No missing data!')
print(msg)


# In[5]:


def three_month_to_daily(x, N=63):
    '''
    Convert three-month (i.e. quarterly) interest rates to
    daily interest rates. A quarter is roughly 63 days.
    '''
    return 100*((1 + x/100)**(1/N) - 1)


# In[6]:


libor = three_month_to_daily(libor)


# In[7]:


libor.head()


# ## S&P 500 Data
# 
# Data downloaded from https://finance.yahoo.com/quote/%5EGSPC/

# In[8]:


sp500 = pd.read_csv('sp500.csv', index_col=0, parse_dates=True)

# For returns, take the percent change of the opening prices.
sp500 = sp500['Open'].pct_change()

# Coerce data to numeric values. Non-numeric data becomes NaNs.
sp500 = pd.to_numeric(sp500, errors='coerce')

# Forward-fill missing values
sp500 = sp500.fillna(method='ffill')

# We still have one last NaN: the first day (which is NaN
# because we computed percent change). Backfill for this one.
sp500 = sp500.fillna(method='bfill')

# Check for missing data
msg = ('Data missing' if sp500.isna().any()
       else 'No missing data!')
print(msg)


# In[9]:


sp500.head()


# ## Aggregate Data into a Single DataFrame

# In[10]:


df = ff
df['LIBOR'] = libor
df['SP500'] = sp500

# Putting all our data together, we see that we are missing LIBOR
# data for the very first day. This is the only missing datapoint.
# Backfill for this.
df = df.fillna(method='bfill')


# In[11]:


msg = ('Data missing!' if df.isna().any().any()
       else 'No missing data!')
print(msg)


# In[12]:


df.head()


# ## Save Data by Year

# In[13]:


for yr in range(2000, 2017):
    df.loc[df.index.year == yr].to_csv('{}_data.csv'.format(yr))

