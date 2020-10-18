import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# How to get data
# https://blog.csdn.net/dengxun7056/article/details/102054815
# How to process data
# https://blog.csdn.net/csqazwsxedc/article/details/51336322

# Use of dataframe
# Original Reference https://pandas.pydata.org/pandas-docs/stable/reference/index.html
# Basic Functions
# https://blog.csdn.net/daydayup_668819/article/details/82315565?utm_medium=distribute.pc_relevant.none-task-blog-title-1&spm=1001.2101.3001.4242
# Log and exponential
# https://blog.csdn.net/jp_666/article/details/104741629?utm_medium=distribute.pc_relevant.none-task-blog-OPENSEARCH-3.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-OPENSEARCH-3.channel_param

# Data Process
JD = pd.read_csv('JD.csv')
JD = JD.set_index("Date")  # Use Date as index

BABA = pd.read_csv('BABA.csv')
BABA = BABA.set_index("Date")

close_JD = JD["Close"]
close_BABA = BABA["Close"]

stock = pd.merge(JD, BABA, left_index=True, right_index=True)
stock = stock[["Close_x", "Close_y"]]
stock.columns = ["JD", "BABA"]

daily_return = (stock.diff() / stock.shift(periods=1)).dropna()
daily_log_return = (daily_return.apply(np.log1p)).dropna()

# the data and the plots of daily log returns for year-1
past_log = daily_log_return.loc['2018-10-18':'2019-10-18']
print("The daily log return for year-1 is: ", past_log)

fig1, ax1 = plt.subplots(nrows=1, ncols=2, figsize=(13, 4))
sns.distplot(past_log["JD"], ax=ax1[0])
ax1[0].set_title("JD Daily Log Returns for Year-1")
sns.distplot(past_log["BABA"], ax=ax1[1])
ax1[1].set_title("BABA Daily Log Returns for Year-1")
# plt.show()

JDmu = past_log["JD"].mean()
JDstd = past_log["JD"].std()
JDx = np.linspace(JDmu - 3*JDstd, JDmu + 3*JDstd, 50)
JDy = np.exp(-(JDx - JDmu) ** 2 / (2 * JDstd ** 2))/(math.sqrt(2*math.pi)*JDstd)
plt.plot(JDx, JDy, "r-", linewidth=2)

BAmu = past_log["BABA"].mean()
BAstd = past_log["BABA"].std()
BAx = np.linspace(BAmu - 3*BAstd, BAmu + 3*BAstd, 50)
BAy = np.exp(-(BAx - BAmu) ** 2 / (2 * BAstd ** 2))/(math.sqrt(2*math.pi)*BAstd)
plt.plot(BAx, BAy, "k-", linewidth=2)
# plt.show()

# Boxplot
# past_log OR past_daily OR past_stock ???
past_JD = stock["JD"].loc['2018-10-18':'2019-10-18']
past_BA = stock["BABA"].loc['2018-10-18':'2019-10-18']
fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(7, 5))
past_JD.plot.box(fontsize=7, notch=True, sym='.', grid=False, ax=ax2[0])
ax2[0].set_title("JD Data Sets for Year-1")
past_BA.plot.box(fontsize=7, notch=True, sym='.', grid=False, ax=ax2[1])
ax2[1].set_title("BABA Data Sets for Year-1")
plt.show()

# Skewness and Kurtosis
past_daily = daily_return.loc['2018-10-18':'2019-10-18']
print("The skewness for the year-1 daily returns is: ", past_daily.skew())
print("The kurtosis for the year-1 daily returns is: ", past_daily.kurt())