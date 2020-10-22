import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy import stats


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

IVV = pd.read_csv('IVV.csv')
IVV = IVV.set_index("Date")

close_JD = JD["Close"]
close_BABA = BABA["Close"]
close_IVV = IVV["Close"]

stock = pd.merge(JD, BABA, left_index=True, right_index=True)
stock = stock[["Close_x", "Close_y"]]
stock.columns = ["JD", "BABA"]

daily_return = (stock.diff() / stock.shift(periods=1)).dropna()
daily_log_return = (daily_return.apply(np.log1p)).dropna()

IVV_return = (close_IVV.diff() / close_IVV.shift(periods=1)).dropna()
IVV_log_return = (IVV_return.apply(np.log1p)).dropna()

# the data and the plots of daily log returns for year-1
# pd.set_option('display.max_columns', None)    shows the complete column
# pd.set_option('display.max_rows', None)  shows the complete row
past_log_stk = daily_log_return.loc['2018-10-19':'2019-10-18']
past_log_IVV = IVV_log_return.loc['2018-10-19':'2019-10-18']
print("JD and BABA's daily log return for year-1 is:")
print(past_log_stk)
print("IVV daily log return for year-1 is:")
print(past_log_IVV)

# The plot of the original data
fig0, ax0 = plt.subplots(nrows=1, ncols=3, figsize=(13, 4))
close_JD.plot(ax=ax0[0])
ax0[0].set_title("JD Close Price for Year-1")
close_JD.plot(ax=ax0[1])
ax0[1].set_title("BABA Close Price for Year-1")
close_IVV.plot(ax=ax0[2])
ax0[2].set_title("IVV Close Price for Year-1")
plt.show()

# The Histogram of daily log returns (Stocks and ETF)
fig1, ax1 = plt.subplots(nrows=1, ncols=3, figsize=(13, 4))
sns.distplot(past_log_stk["JD"], ax=ax1[0])
ax1[0].set_title("JD Daily Log Returns for Year-1")
sns.distplot(past_log_stk["BABA"], ax=ax1[1])
ax1[1].set_title("BABA Daily Log Returns for Year-1")
sns.distplot(past_log_IVV, ax=ax1[2])
ax1[2].set_title("IVV Daily Log Returns for Year-1")

JDmu = past_log_stk["JD"].mean()
JDstd = past_log_stk["JD"].std()
JDx = np.linspace(JDmu - 3*JDstd, JDmu + 3*JDstd, 50)  # obey the 3-\sigma rule
JDy = np.exp(-(JDx - JDmu) ** 2 / (2 * JDstd ** 2))/(math.sqrt(2*math.pi)*JDstd)
JDnormal = pd.DataFrame(JDy, JDx)
JDnormal.plot(ax=ax1[0])
# plt.plot(JDx, JDy, "r-", linewidth=2)

BAmu = past_log_stk["BABA"].mean()
BAstd = past_log_stk["BABA"].std()
BAx = np.linspace(BAmu - 3*BAstd, BAmu + 3*BAstd, 50)
BAy = np.exp(-(BAx - BAmu) ** 2 / (2 * BAstd ** 2))/(math.sqrt(2*math.pi)*BAstd)
BAnormal = pd.DataFrame(BAy, BAx)
BAnormal.plot(ax=ax1[1])
# plt.plot(BAx, BAy, "w*", linewidth=2) To verify the use of DataFrame is correct

IVVmu = past_log_IVV.mean()
IVVstd = past_log_IVV.std()
IVVx = np.linspace(IVVmu - 3*IVVstd, IVVmu + 3*IVVstd, 50)
IVVy = np.exp(-(IVVx - IVVmu) ** 2 / (2 * IVVstd ** 2))/(math.sqrt(2*math.pi)*IVVstd)
IVVnormal = pd.DataFrame(IVVy, IVVx)
IVVnormal.plot(ax=ax1[2])

plt.show()

# QQ plot using the CDF of the normal distribution
past_JD = stock["JD"].loc['2018-10-18':'2019-10-18']
past_BA = stock["BABA"].loc['2018-10-18':'2019-10-18']
past_daily_stk = daily_return.loc['2018-10-19':'2019-10-18']
past_daily_IVV = IVV_return.loc['2018-10-19':'2019-10-18']

# year-1 stock sample data
stats.probplot(past_JD, dist="norm", plot=plt)
# plt.show()
# year-1 daily returns
stats.probplot(past_daily_stk["JD"], dist="norm", plot=plt)
# plt.show()
# year-1 daily log returns
stats.probplot(past_log_stk["JD"], dist="norm", plot=plt)
# plt.show()
stats.probplot(past_BA, dist="norm", plot=plt)
# plt.show()

# QQ plot using another distribution
# dist 默认正态norm,我查到的有指数expon, 对数正态logistic (但都会很奇怪
# 打算用t分布 但我现在还不确定咋搞
# https://vimsky.com/examples/usage/python-scipy.stats.probplot.html
# https://www.cnblogs.com/king-lps/p/7840268.html
stats.probplot(past_JD, dist="expon", plot=plt)

# Boxplot
# past_log OR past_daily OR past_stock ???
# past stock will have no outliers.
fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(7, 5))
past_log_stk["JD"].plot.box(fontsize=7, notch=True, sym='.', grid=False, ax=ax2[0])
ax2[0].set_title("JD Data Sets for Year-1")
past_log_stk["BABA"].plot.box(fontsize=7, notch=True, sym='.', grid=False, ax=ax2[1])
ax2[1].set_title("BABA Data Sets for Year-1")
# plt.show()

# Skewness and Kurtosis
past_number_stk = past_daily_stk.count()
past_number_IVV = past_daily_IVV.count()
num1 = past_daily_stk["JD"].sub(past_daily_stk["JD"].mean())
num2 = num1.div(past_daily_stk["JD"].std())
skJD = num2.pow(3).mean()
kurtJD = num2.pow(4).mean()
# print("JD skewness for the year-1 daily returns is: ", skJD)
# print("JD kurtosis for the year-1 daily returns is: ", kurtJD)

num3 = past_daily_stk["BABA"].sub(past_daily_stk["BABA"].mean())
num4 = num3.div(past_daily_stk["BABA"].std())
skBA = num4.pow(3).mean()
kurtBA = num4.pow(4).mean()
# print("BABA skewness for the year-1 daily returns is: ", skBA)
# print("BABA kurtosis for the year-1 daily returns is: ", kurtBA)

num5 = past_daily_IVV.sub(past_daily_IVV.mean())
num6 = num5.div(past_daily_IVV.std())
skIVV = num6.pow(3).mean()
kurtIVV = num6.pow(4).mean()
# print("IVV skewness for the year-1 daily returns is: ", skIVV)
# print("IVV kurtosis for the year-1 daily returns is: ", kurtIVV)

# Use the function to get the skewness and kurtosis directly
# print("The skewness for the year-1 daily returns is:")
# print(past_daily_stk.skew())
# print("The kurtosis for the year-1 daily returns is: ")
# print(past_daily_stk.kurt())

# Use Jarque-Bera Test to test whether follow normal distributed
T_JD = 1/6*past_number_stk["JD"]*(skJD ** 2 + 1/4*((kurtJD-3) ** 2))
T_BA = 1/6*past_number_stk["BABA"]*(skBA ** 2 + 1/4*((kurtBA-3) ** 2))
T_IVV = 1/6*past_number_IVV*(skIVV ** 2 + 1/4*(kurtIVV-3) ** 2)
print(T_JD, "...", T_BA, "...", T_IVV)

# sk_stk = past_daily_stk.skew()
# kurt_stk = past_daily_stk.kurt()
# sk_IVV = past_daily_IVV.skew()
# kurt_IVV = past_daily_IVV.kurt()
# T_stk = 1/6*251*(sk_stk ** 2 + 1/4*(kurt_stk ** 2))
# T_IVV = 1/6*251*(sk_IVV ** 2 + 1/4*(kurt_IVV ** 2))
