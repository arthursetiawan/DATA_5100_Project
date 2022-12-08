import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('https://github.com/arthursetiawan/DATA_5100_Project/blob/3ef28a95388bcf920158a93f95f13e64e68d1bfc/Fremont_Bridge_Bicycle_Counter.csv?raw=true',index_col=0)

# Remove unnecessary columns
df = df.drop(columns=['Fremont Bridge East Sidewalk','Fremont Bridge West Sidewalk'])
# Change index to DatetimeIndex
df.index=pd.to_datetime(df.index)
# Replace empty data with quadratic interpolated data
df['Fremont Bridge Total'] = df['Fremont Bridge Total'].interpolate(method='quadratic')
# Summarize data on daily, monthly, yearly bases
df_daily = df.resample('D').sum()
df_weekly = df.resample('W-MON').sum()
df_monthly = df.resample('M').sum()
df_quarterly = df.resample('Q').sum()
df_yearly = df.resample('Y').sum()

df_weekly.plot()

import matplotlib as mpl
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#Weekly Seasonal Decomposition
mpl.rcParams['figure.figsize'] = (12,8)

result = seasonal_decompose(df_weekly)
result.plot()

plt.show()

fig, ax = plt.subplots(figsize=(8,5))
plot_acf(df_weekly, ax=ax)
plt.show()

# STL Non-Robust, Seasonal Smoother=7
mpl.rcParams['figure.figsize'] = (12,8)
from statsmodels.tsa.seasonal import STL
res = STL(df_weekly, robust=False).fit()
res.plot()
plt.show()

mpl.rcParams['figure.figsize'] = (12,8)
from statsmodels.tsa.seasonal import STL
res = STL(df_weekly, robust=True).fit()
res.plot()
plt.show()

def add_stl_plot(fig, res, legend):
    """Add 3 plots from a second STL fit"""
    axs = fig.get_axes()
    comps = ["trend", "seasonal", "resid"]
    for ax, comp in zip(axs[1:], comps):
        series = getattr(res, comp)
        if comp == "resid":
            ax.plot(series, marker="o", linestyle="none")
        else:
            ax.plot(series)
            if comp == "trend":
                ax.legend(legend, frameon=False)


stl = STL(df_weekly, period=52, robust=True)
res_robust = stl.fit()
fig = res_robust.plot()
res_non_robust = STL(df_weekly, period=52, robust=False).fit()
add_stl_plot(fig, res_non_robust, ["STL Robust", "STL Non-Robust"])

def add_nonrob_plot(fig, res, legend):
    """Add 3 plots from a second STL fit"""
    axs = fig.get_axes()
    comps = ["trend", "seasonal", "resid"]
    for ax, comp in zip(axs[1:], comps):
        series = getattr(res, comp)
        if comp == "resid":
            ax.plot(series, marker="o", linestyle="none")
        else:
            ax.plot(series)
            if comp == "trend":
                ax.legend(legend, frameon=False)

res_non_robust = STL(df_weekly, period=52, robust=False).fit()
fig = seasonal_decompose(df_weekly).plot()
add_nonrob_plot(fig, res_non_robust, ["Classical", "STL Non-robust"])



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

sns.set_style("whitegrid")

fig, ax = plt.subplots(1,1, figsize=(12,8))
stats.probplot(result.resid, plot=ax, fit=False);
stats.probplot(res_non_robust.resid, plot=ax, fit=False);
stats.probplot(res_robust.resid, plot=ax, fit=False);

ax.get_lines()[0].set_markerfacecolor('c')
ax.get_lines()[1].set_markerfacecolor('m')
ax.get_lines()[2].set_markerfacecolor('y')

ax.get_lines()[0].set_markeredgewidth(0)
ax.get_lines()[1].set_markeredgewidth(0)
ax.get_lines()[2].set_markeredgewidth(0)

plt.legend(title='Model', loc='upper left', labels=['Classical', 'Non-Robust STL','Robust STL'])

mpl.rcParams['figure.figsize'] = (12,8)
sns.kdeplot(data=res_robust.resid)
sns.kdeplot(data=res_non_robust.resid)
sns.kdeplot(data=result.resid)

plt.legend(title='Model', loc='upper left', labels=['Robust STL', 'Non-Robust STL','Classical'])


# STLForecast w/ ARIMA forecast model

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast

df_weekly.index.freq = df_weekly.index.inferred_freq
stlf = STLForecast(df_weekly, ARIMA, 
                   model_kwargs=dict(order=(1, 1, 0), trend="t"), 
                   robust=False, seasonal=13)
stlf_res = stlf.fit()

forecast = stlf_res.forecast(72)
plt.plot(df_weekly)
plt.plot(forecast)
plt.show()

print(stlf_res.summary())

# STLForecast w/ Exponential Smoothing forecast model

from statsmodels.tsa.statespace import exponential_smoothing
from statsmodels.tsa.forecasting.stl import STLForecast

df_weekly.index.freq = df_weekly.index.inferred_freq

ES = exponential_smoothing.ExponentialSmoothing
stlf = STLForecast(df_weekly, ES, model_kwargs={"trend": True}, 
                   robust=False)
res = stlf.fit()
forecasts = res.forecast(72)

fig,ax=plt.subplots(1,1,figsize=(15,8))

ax.plot(df_weekly, label='Historic')
ax.plot(forecasts, label='Robust STL')

ax.set_xlabel("Year")
ax.set_ylabel("Bike Count")
ax.legend(loc='upper right')

#References
#Statsmodels: https://www.statsmodels.org/dev/index.html 
#Statsmodels.tsa: https://www.statsmodels.org/stable/tsa.html 
#Statsmodels.tsa.seasonal.seasonal_decompose: https://www.statsmodels.org/devel/generated/statsmodels.tsa.seasonal.seasonal_decompose.html 
#Statsmodels.tsa.seasonal.STL: https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.STL.html 
#STLForecast: https://www.statsmodels.org/dev/generated/statsmodels.tsa.forecasting.stl.STLForecast.html 
#Exponential Smoothing: https://www.statsmodels.org/dev/examples/notebooks/generated/exponential_smoothing.html 
#ARIMA: https://www.statsmodels.org/dev/generated/statsmodels.tsa.arima.model.ARIMA.html 
