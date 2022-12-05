# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats

# Import Statsmodels libraries
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.statespace import exponential_smoothing

# Import Facebook Prophet libraries
from prophet import Prophet

# Import data
df = pd.read_csv('https://github.com/arthursetiawan/DATA_5100_Project/blob/3ef28a95388bcf920158a93f95f13e64e68d1bfc/Fremont_Bridge_Bicycle_Counter.csv?raw=true',index_col=0)

# Remove unnecessary columns
df = df.drop(columns=['Fremont Bridge East Sidewalk','Fremont Bridge West Sidewalk'])

# Change index to DatetimeIndex
df.index = pd.to_datetime(df.index)

# Replace empty data with quadratic interpolated data
df['Fremont Bridge Total'] = df['Fremont Bridge Total'].interpolate(method='quadratic')

# Summarize data on daily, monthly, yearly bases
df_daily = df.resample('D').sum()
df_weekly = df.resample('W-MON').sum()
df_monthly = df.resample('M').sum()
df_quarterly = df.resample('Q').sum()
df_yearly = df.resample('Y').sum()

# Make a copy of the dataframe for Prophet using the required format (ds & y columns)
# Define a function for this step
def desireddf(df):
  df_copy = df.copy()
  df_copy.reset_index(inplace=True)
  df_copy.rename(columns={"Date":"ds","Fremont Bridge Total":"y"},inplace=True)
  return df_copy

# Fit a model WITHOUT holiday conditionality

# Multiplicative seasonality mode works best
m = Prophet(seasonality_mode='multiplicative', interval_width=0.95).fit(desireddf(df_weekly));

# Weekly prediction
future = m.make_future_dataframe(periods=72,freq='w');
forecast = m.predict(future);

proph = pd.merge(desireddf(df_weekly), forecast, on=['ds'], how='inner')

# Evaluate residuals
proph["residual"] = desireddf(df_weekly)['y']-forecast.yhat

# Fit a model WITH holiday conditionality

# Define COVID-19 lockdown periods (these will be treated as holiday periods)
lockdowns = pd.DataFrame([
    {'holiday': 'lockdown_1', 'ds': '2020-03-21', 'lower_window': 0, 'ds_upper': '2020-06-06'},
    {'holiday': 'lockdown_2', 'ds': '2020-07-09', 'lower_window': 0, 'ds_upper': '2020-10-27'},
    {'holiday': 'lockdown_3', 'ds': '2021-02-13', 'lower_window': 0, 'ds_upper': '2021-02-17'},
    {'holiday': 'lockdown_4', 'ds': '2021-05-28', 'lower_window': 0, 'ds_upper': '2021-06-10'},
])
for t_col in ['ds', 'ds_upper']:
    lockdowns[t_col] = pd.to_datetime(lockdowns[t_col])
lockdowns['upper_window'] = (lockdowns['ds_upper'] - lockdowns['ds']).dt.days

# Covid affected data
df=desireddf(df_weekly)
df2 = df.copy()
df2['pre_covid'] = pd.to_datetime(df2['ds']) < pd.to_datetime('2020-03-21')
df2['post_covid'] = ~df2['pre_covid']

# Include lockdowns as holiday periods, multiplicative seasonality mode works best
m2 = Prophet(holidays=lockdowns,seasonality_mode='multiplicative', weekly_seasonality=False,interval_width=0.95)
m2.add_seasonality(
    name='weekly_pre_covid',
    period=7,
    fourier_order=3,
    condition_name='pre_covid',
)
m2.add_seasonality(
    name='weekly_post_covid',
    period=7,
    fourier_order=3,
    condition_name='post_covid',
);

#clipper function
def clipper(df):
  for col in ['yhat', 'yhat_lower', 'yhat_upper']:
    df[col] = df[col].clip(lower=0.0)
  return df

# Weekly prediction
m2.fit(df2)
future4 = m2.make_future_dataframe(periods=72,freq='w')
future4['pre_covid'] = pd.to_datetime(future4['ds']) < pd.to_datetime('2020-03-21')
future4['post_covid'] = ~future4['pre_covid']
forecast4 = m2.predict(future4)
forecast4=clipper(forecast4)

proph_con = pd.merge(desireddf(df_weekly), forecast4, on=['ds'], how='inner')

# Evaluate residuals
proph_con["residual"] = desireddf(df_weekly)['y']-forecast4.yhat

# Fit a model using Seasonal Decomposition
classic = seasonal_decompose(df_weekly)

# Fit a model using Season Trend LOESS, non-robust
res_non_robust = STL(df_weekly, period=52, robust=False).fit()

# Fit a model using Season Trend LOESS, robust
res_robust = STL(df_weekly, period=52, robust=True).fit()

# Q-Q Plot
sns.set_style("whitegrid")

fig, ax = plt.subplots(1,1, figsize=(12,10))
stats.probplot(proph.residual, plot=ax, fit=False);
stats.probplot(proph_con.residual, plot=ax, fit=False);
stats.probplot(res_non_robust.resid, plot=ax, fit=False);
stats.probplot(res_robust.resid, plot=ax, fit=False);

ax.get_lines()[0].set_markerfacecolor('c')
ax.get_lines()[0].set_markeredgewidth(0)
ax.get_lines()[1].set_markerfacecolor('m')
ax.get_lines()[1].set_markeredgewidth(0)
ax.get_lines()[2].set_markerfacecolor('y')
ax.get_lines()[2].set_markeredgewidth(0)
ax.get_lines()[3].set_markerfacecolor('k')
ax.get_lines()[3].set_markeredgewidth(0)

plt.legend(title='Model', loc='upper left', labels=['Prophet w/o Holiday', 'Prophet with Holiday', 'Non-Robust STL','Robust STL'])
plt.ylim((-11000,11000));

# Residuals Plot
sns.set_style("whitegrid")

fig, ax = plt.subplots(1,1, figsize=(12,8))

ax.plot(proph.ds, proph.residual, marker="o", linestyle="none",label="Prophet w/o Holiday")
ax.plot(proph.ds, proph_con.residual, marker="o", linestyle="none",label="Prophet with Holiday")
ax.plot(proph.ds, res_non_robust.resid, marker="o", linestyle="none",label="STL Non-Robust")
ax.plot(proph.ds, res_robust.resid, marker="o", linestyle="none",label="STL Robust")

ax.get_lines()[0].set_markerfacecolor('c')
ax.get_lines()[1].set_markerfacecolor('m')
ax.get_lines()[2].set_markerfacecolor('y')
ax.get_lines()[3].set_markerfacecolor('k')

ax.get_lines()[0].set_markeredgewidth(0)
ax.get_lines()[1].set_markeredgewidth(0)
ax.get_lines()[2].set_markeredgewidth(0)
ax.get_lines()[3].set_markeredgewidth(0)

ax.legend(loc='best')
ax.axhline(0,color="black")
plt.ylim((-11000,11000));

mpl.rcParams['figure.figsize'] = (12,8)
sns.kdeplot(data=proph.residual, color='c')
sns.kdeplot(data=proph_con.residual, color='m')
sns.kdeplot(data=res_non_robust.resid, color='y')
sns.kdeplot(data=res_robust.resid, color='k')

plt.legend(title='Model', loc='upper left', labels=['Prophet w/o Holiday', 'Prophet with Holiday','STL Non-Robust','STL Robust'])


# Q-Q Plot
sns.set_style("whitegrid")

fig, ax = plt.subplots(1,1, figsize=(12,10))
stats.probplot(proph.residual, plot=ax, fit=False);
stats.probplot(proph_con.residual, plot=ax, fit=False);

ax.get_lines()[0].set_markerfacecolor('c')
ax.get_lines()[0].set_markeredgewidth(0)
ax.get_lines()[1].set_markerfacecolor('m')
ax.get_lines()[1].set_markeredgewidth(0)

plt.legend(title='Model', loc='upper left', labels=['Prophet w/o Holiday', 'Prophet with Holiday'])
plt.ylim((-11000,11000));


# Residuals Plot
sns.set_style("whitegrid")

fig, ax = plt.subplots(1,1, figsize=(12,8))

ax.plot(proph.ds, proph.residual, marker="o", linestyle="none",label="Prophet w/o Holiday")
ax.plot(proph.ds, proph_con.residual, marker="o", linestyle="none",label="Prophet with Holiday")

ax.get_lines()[0].set_markerfacecolor('c')
ax.get_lines()[1].set_markerfacecolor('m')

ax.get_lines()[0].set_markeredgewidth(0)
ax.get_lines()[1].set_markeredgewidth(0)

ax.legend(loc='best')
ax.axhline(0,color="black")
plt.ylim((-11000,11000));

mpl.rcParams['figure.figsize'] = (12,8)
sns.kdeplot(data=proph.residual, color='c')
sns.kdeplot(data=proph_con.residual, color='m')


plt.legend(title='Model', loc='upper left', labels=['Prophet w/o Holiday', 'Prophet with Holiday'])

# Q-Q Plot
sns.set_style("whitegrid")

fig, ax = plt.subplots(1,1, figsize=(12,10))

stats.probplot(res_non_robust.resid, plot=ax, fit=False);
stats.probplot(res_robust.resid, plot=ax, fit=False);

ax.get_lines()[0].set_markerfacecolor('y')
ax.get_lines()[0].set_markeredgewidth(0)
ax.get_lines()[1].set_markerfacecolor('k')
ax.get_lines()[1].set_markeredgewidth(0)

plt.legend(title='Model', loc='upper left', labels=['Non-Robust STL','Robust STL'])
plt.ylim((-11000,11000));

# Residuals Plot
sns.set_style("whitegrid")

fig, ax = plt.subplots(1,1, figsize=(12,8))

ax.plot(proph.ds, classic.resid, marker="o", linestyle="none",label="Classical SD")
ax.plot(proph.ds, res_non_robust.resid, marker="o", linestyle="none",label="STL Non-Robust")
ax.plot(proph.ds, res_robust.resid, marker="o", linestyle="none",label="STL Robust")

ax.get_lines()[0].set_markerfacecolor('r')
ax.get_lines()[1].set_markerfacecolor('y')
ax.get_lines()[2].set_markerfacecolor('k')

ax.get_lines()[0].set_markeredgewidth(0)
ax.get_lines()[1].set_markeredgewidth(0)
ax.get_lines()[2].set_markeredgewidth(0)

ax.legend(loc='best')
ax.axhline(0,color="black")
plt.ylim((-11000,11000));

# STLForecast w/ Exponential Smoothing forecast model

df_weekly.index.freq = df_weekly.index.inferred_freq

ES = exponential_smoothing.ExponentialSmoothing
stlf = STLForecast(df_weekly, ES, model_kwargs={"trend": True}, 
                   robust=False)
res = stlf.fit()
forecasts = res.forecast(72)

plt.plot(df_weekly)
plt.plot(forecasts)
plt.show()

# STLForecast w/ ARIMA forecast model

df_weekly.index.freq = df_weekly.index.inferred_freq

stlf = STLForecast(df_weekly, ARIMA, model_kwargs=dict(order=(1, 1, 0), trend="t"), 
                   robust=True, seasonal=13)
res = stlf.fit()
robust_forecast = res.forecast(72)

plt.plot(df_weekly)
plt.plot(robust_forecast)
plt.show()


previous=desireddf(df_weekly)
previous = previous.loc[previous['ds']>'2020-12-31 00:00:00']

df_weekly.index.freq = df_weekly.index.inferred_freq

ES = exponential_smoothing.ExponentialSmoothing
stlfr = STLForecast(df_weekly, ES, model_kwargs={"trend": True}, 
                   robust=True)
res = stlfr.fit()
robust_forecasts = res.forecast(72)

ES = exponential_smoothing.ExponentialSmoothing
stlfu = STLForecast(df_weekly, ES, model_kwargs={"trend": True}, 
                   robust=False)
res = stlfu.fit()
unrobust_forecasts = res.forecast(72)

pred_holiday=forecast4.loc[forecast4['ds']>='2022-09-30 00:00:00']
predicted=forecast.loc[forecast['ds']>='2022-09-30 00:00:00']

fig,ax= plt.subplots(1,1,figsize=(15,8)) 
ax.plot(previous.ds,previous.y,label='Historic')
ax.plot(pred_holiday.ds,pred_holiday.yhat,label='Prophet with Holiday',color='m')
ax.plot(predicted.ds,predicted.yhat, label='Prophet w/o Holiday',color='c')
ax.plot(robust_forecasts,color='k',label='Robust STL')
ax.plot(unrobust_forecasts,color='y',label='Non-Robust STL')

ax.legend(loc='best')
ax.set_xlabel("Year")
ax.set_ylabel("Bike Count")


# 72 week forecast using Robust STL and ARIMA / Exponential Smoothing

previous=desireddf(df_weekly)
previous = previous.loc[previous['ds']>'2020-12-31 00:00:00']

df_weekly.index.freq = df_weekly.index.inferred_freq

stlf = STLForecast(df_weekly, ARIMA, model_kwargs=dict(order=(1, 1, 0), trend="t"), 
                   robust=True, seasonal=13)
res = stlf.fit()
robust_forecast = res.forecast(72)

ES = exponential_smoothing.ExponentialSmoothing
stlfr = STLForecast(df_weekly, ES, model_kwargs={"trend": True}, 
                   robust=True)
res = stlfr.fit()
robust_forecasts = res.forecast(72)

fig,ax= plt.subplots(1,1,figsize=(15,8)) 
ax.plot(previous.ds,previous.y,label='Historic')
ax.plot(robust_forecast,color='k',label='Robust STL ARIMA')
ax.plot(robust_forecasts,color='g',label='Robust STL Exponential Smoothing')

ax.legend(loc='best')
ax.set_xlabel("Year")
ax.set_ylabel("Bike Count")

# 72 week forecast with Facebook Prophet & STL

previous=desireddf(df_weekly)
previous = previous.loc[previous['ds']>'2020-12-31 00:00:00']

df_weekly.index.freq = df_weekly.index.inferred_freq

stlf = STLForecast(df_weekly, ARIMA, model_kwargs=dict(order=(1, 1, 0), trend="t"), 
                   robust=True, seasonal=13)
res = stlf.fit()
robust_forecast = res.forecast(72)

ES = exponential_smoothing.ExponentialSmoothing
stlfr = STLForecast(df_weekly, ES, model_kwargs={"trend": True}, 
                   robust=True)
res = stlfr.fit()
robust_forecasts = res.forecast(72)

pred_holiday=forecast4.loc[forecast4['ds']>='2022-09-30 00:00:00']
predicted=forecast.loc[forecast['ds']>='2022-09-30 00:00:00']

fig,ax= plt.subplots(1,1,figsize=(15,8)) 
ax.plot(previous.ds,previous.y,label='Historic')
ax.plot(pred_holiday.ds,pred_holiday.yhat,label='Prophet with Holiday',color='m')
ax.plot(predicted.ds,predicted.yhat, label='Prophet w/o Holiday',color='c')
ax.plot(robust_forecast,color='k',label='Robust STL ARIMA')
ax.plot(robust_forecasts,color='g',label='Robust STL Exponential Smoothing')

ax.legend(loc='best')
ax.set_xlabel("Year")
ax.set_ylabel("Bike Count")
