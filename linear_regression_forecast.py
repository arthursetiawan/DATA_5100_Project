# Import pandas, numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.linalg import lstsq

def pred_values(df, feature_columns, pred_column):
  A = df[feature_columns].values
  y = df[pred_column]
  p, res, rnk, s = lstsq(A, y)
  return A.dot(p)

def assign_sma(df):
  data_vals = df["Fremont Bridge Total"].values
  data_vals[data_vals == 0] = 1e-6
  df["Fremont Bridge Total"] = data_vals
  df["Fremont Bridge Total Log"] = np.log(data_vals)
  df["log sma30"] = df["Fremont Bridge Total Log"].rolling(30).mean().bfill().values
  df["log sma3"] = df["Fremont Bridge Total Log"].rolling(3).mean().bfill().values
  return df

# Download dataset
df = pd.read_csv('https://github.com/arthursetiawan/DATA_5100_Project/blob/3ef28a95388bcf920158a93f95f13e64e68d1bfc/Fremont_Bridge_Bicycle_Counter.csv?raw=true',index_col=0)

# Replace empty data with '0'
df.fillna(0,inplace=True)
df.index = pd.DatetimeIndex(df.index)
df_index = df.index

print(df.isnull().sum()) #[Answer: 0 rows]
df_daily = df.resample('D').sum()
df_weekly = df.resample('W-MON').sum()
df_monthly = df.resample('M').sum()
df_quarterly = df.resample('Q').sum()
df_yearly = df.resample('Y').sum()
daily_total = dict(list(zip(df_daily.index.date, df_daily["Fremont Bridge Total"])))
indices = list(range(len(daily_total)))

df["sma30"] = df["Fremont Bridge Total"].rolling(30).mean().bfill().values
df_daily["sma30"] = df_daily["Fremont Bridge Total"].rolling(30).mean().bfill().values
df_weekly["sma30"] = df_weekly["Fremont Bridge Total"].rolling(30).mean().bfill().values
df_monthly["sma30"] = df_monthly["Fremont Bridge Total"].rolling(30).mean().bfill().values
df_quarterly["sma30"] = df_quarterly["Fremont Bridge Total"].rolling(30).mean().bfill().values
df_yearly["sma30"] = df_yearly["Fremont Bridge Total"].rolling(30).mean().bfill().values

df["sma3"] = df["Fremont Bridge Total"].rolling(3).mean().bfill().values
df_daily["sma3"] = df_daily["Fremont Bridge Total"].rolling(3).mean().bfill().values
df_weekly["sma3"] = df_weekly["Fremont Bridge Total"].rolling(3).mean().bfill().values
df_monthly["sma3"] = df_monthly["Fremont Bridge Total"].rolling(3).mean().bfill().values
df_quarterly["sma3"] = df_quarterly["Fremont Bridge Total"].rolling(3).mean().bfill().values
df_yearly["sma3"] = df_yearly["Fremont Bridge Total"].rolling(3).mean().bfill().values

df["res_sma3"] = df["Fremont Bridge Total"] - df["sma3"]
df_daily["res_sma3"] = df_daily["Fremont Bridge Total"] - df_daily["sma3"]
df_weekly["res_sma3"] = df_weekly["Fremont Bridge Total"] - df_weekly["sma3"]
df_monthly["res_sma3"] = df_monthly["Fremont Bridge Total"] - df_monthly["sma3"]
df_quarterly["res_sma3"] = df_quarterly["Fremont Bridge Total"] - df_quarterly["sma3"]
df_yearly["res_sma3"] = df_yearly["Fremont Bridge Total"] - df_yearly["sma3"]

df["week"] = df.index.week
df_daily["week"] = df_daily.index.week
df_weekly["week"] = df_weekly.index.week

df["month"] = df.index.month
df_daily["month"] = df_daily.index.month
df_weekly["month"] = df_weekly.index.month

df["year"] = df.index.year
df_daily["year"] = df_daily.index.year
df_weekly["year"] = df_weekly.index.year

df = assign_sma(df)
df_daily = assign_sma(df_daily)
df_weekly = assign_sma(df_weekly)

feature_columns = ["week", "month", "year", "sma30", "sma3"]
pred_column = "Fremont Bridge Total"
df[pred_column + " Pred"] = pred_values(df, feature_columns, pred_column)
df_daily[pred_column + " Pred"] = pred_values(df_daily, feature_columns, pred_column)
df_weekly[pred_column + " Pred"] = pred_values(df_weekly, feature_columns, pred_column)

feature_columns = ["week", "month", "year", "log sma30", "log sma3"]
pred_column = "Fremont Bridge Total Log"
df[pred_column + " Pred"] = pred_values(df, feature_columns, pred_column)
df_daily[pred_column + " Pred"] = pred_values(df_daily, feature_columns, pred_column)
df_weekly[pred_column + " Pred"] = pred_values(df_weekly, feature_columns, pred_column)