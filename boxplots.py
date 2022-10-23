# Colab notebook here: https://colab.research.google.com/drive/165hYmhGoPJU5axKqMLfl8VhB-sDbzq6Q?usp=sharing

## Run code from data_upload.py

# Import pandas, numpy
import pandas as pd
import numpy as np

# Download dataset
df = pd.read_csv('https://github.com/arthursetiawan/DATA_5100_Project/blob/3ef28a95388bcf920158a93f95f13e64e68d1bfc/Fremont_Bridge_Bicycle_Counter.csv?raw=true',index_col=0)

# Remove unnecessary columns
df = df.drop(columns=['Fremont Bridge East Sidewalk','Fremont Bridge West Sidewalk'])

# Change index to DatetimeIndex
df.index=pd.to_datetime(df.index)

# Replace empty data with quadratic interpolated data
df['Fremont Bridge Total'] = df['Fremont Bridge Total'].interpolate(method='quadratic')

## End code from data_upload.py

## Run code from data_date_indexing.py

# Summarize data on daily, monthly, yearly bases
df_daily = df.resample('D').sum()
df_weekly = df.resample('W-MON').sum()
df_monthly = df.resample('M').sum()
df_quarterly = df.resample('Q').sum()
df_yearly = df.resample('Y').sum()

## End code from data_date_indexing.py

# Import matplotlib.pyplot, seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Plot distributions by year (all data)
fig, ax = plt.subplots(figsize=(24,10))
sns.boxplot(x=df.index.year, y=df['Fremont Bridge Total'], ax=ax)

# Plot distributions by quarter (all data)
fig, ax = plt.subplots(figsize=(24,10))
sns.boxplot(x=df.index.quarter, y=df['Fremont Bridge Total'], ax=ax, fliersize=0, color="w", linewidth=4)
sns.stripplot(x=df.index.quarter, y=df['Fremont Bridge Total'], ax=ax, alpha=.25)

# Plot distributions by month (all data)
fig, ax = plt.subplots(figsize=(24,10))
sns.boxplot(x=df.index.month, y=df['Fremont Bridge Total'], ax=ax)

# Plot distributions by week (all data)
fig, ax = plt.subplots(figsize=(24,10))
sns.boxplot(x=df.index.week, y=df['Fremont Bridge Total'], ax=ax)

# Plot distributions by day of week (all data)
fig, ax = plt.subplots(figsize=(24,10))
sns.boxplot(x=df.index.dayofweek, y=df['Fremont Bridge Total'], ax=ax)
