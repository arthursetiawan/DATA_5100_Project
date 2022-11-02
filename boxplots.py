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
sns.boxplot(x=df.index.isocalendar().week, y=df['Fremont Bridge Total'], ax=ax)

# Plot distributions by day of week (all data)
fig, ax = plt.subplots(figsize=(24,10))
sns.boxplot(x=df.index.dayofweek, y=df['Fremont Bridge Total'], ax=ax)

# All plots used for paper are listed below:

# Plot distributions by month (2013-2019)
comp_1319 = df_daily.loc[df_daily['year'].isin(range(2013,2020))]
fig, ax = plt.subplots(figsize=(24,10));
sns.set_color_codes("muted")
sns.boxplot(x=comp_1319.index.month, y=comp_1319['Fremont Bridge Total'], ax=ax, color='b');
ax.tick_params(axis='both', which='major', labelsize=15);
plt.ylabel('Daily Bicycle Traffic Count', fontsize=20);
plt.xlabel('Month', fontsize=20);
ax.set_xticklabels(['January','February','March','April','May','June','July','August','September','October','November','December']);
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

#Summer months distribution comparison years 2013 to 2019
comp_1920 = df_daily.loc[df_daily['month'].isin(range(5,9))]
comp_1319 = comp_1920.loc[comp_1920['year'].isin(range(2013,2020))]
fig, axs = plt.subplots(figsize=(24,10))
gfg = sns.boxplot(x=comp_1319.index.month, y=comp_1319['Fremont Bridge Total'], hue=comp_1319.year, ax=axs);
axs.tick_params(axis='both', which='major', labelsize=15);
axs.set_xlabel('Month',fontsize=20)
axs.set_ylabel('Daily Bicycle Traffic Count',fontsize=20)
axs.set_ylim(500,6500)
axs.set_xticklabels(['May','June','July','August'])
axs.legend(loc='lower center',ncol=4,title='Year')
plt.setp(gfg.get_legend().get_texts(), fontsize='15');
plt.setp(gfg.get_legend().get_title(), fontsize='20');
axs.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'));


