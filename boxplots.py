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

#Summer months linear regression showing increase year-on-year trend
comp_1319 = comp_1920.loc[comp_1920['year'].isin(range(2013,2020))]
g = sns.FacetGrid(comp_1319, col = "month")
g.map(sns.regplot, "year", "Fremont Bridge Total", x_estimator=np.mean)
g.figure.set_size_inches(24, 10)
for ax, title in zip(g.axes.flat, ['May', 'June','July','August']):
    ax.set_title(title,fontsize=25)
    ax.set_ylabel('Daily Bicycle Traffic Count', fontsize='x-large')
    ax.tick_params(axis='both', which='major', labelsize=15);
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'));
    ax.set_xlabel('Year',fontsize='x-large')
    
#Comparison of 2019 and 2020 hourly traffic distributions
comp_1920 = df.loc[df['year'].isin(range(2019,2021))]
fig, ax = plt.subplots(figsize=(24,10));
gfg = sns.boxplot(x=comp_1920.index.year, y=comp_1920['Fremont Bridge Total'], ax=ax, hue=comp_1920.hour);
ax.tick_params(axis='both', which='major', labelsize=15);
plt.ylabel('Hourly Bicycle Traffic Count', fontsize=20);
plt.xlabel('Year', fontsize=20);
plt.legend(title='Hour',ncol=2);
plt.setp(gfg.get_legend().get_texts(), fontsize='15');
plt.setp(gfg.get_legend().get_title(), fontsize='20');

#Comparison of 2019 weekday and weekend  hourly traffic distributions
y19 = df.loc[df.index.year == 2019]
wd = y19.loc[y19.index.dayofweek.isin(range(0,5))]
we = y19.loc[y19.index.dayofweek.isin(range(5,6))]
fig, axs = plt.subplots(ncols=2,figsize=(24,10),sharey=True)
gfg = sns.boxplot(x=wd.year, y=wd['Fremont Bridge Total'], hue=wd.hour, ax=axs[0]);
ghg = sns.boxplot(x=we.year, y=we['Fremont Bridge Total'], hue=we.hour, ax=axs[1]);
axs[0].tick_params(axis='both', which='major', labelsize=18);
axs[0].set_xlabel('')
axs[0].set_ylabel('Hourly Bicycle Traffic Count',fontsize=20)
axs[0].set_ylim(0,1200)
axs[0].set_xticklabels(['Weekday (Mon-Fri)'])
axs[0].legend(loc='upper right',ncol=2,title='Hour')
plt.setp(gfg.get_legend().get_texts(), fontsize='15');
plt.setp(gfg.get_legend().get_title(), fontsize='18');
axs[0].yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'));
axs[1].tick_params(axis='both', which='major', labelsize=18);
axs[1].set_xlabel('')
axs[1].set_ylabel('')
axs[1].set_xticklabels(['Weekend (Sat-Sun)'])
axs[1].legend(loc='upper right',ncol=2,title='Hour')
plt.setp(ghg.get_legend().get_texts(), fontsize='15');
plt.setp(ghg.get_legend().get_title(), fontsize='18');
axs[1].yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'));
plt.tight_layout()

#Comparison of 2021 weekday and weekend  hourly traffic distributions
y19 = df.loc[df.index.year == 2021]
wd = y19.loc[y19.index.dayofweek.isin(range(0,5))]
we = y19.loc[y19.index.dayofweek.isin(range(5,6))]
fig, axs = plt.subplots(ncols=2,figsize=(24,10),sharey=True)
gfg = sns.boxplot(x=wd.year, y=wd['Fremont Bridge Total'], hue=wd.hour, ax=axs[0]);
ghg = sns.boxplot(x=we.year, y=we['Fremont Bridge Total'], hue=we.hour, ax=axs[1]);
axs[0].tick_params(axis='both', which='major', labelsize=18);
axs[0].set_xlabel('')
axs[0].set_ylabel('Hourly Bicycle Traffic Count',fontsize=20)
axs[0].set_ylim(0,1200)
axs[0].set_xticklabels(['Weekday (Mon-Fri)'])
axs[0].legend(loc='upper right',ncol=2,title='Hour')
plt.setp(gfg.get_legend().get_texts(), fontsize='15');
plt.setp(gfg.get_legend().get_title(), fontsize='18');
axs[0].yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'));
axs[1].tick_params(axis='both', which='major', labelsize=18);
axs[1].set_xlabel('')
axs[1].set_ylabel('')
axs[1].set_xticklabels(['Weekend (Sat-Sun)'])
axs[1].legend(loc='upper right',ncol=2,title='Hour')
plt.setp(ghg.get_legend().get_texts(), fontsize='15');
plt.setp(ghg.get_legend().get_title(), fontsize='18');
axs[1].yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'));
plt.tight_layout()

#Comparison of 2018, 2019, 2020, and 2021 average hourly traffic 
comp_1920 = df.loc[df['year'].isin(range(2018,2022))]

fig, ax = plt.subplots(figsize=(24,10));
gfg = sns.lineplot(data=comp_1920, x="hour", y="Fremont Bridge Total", hue="year", err_style="bars",palette=sns.color_palette("RdBu", 4));
ax.tick_params(axis='both', which='major', labelsize=15);
plt.ylabel('Hourly Bicycle Traffic Count', fontsize=20);
plt.xlabel('Hour', fontsize=20);
plt.legend(title='Year');
plt.setp(gfg.get_legend().get_texts(), fontsize='15');
plt.setp(gfg.get_legend().get_title(), fontsize='20');
plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],['1am','2am','3am','4am','5am','6am','7am','8am','9am','10am','11am','12pm','1pm','2pm','3pm','4pm','5pm','6pm','7pm','8pm','9pm','10pm','11pm','12am']);
