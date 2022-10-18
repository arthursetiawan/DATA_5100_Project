# Change index to DatetimeIndex
df.index=pd.to_datetime(df.index)

# Summarize data on daily, monthly bases
df_daily = df.resample('D').sum()
df_monthly = df.resample('M').sum()
