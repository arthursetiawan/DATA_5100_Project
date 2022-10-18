df['Timestamp']=df.index
df['Month'] = pd.DatetimeIndex(df['Timestamp']).month
df['Year'] = pd.DatetimeIndex(df['Timestamp']).year
df = df.reindex(columns=['Timestamp','Year','Month','Fremont Bridge Total'])
