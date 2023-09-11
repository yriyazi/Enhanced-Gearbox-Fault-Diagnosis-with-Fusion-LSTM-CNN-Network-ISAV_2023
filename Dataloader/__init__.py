from .Reader import dataser_reader
df = dataser_reader()

_datas = 20
X_numpy = df['Data' ].values.reshape(_datas,len(df['Data'])//_datas)
Y_numpy = df['Faulty'].values.reshape(_datas,len(df['Data'])//_datas)