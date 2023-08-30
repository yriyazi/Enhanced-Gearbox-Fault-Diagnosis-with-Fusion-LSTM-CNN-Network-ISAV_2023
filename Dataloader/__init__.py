import pandas as pd
import Utils

df = pd.read_csv('Dataset\Dataset_BinaryClass.csv')
df['Faulty'] = df['Faulty'].replace({False: 0, True: 1})

"""
print(df.head())

its better to keep numpy format because CWT uses numpy and before network make it pytorch

X = torch.tensor(df['Datas' ].values).reshape(Total_test,samples_per_test)
Y = torch.tensor(df['Faulty'].values).reshape(Total_test,samples_per_test)
"""
Utils.Total_test = df['Datas' ].values.shape[0]//Utils.samples_per_test
X_numpy = df['Datas' ].values.reshape(Utils.Total_test,Utils.samples_per_test)
Y_numpy = df['Faulty'].values.reshape(Utils.Total_test,Utils.samples_per_test)