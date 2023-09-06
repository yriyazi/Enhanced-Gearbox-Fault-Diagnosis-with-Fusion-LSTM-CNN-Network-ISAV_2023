import pandas as pd
import Utils

df = pd.read_csv('E:\Thesis\ISAV2023V2\ISAV_2023_V2\Dataset\Dataset-Raja.csv')
#df['Faulty'] = df['Faulty'].replace({False: 0, True: 1})

"""
print(df.head())

its better to keep numpy format because CWT uses numpy and before network make it pytorch

X = torch.tensor(df['Datas' ].values).reshape(Total_test,samples_per_test)
Y = torch.tensor(df['Faulty'].values).reshape(Total_test,samples_per_test)
"""

"""

"""

mask = df['Test_ID'] != 2 
df_filt = df[mask]

mask2 = df['Test_ID'] != 10
df_filtered = df_filt[mask2]

df_filtered['Test_ID'].unique()

"""

Test 2 & test 10 are removed

"""
df_loaded   = df_filtered[df_filtered['Generator']==1]
df_filtered = df_filtered[df_filtered['Generator']!=1]
df_filtered = df_filtered[df_filtered['C6']!=1]

df_filtered['label'] = df_filtered['C5'].apply(lambda x: 1 if x==1 else 0)


data = df_filtered.drop(columns=['Column1', 'IDs', 'Time', 'Name_file', 'C5', 'C6', 'Manifold', 'Generator'])
data.head()


Utils.Total_test = data['Datas' ].values.shape[0]//Utils.samples_per_test
X_numpy = data['Datas' ].values.reshape(Utils.Total_test,Utils.samples_per_test)
Test_ID_numpy = data['Test_ID'].values.reshape(Utils.Total_test,Utils.samples_per_test)
Y_numpy = data['label'].values.reshape(Utils.Total_test,Utils.samples_per_test)