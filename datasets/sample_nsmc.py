import pandas as pd
from sklearn.model_selection import train_test_split

df_test = pd.read_csv('./nsmc/ratings_test.txt', sep='\t')
df_train = pd.read_csv('./nsmc/ratings_train.txt', sep='\t')


df0_test = df_test[df_test['label'] == 0]
df4_test = df_test[df_test['label'] == 1]
df0_train = df_train[df_train['label'] == 0]
df4_train = df_train[df_train['label'] == 1]

# change label 1 to label 4 for match MARC
df4_test.loc[df4_test['label']==1, ['label']] = 4
df4_train.loc[df4_train['label']==1, ['label']] = 4

df0_test = df0_test.rename(columns={'document': 'text'})
df4_test = df4_test.rename(columns={'document': 'text'})
df0_train = df0_train.rename(columns={'document': 'text'})
df4_train = df4_train.rename(columns={'document': 'text'})


# Sampling
df0_test_sample = df0_test.sample(n=2000, random_state=42)
df4_test_sample = df4_test.sample(n=2000, random_state=42)

df0_train_first, df0_train_second = train_test_split(df0_train, test_size=0.5, random_state=42)
df4_train_first, df4_train_second = train_test_split(df4_train, test_size=0.5, random_state=42)
df0_train_first['label'] = 0
df0_train_second['label'] = 1
df4_train_first['label'] = 3
df4_train_second['label'] = 4


df_test_sample = pd.concat([df0_test_sample, df4_test_sample], axis=0)
df_train_sample = pd.concat([df0_train_first, df0_train_second, df4_train_first, df4_train_second], axis=0)

df_test_sample.to_csv('./nsmc/eno_test.csv', header=True, index=False)
df_train_sample.to_csv('./nsmc/eno_train.csv', header=True, index=False)