import csv
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('./Prdect-ID/dataset.csv')

df['Customer Rating'] = df['Customer Rating']-1

df0 = df[df['Customer Rating'] == 0]
df1 = df[df['Customer Rating'] == 1]
df3 = df[df['Customer Rating'] == 3]
df4 = df[df['Customer Rating'] == 4]

df0 = df0.loc[:,['Customer Rating','Customer Review']].rename(columns={'Customer Rating': 'label', 'Customer Review': 'text'})
df1 = df1.loc[:,['Customer Rating','Customer Review']].rename(columns={'Customer Rating': 'label', 'Customer Review': 'text'})
df3 = df3.loc[:,['Customer Rating','Customer Review']].rename(columns={'Customer Rating': 'label', 'Customer Review': 'text'})
df4 = df4.loc[:,['Customer Rating','Customer Review']].rename(columns={'Customer Rating': 'label', 'Customer Review': 'text'})

min_0_4 = min(len(df0), len(df4))
min_1_3 = min(len(df1), len(df3))

# ダウンサンプリング
df0_sample = df0.sample(n=min_0_4, random_state=42)
df1_sample = df1.sample(n=min_1_3, random_state=42)
df3_sample = df3.sample(n=min_1_3, random_state=42)
df4_sample = df4.sample(n=min_0_4, random_state=42)

# trialとtestに分割
df0_sample_trial, df0_sample_test = train_test_split(df0_sample, test_size=0.9, random_state=42)
df1_sample_trial, df1_sample_test = train_test_split(df1_sample, test_size=0.9, random_state=42)
df3_sample_trial, df3_sample_test = train_test_split(df3_sample, test_size=0.9, random_state=42)
df4_sample_trial, df4_sample_test = train_test_split(df4_sample, test_size=0.9, random_state=42)

df_trial = pd.concat([df0_sample_trial, df1_sample_trial, df3_sample_trial, df4_sample_trial], axis=0)
df_test = pd.concat([df0_sample_test, df1_sample_test, df3_sample_test, df4_sample_test], axis=0)

df_trial.to_csv('./Prdect-ID/eno_trial.csv', header=True, index=False)
df_test.to_csv('./Prdect-ID/eno_test.csv', header=True, index=False)
