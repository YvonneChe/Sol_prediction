import pandas as pd

df=pd.read_csv('delaney-processed.csv')
df.columns
new_df=df[['measured log solubility in mols per litre', 'smiles']]
new_df.columns
new_df.to_csv('logp.csv')
new_df.describe()
new_df.notnull().sum()
df['measured log solubility in mols per litre'].plot.hist(bins=10)

train_set=new_df[:878]

train_set.describe()

train_set.to_csv('logP_train.csv')

valid_set=new_df[878:978]
valid_set.to_csv('logP_valid.csv')

test_set=new_df[978:1128]
test_set.to_csv('logP_test.csv')
