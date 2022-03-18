#%%
import pickle
import pandas as pd

file = open("output_stats.pkl",'rb')
object_file = pickle.load(file)
df = pd.DataFrame(object_file)

file = open("output_stats_old.pkl",'rb')
object_file = pickle.load(file)
df2 = pd.DataFrame(object_file)
# %%
df = pd.concat([df,df2])
#%%
precision = df.groupby('mode').sum()['merged_set_length']/ (df.groupby('mode').sum()['merged_set_length']+ df.groupby('mode').sum()['false_positives_length'])
recall = df.groupby('mode').sum()['merged_set_length']/ (df.groupby('mode').sum()['merged_set_length']+ df.groupby('mode').sum()['false_negatives_length'])
# %%
