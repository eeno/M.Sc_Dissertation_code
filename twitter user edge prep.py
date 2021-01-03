

import csv
from operator import itemgetter
import networkx as nx
from networkx.algorithms import community #This part of networkx, for community detection, needs to be imported separately.
import pandas as pd


df = pd.read_csv('C:/Users/Ian/Desktop/Twitter Ids/node_list.csv', index_col=False)

df['user'] = df['user'].astype('int64')

df = df[['user']]

df_dict = pd.read_csv('C:/Users/Ian/Desktop/Twitter Ids/dict.csv', index_col=False)

df_dict = df_dict.rename(columns = {'user_id':'ffollower_id','follower_id':'fffollower_id' })

df2 = pd.read_csv('C:/Users/Ian/Desktop/Twitter Ids/follwers_followers.csv', index_col=False)



print(df.shape)
print(df_dict.shape)
print(df2.shape)

df2 = df2.append(df_dict, ignore_index = True)

print(df2.dtypes)

df.astype(int)
df2['ffollower_id'].astype('int64')
df2['fffollower_id'].astype('int64')


print(df['user'].dtypes)
print(type(df2['ffollower_id']))
print(type(df2['fffollower_id']))



print(df.tail(20))

print(df2.tail(20))
print(df2.shape)
user_list = list(df['user'])

user_list = list(user_list)




df_complete = pd.merge(df2, 
                        df,
                        left_on = "ffollower_id",
                        right_on = "user",
                        how='left')




df_complete = pd.merge(df_complete, 
                        df,
                        left_on = "fffollower_id",
                        right_on = "user",
                        how='left')  




print(df_complete.shape)



testdf = df_complete[df_complete[['user_x','user_y']].notnull().all(axis=1)]

test_df = pd.DataFrame(testdf)
print(testdf.isnull().sum())

testdf = testdf[['ffollower_id','fffollower_id']]
print(testdf.isnull().sum())
print(testdf.shape)
print(testdf.tail(20))

#print(df_complete.shape)

testdf.to_csv (r'C:/Users/Ian/Desktop/Twitter Ids/edge_list_complete.csv', index = False, header=True)


