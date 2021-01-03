# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np


df = pd.read_csv('C:/Users/Ian/Desktop/Twitter Ids/node_list.csv', index_col=False)


df['fofo_ratio'] = df['friend_count'] / df['follower_count'] 

df['created_at'] = pd.to_datetime(df['created_at'])

df['age_of_profile'] =  pd.Timestamp('2020-12-4') - df['created_at']

df['age_of_profile'] = df['age_of_profile'].astype(np.int64)
df['age_of_profile'] = df['age_of_profile'] / 86400


df['posting_rate'] =   df['staus_count'] / df['age_of_profile']

del df['created_at']

df.to_csv (r'C:/Users/Ian/Desktop/Twitter Ids/user features.csv', index = False, header=True)


print(df.head(5))

print(df.info())

# x = np.timedelta64(2069211000000000, 'ns')
 
# days = x.astype('timedelta64[D]') 

# days / np.timedelta64(1, 'D')
