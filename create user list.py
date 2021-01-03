# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 12:53:18 2020

@author: Ian
"""

import pandas as pd

df = pd.read_csv('C:/Users/Ian/Desktop/Twitter Ids/dict.csv', index_col=False)

df = df[['follower_id']]
print("df lenght:", len(df['follower_id']))
df = df.drop_duplicates(subset=['follower_id']) #remove duplicates from followerid 
print("df lenght without dupes:", len(df['follower_id']))


df_user = df[['user_id']]
print("df lenght:", len(df[['user_id']]))
df_user = df.drop_duplicates(subset=['user_id'])

print("df_user lenght without dupes:", len(df_user['user_id']))


df2 = pd.read_csv('C:/Users/Ian/Desktop/Twitter Ids/follwers_followers.csv', index_col=False)

df2_user = df2[['ffollower_id']]
print("df2 user lenght :", len(df2_user['ffollower_id']))

df2_user = df2_user.drop_duplicates( subset=['ffollower_id'] )
print("df2 user lenght without dupes:", len(df2_user['ffollower_id']))

df2_follower = df2[['fffollower_id']]
print("df2 follower lenght :", len(df2_follower['fffollower_id']))


df2_follower  = df2_follower.drop_duplicates( subset=['fffollower_id'] )
print("df2 follower lenght without dupes :", len(df2_follower['fffollower_id']))

df.columns=['users']
df_user.columns=['users']
df2_user.columns=['users']
df2_follower.columns=['users']



df3 = df.append(df2_user, ignore_index = True)

df3 = df3.append(df2_follower, ignore_index = True)

df3 = df3.append(df_user, ignore_index = True)

print("total length of users:", len(df['user_id']) + len(df['users']) + len(df2_user['users']) + len(df2_follower['users']))


print("length of user from new df", len(df3))



df.to_csv (r'C:/Users/Ian/Desktop/Twitter Ids/users.csv', index = False, header=True)