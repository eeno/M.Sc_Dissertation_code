# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 20:52:38 2020

@author: Ian
"""

import pandas as pd


dftwsim = pd.read_csv('C:/Users/Ian/Desktop/Twitter Ids/tweet_simiarity scores.csv', index_col =False)


dfusefeat = pd.read_csv('C:/Users/Ian/Desktop/Twitter Ids/user features.csv', index_col =False)


dfgraphfeat = pd.read_csv('C:/Users/Ian/Desktop/Twitter Ids/graph_data.csv')



dfcomplete = pd.merge(dfusefeat,dftwsim,)


df_complete = pd.merge(dfusefeat,
                       dftwsim[['user', 'tweet_sim_score']],
                       on = 'user',
                       how = 'left')


df_complete = pd.merge(df_complete,
                       dfgraphfeat['user','pred_link_count',"user","med_of_neigh_followers","degree",'bdirectional ratio'],
                       on = 'user',
                       how = 'left')


