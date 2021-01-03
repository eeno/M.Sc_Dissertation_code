# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 10:48:30 2020

@author: Ian
"""

import pandas as pd

df = pd.read_csv('C:/Users/Ian/Desktop/Twitter Ids/bot_scores.csv', index_col=False)

df2 = pd.read_csv('C:/Users/Ian/Desktop/Twitter Ids/bot_scores2.csv', index_col=False)

df3 = pd.read_csv('C:/Users/Ian/Desktop/Twitter Ids/bot_scores3.csv', index_col=False)

df4 = pd.read_csv('C:/Users/Ian/Desktop/Twitter Ids/bot_scores4.csv', index_col=False)



df_all = df.append([df2,df3,df4], ignore_index = True)




df_all.to_csv (r'C:/Users/Ian/Desktop/Twitter Ids/bot_scores_all.csv', index = False, header=True)


