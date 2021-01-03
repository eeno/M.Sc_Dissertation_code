

import botometer
import pandas as pd
import csv
import time
import numpy as np



df = pd.read_csv("bot_check_ids4.csv")



accounts = df.userid.unique().tolist()

print(len(accounts))

accounts = [accounts[x:x+100] for x in range(0, len(accounts), 100)]

print(len(accounts))
rapidapi_key = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" # now it's called rapidapi key
twitter_app_auth = {
    'consumer_key': "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
    'consumer_secret': "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
    'access_token': "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
    'access_token_secret': "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
    }

blt_twitter = botometer.BotometerLite(wait_on_ratelimit=True,rapidapi_key=rapidapi_key, **twitter_app_auth)




i = 0
header_list = [['botscore','tweet_id','user_id']]

with open('bot_scores4.csv', 'a') as bt:
    btwrite = csv.writer(bt)
    btwrite.writerow(header_list)
    
    for account in accounts:
        blt_scores = blt_twitter.check_accounts_from_user_ids(account)
        keys = blt_scores[0].keys()
        dict_writer = csv.DictWriter(bt, keys)
        #dict_writer.writeheader()
        dict_writer.writerows(blt_scores)
        bt.flush
        blt_scores
        i = i+1
        print(i,len(accounts))
    
    
    
