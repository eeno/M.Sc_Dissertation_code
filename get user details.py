"""
Your module description
"""
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import json
import boto3
import time
import pandas as pd
import csv

import tweepy

#Variables that contains the user credentials to access Twitter API
consumer_key = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
consumer_secret = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
access_token = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
access_token_secret = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)





api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify = True)

df = pd.read_csv('s3://ian-twitter-data/users.csv', index_col=False)

#df = df.loc[df['users'] != 2342119674]

df.users = df.users.astype(str)

user_list = df.users.tolist()
header_list = [ 'user','follower_count', 'friend_count','staus_count', 'created_at']
details = []



i = 1 
with open('user_details.csv', 'a', newline='') as ud:
    udwrite = csv.writer(ud)
    udwrite.writerow(header_list)
    
   
    for page in user_list:
         try:
                user = api.get_user(page)
                details.append([str(page),str(user.followers_count),str(user.friends_count),str(user.statuses_count),str(user.created_at)])
                udwrite.writerows(details)
                ud.flush()
                print(details, i)
                details.clear()
                i +=1

         except tweepy.TweepError as e:
                continue
print("DONE!!!!")