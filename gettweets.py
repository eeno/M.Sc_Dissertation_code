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


df = pd.read_csv('s3://ian-twitter-data/user_details.csv', index_col=False)
#df.user = df.user.astype(str)

user_list = df.user.tolist()

header_list = [ 'user','screen_name','created_at' ,'tweet_text', 'source', 'tweet_id']
counter=0
details = []

with open('tweet_details.csv', 'a', newline='') as td:
    tdwrite = csv.writer(td)
    tdwrite.writerow(header_list)

    for page in user_list:
        try:
            
            for status in tweepy.Cursor(api.user_timeline,user_id=page,  include_rts = True,tweet_mode = 'extended').items(10):
                counter = counter + 1
                details.append([str(status.user.id),str(status.user.screen_name),str(status.created_at),str(status.full_text.replace('\n',' ').replace('\r',' ')),str(status.source),str(status.id)])
                tdwrite.writerows(details)
                td.flush()
                print( counter, len(user_list))
                details.clear()
                         
        except tweepy.TweepError as e:
                    print(e)
                    continue
print("done!!!!!!")