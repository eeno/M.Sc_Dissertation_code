"""
Your module description
"""


import subprocess
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import tweepy
import pandas as pd
import csv

#Variables that contains the user credentials to access Twitter API
consumer_key = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
consumer_secret = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
access_token = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
access_token_secret = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"


auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
# The api object gives you access to all of the http calls that Twitter accepts 
api = tweepy.API(auth,wait_on_rate_limit=True, wait_on_rate_limit_notify = True)

df = pd.read_csv('s3://ian-twitter-data/dict.csv', index_col=False)

df = df.drop_duplicates(subset=['follower_id'])


df['follower_id'] = df['follower_id'].astype(str) 

user_list = df.follower_id.tolist()




header_list = ['ffollower_id','fffollower_id']

followers_dict = {}
follower_list=[]

i = 1 
with open('followers_followers.csv', 'a') as ff:
    ffwrite = csv.writer(ff)
    ffwrite.writerow(header_list)
    for user in user_list:
        print(user,i,len(user_list))
        i +=1
        
        try:
        
            user_follower = tweepy.Cursor(api.followers_ids, user_id = user )
            for page in user_follower.items():
                page = str(page)
                follower_list.append([user,page])
                ffwrite.writerows(follower_list)
                follower_list.clear()
                
                
                
        except tweepy.TweepError as e:
            print(e)
    ffwrite.writerows(follower_list)

print("end")

  #transfer to s3 using bash
command = "s3cmd put followers_followers.csv s3://ian-twitter-data/follwers_followers.csv"
process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
output, error = process.communicate() 