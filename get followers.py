"""
This will gather folowers for users contaned in a file

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


df = pd.read_csv('s3://ian-twitter-data/twitteroutput.csv', index_col=False)
 
df['user_id'] = df['user_id'].astype(str) 

print(df.user_id) 

user_list = df.user_id.tolist()

user_list = list(dict.fromkeys(user_list)) #remove duplicates from the list

del df

print(user_list)

header_list = ['user_id','follower_id']

followers_dict = {}
follower_list=[]
i = 1 
with open('dict.csv', 'a',newline='') as ff:
    ffwrite = csv.writer(ff)
    ffwrite.writerow(header_list)
    for user in user_list:
        print(user, i)
        i +=1
        try:
        
            user_follower = tweepy.Cursor(api.followers_ids, user_id = user )
            for page in user_follower.items():
                
                follower_list.append([user,page])
                ffwrite.writerows(follower_list) 
                follower_list.clear()
                
                
               
                    
        except tweepy.TweepError as e:
            print(e)
    
#print(followers_dict)
print(follower_list)

  #transfer to s3 using bash
command = "s3cmd put dict.csv s3://ian-twitter-data/dict.csv"
process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
output, error = process.communicate() 




