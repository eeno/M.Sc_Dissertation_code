"""
Your module description
"""
import boto3
import subprocess
import csv
import pandas as pd

# Iterates through all the objects, doing the pagination for you. Each obj
# is an ObjectSummary, so it doesn't contain the body. You'll need to call
# get to get the whole body.
header_list = ['user_id','user_name','user_screen_name','tweet_id','tweet_text','user_followers_count','user_friends_count','user_location','user_url','tweet_created_at','tweet_source']

s3 = boto3.resource('s3')
bucket = s3.Bucket('ian-twitter-data')

for obj in bucket.objects.all():
    key = obj.key
    body = obj.get()['Body'].read()
    #print(type(body))
    new_tweet =  body.decode("utf-8")
    with open('twitteroutput.txt', 'a', encoding='utf-8') as tf:
        tf.write(new_tweet)
    
    with open('twitteroutput.txt', 'r', encoding='utf-8') as tft:
        in_reader = csv.reader(tft,delimiter = '\t')
        with open('twitteroutput.csv', 'w') as out_csv:
            out_writer =  csv.writer(out_csv)
            out_writer.writerow(header_list)
            for row in in_reader:
                out_writer.writerow(row)
    




    #transfer to s3 using bash
command = "s3cmd put twitteroutput.csv s3://ian-twitter-data/twitteroutput.csv"
process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
output, error = process.communicate()    
    
   
cli = boto3.client('s3')