# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 13:14:38 2020

@author: Ian
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np     
import statistics

df = pd.read_csv('C:/Users/Ian/Desktop/Twitter Ids/tweet_details_complete.csv', index_col=False)

#remove URL
df['tweet_text_clean'] = df['tweet_text'].replace('https?://[A-Za-z./]*',' ',regex=True)

#rmove special characters i.e everything that isnt a letter or a number.
df['tweet_text_clean'] = df['tweet_text_clean'].replace('[^A-Za-z0-9]+',' ',regex=True)

#filter tthe datframe so that tehre are no empty tweet_text_clean col values
df = df[df['tweet_text_clean'] != ' ' ]
user_list = df.user.unique()
sim_list = []


for userid in user_list:
    df_tweet = df[df['user'] == userid]
    corpus = df_tweet['tweet_text_clean'].to_list()
    corpus = [str(i) for i in corpus]
    print(len(corpus))
    print(max(corpus,key=len))
    if len(corpus) > 1 and len(max(corpus,key=len)) > 1:
        print(corpus)
        vect = TfidfVectorizer(min_df=1)
        tfidf = vect.fit_transform(corpus) 
        cosine_similarity = tfidf * tfidf.T 
        arr = cosine_similarity.toarray()     
        print(arr)
        np.fill_diagonal(arr, np.nan)
        arr = arr[~np.isnan(arr)]
        print(arr)
        tweet_sim  = np.mean(arr)
        sim_list.append([userid, tweet_sim])
    
print(sim_list)    
    
dftwsim = pd.DataFrame(sim_list,columns=('user', 'tweet_sim_score'))

dftwsim.to_csv('C:/Users/Ian/Desktop/Twitter Ids/tweet_simiarity scores.csv')




