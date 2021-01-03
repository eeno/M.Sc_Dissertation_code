

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/Ian/Desktop/Twitter Ids/complete data2.csv', index_col=False)

user_list = df.user.tolist()

df.loc[(df['botscore'] >= .75),'bot_indicator)' ] = 1
#df['bot_indicator'] = np.where((df['botscore'] >= .75, 1))



#df['High Value Indicator'] = np.where((df.Value_1 > 1000) | (df.Value_2 > 15000), 'Y', 'N')

df2 = pd.read_csv('C:/Users/Ian/Desktop/Twitter Ids/tweet_simiarity scores.csv', index_col=False)
print(df2.shape)
df2 = df2[df2['user'].isin(user_list)]

print(df.shape)
print(df2.shape)

df3 = pd.merge(df,
               df2[['user','tweet_sim_score']],
               on =  'user',
               how = 'left')


df3.fillna(0, inplace=True)
print(df3.shape)

df3.to_csv('C:/Users/Ian/Desktop/Twitter Ids/all_model_data.csv', index = False, header=True)
