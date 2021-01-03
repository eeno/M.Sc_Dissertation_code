# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 22:01:09 2020

@author: Ian
"""

import csv

import networkx as nx

import pandas as pd

with open('C:/Users/Ian/Desktop/Twitter Ids/user features.csv', 'r') as nodecsv: # Open the file
    nodereader = csv.reader(nodecsv) # Read the csv
    # Retrieve the data (using Python list comprhension and list slicing to remove the header row)
    nodes = [n for n in nodereader][1:]

node_names = [n[0] for n in nodes] # Get a list of only the node names

with open('C:/Users/Ian/Desktop/Twitter Ids/edge_list_complete.csv', 'r') as edgecsv: # Open the file
    edgereader = csv.reader(edgecsv) # Read the csv
    edges = [tuple(e) for e in edgereader][1:] # Retrieve the data

df = pd.read_csv("C:/Users/Ian/Desktop/Twitter Ids/predictions.csv")
    

df = df[df['prediction'] ==1 ]

bot_list = df['user'].astype(str).tolist()

print(len(node_names))

print(len(edges))

G = nx.Graph()

G.add_nodes_from(node_names)
G.add_edges_from(edges)

total_nodes = G.number_of_nodes()

print(nx.info(G))

print(total_nodes)

bot_score_dict = {}
follower_count_dict = {}
friend_count_dict = {}
staus_count_dict = {}
fofo_ratio_dict = {}
age_of_profile_dict = {}
posting_rate_dict = {}



for node in nodes: # Loop through the list, one row at a time
    bot_score_dict[node[0]] = node[1]
    follower_count_dict[node[0]] = node[2]
    friend_count_dict[node[0]] = node[3]
    staus_count_dict[node[0]] = node[4]
    fofo_ratio_dict[node[0]] = node[5]
    age_of_profile_dict[node[0]] = node[6]
    posting_rate_dict[node[0]] = node[7]
    

#print(  user_followers_count_dict)
nx.set_node_attributes(G, bot_score_dict, 'bot_score')
nx.set_node_attributes(G, follower_count_dict, 'follower_count')
nx.set_node_attributes(G, friend_count_dict, 'friend_count')
nx.set_node_attributes(G, staus_count_dict, 'staus_count')
nx.set_node_attributes(G, fofo_ratio_dict, 'fofo_ratio')
nx.set_node_attributes(G, age_of_profile_dict, 'age_of_profile')
nx.set_node_attributes(G, posting_rate_dict, 'posting_rate')


DG = nx.DiGraph()
DG.add_nodes_from(node_names)
DG.add_edges_from(edges)

nx.set_node_attributes(DG, bot_score_dict, 'bot_score')
nx.set_node_attributes(DG, follower_count_dict, 'follower_count')
nx.set_node_attributes(DG, friend_count_dict, 'friend_count')
nx.set_node_attributes(DG, staus_count_dict, 'staus_count')
nx.set_node_attributes(DG, fofo_ratio_dict, 'fofo_ratio')
nx.set_node_attributes(DG, age_of_profile_dict, 'age_of_profile')
nx.set_node_attributes(DG, posting_rate_dict, 'posting_rate')

print(nx.info(DG))



influence_list = []

influence = nx.eigenvector_centrality(DG)


for node in influence:
    if node in bot_list:
         influence_list.append([node,influence[node]])


#bot influnce list
df_inf = pd.DataFrame(influence_list, columns = ['user','eigenvetor_score'])
#save bot inlfuenc list to csv 
df_inf.to_csv("C:/Users/Ian/Desktop/Twitter Ids/influence_scores.csv", index = False, header=True)
# eigenvalue score for whole graph
dfev = pd.DataFrame(list(influence.items()),columns = ['user','Eigenvector score'])
#save to csv
dfev.to_csv("C:/Users/Ian/Desktop/Twitter Ids/all_influence_scores.csv", index = False, header=True)

