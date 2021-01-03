
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


cluster_coeff = nx.clustering(DG)
#dataframe with clustering coeffcient
dfclus = pd.DataFrame(list(cluster_coeff.items()),columns = ['user','clustering_coefficient'])

dfclus['user'] = dfclus['user'].astype('int64')


print("clustering done done")





bw_centrality = nx.betweenness_centrality(DG, normalized=True)

#dataframe with betweenness score
dfbw = pd.DataFrame(list(bw_centrality.items()),columns = ['user','bw_score'])

dfbw['user'] = dfbw['user'].astype('int64')



#get the followers of each node
num_of_preds = []
for node in DG.nodes:
    num_of_preds.append([node , len(list(DG.predecessors(node)))])
    

  #dataframe of predecessor count
dfpredcount = pd.DataFrame(num_of_preds,columns = ['user','pred_link_count'])


def have_bidirectional_relationship(DG, node1, node2):    
    return DG.has_edge(node1, node2) and DG.has_edge(node2, node1)



biconnections = set()
for u, v in DG.edges():
    if u > v:  # Avoid duplicates, such as (1, 2) and (2, 1)
        v, u = u, v
    if have_bidirectional_relationship(DG, u, v):
        biconnections.add((u, v))

biconnections = list(biconnections )

col_names = ["user1", "user2"]

df= pd.DataFrame(biconnections, columns=col_names)


dfuser1 = df[['user1']]
dfuser2 = df[['user2']]

dfuser1.columns=['user']
dfuser2.columns=['user']

df2 = dfuser1.append(dfuser2, ignore_index = True)


df2 = df2.groupby('user')['user'].count().reset_index(name='Bidirectional_count') 

# datframe bidriectional count 
df2 = df2.rename(columns = {"" : "Bidirectional_count"})

node_degrees = set()
for node in G.nodes:
    deg = G.degree(node)
    node_degrees.add((node,deg))
    
#print(node_degrees)    

node_degrees = list(node_degrees)

print("degrees done done")
col_names2 = ["user", "degree"]

#df degree
df_deg = pd.DataFrame(node_degrees, columns=col_names2)

df_deg['user'] = df_deg['user'].astype('int64')

all_nodes = DG.nodes(data=True)
#out[pred_node] = all_nodes[pred_node]['follower_count']
pred = []
foll_count = []
out={}
for node in DG.nodes:

    if len(list(DG.predecessors(node))) > 0:
        for pred_node in DG.predecessors(node):
           

            

            node_foll_count = DG.nodes[pred_node]['follower_count']

            
            foll_count.append([node, pred_node, node_foll_count])
           
 

col_names3 = ["user", "pred_node", "med_of_neigh_followers"]
df_med = pd.DataFrame(foll_count, columns=col_names3)


df_med["med_of_neigh_followers"] = df_med["med_of_neigh_followers"].astype('int')

#datframe of median followers count
df_med = df_med.groupby(['user'])['med_of_neigh_followers'].median().reset_index(name='med_of_neigh_followers') 




df_complete = pd.merge(dfclus,
                        dfbw[['user','bw_score']],
                        on = 'user',
                        how = 'left')



df_complete = pd.merge(df_complete,
                        dfpredcount[['user','pred_link_count']],
                        on = 'user',
                        how = 'left')





df_complete = pd.merge(df_complete,
                        df2[['user','Bidirectional_count']],
                        on = 'user',
                        how = 'left')


df_complete = pd.merge(df_complete,
                        df_med[["user","med_of_neigh_followers"]],
                        on = 'user',
                        how = 'left')


df_complete.fillna(0, inplace=True)



df_complete['bdirectional ratio'] = df_complete['Bidirectional_count'] / df_complete['pred_link_count']

df_user_deet = pd.read_csv('C:/Users/Ian/Desktop/Twitter Ids/user features.csv', index_col=False )

df_user_deet['user'] = df_user_deet['user'].astype('int64')

df_complete = pd.merge(df_complete,
                        df_user_deet[['user','botscore','follower_count','friend_count', 'staus_count', 'fofo_ratio','age_of_profile', 'posting_rate']],
                        on = 'user',
                        how = 'left')

df_complete.fillna(0, inplace=True)

df_complete.to_csv('C:/Users/Ian/Desktop/Twitter Ids/complete data2.csv', index = False, header=True)



