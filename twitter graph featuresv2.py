
import csv
from operator import itemgetter
import networkx as nx
from networkx.algorithms import community #This part of networkx, for community detection, needs to be imported separately.
import pandas as pd

with open('C:/Users/Ian/Desktop/Twitter Ids/node_list.csv', 'r') as nodecsv: # Open the file
    nodereader = csv.reader(nodecsv) # Read the csv
    # Retrieve the data (using Python list comprhension and list slicing to remove the header row, see footnote 3)
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



print(nx.info(G))




username_dict = {}
user_screen_name_dict = {}
tweetid_dict = {}
tweet_text_dict = {}
user_followers_count_dict = {}
user_friends_count_dict = {}
user_location_dict = {}
user_url_dict = {}
tweet_created_at_dict = {}
tweet_source_dict = {}




for node in nodes: # Loop through the list, one row at a time
    username_dict[node[0]] = node[1]
    user_screen_name_dict[node[0]] = node[2]
    tweetid_dict[node[0]] = node[3]
    tweet_text_dict[node[0]] = node[4]
    user_followers_count_dict[node[0]] = node[5]
    user_friends_count_dict[node[0]] = node[6]
    user_location_dict = node[7]
    user_url_dict = node[8]
    tweet_created_at_dict = node[9]
    tweet_source_dict = node[10]

#print(  user_followers_count_dict)

nx.set_node_attributes(G, username_dict, 'username')
nx.set_node_attributes(G, user_screen_name_dict, 'screen_name')
nx.set_node_attributes(G, tweetid_dict, 'tweetid')
nx.set_node_attributes(G, tweet_text_dict, 'tweet_text')
nx.set_node_attributes(G, user_followers_count_dict, 'follower_count')
nx.set_node_attributes(G, user_friends_count_dict, 'friend_count')
nx.set_node_attributes(G, user_location_dict, 'location')
nx.set_node_attributes(G, user_url_dict, 'URL')
nx.set_node_attributes(G, tweet_created_at_dict, 'tweet_creation_time')
nx.set_node_attributes(G, tweet_source_dict, 'tweet_source')


DG = nx.DiGraph()
DG.add_nodes_from(node_names)
DG.add_edges_from(edges)
total_nodes = DG.number_of_nodes()
print(total_nodes)


nx.set_node_attributes(DG, username_dict, 'username')
nx.set_node_attributes(DG, user_screen_name_dict, 'screen_name')
nx.set_node_attributes(DG, tweet_text_dict, 'tweet_text')
nx.set_node_attributes(DG, tweetid_dict, 'tweetid')
nx.set_node_attributes(DG, user_followers_count_dict, 'follower_count')
nx.set_node_attributes(DG, user_friends_count_dict, 'friend_count')
nx.set_node_attributes(DG, user_location_dict, 'location')
nx.set_node_attributes(DG, user_url_dict, 'URL')
nx.set_node_attributes(DG, tweet_created_at_dict, 'tweet_creation_time')
nx.set_node_attributes(DG, tweet_source_dict, 'tweet_source')








cluster_coeff = nx.clustering(DG)
print(type(cluster_coeff))

#dataframe with clustering coeffcient
dfclus = pd.DataFrame(list(cluster_coeff.items()),columns = ['user','clustering_coefficient'])

 

bw_centrality = nx.betweenness_centrality(DG, normalized=True)

print(type(bw_centrality))

#dataframe with betweenness score
dfbw = pd.DataFrame(list(bw_centrality.items()),columns = ['user','bw_score'])


#get the foloowers of each node
num_of_preds = []
for node in DG.nodes:
    num_of_preds.append([node , len(list(DG.predecessors(node)))])
    

 #dataframe of predecessor count
dfpredcount = pd.DataFrame(num_of_preds,columns = ['user','pred_link_count'])

print(dfpredcount.head(5))



def have_bidirectional_relationship(DG, node1, node2):     #https://stackoverflow.com/questions/64787089/is-there-a-way-to-find-a-bidirectional-relationship-between-nodes-in-networkx/64787324?noredirect=1#comment114549339_64787324
    return DG.has_edge(node1, node2) and DG.has_edge(node2, node1)

#print('biconnections**********************************')

biconnections = set()
for u, v in DG.edges():
    if u > v:  # Avoid duplicates, such as (1, 2) and (2, 1)
        v, u = u, v
    if have_bidirectional_relationship(DG, u, v):
        biconnections.add((u, v))

biconnections = list(biconnections )

# with open('bidirectional_links.csv', 'w' ,newline='') as bd:
#     bdwrite = csv.writer(bd)
#     bdwrite.writerow(["user1", "user2"])
#     bdwrite.writerows(biconnections)
    

col_names = ["user1", "user2"]


df= pd.DataFrame(biconnections, columns=col_names)

print(df.head(5))




dfuser1 = df[['user1']]
dfuser2 = df[['user2']]

dfuser1.columns=['user']
dfuser2.columns=['user']

df2 = dfuser1.append(dfuser2, ignore_index = True)

print(df2.head(5))

df2 = df2.groupby('user')['user'].count().reset_index(name='Bidirectional_count') 

# datframe bidriectional count 
df2 = df2.rename(columns = {"" : "Bidirectional_count"})


print(df2.head(5))

node_degrees = set()
for node in G.nodes:
    deg = G.degree(node)
    node_degrees.add((node,deg))
    
#print(node_degrees)    

node_degrees = list(node_degrees)


col_names2 = ["user", "degree"]

#df degree
df_deg = pd.DataFrame(node_degrees, columns=col_names2)


print(df_deg.info)







 

all_nodes = DG.nodes(data=True)
pred = []
out={}
foll_count = []

i=1
#followings to median neighbours followers use in edge
for node in DG.nodes:

    if len(list(DG.predecessors(node))) > 0:
        for pred_node in DG.predecessors(node):
           

            out[pred_node] = all_nodes[pred_node]['follower_count']

            a = DG.nodes[pred_node]['follower_count']

            i += 1
            foll_count.append([node, pred_node, a])
           
 
       
            pred.append([node , foll_count])


col_names3 = ["user", "pred_node", "med_of_neigh_followers"]
df_med = pd.DataFrame(foll_count, columns=col_names3)

df_med["med_of_neigh_followers"] = df_med["med_of_neigh_followers"].astype('int')

print(df_med.dtypes)

#datframe of median followers count
df_med = df_med.groupby(['user'])['med_of_neigh_followers'].median().reset_index(name='med_of_neigh_followers') 

print(df_med.head(5))






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

df_complete = pd.merge(df_complete,
                       df_deg[["user", "degree"]],
                       on = 'user',
                       how = 'left')

df_complete.fillna(0, inplace=True)

df_complete['bdirectional ratio'] = df_complete['Bidirectional_count'] / df_complete['pred_link_count']

print(df_complete.info())


df_complete.to_csv('graph_data.csv', index = False, header=True)
 
