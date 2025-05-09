import pandas as pd
import numpy as np
import networkx as nx
import time
import random

import warnings
warnings.filterwarnings("ignore") # GP struggles to converge sometimes, ignored for now

# Graphs
edgelist = pd.read_csv('data/edgelists.csv')
graphids_ = {0: 'Max Max Betweenness', 1: 'Min Max Closeness', 2: 'Max Avg Betweenness',
             3: 'Max Avg Clustering', 4: 'Max Var Constraint', 162: 'Min Avg Betweenness',
             163: 'Max Max Closeness', 166: 'Min Avg Clustering'}
graphs = {}
for graph in edgelist['graphid'].unique():
    temp = edgelist[edgelist['graphid']==graph].copy()
    edges = list(zip(temp['node_a'], temp['node_b']))
    graphs[graphids_[graph]] = nx.from_edgelist(edges)
solo = nx.Graph()
solo.add_nodes_from([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
graphs['Independent'] = solo

# Landscapes
lsdf = pd.read_csv('data/MasonWattsLS.csv')
landscapes = {}
for probid in lsdf['probid'].unique():
    temp = lsdf[lsdf['probid']==probid][['x','y','score']].copy()
    landscapes[probid] = np.array(temp.pivot(index='x', columns='y', values='score')) / 100
landscape_index = [loc for loc, val in np.ndenumerate(landscapes[1043])]

# Experiment Data
df = pd.read_csv('data/MasonWattsExpData.csv', low_memory=False)
df = df[['node','guessX','guessY','pts_earned','expid','trial','round','graphid']].copy()
df = df[(df['guessX']<100)&(df['guessY']<100)].copy()
df.reset_index(drop=True,inplace=True)

