import random
import pickle

# Edgelists
premade = True
graph_file = 'DesignedGraphs.pickle'
if premade:
    with open(graph_file, 'rb') as f:
        edgelists = pickle.load(f) 
        
class Graph():
    def __init__(self, edgelist, id_):
        self.id = id_
        self.edgelist = edgelist
graphs = []
for i, edgelist in enumerate(edgelists):
    graphs.append(Graph(edgelist, i))

# Sim Properties
num_searches = [10]
num_sims = 1    
    
# Landscape Properties     
N = [11]
K = [2,3]
num_of_ls = [32]

# Agent Properties
num_agents = len(list(edgelists[0].keys()))
search_costs_ = [0]
search_costs = [[i for _ in range(num_agents)] for i in search_costs_]
memorys_ = [2,3,4]
memorys = [[i for _ in range(num_agents)] for i in memorys_]
myopia = [False]


