from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import pandas as pd
import random as random
import numpy as np
import networkx as nx

import nklsgen

def extract_edgelist(G):
    nodes = list(G.nodes)
    edges = []
    
    for node in nodes:
        edges.append(sorted(list(G[node])))
        
    return dict(zip(nodes, edges))


def random_starts(landscape, points):
    items = list(landscape.items()).copy()
    picks = random.choices(items, k=points)
    
    start_points = [[pick[0], pick[1]] for pick in picks]
    #start_points = [[nklsgen.str_to_bin(pick[0]), pick[1]] for pick in picks]
    return start_points

def bin_to_str(code):
    string = str()
    for each in code:
        string += (str(each))
    return string


def str_to_bin(code):
    return [int(each) for each in code]


def distances_from_current(landscape_index, locs_bin):
    distances = [distance_from_many_to_one(point, locs_bin) for point in landscape_index]
    return distances


def distance_from_many_to_one(point, locs_bin): 
    distances = [euclidean_distance(point, loc) for loc in locs_bin] 
    return min(distances)


def euclidean_distance(a, b):
    val = np.sum([(b[i]-a[i])**2 for i in range(len(a))])
    return np.sqrt(val)

class Landscape():
    def __init__(self, landscape_, id_):
        self.id_ = id_
        self.landscape = landscape_
        
class Graph():
    def __init__(self, edgelist, id_, density=None):
        self.id = id_
        self.edgelist = edgelist
        self.density = density
        
class Agent():

    def __init__(self, search_cost=0, memory=None, myopic=False, 
                 edgelist=None, id_=None, start_point=None, landscape_index=None):
        
        self.id = id_
        self.myopic = myopic
        self.landscape_index = landscape_index
            
        self.search_cost = search_cost    
        self.memory = 0 if myopic else memory
        self.edgelist = edgelist
        self.neighbors = []
        
        start_point.append(self.id)
        start_point.append(0)
        self.search_hist = pd.DataFrame(columns=['Location', 'Value', 'AgentID', 'Step'])
        self.search_hist.loc[len(self.search_hist)] = start_point
        
        self.shared_search_hist = self.search_hist.copy()
        
        # GP kernel parameters
        self.kernel = RBF(length_scale=5,
                          length_scale_bounds=(1e-5,1e8))
        self.gp = GaussianProcessRegressor(kernel=self.kernel,
                                           n_restarts_optimizer=100)
        
        
    def update_shared_hist(self, teach=False, learn=True):
        for agent in self.neighbors:
            new_data = agent.search_hist.iloc[-1,:].values.tolist()
            self.shared_search_hist.loc[len(self.shared_search_hist)] = new_data
            
            
    def survey(self):                
        if self.myopic:
            amount = ((len(self.neighbors)+1) * 1)
            relevant_hist = self.shared_search_hist.sort_values(by=['Step'])[-amount:].copy()            
        if not self.myopic:
            amount = ((len(self.neighbors)+1) * self.memory)
            relevant_hist = self.shared_search_hist.sort_values(by=['Step'])[-amount:].copy()
            
        locs = relevant_hist['Location'].apply(nklsgen.str_to_bin).tolist()
        vals = relevant_hist['Value'].tolist()

        self.gp.fit(locs, vals)
        y_est, std = self.gp.predict(self.landscape_index, return_std=True)
        
        return y_est, std

    
    def choose_new_point(self): 
        
        # Expected Values + Curiosity
        y_est, std = self.survey()
        expected_value = [(a[0] + a[1]*0.50) for a in zip(y_est,std)] # <-- UNCERTAINTY VALUE HERE
        
        # Distances
        amount = ((len(self.neighbors)+1) * self.memory)
        relevant_hist = self.shared_search_hist.sort_values(by=['Step'])[-amount:].copy() 
        locs_bin = relevant_hist['Location'].apply(nklsgen.str_to_bin).tolist()
        distances = distances_from_current(self.landscape_index, locs_bin)
        
        ## Search Cost + Softmax Choice
        values = np.array([(a[0] - (a[1]*self.search_cost)) for a in zip(expected_value,distances)])
        tau = 0.02
        softmax = np.exp(values/tau) / sum(np.exp(values/tau))
        softmax_dict = dict(zip(softmax.copy(), values.copy()))
        options = dict(sorted(softmax_dict.items(), reverse=True))
        choice = random.choices(list(options.values()), weights=list(options.keys()), k=1) 
        loc = np.where(values == choice)[0][0]            

        return self.landscape_index[loc]
    
    
    def choose_and_move(self, landscape):
        new_loc_bin = self.choose_new_point()
        new_val = landscape[bin_to_str(new_loc_bin)]
        new_loc = bin_to_str(new_loc_bin)
        
        self.search_hist.loc[len(self.search_hist)] = [new_loc, new_val, self.id, len(self.search_hist)]
    
    
    def explore(self, landscape, steps):
        for _ in range(steps):
            self.choose_and_move(landscape)
        

class Population():
    
    def __init__(self, landscape, num_searches, graph,
                 search_costs, memorys, myopia,
                 pop_id=None, graph_id=None):

        self.landscape = landscape.copy()
        self.landscape_index = [nklsgen.str_to_bin(each) for each in list(self.landscape.keys())]
        self.num_searches = num_searches
        
        # Agents and Properties
        self.search_costs = search_costs
        self.memorys = memorys
        self.myopia = myopia
        self.graph = graph
        self.edgelist = self.graph.edgelist
        self.graph_id = self.graph.id
        
        # Create Agents
        self.agents = self.create_agents()
        
    def create_agents(self):
        starts = random_starts(self.landscape, len(edgelist))
        agents_ = []
        for i in range(len(edgelist)):
            agents_.append(Agent(self.search_costs[i], self.memorys[i], self.myopia, 
                                 self.edgelist[i], i, starts[i], self.landscape_index))
        
        for i in range(len(edgelist)):
            neighbors_ = [agents_[a] for a in self.edgelist[i]]
            agents_[i].neighbors = neighbors_
            agents_[i].update_shared_hist()
        
        return agents_
    
    
    def report_step_averages(self, step):
        
        agent_vals = []
        for agent in self.agents:
            agent_vals.append(agent.search_hist['Value'][step])
        
        return agent_vals
    
    
    def explore(self, sync=True):
        # Pop level explore function (round-based search)
        ## Could weave search and update together pretty easily
        ### Done by reversing the updating function
        
        for _ in range(self.num_searches):
            for agent in self.agents:
                agent.explore(self.landscape, 1)
                
                if not sync:
                    agent.update_shared_hist()
             
            if sync:
                for agent in self.agents:
                    agent.update_shared_hist()
                    
        print(f'Sim Complete!')
        
        
def run_model(p_combo):
    pop = Population(landscape = p_combo['landscape'],  
                     num_searches = p_combo['num_searches'], 
                     graph = p_combo['graphs'][0],
                     search_costs = p_combo['search_costs'],
                     memorys = p_combo['memorys'],
                     myopia = p_combo['myopia'],
                     pop_id = p_combo['pop_id'])
 
    pop.explore()
    
    return pop