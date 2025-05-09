#!/usr/bin/env python
# coding: utf-8

import warnings
import sys
import multiprocessing
import networkx as nx
from sklearn.model_selection import ParameterGrid
from matplotlib import pyplot as plt
import time
import os

# ABM Module
from ABM_Func import *
# Input Parameters
from input_parameters import *

# New Parallelization Package
from mpi4py import MPI
# Retrieves MPI environment
comm = MPI.COMM_WORLD
# set variable "size" as the total number of MPI processes
size = comm.Get_size()
# set variable "rank" as the specific MPI rank on each MPI process
rank = comm.Get_rank()


#### Rank==0 Prep Phase (includes testing code)
if rank == 0:

    from input_parameters import *

    start = time.time()

    # Exception graph construction for single agents
    if num_agents == 1:
        
        graphs = [nx.Graph() for _ in range(num_graphs)]
        for graph in graphs:
            graph.add_node(1)
            
        graph_labels = dict(zip(graphs, range(len(graphs)))) 

    # Create graph/graphs for multiple agents
    elif num_agents > 1:
        
        graphs = []     
        if len(edges) != 1:
            for edge in edges:
                graphs.append([nx.gnm_random_graph(num_agents, edge) for _ in range(num_graphs)])
            graphs = [graph[i] for graph in graphs for i in range(len(graph))] 
 
 
        # Specially designed graphs (CURRENTLY 16 OF THEM)
        else:
            if edges == [0]:
                df = pd.read_csv('DesignedGraphs_Complete.csv')
                df.drop(labels=['Unnamed: 0'], axis=1, inplace=True)
                
                #graphs = []
                for i in range(df.shape[1]):
                    graphs.append(nx.from_edgelist(list(eval(df.iloc[0,i]))))   
        
        graph_labels = dict(zip(graphs, range(len(graphs))))    
            
    # Create Parameter Grid
    parameter_grid = {'landscape': landscapes, 
                    'num_agents': [num_agents], 
                    'num_searches': [num_searches], 
                    'graph': graphs, 
                    'memory': memory,
                    'distance_costs': distance_costs,
                    'myopia': [myopia],
                    'sync': [True]}

    # Create Parameter Space
    parameter_combos = list(ParameterGrid(parameter_grid))
    parameter_combos = [a for a in parameter_combos.copy() for _ in range(num_sims)]

    # Labels Param_Combos with IDs for Identification
    for i, pcombo in enumerate(parameter_combos):
        # Add id key
        pcombo.update({'id': i})
        # Add graph_label key
        pcombo.update({'graph_label': graph_labels[pcombo['graph']]})

    #Shuffle so batches aren't unbalanced
    random.shuffle(parameter_combos)

    print(f'This batch contains {len(parameter_combos)} simulations!')

    ## Split Parameter Space for Distribution
    split_seeds = np.array_split(parameter_combos, size, axis=0)

else:
    split_seeds = None
    start = None
    i = None


# Distributed Operations
parameter_space_ = comm.scatter(split_seeds, root = 0)

if len(parameter_space_) != 0:

    ### Run model on parameter_space_
    warnings.filterwarnings("ignore") # GP struggles to converge sometimes, ignored for now
    _sim_history = []
    
    for each in parameter_space_:
        _sim_history.append(run_model(each)) ### <-- MYOPIA PARAMETER HERE

    folder_ = sys.argv[1]

    
### NODE-SPECIFIC DATA PROCESSING

    ## PERFORMANCE DATA
    sim_index = []
    step_index = []
    avg_value = []
    distance_costs_= []
    densities_ = []
    landscapes_ = []
    ids_ = []
    graphids_ = []
    
    for i, pop in enumerate(_sim_history):
        
        filename_ = f'./{folder_}/landscapes/'
        pop.landscape.tofile(f'{filename_}{pop.id_}_landscape.csv', sep=',')
        
        for j in range(pop.num_searches+1):
            
            sim_index.append(f'{rank}_{i+1}')
            step_index.append(j)
    
            agent_vals = []
            for agent in pop.agents:
                agent_vals.append(agent.search_hist['Value'][j])
    
            avg_value.append((sum(agent_vals) / len(agent_vals)))
    
            distance_costs_.append(pop.distance_cost)
    
            densities_.append(nx.density(pop.graph))
    
            landscapes_.append(pop.landscape_file)
    
            ids_.append(pop.id_)
            
            graphids_.append(pop.graphid)
            
    data = {'Sim_Index': sim_index, 
            'Step_Index': step_index, 
            'AVG_Value': avg_value, 
            'Distance_Cost': distance_costs_,
            'Densities': densities_,
            'Landscape': landscapes_,
            'IDs': ids_,
            'GraphID': graphids_}
    
    
    if len(_sim_history) != 0:    
        performance_df = pd.DataFrame(data)
        
        filename_ = f'./temp_data/rank{rank}'
        performance_df.to_csv(f'{filename_}_performance.csv', index=False)
    
    
    ## HISTORICAL DATA
    agent_hists = []
    
    for i, pop in enumerate(_sim_history):
    
        agent_id = 0
        for agent in pop.agents:
            temp = agent.search_hist.copy()
            #print(temp)
    
            temp['Memory'] = agent.memory
            temp['IDs'] = f'{pop.id_}'
            temp['Agent_ID'] = agent_id
            temp['GraphID'] = pop.graphid
    
            agent_hists.append(temp)
    
            agent_id += 1
    
    if len(_sim_history) != 0:    
        histdata_df = pd.concat(agent_hists, ignore_index=True)
        
        filename_ = f'./temp_data/rank{rank}'
        histdata_df.to_csv(f'{filename_}_histdata.csv', index=False)



### BARRIER() TO SYNCHRONIZE
comm.Barrier()


### RECONSTITUTION OF NODE-SPECIFIC DATA

if rank == 0:
    
    # Name output files
    filename = ''
    for i in range(1, len(sys.argv) - 1):
        filename += f'{sys.argv[i]}_'
    
    folder_ = sys.argv[1]
    
    # Timing Report
    end = time.time()
    time_ = round(end-start, 0) / 60
    
    with open(f'{time_}_{filename}.txt', 'w') as f:
        f.write(f'{size}\n')
        f.write(f'{time_}\n')
    
    
    
    # Read through and concat PERFORMANCE data
    dfs = []
    for (root,dirs,files) in os.walk(f'./temp_data/', topdown=True):
        for file in files:
            if 'performance' in file:
                dfs.append(pd.read_csv(f'./temp_data/{file}'))

    final_perf_df = pd.concat(dfs)
    
    # Export Final Data
    final_perf_df.to_csv(f'{folder_}/performance.csv', index=False)    
    #final_perf_df.to_csv(f'{filename}_performance.csv', index=False)



    # Read through and concat HISTORICAL data
    dfs = []
    for (root,dirs,files) in os.walk(f'./temp_data/', topdown=True):
        for file in files:
            if 'histdata' in file:
                dfs.append(pd.read_csv(f'./temp_data/{file}'))

    final_hist_df = pd.concat(dfs)
    
    # Export Final Data
    final_hist_df.to_csv(f'{folder_}/histdata.csv', index=False)
    #final_hist_df.to_csv(f'{filename}_histdata.csv', index=False)



else: 
    end = None



