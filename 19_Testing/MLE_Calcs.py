import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
import networkx as nx
import time
import random
import os
import pickle
import math

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import ParameterGrid
import warnings
warnings.filterwarnings("ignore") # GP struggles to converge sometimes, ignored for now

# Custom Functions
import utils

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Data loaded in all subprocesses
from input_parameters import *

# Prep for Scatter()
if rank == 0:
    
    start = time.time()

    trial_ids = utils.get_trial_ids(df)
    # trial_data = [utils.get_tg_data(df,trial_id,graphs) for trial_id in trial_ids] 

    # #PRIMARY PARAM SETUP
    # agent_param_grid = {'memory': [a for a in range(1,20)], #[a for a in range(1,51)]
    #                     'ei': [0]+[random.uniform(0,0.5) for _ in range(30)], #[0]+[random.uniform(0,0.5) for _ in range(30)]
    #                     'search_cost': [0]+[random.uniform(0,0.5) for _ in range(30)], #[0]+[random.uniform(0,0.5) for _ in range(30)]
    #                   }
    # agent_params = list(ParameterGrid(agent_param_grid))
    # print(f"This job contains {len(agent_params)} model parameter presets!")
    
    # SECONDARY PARAM SETUP
    temp = [{'ei':random.uniform(0,0.5),
             'search_cost':random.uniform(0,0.5),
             'tau':random.uniform(0,1)} 
             for a in range(10)]
    agent_params = []
    for temp_ in temp:
        for mem in range(5,6):
            new = temp_.copy()
            new['memory'] = mem
            agent_params.append(new)
    print(f"This job contains {len(agent_params)} model parameter presets!")
    
    parameter_grid = {'trial_id': trial_ids,
                    'agent_params': agent_params,
                    'landscape_index': [landscape_index]}
    parameter_combos = list(ParameterGrid(parameter_grid))

    random.shuffle(parameter_combos)

    print(f'This batch contains {len(parameter_combos)} jobs!')

    split_seeds = np.array_split(parameter_combos, size, axis=0)

else:
    split_seeds = None
    start = None
    i = None

# SCATTER() 
parameter_space_ = comm.scatter(split_seeds, root=0)

if len(parameter_space_) != 0:
    warnings.filterwarnings("ignore") # GP struggles to converge sometimes, ignored for now
    parameter_space_ = utils.process_data(parameter_space_, df, graphs)
    
    output_dfs = [pd.DataFrame(each['results']) for each in parameter_space_]
    partial_df = pd.concat(output_dfs, ignore_index=True)
    partial_df.to_csv(f"./temp_data/rank{rank}_estimates.csv", index=False)


### BARRIER() TO SYNCHRONIZE
comm.Barrier()

# # GATHER() to collect data
# parameter_space = comm.gather(parameter_space_,root=0)

if rank == 0:

    # Read through and concat PERFORMANCE data
    dfs = []
    for (root,dirs,files) in os.walk(f'./temp_data/', topdown=True):
        for file in files:
            if 'estimates' in file:
                dfs.append(pd.read_csv(f'./temp_data/{file}'))
 
    final_df = pd.concat(dfs, ignore_index=True)
    final_df['6'] = final_df['5'].apply(np.log10)
    final_df = final_df[['4','6']].groupby(['4']).sum()
    final_df = final_df.reset_index()
    final_df.to_csv('output/results.csv', index=False)
    
    end = time.time()
    print(f"Total Time Taken: {end-start} seconds!")
    print(f"Distributed over {size-1} cores!")
    print(f"Time per job: {(end-start)/(size-1)} seconds!")


else:
    end = None




    
