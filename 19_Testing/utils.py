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

# Misc
def im_late(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        data = func(*args, **kwargs)
        time_taken = time.time() - start
        print(f'Time Taken: {time_taken} seconds!')
        return data
    return wrapper

def vectorized_dist(curr_loc, neigh_locs, landscape_index):
    curr_dists = np.array([math.inf for _ in range(len(landscape_index))]).reshape(-1,1) 
    neigh_dists = np.array([math.inf for _ in range(len(landscape_index))]).reshape(-1,1)
    if len(neigh_locs) > 0:
        neigh_dists = sp.spatial.distance.cdist(landscape_index, neigh_locs, 'cityblock') + 1 #are neighbors free to move to?
    if len(curr_loc) > 0:
        curr_dists = sp.spatial.distance.cdist(landscape_index, curr_loc, 'cityblock')
    all_dists = np.concatenate([curr_dists, neigh_dists], axis=1)
    return all_dists.min(axis=1)

# Outer Functions
def get_trial_ids(df):
    # This returns a list of tuples [(id,trial#),...]
    trial_ids = [(id,trial) for id in df['expid'].unique() for trial in df[df['expid']==id]['trial'].unique()]
    return trial_ids

def get_trial_df(df,trial_id):
    # This returns a trimmed df with data relevant to a specific trial
    return df[(df['expid']==trial_id[0]) & (df['trial']==trial_id[1])].copy() 

def get_graph(trial_df,graphs):
    # This returns the graph associated with a specific trial
    return graphs[trial_df['graphid'].unique()[0]].copy()

def get_tg_data(df,trial_id,graphs):
    # This returns a dictionary containing 'Trial_Data_DF', 'NX_Graph'
    trial_df = get_trial_df(df,trial_id)
    return trial_df, get_graph(trial_df,graphs)
    # return {'Trial_Data_DF': trial_df, 'NX_Graph': get_graph(trial_df,graphs)}

# Inner Functions
def isolate_ego_net(node,graph,df):
    # nodes = [node] + list(graph.neighbors(node)) #Include neighbors
    nodes = [node]                               #Don't include neighbors
    return df[df['node'].isin(nodes)].copy()

def get_round_range(subnet,node):
    range_ = subnet[subnet['node']==node]['round'].unique().tolist()
    range_.remove(min(range_))
    return range_
    
def prob_to_choose(subnet,round_,node,gp,agent_params,landscape_index):
    new_point_ = subnet[(subnet['round']==round_) & (subnet['node']==node)].copy()
    new_point = list(zip(new_point_['guessX'],new_point_['guessY']))[0]
    inst = subnet[subnet['round']<round_].sort_values(by='round')[-agent_params['memory']:].copy()
    # Input Data
    locs = list(zip(inst['guessX'],inst['guessY']))
    vals = inst['pts_earned'].tolist()
    # Fit & Trim
    gp.fit(locs,vals)
    y_est, std = gp.predict(landscape_index, return_std=True)
    # y_est[y_est < 0.001] = 0
    # Curiosity
    values = y_est + std*agent_params['ei']
    # Normalization
    values = values / (values.max())
    # values[values < 0] = 0
    # Search Cost
    self_prior = subnet[(subnet['node']==node) & (subnet['round']<round_)]['round'].max()
    curr_loc_df = inst[(inst['node']==node) & (inst['round']==self_prior)].copy()
    curr_loc = np.array(list(zip(curr_loc_df['guessX'],curr_loc_df['guessY'])))
    # print(round_,self_prior,curr_loc)

    neigh_loc_df = inst.drop(inst[(inst['node']==node) & (inst['round']==self_prior)].index).copy()
    neigh_locs = np.array(list(zip(neigh_loc_df['guessX'],neigh_loc_df['guessY'])))
    distances = vectorized_dist(curr_loc, neigh_locs, np.array(landscape_index.copy()))
    # print(distances)
    search_loss = distances * np.array(agent_params['search_cost'])
    values = values - search_loss
    values[values < 0] = 0.001
    
    # Softmax
    # tau = 0.02
    tau = agent_params['tau'] #Tau as variable parameter
    softmax = np.exp(values/tau) / sum(np.exp(values/tau))
    prob = softmax[landscape_index.index(new_point)]
    return prob

def process_data(parameter_combos, df, graphs):
    for p_combo in parameter_combos:
        results = []
        kernel = RBF(length_scale=5,length_scale_bounds=(1e-5,1e8))
        gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=10)
        
        data, graph = get_tg_data(df,p_combo['trial_id'],graphs)
        
        # data = p_combo['trial_data']['Trial_Data_DF']
        # graph = p_combo['trial_data']['NX_Graph']
        expid = data['expid'].unique()[0]
        trial = data['trial'].unique()[0]
        
        landscape_index = p_combo['landscape_index']
        agent_params = p_combo['agent_params']
    
        for node in data['node'].unique():
            subnet = isolate_ego_net(node,graph,data)
            range_ = get_round_range(subnet,node)
    
            for i,round_ in enumerate(range_):
                prob = prob_to_choose(subnet,round_,node,gp,agent_params,landscape_index)
                # print(f"Experiment: {expid} \nTrial: {trial} \nNode: {node} \nRound: {round_} \nProbability: {prob}")
                results.append([expid,trial,node,round_,agent_params,prob])
        p_combo['results'] = results

    return parameter_combos
