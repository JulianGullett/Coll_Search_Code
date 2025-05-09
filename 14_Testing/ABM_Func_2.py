
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import pandas as pd
import random as random
import numpy as np

class Agent():
    
    # Single actor who can search and share info with neighbors
    ## Contains survey/explore functions (evaluation of landscape and choice of new position)
    
    def __init__(self, landscape, points = 1, distance_cost = 0, memory = None, search_hist = None, neighbors = None):
        
        self.landscape = landscape
        if memory == None:
            self.memory = 10**10
        else:
            self.memory = memory
        
        self.distance_cost = distance_cost
        
        if search_hist is None:
            self.search_hist = pd.DataFrame(columns=['Location', 'Value', 'Step'])
            self.random_start(points)
        
        self.current_position = self.search_hist['Location'].iloc[-1]
        
        if neighbors is None:
            self.neighbors = []
            
        self.shared_search_hist = pd.DataFrame(columns=['Location', 'Value', 'Step'])
        
        self.kernel = RBF(length_scale=5,
                          length_scale_bounds=(1e-5,1e8))
        self.gp = GaussianProcessRegressor(kernel=self.kernel,
                                           n_restarts_optimizer=100)
        
        self.start_time = 0
        self.end_time = 0
        
        
    def random_start(self, points):
        for i in range(points):
            temp_loc = [random.choice(range(1,int(len(self.landscape)**(1/2)+1))) for _ in range(2)]
            temp_val = self.landscape[temp_loc[0], temp_loc[1]]
            self.search_hist.loc[len(self.search_hist)] = [temp_loc, temp_val, i]
        
        
    def add_to_shared_from(self, agent_, reverse = False):
        #temp_df = agent_.search_hist.copy()
        #self.shared_search_hist = pd.concat([self.shared_search_hist, temp_df], ignore_index=True)
        
        if reverse:
            agent_.shared_search_hist.loc[len(agent_.shared_search_hist)] = self.search_hist.iloc[-1,:].values.tolist()
        else:
            self.shared_search_hist.loc[len(self.shared_search_hist)] = agent_.search_hist.iloc[-1,:].values.tolist()
        
        
    def update_shared_searches(self, reverse = False):        
        #self.shared_search_hist = pd.DataFrame(columns=['Location', 'Value', 'Step'])
        #self.shared_search_hist = pd.concat([self.shared_search_hist, self.search_hist.copy()], ignore_index=True)
        self.shared_search_hist.loc[len(self.shared_search_hist)] = self.search_hist.iloc[-1,:].values.tolist()
        
        if self.neighbors != []:
            for neighbor in self.neighbors:
                self.add_to_shared_from(neighbor, reverse)
        
        
    def update_neighbors(self, neighbors_list):
        self.neighbors = neighbors_list
        
    
    def add_neighbor(self, new_neighbor):
        self.neighbors.append(new_neighbor)
        
    
    def remove_neighbor(self, neighbor):
        self.neighbors.remove(neighbor)
        
        
    def survey(self, shared=True):
        
        # CHANGE SHARED TO FALSE FOR INDIVIDUAL TESTING
        
        #start_time = time.time()
        
        temp_search_hist = self.shared_search_hist.sort_values(by=['Step'])[-((len(self.neighbors)+1)*self.memory):]
            
        if shared:
            self.gp.fit(temp_search_hist['Location'].tolist(),
                        temp_search_hist['Value'].tolist())
            landscape_index = [list(a) for a in self.landscape.index]           
        else:
            self.gp.fit(self.search_hist['Location'].tolist(),
                        self.search_hist['Value'].tolist())
            landscape_index = [list(a) for a in self.landscape.index]
        
        #end_time = time.time()
        #(f'Survey time required: {end_time - start_time}')
        
        y_est, std = self.gp.predict(landscape_index, return_std=True)
        return y_est, std
    
    
    def choose_max_point(self, shared=False): 
        
        # This includes y_est, multiplier*std(uncertainty value), and -multiplier*distance from current point
        
        y_est, std = self.survey()
        expected_value = [(a[0] + a[1]*0.50) for a in zip(y_est,std)] # <-- UNCERTAINTY VALUE HERE
        
        distances = self.distances_from_current()

        
        ## THIS IS FOR GP-UCB AGENTS
        values = np.array([(a[0] - (a[1]*self.distance_cost)) for a in zip(expected_value,distances)])
        tau = 0.02
        softmax = np.exp(values/tau) / sum(np.exp(values/tau))
        softmax_dict = dict(zip(softmax.copy(), values.copy()))
        options = dict(sorted(softmax_dict.items(), reverse=True))
        choice = random.choices(list(options.values()), weights=list(options.keys()), k=1) 
        loc = np.where(values == choice)[0][0]
        
#         ### THIS IS FOR MYOPIC AGENTS
#         values = np.array([1 if a[1] <= 1.0 else 0 for a in zip(expected_value,distances)])
#         choice = random.choices(values, weights=values, k=1)
#         locs = np.where(values == choice)[0].tolist()
#         loc = random.choices(locs, k=1)[0]
#         #print(loc)

        
        if shared:
            landscape_index = [list(a) for a in self.landscape.index]
        else:
            landscape_index = [list(a) for a in self.landscape.index]
            
        return landscape_index[loc]
        
        
    def distance_2_points(self, a, b):
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    
    
    def distance_from_many_to_one(self, new_point, shared=True):  
        #### BE CAREFUL, THIS WONT WORK IF YOU INITIALIZE AGENTS OUTSIDE OF POPULATIONS
        
        # Agents wont have shared search histories, make shared=False
        ## I think this is fixed now, requires specific testing
        
        if shared:
            distances = [self.distance_2_points(a, new_point)
                         for a in self.shared_search_hist['Location'].tolist()] 
        else:
            distances = [self.distance_2_points(a, new_point)
                         for a in self.search_hist['Location'].tolist()]

        return min(distances)
    
    
    def distances_from_current(self):
        
        landscape_index = [list(a) for a in self.landscape.index]
        
        distances = [self.distance_from_many_to_one(each) for each in landscape_index]
        
        return distances
    
    
    def move_across_landscape(self):
        #Max value location
        new_location = self.choose_max_point()
        
        #Determine actual value of new location
        new_val = self.landscape[new_location[0], new_location[1]]
        
        #Add new location to search_hist
        self.search_hist.loc[len(self.search_hist)] = [new_location, new_val, len(self.search_hist)]
        
        #Change current_position
        ##Auto-updates based on .iloc[-1]
        
#         global progress_counter
#         global complete_
#         print(f'{progress_counter}/{complete_} steps taken!')
#         print(f'Job {round(progress_counter/complete_, 3)*100}% complete!')
#         progress_counter += 1
        
    def explore(self, steps):
        # Repeated move_across_landscape()
        for _ in range(steps):
            self.move_across_landscape()


class Agent_Myopic():
    
    # Single actor who can search and share info with neighbors
    ## Contains survey/explore functions (evaluation of landscape and choice of new position)
    
    def __init__(self, landscape, points = 1, distance_cost = 0, memory = 1, search_hist = None, neighbors = None):
        
        self.landscape = landscape
        if memory == None:
            self.memory = 10**10
        else:
            self.memory = memory
        
        self.distance_cost = distance_cost
        
        if search_hist is None:
            self.search_hist = pd.DataFrame(columns=['Location', 'Value', 'Step'])
            self.random_start(points)
        
        self.current_position = self.search_hist['Location'].iloc[-1]
        
        if neighbors is None:
            self.neighbors = []
            
        self.shared_search_hist = pd.DataFrame(columns=['Location', 'Value', 'Step'])
        
        self.kernel = RBF(length_scale=5,
                          length_scale_bounds=(1e-5,1e8))
        self.gp = GaussianProcessRegressor(kernel=self.kernel,
                                           n_restarts_optimizer=100)
        
        self.start_time = 0
        self.end_time = 0
        
        
    def random_start(self, points):
        for i in range(points):
            temp_loc = [random.choice(range(1,int(len(self.landscape)**(1/2)+1))) for _ in range(2)]
            temp_val = self.landscape[temp_loc[0], temp_loc[1]]
            self.search_hist.loc[len(self.search_hist)] = [temp_loc, temp_val, i]
        
        
    def add_to_shared_from(self, agent_, reverse = False):
        #temp_df = agent_.search_hist.copy()
        #self.shared_search_hist = pd.concat([self.shared_search_hist, temp_df], ignore_index=True)
        
        if reverse:
            agent_.shared_search_hist.loc[len(agent_.shared_search_hist)] = self.search_hist.iloc[-1,:].values.tolist()
        else:
            self.shared_search_hist.loc[len(self.shared_search_hist)] = agent_.search_hist.iloc[-1,:].values.tolist()
        
        
    def update_shared_searches(self, reverse = False):        
        #self.shared_search_hist = pd.DataFrame(columns=['Location', 'Value', 'Step'])
        #self.shared_search_hist = pd.concat([self.shared_search_hist, self.search_hist.copy()], ignore_index=True)
        self.shared_search_hist.loc[len(self.shared_search_hist)] = self.search_hist.iloc[-1,:].values.tolist()
        
        if self.neighbors != []:
            for neighbor in self.neighbors:
                self.add_to_shared_from(neighbor, reverse)
        
        
    def update_neighbors(self, neighbors_list):
        self.neighbors = neighbors_list
        
    
    def add_neighbor(self, new_neighbor):
        self.neighbors.append(new_neighbor)
        
    
    def remove_neighbor(self, neighbor):
        self.neighbors.remove(neighbor)
        
        
    def survey(self, shared=True):
        
        # CHANGE SHARED TO FALSE FOR INDIVIDUAL TESTING
        
        #start_time = time.time()
        
#         ### THIS IS FOR GP-UCB AGENTS
#         if shared:
#             temp_search_hist = self.shared_search_hist.sort_values(by=['Step'])[-self.memory:]
#             self.gp.fit(temp_search_hist['Location'].tolist(),
#                         temp_search_hist['Value'].tolist())
#             landscape_index = [list(a) for a in self.landscape.index]           
#         else:
#             temp_search_hist = self.search_hist.sort_values(by=['Step'])[-self.memory:]
#             self.gp.fit(temp_search_hist['Location'].tolist(),
#                         temp_search_hist['Value'].tolist())
#             landscape_index = [list(a) for a in self.landscape.index]
            
        ### THIS IS FOR MYOPIC AGENTS
        temp_search_hist = self.search_hist.sort_values(by=['Step'])[-1:]
        self.gp.fit(temp_search_hist['Location'].tolist(),
                    temp_search_hist['Value'].tolist())
        landscape_index = [list(a) for a in self.landscape.index]
        
        #end_time = time.time()
        #print(f'Survey time required: {end_time - start_time}')
        
        y_est, std = self.gp.predict(landscape_index, return_std=True)
        return y_est, std
    
    
    def choose_max_point(self, shared=False): 
        
        # This includes y_est, multiplier*std(uncertainty value), and -multiplier*distance from current point
        
        recent_neighbor_vals = self.shared_search_hist.sort_values(by=['Step'])[-(self.memory*len(self.neighbors)):]
        recent_self_val = self.search_hist.sort_values(by=['Step'])[-1:]

        max_neighbor = recent_neighbor_vals.sort_values(by = ['Value'], ascending=False).iloc[0,:]

        if max_neighbor['Value'] > recent_self_val['Value'].tolist()[0]:
            loc = np.where(self.landscape == max_neighbor['Value'])[0][0]
        
        else:
#             y_est, std = self.survey()
            y_est = self.landscape.to_numpy()
            std = [0]*len(y_est)
            expected_value = [(a[0] + a[1]*0) for a in zip(y_est,std)] # <-- UNCERTAINTY VALUE HERE

            distances = self.distances_from_current()


            ### THIS IS FOR GP-UCB AGENTS
    #         values = np.array([(a[0] - (a[1]*self.distance_cost)) for a in zip(expected_value,distances)])
    #         tau = 0.02
    #         softmax = np.exp(values/tau) / sum(np.exp(values/tau))
    #         softmax_dict = dict(zip(softmax.copy(), values.copy()))
    #         options = dict(sorted(softmax_dict.items(), reverse=True))
    #         for option in options.items():
    #             print(option)

    #         choice = random.choices(list(options.values()), weights=list(options.keys()), k=1) 
    #         loc = np.where(values == choice)[0][0]

            ### THIS IS FOR MYOPIC AGENTS VALUE ARRAY
            values = np.array([(a[0] - (a[1]*self.distance_cost)) if a[1] <= 1.0 else 0 for a in zip(expected_value,distances)])  

            ### THIS IS FOR RANDOMLY CHANGING ONE DIGIT
            curr_val = recent_self_val['Value'].tolist()[0]
            new_loc = random.choices(np.where( (values != 0) & (values != curr_val))[0].tolist(), k=1)
            
            if values[new_loc] > recent_self_val['Value'].tolist()[0]:
                loc = new_loc[0]
            else:
                loc = np.where(values == recent_self_val['Value'].tolist()[0])[0].tolist()[0] 
            
#             ### THIS IS FOR GRADIENT ASCENT
#             if max(values) > recent_self_val['Value'].tolist()[0]:
#                 locs = np.where(values == max(values))[0].tolist()
#             else:
#                 locs = np.where(values == recent_self_val['Value'].tolist()[0])[0].tolist()
                              
#             choice = random.choices(values, weights=values, k=1)
#             locs = np.where(values == choice)[0].tolist()
#             loc = random.choices(locs, k=1)[0]
#             #print(loc)

        
        if shared:
            landscape_index = [list(a) for a in self.landscape.index]
        else:
            landscape_index = [list(a) for a in self.landscape.index]
            
        return landscape_index[loc]
        
        
    def distance_2_points(self, a, b):
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    
    
    def distance_from_many_to_one(self, new_point, shared=True):  
        #### BE CAREFUL, THIS WONT WORK IF YOU INITIALIZE AGENTS OUTSIDE OF POPULATIONS
        
        # Agents wont have shared search histories, make shared=False
        ## I think this is fixed now, requires specific testing
        
        if shared:
            distances = [self.distance_2_points(a, new_point)
                         for a in self.shared_search_hist['Location'].tolist()] 
        else:
            distances = [self.distance_2_points(a, new_point)
                         for a in self.search_hist['Location'].tolist()]

        return min(distances)
    
    
    def distances_from_current(self):
        
        landscape_index = [list(a) for a in self.landscape.index]
        
        distances = [self.distance_from_many_to_one(each) for each in landscape_index]
        
        return distances
    
    
    def move_across_landscape(self):
        #Max value location
        new_location = self.choose_max_point()
        
        #Determine actual value of new location
        new_val = self.landscape[new_location[0], new_location[1]]
        
        #Add new location to search_hist
        self.search_hist.loc[len(self.search_hist)] = [new_location, new_val, len(self.search_hist)]
        
        #Change current_position
        ##Auto-updates based on .iloc[-1]
        
#         global progress_counter
#         global complete_
#         print(f'{progress_counter}/{complete_} steps taken!')
#         print(f'Job {round(progress_counter/complete_, 3)*100}% complete!')
#         progress_counter += 1
        
    def explore(self, steps):
        # Repeated move_across_landscape()
        for _ in range(steps):
            self.move_across_landscape()            
            
            
class Population():
     
    def __init__(self, landscape_file, num_agents, graph, num_searches, distance_cost = 0, memory = None, id_ = None, graphid_ = 1, myopia = False):
        
        self.myopia = myopia
        
        self.id_ = id_
        
        self.num_agents = num_agents
        
        self.num_searches = num_searches
        
        self.memory = memory
        
        self.distance_cost = distance_cost
        
        self.graph = graph
        
        self.graphid = graphid_
        
        self.landscape_file = landscape_file
        self.landscape = self.read_in_landscape()
        
        self.agents = self.initialize_population()
    
        self.progress_counter = 0
        self.complete_ = num_agents*num_searches
                
        
    def read_in_landscape(self):
        df = pd.read_csv(self.landscape_file, header=None)
        df = df.iloc[1:,1:]
        landscape = df.stack()
        landscape = landscape.astype('float')
        return landscape
    
    
    def add_neighbors_from_graph(self, agents, graph):
        edge_list = list(list(a) for a in graph.edges)
        
        for edge in edge_list:
            agents[edge[0]].add_neighbor(agents[edge[1]])
            agents[edge[1]].add_neighbor(agents[edge[0]])
    
    
    def initialize_population(self): 
        
        if self.myopia == False:
            agents = []
            for mem in self.memory:
                agents.append(Agent(landscape=self.landscape, 
                                    points=1, 
                                    memory=mem,
                                    distance_cost=self.distance_cost))
        
        if self.myopia == True:
            agents = []
            for mem in self.memory:
                agents.append(Agent_Myopic(landscape=self.landscape, 
                                    points=1, 
                                    memory=mem,
                                    distance_cost=self.distance_cost)) 
        
#         if self.myopia == False:
#             agents = [Agent(landscape=self.landscape,
#                             points=1,
#                             memory=self.memory,
#                             distance_cost=self.distance_cost) for i in range(self.num_agents)]
        
#         if self.myopia == True:
#             agents = [Agent_Myopic(landscape=self.landscape,
#                                    points=1,
#                                    memory=self.memory,
#                                    distance_cost=self.distance_cost) for i in range(self.num_agents)]            
          
        
        self.add_neighbors_from_graph(agents, self.graph)
        
        for agent in agents:
            agent.update_shared_searches()
        
        print(f'Population initialized!')
            
        return agents
    
    
    def report_step_averages(self, step):
        
        agent_vals = []
        for agent in self.agents:
            agent_vals.append(agent.search_hist['Value'][step])
        
        return agent_vals
    
    
    def explore(self, steps = 1, sync=True):
        # Pop level explore function (round-based search)
        ## Could weave search and update together pretty easily
        ### Done by reversing the updating function
        
        for _ in range(steps):
            for agent in self.agents:
                agent.explore(1)
                
                if not sync:
                    agent.update_shared_searches(reverse=True)
                
                # Progress Markers
                #print(f'{progress_counter}/{complete_} steps taken!')
                self.progress_counter += 1
                #print(f'Search Progress: {int(round(self.progress_counter/self.complete_, 3)*100)}%!')
             
            if sync:
                for agent in self.agents:
                    agent.update_shared_searches(reverse=True)
                    continue
                    
        print(f'Sim Complete!')
                    
                    
def run_model(p_combo):

    memory_ = []
    for each in p_combo['memory']:
        memory_ += [each[1]]*each[0]
    random.shuffle(memory_)
    print(memory_)
    
    # Create one population to work with
    pop = Population(landscape_file = p_combo['landscape'], 
                     num_agents = p_combo['num_agents'], 
                     graph = p_combo['graph'], 
                     num_searches = p_combo['num_searches'], 
                     distance_cost = p_combo['distance_costs'],
                     memory = memory_,
                     id_ = p_combo['id'],
                     graphid_ = p_combo['graph_label'])
    
    # Explore 
    pop.explore(p_combo['num_searches'], sync=p_combo['sync'])
    
    # Add pop to history list
    #sim_history.append(pop)
    
    # Announcements
    # print(f'Sim {pop.id_} Complete!')
    
    # Return pop object
    return pop  


def run_model_myopic(p_combo):

    # Create one population to work with
    pop = Population(landscape_file = p_combo['landscape'], 
                     num_agents = p_combo['num_agents'], 
                     graph = p_combo['graph'], 
                     num_searches = p_combo['num_searches'], 
                     distance_cost = p_combo['distance_costs'],
                     memory = 1,
                     id_ = p_combo['id'],
                     graphid_ = p_combo['graph_label'],
                     myopia = True)
    
    # Explore 
    pop.explore(p_combo['num_searches'], sync=p_combo['sync'])
    
    # Add pop to history list
    #sim_history.append(pop)
    
    # Announcements
    # print(f'Sim {pop.id_} Complete!')
    
    # Return pop object
    return pop 


def sys_argv_extract(sys_arg, log=False):
    
    n = len(sys_arg)
    c = sys_arg[1:n-1]
    arg_split = c.split(',')
    
    if len(arg_split) == 1:
        params = [float(arg_split[0])]
        
    if len(arg_split) == 3:
        min_param = float(arg_split[0])
        max_param = float(arg_split[1])
        increment = int(arg_split[2])    
        
        if log:
            params = np.logspace(min_param, max_param, increment, endpoint=True)
        else:
            params = np.linspace(min_param, max_param, increment, endpoint=True)        
        
    if len(arg_split) > 3:
        params = []
        for item in arg_split:
            params.append(float(item))
    
    return params



