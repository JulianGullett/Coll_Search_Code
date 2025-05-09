
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import pandas as pd
import random as random
import numpy as np

class Agent():
    
    # Single actor who can search and share info with neighbors
    ## Contains survey/explore functions (evaluation of landscape and choice of new position)
    
    def __init__(self, landscape, points = 1, distance_cost = 0, memory = None, search_hist = None, neighbors = None, myopic=False):
        
        self.landscape = landscape
        if memory == None:
            self.memory = 10**10
        else:
            self.memory = memory
        
        if myopic:
            self.distance_cost = 10**6
        else:
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
        if self.memory == 0:
            temp_search_hist = self.search_hist.sort_values(by=['Step'])[-1:]
        if self.memory != 0:
            temp_search_hist = self.shared_search_hist.sort_values(by=['Step'])[-((len(self.neighbors)+1)*self.memory):]
            
        if shared:
            self.gp.fit(temp_search_hist['Location'].tolist(),
                        temp_search_hist['Value'].tolist())
            landscape_index = np.indices(self.landscape.shape).reshape(-1, len(self.landscape.shape)).tolist()
            #landscape_index = [list(a) for a in self.landscape.index]           
        else:
            self.gp.fit(self.search_hist['Location'].tolist(),
                        self.search_hist['Value'].tolist())
            landscape_index = np.indices(self.landscape.shape).reshape(-1, len(self.landscape.shape)).tolist()
            #landscape_index = [list(a) for a in self.landscape.index]
        
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

        landscape_index = np.indices(self.landscape.shape).reshape(-1, len(self.landscape.shape)).tolist()
            
        return landscape_index[loc]
        
        
    def distances_from_current(self):
        
        landscape_index = np.indices(self.landscape.shape).reshape(-1, len(self.landscape.shape)).tolist()
        #landscape_index = [list(a) for a in self.landscape.index]
        
        distances = [self.distance_from_many_to_one(each) for each in landscape_index]
        
        return distances 
       

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

       
    def distance_2_points(self, a, b):
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    

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
     
    def __init__(self, landscape_params, num_agents, graph, num_searches, distance_cost = 0, memory = None, id_ = None, graphid_ = 1, myopia = False):
        
        self.myopia = myopia
        
        self.id_ = id_
        
        self.num_agents = num_agents
        
        self.num_searches = num_searches
        
        self.memory = memory
        
        self.distance_cost = distance_cost
        
        self.graph = graph
        
        self.graphid = graphid_
        
        self.landscape_params = landscape_params
        self.landscape = self.produce_landscape()
        
        self.agents = self.initialize_population()
    
        self.progress_counter = 0
        self.complete_ = num_agents*num_searches
                
        
    def read_in_landscape(self):
        df = pd.read_csv(self.landscape_file, header=None)
        df = df.iloc[1:,1:]
        landscape = df.stack()
        landscape = landscape.astype('float')
        return landscape
        
        
    def produce_landscape(self):
        width, height, scale = self.landscape_params
        return generate_perlin_noise(width, height, scale)
    
    
    def add_neighbors_from_graph(self, agents, graph):
        edge_list = list(list(a) for a in graph.edges)
        
        for edge in edge_list:
            agents[edge[0]].add_neighbor(agents[edge[1]])
            agents[edge[1]].add_neighbor(agents[edge[0]])
    
    
    def initialize_population(self): 
        
        agents = []
        for mem in self.memory:
            agents.append(Agent(landscape=self.landscape, 
                                points=1, 
                                memory=mem,
                                distance_cost=self.distance_cost,
                                myopic=self.myopia))        
        
        # if self.myopia == False:
        #     agents = []
        #     for mem in self.memory:
        #         agents.append(Agent(landscape=self.landscape, 
        #                             points=1, 
        #                             memory=mem,
        #                             distance_cost=self.distance_cost))
        
        # if self.myopia == True:
        #     agents = []
        #     for mem in self.memory:
        #         agents.append(Agent_Myopic(landscape=self.landscape, 
        #                             points=1, 
        #                             memory=mem,
        #                             distance_cost=self.distance_cost)) 
        
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
    #print(memory_)
    
    # Create one population to work with
    pop = Population(landscape_params = p_combo['landscape'], 
                     num_agents = p_combo['num_agents'], 
                     graph = p_combo['graph'], 
                     num_searches = p_combo['num_searches'], 
                     distance_cost = p_combo['distance_costs'],
                     memory = memory_,
                     id_ = p_combo['id'],
                     graphid_ = p_combo['graph_label'],
                     myopia = p_combo['myopia'])
    
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


def generate_perlin_noise(width, height, scale):
    """
    Generate Perlin noise using the given parameters.
    
    Parameters:
    - width (int): Width of the noise array.
    - height (int): Height of the noise array.
    - scale (int): Scale factor for generating the noise.
    
    Returns:
    - noise (n-dimensional array): Perlin noise array of shape (height, width).
    """

    
    # Create an empty noise array
    noise = np.zeros((height, width))
    
    # Generate random gradient vectors
    gradients = np.random.randn(height // scale + 2, width // scale + 2, 2)

    # Iterate over each pixel in the noise array
    for y in range(height):
        for x in range(width):
            # Calculate the grid cell coordinates for the current pixel
            cell_x = x // scale
            cell_y = y // scale

            # Calculate the position within the cell as fractional offsets
            cell_offset_x = x / scale - cell_x
            cell_offset_y = y / scale - cell_y

            # Calculate the dot products between gradients and offsets
            dot_product_tl = np.dot([cell_offset_x, cell_offset_y], gradients[cell_y, cell_x])
            dot_product_tr = np.dot([cell_offset_x - 1, cell_offset_y], gradients[cell_y, cell_x + 1])
            dot_product_bl = np.dot([cell_offset_x, cell_offset_y - 1], gradients[cell_y + 1, cell_x])
            dot_product_br = np.dot([cell_offset_x - 1, cell_offset_y - 1], gradients[cell_y + 1, cell_x + 1])
          
            # Interpolate the dot products using smoothstep function
            weight_x = smoothstep(cell_offset_x)
            weight_y =  smoothstep(cell_offset_y)
            interpolated_top = lerp(dot_product_tl, dot_product_tr, weight_x)
            interpolated_bottom = lerp(dot_product_bl, dot_product_br, weight_x)
            interpolated_value = lerp(interpolated_top, interpolated_bottom, weight_y)

            # Store the interpolated value in the noise array
            noise[y, x] = interpolated_value
            
    # Normalize the noise values within the range of 0 to 1
    noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))
    noise[noise==0] = np.min(noise[np.nonzero(noise)])
    
    return noise

def smoothstep(t):
    """
    Smoothstep function for interpolation.
    
    Parameters:
    - t (float): Interpolation value between 0.0 and 1.0.
    
    Returns:
    - result (float): Smoothstep interpolated value.
    """
    return t * t * (3 - 2 * t)

def lerp(a, b, t):
    """
    Linear interpolation between two values.
    
    Parameters:
    - a (float): Start value.
    - b (float): End value.
    - t (float): Interpolation factor between 0.0 and 1.0.
    
    Returns:
    - result (float): Interpolated value between a and b.
    """
    return a + t * (b - a)

