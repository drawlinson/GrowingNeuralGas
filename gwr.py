# coding: utf-8

import numpy as np
from scipy import spatial
import networkx as nx
import matplotlib.pyplot as plt
from sklearn import decomposition
import copy

__authors__ = 'David Rawlinson'
__email__ = 'dave@agi.io'

'''
Implementation of Growing When Required (GWR) based on:
"Lifelong learning of human actions with deep neural network self-organization"
German I. Parisi, Jun Tani, Cornelius Weber, Stefan Wermter
Neural Networks 96 (2017) pp 137–149

Additional details of GWR from: 
"A self-organising network that grows when required"
Stephen Marsland, Jonathan Shapiro, Ulrich Nehmzow
Neural Networks 15(8-9) (2002) pp 1041-58

Original code forked from GNG implementation at:
https://github.com/AdrienGuille/GrowingNeuralGas
'''


class GrowingWhenRequired:

    GrowthStrategyInterval = 'interval'  # i.e. GNG
    GrowthStrategyRequired = 'required'  # i.e. GWR

    def __init__(self, hparams):#input_data):
        self.network = None
        self.units_created = 0
        self.hparams = copy.deepcopy(hparams)
        #self.num_inputs = num_inputs
        #self.data = input_data
        #plt.style.use('ggplot')
        
    @staticmethod
    def default_hparams():
        """Hyperparameter input_shape must be 4d for compatibility with other conv ANN. The dimensions are batch_size, h, w, channels or depth."""
        hparams = {
            'growth_strategy':GrowingWhenRequired.GrowthStrategyInterval,
            'topology':True,  # False for NeuralGas model
            'input_shape':None,
            'init_range':2.0,  # Uniform distribution
            'learning_rate_best': 0.1,
            'learning_rate_neighbour': 0.006,
            'max_age': 10,  # Max age of an edge before it is removed
            'max_size':-1,
            'growth_interval': 200,  # Only if GNG/NG
            'activity_threshold':0.0,  # Only if GWR, unit value. "The value of the insertion threshold aT does make a large difference, however. If the value is set very close to 1 then more nodes are produced and the input is represented very well. For lower values of aT fewer nodes are added."
            'usage_threshold':0.1,  # Threshold at which to allow growth
            'usage_initial':1.0,  # "values used in the experiments were: h_0 = 1"
            'usage_decay':0.8,  # Should be unit range
            'error_split': 0.5,  # What happens to accumulated error on split
            'error_decay': 0.995  # Learning rate of error
        }
        return hparams

    def find_nearest_units(self, observation):
        #distance = []
        dist_b = 0
        dist_n = 0
        node_b = None
        node_n = None
        for u, attributes in self.network.nodes(data=True):
            vector = attributes['vector']
            #dist = spatial.distance.euclidean(vector, observation)
            dist = self.distance(vector, observation)
            if node_b is None:
                node_b = u
                dist_b = dist
            elif node_n is None:
                if dist < dist_b:
                    node_n = node_b
                    dist_n = dist_b
                    node_b = u
                    dist_b = dist
                else:
                    node_n = u
                    dist_n = dist
            elif dist < dist_n:
                if dist < dist_b:
                    node_n = node_b
                    dist_n = dist_b
                    node_b = u
                    dist_b = dist
                else:
                    node_n = u
                    dist_n = dist

            #distance.append((u, dist))
        #distance.sort(key=lambda x: x[1])
        #ranking = [u for u, dist in distance]
        #return ranking
        return [node_b, node_n]

    def distance(self, vector, observation):
        dist = spatial.distance.euclidean(vector, observation)
        return dist

    def prune_connections(self):
        listToRemoveE = []
        listToRemoveU = []

        # Remove edges that haven't been used for a while
        max_age = self.hparams['max_age']
        for u, v, attributes in self.network.edges(data=True):
            if attributes['age'] > max_age:
                #print('removing edge:', u,v )
                #self.network.remove_edge(u, v)
                listToRemoveE.append([u, v])

        # Remove nodes with no edges
        for u in self.network.nodes():
            if self.network.degree(u) == 0:
                #print('!!!!!!!!!!!!!!!!removing node', u )
                #self.network.remove_node(u)
                listToRemoveU.append(u)

        try:
            self.network.remove_edges_from(listToRemoveE)
            self.network.remove_nodes_from(listToRemoveU)
        except:
            print('Error while removing...')
            print('Edges to remove',listToRemoveE)
            print('Nodes to remove',listToRemoveU)

    def number_of_clusters(self):
        return nx.number_connected_components(self.network)

    def plot_clusters(self, clustered_data):
        number_of_clusters = nx.number_connected_components(self.network)
        plt.clf()
        plt.title('Cluster affectation')
        color = ['r', 'b', 'g', 'k', 'm', 'r', 'b', 'g', 'k', 'm']
        for i in range(number_of_clusters):
            observations = [observation for observation, s in clustered_data if s == i]
            if len(observations) > 0:
                observations = np.array(observations)
                plt.scatter(observations[:, 0], observations[:, 1], color=color[i], label='cluster #'+str(i))
        plt.legend()
        plt.savefig('visualization/clusters.png')


    def init(self):
        #w_shape = np.shape(self.data)[1]
        input_shape = self.hparams['input_shape']
        num_inputs = np.prod(input_shape)
        init_range = self.hparams['init_range']
        w = [np.random.uniform(-init_range, init_range) for _ in range(num_inputs)]
        return w

    def add_node(self, vector):
        """Ensure we push all node creation through this member fn to standardize properties of the nodes"""
        error_initial = 0.0
        usage_initial = self.hparams['usage_initial']
        node_id = self.units_created
        self.network.add_node(node_id, vector=vector, error=error_initial, usage=usage_initial)
        self.units_created += 1
        return node_id

    def reset(self):
        # 0. start with two units a and b at random position w_a and w_b
        self.units_created = 0
        w_a = self.init()
        w_b = self.init()
        self.network = nx.Graph()
        self.add_node(w_a)
        self.add_node(w_b)
        #self.network.add_node(self.units_created, vector=w_a, error=0)
        #self.units_created += 1
        #self.network.add_node(self.units_created, vector=w_b, error=0)
        #self.units_created += 1

    def can_grow(self):
        max_size = self.hparams['max_size']
        num_nodes = len(self.network.nodes())
        #print('grow? sz=', num_nodes,'max',max_size)
        if (max_size < 0) or (num_nodes < max_size):
            return True
        #print('cant grow')
        return False

    def split_cells(self, q, f):
        w_r = 0.5 * (np.add(self.network.node[q]['vector'], self.network.node[f]['vector']))
        r = self.units_created  # TODO allow node recycling. This line implies endless growth
        # 8.c insert edges connecting the new unit r with q and f
        #     remove the original edge between q and f
        #self.network.add_node(r, vector=w_r, error=0)
        self.units_created += 1
        r = self.add_node(w_r)
        self.network.add_edge(r, q, age=0)
        self.network.add_edge(r, f, age=0)
        self.network.remove_edge(q, f)
        # 8.d decrease the error variables of q and f by multiplying them with a
        #     initialize the error variable of r with the new value of the error variable of q
        a = self.hparams['error_split']
        self.network.node[q]['error'] *= a
        self.network.node[f]['error'] *= a
        self.network.node[r]['error'] = self.network.node[q]['error']
        
    def update_usage(self, node_id, is_best):
        """Unlike the original GWR paper, I'm just going with a simple exponential decay."""
        node = self.network.node[node_id]
        old_usage = node['usage']
        #if is_best:  # Winner
        #    alpha = 1.05
        #    tau =
        #else:  # Neighbour
        #    alpha = 1.05
        #z = num / den * (1.0 - np.exp(-alpha/tau)
        usage_decay = self.hparams['usage_decay']
        new_usage = old_usage * usage_decay
        node['usage'] = new_usage

    def update(self, observation, step):
        # Get hyperparameters
        growth_strategy = self.hparams['growth_strategy']
        e_b = self.hparams['learning_rate_best']
        e_n = self.hparams['learning_rate_neighbour']
        l = self.hparams['growth_interval']
        #a = self.hparams['error_split']
        #d = self.hparams['error_decay']
        
        # 2. find the nearest unit s_1 and the second nearest unit s_2
        nearest_units = self.find_nearest_units(observation)
        s_1 = nearest_units[0]
        s_2 = nearest_units[1]

        # 3. increment the age of all edges emanating from s_1
        #for u, v, attributes in self.network.edges_iter(data=True, nbunch=[s_1]):
        for u, v, attributes in self.network.edges(data=True, nbunch=[s_1]):
            #print('edge:',u,'->',v)
            #print('edge:',u,'->',v,' age',self.network.edges[u,v]['age'])
            #self.network.add_edge(u, v, age=attributes['age']+1)
            self.network.edges[u,v]['age'] += 1

        # 3.1 If there is no edge between winner and runner-up, create it.
        # Otherwise, set the edge age to zero.
        # This is done here by replacing the edge.
        # https://networkx.github.io/documentation/stable/reference/classes/graph.html
        if self.network.has_edge(s_1, s_2):
            self.network.edges[s_1, s_2]['age'] = 0
        else:
            self.network.add_edge(s_1, s_2, age=0)

        # 4. add the squared distance between the observation and the nearest unit in input space
        #dist = spatial.distance.euclidean(observation, self.network.node[s_1]['vector'])**2
        node = self.network.node[s_1]['vector']
        dist = self.distance(observation, node) **2
        self.network.node[s_1]['error'] += dist  # cumulative error update

        did_split = False
        if growth_strategy == GrowingWhenRequired.GrowthStrategyRequired:
            # update firing (usage) rate for winning node and neighbours
            best_usage = self.network.node[s_1]['usage']  # Keep old value
            self.update_usage(s_1, True)
            for u in self.network.neighbors(s_1):
                self.update_usage(u, False)

            if self.can_grow():
                #If the activity a < activity threshold aT
                # And firing counter < firing threshold hT
                # .. then a new node should
                #be added between the two best matching matching
                #nodes (s and t)
                # See [Marsland, (2002): A self-organising network that grows when required]
                euc_norm = np.sqrt(dist)  # ie vec len
                best_activity = np.exp(-euc_norm)  # ie unit value
                activity_threshold = self.hparams['activity_threshold']
                usage_threshold = self.hparams['usage_threshold']
                activity_test = best_activity < activity_threshold
                usage_test = best_usage < usage_threshold
                #print('sz',len(self.network.nodes()),activity_test,usage_test,' best act/thr', best_activity,activity_threshold, '----- best usage', best_usage, usage_threshold)
                if activity_test and usage_test:
                    self.split_cells(s_1,s_2)  # Create a new cell between s_1 and s_2
                    did_split = True


        # 5 .move s_1 and its direct topological neighbors towards the observation by the fractions
        #    e_b and e_n, respectively, of the total distance
        if did_split is False:
            b_vector = self.network.node[s_1]['vector']
            update_w_s_1 = e_b * (np.subtract(observation, b_vector))
            self.network.node[s_1]['vector'] = np.add(b_vector, update_w_s_1)
            #update_w_s_n = e_n * (np.subtract(observation, best_vector))  This looks like a bug
            for neighbor in self.network.neighbors(s_1):
                n_vector = self.network.node[neighbor]['vector']
                update_w_s_n = e_n * (np.subtract(observation, n_vector))
                self.network.node[neighbor]['vector'] = np.add(n_vector, update_w_s_n)
            # 6. if s_1 and s_2 are connected by an edge, set the age of this edge to zero
            #    if such an edge doesn't exist, create it
            self.network.add_edge(s_1, s_2, age=0)

        # 7. remove edges with an age larger than max_age
        #    if this results in units having no emanating edges, remove them as well
        self.prune_connections()

        # 8. Interval strategy: if the number of steps so far is an integer multiple of parameter l, insert a new unit
        step += 1
        if growth_strategy == GrowingWhenRequired.GrowthStrategyInterval:
            if self.can_grow():
                if step % l == 0:
                    # 8.a determine the unit q with the maximum accumulated error
                    q = 0
                    error_max = 0
                    #for u in self.network.nodes_iter():
                    for u in self.network.nodes():
                        if self.network.node[u]['error'] > error_max:
                            error_max = self.network.node[u]['error']
                            q = u
                    # 8.b insert a new unit r halfway between q and its neighbor f with the largest error variable
                    f = -1
                    largest_error = -1
                    for u in self.network.neighbors(q):
                        if self.network.node[u]['error'] > largest_error:
                            largest_error = self.network.node[u]['error']
                            f = u
                    self.split_cells(q,f)  # Create a new cell between q and f

        # 9. decrease all error variables by multiplying them with a constant d
        error = 0
        for u in self.network.nodes():
            error += self.network.node[u]['error']
        #accumulated_local_error.append(error)
        return error
    
    def fit_network(self, data, passes=1, plot_evolution=False):
        # logging variables
        accumulated_local_error = []
        global_error = []
        network_order = []
        network_size = []
        total_units = []
        
        def compute_global_error(gng, data):
            global_error = 0
            for observation in data:
                nearest_units = gng.find_nearest_units(observation)
                s_1 = nearest_units[0]
                #global_error += spatial.distance.euclidean(observation, gng.network.node[s_1]['vector'])**2
                node = gng.network.node[s_1]['vector']
                dist = self.distance(observation, node)**2
                global_error += dist
            return global_error
        
        
        l = self.hparams['growth_interval']
        d = self.hparams['error_decay']
        
        ## 0. start with two units a and b at random position w_a and w_b
        self.reset()

        # 1. iterate through the data
        sequence = 0
        for p in range(passes):
            print('   Pass #%d' % (p + 1))
            np.random.shuffle(data)
            step = 0
            for observation in data:
                error = self.update(observation, step)
                accumulated_local_error.append(error)
                
                #if step % l == 0:
                if plot_evolution:
                    self.plot_network(data, 'visualization/sequence/' + str(sequence) + '.png')
                sequence += 1
                
                network_order.append(self.network.order())
                network_size.append(self.network.size())
                total_units.append(self.units_created)
                
                # Node error
                for u in self.network.nodes():
                    self.network.node[u]['error'] *= d
                    if self.network.degree(nbunch=[u]) == 0:
                        print(u)

                step += 1
            global_error.append(compute_global_error(self, data))
        plt.clf()
        plt.title('Accumulated local error')
        plt.xlabel('iterations')
        plt.plot(range(len(accumulated_local_error)), accumulated_local_error)
        plt.savefig('visualization/accumulated_local_error.png')
        plt.clf()
        plt.title('Global error')
        plt.xlabel('passes')
        plt.plot(range(len(global_error)), global_error)
        plt.savefig('visualization/global_error.png')
        plt.clf()
        plt.title('Neural network properties')
        plt.plot(range(len(network_order)), network_order, label='Network order')
        plt.plot(range(len(network_size)), network_size, label='Network size')
        plt.legend()
        plt.savefig('visualization/network_properties.png')

    def cluster_data(self, data):
        unit_to_cluster = np.zeros(self.units_created)
        cluster = 0
        for c in nx.connected_components(self.network):
            for unit in c:
                unit_to_cluster[unit] = cluster
            cluster += 1
        clustered_data = []
        for observation in data:
            nearest_units = self.find_nearest_units(observation)
            s = nearest_units[0]
            clustered_data.append((observation, unit_to_cluster[s]))
        return clustered_data

        
    def plot_network(self, data, file_path):
        plt.clf()
        plt.scatter(data[:, 0], data[:, 1], c='b')
        node_pos = {}
        for u in self.network.nodes():
            vector = self.network.node[u]['vector']
            node_pos[u] = (vector[0], vector[1])
        nx.draw(self.network, pos=node_pos, node_color='r')
        plt.draw()
        plt.savefig(file_path)

            