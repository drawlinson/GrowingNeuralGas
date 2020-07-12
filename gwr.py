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

A note on the intended dynamics.

0. We desire to have an upper bound on resources (cells).

1. Edges age when they compete with a winning edge, and aren't between winner and 2nd best.
This means that cells representing rare input can persist for an arbitrarily long time.
Useless cells will be removed when nearby useful ones are used instead.

2. If the network has headroom to grow, we are OK to grow and prune at our leisure.
This means the network has less than min_size quantity of cells. Min_size envisioned to be
about 90% of max_size (say, reserviing 10% for rapid growth).

3. The network can temporarily grow to max_size to accomodate new input. If the new input
is transient, or it replaces old input that no longer occurs, the network will gradually
shrink down again. Remember, the network only grows when required, so it may reach max_size.
Whether this happens is controlled by activity_threshold hyperparameter.

4. The optimality of cell distribution across the input space depends on being able to add
and remove cells. Therefore if the network is > min_size, there's no guarantee the resource
allocation is optimal.

5. If the network exceeds min_size there is no room to learn new cells. If we require fast
adaptation there will be a non-optimal allocation of resources. Therefore, we'd like to
remove resources where an excess exists.

6. The original GNG paper suggests not to prune cells based on edge-age, but to measure
utility instead. I don't see any harm in keeping the age-based pruning rule. It can be
disabled by making it negative (in this implementation).

7. If we use utility to select cells for removal, we should only do it to "older" cells
where the utility has had time to be defined. This is captured by the usage tracker.
So we will remove cells with min utility, and enough usage, until we get down to min_size again.

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
            'max_age': 10,  # Max age of an edge before it is removed. If negative, no limit
            'max_size':-1,  # Max number of cells; if negative, no limit
            'min_size':-1,  # Min number of cells; if negative, no limit. Min cells is used to reserve capacity for continual learning
            'growth_interval': 200,  # Only if GNG/NG
            'activity_threshold':0.0,  # Only if GWR, unit value. "The value of the insertion threshold aT does make a large difference, however. If the value is set very close to 1 then more cells are produced and the input is represented very well. For lower values of aT fewer cells are added."
            'usage_threshold':0.1,  # Threshold at which to allow growth
            'usage_initial':1.0,  # "values used in the experiments were: h_0 = 1"
            'usage_decay':0.8,  # Should be unit range
            'error_split': 0.5,  # What happens to accumulated error on split
            'error_decay': 0.995,  # Learning rate of error
            'utility_decay': 0.995,  # Learning rate of utility
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
        return [node_b, node_n], [dist_b, dist_n]

    def distance(self, vector, observation):
        dist = spatial.distance.euclidean(vector, observation)
        return dist

    def prune_connections(self):
        max_age = self.hparams['max_age']
        if max_age < 0:
            return  # Don't prune, option disabled

        # Remove edges that haven't been used for a while
        listToRemoveE = []
        listToRemoveU = []

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
            print('Cells to remove',listToRemoveU)

    def number_of_clusters(self):
        return nx.number_connected_components(self.network)

    def cluster_data(self, data):
        """Cluster the data using connected components of the network graph"""
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

    def plot_clusters(self, clustered_data):
        """Plot the clusters defined by the graph"""
        number_of_clusters = nx.number_connected_components(self.network)
        plt.clf()
        plt.title('Cluster affectation')
        color = ['r', 'b', 'g', 'k', 'm', 'r', 'b', 'g', 'k', 'm']
        for i in range(number_of_clusters):
            observations = []
            for observation, s in clustered_data:
                if s.any() == i:
                    observations.append(observation)
            #observations = [observation for observation, s in clustered_data if s == i]
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

    def add_cell(self, vector):
        """Ensure we push all cell creation through this member fn to standardize properties of the cells"""
        error_initial = 0.0
        usage_initial = self.hparams['usage_initial']  # e.g. 1
        utility_initial = 0.0
        cell_id = self.units_created
        self.network.add_node(cell_id, vector=vector, error=error_initial, usage=usage_initial, utility=utility_initial)
        self.units_created += 1
        return cell_id

    def reset(self):
        # 0. start with two units a and b at random position w_a and w_b
        self.units_created = 0
        w_a = self.init()
        w_b = self.init()
        self.network = nx.Graph()
        self.add_cell(w_a)
        self.add_cell(w_b)

    def can_remove(self):
        min_size = self.hparams['min_size']
        num_cells = len(self.network.nodes())
        #print('grow? sz=', num_cells,'max',max_size)
        if (min_size < 0):
            return True  # Can always prune down to any size

        if (num_cells > min_size):
            return True  # Can prune above this size

        #print('cant prune')
        return False

    def can_grow(self):
        max_size = self.hparams['max_size']
        num_cells = len(self.network.nodes())
        #print('grow? sz=', num_nodes,'max',max_size)
        if (max_size < 0) or (num_cells < max_size):
            return True
        #print('cant grow')
        return False

    def get_size(self):
        num_cells = len(self.network.nodes())
        return num_cells

    def remove_cell(self, cell_id):
        """Removes cell c and all edges to it"""
        # Remove all edges to this vertex
        listToRemoveE = []
        listToRemoveU = []

        for u, v, attributes in self.network.edges(data=True, nbunch=[cell_id]):
            listToRemoveE.append([u, v])

        # Remove cell
        listToRemoveU.append(cell_id)

        try:
            self.network.remove_edges_from(listToRemoveE)
            self.network.remove_nodes_from(listToRemoveU)
        except:
            print('Error while removing...')
            print('Edges to remove',listToRemoveE)
            print('Cells to remove',listToRemoveU)

    def choose_remove_cell(self):
        if not self.can_remove():
            #print('cant remove, size=', self.get_size())
            return None  # Can't remove any

        # Find the cell with usage > threshold and min utility
        usage_threshold = self.hparams['usage_threshold']
        eligible_cells = []
        for cell_id in self.network.nodes():
            cell = self.network.node[cell_id]
            usage = cell['usage']
            if usage < usage_threshold:
                eligible_cells.append(cell_id)

        num_eligible_cells = len(eligible_cells)
        if num_eligible_cells == 0:
            print('cant remove, no eligible, size=', self.get_size())
            return None

        # Find the cell with min utility
        min_utility = None
        min_utility_cell = None
        for cell_id in eligible_cells:
            cell = self.network.node[cell_id]
            utility = cell['utility']
            if min_utility_cell is None:
                min_utility_cell = cell_id
                min_utility = utility
            elif utility < min_utility:
                min_utility_cell = cell_id
                min_utility = utility

        print('remove, Eligible cell, util=',min_utility, ' size=', self.get_size())
        return min_utility_cell

    def split_cells(self, q, f):
        w_r = 0.5 * (np.add(self.network.node[q]['vector'], self.network.node[f]['vector']))
        r = self.units_created  # TODO allow node recycling. This line implies endless growth
        # 8.c insert edges connecting the new unit r with q and f
        #     remove the original edge between q and f
        #self.network.add_node(r, vector=w_r, error=0)
        self.units_created += 1
        r = self.add_cell(w_r)
        self.network.add_edge(r, q, age=0)
        self.network.add_edge(r, f, age=0)
        self.network.remove_edge(q, f)
        # 8.d decrease the error variables of q and f by multiplying them with a
        #     initialize the error variable of r with the new value of the error variable of q
        a = self.hparams['error_split']
        self.network.node[q]['error'] *= a
        self.network.node[f]['error'] *= a
        self.network.node[r]['error'] = self.network.node[q]['error']
        
        # Usage is set to default usage - ie high.
        # Assume utility adjusts over time - we won't remove unless usage also low.

    def update_usage(self, cell_id, is_best):
        """Unlike the original GWR paper, I'm just going with a simple exponential decay."""
        cell = self.network.node[cell_id]
        old_usage = cell['usage']
        #if is_best:  # Winner
        #    alpha = 1.05
        #    tau =
        #else:  # Neighbour
        #    alpha = 1.05
        #z = num / den * (1.0 - np.exp(-alpha/tau)
        usage_decay = self.hparams['usage_decay']
        new_usage = old_usage * usage_decay
        cell['usage'] = new_usage

    def update_utility(self, cell_id, current_utility):
        """Update the utility of the specified cell"""
        cell = self.network.node[cell_id]
        old_utility = cell['utility']
        utility_decay = self.hparams['utility_decay']
        new_utility = (old_utility + current_utility) * utility_decay
        cell['usage'] = new_utility

    def update(self, observation, step):
        # Get hyperparameters
        growth_strategy = self.hparams['growth_strategy']
        e_b = self.hparams['learning_rate_best']
        e_n = self.hparams['learning_rate_neighbour']
        l = self.hparams['growth_interval']

        # Consider removing low-utility cells before selecting winners.
        remove_cell_id = self.choose_remove_cell()
        if remove_cell_id is not None:
            self.remove_cell(remove_cell_id)

        # 2. find the nearest unit s_1 and the second nearest unit s_2
        nearest_units, nearest_dists = self.find_nearest_units(observation)
        s_1 = nearest_units[0]
        s_2 = nearest_units[1]
        error_1 = nearest_dists[0]
        error_2 = nearest_dists[1]

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
        cell_vector = self.network.node[s_1]['vector']
        dist = self.distance(observation, cell_vector) **2
        self.network.node[s_1]['error'] += dist  # cumulative error update

        did_split = False
        if growth_strategy == GrowingWhenRequired.GrowthStrategyRequired:
            # update firing (usage) rate for winning node and neighbours
            utility = error_2 - error_1  # NB this is always positive as error 2 >= error 1, by definition
            best_usage = self.network.node[s_1]['usage']  # Keep old value
            self.update_usage(s_1, True)
            self.update_utility(s_1, utility)
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
    
    def fit_network(self, data, passes=1, plot_interval=None):
        # logging variables
        accumulated_local_error = []
        global_error = []
        network_order = []
        network_size = []
        total_units = []
        
        def compute_global_error(gng, data):
            global_error = 0
            for observation in data:
                nearest_units, nearest_dists = gng.find_nearest_units(observation)
                s_1 = nearest_units[0]
                #global_error += spatial.distance.euclidean(observation, gng.network.node[s_1]['vector'])**2
                cell_vector = gng.network.node[s_1]['vector']
                dist = self.distance(observation, cell_vector)**2
                global_error += dist
            return global_error
        
        
        d = self.hparams['error_decay']
        
        ## 0. start with two units a and b at random position w_a and w_b
        self.reset()

        # 1. iterate through the data
        step = 0
        for p in range(passes):
            print('   Pass #%d' % (p + 1))
            np.random.shuffle(data)
            for i, observation in enumerate(data):
                #print('i',i)
                early_stopping = -1
                if (i > early_stopping) and (early_stopping >= 0):
                    break
                error = self.update(observation, step)
                step += 1
                accumulated_local_error.append(error)
                
                if plot_interval is not None:
                    if step % plot_interval == 0:
                        # Draw the network to file
                        self.plot_network(data, 'visualization/sequence/' + str(step) + '.png')
                
                network_order.append(self.network.order())  # The order of a graph G is the cardinality of its vertex set
                network_size.append(self.network.size())  # The size of a graph G is the cardinality of its vertex set,
                total_units.append(self.units_created)  # Total number created, ever

                # Node error
                for u in self.network.nodes():
                    self.network.node[u]['error'] *= d
                    if self.network.degree(nbunch=[u]) == 0:
                        print(u)

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

            