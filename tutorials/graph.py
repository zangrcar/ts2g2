import matplotlib.pyplot as plt
import random


class GraphMaster:
    """Superclass of classes GraphSlidWin and Graph"""
    def __init__(self, graph, strategy):
        self.graph = graph
        self.next_node_strategy = "random"
        self.next_value_strategy = "random"
        self.skip_values = 0
        self.time_series_len = 100
        self.sequences = None
        self.walk = "one"
        self.switch_graphs = 1
        self.colors = None
        self.nodes = None
        self.data_nodes = None
        self.strategy = strategy
        self.att = 'value'
    
    def set_nodes(self, nodes, data_nodes):
        """Sets parameters to be used if we have multivariate graph and need nodes from specific original graph."""
        pass
    
    def set_nodes(self, dict: dict):
        pass

    def set_attribute(self, att):
        self.att = att

    """Next 6 function set the parameters, that are later on used as a strategy for converting graph to time series."""
    """--->"""
    def walk_through_all(self):
        self.walk = "all"
        return self
    
    def change_graphs_every_x_steps(self, x):
        self.switch_graphs = x
        return self
    
    def choose_next_node(self, strategy):
        self.next_node_strategy = strategy
        return self
    
    def choose_next_value(self, strategy):
        self.next_value_strategy = strategy
        return self
    
    def skip_every_x_steps(self, x):
        self.skip_values = x
        return self
    
    def ts_length(self, x):
        self.time_series_len = x
        return self
    """<---"""

    def to_time_sequence(self):
        """Adjusts parameters nodes and data_nodes to fit function to_multiple_time_sequences."""
        pass
    
    def to_multiple_time_sequences(self):
        """Converts graph into time sequences."""
        pass

    def is_equal(self, graph_1, graph_2):
        """Compares two graphs if they are equal."""
        if(graph_1.nodes != graph_2.nodes): return False
        if(graph_1.edges != graph_2.edges): return False
        for i in range(len(graph_1.nodes)):
            if list(list(graph_1.nodes(data=True))[i][1][self.att]) != list(list(graph_2.nodes(data=True))[i][1][self.att]):
                    return False
        return True

    def plot_timeseries(self, sequence, title, x_legend, y_legend, color):
        """Function to sets parameters to draw time series."""
        plt.figure(figsize=(10, 6))
        plt.plot(sequence, linestyle='-', color=color)
        
        plt.title(title)
        plt.xlabel(x_legend)
        plt.ylabel(y_legend)
        plt.grid(True)

    def draw(self):
        """Draws time series."""
        if self.colors == None:
            self.colors = []
            for j in range(len(self.sequences)):
                self.colors.append("black")
        
        for j in range(len(self.sequences)):
            self.plot_timeseries(self.sequences[j], f"walk = {self.walk}, next_node_strategy = {self.next_node_strategy}, next_value = {self.next_value_strategy}", "Date", "Value", self.colors[j])
        plt.show()

class GraphSlidWin(GraphMaster):
    """Class that converts graphs made using sliding window mechanism back to time series"""
    def __init__(self, graph):
        super().__init__(graph, "slid_win")
    
    """
    def set_nodes(self, nodes):
        self.nodes = nodes
        return self
        """
    
    def set_nodes(self, dicts: dict):

        if isinstance(dicts, list):
            graphs = [{} for _ in range(len(dicts))]
            for i in range(len(dicts)):
                for j in range(len(dicts[i].values())):
                    graphs[i][list(dicts[i].items())[j]] = list(dicts[i].values())[j]
            dicts = graphs

        self.nodes = [[] for _ in range(len(dicts))]

        for i in range(len(dicts)):
            for graph in dicts[i].values():
                self.nodes[i].append(graph)
        return self

    def to_time_sequence(self):
        #self.nodes = [list(self.nodes)]
        return self.to_multiple_time_sequences()


    def to_multiple_time_sequences(self):

        self.sequences = [[] for _ in range(len(self.nodes))]

        current_nodes = [None for _ in range(len(self.nodes))]

        for i in range(len(self.nodes)):
            current_nodes[i] = list(self.nodes[i])[0]
        
        dictionaries = [{} for _ in range(len(self.nodes))]
        for i in range(len(self.nodes)):
            for j in range(len(list(self.nodes[i]))):
                dictionaries[i][j] = 0

        
        strategy = ChooseStrategySlidWin(self.walk, self.next_node_strategy, self.next_value_strategy, self.graph, self.nodes, dictionaries, self.att)

        i = 0
        while len(self.sequences[0]) < self.time_series_len:
            for j in range(len(self.sequences)):

                index = 0
                for i in range(len(self.nodes[j])):
                    if(self.is_equal(current_nodes[j], list(self.graph.nodes)[i])):
                        index = i
                        break

                self.sequences[j] = strategy.append(self.sequences[j], current_nodes[j], j, index)
                if self.sequences[j][-1] == None:
                    return self
                
            for j in range(self.skip_values + 1):
                for k in range(len(current_nodes)):

                    current_nodes[k] = strategy.next_node(i, k, current_nodes, self.switch_graphs)

                    if(current_nodes[k] == None):
                        return self
                    
            
            """
            for k in range(len(current_nodes)):
                    current_nodes[k] = strategy.next_node(i, k, current_nodes, self.switch_graphs)
                    if(current_nodes[k] == None):
                        break
            """
            i += 1
        return self

class GraphToTS(GraphMaster):
    """Class that turns ordinary graphs back to time series."""
    def __init__(self, graph):
        super().__init__(graph, "classic")
    
    def set_nodes(self, nodes, data_nodes):
        self.nodes = nodes
        self.data_nodes = data_nodes
        return self

    def set_nodes(self, dict: dict):
        
        if isinstance(dict, list):
            graphs = {}
            for i in range(len(dict)):
                graphs[list(dict[i].items())[0]] = list(dict[i].values())[0]
            dict = graphs

        self.nodes = []
        self.data_nodes = []
        for graph in dict.values():
            self.nodes.append(list(graph.nodes))
            self.data_nodes.append(list(graph.nodes(data=True)))
        return self

    def to_time_sequence(self):
        self.nodes = [list(self.nodes)]
        self.data_nodes = [list(self.data_nodes)]
        return self.to_multiple_time_sequences()

    def to_multiple_time_sequences(self):
    
        self.sequences = [[] for _ in range(len(self.nodes))]

        current_nodes = [None for _ in range(len(self.nodes))]
        current_nodes_data = [None for _ in range(len(self.data_nodes))]

        for i in range(len(self.nodes)):
            current_nodes[i] = self.nodes[i][0]
            current_nodes_data[i] = self.data_nodes[i][0]
        
        dictionaries = [{} for _ in range(len(self.nodes))]
        for i in range(len(self.nodes)):
            for j in range(len(list(self.nodes[i]))):
                dictionaries[i][j] = 0

        strategy = ChooseStrategy(self.walk, self.next_node_strategy, self.next_value_strategy, self.graph, self.nodes, dictionaries, self.att)

        i = 0
        while len(self.sequences[0]) < self.time_series_len:
            
            for j in range(len(current_nodes)):

                index = 0

                for i in range(len(list(self.nodes[j]))):
                    if(current_nodes_data[j] == self.data_nodes[j][i]):
                        index = i
                        break
                
                self.sequences[j] = strategy.append(self.sequences[j], current_nodes_data[j], j, index)
                if self.sequences[j][-1] == None:
                    return

            for j in range(self.skip_values+1):
                for k in range(len(current_nodes)):
                    current_nodes[k] = strategy.next_node(i, k, current_nodes, self.switch_graphs)

                    new_index = self.nodes[k].index(current_nodes[k])
                    current_nodes_data[k] = self.data_nodes[k][new_index]
                    if(current_nodes[k] == None):
                        break
            
            i += 1
        return self

class ChooseStrategyMaster:
    def __init__(self, walk, next_node_strategy, value, graph, nodes, dictionaries, att):
        self.next_node_strategy = next_node_strategy
        self.walk = walk
        self.value = value
        self.graph = graph
        self.nodes = nodes
        self.dictionaries = dictionaries
        self.att = att
    
    def append_random(self, sequence, graph):
        """To a sequence appends a random value of a node it is currently on."""
        pass

    def append_lowInd(self, sequence, graph, graph_index, index):
        """To a sequence appends a successive value of a node it is currently on."""
        pass

    def append(self, sequence, graph, graph_index, index):
        if(self.value) == "random":
            return self.append_random(sequence, graph)
        elif self.value == "sequential" :
           return self.append_lowInd(sequence, graph, graph_index, index)
        else:
            print("you chose non-existent method of value selection")
            print("please choose between: random, sequential")
            return None
    
    def next_node_one_random(self, graph_index, node):
        """From neighbors of the previous node randomly chooses next node."""
        neighbors = set(self.graph.neighbors(node))
        neighbors = list(set(self.nodes[graph_index]) & neighbors)
        
        return random.choice(neighbors)

    def next_node_one_weighted(self, graph_index, node):
        """From neighbors of the previous node chooses next one based on number of connections between them."""
        neighbors = set(self.graph.neighbors(node))
        neighbors = list(set(self.nodes[graph_index]) & neighbors)

        weights = []
        total = 0
        for neighbor in neighbors:
            num = self.graph.number_of_edges(node, neighbor)
            weights.append(num)
            total += num
        for element in weights:
            element /= total
        
        return random.choices(neighbors, weights=weights, k=1)[0]

    def next_node_one(self, graph_index, node):
        """If we have multivariate graph this function walks on first ones and 
        for others graphs chooses next node based on neighbors of the node in first graph."""
        if self.next_node_strategy == "random":
            return self.next_node_one_random(graph_index, node)
        elif  self.next_node_strategy == "weighted":
            return self.next_node_one_weighted(graph_index, node)
        else:
            print("you chose non-existent next_node_strategy.")
            print("please choose between: random, weighted")
            return None
    
    def next_node_all_random(self, i, graph_index, nodes, switch):
        """From neighbors of the previous node randomly chooses next node."""
        index = int((i/switch) % len(nodes))
        neighbors = set(self.graph.neighbors(nodes[index]))
        
        neighbors = list(set(self.nodes[graph_index]) & neighbors)
        return random.choice(neighbors)
    
    def next_node_all_weighted(self, i, graph_index, nodes, switch):
        """From neighbors of the previous node chooses next one based on number of connections between them."""
        index = int((i/switch) % len(nodes))
        neighbors = set(self.graph.neighbors(nodes[index]))
        neighbors = list(set(self.nodes[graph_index]) & neighbors)
        
        weights = []
        total = 0

        for neighbor in neighbors:
            num = self.graph.number_of_edges(nodes[index], neighbor)
            weights.append(num)
            total += num
        
        for element in weights:
            element /= total
        
        return random.choices(neighbors, weights=weights, k=1)[0]
    
    def next_node_all(self, i, graph_index, nodes, switch):
        """If we have multivariate graph this function walks on all graphs and switches between them every 'switch' steps and
        for others graphs chooses next node based on neighbors of the node in cuurent graph."""
        if self.next_node_strategy == "random":
            return self.next_node_all_random(i, graph_index, nodes, switch)
        elif  self.next_node_strategy == "weighted":
            return self.next_node_all_weighted(i, graph_index, nodes, switch)
        else:
            print("you chose non-existent next_node_strategy.")
            print("please choose between: random, weighted")
            return None
    
    def next_node(self, i, graph_index, nodes, switch):
        if self.walk == "one":
            return self.next_node_one(graph_index, nodes[0])
        elif self.walk == "all":
            return self.next_node_all(i, graph_index, nodes, switch)
        else:
            print("you chose non-existent walk")
            print("please choose between: one, all")
            return None
    
class ChooseStrategy(ChooseStrategyMaster):
    """Subclass that alters few methods to fit normal graph."""
    def __init__(self, walk, next_node_strategy, value, graph, nodes, dictionaries, att):
        super().__init__(walk, next_node_strategy, value, graph, nodes, dictionaries, att)
    
    def append_random(self, sequence, graph):
        index = random.randint(0, len(graph[1][self.att]) - 1)
        sequence.append(graph[1][self.att][index])
        return sequence
    
    def append_lowInd(self, sequence, graph, graph_index, index):
        if int(self.dictionaries[graph_index][index]/2) >= len(list(graph[1][self.att])):
            self.dictionaries[graph_index][index] = 0
        
        ind = int(self.dictionaries[graph_index][index]/2)
        sequence.append(graph[1][self.att][ind])
        self.dictionaries[graph_index][index] += 1
        return sequence

class ChooseStrategySlidWin(ChooseStrategyMaster):
    """Subclass that alters few methods to fit graph made using sliding window mechanism."""
    def __init__(self, walk, next_node_strategy, value, graph, nodes, dictionaries, att):
        super().__init__(walk, next_node_strategy, value, graph, nodes, dictionaries, att)
    
    def append_random(self, sequence, graph):

        nodes = list(graph.nodes(data = True))
        random.shuffle(nodes)

        for node in nodes:
            index = random.randint(0, len(node[1][self.att]) - 1)
            sequence.append(node[1][self.att][index])
        return sequence

    def append_lowInd(self, sequence, graph, graph_index, index):
        
        if int(self.dictionaries[graph_index][index]/2) >= len(list(list(graph.nodes(data=True))[0][1][self.att])):
            self.dictionaries[graph_index][index] = 0
    
        ind = int(self.dictionaries[graph_index][index]/2)

        for node in graph.nodes(data=True):
            sequence.append(node[1][self.att][ind])
    
        self.dictionaries[graph_index][index] += 1
        return sequence
