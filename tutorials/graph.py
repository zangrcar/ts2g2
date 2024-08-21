import matplotlib.pyplot as plt
import random


class NextValue:
    def __init__(self):
        self.skip = 0
        self.att = 'value'
        self.dictionaries = None

    def append(self, sequence, graph, graph_index, index):
        pass

    def skip_every_x_steps(self, x):
        self.skip = x
        return self

    def get_skip(self):
        return self.skip

    def set_arguments(self, dictionary, att):
        self.dictionaries = dictionary
        self.att = att

    def get_name(self):
        pass

class NextValueRandom(NextValue):
    def __init__(self):
        super().__init__()
        
    def append(self, sequence, graph, graph_index, index):
        index = random.randint(0, len(graph[1][self.att]) - 1)
        sequence.append(graph[1][self.att][index])
        return sequence

    def get_name(self):
        return "random"

class NextValueRandomSlidWin(NextValue):
    def __init__(self):
        super().__init__()

    def append(self, sequence, graph, graph_index, index):
        nodes = list(graph.nodes(data = True))
        random.shuffle(nodes)

        for node in nodes:
            index = random.randint(0, len(node[1][self.att]) - 1)
            sequence.append(node[1][self.att][index])
        return sequence

    def get_name(self):
        return "random"

class NextValueSequential(NextValue):
    def __init__(self):
        super().__init__()

    def append(self, sequence, graph, graph_index, index):
        if int(self.dictionaries[graph_index][index]/2) >= len(list(graph[1][self.att])):
            self.dictionaries[graph_index][index] = 0
        
        ind = int(self.dictionaries[graph_index][index]/2)
        sequence.append(graph[1][self.att][ind])
        self.dictionaries[graph_index][index] += 1
        return sequence

    def get_name(self):
        return "sequential"

class NextValueSequentialSlidWin(NextValue):
    def __init__(self):
        super().__init__()

    def append(self, sequence, graph, graph_index, index):
        if int(self.dictionaries[graph_index][index]/2) >= len(list(list(graph.nodes(data=True))[0][1][self.att])):
            self.dictionaries[graph_index][index] = 0
    
        ind = int(self.dictionaries[graph_index][index]/2)

        for node in graph.nodes(data=True):
            sequence.append(node[1][self.att][ind])
    
        self.dictionaries[graph_index][index] += 1
        return sequence


    def get_name(self):
        return "sequential"


class NextNode:
    def __init__(self):
        self.change_graphs = 1
        self.graph = None
        self.nodes = None
        self.dictionaries = None
        self.att = 'value'

    def next_node(self, i, graph_index, nodes, switch, node):
        pass

    def change_graphs_every_x_steps(self, x):
        self.change_graphs = x
        return self
    
    def get_change(self):
        return self.change_graphs

    def set_arguments(self, graph, nodes, dictionaries, att):
        self.graph = graph
        self.nodes = nodes
        self.dictionaries = dictionaries
        self.att = att
    
    def get_name(self):
        pass

class NextNodeAllRandom(NextNode):
    def __init__(self):
        super().__init__()

    def next_node(self, i, graph_index, nodes, switch, node):
        """From neighbors of the previous node randomly chooses next node."""
        index = int((i/switch) % len(nodes))
        neighbors = set(self.graph.neighbors(nodes[index]))
        
        neighbors = list(set(self.nodes[graph_index]) & neighbors)
        return random.choice(neighbors)

    def get_name(self):
        return "walkthrough all graphs randomly"

class NextNodeOneRandom(NextNode):
    def __init__(self):
        super().__init__()

    def next_node(self, i, graph_index, nodes, switch, node):
        """From neighbors of the previous node randomly chooses next node."""
        neighbors = set(self.graph.neighbors(node))
        neighbors = list(set(self.nodes[graph_index]) & neighbors)
        
        return random.choice(neighbors)

    def get_name(self):
        return "walkthrough one graph randomly"

class NextNodeAllWeighted(NextNode):
    def __init__(self):
        super().__init__()

    def next_node(self, i, graph_index, nodes, switch, node):
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

    def get_name(self):
        return "walkthrough all graphs weighted"

class NextNodeOneWeighted(NextNode):
    def __init__(self):
        super().__init__()

    def next_node(self, i, graph_index, nodes, switch, node):
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

    def get_name(self):
        return "walkthrough one graph weighted"


class GraphMaster:
    """Superclass of classes GraphSlidWin and Graph"""
    def __init__(self, graph, strategy):
        self.graph = graph
        self.node_strategy = None
        self.value_strategy = None
        self.skip_values = 0
        self.time_series_len = 100
        self.sequences = None
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

    """Next 4 function set the parameters, that are later on used as a strategy for converting graph to time series."""
    """--->"""
    def next_node_strategy(self, strategy: NextNode):
        self.node_strategy = strategy
        self.switch_graphs = strategy.get_change()
        return self
    
    def next_value_strategy(self, strategy: NextValue):
        self.value_strategy = strategy
        self.skip_values = strategy.get_skip()
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
            self.plot_timeseries(self.sequences[j], f"next_node_strategy = {self.node_strategy.get_name()}, next_value = {self.value_strategy.get_name()}", "Date", "Value", self.colors[j])
        plt.show()

class GraphSlidWin(GraphMaster):
    """Class that converts graphs made using sliding window mechanism back to time series"""
    def __init__(self, graph):
        super().__init__(graph, "slid_win")
    
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


        self.value_strategy.set_arguments(dictionaries, self.att)
        self.node_strategy.set_arguments(self.graph, self.nodes, dictionaries, self.att)


        i = 0
        while len(self.sequences[0]) < self.time_series_len:
            for j in range(len(self.sequences)):

                index = 0
                for i in range(len(self.nodes[j])):
                    if(self.is_equal(current_nodes[j], list(self.graph.nodes)[i])):
                        index = i
                        break

                self.sequences[j] = self.value_strategy.append(self.sequences[j], current_nodes[j], j, index)
                if self.sequences[j][-1] == None:
                    return self
                
            for j in range(self.skip_values + 1):
                for k in range(len(current_nodes)):

                    current_nodes[k] = self.node_strategy.next_node(i, k, current_nodes, self.switch_graphs, current_nodes[0])

                    if(current_nodes[k] == None):
                        return self
                    
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
            
        self.value_strategy.set_arguments(dictionaries, self.att)
        self.node_strategy.set_arguments(self.graph, self.nodes, dictionaries, self.att)

        i = 0
        while len(self.sequences[0]) < self.time_series_len:
            
            for j in range(len(current_nodes)):

                index = 0

                for i in range(len(list(self.nodes[j]))):
                    if(current_nodes_data[j] == self.data_nodes[j][i]):
                        index = i
                        break
                
                self.sequences[j] = self.value_strategy.append(self.sequences[j], current_nodes_data[j], j, index)
                if self.sequences[j][-1] == None:
                    return

            for j in range(self.skip_values+1):
                for k in range(len(current_nodes)):
                    current_nodes[k] = self.node_strategy.next_node(i, k, current_nodes, self.switch_graphs, current_nodes[0])

                    new_index = self.nodes[k].index(current_nodes[k])
                    current_nodes_data[k] = self.data_nodes[k][new_index]
                    if(current_nodes[k] == None):
                        break
            
            i += 1
        return self
