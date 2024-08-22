import random


class NextValue:
    """Master class for choosing next value, when transforming graph back to time series."""
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
    """Chooses next value randomly."""
    def __init__(self):
        super().__init__()

    def append(self, sequence, graph, graph_index, index):
        index = random.randint(0, len(graph[1][self.att]) - 1)
        sequence.append(graph[1][self.att][index])
        return sequence

    def get_name(self):
        return "random"


class NextValueRandomSlidWin(NextValue):
    """Chooses next value randomly for graph made with sliding window mechanism."""
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
    """Chooses next value sequentially, in the same order as they were saved."""
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
    """Chooses next value sequentially for graph made with sliding window mechanism, in the same order as they were saved."""
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
    """Mother class for choosing next node among the neighbors of the previous node."""
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
    """Walks through all graphs in a multivariate graph and chooses next node randomly."""
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
    """Walks through one graph and chooses next node randomly.s"""
    def __init__(self):
        super().__init__()

    def next_node(self, i, graph_index, nodes, switch, node):
        """From neighbors of the previous node randomly chooses next node."""
        neighbors = set(self.graph.neighbors(node))
        neighbors = list(set(self.nodes[graph_index]) & neighbors)

        return random.choice(neighbors)

    def get_name(self):
        return "walkthrough one graph randomly"