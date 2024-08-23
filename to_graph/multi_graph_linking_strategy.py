import hashlib
import networkx as nx


class StrategyLinkingMultipleGraphs:
    """Links multiple graphs together."""
    def __init__(self, graphs, num):
        self.graphs = graphs
        self.graph = None
        self.num = num

    def get_num(self):
        return self.num

    def set_graphs(self, graphs, order):
        pass

    def apply(self):
        pass


class StrategyLinkingMultipleGraphsByTimeCooccurrence(StrategyLinkingMultipleGraphs):
    """Links nodes from multiple graphs based on their sequential order."""
    def __init__(self, graphs):
        super().__init__(graphs, 1)

    def set_graphs(self, graphs, order):
        self.graphs = graphs
        return self

    def apply(self):
        """Connects graphs in a multivariate graph based on time co-ocurrence."""
        g = nx.Graph()

        min_size = None

        if isinstance(self.graphs, list):
            graphs = {}
            for i in range(len(self.graphs)):
                graphs[list(self.graphs[i].items())[0]] = list(self.graphs[i].values())[0]
            self.graphs = graphs

        for graph in self.graphs.values():
            if min_size == None or len(graph.nodes) < min_size:
                min_size = len(graph.nodes)

        for hash, graph in self.graphs.items():
            nx.set_node_attributes(graph, hash, 'id')
            i = 0
            for node in list(graph.nodes(data = True)):
                node[1]['order'] = i
                i += 1

        for graph in self.graphs.values():
            g = nx.compose(g, graph)

        i = 0
        j = 0
        for (node_11, node_12) in zip(list(g.nodes(data = True)), list(g.nodes)):

            i = 0
            for (node_21, node_22) in zip(list(g.nodes(data = True)), list(g.nodes)):
                if i == j:
                    i+=1
                    continue

                if node_11[1]['order'] == node_21[1]['order']:
                    g.add_edge(node_12, node_22, intergraph_binding = 'positional')
                i+=1
            j+=1

        self.graph = g

        return self.graph, self.graphs



class StrategyLinkingMultipleGraphsSlidingWindow(StrategyLinkingMultipleGraphs):
    """Sequentially links graphs made by sliding window mechanism."""
    def __init__(self, graphs, graph_order):
        super().__init__(graphs, 0)
        self.graph_order = graph_order

    def set_graphs(self, graphs, order):
        self.graphs = graphs
        self.graph_order = order
        return self
    
    def hash(self, graph):
        """Returns unique hash of this graph."""
        str_to_hash = str(graph.nodes()) + str(graph.edges())
        return hashlib.md5(str_to_hash.encode()).hexdigest()

    def apply(self):
        g = nx.Graph()
        graphs = {}

        for j in range(len(self.graphs)):
            h = nx.Graph()

            for i in range(len(self.graph_order[j])-1):
                g.add_edge(self.graphs[j][self.graph_order[j][i]], self.graphs[j][self.graph_order[j][i+1]])
                h.add_edge(self.graphs[j][self.graph_order[j][i]], self.graphs[j][self.graph_order[j][i+1]])
            graphs[self.hash(h)] = h

        self.graph = g
        self.graphs = graphs

        return self.graph, self.graphs


class LinkGraphs:
    """Builder class for linking multiple graphs, through which we can access linking strategies."""
    def __init__(self):
        self.graphs = None
        self.graph = None
        self.graph_order = None
        self.command_array = []

    def time_cooccurrence(self):
        """Notes that we want to connect graphs in a multivariate graph based on time co-ocurrance."""
        self.command_array.append(StrategyLinkingMultipleGraphsByTimeCooccurrence(self.graphs))
        return self

    def sliding_window(self):
        self.command_array.append(StrategyLinkingMultipleGraphsSlidingWindow(self.graphs, self.graph_order))
        return self

    def succession(self, strategy):
        return strategy.get_num()

    def link(self, graphs, graph_order):
        self.graphs = graphs
        self.graph_order = graph_order

        self.command_array.sort(key=self.succession)

        for strat in self.command_array:
            strat.set_graphs(self.graphs, graph_order)
            self.graph, self.graphs = strat.apply()

        return self.graph