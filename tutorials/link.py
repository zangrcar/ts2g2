import networkx as nx

class Link:
    def __init__(self, graph = None, multi = False, att = 'value'):
        self.seasonalites = False
        self.same_timestep = -1
        self.graph = graph
        self.coocurrence = False
        self.multi = multi
        self.period = None
        self.attribute = att
    
    def seasonalities(self, period):
        """Notes that we want to connect based on seasonalities, ad sets the period parameter."""
        self.seasonalites = True
        self.period = period
        return self

    
    def by_value(self, strategy):
        """Notes that we want to connect nodes based on values and strategy."""
        if isinstance(strategy, SameValue):
            self.same_timestep = strategy.get_all_diff()
        return self

    def time_coocurence(self):
        """Notes that we want to connect graphs in a multivariate graph based on time co-ocurrance."""
        self.coocurrence = True
        return self
    
    def link_positional(self, graph):
        """Connects graphs in a multivariate graph based on time co-ocurrance."""
        g = None
        if self.multi:
            g = nx.MultiGraph()
        else :
            g = nx.Graph()

        min_size = None
        
        for graph in self.graph.values():
            if min_size == None or len(graph.nodes) < min_size:
                min_size = len(graph.nodes)

        for hash, graph in self.graph.items():
            nx.set_node_attributes(graph, hash, 'id')
            i = 0
            for node in list(graph.nodes(data = True)):
                node[1]['order'] = i
                i += 1
        
        for graph in self.graph.values():
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

    def link_seasonalities(self):
        """Links nodes that are self.period instances apart."""
        for i in range(len(self.graph.nodes) - self.period):
            self.graph.add_edge(list(self.graph.nodes)[i], list(self.graph.nodes)[i+self.period], intergraph_binding='seasonality')

    def link_same_timesteps(self):
        """Links nodes whose values are at most sels.same_timestep apart."""
        for node_11, node_12 in zip(self.graph.nodes(data=True), self.graph.nodes):
            for node_21, node_22 in zip(self.graph.nodes(data=True), self.graph.nodes):
                if  abs(node_11[1][self.attribute][0] - node_21[1][self.attribute][0]) < self.same_timestep and node_12 != node_22:
                    self.graph.add_edge(node_12, node_22, intergraph_binding = 'timesteps')

    def link(self, graph):
        """Calls functions to link nodes based on what we set before."""
        self.graph = graph

        if self.seasonalites:
            self.link_seasonalities()

        if self.same_timestep > 0:
            self.link_same_timesteps()
        
        if self.coocurrence:
            self.link_positional(graph)

        return self.graph

class SameValue:
    "Class that notes that we want to connect nodes based on similarity of values."
    def __init__(self, allowed_difference):
        self.allowed_difference = allowed_difference
    
    def get_all_diff(self):
        return self.allowed_difference