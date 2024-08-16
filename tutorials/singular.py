import networkx as nx
from core import model
import matplotlib.pyplot as plt
import hashlib
import builder.graph_strategy as gs
import link

class TimeSeries:
    def __init__(self, time_series):
        self.time_series = time_series
    
    def get_ts(self):
        return self.time_series

class TimeSeriesPreprocessing:
    def __init__(self, ts: TimeSeries):
        self.ts = ts.get_ts()
        self.segments = None
    
    def segmentation(self, segment_start, segment_end):
        self.ts = self.ts[segment_start:segment_end]
        return self
    
    def sliding_window(self, win_size, move_len = 1):
        segments = []
        for i in range(0, len(self.ts) - win_size, move_len):
            segments.append(self.ts[i:i + win_size])
        self.segments = segments
        return self
    
    def process(self):
        if self.segments == None:
            return TimeSeriesView([self.ts])
        else:
            return TimeSeriesView(self.segments)

class TimeSeriesView:
    def __init__(self, ts, attribute = 'value'):
        self.ts = ts
        self.graphs = {}
        self.attribute = attribute
        self.graph = None
        self.graph_order = []
    
    def get_ts(self):
        return self.ts[0]

    def add(self, ts: TimeSeries):
        self.ts.append(ts.get_ts())
        return self

    def to_graph(self, strategy: gs.Strategy):
        
        for time_series in self.ts:
            g =  strategy.to_graph(model.TimeseriesArrayStream(time_series))
            g = g.graph

            for i in range(len(g.nodes)):
                old_value = g.nodes[i][self.attribute]
                new_value = [old_value]
                g.nodes[i][self.attribute] = new_value
            
            hash = self.hash(g)
            mapping = {node: f"{hash}_{node}" for node in g.nodes}
            g = nx.relabel_nodes(g, mapping)

            nx.set_edge_attributes(g, strategy.get_name(), "strategy")
            self.graphs[self.hash(g)] = g
            self.graph_order.append(self.hash(g))
            if self.graph == None:
                self.graph = g

        if len(self.ts) == 1:
            return Graph(self.graph)
        else: 
            return self
    
    def link(self, link_strategy: link.LinkGraphs):
        return Graph(link_strategy.link(self.graphs, self.graph_order), graphs = self.graphs)

    def get_graphs(self):
        return self.graphs
    
    def get_graph(self):
        if self.graph == None:
            return Graph(list(self.graphs.values())[0])
        else:
            return Graph(self.graph)
    
    def hash(self, graph):
        """Returns unique hash of this graph."""
        str_to_hash = str(graph.nodes()) + str(graph.edges())
        return hashlib.md5(str_to_hash.encode()).hexdigest()
            
class Graph:
    def __init__(self, graph, graphs = None):
        self.graph = graph
        self.orig_graph = None
        self.graphs = graphs
    
    def get_graph(self):
        return self.graph
    
    def get_graphs(self):
        return self.graphs

    def add_edge(self, node_1, node_2, weight=None):
        """Adds edge between node_1 and node_2."""
        if weight == None:
            self.graph.add_edge(list(self.graph.nodes)[node_1], list(self.graph.nodes)[node_2])
        else:
            self.graph.add_edge(list(self.graph.nodes)[node_1], list(self.graph.nodes)[node_2], weight = weight)
        return self
    
    def link(self, link_strategy: link.LinkOne):
        """Links nodes based on link_strategy. link_strategy is object of class Link."""
        self.graph = link_strategy.link(self)
        return self

    def hash(self):
        """Returns unique hash of this graph."""
        str_to_hash = str(self.graph.nodes()) + str(self.graph.edges())
        return hashlib.md5(str_to_hash.encode()).hexdigest()
    
    def combine_identical_nodes_slid_win(self):
        """Combines nodes that have same value of attribute self.attribute if graph is classical graph and
        nodes that are identical graphs if graph is created using sliding window mechanism."""
        self.orig_graph = self.graph.copy()

        for i, node_1 in enumerate(list(self.graph.nodes)):
            if node_1 not in self.graph:
                continue

            for node_2 in list(self.graph.nodes)[i+1:]:
                if node_2 == None:
                    break
                if node_2 not in self.graph:
                    continue

                if(set(list(node_1.edges)) == set(list(node_2.edges))):
                    self.graph = self.__combine_nodes_win(self.graph, node_1, node_2, self.attribute)

        return self

    def __combine_nodes_win(self, graph, node_1, node_2, att):
        """Combines nodes node_1 and node_2, that are graphs."""
        for i in range(len(list(node_1.nodes(data=True)))):
            for j in range(len(list(node_2.nodes(data=True))[i][1][att])):
                list(node_1.nodes(data=True))[i][1][att].append(list(node_2.nodes(data=True))[i][1][att][j])
        
        for neighbor in list(graph.neighbors(node_2)):
            graph.add_edge(node_1, neighbor)

        graph.remove_node(node_2)
        return graph
    
    def combine_identical_nodes(self):
        """Combines nodes that have same value of attribute self.attribute if graph is classical graph and
        nodes that are identical graphs if graph is created using sliding window mechanism."""
        self.orig_graph = self.graph.copy()
        
        for i, node_1 in enumerate(list(self.graph.nodes(data=True))):
            if node_1 not in self.graph:
                continue

            for node_2 in list(self.graph.nodes(data=True))[i+1:]:
                if node_2 == None:
                    break
                if node_2 not in self.graph:
                    continue

                if(node_1[self.attribute] == node_2[self.attribute]):
                    self.graph = self.__combine_nodes(self.graph, node_1, node_2, self.attribute)
            
        return self
    
    def __combine_nodes(self, graph, node_1, node_2, att):
        """Combines nodes node_1 and node_2."""
        node_1[att].append(node_2[att])
        for neighbor in list(graph.neighbors(node_2)):
            graph.add_edge(node_1, neighbor)
        
        graph.remove_node(node_2)
        return graph

    def draw(self, color = "black"):
        """Draws the created graph"""
        pos=nx.spring_layout(self.graph, seed=1)
        nx.draw(self.graph, pos, node_size=40, node_color=color)
        plt.show()
        return self
