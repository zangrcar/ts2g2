import networkx as nx
from core import model
import matplotlib.pyplot as plt
import hashlib


class TimeSeriesToGraphMaster:
    def __init__(self, time_series = None, attribute = 'value'):
        self.time_series = time_series
        self.strategy = None
        self.graph = None
        self.orig_graph = None
        self.slid_graphs = []
        self.attribute = attribute
        self.base_ts = self

    def set_base(self, ts):
        self.base_ts = ts

    def from_csv(self, csv_read):
        """Gets data from csv file."""
        self.time_series = csv_read.from_csv()
        return self
    
    def from_xml(self, xml_read):
        """Gets data from xml file."""
        self.time_series = xml_read.from_xml()
        return self
    
    def return_graph(self):
        """Returns graph."""
        return self.graph

    def return_original_graph(self):
        """Returns graph that still has all of the nodes."""
        if self.orig_graph == None:
            return self.graph
        return self.orig_graph
    
    def process(self, ts_processing_strategy = None):
        """Returns a TimeSeriesToGraph object that is taylored by ts_processing_strategy. 
        ts_processing_strategy is expected to be an object of a subclass of class TSprocess."""
        if ts_processing_strategy == None:
            return self

        return ts_processing_strategy.process(self.time_series)
        #to do: how to return efficiently

    def to_graph(self, strategy):
        pass

    def combine_identical_nodes(self):
        pass

    def draw(self, color = "black"):
        """Draws the created graph"""
        pos=nx.spring_layout(self. graph, seed=1)
        nx.draw(self.graph, pos, node_size=40, node_color=color)
        plt.show()
        return self
    
    def link(self, link_strategy):
        """Links nodes based on link_strategy. link_strategy is object of class Link."""
        self.graph = link_strategy.link(self.graph)
        return self
    
    def add_edge(self, node_1, node_2, weight=None):
        """Adds edge between node_1 and node_2."""
        if weight == None:
            self.graph.add_edge(list(self.graph.nodes)[node_1], list(self.graph.nodes)[node_2])
        else:
            self.graph.add_edge(list(self.graph.nodes)[node_1], list(self.graph.nodes)[node_2], weight = weight)
        return self
    
    def hash(self):
        """Returns unique hash of this graph."""
        str_to_hash = str(self.graph.nodes()) + str(self.graph.edges())
        return hashlib.md5(str_to_hash.encode()).hexdigest()


class TimeSeriesToGraph(TimeSeriesToGraphMaster):
    
    def __init__(self, time_series = None, attribute = 'value'):
        super().__init__(time_series = time_series, attribute = attribute)

    def to_graph(self, strategy):
        """Converts time serie to graph using strategy strategy. 
        Parameter strategy must be of class Strategy or any of its subclasses."""
        self.strategy = strategy
        
        g =  self.strategy.to_graph(model.TimeseriesArrayStream(self.time_series))
        self.graph = g.graph

        for i in range(len(self.graph.nodes)):
            old_value = self.graph.nodes[i][self.attribute]
            new_value = [old_value]
            self.graph.nodes[i][self.attribute] = new_value
        
        hash = self.hash()
        mapping = {node: f"{hash}_{node}" for node in self.graph.nodes}
        self.graph = nx.relabel_nodes(self.graph, mapping)

        nx.set_edge_attributes(self.graph, strategy.get_name(), "strategy")

        return self
    
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


class TimeSeriesToGraphSlidWin(TimeSeriesToGraphMaster):
    def __init__(self, time_series = None, attribute = 'value', segments = None):
        super().__init__(time_series = time_series, attribute = attribute)
        self.segments = segments

    def to_graph(self, strategy):
        """Converts time serie to graph using strategy strategy. 
        Parameter strategy must be of class Strategy or any of its subclasses."""
        self.strategy = strategy

        self.graph = nx.MultiGraph()
        for i in range(len(self.time_series)):
            self.slid_graphs.append(self.time_series[i].to_graph(strategy).return_graph())
        
        for i in range(len(self.slid_graphs)-1):
            self.graph.add_edge(self.slid_graphs[i], self.slid_graphs[i+1])
        
        for graph in self.graph.nodes:
            for i in range(len(graph.nodes)):
                old_value = list(graph.nodes(data = True))[i][1][self.attribute]
                new_value = [old_value]
                list(graph.nodes(data=True))[i][1][self.attribute] = new_value
        
        nx.set_edge_attributes(self.graph, "sliding window connection", "strategy")


        return self
    
    def combine_identical_nodes(self):
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
    


class MultivariateTimeSeriesToGraph:
    """Class that combines multiple time series."""
    def __init__(self):
        self.graphs = {}
        self.multi_graph = None
        self.attribute = 'value'
    
    def set_attribute(self, att):
        self.attribute = att
        return self
    
    def link(self, link_strategy):
        """links nodes based on object link_strategy of class Link"""
        if self.multi_graph == None:
            self.multi_graph = link_strategy.link(self.graphs)
        else:
            self.multi_graph = link_strategy.link(self.multi_graph)
        return self
    
    def return_graph(self):
        """Returns graph."""
        return self.multi_graph

    def add(self, time_serie):
        """Adds object time_serie of class TimeSeriesToGraph to a dictionary."""
        self.graphs[time_serie.hash()] = time_serie.return_original_graph()
        return self
    
    def combine_identical_nodes_win(self):
        """Combines nodes that are identical graphs in a graph made using sliding window mechanism."""
        for graph in self.graphs.values():
            
            for i, node_1 in enumerate(list(graph.nodes)):
                if node_1 not in self.multi_graph:
                    continue

                for node_2 in list(graph.nodes)[i+1:]:
                    if node_2 == None:
                        break
                    if node_2 not in self.multi_graph:
                        continue

                    if(self.hash(node_1) == self.hash(node_2)):
                        graph = self.__combine_nodes_win(graph, node_1, node_2, self.attribute)
            
        
        return

    def hash(self, graph):
        """Returns unique hash of this graph."""
        str_to_hash = str(graph.nodes()) + str(graph.edges())
        return hashlib.md5(str_to_hash.encode()).hexdigest()

    def combine_identical_nodes(self):
        """Combines nodes that have same value of attribute self.attribute"""
        if isinstance(self.multi_graph, nx.MultiGraph):
            self.combine_identical_nodes_win()
            return self

        for graph in self.graphs.values():

            for i, node_1 in enumerate(list(graph.nodes(data=True))):
                if node_1 not in graph:
                    continue

                for node_2 in list(graph.nodes(data=True))[i+1:]:
                    if node_2 == None:
                        break
                    if node_2 not in graph:
                        continue

                    if(node_1[self.attribute] == node_2[self.attribute]):
                        graph = self.__combine_nodes(graph, node_1, node_2, self.attribute)
        return self
    
    def get_graph_nodes(self):
        """returns all nodes of graph"""
        nodes = []
        for graph in self.graphs.values():
            nodes.append(list(graph.nodes))
        
        return nodes

    def get_graph_nodes_data(self):
        """Returns all nodes of graphs with their data."""
        nodes = []
        for graph in self.graphs.values():
            nodes.append(list(graph.nodes(data = True)))
        
        return nodes
    
    def draw(self, color = "black"):
        """Draws graph."""
        pos=nx.spring_layout(self.multi_graph, seed=1)
        colors = []

        #if you want to have different colored nodes from different graphs
        if isinstance(color, list):
            for i, graph in enumerate(self.get_graph_nodes()):
                for node in graph:
                    colors.append(color[i])
            
            print(len(list(colors)))

            nx.draw(self.multi_graph, pos, node_size=40, node_color=colors)
        
        else:
            nx.draw(self.multi_graph, pos, node_size=40, node_color=color)

        plt.show()
        return self

    def __combine_nodes(self, graph, node_1, node_2, att):
        """Combines nodes node_1 and node_2."""
        node_1[att].append(node_2[att])
        for neighbor in list(graph.neighbors(node_2)):
            graph.add_edge(node_1, neighbor)
        
        graph.remove_node(node_2)
        return graph

    def __combine_nodes_win(self, graph, node_1, node_2, att):
        """Combines nodes node_1 and node_2, that are graphs."""
        
        for i in range(len(list(node_1.nodes(data=True)))):
            for j in range(len(list(node_2.nodes(data=True))[i][1][att])):
                list(node_1.nodes(data=True))[i][1][att].append(list(node_2.nodes(data=True))[i][1][att][j])
        
        for neighbor in list(self.multi_graph.neighbors(node_2)):
            self.multi_graph.add_edge(node_1, neighbor)
        
        #for neighbor in list(graph.neighbors(node_2)):
            #graph.add_edge(node_1, neighbor)

        self.multi_graph.remove_node(node_2)
        #graph.remove_node(node_2)
        return graph

