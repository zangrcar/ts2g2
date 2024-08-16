"""This is just something I was tring to do but it failed"""



import networkx as nx
from core import model
import matplotlib.pyplot as plt
import hashlib



class TimeSeriesSource:
    def __init__(self, time_series = None, attribute = 'value'):
        self.time_series = time_series
        self.observers = []
    

    def from_csv(self, csv_read):
        """Gets data from csv file."""
        self.time_series = csv_read.from_csv()
        self.notify_observers()
        return self
    
    def from_xml(self, xml_read):
        """Gets data from xml file."""
        self.time_series = xml_read.from_xml()
        self.notify_observers()
        return self
    
    def data(self):
        return self.time_series
    
    def notify_observers(self):

        for observer in self.observers:
            observer.update()

    def add_observer(self, observer):
        if observer not in self.observers:
            self.observers.append(observer)
    
    def remove_observer(self, observer):
        if observer in self.observers:
            self.observers.remove(observer)
    

class TimeSeriesViewMaster:

    def __init__(self, source, time_series = None, attribute = 'value'):
        self.strategy = None
        self.graph = None
        self.orig_graph = None
        self.slid_graphs = []
        self.attribute = attribute
        self.source = source
        self.source.add_observer(self)
        if time_series is None:
            self.time_series = source.data()
        else:
            self.time_series = time_series
        self.function_sequence = []
        self.base_ts = self

    def set_base(self, ts):
        self.base_ts = ts
    
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
        
        func = (self.process, (), {'ts_processing_strategy': ts_processing_strategy})

        if func not in self.function_sequence:
            self.function_sequence.append(func)
        
        if ts_processing_strategy == None:
            return self

        return ts_processing_strategy.process(self.source, self.time_series)
        #to do: how to return efficiently

    def to_graph(self, strategy):
        pass

    def combine_identical_nodes(self):
        pass

    def draw(self, color = "black"):

        func = (self.draw, (), {'color': color})
        
        if func not in self.function_sequence:
            self.function_sequence.append(func)


        """Draws the created graph"""
        pos=nx.spring_layout(self. graph, seed=1)
        nx.draw(self.graph, pos, node_size=40, node_color=color)
        plt.show()
        return self
    
    def link(self, link_strategy):
        """Links nodes based on link_strategy. link_strategy is object of class Link."""
        
        self.function_sequence.append((self.link, (link_strategy), {}))
        self.graph = link_strategy.link(self.graph)
        return self
    
    def add_edge(self, node_1, node_2, weight=None):
        """Adds edge between node_1 and node_2."""

        self.function_sequence.append((self.add_edge, (node_1, node_2), {'weight': weight}))

        if weight == None:
            self.graph.add_edge(list(self.graph.nodes)[node_1], list(self.graph.nodes)[node_2])
        else:
            self.graph.add_edge(list(self.graph.nodes)[node_1], list(self.graph.nodes)[node_2], weight = weight)
        return self
    
    def hash(self):
        """Returns unique hash of this graph."""
        str_to_hash = str(self.graph.nodes()) + str(self.graph.edges())
        return hashlib.md5(str_to_hash.encode()).hexdigest()

    def update(self):
        self.time_series = self.source.data()
        for func, args, kwargs in self.function_sequence:
            func(*args, **kwargs)

        self.function_sequence = []
        
        return self

    
class TimeSeriesView(TimeSeriesViewMaster):
    def __init__(self, source, time_series = None, attribute = 'value'):
        super().__init__(source, time_series = time_series, attribute = attribute)
    
    def to_graph(self, strategy):
        """Converts time serie to graph using strategy strategy. 
        Parameter strategy must be of class Strategy or any of its subclasses."""
        
        func = (self.to_graph, (strategy), {})

        if func not in self.function_sequence:
            self.function_sequence.append(func)


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

        self.function_sequence.append((self.combine_identical_nodes, (), {}))

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

class TimeSeriesViewSlidWin(TimeSeriesViewMaster):
    def __init__(self, source, time_series = None, attribute = 'value', segments = None):
        super().__init__(source, time_series = time_series, attribute = attribute)
        self.segments = segments
    
    def to_graph(self, strategy):
        """Converts time serie to graph using strategy strategy. 
        Parameter strategy must be of class Strategy or any of its subclasses."""

        self.function_sequence.append((self.to_graph, (strategy), {}))

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

        self.function_sequence.append((self.combine_identical_nodes, (), {}))

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
    