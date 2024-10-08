from deprecated import deprecated
from from_graph.strategy_to_time_sequence import StrategyNextValueInNode
from to_graph.strategy_linking_graph import LinkNodesWithinGraph
from from_graph.strategy_to_time_sequence import StrategySelectNextNode

import matplotlib.pyplot as plt
import networkx as nx
import hashlib
import numpy as np

from to_graph.strategy_linking_multi_graphs import LinkGraphs
import to_graph.strategy_to_graph
from to_graph.strategy_to_graph import BuildStrategyForTimeseriesToGraph
import copy
import math


class StrategyNotImplementedError(Exception):
    """Custom exception for strategies that are not implemented."""
    pass

class Timeseries:
    """Saves extracted data as timeseries."""
    def __init__(self, timeseries):
        self.timeseries = timeseries

    def get_ts(self):
        return self.timeseries
    
    def with_preprocessing(self, strategy):
        return strategy.process(self.timeseries)

class TimeseriesPreprocessing:
    """Processes timeseries."""
    def __init__(self):
        self.ts = None

    def process(self, ts):
        self.ts = ts
        return TimeseriesView([ts])


class TimeseriesPreprocessingSegmentation(TimeseriesPreprocessing):
    """
    Returns a designated segment from timeseries.
    
    **Attributes:**

    - `segmen_start`: start of the segment
    - `segment_end`: end of the segment
    
    """
    def __init__(self, segment_start, segment_end):
        self.seg_st = segment_start
        self.seg_end = segment_end
        self.ts = None

    def process(self, ts):
        self.ts = ts[self.seg_st:self.seg_end]
        return TimeseriesView([self.ts])


class TimeseriesPreprocessingSlidingWindow(TimeseriesPreprocessing):
    """
    Returns an array of segments made with sliding window mechanism.
    
    **Attributes:**

    - `win_size`: size of the creted segments
    - `move_len`: tells for how many data does window move until next segment
    
    """
    def __init__(self, win_size, move_len = 1):
        self.win_size = win_size
        self.move_len = move_len
        self.segments = []

    def process(self, ts):
        if isinstance(ts, list):
            ts = ts[0]
        for i in range(0, len(ts) - self.win_size, self.move_len):
            self.segments.append(ts[i:i + self.win_size])
        return TimeseriesView(self.segments)


class TimeseriesPreprocessingComposite():
    """
    Composites processing strategies, allowing us to use multiple of them.
    
    **Attributes:**

    - `ts`: Timeseries object with extracted timeseries
    """
    def __init__(self):
        self.ts = None
        self.segments = None
        self.strategy = []

    def add(self, strat):
        self.strategy.append(strat)
        return self
    
    def process(self, ts):
        self.ts = ts
        for strat in self.strategy:
            self.ts = strat.process(self.ts).get_ts()[0]
        return TimeseriesView(self.ts)

class TimeseriesStream:
  def read(self):
      return None

class TimeseriesArrayStream(TimeseriesStream):
    def __init__(self, array):
        self.array = copy.deepcopy(array)

    def read(self):
        return copy.deepcopy(self.array)


class TimeseriesView:
    """
    Stores one or more already processed timeseries, then changes them to graph using provided strategy.
    If we have multiple timie series turned to graphs, we can link them into one multivariate graph.
    
    **Attributes:**

    - `ts`: processed timeseries
    - `graph`: networkx.Graph object
    
    """
    
    def __init__(self, ts, attribute = 'value'):
        self.ts = [ts]
        self.graphs = []
        self.attribute = attribute
        self.graph = None
        self.graph_order = []
        self.histogram_frequencies = None
        self.histogram_bins = None
        self.w = 1
        self.tau = 1
        self.is_implemented = True
        self.quantiles = None
        self.quantile_values = None

    def get_ts(self):
        return self.ts

    def add(self, ts):
        series = ts.get_ts()
        for time_ser in series:
            self.ts.append(time_ser)
        return self

    def to_graph(self, strategy: to_graph.strategy_to_graph):
        pool = ["#000000", "#101010", "#202020", "#303030", "#404040", "#505050", "#606060", "#707070", "#808080", "#909090", "#a0a0a0", "#b0b0b0", "#c0c0c0", "#d0d0d0", "#e0e0e0", "#f0f0f0"]
        num = math.floor(16/len(self.ts))
        if len(self.ts) > 16:
            num = 1
        color_counter = 0
        self.w, self.tau = strategy._get_w_tau()
        for ts in self.ts:
            graph_dict = {}
            order=[]

            counter = 0
            for timeseries in ts:
                g =  strategy.to_graph(TimeseriesArrayStream(timeseries))
                g = g.graph
                
                if strategy._has_value():
                    for i in range(len(g.nodes)):
                        old_value = g.nodes[i][self.attribute]
                        new_value = [old_value]
                        g.nodes[i][self.attribute] = new_value

                hash = self._hash(g)
                mapping = {node: f"{hash}_{node}" for node in g.nodes}
                g = nx.relabel_nodes(g, mapping)

                nx.set_node_attributes(g, pool[color_counter], "color")
                color_counter += num
                if color_counter > 15:
                    color_counter = 0
                

                nx.set_edge_attributes(g, strategy._get_name(), "strategy")
                graph_dict[self._hash(g) + f"_{counter}"] = g
                order.append(self._hash(g) + f"_{counter}")

                if self.graph == None:
                    self.graph = g

                counter+=1

            self.graphs.append(graph_dict)
            self.graph_order.append(order)
        
        self.quantiles, self.quantile_values = strategy._get_bins()
        
        if (len(self.ts) == 1 and len(self.ts[0]) == 1):
            return TimeGraph(self.graph, graphs = self.graphs[0], is_implemented=strategy._has_implemented_to_ts(), histogram_frequencies = self.histogram_frequencies, histogram_bins = self.histogram_bins, w = self.w, tau = self.tau, quantiles = self.quantiles, quantile_values=self.quantile_values)
        else:
            self.is_implemented = strategy._has_implemented_to_ts()
            return self

    def link(self, link_strategy: LinkGraphs):
        return TimeGraph(link_strategy.link(self.graphs, self.graph_order, self.ts), graphs = self.graphs, is_implemented=self.is_implemented, histogram_frequencies = self.histogram_frequencies, histogram_bins = self.histogram_bins, w = self.w, tau = self.tau, quantiles = self.quantiles, quantile_values=self.quantile_values)

    def _get_graphs(self):
        return self.graphs

    def _get_graph(self):
        if self.graph == None:
            return TimeGraph(list(self.graphs[0].values())[0])
        else:
            return TimeGraph(self.graph)

    def _hash(self, graph):
        """Returns unique hash of this graph."""
        str_to_hash = str(graph.nodes()) + str(graph.edges())
        return hashlib.md5(str_to_hash.encode()).hexdigest()
    
    def to_histogram(self, bins):
        self.histogram_frequencies = []
        self.histogram_bins = []
        for i in range(len(self.ts)):
            a, b = np.histogram(self.ts[i], bins = bins)
            self.histogram_frequencies.append(a)
            self.histogram_bins.append(b)

        return self


class TimeGraph:
    """
    Stores already made graph, allows us to add edges and links between nodes.
    
    **Attributes:**

    - `graph`: object networkx.Graph
    
    """
    
    def __init__(self, graph, graphs = None, is_implemented = True, histogram_frequencies = None, histogram_bins = None, w = 1, tau = 1, quantiles = None, quantile_values = None):
        self.graph = graph
        self.orig_graph = None
        self.graphs = graphs
        self.attribute = 'value'
        self.sequences = None
        self.sequence_visitor = None
        self.is_implemented = is_implemented
        self.histogram_frequencies = histogram_frequencies
        self.histogram_bins = histogram_bins
        self.w = w
        self.tau = tau
        self.embeddings = None
        self.quantiles = quantiles
        self.quantile_values = quantile_values

    def get_is_implemented(self):
        return self.is_implemented

    def _get_graph(self):
        return self.graph

    def _get_graphs(self):
        return self.graphs

    def add_edge(self, node_1, node_2, weight=None):
        """Adds edge between node_1 and node_2."""
        if weight == None:
            self.graph.add_edge(list(self.graph.nodes)[node_1], list(self.graph.nodes)[node_2], color = "#66ffff")
        else:
            self.graph.add_edge(list(self.graph.nodes)[node_1], list(self.graph.nodes)[node_2], weight = weight, color = "#66ffff")
        return self

    def link(self, link_strategy: LinkNodesWithinGraph):
        """Links nodes based on link_strategy. link_strategy is object of class Link."""
        self.graph = link_strategy.link(self)
        return self

    def _hash(self):
        """Returns unique hash of this graph."""
        str_to_hash = str(self.graph.nodes()) + str(self.graph.edges())
        return hashlib.md5(str_to_hash.encode()).hexdigest()

    def combine_identical_subgraphs(self):
        """Combines nodes that have same value of attribute self.attribute if graph is classical graph and
        nodes that are identical graphs if graph is created using sliding window mechanism."""
        self.orig_graph = self.graph.copy()
        for j in range(len(self.graphs)):
            for i, node_1 in enumerate(list(self.graphs[j].values())):
                if node_1 not in self.graph:
                    continue

                for key, node_2 in list(self.graphs[j].items())[i+1:]:
                    if node_2 == None:
                        break
                    if node_2 not in self.graph:
                        continue

                    if(set(list(node_1.edges)) == set(list(node_2.edges))):
                        self.graph = self._combine_subgraphs(self.graph, node_1, node_2, self.attribute)
                        del self.graphs[j][key]
                        

        return self

    def _combine_subgraphs(self, graph, node_1, node_2, att):
        """Combines nodes node_1 and node_2, that are graphs."""
        for i in range(len(list(node_1.nodes(data=True)))):
            for j in range(len(list(node_2.nodes(data=True))[i][1][att])):
                list(node_1.nodes(data=True))[i][1][att].append(list(node_2.nodes(data=True))[i][1][att][j])

        for neighbor in list(graph.neighbors(node_2)):
            graph.add_edge(node_1, neighbor, graph_binding = "sliding window", color = "#00ff00")

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
                    self.graph = self._combine_nodes(self.graph, node_1, node_2, self.attribute)

        return self

    def _combine_nodes(self, graph, node_1, node_2, att):
        """Combines nodes node_1 and node_2."""
        node_1[att].append(node_2[att])
        for neighbor in list(graph.neighbors(node_2)):
            graph.add_edge(node_1, neighbor)

        graph.remove_node(node_2)
        return graph

    #TODO: add node weights and names to draw function
    def draw(self):
        """Draws the created graph"""
        
        pos=nx.spring_layout(self.graph, seed=1)
        
        nodes_missing_color = [(u, d) for u, d in self.graph.nodes(data=True) if 'color' not in d]

        edge_colors = [self.graph[u][v]['color'] for u, v in self.graph.edges()]
        nodes_colors = [d["color"] for u, d in self.graph.nodes(data=True)]
        
        # Get edge weights to adjust edge thickness
        edges = self.graph.edges(data=True)
        weights = []
        for _, _, data in edges:
            weights.append(data.get('weight', 1))
        #weights = [data['weight'] for _, _, data in edges]  # Extract weights

        # Normalize weights for better visual scaling (optional, depending on your range of weights)
        max_weight = max(weights) if weights else 1  # Avoid division by zero
        min_weight = min(weights) if weights else 0
        if max_weight == min_weight:
            normalized_weights = [1 for weight in weights]
        else:
            normalized_weights = [(1 + 4 * (weight - min_weight) / (max_weight - min_weight)) for weight in
                              weights]

        nx.draw(self.graph, pos, node_size=100, node_color=nodes_colors, with_labels=False, edge_color=edge_colors, width=normalized_weights)
        plt.show()
        return self
    
    def to_sequence(self, sequence_visitor):
        if not self.is_implemented:
            raise StrategyNotImplementedError(f"This function is not yet implemented for this type of graph.")
        self.sequence_visitor = sequence_visitor
        self.sequences = sequence_visitor.to_sequence(self)
        return self
    
    def draw_sequence(self):
        """Draws timeseries."""
        if not self.is_implemented:
            raise StrategyNotImplementedError(f"This function is not yet implemented for this type of graph.")
        colors = []
        for j in range(len(self.sequences)):
            colors.append("black")

        for j in range(len(self.sequences)):
            self.plot_timeseries(self.sequences[j], f"next_node_strategy = {self.sequence_visitor._get_node_strategy_name()}, next_value = {self.sequence_visitor._get_value_strategy_name()}", "Date", "Value", colors[j])
        plt.show()
        return self

    def plot_timeseries(self, sequence, title, x_legend, y_legend, color):
        """Function to sets parameters to draw timeseries."""
        plt.figure(figsize=(10, 6))
        plt.plot(sequence, linestyle='-', color=color)

        plt.title(title)
        plt.xlabel(x_legend)
        plt.ylabel(y_legend)
        plt.grid(True)
    
    def _get_histogram(self):
        return self.histogram_frequencies, self.histogram_bins
    
    def _get_w_tau(self):
        return self.w, self.tau

    def to_embedding(self, embedding_visitor):
        self.embeddings = embedding_visitor.get_graph_embedding()
        return self
    
    def get_embedding(self):
        return self.embeddings

    def _get_quantiles(self):
        return self.quantiles, self.quantile_values

class VisitorGraphEmbedding:
    def __init__(self):
        self.embedding = None
    
    def get_graph_embedding(self, graph):
        self.embedding = np.array(list(nx.eigenvector_centrality_numpy(graph)))
        return self.embedding

class ToSequenceVisitorMaster:
    """
    Turns graphs back to timeseries.
    
    **Attributes:**

    - `graph`: networkx.Graph object
    - `node_strategy`: strategy_to_time_sequence.StrategySelectNextNode object
    - `value_strategy`: strategy_to_time_sequence.StrategyNextValueInNode object

    """
    def __init__(self):
        self.ts_view = None
        self.graph = None
        self.node_strategy = None
        self.value_strategy = None
        self.skip_values = 0
        self.timeseries_len = 100
        self.sequences = None
        self.switch_graphs = 1
        self.nodes = None
        self.data_nodes = None
        self.att = 'value'

    def _set_nodes(self, dict: dict):
        pass

    def set_attribute(self, att):
        self.att = att

    """Next 4 function set the parameters, that are later on used as a strategy for converting graph to timeseries."""
    """--->"""
    def next_node_strategy(self, strategy: StrategySelectNextNode):
        self.node_strategy = strategy
        self.switch_graphs = strategy.get_change()
        return self

    def next_value_strategy(self, strategy: StrategyNextValueInNode):
        self.value_strategy = strategy
        self.skip_values = strategy.get_skip()
        return self

    def ts_length(self, x):
        self.timeseries_len = x
        return self
    """<---"""

    def _get_node_strategy_name(self):
        return self.node_strategy.get_name()
    
    def _get_value_strategy_name(self):
        return self.value_strategy.get_name()

    def to_sequence(self, graph):
        """Converts graph into time sequences."""
        pass

    def _is_equal(self, graph_1, graph_2):
        """Compares two graphs if they are equal."""
        if(graph_1.nodes != graph_2.nodes): return False
        if(graph_1.edges != graph_2.edges): return False
        for i in range(len(graph_1.nodes)):
            if list(list(graph_1.nodes(data=True))[i][1][self.att]) != list(list(graph_2.nodes(data=True))[i][1][self.att]):
                    return False
        return True


class ToSequenceVisitorSlidingWindow(ToSequenceVisitorMaster):
    """Converts graphs made using sliding window mechanism back to timeseries"""
    def __init__(self):
        super().__init__()

    def _set_nodes(self, dicts: dict):

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

    def to_sequence(self, graph):
        
        self.ts_view = graph
        self.graph = graph._get_graph()
        self._set_nodes(graph._get_graphs())


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


        ts_len = 0
        while len(self.sequences[0]) < self.timeseries_len:
            for j in range(len(self.sequences)):
                
                index = 0
                for i in range(len(list(self.nodes[j]))):
                
                    if(self._is_equal(current_nodes[j], list(self.graph.nodes)[i])):
                        index = i
                        break

                self.sequences[j] = self.value_strategy.append(self.sequences[j], current_nodes[j], j, index)
                if self.sequences[j][-1] == None:
                    return self

            for j in range(self.skip_values + 1):
                for k in range(len(current_nodes)):

                    current_nodes[k] = self.node_strategy.next_node(ts_len, k, current_nodes, self.switch_graphs, current_nodes[k])

                    if(current_nodes[k] == None):
                        return self

            ts_len += 1
        return self.sequences

class ToSequenceVisitor(ToSequenceVisitorMaster):
    """Converts ordinary graphs back to timeseries."""
    def __init__(self):
        super().__init__()

    def _set_nodes(self, dict: dict):

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

    def to_sequence(self, graph):

        self.ts_view = graph
        self.graph = graph._get_graph()
        self._set_nodes(graph._get_graphs())

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
        while len(self.sequences[0]) < self.timeseries_len:

            for j in range(len(current_nodes)):

                index = 0

                for b in range(len(list(self.nodes[j]))):
                    if(current_nodes_data[j] == self.data_nodes[j][b]):
                        index = b
                        break

                self.sequences[j] = self.value_strategy.append(self.sequences[j], current_nodes_data[j], j, index)
                if self.sequences[j][-1] == None:
                    return
                
            for j in range(self.skip_values+1):
                for k in range(len(current_nodes)):
                    current_nodes[k] = self.node_strategy.next_node(i, k, current_nodes, self.switch_graphs, current_nodes[k])
                    
                    new_index = self.nodes[k].index(current_nodes[k])
                    current_nodes_data[k] = self.data_nodes[k][new_index]
                    if(current_nodes[k] == None):
                        break

            i += 1
        return self.sequences



class ToSequenceVisitorOrdinalPartition(ToSequenceVisitorMaster):
    """Converts graphs made using ordinal partition mechanism back to timeseries"""
    def __init__(self):
        super().__init__()
        self.histogram_frequencies = None
        self.histogram_bins = None
        self.w = 1
        self.tau = 1

    def _set_nodes(self, dict: dict):

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
    
    def to_sequence(self, graph):


        self.graph = graph._get_graph()

        self.histogram_frequencies, self.histogram_bins = graph._get_histogram()
        self._set_nodes(graph._get_graphs())
        self.w, self.tau = graph._get_w_tau()
        one_ts_length = self.timeseries_len/self.tau
        short_series = [[[] for i in range(self.tau)] for i in range(len(graph._get_graphs()))]
        current_nodes = [None for _ in range(len(self.nodes))]
        current_nodes_data = [None for _ in range(len(self.data_nodes))]
        for i in range(len(self.nodes)):
            current_nodes[i] = self.nodes[i][0]
            current_nodes_data[i] = self.data_nodes[i][self.nodes[i].index(current_nodes[i])]

        self.node_strategy.set_arguments(self.graph, self.nodes, {}, self.att)

        i = 0
        while(len(short_series[0][0]) < one_ts_length):
            for k in range(len(graph._get_graphs())):
                for i in range(self.tau):
                    if(len(short_series[k][i]) == 0):
                        short_series[k][i] = self.value_strategy.append_start(short_series[k][i], k, current_nodes_data[k], self.histogram_frequencies, self.histogram_bins, self.w)
                    else:
                        short_series[k][i] = self.value_strategy.append(short_series[k][i], k, current_nodes_data[k], self.histogram_frequencies, self.histogram_bins, self.w)
                    if(i < self.tau-1):
                        current_nodes[k] = self.node_strategy.next_node(i, k, current_nodes, self.switch_graphs, current_nodes[k])
                        current_nodes_data[k] = self.data_nodes[k][self.nodes[k].index(current_nodes[k])]
            
            for j in range(self.skip_values+1):
                for k in range(len(graph._get_graphs())):
                    current_nodes[k] = self.node_strategy.next_node(i, k, current_nodes, self.switch_graphs, current_nodes[k])
                    current_nodes_data[k] = self.data_nodes[k][self.nodes[k].index(current_nodes[k])]
            
            i+=1
        
        
        self.sequences = [[] for i in range(len(graph._get_graphs()))]
        for k in range(len(graph._get_graphs())):
            for j in range(self.tau):
                for i in range(len(short_series[k][j])):
                    self.sequences[k].append(short_series[k][j][i])
        
        return self.sequences

class ToSequenceVisitorQuantile(ToSequenceVisitorMaster):
    def __init__(self):
        super().__init__()
        self.bins = None

    def _set_nodes(self, dict: dict):

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
    
    def to_sequence(self, graph):
        self.graph = graph._get_graph()
        self.bins, self.values = graph._get_quantiles()
        self._set_nodes(graph._get_graphs())

        self.sequences = [[] for _ in range(len(self.nodes))]

        current_nodes = [None for _ in range(len(self.nodes))]
        current_nodes_data = [None for _ in range(len(self.data_nodes))]

        self.node_strategy.set_arguments(self.graph, self.nodes, {}, self.att)

        for i in range(len(self.nodes)):
            current_nodes[i] = self.nodes[i][0]
            current_nodes_data[i] = self.data_nodes[i][0]

        
        i = 0
        while len(self.sequences[0]) < self.timeseries_len:

            for j in range(len(current_nodes)):

                self.sequences[j] = self.value_strategy.append(self.sequences[j], current_nodes_data[j], self.bins[j], self.values[j])
                
                
            for j in range(self.skip_values+1):
                for k in range(len(current_nodes)):
                    current_nodes[k] = self.node_strategy.next_node(i, k, current_nodes, self.switch_graphs, current_nodes[k])
                    
                    new_index = self.nodes[k].index(current_nodes[k])
                    current_nodes_data[k] = self.data_nodes[k][new_index]
                    if(current_nodes[k] == None):
                        break

            i += 1
        return self.sequences