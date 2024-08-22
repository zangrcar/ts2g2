class GraphLinkingstrategy:
    """Mother class of linking strategies for linking nodes within same graph."""
    def __init__(self, graph, num):
        self.graph = graph
        self.num = num

    def set_graph(self, graph):
        self.graph = graph

    def get_num(self):
        return self.num

    def apply(self):
        pass


class LinkSeasonalities(GraphLinkingstrategy):
    """Links all nodes that are sequentially self.period apart."""
    def __init__(self, period):
        super().__init__(None, 0)
        self.period = period

    def apply(self):
        for i in range(len(self.graph.nodes) - self.period):
            self.graph.add_edge(list(self.graph.nodes)[i], list(self.graph.nodes)[i+self.period], intergraph_binding='seasonality')
        return self.graph


class ByValue(GraphLinkingstrategy):
    """Links nodes based n their value."""
    def __init__(self, graph):
        super().__init__(graph, 1)
        self.attribute = 'value'

    def apply(self):
        pass


class SameValue(ByValue):
    "Class that notes that we want to connect nodes based on similarity of values."
    def __init__(self, allowed_difference):
        super().__init__(None)
        self.allowed_difference = allowed_difference

    def apply(self):
        for node_11, node_12 in zip(self.graph.nodes(data=True), self.graph.nodes):
            for node_21, node_22 in zip(self.graph.nodes(data=True), self.graph.nodes):
                if  abs(node_11[1][self.attribute][0] - node_21[1][self.attribute][0]) < self.allowed_difference and node_12 != node_22:
                    self.graph.add_edge(node_12, node_22, intergraph_binding = 'timesteps')

        return self.graph


class LinkNodesWithinGraph:
    """Control class for linking nodes within one graph, through which we can access linking strategies."""
    def __init__(self):
        self.graph = None
        self.attribute = 'value'
        self.command_array = []

    def link(self, graph):
        self.graph = graph.get_graph()

        self.command_array.sort(key=self.succession)

        for strat in self.command_array:
            strat.set_graph(self.graph)
            self.graph = strat.apply()

        return self.graph

    def succession(self, strategy):
        return strategy.get_num()

    def seasonalities(self, period):
        """Notes that we want to connect based on seasonalities, ad sets the period parameter."""
        self.command_array.append(LinkSeasonalities(period))
        return self

    def by_value(self, strategy):
        """Notes that we want to connect nodes based on values and strategy."""
        self.command_array.append(strategy)
        return self