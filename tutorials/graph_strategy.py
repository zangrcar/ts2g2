from timeseries.strategies import TimeseriesToGraphStrategy, TimeseriesEdgeVisibilityConstraintsNatural, TimeseriesEdgeVisibilityConstraintsHorizontal, EdgeWeightingStrategyNull, TimeseriesEdgeVisibilityConstraintsVisibilityAngle

class Strategy:
    """Superclass of classes NaturalVisibility and HorizontalVisibility. Sets and returns a strategy with which we can
    convert time series into graphs."""

    def __init__(self):
        self.visibility = []
        self.graph_type = "undirected"
        self.edge_weighting_strategy = EdgeWeightingStrategyNull()
        self.str_name = None

    def with_angle(self, angle):
        """Sets an angle in which range must a node be to be considered for connection."""
        self.visibility.append(TimeseriesEdgeVisibilityConstraintsVisibilityAngle(angle))
        self.str_name += (f" with angle({angle})")
        return self

    def with_limit(self, limit):
        """Sets a limit as to how many data instances two nodes must be apart to be considered for connection."""
        pass
    
    def strategy_name(self):
        """Returns name of used strategy."""
        return self.str_name

    def get_strategy(self):
        """Returns strategy."""
        return TimeseriesToGraphStrategy(
            visibility_constraints = self.visibility,
            graph_type= self.graph_type,
            edge_weighting_strategy=self.edge_weighting_strategy
        )

class NaturalVisibility(Strategy):
    """As initial strategy sets Natural visibility strategy."""
    def __init__(self):
        super().__init__()
        self.visibility = [TimeseriesEdgeVisibilityConstraintsNatural()]
        self.str_name = "Natural visibility strategy"
    
    def with_limit(self, limit):
        self.visibility[0] = TimeseriesEdgeVisibilityConstraintsNatural(limit)
        self.str_name += (f" with limit({limit})")
        return self
    
class HorizontalVisibility(Strategy):
    """As initial strategy sets Horizontal visibility strategy."""
    def __init__(self):
        super().__init__()
        self.visibility = [TimeseriesEdgeVisibilityConstraintsHorizontal()]
        self.str_name = "Horizontal visibility strategy"
    
    def with_limit(self, limit):
        self.visibility[0] = TimeseriesEdgeVisibilityConstraintsHorizontal(limit)
        self.str_name += (f" with limit({limit})")
        return self
