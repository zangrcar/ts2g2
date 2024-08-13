import ts_to_graph


class TSprocess:
    """Superclass of classes Segment and SlidingWindow"""
    def __init__(self):
        pass

    def process(time_series):
        pass

class Segment(TSprocess):

    def __init__(self, segment_start, segment_end):
        self.segment_start = segment_start
        self.segment_end = segment_end
    
    def process(self, time_series):
        """returns a TimeSeriesToGraph object, that has a wanted segment of original time serie"""
        g = ts_to_graph.TimeSeriesToGraph(time_series[self.segment_start:self.segment_end])
        g.set_base(time_series)
        return g

class SlidingWindow(TSprocess):

    def __init__(self, window_size, win_move_len = 1):
        self.window_size = window_size
        self.win_move_len = win_move_len
    
    def process(self, time_series):
        """Returns a TimeSeriesToGraph object, that has an array of segments of window_size size and each win_move_len data apart.
        This function of this class is called, when we want to create a graph using sliding window mehanism."""
        segments = []
        for i in range(0, len(time_series) - self.window_size, self.win_move_len):
            segments.append(time_series[i:i + self.window_size])
        
        new_series = []
        for i in range(len(segments)):
            new_series.append(ts_to_graph.TimeSeriesToGraph(segments[i]))
        
        g = ts_to_graph.TimeSeriesToGraphSlidWin(new_series, segments = segments)
        g.set_base(time_series)
        return g

