import pandas as pd

class CsvRead:
    """Superclass of all classes for extraxtion of data from csv files.""" 
    def __init__(self):
        pass

    def from_csv(self):
        pass

class CsvStock(CsvRead):
    """Returns proccessed data from csv file with data sorted by date."""
    def __init__(self, path, y_column):
        self.path = path
        self.y_column = y_column
    
    def from_csv(self):
        time_series = pd.read_csv(self.path)
        time_series["Date"] = pd.to_datetime(time_series["Date"])
        time_series.set_index("Date", inplace=True)
        time_series = time_series[self.y_column]
        return time_series
