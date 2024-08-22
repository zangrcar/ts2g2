import pandas as pd
import xml.etree.ElementTree as ET
import core.model as sg

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
        ts = sg.TimeSeries(time_series)
        return ts

class XmlRead:
    """Superclass of all classes for extraxtion of data from xml files."""
    def __init__(self):
        pass

    def from_xml(self):
        pass

class XmlSomething(XmlRead):
    """One of the ways of extraction of data from xml file."""
    def __init__(self, path, item, season = "Annual"):
        self.path = path
        self.item = item
        self.season = season
    
    def from_xml(self):
        tree = ET.parse(self.path)
        root = tree.getroot()

        financial_statements = root.find('FinancialStatements')
        COAMap = financial_statements.find('COAMap')
        
        periods = None
        if self.season.lower() == "annual":
            periods = financial_statements.find("AnnualPeriods")
        else:
            periods = financial_statements.find('InterimPeriods')

        elements = periods.findall(f".//lineItem[@coaCode = '{self.item}']")
        column = []

        for element in elements:
            column.append(float(element.text))
        
        ts = sg.TimeSeries(column)
        return ts
