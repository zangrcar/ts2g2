import pandas as pd
import xml.etree.ElementTree as ET
import core.model as sg

class CsvRead:
    """Extracts data from a csv file.""" 
    def __init__(self):
        pass

    def from_csv(self):
        pass

class CsvStock(CsvRead):
    """Returns proccessed data from csv file sorted by "Date"."""
    def __init__(self, path, y_column):
        self.path = path
        self.y_column = y_column
    
    def from_csv(self):
        timeseries = pd.read_csv(self.path)
        timeseries["Date"] = pd.to_datetime(timeseries["Date"])
        timeseries.set_index("Date", inplace=True)
        timeseries = timeseries[self.y_column]
        ts = sg.Timeseries(timeseries)
        return ts

class XmlRead:
    """Extracts data from an xml file."""
    def __init__(self):
        pass

    def from_xml(self):
        pass

#TODO: rename
class XmlSomething(XmlRead):
    """One of the ways of extracting the data from xml file."""
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
        
        ts = sg.Timeseries(column)
        return ts
