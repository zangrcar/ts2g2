import xml.etree.ElementTree as ET

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
        
        return column
