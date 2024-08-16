import os
import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

#import os
import csv
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from timeseries.strategies import TimeseriesToGraphStrategy, TimeseriesEdgeVisibilityConstraintsNatural, TimeseriesEdgeVisibilityConstraintsHorizontal, EdgeWeightingStrategyNull, TimeseriesEdgeVisibilityConstraintsVisibilityAngle
from generation.strategies import RandomWalkWithRestartSequenceGenerationStrategy, RandomWalkSequenceGenerationStrategy, RandomNodeSequenceGenerationStrategy, RandomNodeNeighbourSequenceGenerationStrategy, RandomDegreeNodeSequenceGenerationStrategy
from core import model 
from sklearn.model_selection import train_test_split
import itertools

import xml.etree.ElementTree as ET
import random
import hashlib

import input as inp
import builder.graph_strategy as gs
import graph
import ts_process_strategy as tps
import ts_to_graph as ttg
import link
import time_series_view as tsv
import singular as sg

amazon_path = os.path.join(os.getcwd(), "amazon", "AMZN.csv")
apple_path = os.path.join(os.getcwd(), "apple", "APPLE.csv")

"""
ttg.TimeSeriesToGraph()\
    .from_csv(inp.CsvStock(amazon_path, "Close"))\
    .process(tps.Segment(60, 90))\
    .to_graph(gs.NaturalVisibility().with_limit(1).get_strategy())\
    .add_edge(0,2)\
    .add_edge(13, 21, weight = 17)\
    .link(link.Link().by_value(link.SameValue(2)).seasonalities(15))\
    .draw("blue")


#--------------------------------------------------------------------------------

ttg.TimeSeriesToGraph()\
    .from_csv(inp.CsvStock(apple_path, "Close"))\
    .process(tps.Segment(60, 120))\
    .process(tps.SlidingWindow(5))\
    .to_graph(gs.NaturalVisibility().get_strategy())\
    .combine_identical_nodes()\
    .draw("red")

#--------------------------------------------------------------------------------


a = ttg.TimeSeriesToGraph().from_csv(inp.CsvStock(amazon_path, "Close"))\
    .process(tps.Segment(60, 80))\
    .to_graph(gs.NaturalVisibility().get_strategy())\
    .link(link.Link().by_value(link.SameValue(1)))

b = ttg.TimeSeriesToGraph().from_csv(inp.CsvStock(apple_path, "Close"))\
    .process(tps.Segment(120, 140))\
    .to_graph(gs.NaturalVisibility().with_limit(1).get_strategy())

c = ttg.TimeSeriesToGraph().from_csv(inp.CsvStock(amazon_path, "Close"))\
    .process(tps.Segment(180, 200))\
    .to_graph(gs.NaturalVisibility().with_angle(120).get_strategy())\
    .combine_identical_nodes()

d = ttg.TimeSeriesToGraph().from_csv(inp.CsvStock(apple_path, "Close"))\
    .process(tps.Segment(240, 260))\
    .to_graph(gs.NaturalVisibility().get_strategy())\
    .link(link.Link().seasonalities(15))

i = ttg.MultivariateTimeSeriesToGraph()\
    .add(a)\
    .add(b)\
    .add(c)\
    .add(d)\
    .link(link.Link().time_coocurence())\
    .link(link.Link().by_value(link.SameValue(0.5)))\
    .combine_identical_nodes()\
    .draw("purple")  

#-------------------------------------------------------------------------------

x = ttg.TimeSeriesToGraph().from_csv(inp.CsvStock(amazon_path, "Close"))\
    .process(tps.Segment(60, 90))\
    .process(tps.SlidingWindow(5))\
    .to_graph(gs.NaturalVisibility().get_strategy())\
    .combine_identical_nodes()


y = ttg.TimeSeriesToGraph().from_csv(inp.CsvStock(apple_path, "Close"))\
    .process(tps.Segment(120, 150))\
    .process(tps.SlidingWindow(5))\
    .to_graph(gs.NaturalVisibility().get_strategy())\
    .combine_identical_nodes()

z = ttg.TimeSeriesToGraph().from_csv(inp.CsvStock(amazon_path, "Close"))\
    .process(tps.Segment(180, 210))\
    .process(tps.SlidingWindow(5))\
    .to_graph(gs.NaturalVisibility().get_strategy())

w = ttg.TimeSeriesToGraph().from_csv(inp.CsvStock(apple_path, "Close"))\
    .process(tps.Segment(240, 270))\
    .process(tps.SlidingWindow(5))\
    .to_graph(gs.NaturalVisibility().get_strategy())\
    .combine_identical_nodes()

j = ttg.MultivariateTimeSeriesToGraph()\
    .add(x)\
    .add(y)\
    .add(z)\
    .add(w)\
    .link(link.LinkMulti().time_coocurence())\
    .combine_identical_nodes()\
    .draw("red")

#------------------------------------------------------------------------------------

graph.Graph(a.return_graph())\
    .set_nodes(i.return_graph().nodes, i.return_graph().nodes(data=True))\
    .walk_through_all()\
    .choose_next_node("weighted")\
    .choose_next_value("sequential")\
    .skip_every_x_steps(1)\
    .ts_length(100)\
    .to_time_sequence()\
    .draw()

#----------------------------------------------------------------------------------

graph.GraphSlidWin(x.return_graph())\
    .set_nodes(x.return_graph().nodes)\
    .choose_next_node("weighted")\
    .choose_next_value("random")\
    .skip_every_x_steps(1)\
    .ts_length(50)\
    .to_time_sequence()\
    .draw()

#---------------------------------------------------------------------------------

graph.Graph(i.return_graph())\
    .set_nodes(i.get_graph_nodes(), i.get_graph_nodes_data())\
    .walk_through_all()\
    .change_graphs_every_x_steps(2)\
    .choose_next_node("random")\
    .choose_next_value("sequential")\
    .skip_every_x_steps(1)\
    .ts_length(50)\
    .to_multiple_time_sequences()\
    .draw()

#-------------------------------------------------------------------------------

graph.GraphSlidWin(j.return_graph())\
    .set_nodes(j.get_graph_nodes())\
    .choose_next_node("weighted")\
    .choose_next_value("sequential")\
    .skip_every_x_steps(0)\
    .ts_length(100)\
    .to_multiple_time_sequences()\
    .draw()



l = ttg.TimeSeriesToGraph().from_csv(inp.CsvStock(amazon_path, "Close"))\
    .process(tps.Segment(60, 65))\
    .to_graph(gs.NaturalVisibility().get_strategy())


m = ttg.TimeSeriesToGraph().from_csv(inp.CsvStock(apple_path, "Close"))\
    .process(tps.Segment(120, 125))\
    .to_graph(gs.NaturalVisibility().get_strategy())

n = ttg.TimeSeriesToGraph().from_csv(inp.CsvStock(amazon_path, "Close"))\
    .process(tps.Segment(180, 185))\
    .to_graph(gs.NaturalVisibility().get_strategy())

o = ttg.TimeSeriesToGraph().from_csv(inp.CsvStock(apple_path, "Close"))\
    .process(tps.Segment(240, 245))\
    .to_graph(gs.NaturalVisibility().get_strategy())

spo = ttg.MultivariateTimeSeriesToGraph()\
    .add(l)\
    .add(m)\
    .add(n)\
    .link(link.Link().time_coocurence())\
    .draw(color = ["grey", "silver", "black"])
"""
"""Change the draw function back after taking the photo"""
"""
source = tsv.TimeSeriesSource()\
    .from_csv(inp.CsvStock(apple_path, "Close"))

view = tsv.TimeSeriesView(source)\
    .process(tps.Segment(60, 90))\
    .to_graph(gs.NaturalVisibility().with_limit(1).get_strategy())\
    .draw("blue")

source.from_csv(inp.CsvStock(amazon_path, "Close"))
"""

test = sg.TimeSeriesPreprocessing(inp.CsvStock(apple_path, "Close").from_csv())\
    .segmentation(60, 90).process()\
    .add(sg.TimeSeriesPreprocessing(inp.CsvStock(apple_path, "Close").from_csv())\
         .segmentation(90, 120).process())\
    .add(sg.TimeSeriesPreprocessing(inp.CsvStock(apple_path, "Close").from_csv())\
         .segmentation(150, 180).process())\
    .to_graph(gs.NaturalVisibility().with_limit(1).get_strategy())\
    .link(link.LinkGraphs().time_coocurence())\
    .link(link.LinkOne().seasonalities(15))\
    .draw("blue")

new_test = sg.TimeSeriesPreprocessing(inp.CsvStock(apple_path, "Close").from_csv())\
    .segmentation(60, 150).sliding_window(5).process()\
    .to_graph(gs.NaturalVisibility().get_strategy())\
    .link(link.LinkGraphs().sliding_window())\
    .combine_identical_nodes_slid_win()\
    .draw("red")

mixed_test = graph.Graph(test.get_graph())\
    .set_nodes(test.get_graphs())\
    .walk_through_all()\
    .choose_next_node("weighted")\
    .choose_next_value("sequential")\
    .skip_every_x_steps(1)\
    .ts_length(100)\
    .to_multiple_time_sequences()\
    .draw()


graph.GraphSlidWin(new_test.get_graph())\
    .set_nodes(new_test.get_graph().nodes)\
    .choose_next_node("weighted")\
    .choose_next_value("random")\
    .skip_every_x_steps(1)\
    .ts_length(50)\
    .to_time_sequence()\
    .draw()

test = sg.TimeSeriesPreprocessing(inp.CsvStock(apple_path, "Close").from_csv())\
    .segmentation(60, 90).process()\
    .to_graph(gs.NaturalVisibility().with_limit(1).get_strategy())\
    .link(link.LinkOne().seasonalities(15))\
    .draw("blue")



