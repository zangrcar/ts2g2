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

import input as inp
import builder.graph_strategy as gs
import graph
import link
import singular as sg

amazon_path = os.path.join(os.getcwd(), "amazon", "AMZN.csv")
apple_path = os.path.join(os.getcwd(), "apple", "APPLE.csv")

"""
test = sg.TimeSeriesPreprocessing(inp.CsvStock(amazon_path, "Close").from_csv())\
    .add_strategy(sg.Segmentation(60, 90))\
    .process()\
    .to_graph(gs.NaturalVisibility().with_limit(1).get_strategy())\
    .add_edge(0,2)\
    .add_edge(13, 21, weight = 17)\
    .link(link.LinkNodesWithinGraph().by_value(link.SameValue(2)).seasonalities(15))\
    .draw("blue")
"""

i = sg.TimeSeriesPreprocessing(inp.CsvStock(apple_path, "Close").from_csv())\
    .add_strategy(sg.Segmentation(60, 90))\
    .process()\
    .add(sg.TimeSeriesPreprocessing(inp.CsvStock(apple_path, "Close").from_csv())\
        .add_strategy(sg.Segmentation(90, 120))\
        .process())\
    .add(sg.TimeSeriesPreprocessing(inp.CsvStock(apple_path, "Close").from_csv())\
        .add_strategy(sg.Segmentation(150, 180))\
        .process())\
    .to_graph(gs.NaturalVisibility().with_limit(1).get_strategy())\
    .link(link.LinkGraphs().time_coocurence())\
    .link(link.LinkNodesWithinGraph().by_value(link.SameValue(0.5)))\
    .combine_identical_nodes()\
    .draw("blue")

sg.TimeSeriesPreprocessing(inp.CsvStock(apple_path, "Close").from_csv())\
    .add_strategy(sg.Segmentation(60, 90))\
    .process()\
    .add(sg.TimeSeriesPreprocessing(inp.CsvStock(apple_path, "Close").from_csv())\
        .add_strategy(sg.Segmentation(90, 120))\
        .process())\
    .add(sg.TimeSeriesPreprocessing(inp.CsvStock(apple_path, "Close").from_csv())\
        .add_strategy(sg.Segmentation(150, 180))\
        .process())\
    .to_graph(gs.NaturalVisibility().with_limit(1).get_strategy())\
    .link(link.LinkGraphs().time_coocurence())\
    .link(link.LinkNodesWithinGraph().by_value(link.SameValue(0.5)))\
    .draw("blue")

"""
sg.TimeSeriesPreprocessing(inp.CsvStock(apple_path, "Close").from_csv())\
    .add_strategy(sg.Segmentation(60, 90))\
    .add_strategy(sg.SlidingWindow(5))\
    .process()\
    .to_graph(gs.NaturalVisibility().get_strategy())\
    .link(link.LinkGraphs().sliding_window())\
    .link(link.LinkNodesWithinGraph().seasonalities(15))\
    .combine_identical_nodes_slid_win()\
    .draw("green")


sg.TimeSeriesPreprocessing(inp.CsvStock(apple_path, "Close").from_csv())\
    .add_strategy(sg.Segmentation(60, 90))\
    .add_strategy(sg.SlidingWindow(5))\
    .process()\
    .add(sg.TimeSeriesPreprocessing(inp.CsvStock(apple_path, "Close").from_csv())\
         .add_strategy(sg.Segmentation(90, 120))\
            .add_strategy(sg.SlidingWindow(5))\
                .process())\
    .add(sg.TimeSeriesPreprocessing(inp.CsvStock(apple_path, "Close").from_csv())\
         .add_strategy(sg.Segmentation(150, 180))\
            .add_strategy(sg.SlidingWindow(5))\
                .process())\
    .to_graph(gs.NaturalVisibility().get_strategy())\
    .link(link.LinkGraphs().sliding_window().time_coocurence())\
    .link(link.LinkNodesWithinGraph().seasonalities(15))\
    .draw("red")


graph.GraphToTS(test.get_graph())\
    .set_nodes(test.get_graphs())\
    .next_node_strategy(graph.NextNodeAllWeighted())\
    .next_value_strategy(graph.NextValueSequential().skip_every_x_steps(1))\
    .ts_length(100)\
    .to_multiple_time_sequences()\
    .draw()


graph.GraphToTS(i.get_graph())\
    .set_nodes(i.get_graphs())\
    .next_node_strategy(graph.NextNodeAllRandom().change_graphs_every_x_steps(2))\
    .next_value_strategy(graph.NextValueSequential().skip_every_x_steps(1))\
    .ts_length(50)\
    .to_multiple_time_sequences()\
    .draw()


x = sg.TimeSeriesPreprocessing(inp.CsvStock(apple_path, "Close").from_csv())\
    .add_strategy(sg.Segmentation(60, 120))\
    .add_strategy(sg.SlidingWindow(5))\
    .process()\
    .to_graph(gs.NaturalVisibility().get_strategy())\
    .link(link.LinkGraphs().sliding_window())\
    .combine_identical_nodes_slid_win()\
    .draw("red")


graph.GraphSlidWin(x.get_graph())\
    .set_nodes(x.get_graphs())\
    .next_node_strategy(graph.NextNodeOneWeighted())\
    .next_value_strategy(graph.NextValueRandomSlidWin().skip_every_x_steps(1))\
    .ts_length(50)\
    .to_time_sequence()\
    .draw()


j = sg.TimeSeriesPreprocessing(inp.CsvStock(apple_path, "Close").from_csv())\
    .add_strategy(sg.Segmentation(60, 110))\
    .add_strategy(sg.SlidingWindow(5))\
    .process()\
    .add(sg.TimeSeriesPreprocessing(inp.CsvStock(apple_path, "Close").from_csv())\
         .add_strategy(sg.Segmentation(90, 140))\
            .add_strategy(sg.SlidingWindow(5))\
                .process())\
    .add(sg.TimeSeriesPreprocessing(inp.CsvStock(apple_path, "Close").from_csv())\
         .add_strategy(sg.Segmentation(150, 200))\
            .add_strategy(sg.SlidingWindow(5))\
                .process())\
    .to_graph(gs.NaturalVisibility().get_strategy())\
    .link(link.LinkGraphs().sliding_window().time_coocurence())\
    .combine_identical_nodes_slid_win()\
    .link(link.LinkNodesWithinGraph().seasonalities(15))\
    .draw("blue")


graph.GraphSlidWin(j.get_graph())\
    .set_nodes(j.get_graphs())\
    .next_node_strategy(graph.NextNodeOneRandom())\
    .next_value_strategy(graph.NextValueSequentialSlidWin())\
    .ts_length(100)\
    .to_multiple_time_sequences()\
    .draw()
"""