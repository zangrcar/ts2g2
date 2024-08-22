import os
import sys

import from_graph.to_time_sequence_strategy
import to_graph.graph_linking_strategy
import to_graph.multi_graph_linking_strategy
import to_graph.to_graph_strategy
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

import IO.input as inp

amazon_path = os.path.join(os.getcwd(), "amazon", "AMZN.csv")
apple_path = os.path.join(os.getcwd(), "apple", "APPLE.csv")


test = model.TimeSeriesPreprocessing(inp.CsvStock(amazon_path, "Close").from_csv())\
    .add_strategy(model.Segmentation(60, 90))\
    .process()\
    .to_graph(to_graph.to_graph_strategy.NaturalVisibility().with_limit(1).get_strategy())\
    .add_edge(0,2)\
    .add_edge(13, 21, weight = 17)\
    .link(to_graph.graph_linking_strategy.LinkNodesWithinGraph().by_value(to_graph.graph_linking_strategy.SameValue(2)).seasonalities(15))\
    .draw("blue")


i = model.TimeSeriesPreprocessing(inp.CsvStock(apple_path, "Close").from_csv())\
    .add_strategy(model.Segmentation(60, 90))\
    .process()\
    .add(model.TimeSeriesPreprocessing(inp.CsvStock(apple_path, "Close").from_csv())\
        .add_strategy(model.Segmentation(90, 120))\
        .process())\
    .add(model.TimeSeriesPreprocessing(inp.CsvStock(apple_path, "Close").from_csv())\
        .add_strategy(model.Segmentation(150, 180))\
        .process())\
    .to_graph(to_graph.to_graph_strategy.NaturalVisibility().with_limit(1).get_strategy())\
    .link(to_graph.multi_graph_linking_strategy.LinkGraphs().time_cooccurrence())\
    .link(to_graph.graph_linking_strategy.LinkNodesWithinGraph().by_value(to_graph.graph_linking_strategy.SameValue(0.5)))\
    .combine_identical_nodes()\
    .draw("blue")


model.TimeSeriesPreprocessing(inp.CsvStock(apple_path, "Close").from_csv())\
    .add_strategy(model.Segmentation(60, 90))\
    .add_strategy(model.SlidingWindow(5))\
    .process()\
    .to_graph(to_graph.to_graph_strategy.NaturalVisibility().get_strategy())\
    .link(to_graph.multi_graph_linking_strategy.LinkGraphs().sliding_window())\
    .link(to_graph.graph_linking_strategy.LinkNodesWithinGraph().seasonalities(15))\
    .combine_identical_nodes_slid_win()\
    .draw("green")


model.TimeSeriesPreprocessing(inp.CsvStock(apple_path, "Close").from_csv())\
    .add_strategy(model.Segmentation(60, 90))\
    .add_strategy(model.SlidingWindow(5))\
    .process()\
    .add(model.TimeSeriesPreprocessing(inp.CsvStock(apple_path, "Close").from_csv())\
         .add_strategy(model.Segmentation(90, 120))\
            .add_strategy(model.SlidingWindow(5))\
                .process())\
    .add(model.TimeSeriesPreprocessing(inp.CsvStock(apple_path, "Close").from_csv())\
         .add_strategy(model.Segmentation(150, 180))\
            .add_strategy(model.SlidingWindow(5))\
                .process())\
    .to_graph(to_graph.to_graph_strategy.NaturalVisibility().get_strategy())\
    .link(to_graph.multi_graph_linking_strategy.LinkGraphs().sliding_window().time_cooccurrence())\
    .link(to_graph.graph_linking_strategy.LinkNodesWithinGraph().seasonalities(15))\
    .draw("red")


model.GraphToTS(test)\
    .next_node_strategy(from_graph.to_time_sequence_strategy.NextNodeAllRandom())\
    .next_value_strategy(from_graph.to_time_sequence_strategy.NextValueSequential().skip_every_x_steps(1))\
    .ts_length(100)\
    .to_multiple_time_sequences()\
    .draw()


model.GraphToTS(i)\
    .next_node_strategy(from_graph.to_time_sequence_strategy.NextNodeAllRandom().change_graphs_every_x_steps(2))\
    .next_value_strategy(from_graph.to_time_sequence_strategy.NextValueSequential().skip_every_x_steps(1))\
    .ts_length(50)\
    .to_multiple_time_sequences()\
    .draw()


x = model.TimeSeriesPreprocessing(inp.CsvStock(apple_path, "Close").from_csv())\
    .add_strategy(model.Segmentation(60, 120))\
    .add_strategy(model.SlidingWindow(5))\
    .process()\
    .to_graph(to_graph.to_graph_strategy.NaturalVisibility().get_strategy())\
    .link(to_graph.multi_graph_linking_strategy.LinkGraphs().sliding_window())\
    .combine_identical_nodes_slid_win()\
    .draw("red")


model.GraphSlidWin(x)\
    .next_node_strategy(from_graph.to_time_sequence_strategy.NextNodeOneRandom())\
    .next_value_strategy(from_graph.to_time_sequence_strategy.NextValueRandomSlidWin().skip_every_x_steps(1))\
    .ts_length(50)\
    .to_time_sequence()\
    .draw()


j = model.TimeSeriesPreprocessing(inp.CsvStock(apple_path, "Close").from_csv())\
    .add_strategy(model.Segmentation(60, 110))\
    .add_strategy(model.SlidingWindow(5))\
    .process()\
    .add(model.TimeSeriesPreprocessing(inp.CsvStock(apple_path, "Close").from_csv())\
         .add_strategy(model.Segmentation(90, 140))\
            .add_strategy(model.SlidingWindow(5))\
                .process()\
                .add(model.TimeSeriesPreprocessing(inp.CsvStock(apple_path, "Close").from_csv())\
                    .add_strategy(model.Segmentation(150, 200))\
                        .add_strategy(model.SlidingWindow(5))\
                            .process()))\
    .to_graph(to_graph.to_graph_strategy.NaturalVisibility().get_strategy())\
    .link(to_graph.multi_graph_linking_strategy.LinkGraphs().sliding_window().time_cooccurrence())\
    .combine_identical_nodes_slid_win()\
    .link(to_graph.graph_linking_strategy.LinkNodesWithinGraph().seasonalities(15))\
    .draw("blue")


model.GraphSlidWin(j)\
    .next_node_strategy(from_graph.to_time_sequence_strategy.NextNodeOneRandom())\
    .next_value_strategy(from_graph.to_time_sequence_strategy.NextValueSequentialSlidWin())\
    .ts_length(100)\
    .to_multiple_time_sequences()\
    .draw()