import os
import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

from core import model

from input_output import input as inp
import from_graph.strategy_to_time_sequence as tts
import to_graph.strategy_linking_graph as gls
import to_graph.strategy_linking_multi_graphs as mgl
import to_graph.strategy_to_graph as tgs

amazon_path = os.path.join(os.getcwd(), "amazon", "AMZN.csv")
apple_path = os.path.join(os.getcwd(), "apple", "APPLE.csv")


test = model.Timeseries(inp.CsvStock(amazon_path, "Close").from_csv())\
    .with_preprocessing(model.TimeseriesPreprocessingSegmentation(60, 90))\
    .to_graph(tgs.BuildTimeseriesToGraphNaturalVisibilityStrategy().with_limit(1).get_strategy())\
    .add_edge(0,2)\
    .add_edge(13, 21, weight = 17)\
    .link(gls.LinkNodesWithinGraph().by_value(gls.StrategyLinkingGraphByValueWithinRange(2)).seasonalities(15))\
    .draw("blue")


x = model.Timeseries(inp.CsvStock(apple_path, "Close").from_csv())\
    .with_preprocessing(model.TimeseriesPreprocessingComposite()\
        .add_strategy(model.TimeseriesPreprocessingSegmentation(60, 120))\
        .add_strategy(model.TimeseriesPreprocessingSlidingWindow(5)))\
    .to_graph(tgs.BuildTimeseriesToGraphNaturalVisibilityStrategy().get_strategy())\
    .link(mgl.LinkGraphs().sliding_window())\
    .combine_identical_nodes_slid_win()\
    .draw("red")

i = model.Timeseries(inp.CsvStock(apple_path, "Close").from_csv())\
    .with_preprocessing(model.TimeseriesPreprocessingSegmentation(60, 90))\
    .add(model.Timeseries(inp.CsvStock(apple_path, "Close").from_csv())\
        .with_preprocessing(model.TimeseriesPreprocessingSegmentation(90, 120)))\
    .add(model.Timeseries(inp.CsvStock(apple_path, "Close").from_csv())\
        .with_preprocessing(model.TimeseriesPreprocessingSegmentation(150, 180)))\
    .to_graph(tgs.BuildTimeseriesToGraphNaturalVisibilityStrategy().with_limit(1).get_strategy())\
    .link(mgl.LinkGraphs().time_cooccurrence())\
    .link(gls.LinkNodesWithinGraph().by_value(gls.StrategyLinkingGraphByValueWithinRange(0.5)))\
    .combine_identical_nodes()\
    .draw("brown")

j = model.Timeseries(inp.CsvStock(apple_path, "Close").from_csv())\
    .with_preprocessing(model.TimeseriesPreprocessingComposite()\
        .add_strategy(model.TimeseriesPreprocessingSegmentation(60, 110))\
        .add_strategy(model.TimeseriesPreprocessingSlidingWindow(5)))\
    .add(model.Timeseries(inp.CsvStock(apple_path, "Close").from_csv())\
        .with_preprocessing(model.TimeseriesPreprocessingComposite()\
            .add_strategy(model.TimeseriesPreprocessingSegmentation(120, 170))\
            .add_strategy(model.TimeseriesPreprocessingSlidingWindow(5)))\
        .add(model.Timeseries(inp.CsvStock(apple_path, "Close").from_csv())\
            .with_preprocessing(model.TimeseriesPreprocessingComposite()\
                    .add_strategy(model.TimeseriesPreprocessingSegmentation(190, 240))\
                    .add_strategy(model.TimeseriesPreprocessingSlidingWindow(5)))))\
    .to_graph(tgs.BuildTimeseriesToGraphNaturalVisibilityStrategy().get_strategy())\
    .link(mgl.LinkGraphs().sliding_window().time_cooccurrence())\
    .combine_identical_nodes_slid_win()\
    .link(gls.LinkNodesWithinGraph().seasonalities(15))\
    .draw("green")


test.to_sequence(model.ToSequenceVisitor()\
        .next_node_strategy(tts.StrategySelectNextNodeRandomlyAcrossGraphs())\
        .next_value_strategy(tts.StrategyNextValueInNodeRoundRobin().skip_every_x_steps(1))\
        .ts_length(100))\
    .draw_sequence()


x.to_sequence(model.ToSequenceVisitorSlidingWindow()\
    .next_node_strategy(tts.StrategySelectNextNodeRandomlyFromFirstGraph())\
    .next_value_strategy(tts.StrategyNextValueInNodeRandomForSlidingWindow().skip_every_x_steps(1))\
    .ts_length(50))\
    .draw_sequence()


i.to_sequence(model.ToSequenceVisitor()\
    .next_node_strategy(tts.StrategySelectNextNodeRandomlyAcrossGraphs().change_graphs_every_x_steps(2))\
    .next_value_strategy(tts.StrategyNextValueInNodeRoundRobin().skip_every_x_steps(1))\
    .ts_length(50))\
    .draw_sequence()


j.to_sequence(model.ToSequenceVisitorSlidingWindow()\
    .next_node_strategy(tts.StrategySelectNextNodeRandomlyFromFirstGraph())\
    .next_value_strategy(tts.StrategyNextValueInNodeRoundRobinForSlidingWindow())\
    .ts_length(100))\
    .draw_sequence()
