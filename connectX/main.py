import sys
import os

cwd = "/kaggle_simulations/agent/"
if os.path.exists(cwd):
    # print(os.getcwd())
    # print(os.listdir())
    # print(os.listdir("/kaggle_simulations/agent/resources/"))
    # os.symlink(
    #     "/kaggle_simulations/agent/resources/", "./resoures", target_is_directory=True
    # )
    # print(os.listdir("resources"))
    # print(os.listdir("resources\models"))
    sys.path.append(cwd)
    sys.path.append(cwd + "mcts/")
    sys.path.append(cwd + "game/")
    sys.path.append(cwd + "game/board/")
    sys.path.append(cwd + "model/")
    sys.path.append(cwd + "agent/")
    sys.path.append(cwd + "logger/")
else:
    cwd = ""

from mcts.nn_mcts import NeuralNetworkMonteCarloTreeSearch
from agent.mcts_agent import MCTSAgent


def act(observation, configuration):
    return MCTSAgent.get_instance(
        configuration, NeuralNetworkMonteCarloTreeSearch.get_instance(configuration)
    ).act(observation)
