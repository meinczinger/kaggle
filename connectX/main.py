import sys
import os

cwd = "/kaggle_simulations/agent/"
print("Before if")
if os.path.exists(cwd):
    print("Installing...")
    sys.path.append(cwd)
    sys.path.append(cwd + "agents/")
    sys.path.append(cwd + "agents/mcts/")
    sys.path.append(cwd + "agents/game/")
    sys.path.append(cwd + "agents/model/")
    sys.path.append(cwd + "agents/agent/")
    os.system("pip install game/")
    os.system("pip install model/")
    os.system("pip install mcts/")
    os.system("pip install agent/")
    os.system("pip freeze")
else:
    cwd = ""

from mcts.nn_mcts import NeuralNetworkMonteCarloTreeSearch
from agent.mcts_agent import MCTSAgent


def act(observation, configuration):
    return MCTSAgent.get_instance(
        configuration, NeuralNetworkMonteCarloTreeSearch(configuration)
    ).act(observation)
