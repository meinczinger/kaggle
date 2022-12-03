import sys
import os

cwd = '/kaggle_simulations/agent/'
if os.path.exists(cwd):
  sys.path.append(cwd)
  sys.path.append(cwd+'agents/')
  sys.path.append(cwd+'agents/mcts/')
else:
  cwd = ''

from my_agents.mcts.nn_mcts import NeuralNetworkMonteCarloTreeSearch
from my_agents.mcts_agent import MCTSAgent



def act(observation, configuration):
    return MCTSAgent.get_instance(
        configuration, NeuralNetworkMonteCarloTreeSearch(configuration)
    ).act(observation)
