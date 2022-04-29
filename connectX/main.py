import sys
import os


sys.path.append("/kaggle_simulations/agent")
sys.path.append("/kaggle_simulations/agent/agents")
sys.path.append("/kaggle_simulations/agent/agents/model")
# sys.path.append("/opt/conda/lib/python3.7/site-packages/kaggle_environments")
if os.environ.get("GFOOTBALL_DATA_DIR", ""):
    os.chdir("/kaggle_simulations/agent/")

from agents.mcts_agent import MCTSAgent

# from agents.nn_agent import NeuralNetworkAgent
from agents.mcts.classic_mcts import ClassicMonteCarloTreeSearch
from agents.mcts.tabular_mcts import TabularMonteCarloTreeSearch
from agents.rl_tabular.monte_carlo import TabularMonteCarlo


# def act_nn(observation, configuration):
#     return NeuralNetworkAgent.get_instance(configuration).act(observation)


def act_mcts(observation, configuration):
    return MCTSAgent.get_instance(
        configuration, ClassicMonteCarloTreeSearch(configuration)
    ).act(observation)

def act_mcts(observation, configuration):
    return MCTSAgent.get_instance(
        configuration, TabularMonteCarloTreeSearch(configuration, TabularMonteCarlo(configuration))
    ).act(observation)
