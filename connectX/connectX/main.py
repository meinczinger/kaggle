import sys
import os


sys.path.append("/kaggle_simulations/agent")
sys.path.append("/kaggle_simulations/agent/agents")
sys.path.append("/kaggle_simulations/agent/agents/model")
# sys.path.append("/opt/conda/lib/python3.7/site-packages/kaggle_environments")
if os.environ.get('GFOOTBALL_DATA_DIR', ''):
    os.chdir('/kaggle_simulations/agent/')

from agents.monte_carlo_agent import TabularMonteCarloAgent
from agents.sarsa_agent import TabularSarsaAgent
from agents.minimax_agent import MinimaxAgent
from agents.q_learning import TabularQLeaerningAgent
from mcts_agent import MCTSAgent
from baseline import BaselineAgent
from nn_agent import NeuralNetworkAgent


def act_tabular_monte_carlo(observation, configuration):
    return TabularMonteCarloAgent.get_instance(configuration).act(observation)


def act_minimax(observation, configuration):
    return MinimaxAgent.get_instance(configuration).act(observation)


def act_tabular_sarsa(observation, configuration):
    return TabularSarsaAgent.get_instance(configuration).act(observation)


def act_tabular_q_learning(observation, configuration):
    return TabularQLeaerningAgent.get_instance(configuration).act(observation)


def act_baseline(observation, configuration):
    return BaselineAgent.get_instance(configuration).act(observation)


def act_mcts(observation, configuration):
    return MCTSAgent.get_instance(configuration).act(observation)


def act_nn(observation, configuration):
    return NeuralNetworkAgent.get_instance(configuration).act(observation)
