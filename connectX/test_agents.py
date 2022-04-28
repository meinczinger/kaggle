from calendar import c
import random
from kaggle_environments.utils import Struct
import cProfile
import os
import numpy as np
from agents.simulator import Simulator
from agents.bitboard import BitBoard
from agents.rl_agent import RLAgent
from agents.rl_tabular.monte_carlo import TabularMonteCarlo
from agents.mcts.classic_mcts import ClassicMonteCarloTreeSearch
from agents.mcts.nn_mcts import NeuralNetworkMonteCarloTreeSearch
from agents.mcts_agent import MCTSAgent
from agents.mcts.tabular_mcts import TabularMonteCarloTreeSearch
from agents.nn_agent import NeuralNetworkAgent
from agents.baseline import BaselineAgent
import threading
import concurrent.futures


config = Struct()
config.columns = 7
config.rows = 6
config.inarow = 4
config.episodeSteps = 0
config.actTimeout = 0.5
config.timeout = 2.0

obs = Struct()
obs.step = 0
obs.board = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]
obs.mark = 1

# agent = NeuralNetworkAgent(config)
# action = agent.act(obs)

# cProfile.run('agent.act(obs)')

# minimax_agent = MinimaxAgent(config, 3)
# sizes = []
# own_player = 1
# gtc = GameTreeControl(config, True)
# prev_size = 0
# base_size = 0
# for i in range(30):
#     gtc.initialize(own_player)
#     gtc.build_tree(10000, 13)
#     gtc._persist()
#     own_player = (own_player % 2) + 1
#     file_size = os.path.getsize("mcts_tree.pickle")
#     if i == 0:
#         base_size = file_size
#     print(i, 'th iteration, file size compared to base ', (file_size - prev_size) / base_size)
#     prev_size = file_size
#     sizes.append(file_size)


# cProfile.run('gtc.build_tree(1000)')
# cProfile.run('agent.act(obs)')


# simClassicVsMonteCarlo = Simulator(
#     config,
#     MCTSAgent(config, ClassicMonteCarloTreeSearch(config)),
#     MCTSAgent(config, TabularMonteCarloTreeSearch(config, TabularMonteCarlo(config))),
# )

agent1 = "Neural network"
agent2 = "MCTS with tabular MonteCarlo"

sim_agent1_agent2 = Simulator(
    config,
    MCTSAgent(
        config,
        NeuralNetworkMonteCarloTreeSearch(
            config,
        ),
    ),
    # NeuralNetworkAgent(config),
    MCTSAgent(config, TabularMonteCarloTreeSearch(config, TabularMonteCarlo(config))),
)

sim_agent2_agent1 = Simulator(
    config,
    MCTSAgent(config, TabularMonteCarloTreeSearch(config, TabularMonteCarlo(config))),
    # NeuralNetworkAgent(config),
    MCTSAgent(
        config,
        NeuralNetworkMonteCarloTreeSearch(
            config,
        ),
    ),
)

# simMonteCarloVsClassic = Simulator(
#     config,
#     MCTSAgent(config, TabularMonteCarloTreeSearch(config, TabularMonteCarlo(config))),
#     MCTSAgent(config, ClassicMonteCarloTreeSearch(config)),
# )

count = 0
reward_agent1_agent2 = 0
reward_agent2_agent1 = 0

for i in range(50):
    random_position = None
    while random_position is None:
        random_position = sim_agent1_agent2.generate_random_position(
            random.randint(0, 4)
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            agent1_agent2_result = executor.submit(
                sim_agent1_agent2.simulate,
                random_position,
                random_position.active_player(),
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            agent2_agent1_result = executor.submit(
                sim_agent2_agent1.simulate,
                random_position,
                random_position.active_player(),
            )

        agent1_agent2_result = agent1_agent2_result.result()
        agent2_agent1_result = agent2_agent1_result.result()

        if agent1_agent2_result == 1:
            reward_agent1_agent2 += 1.0
        else:
            if agent1_agent2_result == 0:
                reward_agent1_agent2 += 0.5
                reward_agent2_agent1 += 0.5
            else:
                reward_agent2_agent1 += 1.0

        if agent2_agent1_result == 1:
            reward_agent2_agent1 += 1.0
        else:
            if agent2_agent1_result == 0:
                reward_agent2_agent1 += 0.5
                reward_agent1_agent2 += 0.5
            else:
                reward_agent1_agent2 += 1.0

        count += 2
        print(
            agent1,
            "vs",
            agent2,
            reward_agent1_agent2,
            "/",
            count,
            agent2,
            "vs",
            agent1,
            reward_agent2_agent1,
            "/",
            count,
            "B2B1:",
        )
        ratio_1 = reward_agent1_agent2 / count
        ratio_2 = reward_agent2_agent1 / count

        print("Evaluate, after iteration", i, ", the reward ratio is", ratio_1, ratio_2)
