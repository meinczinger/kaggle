import sys
import os
sys.path.append(os.path.dirname("/kaggle_simulations/agent/"))
sys.path.append(os.path.dirname("/kaggle_simulations/agent/agents"))

from calendar import c
import random
from kaggle_environments.utils import Struct
import cProfile
import os
import numpy as np
from my_agents.simulator import Simulator
from my_agents.bitboard import BitBoard
from my_agents.rl_agent import RLAgent
from my_agents.rl_tabular.monte_carlo import TabularMonteCarlo
from my_agents.mcts.classic_mcts import ClassicMonteCarloTreeSearch
from my_agents.mcts.nn_mcts import NeuralNetworkMonteCarloTreeSearch
from my_agents.mcts_agent import MCTSAgent
from my_agents.mcts.tabular_mcts import TabularMonteCarloTreeSearch
from my_agents.mcts_with_lookup import MCTSWithLookupAgent
import threading
import concurrent.futures


config = Struct()
config.columns = 7
config.rows = 6
config.inarow = 4
config.episodeSteps = 0
config.actTimeout = 2.0
config.timeout = 2.0


TIME_REDUCTION_EVALUATION = 1.5

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
agent2 = "MCTS"

sim_agent1_agent2 = Simulator(
    config,
    MCTSWithLookupAgent(
        config,
        ClassicMonteCarloTreeSearch(config),
        self_play=False,
        time_reduction=TIME_REDUCTION_EVALUATION,
    ),
    MCTSAgent(
        config,
        NeuralNetworkMonteCarloTreeSearch(
            config,
            self_play=False,
            evaluation=True,
            use_best_player1=True,
            use_best_player2=True,
        ),
        self_play=False,
        time_reduction=TIME_REDUCTION_EVALUATION,
    ),
    # NeuralNetworkAgent(
    #     config,
    #     self_play=False,
    #     evaluation=False,
    #     use_best_player1=True,
    #     use_best_player2=True,
    #     exploration_phase=0,
    #     time_reduction=TIME_REDUCTION_EVALUATION,
    # )
)

sim_agent2_agent1 = Simulator(
    config,
    MCTSAgent(
        config,
        ClassicMonteCarloTreeSearch(config),
        self_play=False,
        time_reduction=TIME_REDUCTION_EVALUATION,
    ),
    # NeuralNetworkAgent(
    #     config,
    #     self_play=False,
    #     evaluation=False,
    #     use_best_player1=True,
    #     use_best_player2=True,
    #     exploration_phase=0,
    #     time_reduction=TIME_REDUCTION_EVALUATION,
    # ),
    MCTSAgent(
        config,
        NeuralNetworkMonteCarloTreeSearch(
            config,
            self_play=False,
            evaluation=True,
            use_best_player1=True,
            use_best_player2=True,
        ),
        self_play=False,
        time_reduction=TIME_REDUCTION_EVALUATION,
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

winner = sim_agent1_agent2.simulate(BitBoard.create_empty_board(config.columns, config.rows, config.inarow, 1), 1)

print("the winner is", winner)

# for i in range(100):
#     random_position = None
#     while random_position is None:
#         random_position = sim_agent1_agent2.generate_random_position(
#             random.randint(0, 0)
#         )

#         with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
#             agent1_agent2_result = executor.submit(
#                 sim_agent1_agent2.simulate,
#                 random_position,
#                 random_position.active_player(),
#             )

#         with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
#             agent2_agent1_result = executor.submit(
#                 sim_agent2_agent1.simulate,
#                 random_position,
#                 random_position.active_player(),
#             )

#         agent1_agent2_result = agent1_agent2_result.result()
#         agent2_agent1_result = agent2_agent1_result.result()

#         if agent1_agent2_result == 1:
#             reward_agent1_agent2 += 1.0
#         else:
#             if agent1_agent2_result == 0:
#                 reward_agent1_agent2 += 0.5
#                 reward_agent2_agent1 += 0.5
#             else:
#                 reward_agent2_agent1 += 1.0

#         if agent2_agent1_result == 1:
#             reward_agent2_agent1 += 1.0
#         else:
#             if agent2_agent1_result == 0:
#                 reward_agent2_agent1 += 0.5
#                 reward_agent1_agent2 += 0.5
#             else:
#                 reward_agent1_agent2 += 1.0

#         count += 2
#         print(
#             agent1,
#             "vs",
#             agent2,
#             reward_agent1_agent2,
#             "/",
#             count,
#             agent2,
#             "vs",
#             agent1,
#             reward_agent2_agent1,
#             "/",
#             count,
#             "B2B1:",
#         )
#         ratio_1 = reward_agent1_agent2 / count
#         ratio_2 = reward_agent2_agent1 / count

#         print("Evaluate, after iteration", i, ", the reward ratio is", ratio_1, ratio_2)
