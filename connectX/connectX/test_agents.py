from kaggle_environments.utils import Struct
import cProfile
import pstats
from nn_agent import NeuralNetworkAgent
import os
import numpy as np
from simulator import Simulator
from bitboard import BitBoard


config = Struct()
config.columns = 7
config.rows = 6
config.inarow = 4
config.episodeSteps = 0
config.actTimeout = 2.0
config.timeout = 2.0

obs = Struct()
obs.step = 0
obs.board = [0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0]
obs.mark = 1

agent = NeuralNetworkAgent(config, True, True, True)
action = agent.act(obs)

# cProfile.run('agent.act(obs)', 'stats')
# p = pstats.Stats('stats')
# p.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(100)

#minimax_agent = MinimaxAgent(config, 3)
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



#cProfile.run('gtc.build_tree(1000)')
#cProfile.run('agent.act(obs)')


# sim = Simulator(config, NeuralNetworkAgent(config, False, False, False, 0),
#                 NeuralNetworkAgent(config, False, False, False, 0))
#
# sim.simulate(BitBoard.create_empty_board(config.columns, config.rows, config.inarow, 1), 1)

