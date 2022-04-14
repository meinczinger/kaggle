from tensorflow.keras.models import load_model
import numpy as np
from nn_agent import NeuralNetworkAgent
from baseline import BaselineAgent
from kaggle_environments.utils import Struct
from simulator import Simulator
from bitboard import BitBoard
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt


config = Struct()
config.columns = 7
config.rows = 6
config.inarow = 4
config.timeout = 2.0
config.actTimeout = 2.0


def profile():
    agent = NeuralNetworkAgent(config, False, True)
    agent.act()

def test_self_play():
    sim = Simulator(config, NeuralNetworkAgent(config, False, True), NeuralNetworkAgent(config, False, True))
    sim.self_play()


def test_simulator():
    candidate_agent = NeuralNetworkAgent(config, True)
    best_agent = NeuralNetworkAgent(config)
    sim = Simulator(config, candidate_agent, best_agent)
    winner = sim.simulate(BitBoard.create_empty_board(config.columns, config.rows, config.inarow, 1), 1)
    print(winner)


def predictions():
    state_value_model = load_model('resources/models/candidate_state_value_model')
    priors_model = load_model('resources/models/candidate_priors_model')

    empty_board = np.zeros(shape=(1, 6, 7, 1))
    first_positions = np.zeros(shape=(7, 6, 7, 1))
    for i in range(7):
        first_positions[i, 5, i, 0] = 1.

    second_positions = np.zeros(shape=(49, 6, 7, 1))
    pos = 0
    for i in range(7):
        second_positions[pos, 5, i, 0] = 1
        for j in range(7):
            if i != j:
                second_positions[pos, 5, j, 0] = 1
            else:
                second_positions[pos, 4, j, 0] = 1
            pos += 1

    # left_first = np.zeros(shape=(7, 6, 7, 1))
    # left_first[0, ]
    end_pos = np.array([0, 0, 0, 2, 0, 0, 0,
                        1, 0, 0, 2, 0, 0, 0,
                        1, 0, 0, 1, 0, 0, 0,
                        1, 0, 0, 2, 0, 0, 1,
                        2, 2, 0, 2, 1, 2, 1,
                        2, 1, 2, 1, 1, 1, 2,
    #
                        2, 1, 2, 1, 2, 0, 1,
                        1, 1, 2, 2, 1, 0, 1,
                        2, 2, 1, 1, 1, 0, 1,
                        1, 1, 2, 2, 2, 1, 2,
                        2, 2, 1, 2, 1, 2, 1,
                        1, 2, 2, 2, 1, 2, 1,
    #
                        0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0,
                        1, 2, 0, 0, 0, 0, 0,
                        2, 2, 0, 0, 0, 0, 0,
                        2, 1, 1, 1, 0, 0, 0
                        ])

    end_pos = end_pos.reshape((3, 6, 7, 1))
    # board_end_position = BitBoard.create_from_board(7, 6, 4, 1,
    #                                           [0, 0, 0, 2, 0, 0, 0,
    #                                            1, 0, 0, 2, 0, 0, 0,
    #                                            1, 0, 0, 1, 0, 0, 0,
    #                                            1, 0, 0, 2, 0, 0, 1,
    #                                            2, 2, 2, 2, 1, 2, 1,
    #                                            2, 1, 2, 1, 1, 1, 2])
    # end_position = BitBoard.bitboard_to_numpy2d(board_end_position.hash(), 6, 7).reshape(1, 6, 7, 1)

    print("Start position:", np.around(state_value_model.predict([empty_board]) , decimals=4))
    print("Start position, priors:", np.around(priors_model.predict([empty_board]), decimals=4))
    print("First positions, state values:", np.around(state_value_model.predict([first_positions]), decimals=4))
    print("First positions, priors:", np.around(priors_model.predict([first_positions]), decimals=4))

    print("Second positions, state values:", np.around(state_value_model.predict([second_positions]), decimals=4))
    print("Second positions, priors:", np.around(priors_model.predict([second_positions]), decimals=4))

    print("End position, state values:", np.around(state_value_model.predict([end_pos]), decimals=4))
    print("End position, priors:", np.around(priors_model.predict([end_pos]), decimals=4))


def first_positions():
    state_values = pd.read_csv('resources/games/train_state_value.csv', delimiter=',', header=None)
    nr_of_records = len(state_values)
    ignore = int(nr_of_records * 0.2)
    state_values = state_values.iloc[ignore:, :]

    first_pos_win = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 1]]

    #state_values = state_values.iloc[0:5001, :]

    legend = []
    averages = defaultdict()

    counts = defaultdict()
    for i in range(7):
        counts[i] = {'win': 0, 'loss': 0}
        legend.append(str(i))
        averages[i] = []
        # counts[i]['win'] = 0
        # counts[i]['loss'] = 0



    plt.clf()
    plt.figure(figsize=(15, 5), dpi= 80, facecolor='w', edgecolor='k')
    plt.grid()

    #for i in range(len(state_values)):
    for index, row in state_values.iterrows():
        for j in range(len(first_pos_win)):
            if list(row.values[0:42]) == first_pos_win[j]:
            #if list(state_values.loc[i, 0:41]) == first_pos_win[j]:
                #if state_values.loc[i, 42] == 1:
                if row.values[42] == 1:
                    counts[j]['win'] += 1
                else:
                    # if state_values.loc[i, 42] == -1:
                    if row.values[42] == 0:
                        counts[j]['loss'] += 1
        if (index % 2000) == 0:
            for k in counts:
                ratio = 0.5 if counts[k]['win'] == counts[k]['loss'] == 0 \
                    else counts[k]['win'] / (counts[k]['win'] + counts[k]['loss'])
                averages[k].append(ratio)
                #counts[k] = {'win': 0, 'loss': 0}

    for k in counts:
        plt.plot(averages[k], linestyle="--")
        plt.legend(legend, loc='upper left')

    plt.show()

    for k in counts:
        ratio = 0.5 if counts[k]['win'] == counts[k]['loss'] == 0 \
            else counts[k]['win'] / (counts[k]['win'] + counts[k]['loss'])
        print(k, counts[k]['win'], counts[k]['loss'], ratio)

#test_self_play()
#test_simulator()
predictions()
#first_positions()