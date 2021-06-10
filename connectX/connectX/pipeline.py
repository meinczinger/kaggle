from simulator import Simulator
from nn_agent import NeuralNetworkAgent
from kaggle_environments.utils import Struct
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras import initializers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from bitboard import BitBoard
import os
from logger import Logger
from baseline import BaselineAgent
import pandas as pd
from nn_model import StateValueNNModel, PriorsNNModel


config = Struct()
config.columns = 7
config.rows = 6
config.inarow = 4
config.timeout = 2.0
config.actTimeout = 2.0

logger = Logger.info_logger("pipeline", target="pipeline.log")

BUFFER_SIZE = 50000
SAMPLE_SIZE = 20000
EXPLORATION_PHASE_SELF_PLAY = 12
EXPLORATION_PHASE_EVALUATION = 4
LEARNING_RATE = 5e-4


def train_state_value_model(train, player):
    train_x_channels = train_y = None
    if train:
        # state_values = np.genfromtxt('resources/games/train_state_value' + '_p' + str(player) + '.csv',
        #                              delimiter=',', dtype=np.int64)

        state_values = pd.read_csv('resources/games/train_state_value' + '_p' + str(player) + '.csv',
                                   delimiter=',', header=None)
        # state_values = state_values[state_values.iloc[:, -1] != 0]
        state_values = state_values[-BUFFER_SIZE:]
        priorities = SAMPLE_SIZE * 0.2 * np.power(10, np.linspace(0, -1.5, num=42))
        sample_values = None
        for priority, distance in zip(priorities, range(42)):
            sv = state_values[state_values.iloc[:, 0] == distance].to_numpy(dtype=np.float)
            sample_indices = np.random.choice(len(sv), min(int(priority), len(sv)))
            print("Adding distance", distance, "with", len(sample_indices), "samples")
            sv = sv[sample_indices, 1:]
            if sample_values is None:
                sample_values = sv
            else:
                sample_values = np.append(sample_values, sv, axis=0)

        sample_indices = np.random.choice(len(sample_values), SAMPLE_SIZE)
        sample_values = sample_values[sample_indices]
        # state_values = state_values.sort_values(by=[col for col in state_values.columns])
        #
        # state_values = state_values.to_numpy(dtype=np.float)
        # prev_state = state_values[0, :]
        # output = None
        # wins = 0
        # losses = 0
        #
        # for state in state_values:
        #     if np.array_equal(prev_state[:-1], state[:-1]):
        #         res = state[-1:]
        #         wins = wins + 1 if res == 1 else wins
        #         losses = losses + 1 if res < 1 else losses
        #     else:
        #         prev_state[-1:] = 1 if wins > losses else -1 if losses > wins else 0
        #         #prev_state[-1:] = 0.5 if (wins == 0) and (losses == 0) else wins / (wins + losses)
        #         if output is None:
        #             output = copy.copy(prev_state).reshape(1, 43)
        #         else:
        #             output = np.append(output, prev_state.reshape(1, 43), axis=0)
        #         prev_state = state
        #         losses = 0
        #         wins = 0
        #
        # state_values = output
        # print(state_values.shape)

        train_x = sample_values[:, :-1]
        train_x = train_x.reshape(train_x.shape[0], 6, 7)
        train_x_channels = np.zeros((train_x.shape[0], 6, 7, 2))
        train_x_channels[:, :, :, 0] = np.where(train_x == 1, 1, 0)
        train_x_channels[:, :, :, 1] = np.where(train_x == 2, 1, 0)
        train_y = sample_values[:, -1:]
        #train_y = np.where(train_y == 1, 1, 0)
        train_y = (train_y + 1.0) / 2.0
        #train_y += 1
        #train_y = to_categorical(train_y, num_classes=3)

    state_value_conv_model = StateValueNNModel('candidate_state_value_model_p' + str(player))

    try:
        state_value_conv_model.load(None, LEARNING_RATE)
    except:
        state_value_conv_model.create_model(5e-4)

    if train:
        state_value_conv_model.train(train_x_channels, train_y, 20, 8)
        logger.info("Loss of state value network is " + str(state_value_conv_model.history().history['loss'][-1]))
        logger.info("MSE of state value network is " + str(state_value_conv_model.history().history['mean_squared_error'][-1]))

    state_value_conv_model.save()
    return state_value_conv_model


def train_priors_model(train, player):
    train_x_channels = train_y = None
    if train:
        priors = pd.read_csv('resources/games/train_priors' + '_p' + str(player) + '.csv',
                             delimiter=',', header=None)

        # priors = np.genfromtxt('resources/games/train_priors' + '_p' + str(player) + '.csv', delimiter=',')
        # priors = np.nan_to_num(priors, copy=False, nan=0.)

        # Ignore the first 20 % of the data
        priors = priors[-BUFFER_SIZE:]
        priorities = SAMPLE_SIZE * 0.2 * np.power(10, np.linspace(0, -1.5, num=42))
        sample_values = None
        for priority, distance in zip(priorities, range(42)):
            sv = priors[priors.iloc[:, 0] == distance].to_numpy(dtype=np.float)
            sv = np.nan_to_num(sv, copy=False, nan=0.)
            sample_indices = np.random.choice(len(sv), min(int(priority), len(sv)))
            sv = sv[sample_indices, 1:]
            if sample_values is None:
                sample_values = sv
            else:
                sample_values = np.append(sample_values, sv, axis=0)

        sample_indices = np.random.choice(len(sample_values), SAMPLE_SIZE)
        sample_values = sample_values[sample_indices]

        # priors = priors.sort_values(by=[col for col in priors.columns])
        #
        # priors = priors.to_numpy(dtype=np.float)
        # priors = np.nan_to_num(priors, copy=False, nan=0.)
        #
        # prev_state = priors[0, :]
        # output = None
        # probs = np.zeros((7, ))
        # counter = 0
        #
        # for state in priors:
        #     if np.array_equal(prev_state[:-1], state[:-1]):
        #         probs = probs + state[42:]
        #         counter += 1
        #     else:
        #         prev_state[42:] = probs / counter
        #         #prev_state[-1:] = 0.5 if (wins == 0) and (losses == 0) else wins / (wins + losses)
        #         if output is None:
        #             output = copy.copy(prev_state).reshape(1, 49)
        #         else:
        #             output = np.append(output, prev_state.reshape(1, 49), axis=0)
        #         prev_state = state
        #         probs = np.zeros((7, ))

        train_x = sample_values[:, :42]
        train_x = train_x.reshape(train_x.shape[0], 6, 7)
        train_x_channels = np.zeros((train_x.shape[0], 6, 7, 2))
        train_x_channels[:, :, :, 0] = np.where(train_x == 1, 1, 0)
        train_x_channels[:, :, :, 1] = np.where(train_x == 2, 1, 0)

        train_y = sample_values[:, 42:]

    priors_conv_model = PriorsNNModel('candidate_priors_model_p' + str(player))

    try:
        priors_conv_model.load()
    except:
        priors_conv_model.create_model(5e-4)

    if train:
        priors_conv_model.train(train_x_channels, train_y, 20, 8)

        logger.info("Loss of prior network is " + str(priors_conv_model.history().history['loss'][-1]))
        logger.info("MSE of prior network is " + str(priors_conv_model.history().history['mean_squared_error'][-1]))

    priors_conv_model.save()
    return priors_conv_model


def self_play(iter):
    sim = Simulator(config, NeuralNetworkAgent(config, True, True, True, EXPLORATION_PHASE_SELF_PLAY))
    logger.info("Starting self play")
    for i in range(iter):
        print("Self play,", i, 'th iteration')
        sim.self_play()


def optimize(train=True):
    logger.info("Starting training")
    train_state_value_model(train, 1)
    train_state_value_model(train, 2)
    train_priors_model(train, 1)
    train_priors_model(train, 2)


def evaluate(iter):
    logger.info("Starting evaluation")
    best1_best2 = Simulator(config, NeuralNetworkAgent(config, False, True, True, EXPLORATION_PHASE_EVALUATION),
                            NeuralNetworkAgent(config, False, True, True, EXPLORATION_PHASE_EVALUATION))
    candidate1_best2 = Simulator(config, NeuralNetworkAgent(config, False, False, True, EXPLORATION_PHASE_EVALUATION),
                                 NeuralNetworkAgent(config, False, True, True, EXPLORATION_PHASE_EVALUATION))
    best1_candidate2 = Simulator(config, NeuralNetworkAgent(config, False, True, True, EXPLORATION_PHASE_EVALUATION),
                                 NeuralNetworkAgent(config, False, True, False, EXPLORATION_PHASE_EVALUATION))

    ratio_1 = ratio_2 = 0
    count = 0

    reward_best1_best2 = 0
    reward_best2_best1 = 0
    reward_candidate1_best2 = 0
    reward_candidate2_best1 = 0

    for i in range(2*iter):
        # Best vs best
        winner = best1_best2.simulate(BitBoard.create_empty_board(config.columns, config.rows, config.inarow, 1), 1)
        if winner == 1:
            reward_best1_best2 += 1.0
        else:
            if winner == 0:
                reward_best1_best2 += 0.5
                reward_best2_best1 += 0.5
            else:
                reward_best2_best1 += 1.0
        # Candidate vs best
        winner = candidate1_best2.simulate(BitBoard.create_empty_board(config.columns, config.rows, config.inarow, 1), 1)
        if winner == 1:
            reward_candidate1_best2 += 1.0
        else:
            if winner == 0:
                reward_candidate1_best2 += 0.5

        # Best vs candidate
        winner = best1_candidate2.simulate(BitBoard.create_empty_board(config.columns, config.rows, config.inarow, 1), 1)
        if winner == 2:
            reward_candidate2_best1 += 1.0
        else:
            if winner == 0:
                reward_candidate2_best1 += 0.5

        count += 1

        ratio_1 = reward_candidate1_best2 / reward_best1_best2 if reward_best1_best2 > 0 else 100
        ratio_2 = reward_candidate2_best1 / reward_best2_best1 if reward_best2_best1 > 0 else 100
        # ratio_1 = (reward_candidate1_best2 - reward_best1_best2) / count
        # ratio_2 = (reward_candidate2_best1 - reward_best2_best1) / count

        print("Candidate1-Best2:", reward_candidate1_best2, "/", count, "Best1-Best2:", reward_best1_best2, "/", count)
        print("Candidate2-Best1:", reward_candidate2_best1, "/", count, "Best2-Best1:", reward_best2_best1, "/", count, )
        print("Evaluate, after iteration", i, ", the reward ratio is", ratio_1, ratio_2)

    logger.info("The final reward ratio is " + str(ratio_1) + ", " + str(ratio_2))

    limit = 1.05
    if ratio_1 >= limit:
        logger.info("Making last candidate agent for player 1 to become the best agent")
        os.system('rm -r resources/models/best_state_value_model_p1/')
        os.system('cp -R resources/models/candidate_state_value_model_p1 resources/models/best_state_value_model_p1')
        os.system('rm -r resources/models/best_priors_model_p1/')
        os.system('cp -R resources/models/candidate_priors_model_p1 resources/models/best_priors_model_p1')
    if ratio_2 >= limit:
        logger.info("Making last candidate agent for player 2 to become the best agent")
        os.system('rm -r resources/models/best_state_value_model_p2/')
        os.system(
            'cp -R resources/models/candidate_state_value_model_p2 resources/models/best_state_value_model_p2')
        os.system('rm -r resources/models/best_priors_model_p2/')
        os.system('cp -R resources/models/candidate_priors_model_p2 resources/models/best_priors_model_p2')

    if (ratio_1 >= limit) or (ratio_2 >= limit):
        evaluate_against_baseline(iter)


def evaluate_against_baseline(iter):
    logger.info("Starting evaluation against baseline")
    sim_B1_C2 = Simulator(config, BaselineAgent(config),
                          NeuralNetworkAgent(config, False, True, True, EXPLORATION_PHASE_EVALUATION))
    sim_C1_B2 = Simulator(config, NeuralNetworkAgent(config, False, True, True, EXPLORATION_PHASE_EVALUATION),
                          BaselineAgent(config))
    ratio_1 = ratio_2 = 0
    count = 0

    reward_b1_c2 = reward_c2_b1 = 0
    reward_c1_b2 = reward_b2_c1 = 0

    for i in range(iter):
        # B1 vs C2
        winner = sim_B1_C2.simulate(BitBoard.create_empty_board(config.columns, config.rows, config.inarow, 1), 1)
        if winner == 1:
            reward_b1_c2 += 1.0
        else:
            if winner == 0:
                reward_b1_c2 += 0.5
                reward_c2_b1 += 0.5
            else:
                reward_c2_b1 += 1.0

        # C1 vs B2
        winner = sim_C1_B2.simulate(BitBoard.create_empty_board(config.columns, config.rows, config.inarow, 1), 1)
        if winner == 1:
            reward_c1_b2 += 1.0
        else:
            if winner == 0:
                reward_c1_b2 += 0.5
                reward_b2_c1 += 0.5
            else:
                reward_b2_c1 += 1.0

        count += 1
        print("C1B2:", reward_c1_b2, "/", count, "B1B2:", "C2B1:", reward_c2_b1, "/", count, "B2B1:")
        ratio_1 = reward_c1_b2 / count
        ratio_2 = reward_c2_b1 / count

        print("Evaluate, after iteration", i, ", the reward ratio is", ratio_1, ratio_2)

    logger.info("The final reward ratio against baseline is " + str(ratio_1) + ", " + str(ratio_2))
    print("The final reward ratio against baseline is " + str(ratio_1) + ", " + str(ratio_2))


def pipeline(iteration, rounds):
    for _ in range(rounds):
        self_play(10 * iteration)
        optimize(True)
        evaluate(iteration)


pipeline(100, 1)

