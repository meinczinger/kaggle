from datetime import datetime
from agents.simulator import Simulator
from agents.nn_agent import NeuralNetworkAgent
from kaggle_environments.utils import Struct
import numpy as np
from agents.bitboard import BitBoard
import os
from agents.logger import Logger
from agents.baseline import BaselineAgent
import pandas as pd
from agents.model.nn_model import StateValueNNModel, PriorsNNModel
from pathlib import Path
import threading
import concurrent.futures
import time
import random


GAMES_FOLDER = Path("resources/games/")
config = Struct()
config.columns = 7
config.rows = 6
config.inarow = 4
config.timeout = 2.0
config.actTimeout = 2.0

logger = Logger.info_logger("pipeline", target="pipeline.log")
baseline_result_logger = Logger.info_logger(
    "baseline_result", target="baseline_result.log"
)
evaluation_result_logger = Logger.info_logger(
    "evaluation_result", target="evaluation_result.log"
)

BUFFER_SIZE = 15000
HISTORY_SIZE = 100000
SAMPLE_SIZE = 20000
EXPLORATION_PHASE_SELF_PLAY = 12
EXPLORATION_PHASE_EVALUATION = 4
LEARNING_RATE = 1e-4
TIME_REDUCTION = 1.5
TIME_REDUCTION_EVALUATION = 1.5
Z_STAT_SIGNIFICANT = 1.5
DEPTH_FOR_RANDOM_GAMES = 20


def cut_games_file(player):
    # cut states files
    games_file = "train_state_value" + "_p" + str(player) + ".csv"
    state_values = pd.read_csv(GAMES_FOLDER / games_file, delimiter=",", header=None)
    state_values = state_values[-HISTORY_SIZE:]
    state_values.to_csv(GAMES_FOLDER / games_file, index=False, header=False)
    # cut priorities files
    games_file = "train_priors" + "_p" + str(player) + ".csv"
    priors = pd.read_csv(GAMES_FOLDER / games_file, delimiter=",", header=None)
    priors = priors[-HISTORY_SIZE:]
    priors.to_csv(GAMES_FOLDER / games_file, index=False, header=False)


def train_state_value_model(train, player):
    train_x_channels = train_y = None
    if train:
        # state_values = np.genfromtxt('resources/games/train_state_value' + '_p' + str(player) + '.csv',
        #                              delimiter=',', dtype=np.int64)
        games_file = "train_state_value" + "_p" + str(player) + ".csv"
        state_values = pd.read_csv(
            GAMES_FOLDER / games_file, delimiter=",", header=None
        )
        # state_values = state_values[state_values.iloc[:, -1] != 0]
        state_values = state_values[-BUFFER_SIZE:]
        # sample_values = state_values[-BUFFER_SIZE:]
        # priorities = SAMPLE_SIZE * 0.2 * np.power(10, np.linspace(0, -1.5, num=42))
        # sample_values = None
        # for priority, distance in zip(priorities, range(42)):
        #     sv = state_values[state_values.iloc[:, 0] == distance].to_numpy(dtype=np.float)
        #     sample_indices = np.random.choice(len(sv), min(int(priority), len(sv)))
        #     print("Adding distance", distance, "with", len(sample_indices), "samples")
        #     sv = sv[sample_indices, 1:]
        #     if sample_values is None:
        #         sample_values = sv
        #     else:
        #         sample_values = np.append(sample_values, sv, axis=0)
        #
        # sample_indices = np.random.choice(len(sample_values), SAMPLE_SIZE)
        # sample_values = sample_values[sample_indices]

        train_data = state_values.to_numpy(dtype=float)
        train_data = train_data[:, 1:]
        train_x = train_data[:, :-1]
        train_x = train_x.reshape(train_x.shape[0], 6, 7)

        # train_x = sample_values[:, :-1]
        # train_x = train_x.reshape(train_x.shape[0], 6, 7)
        train_x_channels = np.zeros((train_x.shape[0], 6, 7, 2))
        train_x_channels[:, :, :, 0] = np.where(train_x == 1, 1, 0)
        train_x_channels[:, :, :, 1] = np.where(train_x == 2, 1, 0)
        train_y = train_data[:, -1:]
        # train_y = np.where(train_y == 1, 1, 0)
        train_y = (train_y + 1.0) / 2.0
        # train_y += 1
        # train_y = to_categorical(train_y, num_classes=3)

    state_value_conv_model = StateValueNNModel(
        "candidate_state_value_model_p" + str(player)
    )

    try:
        state_value_conv_model.load(None, LEARNING_RATE)
    except:
        state_value_conv_model.create_model(5e-4)

    if train:
        state_value_conv_model.train(train_x_channels, train_y, 20, 8)
        logger.info(
            "Loss of state value network is "
            + str(state_value_conv_model.history().history["loss"][-1])
        )
        logger.info(
            "MSE of state value network is "
            + str(state_value_conv_model.history().history["mean_squared_error"][-1])
        )

    state_value_conv_model.save()
    return state_value_conv_model


def train_priors_model(train, player):
    train_x_channels = train_y = None
    if train:
        games_file = "train_priors" + "_p" + str(player) + ".csv"
        priors = pd.read_csv(GAMES_FOLDER / games_file, delimiter=",", header=None)

        # priors = np.genfromtxt('resources/games/train_priors' + '_p' + str(player) + '.csv', delimiter=',')
        # priors = np.nan_to_num(priors, copy=False, nan=0.)

        # Ignore the first 20 % of the data
        # priors = priors[-BUFFER_SIZE:]
        sample_values = priors[-BUFFER_SIZE:]
        # priorities = SAMPLE_SIZE * 0.2 * np.power(10, np.linspace(0, -1.5, num=42))
        # sample_values = None
        # for priority, distance in zip(priorities, range(42)):
        #     sv = priors[priors.iloc[:, 0] == distance].to_numpy(dtype=np.float)
        #     sv = np.nan_to_num(sv, copy=False, nan=0.)
        #     sample_indices = np.random.choice(len(sv), min(int(priority), len(sv)))
        #     sv = sv[sample_indices, 1:]
        #     if sample_values is None:
        #         sample_values = sv
        #     else:
        #         sample_values = np.append(sample_values, sv, axis=0)
        #
        # sample_indices = np.random.choice(len(sample_values), SAMPLE_SIZE)
        # sample_values = sample_values[sample_indices]

        train_data = sample_values.to_numpy(dtype=float)
        train_data = train_data[:, 1:]
        train_x = train_data[:, :42]
        train_x = train_x.reshape(train_x.shape[0], 6, 7)

        # train_x = sample_values[:, :42]
        # train_x = train_x.reshape(train_x.shape[0], 6, 7)
        train_x_channels = np.zeros((train_x.shape[0], 6, 7, 2))
        train_x_channels[:, :, :, 0] = np.where(train_x == 1, 1, 0)
        train_x_channels[:, :, :, 1] = np.where(train_x == 2, 1, 0)

        train_y = train_data[:, 42:]

    priors_conv_model = PriorsNNModel("candidate_priors_model_p" + str(player))

    try:
        priors_conv_model.load(None, LEARNING_RATE)
    except:
        priors_conv_model.create_model(5e-4)

    if train:
        priors_conv_model.train(train_x_channels, train_y, 20, 8)

        logger.info(
            "Loss of prior network is "
            + str(priors_conv_model.history().history["loss"][-1])
        )
        logger.info(
            "MSE of prior network is "
            + str(priors_conv_model.history().history["mean_squared_error"][-1])
        )

    priors_conv_model.save()
    return priors_conv_model


def play_writer(df, name, lock, thread_nr):
    with lock:
        df.to_csv(GAMES_FOLDER / name, index=False, header=False, mode="a")


def self_play(iter, lock, thread_nr):
    print("Starting self play", "iter=", iter)
    time_reduction = TIME_REDUCTION

    sim = Simulator(
        config,
        NeuralNetworkAgent(
            config, True, False, True, True, EXPLORATION_PHASE_SELF_PLAY, time_reduction
        ),
    )
    for i in range(iter):
        random_position = None
        count = 0
        while random_position is None:
            random_position = sim.generate_random_position(
                random.randint(0, DEPTH_FOR_RANDOM_GAMES)
            )
            count += 1
        # print("Took", count, "attempts to get a random position, starting from ply",
        #       random_position.ply())
        print("Thread:", thread_nr, "Play:", i)
        sim.self_play(random_position, play_writer, lock, thread_nr)


def custom_hook(args):
    print(f"Thread failed: {args.exc_value}")


def parallel_self_play(iter):
    logger.info("Starting self play")
    NR_OF_THREADS = 5
    threading.excepthook = custom_hook
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=NR_OF_THREADS) as executor:
        for i in range(NR_OF_THREADS):
            executor.submit(self_play, iter, lock, i)
            time.sleep(5)


def optimize(train=True):
    logger.info("Starting training")
    train_state_value_model(train, 1)
    train_state_value_model(train, 2)
    train_priors_model(train, 1)
    train_priors_model(train, 2)


def zstat(x, y, sample_size):
    pstd = ((sample_size - 1) * np.std(x) ** 2 + (sample_size - 1) * np.std(y) ** 2) / (
        2 * sample_size - 2
    )
    tstat = (np.average(x) - np.average(y)) / (np.sqrt(pstd * 2 / sample_size))
    # zstat = (np.average(x) - np.average(y)) / np.sqrt(np.std(x)**2/sample_size + np.std(y)**2/sample_size)
    # print('tstat/zstat', tstat, zstat)
    return tstat


def evaluate(iterations):
    logger.info("Starting evaluation")
    time_reduction = TIME_REDUCTION_EVALUATION
    best1_best2 = Simulator(
        config,
        NeuralNetworkAgent(
            config,
            False,
            True,
            True,
            True,
            EXPLORATION_PHASE_EVALUATION,
            time_reduction,
        ),
        NeuralNetworkAgent(
            config,
            False,
            True,
            True,
            True,
            EXPLORATION_PHASE_EVALUATION,
            time_reduction,
        ),
    )
    candidate1_best2 = Simulator(
        config,
        NeuralNetworkAgent(
            config,
            False,
            True,
            False,
            True,
            EXPLORATION_PHASE_EVALUATION,
            time_reduction,
        ),
        NeuralNetworkAgent(
            config,
            False,
            True,
            True,
            True,
            EXPLORATION_PHASE_EVALUATION,
            time_reduction,
        ),
    )
    best1_candidate2 = Simulator(
        config,
        NeuralNetworkAgent(
            config,
            False,
            True,
            True,
            True,
            EXPLORATION_PHASE_EVALUATION,
            time_reduction,
        ),
        NeuralNetworkAgent(
            config,
            False,
            True,
            True,
            False,
            EXPLORATION_PHASE_EVALUATION,
            time_reduction,
        ),
    )

    round_length = 10

    best1_best2_buffer = []
    best2_best1_buffer = []
    candidate1_best2_buffer = []
    candidate2_best1_buffer = []
    zstat1, zstat2 = 0, 0

    for i in range(int(2 * iterations / round_length)):
        reward_best1_best2 = 0
        reward_best2_best1 = 0
        reward_candidate1_best2 = 0
        reward_candidate2_best1 = 0

        for j in range(round_length):
            random_position = None
            while random_position is None:
                random_position = best1_best2.generate_random_position(
                    random.randint(0, DEPTH_FOR_RANDOM_GAMES)
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                # Best vs best
                winnerbb = executor.submit(
                    best1_best2.simulate,
                    random_position,
                    random_position.active_player(),
                )
                # winnerbb = best1_best2.simulate(BitBoard.create_empty_board(config.columns, config.rows, config.inarow, 1), 1)
                # Candidate vs best
                # winnercb = candidate1_best2.simulate(BitBoard.create_empty_board(config.columns, config.rows, config.inarow, 1), 1)
                winnercb = executor.submit(
                    candidate1_best2.simulate,
                    random_position,
                    random_position.active_player(),
                )
                # Best vs candidate
                # winnerbc = best1_candidate2.simulate(BitBoard.create_empty_board(config.columns, config.rows, config.inarow, 1), 1)
                winnerbc = executor.submit(
                    best1_candidate2.simulate,
                    random_position,
                    random_position.active_player(),
                )

            winnerbb = winnerbb.result()
            winnercb = winnercb.result()
            winnerbc = winnerbc.result()
            if winnerbb == 1:
                reward_best1_best2 += 1.0
            else:
                if winnerbb == 0:
                    reward_best1_best2 += 0.5
                    reward_best2_best1 += 0.5
                else:
                    reward_best2_best1 += 1.0

            if winnercb == 1:
                reward_candidate1_best2 += 1.0
            else:
                if winnercb == 0:
                    reward_candidate1_best2 += 0.5

            if winnerbc == 2:
                reward_candidate2_best1 += 1.0
            else:
                if winnerbc == 0:
                    reward_candidate2_best1 += 0.5

        best1_best2_buffer.append(reward_best1_best2 / float(round_length))
        best2_best1_buffer.append(reward_best2_best1 / float(round_length))
        candidate1_best2_buffer.append(reward_candidate1_best2 / float(round_length))
        candidate2_best1_buffer.append(reward_candidate2_best1 / float(round_length))

        print("Candidate1 vs best2 averages after round", i, candidate1_best2_buffer)
        print("Best1 vs best2 averages after round", i, best1_best2_buffer)
        print("Candidate2 vs best1 averages after round", i, candidate2_best1_buffer)
        print("Best2 vs best1 averages after round", i, best2_best1_buffer)

        zstat1 = zstat(candidate1_best2_buffer, best1_best2_buffer, round_length)
        zstat2 = zstat(candidate2_best1_buffer, best2_best1_buffer, round_length)
        print(
            "zstats after round",
            i,
            "candidate1/best1 against best2",
            round(zstat1, 2),
            "candidate2/best2 against best1",
            round(zstat2, 2),
        )

    logger.info(
        "zstats - candidate1 vs best1: "
        + str(round(zstat1, 2))
        + ", candidate2 vs best2: "
        + str(round(zstat2, 2))
    )

    evaluation_result_logger.info(str(round(zstat1, 2)) + ", " + str(round(zstat2, 2)))

    if zstat1 >= Z_STAT_SIGNIFICANT:
        evaluation_result_logger.info("--- Swap for player 1")
        logger.info("Making last candidate agent for player 1 to become the best agent")
        os.system(
            "cp resources/models/candidate_state_value_model_p1.h5 resources/models/best_state_value_model_p1.h5"
        )
        os.system(
            "cp resources/models/candidate_priors_model_p1.h5 resources/models/best_priors_model_p1.h5"
        )
    if zstat2 >= Z_STAT_SIGNIFICANT:
        evaluation_result_logger.info("--- Swap for player 2")
        logger.info("Making last candidate agent for player 2 to become the best agent")
        os.system(
            "cp resources/models/candidate_state_value_model_p2.h5 resources/models/best_state_value_model_p2.h5"
        )
        os.system(
            "cp resources/models/candidate_priors_model_p2.h5 resources/models/best_priors_model_p2.h5"
        )

    if (zstat1 >= Z_STAT_SIGNIFICANT) or (zstat2 >= Z_STAT_SIGNIFICANT):
        evaluate_against_baseline(iterations)


def evaluate_against_baseline(iter):
    logger.info("Starting evaluation against baseline")
    time_reduction = TIME_REDUCTION_EVALUATION
    sim_B1_C2 = Simulator(
        config,
        BaselineAgent(config),
        NeuralNetworkAgent(
            config, False, True, True, EXPLORATION_PHASE_EVALUATION, time_reduction
        ),
    )
    sim_C1_B2 = Simulator(
        config,
        NeuralNetworkAgent(
            config,
            False,
            True,
            True,
            True,
            EXPLORATION_PHASE_EVALUATION,
            time_reduction,
        ),
        BaselineAgent(config),
    )
    ratio_1 = ratio_2 = 0
    count = 0

    reward_b1_c2 = reward_c2_b1 = 0
    reward_c1_b2 = reward_b2_c1 = 0

    for i in range(iter):
        random_position = None
        while random_position is None:
            random_position = sim_B1_C2.generate_random_position(
                random.randint(0, DEPTH_FOR_RANDOM_GAMES)
            )

        bitboards = [random_position for _ in range(2)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # B1 vs C2
            winnerBN = executor.submit(
                sim_B1_C2.simulate, random_position, random_position.active_player()
            )

            # C1 vs B2
            winnerNB = executor.submit(
                sim_C1_B2.simulate, random_position, random_position.active_player()
            )

        winnerBN = winnerBN.result()
        winnerNB = winnerNB.result()

        if winnerBN == 1:
            reward_b1_c2 += 1.0
        else:
            if winnerBN == 0:
                reward_b1_c2 += 0.5
                reward_c2_b1 += 0.5
            else:
                reward_c2_b1 += 1.0

        if winnerNB == 1:
            reward_c1_b2 += 1.0
        else:
            if winnerNB == 0:
                reward_c1_b2 += 0.5
                reward_b2_c1 += 0.5
            else:
                reward_b2_c1 += 1.0

        count += 1
        print(
            "C1B2:",
            reward_c1_b2,
            "/",
            count,
            "B1B2:",
            "C2B1:",
            reward_c2_b1,
            "/",
            count,
            "B2B1:",
        )
        ratio_1 = reward_c1_b2 / count
        ratio_2 = reward_c2_b1 / count

        print("Evaluate, after iteration", i, ", the reward ratio is", ratio_1, ratio_2)

    logger.info(
        "The final reward ratio against baseline is "
        + str(ratio_1)
        + ", "
        + str(ratio_2)
    )

    baseline_result_logger.info(str(round(ratio_1, 2)) + ", " + str(round(ratio_2, 2)))

    print(
        "The final reward ratio against baseline is "
        + str(ratio_1)
        + ", "
        + str(ratio_2)
    )


def pipeline(iterations, rounds):
    for _ in range(rounds):
        parallel_self_play(300)
        optimize(True)
        evaluate(iterations)


# pipeline(100, 1)
evaluate_against_baseline(100)
