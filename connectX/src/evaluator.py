import random
import concurrent
import numpy as np
import os
from pathlib import Path

from game.simulator import Simulator
from logger import Logger
from agent.mcts_agent import MCTSAgent
from mcts.nn_mcts import NeuralNetworkMonteCarloTreeSearch
from mcts.classic_mcts import ClassicMonteCarloTreeSearch
from game.config import get_config

Z_STAT_SIGNIFICANT = 1.0

config = get_config()


class Evaluator:

    def __init__(
        self,
        time_reduction: float,
        dept_for_random_games: int,
        prob_for_random_move: float,
        games_folder: Path,
        models_folder: Path,
    ):
        self._time_reduction = time_reduction
        self._dept_for_random_games = dept_for_random_games
        self._prob_for_random_move = prob_for_random_move
        self._games_folder = games_folder
        self._models_folder = models_folder
        self._logger = Logger.info_logger("Evaluator")
        self._evaluation_result_logger = Logger.info_logger(
            "evaluation_result", target="evaluation_result.log"
        )

    def create_agent(
        self, use_best_player_1: bool, use_best_player_2: bool
    ) -> MCTSAgent:
        return MCTSAgent(
            config,
            NeuralNetworkMonteCarloTreeSearch(
                config,
                self_play=False,
                evaluation=True,
                use_best_player1=use_best_player_1,
                use_best_player2=use_best_player_2,
            ),
            self_play=False,
            time_reduction=self._time_reduction,
        )

    def create_simulator(
        self,
        agent_1_use_best_player_1: bool,
        agent_1_use_best_player_2: bool,
        agent_2_use_best_player_1: bool,
        agent_2_use_best_player_2: bool,
    ) -> Simulator:
        return Simulator(
            config,
            self._games_folder,
            self._models_folder,
            self.create_agent(agent_1_use_best_player_1, agent_1_use_best_player_2),
            self.create_agent(agent_2_use_best_player_1, agent_2_use_best_player_2),
        )

    def random_position_generator(self, depth: int, prob: float, lock=None):
        randomGameGenerator = Simulator(
            config,
            self._games_folder,
            self._models_folder,
            MCTSAgent(
                config,
                NeuralNetworkMonteCarloTreeSearch(
                    config,
                    self_play=True,
                    evaluation=False,
                    use_best_player1=True,
                    use_best_player2=True,
                ),
                self_play=True,
                time_reduction=self._time_reduction,
            ),
        )
        random_position = None
        while random_position is None:
            random_position = randomGameGenerator.generate_random_position(depth, prob)
        return random_position

    def zstat(self, x, y, sample_size):
        pstd = (
            (sample_size - 1) * np.std(x) ** 2 + (sample_size - 1) * np.std(y) ** 2
        ) / (2 * sample_size - 2)
        tstat = (np.average(x) - np.average(y)) / (np.sqrt(pstd * 2 / sample_size))
        return tstat

    def evaluate(self, iterations=200, parallelism=4):
        self._logger.info("Starting evaluation")

        best1_best2 = [
            self.create_simulator(True, True, True, True) for _ in range(parallelism)
        ]
        candidate1_best1 = [
            self.create_simulator(False, True, True, True) for _ in range(parallelism)
        ]
        best1_candidate2 = [
            self.create_simulator(True, True, True, False) for _ in range(parallelism)
        ]

        round_length = 5

        best1_best2_buffer = []
        best2_best1_buffer = []
        candidate1_best2_buffer = []
        candidate2_best1_buffer = []
        zstat1, zstat2 = 0, 0

        for it in range(int(iterations / parallelism / round_length)):
            for r in range(round_length):
                random_positions = [
                    self.random_position_generator(
                        random.randint(0, self._dept_for_random_games),
                        self._prob_for_random_move,
                    )
                    for _ in range(parallelism)
                ]

                winnerbbres = [None] * parallelism
                winnercbres = [None] * parallelism
                winnerbcres = [None] * parallelism

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=3 * parallelism
                ) as executor:
                    for j in range(parallelism):
                        # Best vs best
                        winnerbbres[j] = executor.submit(
                            best1_best2[j].simulate,
                            random_positions[j],
                            random_positions[j].active_player(),
                        )
                        # Candidate vs best
                        winnercbres[j] = executor.submit(
                            candidate1_best1[j].simulate,
                            random_positions[j],
                            random_positions[j].active_player(),
                        )
                        # Best vs candidate
                        winnerbcres[j] = executor.submit(
                            best1_candidate2[j].simulate,
                            random_positions[j],
                            random_positions[j].active_player(),
                        )

                reward_best1_best2 = 0
                reward_best2_best1 = 0
                reward_candidate1_best1 = 0
                reward_candidate2_best2 = 0

                for j in range(parallelism):
                    winnerbb = winnerbbres[j].result()
                    winnercb = winnercbres[j].result()
                    winnerbc = winnerbcres[j].result()
                    if winnerbb == 1:
                        reward_best1_best2 += 1.0
                    else:
                        if winnerbb == 0:
                            reward_best1_best2 += 0.5
                            reward_best2_best1 += 0.5
                        else:
                            reward_best2_best1 += 1.0

                    if winnercb == 1:
                        reward_candidate1_best1 += 1.0
                    else:
                        if winnercb == 0:
                            reward_candidate1_best1 += 0.5

                    if winnerbc == 2:
                        reward_candidate2_best2 += 1.0
                    else:
                        if winnerbc == 0:
                            reward_candidate2_best2 += 0.5

                best1_best2_buffer.append(reward_best1_best2 / float(parallelism))
                best2_best1_buffer.append(reward_best2_best1 / float(parallelism))
                candidate1_best2_buffer.append(
                    reward_candidate1_best1 / float(parallelism)
                )
                candidate2_best1_buffer.append(
                    reward_candidate2_best2 / float(parallelism)
                )

            print(
                "Candidate1 vs best1 averages after round", it, candidate1_best2_buffer
            )
            print("Best1 vs best2 averages after round", it, best1_best2_buffer)
            print(
                "Candidate2 vs best2 averages after round", it, candidate2_best1_buffer
            )
            print("Best2 vs best1 averages after round", it, best2_best1_buffer)

            print(
                f"Average c1/b1 is {sum(candidate1_best2_buffer) / len(candidate1_best2_buffer)}, {sum(best1_best2_buffer) / len(best1_best2_buffer)}"
            )
            print(
                f"Average c2/b2 is {sum(candidate2_best1_buffer) / len(candidate2_best1_buffer)}, {sum(best2_best1_buffer) / len(best2_best1_buffer)}"
            )

            zstat1 = self.zstat(
                candidate1_best2_buffer, best1_best2_buffer, (r + 1) * round_length
            )
            zstat2 = self.zstat(
                candidate2_best1_buffer, best2_best1_buffer, (r + 1) * round_length
            )
            print(
                "zstats after round",
                it,
                "candidate1 against best1",
                round(zstat1, 2),
                "candidate2 against best2",
                round(zstat2, 2),
            )

        self._logger.info(
            "zstats - candidate1 vs best1: "
            + str(round(zstat1, 2))
            + ", candidate2 vs best2: "
            + str(round(zstat2, 2))
        )

        self._evaluation_result_logger.info(
            str(round(zstat1, 2)) + ", " + str(round(zstat2, 2))
        )

        if zstat1 >= Z_STAT_SIGNIFICANT:
            self._evaluation_result_logger.info("--- Swap for player 1")
            self._logger.info(
                "Making last candidate agent for player 1 to become the best agent"
            )
            os.system(
                "cp resources/models/candidate_model_p1.keras resources/models/best_model_p1.keras"
            )
        if zstat2 >= Z_STAT_SIGNIFICANT:
            self._evaluation_result_logger.info("--- Swap for player 2")
            self._logger.info(
                "Making last candidate agent for player 2 to become the best agent"
            )
            os.system(
                "cp resources/models/candidate_model_p2.keras resources/models/best_model_p2.keras"
            )
        if (zstat1 >= Z_STAT_SIGNIFICANT) or (zstat2 >= Z_STAT_SIGNIFICANT):
            self.evaluate_against_baseline(iterations / 2)

    def evaluate_against_baseline(self, iter, parallelism=4):
        self._logger.info("Starting evaluation against baseline")
        sim_B1_C2 = [
            Simulator(
                config,
                self._games_folder,
                self._models_folder,
                MCTSAgent(
                    config,
                    ClassicMonteCarloTreeSearch(config),
                    self_play=False,
                    time_reduction=self._time_reduction,
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
                    time_reduction=self._time_reduction,
                ),
            )
            for _ in range(parallelism)
        ]
        sim_C1_B2 = [
            Simulator(
                config,
                self._games_folder,
                self._models_folder,
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
                    time_reduction=self._time_reduction,
                ),
                MCTSAgent(
                    config,
                    ClassicMonteCarloTreeSearch(config),
                    self_play=False,
                    time_reduction=self._time_reduction,
                ),
            )
            for _ in range(parallelism)
        ]
        ratio_1 = ratio_2 = 0
        count = 0

        reward_b1_c2 = reward_c2_b1 = 0
        reward_c1_b2 = reward_b2_c1 = 0

        for i in range(int(iter / parallelism)):
            random_positions = [
                self.random_position_generator(
                    random.randint(0, self._dept_for_random_games),
                    self._prob_for_random_move,
                )
                for _ in range(parallelism)
            ]

            winnerBNres = [None] * parallelism
            winnerNBres = [None] * parallelism

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=2 * parallelism
            ) as executor:
                for proc in range(parallelism):
                    # B1 vs C2
                    winnerBNres[proc] = executor.submit(
                        sim_B1_C2[proc].simulate,
                        random_positions[proc],
                        random_positions[proc].active_player(),
                    )

                    # C1 vs B2
                    winnerNBres[proc] = executor.submit(
                        sim_C1_B2[proc].simulate,
                        random_positions[proc],
                        random_positions[proc].active_player(),
                    )

            for proc in range(parallelism):
                winnerBN = winnerBNres[proc].result()
                winnerNB = winnerNBres[proc].result()

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

            count += parallelism
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

            print(
                "Evaluate, after iteration",
                i,
                ", the reward ratio is",
                ratio_1,
                ratio_2,
            )

        self._evaluation_result_logger.info(
            "The final reward ratio against baseline is "
            + str(ratio_1)
            + ", "
            + str(ratio_2)
        )

        print(
            "The final reward ratio against baseline is "
            + str(ratio_1)
            + ", "
            + str(ratio_2)
        )
