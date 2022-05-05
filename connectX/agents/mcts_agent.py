from agents.mcts.base_mcts import BaseMonteCarloTreeSearch
import logging
import time
import numpy as np


class MCTSAgent:
    _agent = None

    def __init__(
        self,
        configuration,
        mcts: BaseMonteCarloTreeSearch,
        self_play=False,
        time_reduction=0.05,
    ):
        self.logger = logging.getLogger("mcts_agent")
        self.logger.setLevel(logging.ERROR)
        fh = logging.FileHandler("error.log")
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        self._config = configuration
        self._mcts = mcts
        self._time_reduction = time_reduction
        self._self_play = self_play
        self._step = 0

    def act(self, observation, explore=False, step=0):
        """Main method to act on opponents move"""
        board = observation.board

        # it seems sometimes the mark is incorrect so
        own_player = observation.mark

        # Set step
        if step != 0:
            self._step = step
        else:
            if self._step == 0:
                # First step
                self._step = own_player
            else:
                if self._self_play:
                    self._step += 1
                else:
                    self._step += 2

        # Starting a new game
        if self._self_play:
            limit = 1
        else:
            limit = 2

        deadline = time.time() + self._config.actTimeout - self._time_reduction

        if self._step <= limit:
            self._mcts.initialize(board, own_player)

        action = self._mcts.search(board, own_player, self._step, deadline, True)

        return int(action)

    def MCTS(self):
        return self._mcts

    @staticmethod
    def get_instance(configuration, mcts: BaseMonteCarloTreeSearch):
        if MCTSAgent._agent is None:
            MCTSAgent._agent = MCTSAgent(configuration, mcts)
        return MCTSAgent._agent
