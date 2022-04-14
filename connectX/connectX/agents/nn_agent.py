import logging
from agents.nn_mcts import NeuralNetworkMonteCarloTreeSearch
import time
import numpy as np


class NeuralNetworkAgent:
    _agent = None

    def __init__(self, configuration, self_play=False, evaluation=False, use_best_player1=True, use_best_player2=True,
                 exploration_phase=0, time_reduction=0.05):
        self.logger = logging.getLogger('nn agent')
        self._config = configuration
        self._mcts = None
        self._self_play = self_play
        self._mcts = NeuralNetworkMonteCarloTreeSearch(
            self._config, self_play, evaluation, use_best_player1, use_best_player2, exploration_phase)
        self._time_reduction = time_reduction

    def act(self, observation):
        """ Main method to act on opponents move """
        board = observation.board

        # it seems sometimes the mark is incorrect so
        own_player = observation.mark

        # Starting a new game
        if self._self_play:
            limit = 1
        else:
            limit = 2

        deadline = time.time() + self._config.actTimeout - self._time_reduction

        if observation.step < limit:
            self._mcts.initialize(board, own_player)

        action = self._mcts.search(board, own_player, observation.step, deadline, True)

        return int(action)

    def priors(self):
        return self._mcts.priors()

    @staticmethod
    def get_instance(configuration):
        if NeuralNetworkAgent._agent is None:
            NeuralNetworkAgent._agent = NeuralNetworkAgent(configuration)
        return NeuralNetworkAgent._agent


