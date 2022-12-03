from email import header
from pickletools import int4
from turtle import st
from my_agents.mcts.base_mcts import BaseMonteCarloTreeSearch
import logging
import time
import numpy as np
import pandas as pd
import csv


class MCTSWithLookupAgent:
    _agent = None

    def __init__(
        self,
        configuration,
        mcts: BaseMonteCarloTreeSearch,
        self_play=False,
        time_reduction=0.05,
    ):
        self.logger = logging.getLogger("mcts_with_lookup")
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
        self._lookup_table = self.read_lookup_table()

    def act(self, observation, step=0, initialize=False):
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

        if (self._step <= limit) or initialize:
            self._mcts.initialize(board, own_player)

        pos = self.lookup_position(board)
        if pos is None:
            action = self._mcts.search(board, own_player, self._step, deadline, True)
        else:
            action = np.argmax(pos)
        return int(action)

    def read_lookup_table(self):
        lookup_table = "resources/lookup_table/connectx-state-action-value.txt"
        # state_values = pd.read_csv(lookup_table, delimiter=",", header=None, names=['position', '0', '1', '2', '3', '4', '5', '6'], dtype={'position': str,
        # '0': str, '1': str, '2': str, '3': str, '4': str, '5': str, '6': str})
        csv_file = open(lookup_table)
        csv_reader = csv.reader(csv_file, delimiter=',')

        return csv_reader

    def lookup_position(self, position):
        pos = ''.join([str(c) for c in position])
        try:
            item = next(self._lookup_table)
        except StopIteration:
            return None
            
        while item[0] != pos:
            try:
                item = next(self._lookup_table)
            except StopIteration:
                item = None
                break

        if item is not None:
            return [-100 if r == '' else int(r) for r in item[1:]]
        else:
            return None

    def MCTS(self):
        return self._mcts

    @staticmethod
    def get_instance(configuration, mcts: BaseMonteCarloTreeSearch):
        if MCTSWithLookupAgent._agent is None:
            MCTSWithLookupAgent._agent = MCTSWithLookupAgent(configuration, mcts)
        return MCTSWithLookupAgent._agent
