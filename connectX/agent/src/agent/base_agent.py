from mcts.base_mcts import BaseMonteCarloTreeSearch
import logging
import time
import numpy as np


class BaseAgent:
    _agent = None

    def __init__(
        self,
        configuration,
        mcts: BaseMonteCarloTreeSearch,
        self_play: bool = False,
        time_reduction: float = 0.05,
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

    def act(self, observation, step=0, initialize=False):
        """Main method to act on opponents move"""
        return 0

    def MCTS(self):
        return self._mcts

    @staticmethod
    def get_instance(configuration, mcts: BaseMonteCarloTreeSearch):
        if BaseAgent._agent is None:
            BaseAgent._agent = BaseAgent(configuration, mcts)
        return BaseAgent._agent
