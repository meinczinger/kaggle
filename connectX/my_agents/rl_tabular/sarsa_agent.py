from my_agents.rl_agent import RLAgent
from bitboard import BitBoard
import logging
import copy
import pickle


logging.basicConfig(filename='error.log')


class TabularSarsaAgent(RLAgent):
    _agent = None
    ALPHA = 0.5
    GAMMA = 0.9

    def __init__(self, configuration):
        super().__init__(configuration)
        self._prev_action = 0
        self._prev_state = None

    def step(self, state, action, reward, end_state, prev_state, prev_action, prev_reward):
        if prev_state is not None:
            if end_state:
                self._state_action.set_value(state, action, self._state_action.get_value(state, action) +
                                             self.ALPHA * (reward - self._state_action.get_value(state, action)))

            # Get the highest value for the state
            max_value = self._state_action.get_value(state, action)
            self._state_action.set_value(prev_state, prev_action,
                                         self._state_action.get_value(prev_state, prev_action) +
                                         self.ALPHA * (prev_reward + self.GAMMA * max_value -
                                                       self._state_action.get_value(prev_state, prev_action)))

    @staticmethod
    def pickle_name():
        return "sarsa_tabular.pickle"

    def process_epoch(self, epoch):
        pass

    @staticmethod
    def get_instance(configuration):
        if TabularSarsaAgent._agent is None:
            TabularSarsaAgent._agent = TabularSarsaAgent(configuration)
        return TabularSarsaAgent._agent
