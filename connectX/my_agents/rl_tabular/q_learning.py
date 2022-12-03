from my_agents.rl_agent import RLAgent


class TabularQLeaerningAgent(RLAgent):
    _agent = None
    ALPHA = 0.1
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
            max_value = \
                self._state_action.get_value(state,
                                             self._state_action.best_action(state, state.possible_actions(), True))
            self._state_action.set_value(prev_state, prev_action,
                                         self._state_action.get_value(prev_state, prev_action) +
                                         self.ALPHA * (prev_reward + self.GAMMA * max_value -
                                                       self._state_action.get_value(prev_state, prev_action)))

    def process_epoch(self, epoch):
        pass

    @staticmethod
    def pickle_name():
        return "qlearning_tabular.pickle"

    @staticmethod
    def get_instance(configuration):
        if TabularQLeaerningAgent._agent is None:
            TabularQLeaerningAgent._agent = TabularQLeaerningAgent(configuration)
        return TabularQLeaerningAgent._agent
