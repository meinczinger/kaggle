from agents.rl_tabular.tabular_base import TabularBase
import os


# PICKLE_NAME = "/kaggle_simulations/agent/resources/tabular/mc_tabular.pickle"
PICKLE_NAME = "resources/tabular/mc_tabular.pickle"


class TabularMonteCarlo(TabularBase):
    _agent = None
    GAMMA = 0.9

    def __init__(self, configuration):
        super().__init__(configuration)

    def step(
        self, state, action, reward, end_state, prev_state, prev_action, prev_reward
    ):
        # if not end_state:
        self._state_action.add(
            state, action, reward, prev_state, prev_action, prev_reward
        )

    def process_epoch(self, epoch):
        value = 0
        for i in range(len(epoch) - 1, -1, -1):
            state = epoch[i]["state"]
            action = epoch[i]["action"]
            reward = epoch[i]["reward"]

            value = self.GAMMA * value + reward
            # In order to avoid storing all returns, update the average directly, for
            # that we need the current state-action value
            prev_value = self._state_action.get_value(state, action)
            # How often this state/action combination was visited
            action_count = self._state_action.get_action_selected_count(state, action)

            # Update the new state/action values
            self._state_action.set_value(
                state,
                action,
                (action_count - 1.0) / action_count * prev_value
                + value / float(action_count),
            )

    @staticmethod
    def pickle_name():
        return PICKLE_NAME
