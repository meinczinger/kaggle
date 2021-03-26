import numpy as np
from collections import defaultdict, Counter
from bitboard import BitBoard


class StateAction:
    def __init__(self, epsilon):
        self._values = None
        self._counts = None
        self._epoch = []
        self._epsilon = epsilon
        self._counts = defaultdict(Counter)
        self._values = defaultdict(defaultdict)

    def _actions(self, state):
        return [action for action in self._values[state]]

    """ Find best action according to greedy policy """
    def best_action(self, state: BitBoard, possible_actions, ignore_epsilon=False):
        state_hash = state.hash()
        if len(self._actions(state_hash)) == 0:
            random_action = np.random.choice(possible_actions)
            # Set optimistic initial values for the other actions in this state
            for a in possible_actions:
                if a != random_action:
                    self._values[state_hash][a] = 0.0
            return random_action
        else:
            if not ignore_epsilon and (np.random.rand() <= self._epsilon):
                return np.random.choice(possible_actions)
            else:
                max_value = max([self._values[state_hash][a] for a in self._values[state_hash]])
                return np.random.choice([a for a in self._values[state_hash]
                                         if self._values[state_hash][a] >= max_value])

    """ Add step to epoch """
    def add(self, state: BitBoard, action, reward, prev_state, prev_action, prev_reward):
        self._epoch.append({'state': state, 'action': action, 'reward': reward, 'prev_state': prev_state,
                            'prev_action': prev_action, 'prev_reward': prev_reward})
        # Increase the count of how often this state/action combination was visited
        self._counts[state.hash()][action] += 1

    """ Get the number of times an action was visited in a given state """
    def get_action_selected_count(self, state: BitBoard, action):
        return self._counts[state.hash()][action]

    """ Get state-action value """
    def get_value(self, state: BitBoard, action):
        try:
            return self._values[state.hash()][action]
        except:
            return 0

    """ Set state-action value """
    def set_value(self, state: BitBoard, action, value):
        self._values[state.hash()][action] = value
        self._values[state.swap_marks_board()][action] = value

    def update(self, process_epoch):
        u = process_epoch(self._epoch)

        if u is not None:
            for state, value in u.items():
                for action, _ in value.items():
                    self._values[state.hash()][action] = 0.9 * (self.get_value(state, action) + u[state][action])
