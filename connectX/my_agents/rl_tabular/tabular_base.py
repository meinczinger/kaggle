from turtle import st
from my_agents.rl_tabular.state_action import StateAction
from my_agents.bitboard import BitBoard
import copy
import pickle
import logging


EPSILON = 0.05
REWARD_WIN = 1.0
REWARD_DRAW = 0.0
REWARD_LOOSE = -1.0
REWARD_STEP = 0.0
MARKS = [1, 2]


class TabularBase:
    def __init__(self, configuration):
        self.columns = configuration.columns
        self.rows = configuration.rows
        self.inarow = configuration.inarow
        self.episodeSteps = configuration.episodeSteps
        self.actTimeout = configuration.actTimeout
        self.my_player = 1
        self._prev_board = None
        self._state_action = None
        self._logger = logging.getLogger("TabularBase")
        self._logger.setLevel(logging.DEBUG)
        self._logger.addHandler(logging.StreamHandler())
        self.read_pickle()
        self._end_state = False
        self._prev_state = None
        self._prev_action = 0
        self._prev_reward = 0

    def _initialize(self):
        self._end_state = False
        self._prev_state = None
        self._prev_action = 0
        self._prev_reward = 0
        # self.read_pickle()

    def post_action(self, state, action):
        # Make action
        new_state = copy.copy(state)
        new_state.make_action(action)

        reward, self._end_state = self.get_reward(new_state)

        if not self._end_state:
            # Perform step
            self.step(
                state,
                action,
                reward,
                self._end_state,
                self._prev_state,
                self._prev_action,
                self._prev_reward,
            )
        else:
            # We reached the end state
            # Perform actions after an epoch has finished
            self.epoch_finished(self._state_action)
            self._end_state = True
            self.write_pickle()

        self._prev_state = state
        self._prev_action = action
        self._prev_reward = reward

        return int(action)

    """ Get action according to greedy policy """

    def get_action(self, state):
        action = self._state_action.best_action(state, state.possible_actions())
        return action

    def epoch_finished(self, state_action):
        state_action.update(self.process_epoch)
        state_action.clear_epoch()

    def step(
        self, state, action, reward, end_state, prev_state, prev_action, prev_reward
    ):
        raise NotImplementedError

    """ Process the epoch when the epoch has finished (e.g. Monte-Carlo tabular) """

    def process_epoch(self, epoch):
        raise NotImplementedError

    @staticmethod
    def pickle_name():
        raise NotImplementedError

    def write_pickle(self):
        with open(self.pickle_name(), "wb") as f:
            pickle.dump(self._state_action, f)

    def read_pickle(self):
        try:
            with open(self.pickle_name(), "rb") as f:
                self._state_action = pickle.load(f)
                # self._logger.info(
                #     f"pickle size: {self._state_action.get_state_value_size()}"
                # )
                print(f"pickle size: {self._state_action.get_state_value_size()}")
        except (IOError, TypeError) as e:
            self._state_action = StateAction(EPSILON)

    def get_state_value_size(self):
        return self._state_action.get_state_value_size()

    """ Get state value """

    def get_state_value(self, state):
        return self._state_action.get_state_value(state)

    """ Get the reward """

    def get_reward(self, state):
        reward = REWARD_STEP
        end_state = False

        # Check if end state (win or draw)
        if state.is_terminal_state():
            end_state = True
            if state.is_draw():
                reward = REWARD_DRAW
            else:
                reward = REWARD_WIN
        if not end_state:
            # Simulate opponents move
            res = self.opponents_move(state)
            if res == 1:
                end_state = True
                reward = REWARD_LOOSE  # The oppnenent wins
            else:
                if res == 0:
                    end_state = True
                    reward = REWARD_DRAW  # Draw
        return reward, end_state

    """ Mark of the opponent """

    @staticmethod
    def opponent(player):
        return player % 2 + 1

    """ Check if some of the opponents move would cause the game to end """

    @staticmethod
    def opponents_move(state):
        for action in state.possible_actions():
            new_state = copy.copy(state)
            new_state.make_action(action)
            if new_state.is_terminal_state():
                if new_state.is_draw():
                    return 0
                else:
                    return 1
        return -1
