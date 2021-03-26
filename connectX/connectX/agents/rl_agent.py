from state_action import StateAction
from bitboard import BitBoard
import copy
import pickle


EPSILON = 0.05
REWARD_WIN = 1.0
REWARD_DRAW = 0.0
REWARD_LOOSE = -1.0
REWARD_STEP = 0.0
MARKS = [1, 2]


class RLAgent:
    def __init__(self, configuration):
        self.columns = configuration.columns
        self.rows = configuration.rows
        self.inarow = configuration.inarow
        self.episodeSteps = configuration.episodeSteps
        self.actTimeout = configuration.actTimeout
        self.my_player = 1
        self._prev_board = None
        self._state_action = None
        self.read_pickle()
        self._end_state = False
        self._prev_state = None
        self._prev_action = 0
        self._prev_reward = 0
        self._act_count = 0

    def _initialize(self):
        self._end_state = False
        self._prev_state = None
        self._prev_action = 0
        self._prev_reward = 0

    def act(self, observation, sim_board=None):
        reward = 0

        """ Main method to act on opponents move """
        if observation.step <= 1:
            self._initialize()

        # Set my player
        self.my_player = observation.mark

        # Create a state from the board
        if sim_board is None:
            board = observation.board
            state = BitBoard.create_from_board(self.columns, self.rows, self.inarow, self.my_player, board)
        else:
            state = sim_board

        # Get action according to policy
        action = self.get_action(state)

        # Make action
        new_state = copy.copy(state)
        new_state.make_action(action)

        if not self._end_state:
            # Get reward
            reward, end_state = self.get_reward(new_state)

            # Perform step
            self.step(state, action, reward, end_state, self._prev_state, self._prev_action, self._prev_reward)

            # We reached the end state
            if end_state:
                # Perform actions after an epoch has finished
                self.epoch_finished(self._state_action)
                self._end_state = True

        self._prev_state = state
        self._prev_action = action
        self._prev_reward = reward

        self._act_count += 1

        if self._act_count % 1000 == 0:
            self.write_pickle()

        return int(action)

    """ Get action according to greedy policy """
    def get_action(self, state):
        action = self._state_action.best_action(state, state.possible_actions())
        return action

    def epoch_finished(self, state_action):
        state_action.update(self.process_epoch)

    def step(self, state, action, reward, end_state, prev_state, prev_action, prev_reward):
        raise NotImplementedError

    """ Process the epoch when the epoch has finished (e.g. Monte-Carlo tabular) """
    def process_epoch(self, epoch):
        raise NotImplementedError

    @staticmethod
    def pickle_name():
        raise NotImplementedError

    def write_pickle(self):
        with open(self.pickle_name(), 'wb') as f:
            pickle.dump(self._state_action, f)

    def read_pickle(self):
        try:
            with open(self.pickle_name(), "rb") as f:
                self._state_action = pickle.load(f)
        except (IOError, TypeError) as e:
            self._state_action = StateAction(EPSILON)

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

    @staticmethod
    def get_instance(configuration):
        raise NotImplementedError
