from state import State


class Checker:
    _checker = None

    def __init__(self, configuration):
        self.columns = configuration.columns
        self.rows = configuration.rows
        self.inarow = configuration.inarow
        self.episodeSteps = configuration.episodeSteps
        self.actTimeout = configuration.actTimeout
        self.my_player = 1
        self._state = State(self.columns, self.rows, [1, 2], self.inarow)
        self._prev_board = None

    @staticmethod
    def execute_action(state, action, mark):
        """ Executes an action. State is mutable so needs to be copied """
        astate = state.copy()

        astate.make_action(action, mark)
        return astate

    def act(self, observation):
        """ Main method to act on opponents move """
        board = observation["board"]
        self.my_player = observation["mark"]

        state = State.state_from_board(board, self.columns, self.rows, [1, 2], self.inarow, self.my_player)

        # Run minimax with alpha-beta pruning, with depth 6
        action = self.alpha_beta(state, 6)

        return action

    def my_player(self):
        return self.my_player

    def opponent(self):
        return self.my_player % 2 + 1

    def alpha_beta(self, state, depth):
        alpha = -float("inf")
        beta = float("inf")
        return max(state.possible_actions(),
                   key=lambda x: self.beta(self.execute_action(state, x, state.active_player()), depth - 1, alpha, beta))

    def alpha(self, state, depth, alpha, beta):
        if state.is_terminal_state():
            return self.utility(state)
        if depth <= 0:
            return self.score(state)
        value = float("-inf")
        for action in state.possible_actions():
            value = max(value,
                        self.beta(self.execute_action(state, action, state.active_player()), depth - 1, alpha, beta))
            if value >= beta:
                return value
            alpha = max(alpha, value)

        return value

    def beta(self, state, depth, alpha, beta):
        if state.is_terminal_state():
            return self.utility(state)
        if depth <= 0:
            return self.score(state)
        value = float("inf")
        for action in state.possible_actions():
            value = min(value,
                        self.alpha(self.execute_action(state, action, state.active_player()), depth - 1, alpha, beta))
            if value <= alpha:
                return value
            beta = min(beta, value)
        return value

    def utility(self, state):
        if state.active_player() != self.my_player:
            return float("inf")
        else:
            return -float("inf")

    def score(self, state):
        """ Simple heuristic for non-end states """
        top_marks = state.top_marks()
        return top_marks[self.my_player] - top_marks[self.opponent()]

    @staticmethod
    def get_instance(configuration):
        if Checker._checker is None:
            Checker._checker = Checker(configuration)
        return Checker._checker


def act(observation, configuration):
    return Checker.get_instance(configuration).act(observation)



