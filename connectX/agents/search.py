from agents.bitboard import BitBoard
import copy


class MiniMax:
    def __init__(self):
        self.own_player = 1

    def search(self, state: BitBoard, player, depth):
        self.own_player = player
        return self._alpha_beta(state, depth)

    @staticmethod
    def _execute_action(state, action):
        """ Executes an action. State is mutable so needs to be copied """
        astate = copy.copy(state)
        astate.make_action(action)

        return astate

    def _alpha_beta(self, state, depth):
        alpha = -float("inf")
        beta = float("inf")
        return max(state.possible_actions(), key=lambda x: self._beta(self._execute_action(state, x),
                                                                      depth - 1, alpha, beta))

    def _alpha(self, state, depth, alpha, beta):
        if state.is_terminal_state():
            return self._utility(state)
        if depth <= 0:
            return self._score(state)
        value = float("-inf")
        for action in state.possible_actions():
            value = max(value,
                        self._beta(self._execute_action(state, action), depth - 1, alpha, beta))
            if value >= beta:
                return value
            alpha = max(alpha, value)

        return value

    def _beta(self, state, depth, alpha, beta):
        if state.is_terminal_state():
            return self._utility(state)
        if depth <= 0:
            return self._score(state)
        value = float("inf")
        for action in state.possible_actions():
            value = min(value,
                        self._alpha(self._execute_action(state, action), depth - 1, alpha, beta))
            if value <= alpha:
                return value
            beta = min(beta, value)
        return value

    """ 
    Utility function
    We reached the end state, if the active player is the own player, it's a win, otherwise a loose
    """
    def _utility(self, state):
        if state.active_player() != self.own_player:
            return float("inf")
        else:
            return -float("inf")

    """ Very simplistic heuristic """
    def _score(self, state):
        """ Simple heuristic for non-end states """
        top_marks = state.top_marks()
        return top_marks[self.own_player] - top_marks[self._opponent()]

    """ Change player """
    def _opponent(self):
        return self.own_player % 2 + 1
