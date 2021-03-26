import numpy as np
from kaggle_environments.utils import Struct
from bitboard import BitBoard


class Simulator:
    def __init__(self, configuration, agent1=None, agent2=None):
        """ Simulates games
        If no agent is given, it does random simulation, otherwise it uses the agent(s)
        """
        self._config = configuration
        self._agents = []
        self.agents = [agent1, agent2]

    def simulate(self, board: BitBoard, to_play: int) -> int:
        """
        Runs a simulation from the given board
        :param board: the current board position
        :param to_play: the mark of the player who is next to play
        :return: 0 in case of a tie, 1 if the player wins, who plays next, -1 otherwise
        """
        obs = Struct()
        marks = [to_play, (to_play + 1) % 2]
        ply = (to_play + 1) % 2
        obs.step = 0
        # Execute moves before we reach a terminal state
        while not board.is_terminal_state():
            obs.mark = marks[ply]
            if self.agents[ply] is None:
                # Get a random move
                action = np.random.choice(board.possible_actions())
            else:
                # Get a move given by the agent
                action = self.agents[ply].act(obs, board)
            # Make the move
            board.make_action(action)
            # Swap players
            ply = 1 - ply
            # Increase the number of steps
            obs.step += 1
        if board.is_draw():
            return 0
        else:
            return board.last_player()

