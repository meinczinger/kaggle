import numpy as np
from typing import Tuple, List, Union


class GameOvers:
    _game_overs = None

    def __init__(self, width, height, mark_in_row):
        self._width = width
        self._height = height
        self._mark_in_row = mark_in_row
        self._game_overs = None

    def _get_game_overs(self) -> np.ndarray:
        """Creates a set of all winning board positions, over 4 directions: horizontal, vertical and 2 diagonals"""
        game_overs = []

        mask_horizontal = 0
        mask_vertical = 0
        mask_diagonal_dl = 0
        mask_diagonal_ul = 0
        for n in range(self._mark_in_row):  # use prange() with numba(parallel=True)
            mask_horizontal |= 1 << n
            mask_vertical |= 1 << n * self._width
            mask_diagonal_dl |= 1 << n * self._width + n
            mask_diagonal_ul |= 1 << n * self._width + (self._mark_in_row - 1 - n)

        row_inner = self._height - self._mark_in_row
        col_inner = self._width - self._mark_in_row
        for row in range(self._height):         # use prange() with numba(parallel=True)
            for col in range(self._width):  # use prange() with numba(parallel=True)
                offset = col + row * self._width
                if col <= col_inner:
                    game_overs.append( mask_horizontal << offset )
                if row <= row_inner:
                    game_overs.append( mask_vertical << offset )
                if col <= col_inner and row <= row_inner:
                    game_overs.append( mask_diagonal_dl << offset )
                    game_overs.append( mask_diagonal_ul << offset )

        return np.array(game_overs)

    @staticmethod
    def get_instance(width, height, mark_in_row):
        if GameOvers._game_overs is None:
            GameOvers._game_overs = GameOvers(width, height, mark_in_row)
            GameOvers._game_overs._game_overs = GameOvers._game_overs._get_game_overs()
        return GameOvers._game_overs._game_overs


class BitBoard:
    def __init__(self, width, height, mark_in_row, player, bit_board: np.ndarray):
        self._width = width
        self._height = height
        self._mark_in_row = mark_in_row
        self._bitboard = bit_board
        # Payer: 1->0, 2->1
        self._player = (player + 1) % 2
        self._mask_board = (np.int64(1) << self._width * self._height) - 1
        self._mask_legal_moves = (np.int64(1) << self._width) - 1
        self._end_state = False
        self._draw = False
        self._game_overs = GameOvers.get_instance(self._width, self._height, self._mark_in_row)
        self._top_marks = {}
        self._set_top_marks()
        self._matrix = None
        self._last_action = 0

    """ Create from a kaggle board """
    @classmethod
    def create_from_board(cls, width, height, mark_in_row, player, board: list):
        return cls(width, height, mark_in_row, player, BitBoard.list_to_bitboard(board))

    """ Clone from bitboard[0], bitboard[1] """
    @classmethod
    def create_from_bitboard(cls, width, height, mark_in_row, player, bitboard):
        return cls(width, height, mark_in_row, player, bitboard)

    """ Create an empty board """
    @classmethod
    def create_empty_board(cls, width, height, mark_in_row, player):
        return cls(width, height, mark_in_row, player, np.array([0, 0], dtype=np.int64))

    """ Copy a BitBoard instance """
    def __copy__(self):
        bb = BitBoard(self._width, self._height, self._mark_in_row, self._player + 1, np.copy(self._bitboard))
        bb._top_marks = self._top_marks.copy()
        return bb

    """ Convert the kaggle list-board to bitboard """
    @staticmethod
    def list_to_bitboard(board: Union[np.ndarray, List[int]]) -> np.ndarray:
        # bitboard[0] = played, is a square filled             | 0 = empty, 1 = filled
        # bitboard[1] = player, who's token is this, if filled | 0 = empty, 1 = filled
        bitboard_played = np.int64(0)  # 42 bit number for if board square has been played
        bitboard_player = np.int64(0)  # 42 bit number for player 0=p1 1=p2
        if isinstance(board, np.ndarray):
            board = board.flatten()
        for n in range(len(board)):  # prange
            if board[n] != 0:
                bitboard_played |= (1 << n)        # is a square filled (0 = empty | 1 = filled)
                if board[n] == 2:
                    bitboard_player |= (1 << n)    # mark as player 2 square, else assume p1=0 as default
        bitboard = np.array([bitboard_played, bitboard_player], dtype=np.int64)
        return bitboard

    """ Convert bitboard back to list-based board """
    def to_list(self):
        board = []
        for n in range(self._width * self._height):
            is_played = (self._bitboard[0] >> n) & 1
            if is_played:
                player = (self._bitboard[1] >> n) & 1
                board.append(int(player + 1))
            else:
                board.append(int(0))
        return board

    """ Convert bitboard back to list-based board """
    @staticmethod
    def bitboard_to_list(columns, rows, bitboard):
        board = []
        for n in range(columns * rows):
            is_played = (bitboard[0] >> n) & 1
            if is_played:
                player = (bitboard[1] >> n) & 1
                board.append(int(player + 1))
            else:
                board.append(int(0))
        return board

    def bitboard_to_numpy(self) -> np.ndarray:
        global configuration
        rows = self._height
        columns = self._width
        size = rows * columns
        output = np.zeros((size,), dtype=np.int8)
        for i in range(size):  # prange
            is_played = (self._bitboard[0] >> i) & 1
            if is_played:
                player = (self._bitboard[1] >> i) & 1
                output[i] = 1 if player == 0 else 2
        return output.reshape((rows, columns))

    @staticmethod
    def bitboard_to_numpy2d(bitboard, columns=7, rows=6) -> np.ndarray:
        global configuration
        size = rows * columns
        output = np.zeros((size,), dtype=np.int8)
        for i in range(size):  # prange
            is_played = (bitboard[0] >> i) & 1
            if is_played:
                player = (bitboard[1] >> i) & 1
                output[i] = 1 if player == 0 else 2
        return output.reshape((rows, columns))

    @staticmethod
    def board_channels(numpy_data):
        channels = np.zeros((1, 6, 7, 2))
        channels[:, :, :, 0] = np.where(numpy_data == 1, 1, 0)
        channels[:, :, :, 1] = np.where(numpy_data == 2, 1, 0)
        return channels

    """ Convert bitboard back to list-based board """
    @staticmethod
    def from_bitboard_to_list(width, height, bitboard):
        board = []
        for n in range(width * height):
            is_played = (bitboard[0] >> n) & 1
            if is_played:
                player = (bitboard[1] >> n) & 1
                board.append(int(player + 1))
            else:
                board.append(int(0))
        return board

    """ Convert bitboard back to list-based board """
    def to_str(self):
        board = ""
        for n in range(self._width * self._height):
            is_played = (self._bitboard[0] >> n) & 1
            if is_played:
                player = (self._bitboard[1] >> n) & 1
                board += str(player + 1)
            else:
                board += '0'
        return board

    """ Check if the state is a draw """
    def is_draw(self) -> bool:
        """If all the squares on the top row have been played, then there are no more moves"""
        return self._bitboard[0] & self._mask_legal_moves == self._mask_legal_moves

    """ Returns the possible actions for the board """
    def possible_actions(self):
        """ Returns the indexes of columns, which are not yet full """
        possible_actions = self._bitboard[0] & self._mask_legal_moves
        return [i for i in range(self._width) if not ((possible_actions >> i) & 1)]

    """ Find the first empty row so that the new piece can be placed there """
    def _first_empty_row_for_column(self, column):
        for i in range(self._height - 1, -1, -1):
            mask_column = (np.int64(1) << column) << (i * self._width)
            if not (self._bitboard[0] & mask_column):
                return i

    """" Check if the board is an end state """
    def _is_end_state(self):
        return np.any((self._bitboard[0] & self._bitboard[1]) & self._game_overs[:] == self._game_overs[:]) or \
               np.any((self._bitboard[0] & ~self._bitboard[1]) & self._game_overs[:] == self._game_overs[:])
        # return any([g for g in self._game_overs if (self._bitboard[0] & self._bitboard[1]) & g == g or
        #             (self._bitboard[0] & ~self._bitboard[1]) & g == g])

    """ Perform the given action on the board """
    def make_action(self, column):
        # find the lowest mark, which is not yet marked in the given column
        row = self._first_empty_row_for_column(column)
        new_mark_mask = np.int64(1) << (row * self._width + column)
        self._bitboard[0] |= new_mark_mask
        if self._player:
            self._bitboard[1] |= new_mark_mask
        # If we reached an end-state:
        self._end_state = self._is_end_state()
        # Check if there is a draw
        self._draw = self.is_draw()

        self._top_marks[column] = self._player + 1

        # Make the other player the active player
        self._swap_player()

        self._last_action = column

    """ After a move was done, swap the player """
    def _swap_player(self):
        self._player = (self._player + 1) % 2

    """" Who is the active player (the player who moves next) """
    def active_player(self):
        return self._player + 1

    """ Get the last player """
    def last_player(self):
        return (1 - self._player) + 1

    """ Returns whether the board is a terminal state """
    def is_terminal_state(self) -> bool:
        return self._end_state or self._draw

    """ What are the top-marks - for a trivial heuristic """
    def top_marks(self):
        d = dict()
        d[1] = sum([1 for k, v in self._top_marks.items() if v == 1])
        d[2] = sum([1 for k, v in self._top_marks.items() if v == 2])
        return d

    """ Set the top marks after a move """
    def _set_top_marks(self):
        for n in range(self._width * self._height -1, -1, -1):
            mask = np.int64(1) << n
            if self._bitboard[0] & mask:
                if self._bitboard[1] & mask:
                    player = 2
                else:
                    player = 1
                self._top_marks[n % self._width] = player

    """ Get the two bitboards as a hash, this is used when bitboards are stored in a dictionary """
    def hash(self):
        return self._bitboard[0], self._bitboard[1]

    """ Get the last action """
    def last_action(self):
        return self._last_action

    """ After the opponent's move, compare boards to find out what his move was """
    def set_last_action(self, board):
        action_bit = self._bitboard[0] ^ board._bitboard[0]
        masks = []
        for c in range(self._width):
            masks.append(1 << c)
            for i in range(1, self._width):
                masks[c] = (np.int64(1) << (self._width * i + c)) | masks[c]
        for c in masks:
            if (action_bit & c) > 0:
                self._last_action = masks.index(c)
                break

    """ Swap the marks on the board """
    def swap_marks_board(self):
        return self._bitboard[0], ~ self._bitboard[1]

    def mirror_board(self):
        # Get mirrored board
        board = self.bitboard_to_numpy()
        board = np.flip(board, 1)
        return list(board.reshape(self._width * self._height,))

    def channels(self):
        np_boards = np.array(self.bitboard_to_numpy())
        np_boards = np_boards.reshape((1, 6, 7))
        np_boards_channels = np.zeros((1, 6, 7, 2))
        np_boards_channels[:, :, :, 0] = np.where(np_boards == 1, 1, 0)
        np_boards_channels[:, :, :, 1] = np.where(np_boards == 2, 1, 0)
        return np_boards_channels
