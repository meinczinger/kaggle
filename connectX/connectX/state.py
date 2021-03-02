from collections import defaultdict
from enum import Enum


class Column:
    """ Class representing a column (treated as a stack) """
    def __init__(self, height, marks: [int], mark_in_row):
        self._height = height
        self._marks = marks
        self._mark_in_row = mark_in_row
        self._column = ''

    def column(self):
        return self._column

    def adjust(self, mark):
        """ Adjust the column to reflect an action """
        self._column += str(mark)

    def fill_level(self):
        """ How many marks are in the column. Needed to decide whether there is a possible action for this column """
        return len(self._column)

    def is_winning(self):
        """ Are there _mark_in_row marks in the column """
        return any([str(m)*self._mark_in_row in self._column for m in self._marks])


class Row:
    """ Class representing a row """
    def __init__(self, width, marks: [int], mark_in_row):
        self._width = width
        self._marks = marks
        self._mark_in_row = mark_in_row

        self._row = '0' * self._width

    def adjust(self, pos, mark):
        """ Adjust the row to reflect an action """
        self._row = self._row[:pos] + str(mark) + self._row[pos+1:]

    def row(self):
        return self._row

    def is_winning(self):
        """ Are there _mark_in_row marks in the row """
        return any([str(m)*self._mark_in_row in self._row for m in self._marks])


class Orientation(Enum):
    """ Used for the diagonals"""
    RIGHT = 0
    LEFT = 1


class Diagonal:
    """ Class representing a diagonal """
    def __init__(self, length, marks: [int], mark_in_row, orientation: Orientation):
        self._length = length
        self._marks = marks
        self._mark_in_row = mark_in_row
        self._diagonal = '0' * self._length
        self._orientation = orientation

    def diagonal(self):
        return self._diagonal

    def adjust(self, mark, pos):
        """ Adjust the diagonal to reflect an action """
        self._diagonal = self._diagonal[:pos] + str(mark) + self._diagonal[pos + 1:]

    def is_winning(self):
        """ Are there _mark_in_row marks in the diagonal """
        return any([str(m)*self._mark_in_row in self._diagonal for m in self._marks])


class Diagonals:
    """ Class representing diagonals
    TODO: Find a better representation
    """
    def __init__(self, width, height, mark_in_row, marks):
        self._width = width
        self._height = height
        self._mark_in_row = mark_in_row
        self._marks = marks
        self._diagonals = self._init_diagonals()

    """ Custom copy to avoid deepcopy """
    def copy(self):
        d = Diagonals(self._width, self._height, self._mark_in_row, self._marks)
        for r, _ in d._diagonals.items():
            d._diagonals[r]._diagonal = self._diagonals[r].diagonal()
        return d

    """ Initial all potential diagonals """
    def _init_diagonals(self):
        d = defaultdict()
        # Lower part
        for i in range(0, self._width):
            d[(i, 0, Orientation.RIGHT)] = \
                Diagonal(min(self._width, self._height), self._marks, self._mark_in_row, Orientation.RIGHT)
            d[(i, 0, Orientation.LEFT)] = \
                Diagonal(min(self._width, self._height), self._marks, self._mark_in_row, Orientation.LEFT)
        # Upper part
        for i in range(1, self._height):
            d[(0, i, Orientation.RIGHT)] = \
                Diagonal(min(self._width, self._height), self._marks, self._mark_in_row, Orientation.RIGHT)
            d[(self._width - 1, i, Orientation.LEFT)] = \
                Diagonal(min(self._width, self._height), self._marks, self._mark_in_row, Orientation.LEFT)

        return d

    def diagonal_origins(self, x, y):
        """ For a given point, get the originals of diagonals passing through this point """
        origins = []
        if x - y >= 0:
            origins.append([x - y, 0, Orientation.RIGHT])
        else:
            origins.append([0, y - x, Orientation.RIGHT])

        if x + y < self._width:
            origins.append([x + y, 0, Orientation.LEFT])
        else:
            origins.append([self._width - 1, (x + y) - self._width + 1, Orientation.LEFT])

        return origins

    def diagonal(self, x, y, orientation):
        return self._diagonals[(x, y, orientation)]

    def diagonals(self):
        return self._diagonals

    def update_diagonal(self, x, y, mark):
        """ Update the diagonals which contains the point x, y """
        origins = self.diagonal_origins(x, y)
        for o in origins:
            self._diagonals[(o[0], o[1], o[2])].adjust(mark, y - o[1])


class State:
    def __init__(self, width, height, marks: [int], mark_in_row, player=1):
        self._marks = marks
        self._width = width
        self._height = height
        self._mark_in_row = mark_in_row
        self._player = player
        self._rows = {key: Row(self._width, self._marks, self._mark_in_row) for key in range(self._height)}
        self._columns = {key: Column(self._height, self._marks, self._mark_in_row) for key in range(self._width)}
        self._diagonals = Diagonals(self._width, self._height
                                    , self._mark_in_row, self._marks)

    @staticmethod
    def state_from_board(board, width, height, marks: [int], mark_in_row, player):
        state = State(width, height, marks, mark_in_row, player)
        for i in range(len(board)):
            pos = len(board) - i - 1
            if board[pos] != 0:
                state.make_action(pos % width, board[pos])

        state._player = player
        return state

    def copy(self):
        state = State(self._width, self._height, self._marks, self._mark_in_row, self._player)
        for r, _ in state._rows.items():
            state._rows[r]._row = self._rows[r].row()
        for c, _ in state._columns.items():
            state._columns[c]._column = self._columns[c].column()
        #state._columns = self._columns.copy()
        state._diagonals = self._diagonals.copy()
        return state

    def width(self):
        return self._width

    def possible_actions(self):
        """ Returns the indexes of columns, which are not yet full """
        return [k for k, c in self._columns.items() if c.fill_level() < self._height]

    def swap_player(self):
        self._player = self._player % 2 + 1

    def make_action(self, action, mark):
        # First remember which row we will have to update
        row_to_update = self._columns[action].fill_level()
        # Update the column
        self._columns[action].adjust(mark)
        # Update the corresponding row
        self._rows[row_to_update].adjust(action, mark)
        # Update the corresponding diagonals
        self._diagonals.update_diagonal(action, row_to_update, mark)
        # Make the other player the active player
        self.swap_player()

    def columns(self):
        return [c.column() for _, c in self._columns.items()]

    def rows(self):
        return [r.row() for _, r in self._rows.items()]

    def diagonals(self):
        return [k for k, d in self._diagonals.diagonals().items()]

    def diagonal(self, x, y, orientation):
        return self._diagonals.diagonal(x, y, orientation).diagonal()

    def is_terminal_state(self):
        return any([c.is_winning() for _, c in self._columns.items()]) or \
               any([r.is_winning() for _, r in self._rows.items()]) or \
               any([d.is_winning() for _, d in self._diagonals.diagonals().items()])

    def active_player(self):
        return self._player

    @staticmethod
    def opponent(player):
        return player % 2 + 1

    def top_marks(self):
        return {mark: max(0, sum([1 for _, c in self._columns.items()
                                  if len(c.column()) > 0 and c.column()[-1] == str(mark)]))
                for mark in self._marks}


