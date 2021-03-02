import unittest
from state import State, Orientation


class TestState(unittest.TestCase):
    def test_state_basic(self):
        s = State(4, 3, [1, 2], 3, 1)
        s.make_action(2, 1)
        self.assertListEqual(['', '', '1', ''], s.columns(), "Columns incorrect after first move")
        self.assertListEqual(['0010', '0000', '0000'], s.rows(), "Rows incorrect after first move")

    def test_state_two_moves(self):
        s = State(4, 3, [1, 2], 3, 1)
        s.make_action(2, 1)
        s.make_action(0, 2)
        self.assertListEqual(['2', '', '1', ''], s.columns(), "Columns incorrect after first move")
        self.assertListEqual(['2010', '0000', '0000'], s.rows(), "Rows incorrect after first move")

    def test_state_higher_row(self):
        s = State(4, 3, [1, 2], 3, 1)
        s.make_action(2, 1)
        s.make_action(0, 2)
        s.make_action(2, 1)
        self.assertListEqual(['2', '', '11', ''], s.columns(), "Columns incorrect after first move")
        self.assertListEqual(['2010', '0010', '0000'], s.rows(), "Rows incorrect after first move")

    def test_end_state_in_column(self):
        s = State(4, 3, [1, 2], 3, 1)
        s.make_action(2, 1)
        s.make_action(0, 2)
        s.make_action(2, 1)
        s.make_action(2, 1)
        self.assertEqual(True, s.is_terminal_state(), "This is an end state")

    def test_end_state_in_row(self):
        s = State(4, 3, [1, 2], 3, 1)
        s.make_action(3, 1)
        s.make_action(0, 2)
        s.make_action(3, 1)
        s.make_action(1, 2)
        s.make_action(2, 2)
        self.assertEqual(True, s.is_terminal_state(), "This is an end state")

    def test_possible_actions(self):
        s = State(4, 3, [1, 2], 3, 1)
        s.make_action(2, 1)
        s.make_action(0, 2)
        s.make_action(2, 1)
        s.make_action(2, 1)
        self.assertListEqual([0, 1, 3], s.possible_actions(), "Action 2 is not possible")

    def test_diagonals(self):
        s = State(4, 4, [1, 2], 3, 1)
        s.make_action(2, 1)
        s.make_action(0, 2)
        s.make_action(2, 1)
        s.make_action(0, 1)
        self.assertEqual("1000", s.diagonal(0, 1, Orientation.RIGHT), "Action 2 is not possible")

    def test_diagonals_middle(self):
        s = State(6, 4, [1, 2], 3, 1)
        s.make_action(2, 1)
        s.make_action(0, 2)
        s.make_action(2, 1)
        s.make_action(0, 2)
        s.make_action(1, 1)
        s.make_action(5, 2)
        s.make_action(5, 1)
        self.assertEqual("1100", s.diagonal(1, 0, Orientation.RIGHT), "Incorrect diagonal")
        self.assertEqual("1200", s.diagonal(1, 0, Orientation.LEFT), "Incorrect diagonal")
        self.assertEqual("2000", s.diagonal(0, 1, Orientation.RIGHT), "Incorrect diagonal")
        self.assertEqual("2000", s.diagonal(5, 0, Orientation.RIGHT), "Incorrect diagonal")
        self.assertEqual("1000", s.diagonal(5, 1, Orientation.LEFT), "Incorrect diagonal")
        self.assertEqual("0100", s.diagonal(4, 0, Orientation.RIGHT), "Incorrect diagonal")

    def active_player(self):
        s = State(6, 4, [1, 2], 3, 1)
        self.assertEqual(1, s.active_player(), "Active player should be 1")
        s.make_action(2, 1)
        s.make_action(0, 2)
        self.assertEqual(1, s.active_player(), "Active player should be 1")
        s.make_action(0, 1)
        self.assertEqual(2, s.active_player(), "Active player should be 2")

    def state_from_board(self):
        state = State(6, 4, [1, 2], 3, 1)
        board = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 2, 0, 0, 1, 0, 0, 1]
        s = State.state_from_board(board, state)
        self.assertListEqual(['22', '', '', '', '', '11'], s.columns(), "Columns incorrect after first move")




if __name__ == '__main__':
    unittest.main()