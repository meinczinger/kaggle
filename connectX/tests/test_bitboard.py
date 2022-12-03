import unittest
from my_agents.bitboard import BitBoard
import copy


class TestState(unittest.TestCase):
    def test_state_basic(self):
        board = [0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0]
        s = BitBoard.create_from_board(7, 6, 4, 1, board)
        state = copy.copy(s)
        self.assertListEqual([0, 1, 2, 3, 4, 5, 6], state.possible_actions(), "Columns incorrect after first move")
        state.make_action(0)
        state.make_action(0)
        state.make_action(0)
        state.make_action(0)
        state.make_action(0)
        state.make_action(0)
        self.assertListEqual([1, 2, 3, 4, 5, 6], state.possible_actions(), "Columns incorrect after first move")

    def test_game_over(self):
        board = [0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0]
        state = BitBoard.create_from_board(7, 6, 4, 1, board)
        state.make_action(0)
        state.make_action(0)
        state.make_action(1)
        state.make_action(0)
        state.make_action(2)
        state.make_action(0)
        state.make_action(3)
        self.assertEqual(True, state.is_terminal_state())

    def test_2(self):
        board = [1, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 1,
                 1, 0, 0, 0, 0, 0, 1,
                 1, 0, 2, 0, 0, 2, 2,
                 2, 2, 1, 2, 2, 1, 1,
                 1, 1, 2, 1, 1, 2, 1]
        state = BitBoard.create_from_board(7, 6, 4, 1, board)
        self.assertListEqual([1, 2, 3, 4, 5, 6], state.possible_actions(), "Columns incorrect after first move")

    def test_end_state(self):
        board = [0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 1, 1, 0, 0, 0]
        state = BitBoard.create_from_board(7, 6, 4, 1, board)
        new_state = copy.copy(state)
        new_state.make_action(0)
        self.assertEqual(True, new_state.is_terminal_state())

    def test_end_state2(self):
        board = [0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 1, 0, 0,
                 0, 0, 2, 1, 0, 0, 0,
                 0, 2, 2, 0, 0, 2, 0,
                 0, 1, 1, 0, 2, 0, 2]
        state = BitBoard.create_from_board(7, 6, 4, 1, board)
        state.make_action(0)
        self.assertEqual(False, state.is_terminal_state())

    def test_to_list(self):
        board = [1, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 1,
                 1, 0, 0, 0, 0, 0, 1,
                 1, 0, 2, 0, 0, 2, 2,
                 2, 2, 1, 2, 2, 1, 1,
                 1, 1, 2, 1, 1, 2, 1]
        state = BitBoard.create_from_board(7, 6, 4, 1, board)
        board2 = state.to_list()
        self.assertListEqual(board, board2)

    def test_full_board(self):
        board = [0, 1, 2, 1, 2, 1, 2,
                 1, 2, 2, 2, 1, 2, 1,
                 2, 1, 1, 2, 1, 2, 2,
                 1, 1, 2, 2, 1, 1, 1,
                 2, 1, 2, 1, 2, 1, 2,
                 1, 2, 1, 2, 1, 2, 1]
        state = BitBoard.create_from_board(7, 6, 4, 1, board)
        state.make_action(0)
        self.assertEqual(True, state.is_terminal_state())

    def test_from_bitboard_to_list(self):
        board = [0, 1, 2, 1, 2, 1, 2,
                 1, 2, 2, 2, 1, 2, 1,
                 2, 1, 1, 2, 1, 2, 2,
                 1, 1, 2, 2, 1, 1, 1,
                 2, 1, 2, 1, 2, 1, 2,
                 1, 2, 1, 2, 1, 2, 1]
        state = BitBoard.create_from_board(7, 6, 4, 1, board)
        bb = state.hash()
        self.assertListEqual(board, BitBoard.from_bitboard_to_list(7, 6, bb))

    def test_mirror(self):
        board = [0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 2, 0, 0, 1, 0,
                 0, 1, 1, 1, 0, 2, 0]
        state = BitBoard.create_from_board(7, 6, 4, 1, board)
        mirror = state.mirror_board()
        print(mirror)


if __name__ == '__main__':
    unittest.main()
