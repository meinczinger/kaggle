import unittest
from agents.SimpleState import SimpleState
from agents.search import MiniMax


class TestState(unittest.TestCase):
    def test_forced_win(self):
        board = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0]
        state = SimpleState(7, 6, [1, 2], 4, board, 1)
        self.assertEqual(False, state.is_terminal_state())
        state.make_action(5)
        state.make_action(3)
        state.make_action(2)
        # state.make_action(4, 1)
        self.assertEqual(False, state.is_terminal_state())
        state.make_action(3)
        state.make_action(4)
        state.make_action(1)
        state.make_action(5)
        state.make_action(4)
        state.make_action(6)
        state.make_action(4)
        state.make_action(0)
        state.make_action(2)
        state.make_action(1)
        state.make_action(3)
        state.make_action(4)
        state.make_action(0)
        state.make_action(3)
        state.make_action(2)
        state.make_action(4)
        state.make_action(0)
        state.make_action(1)
        state.make_action(0)
        state.make_action(0)
        state.make_action(2)
        state.make_action(0)

        self.assertEqual(5, MiniMax.forced_win(state, 2, 3)[0])
        self.assertEqual(True, MiniMax.forced_win(state, 2, 3)[1])


if __name__ == '__main__':
    unittest.main()