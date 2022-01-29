import random

from typing import List, Optional, Iterable
from typing_extensions import Self

from numpy import append

from AI import QModel



class Game:
    def __init__(self, model: QModel) -> None:
        # AI model
        self._model = model

        # game state
        self.TIC = model.TIC
        self.TAC = model.TAC
        

        self.TIC_WIN = 1
        self.TAC_WIN = -1
        self.DRAW = 0
        self.IN_PROGRESS = None

        self.TIC_WIN_COUNT = 0
        self.TAC_WIN_COUNT = 0
        self.DRAW_COUNT = 0

        self._board = None

        self._tac_win_msg = 'Congrats Human win!!!'
        self._tic_win_msg = 'AI win!!!'
        self._draw_msg = 'Draw!!!'

        self._turn = self.TAC

    def _flip_turn(self):
        if self._turn == self.TIC:
            self._turn = self.TAC
        else:
            self._turn = self.TIC

    def _init_board(self) -> None:
        board = [0, 0,   0,   0,   0,   0,   0,   0,   0]
        board[random.randint(0, 8)] = self.TIC
        self._board = board
        self._turn = self.TAC

    def _print_statistics(self):
        print('-'*30)
        print("AI won %d times" % self.TIC_WIN_COUNT)
        print("Human won %d times" % self.TAC_WIN_COUNT)
        print("Draws %d" % self.DRAW_COUNT)
        print('-'*30)


    def _print_board(self):
        board = []
        for item in self._board:
            if item == self.TAC:
                board.append('O')
            elif item == self.TIC:
                board.append('X')
            else:
                board.append(' ')

        s = []
        for i in range(0, 9, 3):
            s.append('|'.join(board[i:i+3]))
            s.append('\n')
            s.append("-----")
            s.append('\n')
        s = s[:len(s) -2]
        print()
        print(''.join(s))


    def _state_horizontals(self, state: List[int]) -> tuple[tuple[int]]:
        """..."""
        return (
            state[:3],
            state[3:6],
            state[6:9],
        )

    def _state_verticals(self, state: List[int]) -> tuple[tuple[int]]:
        """..."""
        return (
            (state[0], state[3], state[6]),
            (state[1], state[4], state[7]),
            (state[2], state[5], state[8]),
        )

    def _state_diagonals(self, state) -> tuple[tuple[int]]:
        """..."""
        return (
            (state[0], state[4], state[8]),
            (state[2], state[4], state[6]),
        )

    def _main_states(self) -> tuple[tuple[int]]:
        """..."""
        _states = self._state_horizontals(self._board)
        _states += self._state_verticals(self._board)
        _states += self._state_diagonals(self._board)
        return _states

    def _is_win_row_for(self, tic: int, row: Iterable[int]) -> bool:
        """..."""
        return all(list(map(lambda x: x==tic, row)))


    def _board_state(self) -> Optional[str]:
        """..."""
        main_states = self._main_states()
        for sub_state in main_states:
            if self._is_win_row_for(self.TIC, sub_state):
                return self.TIC_WIN
            if self._is_win_row_for(self.TAC, sub_state):
                return self.TAC_WIN

        draw = self.DRAW
        for el in self._board:
            if el == 0:
                draw = self.IN_PROGRESS
        return draw

    def run(self) -> None:
        self._init_board()
        
        self._print_board()
        while True:
            game_progress = self._board_state()
            if game_progress == self.TIC_WIN:
                self.TIC_WIN_COUNT += 1
                print(self._tic_win_msg)
                self._init_board()
                self._print_board()
            elif game_progress == self.TAC_WIN:
                self.TAC_WIN_COUNT += 1
                print(self._tac_win_msg)
                self._init_board()
                self._print_board()
            elif game_progress == self.DRAW:
                self.DRAW_COUNT += 1
                print(self._draw_msg)
                self._init_board()
                self._print_board()
            else:
                if self._turn == self.TAC:
                    command = input("[Statistics/Restart/Exit/Hint[1-9]]: ")
                    if command == 'R' or command == 'Restart':
                        self._init_board()
                        self._print_board()
                    elif command == 'S' or command == 'Statistics':
                        self._print_statistics()
                        self._print_board()
                    elif command == 'E' or command == 'Exit':
                        exit(0)
                    else:
                        try:
                            i = int(command) - 1
                        except ValueError:
                            print('error while processing input')
                            continue
                        self._board[i] = self.TAC
                        self._flip_turn()
                else:
                    best_action, _ = self._model.predict(self._board)
                    self._board[best_action] = self.TIC
                    self._flip_turn()
                    self._print_board()

