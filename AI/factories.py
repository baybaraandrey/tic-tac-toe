from itertools import permutations
from typing import Dict, Iterable, List, Tuple

import numpy as np



class InitialStateFactory:
    """..."""

    def __init__(self, tic, tac):
        """..."""
        self.tic = tic
        self.tac = tac

    def create(self) -> List[List[int]]:
        """creates base for states/environment."""
        return [
            [self.tic, 0,   0,   0,   0,   0,   0,   0,   0],
            [self.tic, self.tac, 0,   0,   0,   0,   0,   0,   0],
            [self.tic, self.tac, self.tic, 0,   0,   0,   0,   0,   0],
            [self.tic, self.tac, self.tic, self.tac, 0,   0,   0,   0,   0],
            [self.tic, self.tac, self.tic, self.tac, self.tic, 0,   0,   0,   0],
            [self.tic, self.tac, self.tic, self.tac, self.tic, self.tac, 0,   0,   0],
            [self.tic, self.tac, self.tic, self.tac, self.tic, self.tac, self.tic, 0,   0],
            [self.tic, self.tac, self.tic, self.tac, self.tic, self.tac, self.tic, self.tac, 0],
            [self.tic, self.tac, self.tic, self.tac, self.tic, self.tac, self.tic, self.tac, self.tic],
        ]


class PossibleStatesFactory:
    """..."""
    def __init__(self) -> None:
        self.state = None

    def from_state(self, state):
        self.state = state
        return self

    def create(self) -> List[List[int]]:
        """generate all possible states."""
        states = []
        for s in self.state:
            # TODO: find better way
            perms = list(set(permutations(s)))
            states.extend(perms)
        return states


class QFactory:
    """..."""

    def __init__(self, action_list_length) -> None:
        self.states = None
        self._action_list_length = action_list_length

    def from_states(self, states):
        self.states = states
        return self

    def create(self) -> Dict[Tuple[int], np.array]:
        """..."""
        q = {}
        for state in self.states:
            q[state] = np.zeros((self._action_list_length, ))
        return q

