import copy
from typing import Iterable, List, Optional, Dict, Tuple

import numpy as np


class Reward:
    """..."""

    def __init__(
        self,
        tic,
        tac,
        win_reward,
        lose_reward,
        common_reward,
        action_list_length,
    ) -> None:
        self.states = None

        self._tic = tic
        self._tac = tac
        self._win_reward = win_reward
        self._lose_reward = lose_reward
        self._common_reward = common_reward
        self._action_list_length = action_list_length
        self._reward_table = None

    def _state_horizontals(self, state: List[int]) -> Tuple[Tuple[int]]:
        """..."""
        return (
            state[:3],
            state[3:6],
            state[6:9],
        )

    def _state_verticals(self, state: List[int]) -> Tuple[Tuple[int]]:
        """..."""
        return (
            (state[0], state[3], state[6]),
            (state[1], state[4], state[7]),
            (state[2], state[5], state[8]),
        )

    def _state_diagonals(self, state) -> Tuple[Tuple[int]]:
        """..."""
        return (
            (state[0], state[4], state[8]),
            (state[2], state[4], state[6]),
        )

    def _main_states(self, state: List[int]) -> Tuple[Tuple[int]]:
        """..."""
        _states = self._state_horizontals(state)
        _states += self._state_verticals(state)
        _states += self._state_diagonals(state)
        return _states

    def _get_available_action_idxs(self, state: List[int]) -> List[int]:
        """..."""
        return [
            i for i in range(self._action_list_length)
            if state[i] not in [self._tic, self._tac]
        ]

    def is_win_row_for(self, tic: int, row: Iterable[int]) -> bool:
        """..."""
        return all(list(map(lambda x: x==tic, row)))

    def _get_reward_from_current_state(self, state: List[int]) -> float:
        """..."""
        main_states = self._main_states(state)
        for sub_state in main_states:
            if self.is_win_row_for(self._tic, sub_state):
                return self._win_reward
            if self.is_win_row_for(self._tac, sub_state):
                return self._lose_reward

        return self._common_reward


    def _get_actions_rewards(self, state: List[int]) -> np.array:
        """..."""
        state_copy = copy.deepcopy(list(state))
        actions = np.zeros((self._action_list_length,))

        for action_idx in self._get_available_action_idxs(state):
            if state_copy[action_idx] in [self._tic, self._tac]:
                continue

            prev_state_val = state_copy[action_idx]
            state_copy[action_idx] = self._tic

            actions[action_idx] = self._get_reward_from_current_state(state_copy)

            state_copy[action_idx] = prev_state_val

        return actions

    def from_states(self, states: List[int]):
        self.states = states
        return self


    def build(self):
        """..."""
        rewards = {}
        for state in self.states:
            rewards[state] = self._get_actions_rewards(state)
        self._reward_table = rewards

    def is_next_action_win(self, state: List[int], turn: int) ->  bool:
        """..."""
        available_actions = self._get_available_action_idxs(state)
        for action in available_actions:
            state_copy = list(copy.deepcopy(state))
            state_copy[action] = turn
            main_states = self._main_states(state_copy)
            for sub_state in main_states:
                if self.is_win_row_for(turn, sub_state):
                    return True
        return False

    def get_reward(self, state: List[int], action: int) -> float:
        """..."""
        state_copy = list(copy.deepcopy(state))
        state_copy[action] = self._tic

        if self._reward_table[tuple(state)][action] == self._win_reward:
            return self._win_reward

        if self.is_next_action_win(state_copy, self._tac):
            return self._lose_reward

        return self._reward_table[tuple(state)][action]

