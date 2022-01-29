import random
import copy
import math

from typing import List, Tuple

from tqdm import tqdm
import numpy as np

from .factories import (
    InitialStateFactory,
    QFactory,
    PossibleStatesFactory,
)
from .reward import Reward


TIC = 1 # AI flag
TAC = -1 # opponent flag


class QModel:
    """..."""

    TIC = 1
    TAC = -1

    def __init__(
        self,
    ) -> None:
        # will be initialized when fit will call
        self._initial_states = None
        self._possible_states = None
        self._Q = None

        self._win_reward = 100
        self._lose_reward = -7
        self._common_reward = 0.5
        self._action_list_length = 9

        self._initial_state_factory = InitialStateFactory(self.TIC, self.TAC)
        self._possible_states_factory = PossibleStatesFactory()
        self._rewards = Reward(
            tic=self.TIC,
            tac=self.TAC,
            win_reward=self._win_reward,
            lose_reward=self._lose_reward,
            common_reward=self._common_reward,
            action_list_length=self._action_list_length,
        )
        self._q_factory = QFactory(
            action_list_length=self._action_list_length,
        )


    def _prefit(self) -> None:
        """..."""
        self._initial_state = self._initial_state_factory.create()
        self._possible_states = self._possible_states_factory.from_state(
            state=self._initial_state,
        ).create()
        self._rewards.from_states(
            states=self._possible_states,
        ).build()
        self._Q = self._q_factory.from_states(
            states=self._possible_states,
        ).create()

    def _ai_turn_states(self) -> None:
        """..."""
        return [
            st for st in self._possible_states
            if sum(st) == 0
        ]

    def _get_next_state_from_actions(self, state: List[int], next_action: int, turn: int):
        next_state = copy.deepcopy(list(state))
        next_state[next_action] = turn
        next_state = tuple(next_state)
        return next_state

    def _best_available_action(self, q_values_in_state: np.array, available: List[int]) -> int:
        """..."""
        idx = -1
        m = -math.inf
        for i in range(len(q_values_in_state)):
            val = q_values_in_state[i]
            if i in available and val > m:
                idx = i
                m = val
        return idx

    def next_random_action_state(self, state: Tuple[int], turn: int) -> Tuple[int, Tuple[int]]:
        """..."""
        available_actions = self._rewards._get_available_action_idxs(state)
        next_action = np.random.choice(available_actions)
        next_state = self._get_next_state_from_actions(state, next_action, turn)
        return next_action, next_state

    def fit(
        self,
        alpha=0.9,
        gamma=0.75,
        epoch=100000,
    ) -> None:
        """fit the model."""
        self._prefit()
        states = self._ai_turn_states()
        for _ in tqdm(range(epoch)):
            current_state = random.choice(states)

            next_action, next_state = self.next_random_action_state(current_state, self.TIC)

            available_computer_actions = self._rewards._get_available_action_idxs(next_state)
            if not available_computer_actions:
                #print('no available actions for the computer', next_state)
                current_state_after_computer_turn = next_state
            else:
                _, current_state_after_computer_turn = self.predict(next_state)

            TD = self._rewards.get_reward(current_state, next_action) + gamma * np.argmax(self._Q[current_state_after_computer_turn]) - self._Q[current_state][next_action]
            self._Q[current_state][next_action] += alpha * TD



    def predict(self, state: Tuple[int]) -> Tuple[int, Tuple[int]]:
        """predicts the best hint."""
        available_actions = self._rewards._get_available_action_idxs(state)
        best_action = self._best_available_action(self._Q[tuple(state)], available_actions)
        new_state = self._get_next_state_from_actions(state, best_action, self.TAC)
        return best_action, new_state
