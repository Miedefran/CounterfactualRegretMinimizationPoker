from abc import ABC, abstractmethod
from typing import Any

from envs.base.combinaton_generator import PokerCombinationGenerator
from training.config import TrainingConfig


class TrainingGame(ABC):
    CHANCE_PLAYER = -1
    suit_abstraction: bool = False
    combination_generator: PokerCombinationGenerator

    def __init__(self):
        self.history = []
        self.state_stack = []
        self.starting_player = 0
        self._chance_targets = []
        self._chance_context = None
        self.pot = 0
        self._betting_round: int = 0
        self.done = False
        self.current_player = self.CHANCE_PLAYER

    @classmethod
    @abstractmethod
    def game_name(cls) -> str:
        """Returns the game's string identifier."""
        pass

    @abstractmethod
    def get_big_blind_equivalent(self) -> int:
        pass

class TrainingSolver(ABC):
    @staticmethod
    @abstractmethod
    def evaluate_solver(config: 'TrainingConfig') -> bool:
        """Evaluates if this solver is the one requested in the config."""
        pass

    @staticmethod
    @abstractmethod
    def create_solver(config: 'TrainingConfig', game: Any, combo_gen: Any) -> 'TrainingSolver':
        """Creates an instance of the solver."""
        pass

    @staticmethod
    @abstractmethod
    def supports_alternating_updates() -> bool:
        """Returns True if this solver supports alternating updates (for path generation)."""
        pass

    @staticmethod
    @abstractmethod
    def supports_partial_pruning() -> bool:
        """Returns True if this solver supports partial pruning (for path generation)."""
        pass

    @staticmethod
    @abstractmethod
    def supports_squared_weights() -> bool:
        """Returns True if this solver supports squared weights (for path generation)."""
        pass
