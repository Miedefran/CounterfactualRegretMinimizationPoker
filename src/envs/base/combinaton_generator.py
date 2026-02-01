from abc import ABC, abstractmethod


class PokerCombinationGenerator(ABC):
    @abstractmethod
    def get_all_combinations(self):
        raise NotImplementedError

    @abstractmethod
    def setup_game_with_combination(self, game, combination):
        raise NotImplementedError
