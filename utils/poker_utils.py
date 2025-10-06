GAME_CONFIGS = {
    'kuhn_case1': {
        'ante': 2,
        'bet_size': 1
    },
    'kuhn_case2': {
        'ante': 1,
        'bet_size': 1
    },
    'kuhn_case3': {
        'ante': 1,
        'bet_size': 1.5
    },
    'kuhn_case4': {
        'ante': 1,
        'bet_size': 2
    },
    'leduc': {
        'ante': 1,
        'bet_sizes': [2, 4],
        'bet_limit': 2
    }
}

class PokerCombinationGenerator:
    
    def get_all_combinations(self):
        raise NotImplementedError
    
    def setup_game_with_combination(self, game, combination):
        raise NotImplementedError


class KuhnPokerCombinations(PokerCombinationGenerator):
    
    def get_all_combinations(self):
        cards = ['J', 'Q', 'K']
        combinations = []
        for card1 in cards:
            for card2 in cards:
                if card1 != card2:
                    combinations.append((card1, card2))
        return combinations
    
    def setup_game_with_combination(self, game, combination):
        game.reset(0)
        game.players[0].private_card = combination[0]
        game.players[1].private_card = combination[1]


class LeducHoldemCombinations(PokerCombinationGenerator):
    
    def get_all_combinations(self):
        pass
    
    def setup_game_with_combination(self, game, combination):
        pass
