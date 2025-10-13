import os

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
    },
    'rhode_island': {
        'ante': 5,
        'bet_sizes': [10, 20, 20],
        'bet_limit': 3
    }
}

def get_model_path(game, iterations):
    base_dir = 'models'
    filename = f"{game}_{iterations}.pkl.gz"
    
    if game.startswith('kuhn'):
        case = game.split('_')[1]
        path = os.path.join(base_dir, 'kuhn', case, filename)
    elif game == 'leduc':
        path = os.path.join(base_dir, 'leduc', filename)
    elif game == 'rhode_island':
        path = os.path.join(base_dir, 'rhode_island', filename)
    else:
        path = os.path.join(base_dir, filename)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

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
        deck = ['J', 'J', 'Q', 'Q', 'K', 'K']
        combinations = []
        
        for i, card1 in enumerate(deck):
            for j, card2 in enumerate(deck):
                if i != j:
                    for k, public in enumerate(deck):
                        if k != i and k != j:
                            combinations.append((card1, card2, public))
        
        return combinations
    
    def setup_game_with_combination(self, game, combination):
        game.reset(0)
        game.players[0].private_card = combination[0]
        game.players[1].private_card = combination[1]
        game.dealer.deck = [combination[2]]


class RhodeIslandCombinations(PokerCombinationGenerator):
    
    def get_all_combinations(self):
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = ['s', 'h', 'd', 'c']
        deck = [rank + suit for rank in ranks for suit in suits]
        combinations = []
        
        for i, card1 in enumerate(deck):
            for j, card2 in enumerate(deck):
                if i != j:
                    for k, public1 in enumerate(deck):
                        if k != i and k != j:
                            for l, public2 in enumerate(deck):
                                if l != i and l != j and l != k:
                                    combinations.append((card1, card2, public1, public2))
        
        return combinations
    
    def setup_game_with_combination(self, game, combination):
        game.reset(0)
        game.players[0].private_card = combination[0]
        game.players[1].private_card = combination[1]
        game.dealer.deck = [combination[2], combination[3]]
