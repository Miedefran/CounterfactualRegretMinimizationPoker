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
    },
    'twelve_card_poker': {
        'ante': 1,
        'bet_sizes': [2, 4, 8],
        'bet_limit': 2
    },
    'royal_holdem': {
        'ante': 1,
        'bet_sizes': [2, 4, 8, 8],
        'bet_limit': 3
    },
    'small_island_holdem': {
        'ante': 5,
        'bet_sizes': [10, 20, 20],
        'bet_limit': 2
    },
    'limit_holdem': {
        'small_blind': 5,
        'big_blind': 10,
        'bet_sizes': [10, 10, 20, 20],
        'bet_limit': 4
    }
}

def get_model_path(game, iterations, algorithm='cfr'):
    base_dir = 'data/models'
    filename = f"{game}_{iterations}.pkl.gz"
    iterations_dir = str(iterations)
    
    if game.startswith('kuhn'):
        case = game.split('_')[1]
        path = os.path.join(base_dir, 'kuhn', case, algorithm, iterations_dir, filename)
    elif game == 'leduc':
        path = os.path.join(base_dir, 'leduc', algorithm, iterations_dir, filename)
    elif game == 'rhode_island':
        path = os.path.join(base_dir, 'rhode_island', algorithm, iterations_dir, filename)
    elif game == 'twelve_card_poker':
        path = os.path.join(base_dir, 'twelve_card_poker', algorithm, iterations_dir, filename)
    elif game == 'royal_holdem':
        path = os.path.join(base_dir, 'royal_holdem', algorithm, iterations_dir, filename)
    elif game == 'small_island_holdem':
        path = os.path.join(base_dir, 'small_island_holdem', algorithm, iterations_dir, filename)
    elif game == 'limit_holdem':
        path = os.path.join(base_dir, 'limit_holdem', algorithm, iterations_dir, filename)
    else:
        path = os.path.join(base_dir, algorithm, iterations_dir, filename)
    
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
        deck = ['Js', 'Jh', 'Qs', 'Qh', 'Ks', 'Kh']
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


class LeducHoldemCombinationsAbstracted(PokerCombinationGenerator):
    """
    Suit-abstracted Combination Generator für Leduc Holdem.
    Verwendet nur Ranks (J, Q, K) ohne Suits.
    Generiert nur eindeutige Kombinationen nach Rank (nicht nach Index),
    um die Anzahl der Root Nodes zu reduzieren.
    """
    
    def get_all_combinations(self):
        ranks = ['J', 'Q', 'K']
        combinations = []
        seen = set()
        
        # Generiere alle Kombinationen, aber filtere Duplikate nach Rank
        # WICHTIG: Reihenfolge der private cards ist wichtig (p0, p1)
        for r1 in ranks:
            for r2 in ranks:
                    for r3 in ranks:
                        # Kombination: (p0_card, p1_card, public_card)
                        # NICHT sortieren, da Reihenfolge wichtig ist!
                        combo_key = (r1, r2, r3)
                        if combo_key not in seen:
                            seen.add(combo_key)
                            combinations.append((r1, r2, r3))
        
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


class TwelveCardPokerCombinations(PokerCombinationGenerator):
    
    def get_all_combinations(self):
        ranks = ['J', 'Q', 'K', 'A']
        suits = ['s', 'h', 'd']
        deck = [rank + suit for rank in ranks for suit in suits]
        combinations = []
        
        for i, p0_card in enumerate(deck):
            for j, p1_card in enumerate(deck):
                if i != j:
                    for k, public1 in enumerate(deck):
                        if k != i and k != j:
                            for l, public2 in enumerate(deck):
                                if l != i and l != j and l != k:
                                    combinations.append((p0_card, p1_card, public1, public2))
        
        return combinations
    
    def setup_game_with_combination(self, game, combination):
        game.reset(0)
        game.players[0].private_card = combination[0]
        game.players[1].private_card = combination[1]
        game.dealer.deck = [combination[2], combination[3]]


class TwelveCardPokerCombinationsAbstracted(PokerCombinationGenerator):
    """
    Suit-abstracted Combination Generator für Twelve Card Poker.
    Verwendet nur Ranks (J, Q, K, A) ohne Suits.
    Generiert nur eindeutige Kombinationen nach Rank (nicht nach Index),
    um die Anzahl der Nodes zu reduzieren.
    """
    
    def get_all_combinations(self):
        ranks = ['J', 'Q', 'K', 'A']
        combinations = []
        seen = set()
        
        # Generiere alle Kombinationen, aber filtere Duplikate nach Rank
        for r1 in ranks:
            for r2 in ranks:
                if r1 != r2:  # Verschiedene private cards
                    for r3 in ranks:
                        if r3 != r1 and r3 != r2:  # Public card 1
                            for r4 in ranks:
                                if r4 != r1 and r4 != r2 and r4 != r3:  # Public card 2
                                    # Kombination als Tuple von Ranks (ohne Index)
                                    # Sortiere private cards und public cards für Eindeutigkeit
                                    combo_key = tuple(sorted([r1, r2])) + tuple(sorted([r3, r4]))
                                    if combo_key not in seen:
                                        seen.add(combo_key)
                                        # Speichere in ursprünglicher Reihenfolge
                                        combinations.append((r1, r2, r3, r4))
        
        return combinations
    
    def setup_game_with_combination(self, game, combination):
        game.reset(0)
        game.players[0].private_card = combination[0]
        game.players[1].private_card = combination[1]
        game.dealer.deck = [combination[2], combination[3]]


class RoyalHoldemCombinations(PokerCombinationGenerator):
    
    def get_all_combinations(self):
        ranks = ['T', 'J', 'Q', 'K', 'A']
        suits = ['s', 'h', 'd', 'c']
        deck = [rank + suit for rank in ranks for suit in suits]
        combinations = []
        
        for i, p0c1 in enumerate(deck):
            for j, p0c2 in enumerate(deck):
                if i != j:
                    for k, p1c1 in enumerate(deck):
                        if k != i and k != j:
                            for l, p1c2 in enumerate(deck):
                                if l != i and l != j and l != k:
                                    for m, public1 in enumerate(deck):
                                        if m != i and m != j and m != k and m != l:
                                            for n, public2 in enumerate(deck):
                                                if n != i and n != j and n != k and n != l and n != m:
                                                    for o, public3 in enumerate(deck):
                                                        if o != i and o != j and o != k and o != l and o != m and o != n:
                                                            for p, public4 in enumerate(deck):
                                                                if p != i and p != j and p != k and p != l and p != m and p != n and p != o:
                                                                    for q, public5 in enumerate(deck):
                                                                        if q != i and q != j and q != k and q != l and q != m and q != n and q != o and q != p:
                                                                            combinations.append((p0c1, p0c2, p1c1, p1c2, public1, public2, public3, public4, public5))
        
        return combinations
    
    def setup_game_with_combination(self, game, combination):
        game.reset(0)
        game.players[0].set_private_cards(combination[0], combination[1])
        game.players[1].set_private_cards(combination[2], combination[3])
        game.dealer.deck = [combination[4], combination[5], combination[6], combination[7], combination[8]]


class SmallIslandHoldemCombinations(PokerCombinationGenerator):
    
    def get_all_combinations(self):
        ranks = ['T', 'J', 'Q', 'K', 'A']
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


class LimitHoldemCombinations(PokerCombinationGenerator):
    
    def get_all_combinations(self):
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = ['s', 'h', 'd', 'c']
        deck = [rank + suit for rank in ranks for suit in suits]
        combinations = []
        
        for i, p0c1 in enumerate(deck):
            for j, p0c2 in enumerate(deck):
                if i != j:
                    for k, p1c1 in enumerate(deck):
                        if k != i and k != j:
                            for l, p1c2 in enumerate(deck):
                                if l != i and l != j and l != k:
                                    for m, public1 in enumerate(deck):
                                        if m != i and m != j and m != k and m != l:
                                            for n, public2 in enumerate(deck):
                                                if n != i and n != j and n != k and n != l and n != m:
                                                    for o, public3 in enumerate(deck):
                                                        if o != i and o != j and o != k and o != l and o != m and o != n:
                                                            for p, public4 in enumerate(deck):
                                                                if p != i and p != j and p != k and p != l and p != m and p != n and p != o:
                                                                    for q, public5 in enumerate(deck):
                                                                        if q != i and q != j and q != k and q != l and q != m and q != n and q != o and q != p:
                                                                            combinations.append((p0c1, p0c2, p1c1, p1c2, public1, public2, public3, public4, public5))
        
        return combinations
    
    def setup_game_with_combination(self, game, combination):
        game.reset(0)
        game.players[0].set_private_cards(combination[0], combination[1])
        game.players[1].set_private_cards(combination[2], combination[3])
        game.dealer.deck = [combination[4], combination[5], combination[6], combination[7], combination[8]]
