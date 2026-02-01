import os
from typing import List, Optional, Type

from envs.base.combinaton_generator import PokerCombinationGenerator
from training.registry import TrainingGame


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


# Game registry - populated after imports
ALL_GAMES: List[Type[TrainingGame]] = []


def _populate_all_games():
    """Populate ALL_GAMES list with all game classes."""
    global ALL_GAMES
    if ALL_GAMES:
        return  # Already populated

    from envs.kuhn_poker.game import KuhnPokerGame, KuhnCase1Game, KuhnCase3Game, KuhnCase4Game
    from envs.leduc_holdem.game import LeducHoldemGame, LeducHoldemAbstractedGame
    from envs.rhode_island.game import RhodeIslandGame
    from envs.twelve_card_poker.game import TwelveCardPokerGame, TwelveCardPokerAbstractedGame
    from envs.royal_holdem.game import RoyalHoldemGame
    from envs.small_island_holdem.game import SmallIslandHoldemGame
    from envs.limit_holdem.game import LimitHoldemGame

    ALL_GAMES = [
        KuhnPokerGame,
        KuhnCase1Game,
        KuhnCase3Game,
        KuhnCase4Game,
        LeducHoldemGame,
        LeducHoldemAbstractedGame,
        RhodeIslandGame,
        TwelveCardPokerGame,
        TwelveCardPokerAbstractedGame,
        RoyalHoldemGame,
        SmallIslandHoldemGame,
        LimitHoldemGame,
    ]


def find_game_class_for_abstraction(game_name: str, use_suit_abstraction: bool) -> Optional[Type[TrainingGame]]:
    """Find appropriate game class considering abstraction."""
    _populate_all_games()
    for game_cls in ALL_GAMES:
        if game_cls.game_name() == game_name and game_cls.suit_abstraction == use_suit_abstraction:
            return game_cls

    return None


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
    Suit-abstracted combination generator for Leduc Holdem.
    Uses only ranks (J, Q, K) without suits.
    Generates only unique combinations by rank (not by index),
    to reduce the number of root nodes.
    """

    def get_all_combinations(self):
        ranks = ['J', 'Q', 'K']
        combinations = []
        seen = set()

        # Generate all combinations, but filter duplicates by rank
        # Preserve order of private cards (p0, p1)
        for r1 in ranks:
            for r2 in ranks:
                for r3 in ranks:
                    # Combination: (p0_card, p1_card, public_card); do not sort
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
    Suit-abstracted combination generator for Twelve Card Poker.
    Uses only ranks (J, Q, K, A) without suits.
    Generates only unique combinations by rank (not by index),
    to reduce the number of nodes.
    """

    def get_all_combinations(self):
        ranks = ['J', 'Q', 'K', 'A']
        combinations = []
        seen = set()

        # Generate all combinations, but filter duplicates by rank
        for r1 in ranks:
            for r2 in ranks:
                if r1 != r2:  # Different private cards
                    for r3 in ranks:
                        if r3 != r1 and r3 != r2:  # Public card 1
                            for r4 in ranks:
                                if r4 != r1 and r4 != r2 and r4 != r3:  # Public card 2
                                    # Combination as tuple of ranks (without index)
                                    # Sort private cards and public cards for uniqueness
                                    combo_key = tuple(sorted([r1, r2])) + tuple(sorted([r3, r4]))
                                    if combo_key not in seen:
                                        seen.add(combo_key)
                                        # Store in original order
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
                                                                            combinations.append(
                                                                                (p0c1, p0c2, p1c1, p1c2, public1,
                                                                                 public2, public3, public4, public5))

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
                                                                            combinations.append(
                                                                                (p0c1, p0c2, p1c1, p1c2, public1,
                                                                                 public2, public3, public4, public5))

        return combinations

    def setup_game_with_combination(self, game, combination):
        game.reset(0)
        game.players[0].set_private_cards(combination[0], combination[1])
        game.players[1].set_private_cards(combination[2], combination[3])
        game.dealer.deck = [combination[4], combination[5], combination[6], combination[7], combination[8]]
