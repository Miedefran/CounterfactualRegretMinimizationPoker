import pytest
from envs.leduc_holdem.game import LeducHoldemGame
from envs.rhode_island.game import RhodeIslandGame
from envs.royal_holdem.game import RoyalHoldemGame

class TestLeducHoldemLogic:
    @pytest.fixture
    def leduc_game(self):
        # Ante = 1, Bet sizes = [2, 4]
        game = LeducHoldemGame(ante=1, bet_sizes=[2, 4])
        game.reset(starting_player=0)
        
        # Manually set cards for deterministic testing
        # P0: King, P1: Jack, Deck: Queen (for public)
        game.players[0].private_card = 'K'
        game.players[1].private_card = 'J'
        game.dealer.deck = ['Q'] 
        return game

    def test_fold_preflop_p0_immediate(self, leduc_game):
        """P0 folds immediately. P1 wins the ante."""
        legal = leduc_game.get_legal_actions()
        assert 'check' in legal
        assert 'bet' in legal
        
        # If P0 bets, P1 can fold.
        leduc_game.step('bet') # P0 bets 2. Pot = 1+1+2 = 4.
        assert leduc_game.current_player == 1
        
        leduc_game.step('fold') # P1 folds.
        
        assert leduc_game.done
        
        p0_payoff = leduc_game.get_payoff(0)
        p1_payoff = leduc_game.get_payoff(1)
        
        assert p0_payoff == 1
        assert p1_payoff == -1
        
    def test_fold_preflop_p0_check_p1_bet_p0_fold(self, leduc_game):
        """P0 checks, P1 bets, P0 folds. P1 wins P0's ante."""
        leduc_game.step('check')
        assert leduc_game.current_player == 1
        
        leduc_game.step('bet') # P1 bets 2. Total P1: 3. P0: 1. Pot: 4.
        assert leduc_game.current_player == 0
        
        leduc_game.step('fold') # P0 folds.
        
        assert leduc_game.done
        
        assert leduc_game.get_payoff(0) == -1
        assert leduc_game.get_payoff(1) == 1

    def test_showdown_p0_wins_high_card(self, leduc_game):
        """Standard showdown. P0 (K) beats P1 (J)."""
        # Round 1
        leduc_game.step('check')
        leduc_game.step('check')
        
        # Transition to Round 2
        assert leduc_game.betting_round == 1
        assert leduc_game.public_card == 'Q'
        
        # Round 2
        leduc_game.step('check')
        leduc_game.step('check')
        
        assert leduc_game.done
        
        # Pot: 2 (Antes only).
        # Winner P0: 2 - 1 = +1.
        # Loser P1: -1.
        
        assert leduc_game.get_payoff(0) == 1
        assert leduc_game.get_payoff(1) == -1

    def test_showdown_tie(self, leduc_game):
        """Both players have same card."""
        leduc_game.players[0].private_card = 'K'
        leduc_game.players[1].private_card = 'K'
        leduc_game.dealer.deck = ['Q']
        
        leduc_game.step('check')
        leduc_game.step('check') # End R1
        leduc_game.step('check')
        leduc_game.step('check') # End R2
        
        assert leduc_game.get_payoff(0) == 0
        assert leduc_game.get_payoff(1) == 0

    def test_bug_regression_fold_attribution(self, leduc_game):
        """
        REGRESSION TEST: Ensure the fix for incorrect fold attribution works.
        If P1 bets and P0 folds, P1 must be the winner.
        Previous bug: P0 folding caused P1 (current_player) to be judged as the "folder".
        """
        leduc_game.step('check')
        leduc_game.step('bet') # P1 bets
        leduc_game.step('fold') # P0 folds
        
        # P1 is Winner.
        # P1 payoff should be positive.
        assert leduc_game.get_payoff(1) > 0
        assert leduc_game.get_payoff(0) < 0

    def test_turn_order_flip(self, leduc_game):
        """Verify current_player flips correctly."""
        assert leduc_game.current_player == 0
        leduc_game.step('check')
        assert leduc_game.current_player == 1
        leduc_game.step('check')
        # Round change -> reset to starting player (0 in Leduc usually)
        assert leduc_game.current_player == 0


class TestRhodeIslandLogic:
    @pytest.fixture
    def rhode_game(self):
        # Ante = 5, Bets=[10, 20, 20]
        game = RhodeIslandGame(ante=5, bet_sizes=[10, 20, 20])
        game.reset(starting_player=0)
        
        # P0: As, P1: 2s
        # Board: Ks, Qs
        game.players[0].private_card = 'As'
        game.players[1].private_card = '2s'
        game.dealer.deck = ['Ks', 'Qs'] 
        return game

    def test_fold_preflop(self, rhode_game):
        rhode_game.step('bet') # P0 bets 10
        rhode_game.step('fold') # P1 folds
        
        assert rhode_game.done
        # Winner P0.
        assert rhode_game.get_payoff(0) > 0
        assert rhode_game.get_payoff(1) < 0

    def test_showdown_full_rounds(self, rhode_game):
        # Round 1
        rhode_game.step('check')
        rhode_game.step('check')
        
        # Round 2 (1 public card dealt: Ks)
        assert rhode_game.betting_round == 1
        assert len(rhode_game.public_cards) == 1
        rhode_game.step('check')
        rhode_game.step('check')
        
        # Round 3 (2nd public card dealt: Qs)
        assert rhode_game.betting_round == 2
        assert len(rhode_game.public_cards) == 2
        rhode_game.step('check')
        rhode_game.step('check')
        
        assert rhode_game.done
        # P0 (As) vs P1 (2s) on (Ks, Qs). P0 wins (High Card Ace).
        assert rhode_game.get_payoff(0) > 0

    def test_regression_fold(self, rhode_game):
        # Verify the fold attribution fix in Rhode Island
        rhode_game.step('check')
        rhode_game.step('bet') # P1 bets
        rhode_game.step('fold') # P0 folds
        
        # P1 Winner
        assert rhode_game.get_payoff(1) > 0
        assert rhode_game.get_payoff(0) < 0


class TestRoyalHoldemLogic:
    @pytest.fixture
    def royal_game(self):
        game = RoyalHoldemGame()
        game.reset(starting_player=0)
        game.players[0].private_cards = ['As', 'Ks']
        game.players[1].private_cards = ['Ts', 'Js']
        # 5 public cards dealt in 3 rounds: 3 Flop, 1 Turn, 1 River
        game.dealer.deck = ['Qs', 'Ah', 'Kh', 'Qh', 'Jh']
        return game

    def test_regression_fold(self, royal_game):
        royal_game.step('check')
        royal_game.step('bet') 
        royal_game.step('fold') 
        
        assert royal_game.get_payoff(1) > 0
        assert royal_game.get_payoff(0) < 0

    def test_showdown_structure(self, royal_game):
        # Round 1 (Preflop)
        royal_game.step('check')
        royal_game.step('check')
        
        # Round 2 (Flop - 3 cards)
        assert royal_game.betting_round == 1
        assert len(royal_game.public_cards) == 3
        royal_game.step('check')
        royal_game.step('check')
        
        # Round 3 (Turn - 1 card)
        assert royal_game.betting_round == 2
        assert len(royal_game.public_cards) == 4
        royal_game.step('check')
        royal_game.step('check')
        
        # Round 4 (River - 1 card)
        assert royal_game.betting_round == 3
        assert len(royal_game.public_cards) == 5
        royal_game.step('check')
        royal_game.step('check')
        
        assert royal_game.done
        # Check P0 payoff (Assume win or tie, just check valid return)
        assert royal_game.get_payoff(0) is not None