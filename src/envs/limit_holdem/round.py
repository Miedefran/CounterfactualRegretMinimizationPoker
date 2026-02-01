from envs.royal_holdem.round import RoyalHoldemRound


class LimitHoldemRound(RoyalHoldemRound):
    def start_new_round(self, game, starting_player, betting_round=0):
        super().start_new_round(game, starting_player, betting_round)

        # Preflop: small blind (button) acts first. Postflop: big blind acts first.
        if betting_round == 0:
            sb = starting_player
            bb = 1 - starting_player
            self.round_bets = [0, 0]
            self.round_bets[sb] = game.small_blind
            self.round_bets[bb] = game.big_blind
            game.current_player = sb
        else:
            game.current_player = 1 - starting_player
