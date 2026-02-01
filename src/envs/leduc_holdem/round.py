class LeducHoldemRound:

    def __init__(self, bet_sizes=[2, 4], bet_limit=2):
        self.bet_limit = bet_limit
        self.bet_sizes = bet_sizes
        self.current_bet_size = bet_sizes[0]
        self.bet_count = 0
        self.passive_action_count = 0
        self.round_bets = [0, 0]

    def start_new_round(self, game, starting_player, betting_round=0):
        self.bet_count = 0
        self.passive_action_count = 0
        self.round_bets = [0, 0]
        self.current_bet_size = self.bet_sizes[betting_round]
        # New betting rounds always start with the starting player.
        # Otherwise, first-to-act depends on how the previous round ended
        # (because proceed_round() flips current_player after every action).
        game.current_player = starting_player

    def get_amount_to_call(self, game):
        return max(self.round_bets) - self.round_bets[game.current_player]

    def proceed_round(self, game, action):
        if action == 'call':
            call_amount = self.get_amount_to_call(game)
            self.round_bets[game.current_player] += call_amount
            game.total_bets[game.current_player] += call_amount
            game.pot += call_amount
            self.passive_action_count += 1

        elif action == 'bet':
            call_amount = self.get_amount_to_call(game)
            total_amount = call_amount + self.current_bet_size
            self.round_bets[game.current_player] += total_amount
            game.total_bets[game.current_player] += total_amount
            game.pot += total_amount
            self.bet_count += 1
            self.passive_action_count = 1

        elif action == 'fold':
            pass

        elif action == 'check':
            self.passive_action_count += 1

        game.current_player = 1 - game.current_player
        return game.current_player

    def is_round_complete(self):
        return self.passive_action_count >= 2

    def get_legal_actions(self, game):
        actions = []
        amount_to_call = self.get_amount_to_call(game)

        if amount_to_call > 0:
            actions.extend(['call', 'fold'])
        else:
            actions.append('check')

        if self.bet_count < self.bet_limit:
            actions.append('bet')

        return actions
