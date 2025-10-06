class LeducHoldemRound:
    
    def __init__(self, bet_sizes=[2, 4], bet_limit=2):
        self.bet_limit = bet_limit
        self.bet_sizes = bet_sizes  
        self.current_bet_size = bet_sizes[0]  
        self.current_player = 0
        self.player_folded = False
        self.pot = 0
        self.bet_count = 0
        self.passive_action_count = 0
        self.round_bets = [0, 0]
        self.total_bets = [0, 0]
        
    def start_new_round(self, starting_player, betting_round=0):
        self.current_player = starting_player
        self.bet_count = 0
        self.passive_action_count = 0
        self.round_bets = [0, 0] 
        self.player_folded = False
        self.current_bet_size = self.bet_sizes[betting_round]
    
    def get_amount_to_call(self):
        return max(self.round_bets) - self.round_bets[self.current_player]
        
    def proceed_round(self, players, action):
        if action == 'call':
            call_amount = self.get_amount_to_call()
            self.round_bets[self.current_player] += call_amount
            self.total_bets[self.current_player] += call_amount
            self.pot += call_amount
            self.passive_action_count += 1
            
        elif action == 'bet':
            call_amount = self.get_amount_to_call()
            total_amount = call_amount + self.current_bet_size
            self.round_bets[self.current_player] += total_amount
            self.total_bets[self.current_player] += total_amount
            self.pot += total_amount
            self.bet_count += 1
            self.passive_action_count = 1
            
        elif action == 'fold':
            self.player_folded = True
            
        elif action == 'check':
            self.passive_action_count += 1
            
        self.current_player = 1 - self.current_player
        return self.current_player
        
    def is_round_complete(self):
        return self.passive_action_count >= 2 or self.player_folded
        
    def get_legal_actions(self):
        actions = []
        amount_to_call = self.get_amount_to_call()
        
        if amount_to_call > 0:
            actions.extend(['call', 'fold'])
        else:
            actions.append('check')
        
        if self.bet_count < self.bet_limit:
            actions.append('bet')

        return actions
    