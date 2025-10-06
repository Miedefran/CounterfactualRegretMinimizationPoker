from envs.kuhn_poker.dealer import KuhnPokerDealer

class LeducHoldemDealer(KuhnPokerDealer):
    
    def __init__(self):
        super().__init__()
    
    def reset(self):
        self.deck = ['J', 'J', 'Q', 'Q', 'K', 'K']
