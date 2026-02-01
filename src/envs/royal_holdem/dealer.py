from envs.kuhn_poker.dealer import KuhnPokerDealer


class RoyalHoldemDealer(KuhnPokerDealer):

    def __init__(self):
        super().__init__()

    def reset(self):
        ranks = ['T', 'J', 'Q', 'K', 'A']
        suits = ['s', 'h', 'd', 'c']
        self.deck = [rank + suit for rank in ranks for suit in suits]
