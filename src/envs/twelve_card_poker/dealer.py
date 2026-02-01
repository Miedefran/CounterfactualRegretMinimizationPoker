from envs.kuhn_poker.dealer import KuhnPokerDealer


class TwelveCardPokerDealer(KuhnPokerDealer):

    def __init__(self):
        super().__init__()

    def reset(self):
        ranks = ['J', 'Q', 'K', 'A']
        suits = ['s', 'h', 'd']
        self.deck = [rank + suit for rank in ranks for suit in suits]
