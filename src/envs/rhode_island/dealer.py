from envs.kuhn_poker.dealer import KuhnPokerDealer


class RhodeIslandDealer(KuhnPokerDealer):

    def __init__(self):
        super().__init__()

    def reset(self):
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = ['s', 'h', 'd', 'c']
        self.deck = [rank + suit for rank in ranks for suit in suits]
