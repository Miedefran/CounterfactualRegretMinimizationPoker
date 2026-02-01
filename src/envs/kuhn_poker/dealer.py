import random


class KuhnPokerDealer:

    def __init__(self):
        self.reset()

    def reset(self):
        self.deck = ['J', 'Q', 'K']

    def shuffle(self):
        random.shuffle(self.deck)

    def deal_card(self):
        if not self.deck:
            raise ValueError("Deck is empty!")
        return self.deck.pop()
