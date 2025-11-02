from envs.leduc_holdem.judger import LeducHoldemJudger

class TwelveCardPokerJudger(LeducHoldemJudger):
  
    def __init__(self):
        super().__init__()
        self.hand_rank = {'J': 11, 'Q': 12, 'K': 13, 'A': 14}

    def evaluate_hand(self, player):
        private = player.private_card
        public_cards = player.public_cards
        all_cards = [private] + public_cards
        
        ranks = [card[0] for card in all_cards]
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        
        if 3 in rank_counts.values():
            trip_rank = [self.hand_rank[r] for r, c in rank_counts.items() if c == 3][0]
            return (2, trip_rank, 0)
        
        if 2 in rank_counts.values():
            pair_rank = [self.hand_rank[r] for r, c in rank_counts.items() if c == 2][0]
            kicker = [self.hand_rank[r] for r, c in rank_counts.items() if c == 1][0]
            return (1, pair_rank, kicker)
        
        cards = sorted([self.hand_rank[card[0]] for card in all_cards], reverse=True)
        return (0, cards[0], cards[1])