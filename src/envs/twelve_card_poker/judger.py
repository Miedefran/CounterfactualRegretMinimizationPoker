from envs.leduc_holdem.judger import LeducHoldemJudger


class TwelveCardPokerJudger(LeducHoldemJudger):

    def __init__(self):
        super().__init__()
        self.hand_rank = {'J': 11, 'Q': 12, 'K': 13, 'A': 14}

    def evaluate_hand(self, player):
        private = player.private_card
        public_cards = player.public_cards
        all_cards = [private] + public_cards

        # During gameplay the evaluator may be called with fewer than 3 cards
        # (e.g. preflop or after only 1 public card). The original logic assumes
        # there are at least 2 cards for the high-card case (cards[1]).
        if len(all_cards) < 2:
            # Not enough cards to distinguish much: treat as high card.
            hi = self.hand_rank[all_cards[0][0]] if all_cards else 0
            return (0, hi, 0)

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
        c0 = cards[0]
        c1 = cards[1] if len(cards) > 1 else 0
        c2 = cards[2] if len(cards) > 2 else 0
        return (0, c0, c1, c2)
