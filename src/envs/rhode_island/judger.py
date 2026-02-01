from envs.leduc_holdem.judger import LeducHoldemJudger


class RhodeIslandJudger(LeducHoldemJudger):

    def __init__(self):
        super().__init__()
        self.hand_rank = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                          '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}

    def evaluate_hand(self, player):
        private = player.private_card
        public_cards = player.public_cards
        all_cards = [private] + public_cards

        ranks = [card[0] for card in all_cards]
        suits = [card[1] for card in all_cards]
        rank_vals = sorted([self.hand_rank[r] for r in ranks], reverse=True)

        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1

        # During gameplay the evaluator may be called with fewer than 3 cards
        # (e.g. preflop or after the first public card). The original logic
        # assumes exactly 3 cards and would crash on rank_vals[2].
        #
        # For <3 cards we only distinguish Pair vs High Card and pad kickers.
        if len(rank_vals) < 3:
            if 2 in rank_counts.values():
                pair_rank = [self.hand_rank[r] for r, c in rank_counts.items() if c == 2][0]
                kicker = [self.hand_rank[r] for r, c in rank_counts.items() if c == 1]
                kicker_val = kicker[0] if kicker else 0
                return (1, pair_rank, kicker_val)

            hi = rank_vals[0] if len(rank_vals) >= 1 else 0
            lo = rank_vals[1] if len(rank_vals) >= 2 else 0
            return (0, hi, lo, 0)

        is_flush = len(set(suits)) == 1
        is_straight = (rank_vals[0] - rank_vals[2] == 2 and len(set(rank_vals)) == 3)

        if rank_vals == [14, 3, 2]:
            is_straight = True
            rank_vals = [3, 2, 1]

        # Straight Flush (höher als Three of a Kind)
        if is_flush and is_straight:
            return (5, rank_vals[0], rank_vals[1], rank_vals[2])

        if 3 in rank_counts.values():
            trip_rank = [self.hand_rank[r] for r, c in rank_counts.items() if c == 3][0]
            return (4, trip_rank, 0)

        if is_straight:
            return (3, rank_vals[0], rank_vals[1], rank_vals[2])

        if is_flush:
            return (2, rank_vals[0], rank_vals[1], rank_vals[2])

        if 2 in rank_counts.values():
            pair_rank = [self.hand_rank[r] for r, c in rank_counts.items() if c == 2][0]
            kicker = [self.hand_rank[r] for r, c in rank_counts.items() if c == 1][0]
            return (1, pair_rank, kicker)

        return (0, rank_vals[0], rank_vals[1], rank_vals[2])
