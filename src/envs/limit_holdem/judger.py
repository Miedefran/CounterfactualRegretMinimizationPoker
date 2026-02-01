from envs.royal_holdem.judger import RoyalHoldemJudger
import itertools


class LimitHoldemJudger(RoyalHoldemJudger):

    def __init__(self):
        super().__init__()
        self.hand_rank = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                          '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}

    def evaluate_hand(self, player):
        private_cards = player.private_cards
        public_cards = player.public_cards
        all_cards = private_cards + public_cards

        if len(all_cards) < 5:
            return (0, 0, 0, 0, 0)

        best_hand = (0, 0, 0, 0, 0)
        for combo in itertools.combinations(all_cards, 5):
            cards = list(combo)
            ranks = [card[0] for card in cards]
            suits = [card[1] for card in cards]
            rank_vals = sorted([self.hand_rank[r] for r in ranks], reverse=True)

            rank_counts = {}
            for r in ranks:
                rank_counts[r] = rank_counts.get(r, 0) + 1

            is_flush = len(set(suits)) == 1

            is_straight = False
            if len(set(rank_vals)) == 5:
                if rank_vals[0] - rank_vals[4] == 4:
                    is_straight = True
                    high_card = rank_vals[0]
                elif rank_vals == [14, 13, 12, 11, 10]:
                    is_straight = True
                    high_card = 14
                elif rank_vals == [14, 5, 4, 3, 2]:
                    is_straight = True
                    high_card = 5

            if is_flush and is_straight:
                if high_card == 14 and rank_vals == [14, 13, 12, 11, 10]:
                    hand_value = (9, 14, 0, 0, 0)
                else:
                    hand_value = (8, high_card, 0, 0, 0)
            else:
                four_of_kind_rank = None
                for rank, count in rank_counts.items():
                    if count == 4:
                        four_of_kind_rank = self.hand_rank[rank]
                        break

                if four_of_kind_rank:
                    kicker = max([self.hand_rank[r] for r, c in rank_counts.items() if c == 1])
                    hand_value = (7, four_of_kind_rank, kicker, 0, 0)
                else:
                    three_of_kind_rank = None
                    pair_ranks = []
                    for rank, count in rank_counts.items():
                        if count == 3:
                            three_of_kind_rank = self.hand_rank[rank]
                        elif count == 2:
                            pair_ranks.append(self.hand_rank[rank])

                    if three_of_kind_rank and pair_ranks:
                        pair_ranks.sort(reverse=True)
                        hand_value = (6, three_of_kind_rank, pair_ranks[0], 0, 0)
                    elif is_flush:
                        hand_value = (5, rank_vals[0], rank_vals[1], rank_vals[2], rank_vals[3])
                    elif is_straight:
                        hand_value = (4, high_card, 0, 0, 0)
                    elif three_of_kind_rank:
                        kickers = sorted([self.hand_rank[r] for r, c in rank_counts.items() if c == 1], reverse=True)
                        hand_value = (3, three_of_kind_rank, kickers[0], kickers[1], 0)
                    elif len(pair_ranks) >= 2:
                        pair_ranks.sort(reverse=True)
                        kicker = max([self.hand_rank[r] for r, c in rank_counts.items() if c == 1])
                        hand_value = (2, pair_ranks[0], pair_ranks[1], kicker, 0)
                    elif len(pair_ranks) == 1:
                        kickers = sorted([self.hand_rank[r] for r, c in rank_counts.items() if c == 1], reverse=True)
                        hand_value = (1, pair_ranks[0], kickers[0], kickers[1], kickers[2])
                    else:
                        hand_value = (0, rank_vals[0], rank_vals[1], rank_vals[2], rank_vals[3])

            if hand_value > best_hand:
                best_hand = hand_value

        return best_hand
