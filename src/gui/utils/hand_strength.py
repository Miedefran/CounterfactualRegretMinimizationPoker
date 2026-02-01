from __future__ import annotations

from types import SimpleNamespace
from typing import Iterable, List, Optional


def hand_strength_text(
        private_cards: Iterable[str] | None,
        public_cards: Iterable[str] | None,
        game: Optional[str] = None,
) -> str:
    """
    Returns a short human-readable hand category for the given cards.

    Notes:
    - Implemented inside gui/ on purpose (no edits outside gui/).
    - Reuses existing env Judger.evaluate_hand() implementations.
    """
    priv = _normalize_cards(private_cards)
    pub = _normalize_cards(public_cards)

    if not priv:
        return ""

    # If the server tells us the game explicitly, we can avoid heuristics.
    if game is not None:
        # Kuhn-style (rank only: "J", "Q", "K")
        if game == "kuhn":
            if len(priv) == 1:
                return f"Card: {priv[0]}"
            return ""

        # Leduc: 1 private + 1 public. Preflop: show nothing.
        if game == "leduc":
            if len(priv) != 1 or len(pub) != 1:
                return ""
            from envs.leduc_holdem.judger import LeducHoldemJudger

            judger = LeducHoldemJudger()
            dummy = SimpleNamespace(private_card=priv[0], public_card=pub[0])
            value = judger.evaluate_hand(dummy)
            return {1: "Pair", 0: "High Card"}.get(_first(value), "High Card")

        # Twelve-card poker: 1 private + >=1 public. Preflop: show nothing.
        if game == "twelve_card":
            if len(priv) != 1 or len(pub) < 1:
                return ""
            from envs.twelve_card_poker.judger import TwelveCardPokerJudger

            judger = TwelveCardPokerJudger()
            dummy = SimpleNamespace(private_card=priv[0], public_cards=pub)
            value = judger.evaluate_hand(dummy)
            return {2: "Three of a Kind", 1: "Pair", 0: "High Card"}.get(_first(value), "High Card")

        # Rhode Island: 1 private + (1..2) public. Preflop: show nothing.
        if game == "rhode_island":
            if len(priv) != 1 or len(pub) < 1:
                return ""
            from envs.rhode_island.judger import RhodeIslandJudger

            judger = RhodeIslandJudger()
            dummy = SimpleNamespace(private_card=priv[0], public_cards=pub)
            value = judger.evaluate_hand(dummy)
            return {5: "Straight Flush", 4: "Three of a Kind", 3: "Straight", 2: "Flush", 1: "Pair",
                    0: "High Card"}.get(
                _first(value), "High Card"
            )

        # Hold'em-like variants (2 private cards). Preflop: show nothing.
        if game in {"royal_holdem", "limit_holdem"}:
            if len(priv) != 2 or len(pub) < 3:
                return ""

            if game == "royal_holdem":
                from envs.royal_holdem.judger import RoyalHoldemJudger

                judger = RoyalHoldemJudger()
                dummy = SimpleNamespace(private_cards=priv, public_cards=pub)
                value = judger.evaluate_hand(dummy)
                return _holdem_category_from_value(value)

            from envs.limit_holdem.judger import LimitHoldemJudger

            judger = LimitHoldemJudger()
            dummy = SimpleNamespace(private_cards=priv, public_cards=pub)
            value = judger.evaluate_hand(dummy)
            return _holdem_category_from_value(value)

        # Unknown/unsupported game id: don't guess.
        return ""

    # Kuhn-style (rank only: "J", "Q", "K")
    if all(isinstance(c, str) and len(c) == 1 for c in priv) and all(isinstance(c, str) and len(c) == 1 for c in pub):
        if len(priv) == 1:
            return f"Card: {priv[0]}"
        return "Cards: " + " ".join(priv)

    # Hold'em-like variants (2 private cards)
    if len(priv) == 2:
        # Distinguish Royal (T-A only) vs full-deck (Limit)
        rank_set = {c[0] for c in (priv + pub) if isinstance(c, str) and len(c) >= 1}
        is_royal_deck = rank_set.issubset({"T", "J", "Q", "K", "A"})

        if is_royal_deck:
            from envs.royal_holdem.judger import RoyalHoldemJudger

            judger = RoyalHoldemJudger()
            dummy = SimpleNamespace(private_cards=priv, public_cards=pub)
            value = judger.evaluate_hand(dummy)
            return _holdem_category_from_value(value)

        from envs.limit_holdem.judger import LimitHoldemJudger

        judger = LimitHoldemJudger()
        dummy = SimpleNamespace(private_cards=priv, public_cards=pub)
        value = judger.evaluate_hand(dummy)
        return _holdem_category_from_value(value)

    # Single-private-card variants
    if len(priv) == 1:
        # Preflop / not enough public cards: show nothing (avoid calling judgers).
        if len(pub) == 0:
            return ""

        # Leduc (exactly 1 public card)
        if len(pub) == 1:
            from envs.leduc_holdem.judger import LeducHoldemJudger

            judger = LeducHoldemJudger()
            dummy = SimpleNamespace(private_card=priv[0], public_card=pub[0])
            value = judger.evaluate_hand(dummy)
            return {1: "Pair", 0: "High Card"}.get(_first(value), "High Card")

        # Twelve-card poker (ranks JQKA, suits typically s/h/d)
        ranks = {c[0] for c in (priv + pub) if isinstance(c, str) and len(c) >= 1}
        suits = {c[1] for c in (priv + pub) if isinstance(c, str) and len(c) >= 2}
        if len(pub) >= 1 and ranks.issubset({"J", "Q", "K", "A"}) and suits.issubset({"s", "h", "d"}):
            from envs.twelve_card_poker.judger import TwelveCardPokerJudger

            judger = TwelveCardPokerJudger()
            dummy = SimpleNamespace(private_card=priv[0], public_cards=pub)
            value = judger.evaluate_hand(dummy)
            return {2: "Three of a Kind", 1: "Pair", 0: "High Card"}.get(_first(value), "High Card")

        # Rhode Island (up to 2 public cards)
        from envs.rhode_island.judger import RhodeIslandJudger

        judger = RhodeIslandJudger()
        dummy = SimpleNamespace(private_card=priv[0], public_cards=pub)
        value = judger.evaluate_hand(dummy)
        return {4: "Three of a Kind", 3: "Straight", 2: "Flush", 1: "Pair", 0: "High Card"}.get(
            _first(value), "High Card"
        )

    return ""


def _normalize_cards(cards: Iterable[str] | None) -> List[str]:
    if not cards:
        return []
    if isinstance(cards, str):
        return [cards]
    return [c for c in cards if c]


def _first(value) -> int:
    if isinstance(value, tuple) and len(value) > 0:
        return int(value[0])
    return 0


def _holdem_category_from_value(value) -> str:
    # For Hold'em-like judgers, a "not enough cards" result is (0,0,0,0,0).
    if isinstance(value, tuple) and len(value) >= 5 and all(v == 0 for v in value):
        return "Preflop"

    return {
        9: "Royal Flush",
        8: "Straight Flush",
        7: "Four of a Kind",
        6: "Full House",
        5: "Flush",
        4: "Straight",
        3: "Three of a Kind",
        2: "Two Pair",
        1: "One Pair",
        0: "High Card",
    }.get(_first(value), "High Card")
