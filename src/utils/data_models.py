def _card_to_rank(card):
    """Reduce single card (e.g. 'Js', 'Jh') to rank (e.g. 'J'). Already rank-only (e.g. 'J') unchanged."""
    if card is None:
        return None
    s = str(card).strip()
    if len(s) <= 1:
        return s
    return s[0]


def infoset_key_to_suit_abstracted(key):
    """
    Convert an infoset key with full card designations (e.g. 'Js', 'Jh')
    to suit-abstracted form (rank only, e.g. 'J'), as used during training with
    abstract_suits=True.

    This allows a strategy trained with suit abstraction to be correctly looked up
    in the GUI (without abstraction).

    Args:
        key: (private_card, public_cards, history, player_id)
             private_card: str or Tuple[str], public_cards: Tuple[str]

    Returns:
        Key in the same format, but all cards replaced by their rank.
    """
    private_card, public_cards, history, player_id = key
    if isinstance(private_card, tuple):
        priv_abs = tuple(sorted(_card_to_rank(c) for c in private_card))
    else:
        priv_abs = _card_to_rank(private_card)
    pub_abs = tuple(_card_to_rank(c) for c in (public_cards or ()))
    return (priv_abs, pub_abs, history, player_id)


class KeyGenerator:
    """
    Generate uniform information-set keys for training (CFR) and evaluation (Best Response/Public Tree).
    Canonical format: (private_card, public_cards, history, player_id);
    history without '|', private_card str or Tuple[str], public_cards Tuple[str].
    """

    @staticmethod
    def get_info_set_key(game, player_id: int):

        if hasattr(game.players[player_id], 'private_cards'):
            private_cards = game.players[player_id].private_cards
            private_card = tuple(sorted(private_cards))
        else:
            private_card = game.players[player_id].private_card
        public_cards = ()

        # Public card(s) depending on game
        if hasattr(game, 'public_cards'):
            if game.public_cards is not None:
                public_cards = tuple(game.public_cards)
        elif hasattr(game, 'public_card'):
            if game.public_card is not None and game.public_card != 'None':
                public_cards = (game.public_card,)

        # Filter history to actions (no round separators) for uniform keys (Training/State-Tree)
        raw_history = game.history
        clean_history = tuple(a for a in raw_history if a in {'bet', 'check', 'call', 'fold'})

        return (private_card, public_cards, clean_history, player_id)
