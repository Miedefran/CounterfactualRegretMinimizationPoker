class KeyGenerator:
    """
    Centralized authority for generating Information Set Keys.
    Ensures consistency between Training (CFR) and Evaluation (Best Response/Public Tree).
    
    Canonical Key Format:
    (
        private_card: str | Tuple[str],  # e.g., 'Js' (single card) or ('As', 'Ks') (multiple cards)
        public_cards: Tuple[str],         # e.g., ('Qs', 'Kh') or ()
        history: Tuple[str],              # e.g., ('check', 'bet', 'call') - NO '|'
        player_id: int                    # 0 or 1
    )
    """
    
    @staticmethod
    def get_info_set_key(game, player_id: int):

        if hasattr(game.players[player_id], 'private_cards'):
            private_cards = game.players[player_id].private_cards
            private_card = tuple(sorted(private_cards))
        else:
              private_card = game.players[player_id].private_card
        public_cards = ()
        
        # Get Public Card/Cards depending on game
        if hasattr(game, 'public_cards'):
            if game.public_cards is not None:
                public_cards = tuple(game.public_cards)
        elif hasattr(game, 'public_card'):
            if game.public_card is not None and game.public_card != 'None':
                public_cards = (game.public_card,)
        
        #Filter | round marker to avoid key missmatch between training and statetree
        raw_history = game.history
        clean_history = tuple(a for a in raw_history if a in {'bet', 'check', 'call', 'fold'})
        
        return (private_card, public_cards, clean_history, player_id)
