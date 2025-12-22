import collections
import functools


def _memoize_method(key_fn=lambda x: x):
    def memoizer(method):
        cache_name = "cache_" + method.__name__

        def wrap(self, arg):
            key = key_fn(arg)
            cache = vars(self).setdefault(cache_name, {})
            if key not in cache:
                cache[key] = method(self, arg)
            return cache[key]

        return wrap
    return memoizer


def _state_history_key(game):
    return tuple(game.history)


class BestResponsePolicy:
    def __init__(self, game_class, game_config, player_id, policy, combination_generator=None, cut_threshold=0.0):
        self.game_class = game_class
        self.game_config = game_config
        self._player_id = player_id
        self._policy = policy
        self._cut_threshold = cut_threshold
        self._combination_generator = combination_generator
        
        root_game = self._create_game()
        root_game.reset(0)
        self._root_state = root_game
        
        self.infosets = self.info_sets(root_game)
    
    def _create_game(self):
        return self.game_class(**self.game_config)
    
    def _clone_game(self, game):
        if hasattr(game, 'save_state') and hasattr(game, 'restore_state'):
            new_game = self._create_game()
            saved_state = game.save_state()
            new_game.reset(saved_state.get('starting_player', 0) if hasattr(game, 'starting_player') else 0)
            
            if 'hand_p0' in saved_state:
                new_game.players[0].private_card = saved_state['hand_p0']
            if 'hand_p1' in saved_state:
                new_game.players[1].private_card = saved_state['hand_p1']
            
            for action in game.history:
                if action != '|':
                    new_game.step(action)
                else:
                    if hasattr(new_game, 'deal_public_card'):
                        new_game.deal_public_card()
                    new_game.history.append('|')
            
            if hasattr(game, 'restore_state'):
                new_game.restore_state(saved_state)
            return new_game
        
        new_game = self._create_game()
        starting_player = game.starting_player if hasattr(game, 'starting_player') else 0
        new_game.reset(starting_player)
        
        if hasattr(game, 'players'):
            new_game.players[0].private_card = game.players[0].private_card
            new_game.players[1].private_card = game.players[1].private_card
        
        for action in game.history:
            if action != '|':
                new_game.step(action)
            else:
                if hasattr(new_game, 'deal_public_card'):
                    new_game.deal_public_card()
                new_game.history.append('|')
        
        return new_game
    
    def info_sets(self, game):
        infosets = collections.defaultdict(list)
        for state, cf_prob in self.decision_nodes(game):
            info_state = self._get_information_state(state)
            infosets[info_state].append((state, cf_prob))
        return dict(infosets)
    
    def decision_nodes(self, parent_game):
        if parent_game.done:
            return
        
        if self._is_initial_card_distribution_node(parent_game):
            for action, p_action in self.transitions(parent_game):
                child_game = self._apply_action(parent_game, action)
                if child_game:
                    for state, p_state in self.decision_nodes(child_game):
                        yield (state, p_state * p_action)
            return
        
        if parent_game.current_player == self._player_id:
            if hasattr(parent_game, 'players') and parent_game.players:
                if hasattr(parent_game.players[self._player_id], 'private_card'):
                    if parent_game.players[self._player_id].private_card is not None:
                        yield (parent_game, 1.0)
        
        for action, p_action in self.transitions(parent_game):
            child_game = self._apply_action(parent_game, action)
            if child_game:
                for state, p_state in self.decision_nodes(child_game):
                    yield (state, p_state * p_action)
    
    def transitions(self, game):
        if game.done:
            return []
        
        if self._is_initial_card_distribution_node(game):
            return self._initial_card_distributions(game)
        
        if game.current_player == self._player_id:
            legal_actions = game.get_legal_actions()
            return [(action, 1.0) for action in legal_actions]
        
        if self._is_chance_node(game):
            return self._chance_outcomes(game)
        
        return self._policy_transitions(game)
    
    def _is_initial_card_distribution_node(self, game):
        if self._combination_generator is None:
            return False
        if hasattr(game, 'players') and game.players:
            if hasattr(game.players[0], 'private_card') and game.players[0].private_card is None:
                return True
        return False
    
    def _is_chance_node(self, game):
        if hasattr(game, 'history') and game.history:
            return game.history[-1] == '|'
        return False
    
    def _initial_card_distributions(self, game):
        if not self._combination_generator:
            return []
        
        combinations = self._combination_generator.get_all_combinations()
        num_combinations = len(combinations)
        if num_combinations == 0:
            return []
        
        prob = 1.0 / num_combinations
        return [(combination, prob) for combination in combinations]
    
    def _chance_outcomes(self, game):
        if not self._is_chance_node(game):
            return []
        
        if hasattr(game, 'public_card') and game.public_card is None:
            if hasattr(game, 'dealer') and hasattr(game.dealer, 'deck'):
                deck = game.dealer.deck
                num_cards = len(deck)
                if num_cards > 0:
                    prob = 1.0 / num_cards
                    outcomes = []
                    for card in deck:
                        outcomes.append((card, prob))
                    return outcomes
        
        return []
    
    def _policy_transitions(self, game):
        state_dict = game.get_state(game.current_player)
        info_set_key = game.get_info_set_key(game.current_player)
        
        if info_set_key in self._policy:
            action_probs = self._policy[info_set_key]
            legal_actions = game.get_legal_actions()
            result = []
            for action in legal_actions:
                prob = action_probs.get(action, 0.0)
                result.append((action, prob))
            return result
        else:
            legal_actions = game.get_legal_actions()
            uniform_prob = 1.0 / len(legal_actions) if legal_actions else 0.0
            return [(action, uniform_prob) for action in legal_actions]
    
    def _apply_action(self, game, action):
        cloned_game = self._clone_game(game)
        
        if isinstance(action, tuple) and self._is_initial_card_distribution_node(cloned_game):
            if self._combination_generator:
                self._combination_generator.setup_game_with_combination(cloned_game, action)
                return cloned_game
            return None
        
        if self._is_chance_node(cloned_game):
            if hasattr(cloned_game, 'public_card') and cloned_game.public_card is None:
                if hasattr(cloned_game, 'dealer') and action in cloned_game.dealer.deck:
                    cloned_game.public_card = action
                    cloned_game.players[0].set_public_card(action)
                    cloned_game.players[1].set_public_card(action)
                    cloned_game.dealer.deck.remove(action)
                    return cloned_game
            return None
        
        cloned_game.step(action)
        return cloned_game
    
    def _get_information_state(self, game):
        return game.get_info_set_key(self._player_id)
    
    @_memoize_method(key_fn=_state_history_key)
    def value(self, game):
        if game.done:
            return game.get_payoff(self._player_id)
        
        if self._is_initial_card_distribution_node(game):
            total = 0.0
            for action, prob in self.transitions(game):
                if prob > self._cut_threshold:
                    total += prob * self.q_value(game, action)
            return total
        
        if self._is_chance_node(game):
            total = 0.0
            for action, prob in self.transitions(game):
                if prob > self._cut_threshold:
                    total += prob * self.q_value(game, action)
            return total
        
        if game.current_player == self._player_id:
            info_state = self._get_information_state(game)
            action = self.best_response_action(info_state)
            if action is None:
                return 0.0
            return self.q_value(game, action)
        
        total = 0.0
        for action, prob in self.transitions(game):
            if prob > self._cut_threshold:
                total += prob * self.q_value(game, action)
        return total
    
    def q_value(self, game, action):
        child_game = self._apply_action(game, action)
        if child_game:
            return self.value(child_game)
        return 0.0
    
    @_memoize_method()
    def best_response_action(self, infostate):
        if infostate not in self.infosets:
            return None
        
        infoset = self.infosets[infostate]
        
        if not infoset:
            return None
        
        first_state = infoset[0][0]
        legal_actions = first_state.get_legal_actions()
        
        if not legal_actions:
            return None
        
        best_action = None
        best_value = float('-inf')
        
        for action in legal_actions:
            action_value = 0.0
            for state, cf_prob in infoset:
                if cf_prob > self._cut_threshold:
                    q_val = self.q_value(state, action)
                    action_value += cf_prob * q_val
            
            if action_value > best_value:
                best_value = action_value
                best_action = action
        
        return best_action
    
    def action_probabilities(self, game, player_id=None):
        if player_id is None:
            player_id = game.current_player
        
        if player_id != self._player_id:
            state_dict = game.get_state(player_id)
            info_set_key = game.get_info_set_key(player_id)
            
            if info_set_key in self._policy:
                return self._policy[info_set_key]
            else:
                legal_actions = game.get_legal_actions()
                uniform_prob = 1.0 / len(legal_actions) if legal_actions else 0.0
                return {action: uniform_prob for action in legal_actions}
        
        info_state = self._get_information_state(game)
        best_action = self.best_response_action(info_state)
        
        if best_action is None:
            legal_actions = game.get_legal_actions()
            return {action: 0.0 for action in legal_actions}
        
        legal_actions = game.get_legal_actions()
        probs = {action: 0.0 for action in legal_actions}
        if best_action in probs:
            probs[best_action] = 1.0
        return probs
    
    def get_action(self, game):
        state_dict = game.get_state(self._player_id)
        info_state = self._get_information_state(game)
        return self.best_response_action(info_state)

