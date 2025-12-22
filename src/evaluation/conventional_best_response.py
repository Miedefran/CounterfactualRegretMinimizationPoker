import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BestResponse:
    def __init__(self, game_class, game_config, best_responder, policy):
        self.game_class = game_class
        self.game_config = game_config
        self.best_responder = best_responder
        self.policy = policy
        
        root_state = self._create_root_state()
        self.tree = self._build_history_tree(root_state)
        self.value_cache = {}
        self.best_response_policy = {}
    
    def _create_root_state(self):
        game = self.game_class(**self.game_config)
        game.reset(0)
        return game
    
    def _history_to_string(self, state):
        return ','.join(str(a) for a in state.history)
    
    def _get_state_type(self, state):
        if state.done:
            return 'terminal'
        if state.history and state.history[-1] == '|':
            return 'chance'
        return 'decision'
    
    def _apply_action(self, state, action):
        new_state = self.game_class(**self.game_config)
        new_state.reset(state.starting_player if hasattr(state, 'starting_player') else 0)
        for a in state.history:
            new_state.step(a)
        new_state.step(action)
        return new_state
    
    def _build_history_tree(self, root_state):
        tree = {}
        queue = [(root_state, "")]
        
        while queue:
            state, _ = queue.pop(0)
            history_str = self._history_to_string(state)
            
            if history_str in tree:
                continue
            
            tree[history_str] = {
                'state': state,
                'type': self._get_state_type(state),
                'children': {}
            }
            
            if not state.done:
                for action in state.get_legal_actions():
                    child_state = self._apply_action(state, action)
                    child_history = self._history_to_string(child_state)
                    tree[history_str]['children'][action] = child_history
                    queue.append((child_state, child_history))
        
        return tree

