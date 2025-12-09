import os
import pickle
import gzip
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.poker_utils import GAME_CONFIGS

def build_public_state_tree(game_class, game_config):
    public_states = {}
    
    def get_all_cards_for_game(game):
        game_name = game.__class__.__name__
        if 'Kuhn' in game_name:
            return ['J', 'Q', 'K']
        elif 'Leduc' in game_name:
            return ['Js', 'Jh', 'Qs', 'Qh', 'Ks', 'Kh']
        elif 'Rhode' in game_name:
            ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
            suits = ['s', 'h', 'd', 'c']
            return [rank + suit for rank in ranks for suit in suits]
        elif 'Twelve' in game_name:
            ranks = ['J', 'Q', 'K', 'A']
            suits = ['s', 'h', 'd']
            return [rank + suit for rank in ranks for suit in suits]
        return []
    
    def extract_public_history(history):
        public = []
        for item in history:
            if item != '|' and item not in get_all_cards_for_game(type('Game', (), {})()):
                if isinstance(item, str) and len(item) == 2 and item[1] in ['s', 'h', 'd', 'c']:
                    continue
                public.append(item)
        return public
    
    def get_info_sets_for_public_state(public_hist, game):
        game_name = game.__class__.__name__
        all_cards = get_all_cards_for_game(game)
        
        public_cards = []
        clean_history = []
        for item in public_hist:
            if isinstance(item, str) and len(item) >= 2 and item in all_cards:
                public_cards.append(item)
            else:
                clean_history.append(item)
        
        available_cards = [card for card in all_cards if card not in public_cards]
        
        if 'Kuhn' in game_name:
            player0_info_sets = [(card, tuple(clean_history), 0) for card in available_cards]
            player1_info_sets = [(card, tuple(clean_history), 1) for card in available_cards]
        elif 'Leduc' in game_name:
            public_card = public_cards[0] if public_cards else 'None'
            player0_info_sets = [(card, public_card, tuple(clean_history), 0) for card in available_cards]
            player1_info_sets = [(card, public_card, tuple(clean_history), 1) for card in available_cards]
        elif 'Rhode' in game_name or 'Twelve' in game_name:
            player0_info_sets = [(card, tuple(public_cards), tuple(clean_history), 0) for card in available_cards]
            player1_info_sets = [(card, tuple(public_cards), tuple(clean_history), 1) for card in available_cards]
        else:
            player0_info_sets = [(card, tuple(clean_history), 0) for card in available_cards]
            player1_info_sets = [(card, tuple(clean_history), 1) for card in available_cards]
        
        return player0_info_sets, player1_info_sets
    
    def set_dummy_cards(game):
        game_name = game.__class__.__name__
        all_cards = get_all_cards_for_game(game)
        if all_cards:
            game.players[0].private_card = all_cards[0]
            game.players[1].private_card = all_cards[1] if len(all_cards) > 1 else all_cards[0]
    
    def traverse(game, public_hist):
        public_key = tuple(public_hist)
        
        if public_key in public_states:
            return
        
        game_name = game.__class__.__name__
        public_cards_in_history = [item for item in public_hist if isinstance(item, str) and len(item) == 2 and item[1] in ['s', 'h', 'd', 'c']]
        has_marker = '|' in public_hist
        
        is_chance = False
        chance_outcomes = []
        
        if hasattr(game, 'round') and hasattr(game.round, 'is_round_complete'):
            if 'Leduc' in game_name:
                if not has_marker and len(public_cards_in_history) == 0:
                    temp_check = game_class(**game_config)
                    temp_check.reset(0)
                    set_dummy_cards(temp_check)
                    passive_count = 0
                    betting_round_at_start = temp_check.betting_round
                    
                    for a in public_hist:
                        if a in ['check', 'call']:
                            passive_count += 1
                        elif a == 'bet':
                            passive_count = 0
                        temp_check.step(a)
                    
                    if passive_count >= 2 and betting_round_at_start == 0 and temp_check.betting_round == 1:
                        is_chance = True
                        chance_outcomes = ['Js', 'Jh', 'Qs', 'Qh', 'Ks', 'Kh']
            elif 'Rhode' in game_name or 'Twelve' in game_name:
                if not has_marker:
                    temp_check = game_class(**game_config)
                    temp_check.reset(0)
                    set_dummy_cards(temp_check)
                    for a in public_hist:
                        temp_check.step(a)
                    if temp_check.round.is_round_complete() and not temp_check.done and temp_check.betting_round < 2:
                        is_chance = True
                        if 'Rhode' in game_name:
                            ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
                            suits = ['s', 'h', 'd', 'c']
                            chance_outcomes = [rank + suit for rank in ranks for suit in suits]
                        else:
                            ranks = ['J', 'Q', 'K', 'A']
                            suits = ['s', 'h', 'd']
                            chance_outcomes = [rank + suit for rank in ranks for suit in suits]
        
        if is_chance:
            children = {}
            for outcome in chance_outcomes:
                child_hist = list(public_hist) + [outcome]
                child_key = tuple(child_hist)
                
                game_new = game_class(**game_config)
                game_new.reset(0)
                set_dummy_cards(game_new)
                for a in public_hist:
                    game_new.step(a)
                game_new.step('|')
                if 'Leduc' in game_name:
                    game_new.public_card = outcome
                    if hasattr(game_new.players[0], 'set_public_card'):
                        game_new.players[0].set_public_card(outcome)
                        game_new.players[1].set_public_card(outcome)
                elif 'Rhode' in game_name or 'Twelve' in game_name:
                    if not hasattr(game_new, 'public_cards'):
                        game_new.public_cards = []
                    game_new.public_cards.append(outcome)
                    game_new.players[0].public_cards.append(outcome)
                    game_new.players[1].public_cards.append(outcome)
                
                children[outcome] = child_key
                traverse(game_new, child_hist)
            
            temp_for_info_sets = game_class(**game_config)
            temp_for_info_sets.reset(0)
            set_dummy_cards(temp_for_info_sets)
            for a in public_hist:
                temp_for_info_sets.step(a)
            
            player0_info_sets, player1_info_sets = get_info_sets_for_public_state(public_hist, temp_for_info_sets)
            player_bets = temp_for_info_sets.total_bets if hasattr(temp_for_info_sets, 'total_bets') else temp_for_info_sets.player_bets
            
            public_states[public_key] = {
                'type': 'chance',
                'pot': temp_for_info_sets.pot,
                'player_bets': list(player_bets),
                'children': children,
                'player0_info_sets': player0_info_sets,
                'player1_info_sets': player1_info_sets
            }
            return
        
        set_dummy_cards(game)
        
        if game.done:
            player0_info_sets, player1_info_sets = get_info_sets_for_public_state(public_hist, game)
            player_bets = game.total_bets if hasattr(game, 'total_bets') else game.player_bets
            
            public_states[public_key] = {
                'type': 'terminal',
                'pot': game.pot,
                'player_bets': list(player_bets),
                'player0_info_sets': player0_info_sets,
                'player1_info_sets': player1_info_sets
            }
            return
        
        
        legal_actions = game.get_legal_actions()
        
        if not legal_actions:
            player0_info_sets, player1_info_sets = get_info_sets_for_public_state(public_hist, game)
            player_bets = game.total_bets if hasattr(game, 'total_bets') else game.player_bets
            
            public_states[public_key] = {
                'type': 'terminal',
                'pot': game.pot,
                'player_bets': list(player_bets),
                'player0_info_sets': player0_info_sets,
                'player1_info_sets': player1_info_sets
            }
            return
        
        children = {}
        for action in legal_actions:
            child_hist = list(public_hist) + [action]
            child_key = tuple(child_hist)
            
            game_new = game_class(**game_config)
            game_new.reset(0)
            set_dummy_cards(game_new)
            for a in child_hist:
                game_new.step(a)
            
            children[action] = child_key
            traverse(game_new, child_hist)
        
        player0_info_sets, player1_info_sets = get_info_sets_for_public_state(public_hist, game)
        player_bets = game.total_bets if hasattr(game, 'total_bets') else game.player_bets
        
        public_states[public_key] = {
            'type': 'choice',
            'player': game.current_player,
            'pot': game.pot,
            'player_bets': list(player_bets),
            'children': children,
            'player0_info_sets': player0_info_sets,
            'player1_info_sets': player1_info_sets
        }
    
    game = game_class(**game_config)
    game.reset(0)
    set_dummy_cards(game)
    traverse(game, [])
    
    return {'public_states': public_states}


def save_public_state_tree(game_name, tree, output_dir='evaluation/public_state_trees'):
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{game_name}_public_tree_v2.pkl.gz"
    path = os.path.join(output_dir, filename)
    with gzip.open(path, 'wb') as f:
        pickle.dump(tree, f)
    print(f"Saved: {path}")
    return path


if __name__ == "__main__":
    from envs.kuhn_poker.game import KuhnPokerGame
    from envs.leduc_holdem.game import LeducHoldemGame
    from envs.rhode_island.game import RhodeIslandGame
    
    game_name = sys.argv[1] if len(sys.argv) > 1 else 'kuhn_case2'
    
    if game_name.startswith('kuhn'):
        game_class = KuhnPokerGame
        game_config = GAME_CONFIGS[game_name]
        save_name = 'kuhn'
    elif game_name == 'leduc':
        game_class = LeducHoldemGame
        game_config = GAME_CONFIGS['leduc']
        save_name = 'leduc'
    elif game_name == 'rhode_island':
        game_class = RhodeIslandGame
        game_config = GAME_CONFIGS['rhode_island']
        save_name = 'rhode_island'
    else:
        print(f"Unknown game: {game_name}")
        sys.exit(1)
    
    tree = build_public_state_tree(game_class, game_config)
    states = tree['public_states']
    
    num_choice = sum(1 for s in states.values() if s['type'] == 'choice')
    num_chance = sum(1 for s in states.values() if s['type'] == 'chance')
    num_terminal = sum(1 for s in states.values() if s['type'] == 'terminal')
    
    print(f"Total public states: {len(states)}")
    print(f"Choice nodes: {num_choice}, Chance nodes: {num_chance}, Terminal nodes: {num_terminal}")
    
    save_public_state_tree(save_name, tree)
