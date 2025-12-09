import os
import json
import pickle
import gzip
def build_public_state_tree(game_class, game_config):
    public_states = {}
    
    # Rekursiv alle Histories durchgehen
    traverse_public_states(game_class, game_config, public_states, history=())
    
    return {
        'public_states': public_states
    }

def traverse_public_states(game_class, game_config, public_states, history=()):
    game = game_class(**game_config)
    game.reset(0)
    
    game_name_str = game.__class__.__name__.lower()
    if 'kuhn' in game_name_str:
        game.players[0].private_card = 'J'
        game.players[1].private_card = 'Q'
    elif 'leduc' in game_name_str:
        game.players[0].private_card = 'Js'
        game.players[1].private_card = 'Qh'
    else:
        game.players[0].private_card = 'Js'
        game.players[1].private_card = 'Qh'
    
    for a in history:
        game.step(a)

    key = tuple(history)
    if key in public_states:
        return

    if game.done:
        game_name_str = game.__class__.__name__.lower()
        player0_info_sets, player1_info_sets = get_info_sets_in_public_state(history, game_name=game_name_str)
        public_states[key] = {
            'type': 'terminal',
            'pot': game.pot,
            'player_bets': list(game.total_bets) if hasattr(game, 'total_bets') else list(game.player_bets),
            'player0_info_sets': player0_info_sets,
            'player1_info_sets': player1_info_sets
        }
        return

    #Check if this is a public chance node
    if game.history and game.history[-1] == '|':
        # Public chance node detected
        history_with_marker = history + ('|',)
        deck = get_public_chance_deck(game)
        children = {}
        for card in deck:
            child_hist = history_with_marker + (card,)
            children[card] = child_hist
            traverse_public_states(game_class, game_config, public_states, child_hist)
        
        player0_info_sets, player1_info_sets = get_info_sets_in_public_state(history_with_marker, game_name=game_name_str)
        public_states[key] = {
            'type': 'chance',
            'pot': game.pot,
            'player_bets': list(game.total_bets) if hasattr(game, 'total_bets') else list(game.player_bets),
            'children': {card: tuple(h) for card, h in children.items()},
            'player0_info_sets': player0_info_sets,
            'player1_info_sets': player1_info_sets
        }
        return


    # 3) Choice Public State
    legal_actions = game.get_legal_actions()
    children = {}
    for a in legal_actions:
        child_hist = history + (a,)
        children[a] = child_hist
        traverse_public_states(game_class, game_config, public_states, child_hist)

    game_name_str = game.__class__.__name__.lower()
    player0_info_sets, player1_info_sets = get_info_sets_in_public_state(history, game_name=game_name_str)
    public_states[key] = {
        'type': 'choice',
        'player': game.current_player,
        'pot': game.pot,
        'player_bets': list(game.total_bets) if hasattr(game, 'total_bets') else list(game.player_bets),
        'children': {a: tuple(h) for a, h in children.items()},
        'player0_info_sets': player0_info_sets,
        'player1_info_sets': player1_info_sets
    }
    
def get_info_sets_in_public_state(history, game_name='kuhn'):
    deck = []
    if game_name.startswith('kuhn'):
        deck = ['J', 'Q', 'K']
        player0_info_sets = [(c, tuple(history), 0) for c in deck]
        player1_info_sets = [(c, tuple(history), 1) for c in deck]
        return player0_info_sets, player1_info_sets
    if game_name == 'leducholdemgame' or game_name.startswith('leduc'):
        deck = ['Js', 'Jh', 'Qs', 'Qh', 'Ks', 'Kh']
        public_card = None
        history_without_cards = []
        
        for item in history:
            if isinstance(item, str) and len(item) == 2 and item[1] in ['s', 'h'] and item[0] in ['J', 'Q', 'K', 'A']:
                public_card = item
                
            else:
                history_without_cards.append(item)
        
        player0_info_sets = [(c, public_card, tuple(history_without_cards), 0) for c in deck]
        player1_info_sets = [(c, public_card, tuple(history_without_cards), 1) for c in deck]
        return player0_info_sets, player1_info_sets
    if 'rhode' in game_name.lower():
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = ['s', 'h', 'd', 'c']
        full_deck = [rank + suit for rank in ranks for suit in suits]
        deck = full_deck.copy()
        public_cards = []
        history_without_cards = []
        for item in history:
            if isinstance(item, str) and len(item) == 2 and item[1] in suits and item[0] in ranks:
                if item in deck:
                    deck.remove(item)
                public_cards.append(item)
            else:
                history_without_cards.append(item)
        player0_info_sets = [(c, tuple(public_cards), tuple(history_without_cards), 0) for c in deck]
        player1_info_sets = [(c, tuple(public_cards), tuple(history_without_cards), 1) for c in deck]
        return player0_info_sets, player1_info_sets
    if 'twelve' in game_name.lower():
        ranks = ['J', 'Q', 'K', 'A']
        suits = ['s', 'h', 'd']
        full_deck = [rank + suit for rank in ranks for suit in suits]
        deck = full_deck.copy()
        public_cards = []
        history_without_cards = []
        for item in history:
            if isinstance(item, str) and len(item) == 2 and item[1] in suits and item[0] in ranks:
                if item in deck:
                    deck.remove(item)
                public_cards.append(item)
            else:
                history_without_cards.append(item)
        player0_info_sets = [(c, tuple(public_cards), tuple(history_without_cards), 0) for c in deck]
        player1_info_sets = [(c, tuple(public_cards), tuple(history_without_cards), 1) for c in deck]
        return player0_info_sets, player1_info_sets
    player0_info_sets = [(c, tuple(history), 0) for c in deck]
    player1_info_sets = [(c, tuple(history), 1) for c in deck]
    return player0_info_sets, player1_info_sets

def get_public_chance_deck(game):
    if 'Leduc' in game.__class__.__name__:
        return ['Js', 'Jh', 'Qs', 'Qh', 'Ks', 'Kh']
    elif 'Rhode' in game.__class__.__name__:
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = ['s', 'h', 'd', 'c']
        return [rank + suit for rank in ranks for suit in suits]
    elif 'Twelve' in game.__class__.__name__:
        ranks = ['J', 'Q', 'K', 'A']
        suits = ['s', 'h', 'd']
        return [rank + suit for rank in ranks for suit in suits]
    return []
    
    
def print_public_state_tree(tree):
    states = tree['public_states']
    def print_node(hist_tuple, visited, prefix="", is_last=True, skip_print=False):
        if hist_tuple in visited:
            return
        visited.add(hist_tuple)
        node = states[hist_tuple]
        hist_str = '|'.join(hist_tuple) if hist_tuple else 'root'
        
        if not skip_print:
            connector = "└─ " if is_last else "├─ "
            print(f"{prefix}{connector}{hist_str}", end="")
        
        if node['type'] == 'choice':
            pot = node.get('pot', 0)
            player_bets = node.get('player_bets', [0, 0])
            print(f" [Player {node['player']}, pot={pot}, bets={player_bets}]")
            children_list = list(node['children'].items())
            for idx, (action, child_hist) in enumerate(children_list):
                is_child_last = idx == len(children_list) - 1
                child_prefix = prefix + ("   " if is_last else "│  ")
                child_hist_str = '|'.join(child_hist)
                print(f"{child_prefix}{'└─ ' if is_child_last else '├─ '}{child_hist_str}", end="")
                print_node(child_hist, visited, child_prefix, is_child_last, skip_print=True)
        elif node['type'] == 'chance':
            pot = node.get('pot', 0)
            player_bets = node.get('player_bets', [0, 0])
            print(f" [CHANCE, pot={pot}, bets={player_bets}]")
            children_list = list(node['children'].items())
            for idx, (action, child_hist) in enumerate(children_list):
                is_child_last = idx == len(children_list) - 1
                child_prefix = prefix + ("   " if is_last else "│  ")
                child_hist_str = '|'.join(child_hist)
                print(f"{child_prefix}{'└─ ' if is_child_last else '├─ '}{child_hist_str}", end="")
                print_node(child_hist, visited, child_prefix, is_child_last, skip_print=True)
        else:
            num_player0 = len(node['player0_info_sets'])
            num_player1 = len(node['player1_info_sets'])
            pot = node.get('pot', 0)
            player_bets = node.get('player_bets', [0, 0])
            print(f" [TERMINAL: {num_player0} player0, {num_player1} player1 info sets, pot={pot}, bets={player_bets}]")
    
    print_node((), set(), "", True)

def save_public_state_tree(game_name, tree):
    base_dir = os.path.join('evaluation', 'public_state_trees')
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, f"{game_name}_public_tree.pkl.gz")
    with gzip.open(path, 'wb') as f:
        pickle.dump(tree, f)
    file_size = os.path.getsize(path)
    print(f"file size: {file_size / 1024:.2f} KB")
    return path


if __name__ == "__main__":
    import sys
    from envs.kuhn_poker.game import KuhnPokerGame
    from envs.leduc_holdem.game import LeducHoldemGame
    from envs.rhode_island.game import RhodeIslandGame
    from envs.twelve_card_poker.game import TwelveCardPokerGame
    from utils.poker_utils import GAME_CONFIGS

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
    elif game_name == 'twelve_card':
        game_class = TwelveCardPokerGame
        game_config = {'ante': 1, 'bet_sizes': [2, 4, 8], 'bet_limit': 2}
        save_name = 'twelve_card'
    else:
        print(f"Unknown game: {game_name}")
        sys.exit(1)

    tree = build_public_state_tree(game_class, game_config)
    states = tree['public_states']
    num_choice = sum(1 for s in states.values() if s['type'] == 'choice')
    num_chance = sum(1 for s in states.values() if s['type'] == 'chance')
    num_terminal = sum(1 for s in states.values() if s['type'] == 'terminal')
    print({'total_states': len(states), 'choice_states': num_choice, 'chance_states': num_chance, 'terminal_states': num_terminal})
    path = save_public_state_tree(save_name, tree)
    print_public_state_tree(tree)
    print(f"saved: {path}")
