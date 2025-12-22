import os
import pickle
import gzip
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.kuhn_poker.game import KuhnPokerGame
from envs.leduc_holdem.game import LeducHoldemGame
from envs.rhode_island.game import RhodeIslandGame
from envs.royal_holdem.game import RoyalHoldemGame
from envs.limit_holdem.game import LimitHoldemGame
from envs.twelve_card_poker.game import TwelveCardPokerGame
from utils.poker_utils import GAME_CONFIGS
from utils.data_models import KeyGenerator

def build_public_state_tree(game_class, game_config, progress_interval=1000, use_cache=True):
    """
    Baut den Public State Tree.
    
    Args:
        game_class: Die Game-Klasse
        game_config: Die Game-Konfiguration
        progress_interval: Alle N States wird der Fortschritt ausgegeben (default: 1000)
        use_cache: Wenn True, werden Game States gecacht für bessere Performance (default: True)
    """
    public_states = {}
    states_visited = 0
    stats = {'chance': 0, 'choice': 0, 'terminal': 0}
    
    # Cache für Game States: public_hist -> saved_state
    # Nutzt die native save_state/restore_state Funktion der Game-Klassen
    game_state_cache = {} if use_cache else None
    
    def get_all_cards_for_game(game):
        dealer_class = game.dealer.__class__
        dealer = dealer_class()
        dealer.reset()
        return list(dealer.deck)
    
    "Get all Infosets for a public state"
    def get_info_sets_simulated(game, player_id):
        all_cards = get_all_cards_for_game(game)
        
        # Determine public cards from the game state to filter availability
        current_public = []
        if hasattr(game, 'public_cards') and game.public_cards:
            current_public = list(game.public_cards)
        elif hasattr(game, 'public_card') and game.public_card and game.public_card != 'None':
            current_public = [game.public_card]
        
        #Get all available cards except public card
        available_cards = [c for c in all_cards if c not in current_public]
        
        # Save original state with dummy card 
        original_card = game.players[player_id].private_card
        
        #Generate Infosets for all available private cards
        info_sets = []
        for card in available_cards:
            game.players[player_id].private_card = card
            
            # Generate key via authoritative source
            key = KeyGenerator.get_info_set_key(game, player_id)
            info_sets.append(key)
            
        # Restore state with dummy card
        game.players[player_id].private_card = original_card
        
        return info_sets

    def set_dummy_cards(game):
        game_name = game.__class__.__name__
        all_cards = get_all_cards_for_game(game)
        if all_cards:
            game.players[0].private_card = all_cards[0]
            game.players[1].private_card = all_cards[1] if len(all_cards) > 1 else all_cards[0]
            # Ensure no crash on get_info_set_key before we swap cards in simulation
    
    "Construct Game state from public history"
    def replay_public_history(game, public_hist):
        public_hist_key = tuple(public_hist)
        
        # Prüfe Cache: Wenn wir diesen State schon mal hatten, restore ihn
        if use_cache and game_state_cache is not None and public_hist_key in game_state_cache:
            if hasattr(game, 'restore_state'):
                game.restore_state(game_state_cache[public_hist_key])
                # Stelle sicher, dass Dummy-Cards gesetzt sind (für Info-Set-Generierung)
                set_dummy_cards(game)
                return
            # Fallback falls restore_state nicht verfügbar
            # (sollte nicht passieren, aber sicherheitshalber)
        
        # State nicht im Cache: Replay die History
        valid_actions = {'bet', 'check', 'call', 'fold'}
        game_name = game.__class__.__name__
        
        for item in public_hist:
            if item in valid_actions:
                game.step(item)
            else:
                # It is a chance outcome (public card)
                if 'Leduc' in game_name:
                    if hasattr(game, 'public_card'):
                        game.public_card = item
                        game.players[0].set_public_card(item)
                        game.players[1].set_public_card(item)
                
                elif 'Rhode' in game_name or 'Twelve' in game_name or 'Royal' in game_name or 'Limit' in game_name:
                    
                    if hasattr(game, 'public_cards'):
                        #If step added card, overwrite it / if not add it to list
                        if item not in game.public_cards:
                            if len(game.public_cards) > 0:
                                # Overwrite card added by step
                                game.public_cards[-1] = item
                                game.players[0].public_cards[-1] = item
                                game.players[1].public_cards[-1] = item
                            else:
                                # list empty, add card
                                game.public_cards.append(item)
                                game.players[0].public_cards.append(item)
                                game.players[1].public_cards.append(item)
        
        # Speichere den State im Cache für zukünftige Verwendung
        # WICHTIG: Speichere NACH set_dummy_cards, damit die Dummy-Cards im Cache sind
        if use_cache and game_state_cache is not None and hasattr(game, 'save_state'):
            game_state_cache[public_hist_key] = game.save_state()

    def traverse(game, public_hist):
        nonlocal states_visited, stats
        
        public_key = tuple(public_hist)
        
        if public_key in public_states:
            return
        
        states_visited += 1
        if states_visited % progress_interval == 0:
            print(f"  Progress: {states_visited} states visited | "
                  f"Chance: {stats['chance']}, Choice: {stats['choice']}, Terminal: {stats['terminal']}")
        
        game_name = game.__class__.__name__
        public_cards_in_history = [item for item in public_hist if isinstance(item, str) and len(item) == 2 and item[1] in ['s', 'h', 'd', 'c']]
        has_marker = '|' in public_hist
        
        is_chance = False
        chance_outcomes = []
        
        # Chance Node Detection Logic (Game-Specific)
        if hasattr(game, 'round') and hasattr(game.round, 'is_round_complete'):
            if 'Leduc' in game_name:
                if not has_marker and len(public_cards_in_history) == 0:
                    temp_check = game_class(**game_config)
                    temp_check.reset(0)
                    set_dummy_cards(temp_check)
                    betting_round_at_start = temp_check.betting_round
                    
                    replay_public_history(temp_check, public_hist)
                    
                    if betting_round_at_start == 0 and temp_check.betting_round == 1:
                        is_chance = True
                        chance_outcomes = ['Js', 'Jh', 'Qs', 'Qh', 'Ks', 'Kh']
            elif 'Rhode' in game_name or 'Twelve' in game_name or 'Royal' in game_name:
                if not has_marker:
                    # Bestimme betting_round BEVOR wir die letzte Action replayen
                    # Dazu replayen wir die History bis zur vorletzten Action
                    temp_check_before = game_class(**game_config)
                    temp_check_before.reset(0)
                    set_dummy_cards(temp_check_before)
                    
                    betting_round_before_last_action = 0
                    if len(public_hist) > 0:
                        # Replay alle Actions außer der letzten
                        replay_public_history(temp_check_before, public_hist[:-1])
                        betting_round_before_last_action = temp_check_before.betting_round
                    
                    # Jetzt replayen wir die komplette History
                    temp_check = game_class(**game_config)
                    temp_check.reset(0)
                    set_dummy_cards(temp_check)
                    replay_public_history(temp_check, public_hist)
                    
                    betting_round_after = temp_check.betting_round
                    betting_round_changed = betting_round_before_last_action != betting_round_after
                    
                    # Spielspezifisch hardcoded:
                    # Rhode Island und Twelve Card Poker: 2 Betting-Runden
                    #   Chance-Node nach Round 0: betting_round 0 → 1
                    #   Chance-Node nach Round 1: betting_round 1 → 2
                    # Royal Hold'em: 3 Betting-Runden
                    #   Chance-Node nach Round 0: betting_round 0 → 1 (3 Public Cards)
                    #   Chance-Node nach Round 1: betting_round 1 → 2 (1 Public Card)
                    #   Chance-Node nach Round 2: betting_round 2 → 3 (1 Public Card)
                    
                    if 'Royal' in game_name:
                        is_first_chance = (betting_round_before_last_action == 0 and betting_round_after == 1)
                        is_second_chance = (betting_round_before_last_action == 1 and betting_round_after == 2)
                        is_third_chance = (betting_round_before_last_action == 2 and betting_round_after == 3)
                        is_valid_chance = is_first_chance or is_second_chance or is_third_chance
                    else:  # Rhode Island oder Twelve Card Poker
                        is_first_chance = (betting_round_before_last_action == 0 and betting_round_after == 1)
                        is_second_chance = (betting_round_before_last_action == 1 and betting_round_after == 2)
                        is_valid_chance = is_first_chance or is_second_chance
                    
                    if betting_round_changed and is_valid_chance and not temp_check.done:
                        is_chance = True
                        if 'Rhode' in game_name:
                            ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
                            suits = ['s', 'h', 'd', 'c']
                            all_cards = [rank + suit for rank in ranks for suit in suits]
                        elif 'Twelve' in game_name:
                            ranks = ['J', 'Q', 'K', 'A']
                            suits = ['s', 'h', 'd']
                            all_cards = [rank + suit for rank in ranks for suit in suits]
                        else:  # Royal Hold'em
                            ranks = ['T', 'J', 'Q', 'K', 'A']
                            suits = ['s', 'h', 'd', 'c']
                            all_cards = [rank + suit for rank in ranks for suit in suits]
                        
                        # Deck reduction: Remove already dealt public cards
                        # Note: Private cards are NOT part of the public state tree
                        chance_outcomes = [card for card in all_cards if card not in public_cards_in_history]
        
        if is_chance:
            children = {}
            for outcome in chance_outcomes:
                child_hist = list(public_hist) + [outcome]
                child_key = tuple(child_hist)

                game_new = game_class(**game_config)
                game_new.reset(0)
                set_dummy_cards(game_new)
                replay_public_history(game_new, public_hist)
                
                # Removed: game_new.step('|')
                # The game state is already at the end of the round (due to replay of actions like 'call').
                # Manually calling step('|') triggers proceed_round('|') -> default case -> flips player -> Wrong!
                
                # Apply chance outcome
                if 'Leduc' in game_name:
                    game_new.public_card = outcome
                    if hasattr(game_new.players[0], 'set_public_card'):
                        game_new.players[0].set_public_card(outcome)
                        game_new.players[1].set_public_card(outcome)
                elif 'Rhode' in game_name or 'Twelve' in game_name or 'Royal' in game_name:
                    if not hasattr(game_new, 'public_cards'):
                        game_new.public_cards = []
                    game_new.public_cards.append(outcome)
                    game_new.players[0].public_cards.append(outcome)
                    game_new.players[1].public_cards.append(outcome)
                
                children[outcome] = child_key
                traverse(game_new, child_hist)
            
            # For the chance node itself, we need info sets (though usually not used for value lookup, good for debug)
            # We recreate the state BEFORE the chance event
            temp_for_info_sets = game_class(**game_config)
            temp_for_info_sets.reset(0)
            set_dummy_cards(temp_for_info_sets)
            replay_public_history(temp_for_info_sets, public_hist)
            
            # Fix: The game engine might have eagerly dealt a card upon completing the round.
            # But this Chance Node represents the state *before* the card is revealed.
            # We must force the game state to appear as if no card is public yet.
            if 'Leduc' in game_name:
                temp_for_info_sets.public_card = None
                temp_for_info_sets.players[0].set_public_card(None)
                temp_for_info_sets.players[1].set_public_card(None)
            elif 'Rhode' in game_name or 'Twelve' in game_name or 'Royal' in game_name:
                temp_for_info_sets.public_cards = []
                temp_for_info_sets.players[0].public_cards = []
                temp_for_info_sets.players[1].public_cards = []
            
            player0_info_sets = get_info_sets_simulated(temp_for_info_sets, 0)
            player1_info_sets = get_info_sets_simulated(temp_for_info_sets, 1)
            player_bets = temp_for_info_sets.total_bets if hasattr(temp_for_info_sets, 'total_bets') else temp_for_info_sets.player_bets
            
            public_states[public_key] = {
                'type': 'chance',
                'pot': temp_for_info_sets.pot,
                'player_bets': list(player_bets),
                'children': children,
                'player0_info_sets': player0_info_sets,
                'player1_info_sets': player1_info_sets
            }
            stats['chance'] += 1
            return
        
        set_dummy_cards(game)
        
        # Check Terminal
        if game.done:
            player0_info_sets = get_info_sets_simulated(game, 0)
            player1_info_sets = get_info_sets_simulated(game, 1)
            player_bets = game.total_bets if hasattr(game, 'total_bets') else game.player_bets
            
            # Identify who acted last. 
            # In Kuhn/Leduc, step() flips current_player. So last actor is 1 - current.
            # We store this so BestResponseAgent doesn't have to guess based on history length.
            last_actor = 1 - game.current_player
            
            public_states[public_key] = {
                'type': 'terminal',
                'pot': game.pot,
                'player_bets': list(player_bets),
                'last_actor': last_actor,
                'player0_info_sets': player0_info_sets,
                'player1_info_sets': player1_info_sets
            }
            stats['terminal'] += 1
            return
        
        
        legal_actions = game.get_legal_actions()
        
        if not legal_actions:
            # Should be terminal if no actions
            player0_info_sets = get_info_sets_simulated(game, 0)
            player1_info_sets = get_info_sets_simulated(game, 1)
            player_bets = game.total_bets if hasattr(game, 'total_bets') else game.player_bets
            
            last_actor = 1 - game.current_player
            
            public_states[public_key] = {
                'type': 'terminal',
                'pot': game.pot,
                'player_bets': list(player_bets),
                'last_actor': last_actor,
                'player0_info_sets': player0_info_sets,
                'player1_info_sets': player1_info_sets
            }
            stats['terminal'] += 1
            return
        
        children = {}
        for action in legal_actions:
            child_hist = list(public_hist) + [action]
            child_key = tuple(child_hist)
            
            game_new = game_class(**game_config)
            game_new.reset(0)
            set_dummy_cards(game_new)
            # Replay
            replay_public_history(game_new, child_hist)
            
            children[action] = child_key
            traverse(game_new, child_hist)
        
        player0_info_sets = get_info_sets_simulated(game, 0)
        player1_info_sets = get_info_sets_simulated(game, 1)
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
        stats['choice'] += 1
    
    cache_status = "enabled" if use_cache else "disabled"
    print(f"Building public state tree for {game_class.__name__}... (Cache: {cache_status})")
    start_time = time.time()
    
    game = game_class(**game_config)
    game.reset(0)
    set_dummy_cards(game)
    traverse(game, [])
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\nTree building complete!")
    print(f"  Total states: {len(public_states)}")
    print(f"  Chance nodes: {stats['chance']}")
    print(f"  Choice nodes: {stats['choice']}")
    print(f"  Terminal nodes: {stats['terminal']}")
    print(f"  Time elapsed: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    if use_cache and game_state_cache is not None:
        print(f"  Cache hits: {len(game_state_cache)} unique states cached")
    
    return {'public_states': public_states}


def print_public_state_tree(public_states, root_key=(), indent=""):
    """
    Recursively prints the public state tree in a readable format.
    """
    if root_key not in public_states:
        return

    node = public_states[root_key]
    node_type = node['type']

    # Format node information string
    info = f"[{node_type.upper()}]"
    if node_type == 'choice':
        info += f" Player: {node['player']}"
    elif node_type == 'chance':
        info += " (Chance/Nature)"
    elif node_type == 'terminal':
        info += f" Pot: {node['pot']} | Bets: {node['player_bets']}"

    # Print the current node's info (newline handled by parent loop or initial call)
    print(info)

    if 'children' in node:
        # Sort children for consistent output order
        # (Actions like 'bet', 'check' first, then card deals)
        sorted_children = sorted(node['children'].items(), key=lambda x: str(x[0]))

        for i, (action, child_key) in enumerate(sorted_children):
            is_last = (i == len(sorted_children) - 1)

            # Draw tree branches
            branch = "└──" if is_last else "├──"
            edge_str = f"{indent}{branch} {action} --> "

            # Print the edge (action) without newline, then recurse
            print(edge_str, end="")

            # Update indentation for the child: spaces for last item, pipe for others
            next_indent = indent + ("    " if is_last else "│   ")
            print_public_state_tree(public_states, child_key, next_indent)


def save_public_state_tree(game_name, tree, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'public state trees')
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{game_name}_public_tree_v2.pkl.gz"
    path = os.path.join(output_dir, filename)
    with gzip.open(path, 'wb') as f:
        pickle.dump(tree, f)
    print(f"Saved: {path}")
    return path


if __name__ == "__main__":
    # Parse command line arguments
    use_cache = True
    game_name = None
    
    for arg in sys.argv[1:]:
        if arg == '--no-cache':
            use_cache = False
        elif not arg.startswith('--'):
            game_name = arg
    
    if game_name is None:
        game_name = 'kuhn_case2'
    
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
    elif game_name == 'royal_holdem' or game_name.lower() == 'royal':
        game_class = RoyalHoldemGame
        game_config = GAME_CONFIGS['royal_holdem']
        save_name = 'royal_holdem'
    elif game_name == 'twelve_card_poker' or game_name.lower() == 'twelve_card':
        game_class = TwelveCardPokerGame
        game_config = GAME_CONFIGS['twelve_card_poker']
        save_name = 'twelve_card_poker'
    elif game_name == 'limit_holdem' or game_name.lower() == 'limit':
        game_class = LimitHoldemGame
        game_config = GAME_CONFIGS['limit_holdem']
        save_name = 'limit_holdem'
    else:
        # Defaults or strict error
        print(f"Unknown game: {game_name}")
        sys.exit(1)
    
    tree = build_public_state_tree(game_class, game_config, use_cache=use_cache)
    states = tree['public_states']

    num_choice = sum(1 for s in states.values() if s['type'] == 'choice')
    num_chance = sum(1 for s in states.values() if s['type'] == 'chance')
    num_terminal = sum(1 for s in states.values() if s['type'] == 'terminal')

    print(f"Total public states: {len(states)}")
    print(f"Choice nodes: {num_choice}, Chance nodes: {num_chance}, Terminal nodes: {num_terminal}")
    #print_public_state_tree(states, root_key=())

    save_public_state_tree(save_name, tree)