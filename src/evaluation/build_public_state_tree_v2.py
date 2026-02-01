import os
import pickle
import gzip
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.poker_utils import find_game_class_for_abstraction
from utils.data_models import KeyGenerator
from utils.tree_registry import record_tree_stats


def build_public_state_tree(game_class, progress_interval=10000, use_cache=False, abstract_suits=False):
    """
    Build the Public State Tree.

    Args:
        game_class: The game class
        progress_interval: Progress is printed every N states (default: 1000)
        use_cache: If True, game states are cached for better performance (default: True)
        abstract_suits: If True, only unique ranks are used (Suit Abstraction)
    """
    public_states = {}
    states_visited = 0
    stats = {'chance': 0, 'choice': 0, 'terminal': 0}

    # Cache for game states: public_hist -> saved_state
    # Uses native save_state/restore_state function of game classes
    game_state_cache = {} if use_cache else None

    VALID_ACTIONS = {'bet', 'check', 'call', 'fold'}

    def new_game():
        # Games are now instantiated without parameters
        return game_class()

    def get_all_cards_for_game(game):
        dealer_class = game.dealer.__class__
        dealer = dealer_class()
        dealer.reset()
        all_cards = list(dealer.deck)

        if abstract_suits:
            unique_ranks = set()
            for card in all_cards:
                rank = card[0] if len(card) > 1 else card
                unique_ranks.add(rank)
            return sorted(list(unique_ranks))

        return all_cards

    def get_current_public_cards(game):
        if hasattr(game, 'public_cards'):
            return list(game.public_cards) if game.public_cards else []
        if hasattr(game, 'public_card') and game.public_card is not None and game.public_card != 'None':
            return [game.public_card]
        return []

    def get_info_sets_simulated(game, player_id):
        all_cards = get_all_cards_for_game(game)
        current_public = get_current_public_cards(game)

        if abstract_suits:
            # With suit abstraction: ranks can occur multiple times (e.g. Leduc); determine possible private cards via remaining deck (after replay), not via rank equality.
            try:
                available_cards = sorted(list(set(game.dealer.deck)), key=lambda x: str(x))
            except Exception:
                available_cards = list(all_cards)
        else:
            # Non-abstracted: exact physical cards cannot be both public and private
            available_cards = [c for c in all_cards if c not in current_public]

        original = None
        if hasattr(game.players[player_id], 'private_cards'):
            original = list(game.players[player_id].private_cards)
        else:
            original = game.players[player_id].private_card

        info_sets = []
        seen_keys = set()
        for card in available_cards:
            if hasattr(game.players[player_id], 'private_cards'):
                # Keep list length stable (2 cards) if applicable
                if len(game.players[player_id].private_cards) >= 2:
                    game.players[player_id].private_cards = [card, card]
                else:
                    game.players[player_id].private_cards = [card]
                game.players[player_id].private_card = game.players[player_id].private_cards[0]
            else:
                game.players[player_id].private_card = card

            key = KeyGenerator.get_info_set_key(game, player_id)
            if abstract_suits:
                if key not in seen_keys:
                    seen_keys.add(key)
                    info_sets.append(key)
            else:
                info_sets.append(key)

        # restore
        if hasattr(game.players[player_id], 'private_cards'):
            game.players[player_id].private_cards = list(original)
            game.players[player_id].private_card = game.players[player_id].private_cards[0] if game.players[
                player_id].private_cards else None
        else:
            game.players[player_id].private_card = original
        return info_sets

    def prepare_public_root_state(game):
        """
        Public Tree root: AFTER private deals (unobserved), BEFORE first player action.
        We set deterministic dummy private cards WITHOUT removing them from the deck,
        so public chance outcomes remain independent of private cards (public abstraction).
        """
        game.reset(0)

        all_cards = get_all_cards_for_game(game)
        if not all_cards:
            return

        # Assign dummy private cards (do NOT remove from deck)
        needs_two = hasattr(game.players[0], 'private_cards')
        if needs_two:
            c0, c1, c2, c3 = all_cards[0], all_cards[1], all_cards[2], all_cards[3]
            game.players[0].private_cards = [c0, c1]
            game.players[1].private_cards = [c2, c3]
            game.players[0].private_card = c0
            game.players[1].private_card = c2
        else:
            game.players[0].private_card = all_cards[0]
            game.players[1].private_card = all_cards[1] if len(all_cards) > 1 else all_cards[0]

        # Skip private chance node for public-tree simulation
        if hasattr(game, '_chance_targets'):
            game._chance_targets = []
        if hasattr(game, '_chance_context'):
            game._chance_context = None

        # Move to first decision node using the game's own hook
        if hasattr(game, '_after_private_deal'):
            game._after_private_deal()

    def replay_public_history(game, public_hist):
        public_hist_key = tuple(public_hist)

        if use_cache and game_state_cache is not None and public_hist_key in game_state_cache:
            game.restore_state(game_state_cache[public_hist_key])
            return

        prepare_public_root_state(game)

        for item in public_hist:
            if item in VALID_ACTIONS:
                if hasattr(game, 'is_chance_node') and game.is_chance_node():
                    raise ValueError(f"Replay desync: expected decision, got chance before action {item}")
                game.step(item)
            else:
                # public chance outcome (card)
                if not (hasattr(game, 'is_chance_node') and game.is_chance_node()):
                    raise ValueError(f"Replay desync: expected chance, got decision before card {item}")
                game.step(item)

        if use_cache and game_state_cache is not None:
            game_state_cache[public_hist_key] = game.save_state()

    def traverse(public_hist):
        nonlocal states_visited, stats
        public_key = tuple(public_hist)
        if public_key in public_states:
            return

        states_visited += 1
        if states_visited % progress_interval == 0:
            print(
                f"  Progress: {states_visited} states visited | "
                f"Chance: {stats['chance']}, Choice: {stats['choice']}, Terminal: {stats['terminal']}"
            )

        game = new_game()
        replay_public_history(game, public_hist)

        player0_info_sets = get_info_sets_simulated(game, 0)
        player1_info_sets = get_info_sets_simulated(game, 1)
        player_bets = game.total_bets if hasattr(game, 'total_bets') else game.player_bets

        # Terminal
        if game.done:
            last_actor = 1 - game.current_player
            public_states[public_key] = {
                'type': 'terminal',
                'pot': game.pot,
                'player_bets': list(player_bets),
                'last_actor': last_actor,
                'player0_info_sets': player0_info_sets,
                'player1_info_sets': player1_info_sets,
            }
            stats['terminal'] += 1
            return

        # Public chance node (cards)
        if hasattr(game, 'is_chance_node') and game.is_chance_node():
            outcomes_with_probs = game.get_chance_outcomes_with_probs()
            outcomes = sorted(list(outcomes_with_probs.keys()), key=lambda x: str(x))
            children = {}
            for outcome in outcomes:
                child_hist = list(public_hist) + [outcome]
                child_key = tuple(child_hist)
                children[outcome] = child_key
                traverse(child_hist)

            public_states[public_key] = {
                'type': 'chance',
                'pot': game.pot,
                'player_bets': list(player_bets),
                'children': children,
                'chance_probs': dict(outcomes_with_probs),
                'player0_info_sets': player0_info_sets,
                'player1_info_sets': player1_info_sets,
            }
            stats['chance'] += 1
            return

        # Choice node
        legal_actions = game.get_legal_actions()
        if not legal_actions:
            last_actor = 1 - game.current_player
            public_states[public_key] = {
                'type': 'terminal',
                'pot': game.pot,
                'player_bets': list(player_bets),
                'last_actor': last_actor,
                'player0_info_sets': player0_info_sets,
                'player1_info_sets': player1_info_sets,
            }
            stats['terminal'] += 1
            return

        children = {}
        for action in legal_actions:
            child_hist = list(public_hist) + [action]
            child_key = tuple(child_hist)
            children[action] = child_key
            traverse(child_hist)

        public_states[public_key] = {
            'type': 'choice',
            'player': game.current_player,
            'pot': game.pot,
            'player_bets': list(player_bets),
            'children': children,
            'player0_info_sets': player0_info_sets,
            'player1_info_sets': player1_info_sets,
        }
        stats['choice'] += 1

    cache_status = "enabled" if use_cache else "disabled"
    abstraction_str = " (suit abstracted)" if abstract_suits else ""
    print(f"Building public state tree for {game_class.__name__}{abstraction_str}... (Cache: {cache_status})")
    start_time = time.time()

    traverse([])

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\nTree building complete!")
    print(f"  Total states: {len(public_states)}")
    print(f"  Chance nodes: {stats['chance']}")
    print(f"  Choice nodes: {stats['choice']}")
    print(f"  Terminal nodes: {stats['terminal']}")
    print(f"  Time elapsed: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
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


def save_public_state_tree(game_name, tree, output_dir=None, abstract_suits=False):
    if output_dir is None:
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(script_dir, 'data', 'trees', 'public_state_trees')
    os.makedirs(output_dir, exist_ok=True)

    # If abstract_suits=False, add "_NOT_abstracted" to name
    # If abstract_suits=True, use standard name
    if abstract_suits:
        filename = f"{game_name}_public_tree_v2.pkl.gz"
    else:
        filename = f"{game_name}_public_tree_v2_NOT_abstracted.pkl.gz"

    path = os.path.join(output_dir, filename)
    with gzip.open(path, 'wb') as f:
        pickle.dump(tree, f)
    print(f"Saved: {path}")

    # Registry logging (Public State Tree Save): sizes + node types + infosets.
    try:
        states = tree.get("public_states", {}) if isinstance(tree, dict) else {}
        type_counts = {"chance": 0, "choice": 0, "terminal": 0}
        p0_infosets = set()
        p1_infosets = set()
        for node in states.values():
            t = node.get("type")
            if t in type_counts:
                type_counts[t] += 1
            # Infosets, if present (carried at nodes in PST)
            for k in node.get("player0_info_sets", []) or []:
                p0_infosets.add(k)
            for k in node.get("player1_info_sets", []) or []:
                p1_infosets.add(k)
        record_tree_stats(
            {
                "schema_version": 1,
                "tree_kind": "public_state_tree_v2",
                "game": str(game_name),
                "abstract_suits": bool(abstract_suits),
                "num_states": int(len(states)),
                "num_infosets_p0": int(len(p0_infosets)),
                "num_infosets_p1": int(len(p1_infosets)),
                "node_type_counts": type_counts,
                "tree_path": path,
            }
        )
    except Exception:
        # Registry is optional; saving must never fail.
        pass

    return path


if __name__ == "__main__":
    # Parse command line arguments
    use_cache = True
    game_name = None
    abstract_suits = None  # None means: automatically determine

    for arg in sys.argv[1:]:
        if arg == '--no-cache':
            use_cache = False
        elif arg == '--abstract-suits' or arg == '--abstract_suits':
            abstract_suits = True
        elif arg == '--no-suit-abstraction' or arg == '--no_suit_abstraction':
            abstract_suits = False
        elif arg.startswith('--abstract-suits=') or arg.startswith('--abstract_suits='):
            # Support --abstract-suits=true/false and --abstract_suits=True/False
            value = arg.split('=', 1)[1].lower()
            if value in ('true', '1', 'yes'):
                abstract_suits = True
            elif value in ('false', '0', 'no'):
                abstract_suits = False
            else:
                print(f"WARNING: Unknown value for abstract-suits: {value}. Using automatic detection.")
        elif not arg.startswith('--'):
            game_name = arg

    if game_name is None:
        game_name = 'kuhn_case2'

    # Determine if suit abstraction should be used
    # Default for leduc and twelve_card_poker, unless explicitly disabled
    if abstract_suits is None:
        # Automatically determine based on game
        abstract_suits = (game_name in ['leduc', 'twelve_card_poker'])

    # Find the game class using the new utility
    game_class = find_game_class_for_abstraction(game_name, abstract_suits)
    if game_class is None:
        print(f"Unknown game: {game_name}")
        sys.exit(1)

    # Determine save_name
    if game_name.startswith('kuhn'):
        save_name = 'kuhn'
    else:
        save_name = game_name

    tree = build_public_state_tree(game_class, use_cache=use_cache, abstract_suits=abstract_suits)
    states = tree['public_states']

    num_choice = sum(1 for s in states.values() if s['type'] == 'choice')
    num_chance = sum(1 for s in states.values() if s['type'] == 'chance')
    num_terminal = sum(1 for s in states.values() if s['type'] == 'terminal')

    print(f"Total public states: {len(states)}")
    print(f"Choice nodes: {num_choice}, Chance nodes: {num_chance}, Terminal nodes: {num_terminal}")
    # print_public_state_tree(states, root_key=())

    save_public_state_tree(save_name, tree, abstract_suits=abstract_suits)
