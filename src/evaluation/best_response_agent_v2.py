import gzip
import pickle
import argparse
import sys
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TIME_STATS = {
    'compute_payoff': 0.0,
    'terminal': 0.0,
    'chance': 0.0,
    'our_choice': 0.0,
    'opponent_choice': 0.0,
    'traverse_routing': 0.0,
}

NUM_WORKERS = None


def load_public_tree(path):
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)


def load_average_strategy(path):
    with gzip.open(path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and 'average_strategy' in data:
        return data['average_strategy']
    return data


def get_action_probability(info_set_key, action, avg_strategy, legal_actions):
    # Direct Lookup Only - Relying on Canonical Key Format
    strategy = avg_strategy.get(info_set_key)
    
    if not strategy:
        # Fallback to check if it's 0 probability for this action (or if missing completely)
        # If missing completely, it might be an unreachable state in training but reachable in tree?
        # Or simply unexplored. Return 0.
        print(f"WARNING: undefined key {info_set_key}")
        return 0.0

    return strategy.get(action, 0.0)


# Global cache for judgers
JUDGER_CACHE = {}

class PlayerDummy:
    pass

def get_judger(game_name):
    if game_name in JUDGER_CACHE:
        return JUDGER_CACHE[game_name]
    
    judger = None
    if 'kuhn' in game_name.lower():
        from envs.kuhn_poker.judger import KuhnPokerJudger
        judger = KuhnPokerJudger()
    elif 'leduc' in game_name.lower():
        from envs.leduc_holdem.judger import LeducHoldemJudger
        judger = LeducHoldemJudger()
    elif 'twelve' in game_name.lower():
        from envs.twelve_card_poker.judger import TwelveCardPokerJudger
        judger = TwelveCardPokerJudger()
    elif 'rhode' in game_name.lower():
        from envs.rhode_island.judger import RhodeIslandJudger
        judger = RhodeIslandJudger()
    elif 'royal' in game_name.lower():
        from envs.royal_holdem.judger import RoyalHoldemJudger
        judger = RoyalHoldemJudger()
    
    if judger:
        JUDGER_CACHE[game_name] = judger
    return judger

def compute_payoff(game_name, our_info, opp_info, pot, player_bets, player_id, node):
    t_start = time.time()
    
    # New Key Format: (private_card, public_cards_tuple, history_tuple, pid)
    
    # Extract info based on who is who
    if our_info[-1] == 0:
        p0_info, p1_info = our_info, opp_info
    else:
        p0_info, p1_info = opp_info, our_info
    
    # Unpack keys
    p0_card, p0_public, p0_hist, _ = p0_info
    p1_card, p1_public, p1_hist, _ = p1_info
    
    # History should be identical for both players in terms of actions
    history = list(p0_hist)
    
    p0 = PlayerDummy()
    p1 = PlayerDummy()
    
    judger = get_judger(game_name)
    if not judger:
        return 0.0

    # Setup dummy players based on game type (using checking logic from original code)
    if 'kuhn' in game_name.lower():
        p0.private_card = p0_card
        p1.private_card = p1_card
    
    elif 'leduc' in game_name.lower():
        p0.private_card = p0_card
        p1.private_card = p1_card
        
        # Public cards are in the tuple
        if len(p0_public) > 0:
            p0.public_card = p0_public[0]
            p1.public_card = p1_public[0] # Should be same
        else:
            p0.public_card = None
            p1.public_card = None
            
        if (p0.public_card is None) and ('fold' in history):
             # Dummy public card to prevent crash if judger checks it (though it shouldn't for fold)
             p0.public_card = 'Js'
             p1.public_card = 'Js'

    elif 'twelve' in game_name.lower():
        p0.private_card = p0_card
        p1.private_card = p1_card
        p0.public_cards = list(p0_public)
        p1.public_cards = list(p1_public)
    elif 'rhode' in game_name.lower():
        p0.private_card = p0_card
        p1.private_card = p1_card
        p0.public_cards = list(p0_public)
        p1.public_cards = list(p1_public)
    elif 'royal' in game_name.lower():
        p0.private_card = p0_card
        p1.private_card = p1_card
        p0.public_cards = list(p0_public)
        p1.public_cards = list(p1_public)
    else:
        return 0.0
    
    players = [p0, p1]
    
    # Use reliable actor from tree node if available
    if 'last_actor' in node:
        current_player = node['last_actor']
    else:
        # print("last actor not found") # Silence print for performance
        current_player = None

    payoffs = judger.judge(players, history, current_player, pot, player_bets)
    result = payoffs[player_id]
    TIME_STATS['compute_payoff'] += time.time() - t_start
    return result


def _traverse_subtree_worker(args):
    game_name, player_id, tree, avg_strategy, public_hist, r_opp = args
    return traverse_public_tree(game_name, player_id, tree, avg_strategy, public_hist, r_opp)


def traverse_public_tree(game_name, player_id, tree, avg_strategy, public_hist, r_opp):
    t_start = time.time()
    
    node = tree['public_states'].get(public_hist)
    if not node:
        raise ValueError(f"Public state not found: {public_hist}")
    
    node_type = node['type']
    our_infosets = node[f'player{player_id}_info_sets']
    opp_infosets = node[f'player{1-player_id}_info_sets']
    
    if len(r_opp) != len(opp_infosets):
        raise ValueError(f"Reach probability mismatch: {len(r_opp)} vs {len(opp_infosets)}")
    
    routing_time = time.time() - t_start
    TIME_STATS['traverse_routing'] += routing_time
    
    if node_type == 'terminal':
        return terminal_value(game_name, player_id, node, r_opp)
    
    elif node_type == 'chance':
        return chance_value(game_name, player_id, tree, avg_strategy, node, public_hist, r_opp)
    
    elif node_type == 'choice':
        if node['player'] == player_id:
            return our_choice_value(game_name, player_id, tree, avg_strategy, node, public_hist, r_opp)
        else:
            return opponent_choice_value(game_name, player_id, tree, avg_strategy, node, public_hist, r_opp)
    
    raise ValueError(f"Unknown node type: {node_type}")


def get_hand_strength_for_info_set(game_name, info_set):
    judger = get_judger(game_name)
    if not judger:
        return None
    
    our_card = info_set[0]
    our_public = info_set[1]
    
    p = PlayerDummy()
    
    if 'kuhn' in game_name.lower():
        p.private_card = our_card
        if hasattr(judger, 'hand_rank'):
            return (judger.hand_rank.get(our_card, 0),)
        return (0,)
    elif 'leduc' in game_name.lower():
        p.private_card = our_card
        if len(our_public) > 0:
            p.public_card = our_public[0]
        else:
            p.public_card = None
        if hasattr(judger, 'evaluate_hand'):
            return judger.evaluate_hand(p)
    elif 'twelve' in game_name.lower() or 'rhode' in game_name.lower() or 'royal' in game_name.lower():
        p.private_card = our_card
        p.public_cards = list(our_public)
        if hasattr(judger, 'evaluate_hand'):
            return judger.evaluate_hand(p)
    
    return None

def terminal_value(game_name, player_id, node, r_opp):
    t_start = time.time()
    
    our_infosets = node[f'player{player_id}_info_sets']
    opp_infosets = node[f'player{1-player_id}_info_sets']
    pot = node['pot']
    player_bets = node['player_bets']
    
    if len(our_infosets) == 0 or len(opp_infosets) == 0:
        TIME_STATS['terminal'] += time.time() - t_start
        return [0.0] * len(our_infosets)
    
    sample_our_info = our_infosets[0]
    sample_hist = sample_our_info[2]
    is_fold = len(sample_hist) > 0 and sample_hist[-1] == 'fold'
    
    if is_fold:
        last_actor = node.get('last_actor', None)
        if last_actor is not None:
            fold_player = last_actor
            winner = 1 - fold_player
            if winner == player_id:
                win_util = pot - player_bets[player_id]
            else:
                win_util = -player_bets[player_id]
            
            result = []
            for our_info in our_infosets:
                our_card = our_info[0]
                value = 0.0
                for j, opp_info in enumerate(opp_infosets):
                    if r_opp[j] == 0:
                        continue
                    opp_card = opp_info[0]
                    if our_card == opp_card:
                        continue
                    value += r_opp[j] * win_util
                result.append(value)
            
            TIME_STATS['terminal'] += time.time() - t_start
            return result
    
    our_with_strength = []
    for i, our_info in enumerate(our_infosets):
        strength = get_hand_strength_for_info_set(game_name, our_info)
        our_with_strength.append((i, our_info, strength))
    
    opp_with_strength = []
    for j, opp_info in enumerate(opp_infosets):
        if r_opp[j] == 0:
            continue
        strength = get_hand_strength_for_info_set(game_name, opp_info)
        opp_with_strength.append((j, opp_info, strength, r_opp[j]))
    
    if len(opp_with_strength) == 0:
        TIME_STATS['terminal'] += time.time() - t_start
        return [0.0] * len(our_infosets)
    
    our_with_strength.sort(key=lambda x: x[2] if x[2] is not None else (float('-inf'),))
    opp_with_strength.sort(key=lambda x: x[2] if x[2] is not None else (float('-inf'),))
    
    win_util = pot - player_bets[player_id]
    lose_util = -player_bets[player_id]
    tie_util = 0.0
    
    result = [0.0] * len(our_infosets)
    
    for our_idx, our_info, our_strength in our_with_strength:
        our_card = our_info[0]
        
        if our_strength is None:
            for j, opp_info, opp_strength, r_val in opp_with_strength:
                opp_card = opp_info[0]
                if our_card == opp_card:
                    continue
                util = compute_payoff(game_name, our_info, opp_info, pot, player_bets, player_id, node)
                result[our_idx] += r_val * util
            continue
        
        worse_prob = 0.0
        equal_prob = 0.0
        better_prob = 0.0
        
        for j, opp_info, opp_strength, r_val in opp_with_strength:
            opp_card = opp_info[0]
            if our_card == opp_card:
                continue
            
            if opp_strength is None:
                util = compute_payoff(game_name, our_info, opp_info, pot, player_bets, player_id, node)
                result[our_idx] += r_val * util
                continue
            
            if opp_strength < our_strength:
                worse_prob += r_val
            elif opp_strength == our_strength:
                equal_prob += r_val
            else:
                better_prob += r_val
        
        result[our_idx] = worse_prob * win_util + equal_prob * tie_util + better_prob * lose_util
    
    TIME_STATS['terminal'] += time.time() - t_start
    return result


def chance_value(game_name, player_id, tree, avg_strategy, node, public_hist, r_opp):
    t_start = time.time()
    
    our_infosets = node[f'player{player_id}_info_sets']
    opp_infosets = node[f'player{1-player_id}_info_sets']
    children = node['children']

    result = [0.0] * len(our_infosets)
    
    num_unknown_cards = len(children)
    if num_unknown_cards > 2:
        chance_prob = 1.0 / (num_unknown_cards - 2)
    else:
        chance_prob = 0.0

    parent_card_to_r_opp = {info[0]: r for info, r in zip(opp_infosets, r_opp)}

    tasks = []
    for outcome, child_hist in children.items():
        child_node = tree['public_states'].get(child_hist)
        if not child_node:
            continue

        child_opp_infosets = child_node[f'player{1-player_id}_info_sets']
        child_our_infosets = child_node[f'player{player_id}_info_sets']

        r_child = []
        has_mass = False
        for info in child_opp_infosets:
            child_card = info[0]
            if child_card == outcome: 
                r_child.append(0.0)
                continue

            parent_r = parent_card_to_r_opp.get(child_card, 0.0)
            val = parent_r * chance_prob
            r_child.append(val)
            if val > 0: has_mass = True

        if has_mass:
            tasks.append((outcome, child_hist, child_our_infosets, r_child))

    if not tasks:
        TIME_STATS['chance'] += time.time() - t_start
        return result

    use_parallel = NUM_WORKERS is not None and NUM_WORKERS > 1 and len(tasks) > 1

    if use_parallel:
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {}
            for outcome, child_hist, child_our_infosets, r_child in tasks:
                args = (game_name, player_id, tree, avg_strategy, child_hist, r_child)
                future = executor.submit(_traverse_subtree_worker, args)
                futures[future] = (outcome, child_our_infosets)

            for future in as_completed(futures):
                outcome, child_our_infosets = futures[future]
                try:
                    v_child = future.result()
                    card_to_child_value = {info[0]: val for info, val in zip(child_our_infosets, v_child)}

                    for i, our_info in enumerate(our_infosets):
                        our_card = our_info[0]
                        if our_card == outcome:
                            continue

                        if our_card in card_to_child_value:
                            result[i] += card_to_child_value[our_card]
                except Exception as e:
                    print(f"Error in parallel chance computation: {e}")
                    raise
    else:
        for outcome, child_hist, child_our_infosets, r_child in tasks:
            t_before_recursion = time.time()
            v_child = traverse_public_tree(game_name, player_id, tree, avg_strategy, child_hist, r_child)
            t_after_recursion = time.time()
            TIME_STATS['chance'] += (t_before_recursion - t_start) + (time.time() - t_after_recursion)
            t_start = time.time()

            card_to_child_value = {info[0]: val for info, val in zip(child_our_infosets, v_child)}

            for i, our_info in enumerate(our_infosets):
                our_card = our_info[0]
                if our_card == outcome:
                    continue

                if our_card in card_to_child_value:
                    result[i] += card_to_child_value[our_card]

    TIME_STATS['chance'] += time.time() - t_start
    return result


def our_choice_value(game_name, player_id, tree, avg_strategy, node, public_hist, r_opp):
    t_start = time.time()
    
    our_infosets = node[f'player{player_id}_info_sets']
    children = node['children']
    
    if not children:
        return [0.0] * len(our_infosets)
    
    tasks = []
    for action, child_hist in children.items():
        child_node = tree['public_states'].get(child_hist)
        if child_node:
            child_our_infosets = child_node[f'player{player_id}_info_sets']
            tasks.append((action, child_hist, child_our_infosets))
    
    if not tasks:
        return [0.0] * len(our_infosets)
    
    use_parallel = NUM_WORKERS is not None and NUM_WORKERS > 1 and len(tasks) > 1
    
    action_values = {}
    
    if use_parallel:
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {}
            for action, child_hist, child_our_infosets in tasks:
                args = (game_name, player_id, tree, avg_strategy, child_hist, r_opp)
                future = executor.submit(_traverse_subtree_worker, args)
                futures[future] = (action, child_our_infosets)
            
            for future in as_completed(futures):
                action, child_our_infosets = futures[future]
                try:
                    v_child = future.result()
                    action_values[action] = (v_child, child_our_infosets)
                except Exception as e:
                    print(f"Error in parallel our_choice computation: {e}")
                    raise
    else:
        for action, child_hist, child_our_infosets in tasks:
            t_before_recursion = time.time()
            v_child = traverse_public_tree(game_name, player_id, tree, avg_strategy, child_hist, r_opp)
            t_after_recursion = time.time()
            TIME_STATS['our_choice'] += (t_before_recursion - t_start) + (time.time() - t_after_recursion)
            t_start = time.time()
            action_values[action] = (v_child, child_our_infosets)
    
    if not action_values:
        TIME_STATS['our_choice'] += time.time() - t_start
        return [0.0] * len(our_infosets)
    
    result = []
    for our_info in our_infosets:
        our_card = our_info[0]
        max_val = float('-inf')
        
        for action in children.keys():
            if action in action_values:
                v_child, child_our_infosets = action_values[action]
                for j, child_info in enumerate(child_our_infosets):
                    if child_info[0] == our_card:
                        if len(v_child) > j:
                            if v_child[j] > max_val:
                                max_val = v_child[j]
                        break
        
        if max_val == float('-inf'):
            max_val = 0.0
        result.append(max_val)
    
    TIME_STATS['our_choice'] += time.time() - t_start
    return result


def opponent_choice_value(game_name, player_id, tree, avg_strategy, node, public_hist, r_opp):
    t_start = time.time()
    
    our_infosets = node[f'player{player_id}_info_sets']
    opp_infosets = node[f'player{1-player_id}_info_sets']
    children = node['children']
    
    if not children:
        return [0.0] * len(our_infosets)
    
    legal_actions = list(children.keys())
    
    result = None
    
    for action in legal_actions:
        child_hist = children[action]
        
        r_child = []
        for k, opp_info in enumerate(opp_infosets):
            prob = get_action_probability(opp_info, action, avg_strategy, legal_actions)
            r_child.append(r_opp[k] * prob)
        
        t_before_recursion = time.time()
        v_child = traverse_public_tree(game_name, player_id, tree, avg_strategy, child_hist, r_child)
        t_after_recursion = time.time()
        TIME_STATS['opponent_choice'] += (t_before_recursion - t_start) + (time.time() - t_after_recursion)
        t_start = time.time()
        
        child_node = tree['public_states'].get(child_hist)
        if child_node:
            child_our_infosets = child_node[f'player{player_id}_info_sets']
            card_to_child_value = {child_info[0]: v_child[i] for i, child_info in enumerate(child_our_infosets)}
            
            if result is None:
                result = []
                for our_info in our_infosets:
                    our_card = our_info[0]
                    if our_card in card_to_child_value:
                        result.append(card_to_child_value[our_card])
                    else:
                        result.append(0.0)
            else:
                for i, our_info in enumerate(our_infosets):
                    our_card = our_info[0]
                    if our_card in card_to_child_value:
                        result[i] += card_to_child_value[our_card]
    
    if result is None:
        TIME_STATS['opponent_choice'] += time.time() - t_start
        return [0.0] * len(our_infosets)
    
    TIME_STATS['opponent_choice'] += time.time() - t_start
    return result


def compute_best_response_value(game_name, player_id, tree, avg_strategy, root_hist=(), num_workers=None):
    global TIME_STATS, NUM_WORKERS
    NUM_WORKERS = num_workers
    TIME_STATS = {
        'compute_payoff': 0.0,
        'terminal': 0.0,
        'chance': 0.0,
        'our_choice': 0.0,
        'opponent_choice': 0.0,
        'traverse_routing': 0.0,
    }
    
    t_total_start = time.time()
    
    root_node = tree['public_states'].get(root_hist)
    if not root_node:
        raise ValueError(f"Root public state not found: {root_hist}")
    
    our_infosets = root_node[f'player{player_id}_info_sets']
    opp_infosets = root_node[f'player{1-player_id}_info_sets']
    
    total_value = 0.0
    
    for i, our_info in enumerate(our_infosets):
        our_card = our_info[0]
        
        valid_opp_indices = [j for j, opp_info in enumerate(opp_infosets) if opp_info[0] != our_card]
        num_valid = len(valid_opp_indices)
        
        if num_valid == 0:
            continue
        
        r_opp_conditional = [0.0] * len(opp_infosets)
        for j in valid_opp_indices:
            r_opp_conditional[j] = 1.0 / num_valid
        
        v_vector = traverse_public_tree(game_name, player_id, tree, avg_strategy, root_hist, r_opp_conditional)
        total_value += (1.0 / len(our_infosets)) * v_vector[i]
    
    t_total = time.time() - t_total_start
    
    total_measured = sum(TIME_STATS.values())
    
    print(f"\n[BR Zeitverteilung Player {player_id}] Gesamt: {t_total:.2f}s")
    if t_total > 0:
        print(f"  Terminal: {TIME_STATS['terminal']:.2f}s ({TIME_STATS['terminal']/t_total*100:.1f}%)")
        print(f"  Compute Payoff: {TIME_STATS['compute_payoff']:.2f}s ({TIME_STATS['compute_payoff']/t_total*100:.1f}%)")
        print(f"  Chance: {TIME_STATS['chance']:.2f}s ({TIME_STATS['chance']/t_total*100:.1f}%)")
        print(f"  Our Choice: {TIME_STATS['our_choice']:.2f}s ({TIME_STATS['our_choice']/t_total*100:.1f}%)")
        print(f"  Opponent Choice: {TIME_STATS['opponent_choice']:.2f}s ({TIME_STATS['opponent_choice']/t_total*100:.1f}%)")
        print(f"  Traverse Routing: {TIME_STATS['traverse_routing']:.2f}s ({TIME_STATS['traverse_routing']/t_total*100:.1f}%)")
        print(f"  Summe gemessen: {total_measured:.2f}s ({total_measured/t_total*100:.1f}%)")
    
    return total_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', required=True)
    parser.add_argument('--player', type=int, required=True)
    parser.add_argument('--public-tree', required=True)
    parser.add_argument('--strategy', required=True)
    parser.add_argument('--num-workers', type=int, default=4, 
                        help='Number of parallel workers (default: None = sequential)')
    args = parser.parse_args()
    
    tree = load_public_tree(args.public_tree)
    avg_strategy = load_average_strategy(args.strategy)
    value = compute_best_response_value(args.game, args.player, tree, avg_strategy, 
                                       root_hist=(), num_workers=args.num_workers)
    print(f"best_response_value={value}")


if __name__ == '__main__':
    main()