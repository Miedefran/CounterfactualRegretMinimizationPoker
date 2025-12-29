import gzip
import pickle
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def compute_payoff(game_name, our_info, opp_info, pot, player_bets, player_id, node):
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
    
    class PlayerDummy:
        pass
    
    p0 = PlayerDummy()
    p1 = PlayerDummy()
    
    judger = None
    
    if 'kuhn' in game_name.lower():
        from envs.kuhn_poker.judger import KuhnPokerJudger
        p0.private_card = p0_card
        p1.private_card = p1_card
        judger = KuhnPokerJudger()
    
    elif 'leduc' in game_name.lower():
        from envs.leduc_holdem.judger import LeducHoldemJudger
        p0.private_card = p0_card
        p1.private_card = p1_card
        
        # Public cards are in the tuple
        if len(p0_public) > 0:
            p0.public_card = p0_public[0]
            p1.public_card = p1_public[0] # Should be same
        else:
            p0.public_card = None
            p1.public_card = None
            
        judger = LeducHoldemJudger()
        
        if (p0.public_card is None) and ('fold' in history):
             # Dummy public card to prevent crash if judger checks it (though it shouldn't for fold)
             p0.public_card = 'Js'
             p1.public_card = 'Js'
            
    elif 'twelve' in game_name.lower():
        from envs.twelve_card_poker.judger import TwelveCardPokerJudger
        p0.private_card = p0_card
        p1.private_card = p1_card
        p0.public_cards = list(p0_public)
        p1.public_cards = list(p1_public)
        judger = TwelveCardPokerJudger()
    elif 'rhode' in game_name.lower():
        from envs.rhode_island.judger import RhodeIslandJudger
        p0.private_card = p0_card
        p1.private_card = p1_card
        p0.public_cards = list(p0_public)
        p1.public_cards = list(p1_public)
        judger = RhodeIslandJudger()
    elif 'royal' in game_name.lower():
        from envs.royal_holdem.judger import RoyalHoldemJudger
        p0.private_card = p0_card
        p1.private_card = p1_card
        p0.public_cards = list(p0_public)
        p1.public_cards = list(p1_public)
        judger = RoyalHoldemJudger()    
    else:
        return 0.0
    
    players = [p0, p1]
    
    # Use reliable actor from tree node if available
    if 'last_actor' in node:
        current_player = node['last_actor']
    else: 
        print("last actor not found")
        current_player = None
    
    
    payoffs = judger.judge(players, history, current_player, pot, player_bets)
    return payoffs[player_id]


def traverse_public_tree(game_name, player_id, tree, avg_strategy, public_hist, r_opp):
    node = tree['public_states'].get(public_hist)
    if not node:
        raise ValueError(f"Public state not found: {public_hist}")
    
    node_type = node['type']
    our_infosets = node[f'player{player_id}_info_sets']
    opp_infosets = node[f'player{1-player_id}_info_sets']
    
    if len(r_opp) != len(opp_infosets):
        raise ValueError(f"Reach probability mismatch: {len(r_opp)} vs {len(opp_infosets)}")
    
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


def terminal_value(game_name, player_id, node, r_opp):
    our_infosets = node[f'player{player_id}_info_sets']
    opp_infosets = node[f'player{1-player_id}_info_sets']
    pot = node['pot']
    player_bets = node['player_bets']
    
    result = []
    
    for our_info in our_infosets:
        our_card = our_info[0]
        value = 0.0
        total_valid_r = 0.0
        
        for j, opp_info in enumerate(opp_infosets):
            opp_card = opp_info[0]
            if our_card == opp_card:
                continue
            total_valid_r += r_opp[j]
        
        if total_valid_r > 0:
            for j, opp_info in enumerate(opp_infosets):
                opp_card = opp_info[0]
                if our_card == opp_card:
                    continue
                # Pass 'node' to compute_payoff
                util = compute_payoff(game_name, our_info, opp_info, pot, player_bets, player_id, node)
                value += r_opp[j] * util
        
        result.append(value)
    
    return result


def chance_value(game_name, player_id, tree, avg_strategy, node, public_hist, r_opp):
    our_infosets = node[f'player{player_id}_info_sets']
    opp_infosets = node[f'player{1-player_id}_info_sets']
    children = node['children']

    result = [0.0] * len(our_infosets)
    chance_prob = 1.0 / len(children)

    parent_card_to_r_opp = {info[0]: r for info, r in zip(opp_infosets, r_opp)}

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

        if not has_mass:
            continue
        
    


        v_child = traverse_public_tree(game_name, player_id, tree, avg_strategy, child_hist, r_child)

        card_to_child_value = {info[0]: val for info, val in zip(child_our_infosets, v_child)}

        for i, our_info in enumerate(our_infosets):
            our_card = our_info[0]
            if our_card == outcome:
                continue

            if our_card in card_to_child_value:
                result[i] += card_to_child_value[our_card]

    return result


def our_choice_value(game_name, player_id, tree, avg_strategy, node, public_hist, r_opp):
    our_infosets = node[f'player{player_id}_info_sets']
    children = node['children']
    
    if not children:
        return [0.0] * len(our_infosets)
    
    action_values = {}
    
    for action, child_hist in children.items():
        child_node = tree['public_states'].get(child_hist)
        if child_node:
            child_our_infosets = child_node[f'player{player_id}_info_sets']
            v_child = traverse_public_tree(game_name, player_id, tree, avg_strategy, child_hist, r_opp)
            action_values[action] = (v_child, child_our_infosets)
    
    if not action_values:
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
    
    return result


def opponent_choice_value(game_name, player_id, tree, avg_strategy, node, public_hist, r_opp):
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
        
        v_child = traverse_public_tree(game_name, player_id, tree, avg_strategy, child_hist, r_child)
        
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
        return [0.0] * len(our_infosets)
    
    return result


def compute_best_response_value(game_name, player_id, tree, avg_strategy, root_hist=()):
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
    
    return total_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', required=True)
    parser.add_argument('--player', type=int, required=True)
    parser.add_argument('--public-tree', required=True)
    parser.add_argument('--strategy', required=True)
    args = parser.parse_args()
    
    tree = load_public_tree(args.public_tree)
    avg_strategy = load_average_strategy(args.strategy)
    value = compute_best_response_value(args.game, args.player, tree, avg_strategy, root_hist=())
    print(f"best_response_value={value}")


if __name__ == '__main__':
    main()