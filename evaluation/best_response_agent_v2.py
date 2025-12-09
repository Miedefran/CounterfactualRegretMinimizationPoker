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
    if isinstance(info_set_key, tuple) and len(info_set_key) >= 3:
        if len(info_set_key) == 4 and info_set_key[1] is None:
            modified_key = (info_set_key[0], 'None', info_set_key[2], info_set_key[3])
            strategy = avg_strategy.get(modified_key, {})
        else:
            strategy = avg_strategy.get(info_set_key, {})
    else:
        strategy = avg_strategy.get(info_set_key, {})
    
    if not strategy and legal_actions:
        return 1.0 / len(legal_actions)
    
    return strategy.get(action, 0.0)


def compute_payoff(game_name, our_info, opp_info, pot, player_bets, player_id):
    player0_info = our_info if our_info[-1] == 0 else opp_info
    player1_info = opp_info if our_info[-1] == 0 else our_info
    
    class Player0:
        pass
    class Player1:
        pass
    
    p0 = Player0()
    p1 = Player1()
    
    if 'kuhn' in game_name.lower():
        from envs.kuhn_poker.judger import KuhnPokerJudger
        p0.private_card = player0_info[0]
        p1.private_card = player1_info[0]
        history = list(player0_info[1])
        judger = KuhnPokerJudger()
    
    elif 'leduc' in game_name.lower():
        from envs.leduc_holdem.judger import LeducHoldemJudger
        p0.private_card = player0_info[0]
        p1.private_card = player1_info[0]
        public_card = player0_info[1]
        history = list(player0_info[2])
        
        if public_card == 'None' or public_card is None:
            if 'fold' in history:
                p0.public_card = 'Js'
                p1.public_card = 'Js'
            else:
                return 0.0
        else:
            p0.public_card = public_card
            p1.public_card = public_card
        
        judger = LeducHoldemJudger()
    
    elif 'rhode' in game_name.lower() or 'twelve' in game_name.lower():
        from envs.rhode_island.judger import RhodeIslandJudger
        p0.private_card = player0_info[0]
        p1.private_card = player1_info[0]
        p0.public_cards = list(player0_info[1])
        p1.public_cards = list(player1_info[1])
        history = list(player0_info[2])
        judger = RhodeIslandJudger()
    else:
        return 0.0
    
    players = [p0, p1]
    betting_actions = [a for a in history if a in ['check', 'bet', 'call', 'fold']]
    
    if betting_actions and betting_actions[-1] == 'fold':
        current_player = (len(betting_actions) - 1) % 2
    else:
        current_player = len(betting_actions) % 2
    
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
                util = compute_payoff(game_name, our_info, opp_info, pot, player_bets, player_id)
                # r_opp may be normalized or not - we normalize here
                value += (r_opp[j] / total_valid_r) * util
        
        result.append(value)
    
    return result


def chance_value(game_name, player_id, tree, avg_strategy, node, public_hist, r_opp):
    our_infosets = node[f'player{player_id}_info_sets']
    opp_infosets = node[f'player{1-player_id}_info_sets']
    children = node['children']
    
    if not children:
        return [0.0] * len(our_infosets)
    
    num_outcomes = len(children)
    uniform_prob = 1.0 / num_outcomes
    
    result = None
    
    for outcome, child_hist in children.items():
        child_node = tree['public_states'].get(child_hist)
        if not child_node:
            continue
        
        child_opp_infosets = child_node[f'player{1-player_id}_info_sets']
        child_our_infosets = child_node[f'player{player_id}_info_sets']
        
        # For chance nodes: remaining cards are uniform conditional on the outcome
        num_remaining = len(child_opp_infosets)
        if num_remaining > 0:
            uniform_conditional = 1.0 / num_remaining
            # Normalize first, then pass (terminal_value needs normalized probs)
            r_child_normalized = [uniform_conditional] * num_remaining
        else:
            r_child_normalized = [0.0] * len(child_opp_infosets)
        
        v_child = traverse_public_tree(game_name, player_id, tree, avg_strategy, child_hist, r_child_normalized)
        
        card_to_child_value = {child_info[0]: v_child[i] for i, child_info in enumerate(child_our_infosets)}
        
        # Sum over chance outcomes weighted by uniform_prob (like OpenSpiel: prob * Value)
        if result is None:
            result = []
            for our_info in our_infosets:
                our_card = our_info[0]
                if our_card in card_to_child_value:
                    result.append(uniform_prob * card_to_child_value[our_card])
                else:
                    result.append(0.0)
        else:
            for i, our_info in enumerate(our_infosets):
                our_card = our_info[0]
                if our_card in card_to_child_value:
                    result[i] += uniform_prob * card_to_child_value[our_card]
    
    if result is None:
        return [0.0] * len(our_infosets)
    
    return result


def our_choice_value(game_name, player_id, tree, avg_strategy, node, public_hist, r_opp):
    our_infosets = node[f'player{player_id}_info_sets']
    children = node['children']
    
    if not children:
        return [0.0] * len(our_infosets)
    
    action_values = {}
    child_infosets_map = {}
    
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
    
    # Following OpenSpiel: value += prob * Value(child)
    # Following PokerRL: sum over actions (children already have correct values)
    result = None
    
    for action in legal_actions:
        child_hist = children[action]
        
        # Compute reach probabilities after this action (like PokerRL StrategyFiller line 134)
        # Multiply by strategy probability, DO NOT normalize (like PokerRL)
        r_child = []
        for k, opp_info in enumerate(opp_infosets):
            prob = get_action_probability(opp_info, action, avg_strategy, legal_actions)
            r_child.append(r_opp[k] * prob)
        
        # Pass non-normalized reach probs (like PokerRL)
        # Then just sum over actions (like PokerRL line 88: np.sum(ev_all_actions[:, opp], axis=0))
        v_child = traverse_public_tree(game_name, player_id, tree, avg_strategy, child_hist, r_child)
        
        child_node = tree['public_states'].get(child_hist)
        if child_node:
            child_our_infosets = child_node[f'player{player_id}_info_sets']
            card_to_child_value = {child_info[0]: v_child[i] for i, child_info in enumerate(child_our_infosets)}
            
            # Sum over actions (like PokerRL line 88)
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
