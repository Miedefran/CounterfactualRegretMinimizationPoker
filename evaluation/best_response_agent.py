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
    return data['average_strategy']


def traverse(game_name, player_id, tree, avg_strategy, hist, r_opp):
    node = get_node(tree, hist)
    node_type = node['type']
    
    if node_type == 'terminal':
        return terminal_vector(game_name, player_id, node, r_opp)
    
    if node_type == 'chance':
        return average_over_children(game_name, player_id, tree, avg_strategy, hist, r_opp)
    
    if node_type == 'choice':
        if node['player'] == (1 - player_id):
            return opponent_choice_vector(game_name, player_id, tree, avg_strategy, node, hist, r_opp)
        else:
            return our_choice_vector(game_name, player_id, tree, avg_strategy, node, hist, r_opp)
    
    raise ValueError(f"Unknown node type: {node_type}")


def opponent_choice_vector(game_name, player_id, tree, avg_strategy, node, hist, r_opp):
    opponent_id = 1 - player_id
    legal_actions = list(node['children'].keys())
    opp_infosets = node[f'player{opponent_id}_info_sets']
    our_infosets = node[f'player{player_id}_info_sets']
    sigma = policy_matrix(node, opponent_id, avg_strategy, legal_actions)
    
    if len(sigma) != len(r_opp):
        raise ValueError(f"Mismatch: policy matrix has {len(sigma)} rows but r_opp has {len(r_opp)} entries")
    
    result = None
    for j, action in enumerate(legal_actions):
        sigma_column = [row[j] for row in sigma]
        r_child = [r_opp[k] * sigma_column[k] for k in range(len(r_opp))]
        r_sum = sum(r_child)
        
        if r_sum > 0:
            r_child_normalized = [x / r_sum for x in r_child]
            child_hist = node['children'][action]
            v_j = traverse(game_name, player_id, tree, avg_strategy, child_hist, r_child_normalized)
            
            if result is None:
                result = [r_sum * v for v in v_j]
            else:
                result = [result[i] + r_sum * v_j[i] for i in range(len(result))]
    
    if result is None:
        return [0.0] * len(our_infosets)
    return result


def our_choice_vector(game_name, player_id, tree, avg_strategy, node, hist, r_opp):
    legal_actions = list(node['children'].keys())
    our_infosets = node[f'player{player_id}_info_sets']
    action_values = []
    
    for action in legal_actions:
        child_hist = node['children'][action]
        v = traverse(game_name, player_id, tree, avg_strategy, child_hist, r_opp)
        action_values.append(v)
    
    if not action_values:
        return [0.0] * len(our_infosets)
    
    num_infosets = len(action_values[0])
    result = []
    for i in range(num_infosets):
        max_val = max(v[i] for v in action_values)
        result.append(max_val)
    
    return result


def average_over_children(game_name, player_id, tree, avg_strategy, hist, r_opp):
    node = get_node(tree, hist)
    our_infosets = node[f'player{player_id}_info_sets']
    children_values = []
    public_cards = []
    
    if 'leduc' in game_name.lower():
        opp_infosets = node[f'player{1-player_id}_info_sets']
        
        for action, child_hist in node['children'].items():
            public_card = action
            public_cards.append(public_card)
            
            r_child = []
            for j, opp_info in enumerate(opp_infosets):
                opp_card = opp_info[0]
                if opp_card == public_card:
                    r_child.append(0.0)
                else:
                    r_child.append(r_opp[j])
            
            r_sum = sum(r_child)
            if r_sum > 0:
                r_child_normalized = [x / r_sum for x in r_child]
            else:
                r_child_normalized = [0.0] * len(r_child)
            
            v = traverse(game_name, player_id, tree, avg_strategy, child_hist, r_child_normalized)
            children_values.append(v)
    else:
        for child_hist in node['children'].values():
            v = traverse(game_name, player_id, tree, avg_strategy, child_hist, r_opp)
            children_values.append(v)
    
    if not children_values:
        return [0.0] * len(our_infosets)
    
    result = []
    for i in range(len(our_infosets)):
        our_card = our_infosets[i][0]
        
        if 'leduc' in game_name.lower() or 'rhode' in game_name.lower() or 'twelve' in game_name.lower():
            valid_values = []
            for j, public_card in enumerate(public_cards):
                if our_card != public_card:
                    valid_values.append(children_values[j][i])
            
            if len(valid_values) > 0:
                avg = sum(valid_values) / len(valid_values)
            else:
                avg = 0.0
        else:
            avg = sum(v[i] for v in children_values) / len(children_values)
        
        result.append(avg)
    
    return result


def policy_matrix(node, node_player, avg_strategy, legal_actions):
    infosets = node[f'player{node_player}_info_sets']
    matrix = []
    
    for info_set_key in infosets:
        key = info_set_key
        if isinstance(info_set_key, tuple) and len(info_set_key) == 4:
            private, public, hist, player = info_set_key
            if public is None:
                key = (private, 'None', hist, player)
        
        strategy = avg_strategy.get(key, {})
        row = [strategy.get(action, 0.0) for action in legal_actions]
        matrix.append(row)
    
    return matrix


def terminal_vector(game_name, player_id, node, r_opp):
    our_infosets = node[f'player{player_id}_info_sets']
    opp_infosets = node[f'player{1-player_id}_info_sets']
    pot = node.get('pot', 0)
    player_bets = node.get('player_bets', [0, 0])
    
    if len(opp_infosets) != len(r_opp):
        raise ValueError(f"Mismatch at terminal: opp_infosets has {len(opp_infosets)} but r_opp has {len(r_opp)}")
    
    result = []
    for our_info in our_infosets:
        our_card = our_info[0]
        value = 0.0
        if 'kuhn' in game_name.lower():
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
                    value += (r_opp[j] / total_valid_r) * util
        else:
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
                    value += (r_opp[j] / total_valid_r) * util
        
        result.append(value)
    
    return result


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
        p0.public_card = player0_info[1]
        p1.public_card = player1_info[1]
        history = list(player0_info[2])
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


def get_node(tree, hist):
    return tree['public_states'][hist]


def compute_best_response_value(game_name, player_id, tree, avg_strategy, root_hist=()):
    root_node = get_node(tree, root_hist)
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
        
        v_vector = traverse(game_name, player_id, tree, avg_strategy, root_hist, r_opp_conditional)
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
