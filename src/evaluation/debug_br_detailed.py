import gzip
import pickle
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.best_response_agent import (
    load_public_tree, load_average_strategy, get_node, policy_matrix,
    terminal_vector, compute_payoff
)


def debug_traverse(game_name, player_id, tree, avg_strategy, hist, r_opp, depth=0):
    node = get_node(tree, hist)
    node_type = node['type']
    indent = "  " * depth
    
    our_info_count = len(node[f'player{player_id}_info_sets'])
    opp_info_count = len(node[f'player{1-player_id}_info_sets'])
    
    print(f"{indent}{'='*60}")
    print(f"{indent}[DEPTH {depth}] Node: {str(hist)}")
    print(f"{indent}Type: {node_type}")
    print(f"{indent}Our Info-Sets: {our_info_count}, Opp Info-Sets: {opp_info_count}")
    
    our_infosets = node[f'player{player_id}_info_sets']
    opp_infosets = node[f'player{1-player_id}_info_sets']
    print(f"{indent}Our Info-Set Keys: {our_infosets}")
    print(f"{indent}Opp Info-Set Keys: {opp_infosets}")
    
    print(f"{indent}r_opp IN: {[f'{x:.3f}' for x in r_opp]}")
    
    if node_type == 'terminal':
        print(f"{indent}>>> TERMINAL NODE")
        result = terminal_vector(game_name, player_id, node, r_opp)
        print(f"{indent}Result: {[f'{x:.3f}' for x in result]}")
        print(f"{indent}{'='*60}")
        return result
    
    if node_type == 'chance':
        print(f"{indent}>>> CHANCE NODE (Leduc: {('leduc' in game_name.lower())})")
        children_values = []
        public_cards = []
        
        if 'leduc' in game_name.lower():
            opp_infosets = node[f'player{1-player_id}_info_sets']
            print(f"{indent}Opp Info-Sets: {[info[0] for info in opp_infosets]}")
            
            for action, child_hist in node['children'].items():
                public_card = action
                public_cards.append(public_card)
                print(f"{indent}  Child: Public Card = '{public_card}'")
                
                r_child = []
                for j, opp_info in enumerate(opp_infosets):
                    opp_card = opp_info[0]
                    if opp_card == public_card:
                        r_child.append(0.0)
                        print(f"{indent}    r[{j}] ({opp_card}) = 0.0 (public)")
                    else:
                        r_child.append(r_opp[j])
                        print(f"{indent}    r[{j}] ({opp_card}) = {r_opp[j]:.3f}")
                
                r_sum = sum(r_child)
                print(f"{indent}    r_sum = {r_sum:.3f}")
                
                if r_sum > 0:
                    r_child_normalized = [x / r_sum for x in r_child]
                else:
                    r_child_normalized = [0.0] * len(r_child)
                
                print(f"{indent}    r_child_normalized: {[f'{x:.3f}' for x in r_child_normalized]}")
                
                v = debug_traverse(game_name, player_id, tree, avg_strategy, child_hist, r_child_normalized, depth+1)
                print(f"{indent}    Child returned: {[f'{x:.3f}' for x in v]}")
                children_values.append(v)
        else:
            for child_hist in node['children'].values():
                v = debug_traverse(game_name, player_id, tree, avg_strategy, child_hist, r_opp, depth+1)
                children_values.append(v)
        
        if not children_values:
            result = [0.0] * our_info_count
            print(f"{indent}No children, returning zeros")
        else:
            our_infosets = node[f'player{player_id}_info_sets']
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
                        print(f"{indent}Info-Set {i} (card={our_card}): averaged {len(valid_values)} valid values (excluded {len(public_cards) - len(valid_values)} impossible)")
                    else:
                        avg = 0.0
                        print(f"{indent}Info-Set {i} (card={our_card}): no valid values!")
                else:
                    avg = sum(v[i] for v in children_values) / len(children_values)
                
                result.append(avg)
            print(f"{indent}Averaged children for {len(our_infosets)} info-sets")
        
        print(f"{indent}Result: {[f'{x:.3f}' for x in result]}")
        print(f"{indent}{'='*60}")
        return result
    
    if node_type == 'choice':
        if node['player'] == (1 - player_id):
            print(f"{indent}>>> OPPONENT CHOICE")
            legal_actions = list(node['children'].keys())
            print(f"{indent}Legal Actions: {legal_actions}")
            
            sigma = policy_matrix(node, 1-player_id, avg_strategy, legal_actions)
            print(f"{indent}Policy Matrix ({len(sigma)} rows):")
            for i, row in enumerate(sigma):
                print(f"{indent}  Info-Set {i}: {[f'{x:.3f}' for x in row]}")
            
            result = None
            for j, action in enumerate(legal_actions):
                print(f"{indent}  Action '{action}':")
                sigma_column = [row[j] for row in sigma]
                r_child = [r_opp[k] * sigma_column[k] for k in range(len(r_opp))]
                r_sum = sum(r_child)
                print(f"{indent}    r_sum = {r_sum:.3f}")
                
                if r_sum > 0:
                    r_child_normalized = [x / r_sum for x in r_child]
                    child_hist = node['children'][action]
                    v_j = debug_traverse(game_name, player_id, tree, avg_strategy, child_hist, r_child_normalized, depth+1)
                    
                    weighted_v_j = [r_sum * v for v in v_j]
                    print(f"{indent}    v_j weighted: {[f'{x:.3f}' for x in weighted_v_j]}")
                    
                    if result is None:
                        result = weighted_v_j
                    else:
                        result = [result[i] + weighted_v_j[i] for i in range(len(result))]
            
            if result is None:
                result = [0.0] * our_info_count
            
            print(f"{indent}Result: {[f'{x:.3f}' for x in result]}")
        else:
            print(f"{indent}>>> OUR CHOICE")
            legal_actions = list(node['children'].keys())
            print(f"{indent}Legal Actions: {legal_actions}")
            
            action_values = []
            for action in legal_actions:
                print(f"{indent}  Action '{action}':")
                child_hist = node['children'][action]
                v = debug_traverse(game_name, player_id, tree, avg_strategy, child_hist, r_opp, depth+1)
                print(f"{indent}    Action value: {[f'{x:.3f}' for x in v]}")
                action_values.append(v)
            
            if not action_values:
                result = [0.0] * our_info_count
            else:
                num_infosets = len(action_values[0])
                result = []
                for i in range(num_infosets):
                    max_val = max(v[i] for v in action_values)
                    result.append(max_val)
            
            print(f"{indent}Result (max): {[f'{x:.3f}' for x in result]}")
        
        print(f"{indent}{'='*60}")
        return result
    
    return []


def main():
    tree_path = 'evaluation/public_state_trees/leduc_public_tree.pkl.gz'
    strategy_path = 'models/leduc/cfr/leduc_1000.pkl.gz'
    output_file = 'evaluation/debug_br_output.txt'
    
    with open(output_file, 'w') as f:
        sys.stdout = f
        
        print("Loading tree...")
        tree = load_public_tree(tree_path)
        print(f"Tree loaded: {len(tree['public_states'])} public states")
        
        print("\nLoading strategy...")
        avg_strategy = load_average_strategy(strategy_path)
        print(f"Strategy loaded: {len(avg_strategy)} info sets")
        
        print("\n" + "="*80)
        print("STRATEGY SAMPLE (first 10 info sets):")
        print("="*80)
        for i, (key, actions) in enumerate(list(avg_strategy.items())[:10]):
            print(f"  {key}")
            print(f"    -> {actions}")
        
        print(f"\n{'='*80}")
        print("TRAVERSING FOR PLAYER 0")
        print(f"{'='*80}\n")
        
        root = get_node(tree, ())
        print(f"Root node info:")
        print(f"  Type: {root['type']}")
        print(f"  Player 0 Info-Sets: {root['player0_info_sets']}")
        print(f"  Player 1 Info-Sets: {root['player1_info_sets']}")
        
        r_opp_initial = [1.0/6] * 6
        print(f"\nInitial r_opp: {r_opp_initial}")
        print()
        
        result = debug_traverse('leduc', 0, tree, avg_strategy, (), r_opp_initial, depth=0)
        
        print("\n" + "="*80)
        print(f"FINAL RESULT VECTOR ({len(result)} values):")
        for i, val in enumerate(result):
            info_set = root['player0_info_sets'][i]
            print(f"  [{i}] {info_set}: {val:.6f}")
        print(f"\nExpected: {len(root['player0_info_sets'])} values")
        print(f"Got: {len(result)} values")
        print("="*80)
        
        sys.stdout = sys.__stdout__
    
    print(f"Debug output written to {output_file}")


if __name__ == '__main__':
    main()

