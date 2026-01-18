"""
Zusammengefasste Regret-Analyse-Tools.

Enthält Funktionen für:
- Berechnung von Spielstatistiken (M-Wert, Infosets, etc.)
- Berechnung theoretischer Regret Bounds
- Berechnung praktischen Regrets aus trainierten Modellen
- Vergleich von praktischem und theoretischem Regret

Usage:
    # Spielstatistiken berechnen
    python src/evaluation/regret_analysis.py compute_stats <game_name>
    
    # Theoretische Bounds berechnen
    python src/evaluation/regret_analysis.py compute_bounds <game_name> --iterations 10000
    
    # Praktisches Regret berechnen
    python src/evaluation/regret_analysis.py compute_regret <model_path>
    
    # Vergleich
    python src/evaluation/regret_analysis.py compare <model_path> [--algorithm ALGORITHM]
"""

import os
import sys
import json
import math
import gzip
import pickle as pkl
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.build_game_tree import build_game_tree, load_game_tree, save_game_tree
from envs.kuhn_poker.game import KuhnPokerGame
from envs.leduc_holdem.game import LeducHoldemGame
from envs.rhode_island.game import RhodeIslandGame
from envs.twelve_card_poker.game import TwelveCardPokerGame
from envs.royal_holdem.game import RoyalHoldemGame
from envs.limit_holdem.game import LimitHoldemGame
from utils.poker_utils import (
    GAME_CONFIGS,
    KuhnPokerCombinations,
    LeducHoldemCombinations,
    RhodeIslandCombinations,
    TwelveCardPokerCombinations,
    RoyalHoldemCombinations,
    LimitHoldemCombinations,
)


# ============================================================================
# Spielstatistiken
# ============================================================================

def compute_m_value(tree, player_id):
    """Berechnet den M-Wert für einen Spieler."""
    action_sequences_to_infosets = defaultdict(set)
    
    def traverse(node_id, player_action_sequence):
        node = tree.nodes[node_id]
        if node.type == 'terminal':
            return
        
        if node.type == 'decision':
            current_player = node.player
            if current_player == player_id and node.infoset_key is not None:
                action_seq_tuple = tuple(player_action_sequence)
                action_sequences_to_infosets[action_seq_tuple].add(node.infoset_key)
            
            for action, child_id in node.children.items():
                new_sequence = list(player_action_sequence)
                if current_player == player_id:
                    new_sequence.append(action)
                traverse(child_id, new_sequence)
    
    for root_id in tree.root_nodes:
        traverse(root_id, [])
    
    m_value = 0.0
    for action_sequence, infoset_set in action_sequences_to_infosets.items():
        num_infosets = len(infoset_set)
        if num_infosets > 0:
            m_value += math.sqrt(num_infosets)
    
    return m_value


def compute_infoset_counts(tree):
    """Zählt die Anzahl der Infosets pro Spieler."""
    player_infosets = {0: set(), 1: set()}
    
    for infoset_key, node_ids in tree.infoset_to_nodes.items():
        if node_ids:
            first_node = tree.nodes[node_ids[0]]
            if first_node.type == 'decision' and first_node.player is not None:
                player_id = first_node.player
                player_infosets[player_id].add(infoset_key)
    
    return {
        0: len(player_infosets[0]),
        1: len(player_infosets[1])
    }


def compute_utility_range(tree, player_id):
    """Berechnet den Nutzenbereich Delta_u für einen Spieler."""
    min_util = float('inf')
    max_util = float('-inf')
    
    for node_id, node in tree.nodes.items():
        if node.type == 'terminal' and node.payoffs is not None:
            utility = node.payoffs[player_id]
            min_util = min(min_util, utility)
            max_util = max(max_util, utility)
    
    delta_u = max_util - min_util if max_util != float('-inf') and min_util != float('inf') else 0.0
    return min_util, max_util, delta_u


def compute_max_actions(tree, player_id):
    """Berechnet die maximale Anzahl an Aktionen pro Spieler."""
    max_actions = 0
    for node_id, node in tree.nodes.items():
        if node.type == 'decision' and node.player == player_id:
            num_actions = len(node.legal_actions) if node.legal_actions else 0
            max_actions = max(max_actions, num_actions)
    return max_actions


def compute_game_stats(game_name, build_if_missing=True):
    """Berechnet alle Spielstatistiken für ein gegebenes Spiel."""
    print(f"\n=== Berechne Statistiken für {game_name} ===\n")
    
    try:
        print(f"Lade Game Tree für {game_name}...")
        tree = load_game_tree(game_name)
    except FileNotFoundError:
        if not build_if_missing:
            raise FileNotFoundError(f"Game Tree für {game_name} nicht gefunden")
        
        print(f"Game Tree nicht gefunden, baue Tree für {game_name}...")
        config = GAME_CONFIGS[game_name]
        
        if game_name.startswith('kuhn'):
            game = KuhnPokerGame(ante=config['ante'], bet_size=config['bet_size'])
            combo_gen = KuhnPokerCombinations()
        elif game_name == 'leduc':
            game = LeducHoldemGame(ante=config['ante'], bet_sizes=config['bet_sizes'], bet_limit=config['bet_limit'])
            combo_gen = LeducHoldemCombinations()
        elif game_name == 'rhode_island':
            game = RhodeIslandGame(ante=config['ante'], bet_sizes=config['bet_sizes'], bet_limit=config['bet_limit'])
            combo_gen = RhodeIslandCombinations()
        elif game_name == 'twelve_card_poker':
            game = TwelveCardPokerGame(ante=config['ante'], bet_sizes=config['bet_sizes'], bet_limit=config['bet_limit'])
            combo_gen = TwelveCardPokerCombinations()
        elif game_name == 'royal_holdem':
            game = RoyalHoldemGame(ante=config['ante'], bet_sizes=config['bet_sizes'], bet_limit=config['bet_limit'])
            combo_gen = RoyalHoldemCombinations()
        elif game_name == 'limit_holdem':
            game = LimitHoldemGame(small_blind=config['small_blind'], big_blind=config['big_blind'], 
                                  bet_sizes=config['bet_sizes'], bet_limit=config['bet_limit'])
            combo_gen = LimitHoldemCombinations()
        else:
            raise ValueError(f"Unbekanntes Spiel: {game_name}")
        
        tree = build_game_tree(game, combo_gen, game_name=game_name, game_config=config)
        save_game_tree(tree, game_name)
    
    print(f"Tree geladen: {len(tree.nodes)} Nodes, {len(tree.infoset_to_nodes)} unique Infosets\n")
    
    infoset_counts = compute_infoset_counts(tree)
    m_values = {}
    utility_ranges = {}
    max_actions = {}
    
    for player_id in [0, 1]:
        m_values[player_id] = compute_m_value(tree, player_id)
        min_util, max_util, delta_u = compute_utility_range(tree, player_id)
        utility_ranges[player_id] = {'min': min_util, 'max': max_util, 'delta': delta_u}
        max_actions[player_id] = compute_max_actions(tree, player_id)
    
    stats = {
        'game_name': game_name,
        'num_nodes': len(tree.nodes),
        'num_infosets_total': len(tree.infoset_to_nodes),
        'players': {
            0: {
                'num_infosets': infoset_counts[0],
                'm_value': m_values[0],
                'utility_range': utility_ranges[0],
                'max_actions': max_actions[0]
            },
            1: {
                'num_infosets': infoset_counts[1],
                'm_value': m_values[1],
                'utility_range': utility_ranges[1],
                'max_actions': max_actions[1]
            }
        }
    }
    
    return stats


def save_stats(stats, output_dir=None):
    """Speichert die Statistiken in einer JSON-Datei."""
    if output_dir is None:
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(script_dir, 'data', 'game_stats')
    
    os.makedirs(output_dir, exist_ok=True)
    
    game_name = stats['game_name']
    filename = f"{game_name}_stats.json"
    filepath = os.path.join(output_dir, filename)
    
    def convert_to_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (int, float)):
            return float(obj) if isinstance(obj, float) else int(obj)
        else:
            return obj
    
    serializable_stats = convert_to_json_serializable(stats)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
    
    print(f"\nStatistiken gespeichert in: {filepath}")
    return filepath


# ============================================================================
# Theoretische Regret Bounds
# ============================================================================

def load_game_stats_from_file(game_name):
    """Lädt die Spielstatistiken aus der JSON-Datei."""
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    stats_file = os.path.join(script_dir, 'data', 'game_stats', f"{game_name}_stats.json")
    
    if not os.path.exists(stats_file):
        raise FileNotFoundError(f"Statistiken für {game_name} nicht gefunden. Bitte zuerst compute_stats ausführen.")
    
    with open(stats_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_vanilla_cfr_bound(delta_u, m_value, max_actions, T):
    """Berechnet das Regret Bound für Vanilla CFR."""
    return delta_u * m_value * math.sqrt(max_actions) / math.sqrt(T)


def compute_external_sampling_bound(delta_u, m_value, max_actions, T, p=0.05):
    """Berechnet das Regret Bound für External Sampling MCCFR."""
    constant_factor = 1 + math.sqrt(2 / p)
    return constant_factor * delta_u * m_value * math.sqrt(max_actions) / math.sqrt(T)


def compute_outcome_sampling_bound(delta_u, m_value, max_actions, T, p=0.05, delta=0.1):
    """Berechnet das Regret Bound für Outcome Sampling MCCFR."""
    constant_factor = 1 + math.sqrt(2 / p) * (1 / math.sqrt(delta))
    return constant_factor * delta_u * m_value * math.sqrt(max_actions) / math.sqrt(T)


def compute_cfr_plus_bound(total_infosets, delta_u, max_actions, T):
    """Berechnet das Regret Bound für CFR+."""
    return 2 * total_infosets * delta_u * math.sqrt(max_actions) / math.sqrt(T)


def compute_bounds_for_game(game_name, T=10000, p=0.05, delta=0.1):
    """Berechnet alle Regret Bounds für ein gegebenes Spiel."""
    stats = load_game_stats_from_file(game_name)
    
    print(f"\n=== Regret Bounds für {game_name} (T = {T}) ===\n")
    
    bounds = {
        'game_name': game_name,
        'iterations': T,
        'p': p,
        'delta': delta,
        'players': {}
    }
    
    for player_id in ['0', '1']:
        player_data = stats['players'][player_id]
        delta_u = player_data['utility_range']['delta']
        m_value = player_data['m_value']
        max_actions = player_data['max_actions']
        
        vanilla_bound = compute_vanilla_cfr_bound(delta_u, m_value, max_actions, T)
        external_bound = compute_external_sampling_bound(delta_u, m_value, max_actions, T, p)
        outcome_bound = compute_outcome_sampling_bound(delta_u, m_value, max_actions, T, p, delta)
        
        bounds['players'][player_id] = {
            'bounds': {
                'vanilla_cfr': vanilla_bound,
                'external_sampling': external_bound,
                'outcome_sampling': outcome_bound
            }
        }
    
    total_infosets = stats['num_infosets_total']
    max_delta_u = max(stats['players']['0']['utility_range']['delta'], 
                      stats['players']['1']['utility_range']['delta'])
    max_actions_global = max(stats['players']['0']['max_actions'], 
                              stats['players']['1']['max_actions'])
    
    cfr_plus_bound = compute_cfr_plus_bound(total_infosets, max_delta_u, max_actions_global, T)
    bounds['cfr_plus'] = {'bound': cfr_plus_bound}
    
    return bounds


# ============================================================================
# Praktisches Regret
# ============================================================================

def load_model(filepath):
    """Lädt ein gespeichertes CFR Modell."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Modell nicht gefunden: {filepath}")
    
    with gzip.open(filepath, 'rb') as f:
        return pkl.load(f)


def compute_average_regret_from_regrets(cumulative_regret, T):
    """Berechnet das durchschnittliche Regret aus akkumulierten counterfactual regrets."""
    if T == 0:
        return 0.0
    
    total_regret = 0.0
    for infoset_key, action_regrets in cumulative_regret.items():
        if not action_regrets:
            continue
        
        max_positive_regret = 0.0
        for action, regret in action_regrets.items():
            positive_regret = max(regret, 0.0)
            max_positive_regret = max(max_positive_regret, positive_regret)
        
        total_regret += max_positive_regret
    
    return total_regret / T


# ============================================================================
# Vergleich
# ============================================================================

def determine_algorithm_from_path(model_path):
    """Versucht den Algorithmus aus dem Pfad zu bestimmen."""
    path_lower = model_path.lower()
    if 'external_sampling' in path_lower:
        return 'external_sampling'
    elif 'outcome_sampling' in path_lower:
        return 'outcome_sampling'
    elif 'cfr_plus' in path_lower or 'cfr+' in path_lower:
        return 'cfr_plus'
    elif 'cfr_with_tree' in path_lower or 'cfr' in path_lower:
        return 'vanilla_cfr'
    return None


def determine_game_name_from_path(model_path):
    """Versucht den Spielnamen aus dem Pfad zu bestimmen."""
    path_parts = model_path.split('/')
    for part in path_parts:
        if part in ['leduc', 'kuhn_case1', 'kuhn_case2', 'kuhn_case3', 'kuhn_case4', 
                    'rhode_island', 'twelve_card_poker', 'royal_holdem', 'limit_holdem']:
            return part
    return None


# ============================================================================
# CLI
# ============================================================================

def cmd_compute_stats(args):
    """Befehls-Handler für compute_stats."""
    stats = compute_game_stats(args.game_name)
    save_stats(stats)
    print("\n=== Fertig ===")


def cmd_compute_bounds(args):
    """Befehls-Handler für compute_bounds."""
    bounds = compute_bounds_for_game(args.game_name, args.iterations, args.p, args.delta)
    
    if args.save:
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(script_dir, 'data', 'game_stats')
        os.makedirs(output_dir, exist_ok=True)
        
        game_name = bounds['game_name']
        T = bounds['iterations']
        filename = f"{game_name}_bounds_T{T}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(bounds, f, indent=2, ensure_ascii=False)
        
        print(f"\nBounds gespeichert in: {filepath}")
    
    print("\n=== Fertig ===")


def cmd_compute_regret(args):
    """Befehls-Handler für compute_regret."""
    print(f"\n=== Lade Modell: {args.model_path} ===\n")
    data = load_model(args.model_path)
    
    cumulative_regret = data.get('cumulative_regret') or data.get('regret_sum') or data.get('Q')
    iteration_count = data.get('iteration_count', 0)
    
    if cumulative_regret is None:
        print("FEHLER: Keine Regret-Daten im Modell gefunden!")
        sys.exit(1)
    
    print(f"Iterationen: {iteration_count}")
    print(f"Anzahl Infosets mit Regrets: {len(cumulative_regret)}\n")
    
    if iteration_count > 0:
        practical_regret = compute_average_regret_from_regrets(cumulative_regret, iteration_count)
        print(f"Durchschnittliches Regret: R^T = {practical_regret:.6f}")
    else:
        print("WARNUNG: iteration_count ist 0, kann Regret nicht berechnen")
    
    print("\n=== Fertig ===")


def cmd_compare(args):
    """Befehls-Handler für compare."""
    print(f"\n=== Lade Modell: {args.model_path} ===\n")
    data = load_model(args.model_path)
    
    cumulative_regret = data.get('cumulative_regret') or data.get('regret_sum') or data.get('Q')
    iteration_count = data.get('iteration_count', 0)
    
    if cumulative_regret is None:
        print("FEHLER: Keine Regret-Daten im Modell gefunden!")
        sys.exit(1)
    
    practical_regret = compute_average_regret_from_regrets(cumulative_regret, iteration_count)
    
    game_name = determine_game_name_from_path(args.model_path)
    algorithm = args.algorithm or determine_algorithm_from_path(args.model_path)
    
    if not game_name:
        print("WARNUNG: Konnte Spielname nicht bestimmen. Verwende 'leduc'.")
        game_name = 'leduc'
    
    if not algorithm:
        print("WARNUNG: Konnte Algorithmus nicht bestimmen. Verwende 'vanilla_cfr'.")
        algorithm = 'vanilla_cfr'
    
    print(f"Spiel: {game_name}")
    print(f"Algorithmus: {algorithm}")
    print(f"Praktisches Regret: R^T = {practical_regret:.6f}")
    print()
    
    try:
        stats = load_game_stats_from_file(game_name)
    except FileNotFoundError:
        print(f"FEHLER: Statistiken für {game_name} nicht gefunden!")
        print("Bitte zuerst 'compute_stats' ausführen.")
        sys.exit(1)
    
    print("=== Theoretische Bounds ===\n")
    
    max_bound = 0.0
    for player_id in ['0', '1']:
        player_data = stats['players'][player_id]
        delta_u = player_data['utility_range']['delta']
        m_value = player_data['m_value']
        max_actions = player_data['max_actions']
        
        if algorithm == 'vanilla_cfr':
            bound = compute_vanilla_cfr_bound(delta_u, m_value, max_actions, iteration_count)
        elif algorithm == 'external_sampling':
            bound = compute_external_sampling_bound(delta_u, m_value, max_actions, iteration_count, args.p)
        elif algorithm == 'outcome_sampling':
            bound = compute_outcome_sampling_bound(delta_u, m_value, max_actions, iteration_count, args.p, args.delta)
        else:
            bound = 0.0
        
        max_bound = max(max_bound, bound)
    
    if algorithm == 'cfr_plus':
        total_infosets = stats['num_infosets_total']
        max_delta_u = max(stats['players']['0']['utility_range']['delta'], 
                          stats['players']['1']['utility_range']['delta'])
        max_actions_global = max(stats['players']['0']['max_actions'], 
                                  stats['players']['1']['max_actions'])
        max_bound = compute_cfr_plus_bound(total_infosets, max_delta_u, max_actions_global, iteration_count)
    
    print(f"Theoretischer Bound (max): {max_bound:.6f}\n")
    print("=== Vergleich ===\n")
    
    if practical_regret <= max_bound:
        ratio = practical_regret / max_bound if max_bound > 0 else 0
        print(f"✓ OK: Praktisches Regret ≤ Theoretischer Bound ({ratio:.2%})")
    else:
        ratio = practical_regret / max_bound if max_bound > 0 else float('inf')
        print(f"✗ PROBLEM: Praktisches Regret > Theoretischer Bound ({ratio:.2%})")
    
    print("\n=== Fertig ===")


def main():
    parser = argparse.ArgumentParser(description='Regret-Analyse-Tools')
    subparsers = parser.add_subparsers(dest='command', help='Befehl')
    
    # compute_stats
    parser_stats = subparsers.add_parser('compute_stats', help='Berechnet Spielstatistiken')
    parser_stats.add_argument('game_name', type=str, help='Name des Spiels')
    
    # compute_bounds
    parser_bounds = subparsers.add_parser('compute_bounds', help='Berechnet theoretische Regret Bounds')
    parser_bounds.add_argument('game_name', type=str, help='Name des Spiels')
    parser_bounds.add_argument('--iterations', '-T', type=int, default=10000, help='Anzahl Iterationen')
    parser_bounds.add_argument('--p', type=float, default=0.05, help='Konfidenzparameter p')
    parser_bounds.add_argument('--delta', type=float, default=0.1, help='Sampling policy Parameter δ')
    parser_bounds.add_argument('--save', action='store_true', help='Speichere Bounds in JSON')
    
    # compute_regret
    parser_regret = subparsers.add_parser('compute_regret', help='Berechnet praktisches Regret')
    parser_regret.add_argument('model_path', type=str, help='Pfad zur Modell-Datei')
    
    # compare
    parser_compare = subparsers.add_parser('compare', help='Vergleicht praktisches mit theoretischem Regret')
    parser_compare.add_argument('model_path', type=str, help='Pfad zur Modell-Datei')
    parser_compare.add_argument('--algorithm', type=str, 
                                choices=['vanilla_cfr', 'external_sampling', 'outcome_sampling', 'cfr_plus'],
                                help='Algorithmus (wird automatisch bestimmt falls nicht angegeben)')
    parser_compare.add_argument('--p', type=float, default=0.05, help='Konfidenzparameter p')
    parser_compare.add_argument('--delta', type=float, default=0.1, help='Sampling policy Parameter δ')
    
    args = parser.parse_args()
    
    if args.command == 'compute_stats':
        cmd_compute_stats(args)
    elif args.command == 'compute_bounds':
        cmd_compute_bounds(args)
    elif args.command == 'compute_regret':
        cmd_compute_regret(args)
    elif args.command == 'compare':
        cmd_compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
