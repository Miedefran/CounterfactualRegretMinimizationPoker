"""
Detaillierte Analyse: Warum gibt es genau 25 Chance-Nodes nach Round 1?
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.twelve_card_poker.game import TwelveCardPokerGame
from utils.poker_utils import GAME_CONFIGS
from evaluation.build_public_state_tree_v2 import build_public_state_tree


def analyze_round1_chance_nodes():
    """Analysiere im Detail, warum es 25 Chance-Nodes nach Round 1 gibt."""
    print("\n" + "=" * 80)
    print("DETAILLIERTE ANALYSE: Warum 25 Chance-Nodes nach Round 1?")
    print("=" * 80)
    
    game_class = TwelveCardPokerGame
    game_config = GAME_CONFIGS['twelve_card_poker']
    
    print("\nBauen Tree...")
    tree = build_public_state_tree(game_class, game_config)
    public_states = tree['public_states']
    
    # Finde alle zweiten Chance-Nodes (mit genau 1 Public Card)
    second_chance_nodes = []
    for public_key, node in public_states.items():
        if node['type'] == 'chance':
            public_cards = [item for item in public_key 
                           if isinstance(item, str) and len(item) == 2 and item[1] in ['s', 'h', 'd', 'c']]
            if len(public_cards) == 1:
                second_chance_nodes.append((public_key, node, public_cards[0]))
    
    print(f"\nGefunden: {len(second_chance_nodes)} zweite Chance-Nodes")
    
    # Gruppiere nach Public Card
    nodes_by_card = {}
    for public_key, node, card in second_chance_nodes:
        if card not in nodes_by_card:
            nodes_by_card[card] = []
        nodes_by_card[card].append((public_key, node))
    
    # Analysiere eine Public Card im Detail (z.B. 'Js')
    example_card = 'Js'
    if example_card in nodes_by_card:
        print(f"\n" + "=" * 80)
        print(f"DETAILLIERTE ANALYSE FÜR PUBLIC CARD: {example_card}")
        print("=" * 80)
        
        nodes = nodes_by_card[example_card]
        print(f"\nEs gibt {len(nodes)} verschiedene Public States, die zu einer Chance-Node führen")
        print(f"nachdem {example_card} als erste Public Card ausgeteilt wurde.\n")
        
        # Analysiere die Action-Sequenzen in Round 1
        print("Diese Public States unterscheiden sich durch die Actions in Round 1:")
        print("-" * 80)
        
        round1_actions_by_node = []
        for public_key, node in nodes:
            # Extrahiere nur die Actions NACH der Public Card
            actions = []
            found_card = False
            for item in public_key:
                if isinstance(item, str) and len(item) == 2 and item[1] in ['s', 'h', 'd', 'c']:
                    if item == example_card:
                        found_card = True
                        continue
                if found_card and item in ['bet', 'check', 'call', 'fold']:
                    actions.append(item)
            
            round1_actions_by_node.append((tuple(actions), public_key, node))
        
        # Gruppiere nach Action-Sequenz
        actions_patterns = {}
        for actions_tuple, public_key, node in round1_actions_by_node:
            if actions_tuple not in actions_patterns:
                actions_patterns[actions_tuple] = []
            actions_patterns[actions_tuple].append((public_key, node))
        
        print(f"\nVerschiedene Action-Sequenzen in Round 1 (nach {example_card}):")
        print(f"Anzahl verschiedener Sequenzen: {len(actions_patterns)}\n")
        
        # Zeige alle verschiedenen Sequenzen
        for i, (actions_tuple, examples) in enumerate(sorted(actions_patterns.items(), key=lambda x: (len(x[0]), x[0])), 1):
            print(f"{i:2d}. Round 1 Actions: {list(actions_tuple)}")
            print(f"    → {len(examples)} verschiedene Public State(s) mit dieser Sequenz")
            
            # Analysiere warum es mehrere Public States gibt
            round0_patterns = {}
            for public_key, node in examples:
                # Finde die Actions VOR der Public Card (Round 0)
                actions_before_card = []
                for item in public_key:
                    if item == example_card:
                        break
                    if item in ['bet', 'check', 'call', 'fold']:
                        actions_before_card.append(item)
                
                pattern = tuple(actions_before_card)
                if pattern not in round0_patterns:
                    round0_patterns[pattern] = []
                round0_patterns[pattern].append(public_key)
            
            print(f"    Warum {len(examples)} verschiedene States?")
            print(f"    → Es gibt {len(round0_patterns)} verschiedene Wege, Round 0 zu beenden:")
            for pattern, keys in sorted(round0_patterns.items()):
                print(f"      - Round 0: {list(pattern)} → Card: {example_card} → Round 1: {list(actions_tuple)}")
            
            # Zeige ein Beispiel
            if examples:
                example_key = examples[0][0]
                card_pos = None
                for idx, item in enumerate(example_key):
                    if item == example_card:
                        card_pos = idx
                        break
                if card_pos is not None:
                    before = list(example_key[:card_pos])
                    after = list(example_key[card_pos+1:])
                    print(f"    Beispiel-History: {before} → {example_card} → {after}")
            print()
        
        print(f"\n" + "-" * 80)
        print(f"ZUSAMMENFASSUNG:")
        print(f"- Es gibt {len(actions_patterns)} verschiedene Action-Sequenzen in Round 1")
        print(f"- Jede Sequenz führt zu einem anderen Public State")
        print(f"- Alle diese Public States führen zu einer Chance-Node nach Round 1")
        print(f"- Daher: {len(actions_patterns)} Chance-Nodes für {example_card}")
        
        # Prüfe ob alle Karten die gleiche Anzahl haben
        print(f"\n" + "=" * 80)
        print("VERGLEICH MIT ANDEREN PUBLIC CARDS:")
        print("=" * 80)
        for card in sorted(nodes_by_card.keys()):
            count = len(nodes_by_card[card])
            print(f"  {card}: {count} Chance-Nodes")
        
        if len(set(len(nodes_by_card[card]) for card in nodes_by_card.keys())) == 1:
            print(f"\n✓ Alle Public Cards haben die gleiche Anzahl ({len(nodes_by_card[example_card])})")
            print(f"  Das macht Sinn, da die Anzahl nur von den möglichen Action-Sequenzen")
            print(f"  in Round 1 abhängt, nicht von der spezifischen Karte.")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    analyze_round1_chance_nodes()
