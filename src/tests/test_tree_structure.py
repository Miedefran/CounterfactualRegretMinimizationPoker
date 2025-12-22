"""
Prüft die Struktur des Trees im Detail - sind die Chance-Nodes an den richtigen Stellen?
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.twelve_card_poker.game import TwelveCardPokerGame
from utils.poker_utils import GAME_CONFIGS
from evaluation.build_public_state_tree_v2 import build_public_state_tree


def check_tree_structure():
    """Prüft ob die Chance-Nodes an den richtigen Stellen sind."""
    print("\n" + "=" * 80)
    print("STRUKTUR-PRÜFUNG: Sind Chance-Nodes an den richtigen Stellen?")
    print("=" * 80)
    
    game_class = TwelveCardPokerGame
    game_config = GAME_CONFIGS['twelve_card_poker']
    
    print("\nBauen Tree...")
    tree = build_public_state_tree(game_class, game_config)
    public_states = tree['public_states']
    
    # Prüfe: Nach ['check', 'check'] sollte ein Chance-Node kommen
    print("\n" + "=" * 80)
    print("PRÜFUNG 1: Nach Round 0 Complete (check-check)")
    print("=" * 80)
    
    check_check_key = ('check', 'check')
    if check_check_key in public_states:
        node = public_states[check_check_key]
        print(f"✓ Public State gefunden: {list(check_check_key)}")
        print(f"  Type: {node['type']}")
        if node['type'] == 'chance':
            print(f"  ✓ Korrekt: Chance-Node nach Round 0!")
            outcomes = list(node.get('children', {}).keys())
            print(f"  Outcomes: {len(outcomes)} (sollte 12 sein)")
            print(f"  Sample: {outcomes[:5]}")
        else:
            print(f"  ✗ FEHLER: Sollte Chance-Node sein, ist aber {node['type']}")
    else:
        print(f"✗ FEHLER: Public State {list(check_check_key)} nicht gefunden!")
    
    # Prüfe: Nach ['check', 'check', 'Js', 'check', 'check'] sollte ein Chance-Node kommen
    print("\n" + "=" * 80)
    print("PRÜFUNG 2: Nach Round 1 Complete (check-check -> Js -> check-check)")
    print("=" * 80)
    
    check_check_js_check_check_key = ('check', 'check', 'Js', 'check', 'check')
    if check_check_js_check_check_key in public_states:
        node = public_states[check_check_js_check_check_key]
        print(f"✓ Public State gefunden: {list(check_check_js_check_check_key)}")
        print(f"  Type: {node['type']}")
        if node['type'] == 'chance':
            print(f"  ✓ Korrekt: Chance-Node nach Round 1!")
            outcomes = list(node.get('children', {}).keys())
            print(f"  Outcomes: {len(outcomes)} (sollte 11 sein, da Js bereits ausgeteilt)")
            print(f"  Sample: {outcomes[:5]}")
            if 'Js' in outcomes:
                print(f"  ✗ FEHLER: Js sollte nicht in Outcomes sein!")
            else:
                print(f"  ✓ Korrekt: Js ist nicht in Outcomes")
        else:
            print(f"  ✗ FEHLER: Sollte Chance-Node sein, ist aber {node['type']}")
    else:
        print(f"✗ FEHLER: Public State {list(check_check_js_check_check_key)} nicht gefunden!")
    
    # Prüfe: Nach ['check', 'check', 'Js'] sollte KEIN Chance-Node sein (Round 1 noch nicht complete)
    print("\n" + "=" * 80)
    print("PRÜFUNG 3: Nach Public Card, aber Round 1 noch nicht complete")
    print("=" * 80)
    
    check_check_js_key = ('check', 'check', 'Js')
    if check_check_js_key in public_states:
        node = public_states[check_check_js_key]
        print(f"Public State gefunden: {list(check_check_js_key)}")
        print(f"  Type: {node['type']}")
        if node['type'] == 'chance':
            print(f"  ✗ FEHLER: Sollte KEIN Chance-Node sein (Round 1 noch nicht complete)!")
        else:
            print(f"  ✓ Korrekt: Kein Chance-Node (Round 1 noch nicht complete)")
            if node['type'] == 'choice':
                print(f"  Player: {node['player']}")
    else:
        print(f"Public State {list(check_check_js_key)} nicht gefunden")
    
    # Zeige einen Pfad durch den Tree
    print("\n" + "=" * 80)
    print("BEISPIEL-PFAD DURCH DEN TREE:")
    print("=" * 80)
    
    path = [
        ((), "Root"),
        (('check',), "Nach check"),
        (('check', 'check'), "Nach check-check (Round 0 complete)"),
        (('check', 'check', 'Js'), "Nach Js ausgeteilt"),
        (('check', 'check', 'Js', 'check'), "Nach check in Round 1"),
        (('check', 'check', 'Js', 'check', 'check'), "Nach check-check in Round 1 (Round 1 complete)"),
    ]
    
    for key, description in path:
        if key in public_states:
            node = public_states[key]
            print(f"\n{description}:")
            print(f"  Key: {list(key)}")
            print(f"  Type: {node['type']}")
            if node['type'] == 'choice':
                print(f"  Player: {node['player']}")
                if 'children' in node:
                    print(f"  Legal actions: {list(node['children'].keys())[:5]}")
            elif node['type'] == 'chance':
                outcomes = list(node.get('children', {}).keys())
                print(f"  Outcomes: {len(outcomes)}")
                print(f"  Sample: {outcomes[:3]}")
        else:
            print(f"\n{description}:")
            print(f"  ✗ Public State nicht gefunden: {list(key)}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    check_tree_structure()
