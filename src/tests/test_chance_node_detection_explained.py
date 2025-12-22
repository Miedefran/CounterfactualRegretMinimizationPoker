"""
Erklärung: Was macht der Debug-Test und was bedeuten die Zahlen?
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.twelve_card_poker.game import TwelveCardPokerGame
from utils.poker_utils import GAME_CONFIGS
from evaluation.build_public_state_tree_v2 import build_public_state_tree


def explain_test():
    """Erklärt was der Test macht und was die Ergebnisse bedeuten."""
    print("\n" + "=" * 80)
    print("ERKLÄRUNG: Was macht der Debug-Test?")
    print("=" * 80)
    
    print("\n1. WAS TUT DER TEST?")
    print("-" * 80)
    print("Der Test baut den Public State Tree für Twelve Card Poker und analysiert:")
    print("  - Wie viele Chance-Nodes es gibt")
    print("  - Wie viele davon nach Round 0 kommen (erste Chance-Node)")
    print("  - Wie viele davon nach Round 1 kommen (zweite Chance-Node)")
    print("  - Ob die Deck-Reduktion funktioniert (bereits ausgeteilte Karten werden ausgeschlossen)")
    
    print("\n2. WAS BEDEUTET 'Sample outcomes: ['Js', 'Jh', 'Jd', 'Qs', 'Qh']'?")
    print("-" * 80)
    print("Ein Chance-Node ist ein Punkt im Spiel, wo eine zufällige Karte ausgeteilt wird.")
    print("'Outcomes' sind die möglichen Karten, die an diesem Punkt ausgeteilt werden können.")
    print("Beispiel:")
    print("  - Nach Round 0: Es können alle 12 Karten ausgeteilt werden")
    print("  - Nach Round 1: Es können nur noch 11 Karten ausgeteilt werden (1 wurde schon ausgeteilt)")
    print("'Sample outcomes' zeigt die ersten 5 möglichen Karten als Beispiel.")
    
    print("\n3. WIE VIELE WEGE GIBT ES, EINE ROUND ZU BEENDEN?")
    print("-" * 80)
    
    game_class = TwelveCardPokerGame
    game_config = GAME_CONFIGS['twelve_card_poker']
    
    print("\nBauen Tree, um die tatsächliche Anzahl zu zählen...")
    tree = build_public_state_tree(game_class, game_config)
    public_states = tree['public_states']
    
    chance_nodes = [(k, v) for k, v in public_states.items() if v['type'] == 'chance']
    
    # Analysiere erste Chance-Nodes
    first_chance_nodes = []
    second_chance_nodes = []
    
    for public_key, node in chance_nodes:
        public_cards = [item for item in public_key 
                       if isinstance(item, str) and len(item) == 2 and item[1] in ['s', 'h', 'd', 'c']]
        if len(public_cards) == 0:
            first_chance_nodes.append((public_key, node))
        elif len(public_cards) == 1:
            second_chance_nodes.append((public_key, node))
    
    print(f"\nTATSÄCHLICHE ZAHLEN:")
    print(f"  Erste Chance-Nodes (nach Round 0): {len(first_chance_nodes)}")
    print(f"  Zweite Chance-Nodes (nach Round 1): {len(second_chance_nodes)}")
    
    # Zähle wie viele verschiedene Wege es gibt, Round 1 zu beenden
    # (für jede Public Card)
    if second_chance_nodes:
        # Gruppiere nach Public Card
        nodes_by_card = {}
        for public_key, node in second_chance_nodes:
            public_cards = [item for item in public_key 
                           if isinstance(item, str) and len(item) == 2 and item[1] in ['s', 'h', 'd', 'c']]
            if public_cards:
                card = public_cards[0]
                if card not in nodes_by_card:
                    nodes_by_card[card] = []
                nodes_by_card[card].append((public_key, node))
        
        print(f"\n  Zweite Chance-Nodes pro Public Card:")
        for card in sorted(nodes_by_card.keys()):
            count = len(nodes_by_card[card])
            print(f"    {card}: {count} Chance-Nodes")
        
        # Berechne Durchschnitt
        avg = len(second_chance_nodes) / len(nodes_by_card) if nodes_by_card else 0
        print(f"\n  Durchschnitt: {avg:.1f} Wege, Round 1 zu beenden (pro Public Card)")
        print(f"  (Nicht 25, sondern {avg:.1f}!)")
    
    print("\n" + "=" * 80)
    print("FAZIT:")
    print("=" * 80)
    print(f"- Es gibt {len(first_chance_nodes)} verschiedene Public States, die zu einer")
    print(f"  Chance-Node nach Round 0 führen (verschiedene Action-Sequenzen)")
    print(f"- Es gibt {len(second_chance_nodes)} verschiedene Public States, die zu einer")
    print(f"  Chance-Node nach Round 1 führen")
    print(f"- Das sind {len(second_chance_nodes) / 12:.1f} verschiedene Wege pro Public Card,")
    print(f"  Round 1 zu beenden (nicht 25!)")
    print("=" * 80)


if __name__ == "__main__":
    explain_test()
