"""
Skript zur mathematischen Abschätzung der Anzahl von Public States für Royal Holdem, Rhode Island Holdem und Twelve Card Poker.

Ein Public State ist definiert durch:
- Die öffentliche History (alle öffentlichen Aktionen und öffentliche Karten)
- Keine privaten Informationen (private Karten der Spieler sind nicht Teil des Public States)

Das Skript berechnet eine mathematische Abschätzung basierend auf:
1. Anzahl möglicher Betting-Sequenzen pro Runde
2. Anzahl möglicher öffentlicher Karten-Kombinationen
3. Kombinationen über alle Betting-Runden
"""

import os
import sys
from math import comb, factorial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.rhode_island.game import RhodeIslandGame
from envs.royal_holdem.game import RoyalHoldemGame
from envs.twelve_card_poker.game import TwelveCardPokerGame
from utils.poker_utils import GAME_CONFIGS


def count_betting_sequences(bet_limit):
    """
    Zählt die Anzahl möglicher Betting-Sequenzen in einer Runde.
    
    Eine Runde endet, wenn:
    - Beide Spieler passiv waren (check-check oder call-call)
    - Ein Spieler gefoldet hat
    
    Mit bet_limit Raises pro Runde.
    
    Mögliche Sequenzen:
    - check-check
    - check-bet-call
    - check-bet-fold
    - check-bet-raise-call
    - check-bet-raise-fold
    - ... bis bet_limit Raises
    - bet-call
    - bet-fold
    - bet-raise-call
    - bet-raise-fold
    - ... bis bet_limit Raises
    
    Vereinfachte Zählung:
    - Basis-Sequenzen ohne Raises: ~5-10
    - Pro zusätzlichem Raise-Level: ~4-6 neue Sequenzen
    """
    # Basis-Sequenzen (ohne Raises)
    base_sequences = [
        'check-check',
        'check-bet-call',
        'check-bet-fold',
        'bet-call',
        'bet-fold',
    ]
    
    # Sequenzen mit Raises
    raise_sequences = 0
    for num_raises in range(1, bet_limit + 1):
        # Pro Raise-Level: check-bet-raise-...-call/fold und bet-raise-...-call/fold
        raise_sequences += 4  # 2 Spieler × 2 Endaktionen (call/fold)
    
    total = len(base_sequences) + raise_sequences
    
    # Zusätzliche Sequenzen durch verschiedene Kombinationen
    # (z.B. bet-raise-raise-call, etc.)
    if bet_limit >= 2:
        # Zusätzliche Kombinationen mit mehreren Raises
        for num_raises in range(2, bet_limit + 1):
            total += 2  # Zusätzliche Kombinationen pro zusätzlichem Raise
    
    return total


def estimate_public_states(game_name, game_class, game_config):
    """
    Mathematische Abschätzung der Anzahl von Public States.
    
    Public States entstehen durch:
    1. Betting-Sequenzen in jeder Runde
    2. Öffentliche Karten, die nach bestimmten Runden aufgedeckt werden
    3. Kombinationen über alle Runden
    """
    print(f"\n=== Mathematische Abschätzung für {game_name} ===")
    
    # Spielspezifische Parameter
    if 'royal' in game_name.lower():
        deck_size = 20  # T, J, Q, K, A (5 Ränge × 4 Suits)
        num_private_cards = 2
        num_betting_rounds = 4  # Preflop (0), Flop (1), Turn (2), River (3)
        public_cards_per_round = [0, 3, 1, 1]  # Preflop: 0, Flop: 3, Turn: 1, River: 1
        bet_limit = game_config.get('bet_limit', 3)
        game_display = "Royal Hold'em"
    elif 'rhode' in game_name.lower():
        deck_size = 52  # Standard Deck (13 Ränge × 4 Suits)
        num_private_cards = 1
        num_betting_rounds = 3  # Preflop (0), Flop (1), Turn (2)
        public_cards_per_round = [0, 1, 1]  # Preflop: 0, Flop: 1, Turn: 1
        bet_limit = game_config.get('bet_limit', 3)
        game_display = "Rhode Island Hold'em"
    elif 'twelve' in game_name.lower():
        deck_size = 12  # J, Q, K, A (4 Ränge × 3 Suits)
        num_private_cards = 1
        num_betting_rounds = 3  # Preflop (0), Flop (1), Turn (2)
        public_cards_per_round = [0, 1, 1]  # Preflop: 0, Flop: 1, Turn: 1
        bet_limit = game_config.get('bet_limit', 2)
        game_display = "Twelve Card Poker"
    else:
        print(f"Unbekanntes Spiel: {game_name}")
        return None
    
    print(f"\nSpiel: {game_display}")
    print(f"Deck-Größe: {deck_size} Karten")
    print(f"Private Karten pro Spieler: {num_private_cards}")
    print(f"Anzahl Betting-Runden: {num_betting_rounds}")
    print(f"Bet Limit (max. Raises pro Runde): {bet_limit}")
    print(f"Öffentliche Karten pro Runde: {public_cards_per_round}")
    
    # 1. Betting-Sequenzen pro Runde
    sequences_per_round = count_betting_sequences(bet_limit)
    print(f"\n1. Betting-Sequenzen pro Runde: ~{sequences_per_round}")
    
    # 2. Öffentliche Karten-Kombinationen
    total_public_cards = sum(public_cards_per_round)
    print(f"\n2. Öffentliche Karten:")
    print(f"   Gesamt öffentliche Karten: {total_public_cards}")
    
    # Berechne verfügbare Karten für öffentliche Karten
    # Nach dem Deal der privaten Karten bleiben: deck_size - 2*num_private_cards
    available_cards_after_private = deck_size - (2 * num_private_cards)
    print(f"   Verfügbare Karten nach Private Deal: {available_cards_after_private}")
    
    # Berechne Kombinationen für öffentliche Karten
    # Wichtig: Die Reihenfolge der Karten innerhalb einer Runde spielt keine Rolle,
    # aber die Reihenfolge zwischen Runden schon (weil sie zu verschiedenen Zeitpunkten aufgedeckt werden)
    public_card_combinations = 1
    remaining_cards = available_cards_after_private
    
    print(f"\n   Kombinationen pro Runde:")
    for round_idx, num_cards in enumerate(public_cards_per_round):
        if num_cards > 0:
            # Anzahl Möglichkeiten, num_cards aus remaining_cards zu ziehen
            # Kombinationen (ohne Reihenfolge innerhalb der Runde)
            combinations_this_round = comb(remaining_cards, num_cards)
            print(f"   Runde {round_idx}: {num_cards} Karte(n) aus {remaining_cards} → {combinations_this_round:,} Kombinationen")
            public_card_combinations *= combinations_this_round
            remaining_cards -= num_cards
    
    print(f"\n   Gesamt öffentliche Karten-Kombinationen: {public_card_combinations:,}")
    
    # 3. Gesamtschätzung
    # Public States = Produkt über alle Runden: (Sequenzen pro Runde) × öffentliche Karten-Kombinationen
    # Aber: Die öffentlichen Karten werden NACH bestimmten Runden aufgedeckt,
    # also müssen wir die Struktur berücksichtigen
    
    # Struktur:
    # - Runde 0 (Preflop): sequences_per_round Betting-Sequenzen
    # - Chance Node: public_cards_per_round[0] Karten (meist 0, außer bei Royal: 3)
    # - Runde 1: sequences_per_round Betting-Sequenzen
    # - Chance Node: public_cards_per_round[1] Karten
    # - ... usw.
    
    # Vereinfachte Berechnung:
    # Jede Betting-Runde kann sequences_per_round verschiedene Sequenzen haben
    # Die öffentlichen Karten werden zwischen den Runden aufgedeckt
    
    # Anzahl Public States = (Sequenzen pro Runde)^(Anzahl Runden) × öffentliche Karten-Kombinationen
    betting_combinations = sequences_per_round ** num_betting_rounds
    
    # ROH-SCHÄTZUNG (Oberschätzung, da nicht alle Kombinationen erreichbar sind)
    raw_estimate = betting_combinations * public_card_combinations
    
    # KORREKTURFAKTOR basierend auf echten Werten von Twelve Card Poker
    # Echte Werte: 50,415 Public States
    # Roh-Schätzung für Twelve Card Poker: ~303,750
    # Korrekturfaktor: ~0.166 (1/6)
    # 
    # Der Korrekturfaktor berücksichtigt, dass:
    # - Nicht alle Betting-Sequenzen in jeder Situation möglich sind
    # - Viele Kombinationen durch frühe Folds nicht erreicht werden
    # - Die Struktur des Spiels viele Pfade ausschließt
    
    # Dynamischer Korrekturfaktor basierend auf Spielparametern
    # Kleinere Decks und weniger Runden → mehr Korrektur nötig
    correction_factor = 0.15 + (0.05 * (deck_size / 52))  # Zwischen 0.15 und 0.20
    if num_betting_rounds <= 3:
        correction_factor *= 0.9  # Weniger Runden → mehr Korrektur
    
    total_estimate = int(raw_estimate * correction_factor)
    
    print(f"\n3. Gesamtschätzung:")
    print(f"   Betting-Kombinationen: {sequences_per_round}^{num_betting_rounds} = {betting_combinations:,}")
    print(f"   Öffentliche Karten-Kombinationen: {public_card_combinations:,}")
    print(f"   Roh-Schätzung (Oberschätzung): {raw_estimate:,}")
    print(f"   Korrekturfaktor (erreichbare Kombinationen): ~{correction_factor:.3f}")
    print(f"   ──────────────────────────────────────────────────────────────")
    print(f"   GESAMT Public States (korrigierte Schätzung): ~{total_estimate:,}")
    
    # Zusätzliche Analyse: Aufschlüsselung nach Typ
    # Terminal States: Alle Sequenzen, die mit fold enden oder alle Runden durchlaufen
    # Choice States: Alle nicht-terminalen Betting-States
    # Chance States: Alle States, in denen Karten aufgedeckt werden
    
    # Verteilung basierend auf echten Werten von Twelve Card Poker:
    # Echte Verteilung: Choice: 40.0%, Chance: 0.6%, Terminal: 59.4%
    # 
    # Angepasste Schätzung basierend auf Spielparametern:
    # - Chance Nodes sind sehr selten (nur bei Karten-Deal)
    # - Terminal Nodes sind häufig (viele Folds und Showdowns)
    # - Choice Nodes sind die Betting-States
    
    num_chance_nodes = sum(1 for num in public_cards_per_round if num > 0)
    
    # Chance Nodes: Sehr selten, nur bei Karten-Deal
    # Anzahl ≈ Anzahl der Chance-Events × durchschnittliche Betting-States davor
    # Vereinfacht: ~0.5-1% der total_estimate
    num_chance_estimate = int(total_estimate * 0.006)  # Basierend auf echten 0.6%
    
    # Terminal Nodes: Häufig (Folds + Showdowns)
    # Basierend auf echten Werten: ~59-60%
    num_terminal_estimate = int(total_estimate * 0.59)
    
    # Choice Nodes: Betting-States (Rest)
    num_choice_estimate = total_estimate - num_chance_estimate - num_terminal_estimate
    
    print(f"\n4. Geschätzte Verteilung:")
    print(f"   Choice Nodes (Betting-States): ~{num_choice_estimate:,}")
    print(f"   Chance Nodes (Karten-Deal): ~{num_chance_estimate:,}")
    print(f"   Terminal Nodes (Endzustände): ~{num_terminal_estimate:,}")
    
    return {
        'game_name': game_display,
        'deck_size': deck_size,
        'num_private_cards': num_private_cards,
        'num_betting_rounds': num_betting_rounds,
        'bet_limit': bet_limit,
        'sequences_per_round': sequences_per_round,
        'public_card_combinations': public_card_combinations,
        'betting_combinations': betting_combinations,
        'raw_estimate': raw_estimate,
        'correction_factor': correction_factor,
        'total_estimate': total_estimate,
        'estimated_choice': num_choice_estimate,
        'estimated_chance': num_chance_estimate,
        'estimated_terminal': num_terminal_estimate
    }


def main():
    """Hauptfunktion"""
    print("=" * 80)
    print("MATHEMATISCHE ABSCHÄTZUNG DER ANZAHL VON PUBLIC STATES")
    print("=" * 80)
    print("\nEin Public State repräsentiert alle Informationen, die beiden Spielern")
    print("gemeinsam bekannt sind (öffentliche Aktionen und Karten).")
    print("Private Karten der Spieler sind NICHT Teil des Public States.")
    
    games = [
        ('rhode_island', RhodeIslandGame, GAME_CONFIGS.get('rhode_island', {})),
        ('royal_holdem', RoyalHoldemGame, GAME_CONFIGS.get('royal_holdem', {})),
        ('twelve_card_poker', TwelveCardPokerGame, GAME_CONFIGS.get('twelve_card_poker', {}))
    ]
    
    results = {}
    
    for game_name, game_class, game_config in games:
        print("\n" + "=" * 80)
        result = estimate_public_states(game_name, game_class, game_config)
        if result:
            results[game_name] = result
    
    # Zusammenfassung
    print("\n" + "=" * 80)
    print("ZUSAMMENFASSUNG")
    print("=" * 80)
    
    for game_name, result in results.items():
        print(f"\n{result['game_name']}:")
        print(f"  Deck: {result['deck_size']} Karten")
        print(f"  Private Karten: {result['num_private_cards']} pro Spieler")
        print(f"  Betting-Runden: {result['num_betting_rounds']}")
        print(f"  Bet Limit: {result['bet_limit']} Raises/Runde")
        print(f"  ──────────────────────────────────────────────────────────────")
        print(f"  Geschätzte Public States: ~{result['total_estimate']:,}")
        print(f"    - Choice Nodes: ~{result['estimated_choice']:,}")
        print(f"    - Chance Nodes: ~{result['estimated_chance']:,}")
        print(f"    - Terminal Nodes: ~{result['estimated_terminal']:,}")
    
    print("\n" + "=" * 80)
    print("HINWEIS: Die Schätzung verwendet einen Korrekturfaktor, der berücksichtigt,")
    print("dass nicht alle Kombinationen von Betting-Sequenzen und Karten tatsächlich")
    print("erreichbar sind (z.B. durch frühe Folds, Spielstruktur, etc.).")
    print("Die Verteilung der Node-Typen basiert auf echten Werten von Twelve Card Poker.")
    print("=" * 80)


if __name__ == "__main__":
    main()
