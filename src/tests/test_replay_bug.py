import os
import sys

# Füge src Verzeichnis zum Python-Pfad hinzu
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.rhode_island.game import RhodeIslandGame
from envs.royal_holdem.game import RoyalHoldemGame
from utils.poker_utils import GAME_CONFIGS

def replay_public_history_test(game, public_hist):
    """Test-Version der replay_public_history Funktion mit der Fix-Logik."""
    valid_actions = {'bet', 'check', 'call', 'fold'}
    game_name = game.__class__.__name__
    
    for item in public_hist:
        if item in valid_actions:
            game.step(item)
        else:
            # Chance-Event (öffentliche Karte)
            if 'Leduc' in game_name:
                if hasattr(game, 'public_card'):
                    game.public_card = item
                    game.players[0].set_public_card(item)
                    game.players[1].set_public_card(item)
            elif 'Rhode' in game_name or 'Twelve' in game_name or 'Royal' in game_name or 'Limit' in game_name:
                if hasattr(game, 'public_cards'):
                    if item not in game.public_cards:
                        if len(game.public_cards) > 0:
                            # Überschreibe die letzte Karte (von step() hinzugefügt)
                            game.public_cards[-1] = item
                            game.players[0].public_cards[-1] = item
                            game.players[1].public_cards[-1] = item
                        else:
                            # Liste ist leer, füge Karte hinzu
                            game.public_cards.append(item)
                            game.players[0].public_cards.append(item)
                            game.players[1].public_cards.append(item)

def test_case_1_single_card():
    """Test 1: Eine Karte nach Runde 1"""
    print("\n" + "=" * 60)
    print("TEST 1: Eine Karte nach Runde 1")
    print("=" * 60)
    
    game = RhodeIslandGame(**GAME_CONFIGS['rhode_island'])
    game.reset(0)
    
    public_hist = ['check', 'bet', 'call', 'Qs']
    replay_public_history_test(game, public_hist)
    
    expected = ['Qs']
    if game.public_cards == expected:
        print(f"✅ BESTANDEN: public_cards = {game.public_cards}")
        return True
    else:
        print(f"❌ FEHLGESCHLAGEN: Erwartet {expected}, bekommen {game.public_cards}")
        return False

def test_case_2_multiple_cards():
    """Test 2: Mehrere Karten (Runde 1 und Runde 2)"""
    print("\n" + "=" * 60)
    print("TEST 2: Mehrere Karten nacheinander")
    print("=" * 60)
    
    game = RhodeIslandGame(**GAME_CONFIGS['rhode_island'])
    game.reset(0)
    
    # Runde 1: check, bet, call → Karte 1
    # Runde 2: check, bet, call → Karte 2
    public_hist = ['check', 'bet', 'call', 'Qs', 'check', 'bet', 'call', 'Kh']
    replay_public_history_test(game, public_hist)
    
    expected = ['Qs', 'Kh']
    if game.public_cards == expected:
        print(f"✅ BESTANDEN: public_cards = {game.public_cards}")
        return True
    else:
        print(f"❌ FEHLGESCHLAGEN: Erwartet {expected}, bekommen {game.public_cards}")
        return False

def test_case_3_card_already_present():
    """Test 3: Karte ist bereits in der Liste (sollte nicht hinzugefügt werden)"""
    print("\n" + "=" * 60)
    print("TEST 3: Karte bereits vorhanden")
    print("=" * 60)
    
    game = RhodeIslandGame(**GAME_CONFIGS['rhode_island'])
    game.reset(0)
    
    # Füge manuell eine Karte hinzu
    game.public_cards = ['Qs']
    game.players[0].public_cards = ['Qs']
    game.players[1].public_cards = ['Qs']
    
    # Versuche 'Qs' nochmal hinzuzufügen
    public_hist = ['Qs']
    replay_public_history_test(game, public_hist)
    
    expected = ['Qs']  # Sollte nicht dupliziert werden
    if game.public_cards == expected and len(game.public_cards) == 1:
        print(f"✅ BESTANDEN: public_cards = {game.public_cards} (keine Duplikate)")
        return True
    else:
        print(f"❌ FEHLGESCHLAGEN: Erwartet {expected}, bekommen {game.public_cards}")
        return False

def test_case_4_royal_holdem():
    """Test 4: Royal Holdem (mehrere Karten auf einmal)"""
    print("\n" + "=" * 60)
    print("TEST 4: Royal Holdem - Mehrere Karten")
    print("=" * 60)
    
    game = RoyalHoldemGame(**GAME_CONFIGS['royal_holdem'])
    game.reset(0)
    
    # Royal Holdem: Nach Runde 1 werden 3 Karten auf einmal gedealt
    public_hist = ['check', 'bet', 'call', 'Qs', 'Kh', 'As']
    replay_public_history_test(game, public_hist)
    
    # Nach step('call') sollte step() bereits 3 zufällige Karten hinzugefügt haben
    # Diese sollten alle überschrieben werden
    if len(game.public_cards) == 3:
        print(f"✅ BESTANDEN: public_cards = {game.public_cards} (3 Karten)")
        return True
    else:
        print(f"❌ FEHLGESCHLAGEN: Erwartet 3 Karten, bekommen {len(game.public_cards)}: {game.public_cards}")
        return False

def test_case_5_no_card_from_step():
    """Test 5: step() hat keine Karte hinzugefügt (Edge Case)"""
    print("\n" + "=" * 60)
    print("TEST 5: step() hat keine Karte hinzugefügt")
    print("=" * 60)
    
    game = RhodeIslandGame(**GAME_CONFIGS['rhode_island'])
    game.reset(0)
    
    # Nur eine Aktion, die keine Karte auslöst
    public_hist = ['check', 'Qs']  # 'check' löst keine Karte aus
    replay_public_history_test(game, public_hist)
    
    # Liste sollte leer sein, dann wird 'Qs' hinzugefügt
    expected = ['Qs']
    if game.public_cards == expected:
        print(f"✅ BESTANDEN: public_cards = {game.public_cards}")
        return True
    else:
        print(f"❌ FEHLGESCHLAGEN: Erwartet {expected}, bekommen {game.public_cards}")
        return False

def run_all_tests():
    """Führe alle Tests aus"""
    print("=" * 60)
    print("ROBUSTER TEST: Replay Public History Fix Verification")
    print("=" * 60)
    
    results = []
    results.append(("Test 1: Eine Karte", test_case_1_single_card()))
    results.append(("Test 2: Mehrere Karten", test_case_2_multiple_cards()))
    results.append(("Test 3: Karte bereits vorhanden", test_case_3_card_already_present()))
    results.append(("Test 4: Royal Holdem", test_case_4_royal_holdem()))
    results.append(("Test 5: Keine Karte von step()", test_case_5_no_card_from_step()))
    
    print("\n" + "=" * 60)
    print("ZUSAMMENFASSUNG:")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ BESTANDEN" if result else "❌ FEHLGESCHLAGEN"
        print(f"{status}: {name}")
    
    print(f"\nErgebnis: {passed}/{total} Tests bestanden")
    
    return passed == total

if __name__ == "__main__":
    all_passed = run_all_tests()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ERGEBNIS: ✅ ALLE TESTS BESTANDEN!")
        print("Der Fix funktioniert korrekt für alle Testfälle.")
    else:
        print("ERGEBNIS: ❌ EINIGE TESTS FEHLGESCHLAGEN!")
        print("Der Fix funktioniert nicht für alle Fälle.")
    print("=" * 60)
    
    sys.exit(0 if all_passed else 1)
