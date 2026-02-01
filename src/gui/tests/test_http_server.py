import sys
import requests
import time
import threading
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from envs.kuhn_poker.game import KuhnPokerGame
from gui.server.http_server import PokerHTTPServer


def test_server():
    print("=" * 50)
    print("HTTP SERVER TEST")
    print("=" * 50)

    game = KuhnPokerGame()
    server = PokerHTTPServer(game, host='localhost', port=8889)

    server_thread = threading.Thread(target=server.start, daemon=True)
    server_thread.start()

    time.sleep(1)

    base_url = "http://localhost:8889"

    print("\n1. Test: GET /player_id")
    try:
        response = requests.get(f"{base_url}/player_id", timeout=2)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        player_id_1 = response.json()['player_id']
        assert player_id_1 == 0, f"Expected player_id 0, got {player_id_1}"
        print("   ✅ Player ID 1 erhalten")
    except Exception as e:
        print(f"   ❌ Fehler: {e}")
        return False

    print("\n2. Test: GET /player_id (zweiter Client)")
    try:
        response = requests.get(f"{base_url}/player_id", timeout=2)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        player_id_2 = response.json()['player_id']
        assert player_id_2 == 1, f"Expected player_id 1, got {player_id_2}"
        print("   ✅ Player ID 2 erhalten")
    except Exception as e:
        print(f"   ❌ Fehler: {e}")
        return False

    print("\n3. Test: POST /reset (startet neues Spiel)")
    try:
        response = requests.post(
            f"{base_url}/reset",
            json={'starting_player': 0},
            timeout=2
        )
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        assert response.status_code == 200
        print("   ✅ Reset erfolgreich")
    except Exception as e:
        print(f"   ❌ Fehler: {e}")
        return False

    print("\n4. Test: GET /state (nach Reset, sollte Karten haben)")
    try:
        response = requests.get(f"{base_url}/state?player_id=0", timeout=2)
        print(f"   Status: {response.status_code}")
        state = response.json()
        print(f"   Response Keys: {list(state.keys())}")
        print(f"   Private Cards: {state.get('private_cards')}")
        print(f"   Current Player: {state.get('current_player')}")
        print(f"   Legal Actions: {state.get('legal_actions')}")
        assert 'current_player' in state
        assert 'done' in state
        assert 'pot' in state
        assert 'private_cards' in state
        assert len(state['private_cards']) > 0, "Sollte Karten nach Reset haben"
        print("   ✅ State mit Karten erhalten")
    except Exception as e:
        print(f"   ❌ Fehler: {e}")
        return False

    print("\n5. Test: POST /action (gültige Action)")
    try:
        response = requests.post(
            f"{base_url}/action",
            json={'player_id': 0, 'action': 'bet', 'bet_size': 0},
            timeout=2
        )
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        assert response.status_code == 200
        print("   ✅ Action gesendet")
    except Exception as e:
        print(f"   ❌ Fehler: {e}")
        return False

    print("\n6. Test: GET /state (nach Action, sollte sich geändert haben)")
    try:
        response = requests.get(f"{base_url}/state?player_id=0", timeout=2)
        state = response.json()
        print(f"   Current Player: {state.get('current_player')} (sollte jetzt 1 sein)")
        print(f"   History: {state.get('history')}")
        assert state.get('current_player') == 1, "Current Player sollte nach Action wechseln"
        print("   ✅ State nach Action aktualisiert")
    except Exception as e:
        print(f"   ❌ Fehler: {e}")
        return False

    print("\n7. Test: POST /action (ungültige Action - falscher Player)")
    try:
        response = requests.post(
            f"{base_url}/action",
            json={'player_id': 0, 'action': 'bet', 'bet_size': 0},
            timeout=2
        )
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        assert response.status_code == 400, "Sollte 400 sein, da Player 0 nicht mehr dran ist"
        print("   ✅ Ungültige Action korrekt abgelehnt")
    except Exception as e:
        print(f"   ❌ Fehler: {e}")
        return False

    print("\n" + "=" * 50)
    print("✅ ALLE TESTS BESTANDEN")
    print("=" * 50)
    return True


if __name__ == "__main__":
    success = test_server()
    if success:
        print("\n✅ Alle Tests erfolgreich!")
    else:
        print("\n❌ Tests fehlgeschlagen!")
        sys.exit(1)
