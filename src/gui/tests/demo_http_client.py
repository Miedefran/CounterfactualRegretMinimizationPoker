import sys
import time
import threading
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from envs.kuhn_poker.game import KuhnPokerGame
from gui.server.http_server import PokerHTTPServer
from gui.server.http_client import HTTPClient


def test_client():
    print("=" * 50)
    print("HTTP CLIENT TEST")
    print("=" * 50)

    game = KuhnPokerGame()
    server = PokerHTTPServer(game, host='localhost', port=8890)

    server_thread = threading.Thread(target=server.start, daemon=True)
    server_thread.start()

    time.sleep(1)

    app = QApplication(sys.argv)

    client = HTTPClient("http://localhost:8890")

    state_received = []
    errors_received = []

    def on_state_update(state):
        state_received.append(state)
        print(f"\n[State Update] Player {client.player_id}")
        print(f"  Current Player: {state.get('current_player')}")
        print(f"  Private Cards: {state.get('private_cards')}")
        print(f"  Legal Actions: {state.get('legal_actions')}")
        print(f"  Pot: {state.get('pot')}")

    def on_error(error):
        errors_received.append(error)
        print(f"\n[Error] {error}")

    client.state_update_received.connect(on_state_update)
    client.connection_error.connect(on_error)

    print("\n1. Test: Verbindung zum Server")
    if not client.connect():
        print("   ❌ Verbindung fehlgeschlagen")
        return False
    print(f"   ✅ Verbunden, Player ID: {client.player_id}")

    print("\n2. Test: Warte auf State-Updates (Polling)")
    print("   (Warte 1 Sekunde für Polling...)")

    QTimer.singleShot(1100, app.quit)
    app.exec()

    if len(state_received) > 0:
        print(f"   ✅ {len(state_received)} State-Update(s) empfangen")
    else:
        print("   ❌ Keine State-Updates empfangen")
        return False

    print("\n3. Test: Reset-Request senden")
    if client.send_reset_request(starting_player=0):
        print("   ✅ Reset-Request gesendet")
    else:
        print("   ❌ Reset-Request fehlgeschlagen")
        return False

    print("\n4. Test: Warte auf State nach Reset (Polling)")
    state_received.clear()
    QTimer.singleShot(1100, app.quit)
    app.exec()

    if len(state_received) > 0:
        state = state_received[-1]
        if len(state.get('private_cards', [])) > 0:
            print(f"   ✅ State nach Reset empfangen mit Karten: {state.get('private_cards')}")
        else:
            print("   ⚠️ State empfangen, aber keine Karten")
    else:
        print("   ❌ Kein State nach Reset empfangen")
        return False

    print("\n5. Test: Action senden")
    if client.send_action('bet', 0):
        print("   ✅ Action gesendet")
    else:
        print("   ❌ Action senden fehlgeschlagen")
        return False

    print("\n6. Test: Warte auf State nach Action (Polling)")
    state_received.clear()
    QTimer.singleShot(1100, app.quit)
    app.exec()

    if len(state_received) > 0:
        state = state_received[-1]
        print(f"   ✅ State nach Action empfangen")
        print(f"   Current Player: {state.get('current_player')}")
        print(f"   History: {state.get('history')}")
    else:
        print("   ❌ Kein State nach Action empfangen")
        return False

    print("\n" + "=" * 50)
    print("✅ ALLE TESTS BESTANDEN")
    print("=" * 50)
    return True


if __name__ == "__main__":
    success = test_client()
    sys.exit(0 if success else 1)
