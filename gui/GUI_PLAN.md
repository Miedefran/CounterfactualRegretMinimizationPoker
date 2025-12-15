# GUI/Visualizer Projekt - Architektur & Implementierungsplan

## 1. Projekt-Übersicht

Entwicklung eines GUI/Visualizers für CFR-Poker mit drei Hauptmodi:
- **Agent vs Agent**: Visualisierung von zwei Agenten, die gegeneinander spielen (Replayer)
- **Agent vs Human**: Lokales Gameplay gegen einen trainierten Agenten
- **Human vs Human**: Online-Multiplayer über HTTP REST + Polling (✅ IMPLEMENTIERT)

## 2. Architektur-Übersicht

### Trennung: Layout vs. Funktionalität

**Wichtig:** Layouts enthalten NUR UI-Anordnung, keine Game-Logik!

```
gui/
├── layouts/                    # NUR Layouts (UI-Anordnung)
│   ├── base_poker_layout.py    # Basis-Layout mit allen Komponenten
│   ├── agent_vs_human_layout.py # Layout mit Action Buttons
│   └── agent_vs_agent_layout.py # Layout mit Playback Controls
│
├── agent_vs_human.py           # Agent vs Human Funktionalität (nutzt Layout)
├── agent_vs_agent.py           # Agent vs Agent Funktionalität (nutzt Layout)
├── run_agent_vs_human.py       # Start-Script für Agent vs Human
│
├── components/                  # Wiederverwendbare UI-Komponenten
│   ├── poker_table.py          # Grüner Poker-Tisch
│   ├── visual_card.py          # Visuelle Karten
│   ├── hidden_card.py          # Verdeckte Karte (blau)
│   ├── chip.py                 # Poker-Chips
│   ├── pot_display.py          # Pot-Anzeige
│   ├── player_widget.py        # Spieler-Widget (Karten, Name, Status)
│   ├── action_buttons.py       # Action-Buttons (Check, Bet, Call, Fold)
│   ├── playback_controls.py    # Playback Controls (Play/Pause, Step, Speed)
│   └── history_view.py         # History/Log-Anzeige
│
└── audio/                      # Audio-System
    ├── sound_manager.py        # SoundManager für Audio-Wiedergabe
    └── sounds/                 # Audio-Dateien
        ├── Call.wav
        ├── Fold.wav
        ├── Raise.wav
        └── untitled.wav (Check)
```

## 3. Aktuelle Implementierung

### 3.1 Layouts (NUR UI-Anordnung)

#### `base_poker_layout.py`
- Basis-Layout mit allen Komponenten
- `player_top_widget` (oben) - für Agent
- `player_bottom_widget` (unten) - für Human
- `poker_table` - Grüner Tisch
- `community_cards_widget` - Board-Karten
- `pot_display` + `pot_chips` - Pot-Anzeige
- `history_view` - Collapsible History rechts
- `control_area` - Platzhalter für Controls (wird ersetzt)
- `restart_button` - Restart-Button oben rechts

#### `agent_vs_human_layout.py`
- Erbt von `BasePokerLayout`
- Ersetzt `control_area` durch `ActionButtons`

#### `agent_vs_agent_layout.py`
- Erbt von `BasePokerLayout`
- Ersetzt `control_area` durch `PlaybackControls`

### 3.2 Komponenten

Alle Komponenten sind fertig implementiert:
- ✅ `PokerTable` - Grüner ovaler Tisch
- ✅ `VisualCard` - Visuelle Karten mit Rank/Suit
- ✅ `HiddenCard` - Verdeckte Karte (jetzt `VisualCard(None)`)
- ✅ `Chip` - Poker-Chips mit Design
- ✅ `PotDisplay` - Pot-Anzeige
- ✅ `PlayerWidget` - Spieler-Info (Name, Karten, Bet, Status)
- ✅ `ActionButtons` - Check, Bet, Call, Fold Buttons
- ✅ `PlaybackControls` - Play/Pause, Step Forward/Backward, Speed Slider
- ✅ `HistoryView` - Scrollable History-Log

### 3.3 Design-Entscheidungen

**Player-Positionierung:**
- `player_top_widget` = Immer oben (Agent)
- `player_bottom_widget` = Immer unten (Human)
- Unabhängig von Game-Engine Player-ID (0/1)

**State-Mapping:**
- Helper-Methoden für verschiedene Game-Typen:
  - `get_private_cards()` - Unterstützt single card (Kuhn) und list (Hold'em)
  - `get_public_cards()` - Unterstützt `public_card` (Leduc) und `public_cards` (Hold'em)
  - `get_player_bets()` - Unterstützt `player_bets` (Kuhn) und `total_bets` (Hold'em)

## 4. Agent vs Human Modus - ✅ IMPLEMENTIERT

### 4.1 Implementierte Datei: `gui/agent_vs_human.py`

**Struktur:**
```python
class AgentVsHumanGUI(AgentVsHumanLayout):
    # Erbt vom Layout (nur UI)
    # Fügt Game-Logik hinzu
```

### 4.2 Implementierung - ✅ FERTIG

#### Initialisierung - ✅ IMPLEMENTIERT
- Game-Objekt wird übergeben
- Strategy-File optional (wird automatisch geladen wenn vorhanden)
- SoundManager wird initialisiert
- Human/Agent Player-ID Mapping funktioniert
- Action Buttons werden verbunden
- Restart-Button wird eingerichtet

#### Game-Management - ✅ IMPLEMENTIERT

**`reset_game(starting_player=0)`:**
- ✅ Game zurücksetzen
- ✅ Karten austeilen (unterstützt single card und list)
- ✅ `human_player_id` bestimmen (kann 0 oder 1 sein)
- ✅ `agent_player_id = 1 - human_player_id`
- ✅ Strategy-Agent mit korrekter Player-ID erstellen
- ✅ UI updaten

**`update_display()`:**
- ✅ State von Game holen
- ✅ Karten anzeigen (mit Privacy)
- ✅ Pot, Bets, Legal Actions updaten
- ✅ History updaten

#### Action-Handling - ✅ IMPLEMENTIERT

**`handle_action(action, bet_size)`:**
- ✅ Nur wenn Human dran ist (`game.current_player == human_player_id`)
- ✅ Action validieren (`legal_actions`)
- ✅ Sound abspielen für Human-Action
- ✅ `game.step(action)` ausführen
- ✅ UI updaten
- ✅ Wenn Agent jetzt dran ist → automatisch `agent_step()` nach 2-4 Sekunden aufrufen

**`agent_step()`:**
- ✅ Agent ist dran (`game.current_player == agent_player_id`)
- ✅ State holen
- ✅ Action von Strategy-Agent holen
- ✅ Sound abspielen für Agent-Action
- ✅ `game.step(action)` ausführen
- ✅ UI updaten
- ✅ Wenn Human jetzt dran ist → Buttons aktivieren

#### Privacy-Management - ✅ IMPLEMENTIERT

**Karten-Anzeige:**
- ✅ Human-Karten: `player_bottom_widget.set_cards(..., reveal=True)`
- ✅ Agent-Karten: `player_top_widget.set_cards(..., reveal=False)`
- ✅ Am Ende: beide `reveal=True` wenn `game.done`
- ✅ Helper-Methoden: `get_private_cards()`, `get_public_cards()`, `get_player_bets()`
- ✅ Unterstützt alle Game-Typen (Kuhn, Leduc, Rhode Island, Royal/Limit Hold'em)

**Mapping:**
- ✅ Funktioniert korrekt mit `human_player_id` und `agent_player_id`
- ✅ Karten bleiben verdeckt bis Spielende

#### State-Updates - ✅ IMPLEMENTIERT

**`update_cards(state)`:**
- ✅ Private Cards holen (mit Mapping für Human/Agent)
- ✅ Public Cards holen
- ✅ Anzeigen mit Privacy

**`update_actions(state)`:**
- ✅ Legal Actions holen
- ✅ Buttons enable/disable
- ✅ Nur wenn Human dran ist
- ✅ Call Amount anzeigen wenn vorhanden

**`update_players(state)`:**
- ✅ Current Player bestimmen
- ✅ Highlighting für aktuellen Spieler
- ✅ Bets anzeigen

**`update_history(state)`:**
- ✅ History aus Game holen
- ✅ In HistoryView anzeigen
- ✅ Round-Separatoren hinzufügen
- ✅ Game Result anzeigen

### 4.3 Ablauf eines Spiels - ✅ IMPLEMENTIERT

```
1. reset_game(starting_player)
   ✅ Game zurücksetzen
   ✅ Karten austeilen
   ✅ human_player_id bestimmen
   ✅ Agent erstellen
   ✅ update_display()

2. update_display()
   ✅ Karten anzeigen (mit Privacy)
   ✅ Pot, Bets, Legal Actions updaten
   ✅ History updaten

3. Wenn game.current_player == human_player_id:
   ✅ Action Buttons aktivieren
   ✅ Warten auf Button-Click

4. handle_action(action)
   ✅ Sound abspielen
   ✅ game.step(action)
   ✅ update_display()
   ✅ Wenn Agent jetzt dran: agent_step() nach 2-4 Sekunden

5. agent_step()
   ✅ Action von Agent holen
   ✅ Sound abspielen
   ✅ game.step(action)
   ✅ update_display()
   ✅ Wenn Human jetzt dran: Buttons aktivieren

6. Wiederholen bis game.done

7. Am Ende:
   ✅ Beide Karten zeigen (reveal=True)
   ✅ Game Result anzeigen
```

### 4.4 Zusätzliche Features - ✅ IMPLEMENTIERT

- ✅ **SoundManager**: Audio-Feedback für alle Aktionen
  - Check, Bet, Call, Fold Sounds
  - Automatisches Abspielen bei Human- und Agent-Aktionen
  
- ✅ **Restart-Button**: Neue Hand starten
  - Oben rechts im Fenster
  - Startet neue Hand mit zufälligem starting_player
  
- ✅ **Zufällige Agent-Wartezeit**: 2-4 Sekunden
  - Macht Agent-Verhalten natürlicher
  
- ✅ **Call Amount Anzeige**: 
  - `ActionButtons.set_amount_to_call(amount)` wird verwendet
  - Button-Text: "Call (50)" wenn Bet vorhanden

### 4.5 Offene Fragen

- **Strategy-Tipps anzeigen?** (Bet: 60%, Call: 30%, etc.)
  - Optional: Widget für Strategy-Tipps hinzufügen
  - Zeigt Wahrscheinlichkeiten aus Strategy für aktuelle Info-Set

## 5. Human vs Human Modus - ✅ IMPLEMENTIERT

### 5.1 Architektur-Übersicht

**HTTP REST + Polling Multiplayer-Architektur:**
```
┌─────────────────────────────────────────────────────────┐
│              HTTP SERVER (Game Authority)               │
│  - Läuft auf PC 1 (oder separatem Server-PC)          │
│  - Flask/FastAPI Server auf 0.0.0.0:8888               │
│  - Verwaltet Game-Objekt (Source of Truth)              │
│  - REST API: GET /state, POST /action                   │
└─────────────────────────────────────────────────────────┘
                          │
                          │ HTTP (http://)
                          │
        ┌─────────────────┴─────────────────┐
        │                                     │
┌───────▼────────┐                  ┌────────▼───────┐
│  CLIENT 1      │                  │  CLIENT 2      │
│  (PC 1)        │                  │  (PC 2)        │
│  http://IP:8888│                  │  http://IP:8888│
│                │                  │                │
│  - GUI (PyQt)  │                  │  - GUI (PyQt)  │
│  - HTTP Client │                  │  - HTTP Client │
│  - Polling     │                  │  - Polling     │
│  (alle 100ms)  │                  │  (alle 100ms) │
└────────────────┘                  └────────────────┘
```

### 5.2 Technologie-Stack

**HTTP Server:**
- `Flask` (einfach, leichtgewichtig)
- Alternative: `FastAPI` (schneller, aber mehr Features)

**HTTP Client:**
- `requests` (Standard-Library-ähnlich, sehr einfach)

**Abhängigkeiten:**
```python
flask>=2.0.0
requests>=2.28.0
```

**Netzwerk-Konfiguration:**
- Server: bindet auf `0.0.0.0` (alle Interfaces) oder spezifische IP
- Port: Standard 8888, konfigurierbar
- Client: Verbindet zu Server-IP via HTTP (z.B. `http://192.168.1.100:8888`)
- Polling: Client fragt alle 100ms nach Updates (GET /state)

### 5.3 Datei-Struktur

```
gui/
├── server/
│   ├── __init__.py
│   ├── http_server.py            # Flask HTTP Server
│   └── http_client.py            # HTTP Client (requests)
│
├── human_vs_human.py             # GUI für Human vs Human
├── run_server.py                 # Server starten (mit IP/Port-Args)
└── run_client.py                 # Client starten (mit Server-IP-Arg)
```

### 5.4 Server-Design (`http_server.py`)

**Flask-basierter HTTP Server:**
- Einfache REST API mit GET/POST Endpoints
- Threading für Game-Logik (Flask ist thread-safe)
- Keine persistenten Verbindungen nötig

**Kern-Methoden:**
```python
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/state', methods=['GET'])
def get_state():
    """Gibt aktuellen State für Player zurück"""
    player_id = int(request.args.get('player_id', 0))
    state = server.get_state_update(player_id)
    return jsonify(state)

@app.route('/action', methods=['POST'])
def handle_action():
    """Empfängt Action von Client"""
    data = request.json
    player_id = data['player_id']
    action = data['action']
    bet_size = data.get('bet_size', 0)
    server.handle_action(player_id, action, bet_size)
    return jsonify({'status': 'ok'})

@app.route('/reset', methods=['POST'])
def reset_game():
    """Startet neue Hand"""
    starting_player = request.json.get('starting_player', 0)
    server.reset_game(starting_player)
    return jsonify({'status': 'ok'})
```

**Server-Klasse:**
```python
class PokerHTTPServer:
    def __init__(self, game, host='0.0.0.0', port=8888):
        self.game = game
        self.host = host
        self.port = port
        self.clients = {}  # Track connected clients
    
    def get_state_update(self, player_id):
        """Gibt State mit privaten Karten für Player zurück"""
        state = self.game.get_state(self.game.current_player)
        state['private_cards'] = self._get_private_cards(player_id)
        return state
    
    def handle_action(self, player_id, action, bet_size):
        """Validiert & führt Action aus"""
        if self.game.current_player != player_id:
            return
        if self.game.done:
            return
        self.game.step(action)
    
    def reset_game(self, starting_player=0):
        """Startet neue Hand"""
        self.game.reset(starting_player)
        # Karten austeilen...
```

**REST API Endpoints:**

**GET /state?player_id=0**
```json
{
    "current_player": 0,
    "done": false,
    "pot": 2,
    "player_bets": [1, 1],
    "history": [],
    "public_cards": [],
    "legal_actions": ["check", "bet"],
    "private_cards": ["Q"]  // Nur für diesen Player!
}
```

**POST /action**
```json
// Request Body
{
    "player_id": 0,
    "action": "bet",
    "bet_size": 0
}

// Response
{
    "status": "ok"
}
```

**POST /reset**
```json
// Request Body
{
    "starting_player": 0
}

// Response
{
    "status": "ok"
}
```

**GET /player_id**
```json
// Response (wird beim ersten Connect gesendet)
{
    "player_id": 0
}
```

### 5.5 Client-Design (`http_client.py`)

**HTTP Client mit Polling:**
- `requests` Library für HTTP-Requests
- QTimer für Polling (alle 100ms)
- Einfache GET/POST Requests

**Kern-Struktur:**
```python
import requests
from PyQt6.QtCore import QTimer, QObject, pyqtSignal

class HTTPClient(QObject):
    """HTTP Client mit Polling"""
    
    state_update_received = pyqtSignal(dict)  # Qt Signal
    connection_error = pyqtSignal(str)
    
    def __init__(self, server_url, player_id=None, parent=None):
        super().__init__(parent)
        self.server_url = server_url  # z.B. "http://192.168.1.100:8888"
        self.player_id = player_id
        self.poll_timer = QTimer()
        self.poll_timer.timeout.connect(self._poll_state)
        self.poll_timer.setInterval(100)  # 100ms Polling
    
    def connect(self):
        """Verbindet zum Server und startet Polling"""
        if self.player_id is None:
            # Hole player_id vom Server
            try:
                response = requests.get(f"{self.server_url}/player_id", timeout=2)
                self.player_id = response.json()['player_id']
            except Exception as e:
                self.connection_error.emit(str(e))
                return False
        
        self.poll_timer.start()
        return True
    
    def _poll_state(self):
        """Pollt State vom Server (wird alle 100ms aufgerufen)"""
        try:
            response = requests.get(
                f"{self.server_url}/state",
                params={'player_id': self.player_id},
                timeout=1
            )
            state = response.json()
            self.state_update_received.emit(state)
        except Exception as e:
            self.connection_error.emit(str(e))
            self.poll_timer.stop()
    
    def send_action(self, action, bet_size=0):
        """Sendet Action an Server"""
        try:
            requests.post(
                f"{self.server_url}/action",
                json={
                    'player_id': self.player_id,
                    'action': action,
                    'bet_size': bet_size
                },
                timeout=2
            )
            return True
        except Exception as e:
            self.connection_error.emit(str(e))
            return False
    
    def send_reset_request(self, starting_player=0):
        """Sendet Reset-Request"""
        try:
            requests.post(
                f"{self.server_url}/reset",
                json={'starting_player': starting_player},
                timeout=2
            )
            return True
        except Exception as e:
            self.connection_error.emit(str(e))
            return False
```

### 5.6 GUI-Design (`human_vs_human.py`)

**Verantwortlichkeiten:**
- Nutzt `AgentVsHumanLayout` (gleiches Layout wie Agent vs Human)
- Verbindet HTTP-Client mit GUI
- UI-Updates basierend auf Server-State (via Polling)
- Action-Buttons senden Actions an Server

**Kern-Methoden:**
```python
class HumanVsHumanGUI(AgentVsHumanLayout):
    def __init__(self, server_url, human_name="Player", parent=None):
        super().__init__(parent)
        self.client = HTTPClient(server_url, parent=self)
        
        # Verbinde Client-Signals
        self.client.state_update_received.connect(self._on_state_update)
        self.client.connection_error.connect(self._on_connection_error)
        
        # Verbinde Action-Buttons
        if hasattr(self, 'action_buttons'):
            self.action_buttons.action_selected.connect(self.handle_action)
        
        # Starte Verbindung (startet Polling)
        if not self.client.connect():
            print("Fehler: Konnte nicht zum Server verbinden!")
    
    def _on_state_update(self, state):
        """Callback: State-Update vom Server (via Polling)"""
        self._update_from_server_state(state)
    
    def handle_action(self, action, bet_size):
        """Sendet Action an Server"""
        self.client.send_action(action, bet_size)
    
    def restart_hand(self):
        """Sendet Reset-Request an Server"""
        import random
        starting_player = random.randint(0, 1)
        self.client.send_reset_request(starting_player)
```

**Wichtige Unterschiede zu AgentVsHumanGUI:**
- ❌ Kein lokales Game-Objekt
- ❌ Kein Agent
- ✅ State kommt vom Server (via Polling alle 100ms)
- ✅ Actions werden an Server gesendet (HTTP POST)
- ✅ Polling läuft automatisch im Hintergrund

### 5.7 Datenfluss

**Spiel-Start:**
1. Server startet Flask-Server auf Port 8888
2. Client 1 verbindet → GET `/player_id` → erhält `player_id: 0`
3. Client 2 verbindet → GET `/player_id` → erhält `player_id: 1`
4. Beide Clients starten Polling (GET `/state` alle 100ms)
5. Server startet Spiel (POST `/reset`) → Clients sehen State im nächsten Poll
6. Clients zeigen Karten (nur eigene aus `private_cards`)

**Action-Flow:**
1. Player 0 klickt "Bet" → Client sendet POST `/action` an Server
2. Server validiert → führt `game.step(action)` aus
3. Beide Clients pollen weiter (alle 100ms) → sehen neuen State
4. Client 0: sieht eigenen State mit `private_cards`
5. Client 1: sieht eigenen State (ohne Karten von Player 0)
6. UI wird aktualisiert

**Spiel-Ende:**
1. Server erkennt `game.done == True`
2. State enthält `opponent_cards` für Reveal
3. Clients sehen State im nächsten Poll → zeigen alle Karten an

### 5.8 Run-Scripts

**`run_server.py`:**
```python
import argparse
from envs.kuhn_poker.game import KuhnPokerGame
from envs.leduc_holdem.game import LeducHoldemGame
from gui.server.http_server import PokerHTTPServer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0', 
                       help='Server IP (0.0.0.0 = alle Interfaces)')
    parser.add_argument('--port', type=int, default=8888)
    parser.add_argument('--game', default='kuhn', choices=['kuhn', 'leduc'])
    
    args = parser.parse_args()
    
    # Erstelle Game
    if args.game == 'kuhn':
        game = KuhnPokerGame(ante=1, bet_size=1)
    elif args.game == 'leduc':
        game = LeducHoldemGame(ante=1, bet_sizes=[2, 4], bet_limit=2)
    
    # Starte Server
    server = PokerHTTPServer(game, host=args.host, port=args.port)
    server.start()  # Flask app.run()
```

**`run_client.py`:**
```python
import argparse
import sys
from PyQt6.QtWidgets import QApplication
from gui.human_vs_human import HumanVsHumanGUI

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', required=True,
                       help='Server URL (z.B. http://192.168.1.100:8888)')
    parser.add_argument('--name', default='Player')
    
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    window = HumanVsHumanGUI(args.server, human_name=args.name)
    window.showMaximized()
    sys.exit(app.exec())
```

### 5.9 Netzwerk-Setup

**PC 1 (Server):**
```bash
# Server starten
python gui/run_server.py --host 0.0.0.0 --port 8888 --game kuhn

# Server läuft auf http://0.0.0.0:8888
# Erreichbar über: http://192.168.1.100:8888 (lokale IP)
# Oder: http://localhost:8888 (lokal)
```

**PC 1 (Client 1):**
```bash
# Client auf gleichem PC
python gui/run_client.py --server http://localhost:8888 --name "Player1"
```

**PC 2 (Client 2):**
```bash
# Client auf anderem PC (mit Server-IP)
python gui/run_client.py --server http://192.168.1.100:8888 --name "Player2"
```

**Über verschiedene Netzwerke (mit ngrok):**
```bash
# PC 1: Server starten
python gui/run_server.py --host 0.0.0.0 --port 8888

# PC 1: ngrok Tunnel (in neuem Terminal)
ngrok http 8888
# Gibt: https://abc123.ngrok.io

# PC 2 (irgendwo): Client verbindet zu ngrok-URL
python gui/run_client.py --server https://abc123.ngrok.io --name "Player2"
```

### 5.10 Implementierungs-Reihenfolge

**Phase 1: HTTP-Server-Grundgerüst**
1. ✅ `http_server.py` - Flask-Server mit REST API
2. ✅ GET `/state` Endpoint implementieren
3. ✅ POST `/action` Endpoint implementieren
4. ✅ GET `/player_id` Endpoint implementieren
5. ✅ POST `/reset` Endpoint implementieren
6. ✅ Basis-Kommunikation testen (mit Browser/curl)

**Phase 2: Game-Integration im Server**
1. ✅ Game-Objekt in Server integrieren
2. ✅ `reset_game()` implementieren
3. ✅ `get_state_update()` mit privaten Karten
4. ✅ State-Format für verschiedene Game-Typen

**Phase 3: Action-Handling im Server**
1. ✅ `handle_action()` implementieren
2. ✅ Actions validieren (current_player, legal_actions)
3. ✅ Game-State aktualisieren
4. ✅ Thread-Safety prüfen (Flask ist thread-safe)

**Phase 4: HTTP-Client**
1. ✅ `http_client.py` - HTTP Client mit `requests`
2. ✅ QTimer für Polling (alle 100ms)
3. ✅ Qt-Signals für State-Updates
4. ✅ Action-Sending (POST)
5. ✅ Error-Handling (Connection-Errors)

**Phase 5: GUI-Integration**
1. ✅ `human_vs_human.py` - GUI erstellen
2. ✅ Client-Signals verbinden
3. ✅ UI-Updates basierend auf Server-State
4. ✅ Action-Buttons mit Server verbinden
5. ✅ Restart-Button implementieren

**Phase 6: Run-Scripts**
1. ✅ `run_server.py` - Server starten
2. ✅ `run_client.py` - Client starten
3. ✅ Unterstützung für alle Game-Varianten (Kuhn, Leduc, Twelve Card, Rhode Island, Royal Hold'em, Limit Hold'em)

**Phase 7: Testing**
1. ⏳ Lokale Tests (Server + 2 Clients auf gleichem PC)
2. ⏳ Netzwerk-Tests (2 verschiedene PCs im gleichen WLAN)
3. ⏳ ngrok-Tests (verschiedene Netzwerke)
4. ⏳ Edge Cases (Server-Down, Disconnect, Reconnect)

### 5.11 Vorteile von HTTP REST + Polling

- ✅ **Sehr einfach**: Nur `requests` Library, kein Socket-Management
- ✅ **Nutzt bestehende GUI**: PyQt-GUI kann direkt genutzt werden
- ✅ **HTTP-kompatibel**: Funktioniert über Firewalls, Port 80/443 möglich
- ✅ **Einfaches Debugging**: Kann mit Browser/curl getestet werden
- ✅ **ngrok-Support**: HTTP funktioniert perfekt mit ngrok (verschiedene Netzwerke)
- ✅ **Keine Threading-Komplexität**: QTimer für Polling ist einfach
- ⚠️ **Latenz**: ~100ms durch Polling (für Poker akzeptabel)
- ⚠️ **Server-Last**: Mehr Requests als WebSockets (bei 2 Clients egal)

### 5.12 Offene Fragen

1. **Port-Konfiguration**: Standard 8888 oder konfigurierbar? → ✅ Standard 8888, konfigurierbar (implementiert)
2. **Reconnect**: Automatisch bei Verbindungsabbruch? → ⏳ Später implementieren
3. **Firewall**: Hinweise für Port-Öffnung? → ⏳ Dokumentation hinzufügen
4. **Verschlüsselung**: HTTPS/WSS Support? → ⏳ Später (aktuell HTTP)

## 6. Nächste Schritte: Agent vs Agent Modus

### 5.1 Neue Datei: `gui/agent_vs_agent.py`

**Struktur:**
```python
class AgentVsAgentGUI(AgentVsAgentLayout):
    # Erbt vom Layout (nur UI)
    # Fügt Playback-Funktionalität hinzu
```

### 5.2 Implementierung

- Strategy-Agents für beide Spieler laden
- Auto-Play mit QTimer
- Step Forward/Backward
- Speed Control
- Beide Spieler-Karten immer sichtbar (`reveal=True`)

## 7. Implementierungs-Reihenfolge

### Phase 1: Agent vs Human - ✅ FERTIG
1. ✅ Layouts erstellt
2. ✅ `agent_vs_human.py` implementiert
3. ✅ Game-Integration funktioniert
4. ✅ Strategy-Loading funktioniert
5. ✅ SoundManager integriert
6. ✅ Restart-Button hinzugefügt
7. ✅ Privacy-Management funktioniert

### Phase 2: Agent vs Agent
1. ✅ Layout erstellt
2. ⏳ `agent_vs_agent.py` implementieren
3. ⏳ Playback-Funktionalität testen

### Phase 3: Human vs Human - ✅ IMPLEMENTIERT
1. ✅ HTTP-Server (`http_server.py`)
2. ✅ HTTP-Client (`http_client.py`)
3. ✅ GUI-Integration (`human_vs_human.py`)
4. ✅ Run-Scripts (`run_server.py`, `run_client.py`)
5. ✅ Unterstützung für alle Game-Varianten
6. ⏳ Netzwerk-Testing (2 verschiedene PCs) - Phase 7

## 8. Notizen

- **Code-Konventionen**: Keine Kommentare (außer Paper-Referenzen)
- **Vererbung**: Layouts → GUI-Klassen (Layout + Funktionalität)
- **Modularität**: Components wiederverwendbar
- **Erweiterbarkeit**: Einfach neue Game-Varianten hinzufügen

## 9. Neue Komponenten & Erweiterungen

### 8.1 StrategyAgent Erweiterung - ✅ IMPLEMENTIERT
- **Datei**: `agents/strategy_agent.py`
- **Änderung**: Unterstützt jetzt alle Game-Typen
- **Neue Parameter**: `game` Parameter hinzugefügt (optional, rückwärtskompatibel)
- **Funktion**: Nutzt `game.get_info_set_key(player_id)` für korrekte Info-Set-Keys
- **Unterstützt**: Kuhn, Leduc, Rhode Island, Royal Hold'em, Limit Hold'em

### 8.2 SoundManager - ✅ IMPLEMENTIERT
- **Datei**: `gui/audio/sound_manager.py`
- **Funktionen**: 
  - `play_action(action)`: Spielt Sound für Aktion ab
  - `set_volume()`, `set_enabled()`, `toggle()`
- **Audio-Dateien**: `gui/audio/sounds/`
  - `Call.wav`, `Fold.wav`, `Raise.wav`, `untitled.wav` (Check)

### 8.2 Restart-Button - ✅ IMPLEMENTIERT
- **Position**: Oben rechts im Fenster
- **Funktion**: Startet neue Hand mit `restart_hand()`
- **Integration**: In `BasePokerLayout` und `AgentVsHumanGUI`

## 10. Bekannte Probleme / To-Do

- ✅ Restart-Button Positionierung (implementiert)
- ⏳ Strategy-Tipps Widget (optional)
- ⏳ Agent vs Agent Modus implementieren (`agent_vs_agent.py` fehlt noch)
