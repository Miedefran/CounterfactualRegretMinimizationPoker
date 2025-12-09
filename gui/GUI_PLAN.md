# GUI/Visualizer Projekt - Implementierungsplan

## 1. Projekt-Übersicht

Entwicklung eines GUI/Visualizers für CFR-Poker mit drei Hauptmodi:
- **Agent vs Agent**: Visualisierung von zwei Agenten, die gegeneinander spielen
- **Agent vs Human**: Lokales Gameplay gegen einen trainierten Agenten
- **Human vs Human**: Online-Multiplayer über WebSockets

## 2. Ordnerstruktur

```
gui/
├── __init__.py
├── GUI_PLAN.md              # Dieser Plan
├── base_gui.py              # Basis-Klasse mit gemeinsamer Funktionalität
├── agent_vs_agent.py        # Agent vs Agent Visualisierung
├── agent_vs_human.py        # Agent vs Human (lokales Gameplay)
├── human_vs_human.py        # Human vs Human (WebSocket-Server/Client)
├── components/              # Wiederverwendbare UI-Komponenten
│   ├── __init__.py
│   ├── card_display.py      # Karten-Darstellung
│   ├── pot_display.py       # Pot, Bets, Chips-Anzeige
│   ├── action_buttons.py    # Action-Buttons (Check, Bet, Call, Fold)
│   └── history_view.py      # History/Log-Anzeige
└── server/                  # WebSocket-Server für Human vs Human
    ├── __init__.py
    ├── game_server.py       # Flask-Server mit Game-Logik
    └── websocket_handler.py # WebSocket-Event-Handler
```

## 3. Technologie-Stack

### PyQt6 (Haupt-Framework)
- **Warum**: Native Desktop-App, gut für alle drei Modi
- **Komponenten**:
  - `QMainWindow`: Hauptfenster
  - `QWidget`: Container für UI-Elemente
  - `QPushButton`: Buttons für Actions/Controls
  - `QLabel`: Text/Karten-Anzeige
  - `QTimer`: Auto-Play für Agent vs Agent
  - `QVBoxLayout`, `QHBoxLayout`: Layout-Management

### Flask + SocketIO (für Human vs Human)
- **Flask**: HTTP-Server
- **Flask-SocketIO**: WebSocket-Integration
- **Alternative**: PyQt6 QWebSocket (einheitliche Technologie)

### Abhängigkeiten
```python
PyQt6>=6.0.0
flask>=2.0.0
flask-socketio>=5.0.0
python-socketio>=5.0.0
```

## 4. Basis-GUI-Klasse (`base_gui.py`)

### Gemeinsame Funktionalität

**Game-Management:**
- Game-Instanz verwalten (Kuhn, Leduc, Rhode Island, etc.)
- State-Tracking (`state_stack` für Step Backward)
- Reset-Funktionalität

**Display-Funktionen:**
- State-Display (Karten, Pot, History, Legal Actions)
- Card-Rendering (visuelle Darstellung)
- Pot/Bets-Anzeige
- History-Log

**Navigation:**
- Step Forward (nutzt `game.step()`)
- Step Backward (nutzt `game.state_stack`)
- Reset Game

**Interface:**
```python
class BasePokerGUI(QMainWindow):
    def __init__(self, game, parent=None):
        # Game-Instanz
        # UI-Setup
        # State-Initialisierung
    
    def update_display(self, state):
        # Aktualisiert alle UI-Elemente basierend auf State
    
    def step_forward(self):
        # Führt nächsten Step aus
    
    def step_backward(self):
        # Geht einen Step zurück
    
    def reset_game(self):
        # Startet neues Spiel
    
    def render_cards(self, cards, position):
        # Rendert Karten visuell
    
    def update_pot(self, pot, bets):
        # Aktualisiert Pot/Bets-Anzeige
    
    def log_action(self, action, player):
        # Fügt Action zu History hinzu
```

## 5. Agent vs Agent (`agent_vs_agent.py`)

### Features

**Playback-Kontrolle:**
- Play/Pause-Button
- Step Forward/Backward
- Geschwindigkeitskontrolle (Delay zwischen Steps)
- Auto-Play mit Pause bei wichtigen Events

**Visualisierung:**
- Beide Spieler-Karten sichtbar
- Board/Public Cards
- Pot, Bets, History
- Strategie-Anzeige (optional: Wahrscheinlichkeiten pro Aktion)

**UI-Layout:**
```
┌─────────────────────────────────────────┐
│  Player 0: [Karten]  |  Player 1: [Karten] │
├─────────────────────────────────────────┤
│         [Board/Public Cards]            │
├─────────────────────────────────────────┤
│  Pot: XX  |  Bets: [XX, XX]            │
├─────────────────────────────────────────┤
│  [Play] [Pause] [Step] [Back] [Speed] │
├─────────────────────────────────────────┤
│  History:                               │
│  - Player 0: bet                        │
│  - Player 1: call                       │
└─────────────────────────────────────────┘
```

### Implementierung

**Agent-Integration:**
- Zwei `StrategyAgent` Instanzen
- Strategy-Files laden
- Auto-Action-Selection basierend auf State

**Timing:**
- `QTimer` für Auto-Play
- Konfigurierbare Delay (z.B. 500ms, 1000ms, 2000ms)

## 6. Agent vs Human (`agent_vs_human.py`)

### Features

**Action-Input:**
- Action-Buttons (Check, Bet, Call, Fold)
- Validierung durch Game-Logik
- Disabled-Buttons für illegale Actions

**Strategie-Tipps:**
- Wahrscheinlichkeiten aus trainierten Strategien
- Anzeige: "Bet: 60%, Call: 30%, Fold: 10%"
- Optional: Empfehlung basierend auf höchster Wahrscheinlichkeit

**Privacy:**
- Opponent-Karten verdeckt
- Nur eigene Karten sichtbar
- Reveal am Ende des Spiels

**UI-Layout:**
```
┌─────────────────────────────────────────┐
│  Opponent: [??] [??]                    │
├─────────────────────────────────────────┤
│         [Board/Public Cards]            │
├─────────────────────────────────────────┤
│  You: [Karte] [Karte]                   │
│  [Check] [Bet] [Call] [Fold]           │
├─────────────────────────────────────────┤
│  Pot: XX  |  Your Bet: XX              │
├─────────────────────────────────────────┤
│  Strategy Tips:                         │
│  Bet: 60% | Call: 30% | Fold: 10%     │
└─────────────────────────────────────────┘
```

### Implementierung

**Strategy-Loading:**
- Strategy-File-Pfad als Parameter
- `StrategyAgent` für Opponent
- Strategy-Lookup für Tipps

**Action-Handling:**
- Button-Clicks → Game-Actions
- Validierung via `game.get_legal_actions()`
- Update nach jeder Action

## 7. Human vs Human (`human_vs_human.py`)

### Architektur

**Server (Flask + SocketIO):**
- Game-Instanz auf Server
- Action-Validierung
- State-Broadcasting
- Room-Management (1v1 Games)

**Client (PyQt6 oder Web):**
- Action-Sending
- State-Updates empfangen
- UI-Updates basierend auf State

### Server-Funktionalität (`server/game_server.py`)

```python
from flask import Flask
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

games = {}  # room_id -> game_instance

@socketio.on('connect')
def handle_connect():
    # Room-Erstellung/Join

@socketio.on('action')
def handle_action(data):
    # Action-Validierung
    # Game-Step
    # State-Broadcast

@socketio.on('disconnect')
def handle_disconnect():
    # Cleanup
```

### Client-Funktionalität

**PyQt6 Client:**
- WebSocket-Client (QWebSocket oder python-socketio)
- UI identisch zu Agent vs Human
- Network-Events → UI-Updates

**Web Client (Optional):**
- HTML/JS Frontend
- SocketIO-Client
- Für Demo/Vortrag

## 8. Implementierungs-Reihenfolge

### Phase 1: Basis + Agent vs Agent (MVP)
**Ziel**: Funktionierender Agent vs Agent Visualizer

1. **Basis-GUI** (`base_gui.py`)
   - QMainWindow Setup
   - Game-Integration
   - Basic Display (Text-basiert)
   - Step Forward/Backward

2. **Agent vs Agent** (`agent_vs_agent.py`)
   - Play/Pause-Button
   - Auto-Play mit Timer
   - Strategy-Agent-Integration
   - Test mit Kuhn Poker

3. **Card-Rendering** (`components/card_display.py`)
   - Einfache Text-Darstellung (später erweitern)
   - Card-Format: "As", "Kh", etc.

4. **Pot-Display** (`components/pot_display.py`)
   - Pot, Bets, Player-Bets

**Zeitaufwand**: ~2-3 Tage

### Phase 2: Agent vs Human
**Ziel**: Spielbar gegen Agenten

1. **Action-Buttons** (`components/action_buttons.py`)
   - Check, Bet, Call, Fold
   - Dynamic Enable/Disable

2. **Strategy-Loading**
   - Strategy-File-Parsing
   - StrategyAgent-Integration

3. **Strategie-Tipps-Anzeige**
   - Probability-Display
   - Optional: Empfehlung

4. **Privacy-Features**
   - Opponent-Karten verdeckt
   - Reveal am Ende

**Zeitaufwand**: ~2 Tage

### Phase 3: Human vs Human
**Ziel**: Online-Multiplayer

1. **Flask-Server Setup** (`server/game_server.py`)
   - Basic Flask-App
   - SocketIO-Integration
   - Room-Management

2. **WebSocket-Handler** (`server/websocket_handler.py`)
   - Action-Events
   - State-Broadcasting
   - Connection-Management

3. **Client-Implementierung** (`human_vs_human.py`)
   - WebSocket-Client
   - UI (ähnlich Agent vs Human)
   - Synchronisation

4. **Testing**
   - Lokale Tests (2 Clients)
   - Netzwerk-Tests

**Zeitaufwand**: ~3-4 Tage

## 9. Offene Fragen & Entscheidungen

### Card-Rendering
- **MVP**: Text-basiert ("As", "Kh")
- **Später**: Visuelle Karten (Bilder/SVG)
- **Entscheidung**: Start mit Text, später erweitern

### Strategie-Tipps
- **Anzeige**: Immer oder optional?
- **Format**: "Bet: 60%, Call: 30%, Fold: 10%"
- **Empfehlung**: Optional mit Toggle-Button

### Game-Auswahl
- **UI**: Dropdown für Varianten (Kuhn, Leduc, etc.)
- **Strategie**: File-Picker für Strategy-Files
- **Config**: Game-Config-Auswahl

### WebSocket vs. PyQt QWebSocket
- **Empfehlung**: Flask-SocketIO für Flexibilität
- **Vorteil**: Web + Desktop Clients möglich

## 10. Testing-Strategie

### Unit-Tests
- Basis-GUI-Funktionen
- Component-Tests
- Game-Integration

### Integration-Tests
- Agent vs Agent: Vollständiges Spiel
- Agent vs Human: Action-Flow
- Human vs Human: Server-Client-Kommunikation

### Demo-Vorbereitung
- Test-Szenarien für Vortrag
- Pre-loaded Strategies
- Screenshots/Recordings

## 11. Nächste Schritte

1. ✅ Ordnerstruktur erstellt
2. ⏳ Phase 1 starten: `base_gui.py` implementieren
3. ⏳ Agent vs Agent MVP
4. ⏳ Testing mit Kuhn Poker
5. ⏳ Phase 2: Agent vs Human
6. ⏳ Phase 3: Human vs Human

## 12. Notizen

- **Code-Konventionen**: Keine Kommentare (außer Paper-Referenzen)
- **Vererbung**: Basis-GUI → Spezifische Modi
- **Modularität**: Components wiederverwendbar
- **Erweiterbarkeit**: Einfach neue Game-Varianten hinzufügen

