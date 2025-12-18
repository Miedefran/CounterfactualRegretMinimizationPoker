## GUI Architektur (Kurzüberblick)

### Ziel
- Ein gemeinsames GUI für mehrere Poker Environments (Kuhn, Leduc, Rhode Island, Royal, Limit)
- Drei Modi: Agent vs Human (lokal), Human vs Human (über Netzwerk), Agent vs Agent (Playback, Layout vorhanden)

### Grundprinzip: UI Layout vs Logik
- `gui/layouts/*`
  - Enthält nur die Anordnung der UI Elemente (keine Game Logik)
  - `BasePokerLayout`: Table, Player Widgets (oben/unten), Community Cards, Pot, History Toggle, Control Area
  - `AgentVsHumanLayout`: ersetzt Control Area durch Action Buttons, fügt Restart Button ein
  - `AgentVsAgentLayout`: ersetzt Control Area durch Playback Controls
- `gui/*.py` (Modi)
  - Implementiert das Verhalten (State lesen, Aktionen ausführen, UI updaten)

### Wiederverwendbare UI Komponenten
- `gui/components/*`
  - `PlayerWidget` (Top/Bottom): Name, Karten (reveal/hidden), Turn Highlight, optional Hand Text
  - `VisualCard`: Karte anzeigen oder verdeckt (bei `None`)
  - `ActionButtons`: check / bet / call / fold (Buttons nur aktiv wenn legal)
  - `HistoryView`: Aktionen + Round Separatoren, Winner Anzeige
  - `PokerTable`, `PotDisplay`, `Chip`: Visualisierung von Tisch und Pot

### Modus 1: Agent vs Human (lokal)
- Datei: `gui/agent_vs_human.py`
- Ablauf
  - GUI hält ein lokales `game` Objekt
  - Human klickt → GUI validiert `legal_actions` → `game.step(action)` → UI Update
  - Agent Zug: `StrategyAgent` wählt Aktion aus trainierter Strategie → GUI führt `game.step(action)` aus
- Wichtige Details
  - Human Player ID wird über `starting_player` gesetzt (Human ist nicht immer Player 0)
  - Karten austeilen ist game agnostisch: `set_private_card` (1 Karte) vs `set_private_cards` (2 Karten)
  - Privacy: Gegner Karten bleiben verdeckt bis Spielende
  - Optionaler „Hand Strength“ Text im Player Widget
  - Sound pro Aktion über `SoundManager`

### Modus 2: Human vs Human (Netzwerk)
- Dateien
  - Server: `gui/server/http_server.py`
  - Client: `gui/server/http_client.py`
  - GUI: `gui/human_vs_human.py`
  - Runner: `gui/run_server.py`, `gui/run_client.py`
- Architektur
  - Server ist Game Authority: genau ein `game` Objekt läuft zentral
  - Clients sind View + Input: pollen State und senden Actions
- Endpoints (REST)
  - `GET /player_id?name=...` → vergibt Slot 0/1
  - `GET /state?player_id=...` → State + `private_cards` (nur für diesen Player)
  - `POST /action` → Server validiert Turn + `legal_actions`, führt `game.step` aus
  - `POST /reset` → neue Hand
  - `POST /disconnect` → Slot freigeben
- Synchronisation
  - Clients pollen alle 100 ms (QTimer)
  - Server liefert zusätzlich `reset_id`, `player_names`, `history_events`, `done`, `game`
  - Reveal am Ende: Server fügt `opponent_cards` + `payoffs` in den State

### Modus 3: Agent vs Agent (Playback)
- Layout vorhanden: `gui/layouts/agent_vs_agent_layout.py`
- UI Komponente vorhanden: `gui/components/playback_controls.py`
- Logik Datei `gui/agent_vs_agent.py` ist (laut Plan) noch nicht final implementiert

---

## Verwendete Technologien (und wofür)

### PyQt6
- Desktop GUI Framework
- Widgets, Layouts, Signale/Slots
- `QTimer` für:
  - Polling im HTTP Client
  - verzögerte Agent Aktion (Agent vs Human)
  - UI Repositioning nach State Updates

### HTTP + requests (Client)
- `requests` sendet:
  - GET `/player_id`, GET `/state`
  - POST `/action`, POST `/reset`, POST `/disconnect`
- Timeout Handling: Polling Timeouts werden toleriert (Connection bleibt bestehen)

### Flask (Server)
- REST API für Multiplayer
- Threaded server + Lock für thread safety um das zentrale `game` Objekt

### Pickle + gzip (Strategie Dateien)
- Strategien werden als `.pkl.gz` geladen
- Unterstützt zwei Varianten:
  - direkt `average_strategy`
  - oder `strategy_sum` → Umrechnung via `CFRSolver.average_from_strategy_sum`

### Audio
- `gui/audio/sound_manager.py` spielt Sounds für Aktionen ab (check/bet/call/fold)
