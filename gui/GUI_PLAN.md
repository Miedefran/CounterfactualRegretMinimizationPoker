# GUI/Visualizer Projekt - Architektur & Implementierungsplan

## 1. Projekt-Übersicht

Entwicklung eines GUI/Visualizers für CFR-Poker mit drei Hauptmodi:
- **Agent vs Agent**: Visualisierung von zwei Agenten, die gegeneinander spielen (Replayer)
- **Agent vs Human**: Lokales Gameplay gegen einen trainierten Agenten
- **Human vs Human**: Online-Multiplayer über WebSockets (später)

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
│
└── components/                  # Wiederverwendbare UI-Komponenten
    ├── poker_table.py          # Grüner Poker-Tisch
    ├── visual_card.py          # Visuelle Karten
    ├── hidden_card.py          # Verdeckte Karte (blau)
    ├── chip.py                 # Poker-Chips
    ├── pot_display.py          # Pot-Anzeige
    ├── player_widget.py        # Spieler-Widget (Karten, Name, Status)
    ├── action_buttons.py       # Action-Buttons (Check, Bet, Call, Fold)
    ├── playback_controls.py    # Playback Controls (Play/Pause, Step, Speed)
    └── history_view.py         # History/Log-Anzeige
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

## 4. Nächste Schritte: Agent vs Human Modus

### 4.1 Neue Datei: `gui/agent_vs_human.py`

**Struktur:**
```python
class AgentVsHumanGUI(AgentVsHumanLayout):
    # Erbt vom Layout (nur UI)
    # Fügt Game-Logik hinzu
```

### 4.2 Implementierung

#### Initialisierung
```python
def __init__(self, game, strategy_file=None, human_name="Friedemann", parent=None):
    super().__init__(parent)  # Layout initialisieren
    self.game = game
    self.strategy_file = strategy_file
    self.human_player_id = None  # Wird beim reset_game() gesetzt
    
    # Namen setzen
    self.player_bottom_widget.set_player_name(human_name)
    self.player_top_widget.set_player_name("Strategy Agent")
    
    # Strategy-Agent für Gegner laden
    if strategy_file:
        strategy = self.load_strategy(strategy_file)
        # Agent-Player-ID wird beim reset_game() bestimmt
        self.agent = StrategyAgent(strategy, agent_player_id)
    
    # Action Buttons verbinden
    self.action_buttons.action_selected.connect(self.handle_action)
    
    self.reset_game(0)
```

#### Game-Management

**`reset_game(starting_player=0)`:**
- Game zurücksetzen
- Karten austeilen
- `human_player_id` bestimmen (kann 0 oder 1 sein)
- `agent_player_id = 1 - human_player_id`
- Strategy-Agent mit korrekter Player-ID erstellen
- UI updaten

**`update_display()`:**
- State von Game holen
- Karten anzeigen (mit Privacy)
- Pot, Bets, Legal Actions updaten
- History updaten

#### Action-Handling

**`handle_action(action, bet_size)`:**
- Nur wenn Human dran ist (`game.current_player == human_player_id`)
- Action validieren (`legal_actions`)
- `game.step(action)` ausführen
- UI updaten
- Wenn Agent jetzt dran ist → automatisch `agent_step()` aufrufen

**`agent_step()`:**
- Agent ist dran (`game.current_player == agent_player_id`)
- State holen
- Action von Strategy-Agent holen
- `game.step(action)` ausführen
- UI updaten
- Wenn Human jetzt dran ist → Buttons aktivieren

#### Privacy-Management

**Karten-Anzeige:**
- Human-Karten: `player_bottom_widget.set_cards(..., reveal=True)`
- Agent-Karten: `player_top_widget.set_cards(..., reveal=False)`
- Am Ende: beide `reveal=True`

**Mapping:**
```python
if human_player_id == 0:
    human_cards = game.players[0].private_cards
    agent_cards = game.players[1].private_cards
else:
    human_cards = game.players[1].private_cards
    agent_cards = game.players[0].private_cards
```

#### State-Updates

**`update_cards(state)`:**
- Private Cards holen (mit Mapping für Human/Agent)
- Public Cards holen
- Anzeigen mit Privacy

**`update_actions(state)`:**
- Legal Actions holen
- Buttons enable/disable
- Nur wenn Human dran ist

**`update_players(state)**:**
- Current Player bestimmen
- Highlighting für aktuellen Spieler
- Bets anzeigen

**`update_history(state)`:**
- History aus Game holen
- In HistoryView anzeigen
- Round-Separatoren hinzufügen

### 4.3 Ablauf eines Spiels

```
1. reset_game(starting_player)
   → Game zurücksetzen
   → Karten austeilen
   → human_player_id bestimmen
   → Agent erstellen
   → update_display()

2. update_display()
   → Karten anzeigen (mit Privacy)
   → Pot, Bets, Legal Actions updaten
   → History updaten

3. Wenn game.current_player == human_player_id:
   → Action Buttons aktivieren
   → Warten auf Button-Click

4. handle_action(action)
   → game.step(action)
   → update_display()
   → Wenn Agent jetzt dran: agent_step()

5. agent_step()
   → Action von Agent holen
   → game.step(action)
   → update_display()
   → Wenn Human jetzt dran: Buttons aktivieren

6. Wiederholen bis game.done

7. Am Ende:
   → Beide Karten zeigen (reveal=True)
   → Game Result anzeigen
```

### 4.4 Offene Fragen

- **Strategy-Tipps anzeigen?** (Bet: 60%, Call: 30%, etc.)
  - Optional: Widget für Strategy-Tipps hinzufügen
  - Zeigt Wahrscheinlichkeiten aus Strategy für aktuelle Info-Set

- **Call Amount anzeigen?**
  - `ActionButtons.set_amount_to_call(amount)` existiert bereits
  - Button-Text: "Call (50)" wenn Bet vorhanden

## 5. Nächste Schritte: Agent vs Agent Modus

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

## 6. Implementierungs-Reihenfolge

### Phase 1: Agent vs Human (Aktuell)
1. ✅ Layouts erstellt
2. ⏳ `agent_vs_human.py` implementieren
3. ⏳ Game-Integration testen
4. ⏳ Strategy-Loading testen

### Phase 2: Agent vs Agent
1. ✅ Layout erstellt
2. ⏳ `agent_vs_agent.py` implementieren
3. ⏳ Playback-Funktionalität testen

### Phase 3: Human vs Human (Später)
1. ⏳ WebSocket-Server
2. ⏳ Client-Implementierung

## 7. Notizen

- **Code-Konventionen**: Keine Kommentare (außer Paper-Referenzen)
- **Vererbung**: Layouts → GUI-Klassen (Layout + Funktionalität)
- **Modularität**: Components wiederverwendbar
- **Erweiterbarkeit**: Einfach neue Game-Varianten hinzufügen
