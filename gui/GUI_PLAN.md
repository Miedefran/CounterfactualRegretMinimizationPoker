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

### Phase 3: Human vs Human (Später)
1. ⏳ WebSocket-Server
2. ⏳ Client-Implementierung

## 7. Notizen

- **Code-Konventionen**: Keine Kommentare (außer Paper-Referenzen)
- **Vererbung**: Layouts → GUI-Klassen (Layout + Funktionalität)
- **Modularität**: Components wiederverwendbar
- **Erweiterbarkeit**: Einfach neue Game-Varianten hinzufügen

## 8. Neue Komponenten & Erweiterungen

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

## 9. Bekannte Probleme / To-Do

- ⏳ Restart-Button wird manchmal nicht angezeigt (Positionierung)
- ⏳ Strategy-Tipps Widget (optional)
- ⏳ Agent vs Agent Modus refactoren (nutzt noch altes Layout)
