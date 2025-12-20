---
marp: true
title: CFR-Poker GUI
paginate: true
style: |
  section {
    /* Clean poker look: green felt + subtle vignette, red accent */
    background:
      radial-gradient(1200px 700px at 50% 18%, rgba(255,255,255,0.10) 0%, rgba(255,255,255,0) 55%),
      radial-gradient(1600px 900px at 50% 60%, rgba(0,0,0,0) 45%, rgba(0,0,0,0.40) 100%),
      linear-gradient(180deg, #0f6b4b 0%, #0a4e38 55%, #083528 100%);
    color: #f8fafc;
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    font-size: 32px;
    line-height: 1.25;
    padding: 70px;
  }

  h1, h2 {
    color: #f8fafc;
    letter-spacing: 0.2px;
  }

  h1 { font-size: 58px; margin-bottom: 18px; }
  h2 { font-size: 48px; margin-bottom: 16px; }

  code {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.12);
    padding: 0.12em 0.35em;
    border-radius: 8px;
    font-size: 0.9em;
  }

  /* Improve bullets readability on dark felt */
  ul { margin: 0.5em 0 0 1.05em; }
  li { margin: 0.28em 0; }
  li::marker { color: rgba(248, 250, 252, 0.9); }

  section::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    width: 100%;
    height: 10px;
    background: linear-gradient(90deg, #b91c1c, #ef4444, #b91c1c);
    box-shadow: 0 2px 0 rgba(0,0,0,0.25), 0 10px 30px rgba(0,0,0,0.25);
  }

  /* Page number + subtle bottom shadow (no wooden rail) */
  section::after {
    color: rgba(229, 231, 235, 0.6);
    font-size: 18px;
    text-shadow: 0 2px 10px rgba(0,0,0,0.35);
  }

  /* Red accent for emphasis (optional via <span class="accent">) */
  .accent { color: #ef4444; }
---

## CFR-Poker GUI
Visualizer + Multiplayerdemo für CFR Poker

Modul: Advanced Python Wise25/26
Dozent: Markus Schubert

---

## Bachelorarbeit
- CFR lernt Strategien für Poker Environments
- Trainierte Strategie = Dictionary:
  - Key: Information Set / Spielzustand
  - Value: Wahrscheinlichkeiten für legale Aktionen
- StrategyAgent sampelt daraus die nächste Aktion
- Game Enviroment verwaltet Legale Aktionen in einem bestimmten Infoset, Pot, Runden, Deck, Hand Evaluation, ...

---

## Ziel dieses Projekts
- Ein GUI für mehrere Varianten (Kuhn → Leduc → Twelve Card Poker → Rhode → Royal → Limit)
- Normalisierung: GUI vereinheitlicht unterschiedliche Env States (z.B. `public_card` vs `public_cards`, `player_bets` vs `total_bets`)
- Drei Nutzungsfälle:
  - Agent vs Human (Gegen eine trainierte Strategie spielen)
  - Human vs Human (online spielen)
  - Agent vs Agent (Visualizer / Replay)

---

## Aufbau
- `layouts/*`: nur Anordnung (Table, Player, History, Control Area)
- `agent_vs_human.py` / `human_vs_human.py`: Logik + State Updates
- `components/*`: PokerTable, PlayerWidgets, VisualCard, PotDisplay+Chips, HistoryView, ActionButtons, PlaybackControls
- `audio/*`: Action-Sounds
- `runner/*`: CLI-Argumente parsen 


---

### Layout (Agent/Human vs Human)

**Hierarchische Struktur:**
- Game Area: `QHBoxLayout` (horizontal, teilt Links/Rechts)
  - Links: `QVBoxLayout` (vertikal)
    - Player Top → Table → Player Bottom → Controls
  - Rechts: History View (ausklappbar)
- Table: Community Cards (`QHBoxLayout`), Pot Display, Chips
  - Pot: Label zeigt Wert + Chips zufällig angeordnet
- Player Widgets: Karten horizontal (`QHBoxLayout`), Name, Hand Strength


---

### Grafische Komponenten

- **Karten**: `QLabel` rendert HTML-Code → Karte wird mit HTML/CSS "gemalt" (Rank/Suit Layout, Unicode-Symbole ♠♥♦♣, Farben rot/schwarz)
- **Verdeckte Karten**: `QLabel` mit CSS-Styling (blauer Hintergrund #1a5490)
- **Tisch**: `QPainter` zeichnet grünen Poker-Tisch (`drawRoundedRect`)
- **Chips**: `QPainter` zeichnet Ellipsen (Ring + innerer Kreis)

---

### Modus 1: Agent vs Human

- GUI hält ein lokales `game` Objekt
- Human klickt → GUI prüft `legal_actions` → `game.step(action)` → UI Update
- Agent Zug → `StrategyAgent` sampelt Aktion aus Strategie (Dictionary: InfoSet → Wahrscheinlichkeiten) → `game.step(action)` → UI Update
- Am Ende werden Karten aufgedeckt, Gewinner/Payoff steht im History View

---

### Modus 2: Human vs Human 

- Server = Game Authority, Clients = View + Input
- **REST API**: Architekturstil für Client-Server-Kommunikation über HTTP; Server stellt Endpoints bereit, Client sendet Anfragen (GET = Daten abrufen, POST = Aktion ausführen)
- **Flask**: Python-Framework zum einfachen Erstellen von REST APIs; Client sendet HTTP-Anfragen (GET/POST); **QTimer** für Polling alle 100ms
- Endpoints: `GET /player_id` (Slot 0/1), `GET /state` (State + private_cards), `POST /action`, `POST /reset`, `POST /disconnect`
- Thread-Safety: Flask threaded Server (mehrere Clients gleichzeitig) + Lock um zentrales `game` Objekt (verhindert Race Conditions)


---

## Ausblick
- Agent vs Agent Modus
- Optional: Strategietipps im UI 
- Multiplayer ausserhalb des selben Netzwerks, Scoreboard, begrenzter Chip Stack, No Limit Holdem, Über UI zu Server verbinden anstatt Terminal

