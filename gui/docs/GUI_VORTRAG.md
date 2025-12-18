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

---

## Kontext (Bachelorarbeit)
- CFR lernt Strategien für Poker Environments
- Trainierte Strategie = Dictionary:
  - Key: Information Set / Spielzustand
  - Value: Wahrscheinlichkeiten für legale Aktionen
- StrategyAgent sampelt daraus die nächste Aktion

---

## Ziel dieses GUI-Projekts
- Ein UI für mehrere Varianten (Kuhn → Leduc → Rhode → Royal → Limit)
- Dafür normalisiert die GUI Unterschiede im Env State (z.B. `public_card` vs `public_cards`, `player_bets` vs `total_bets`).
- Drei Nutzungsfälle:
  - Agent vs Human (lokal spielen)
  - Human vs Human (online spielen)
  - Agent vs Agent (Visualizer / Replay)

---

## Was ist „Blackbox“?
- Game Environment: Regeln, Legal Actions, Pot/Bets, Terminal Payoffs
- Trainierte Strategie: nur Lookup + Sampling

GUI macht:
- Anzeige + Eingabe
- State in Widgets übersetzen
- (Multiplayer) Transport von Actions/State

---

## Architektur: Layout vs Logik
- `layouts/*`: nur Anordnung (Table, Player, History, Control Area)
- `agent_vs_human.py` / `human_vs_human.py`: Logik + State Updates
- `components/*`: wiederverwendbare Widgets
- `audio/*`: Action-Sounds
- Technologien: PyQt6 (Widgets/Signals/QTimer), Flask (Server), requests (Client), pickle+gzip (Strategie Dateien)

---

## Modus 1: Agent vs Human (lokal)
- Lokales `game` Objekt: UI validiert legal_actions und führt Action via `game.step(action)` aus
- StrategyAgent spielt Gegnerzug aus Strategie Datei (`.pkl.gz`): lädt `average_strategy` oder berechnet sie aus `strategy_sum`
- QTimer für Flow: Agent Delay (2–4s), UI Updates nach jedem Schritt; Sounds + Reveal der Gegnerkarten am Ende

---

## Modus 2: Human vs Human (online)
Prinzip:
- Server ist „Game Authority“ (ein Game-Objekt)
- Zwei Clients sind „View + Input“
- Transport: HTTP + Polling (pokertaugliche Latenz, einfach zu debuggen)
- REST grob: `GET /player_id`, Poll `GET /state`, Actions via `POST /action`, neue Hand via `POST /reset`

---

## Live Demo (Human vs Human)
Setup:
- Server starten (Game auswählen)
- 2 Clients verbinden (Namen)

Während 1–2 Händen zeigen:
- private_cards sind pro Client privat
- Turn Handling (Buttons nur beim eigenen Turn)
- History synchron + Round Separatoren
- Showdown: Reveal + Winner/Payoff

---

## Ausblick
- Agent vs Agent: Playback fertigstellen (step_back/auto-play)
- Optional: Strategietipps im UI (Wahrscheinlichkeiten aus dem Modell)
- Multiplayer: Tests im WLAN/über das Internet sowie einfache Lobby und Scoreboard Funktionen

