# Start-Anleitung (ohne ngrok, lokal)

Öffne **3 Terminal-Tabs** in Cursor und führe jeweils einen der folgenden Befehle aus:

## Terminal 1: Server
```bash
python3 gui/runner/run_server.py --host 0.0.0.0 --port 8888 --game leduc
```

## Terminal 2: Client 1 (Player1)
```bash
python3 gui/runner/run_client.py --ip localhost --port 8888 --name Player1
```

## Terminal 3: Client 2 (Player2)
```bash
python3 gui/runner/run_client.py --ip localhost --port 8888 --name Player2
```

---

**Hinweis (Game wechseln):**
Setze beim Server `--game` auf eins von:
`kuhn`, `leduc`, `twelve_card`, `rhode_island`, `royal_holdem`, `limit_holdem`

---

# Agent vs Human (lokal)

Öffne **1 Terminal-Tab** und starte die GUI direkt:

```bash
python3 gui/runner/run_agent_vs_human.py --game kuhn
```

Für Leduc:

```bash
python3 gui/runner/run_agent_vs_human.py --game leduc
```

Optional: eigenes Strategy-File angeben:

```bash
python3 gui/runner/run_agent_vs_human.py --game leduc --strategy "models/leduc/cfr/leduc_1000.pkl.gz"
```


