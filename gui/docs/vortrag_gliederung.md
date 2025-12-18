## 10–15 Min Vortrag: GUI/Visualizer für CFR-Poker (Advanced Python)

### Einführung: Kontext Bachelorarbeit (30 sec - 1min)
- Ziel: Berechnung von Nash-Gleichgewichts-Strategien mittels CFR
- es existieren bereits game enviroments welche die Poker Logik extern händeln
- Es gibt trainierte Strategien welche mittels dictionarys gespeichert werden
- Key: aktueller Spielzustand; Value: Wahrscheinlichkeiten für legale Aktionen
- Strategy Agent sampelt eine Aktion gemäß diesen Wahrscheinlichkeiten

### Einführung: Ziel und Abgrenzung dieses Projekts (1 min)
- Ziel ist die Entwicklung eines GUI welches für verschiedene Anwendungen genutzt werden kann
- Agent vs Agent Visualisierung
- Agent vs Human Lokales Gameplay
- Human vs Human Online Gameplay

### Theoretische Grundlagen (2 min)
- Software Architektur erklären mit Game Enviroment und trainierten Strategien als Blackbox
  - Basis GUI Klasse; die drei Modi erben davon
  - Agent vs Agent: Step-Mechaniken nutzen Traversierung der Game Klasse
  - Agent vs Human: User gibt Aktionen ein, Validierung durch Game Logik
  - Human vs Human: Kommunikation über Server mit WebSockets (Echtzeit)
- Genutzte Technologien kurz nennen/beschreiben PyQT, Flask, websocket 

### Live Demo (Hauptteil 8 min)
- Für alle sichtbar Agent vs Agent Visualisierung vorstellen, kurz durchsteppen, verschiedene Funktionalitäten vorstellen
  - Play, Pause, Step Forward, Step Backward zeigen
  - Anzeige von aktuellem Zustand, legalen Aktionen, Potsize, ...
- Um das Publikum mit einzubeziehen, Human vs Human vorstellen mit einer Person des Publikums (Anschaulicher als Agent vs Human)
  - 1 - 2 Runden spielen und dabei live auf die Funktionalitäten eingehen
  - Fokus liegt hierbei auf der Synchronisation und der Echtzeitfähigkeit
- Optional: Agent vs Human mit Strategietipps vorstellen
  - 1 Spiel gegen einen Agent spielen, oder eine Publikumsperson gegen einen Agent spielen lassen
  - Strategietipps mittels meiner trainierten Strategien stehen im Fokus


### Ausblick (30 sec)
- Mögliche Erweiterungen
  - Scoreboard/Leaderboard

### 6. Q&A (2 -3 min)


### Alternativer Plan ### 
Ich denke, es könnte sinnvoll sein, die theoretischen Grundlagen, die Live Demo und das Q&A im Vortrag zu vermischen.
-> Am Anfang kurze Abgrenzung und Ziel des Projekts -> dann direkt Live Demo; während der Live Demo fachliche Inhalte erklären und gleichzeitig bereits Fragen entgegennehmen

So könnte ich die kurze Vortragszeit mehr auf die Themen fokussieren, die das Publikum wirklich interessieren.
So könnte man durch hohe Publikumseinbindung von Anfang an die Aufmerksamkeit der Zuhörer besser halten.

Ausserdem liegen mir Vorträge mit einer "freieren" Struktur einfach besser.
