# Counterfactual Regret Minimization Poker

## Disclaimer

Der Code im Ordner GUI wurde im Rahmen des Moduls Advanced Python erstellt und ist nicht Teil dieser Bachelorarbeit. Die Arbeit nutzt das GUI lediglich, um trainierte Strategien spielbar zu machen.

---

## Strategie trainieren und dagegen spielen

1. **Abhängigkeiten installieren**
   ```
   uv sync
   ```

2. **TUI starten**
   ```
   uv run python src/tui/app.py
   ```
   oder mit mise:
   ```
   mise run tui
   ```

3. **Algorithmus trainieren**
   - Im Tab **Training Queue** wird ein neuer Trainings-Task für ein beliebiges Spiel angelegt.
   - Empfehlung für wenig Zeit: Twelve Card Poker, DCFR oder CFR+ with Flat Tree, bis 1 mb/g mit Early Exit, 10k iterations
   - Empfehlung für mehr Zeit: Small Island Hold'em, DCFR oder CFR+ with Flat Tree, bis 1mb/g Early exit, 10k iterations

4. **Evaluation ansehen**
   - Im Tab **Current Task** werden während und nach dem Training Fortschritt und Exploitability-Kurve des aktuellen Laufs angezeigt.

5. **Gegen eine Strategie spielen**
   - Im Tab **Models** werden die trainierten Modelle aufgelistet. Nach Auswahl eines Modells wird mit der Tastenkombination **g** das GUI gestartet; es wird gegen die gewählte Strategie gespielt.

---

## Hinweise

GUI und TUI sind Work in Progess.
Das heisst es können dabei Fehler auftreten.

Bei Beendigung einer über die TUI gestarteten GUI und erneutem Öffnen einer weiteren Instanz (z. B. für ein anderes Modell) kann die Anwendung abstürzen. In diesem Fall muss das Terminal beendet werden. Um gegen ein anderes Modell zu spielen, wird empfohlen, die TUI zu schließen und neu zu starten.
