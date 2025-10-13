kuhncase2 nach 50 oder 100 iterations ist bei den constraints noch weit von einem nash equi entfernt, jedoch wenn ich selfplay auf den algorithmus runne mit sehr vielen iterationen(1mio), dann nähert sich der erwartungswert erstaunlich nah an -1/18 an 

anscheinend ist der test ob eine strategie ein nash equi ist über den erwartungswert überhaupt nicht ausreichend
es ist eher ne loose überprüfung
 wenn beide agents ne suboptimale strategie spielen trotzdem nenähnliucher erwartungswert rauskommen kann

muss mir demnächst best response agent angucken 

gibt beim trainieren der modelle verschiedenste sachen die einfluss auf die trainingsgeschwindkeit haben:
meine dicts sind nicht optimal aber das ist zu spät da jetzt meine architektur zu ändern

#######TODO:
ich kann was rausholen an der stelle setupgame with combinations da muss ich nicht immer nen vollen game.reset machen, dealer funcs kann ich mir sparen 


einzige quelle zu leduc benchmarks die ich bisher gefunden hab 
https://cs.stackexchange.com/questions/169593/nash-equilibrium-details-for-leduc-holdem?utm_source=openai


nach 50k iterationen sind wir weiter weg vom benchmark xDDDD












































## **Multithreading MCCFR Funktions-Liste:**

**1. MCCFR Sampling:**
```python
def sample_combinations(combinations, sample_size):
    # Zufällige Teilmenge der Kombinationen
    # Für jeden Thread unterschiedlich
```

**2. Thread-safe Regret Updates:**
```python
def thread_safe_regret_update(info_set_key, action, regret):
    # Mit Lock: regret_sum updaten
    # Thread-sichere Dictionary-Operationen
```

**3. Parallel CFR Iteration:**
```python
def parallel_cfr_iteration(num_threads):
    # ThreadPoolExecutor setup
    # Jeder Thread bekommt eigene Kombinationen
    # Parallel processing
```

**4. Regret Accumulation:**
```python
def accumulate_thread_regrets(thread_results):
    # Regrets von allen Threads sammeln
    # Thread-sichere Merging
```

**5. Thread-safe Strategy Updates:**
```python
def thread_safe_strategy_update(info_set_key, legal_actions):
    # Strategy updates mit Locks
    # Regret-Matching parallel
```

**6. Synchronization Management:**
```python
def synchronize_threads(thread_futures):
    # Warten bis alle Threads fertig
    # Exception handling
    # Cleanup
```

**Das sind 6 Hauptfunktionen** - mehr als Bucketing, aber auch nicht extrem komplex.

**Ist das akkurat?**