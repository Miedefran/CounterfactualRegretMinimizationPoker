# Implementierung

# Hier grade am debuggen
1. Evaluationstools                 
1.1 Best Response Agent 
1.2 Exploitability Berechnung
1.3 Konvergenzplots
1.4 Andere Plots EV, ?

2.Optimierte CFR Algorithmen
2.1 CFR+ (sollte einfach sein) 	✅										
2.2 MCCFR (same)
(2.3 MCCFR mit Multithreading)

3. Abstraktionsmethoden
(3.1 Bucketing via Handstrength)




# Dokumentation:

# Kapitel 1: Einleitung


# Kapitel 2: Theoretische Grundlagen

# OsborneRubinsteinMasterpiece 1994
2.1 Grundlagen der Spieltheorie
2.1.1 Definition und Konzepte 
    Was ist spieltheorie 
    Was ist ein Spiel
    Was ist eine Lösung
    Spieltypen(Strategic Games, Extensive Games(Perfect/Imperfect Information)) 
2.1.2 Extensive Games with Imperfect Information
    Information Sets
    Strategien in Extensive Games
    Mixed Stratgies
2.1.3 Nash Equilibrium
    Definition
    Nash Equilibrium in Extensive Games 
    Manuelle Berechnung Kuhn 1951
    Grenzen der Manuellen Berechnung

# Zinkevich 2007 Regret Minimization in Games with Imperfect Information
2.2 Counter Factual Regret Minimization 
2.2.1 Regret Minimization
    - Was ist Regret?
    - Verbindung zu Nash Equilibrium
    - Problem: Regret Minimization in Extensive Games

2.2.2 Counterfactual Regret
    - Was ist Counterfactual Utility?
    - Was ist Immediate Counterfactual Regret?
    - Overall Regret ≤ Sum of Counterfactual Regret

2.2.3 Regret Matching
    - Strategy Update
    - Konvergenz zum Nash Equilibrium

2.2.4 CFR Algorithmus
    - CFR Iteration
    - Average Strategy

# Lanctot et al. 2009 Monte Carlo Sampling for Regret Minimization
2.4 MCCFR
2.4.1 Motivation und Framework
    - Problem: Vanilla CFR traversiert gesamten Game Tree
    - MCCFR Framework: Sampling mit Blocks
    - Unbiased Estimator
2.4.2 Sampling-Strategien
    - Outcome-Sampling MCCFR
    - External-Sampling MCCFR
2.4.3 Theoretische Eigenschaften
    - Regret Bounds
    - Konvergenz-Eigenschaften
    - Vergleich zu CFR

# Tammelin et al. 2015 Solving Heads-Up Limit Texas Hold'em
2.3 CFR+
2.3.1 Regret-Matching+
    - Unterschied zu Regret-Matching
    - Tracking Regret

2.3.2 Linearly Weighted Average
    - Weighted Average Strategy

2.3.3 Theoretische Eigenschaften
    - Regret Bounds
    - Konvergenz-Eigenschaften
    - Vergleich zu CFR

(2.4 Bucketing)

# Kapitel 3: Analyse
3.1 Spielvarianten
    - Kuhn Poker
    - Leduc Hold'em
    - Eigene 12 Karten Variante
    - Rhode Island Hold'em

3.2 CFR Solver
    - Vanilla CFR
    - CFR+
    - MCCFR

3.3 Evaluationsmethoden
    - Erwartungswert Selfplay
    - Best Response Agent / Exploitability
    - Konvergenzplots


# Kapitel 4: Design/Architektur 
UML 
Klassendiagramme Game enviroment 
CFR Solver Klassendiagramm + Sequenzdiagramm einer Iteration

# Kapitel 5: Implementierung
5.1 Game Environment
    - Grundstruktur player, judger, dealer, game, round
    - Vererbung
    - State methods

5.2 CFR Solver
    - Initialisierung, Combination Generator
    - Training Loop 
    - Regret Matching
    - Average Strategy 
    - Storage 

5.3 Evaluationsmethoden
    - Best Response Agent
    - Exploitability Berechnung
    - Selfplay

# Kapitel 6: Evaluation
6.1 Ergebnisse
    - Kuhn Poker Ergebnisse
    - Leduc Hold'em Ergebnisse
    - Eigene Version
    - (Rhode Island Hold'em Ergebnisse)

6.2 Vergleich der Algorithmen
    - CFR vs. CFR+
    - CFR vs. MCCFR
    - Konvergenzplots
    - (sbstraktionsmethoden, Multithread)

6.3 Vergleich mit Literaturwerten/Anderen Modellen
    - Kuhn 1951 (analytische Lösung)
    - Zinkevich 2007 (CFR)
    - Lanctot et al. 2009 (MCCFR)
    - Tammelin et al. 2015 (CFR+)
    - OpenSpiel/RL Card 
    - random Leute in Foren 
    https://cs.stackexchange.com/questions/169593/nash-equilibrium-details-for-leduc-holdem

# Kapitel 7: Fazit
7.1 Zusammenfassung
    - Ergebnisse
    - Erkenntnisse

7.2 Lessons Learned
    - Herausforderungen
    - Lösungsansätze

7.3 Future Work
    -gucken




