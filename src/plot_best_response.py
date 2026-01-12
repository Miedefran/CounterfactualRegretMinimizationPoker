"""
Skript zum Erstellen von Plots aus Best Response pkl.gz Dateien.

Kann einzelne oder mehrere Dateien plotten, um verschiedene Algorithmen zu vergleichen.
"""

import os
import sys
import argparse
import gzip
import pickle
import math
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.best_response_evaluator import BestResponseTracker


def load_best_response_data(filepath, suppress_warnings=True):
    """
    Lädt Best Response Daten aus einer pkl.gz Datei.
    
    Args:
        filepath: Pfad zur pkl.gz Datei
        suppress_warnings: Wenn True, werden Warnungen unterdrückt
    
    Returns:
        BestResponseTracker Objekt oder None bei Fehler
    """
    try:
        # Lade direkt aus der Datei, ohne BestResponseTracker.__init__ aufzurufen
        # (um die Public State Tree Warnung zu vermeiden)
        with gzip.open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Erstelle ein minimales Tracker-Objekt ohne __init__ aufzurufen
        # Wir verwenden object.__new__ um __init__ zu umgehen
        tracker = object.__new__(BestResponseTracker)
        
        # Setze alle notwendigen Attribute manuell
        tracker.game_name = data.get('game_name', 'unknown')
        loaded_values = data.get('values', [])
        
        # Rückwärtskompatibilität: Konvertiere alte Formate zu neuem Format
        if loaded_values:
            if len(loaded_values[0]) == 2:
                # Altes Format: (iteration, exploitability)
                tracker.values = [(it, exp, None, None, 0.0) for it, exp in loaded_values]
            elif len(loaded_values[0]) == 4:
                # Altes Format: (iteration, exploitability_mb, br_value_p0, br_value_p1)
                tracker.values = [(it, exp, br0, br1, 0.0) for it, exp, br0, br1 in loaded_values]
            else:
                # Neues Format mit Zeit
                tracker.values = loaded_values
        else:
            tracker.values = []
        
        tracker.total_br_time = 0.0
        tracker.last_eval_iteration = 0
        
        # Schedule Config
        if 'schedule_config' in data:
            tracker.schedule_config = data['schedule_config']
            tracker.schedule_type = tracker.schedule_config.get("type", "fixed")
            
            if tracker.schedule_type == "fixed":
                tracker.interval = tracker.schedule_config.get("interval", 100)
            elif tracker.schedule_type == "logarithmic":
                tracker.base_interval = tracker.schedule_config.get("base_interval", 10)
                tracker.target_iteration = tracker.schedule_config.get("target_iteration", 100)
                tracker.target_interval = tracker.schedule_config.get("target_interval", 100)
                tracker.unbounded = tracker.schedule_config.get("unbounded", False)
            elif tracker.schedule_type == "custom":
                tracker.custom_schedule = tracker.schedule_config.get("schedule", [])
        else:
            # Fallback für alte Dateien
            tracker.schedule_config = {"type": "fixed", "interval": 100}
            tracker.schedule_type = "fixed"
            tracker.interval = 100
        
        # Tree wird nicht geladen (nicht benötigt für Plots)
        tracker.tree = None
        
        return tracker
    except Exception as e:
        if not suppress_warnings:
            print(f"Fehler beim Laden von {filepath}: {e}")
        return None


def extract_algorithm_name(filepath):
    """
    Extrahiert den Algorithmus-Namen aus dem Dateipfad.
    
    Beispiel: data/models/leduc/cfr_optimized/1000/leduc_1000_best_response.pkl.gz
    -> "cfr_optimized"
    """
    parts = filepath.split(os.sep)
    # Suche nach bekannten Algorithmus-Namen im Pfad
    for part in parts:
        if any(alg in part for alg in ['cfr', 'mccfr', 'chance', 'external']):
            return part
    # Fallback: Verwende den Dateinamen
    filename = os.path.basename(filepath)
    return filename.replace('_best_response.pkl.gz', '').replace('.pkl.gz', '')


def load_model_data(filepath):
    """
    Lädt training_time und iteration_count aus einer Model-Datei.
    
    Args:
        filepath: Pfad zur Model pkl.gz Datei (nicht Best Response Datei)
    
    Returns:
        Tuple (training_time, iteration_count) oder (None, None) bei Fehler
    """
    try:
        with gzip.open(filepath, 'rb') as f:
            data = pickle.load(f)
        training_time = data.get('training_time', None)
        iteration_count = data.get('iteration_count', None)
        return training_time, iteration_count
    except Exception as e:
        return None, None


def get_model_filepath(best_response_filepath):
    """
    Konvertiert einen Best Response Dateipfad zu einem Model Dateipfad.
    
    Args:
        best_response_filepath: Pfad zur Best Response Datei
    
    Returns:
        Pfad zur Model Datei
    """
    # Ersetze "_best_response.pkl.gz" mit ".pkl.gz"
    return best_response_filepath.replace('_best_response.pkl.gz', '.pkl.gz')


def plot_manual_time_comparison(filepaths, output_path=None, title=None, log_log=True, custom_labels=None):
    """
    Plottet manuell ausgewählte Best Response Dateien gegen Zeit.
    
    Diese Funktion ist speziell für manuelle Dateiauswahl gedacht und bietet
    mehr Kontrolle über Labels und Darstellung.
    
    Args:
        filepaths: Liste von Pfaden zu pkl.gz Dateien
        output_path: Optionaler Pfad zum Speichern des Plots
        title: Optionaler Titel für den Plot
        log_log: Wenn True, werden beide Achsen logarithmisch skaliert
        custom_labels: Optionales Dictionary {filepath: label} für benutzerdefinierte Labels
    
    Returns:
        True wenn erfolgreich, False bei Fehler
    """
    if not filepaths:
        print("Fehler: Keine Dateien zum Plotten angegeben")
        return False
    
    # Lade alle Daten
    trackers = []
    labels = []
    model_times = []
    valid_filepaths = []
    
    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(f"Warnung: Datei nicht gefunden: {filepath}")
            continue
        
        tracker = load_best_response_data(filepath)
        if tracker is not None:
            # Lade Model-Datei um training_time und iteration_count zu bekommen
            model_path = get_model_filepath(filepath)
            training_time, iteration_count = load_model_data(model_path)
            
            if training_time is None or iteration_count is None:
                print(f"Warnung: Konnte training_time/iteration_count nicht aus {model_path} laden, überspringe {filepath}")
                continue
            
            trackers.append(tracker)
            model_times.append((training_time, iteration_count))
            valid_filepaths.append(filepath)
            
            # Verwende custom_label falls vorhanden, sonst extrahiere Algorithmus-Namen
            if custom_labels and filepath in custom_labels:
                labels.append(custom_labels[filepath])
            else:
                labels.append(extract_algorithm_name(filepath))
        else:
            print(f"Warnung: Konnte Daten nicht aus {filepath} laden, überspringe")
    
    if not trackers:
        print("Fehler: Keine gültigen Daten zum Plotten gefunden")
        return False
    
    print(f"\nPlotte {len(trackers)} Dateien:")
    for filepath, label in zip(valid_filepaths, labels):
        print(f"  - {label}: {filepath}")
    
    # Erstelle Plot
    plt.rcParams['text.usetex'] = False
    plt.rcParams['mathtext.default'] = 'regular'
    
    plt.figure(figsize=(14, 8))
    
    # Farben für verschiedene Algorithmen
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Marker für verschiedene Algorithmen
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'x', '+']
    
    for i, (tracker, label, (total_time, total_iterations)) in enumerate(zip(trackers, labels, model_times)):
        if not tracker.values:
            print(f"Warnung: {label} hat keine Werte")
            continue
        
        # Extrahiere Daten
        iterations = [x[0] for x in tracker.values]
        exploitability = [x[1] for x in tracker.values]
        
        # Prüfe ob Zeit bereits in den values gespeichert ist (neues Format)
        if len(tracker.values[0]) >= 5:
            # Neues Format: Zeit ist bereits gespeichert
            times = [x[4] for x in tracker.values]
        else:
            # Altes Format: Berechne Zeit linear (Fallback)
            if total_iterations > 0 and total_time > 0:
                time_per_iteration = total_time / total_iterations
            else:
                # Fallback: Verwende max_iter aus den Best Response Werten
                max_iter = max(iterations) if iterations else 1
                time_per_iteration = total_time / max_iter if max_iter > 0 else 0
            
            # Konvertiere Iterationen zu kumulativer Zeit
            times = [time_per_iteration * iter for iter in iterations]
        
        # Normalisiere Zeit relativ zum ersten Messpunkt (alle beginnen bei 0)
        if times:
            first_time = times[0]
            times = [t - first_time for t in times]
        
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        plt.plot(times, exploitability, 
                label=label, 
                marker=marker, 
                linewidth=2, 
                markersize=6,
                color=color,
                markevery=max(1, len(times) // 20))
    
    # Achsen-Skalierung
    if log_log:
        plt.xscale('log')
        plt.yscale('log')
        xlabel = 'Time (seconds, log scale)'
        ylabel = 'Exploitability (mb/g, log scale)'
    else:
        xlabel = 'Time (seconds)'
        ylabel = 'Exploitability (mb/g)'
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    # Titel
    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    elif trackers:
        game_name = trackers[0].game_name
        plt.title(f'Exploitability vs Time Comparison ({game_name})', fontsize=14, fontweight='bold')
    
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, which='both')
    
    # Setze sinnvolle Ticks mit wissenschaftlicher Notation (10^x)
    if trackers:
        all_times = []
        all_exploitability = []
        for tracker, (total_time, total_iterations) in zip(trackers, model_times):
            if tracker.values:
                iterations = [x[0] for x in tracker.values]
                exploitability = [x[1] for x in tracker.values if x[1] > 0]
                
                # Prüfe ob Zeit bereits gespeichert ist
                if len(tracker.values[0]) >= 5:
                    # Neues Format: Verwende gespeicherte Zeit
                    times = [x[4] for x in tracker.values if x[1] > 0]
                    # Normalisiere Zeit relativ zum ersten Messpunkt
                    if times:
                        first_time = times[0]
                        times = [t - first_time for t in times]
                else:
                    # Altes Format: Berechne linear
                    if total_iterations > 0 and total_time > 0:
                        time_per_iteration = total_time / total_iterations
                        times = [time_per_iteration * iter for iter in iterations if any(x[0] == iter and x[1] > 0 for x in tracker.values)]
                        # Normalisiere Zeit relativ zum ersten Messpunkt
                        if times:
                            first_time = times[0]
                            times = [t - first_time for t in times]
                    else:
                        times = []
                
                all_times.extend(times)
                all_exploitability.extend(exploitability)
        
        if all_times and all_exploitability:
            min_time = min(t for t in all_times if t > 0)
            max_time = max(all_times)
            min_exp = min(all_exploitability)
            max_exp = max(all_exploitability)
            
            if log_log:
                # Log-Log Skalierung: Verwende wissenschaftliche Notation
                from matplotlib.ticker import LogLocator, LogFormatter
                plt.gca().xaxis.set_major_locator(LogLocator(base=10, numticks=10))
                plt.gca().yaxis.set_major_locator(LogLocator(base=10, numticks=10))
                plt.gca().xaxis.set_major_formatter(LogFormatter(labelOnlyBase=False))
                plt.gca().yaxis.set_major_formatter(LogFormatter(labelOnlyBase=False))
    
    # Speichere Plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot gespeichert: {output_path}")
    else:
        plt.show()
    
    plt.close()
    return True


def plot_multiple_best_responses_by_time(filepaths, output_path=None, title=None, log_log=True):
    """
    Plottet mehrere Best Response Dateien gegen Zeit statt gegen Iterationen.
    
    Args:
        filepaths: Liste von Pfaden zu pkl.gz Dateien
        output_path: Optionaler Pfad zum Speichern des Plots
        title: Optionaler Titel für den Plot
        log_log: Wenn True, werden beide Achsen logarithmisch skaliert
    """
    if not filepaths:
        print("Keine Dateien zum Plotten angegeben")
        return
    
    # Lade alle Daten
    trackers = []
    labels = []
    model_times = []
    
    for filepath in filepaths:
        tracker = load_best_response_data(filepath)
        if tracker is not None:
            # Lade Model-Datei um training_time und iteration_count zu bekommen
            model_path = get_model_filepath(filepath)
            training_time, iteration_count = load_model_data(model_path)
            
            if training_time is None or iteration_count is None:
                print(f"Warnung: Konnte training_time/iteration_count nicht aus {model_path} laden, überspringe {filepath}")
                continue
            
            trackers.append(tracker)
            model_times.append((training_time, iteration_count))
            # Extrahiere Algorithmus-Namen für Label
            alg_name = extract_algorithm_name(filepath)
            labels.append(alg_name)
        else:
            print(f"Überspringe {filepath} wegen Fehler")
    
    if not trackers:
        print("Keine gültigen Daten zum Plotten gefunden")
        return
    
    # Erstelle Plot
    plt.rcParams['text.usetex'] = False
    plt.rcParams['mathtext.default'] = 'regular'
    
    plt.figure(figsize=(14, 8))
    
    # Farben für verschiedene Algorithmen
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Marker für verschiedene Algorithmen
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'x', '+']
    
    for i, (tracker, label, (total_time, total_iterations)) in enumerate(zip(trackers, labels, model_times)):
        if not tracker.values:
            print(f"Warnung: {label} hat keine Werte")
            continue
        
        # Extrahiere Daten
        iterations = [x[0] for x in tracker.values]
        exploitability = [x[1] for x in tracker.values]
        
        # Prüfe ob Zeit bereits in den values gespeichert ist (neues Format)
        if len(tracker.values[0]) >= 5:
            # Neues Format: Zeit ist bereits gespeichert
            times = [x[4] for x in tracker.values]
        else:
            # Altes Format: Berechne Zeit linear (Fallback)
            if total_iterations > 0 and total_time > 0:
                time_per_iteration = total_time / total_iterations
            else:
                # Fallback: Verwende max_iter aus den Best Response Werten
                max_iter = max(iterations) if iterations else 1
                time_per_iteration = total_time / max_iter if max_iter > 0 else 0
            
            # Konvertiere Iterationen zu kumulativer Zeit
            times = [time_per_iteration * iter for iter in iterations]
        
        # Normalisiere Zeit relativ zum ersten Messpunkt (alle beginnen bei 0)
        if times:
            first_time = times[0]
            times = [t - first_time for t in times]
        
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        plt.plot(times, exploitability, 
                label=label, 
                marker=marker, 
                linewidth=2, 
                markersize=6,
                color=color,
                markevery=max(1, len(times) // 20))
    
    # Achsen-Skalierung
    if log_log:
        plt.xscale('log')
        plt.yscale('log')
        xlabel = 'Time (seconds, log scale)'
        ylabel = 'Exploitability (mb/g, log scale)'
    else:
        xlabel = 'Time (seconds)'
        ylabel = 'Exploitability (mb/g)'
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    # Titel
    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    elif trackers:
        game_name = trackers[0].game_name
        plt.title(f'Exploitability vs Time Comparison ({game_name})', fontsize=14, fontweight='bold')
    
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, which='both')
    
    # Setze sinnvolle Ticks mit wissenschaftlicher Notation (10^x)
    if trackers:
        all_times = []
        all_exploitability = []
        for tracker, (total_time, total_iterations) in zip(trackers, model_times):
            if tracker.values:
                iterations = [x[0] for x in tracker.values]
                exploitability = [x[1] for x in tracker.values if x[1] > 0]
                
                # Prüfe ob Zeit bereits gespeichert ist
                if len(tracker.values[0]) >= 5:
                    # Neues Format: Verwende gespeicherte Zeit
                    times = [x[4] for x in tracker.values if x[1] > 0]
                    # Normalisiere Zeit relativ zum ersten Messpunkt
                    if times:
                        first_time = times[0]
                        times = [t - first_time for t in times]
                else:
                    # Altes Format: Berechne linear
                    if total_iterations > 0 and total_time > 0:
                        time_per_iteration = total_time / total_iterations
                        times = [time_per_iteration * iter for iter in iterations if any(x[0] == iter and x[1] > 0 for x in tracker.values)]
                        # Normalisiere Zeit relativ zum ersten Messpunkt
                        if times:
                            first_time = times[0]
                            times = [t - first_time for t in times]
                    else:
                        times = []
                
                all_times.extend(times)
                all_exploitability.extend(exploitability)
        
        if all_times:
            max_time = max(all_times)
            if log_log:
                # X-Achsen Ticks - verwende nur Potenzen von 10
                min_time = min([t for t in all_times if t > 0])
                min_power = int(math.floor(math.log10(min_time))) if min_time > 0 else -2
                max_power = int(math.ceil(math.log10(max_time))) if max_time > 0 else 2
                x_ticks = [10**i for i in range(min_power, max_power + 1)]
                x_ticks = [t for t in x_ticks if t <= max_time * 1.1]
                x_tick_labels = [f'$10^{{{int(math.log10(t))}}}$' for t in x_ticks]
                plt.xticks(x_ticks, x_tick_labels)
                
                # Y-Achsen Ticks - verwende nur Potenzen von 10
                if all_exploitability:
                    min_exp = min(all_exploitability)
                    max_exp = max(all_exploitability)
                    
                    min_power = int(math.floor(math.log10(min_exp))) if min_exp > 0 else -2
                    max_power = int(math.ceil(math.log10(max_exp))) if max_exp > 0 else 2
                    y_ticks = [10**i for i in range(min_power, max_power + 1)]
                    y_ticks = [t for t in y_ticks if min_exp * 0.5 <= t <= max_exp * 2]
                    
                    if y_ticks:
                        y_tick_labels = [f'$10^{{{int(math.log10(t))}}}$' for t in y_ticks]
                        plt.yticks(y_ticks, y_tick_labels)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot gespeichert: {output_path}")
    else:
        plt.show()


def plot_multiple_best_responses(filepaths, output_path=None, title=None, log_log=True):
    """
    Plottet mehrere Best Response Dateien auf einem Graph.
    
    Args:
        filepaths: Liste von Pfaden zu pkl.gz Dateien
        output_path: Optionaler Pfad zum Speichern des Plots
        title: Optionaler Titel für den Plot
        log_log: Wenn True, werden beide Achsen logarithmisch skaliert
    """
    if not filepaths:
        print("Keine Dateien zum Plotten angegeben")
        return
    
    # Lade alle Daten
    trackers = []
    labels = []
    
    for filepath in filepaths:
        tracker = load_best_response_data(filepath)
        if tracker is not None:
            trackers.append(tracker)
            # Extrahiere Algorithmus-Namen für Label
            alg_name = extract_algorithm_name(filepath)
            labels.append(alg_name)
        else:
            print(f"Überspringe {filepath} wegen Fehler")
    
    if not trackers:
        print("Keine gültigen Daten zum Plotten gefunden")
        return
    
    # Erstelle Plot
    # Aktiviere LaTeX-Rendering für wissenschaftliche Notation
    plt.rcParams['text.usetex'] = False  # Deaktiviere LaTeX (braucht LaTeX Installation)
    plt.rcParams['mathtext.default'] = 'regular'  # Verwende normale Math-Text
    
    plt.figure(figsize=(14, 8))
    
    # Farben für verschiedene Algorithmen
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Marker für verschiedene Algorithmen
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'x', '+']
    
    for i, (tracker, label) in enumerate(zip(trackers, labels)):
        if not tracker.values:
            print(f"Warnung: {label} hat keine Werte")
            continue
        
        iterations = [x[0] for x in tracker.values]
        exploitability = [x[1] for x in tracker.values]
        
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        plt.plot(iterations, exploitability, 
                label=label, 
                marker=marker, 
                linewidth=2, 
                markersize=6,
                color=color,
                markevery=max(1, len(iterations) // 20))  # Zeige nicht alle Marker
    
    # Achsen-Skalierung
    if log_log:
        plt.xscale('log')
        plt.yscale('log')
        xlabel = 'Iterations (log scale)'
        ylabel = 'Exploitability (mb/g, log scale)'
    else:
        xlabel = 'Iterations'
        ylabel = 'Exploitability (mb/g)'
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    # Titel
    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    elif trackers:
        game_name = trackers[0].game_name
        plt.title(f'Exploitability Comparison ({game_name})', fontsize=14, fontweight='bold')
    
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, which='both')
    
    # Setze sinnvolle Ticks mit wissenschaftlicher Notation (10^x)
    if trackers:
        all_iterations = []
        all_exploitability = []
        for tracker in trackers:
            if tracker.values:
                all_iterations.extend([x[0] for x in tracker.values])
                all_exploitability.extend([x[1] for x in tracker.values if x[1] > 0])
        
        if all_iterations:
            max_iter = max(all_iterations)
            if log_log:
                # X-Achsen Ticks - verwende nur Potenzen von 10
                import math
                min_iter = min(all_iterations)
                min_power = int(math.floor(math.log10(min_iter))) if min_iter > 0 else 0
                max_power = int(math.ceil(math.log10(max_iter))) if max_iter > 0 else 0
                x_ticks = [10**i for i in range(min_power, max_power + 1)]
                x_ticks = [t for t in x_ticks if t <= max_iter * 1.1]
                x_tick_labels = [f'$10^{{{int(math.log10(t))}}}$' for t in x_ticks]
                plt.xticks(x_ticks, x_tick_labels)
                
                # Y-Achsen Ticks - verwende nur Potenzen von 10
                if all_exploitability:
                    min_exp = min(all_exploitability)
                    max_exp = max(all_exploitability)
                    
                    min_power = int(math.floor(math.log10(min_exp))) if min_exp > 0 else -2
                    max_power = int(math.ceil(math.log10(max_exp))) if max_exp > 0 else 2
                    y_ticks = [10**i for i in range(min_power, max_power + 1)]
                    y_ticks = [t for t in y_ticks if min_exp * 0.5 <= t <= max_exp * 2]
                    
                    if y_ticks:
                        y_tick_labels = [f'$10^{{{int(math.log10(t))}}}$' for t in y_ticks]
                        plt.yticks(y_ticks, y_tick_labels)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot gespeichert: {output_path}")
    else:
        plt.show()


def is_sampling_algorithm(filepath):
    """
    Prüft ob ein Algorithmus ein Sampling-Algorithmus ist.
    
    Args:
        filepath: Pfad zur Best Response Datei
    
    Returns:
        True wenn Sampling-Algorithmus, sonst False
    """
    sampling_keywords = ['chance_sampling', 'external_sampling', 'mccfr']
    return any(keyword in filepath for keyword in sampling_keywords)


def find_best_response_files(game, iterations, include_sampling=False):
    """
    Findet automatisch alle Best Response Dateien für ein Spiel und eine Iterationsanzahl.
    
    Args:
        game: Name des Spiels (z.B. 'leduc')
        iterations: Anzahl der Iterationen (z.B. 1000)
        include_sampling: Wenn True, werden auch Sampling-Algorithmen inkludiert
    
    Returns:
        Liste von Dateipfaden
    """
    base_dir = 'data/models'
    game_dir = os.path.join(base_dir, game)
    
    if not os.path.exists(game_dir):
        print(f"Warnung: Verzeichnis nicht gefunden: {game_dir}")
        return []
    
    found_files = []
    iterations_str = str(iterations)
    
    # Suche in allen Unterverzeichnissen nach passenden Dateien
    for root, dirs, files in os.walk(game_dir):
        # Prüfe ob das Verzeichnis genau die gewünschte Iterationsanzahl enthält
        # Verwende os.path.basename um nur den Verzeichnisnamen zu prüfen
        dir_name = os.path.basename(root)
        if dir_name == iterations_str:
            for file in files:
                if file.endswith('_best_response.pkl.gz'):
                    # Prüfe ob der Dateiname genau die Iterationsanzahl enthält
                    # z.B. "leduc_1000_best_response.pkl.gz" sollte matchen, aber nicht "leduc_10000"
                    if f"_{iterations_str}_" in file or file.endswith(f"_{iterations_str}_best_response.pkl.gz"):
                        filepath = os.path.join(root, file)
                        # Filtere Sampling-Algorithmen wenn nicht gewünscht
                        if include_sampling or not is_sampling_algorithm(filepath):
                            found_files.append(filepath)
    
    return sorted(found_files)


def main():
    parser = argparse.ArgumentParser(
        description='Erstelle Plots aus Best Response pkl.gz Dateien',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Standard: Erstellt BEIDE Plots automatisch
  # 1. Nicht-Sampling-Algorithmen gegen Iterationen
  # 2. Alle Algorithmen gegen Zeit
  uv run python src/plot_best_response.py --game leduc --iterations 1000

  # Nur gegen Iterationen plotten (nicht-Sampling)
  uv run python src/plot_best_response.py --game leduc --iterations 1000 --iterations-only

  # Nur gegen Zeit plotten (alle Algorithmen)
  uv run python src/plot_best_response.py --game leduc --iterations 1000 --time-only

  # Nur Sampling-Algorithmen gegen Zeit plotten
  uv run python src/plot_best_response.py --game leduc --iterations 1000 --sampling

  # Mit manuellen Dateien (Legacy-Modus)
  uv run python src/plot_best_response.py file1.pkl.gz file2.pkl.gz

  # Manuelle Dateien gegen Zeit plotten
  uv run python src/plot_best_response.py --time file1.pkl.gz file2.pkl.gz

  # Ohne log-log
  uv run python src/plot_best_response.py --game leduc --iterations 1000 --no-log-log
        """
    )
    
    parser.add_argument('files', nargs='*', 
                       help='Pfade zu Best Response pkl.gz Dateien (optional, wenn --game und --iterations verwendet werden)')
    parser.add_argument('--game', type=str, default=None,
                       help='Name des Spiels (z.B. leduc, kuhn_case1)')
    parser.add_argument('--iterations', type=int, default=None,
                       help='Anzahl der Iterationen (z.B. 1000)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output-Pfad für den Plot (optional, wird automatisch generiert wenn --game und --iterations verwendet werden)')
    parser.add_argument('--title', type=str, default=None,
                       help='Titel für den Plot')
    parser.add_argument('--no-log-log', action='store_true',
                       help='Deaktiviere log-log Skalierung (nur X-Achse logarithmisch)')
    parser.add_argument('--sampling', action='store_true',
                       help='Plotte nur Sampling-Algorithmen gegen Zeit (überschreibt Standard-Verhalten)')
    parser.add_argument('--iterations-only', action='store_true',
                       help='Plotte nur gegen Iterationen, nicht gegen Zeit (überschreibt Standard-Verhalten)')
    parser.add_argument('--time-only', action='store_true',
                       help='Plotte nur gegen Zeit, nicht gegen Iterationen (überschreibt Standard-Verhalten)')
    parser.add_argument('--time', action='store_true',
                       help='Bei manuellen Dateien: Plotte gegen Zeit statt gegen Iterationen')
    
    args = parser.parse_args()
    
    # Bestimme welche Dateien verwendet werden sollen
    if args.game and args.iterations:
        output_dir = 'data/plots/comparisons'
        os.makedirs(output_dir, exist_ok=True)
        
        # Standard-Verhalten: Beide Plots erstellen
        # 1. Nicht-Sampling gegen Iterationen
        # 2. Alle gegen Zeit
        
        if args.sampling:
            # Nur Sampling-Algorithmen gegen Zeit
            print(f"Suche nach Sampling-Algorithmen für {args.game} mit {args.iterations} Iterationen...")
            valid_files = find_best_response_files(args.game, args.iterations, include_sampling=True)
            valid_files = [f for f in valid_files if is_sampling_algorithm(f)]
            
            if not valid_files:
                print(f"Keine Sampling-Algorithmen gefunden für {args.game} mit {args.iterations} Iterationen")
                return
            
            print(f"Gefundene Dateien ({len(valid_files)}):")
            for f in valid_files:
                print(f"  - {f}")
            
            output_path = args.output or os.path.join(output_dir, f"{args.game}_{args.iterations}_sampling_time_comparison.png")
            title = args.title or f"Exploitability vs Time Comparison - Sampling Only ({args.game}, {args.iterations} iterations)"
            
            plot_multiple_best_responses_by_time(
                valid_files,
                output_path=output_path,
                title=title,
                log_log=not args.no_log_log
            )
            
        elif args.iterations_only:
            # Nur nicht-Sampling gegen Iterationen
            print(f"Suche nach nicht-Sampling-Algorithmen für {args.game} mit {args.iterations} Iterationen...")
            valid_files = find_best_response_files(args.game, args.iterations, include_sampling=False)
            
            if not valid_files:
                print(f"Keine Dateien gefunden für {args.game} mit {args.iterations} Iterationen")
                return
            
            print(f"Gefundene Dateien ({len(valid_files)}):")
            for f in valid_files:
                print(f"  - {f}")
            
            output_path = args.output or os.path.join(output_dir, f"{args.game}_{args.iterations}_comparison.png")
            title = args.title or f"Exploitability Comparison ({args.game}, {args.iterations} iterations)"
            
            plot_multiple_best_responses(
                valid_files,
                output_path=output_path,
                title=title,
                log_log=not args.no_log_log
            )
            
        elif args.time_only:
            # Nur alle gegen Zeit
            print(f"Suche nach allen Algorithmen für {args.game} mit {args.iterations} Iterationen (Zeit-Plot)...")
            valid_files = find_best_response_files(args.game, args.iterations, include_sampling=True)
            
            if not valid_files:
                print(f"Keine Dateien gefunden für {args.game} mit {args.iterations} Iterationen")
                return
            
            print(f"Gefundene Dateien ({len(valid_files)}):")
            for f in valid_files:
                print(f"  - {f}")
            
            output_path = args.output or os.path.join(output_dir, f"{args.game}_{args.iterations}_time_comparison.png")
            title = args.title or f"Exploitability vs Time Comparison ({args.game}, {args.iterations} iterations)"
            
            plot_multiple_best_responses_by_time(
                valid_files,
                output_path=output_path,
                title=title,
                log_log=not args.no_log_log
            )
        else:
            # Standard-Verhalten: Beide Plots erstellen
            # 1. Nicht-Sampling gegen Iterationen
            print(f"Suche nach nicht-Sampling-Algorithmen für {args.game} mit {args.iterations} Iterationen...")
            non_sampling_files = find_best_response_files(args.game, args.iterations, include_sampling=False)
            
            if non_sampling_files:
                print(f"Gefundene nicht-Sampling-Dateien ({len(non_sampling_files)}):")
                for f in non_sampling_files:
                    print(f"  - {f}")
                
                output_path = args.output or os.path.join(output_dir, f"{args.game}_{args.iterations}_comparison.png")
                title = args.title or f"Exploitability Comparison ({args.game}, {args.iterations} iterations)"
                
                plot_multiple_best_responses(
                    non_sampling_files,
                    output_path=output_path,
                    title=title,
                    log_log=not args.no_log_log
                )
            else:
                print(f"Keine nicht-Sampling-Algorithmen gefunden für {args.game} mit {args.iterations} Iterationen")
            
            # 2. Alle gegen Zeit
            print(f"\nSuche nach allen Algorithmen für {args.game} mit {args.iterations} Iterationen (Zeit-Plot)...")
            all_files = find_best_response_files(args.game, args.iterations, include_sampling=True)
            
            if all_files:
                print(f"Gefundene Dateien für Zeit-Plot ({len(all_files)}):")
                for f in all_files:
                    print(f"  - {f}")
                
                output_path_time = os.path.join(output_dir, f"{args.game}_{args.iterations}_time_comparison.png")
                title_time = f"Exploitability vs Time Comparison ({args.game}, {args.iterations} iterations)"
                
                plot_multiple_best_responses_by_time(
                    all_files,
                    output_path=output_path_time,
                    title=title_time,
                    log_log=not args.no_log_log
                )
            else:
                print(f"Keine Dateien gefunden für Zeit-Plot")
            
    elif args.files:
        # Legacy-Modus: Verwende übergebene Dateien
        valid_files = []
        for filepath in args.files:
            if os.path.exists(filepath):
                valid_files.append(filepath)
            else:
                print(f"Warnung: Datei nicht gefunden: {filepath}")
        
        if not valid_files:
            print("Fehler: Keine gültigen Dateien gefunden")
            return
        
        output_path = args.output or 'comparison.png'
        title = args.title or "Exploitability Comparison"
        
        if args.time:
            # Manueller Zeit-Vergleich
            if not output_path.endswith('.png'):
                output_path = output_path.replace('.png', '_time.png')
            if output_path == 'comparison.png':
                output_path = 'comparison_time.png'
            title = args.title or "Exploitability vs Time Comparison"
            
            plot_manual_time_comparison(
                valid_files,
                output_path=output_path,
                title=title,
                log_log=not args.no_log_log
            )
        else:
            # Standard: Plot gegen Iterationen
            plot_multiple_best_responses(
                valid_files,
                output_path=output_path,
                title=title,
                log_log=not args.no_log_log
            )
    else:
        print("Fehler: Bitte entweder --game und --iterations angeben oder Dateien als Argumente übergeben")
        parser.print_help()
        return


if __name__ == "__main__":
    main()
