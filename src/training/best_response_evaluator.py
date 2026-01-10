"""
Best Response Evaluator für CFR Training.

Diese Datei stellt Funktionen bereit, um während des Trainings
Best Response Werte zu berechnen und zu speichern.
"""

import os
import gzip
import pickle
import time
import json
import math


def get_big_blind_equivalent(game_name):
    from utils.poker_utils import GAME_CONFIGS
    
    if game_name not in GAME_CONFIGS:
        if game_name.startswith('kuhn'):
            base_name = 'kuhn_case2'
        else:
            return 1.0
    else:
        base_name = game_name
    
    config = GAME_CONFIGS[base_name]
    
    if 'big_blind' in config:
        return config['big_blind']
    elif 'ante' in config:
        return config['ante']
    else:
        return 1.0


def to_milliblinds_per_game(value, game_name):
    big_blind_equiv = get_big_blind_equivalent(game_name)
    return (value / big_blind_equiv) * 1000.0


def get_public_state_tree_path(game_name):
    """
    Ermittelt den Pfad zum Public State Tree für ein gegebenes Spiel.
    
    Args:
        game_name: Name des Spiels (z.B. 'leduc', 'kuhn_case1', etc.)
    
    Returns:
        Pfad zur Public State Tree Datei
    """
    # Normalisiere game_name für den Dateinamen
    # Für kuhn_caseX verwenden wir 'kuhn'
    if game_name.startswith('kuhn'):
        save_name = 'kuhn'
    else:
        save_name = game_name
    
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    base_dir = os.path.join(script_dir, 'data', 'trees', 'public_state_trees')
    filename = f"{save_name}_public_tree_v2.pkl.gz"
    path = os.path.join(base_dir, filename)
    
    return path


def evaluate_best_response(game_name, tree, average_strategy, iteration):
    """
    Berechnet den Best Response Wert für beide Spieler.
    
    Args:
        game_name: Name des Spiels
        tree: Public State Tree (bereits geladen)
        average_strategy: Dictionary mit der aktuellen Average Strategy
        iteration: Aktuelle Iterationsnummer
    
    Returns:
        Tuple (br_value_p0, br_value_p1, elapsed_time) oder None bei Fehler
    """
    try:
        start_time = time.time()
        
        # Importiere Best Response Funktionen
        from evaluation.best_response_agent_v2 import compute_best_response_value
        
        # Berechne Best Response für beide Spieler
        br_value_p0 = compute_best_response_value(game_name, 0, tree, average_strategy)
        br_value_p1 = compute_best_response_value(game_name, 1, tree, average_strategy)
        
        elapsed_time = time.time() - start_time
        
        # Exploitability = (BR Wert P0 + BR Wert P1) / 2
        exploitability = (br_value_p0 + br_value_p1) / 2.0
        
        # Umrechnung in milliblinds per game
        br_value_p0_mb = to_milliblinds_per_game(br_value_p0, game_name)
        br_value_p1_mb = to_milliblinds_per_game(br_value_p1, game_name)
        exploitability_mb = to_milliblinds_per_game(exploitability, game_name)
        
        print(f"  Best Response @ Iteration {iteration}: P0={br_value_p0_mb:.2f} mb/g, P1={br_value_p1_mb:.2f} mb/g, Exploitability={exploitability_mb:.2f} mb/g (took {elapsed_time:.2f}s)")
        
        return (br_value_p0, br_value_p1, elapsed_time)
        
    except Exception as e:
        print(f"WARNING: Fehler bei Best Response Evaluation: {e}")
        return None


def load_schedule_config(config_path=None):
    """
    Lädt eine Schedule-Konfiguration aus einer JSON-Datei.
    
    Args:
        config_path: Pfad zur JSON-Datei oder Name eines vordefinierten Schedules
    
    Returns:
        Dictionary mit Schedule-Konfiguration oder None
    """
    if config_path is None:
        return None
    
    if config_path.endswith('.json'):
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Schedule config nicht gefunden: {config_path}")
    else:
        default_config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'config',
            'br_eval_schedules.json'
        )
        if os.path.exists(default_config_path):
            with open(default_config_path, 'r') as f:
                all_schedules = json.load(f)
                if config_path in all_schedules:
                    return all_schedules[config_path]
                else:
                    raise ValueError(f"Schedule '{config_path}' nicht in {default_config_path} gefunden")
        else:
            raise FileNotFoundError(f"Default schedule config nicht gefunden: {default_config_path}")


class BestResponseTracker:
    """
    Verwaltet Best Response Werte während des Trainings.
    Der Public State Tree wird einmal beim Initialisieren geladen und dann wiederverwendet.
    """
    
    def __init__(self, game_name, schedule_config=None):
        """
        Args:
            game_name: Name des Spiels
            schedule_config: Schedule-Konfiguration (Dict, JSON-Pfad, oder Schedule-Name)
                           Falls None oder Integer, wird fester Intervall verwendet (Rückwärtskompatibilität)
        """
        self.game_name = game_name
        self.values = []  # Liste von (iteration, exploitability_mb, br_value_p0, br_value_p1)
        self.total_br_time = 0.0  # Kumulierte Zeit für Best Response Evaluierungen
        self.last_eval_iteration = 0
        
        if schedule_config is None:
            schedule_config = {"type": "fixed", "interval": 100}
        elif isinstance(schedule_config, int):
            schedule_config = {"type": "fixed", "interval": schedule_config}
        elif isinstance(schedule_config, str):
            schedule_config = load_schedule_config(schedule_config)
        
        self.schedule_config = schedule_config
        self.schedule_type = schedule_config.get("type", "fixed")
        
        if self.schedule_type == "fixed":
            self.interval = schedule_config.get("interval", 100)
        elif self.schedule_type == "logarithmic":
            self.base_interval = schedule_config.get("base_interval", 10)
            self.target_iteration = schedule_config.get("target_iteration", 100)
            self.target_interval = schedule_config.get("target_interval", 100)
            # unbounded: Wenn True, wächst das Intervall auch nach target_iteration weiter
            # target_iteration/target_interval definieren den Wachstumsverlauf (wie schnell es wächst)
            self.unbounded = schedule_config.get("unbounded", False)
        elif self.schedule_type == "custom":
            self.custom_schedule = schedule_config.get("schedule", [])
            if not self.custom_schedule:
                raise ValueError("Custom schedule benötigt 'schedule' Liste")
        else:
            raise ValueError(f"Unbekannter Schedule-Typ: {self.schedule_type}")
        
        # Lade Public State Tree einmal beim Initialisieren
        self.tree = None
        self._load_tree()
    
    def _load_tree(self):
        """Lädt den Public State Tree einmal beim Initialisieren."""
        try:
            from evaluation.best_response_agent_v2 import load_public_tree
            
            tree_path = get_public_state_tree_path(self.game_name)
            if not os.path.exists(tree_path):
                print(f"WARNING: Public State Tree nicht gefunden: {tree_path}")
                print("Best Response Evaluation wird deaktiviert")
                return
            
            self.tree = load_public_tree(tree_path)
            print(f"Public State Tree geladen: {tree_path}")
            
        except Exception as e:
            print(f"WARNING: Fehler beim Laden des Public State Trees: {e}")
            print("Best Response Evaluation wird deaktiviert")
    
    def get_current_interval(self, iteration):
        """
        Berechnet das aktuelle Evaluierungsintervall basierend auf der Iteration.
        
        Args:
            iteration: Aktuelle Iterationsnummer
        
        Returns:
            Intervall für diese Iteration
        """
        if self.schedule_type == "fixed":
            return self.interval
        
        elif self.schedule_type == "logarithmic":
            if iteration <= self.base_interval:
                return self.base_interval
            
            log_factor = math.log(iteration / self.base_interval + 1)
            target_log = math.log(self.target_iteration / self.base_interval + 1)
            
            if target_log == 0:
                return self.base_interval
            
            interval = self.base_interval + (self.target_interval - self.base_interval) * (log_factor / target_log)
            
            # Wenn unbounded: Keine obere Begrenzung, Intervall wächst kontinuierlich weiter
            if self.unbounded:
                return max(self.base_interval, int(interval))
            else:
                return max(self.base_interval, min(int(interval), self.target_interval))
        
        elif self.schedule_type == "custom":
            for i, (threshold, interval) in enumerate(self.custom_schedule):
                if i == len(self.custom_schedule) - 1 or iteration < self.custom_schedule[i + 1][0]:
                    return interval
            return self.custom_schedule[-1][1]
        
        return 100
    
    def should_evaluate(self, iteration):
        """
        Prüft, ob bei dieser Iteration evaluiert werden soll.
        
        Args:
            iteration: Aktuelle Iterationsnummer
        
        Returns:
            True wenn evaluiert werden soll, sonst False
        """
        if self.schedule_type == "fixed":
            return iteration % self.interval == 0
        
        if self.schedule_type == "custom":
            if iteration < self.custom_schedule[0][0]:
                return False
        
        current_interval = self.get_current_interval(iteration)
        return (iteration - self.last_eval_iteration) >= current_interval
    
    def add_value(self, iteration, exploitability_mb, br_value_p0, br_value_p1):
        """
        Fügt einen Exploitability Wert und BR-Werte hinzu.
        
        Args:
            iteration: Iterationsnummer
            exploitability_mb: Exploitability in milliblinds per game
            br_value_p0: Best Response Wert für Spieler 0 (roh)
            br_value_p1: Best Response Wert für Spieler 1 (roh)
        """
        self.values.append((iteration, exploitability_mb, br_value_p0, br_value_p1))
    
    def evaluate_and_add(self, average_strategy, iteration):
        """
        Berechnet Best Response Wert und fügt ihn hinzu.
        
        Args:
            average_strategy: Aktuelle Average Strategy
            iteration: Aktuelle Iterationsnummer
        
        Returns:
            Verbrauchte Zeit für die Evaluierung (in Sekunden)
        """
        if self.tree is None:
            # Tree konnte nicht geladen werden, überspringe Evaluation
            return 0.0
        
        result = evaluate_best_response(self.game_name, self.tree, average_strategy, iteration)
        if result is not None:
            br_value_p0, br_value_p1, elapsed_time = result
            self.total_br_time += elapsed_time
            
            # Exploitability = (BR Wert P0 + BR Wert P1) / 2
            exploitability = (br_value_p0 + br_value_p1) / 2.0
            exploitability_mb = to_milliblinds_per_game(exploitability, self.game_name)
            self.add_value(iteration, exploitability_mb, br_value_p0, br_value_p1)
            
            return elapsed_time
        
        return 0.0
    
    def get_total_br_time(self):
        """
        Gibt die kumulierte Zeit zurück, die für Best Response Evaluierungen verwendet wurde.
        
        Returns:
            Totale Zeit in Sekunden
        """
        return self.total_br_time
    
    def plot(self, output_path=None, log_scale=True):
        """
        Plottet die Exploitability über die Iterationen.
        
        Args:
            output_path: Optionaler Pfad zum Speichern des Plots
            log_scale: Wenn True, wird die X-Achse logarithmisch skaliert (default: True)
        """
        if not self.values:
            print("Keine Exploitability Werte zum Plotten vorhanden")
            return
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            iterations = [x[0] for x in self.values]
            exploitability = [x[1] for x in self.values]  # Nur Exploitability für Plot
            
            plt.figure(figsize=(12, 6))
            plt.plot(iterations, exploitability, label='Exploitability', marker='o', linewidth=2, markersize=6)
            
            if log_scale:
                plt.xscale('log')
                plt.xlabel('Iteration (log scale)')
            else:
                plt.xlabel('Iteration')
            
            plt.ylabel('Exploitability (mb/g)')
            plt.title(f'Exploitability During Training ({self.game_name})')
            plt.legend()
            plt.grid(True, alpha=0.3, which='both')  # 'both' für log und linear grid
            
            # Setze sinnvolle X-Achsen-Ticks für log scale
            if log_scale and iterations:
                max_iter = max(iterations)
                # Erstelle logarithmische Ticks
                if max_iter <= 100:
                    ticks = [1, 2, 5, 10, 20, 50, 100]
                elif max_iter <= 1000:
                    ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
                elif max_iter <= 10000:
                    ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
                else:
                    ticks = [1, 10, 100, 1000, 10000, 100000]
                
                # Filtere Ticks die größer als max_iter sind
                ticks = [t for t in ticks if t <= max_iter]
                plt.xticks(ticks, [str(t) for t in ticks])
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"Plot gespeichert: {output_path}")
            else:
                plt.show()
                
        except ImportError:
            print("WARNING: matplotlib nicht verfügbar, Plot wird übersprungen")
        except Exception as e:
            print(f"WARNING: Fehler beim Plotten: {e}")
    
    def save(self, filepath):
        """
        Speichert die Best Response Werte in eine Datei.
        
        Args:
            filepath: Pfad zur Datei
        """
        data = {
            'game_name': self.game_name,
            'schedule_config': self.schedule_config,
            'values': self.values
        }
        
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Best Response Werte gespeichert: {filepath}")
    
    def load(self, filepath):
        """
        Lädt Best Response Werte aus einer Datei.
        
        Args:
            filepath: Pfad zur Datei
        """
        with gzip.open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.game_name = data['game_name']
        loaded_values = data['values']
        
        if 'schedule_config' in data:
            self.schedule_config = data['schedule_config']
            self.schedule_type = self.schedule_config.get("type", "fixed")
            if self.schedule_type == "fixed":
                self.interval = self.schedule_config.get("interval", 100)
            elif self.schedule_type == "logarithmic":
                self.base_interval = self.schedule_config.get("base_interval", 10)
                self.target_iteration = self.schedule_config.get("target_iteration", 100)
                self.target_interval = self.schedule_config.get("target_interval", 100)
            elif self.schedule_type == "custom":
                self.custom_schedule = self.schedule_config.get("schedule", [])
        else:
            eval_interval = data.get('eval_interval', 100)
            self.schedule_config = {"type": "fixed", "interval": eval_interval}
            self.schedule_type = "fixed"
            self.interval = eval_interval
        
        # Rückwärtskompatibilität: Alte Dateien haben nur (iteration, exploitability)
        # Neue Dateien haben (iteration, exploitability_mb, br_value_p0, br_value_p1)
        if loaded_values and len(loaded_values[0]) == 2:
            # Alte Format: Konvertiere zu neuem Format (BR-Werte fehlen)
            self.values = [(it, exp, None, None) for it, exp in loaded_values]
            print(f"WARNING: Alte Datei-Format erkannt, BR-Werte fehlen")
        else:
            self.values = loaded_values
        
        print(f"Best Response Werte geladen: {filepath}")

