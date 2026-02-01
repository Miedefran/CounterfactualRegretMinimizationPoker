import sys
import os
import argparse
import gzip
import pickle as pkl
from pathlib import Path

# project_root ist das Verzeichnis, das src/ enthält
# __file__ ist src/gui/runner/run_agent_vs_human.py
# Also: .parent = src/gui/runner/, .parent.parent = src/gui/, .parent.parent.parent = src/, .parent.parent.parent.parent = root
project_root = Path(__file__).parent.parent.parent.parent
# sys.path muss auf src/ zeigen, nicht auf root
sys.path.insert(0, str(project_root / 'src'))

from PyQt6.QtWidgets import QApplication
from envs.kuhn_poker.game import KuhnPokerGame
from envs.leduc_holdem.game import LeducHoldemGame
from gui.agent_vs_human import AgentVsHumanGUI


def get_exploitability_from_br_file(br_file_path):
    """
    Extrahiert die finale Exploitability aus einer Best Response Datei.
    
    Returns:
        float: Finale Exploitability (mb/g) oder None falls nicht verfügbar
    """
    try:
        with gzip.open(br_file_path, 'rb') as f:
            data = pkl.load(f)

        values = data.get('values', [])
        if not values:
            return None

        # Neues Format: (iteration, exploitability_mb, br_value_p0, br_value_p1, cumulative_training_time)
        # Nimm den letzten Wert (höchste Iteration = finale Exploitability)
        last_value = values[-1]

        # Format: (iteration, exploitability_mb, ...)
        if len(last_value) >= 2:
            return float(last_value[1])  # exploitability_mb

        return None
    except Exception as e:
        return None


def find_best_strategy_by_exploitability(project_root, game_name):
    """
    Findet die Strategie-Datei mit der niedrigsten Exploitability für ein Spiel.
    
    Args:
        project_root: Root-Verzeichnis des Projekts (Path-Objekt oder str)
        game_name: Name des Spiels (z.B. 'kuhn_case2', 'leduc')
    
    Returns:
        str: Pfad zur Strategie-Datei mit niedrigster Exploitability, oder None falls keine gefunden
    """
    print(f"[DEBUG find_best_strategy] Called with game_name={game_name}, project_root={project_root}")

    # Konvertiere project_root zu Path falls nötig
    if isinstance(project_root, str):
        project_root = Path(project_root)

    models_dir = project_root / 'data' / 'models'
    print(f"[DEBUG find_best_strategy] models_dir={models_dir}, exists={models_dir.exists()}")

    if not models_dir.exists():
        print(f"[DEBUG find_best_strategy] models_dir does not exist, returning None")
        return None

    strategy_candidates = []

    # Suche rekursiv nach allen Strategie-Dateien für dieses Spiel
    for root, dirs, files in os.walk(str(models_dir)):
        for file in files:
            # Strategie-Dateien enden mit .pkl.gz, aber nicht mit _best_response.pkl.gz
            if file.endswith('.pkl.gz') and not file.endswith('_best_response.pkl.gz'):
                # Prüfe ob der Dateiname oder Pfad zum Spiel passt
                file_lower = file.lower()
                path_lower = str(root).lower()

                # Für Kuhn: suche nach "kuhn" im Pfad/Dateinamen
                # Für Leduc: suche nach "leduc" im Pfad/Dateinamen
                if game_name.startswith('kuhn'):
                    if 'kuhn' in file_lower or 'kuhn' in path_lower:
                        strategy_path = Path(root) / file
                        strategy_candidates.append(strategy_path)
                elif 'leduc' in game_name.lower():
                    if 'leduc' in file_lower or 'leduc' in path_lower:
                        strategy_path = Path(root) / file
                        strategy_candidates.append(strategy_path)

    if not strategy_candidates:
        print(f"[DEBUG] No strategy candidates found for {game_name}")
        return None

    print(f"[DEBUG] Found {len(strategy_candidates)} strategy candidates for {game_name}")

    # Für jede Strategie-Datei: suche nach zugehöriger Best Response Datei
    best_strategy = None
    best_exploitability = float('inf')
    strategies_with_br = 0

    for strategy_path in strategy_candidates:
        br_path = Path(str(strategy_path).replace('.pkl.gz', '_best_response.pkl.gz'))

        exploitability = None
        if br_path.exists():
            exploitability = get_exploitability_from_br_file(br_path)
            if exploitability is not None:
                strategies_with_br += 1

        if exploitability is not None:
            if exploitability < best_exploitability:
                best_exploitability = exploitability
                best_strategy = strategy_path
        elif best_strategy is None:
            # Falls keine Best Response Datei existiert, nimm die erste gefundene Strategie als Fallback
            best_strategy = strategy_path

    if best_strategy:
        print(f"Found best strategy: {best_strategy}")
        if best_exploitability != float('inf'):
            print(
                f"  Exploitability: {best_exploitability:.6f} mb/g (from {strategies_with_br} strategies with BR data)")
        else:
            print(f"  (No BR data found, using first available strategy)")
        return str(best_strategy)

    return None


def find_leduc_strategy(project_root):
    """Sucht nach einer Leduc-Strategy-Datei, bevorzugt die mit niedrigster Exploitability."""
    # Versuche zuerst die beste Strategie nach Exploitability zu finden
    best = find_best_strategy_by_exploitability(project_root, 'leduc')
    if best:
        return best

    # Fallback: Alte Logik
    possible_paths = [
        project_root / 'data' / 'models' / 'leduc' / 'cfr' / 'leduc_1000.pkl.gz',
        project_root / 'data' / 'models' / 'leduc' / 'leduc_1000.pkl.gz',
        project_root / 'data' / 'models' / 'leduc.pkl.gz',
        project_root / 'leduc_1000.pkl.gz',
    ]

    for path in possible_paths:
        if path.exists():
            return str(path)

    # Rekursive Suche nach leduc-Dateien
    for root, dirs, files in os.walk(project_root):
        for file in files:
            if 'leduc' in file.lower() and file.endswith('.pkl.gz') and not file.endswith('_best_response.pkl.gz'):
                found_path = Path(root) / file
                return str(found_path)

    return None


def main():
    parser = argparse.ArgumentParser(description='Run Agent vs Human GUI')
    parser.add_argument('--game', type=str, default='leduc',
                        choices=['kuhn', 'leduc'],
                        help='Game type (default: leduc)')
    parser.add_argument('--strategy', type=str, default=None,
                        help='Path to strategy file (.pkl.gz). If not provided, uses default model.')

    args = parser.parse_args()

    app = QApplication(sys.argv)

    strategy_file = args.strategy

    if args.game == 'kuhn':
        game = KuhnPokerGame()
        if not strategy_file:
            # Versuche zuerst die beste Strategie nach Exploitability zu finden
            best_strategy = find_best_strategy_by_exploitability(project_root, 'kuhn_case2')
            if best_strategy:
                strategy_file = best_strategy
                print(f"Using best Kuhn strategy (lowest exploitability): {strategy_file}")
            else:
                # Fallback: Alte Logik
                default_strategy = project_root / 'data' / 'models' / 'kuhn' / 'case2' / 'cfr' / 'kuhn_case2_100000.pkl.gz'
                if default_strategy.exists():
                    strategy_file = str(default_strategy)
                    print(f"Using default Kuhn strategy: {strategy_file}")
    elif args.game == 'leduc':
        game = LeducHoldemGame()
        if not strategy_file:
            # Standard: Discounted CFR mit Flat Tree (suche nach allen Varianten)
            # Suche rekursiv nach discounted_cfr_with_flat_tree Modellen
            models_dir = project_root / 'data' / 'models' / 'leduc'
            strategy_files = []

            if models_dir.exists():
                for root, dirs, files in os.walk(models_dir):
                    # Prüfe ob der Pfad "discounted_cfr_with_flat_tree" enthält
                    if 'discounted_cfr_with_flat_tree' in str(root):
                        for file in files:
                            if file.endswith('.pkl.gz') and not file.endswith('_best_response.pkl.gz'):
                                strategy_path = Path(root) / file
                                # Extrahiere Iterationszahl aus dem Pfad
                                try:
                                    iterations = int(Path(root).name)
                                    strategy_files.append((iterations, strategy_path))
                                except ValueError:
                                    pass

            if strategy_files:
                # Sortiere nach Iterationen (höhere Iterationen bevorzugt)
                strategy_files.sort(key=lambda x: x[0], reverse=True)
                strategy_file = str(strategy_files[0][1])
                print(
                    f"Using Discounted CFR with Flat Tree strategy ({strategy_files[0][0]} iterations): {strategy_file}")

            if not strategy_file:
                # Versuche zuerst die beste Strategie nach Exploitability zu finden
                best_strategy = find_best_strategy_by_exploitability(project_root, 'leduc')
                if best_strategy:
                    strategy_file = best_strategy
                    print(f"Using best Leduc strategy (lowest exploitability): {strategy_file}")
                else:
                    # Fallback: Alte Logik
                    found_strategy = find_leduc_strategy(project_root)
                    if found_strategy:
                        strategy_file = found_strategy
                        print(f"Using found Leduc strategy: {found_strategy}")
                    else:
                        print("Warning: No Leduc strategy file found. Agent will not be available.")
                        print(
                            "Please provide a strategy file with --strategy or place a .pkl.gz file with 'leduc' in the name.")

    if strategy_file and not os.path.exists(strategy_file):
        print(f"Warning: Strategy file not found: {strategy_file}")
        strategy_file = None

    window = AgentVsHumanGUI(
        game=game,
        strategy_file=strategy_file,
        human_name="Friedemann"
    )
    window.showMaximized()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
