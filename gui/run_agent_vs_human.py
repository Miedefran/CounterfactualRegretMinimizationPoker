import sys
import os
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication
from envs.kuhn_poker.game import KuhnPokerGame
from envs.leduc_holdem.game import LeducHoldemGame
from gui.agent_vs_human import AgentVsHumanGUI

def main():
    parser = argparse.ArgumentParser(description='Run Agent vs Human GUI')
    parser.add_argument('--game', type=str, default='kuhn',
                       choices=['kuhn', 'leduc'],
                       help='Game type (default: kuhn)')
    parser.add_argument('--strategy', type=str, default=None,
                       help='Path to strategy file (.pkl.gz). If not provided, uses default model.')
    
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    
    strategy_file = args.strategy
    
    if args.game == 'kuhn':
        game = KuhnPokerGame(ante=1, bet_size=1)
        if not strategy_file:
            default_strategy = project_root / 'models' / 'kuhn' / 'case2' / 'cfr' / 'kuhn_case2_100000.pkl.gz'
            if default_strategy.exists():
                strategy_file = str(default_strategy)
                print(f"Using default Kuhn strategy: {strategy_file}")
    elif args.game == 'leduc':
        game = LeducHoldemGame(ante=1, bet_sizes=[2, 4], bet_limit=2)
        if not strategy_file:
            default_strategy = project_root / 'models' / 'leduc' / 'cfr' / 'leduc_1000.pkl.gz'
            if default_strategy.exists():
                strategy_file = str(default_strategy)
                print(f"Using default Leduc strategy: {strategy_file}")
    
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

