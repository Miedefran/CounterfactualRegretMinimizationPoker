import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication
from envs.leduc_holdem.game import LeducHoldemGame
from gui.agent_vs_agent import AgentVsAgentGUI

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    game = LeducHoldemGame()
    
    strategy_file_0 = None
    strategy_file_1 = None
    
    if len(sys.argv) > 1:
        strategy_file_0 = sys.argv[1]
    if len(sys.argv) > 2:
        strategy_file_1 = sys.argv[2]
    
    window = AgentVsAgentGUI(game, strategy_file_0, strategy_file_1)
    window.show()
    
    sys.exit(app.exec())

