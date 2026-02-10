import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication
from envs.leduc_holdem.game import LeducHoldemGame
from gui.agent_vs_agent import AgentVsAgentGUI


def test_gui_init():
    print("Creating QApplication...")
    app = QApplication(sys.argv)

    print("Creating game instance...")
    game = LeducHoldemGame()

    print("Creating GUI window...")
    window = AgentVsAgentGUI(game)

    print("GUI initialized successfully!")
    print("Window title:", window.windowTitle())
    print("Window size:", window.size().width(), "x", window.size().height())

    print("\n✅ All tests passed! GUI is ready to use.")
    print("To start the GUI, run: python gui/run_agent_vs_agent.py")

    return True


if __name__ == "__main__":
    try:
        test_gui_init()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
