import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication
from gui.base_gui import BasePokerGUI

class TestGame:
    def __init__(self):
        self.current_player = 0
        self.done = False
        self.pot = 100
        self.players = [TestPlayer(0), TestPlayer(1)]
        self.history = []
        self.state_stack = []
    
    def get_state(self, player_id):
        return {
            'hand': self.players[player_id].private_cards,
            'public_cards': ['As', 'Kh', 'Qd', 'Jc', 'Ts'],
            'history': list(self.history),
            'current_player': self.current_player,
            'done': self.done,
            'pot': self.pot,
            'total_bets': [50, 50],
            'legal_actions': ['check', 'bet']
        }
    
    def reset(self, starting_player):
        self.current_player = starting_player
        self.done = False
        self.pot = 100
        self.history = []
        self.state_stack = []
        self.players[0].private_cards = ['Ac', 'Kc']
        self.players[1].private_cards = ['Qh', 'Jh']

class TestPlayer:
    def __init__(self, player_id):
        self.player_id = player_id
        self.private_cards = []

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    game = TestGame()
    window = BasePokerGUI(game)
    window.show()
    
    sys.exit(app.exec())
