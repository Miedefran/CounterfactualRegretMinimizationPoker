from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

class BasePokerGUI(QMainWindow):
    
    def __init__(self, game, parent=None):
        super().__init__(parent)
        self.game = game
        self.setWindowTitle("CFR Poker Visualizer")
        self.setMinimumSize(1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)
        
        self.setup_ui()
    
    def setup_ui(self):
        pass
    
    def update_display(self):
        state = self.game.get_state(self.game.current_player)
        self.update_cards()
        self.update_pot()
        self.update_actions()
    
    def update_cards(self):
        pass
    
    def update_pot(self):
        pass
    
    def update_actions(self):
        pass
    
    def step_forward(self):
        if self.game.done:
            return False
        
        current_state = self.game.get_state(self.game.current_player)
        legal_actions = current_state.get('legal_actions', [])
        
        if not legal_actions:
            return False
        
        return True
    
    def step_backward(self):
        if not hasattr(self.game, 'state_stack') or not self.game.state_stack:
            return False
        
        self.game.step_back()
        self.update_display()
        return True
    
    def reset_game(self, starting_player=0):
        self.game.reset(starting_player)
        if hasattr(self.game, 'dealer'):
            self.game.dealer.reset()
            self.game.dealer.shuffle()
        self.update_display()
    
    def format_card(self, card):
        if card is None:
            return "??"
        if isinstance(card, str):
            return card
        if isinstance(card, list):
            return ", ".join(card)
        return str(card)
    
    def get_info_set_key_from_state(self, state):
        hand = state.get('hand')
        history = tuple(state.get('history', []))
        current_player = state.get('current_player', 0)
        
        if isinstance(hand, list):
            hand = tuple(sorted(hand))
        
        return (hand, history, current_player)

