import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel
from envs.leduc_holdem.game import LeducHoldemGame
from gui.base_gui import BasePokerGUI
import random

class TestBaseGUI(BasePokerGUI):
    
    def setup_ui(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QLabel {
                color: white;
                font-size: 14px;
            }
            QPushButton {
                background-color: #4a4a4a;
                color: white;
                border: 1px solid #666;
                border-radius: 4px;
                padding: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        title = QLabel("Base GUI Test - Leduc Hold'em")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)
        
        self.state_label = QLabel("State will be shown here")
        layout.addWidget(self.state_label)
        
        self.pot_label = QLabel("Pot: 0")
        layout.addWidget(self.pot_label)
        
        self.current_player_label = QLabel("Current Player: 0")
        layout.addWidget(self.current_player_label)
        
        button_layout = QVBoxLayout()
        
        step_forward_btn = QPushButton("Step Forward")
        step_forward_btn.clicked.connect(self.step_forward)
        button_layout.addWidget(step_forward_btn)
        
        step_backward_btn = QPushButton("Step Backward")
        step_backward_btn.clicked.connect(self.step_backward)
        button_layout.addWidget(step_backward_btn)
        
        reset_btn = QPushButton("Reset Game")
        reset_btn.clicked.connect(lambda: self.reset_game(random.randint(0, 1)))
        button_layout.addWidget(reset_btn)
        
        layout.addLayout(button_layout)
        layout.addStretch()
        
        self.update_display()
    
    def step_forward(self):
        if self.game.done:
            self.state_label.setText("Game is done!")
            return
        
        current_state = self.game.get_state(self.game.current_player)
        legal_actions = current_state.get('legal_actions', [])
        
        if not legal_actions:
            self.state_label.setText("No legal actions available")
            return
        
        import random
        action = random.choice(legal_actions)
        result = self.game.step(action)
        
        self.update_display()
        
        if result:
            self.state_label.setText(f"Game Over! Result: {result}")
    
    def step_backward(self):
        if super().step_backward():
            self.update_display()
        else:
            self.state_label.setText("Cannot step backward - no saved states")
    
    def update_display(self):
        state = self.game.get_state(self.game.current_player)
        
        state_text = f"""
Game State:
- Hand: {self.format_card(state.get('hand', 'None'))}
- History: {state.get('history', [])}
- Legal Actions: {state.get('legal_actions', [])}
- Betting Round: {state.get('betting_round', 0)}
- Done: {self.game.done}
        """
        self.state_label.setText(state_text.strip())
        
        pot = getattr(self.game, 'pot', 0)
        self.pot_label.setText(f"Pot: {pot}")
        
        current = self.game.current_player
        self.current_player_label.setText(f"Current Player: {current}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    game = LeducHoldemGame()
    window = TestBaseGUI(game)
    window.show()
    
    sys.exit(app.exec())

