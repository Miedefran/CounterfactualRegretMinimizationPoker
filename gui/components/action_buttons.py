from PyQt6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QSpinBox, QLabel
from PyQt6.QtCore import Qt, pyqtSignal

class ActionButtons(QWidget):
    
    action_selected = pyqtSignal(str, int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.layout.setSpacing(10)
        
        self.check_btn = QPushButton("Check")
        self.bet_btn = QPushButton("Bet")
        self.call_btn = QPushButton("Call")
        self.fold_btn = QPushButton("Fold")
        
        self.bet_size_input = QSpinBox()
        self.bet_size_input.setMinimum(1)
        self.bet_size_input.setMaximum(10000)
        self.bet_size_input.setValue(10)
        self.bet_size_input.setEnabled(False)
        
        self.bet_size_label = QLabel("Bet Size:")
        self.bet_size_label.setVisible(False)
        
        self.setup_buttons()
        self.setup_connections()
        self.setup_style()
    
    def setup_buttons(self):
        buttons = [self.check_btn, self.bet_btn, self.call_btn, self.fold_btn]
        for btn in buttons:
            btn.setMinimumSize(100, 50)
            self.layout.addWidget(btn)
        
        self.layout.addStretch()
        self.layout.addWidget(self.bet_size_label)
        self.layout.addWidget(self.bet_size_input)
    
    def setup_connections(self):
        self.check_btn.clicked.connect(lambda: self.emit_action('check', 0))
        self.bet_btn.clicked.connect(lambda: self.emit_action('bet', self.bet_size_input.value()))
        self.call_btn.clicked.connect(lambda: self.emit_action('call', 0))
        self.fold_btn.clicked.connect(lambda: self.emit_action('fold', 0))
    
    def setup_style(self):
        self.setStyleSheet("""
            QPushButton {
                background-color: #1a5490;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover:enabled {
                background-color: #2a6ab0;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #999;
            }
            QPushButton:pressed {
                background-color: #0a3a70;
            }
            QSpinBox {
                background-color: white;
                border: 2px solid #333;
                border-radius: 5px;
                padding: 5px;
                font-size: 14px;
                min-width: 80px;
            }
            QLabel {
                color: white;
                font-size: 14px;
            }
        """)
        
        self.fold_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc143c;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover:enabled {
                background-color: #ff1744;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #999;
            }
            QPushButton:pressed {
                background-color: #b0120a;
            }
        """)
    
    def emit_action(self, action, bet_size):
        self.action_selected.emit(action, bet_size)
    
    def update_legal_actions(self, legal_actions, bet_size_range=None):
        self.check_btn.setEnabled('check' in legal_actions)
        self.bet_btn.setEnabled('bet' in legal_actions)
        self.call_btn.setEnabled('call' in legal_actions)
        self.fold_btn.setEnabled('fold' in legal_actions)
        
        if 'bet' in legal_actions and bet_size_range:
            min_bet, max_bet = bet_size_range
            self.bet_size_input.setMinimum(min_bet)
            self.bet_size_input.setMaximum(max_bet)
            self.bet_size_input.setValue(min_bet)
            self.bet_size_input.setEnabled(True)
            self.bet_size_label.setVisible(True)
        else:
            self.bet_size_input.setEnabled(False)
            self.bet_size_label.setVisible(False)
    
    def set_amount_to_call(self, amount):
        if amount > 0:
            self.call_btn.setText(f"Call ({amount})")
        else:
            self.call_btn.setText("Call")
    
    def disable_all(self):
        self.check_btn.setEnabled(False)
        self.bet_btn.setEnabled(False)
        self.call_btn.setEnabled(False)
        self.fold_btn.setEnabled(False)

