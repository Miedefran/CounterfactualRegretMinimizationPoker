import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

class VisualCard(QLabel):
    
    def __init__(self, card, parent=None):
        super().__init__(parent)
        self.setFixedSize(80, 120)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: white;
                border: 2px solid #333;
                border-radius: 8px;
            }
        """)
        self.set_card(card)
    
    def set_card(self, card):
        if card is None:
            self.setText("??")
            self.setStyleSheet("""
                QLabel {
                    background-color: #1a5490;
                    border: 2px solid #333;
                    border-radius: 8px;
                    color: white;
                    font-size: 24px;
                    font-weight: bold;
                }
            """)
            return
        
        rank, suit, color = self.parse_card(card)
        
        html = f"""
        <div style="
            font-family: Arial;
            font-weight: bold;
            color: {color};
            padding: 5px;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        ">
            <div style="font-size: 20px; text-align: left;">{rank}</div>
            <div style="font-size: 32px; text-align: center; margin: 10px 0;">{suit}</div>
            <div style="font-size: 20px; text-align: right; transform: rotate(180deg);">{rank}</div>
        </div>
        """
        
        self.setText(html)
        self.setStyleSheet(f"""
            QLabel {{
                background-color: white;
                border: 2px solid #333;
                border-radius: 8px;
            }}
        """)
    
    def parse_card(self, card):
        if isinstance(card, str):
            if len(card) == 1:
                return (card, "?", "#000")
            
            rank = card[0]
            suit_char = card[1] if len(card) > 1 else "?"
            
            suit_symbols = {
                's': '♠',
                'h': '♥',
                'd': '♦',
                'c': '♣'
            }
            
            suit_colors = {
                's': '#000000',
                'c': '#000000',
                'h': '#dc143c',
                'd': '#dc143c'
            }
            
            suit = suit_symbols.get(suit_char, suit_char)
            color = suit_colors.get(suit_char, '#000000')
            
            return (rank, suit, color)
        
        return (str(card), "?", "#000")

class TestCardWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Card Visual Test")
        self.setMinimumSize(600, 400)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        title = QLabel("Visual Card Test")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(title)
        
        cards_layout = QHBoxLayout()
        cards_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        test_cards = ['Ks', 'Kh', 'Qd', 'Jc', 'As']
        for card in test_cards:
            card_widget = VisualCard(card)
            cards_layout.addWidget(card_widget)
        
        layout.addLayout(cards_layout)
        
        hidden_layout = QHBoxLayout()
        hidden_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hidden_card = VisualCard(None)
        hidden_layout.addWidget(hidden_card)
        layout.addLayout(hidden_layout)
        
        layout.addStretch()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TestCardWindow()
    window.show()
    sys.exit(app.exec())

