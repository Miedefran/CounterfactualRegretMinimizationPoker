from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt, QTimer
from gui.components.poker_table import PokerTable
from gui.components.player_widget import PlayerWidget
from gui.components.pot_display import PotDisplay
from gui.components.visual_card import VisualCard
from gui.components.chip import Chip
from gui.components.history_view import HistoryView

class BasePokerLayout(QMainWindow):
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CFR Poker Visualizer")
        self.setMinimumSize(1400, 900)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0a1929;
            }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        
        self.community_cards = []
        self.pot_chips = []
        
        self.setup_restart_button()
        self.setup_ui()
        self.add_test_cards()
        QTimer.singleShot(100, self.position_components)
    
    def setup_restart_button(self):
        self.restart_button = QPushButton("🔄 New Hand")
        self.restart_button.setFixedSize(120, 40)
        self.restart_button.setStyleSheet("""
            QPushButton {
                background-color: #1a5490;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2a6ab0;
            }
            QPushButton:pressed {
                background-color: #0a3a70;
            }
        """)
        self.restart_button.setParent(self)
        self.restart_button.raise_()
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'restart_button'):
            button_x = self.width() - self.restart_button.width() - 20
            button_y = 20
            self.restart_button.move(button_x, button_y)
        QTimer.singleShot(50, self.position_components)
    
    def setup_ui(self):
        self.player_top_widget = PlayerWidget(0, "Strategy Agent", self)
        player_top_container = QWidget()
        player_top_layout = QHBoxLayout(player_top_container)
        player_top_layout.setContentsMargins(0, 0, 0, 0)
        player_top_layout.addStretch()
        player_top_layout.addWidget(self.player_top_widget)
        player_top_layout.addStretch()
        
        table_container = QWidget()
        table_container.setStyleSheet("background-color: #0a1929;")
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(10, 10, 10, 10)
        table_layout.setSpacing(0)
        
        self.poker_table = PokerTable()
        self.poker_table.setParent(table_container)
        table_layout.addWidget(self.poker_table)
        
        self.community_cards_widget = QWidget(self.poker_table)
        self.community_cards_widget.setStyleSheet("""
            QWidget {
                background-color: #0d5d2f;
            }
        """)
        self.community_cards_layout = QHBoxLayout(self.community_cards_widget)
        self.community_cards_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.community_cards_layout.setSpacing(10)
        
        self.pot_display = PotDisplay(self.poker_table)
        
        self.pot_chip1 = Chip(5, parent=self.poker_table)
        self.pot_chip2 = Chip(25, parent=self.poker_table)
        self.pot_chip3 = Chip(100, parent=self.poker_table)
        self.pot_chips = [self.pot_chip1, self.pot_chip2, self.pot_chip3]
        
        self.player_bottom_widget = PlayerWidget(1, "Friedemann", self)
        player_bottom_container = QWidget()
        player_bottom_layout = QHBoxLayout(player_bottom_container)
        player_bottom_layout.setContentsMargins(0, 0, 0, 0)
        player_bottom_layout.addStretch()
        player_bottom_layout.addWidget(self.player_bottom_widget)
        player_bottom_layout.addStretch()
        
        self.control_area = QWidget()
        self.control_area.setMinimumHeight(80)
        
        history_toggle_btn = QPushButton("History ▼")
        history_toggle_btn.setMaximumWidth(100)
        history_toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #1a5490;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 3px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #2a6ab0;
            }
        """)
        
        self.history_view = HistoryView()
        self.history_view.setMaximumWidth(250)
        self.history_view.setMinimumWidth(200)
        self.history_view.setVisible(False)
        
        history_container = QWidget()
        history_layout = QVBoxLayout(history_container)
        history_layout.setContentsMargins(0, 0, 0, 0)
        history_layout.setSpacing(5)
        history_layout.addWidget(history_toggle_btn)
        history_layout.addWidget(self.history_view)
        
        self.history_toggle_btn = history_toggle_btn
        self.history_toggle_btn.clicked.connect(self.toggle_history)
        
        game_area = QWidget()
        game_area_layout = QHBoxLayout(game_area)
        game_area_layout.setContentsMargins(0, 0, 0, 0)
        game_area_layout.setSpacing(10)
        
        left_area = QWidget()
        left_area_layout = QVBoxLayout(left_area)
        left_area_layout.setContentsMargins(0, 0, 0, 0)
        left_area_layout.addWidget(player_top_container)
        left_area_layout.addWidget(table_container, 3)
        left_area_layout.addWidget(player_bottom_container)
        left_area_layout.addWidget(self.control_area)
        
        game_area_layout.addWidget(left_area, 4)
        game_area_layout.addWidget(history_container, 0)
        
        self.main_layout.addWidget(game_area)
        
        self.poker_table.show()
        self.community_cards_widget.show()
        self.pot_display.show()
        for chip in self.pot_chips:
            chip.show()
    
    def position_components(self):
        table_rect = self.poker_table.rect()
        if table_rect.width() == 0 or table_rect.height() == 0:
            QTimer.singleShot(100, self.position_components)
            return
        
        cards_count = len(self.community_cards)
        cards_width = max(cards_count * 90, 200) if cards_count > 0 else 200
        cards_height = 140
        cards_x = (table_rect.width() - cards_width) // 2
        cards_y = table_rect.height() // 2 - cards_height // 2
        
        self.community_cards_widget.setGeometry(
            cards_x, cards_y, cards_width, cards_height
        )
        
        pot_width = self.pot_display.width()
        pot_height = self.pot_display.height()
        pot_x = 30
        pot_y = cards_y + (cards_height - pot_height) // 2
        
        self.pot_display.move(pot_x, pot_y)
        
        import random
        random.seed(42)
        
        chip_size = 50
        chip_area_width = 180
        chip_area_height = 100
        chip_area_x = pot_x + pot_width + 15
        chip_area_y = pot_y + (pot_height - chip_area_height) // 2
        
        placed_chips = []
        min_distance = chip_size + 5
        
        for i, chip in enumerate(self.pot_chips):
            max_attempts = 200
            placed = False
            for attempt in range(max_attempts):
                offset_x = random.randint(0, chip_area_width - chip_size)
                offset_y = random.randint(0, chip_area_height - chip_size)
                chip_x = int(chip_area_x + offset_x)
                chip_y = int(chip_area_y + offset_y)
                
                overlaps = False
                for placed_x, placed_y in placed_chips:
                    distance = ((chip_x - placed_x) ** 2 + (chip_y - placed_y) ** 2) ** 0.5
                    if distance < min_distance:
                        overlaps = True
                        break
                
                if not overlaps:
                    chip.move(chip_x, chip_y)
                    placed_chips.append((chip_x, chip_y))
                    placed = True
                    break
            
            if not placed:
                chip_x = int(chip_area_x + (i * (chip_size + 10)))
                chip_y = int(chip_area_y + (i % 2) * (chip_size + 10))
                chip.move(chip_x, chip_y)
                placed_chips.append((chip_x, chip_y))
    
    def toggle_history(self):
        self.history_view.setVisible(not self.history_view.isVisible())
        if self.history_view.isVisible():
            self.history_toggle_btn.setText("History ▲")
        else:
            self.history_toggle_btn.setText("History ▼")
    
    def add_test_cards(self):
        self.player_top_widget.set_cards(['As', 'Kh'], reveal=True)
        self.player_bottom_widget.set_cards(['Qd', 'Jc'], reveal=True)
        
        test_board_cards = ['Ac', 'Kd', 'Qh', 'Js', 'Ts']
        for card in test_board_cards:
            card_widget = VisualCard(card)
            self.community_cards_layout.addWidget(card_widget)
            self.community_cards.append(card_widget)
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'restart_button'):
            button_x = self.width() - self.restart_button.width() - 20
            button_y = 20
            self.restart_button.move(button_x, button_y)
        QTimer.singleShot(50, self.position_components)
