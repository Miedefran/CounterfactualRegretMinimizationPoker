from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from gui.components.poker_table import PokerTable
from gui.components.player_widget import PlayerWidget
from gui.components.pot_display import PotDisplay
from gui.components.visual_card import VisualCard
from gui.components.action_buttons import ActionButtons
from gui.components.chip import Chip

class BasePokerGUI(QMainWindow):
    
    def __init__(self, game, parent=None):
        super().__init__(parent)
        self.game = game
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
        
        self.setup_ui()
        QTimer.singleShot(100, lambda: (self.reset_game(0), self.position_components()))
    
    def setup_ui(self):
        self.player0_widget = PlayerWidget(0, "Player 0", self)
        player0_container = QWidget()
        player0_layout = QHBoxLayout(player0_container)
        player0_layout.setContentsMargins(0, 0, 0, 0)
        player0_layout.addStretch()
        player0_layout.addWidget(self.player0_widget)
        player0_layout.addStretch()
        
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
        
        self.player1_widget = PlayerWidget(1, "Player 1", self)
        player1_container = QWidget()
        player1_layout = QHBoxLayout(player1_container)
        player1_layout.setContentsMargins(0, 0, 0, 0)
        player1_layout.addStretch()
        player1_layout.addWidget(self.player1_widget)
        player1_layout.addStretch()
        
        self.action_buttons = ActionButtons()
        self.action_buttons.action_selected.connect(self.handle_action)
        
        self.main_layout.addWidget(player0_container)
        self.main_layout.addWidget(table_container, 3)
        self.main_layout.addWidget(player1_container)
        self.main_layout.addWidget(self.action_buttons)
        
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
        cards_height = 120
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
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(50, self.position_components)
    
    def update_display(self):
        state = self.game.get_state(self.game.current_player)
        self.update_cards(state)
        self.update_pot(state)
        self.update_actions(state)
        self.update_players(state)
        QTimer.singleShot(50, self.position_components)
    
    def get_private_cards(self, state, player_id):
        if player_id == self.game.current_player:
            hand = state.get('hand')
            if isinstance(hand, list):
                return hand
            elif hand:
                return [hand]
            return []
        else:
            opponent_state = self.game.get_state(1 - self.game.current_player)
            hand = opponent_state.get('hand')
            if isinstance(hand, list):
                return hand
            elif hand:
                return [hand]
            return []
    
    def get_public_cards(self, state):
        public_cards = state.get('public_cards', [])
        if public_cards:
            return public_cards
        public_card = state.get('public_card')
        if public_card:
            return [public_card]
        return []
    
    def get_player_bets(self, state):
        player_bets = state.get('player_bets', [])
        if player_bets:
            return player_bets
        total_bets = state.get('total_bets', [])
        if total_bets:
            return total_bets
        return [0, 0]
    
    def update_cards(self, state):
        player0_cards = self.get_private_cards(state, 0)
        player1_cards = self.get_private_cards(state, 1)
        
        self.player0_widget.set_cards(player0_cards, reveal=True)
        self.player1_widget.set_cards(player1_cards, reveal=False)
        
        public_cards = self.get_public_cards(state)
        
        for card_widget in self.community_cards:
            self.community_cards_layout.removeWidget(card_widget)
            card_widget.deleteLater()
        self.community_cards = []
        
        for card in public_cards:
            if card:
                card_widget = VisualCard(card)
                self.community_cards_layout.addWidget(card_widget)
                self.community_cards.append(card_widget)
    
    def update_pot(self, state):
        pot = state.get('pot', 0)
        self.pot_display.set_pot(pot)
    
    def update_actions(self, state):
        legal_actions = state.get('legal_actions', [])
        self.action_buttons.update_legal_actions(legal_actions)
    
    def update_players(self, state):
        current_player = state.get('current_player', 0)
        player_bets = self.get_player_bets(state)
        
        self.player0_widget.set_current_player(current_player == 0)
        self.player0_widget.set_current_bet(player_bets[0] if len(player_bets) > 0 else 0)
        
        self.player1_widget.set_current_player(current_player == 1)
        self.player1_widget.set_current_bet(player_bets[1] if len(player_bets) > 1 else 0)
    
    def handle_action(self, action, bet_size):
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
        
        if hasattr(self.game, 'deal_private_cards'):
            self.game.deal_private_cards()
        else:
            for player in self.game.players:
                if hasattr(player, 'set_private_card'):
                    if hasattr(self.game.dealer, 'deal_card') and len(self.game.dealer.deck) > 0:
                        card = self.game.dealer.deal_card()
                        player.set_private_card(card)
        
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
