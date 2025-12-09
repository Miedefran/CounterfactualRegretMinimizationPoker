from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QSlider, QGroupBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QPalette, QColor
import gzip
import pickle as pkl
from agents.strategy_agent import StrategyAgent
from training.cfr_solver import CFRSolver
from .base_gui import BasePokerGUI
from .components.card_display import CardDisplay
import random

class AgentVsAgentGUI(BasePokerGUI):
    
    def __init__(self, game, strategy_file_0=None, strategy_file_1=None, parent=None):
        super().__init__(game, parent)
        self.strategy_file_0 = strategy_file_0
        self.strategy_file_1 = strategy_file_1
        self.agent_0 = None
        self.agent_1 = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.auto_step)
        self.is_playing = False
        self.game_speed = 1.0
        self.current_turn = 0
        
        if strategy_file_0:
            strategy_0 = self.load_strategy(strategy_file_0)
            self.agent_0 = StrategyAgent(strategy_0, 0)
        if strategy_file_1:
            strategy_1 = self.load_strategy(strategy_file_1)
            self.agent_1 = StrategyAgent(strategy_1, 1)
        
        self.setup_ui()
        self.reset_game()
    
    def load_strategy(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            data = pkl.load(f)
        
        if 'average_strategy' in data:
            return data['average_strategy']
        else:
            strategy_sum = data['strategy_sum']
            return CFRSolver.average_from_strategy_sum(strategy_sum)
    
    def setup_ui(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QLabel {
                color: white;
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
            QPushButton:pressed {
                background-color: #3a3a3a;
            }
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #666;
            }
        """)
        
        top_bar = QHBoxLayout()
        title = QLabel("CFR Poker Visualizer / Agent vs Agent")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        top_bar.addWidget(title)
        top_bar.addStretch()
        self.main_layout.addLayout(top_bar)
        
        game_area = QHBoxLayout()
        
        left_panel = QVBoxLayout()
        self.player_0_label = QLabel("Player 0")
        self.player_0_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        left_panel.addWidget(self.player_0_label)
        
        self.player_0_cards = CardDisplay()
        left_panel.addWidget(self.player_0_cards)
        
        self.player_0_bet_label = QLabel("Bet: 0")
        left_panel.addWidget(self.player_0_bet_label)
        left_panel.addStretch()
        
        center_area = QVBoxLayout()
        center_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.table_label = QLabel("POKER TABLE")
        self.table_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table_label.setStyleSheet("""
            QLabel {
                background-color: #0d5d2f;
                border: 3px solid #1a7a3e;
                border-radius: 200px;
                padding: 100px;
                font-size: 24px;
                font-weight: bold;
                color: white;
                min-width: 400px;
                min-height: 300px;
            }
        """)
        center_area.addWidget(self.table_label)
        
        self.public_cards_display = CardDisplay()
        center_area.addWidget(self.public_cards_display)
        
        self.pot_label = QLabel("Pot: 0")
        self.pot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pot_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        center_area.addWidget(self.pot_label)
        
        right_panel = QVBoxLayout()
        self.player_1_label = QLabel("Player 1")
        self.player_1_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        right_panel.addWidget(self.player_1_label)
        
        self.player_1_cards = CardDisplay()
        right_panel.addWidget(self.player_1_cards)
        
        self.player_1_bet_label = QLabel("Bet: 0")
        right_panel.addWidget(self.player_1_bet_label)
        right_panel.addStretch()
        
        game_area.addLayout(left_panel, 1)
        game_area.addLayout(center_area, 2)
        game_area.addLayout(right_panel, 1)
        
        self.main_layout.addLayout(game_area)
        
        controls_layout = QHBoxLayout()
        
        self.rewind_start_btn = QPushButton("⏮")
        self.rewind_start_btn.clicked.connect(self.rewind_to_start)
        controls_layout.addWidget(self.rewind_start_btn)
        
        self.prev_step_btn = QPushButton("⏪")
        self.prev_step_btn.clicked.connect(self.step_backward)
        controls_layout.addWidget(self.prev_step_btn)
        
        self.play_pause_btn = QPushButton("⏸ PAUSE")
        self.play_pause_btn.setStyleSheet("background-color: #d32f2f;")
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        controls_layout.addWidget(self.play_pause_btn)
        
        self.next_step_btn = QPushButton("⏩")
        self.next_step_btn.clicked.connect(self.step_forward)
        controls_layout.addWidget(self.next_step_btn)
        
        self.fast_forward_btn = QPushButton("⏭")
        self.fast_forward_btn.clicked.connect(self.fast_forward_to_end)
        controls_layout.addWidget(self.fast_forward_btn)
        
        self.turn_label = QLabel("Turn: 0")
        controls_layout.addWidget(self.turn_label)
        
        controls_layout.addStretch()
        
        speed_label = QLabel("Game Speed:")
        controls_layout.addWidget(speed_label)
        
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(0)
        self.speed_slider.setMaximum(6)
        self.speed_slider.setValue(3)
        self.speed_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.speed_slider.setTickInterval(1)
        self.speed_slider.valueChanged.connect(self.update_speed)
        controls_layout.addWidget(self.speed_slider)
        
        self.speed_label = QLabel("x1")
        controls_layout.addWidget(self.speed_label)
        
        self.main_layout.addLayout(controls_layout)
        
        self.current_player_label = QLabel("Current Player: 0")
        self.current_player_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.current_player_label)
    
    def update_speed(self, value):
        speeds = [0.125, 0.25, 0.5, 1, 2, 4, 8]
        self.game_speed = speeds[value]
        self.speed_label.setText(f"x{self.game_speed}")
        if self.is_playing:
            delay = int(1000 / self.game_speed)
            self.timer.setInterval(delay)
    
    def toggle_play_pause(self):
        if self.is_playing:
            self.pause()
        else:
            self.play()
    
    def play(self):
        if self.game.done:
            return
        
        self.is_playing = True
        self.play_pause_btn.setText("⏸ PAUSE")
        self.play_pause_btn.setStyleSheet("background-color: #d32f2f;")
        delay = int(1000 / self.game_speed)
        self.timer.start(delay)
    
    def pause(self):
        self.is_playing = False
        self.play_pause_btn.setText("▶ PLAY")
        self.play_pause_btn.setStyleSheet("background-color: #388e3c;")
        self.timer.stop()
    
    def auto_step(self):
        if self.game.done:
            self.pause()
            return
        
        self.step_forward()
    
    def step_forward(self):
        if self.game.done:
            return False
        
        current_player = self.game.current_player
        state = self.game.get_state(current_player)
        legal_actions = state.get('legal_actions', [])
        
        if not legal_actions:
            return False
        
        if current_player == 0 and self.agent_0:
            action = self.agent_0.get_action(state)
        elif current_player == 1 and self.agent_1:
            action = self.agent_1.get_action(state)
        else:
            import random
            action = random.choice(legal_actions)
        
        result = self.game.step(action)
        self.current_turn += 1
        self.update_display()
        
        if result:
            self.pause()
            self.show_game_result(result)
        
        return True
    
    def step_backward(self):
        if not hasattr(self.game, 'state_stack') or not self.game.state_stack:
            return False
        
        self.game.step_back()
        self.current_turn = max(0, self.current_turn - 1)
        self.update_display()
        return True
    
    def rewind_to_start(self):
        while self.step_backward():
            pass
    
    def fast_forward_to_end(self):
        self.pause()
        while not self.game.done:
            if not self.step_forward():
                break
    
    def reset_game(self, starting_player=None):
        if starting_player is None:
            starting_player = random.randint(0, 1)
        
        super().reset_game(starting_player)
        
        if hasattr(self.game, 'dealer'):
            if hasattr(self.game.players[0], 'set_private_cards'):
                self.game.players[0].set_private_cards(
                    self.game.dealer.deal_card(), 
                    self.game.dealer.deal_card()
                )
            else:
                self.game.players[0].private_card = self.game.dealer.deal_card()
            
            if hasattr(self.game.players[1], 'set_private_cards'):
                self.game.players[1].set_private_cards(
                    self.game.dealer.deal_card(),
                    self.game.dealer.deal_card()
                )
            else:
                self.game.players[1].private_card = self.game.dealer.deal_card()
        
        self.current_turn = 0
        self.pause()
        self.update_display()
    
    def update_display(self):
        state = self.game.get_state(self.game.current_player)
        
        self.update_cards()
        self.update_pot()
        self.update_current_player()
        self.turn_label.setText(f"Turn: {self.current_turn}")
    
    def update_cards(self):
        if hasattr(self.game.players[0], 'private_cards'):
            cards_0 = self.game.players[0].private_cards
        else:
            cards_0 = [self.game.players[0].private_card] if self.game.players[0].private_card else []
        self.player_0_cards.set_cards(cards_0)
        
        if hasattr(self.game.players[1], 'private_cards'):
            cards_1 = self.game.players[1].private_cards
        else:
            cards_1 = [self.game.players[1].private_card] if self.game.players[1].private_card else []
        self.player_1_cards.set_cards(cards_1)
        
        if hasattr(self.game, 'public_cards'):
            self.public_cards_display.set_cards(self.game.public_cards)
        elif hasattr(self.game, 'public_card') and self.game.public_card:
            self.public_cards_display.set_cards([self.game.public_card])
        else:
            self.public_cards_display.set_cards([])
    
    def update_pot(self):
        pot = getattr(self.game, 'pot', 0)
        self.pot_label.setText(f"Pot: {pot}")
        
        if hasattr(self.game, 'total_bets'):
            bets = self.game.total_bets
        else:
            bets = [0, 0]
        
        self.player_0_bet_label.setText(f"Bet: {bets[0]}")
        self.player_1_bet_label.setText(f"Bet: {bets[1]}")
    
    def update_current_player(self):
        current = self.game.current_player
        self.current_player_label.setText(f"Current Player: {current}")
        
        if current == 0:
            self.player_0_label.setStyleSheet("color: #4caf50; font-weight: bold;")
            self.player_1_label.setStyleSheet("color: white;")
        else:
            self.player_0_label.setStyleSheet("color: white;")
            self.player_1_label.setStyleSheet("color: #4caf50; font-weight: bold;")
    
    def update_actions(self):
        pass
    
    def show_game_result(self, result):
        winner = 0 if result[0] > result[1] else 1
        self.pot_label.setText(f"Game Over! Player {winner} wins! Results: {result}")

