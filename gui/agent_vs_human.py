from PyQt6.QtCore import QTimer
import gzip
import pickle as pkl
import random
from gui.layouts.agent_vs_human_layout import AgentVsHumanLayout
from agents.strategy_agent import StrategyAgent
from training.cfr_solver import CFRSolver
from gui.components.visual_card import VisualCard
from gui.audio.sound_manager import SoundManager

class AgentVsHumanGUI(AgentVsHumanLayout):
    
    def __init__(self, game, strategy_file=None, human_name="Friedemann", parent=None):
        super().__init__(parent)
        self.game = game
        self.strategy_file = strategy_file
        self.human_player_id = None
        self.agent_player_id = None
        self.agent = None
        
        for card_widget in self.community_cards:
            self.community_cards_layout.removeWidget(card_widget)
            card_widget.deleteLater()
        self.community_cards = []
        
        self.player_bottom_widget.set_cards([], reveal=False)
        self.player_top_widget.set_cards([], reveal=False)
        
        self.player_bottom_widget.set_player_name(human_name)
        self.player_top_widget.set_player_name("Strategy Agent")
        
        self.sound_manager = SoundManager()
        
        if strategy_file:
            strategy = self.load_strategy(strategy_file)
            QTimer.singleShot(200, lambda: self.initialize_agent(strategy))
        
        QTimer.singleShot(300, lambda: self.setup_connections())
        QTimer.singleShot(400, lambda: self.reset_game(0))
        
        QTimer.singleShot(500, lambda: self.setup_restart_button())
    
    def load_strategy(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            data = pkl.load(f)
        
        if 'average_strategy' in data:
            return data['average_strategy']
        else:
            strategy_sum = data['strategy_sum']
            return CFRSolver.average_from_strategy_sum(strategy_sum)
    
    def initialize_agent(self, strategy):
        if self.human_player_id is not None:
            self.agent_player_id = 1 - self.human_player_id
            self.agent = StrategyAgent(strategy, self.agent_player_id, self.game)
    
    def setup_connections(self):
        if hasattr(self, 'action_buttons'):
            self.action_buttons.action_selected.connect(self.handle_action)
    
    def setup_restart_button(self):
        if hasattr(self, 'restart_button'):
            self.restart_button.clicked.connect(self.restart_hand)
    
    def reset_game(self, starting_player=0):
        self.game.reset(starting_player)
        
        if hasattr(self.game, 'dealer'):
            self.game.dealer.reset()
            self.game.dealer.shuffle()
        
        self.human_player_id = starting_player
        self.agent_player_id = 1 - self.human_player_id
        
        for player in self.game.players:
            if hasattr(player, 'set_private_cards'):
                if hasattr(self.game.dealer, 'deal_card') and len(self.game.dealer.deck) >= 2:
                    card1 = self.game.dealer.deal_card()
                    card2 = self.game.dealer.deal_card()
                    player.set_private_cards(card1, card2)
            elif hasattr(player, 'set_private_card'):
                if hasattr(self.game.dealer, 'deal_card') and len(self.game.dealer.deck) > 0:
                    card = self.game.dealer.deal_card()
                    player.set_private_card(card)
        
        if self.agent is None and self.strategy_file:
            strategy = self.load_strategy(self.strategy_file)
            self.agent = StrategyAgent(strategy, self.agent_player_id, self.game)
        elif self.agent:
            self.agent.player_id = self.agent_player_id
            self.agent.game = self.game
        
        self.update_display()
    
    def update_display(self):
        if self.game.done:
            self.update_cards(reveal_all=True)
            self.update_actions_final()
        else:
            state = self.game.get_state(self.game.current_player)
            self.update_cards()
            self.update_pot(state)
            self.update_players(state)
            self.update_actions(state)
            self.update_history(state)
        
        QTimer.singleShot(50, self.position_components)
    
    def get_private_cards(self, player_id):
        player = self.game.players[player_id]
        if hasattr(player, 'private_cards'):
            return list(player.private_cards) if player.private_cards else []
        elif hasattr(player, 'private_card') and player.private_card:
            return [player.private_card]
        return []
    
    def get_public_cards(self):
        if hasattr(self.game, 'public_cards'):
            return list(self.game.public_cards) if self.game.public_cards else []
        elif hasattr(self.game, 'public_card') and self.game.public_card:
            return [self.game.public_card]
        return []
    
    def get_player_bets(self):
        if hasattr(self.game, 'total_bets'):
            return list(self.game.total_bets)
        elif hasattr(self.game, 'player_bets'):
            return list(self.game.player_bets)
        return [0, 0]
    
    def update_cards(self, reveal_all=False):
        human_cards = self.get_private_cards(self.human_player_id)
        agent_cards = self.get_private_cards(self.agent_player_id)
        
        if reveal_all:
            self.player_bottom_widget.set_cards(human_cards, reveal=True)
            self.player_top_widget.set_cards(agent_cards, reveal=True)
        else:
            self.player_bottom_widget.set_cards(human_cards, reveal=True)
            self.player_top_widget.set_cards(agent_cards, reveal=False)
        
        public_cards = self.get_public_cards()
        
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
        pot = state.get('pot', getattr(self.game, 'pot', 0))
        self.pot_display.set_pot(pot)
    
    def update_players(self, state):
        current_player = state.get('current_player', self.game.current_player)
        player_bets = self.get_player_bets()
        
        is_human_turn = (current_player == self.human_player_id)
        is_agent_turn = (current_player == self.agent_player_id)
        
        self.player_bottom_widget.set_current_player(is_human_turn)
        self.player_bottom_widget.set_current_bet(player_bets[self.human_player_id] if len(player_bets) > self.human_player_id else 0)
        
        self.player_top_widget.set_current_player(is_agent_turn)
        self.player_top_widget.set_current_bet(player_bets[self.agent_player_id] if len(player_bets) > self.agent_player_id else 0)
    
    def update_actions(self, state):
        if self.game.done:
            self.action_buttons.disable_all()
            return
        
        current_player = state.get('current_player', self.game.current_player)
        
        if current_player == self.human_player_id:
            legal_actions = state.get('legal_actions', self.game.get_legal_actions())
            self.action_buttons.update_legal_actions(legal_actions)
            
            if hasattr(self.game, 'round') and hasattr(self.game.round, 'current_bet_size'):
                call_amount = self.game.round.current_bet_size
                if call_amount > 0:
                    self.action_buttons.set_amount_to_call(call_amount)
        else:
            self.action_buttons.disable_all()
    
    def update_actions_final(self):
        self.action_buttons.disable_all()
    
    def update_history(self, state):
        history = state.get('history', getattr(self.game, 'history', []))
        player_names = {
            self.human_player_id: self.player_bottom_widget.player_name,
            self.agent_player_id: self.player_top_widget.player_name
        }
        self.history_view.set_history(history, player_names=player_names, starting_player=self.human_player_id)
    
    def handle_action(self, action, bet_size):
        if self.game.done:
            return
        
        if self.game.current_player != self.human_player_id:
            return
        
        state = self.game.get_state(self.human_player_id)
        legal_actions = state.get('legal_actions', self.game.get_legal_actions())
        
        if action not in legal_actions:
            return
        
        self.sound_manager.play_action(action)
        
        result = self.game.step(action)
        self.update_display()
        
        if result:
            self.show_game_result(result)
        elif self.game.current_player == self.agent_player_id:
            delay = random.randint(2000, 4000)
            QTimer.singleShot(delay, self.agent_step)
    
    def agent_step(self):
        if self.game.done:
            return
        
        if self.game.current_player != self.agent_player_id:
            return
        
        if not self.agent:
            return
        
        state = self.game.get_state(self.agent_player_id)
        action = self.agent.get_action(state)
        
        self.sound_manager.play_action(action)
        
        result = self.game.step(action)
        self.update_display()
        
        if result:
            self.show_game_result(result)
        elif self.game.current_player == self.human_player_id:
            pass
    
    def show_game_result(self, result):
        if not result:
            return
        
        winner_id = 0 if result[0] > result[1] else 1
        pot_amount = getattr(self.game, 'pot', 0)
        
        if winner_id == self.human_player_id:
            winner_name = self.player_bottom_widget.player_name
        else:
            winner_name = self.player_top_widget.player_name
        
        self.history_view.add_game_result(winner_id, pot_amount)
        
        self.update_cards(reveal_all=True)
    
    def restart_hand(self):
        import random
        starting_player = random.randint(0, 1)
        self.reset_game(starting_player)

