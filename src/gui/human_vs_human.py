from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QMessageBox
from gui.layouts.agent_vs_human_layout import AgentVsHumanLayout
from gui.server.http_client import HTTPClient
from gui.components.visual_card import VisualCard
from gui.audio.sound_manager import SoundManager
from gui.utils.hand_strength import hand_strength_text


class HumanVsHumanGUI(AgentVsHumanLayout):

    def __init__(self, server_url, human_name="Player", parent=None):
        super().__init__(parent)
        self.server_url = server_url
        self.human_name = human_name
        self.player_id = None
        self.opponent_id = None
        self.last_state = None
        self.last_history_length = 0
        self.last_player_bets = [0, 0]
        self.last_pot_value = 0
        self.game_result_shown = False
        self.last_reset_id = None

        for card_widget in self.community_cards:
            self.community_cards_layout.removeWidget(card_widget)
            card_widget.deleteLater()
        self.community_cards = []

        self.player_bottom_widget.set_cards([], reveal=False)
        hidden_cards = [None, None]
        self.player_top_widget.set_cards(hidden_cards, reveal=False)

        self.player_bottom_widget.set_player_name(human_name)
        self.player_top_widget.set_player_name("Opponent")

        self.sound_manager = SoundManager()

        self.client = HTTPClient(server_url, parent=self, player_name=human_name)
        self.client.state_update_received.connect(self._on_state_update)
        self.client.connection_error.connect(self._on_connection_error)

        QTimer.singleShot(200, lambda: self.setup_connections())
        QTimer.singleShot(300, lambda: self.connect_to_server())
        QTimer.singleShot(500, lambda: self.setup_restart_button())

    def setup_connections(self):
        if hasattr(self, 'action_buttons'):
            self.action_buttons.action_selected.connect(self.handle_action)

    def setup_restart_button(self):
        # Rufe die Basis-Methode auf, um den Button zu erstellen
        super().setup_restart_button()
        # Verbinde dann den Click-Handler
        if hasattr(self, 'restart_button'):
            self.restart_button.clicked.connect(self.restart_hand)

    def connect_to_server(self):
        if not self.client.connect():
            QMessageBox.critical(
                self,
                "Connection Error",
                f"Failed to connect to server at {self.server_url}\n\nPlease make sure the server is running."
            )
            return

        self.player_id = self.client.player_id
        self.opponent_id = 1 - self.player_id
        print(f"Connected to server, Player ID: {self.player_id}")

        QTimer.singleShot(1000, lambda: self._auto_reset_if_needed())

    def _auto_reset_if_needed(self):
        if self.player_id == 0:
            if self.last_state is None or len(self.last_state.get('history', [])) == 0:
                import random
                starting_player = random.randint(0, 1)
                result = self.client.send_reset_request(starting_player)
                if result:
                    print(f"Auto-reset sent by Player 0, starting player: {starting_player}")
                else:
                    print(f"Auto-reset failed (not all clients connected). Retrying in 1 second...")
                    QTimer.singleShot(1000, lambda: self._auto_reset_if_needed())

    def _on_state_update(self, state):
        # Clear history on BOTH clients when a new hand starts.
        reset_id = state.get("reset_id")
        if reset_id is not None and reset_id != self.last_reset_id:
            self.last_reset_id = reset_id
            self.history_view.clear()
            self.last_history_length = 0
            self.last_player_bets = [0, 0]
            self.last_pot_value = 0
            self.game_result_shown = False

        # Update displayed player names from server (so "Opponent" becomes real name).
        names_list = state.get("player_names") or []
        if self.player_id is not None and isinstance(names_list, list):
            if 0 <= self.player_id < len(names_list) and names_list[self.player_id]:
                self.player_bottom_widget.set_player_name(str(names_list[self.player_id]))
            if self.opponent_id is not None and 0 <= self.opponent_id < len(names_list) and names_list[
                self.opponent_id]:
                self.player_top_widget.set_player_name(str(names_list[self.opponent_id]))

        self._check_opponent_action(state)
        self.last_state = state
        self._update_from_server_state(state)

    def _check_opponent_action(self, state):
        if not self.last_state:
            return

        last_current_player = self.last_state.get('current_player', -1)
        current_player = state.get('current_player', -1)
        last_history = self.last_state.get('history', [])
        current_history = state.get('history', [])

        if len(current_history) > len(last_history):
            if last_current_player == self.opponent_id:
                last_action = current_history[-1] if current_history else None
                if last_action:
                    self.sound_manager.play_action(last_action)

    def _update_from_server_state(self, state):
        if state.get('done', False):
            self.update_cards_from_state(state, reveal_all=True)
            self.update_actions_final()
            # Make sure the final action (e.g. fold) is visible before showing the winner.
            self.update_history_from_state(state)
            self._check_and_show_game_result(state)
        else:
            self.update_cards_from_state(state)
            self.update_pot(state)
            self.update_players_from_state(state)
            self.update_actions_from_state(state)
            self.update_history_from_state(state)

        QTimer.singleShot(50, self.position_components)

    def update_cards_from_state(self, state, reveal_all=False):
        private_cards = state.get('private_cards', [])
        opponent_cards = state.get('opponent_cards', [])
        public_cards = state.get('public_cards', [])
        game_id = state.get('game')

        self.player_bottom_widget.set_cards(private_cards, reveal=True)
        if hasattr(self.player_bottom_widget, "set_hand_text"):
            self.player_bottom_widget.set_hand_text(hand_strength_text(private_cards, public_cards, game=game_id))

        if reveal_all and opponent_cards:
            self.player_top_widget.set_cards(opponent_cards, reveal=True)
            if hasattr(self.player_top_widget, "set_hand_text"):
                self.player_top_widget.set_hand_text(hand_strength_text(opponent_cards, public_cards, game=game_id))
        else:
            num_cards = len(private_cards) if private_cards else 2
            hidden_cards = [None] * num_cards
            self.player_top_widget.set_cards(hidden_cards, reveal=False)
            if hasattr(self.player_top_widget, "set_hand_text"):
                self.player_top_widget.set_hand_text("")

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
        if pot != self.last_pot_value:
            self.update_pot_chips(pot)
            self.last_pot_value = pot

    def update_players_from_state(self, state):
        current_player = state.get('current_player', 0)
        player_bets = state.get('player_bets', [0, 0])

        is_human_turn = (current_player == self.player_id)
        is_opponent_turn = (current_player == self.opponent_id)

        self.player_bottom_widget.set_current_player(is_human_turn)
        self.player_top_widget.set_current_player(is_opponent_turn)

    def update_actions_from_state(self, state):
        if state.get('done', False):
            if hasattr(self, 'action_buttons'):
                self.action_buttons.disable_all()
            return

        current_player = state.get('current_player', 0)

        if current_player == self.player_id:
            legal_actions = state.get('legal_actions', [])
            if hasattr(self, 'action_buttons'):
                self.action_buttons.update_legal_actions(legal_actions)

                if 'current_bet_size' in state:
                    call_amount = state['current_bet_size']
                    if call_amount > 0:
                        self.action_buttons.set_amount_to_call(call_amount)
        else:
            if hasattr(self, 'action_buttons'):
                self.action_buttons.disable_all()

    def update_history_from_state(self, state):
        # Prefer server-provided history with explicit player attribution.
        events = state.get("history_events")
        if isinstance(events, list):
            history = events
            use_events = True
        else:
            history = state.get('history', [])
            use_events = False

        player_bets = state.get('player_bets', [0, 0])

        if len(history) > self.last_history_length:
            new_actions = history[self.last_history_length:]
            if use_events:
                self._add_new_history_events(new_actions, player_bets, state)
            else:
                self._add_new_history_actions(new_actions, player_bets, state)
            self.last_history_length = len(history)
        elif len(history) < self.last_history_length:
            self.history_view.clear()
            self.last_history_length = 0
            self.last_player_bets = [0, 0]
            if history:
                if use_events:
                    self._add_new_history_events(history, player_bets, state)
                else:
                    self._add_new_history_actions(history, player_bets, state)
                self.last_history_length = len(history)

        if 'player_bets' in state:
            self.last_player_bets = state['player_bets'].copy()

    def _add_new_history_events(self, events, player_bets, state):
        names_list = state.get("player_names") or []

        def _name_for(pid: int) -> str:
            if isinstance(names_list, list) and 0 <= pid < len(names_list) and names_list[pid]:
                return str(names_list[pid])
            if pid == self.player_id:
                return self.human_name
            return "Opponent"

        for ev in events:
            if not isinstance(ev, dict):
                continue

            if ev.get("type") == "separator":
                self.history_view.add_round_separator()
                continue

            if ev.get("type") != "action":
                continue

            pid = ev.get("player_id")
            action = ev.get("action")
            if pid is None or action is None:
                continue

            try:
                pid_int = int(pid)
            except Exception:
                pid_int = pid

            bet_size = None
            if action in ['bet', 'call', 'raise']:
                if isinstance(player_bets, list) and isinstance(self.last_player_bets, list):
                    try:
                        bet_size = player_bets[pid_int] - self.last_player_bets[pid_int]
                    except Exception:
                        bet_size = None

            self.history_view.add_action(pid_int, action, bet_size=bet_size, player_name=_name_for(pid_int))

    def _add_new_history_actions(self, actions, player_bets, state):
        names_list = state.get("player_names") or []

        def _name_for(pid: int) -> str:
            if isinstance(names_list, list) and 0 <= pid < len(names_list) and names_list[pid]:
                return str(names_list[pid])
            if pid == self.player_id:
                return self.human_name
            return "Opponent"

        player_names = {
            self.player_id: _name_for(self.player_id),
            self.opponent_id: _name_for(self.opponent_id),
        }

        if self.last_state and 'current_player' in self.last_state:
            current_player_for_action = self.last_state.get('current_player', 0)
        else:
            current_player_for_action = 0

        for i, action in enumerate(actions):
            if action == '|':
                self.history_view.add_round_separator()
            else:
                player_id = current_player_for_action
                player_name = player_names.get(player_id, f"Player {player_id}")

                bet_size = None
                if action in ['bet', 'call']:
                    if self.last_player_bets:
                        bet_size = player_bets[player_id] - self.last_player_bets[player_id]
                    else:
                        bet_size = player_bets[player_id] if player_bets[player_id] > 0 else None

                self.history_view.add_action(player_id, action, bet_size=bet_size, player_name=player_name)
                current_player_for_action = 1 - current_player_for_action

    def _check_and_show_game_result(self, state):
        if not state.get('done', False):
            self.game_result_shown = False
            return

        if self.game_result_shown:
            return

        pot = state.get('pot', 0)
        payoffs = state.get('payoffs', [0, 0])

        if payoffs[0] > payoffs[1]:
            winner_id = 0
        elif payoffs[1] > payoffs[0]:
            winner_id = 1
        else:
            winner_id = 0

        if winner_id == self.player_id:
            winner_name = self.human_name
        else:
            names_list = state.get("player_names") or []
            if isinstance(names_list, list) and 0 <= winner_id < len(names_list) and names_list[winner_id]:
                winner_name = str(names_list[winner_id])
            else:
                winner_name = "Opponent"

        self.history_view.add_game_result(winner_id, pot, winner_name=winner_name)

        payoff_label = f"\nPayoff: {self.human_name}={payoffs[0]}, Opponent={payoffs[1]}"
        from PyQt6.QtWidgets import QLabel
        from PyQt6.QtGui import QFont
        label = QLabel(payoff_label)
        label.setFont(QFont("Arial", 11))
        label.setStyleSheet("""
            QLabel {
                color: #ffd700;
                padding: 3px;
            }
        """)
        self.history_view.content_layout.addWidget(label)
        self.history_view.history_items.append(label)

        self.game_result_shown = True

    def update_actions_final(self):
        if hasattr(self, 'action_buttons'):
            self.action_buttons.disable_all()

    def handle_action(self, action, bet_size):
        if self.last_state and self.last_state.get('done', False):
            return

        if self.last_state:
            current_player = self.last_state.get('current_player', -1)
            if current_player != self.player_id:
                return

            legal_actions = self.last_state.get('legal_actions', [])
            if action not in legal_actions:
                return

        self.sound_manager.play_action(action)
        self.client.send_action(action, bet_size)

    def restart_hand(self):
        import random
        starting_player = random.randint(0, 1)
        self.last_history_length = 0
        self.last_player_bets = [0, 0]
        self.last_pot_value = 0
        self.game_result_shown = False
        self.client.send_reset_request(starting_player)

    def _on_connection_error(self, error_message):
        QMessageBox.warning(
            self,
            "Connection Error",
            f"Connection error: {error_message}\n\nTrying to reconnect..."
        )
        QTimer.singleShot(2000, lambda: self.client.connect())

    def closeEvent(self, event):
        # Ensure the server frees our player slot (0/1) on restart.
        try:
            if hasattr(self, "client") and self.client is not None:
                self.client.disconnect()
        finally:
            super().closeEvent(event)
