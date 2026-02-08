from flask import Flask, jsonify, request
from typing import Optional
import threading
import time
import random


class PokerHTTPServer:
    def __init__(self, game, host='0.0.0.0', port=8888, game_id: Optional[str] = None):
        self.game = game
        self.host = host
        self.port = port
        self.game_id = game_id
        self.app = Flask(__name__)
        self.max_players = 2
        # player_id -> last_seen timestamp (time.time())
        self.clients: dict[int, float] = {}
        # player_id -> display name
        self.client_names: dict[int, str] = {}
        # If a client hasn't polled state for this long, we consider it gone.
        self.client_stale_seconds = 5.0
        # Increments every time /reset is called so clients can detect a new hand.
        self.reset_id = 0
        # Server-side action log with explicit player attribution.
        # Items are dicts like {"type": "action", "player_id": 0, "action": "check"}
        # or {"type": "separator"} for round separators ("|") added by the env.
        self.history_events: list[dict] = []
        self.lock = threading.Lock()

        self._setup_routes()

    def _prune_stale_clients(self) -> None:
        now = time.time()
        stale_before = now - self.client_stale_seconds
        stale_ids = [pid for pid, last_seen in self.clients.items() if last_seen < stale_before]
        for pid in stale_ids:
            self.clients.pop(pid, None)
            self.client_names.pop(pid, None)

    def _setup_routes(self):
        @self.app.route('/player_id', methods=['GET'])
        def get_player_id():
            with self.lock:
                self._prune_stale_clients()
                name = request.args.get('name')

                for pid in range(self.max_players):
                    if pid not in self.clients:
                        self.clients[pid] = time.time()
                        if isinstance(name, str) and name.strip():
                            self.client_names[pid] = name.strip()
                        return jsonify({'player_id': pid})

                # Server full (two active clients). Tell the client to retry later.
                return jsonify({'status': 'error', 'message': 'Server full (2 players already connected).'}), 409

        @self.app.route('/state', methods=['GET'])
        def get_state():
            player_id = int(request.args.get('player_id', 0))
            state = self._get_state_update(player_id)
            return jsonify(state)

        @self.app.route('/action', methods=['POST'])
        def handle_action():
            data = request.json
            player_id = data.get('player_id')
            action = data.get('action')
            bet_size = data.get('bet_size', 0)

            result = self._handle_action(player_id, action, bet_size)
            if result:
                return jsonify({'status': 'ok'})
            else:
                return jsonify({'status': 'error', 'message': 'Invalid action'}), 400

        @self.app.route('/disconnect', methods=['POST'])
        def disconnect_player():
            data = request.json or {}
            player_id = data.get('player_id')
            try:
                player_id = int(player_id)
            except Exception:
                return jsonify({'status': 'error', 'message': 'Invalid player_id'}), 400

            with self.lock:
                self.clients.pop(player_id, None)
                self.client_names.pop(player_id, None)
            return jsonify({'status': 'ok'})

        @self.app.route('/reset', methods=['POST'])
        def reset_game():
            data = request.json or {}
            starting_player = data.get('starting_player', 0)

            self._reset_game(starting_player)
            return jsonify({'status': 'ok'})

    def _apply_leduc_public_card_fallback(self) -> bool:
        """Fallback wenn Chance-Knoten (öffentliche Karte) keine Outcomes liefert (nur Leduc)."""
        if not hasattr(self.game, '_chance_targets') or not self.game._chance_targets:
            return False
        kind = self.game._chance_targets[0][0] if self.game._chance_targets else None
        if kind != 'public':
            return False
        if not hasattr(self.game, 'public_card') or not hasattr(self.game, 'round'):
            return False
        full_deck = ['Js', 'Jh', 'Qs', 'Qh', 'Ks', 'Kh']
        if getattr(self.game, 'abstract_suits', False):
            full_deck = ['J', 'J', 'Q', 'Q', 'K', 'K']
        used = []
        for i in (0, 1):
            c = getattr(self.game.players[i], 'private_card', None)
            if c is not None:
                used.append(c)
        remaining = list(full_deck)
        for c in used:
            if c in remaining:
                remaining.remove(c)
        if not remaining:
            return False
        card = random.choice(remaining)
        self.game._apply_public_card(card)
        self.game._chance_targets.clear()
        ctx = self.game._chance_context or {}
        self.game._chance_context = None
        if hasattr(self.game, '_after_public_deal'):
            self.game._after_public_deal(ctx)
        return True

    def _drain_chance_nodes(self) -> None:
        """Arbeitet Chance-Knoten (z.B. Karten austeilen, öffentliche Karte bei Leduc) ab,
        bis ein Decision- oder Terminal-Knoten erreicht ist."""
        if not hasattr(self.game, 'is_chance_node') or not self.game.is_chance_node():
            return
        while not self.game.done and self.game.is_chance_node():
            outcomes = self.game.get_chance_outcomes_with_probs()
            if outcomes:
                keys = list(outcomes.keys())
                weights = [outcomes[k] for k in keys]
                outcome = random.choices(keys, weights=weights)[0]
                self.game.step(outcome)
                continue
            if self._apply_leduc_public_card_fallback():
                continue
            break

    def _get_state_update(self, player_id):
        with self.lock:
            self._prune_stale_clients()
            # Mark client as active if it's a valid player id (0/1)
            try:
                pid_int = int(player_id)
            except Exception:
                pid_int = None
            if pid_int in (0, 1):
                self.clients[pid_int] = time.time()

            state = self.game.get_state(self.game.current_player)

            # Expose the explicit game identifier so clients can interpret the
            # state correctly (e.g. hand strength display) without heuristics.
            if self.game_id is not None:
                state['game'] = self.game_id

            # Ensure a consistent contract for all games:
            # some envs don't include 'done' in get_state(), but the GUI relies on it.
            state['done'] = bool(getattr(self.game, 'done', False))

            # Expose player display names for UI (history, headers, etc.)
            state['player_names'] = [
                self.client_names.get(0, ''),
                self.client_names.get(1, ''),
            ]

            # Hand / reset marker so both clients can clear UI state together.
            state['reset_id'] = int(self.reset_id)
            # Prefer this over raw env history on the client (has player attribution).
            state['history_events'] = list(self.history_events)

            state['private_cards'] = self._get_private_cards(player_id)
            state['public_cards'] = self._get_public_cards()

            if hasattr(self.game, 'total_bets'):
                state['player_bets'] = list(self.game.total_bets)
            elif hasattr(self.game, 'player_bets'):
                state['player_bets'] = list(self.game.player_bets)
            else:
                state['player_bets'] = [0, 0]

            if 'legal_actions' not in state:
                state['legal_actions'] = self.game.get_legal_actions()

            if self.game.done:
                opponent_id = 1 - player_id
                state['opponent_cards'] = self._get_private_cards(opponent_id)

                # Judger erwartet den Spieler, der zuletzt gehandelt hat (bei Fold = der Folder).
                # Nach step() hat proceed_round current_player bereits gewechselt → Gewinner.
                history = getattr(self.game, 'history', [])
                if history and history[-1] == 'fold':
                    judge_player = 1 - self.game.current_player  # Folder
                else:
                    judge_player = self.game.current_player
                payoffs = self.game.judger.judge(
                    self.game.players,
                    history,
                    judge_player,
                    self.game.pot,
                    state['player_bets']
                )
                # Wie bei Agent vs Human: anfragender Client bekommt [mein Payoff, Gegner-Payoff]
                try:
                    pid = int(player_id)
                    if pid in (0, 1) and len(payoffs) >= 2:
                        state['payoffs'] = [payoffs[pid], payoffs[1 - pid]]
                    else:
                        state['payoffs'] = list(payoffs)
                except Exception:
                    state['payoffs'] = list(payoffs)

            return state

    def _get_private_cards(self, player_id):
        if player_id >= len(self.game.players):
            return []

        player = self.game.players[player_id]

        if hasattr(player, 'private_cards') and player.private_cards:
            cards = list(player.private_cards) if isinstance(player.private_cards, (list, tuple)) else [
                player.private_cards]
            return cards
        elif hasattr(player, 'private_card') and player.private_card:
            card = [player.private_card]
            return card

        return []

    def _get_public_cards(self):
        if hasattr(self.game, 'public_cards') and self.game.public_cards:
            return list(self.game.public_cards) if isinstance(self.game.public_cards, (list, tuple)) else [
                self.game.public_cards]
        elif hasattr(self.game, 'public_card') and self.game.public_card:
            return [self.game.public_card]
        return []

    def _handle_action(self, player_id, action, bet_size):
        with self.lock:
            if self.game.done:
                return False

            if self.game.current_player != player_id:
                return False

            state = self.game.get_state(self.game.current_player)
            legal_actions = state.get('legal_actions', self.game.get_legal_actions())

            if action not in legal_actions:
                return False

            history_before = list(getattr(self.game, "history", []))
            self.game.step(action)
            history_after = list(getattr(self.game, "history", []))

            # Capture exactly what the env appended (e.g. "|" + new cards).
            appended = history_after[len(history_before):]
            for entry in appended:
                if entry == '|':
                    self.history_events.append({"type": "separator"})
                else:
                    try:
                        pid_int = int(player_id)
                    except Exception:
                        pid_int = player_id
                    self.history_events.append({"type": "action", "player_id": pid_int, "action": entry})

            # Nach Spieler-Aktion ggf. Chance-Nodes abarbeiten (z.B. Leduc: öffentliche Karte)
            self._drain_chance_nodes()
            return True

    def _reset_game(self, starting_player):
        with self.lock:
            self.reset_id += 1
            self.history_events = []
            self.game.reset(starting_player)

            # Spiele mit Chance-Nodes (Kuhn, Leduc, …): Karten werden per _drain_chance_nodes verteilt
            self._drain_chance_nodes()

            # Spiele ohne Chance-Nodes: Karten manuell austeilen
            if not hasattr(self.game, 'is_chance_node') and hasattr(self.game, 'dealer'):
                for i, player in enumerate(self.game.players):
                    if hasattr(player, 'set_private_cards'):
                        if hasattr(self.game.dealer, 'deal_card') and len(self.game.dealer.deck) >= 2:
                            card1 = self.game.dealer.deal_card()
                            card2 = self.game.dealer.deal_card()
                            player.set_private_cards(card1, card2)
                    elif hasattr(player, 'set_private_card'):
                        if hasattr(self.game.dealer, 'deal_card') and len(self.game.dealer.deck) > 0:
                            card = self.game.dealer.deal_card()
                            player.set_private_card(card)

    def start(self):
        print(f"Server läuft auf http://{self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=False, threaded=True)
