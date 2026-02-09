from flask import Flask, jsonify, request
from typing import Optional
from gui.server.game_logic import PokerGameLogic


class PokerHTTPServer:
    def __init__(self, game, host='0.0.0.0', port=8888, game_id: Optional[str] = None):
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        # Nutze gemeinsame Spiellogik
        self.game_logic = PokerGameLogic(game, game_id=game_id)

        self._setup_routes()

    def _setup_routes(self):
        @self.app.route('/player_id', methods=['GET'])
        def get_player_id():
            name = request.args.get('name')
            
            # Versuche Spieler 0 oder 1 zu registrieren
            for pid in range(self.game_logic.max_players):
                if self.game_logic.register_client(pid, name):
                    return jsonify({'player_id': pid})

            # Server full (two active clients). Tell the client to retry later.
            return jsonify({'status': 'error', 'message': 'Server full (2 players already connected).'}), 409

        @self.app.route('/state', methods=['GET'])
        def get_state():
            player_id = int(request.args.get('player_id', 0))
            state = self.game_logic.get_state_update(player_id)
            return jsonify(state)

        @self.app.route('/action', methods=['POST'])
        def handle_action():
            data = request.json
            player_id = data.get('player_id')
            action = data.get('action')
            bet_size = data.get('bet_size', 0)

            result = self.game_logic.handle_action(player_id, action, bet_size)
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

            self.game_logic.unregister_client(player_id)
            return jsonify({'status': 'ok'})

        @self.app.route('/reset', methods=['POST'])
        def reset_game():
            data = request.json or {}
            starting_player = data.get('starting_player', 0)

            self.game_logic.reset_game(starting_player)
            return jsonify({'status': 'ok'})


    def start(self):
        print(f"Server läuft auf http://{self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=False, threaded=True)
