from flask import Flask, jsonify, request
from functools import wraps
from typing import Optional
from gui.server.game_logic import PokerGameLogic
from gui.server.validation import (
    validate_player_id, validate_action, validate_bet_size,
    validate_player_name, validate_starting_player
)
from gui.server.rate_limiter import RateLimiter


class PokerHTTPServer:
    def __init__(self, game, host='localhost', port=8888, game_id: Optional[str] = None, 
                 ssl_cert: Optional[str] = None, ssl_key: Optional[str] = None):
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        # Max Request Size: 1MB
        self.app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
        # Nutze gemeinsame Spiellogik
        self.game_logic = PokerGameLogic(game, game_id=game_id)
        # SSL-Context
        self.ssl_context = None
        if ssl_cert and ssl_key:
            self.ssl_context = (ssl_cert, ssl_key)
        
        # Rate Limiter
        self.rate_limiter = RateLimiter()

        self._setup_routes()
    
    def _rate_limit(self, max_requests: int, window_seconds: float):
        """Decorator für Rate Limiting."""
        def decorator(f):
            @wraps(f)
            def wrapper(*args, **kwargs):
                ip = request.remote_addr or 'unknown'
                if not self.rate_limiter.is_allowed(ip, max_requests, window_seconds):
                    return jsonify({
                        'status': 'error',
                        'message': 'Rate limit exceeded'
                    }), 429
                return f(*args, **kwargs)
            return wrapper
        return decorator

    def _setup_routes(self):
        @self.app.route('/player_id', methods=['GET'])
        @self._rate_limit(max_requests=5, window_seconds=60)
        def get_player_id():
            name = request.args.get('name')
            # Validierung des Namens
            is_valid, sanitized_name, error = validate_player_name(name)
            if not is_valid and error:
                return jsonify({'status': 'error', 'message': error}), 400
            
            # Versuche Spieler 0 oder 1 zu registrieren
            for pid in range(self.game_logic.max_players):
                if self.game_logic.register_client(pid, sanitized_name):
                    return jsonify({'player_id': pid})

            # Server full (two active clients). Tell the client to retry later.
            return jsonify({'status': 'error', 'message': 'Server full (2 players already connected).'}), 409

        @self.app.route('/state', methods=['GET'])
        @self._rate_limit(max_requests=20, window_seconds=10)
        def get_state():
            player_id = request.args.get('player_id')
            is_valid, pid, error = validate_player_id(player_id)
            if not is_valid:
                return jsonify({'status': 'error', 'message': error}), 400
            state = self.game_logic.get_state_update(pid)
            return jsonify(state)

        @self.app.route('/action', methods=['POST'])
        @self._rate_limit(max_requests=10, window_seconds=10)
        def handle_action():
            if not request.json:
                return jsonify({'status': 'error', 'message': 'No JSON data'}), 400
            
            data = request.json
            # Validierung player_id
            is_valid, pid, error = validate_player_id(data.get('player_id'))
            if not is_valid:
                return jsonify({'status': 'error', 'message': error}), 400
            
            # Validierung bet_size
            is_valid, bet_val, error = validate_bet_size(data.get('bet_size', 0))
            if not is_valid:
                return jsonify({'status': 'error', 'message': error}), 400
            
            action = data.get('action')
            # action wird in game_logic.handle_action gegen legal_actions validiert

            result = self.game_logic.handle_action(pid, action, bet_val)
            if result:
                return jsonify({'status': 'ok'})
            else:
                return jsonify({'status': 'error', 'message': 'Invalid action'}), 400

        @self.app.route('/disconnect', methods=['POST'])
        def disconnect_player():
            data = request.json or {}
            is_valid, pid, error = validate_player_id(data.get('player_id'))
            if not is_valid:
                return jsonify({'status': 'error', 'message': error}), 400

            self.game_logic.unregister_client(pid)
            return jsonify({'status': 'ok'})

        @self.app.route('/reset', methods=['POST'])
        @self._rate_limit(max_requests=5, window_seconds=60)
        def reset_game():
            data = request.json or {}
            is_valid, sp, error = validate_starting_player(data.get('starting_player', 0))
            if not is_valid:
                return jsonify({'status': 'error', 'message': error}), 400

            self.game_logic.reset_game(sp)
            return jsonify({'status': 'ok'})


    def start(self):
        protocol = "https" if self.ssl_context else "http"
        print(f"Server läuft auf {protocol}://{self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=False, threaded=True,
                    ssl_context=self.ssl_context)
