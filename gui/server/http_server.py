from flask import Flask, jsonify, request
from typing import Optional
import threading

class PokerHTTPServer:
    def __init__(self, game, host='0.0.0.0', port=8888):
        self.game = game
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.next_player_id = 0
        self.clients = {}
        self.lock = threading.Lock()
        
        self._setup_routes()
    
    def _setup_routes(self):
        @self.app.route('/player_id', methods=['GET'])
        def get_player_id():
            player_id = self.next_player_id
            self.next_player_id += 1
            self.clients[player_id] = True
            return jsonify({'player_id': player_id})
        
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
        
        @self.app.route('/reset', methods=['POST'])
        def reset_game():
            data = request.json or {}
            starting_player = data.get('starting_player', 0)
            
            if len(self.clients) < 2:
                return jsonify({'status': 'error', 'message': 'Not all clients connected'}), 400
            
            self._reset_game(starting_player)
            return jsonify({'status': 'ok'})
    
    def _get_state_update(self, player_id):
        with self.lock:
            state = self.game.get_state(self.game.current_player)
            
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
                
                payoffs = self.game.judger.judge(
                    self.game.players,
                    self.game.history,
                    self.game.current_player,
                    self.game.pot,
                    state['player_bets']
                )
                state['payoffs'] = payoffs
            
            return state
    
    def _get_private_cards(self, player_id):
        if player_id >= len(self.game.players):
            return []
        
        player = self.game.players[player_id]
        
        if hasattr(player, 'private_cards') and player.private_cards:
            cards = list(player.private_cards) if isinstance(player.private_cards, (list, tuple)) else [player.private_cards]
            return cards
        elif hasattr(player, 'private_card') and player.private_card:
            card = [player.private_card]
            return card
        
        return []
    
    def _get_public_cards(self):
        if hasattr(self.game, 'public_cards') and self.game.public_cards:
            return list(self.game.public_cards) if isinstance(self.game.public_cards, (list, tuple)) else [self.game.public_cards]
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
            
            self.game.step(action)
            return True
    
    def _reset_game(self, starting_player):
        with self.lock:
            self.game.reset(starting_player)
            
            if hasattr(self.game, 'dealer'):
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

