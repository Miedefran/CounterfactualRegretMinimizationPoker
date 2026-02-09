"""Rate Limiting für HTTP- und WebSocket-Server."""

from collections import defaultdict
from time import time
from typing import Dict, List
import threading


class RateLimiter:
    """Sliding Window Rate Limiter für HTTP-Server."""
    
    def __init__(self):
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()
    
    def is_allowed(self, ip: str, max_requests: int, window_seconds: float) -> bool:
        """Prüft ob Request erlaubt ist.
        
        Args:
            ip: IP-Adresse des Clients
            max_requests: Maximale Anzahl Requests im Zeitfenster
            window_seconds: Zeitfenster in Sekunden
            
        Returns:
            True wenn erlaubt, False wenn Limit überschritten
        """
        with self.lock:
            now = time()
            # Entferne alte Requests außerhalb des Zeitfensters
            self.requests[ip] = [
                t for t in self.requests[ip] 
                if now - t < window_seconds
            ]
            
            # Prüfe Limit
            if len(self.requests[ip]) >= max_requests:
                return False
            
            # Füge aktuellen Request hinzu
            self.requests[ip].append(now)
            return True
    
    def reset(self, ip: str = None):
        """Setzt Rate Limit für eine IP oder alle IPs zurück."""
        with self.lock:
            if ip:
                self.requests.pop(ip, None)
            else:
                self.requests.clear()


class WebSocketRateLimiter:
    """Rate Limiter für WebSocket-Verbindungen."""
    
    def __init__(self):
        self.message_counts: Dict[int, List[float]] = defaultdict(list)
        self.lock = threading.Lock()
    
    def check_limit(self, player_id: int, max_messages: int = 50, window: float = 10.0) -> bool:
        """Prüft ob Nachricht erlaubt ist.
        
        Args:
            player_id: Player-ID
            max_messages: Maximale Anzahl Nachrichten im Zeitfenster
            window: Zeitfenster in Sekunden
            
        Returns:
            True wenn erlaubt, False wenn Limit überschritten
        """
        with self.lock:
            now = time()
            # Entferne alte Nachrichten außerhalb des Zeitfensters
            self.message_counts[player_id] = [
                t for t in self.message_counts[player_id]
                if now - t < window
            ]
            
            # Prüfe Limit
            if len(self.message_counts[player_id]) >= max_messages:
                return False
            
            # Füge aktuelle Nachricht hinzu
            self.message_counts[player_id].append(now)
            return True
    
    def reset(self, player_id: int = None):
        """Setzt Rate Limit für einen Spieler oder alle Spieler zurück."""
        with self.lock:
            if player_id is not None:
                self.message_counts.pop(player_id, None)
            else:
                self.message_counts.clear()
