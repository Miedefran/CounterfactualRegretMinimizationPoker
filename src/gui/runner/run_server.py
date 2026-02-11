import sys
import argparse
import socket
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


from envs.kuhn_poker.game import KuhnPokerGame
from envs.leduc_holdem.game import LeducHoldemGame
from envs.twelve_card_poker.game import TwelveCardPokerGame
from envs.rhode_island.game import RhodeIslandGame
from envs.royal_holdem.game import RoyalHoldemGame
from envs.limit_holdem.game import LimitHoldemGame
from gui.server.http_server import PokerHTTPServer
from gui.server.ssl_utils import get_or_create_ssl_certificates


def main():
    parser = argparse.ArgumentParser(description='Start HTTP Server for Human vs Human mode')
    parser.add_argument('--host', default='localhost',
                        help='Server IP (localhost = nur lokal, 0.0.0.0 = alle Interfaces, default: localhost)')
    parser.add_argument('--port', type=int, default=8888,
                        help='Server port (default: 8888)')
    parser.add_argument('--game', default='limit_holdem',
                        choices=['kuhn', 'leduc', 'twelve_card', 'rhode_island', 'royal_holdem', 'limit_holdem'],
                        help='Game type (default: limit_holdem)')
    parser.add_argument('--no-ssl', action='store_true',
                        help='Deaktiviert SSL/HTTPS (verwendet unverschlüsseltes HTTP)')
    parser.add_argument('--cert', type=str, default=None,
                        help='Path to SSL certificate file (for HTTPS, optional - verwendet Standard-Zertifikat wenn nicht angegeben)')
    parser.add_argument('--key', type=str, default=None,
                        help='Path to SSL private key file (for HTTPS, optional - verwendet Standard-Zertifikat wenn nicht angegeben)')

    args = parser.parse_args()

    if args.game == 'kuhn':
        game = KuhnPokerGame()
        print(f"Initialized Kuhn Poker game")
    elif args.game == 'leduc':
        game = LeducHoldemGame()
        print(f"Initialized Leduc Hold'em game")
    elif args.game == 'twelve_card':
        game = TwelveCardPokerGame()
        print(f"Initialized Twelve Card Poker game")
    elif args.game == 'rhode_island':
        game = RhodeIslandGame()
        print(f"Initialized Rhode Island Hold'em game")
    elif args.game == 'royal_holdem':
        game = RoyalHoldemGame()
        print(f"Initialized Royal Hold'em game")
    elif args.game == 'limit_holdem':
        game = LimitHoldemGame()
        print(f"Initialized Limit Hold'em game")

    local_ip = get_local_ip()

    # Sicherheitshinweis wenn 0.0.0.0 verwendet wird
    if args.host == '0.0.0.0':
        print("⚠️  WARNING: Server bindet an alle Interfaces (0.0.0.0).")
        print("   Dies macht den Server von außen erreichbar - nur für vertrauenswürdige Netzwerke verwenden!")

    # SSL-Validierung und Context-Erstellung
    ssl_cert = None
    ssl_key = None
    if not args.no_ssl:
        # Verwende benutzerdefinierte Zertifikate falls angegeben, sonst Standard-Zertifikate
        if args.cert and args.key:
            ssl_cert = args.cert
            ssl_key = args.key
        elif args.cert or args.key:
            print("ERROR: --cert und --key müssen beide angegeben werden für SSL")
            sys.exit(1)
        else:
            # Generiere oder verwende Standard-Zertifikate
            ssl_cert, ssl_key = get_or_create_ssl_certificates()
            print(f"SSL enabled: Using auto-generated certificate")

    server = PokerHTTPServer(game, host=args.host, port=args.port, game_id=args.game,
                             ssl_cert=ssl_cert, ssl_key=ssl_key)
    protocol = "https" if ssl_cert else "http"
    print(f"\n🎮 Server gestartet!")
    print(f"📍 Lokale IP: {local_ip}")
    print(f"🔌 Port: {args.port}")
    print(f"🎲 Game: {args.game}")
    print(f"\n💻 Andere Spieler verbinden mit:")
    print(f"   python src/gui/runner/run_client.py --ip {local_ip} --port {args.port}")
    print(f"   URL: {protocol}://{local_ip}:{args.port}")
    print(f"\nPress Ctrl+C to stop the server\n")

    server.start()


if __name__ == "__main__":
    main()
