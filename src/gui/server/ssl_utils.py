"""SSL-Zertifikat-Verwaltung für WebSocket und HTTP Server.

Generiert automatisch selbstsignierte Zertifikate für Entwicklung und lokale Nutzung.
"""
import os
import ipaddress
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Optional
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa


def get_ssl_dir() -> Path:
    """Gibt das Verzeichnis für SSL-Zertifikate zurück."""
    project_root = Path(__file__).parent.parent.parent.parent
    ssl_dir = project_root / "config" / "ssl"
    ssl_dir.mkdir(parents=True, exist_ok=True)
    return ssl_dir


def get_or_create_ssl_certificates() -> Tuple[str, str]:
    """Generiert selbstsignierte SSL-Zertifikate falls nicht vorhanden.
    
    Returns:
        Tuple von (cert_path, key_path) als Strings
    """
    ssl_dir = get_ssl_dir()
    cert_path = ssl_dir / "cert.pem"
    key_path = ssl_dir / "key.pem"
    
    # Wenn beide Dateien existieren, verwende sie
    if cert_path.exists() and key_path.exists():
        return str(cert_path), str(key_path)
    
    # Generiere neues Zertifikat
    # Generiere privaten Schlüssel
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    
    # Erstelle Zertifikat
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "DE"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Berlin"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "Berlin"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Poker CFR"),
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])
    
    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        private_key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.utcnow()
    ).not_valid_after(
        datetime.utcnow() + timedelta(days=365)
    ).add_extension(
        x509.SubjectAlternativeName([
            x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        ]),
        critical=False,
    ).sign(private_key, hashes.SHA256())
    
    # Speichere Zertifikat
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    
    # Speichere privaten Schlüssel
    with open(key_path, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))
    
    # Setze Berechtigungen für privaten Schlüssel (nur für Unix)
    if os.name != 'nt':
        os.chmod(key_path, 0o600)
    
    return str(cert_path), str(key_path)
