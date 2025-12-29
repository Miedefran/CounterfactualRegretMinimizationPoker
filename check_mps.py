#!/usr/bin/env python3
"""Script zum Prüfen der MPS (Metal Performance Shaders) Verfügbarkeit auf macOS
Basierend auf: https://developer.apple.com/metal/pytorch/
"""

import torch

print("=" * 60)
print("PyTorch MPS Verfügbarkeits-Check")
print("=" * 60)
print()

# PyTorch Version
print(f"PyTorch Version: {torch.__version__}")
print()

# MPS Verfügbarkeit (wie in Apple-Dokumentation)
print("MPS (Metal Performance Shaders) Status:")
print("-" * 60)

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print("✅ MPS device ist verfügbar!")
    print(f"   Device: {mps_device}")
    print(f"   Test-Tensor: {x}")
    print()
    print("🎉 Du kannst die GPU auf deinem Mac verwenden!")
else:
    print("❌ MPS device nicht gefunden.")
    print()
    print("Mögliche Gründe:")
    print("  - Du verwendest kein Apple Silicon (M1/M2/M3) oder AMD GPU")
    print("  - macOS Version ist zu alt (benötigt macOS 12.3+)")
    print("  - PyTorch wurde ohne MPS-Unterstützung installiert")
    print()
    print("Installation:")
    print("  pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu")

print()
print("=" * 60)



