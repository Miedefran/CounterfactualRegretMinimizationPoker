#!/bin/bash
# Erstellt ein virtuelles Environment und installiert PyTorch mit MPS

echo "Erstelle virtuelles Environment..."
python3 -m venv venv

echo "Aktiviere virtuelles Environment..."
source venv/bin/activate

echo "Installiere PyTorch mit MPS-Unterstützung..."
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

echo ""
echo "✅ Fertig!"
echo ""
echo "Um das Environment zu aktivieren, führe aus:"
echo "  source venv/bin/activate"
echo ""
echo "Dann kannst du 'python check_mps.py' ausführen."



