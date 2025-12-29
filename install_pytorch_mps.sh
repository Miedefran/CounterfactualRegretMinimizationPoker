#!/bin/bash
# Script zum Installieren von PyTorch mit MPS-Unterstützung auf macOS
# Basierend auf: https://developer.apple.com/metal/pytorch/

echo "=========================================="
echo "PyTorch mit MPS-Unterstützung installieren"
echo "=========================================="
echo ""

# Prüfe ob ein virtuelles Environment aktiv ist
if [ -n "$VIRTUAL_ENV" ]; then
    echo "✅ Virtuelles Environment erkannt: $VIRTUAL_ENV"
    PIP_CMD="pip"
    PYTHON_CMD="python"
else
    echo "ℹ️  Kein virtuelles Environment aktiv."
    echo "   Empfehlung: Aktiviere ein venv oder conda environment"
    PIP_CMD="pip3"
    PYTHON_CMD="python3"
fi

# Prüfe ob pip verfügbar ist
if ! command -v $PIP_CMD &> /dev/null; then
    echo "❌ $PIP_CMD nicht gefunden. Bitte installiere pip zuerst."
    exit 1
fi

echo "Verwende: $PIP_CMD"
echo ""

# Prüfe macOS Version
macos_version=$(sw_vers -productVersion)
echo "macOS Version: $macos_version"
echo ""

# Prüfe ob Xcode Command Line Tools installiert sind
if ! xcode-select -p &> /dev/null; then
    echo "⚠️  Xcode Command Line Tools nicht gefunden."
    echo "   Installiere sie mit: xcode-select --install"
    echo ""
    read -p "Möchtest du sie jetzt installieren? (j/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Jj]$ ]]; then
        xcode-select --install
    fi
fi

echo "Installiere PyTorch mit MPS-Unterstützung..."
echo ""

# Installation gemäß Apple-Dokumentation
# Für stabile Version (PyTorch 1.12+):
# $PIP_CMD install torch torchvision torchaudio

# Für neueste Nightly-Version mit bestem MPS-Support:
$PIP_CMD install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

echo ""
echo "✅ Installation abgeschlossen!"
echo ""
echo "Führe jetzt '$PYTHON_CMD check_mps.py' aus, um zu prüfen ob MPS funktioniert."



