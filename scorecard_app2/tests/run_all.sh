#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.

echo "=========================================="
echo "  1. Génération des données synthétiques"
echo "=========================================="
python tests/generate_synthetic_data.py

echo "=========================================="
echo "  2. Tests de configuration"
echo "=========================================="
python tests/test_config.py

echo "=========================================="
echo "  3. Tests unitaires du moteur"
echo "=========================================="
python tests/test_engine.py

echo "=========================================="
echo "  4. Tests d'intégration des composants"
echo "=========================================="
python tests/test_integration.py

echo "=========================================="
echo "  5. Tests Streamlit (AppTest)"
echo "=========================================="
python tests/test_streamlit_app.py 2>/dev/null

echo "=========================================="
echo "  ✅ TOUS LES TESTS PASSÉS"
echo "=========================================="
