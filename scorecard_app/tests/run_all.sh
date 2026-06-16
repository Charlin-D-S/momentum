#!/usr/bin/env bash
# Lance toute la chaîne de tests : génération de données → unitaires → intégration → Streamlit.
set -e

cd "$(dirname "$0")/.."
export PYTHONPATH=.

echo "=========================================="
echo "  1. Génération des données synthétiques"
echo "=========================================="
python tests/generate_synthetic_data.py
echo ""

echo "=========================================="
echo "  2. Tests unitaires du moteur"
echo "=========================================="
python tests/test_engine.py
echo ""

echo "=========================================="
echo "  3. Tests d'intégration des composants"
echo "=========================================="
python tests/test_integration.py
echo ""

echo "=========================================="
echo "  4. Tests Streamlit (AppTest)"
echo "=========================================="
python tests/test_streamlit_app.py 2>/dev/null
echo ""

echo "=========================================="
echo "  ✅ TOUS LES TESTS PASSÉS"
echo "=========================================="
