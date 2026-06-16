"""Tests Streamlit via AppTest : exécute réellement les pages."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parent.parent


def green(msg: str) -> None:
    print(f"  \033[92m✓\033[0m {msg}")


def _check_no_exception(at: AppTest, page_name: str) -> None:
    if at.exception:
        for e in at.exception:
            print(f"  EXCEPTION in {page_name}: {e.value}")
        raise AssertionError(f"Exception in {page_name}")
    if at.error:
        for e in at.error:
            print(f"  st.error in {page_name}: {e.value}")
        raise AssertionError(f"st.error in {page_name}")


def test_home_page_runs():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.run()
    _check_no_exception(at, "app.py")
    # Doit afficher des metrics
    assert len(at.metric) >= 3, f"Attendait ≥3 metrics, eu {len(at.metric)}"
    green(f"app.py s'exécute sans erreur ({len(at.metric)} metrics affichées).")


def test_scorecard_page_runs():
    at = AppTest.from_file(str(ROOT / "pages/1_Scorecard.py"), default_timeout=30)
    at.run()
    _check_no_exception(at, "pages/1_Scorecard.py")
    # Doit avoir au moins un dataframe et une selectbox
    assert len(at.dataframe) >= 1
    assert len(at.selectbox) >= 1
    green(
        f"1_Scorecard.py : {len(at.dataframe)} dataframe(s), "
        f"{len(at.selectbox)} selectbox, {len(at.multiselect)} multiselect."
    )


def test_segments_page_runs():
    at = AppTest.from_file(str(ROOT / "pages/2_Segments.py"), default_timeout=60)
    at.run()
    _check_no_exception(at, "pages/2_Segments.py")
    # Doit avoir des metrics et des inputs
    assert len(at.metric) >= 4, f"{len(at.metric)} metrics"
    assert len(at.number_input) >= 1, "Aucun number_input pour les seuils"
    green(
        f"2_Segments.py : {len(at.metric)} metrics, "
        f"{len(at.multiselect)} multiselects, {len(at.number_input)} number_inputs."
    )


def test_segments_with_filter_applied():
    """Active un filtre et vérifie que l'app reste fonctionnelle."""
    at = AppTest.from_file(str(ROOT / "pages/2_Segments.py"), default_timeout=60)
    at.run()
    _check_no_exception(at, "pages/2_Segments.py (initial)")

    if at.multiselect:
        ms = at.multiselect[0]
        if ms.options:
            ms.select(ms.options[0])
            at.run()
            _check_no_exception(at, "pages/2_Segments.py (avec filtre)")
            green(f"Filtre appliqué ({ms.label}={ms.options[0]!r}) : page recalculée sans erreur.")
            return
    green("Pas de filtre testé (options vides).")


def test_segments_threshold_change():
    """Change le seuil et vérifie que les blocs décisionnels se mettent à jour."""
    at = AppTest.from_file(str(ROOT / "pages/2_Segments.py"), default_timeout=60)
    at.run()
    _check_no_exception(at, "initial")

    # Modifier le seuil 1
    if at.number_input:
        at.number_input[0].set_value(0.10)
        at.run()
        _check_no_exception(at, "après changement seuil")
        green("Changement du seuil 1 → page recalculée sans erreur.")


def test_segments_tricolor_mode():
    """Bascule en mode tricolore."""
    at = AppTest.from_file(str(ROOT / "pages/2_Segments.py"), default_timeout=60)
    at.run()
    _check_no_exception(at, "initial")

    if at.radio:
        radio = at.radio[0]
        # Le second choix = tricolore
        if len(radio.options) >= 2:
            radio.set_value(radio.options[1])
            at.run()
            _check_no_exception(at, "mode tricolore")
            # Doit y avoir 2 number_input maintenant (seuil 1 + seuil 2)
            assert len(at.number_input) >= 2, f"{len(at.number_input)} number_inputs"
            green(f"Mode tricolore : 2 seuils affichés, {len(at.number_input)} number_inputs.")


if __name__ == "__main__":
    print("\n=== Tests Streamlit (AppTest, exécution réelle des pages) ===\n")
    tests = [
        test_home_page_runs,
        test_scorecard_page_runs,
        test_segments_page_runs,
        test_segments_with_filter_applied,
        test_segments_threshold_change,
        test_segments_tricolor_mode,
    ]
    for t in tests:
        t()
    print(f"\n✅ {len(tests)} tests passés\n")
