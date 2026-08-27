with st.form("params"):
    cible = st.slider("Taux de défaut cible", 0.0, 0.3, 0.05)
    n_min = st.number_input("Volume minimal", 0)
    lance = st.form_submit_button("Lancer")

if lance:
    st.session_state["seuil"] = seuil_pour_defaut(None, cible, n_min=n_min, table=t)
