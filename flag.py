import io

def df_to_excel(df: pd.DataFrame) -> bytes:
    """Convertit le DataFrame agrégé en fichier Excel avec mise en forme."""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_export = df[["segment", "feu", "nb_dossiers", "repartition",
                         "nb_defaut", "tx_defaut", "montant", "mtn_defaut", "mtn_defaut_pct"]]
        df_export.to_excel(writer, index=False, sheet_name="Diagnostic VR", startrow=1)

        wb = writer.book
        ws = writer.sheets["Diagnostic VR"]

        # ── Formats ──────────────────────────────────────────────────────
        fmt_header = wb.add_format({"bold": True, "bg_color": "#1a1a2e", "font_color": "white",
                                     "border": 1, "align": "center"})
        fmt_vert   = wb.add_format({"bg_color": "#d5f5e3", "font_color": "#1e8449", "border": 1})
        fmt_orange = wb.add_format({"bg_color": "#fdebd0", "font_color": "#d35400", "border": 1})
        fmt_rouge  = wb.add_format({"bg_color": "#fadbd8", "font_color": "#c0392b", "border": 1})
        fmt_total  = wb.add_format({"bold": True, "bg_color": "#eaf0fb", "border": 1})
        fmt_normal = wb.add_format({"border": 1})
        fmt_pct    = wb.add_format({"border": 1, "num_format": "0.000%"})

        # ── En-têtes ──────────────────────────────────────────────────────
        headers = ["Segment", "Feu", "Nb dossiers", "Répartition (%)", "Nb défaut",
                   "Taux défaut (%)", "Montant (m€)", "Mtn défaut (m€)", "Mtn défaut (%)"]
        for col_idx, h in enumerate(headers):
            ws.write(1, col_idx, h, fmt_header)

        # ── Largeurs des colonnes ─────────────────────────────────────────
        ws.set_column(0, 0, 22)
        ws.set_column(1, 1, 10)
        ws.set_column(2, 8, 16)

        # ── Colorisation ligne par ligne ──────────────────────────────────
        for row_idx, (_, row) in enumerate(df_export.iterrows()):
            feu = row["feu"]
            fmt = (fmt_vert   if feu == "Vert"   else
                   fmt_orange if feu == "Orange" else
                   fmt_rouge  if feu == "Rouge"  else
                   fmt_total)
            for col_idx, val in enumerate(row):
                ws.write(row_idx + 2, col_idx, val, fmt)

        # ── Titre ─────────────────────────────────────────────────────────
        fmt_titre = wb.add_format({"bold": True, "font_size": 13, "font_color": "#1a1a2e"})
        ws.write(0, 0, "Situation actuelle — Diagnostic Vert / Rouge", fmt_titre)

    return output.getvalue()


# ── Bouton de téléchargement ──────────────────────────────────────────────
st.download_button(
    label="📥 Exporter en Excel",
    data=df_to_excel(df),
    file_name="diagnostic_VR.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
