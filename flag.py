def df_to_excel(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_export = df[["segment", "feu", "nb_dossiers", "repartition",
                         "nb_defaut", "tx_defaut", "montant", "mtn_defaut", "mtn_defaut_pct"]]

        wb = writer.book
        ws = wb.add_worksheet("Diagnostic VR")

        # ── Formats ───────────────────────────────────────────────────────
        fmt_titre  = wb.add_format({"bold": True, "font_size": 13, "font_color": "#1a1a2e"})
        fmt_header = wb.add_format({"bold": True, "bg_color": "#1a1a2e", "font_color": "white",
                                     "border": 1, "align": "center", "valign": "vcenter"})
        fmt_seg    = wb.add_format({"bold": True, "bg_color": "#f0f0f5", "font_color": "#1a1a2e",
                                     "border": 1, "valign": "vcenter", "text_wrap": True})
        fmt_vert   = wb.add_format({"bg_color": "#d5f5e3", "font_color": "#1e8449", "border": 1, "valign": "vcenter"})
        fmt_orange = wb.add_format({"bg_color": "#fdebd0", "font_color": "#d35400", "border": 1, "valign": "vcenter"})
        fmt_rouge  = wb.add_format({"bg_color": "#fadbd8", "font_color": "#c0392b", "border": 1, "valign": "vcenter"})
        fmt_total  = wb.add_format({"bold": True, "bg_color": "#eaf0fb", "border": 1, "valign": "vcenter"})

        def fmt_feu(feu: str):
            return (fmt_vert   if feu == "Vert"   else
                    fmt_orange if feu == "Orange"  else
                    fmt_rouge  if feu == "Rouge"   else
                    fmt_total)

        # ── Largeurs colonnes ─────────────────────────────────────────────
        ws.set_column(0, 0, 22)
        ws.set_column(1, 1, 10)
        ws.set_column(2, 8, 16)

        # ── Titre ─────────────────────────────────────────────────────────
        ws.merge_range(0, 0, 0, 8, "Situation actuelle — Diagnostic Vert / Rouge", fmt_titre)

        # ── En-têtes ──────────────────────────────────────────────────────
        headers = ["Segment", "Feu", "Nb dossiers", "Répartition (%)", "Nb défaut",
                   "Taux défaut (%)", "Montant (m€)", "Mtn défaut (m€)", "Mtn défaut (%)"]
        for col_idx, h in enumerate(headers):
            ws.write(1, col_idx, h, fmt_header)

        # ── Données groupées par segment ──────────────────────────────────
        row_excel = 2  # ligne courante dans Excel (0-indexé)

        for segment, groupe in df_export.groupby("segment", sort=False):
            n_lignes = len(groupe)

            # Fusion des cellules de la colonne segment
            if n_lignes > 1:
                ws.merge_range(
                    row_excel, 0,
                    row_excel + n_lignes - 1, 0,
                    segment, fmt_seg
                )
            else:
                ws.write(row_excel, 0, segment, fmt_seg)

            # Écriture des lignes du groupe
            for _, row in groupe.iterrows():
                feu = row["feu"]
                fmt = fmt_feu(feu)

                ws.write(row_excel, 1, feu,                fmt)
                ws.write(row_excel, 2, row["nb_dossiers"], fmt)
                ws.write(row_excel, 3, row["repartition"], fmt)
                ws.write(row_excel, 4, row["nb_defaut"],   fmt)
                ws.write(row_excel, 5, row["tx_defaut"],   fmt)
                ws.write(row_excel, 6, row["montant"],     fmt)
                ws.write(row_excel, 7, row["mtn_defaut"],  fmt)
                ws.write(row_excel, 8, row["mtn_defaut_pct"], fmt)

                row_excel += 1

        # ── Hauteur des lignes ─────────────────────────────────────────────
        for i in range(2, row_excel):
            ws.set_row(i, 18)

    return output.getvalue()
