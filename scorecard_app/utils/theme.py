"""Palette BNP Paribas et style global de l'application."""

# Palette BNP — vert primaire signature
BNP_GREEN = "#00965E"
BNP_GREEN_DARK = "#006B43"
BNP_GREEN_LIGHT = "#E6F5EE"

# Décision (sémantique métier)
DECISION_GREEN = "#4CAF50"
DECISION_ORANGE = "#FF9800"
DECISION_RED = "#E53935"

# Neutres
TEXT_PRIMARY = "#1A1A1A"
TEXT_SECONDARY = "#6B7280"
BG_SURFACE = "#FFFFFF"
BG_PAGE = "#FAFAFA"
BORDER = "#E5E7EB"

# Plotly — template centralisé
PLOTLY_LAYOUT = {
    "font": {"family": "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
             "color": TEXT_PRIMARY, "size": 12},
    "paper_bgcolor": BG_SURFACE,
    "plot_bgcolor": BG_SURFACE,
    "colorway": [BNP_GREEN, "#0066CC", "#F59E0B", "#8B5CF6", DECISION_RED, TEXT_SECONDARY],
    "xaxis": {"gridcolor": BORDER, "linecolor": BORDER, "zerolinecolor": BORDER},
    "yaxis": {"gridcolor": BORDER, "linecolor": BORDER, "zerolinecolor": BORDER},
    "margin": {"l": 60, "r": 30, "t": 50, "b": 50},
}


def inject_css() -> str:
    """CSS à injecter dans Streamlit pour le branding BNP."""
    return f"""
    <style>
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {BG_SURFACE};
        border-right: 1px solid {BORDER};
    }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {{
        color: {BNP_GREEN_DARK};
    }}

    /* Headers */
    h1, h2, h3 {{
        color: {TEXT_PRIMARY};
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    h1 {{
        border-bottom: 3px solid {BNP_GREEN};
        padding-bottom: 8px;
    }}

    /* Boutons primaires */
    .stButton > button {{
        background-color: {BNP_GREEN};
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: 500;
    }}
    .stButton > button:hover {{
        background-color: {BNP_GREEN_DARK};
        color: white;
    }}

    /* Metric cards */
    [data-testid="stMetric"] {{
        background-color: {BG_SURFACE};
        border: 1px solid {BORDER};
        border-left: 4px solid {BNP_GREEN};
        padding: 12px 16px;
        border-radius: 4px;
    }}

    /* Expander */
    .streamlit-expanderHeader {{
        background-color: {BG_SURFACE};
        font-weight: 500;
    }}

    /* Décision blocks */
    .decision-block {{
        padding: 16px;
        border-radius: 6px;
        color: white;
        text-align: center;
    }}
    .decision-vert {{ background-color: {DECISION_GREEN}; }}
    .decision-orange {{ background-color: {DECISION_ORANGE}; }}
    .decision-rouge {{ background-color: {DECISION_RED}; }}
    .decision-block .label {{
        font-size: 14px; font-weight: 600; letter-spacing: 0.5px;
        text-transform: uppercase; opacity: 0.95;
    }}
    .decision-block .value {{
        font-size: 28px; font-weight: 700; margin-top: 4px;
    }}
    .decision-block .sub {{
        font-size: 12px; opacity: 0.9; margin-top: 6px;
    }}

    /* Profile card */
    .profile-row {{
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid {BORDER};
    }}
    .profile-row:last-child {{ border-bottom: none; }}
    .profile-row .var {{ color: {TEXT_SECONDARY}; font-size: 13px; }}
    .profile-row .bin {{ color: {TEXT_PRIMARY}; font-weight: 500; font-size: 13px; }}
    .profile-row .pts-pos {{ color: {DECISION_GREEN}; font-weight: 600; font-size: 13px; }}
    .profile-row .pts-neg {{ color: {DECISION_RED}; font-weight: 600; font-size: 13px; }}
    .profile-row .pts-zero {{ color: {TEXT_SECONDARY}; font-weight: 500; font-size: 13px; }}

    /* Hide Streamlit chrome */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    </style>
    """
