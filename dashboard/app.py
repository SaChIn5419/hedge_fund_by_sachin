import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from datetime import datetime
import pandas as pd

from data_layer.loader import load_trade_log, compute_daily_summary

# Pre-load to get current regime
trades = load_trade_log()
daily = compute_daily_summary(trades)
current_regime = daily['market_state'].iloc[-1] if not daily.empty else "UNKNOWN"
last_update = daily['date'].iloc[-1].strftime('%Y-%m-%d') if not daily.empty else "N/A"

app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Chimera Quant Desk"

sidebar = html.Div(
    [
        html.H3("CHIMERA v2", className="display-6", style={"color": "#fff", "marginBottom": "20px"}),
        html.Hr(),
        html.P("Quant Desk Dashboard", className="lead", style={"color": "#8B949E", "fontSize": "0.9rem"}),
        dbc.Nav(
            [
                dbc.NavLink(
                    [html.I(className="bi bi-house-door me-2"), f"{page['name']}"],
                    href=page["relative_path"],
                    active="exact",
                    style={"color": "#8B949E"}
                )
                for page in dash.page_registry.values()
            ],
            vertical=True,
            pills=True,
        ),
    ],
    className="sidebar"
)

navbar = html.Div(
    [
        html.Div(
            [
                html.Span("Live Date: ", style={"color": "#8B949E"}),
                html.Span(datetime.now().strftime('%Y-%m-%d %H:%M'), className="monospace"),
            ],
            style={"marginRight": "20px"}
        ),
        html.Div(
            [
                html.Span("Last Refresh: ", style={"color": "#8B949E"}),
                html.Span(last_update, className="monospace"),
            ],
            style={"marginRight": "auto"}
        ),
        html.Div(
            [
                html.Span("Current Regime: ", style={"color": "#8B949E", "marginRight": "10px"}),
                html.Span(current_regime, className=f"regime-badge {current_regime.lower()}")
            ]
        )
    ],
    className="navbar"
)

app.layout = html.Div(
    [
        sidebar,
        html.Div(
            [
                navbar,
                dash.page_container
            ],
            className="content"
        )
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
