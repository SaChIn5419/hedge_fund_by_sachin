import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from datetime import datetime

# Global application setup
app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Chimera Quant OS"

sidebar = html.Div(
    [
        html.Div("CHIMERA OS", className="brand"),
        html.Div("v2.1.0-FIP", className="version"),
        html.Hr(style={"marginBottom": "15px"}),
        
        # We define explicit icons for the specific 7 OS pages
        dbc.Nav(
            [
                dbc.NavLink(
                    [html.I(className="bi bi-pie-chart-fill me-2"), "Portfolio"],
                    href="/", active="exact"
                ),
                dbc.NavLink(
                    [html.I(className="bi bi-cpu-fill me-2"), "Strategy"],
                    href="/02-strategy", active="exact"
                ),
                dbc.NavLink(
                    [html.I(className="bi bi-shield-shaded me-2"), "Risk"],
                    href="/03-risk", active="exact"
                ),
                dbc.NavLink(
                    [html.I(className="bi bi-journal-code me-2"), "Research"],
                    href="/04-research", active="exact"
                ),
                dbc.NavLink(
                    [html.I(className="bi bi-globe2 me-2"), "Market"],
                    href="/05-market", active="exact"
                ),
                dbc.NavLink(
                    [html.I(className="bi bi-lightning-charge-fill me-2"), "Execution"],
                    href="/06-execution", active="exact"
                ),
                dbc.NavLink(
                    [html.I(className="bi bi-eye-fill me-2"), "Explainability"],
                    href="/07-explainability", active="exact"
                ),
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
                html.Span("LIVE: ", className="info"),
                html.Span(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), className="val"),
            ],
            className="status-item"
        ),
        html.Div(
            [
                html.Span("BROKER API: ", className="info"),
                html.Span("CONNECTED", className="val bull"),
            ],
            className="status-item", style={"marginRight": "auto", "marginLeft": "20px"}
        ),
        html.Div(
            [
                html.Span("REGIME: ", className="info"),
                html.Span("UNKNOWN", id="global-regime-badge", className="status-badge bg-chop")
            ],
            className="status-item"
        )
    ],
    className="navbar"
)

app.layout = html.Div(
    [
        sidebar,
        navbar,
        html.Div(
            dash.page_container,
            className="content"
        )
    ]
)

if __name__ == "__main__":
    app.run(debug=True, port=8050)
