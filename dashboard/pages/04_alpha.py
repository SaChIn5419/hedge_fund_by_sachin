import dash
from dash import html

dash.register_page(__name__, path='/alpha', name='Alpha & Signal Lab')

layout = html.Div(
    [
        html.H2("Alpha & Signal Lab"),
        html.Div("FIP Score Leaderboard and Factor Contribution metrics go here.", className="card")
    ]
)
