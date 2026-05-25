import dash
from dash import html

dash.register_page(__name__, path='/validation', name='OOS Validation')

layout = html.Div(
    [
        html.H2("Walk-Forward & OOS Validation"),
        html.Div("In-Sample vs Out-Of-Sample Equity curves and statistics will be displayed here.", className="card")
    ]
)
