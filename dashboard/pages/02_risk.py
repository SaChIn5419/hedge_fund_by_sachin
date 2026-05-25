import dash
from dash import html

dash.register_page(__name__, path='/risk', name='Risk Analytics')

layout = html.Div(
    [
        html.H2("Risk Analytics"),
        html.Div("Rolling Risk Panel & VaR/CVaR tables will be populated here.", className="card")
    ]
)
