import dash
from dash import html, dcc
import plotly.graph_objects as go
import pandas as pd
from data_layer.loader import load_trade_log, compute_daily_summary

dash.register_page(__name__, path='/validation', name='OOS Validation')

trades = load_trade_log()
daily = compute_daily_summary(trades)

def create_oos_split_chart(daily):
    if daily.empty:
        return go.Figure()

    # The README notes Out-of-Sample starts April 2026.
    oos_start = pd.to_datetime('2026-04-01')

    is_data = daily[daily['date'] < oos_start]
    oos_data = daily[daily['date'] >= oos_start]

    fig = go.Figure()

    if not is_data.empty:
        fig.add_trace(go.Scatter(
            x=is_data['date'], y=is_data['equity'],
            mode='lines', name='In-Sample (Training)', line=dict(color='#00B4D8')
        ))

    if not oos_data.empty:
        fig.add_trace(go.Scatter(
            x=oos_data['date'], y=oos_data['equity'],
            mode='lines', name='Out-of-Sample (Walk-Forward)', line=dict(color='#00FF88', width=3)
        ))

        # Add a vertical line to demarcate the split clearly
        fig.add_vline(x=oos_start, line_width=2, line_dash="dash", line_color="#FFC107")
        fig.add_annotation(
            x=oos_start, y=daily['equity'].max(),
            text="OOS Start", showarrow=True, arrowhead=1,
            arrowcolor="#FFC107", font=dict(color="#FFC107")
        )

    fig.update_layout(
        template='plotly_dark', plot_bgcolor='#0D1117', paper_bgcolor='#161B22',
        title="In-Sample vs Walk-Forward Validation (Log Scale)",
        margin=dict(l=40, r=20, t=40, b=40),
        font=dict(color='#E6EDF3', family="JetBrains Mono"),
        yaxis_type="log"
    )
    return fig

layout = html.Div(
    [
        html.H2("Walk-Forward & OOS Validation"),
        html.Div(
            [
                dcc.Graph(figure=create_oos_split_chart(daily))
            ],
            className="card"
        ),
        html.Div(
            [
                html.H4("Validation Stats (Dec 2019 - May 2026)"),
                html.P("Sharpe Ratio: 1.26 | Sortino Ratio: 1.94", style={"color": "#00FF88", "fontWeight": "bold", "fontFamily": "JetBrains Mono"}),
                html.P("Max Drawdown: -28.71% | CAGR: 27.09%", style={"color": "#FF4560", "fontFamily": "JetBrains Mono"}),
                html.P("Out-Of-Sample Sharpe: 3.57 (April - May 2026 Walk Forward)", style={"color": "#00B4D8", "fontWeight": "bold", "fontFamily": "JetBrains Mono"})
            ],
            className="card",
            style={"marginTop": "20px"}
        )
    ]
)
