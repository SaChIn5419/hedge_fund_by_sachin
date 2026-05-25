import dash
from dash import html, dcc
import plotly.graph_objects as go
import pandas as pd
from data_layer.loader import load_trade_log, compute_daily_summary, get_kpis

dash.register_page(__name__, path='/', name='Executive Summary')

trades = load_trade_log()
daily = compute_daily_summary(trades)
kpis = get_kpis(daily)

def create_kpi_card(title, value):
    return html.Div(
        [
            html.Div(title, className="kpi-title"),
            html.Div(value, className="kpi-value kpi-neutral")
        ],
        className="card kpi-tile"
    )

def create_equity_chart(daily):
    if daily.empty:
        return go.Figure()

    fig = go.Figure()

    # Base equity curve
    fig.add_trace(go.Scatter(
        x=daily['date'],
        y=daily['equity'],
        mode='lines',
        name='Chimera',
        line=dict(color='#00B4D8', width=2)
    ))

    # Add regime background shading
    regime_colors = {'BULL': 'rgba(0, 255, 136, 0.1)', 'BEAR': 'rgba(255, 69, 96, 0.1)', 'CHOP': 'rgba(255, 193, 7, 0.1)'}

    # A simple way to add rects: loop through changes
    shapes = []
    if not daily.empty and 'market_state' in daily.columns:
        last_state = daily['market_state'].iloc[0]
        start_date = daily['date'].iloc[0]

        for i, row in daily.iterrows():
            if row['market_state'] != last_state:
                shapes.append(dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=start_date,
                    y0=0,
                    x1=row['date'],
                    y1=1,
                    fillcolor=regime_colors.get(last_state, 'rgba(128, 128, 128, 0.1)'),
                    opacity=1,
                    layer="below",
                    line_width=0,
                ))
                start_date = row['date']
                last_state = row['market_state']

        # Add the final shape
        shapes.append(dict(
            type="rect",
            xref="x",
            yref="paper",
            x0=start_date,
            y0=0,
            x1=daily['date'].iloc[-1],
            y1=1,
            fillcolor=regime_colors.get(last_state, 'rgba(128, 128, 128, 0.1)'),
            opacity=1,
            layer="below",
            line_width=0,
        ))

    fig.update_layout(
        shapes=shapes,
        template='plotly_dark',
        plot_bgcolor='#0D1117',
        paper_bgcolor='#161B22',
        margin=dict(l=40, r=20, t=40, b=40),
        title="Equity Curve with Regime Shading",
        font=dict(color='#E6EDF3', family="JetBrains Mono, Courier New, monospace"),
        yaxis_type="log"
    )
    return fig

layout = html.Div(
    [
        html.H2("Morning Brief"),

        # KPI Bar
        html.Div(
            [
                create_kpi_card(k, v) for k, v in kpis.items()
            ],
            style={"display": "flex", "justifyContent": "space-between", "gap": "15px", "marginBottom": "20px"}
        ),

        # Equity Curve Chart
        html.Div(
            dcc.Graph(figure=create_equity_chart(daily), style={'height': '500px'}),
            className="card"
        ),

        # Placeholder for bottom row (Heatmap, etc)
        html.Div(
            [
                html.Div("Monthly Returns (Coming Soon)", className="card", style={"flex": 1}),
                html.Div("Return Distribution (Coming Soon)", className="card", style={"flex": 1})
            ],
            style={"display": "flex", "gap": "20px"}
        )
    ]
)
