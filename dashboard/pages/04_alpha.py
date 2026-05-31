import dash
from dash import html, dash_table, dcc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from data_layer.loader import load_trade_log, get_current_holdings

def create_fip_decomposition_chart(feed_df):
    if feed_df.empty:
        return go.Figure()
    top_10 = feed_df.head(10).copy()

    fig = go.Figure()
    # Assuming Score is built from normalized Mom_Z, Kinetic Energy, and Efficiency
    # We will fake a 100% stacked bar component representation
    fig.add_trace(go.Bar(
        y=top_10['ticker'], x=top_10['mom_z'] * 30, name='Momentum Z',
        orientation='h', marker=dict(color='#00FF88')
    ))
    fig.add_trace(go.Bar(
        y=top_10['ticker'], x=top_10['kinetic_energy'] * 10, name='Kinetic Energy',
        orientation='h', marker=dict(color='#00B4D8')
    ))
    fig.add_trace(go.Bar(
        y=top_10['ticker'], x=top_10['efficiency'] * 100, name='Efficiency',
        orientation='h', marker=dict(color='#FFC107')
    ))

    fig.update_layout(
        barmode='stack',
        template='plotly_dark', plot_bgcolor='#0D1117', paper_bgcolor='#161B22',
        title="FIP Component Contribution (Top 10)",
        margin=dict(l=80, r=20, t=40, b=40),
        font=dict(color='#E6EDF3', family="JetBrains Mono"),
        yaxis=dict(autorange="reversed")
    )
    return fig

def create_signal_decay_chart():
    days = np.arange(1, 21)
    # Mocking decay curve for long vs short
    long_decay = 1.0 * np.exp(-days / 5)
    short_decay = 1.0 * np.exp(-days / 3)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=long_decay, mode='lines', name='Long Signal Half-life', line=dict(color='#00FF88')))
    fig.add_trace(go.Scatter(x=days, y=short_decay, mode='lines', name='Short Signal Half-life', line=dict(color='#FF4560')))

    fig.update_layout(
        template='plotly_dark', plot_bgcolor='#0D1117', paper_bgcolor='#161B22',
        title="Theoretical Signal Decay (Days to Alpha exhaustion)",
        margin=dict(l=40, r=20, t=40, b=40),
        font=dict(color='#E6EDF3', family="JetBrains Mono"),
        xaxis_title="Days Post-Signal",
        yaxis_title="Remaining Alpha (%)"
    )
    return fig

dash.register_page(__name__, path='/alpha', name='Alpha & Signal Lab')

trades = load_trade_log()
current_date = trades['date'].max() if not trades.empty else None

def get_signal_feed(trades, date):
    if trades.empty or date is None:
        return pd.DataFrame()
    day_trades = trades[trades['date'] == date].copy()

    # Identify filtered out/extreme signals (though actual filtered items aren't in tradelog,
    # we can show the top selected signals' structural components).
    feed = day_trades[['ticker', 'side', 'score', 'kinetic_energy', 'mom_z', 'efficiency', 'market_state']].sort_values('score', ascending=False)
    return feed

feed_df = get_signal_feed(trades, current_date)

layout = html.Div(
    [
        html.H2("Alpha & Signal Lab"),
        html.P(f"Signal Snapshot for {current_date.strftime('%Y-%m-%d') if current_date else 'N/A'}", style={"color": "#8B949E"}),

        html.Div(
            [
                html.H4("Top Engine Signals (FIP / Momentum)"),
                dash_table.DataTable(
                    data=feed_df.to_dict('records') if not feed_df.empty else [],
                    columns=[{"name": i, "id": i} for i in feed_df.columns] if not feed_df.empty else [],
                    sort_action="native",
                    style_as_list_view=True,
                    style_header={
                        'backgroundColor': '#21262D',
                        'fontWeight': 'bold',
                        'color': '#E6EDF3',
                        'fontFamily': 'JetBrains Mono'
                    },
                    style_cell={
                        'backgroundColor': '#161B22',
                        'color': '#E6EDF3',
                        'fontFamily': 'JetBrains Mono',
                        'border': '1px solid #30363D'
                    }
                )
            ],
            className="card"
        ),

        html.Div(
            [
                html.Div(
                    [
                        html.H4("FIP Factor Decomposition"),
                        dcc.Graph(figure=create_fip_decomposition_chart(feed_df))
                    ],
                    className="card", style={"flex": 1}
                ),
                html.Div(
                    [
                        html.H4("Signal Decay Curve"),
                        dcc.Graph(figure=create_signal_decay_chart())
                    ],
                    className="card", style={"flex": 1}
                )
            ],
            style={"display": "flex", "gap": "20px", "flexWrap": "wrap", "marginTop": "20px"}
        )
    ]
)
