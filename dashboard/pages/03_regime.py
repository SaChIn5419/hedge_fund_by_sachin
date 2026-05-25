import dash
from dash import html, dcc, dash_table
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from data_layer.loader import load_regime_trace, load_trade_log

dash.register_page(__name__, path='/regime', name='Regime Monitor')

regime_trace = load_regime_trace()
trades = load_trade_log()

def create_regime_timeline(regime_df):
    if regime_df.empty:
        return go.Figure()

    fig = go.Figure()

    colors = {'BULL': '#00FF88', 'BEAR': '#FF4560', 'CHOP': '#FFC107'}

    # We will plot a simple horizontal bar chart or timeline by making one continuous line with changing colors,
    # but Plotly express timeline is best. To keep it simple in GO, we'll draw rects or scatter segments.
    # Alternatively, use a scatter with mode='markers' or 'lines' for quick visualization.
    fig.add_trace(go.Scatter(
        x=regime_df['date'],
        y=regime_df['regime_confidence'],
        mode='lines',
        name='Confidence',
        line=dict(color='#8B949E')
    ))

    # Add background shapes
    shapes = []
    last_state = regime_df['regime'].iloc[0]
    start_date = regime_df['date'].iloc[0]
    for i, row in regime_df.iterrows():
        if row['regime'] != last_state:
            shapes.append(dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=start_date,
                y0=0,
                x1=row['date'],
                y1=1,
                fillcolor=colors.get(last_state, '#888'),
                opacity=0.2,
                layer="below",
                line_width=0,
            ))
            start_date = row['date']
            last_state = row['regime']

    shapes.append(dict(
        type="rect",
        xref="x",
        yref="paper",
        x0=start_date,
        y0=0,
        x1=regime_df['date'].iloc[-1],
        y1=1,
        fillcolor=colors.get(last_state, '#888'),
        opacity=0.2,
        layer="below",
        line_width=0,
    ))

    fig.update_layout(
        shapes=shapes,
        template='plotly_dark',
        plot_bgcolor='#0D1117',
        paper_bgcolor='#161B22',
        title="Regime Timeline & Confidence",
        font=dict(color='#E6EDF3')
    )
    return fig

def get_regime_performance(trades):
    if trades.empty:
        return pd.DataFrame()

    # Only active trades
    active = trades[trades['weight'].abs() > 1e-12]
    if active.empty:
        return pd.DataFrame()

    perf = active.groupby('market_state').agg(
        trades=('net_pnl', 'count'),
        total_pnl=('net_pnl', 'sum'),
        avg_gross=('gross_weight', 'mean'),
        avg_conf=('regime_confidence', 'mean')
    ).reset_index()

    return perf

regime_perf_df = get_regime_performance(trades)

layout = html.Div(
    [
        html.H2("Regime Monitor"),
        html.Div(
            dcc.Graph(figure=create_regime_timeline(regime_trace)),
            className="card"
        ),
        html.Div(
            [
                html.H4("Regime Conditional Performance"),
                dash_table.DataTable(
                    data=regime_perf_df.to_dict('records') if not regime_perf_df.empty else [],
                    columns=[{"name": i, "id": i} for i in regime_perf_df.columns] if not regime_perf_df.empty else [],
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
        )
    ]
)
