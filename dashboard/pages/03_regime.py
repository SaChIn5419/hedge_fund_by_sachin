import dash
from dash import html, dcc, dash_table
import plotly.graph_objects as go
import pandas as pd
from data_layer.loader import load_regime_trace, load_trade_log

dash.register_page(__name__, path='/regime', name='Regime Monitor')

regime_trace = load_regime_trace()
trades = load_trade_log()

def create_regime_timeline(regime_df):
    if regime_df.empty:
        return go.Figure()

    fig = go.Figure()
    colors = {'BULL': '#00FF88', 'BEAR': '#FF4560', 'CHOP': '#FFC107'}

    fig.add_trace(go.Scatter(
        x=regime_df['date'],
        y=regime_df['regime_confidence'],
        mode='lines',
        name='Confidence',
        line=dict(color='#8B949E')
    ))

    shapes = []
    last_state = regime_df['regime'].iloc[0]
    start_date = regime_df['date'].iloc[0]
    for i, row in regime_df.iterrows():
        if row['regime'] != last_state:
            shapes.append(dict(
                type="rect", xref="x", yref="paper",
                x0=start_date, y0=0, x1=row['date'], y1=1,
                fillcolor=colors.get(last_state, '#888'), opacity=0.2, layer="below", line_width=0,
            ))
            start_date = row['date']
            last_state = row['regime']

    shapes.append(dict(
        type="rect", xref="x", yref="paper",
        x0=start_date, y0=0, x1=regime_df['date'].iloc[-1], y1=1,
        fillcolor=colors.get(last_state, '#888'), opacity=0.2, layer="below", line_width=0,
    ))

    fig.update_layout(
        shapes=shapes, template='plotly_dark', plot_bgcolor='#0D1117', paper_bgcolor='#161B22',
        title="Regime Timeline & Confidence", font=dict(color='#E6EDF3'), margin=dict(l=40, r=20, t=40, b=40)
    )
    return fig

def create_macro_inputs_chart(regime_df):
    if regime_df.empty:
        return go.Figure()

    fig = go.Figure()
    if 'breadth' in regime_df.columns:
        fig.add_trace(go.Scatter(x=regime_df['date'], y=regime_df['breadth'], mode='lines', name='Breadth', line=dict(color='#00B4D8')))
    if 'macro_score' in regime_df.columns:
        fig.add_trace(go.Scatter(x=regime_df['date'], y=regime_df['macro_score'], mode='lines', name='Macro Score', line=dict(color='#FFC107')))

    fig.update_layout(
        template='plotly_dark', plot_bgcolor='#0D1117', paper_bgcolor='#161B22',
        title="Macro Inputs (Breadth & Score)", font=dict(color='#E6EDF3'), margin=dict(l=40, r=20, t=40, b=40)
    )
    return fig

def get_regime_performance(trades):
    if trades.empty:
        return pd.DataFrame()
    active = trades[trades['weight'].abs() > 1e-12]
    if active.empty:
        return pd.DataFrame()
    perf = active.groupby('market_state').agg(
        trades=('net_pnl', 'count'),
        total_pnl=('net_pnl', 'sum'),
        avg_gross=('gross_weight', 'mean'),
        avg_conf=('regime_confidence', 'mean')
    ).reset_index()
    # Format
    perf['total_pnl'] = perf['total_pnl'].apply(lambda x: f"₹{x:,.2f}")
    perf['avg_gross'] = perf['avg_gross'].apply(lambda x: f"{x:.3f}")
    perf['avg_conf'] = perf['avg_conf'].apply(lambda x: f"{x:.2f}")
    return perf

def get_transition_matrix(regime_df):
    if regime_df.empty:
        return pd.DataFrame()
    states = regime_df['regime'].dropna().astype(str).tolist()
    if len(states) < 2:
        return pd.DataFrame()
    pairs = list(zip(states[:-1], states[1:]))
    df = pd.DataFrame(pairs, columns=['from', 'to'])
    table = pd.crosstab(df['from'], df['to'])
    table.reset_index(inplace=True)
    return table

regime_perf_df = get_regime_performance(trades)
transition_df = get_transition_matrix(regime_trace)

layout = html.Div(
    [
        html.H2("Regime Monitor"),
        html.Div(dcc.Graph(figure=create_regime_timeline(regime_trace)), className="card"),

        html.Div(
            [
                html.Div(dcc.Graph(figure=create_macro_inputs_chart(regime_trace)), className="card", style={"flex": 1}),
                html.Div(
                    [
                        html.H4("Transition Matrix"),
                        dash_table.DataTable(
                            data=transition_df.to_dict('records') if not transition_df.empty else [],
                            columns=[{"name": i, "id": i} for i in transition_df.columns] if not transition_df.empty else [],
                            style_as_list_view=True,
                            style_header={'backgroundColor': '#21262D', 'fontWeight': 'bold', 'color': '#E6EDF3', 'fontFamily': 'JetBrains Mono'},
                            style_cell={'backgroundColor': '#161B22', 'color': '#E6EDF3', 'fontFamily': 'JetBrains Mono', 'border': '1px solid #30363D'}
                        )
                    ],
                    className="card", style={"flex": 1}
                )
            ],
            style={"display": "flex", "gap": "20px", "flexWrap": "wrap"}
        ),

        html.Div(
            [
                html.H4("Regime Conditional Performance"),
                dash_table.DataTable(
                    data=regime_perf_df.to_dict('records') if not regime_perf_df.empty else [],
                    columns=[{"name": i, "id": i} for i in regime_perf_df.columns] if not regime_perf_df.empty else [],
                    style_as_list_view=True,
                    style_header={'backgroundColor': '#21262D', 'fontWeight': 'bold', 'color': '#E6EDF3', 'fontFamily': 'JetBrains Mono'},
                    style_cell={'backgroundColor': '#161B22', 'color': '#E6EDF3', 'fontFamily': 'JetBrains Mono', 'border': '1px solid #30363D'}
                )
            ],
            className="card"
        )
    ]
)
