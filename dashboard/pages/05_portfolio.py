import dash
from dash import html, dash_table, dcc
import plotly.graph_objects as go
import pandas as pd
from data_layer.loader import load_trade_log, get_current_holdings

dash.register_page(__name__, path='/portfolio', name='Portfolio Book')

trades = load_trade_log()
current_holdings = get_current_holdings(trades)

def create_allocation_donut(holdings):
    if holdings.empty:
        return go.Figure()

    # Group by side for a simple donut
    grouped = holdings.groupby('side')['Weight %'].sum().reset_index()

    fig = go.Figure(data=[go.Pie(
        labels=grouped['side'],
        values=grouped['Weight %'].abs(),
        hole=.4,
        marker=dict(colors=['#00FF88', '#FF4560'])
    )])

    fig.update_layout(
        template='plotly_dark', plot_bgcolor='#0D1117', paper_bgcolor='#161B22',
        title="Gross Exposure Allocation (Long vs Short)",
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(color='#E6EDF3', family="JetBrains Mono")
    )
    return fig

def create_score_histogram(holdings):
    if holdings.empty:
        return go.Figure()

    fig = go.Figure(data=[go.Histogram(
        x=holdings['score'],
        marker_color='#00B4D8',
        nbinsx=10
    )])

    fig.update_layout(
        template='plotly_dark', plot_bgcolor='#0D1117', paper_bgcolor='#161B22',
        title="Holding Signal Scores (Proxy for Momentum strength)",
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(color='#E6EDF3', family="JetBrains Mono")
    )
    return fig

layout = html.Div(
    [
        html.H2("Current Portfolio Book"),
        html.Div(
            [
                html.Div(dcc.Graph(figure=create_allocation_donut(current_holdings)), className="card", style={"flex": 1}),
                html.Div(dcc.Graph(figure=create_score_histogram(current_holdings)), className="card", style={"flex": 1})
            ],
            style={"display": "flex", "gap": "20px", "marginBottom": "20px"}
        ),
        html.Div(
            [
                dash_table.DataTable(
                    data=current_holdings.to_dict('records') if not current_holdings.empty else [],
                    columns=[{"name": i, "id": i} for i in current_holdings.columns] if not current_holdings.empty else [],
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
        )
    ]
)
