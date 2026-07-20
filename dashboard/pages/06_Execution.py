import dash
from dash import html, dcc
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from data_layer.loader import load_trade_log

dash.register_page(__name__, path='/06-execution', name='Execution')

trades = load_trade_log()

def create_metric_panel(title, value, sub="", color_class="info"):
    return html.Div(
        [
            html.Div(title, className="panel-title"),
            html.Div(value, className=f"metric-val {color_class}"),
            html.Div(sub, className="metric-sub") if sub else None
        ],
        className="os-panel"
    )

def create_slippage_histogram():
    # Mocking slippage distribution
    slippage = np.random.normal(0.05, 0.02, 500)
    
    fig = go.Figure(go.Histogram(
        x=slippage,
        nbinsx=40,
        marker_color='#FFC107'
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=20, t=30, b=30),
        font=dict(family="JetBrains Mono", size=10, color="#8B949E"),
        title="SLIPPAGE DISTRIBUTION (BPS)",
        title_font_size=11,
        bargap=0.1
    )
    return fig

layout = html.Div(
    [
        html.H3("EXECUTION & BROKER OPERATIONS"),
        
        # KPI Row
        html.Div(
            [
                create_metric_panel("FILL RATE", "98.4%", "Last 30 Days", "bull"),
                create_metric_panel("AVG SLIPPAGE", "0.05%", "vs Entry Signal", "chop"),
                create_metric_panel("REJECTED ORDERS", "0", "0% fail rate", "bull"),
                create_metric_panel("API LATENCY", "42ms", "Broker Gateway", "info"),
            ],
            className="os-grid grid-4col", style={"marginBottom": "15px"}
        ),
        
        # Content
        html.Div(
            [
                html.Div(
                    [
                        html.Div("TRADE LOG (RECENT)", className="panel-title"),
                        dash.dash_table.DataTable(
                            data=trades.sort_values('date', ascending=False).head(20).to_dict('records') if not trades.empty else [],
                            columns=[
                                {"name": "DATE", "id": "date", "type": "datetime"},
                                {"name": "TICKER", "id": "ticker"},
                                {"name": "WEIGHT", "id": "weight", "type": "numeric", "format": dash.dash_table.FormatTemplate.percentage(1).sign(dash.dash_table.FormatTemplate.Sign.positive)},
                                {"name": "PNL", "id": "net_pnl", "type": "numeric", "format": dash.dash_table.Format.Format(precision=0, scheme=dash.dash_table.Format.Scheme.fixed).symbol(dash.dash_table.Format.Symbol.yes).symbol_prefix('₹')}
                            ],
                            style_table={'overflowX': 'auto', 'height': '400px', 'overflowY': 'auto'},
                            style_cell={'textAlign': 'left'},
                            style_data_conditional=[
                                {'if': {'filter_query': '{net_pnl} > 0', 'column_id': 'net_pnl'}, 'color': '#00FF88'},
                                {'if': {'filter_query': '{net_pnl} < 0', 'column_id': 'net_pnl'}, 'color': '#FF4560'}
                            ]
                        )
                    ],
                    className="os-panel", style={"gridColumn": "span 2"}
                ),
                
                html.Div(
                    [
                        dcc.Graph(figure=create_slippage_histogram(), config={'displayModeBar': False}, style={"height": "400px"})
                    ],
                    className="os-panel", style={"gridColumn": "span 1"}
                )
            ],
            className="os-grid grid-3col"
        )
    ]
)
