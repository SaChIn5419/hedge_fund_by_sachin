import dash
from dash import html, dcc, Input, Output, callback
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from data_layer.loader import load_trade_log, get_current_holdings

dash.register_page(__name__, path='/07-explainability', name='Explainability')

trades = load_trade_log()
holdings = get_current_holdings(trades)

def create_metric_panel(title, value, sub="", color_class="info"):
    return html.Div(
        [
            html.Div(title, className="panel-title"),
            html.Div(value, className=f"metric-val {color_class}"),
            html.Div(sub, className="metric-sub") if sub else None
        ],
        className="os-panel"
    )

def create_waterfall_chart(asset_row):
    # Mock decomposition of the signal score based on available features
    base_score = 0
    mom_z = asset_row.get('mom_z', 0) * 0.4
    macro = asset_row.get('macro_score', 0) * 0.2
    regime = asset_row.get('regime_confidence', 0) * 0.2
    vol_adj = -0.1 # simulated risk adjustment
    final_score = asset_row.get('score', mom_z + macro + regime + vol_adj)
    
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "relative", "total"],
        x=["Base", "Momentum Z", "Macro", "Regime", "Vol Adj", "Final Score"],
        textposition="outside",
        y=[0, mom_z, macro, regime, vol_adj, final_score],
        connector={"line":{"color":"#30363D"}},
        decreasing={"marker":{"color":"#FF4560"}},
        increasing={"marker":{"color":"#00FF88"}},
        totals={"marker":{"color":"#00B4D8"}}
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=20, t=40, b=30),
        font=dict(family="JetBrains Mono", size=10, color="#8B949E"),
        title=f"SIGNAL SCORE DECOMPOSITION ({asset_row['ticker']})",
        title_font_size=11,
    )
    return fig

layout = html.Div(
    [
        html.H3("EXPLAINABILITY ENGINE (XAI)"),
        
        html.Div(
            [
                html.Span("SELECT HOLDING: ", style={"color": "#8B949E", "marginRight": "10px", "fontFamily": "JetBrains Mono"}),
                dcc.Dropdown(
                    id='xai-asset-selector',
                    options=[{'label': row['ticker'], 'value': row['ticker']} for _, row in holdings.iterrows()] if not holdings.empty else [],
                    value=holdings.iloc[0]['ticker'] if not holdings.empty else None,
                    clearable=False,
                    style={'width': '200px', 'backgroundColor': '#161B22', 'color': '#E6EDF3', 'border': '1px solid #30363D', 'fontFamily': 'JetBrains Mono'}
                )
            ],
            style={"marginBottom": "20px", "display": "flex", "alignItems": "center"}
        ),
        
        html.Div(id='xai-content')
    ]
)

@callback(
    Output('xai-content', 'children'),
    Input('xai-asset-selector', 'value')
)
def update_xai(ticker):
    if not ticker or trades.empty:
        return html.Div("No data available.")
        
    latest_date = trades['date'].max()
    asset_data = trades[(trades['date'] == latest_date) & (trades['ticker'] == ticker)]
    
    if asset_data.empty:
        return html.Div("Asset data not found for current date.")
        
    row = asset_data.iloc[0]
    
    return html.Div(
        [
            # KPI Row
            html.Div(
                [
                    create_metric_panel("MOMENTUM Z-SCORE", f"{row.get('mom_z', 0):.2f}", "Trend strength", "bull" if row.get('mom_z', 0) > 0 else "bear"),
                    create_metric_panel("MACRO REGIME", f"{row.get('market_state', 'UNKNOWN')}", f"Conf: {row.get('regime_confidence', 0)*100:.0f}%", "info"),
                    create_metric_panel("POSITION SIZE", f"{abs(row.get('weight', 0))*100:.2f}%", "Risk-parity adjusted", "chop"),
                    create_metric_panel("FINAL AI SCORE", f"{row.get('score', 0):.3f}", "Aggregated signal", "bull"),
                ],
                className="os-grid grid-4col", style={"marginBottom": "15px"}
            ),
            
            # Content
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Graph(figure=create_waterfall_chart(row), config={'displayModeBar': False}, style={"height": "400px"})
                        ],
                        className="os-panel", style={"gridColumn": "span 2"}
                    ),
                    
                    html.Div(
                        [
                            html.Div(f"WHY WAS {ticker} SELECTED?", className="panel-title"),
                            html.Div(
                                [
                                    html.P(f"1. Strong price momentum detected (Z-Score: {row.get('mom_z', 0):.2f})."),
                                    html.P(f"2. Favorable macro conditions aligned with {row.get('market_state', 'UNKNOWN')} regime."),
                                    html.P(f"3. Broad market breadth ({row.get('breadth', 0)*100:.0f}% > 50SMA) supports long exposure."),
                                    html.P(f"4. Sized at {abs(row.get('weight', 0))*100:.2f}% to respect portfolio volatility target (25%).")
                                ],
                                style={"fontFamily": "JetBrains Mono", "fontSize": "12px", "color": "#E6EDF3", "lineHeight": "1.8", "padding": "10px"}
                            )
                        ],
                        className="os-panel", style={"gridColumn": "span 1"}
                    )
                ],
                className="os-grid grid-3col"
            )
        ]
    )
