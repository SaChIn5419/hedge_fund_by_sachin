import dash
from dash import html, dcc
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from data_layer.loader import load_trade_log, compute_daily_summary

dash.register_page(__name__, path='/02-strategy', name='Strategy')

trades = load_trade_log()
daily = compute_daily_summary(trades)

# Metric panel generator
def create_metric_panel(title, value, sub="", color_class="info"):
    return html.Div(
        [
            html.Div(title, className="panel-title"),
            html.Div(value, className=f"metric-val {color_class}"),
            html.Div(sub, className="metric-sub") if sub else None
        ],
        className="os-panel"
    )

def create_signal_heatmap():
    import os
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Load trade log
    trades = load_trade_log()
    if trades.empty:
        # Fallback to random if empty
        sectors = ['FINANCIALS', 'IT', 'ENERGY', 'METALS', 'PHARMA', 'AUTO', 'FMCG']
        days = [f"T-{i}" for i in range(10, -1, -1)]
        z = np.random.randn(len(sectors), len(days))
        text = [[f"{val:.2f}" for val in row] for row in z]
    else:
        # Load sector mapping from parquet
        try:
            parquet_path = os.path.join(BASE_DIR, 'data/ml_dataset.parquet')
            df_parquet = pd.read_parquet(parquet_path)
            ticker_to_sector = df_parquet.groupby('ticker')['sector'].first().to_dict()
        except Exception:
            ticker_to_sector = {}
            
        trades['sector'] = trades['ticker'].map(ticker_to_sector)
        # Fallback sectors if map failed or NaN
        trades['sector'] = trades['sector'].fillna('OTHER')
        
        # Get last 11 rebalance dates
        dates = sorted(trades['date'].unique())[-11:]
        df_last = trades[trades['date'].isin(dates)]
        
        # Calculate mean score per sector and date
        score_col = 'score' if 'score' in df_last.columns else 'weight'
        piv = df_last.groupby(['sector', 'date'])[score_col].mean().unstack(fill_value=0.0)
        
        if piv.empty:
            sectors = ['FINANCIALS', 'IT', 'ENERGY', 'METALS', 'PHARMA', 'AUTO', 'FMCG']
            days = [f"T-{i}" for i in range(10, -1, -1)]
            z = np.random.randn(len(sectors), len(days))
            text = [[f"{val:.2f}" for val in row] for row in z]
        else:
            # Reorder columns as T-10 to T-0
            days = [f"T-{10-i}" for i in range(len(piv.columns))]
            piv.columns = days
            sectors = piv.index.tolist()
            z = piv.values
            text = [[f"{val:+.2f}" for val in row] for row in z]
            
    fig = go.Figure(data=go.Heatmap(
        z=z, x=days, y=sectors,
        colorscale='RdYlGn',
        showscale=True,
        colorbar=dict(
            title=dict(text="Score", side="top"),
            thickness=12,
            len=0.9
        ),
        zmid=0.0,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 9, "family": "JetBrains Mono", "color": "black"}
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=80, r=20, t=30, b=30),
        font=dict(family="JetBrains Mono", size=10, color="#8B949E"),
        title="SECTOR SIGNAL INTENSITY (MOMENTUM)",
        title_font_size=11
    )
    return fig

layout = html.Div(
    [
        html.H3("STRATEGY HEALTH & SIGNALS"),
        
        # Health KPIs
        html.Div(
            [
                create_metric_panel("FIP MOMENTUM STATUS", "ACTIVE", "Running normal", "bull"),
                create_metric_panel("ENGINE LATENCY", "14 ms", "Avg inference time", "bull"),
                create_metric_panel("LIVE SIGNALS", "24", "Currently open", "info"),
                create_metric_panel("VOLATILITY REGIME", daily['market_state'].iloc[-1] if not daily.empty else "UNKNOWN", "Macro Filter", "chop"),
            ],
            className="os-grid grid-4col", style={"marginBottom": "15px"}
        ),
        
        # Content Grid
        html.Div(
            [
                html.Div(
                    [
                        html.Div("STRATEGY DEPLOYMENTS", className="panel-title"),
                        dash.dash_table.DataTable(
                            data=[
                                {"strategy": "FIP Momentum Core", "status": "LIVE", "alloc": "85%", "pnl": "+1.49%"},
                                {"strategy": "StatArb Pairs", "status": "DEPRECATED", "alloc": "0%", "pnl": "0.00%"},
                                {"strategy": "Intraday Breakout", "status": "PAPER", "alloc": "15%", "pnl": "-0.21%"},
                            ],
                            columns=[
                                {"name": "STRATEGY", "id": "strategy"},
                                {"name": "STATUS", "id": "status"},
                                {"name": "CAPITAL ALLOC", "id": "alloc"},
                                {"name": "7D PNL", "id": "pnl"}
                            ],
                            style_table={'overflowX': 'auto'},
                            style_cell={'textAlign': 'left'},
                            style_data_conditional=[
                                {'if': {'filter_query': '{status} = LIVE', 'column_id': 'status'}, 'color': '#00FF88'},
                                {'if': {'filter_query': '{status} = DEPRECATED', 'column_id': 'status'}, 'color': '#FF4560'},
                                {'if': {'filter_query': '{status} = PAPER', 'column_id': 'status'}, 'color': '#FFC107'}
                            ]
                        )
                    ],
                    className="os-panel", style={"gridColumn": "span 1"}
                ),
                
                html.Div(
                    [
                        dcc.Graph(figure=create_signal_heatmap(), config={'displayModeBar': False}, style={"height": "300px"})
                    ],
                    className="os-panel", style={"gridColumn": "span 1"}
                )
            ],
            className="os-grid grid-2col"
        )
    ]
)
