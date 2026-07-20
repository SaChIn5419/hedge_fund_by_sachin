import dash
from dash import html, dcc, callback, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os

from data_layer.loader import load_trade_log, compute_daily_summary, get_kpis, get_current_holdings

dash.register_page(__name__, path='/', name='Portfolio')

# Load data globally for callbacks
trades = load_trade_log()
daily = compute_daily_summary(trades, capital=1_000_000)

# Base plotly layout template
PLOTLY_TEMPLATE = dict(
    layout=dict(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="JetBrains Mono, monospace", size=10, color="#8B949E"),
        margin=dict(l=40, r=20, t=30, b=30),
        xaxis=dict(showgrid=True, gridcolor="#30363D", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#30363D", zeroline=False)
    )
)

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

def create_equity_chart(daily_df):
    fig = go.Figure()
    if daily_df.empty:
        return fig
    
    # 1. Add Strategy Equity Trace
    fig.add_trace(go.Scatter(
        x=daily_df['date'], y=daily_df['equity'],
        mode='lines',
        name='Chimera v3.0 Core',
        line=dict(color='#00B4D8', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(0, 180, 216, 0.05)'
    ))
    
    # 2. Load Nifty 50 and Gold BeES Benchmarks
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    nifty_path = os.path.join(BASE_DIR, "chimera_data/indices/nifty50.parquet")
    gold_path = os.path.join(BASE_DIR, "chimera_data/macro/goldbees.parquet")
    
    capital = 1_000_000 # Base capital
    
    # Add Nifty trace
    if os.path.exists(nifty_path):
        try:
            nifty = pd.read_parquet(nifty_path)
            nifty['date'] = pd.to_datetime(nifty['Date'])
            nifty_aligned = daily_df[['date']].merge(nifty[['date', 'Close']], on='date', how='left')
            nifty_aligned['Close'] = nifty_aligned['Close'].ffill().bfill()
            
            p0 = nifty_aligned['Close'].iloc[0]
            if p0 > 0:
                nifty_aligned['nifty_val'] = capital * (nifty_aligned['Close'] / p0)
                fig.add_trace(go.Scatter(
                    x=nifty_aligned['date'], y=nifty_aligned['nifty_val'],
                    mode='lines',
                    name='Nifty 50 Index',
                    line=dict(color='#FF9F1C', width=1.5, dash='dash')
                ))
        except Exception as e:
            print(f"Error loading nifty benchmark: {e}")
            
    # Add Gold Bees trace
    if os.path.exists(gold_path):
        try:
            gold = pd.read_parquet(gold_path)
            gold['date'] = pd.to_datetime(gold['Date'])
            gold_aligned = daily_df[['date']].merge(gold[['date', 'Close']], on='date', how='left')
            gold_aligned['Close'] = gold_aligned['Close'].ffill().bfill()
            
            g0 = gold_aligned['Close'].iloc[0]
            if g0 > 0:
                gold_aligned['gold_val'] = capital * (gold_aligned['Close'] / g0)
                fig.add_trace(go.Scatter(
                    x=gold_aligned['date'], y=gold_aligned['gold_val'],
                    mode='lines',
                    name='Gold BeES ETF',
                    line=dict(color='#FFD700', width=1.5, dash='dot')
                ))
        except Exception as e:
            print(f"Error loading gold benchmark: {e}")
            
    fig.update_layout(**PLOTLY_TEMPLATE['layout'])
    fig.update_layout(
        title="PORTFOLIO VALUE COMPARISON vs BENCHMARKS", 
        title_font_size=11, 
        title_x=0,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    fig.update_yaxes(tickformat=",.0f", tickprefix="₹")
    return fig

def create_drawdown_chart(daily_df):
    fig = go.Figure()
    if daily_df.empty:
        return fig
    
    fig.add_trace(go.Scatter(
        x=daily_df['date'], y=daily_df['drawdown'],
        mode='lines',
        name='Drawdown',
        line=dict(color='#FF4560', width=1),
        fill='tozeroy',
        fillcolor='rgba(255, 69, 96, 0.2)'
    ))
    
    fig.update_layout(**PLOTLY_TEMPLATE['layout'])
    fig.update_layout(title="DRAWDOWN", title_font_size=11, title_x=0, height=180)
    fig.update_yaxes(tickformat=".1%")
    return fig

# Main page layout containing the toggle
layout = html.Div(
    [
        html.Div(
            [
                html.H3("PORTFOLIO OVERVIEW", style={"margin": 0}),
                html.Div(
                    [
                        html.Span("PERIOD SELECTOR: ", style={"color": "#8B949E", "marginRight": "10px", "fontSize": "11px"}),
                        dcc.RadioItems(
                            id='period-selector',
                            options=[
                                {'label': ' Out-of-Sample (2023+)', 'value': 'OOS'},
                                {'label': ' Full History', 'value': 'FULL'}
                            ],
                            value='OOS',
                            inline=True,
                            style={"display": "inline-block", "fontFamily": "JetBrains Mono", "fontSize": "11px"},
                            inputStyle={"marginRight": "5px", "marginLeft": "15px"}
                        )
                    ],
                    style={"marginLeft": "auto", "display": "flex", "alignItems": "center"}
                )
            ],
            style={"display": "flex", "alignItems": "center", "marginBottom": "15px"}
        ),
        
        # Dynamic content container
        html.Div(id='portfolio-content')
    ]
)

@callback(
    Output('portfolio-content', 'children'),
    Input('period-selector', 'value')
)
def update_portfolio_view(period):
    if daily.empty:
        return html.Div("No trade logs found.")
        
    if period == 'OOS':
        # Filter OOS period (2023-01-01 onwards)
        daily_filtered = daily[daily['date'] >= '2023-01-01'].copy()
        trades_filtered = trades[trades['date'] >= '2023-01-01'].copy()
        
        # Re-base OOS equity curve to start at exactly 1,000,000
        if not daily_filtered.empty:
            pnl_diff = daily_filtered['net_pnl'].values
            equity_val = [1_000_000]
            for p in pnl_diff:
                equity_val.append(equity_val[-1] + p)
            daily_filtered['equity'] = equity_val[1:]
            
            # Recalculate drawdown relative to new OOS peak
            daily_filtered['peak'] = daily_filtered['equity'].cummax()
            daily_filtered['drawdown'] = daily_filtered['equity'] / daily_filtered['peak'] - 1.0
            
            # Recalculate returns based on OOS equity
            daily_filtered['portfolio_return'] = daily_filtered['net_pnl'] / daily_filtered['equity'].shift(1).fillna(1_000_000)
    else:
        daily_filtered = daily.copy()
        trades_filtered = trades.copy()
        
    kpis = get_kpis(daily_filtered, capital=1_000_000)
    holdings = get_current_holdings(trades_filtered)
    
    return html.Div(
        [
            # KPI Row
            html.Div(
                [
                    create_metric_panel("PORTFOLIO VALUE", f"₹{daily_filtered['equity'].iloc[-1]:,.2f}" if not daily_filtered.empty else "N/A", "Base: ₹1,000,000", "bull"),
                    create_metric_panel("CAGR", kpis.get("CAGR", "N/A"), "Annualized", "bull"),
                    create_metric_panel("SHARPE RATIO", kpis.get("Sharpe Ratio", "N/A"), "Risk-Adjusted", "info"),
                    create_metric_panel("MAX DRAWDOWN", kpis.get("Max Drawdown", "N/A"), "Historical Max", "bear"),
                    create_metric_panel("GROSS EXPOSURE", f"{(daily_filtered['gross_exposure'].iloc[-1] * 100):.1f}%" if not daily_filtered.empty else "N/A", "Current Leverage", "chop"),
                ],
                className="os-grid grid-4col", style={"gridTemplateColumns": "repeat(5, 1fr)", "marginBottom": "15px"}
            ),
            
            # Charts Row
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Graph(figure=create_equity_chart(daily_filtered), config={'displayModeBar': False}, style={"height": "350px"}),
                            dcc.Graph(figure=create_drawdown_chart(daily_filtered), config={'displayModeBar': False}, style={"height": "180px", "marginTop": "10px"})
                        ],
                        className="os-panel", style={"gridColumn": "span 3"}
                    ),
                    
                    html.Div(
                        [
                            html.Div("CURRENT ALLOCATIONS", className="panel-title"),
                            dash.dash_table.DataTable(
                                data=holdings.to_dict('records') if not holdings.empty else [],
                                columns=[
                                    {"name": "TICKER", "id": "ticker"},
                                    {"name": "SIDE", "id": "side"},
                                    {"name": "WEIGHT", "id": "weight", "type": "numeric", "format": dash.dash_table.FormatTemplate.percentage(1)},
                                    {"name": "PNL", "id": "net_pnl", "type": "numeric", "format": dash.dash_table.Format.Format(precision=0, scheme=dash.dash_table.Format.Scheme.fixed).symbol(dash.dash_table.Format.Symbol.yes).symbol_prefix('₹')}
                                ],
                                style_table={'overflowX': 'auto', 'height': '490px', 'overflowY': 'auto'},
                                style_cell={'textAlign': 'left'},
                                style_data_conditional=[
                                    {'if': {'filter_query': '{net_pnl} > 0', 'column_id': 'net_pnl'}, 'color': '#00FF88'},
                                    {'if': {'filter_query': '{net_pnl} < 0', 'column_id': 'net_pnl'}, 'color': '#FF4560'}
                                ]
                            )
                        ],
                        className="os-panel", style={"gridColumn": "span 1"}
                    )
                ],
                className="os-grid grid-4col"
            )
        ]
    )
