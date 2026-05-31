import dash
from dash import html, dcc
import plotly.graph_objects as go
import pandas as pd
from data_layer.loader import load_trade_log, compute_daily_summary

dash.register_page(__name__, path='/risk', name='Risk Analytics')

trades = load_trade_log()
daily = compute_daily_summary(trades)

def create_exposure_chart(daily):
    if daily.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily['date'], y=daily['gross_exposure'],
        mode='lines', name='Gross Exposure', line=dict(color='#00B4D8')
    ))
    fig.add_trace(go.Scatter(
        x=daily['date'], y=daily['net_exposure'],
        mode='lines', name='Net Exposure', line=dict(color='#00FF88')
    ))

    fig.update_layout(
        template='plotly_dark', plot_bgcolor='#0D1117', paper_bgcolor='#161B22',
        title="Historical Exposure (Gross vs Net)",
        margin=dict(l=40, r=20, t=40, b=40),
        font=dict(color='#E6EDF3')
    )
    return fig

def create_drawdown_chart(daily):
    if daily.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily['date'], y=daily['drawdown'] * 100,
        mode='lines', fill='tozeroy', name='Drawdown %', line=dict(color='#FF4560')
    ))

    fig.update_layout(
        template='plotly_dark', plot_bgcolor='#0D1117', paper_bgcolor='#161B22',
        title="Underwater Plot (Drawdown %)",
        margin=dict(l=40, r=20, t=40, b=40),
        font=dict(color='#E6EDF3')
    )
    return fig

import numpy as np

def create_mock_correlation_matrix(trades):
    if trades.empty:
        return go.Figure()

    # Calculate real correlation using recent fwd_returns over time for top tickers
    recent_date = trades['date'].max()
    recent_tickers = trades[trades['date'] == recent_date]['ticker'].unique()[:10]

    # Pivot fwd_return for these tickers over the last N periods
    pivot = trades[trades['ticker'].isin(recent_tickers)].pivot(index='date', columns='ticker', values='fwd_return')
    corr = pivot.corr().fillna(0).values

    # If we don't have enough data, fall back gracefully
    if corr.shape[0] == 0:
        corr = np.eye(len(recent_tickers))

    fig = go.Figure(data=go.Heatmap(
        z=corr,
        x=recent_tickers,
        y=recent_tickers,
        colorscale='RdBu_r',
        zmin=-1, zmax=1,
        showscale=True
    ))
    fig.update_layout(
        template='plotly_dark', plot_bgcolor='#0D1117', paper_bgcolor='#161B22',
        title="Holdings Correlation Matrix (Forward Returns)",
        margin=dict(l=40, r=20, t=40, b=40),
        font=dict(color='#E6EDF3', family="JetBrains Mono")
    )
    return fig

def create_factor_drift_chart(trades):
    if trades.empty:
        return go.Figure()

    fig = go.Figure()

    # Group by date and calculate average factor exposure across the portfolio
    grouped = trades.groupby('date').agg({
        'mom_z': 'mean',
        'kinetic_energy': 'mean',
        'efficiency': 'mean'
    }).reset_index()

    dates = grouped['date']

    # Normalize these to plot on the same axis for drift visualization
    for col in ['mom_z', 'kinetic_energy', 'efficiency']:
        grouped[col] = (grouped[col] - grouped[col].mean()) / grouped[col].std()

    # Smooth them to show drift rather than raw volatility
    grouped['mom_z_smooth'] = grouped['mom_z'].rolling(12, min_periods=1).mean()
    grouped['ke_smooth'] = grouped['kinetic_energy'].rolling(12, min_periods=1).mean()
    grouped['eff_smooth'] = grouped['efficiency'].rolling(12, min_periods=1).mean()

    fig.add_trace(go.Scatter(x=dates, y=grouped['mom_z_smooth'], mode='lines', name='Momentum (Z) Exposure', line=dict(color='#00FF88')))
    fig.add_trace(go.Scatter(x=dates, y=grouped['ke_smooth'], mode='lines', name='Kinetic Energy Exposure', line=dict(color='#00B4D8')))
    fig.add_trace(go.Scatter(x=dates, y=grouped['eff_smooth'], mode='lines', name='Efficiency Exposure', line=dict(color='#FFC107')))

    fig.update_layout(
        template='plotly_dark', plot_bgcolor='#0D1117', paper_bgcolor='#161B22',
        title="Rolling Factor Exposure Drift (Z-Scored)",
        margin=dict(l=40, r=20, t=40, b=40),
        font=dict(color='#E6EDF3', family="JetBrains Mono")
    )
    return fig

layout = html.Div(
    [
        html.H2("Risk Analytics"),
        html.Div(
            [
                html.Div(dcc.Graph(figure=create_exposure_chart(daily)), className="card", style={"flex": 1}),
                html.Div(dcc.Graph(figure=create_drawdown_chart(daily)), className="card", style={"flex": 1})
            ],
            style={"display": "flex", "gap": "20px", "flexWrap": "wrap", "marginBottom": "20px"}
        ),
        html.Div(
            [
                html.Div(dcc.Graph(figure=create_mock_correlation_matrix(trades)), className="card", style={"flex": 1}),
                html.Div(dcc.Graph(figure=create_factor_drift_chart(trades)), className="card", style={"flex": 1})
            ],
            style={"display": "flex", "gap": "20px", "flexWrap": "wrap"}
        )
    ]
)
