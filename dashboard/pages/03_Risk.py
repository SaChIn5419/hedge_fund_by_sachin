import dash
from dash import html, dcc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

from data_layer.loader import load_trade_log, compute_daily_summary, get_current_holdings

dash.register_page(__name__, path='/03-risk', name='Risk')

trades = load_trade_log()
daily = compute_daily_summary(trades, capital=1_000_000)
holdings = get_current_holdings(trades)

def compute_comprehensive_risk(daily_df):
    if daily_df.empty or len(daily_df) < 50:
        return None
        
    df = daily_df.copy()
    rets = df['portfolio_return'].dropna().values
    mu = np.mean(rets)
    sigma = np.std(rets, ddof=1)
    
    # 1. Historical (Empirical) Metrics
    h_var_95 = np.percentile(rets, 5)
    h_var_99 = np.percentile(rets, 1)
    h_cvar_95 = rets[rets <= h_var_95].mean() if len(rets[rets <= h_var_95]) > 0 else h_var_95
    h_cvar_99 = rets[rets <= h_var_99].mean() if len(rets[rets <= h_var_99]) > 0 else h_var_99
    
    # 2. Parametric (Normal) Metrics
    # ppf(0.05) gives the z-score for the 5th percentile
    p_var_95 = mu + sigma * norm.ppf(0.05)
    p_var_99 = mu + sigma * norm.ppf(0.01)
    
    # Expected shortfall for normal distribution: mu - sigma * pdf(z) / (1-alpha)
    p_cvar_95 = mu - sigma * (norm.pdf(norm.ppf(0.05)) / 0.05)
    p_cvar_99 = mu - sigma * (norm.pdf(norm.ppf(0.01)) / 0.01)
    
    # 3. Tail Risk Ratios
    tail_ratio_95 = abs(h_var_95) / abs(p_var_95) if p_var_95 != 0 else 1.0
    tail_ratio_99 = abs(h_var_99) / abs(p_var_99) if p_var_99 != 0 else 1.0
    
    # 4. Rolling 252-day VaR (95%) and Breach Analysis
    df['rolling_h_var_95'] = df['portfolio_return'].rolling(window=252, min_periods=126).apply(lambda x: np.percentile(x, 5), raw=True)
    df['rolling_p_var_95'] = df['portfolio_return'].rolling(window=252, min_periods=126).apply(lambda x: np.mean(x) + np.std(x, ddof=1)*norm.ppf(0.05), raw=True)
    
    # Shift rolling VaR by 1 day to prevent look-ahead bias in breach checking
    df['shifted_var'] = df['rolling_h_var_95'].shift(1)
    
    # Count breaches
    valid_days = df['shifted_var'].notna()
    breaches = df[valid_days & (df['portfolio_return'] < df['shifted_var'])]
    
    total_valid_days = valid_days.sum()
    actual_breaches = len(breaches)
    expected_breaches = total_valid_days * 0.05
    
    return {
        "h_var_95": h_var_95, "h_var_99": h_var_99,
        "h_cvar_95": h_cvar_95, "h_cvar_99": h_cvar_99,
        "p_var_95": p_var_95, "p_var_99": p_var_99,
        "p_cvar_95": p_cvar_95, "p_cvar_99": p_cvar_99,
        "tail_ratio_95": tail_ratio_95, "tail_ratio_99": tail_ratio_99,
        "actual_breaches": actual_breaches,
        "expected_breaches": expected_breaches,
        "total_valid_days": total_valid_days,
        "max_dd": df['drawdown'].min(),
        "df_rolling": df
    }

risk = compute_comprehensive_risk(daily)

def create_metric_panel(title, value, sub="", color_class="info"):
    return html.Div(
        [
            html.Div(title, className="panel-title"),
            html.Div(value, className=f"metric-val {color_class}"),
            html.Div(sub, className="metric-sub") if sub else None
        ],
        className="os-panel"
    )

def create_rolling_var_chart(df):
    fig = go.Figure()
    if df is None or df.empty:
        return fig
        
    # Valid rolling days
    valid_df = df[df['rolling_h_var_95'].notna()]
    
    fig.add_trace(go.Scatter(
        x=valid_df['date'], y=valid_df['portfolio_return'],
        mode='lines', name='Daily Return',
        line=dict(color='#8B949E', width=1), opacity=0.5
    ))
    
    fig.add_trace(go.Scatter(
        x=valid_df['date'], y=valid_df['rolling_h_var_95'],
        mode='lines', name='Rolling H-VaR (95%)',
        line=dict(color='#FFC107', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=valid_df['date'], y=valid_df['rolling_p_var_95'],
        mode='lines', name='Rolling P-VaR (95%)',
        line=dict(color='#00B4D8', width=1, dash='dash')
    ))
    
    fig.update_layout(
        template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=20, t=30, b=30),
        font=dict(family="JetBrains Mono", size=10, color="#8B949E"),
        title="ROLLING 252-DAY VALUE AT RISK (VaR)", title_font_size=11,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(tickformat=".1%")
    return fig

def create_distribution_chart(risk_data):
    fig = go.Figure()
    if not risk_data:
        return fig
        
    df = risk_data['df_rolling']
    rets = df['portfolio_return'].dropna()
    
    fig.add_trace(go.Histogram(
        x=rets, nbinsx=60, marker_color='#30363D', name='Returns', histnorm='probability density'
    ))
    
    # Overlay Normal Distribution Curve
    mu, sigma = np.mean(rets), np.std(rets, ddof=1)
    x_range = np.linspace(rets.min(), rets.max(), 100)
    pdf = norm.pdf(x_range, mu, sigma)
    
    fig.add_trace(go.Scatter(
        x=x_range, y=pdf, mode='lines', name='Normal Dist',
        line=dict(color='#00B4D8', width=1)
    ))
    
    # Tails
    fig.add_vline(x=risk_data['h_var_95'], line_width=2, line_dash="solid", line_color="#FFC107")
    fig.add_vline(x=risk_data['p_var_95'], line_width=1, line_dash="dash", line_color="#FFC107")
    fig.add_vline(x=risk_data['h_cvar_95'], line_width=2, line_dash="solid", line_color="#FF4560")
    
    fig.update_layout(
        template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=20, t=30, b=30),
        font=dict(family="JetBrains Mono", size=10, color="#8B949E"),
        title="EMPIRICAL VS NORMAL RETURN DISTRIBUTION", title_font_size=11,
        showlegend=False, bargap=0.1
    )
    fig.update_xaxes(tickformat=".1%")
    return fig

# Construct layout if risk data is available
if risk:
    table_data = [
        {"metric": "VaR (95%)", "hist": f"{risk['h_var_95']*100:.2f}%", "param": f"{risk['p_var_95']*100:.2f}%"},
        {"metric": "CVaR (95%)", "hist": f"{risk['h_cvar_95']*100:.2f}%", "param": f"{risk['p_cvar_95']*100:.2f}%"},
        {"metric": "VaR (99%)", "hist": f"{risk['h_var_99']*100:.2f}%", "param": f"{risk['p_var_99']*100:.2f}%"},
        {"metric": "CVaR (99%)", "hist": f"{risk['h_cvar_99']*100:.2f}%", "param": f"{risk['p_cvar_99']*100:.2f}%"},
    ]
    
    breach_rate = (risk['actual_breaches'] / risk['total_valid_days']) * 100 if risk['total_valid_days'] > 0 else 0
    breach_status = "PASS" if abs(breach_rate - 5.0) < 1.5 else "WARN"
    breach_color = "bull" if breach_status == "PASS" else "chop"
    
    layout = html.Div(
        [
            html.H3("INSTITUTIONAL RISK & TAIL ANALYSIS"),
            
            # KPI Row
            html.Div(
                [
                    create_metric_panel("MAX DRAWDOWN", f"{risk['max_dd']*100:.2f}%", "Peak-to-Trough", "bear"),
                    create_metric_panel("TAIL RISK RATIO", f"{risk['tail_ratio_95']:.2f}", ">1 indicates fat tails", "chop" if risk['tail_ratio_95'] > 1.1 else "info"),
                    create_metric_panel("VaR BREACHES", f"{risk['actual_breaches']} / {risk['expected_breaches']:.1f}", f"Actual vs Expected (5%)", breach_color),
                    create_metric_panel("BREACH RATE", f"{breach_rate:.2f}%", f"Kupiec expectation: 5.00%", breach_color),
                ],
                className="os-grid grid-4col", style={"marginBottom": "15px"}
            ),
            
            # Middle Content Grid (Table and Distribution)
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("EMPIRICAL VS PARAMETRIC (NORMAL) TAILS", className="panel-title"),
                            dash.dash_table.DataTable(
                                data=table_data,
                                columns=[
                                    {"name": "METRIC", "id": "metric"},
                                    {"name": "HISTORICAL", "id": "hist"},
                                    {"name": "PARAMETRIC", "id": "param"}
                                ],
                                style_table={'overflowX': 'auto'},
                                style_cell={'textAlign': 'left'},
                            ),
                            html.Div(
                                [
                                    html.P(f"• A Tail Risk Ratio of {risk['tail_ratio_95']:.2f} means historical VaR is {risk['tail_ratio_95']*100:.0f}% of the Normal VaR prediction.", style={"marginTop": "15px", "color": "#8B949E"})
                                ],
                                style={"fontFamily": "JetBrains Mono", "fontSize": "11px"}
                            )
                        ],
                        className="os-panel", style={"gridColumn": "span 1"}
                    ),
                    
                    html.Div(
                        [
                            dcc.Graph(figure=create_distribution_chart(risk), config={'displayModeBar': False}, style={"height": "300px"})
                        ],
                        className="os-panel", style={"gridColumn": "span 2"}
                    )
                ],
                className="os-grid grid-3col", style={"marginBottom": "15px"}
            ),
            
            # Bottom Content Grid (Rolling Chart)
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Graph(figure=create_rolling_var_chart(risk['df_rolling']), config={'displayModeBar': False}, style={"height": "350px"})
                        ],
                        className="os-panel", style={"gridColumn": "span 1"}
                    )
                ],
                className="os-grid grid-1col"
            )
        ]
    )
else:
    layout = html.Div(html.H3("Insufficient Data for Risk Modeling (Need 50+ days)"))
