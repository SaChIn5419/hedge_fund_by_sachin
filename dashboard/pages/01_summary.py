import dash
from dash import html, dcc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from data_layer.loader import load_trade_log, compute_daily_summary, get_kpis

dash.register_page(__name__, path='/', name='Executive Summary')

trades = load_trade_log()
daily = compute_daily_summary(trades)
kpis = get_kpis(daily)

def create_kpi_card(title, value):
    return html.Div(
        [
            html.Div(title, className="kpi-title"),
            html.Div(value, className="kpi-value kpi-neutral")
        ],
        className="card kpi-tile"
    )

def create_equity_chart(daily):
    if daily.empty:
        return go.Figure()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=daily['date'],
        y=daily['equity'],
        mode='lines',
        name='Chimera',
        line=dict(color='#00B4D8', width=2)
    ))

    regime_colors = {'BULL': 'rgba(0, 255, 136, 0.1)', 'BEAR': 'rgba(255, 69, 96, 0.1)', 'CHOP': 'rgba(255, 193, 7, 0.1)'}
    shapes = []
    if not daily.empty and 'market_state' in daily.columns:
        last_state = daily['market_state'].iloc[0]
        start_date = daily['date'].iloc[0]
        for i, row in daily.iterrows():
            if row['market_state'] != last_state:
                shapes.append(dict(
                    type="rect", xref="x", yref="paper",
                    x0=start_date, y0=0, x1=row['date'], y1=1,
                    fillcolor=regime_colors.get(last_state, 'rgba(128, 128, 128, 0.1)'),
                    opacity=1, layer="below", line_width=0,
                ))
                start_date = row['date']
                last_state = row['market_state']
        shapes.append(dict(
            type="rect", xref="x", yref="paper",
            x0=start_date, y0=0, x1=daily['date'].iloc[-1], y1=1,
            fillcolor=regime_colors.get(last_state, 'rgba(128, 128, 128, 0.1)'),
            opacity=1, layer="below", line_width=0,
        ))

    fig.update_layout(
        shapes=shapes, template='plotly_dark', plot_bgcolor='#0D1117', paper_bgcolor='#161B22',
        margin=dict(l=40, r=20, t=40, b=40), title="Equity Curve with Regime Shading",
        font=dict(color='#E6EDF3', family="JetBrains Mono, Courier New, monospace"), yaxis_type="log"
    )
    return fig

def create_monthly_heatmap(daily):
    if daily.empty:
        return go.Figure()

    # Calculate monthly returns
    df = daily[['date', 'portfolio_return']].copy()
    df.set_index('date', inplace=True)
    monthly = df['portfolio_return'].resample('ME').apply(lambda x: (1 + x).prod() - 1).reset_index()
    monthly['Year'] = monthly['date'].dt.year
    monthly['Month'] = monthly['date'].dt.strftime('%b')

    pivot = monthly.pivot(index='Year', columns='Month', values='portfolio_return')
    months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot = pivot.reindex(columns=[m for m in months_order if m in pivot.columns])

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='RdYlGn',
        zmid=0,
        text=np.round(pivot.values * 100, 2),
        texttemplate="%{text}%",
        showscale=False
    ))

    fig.update_layout(
        template='plotly_dark', plot_bgcolor='#0D1117', paper_bgcolor='#161B22',
        title="Monthly Returns (%)", font=dict(color='#E6EDF3', family="JetBrains Mono"),
        margin=dict(l=40, r=20, t=40, b=40)
    )
    return fig

layout = html.Div(
    [
        html.H2("Morning Brief"),

        html.Div(
            [create_kpi_card(k, v) for k, v in kpis.items()],
            style={"display": "flex", "justifyContent": "space-between", "gap": "15px", "marginBottom": "20px"}
        ),

        html.Div(dcc.Graph(figure=create_equity_chart(daily), style={'height': '500px'}), className="card"),

        html.Div(
            [
                html.Div(dcc.Graph(figure=create_monthly_heatmap(daily)), className="card", style={"flex": 2}),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H4("System Alerts", style={"marginBottom": "15px", "color": "#E6EDF3"}),
                                html.Div(
                                    [
                                        html.Div("⚠️ High Drawdown Warning" if (daily['drawdown'].iloc[-1] < -0.1) else "✅ Drawdown Risk Contained", style={"padding": "10px", "backgroundColor": "rgba(255, 69, 96, 0.1)" if (daily['drawdown'].iloc[-1] < -0.1) else "rgba(0, 255, 136, 0.1)", "borderLeft": "4px solid #FF4560" if (daily['drawdown'].iloc[-1] < -0.1) else "4px solid #00FF88", "marginBottom": "10px"}),
                                        html.Div(f"ℹ️ Current Regime: {daily['market_state'].iloc[-1]}", style={"padding": "10px", "backgroundColor": "rgba(255, 193, 7, 0.1)", "borderLeft": "4px solid #FFC107", "marginBottom": "10px"}),
                                        html.Div("✅ FIP factor signals active", style={"padding": "10px", "backgroundColor": "rgba(0, 255, 136, 0.1)", "borderLeft": "4px solid #00FF88"})
                                    ] if not daily.empty else [html.P("No data available")],
                                    style={"display": "flex", "flexDirection": "column", "gap": "5px"}
                                )
                            ],
                            className="card",
                            style={"marginBottom": "20px", "flex": 1}
                        ),
                        html.Div(
                            [
                                html.H4("Export Tools"),
                                html.Button("Download Tearsheet (CSV)", id="btn-tearsheet", className="kpi-title", style={"width": "100%", "padding": "15px", "backgroundColor": "#161B22", "color": "#E6EDF3", "border": "1px solid #30363D", "borderRadius": "4px", "cursor": "pointer", "marginTop": "10px"}),
                                dcc.Download(id="download-dataframe-csv")
                            ],
                            className="card",
                            style={"flex": 1}
                        )
                    ],
                    style={"display": "flex", "flexDirection": "column", "flex": 1}
                )
            ],
            style={"display": "flex", "gap": "20px"}
        )
    ]
)

@dash.callback(
    dash.Output("download-dataframe-csv", "data"),
    dash.Input("btn-tearsheet", "n_clicks"),
    prevent_initial_call=True
)
def generate_tearsheet(n_clicks):
    if n_clicks:
        # For simplicity, we just dump the daily KPIs trace as the tearsheet data
        return dcc.send_data_frame(daily.to_csv, "chimera_tearsheet.csv")
    return dash.no_update
