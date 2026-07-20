import dash
from dash import html, dcc, dash_table
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from data_layer.loader import load_geometry_log

dash.register_page(__name__, path='/05-market', name='Market')

df_geom = load_geometry_log()

def create_metric_panel(title, value, sub="", color_class="info"):
    return html.Div(
        [
            html.Div(title, className="panel-title"),
            html.Div(value, className=f"metric-val {color_class}"),
            html.Div(sub, className="metric-sub") if sub else None
        ],
        className="os-panel"
    )

def create_geometry_chart(df):
    fig = go.Figure()
    if df.empty:
        return fig
        
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['pred_breadth'],
        mode='lines',
        name='Predicted Breadth (B_t+1)',
        line=dict(color='#00B4D8', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['pred_mode'] * 50.0,  # Scaled for dual visualization
        mode='lines',
        name='Predicted Mode Strength (PCA x50)',
        line=dict(color='#FFC107', width=2, dash='dot')
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=20, t=30, b=30),
        font=dict(family="JetBrains Mono", size=10, color="#8B949E"),
        title="FORECASTED MARKET GEOMETRY COORDINATES (BREADTH & PCA)",
        title_font_size=11,
    )
    return fig

latest_row = df_geom.iloc[-1] if not df_geom.empty else {}

layout = html.Div(
    [
        html.H3("MARKET GEOMETRY & CONTINUOUS RISK CONTROLLER"),
        
        # KPI Row
        html.Div(
            [
                create_metric_panel("PREDICTED BREADTH", f"{latest_row.get('pred_breadth', 0):.2f}" if not df_geom.empty else "N/A", "Effective Breadth (T+1)", "info"),
                create_metric_panel("PREDICTED PCA MODE", f"{latest_row.get('pred_mode', 0):.4f}" if not df_geom.empty else "N/A", "Market Mode Strength (T+1)", "chop"),
                create_metric_panel("RISK CONTROLLER (κ)", f"{latest_row.get('kappa', 0):.2f}" if not df_geom.empty else "N/A", "Gross Exposure Scaling", "bull"),
                create_metric_panel("SECTOR CONSTRAINTS", f"{latest_row.get('sector_limit', 0)*100:.1f}%" if not df_geom.empty else "N/A", "Dynamic Sector Cap", "info"),
            ],
            className="os-grid grid-4col", style={"marginBottom": "15px"}
        ),
        
        # Content Grid
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(figure=create_geometry_chart(df_geom), config={'displayModeBar': False}, style={"height": "350px"})
                    ],
                    className="os-panel", style={"gridColumn": "span 2"}
                ),
                
                html.Div(
                    [
                        html.Div("RECENT GEOMETRY & RISK REBALANCE LOGS", className="panel-title"),
                        dash_table.DataTable(
                            data=df_geom.tail(8).to_dict('records') if not df_geom.empty else [],
                            columns=[
                                {"name": "DATE", "id": "date"},
                                {"name": "PRED BREADTH", "id": "pred_breadth"},
                                {"name": "PRED PCA", "id": "pred_mode"},
                                {"name": "KAPPA (κ)", "id": "kappa"},
                                {"name": "STOCK CAP", "id": "max_single_stock"},
                                {"name": "SECTOR CAP", "id": "sector_limit"},
                                {"name": "DRAWDOWN", "id": "drawdown"}
                            ],
                            style_table={'overflowX': 'auto'},
                            style_cell={'textAlign': 'left', 'fontSize': '11px'},
                            style_data_conditional=[
                                {'if': {'column_id': 'kappa'}, 'fontWeight': 'bold', 'color': '#00FF88'}
                            ]
                        )
                    ],
                    className="os-panel", style={"gridColumn": "span 1"}
                )
            ],
            className="os-grid grid-3col"
        )
    ]
)
