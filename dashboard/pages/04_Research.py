import dash
from dash import html, dcc, dash_table
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from data_layer.loader import load_evidence_registry, load_geometry_log

dash.register_page(__name__, path='/04-research', name='Research')

def create_metric_panel(title, value, sub="", color_class="info"):
    return html.Div(
        [
            html.Div(title, className="panel-title"),
            html.Div(value, className=f"metric-val {color_class}"),
            html.Div(sub, className="metric-sub") if sub else None
        ],
        className="os-panel"
    )

def create_kappa_chart():
    df_geom = load_geometry_log()
    fig = go.Figure()
    if df_geom.empty:
        return fig
        
    fig.add_trace(go.Scatter(
        x=df_geom['date'], y=df_geom['kappa'],
        mode='lines',
        name='Risk Controller Exposure (κ)',
        line=dict(color='#00FF88', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 255, 136, 0.1)'
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=20, t=30, b=30),
        font=dict(family="JetBrains Mono", size=10, color="#8B949E"),
        title="CONTINUOUS RISK CONTROLLER EXPOSURE FACTOR (κ_t+1)",
        title_font_size=11,
    )
    fig.update_yaxes(range=[0.0, 1.05])
    return fig

def get_evidence_table_data():
    registry = load_evidence_registry()
    table_rows = []
    for entry in registry:
        winning_h = entry.get('winning_hypothesis', 'N/A')
        hypotheses = entry.get('competing_hypotheses', [])
        h_summary = " | ".join(hypotheses) if hypotheses else "N/A"
        table_rows.append({
            "exp_id": entry.get('experiment_id', ''),
            "title": entry.get('title', ''),
            "tier": entry.get('evidence_tier', ''),
            "status": entry.get('status', ''),
            "winner": winning_h,
            "evaluation": entry.get('evidence_evaluation', '')[:90] + "..." if len(entry.get('evidence_evaluation', '')) > 90 else entry.get('evidence_evaluation', '')
        })
    return table_rows

registry_data = get_evidence_table_data()

layout = html.Div(
    [
        html.H3("SCIENTIFIC GOVERNANCE & EVIDENCE REGISTRY"),
        
        # KPI Row
        html.Div(
            [
                create_metric_panel("REGISTERED EXPERIMENTS", str(len(registry_data)), "Formal Competing Hypotheses", "info"),
                create_metric_panel("HIGHEST EVIDENCE TIER", "Level 4", "AUM Capacity Decay Bounds", "bull"),
                create_metric_panel("REPLICATED TIER", "Level 3", "7 Validated Research Phases", "info"),
            ],
            className="os-grid grid-3col", style={"marginBottom": "15px"}
        ),
        
        # Content Grid
        html.Div(
            [
                html.Div(
                    [
                        html.Div("CHIMERA EVIDENCE REGISTRY (COMPETING HYPOTHESES)", className="panel-title"),
                        dash_table.DataTable(
                            data=registry_data,
                            columns=[
                                {"name": "EXP ID", "id": "exp_id"},
                                {"name": "TITLE", "id": "title"},
                                {"name": "EVIDENCE TIER", "id": "tier"},
                                {"name": "WINNER", "id": "winner"},
                                {"name": "STATUS", "id": "status"},
                                {"name": "EMPIRICAL EVALUATION", "id": "evaluation"}
                            ],
                            style_table={'overflowX': 'auto'},
                            style_cell={'textAlign': 'left', 'fontSize': '12px'},
                            style_data_conditional=[
                                {'if': {'filter_query': '{status} = Established', 'column_id': 'status'}, 'color': '#00FF88'},
                                {'if': {'filter_query': '{status} = Replicated', 'column_id': 'status'}, 'color': '#00B4D8'},
                                {'if': {'column_id': 'exp_id'}, 'fontWeight': 'bold', 'color': '#FFC107'}
                            ]
                        )
                    ],
                    className="os-panel", style={"gridColumn": "span 2"}
                ),
                
                html.Div(
                    [
                        dcc.Graph(figure=create_kappa_chart(), config={'displayModeBar': False}, style={"height": "320px"})
                    ],
                    className="os-panel", style={"gridColumn": "span 2"}
                )
            ],
            className="os-grid grid-2col"
        )
    ]
)
