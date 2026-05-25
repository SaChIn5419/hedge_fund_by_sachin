import dash
from dash import html, dash_table
from data_layer.loader import load_trade_log, get_current_holdings

dash.register_page(__name__, path='/portfolio', name='Portfolio Book')

trades = load_trade_log()
current_holdings = get_current_holdings(trades)

layout = html.Div(
    [
        html.H2("Current Portfolio Book"),
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
