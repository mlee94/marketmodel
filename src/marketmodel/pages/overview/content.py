from dash import Dash, html, dcc, Input, Output, State, no_update
import dash_daq as daq
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from marketmodel.pages.overview import callbacks, cards


page_content = [
    html.H1(children='Overview'),
    dbc.Card(children=[
        daq.BooleanSwitch(label='Test Toggle', id='test-switch', on=False, color="blue"),
        dbc.Button('Train New Model', id='train-model', color="primary"),
        ], style={"width": "15%"},
    ),
    html.Div(
        dbc.Container([
            dbc.Row([
                dbc.Col(cards.return_ad_spend_card, width=4),
                dbc.Col(cards.total_ad_spend_card, width=4),
                dbc.Col(cards.revenue_earnings_card, width=4),
            ], className="h-10"
            ),
            dbc.Row([
                dbc.Col(cards.ad_spend_card, width=6),
                dbc.Col(cards.revenue_card),
            ], style={'height': '40%'}
            ),
            dbc.Row([
                dbc.Col(cards.optimal_mix_card, width=6),
                dbc.Col(cards.marginal_return_ad_spend, width=6),
            ]),
            dbc.Row([
                dbc.Col(cards.rerun_mix_card, width=6, style={'display': 'inline-block'}),
            ]),
        ])
    )
]


def get_layout():
    return html.Div(page_content)
