from dash import Dash, html, dcc, Input, Output, State, no_update
import dash_daq as daq
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from marketmodel.pages.overview import callbacks, cards


page_content = [
    html.H1(children='Overview'),
    dbc.Button('Train New Model', id='train-model', color="primary"),
    html.Div(
        dbc.Container([
            dbc.Row([
                dbc.Col(cards.return_ad_spend_card, width=4),
                dbc.Col(cards.total_ad_spend_card, width=4),
                dbc.Col(cards.revenue_earnings_card, width=4),
            ], className="h-15"
            ),
            dbc.Row([
                dbc.Col(cards.ad_spend_card, width=7),
                dbc.Col(cards.revenue_card),
            ], style={'height': '40%'}
            ),
            dbc.Row([
                dbc.Col(cards.optimal_mix_card, width=7),
                dbc.Col(cards.marginal_return_ad_spend)
            ], style={'height': '30%'}
            )
        ])
    )
]


def get_layout():
    return html.Div(page_content)
