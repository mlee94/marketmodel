from dash import Dash, html, dcc, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from marketmodel.pages.overview import callbacks, cards


page_content = [
    html.H1(children='Overview'),
    html.Div(
        dbc.Container([
            dbc.Row([
                dbc.Col(cards.ad_spend_card),
                dbc.Col(cards.revenue_card)
            ], style={'height': '100%'}
            ),
            dbc.Row([
                dbc.Col(cards.pie_card),
            ], style={'height': '30%'}
            )
        ])
    )
]


def get_layout():
    return html.Div(page_content)
