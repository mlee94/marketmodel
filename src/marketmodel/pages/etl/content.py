
from dash import Dash, html, dcc, Input, Output, State, no_update
import dash_bootstrap_components as dbc
from marketmodel.dash_config import app
from marketmodel.pages.etl.callbacks import *
from marketmodel.pages.etl.cards import *




page_content = [
    html.H1(children='Data Upload'),
    dbc.Container([
        dbc.Row([dbc.Col(file_upload_card, width=8)], style={'height': '100%'}),
    ]),
]


def get_layout():
    return html.Div(page_content)

