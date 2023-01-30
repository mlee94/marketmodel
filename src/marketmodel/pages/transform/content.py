from dash import Dash, html, dcc, Input, Output, State, no_update

import dash_daq as daq
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from marketmodel.pages.transform import callbacks, cards

tab1_content = [
    html.H1(children='Model Training'),
    daq.BooleanSwitch(id='train_boolean', on=False),
    html.Button('Load Data', id='load-data', n_clicks=0),
    html.Div(
        dbc.Container([
            dbc.Row([
                dbc.Col(cards.mmm_prediction_card, width=12),
            ],
            ),
            dbc.Row([
                dbc.Col(cards.mmm_summary_stats, width=12),
            ], style={'height': '40%'}
            ),
            dbc.Row([
                dbc.Col(cards.posterior_distributions, width=12),
            ], style={'height': '30%'}
            )
        ])
    )
]

tab2_content = [
    html.H1(children='Model Insights"'),
    html.Div(
        dbc.Container([
            dbc.Row([
                dbc.Col(cards.mmm_prediction_card, width=12),
            ],
            ),
            dbc.Row([
                dbc.Col(cards.mmm_summary_stats, width=12),
            ], style={'height': '40%'}
            ),
            dbc.Row([
                dbc.Col(cards.posterior_distributions, width=12),
            ], style={'height': '30%'}
            )
        ])
    )
]


page_content = dbc.Tabs(
    dbc.Tab(tab1_content, label='Model Training'),
    # dbc.Tab(tab2_content, label='Model Insights'),
)


def get_layout():
    return html.Div(page_content)
