from dash import Dash, html, dcc, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from marketmodel.pages.overview import callbacks


media_time_series_card = dbc.Card([
    dbc.CardHeader(
        html.Div([
            html.Div('Channels', style={'padding': '0 7px', 'flex': '0 1 auto'}),
        ], style={'display': 'flex', 'alignItems': 'center'})
    ),
    dbc.CardBody(
        dcc.Loading(
            id='loading-channel-plots',
            type='default',
            color='#1A2C35',
            children=([
                html.Div(
                    id='channel-chart',
                    style={
                        'display': 'flex',
                        'margin': '0 5px',
                        'alignItems': 'center',
                    }
                ),
            ])
        )
    )
])

page_content = [
    html.H1(children='Overview'),
    html.Div(
        dbc.Container([
            dbc.Row(
                children=dbc.Col(media_time_series_card),
                # dbc.Col(revenue_card),
                style={'height': '100%'}
            ),
            # dbc.Row([
            #     dbc.Col(pie_card),
            # ], style={'height': '30%'}
            # )
        ])
    )
]


def get_layout():
    return html.Div(page_content)
