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


ad_spend_card = dbc.Card([
    dbc.CardHeader(
        html.Div([
            html.Div('Ad Spend per Channel', style={'padding': '0 7px', 'flex': '0 1 auto'}),
        ], style={'display': 'flex', 'alignItems': 'center'})
    ),
    dbc.CardBody(
        dcc.Loading(
            id='loading-adspend-plots',
            type='default',
            color='#1A2C35',
            children=([
                html.Div(
                    id='adspend-chart',
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

revenue_card = dbc.Card([
    dbc.CardBody(
        dcc.Loading(
            id='loading-revenue-plots',
            type='default',
            color='#1A2C35',
            children=([
                html.Div(
                    id='revenue-chart',
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

pie_card = dbc.Card([
    dbc.CardBody(
        dcc.Loading(
            id='loading-pie-plots',
            type='default',
            color='#1A2C35',
            children=([
                html.Div(
                    id='pie-chart',
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