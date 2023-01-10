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


return_ad_spend_card = dbc.Card([
    dbc.CardHeader(
        html.Div([
            html.Div('Return On Ad Spend', style={'padding': '0 7px', 'flex': '0 1 auto'}),
        ], style={'display': 'flex', 'alignItems': 'center'})
    ),
    dbc.CardBody(
        dcc.Loading(
            id='loading-adreturn-card',
            type='default',
            color='#1A2C35',
            children=([
                html.Div(
                    dcc.Graph(
                        id='ad-return-indicator',
                        style={'height': '100%', 'width': '100%'},
                        config={'displayModeBar': False, 'displaylogo': False},
                    )
                ),
            ])
        )
    )
])


total_ad_spend_card = dbc.Card([
    dbc.CardHeader(
        html.Div([
            html.Div('Total Ad Spending', style={'padding': '0 7px', 'flex': '0 1 auto'}),
        ], style={'display': 'flex', 'alignItems': 'center'})
    ),
    dbc.CardBody(
        dcc.Loading(
            id='loading-adspend-card',
            type='default',
            color='#1A2C35',
            children=([
                html.Div(
                    dcc.Graph(
                        id='total-adspend-indicator',
                        style={'height': '100%', 'width': '100%'},
                        config={'displayModeBar': False, 'displaylogo': False},
                    ),
                ),
            ])
        )
    )
])


revenue_earnings_card = dbc.Card([
    dbc.CardHeader(
        html.Div([
            html.Div('Revenue Last 12 Months', style={'padding': '0 7px', 'flex': '0 1 auto'}),
        ], style={'display': 'flex', 'alignItems': 'center'})
    ),
    dbc.CardBody(
        dcc.Loading(
            id='loading-rev-card',
            type='default',
            color='#1A2C35',
            children=([
                html.Div(
                    dcc.Graph(
                        id='rev-indicator',
                        style={'height': '100%', 'width': '100%'},
                        config={'displayModeBar': False, 'displaylogo': False},
                    ),
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
    dbc.Button('🡠', id='back-button', outline=True, size="sm", className='mt-2 ml-2 col-1', style={'display': 'none'}),
    dbc.CardBody(
        dcc.Loading(
            id='loading-adspend-plots',
            type='default',
            color='#1A2C35',
            children=([
                html.Div(
                    dcc.Graph(
                        id='adspend-chart',
                        style={'height': '100%', 'width': '100%'},
                        config={'displayModeBar': False, 'displaylogo': False},
                    ),
                ),
            ])
        )
    )
])


revenue_card = dbc.Card([
    dbc.CardHeader(
        html.Div([
            html.Div('Revenue', style={'padding': '0 7px', 'flex': '0 1 auto'}),
        ], style={'display': 'flex', 'alignItems': 'center'})
    ),
    dbc.CardBody(
        dcc.Loading(
            id='loading-revenue-plots',
            type='default',
            color='#1A2C35',
            children=([
                html.Div(
                    dcc.Graph(
                        id='revenue-chart',
                        style={'height': '100%', 'width': '100%'},
                        config={'displayModeBar': False, 'displaylogo': False},
                    )
                ),
            ])
        )
    )
])

pie_card = dbc.Card([
    dbc.CardHeader(
        html.Div([
            html.Div('Market Mix Strategy', style={'padding': '0 7px', 'flex': '0 1 auto'}),
        ], style={'display': 'flex', 'alignItems': 'center'})
    ),
    dbc.CardBody(
        dcc.Loading(
            id='loading-pie-plots',
            type='default',
            color='#1A2C35',
            children=([
                html.Div(
                    dcc.Graph(
                        id='pie-chart',
                        style={'height': '100%', 'width': '100%'},
                        config={'displayModeBar': False, 'displaylogo': False},
                    ),
                ),
            ])
        )
    )
])