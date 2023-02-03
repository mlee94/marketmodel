from dash import Dash, html, dcc, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from marketmodel.pages.transform import callbacks


posterior_distributions = dbc.Card([
    dbc.CardHeader(
        html.Div([
            html.Div('Posterior Distributions', style={'padding': '0 7px', 'flex': '0 1 auto'}),
        ], style={'display': 'flex', 'alignItems': 'center'})
    ),
    dbc.CardBody(
        dcc.Loading(
            id='loading-dist-plots',
            type='default',
            color='#1A2C35',
            children=([
                html.Div(
                    dcc.Graph(
                        id='posterior-violin-plot',
                        style={'height': '100%', 'width': '100%'},
                        config={'displayModeBar': False, 'displaylogo': False},
                    )
                ),
            ])
        )
    )
])

mmm_prediction_card = dbc.Card([
    dbc.CardHeader(
        html.Div([
            html.Div('Model Predictions vs Actuals', style={'padding': '0 7px', 'flex': '0 1 auto'}),
        ], style={'display': 'flex', 'alignItems': 'center'})
    ),
    dbc.CardBody(
        dcc.Loading(
            id='loading-dist-plots',
            type='default',
            color='#1A2C35',
            children=([
                html.Div(
                    dcc.Graph(
                        id='mmm-prediction-plot',
                        style={'height': '100%', 'width': '100%'},
                        config={'displayModeBar': False, 'displaylogo': False},
                    )
                ),
            ])
        )
    )
])


mmm_summary_stats = dbc.Card([
    dbc.CardHeader(
        html.Div([
            html.Div('Summary Statistics', style={'padding': '0 7px', 'flex': '0 1 auto'}),
        ], style={'display': 'flex', 'alignItems': 'center'})
    ),
    dbc.CardBody(
        dcc.Loading(
            id='loading-mmm-summary',
            type='default',
            color='#1A2C35',
            children=([
                html.Div(
                    id='mmm-summary-table',
                    style={'height': '100%', 'width': '100%'},
                ),
            ])
        )
    )
])


response_curves = dbc.Card([
    dbc.CardHeader(
        html.Div([
            html.Div('Response Curves', style={'padding': '0 7px', 'flex': '0 1 auto'}),
        ], style={'display': 'flex', 'alignItems': 'center'})
    ),
    dbc.CardBody(
        dcc.Loading(
            id='loading-response-plots',
            type='default',
            color='#1A2C35',
            children=([
                html.Div(
                    dcc.Graph(
                        id='response-plots',
                        style={'height': '100%', 'width': '100%'},
                        config={'displayModeBar': False, 'displaylogo': False},
                    )
                ),
            ])
        )
    )
])

roas_mean = dbc.Card([
    dbc.CardHeader(
        html.Div([
            html.Div('Return on Ad Spend (Mean)', style={'padding': '0 7px', 'flex': '0 1 auto'}),
        ], style={'display': 'flex', 'alignItems': 'center'})
    ),
    dbc.CardBody(
        dcc.Loading(
            id='loading-roas-plots',
            type='default',
            color='#1A2C35',
            children=([
                html.Div(
                    dcc.Graph(
                        id='avg-roas-plots',
                        style={'height': '100%', 'width': '100%'},
                        config={'displayModeBar': False, 'displaylogo': False},
                    )
                ),
            ])
        )
    )
])


roas_marginal = dbc.Card([
    dbc.CardHeader(
        html.Div([
            html.Div('Marginal Return on Ad Spend', style={'padding': '0 7px', 'flex': '0 1 auto'}),
        ], style={'display': 'flex', 'alignItems': 'center'})
    ),
    dbc.CardBody(
        dcc.Loading(
            id='loading-marginal-roas-plots',
            type='default',
            color='#1A2C35',
            children=([
                html.Div(
                    dcc.Graph(
                        id='marginal-roas-plots',
                        style={'height': '100%', 'width': '100%'},
                        config={'displayModeBar': False, 'displaylogo': False},
                    )
                ),
            ])
        )
    )
])


roas_summary = dbc.Card([
    dbc.CardHeader(
        html.Div([
            html.Div('Summary Return on Ad Spend (ROAS)', style={'padding': '0 7px', 'flex': '0 1 auto'}),
        ], style={'display': 'flex', 'alignItems': 'center'})
    ),
    dbc.CardBody(
        dcc.Loading(
            id='loading-summary-roas-plot',
            type='default',
            color='#1A2C35',
            children=([
                html.Div(
                    dcc.Graph(
                        id='summary-roas-plot',
                        style={'height': '100%', 'width': '100%'},
                        config={'displayModeBar': False, 'displaylogo': False},
                    )
                ),
            ])
        )
    )
])
