from dash import Dash, html, dcc, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import dash_daq as daq
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
            html.Div(
                html.Img(src='assets/return-on-ad-spend.png', height='60px', style={'filter': 'invert(100%)'}),
                style={'height': '90px', 'width': '90px'}, className='img describeblue',
            ),
            html.Div('Actual Return On Ad Spend', style={'padding': '0 7px', 'flex': '0 1 auto'}),
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
            html.Div(
                html.Img(src='assets/ad-spend-last-12-months.png', height='60px', style={'filter': 'invert(100%)'}),
                style={'height': '90px', 'width': '90px'}, className='img describeblue',
            ),
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
            html.Div(
                html.Img(src='assets/revenue-last-12-months.png', height='60px', style={'filter': 'invert(100%)'}),
                style={'height': '90px', 'width': '90px'}, className='img describeblue',
            ),
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
            html.Div(
                html.Img(src='assets/Metrics.png', height='60px', style={'filter': 'invert(100%)'}),
                style={'height': '90px', 'width': '90px'}, className='img diagnoseblue',
            ),
            html.Div('Ad Spend per Channel', style={'padding': '0 7px', 'flex': '0 1 auto'}),
        ], style={'display': 'flex', 'alignItems': 'center'})
    ),
    dbc.Button(html.I(className='fa fa-arrow-left'), id='back-button', color="dark", outline=True, size="sm", className='mt-2 ml-2 col-1', style={'display': 'none'}),
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
            html.Div(
                html.Img(src='assets/Metrics.png', height='60px', style={'filter': 'invert(100%)'}),
                style={'height': '90px', 'width': '90px'}, className='img predictblue',
            ),
            html.Div('Revenue Projection', style={'padding': '0 7px', 'flex': '0 1 auto'}),
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

optimal_mix_card = dbc.Card([
    dbc.CardHeader(
        html.Div([
            html.Div(
                html.Img(src='assets/Metrics.png', height='60px', style={'filter': 'invert(100%)'}),
                style={'height': '90px', 'width': '90px'}, className='img prescribeblue',
            ),
            html.Div('Market Mix Strategy', style={'padding': '0 7px', 'flex': '0 1 auto'}),
        ], style={'display': 'flex', 'alignItems': 'center'})
    ),
    dbc.CardBody(
        dcc.Loading(
            id='loading-mix-plots',
            type='default',
            color='#1A2C35',
            children=([
                html.Div(id='optimal-mix-chart'),
            ])
        )
    )
])

budget_control = html.Div(
    dbc.Row([
        dbc.Col([
            html.Label('Budget'),
            daq.NumericInput(
                id='budget-control',
                min=0,
                max=100000,
                size=120,
            ),
        ], style={
            'paddingRight': '7px',
            'width': '100%',
            'display': 'inline-block'
        })
    ], justify='center', align='center')
)


time_step_control = html.Div(
    dbc.Row([
        dbc.Col([
            html.Label('Time Steps Ahead'),
            daq.NumericInput(
                id='time-step-control',
                min=1,
                max=365,
                size=120,
            ),
        ], style={
            'paddingRight': '7px',
            'width': '100%',
            'display': 'inline-block'
        })
    ], justify='center', align='center')
)

latest_spending_div = html.Div([
    daq.NumericInput(
        id='latest-daily-spending',
        disabled=True,
    ),
], style={'display': 'none'})

rerun_mix_card = dbc.Card([
    dbc.CardHeader(
        html.Div([
            html.Div(
                html.Img(src='assets/Metrics.png', height='60px', style={'filter': 'invert(100%)'}),
                style={'height': '90px', 'width': '90px'}, className='img prescribeblue',
            ),
            html.Div('Mix Optimiser', style={'padding': '0 7px', 'flex': '0 1 auto'}),
        ], style={'display': 'flex', 'alignItems': 'center'})
    ),
    dbc.CardBody(
        html.Div(children=[
            html.Div([latest_spending_div]),
            html.Div([budget_control], style={'margin-left': '5rem', 'margin-bottom': '1rem'}),
            html.Div([time_step_control], style={'margin-left': '5rem', 'margin-bottom': '1rem'}),
            dbc.Button('Rerun Optimiser', id='rerun-optimiser', color="primary"),
        ], style={'alignItems': 'center'}),
    )
])




marginal_return_ad_spend = dbc.Card([
    dbc.CardHeader(
        html.Div([
            html.Div(
                html.Img(src='assets/Metrics.png', height='60px', style={'filter': 'invert(100%)'}),
                style={'height': '90px', 'width': '90px'}, className='img prescribeblue',
            ),
            html.Div('Marginal Return on Ad Spend', style={'padding': '0 7px', 'flex': '0 1 auto'}),
        ], style={'display': 'flex', 'alignItems': 'center'})
    ),
    dbc.CardBody(
        dcc.Loading(
            id='loading-marginal-roas-card',
            type='default',
            color='#1A2C35',
            children=([
                html.Div(id='marginal-roas-card', style={'height': '100%', 'display': 'flex', 'alignItems': 'center'}),
            ])
        )
    )
])