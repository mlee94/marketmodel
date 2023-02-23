from dash import Dash, html, dcc, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import dash_daq as daq
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from marketmodel.pages.overview import callbacks

targets = dcc.Dropdown(
    id='target',
)
channel_features = dcc.Dropdown(
    id='channel-features',
)
media_features = dcc.Dropdown(
    id='media-features',
    multi=True,
)
date_feature = dcc.Dropdown(
    id='date-feature',
)
cost_features = dcc.Dropdown(
    id='cost-features',
    multi=True,
)
extra_features = dcc.Dropdown(
    id='extra-features',
    multi=True,
)
controls = html.Div(
    children=[
        html.Div(['Date:', date_feature]),
        html.Div(['Channels:', channel_features]),
        html.Div(['Target/KPI:', targets]),
        html.Div(['Media Features (impressions or spending):', media_features]),
        html.Div(['Extra Features:', extra_features]),
        html.Div(['Cost Features (same as media if spending - used as prior):', cost_features]),
    ]
)

upload_div = html.Div(
        [
            dcc.Upload(
                    id='upload-data',
                    children=html.Div(
                            [
                                'Drag and Drop or ',
                                html.A('Select File')
                            ]
                    ),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
            ),
            html.Div(id='output-data-upload'),
            controls,
            dbc.Button('Upload File', id='upload-file-button',  color="primary"),
        ]
),

file_upload_card = dbc.Card([
    dbc.CardHeader(
        html.Div([
            html.Div('Data Upload', style={'padding': '0 7px', 'flex': '0 1 auto'}),
        ], style={'display': 'flex', 'alignItems': 'center'})
    ),
    dbc.CardBody(upload_div)
])

