import jax.numpy as jnp
import numpyro

from dash import Dash, html, dcc, Input, Output, State, no_update
import dash_daq as daq
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lightweight_mmm import lightweight_mmm
from lightweight_mmm import preprocessing
from lightweight_mmm import utils

from marketmodel.dash_config import app

sidebar = html.Div([
        dbc.Col([html.Label('Generate Dummy Data')], style={'margin-bottom': '50px'}),
        dbc.Col([
            daq.NumericInput(
                id='data-length-1',
                label='Training Data Length',
                value=100,
                min=0,
                max=10000,
                style={'margin-right': '10px'}
            ),
            daq.NumericInput(
                id='test-length-1',
                label='Test/Prediction Length',
                value=20,
                min=0,
                max=10000,
                style={'margin-right': '10px'}
            ),
            daq.NumericInput(
                id='channel-size-1',
                label='How many channels',
                value=5,
                min=1,
                max=10,
                style={'margin-right': '10px'},
            ),
            daq.NumericInput(
                id='extra-feature-size-1',
                label='How many extra features',
                value=1,
                min=1,
                max=5,
                style={'margin-right': '10px'},
            ),
            dbc.Button(id='submit-data-1', n_clicks=0, children='Generate Data', color='primary'),
        ], style={'width': '50%', 'display': 'flex'})
], style={"margin-left": "50px"})