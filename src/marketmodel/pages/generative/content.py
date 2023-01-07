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
import time

from lightweight_mmm import lightweight_mmm
from lightweight_mmm import preprocessing
from lightweight_mmm import utils

from marketmodel.pages.features import sidebar
from marketmodel.pages.generative.callbacks import *
from marketmodel.dash_config import app


def generate_sample_data(data_size, test_size, n_channels, n_features):
    media_data, extra_features, target, costs = utils.simulate_dummy_data(
        data_size=data_size + test_size,
        n_media_channels=n_channels,
        n_extra_features=n_features,
    )
    params = [media_data, extra_features, target, costs]
    return params


page_content = [
    # dcc.Download('downloadable-data'),
    html.H1(children='Media Mix Modelling'),
    dbc.Row(children=dbc.Col(sidebar.sidebar)),
]


def get_layout():
    return html.Div(page_content)
