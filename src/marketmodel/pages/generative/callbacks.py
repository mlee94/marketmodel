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
from marketmodel.pages.generative import content


def generate_sample_data(data_size, test_size, n_channels, n_features):
    media_data, extra_features, target, costs = utils.simulate_dummy_data(
        data_size=data_size + test_size,
        n_media_channels=n_channels,
        n_extra_features=n_features,
    )
    params = [media_data, extra_features, target, costs]
    return params


@app.callback(
    Output('sample-data', 'data'),
    # Output('downloadable-data', 'data'),
    Input('submit-data-1', 'n_clicks'),
    State('data-length-1', 'value'),
    State('test-length-1', 'value'),
    State('channel-size-1', 'value'),
    State('extra-feature-size-1', 'value')
)
def generate_data(n_clicks, data_len, test_len, channel_sz, extra_features_sz):
    args = (n_clicks and data_len and test_len and channel_sz and extra_features_sz)

    if not args:
        raise PreventUpdate

    media_data, extra_features, target, costs = generate_sample_data(
        data_len,
        test_len,
        channel_sz,
        extra_features_sz,
    )

    channels_col = [f'channel_{i}' for i in range(1,media_data.shape[1]+1)]
    extra_features_col = [f'extra_feature_{i}' for i in range(1,extra_features.shape[1]+1)]
    columns = ['target'] + channels_col + extra_features_col


    df = pd.DataFrame(
        data=np.hstack((target.reshape(-1, 1), media_data, extra_features)),
        columns=columns
    )
    return df.to_dict(orient='list') # dcc.send_data_frame(df.to_csv, 'mmm_data.csv')