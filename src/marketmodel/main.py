import jax.numpy as jnp
import numpyro


import pandas as pd
import numpy as np
from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from lightweight_mmm import utils

from marketmodel.pages import generative, overview
from marketmodel.dash_config import app


app.layout = html.Div(
    children=[
        dcc.Location(id='url', refresh=False),
        dcc.Store('sample-data', data={}, storage_type='session'),
        html.Div(html.Div(id='page-content')),
    ]
)


@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname'),
)
def load_app(pathname):
    if not pathname:
        return no_update

    if pathname == '/':
        return generative.get_layout()
    elif pathname == '/overview':
        return overview.get_layout()


if __name__ == '__main__':
    app.run_server(debug=True)