import plotly.io as pio
from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix

from dash_extensions.enrich import (
    DashProxy,
    ServersideOutputTransform,
    TriggerTransform,
    MultiplexerTransform,
)
import dash_bootstrap_components as dbc

pio.json.config.default_engine = 'orjson'

server = Flask(f'{__package__}')

server.wsgi_app = ProxyFix(
    server.wsgi_app,
    x_for=1,
    x_proto=1,
    x_host=1,
    x_prefix=1,
)


FA = "https://use.fontawesome.com/releases/v5.15.4/css/all.css"

app = DashProxy(
    f'{__package__}',
    server=server,
    assets_folder='assets',
    # external_stylesheets=[dbc.themes.BOOTSTRAP, FA],
    transforms=[
        TriggerTransform(),
        MultiplexerTransform(),
        ServersideOutputTransform(session_check=False, arg_check=False),
    ]
)
server = app.server
app.config.suppress_callback_exceptions = True
