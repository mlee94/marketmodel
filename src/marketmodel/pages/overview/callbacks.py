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

from marketmodel.pages.overview.content import *
from marketmodel.dash_config import app


def create_chart(df):
    # Create figure with secondary y-axis

    fig = make_subplots(
        rows=len(df.columns),
        cols=1,
        specs=[[{"secondary_y": True}] for i in range(len(df.columns))],
        shared_xaxes=True,
        row_heights=[2] * len(df.columns),
        subplot_titles=[att for att in df.columns],
        vertical_spacing=0.04,
    )

    legend_counter = 0
    for idx, attribute in enumerate(df.columns):
        attribute_plot = df.get(attribute)

        x = attribute_plot.index
        y = attribute_plot

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=attribute,
                # line=dict(color=cols_mapping[c]),
                # legendgroup=c,
                # showlegend=True if legend_counter == 0 else False,
            ),
            row=idx+1,
            col=1,
            secondary_y=False,
        )
    # legend_counter = 1

    fig.update_layout(
        plot_bgcolor="#FFFFFF",
        hovermode='x',
        legend=dict(
            # orientation="h",
            # yanchor="bottom",
            # y=1.10,
            # xanchor="right",
            # x=0.95,
            # Adjust click behavior
            font=dict(size=8, family='sans-serif'),
            itemclick="toggleothers",
            itemdoubleclick="toggle",
        ),
        height=800,
    )
    fig.update_yaxes(showgrid=False, linecolor='#BCCCDC')
    fig.update_xaxes(
        dict(
            linecolor="#BCCCDC",
            showspikes=True,  # Show spike line for X-axis
            # Format spike
            spikethickness=2,
            spikedash="dot",
            spikecolor="#999999",
            spikemode="across",
        )
    )
        # fig.update_yaxes(
        #     title_text="<b>MW</b>",
        #     secondary_y=False,
        #     row=idx+1,
        #     col=1
        # )
        # fig.update_yaxes(
        #     title_text="<b>Price ($/MWh) </b>",
        #     secondary_y=True,
        #     row=idx+1,
        #     col=1
        # )

    dash_chart = dcc.Graph(
        id='raw-media-plot',
        figure=fig,
        style={'height': '100%', 'width': '100%'},
        config={'displayModeBar': False, 'displaylogo': False},
    )

    return dash_chart



def ad_spend_per_channel(df):
    df = df.rename('totals').astype(int).to_frame()
    df.index = df.index.rename('channel')

    fig = px.bar(df.reset_index(), x="totals", y="channel", text='totals', color='channel', orientation='h')
    dash_chart = dcc.Graph(
        id='raw-spend-per-channel-plot',
        figure=fig,
        style={'height': '100%', 'width': '100%'},
        config={'displayModeBar': False, 'displaylogo': False},
    )
    return dash_chart


def revenue_time_series(df):
    fig = px.line(df.get(['target']))
    dash_chart = dcc.Graph(
        id='raw-revenue-plot',
        figure=fig,
        style={'height': '100%', 'width': '100%'},
        config={'displayModeBar': False, 'displaylogo': False},
    )
    return dash_chart


def get_pie_chart(df):
    df = df.rename('totals').astype(int).to_frame()
    df.index = df.index.rename('channel')

    fig = px.pie(df.reset_index(), values='totals', names='channel', title='Market Mix Strategy')
    dash_chart = dcc.Graph(
        id='raw-pie-plot',
        figure=fig,
        style={'height': '100%', 'width': '100%'},
        config={'displayModeBar': False, 'displaylogo': False},
    )
    return dash_chart


@app.callback([
    # Output('channel-chart', 'children'),
    Output('adspend-chart', 'children'),
    Output('revenue-chart', 'children'),
    Output('pie-chart', 'children'),
], Input('sample-data', 'data'),
    # prevent_initial_call=True,
)
def display_widgets(data):
    if not data:
        raise PreventUpdate

    df = pd.DataFrame.from_dict(data)

    col_totals = df.sum(axis=0)
    channel_totals = col_totals[['channel' in s for s in col_totals.index]]
    target_total = (col_totals.loc['target'] / channel_totals.sum()).round(2)

    # CHARTS
    # fig1 = create_chart(df)
    fig2 = ad_spend_per_channel(channel_totals)
    fig3 = revenue_time_series(df)
    fig4 = get_pie_chart(channel_totals)

    return fig2, fig3, fig4