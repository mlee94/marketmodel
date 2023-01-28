import jax.numpy as jnp
import numpyro

from dash import Dash, html, dcc, Input, Output, State, no_update, callback_context
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

    dash_chart = (
        dcc.Graph(
            id='raw-media-plot',
            figure=fig,
            style={'height': '100%', 'width': '100%'},
            config={'displayModeBar': False, 'displaylogo': False},
        ),
    )

    return dash_chart



def ad_spend_per_channel(df):
    df = df.rename('totals').astype(int).to_frame()
    df.index = df.index.rename('channel')

    fig = px.bar(df.reset_index(), x="totals", y="channel", text='totals', color='channel', orientation='h')

    return fig


def revenue_time_series(df):
    fig = px.line(df.reset_index(), x='dates', y='target')
    return fig


def get_pie_chart(df):
    df = df.rename('totals').astype(int).to_frame()
    df.index = df.index.rename('channel')

    fig = px.pie(df.reset_index(), values='totals', names='channel')
    return fig


def get_return_ad_spend_card(total):
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="number",
        value=total,
        number={"font":{"size":35}},
        domain={'row': 0, 'column': 0}
    ))
    fig.update_layout(
        grid={'rows': 1, 'columns': 1, 'pattern': "independent"},
        template={'data': {'indicator': [{
            'mode': "number",
        }]}
        },
        height=200,
    )
    return fig


def get_total_ad_spend_card(total):
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="number",
        value=total,
        number={"prefix": "$", "font":{"size":35}},
        domain={'row': 0, 'column': 0}
    ))
    fig.update_layout(
        grid={'rows': 1, 'columns': 1, 'pattern': "independent"},
        template={'data': {'indicator': [{
            'mode': "number",
        }]}
        },
        height=200,
    )
    return fig


def get_total_revenue_card(total):
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="number",
        value=total,
        number={"prefix": "$", "font":{"size":35}},
        domain={'row': 0, 'column': 0}
    ))
    fig.update_layout(
        grid={'rows': 1, 'columns': 1, 'pattern': "independent"},
        template={'data': {'indicator': [{
            'mode': "number",
        }]},
        },
        height=200,
    )
    return fig


@app.callback([
    # Output('channel-chart', 'children'),
    Output('revenue-chart', 'figure'),
    Output('pie-chart', 'figure'),
    Output('ad-return-indicator', 'figure'),
    Output('total-adspend-indicator', 'figure'),
    Output('rev-indicator', 'figure'),
], Input('sample-data', 'data'),
    # prevent_initial_call=True,
)
def display_widgets(data):
    if not data:
        raise PreventUpdate

    df = pd.DataFrame.from_dict(data)

    col_totals = df.sum(axis=0)
    impression_totals = col_totals[['channel' in s for s in col_totals.index]]
    cost_totals = col_totals[['cost' in s for s in col_totals.index]]
    target_total = col_totals.loc['target']
    return_on_spend = (target_total / cost_totals.sum()).round(2)

    # CHARTS
    # fig1 = create_chart(df)
    fig3 = revenue_time_series(df)
    fig4 = get_pie_chart(cost_totals)
    fig5 = get_return_ad_spend_card(return_on_spend)
    fig6 = get_total_ad_spend_card(cost_totals.sum().astype(int))
    fig7 = get_total_revenue_card(target_total)

    return fig3, fig4, fig5, fig6, fig7




@app.callback(
    Output('adspend-chart', 'figure'),
    Output('back-button', 'style'), #to hide/unhide the back button
    Input('adspend-chart', 'clickData'),    #for getting the vendor name from graph
    Input('back-button', 'n_clicks'),
    Input('sample-data', 'data'),
)
def drilldown(click_data, n_clicks, data):
    # https://community.plotly.com/t/show-and-tell-drill-down-functionality-in-dash-using-callback-context/54403?u=atharvakatre
    if not data:
        raise PreventUpdate

    data = pd.DataFrame.from_dict(data).set_index('dates')

    data_stacked = data.stack()
    data_stacked.index = data_stacked.index.rename('attribute', -1)
    data_stacked = data_stacked.rename('quantity').to_frame().reset_index()

    col_totals = data.sum(axis=0)
    channel_totals = col_totals[['cost' in s for s in col_totals.index]]

    # using callback context to check which input was fired
    # trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    trigger_id = callback_context.triggered_id

    if trigger_id == 'adspend-chart':

        # get vendor name from clickData
        if click_data is not None:
            channel = click_data['points'][0]['label']

            if channel in data_stacked['attribute'].unique():
                # creating df for clicked vendor
                channel_sales_df = data_stacked[data_stacked['attribute'] == channel]

                # generating product sales bar graph
                fig = px.line(channel_sales_df, x='dates', y='quantity', color='attribute')
                fig.update_layout(title=f'<b>{channel} spend<b>', showlegend=False, template='presentation')
                return fig, {'display':'block'}     #returning the fig and unhiding the back button

            else:
                return ad_spend_per_channel(channel_totals), {'display': 'none'}     #hiding the back button

    else:
        return ad_spend_per_channel(channel_totals), {'display':'none'}
