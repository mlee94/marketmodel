import jax.numpy as jnp
import numpyro

from dash import Dash, html, dcc, Input, Output, State, no_update, callback_context
import dash_daq as daq
from dash_extensions.enrich import CycleBreakerInput
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

from marketmodel.pages.overview.content import *
from marketmodel.pages.transform.callbacks import *
from marketmodel.dash_config import app

DATA_PATH = Path(__file__).parents[4].joinpath('data')

_PALETTE = px.colors.qualitative.Plotly


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
    df = df.assign(percentage=lambda x: (x['totals'] / x['totals'].sum()).round(3))

    fig = px.bar(
        df.reset_index(), x="percentage", y="channel", text='percentage',
        hover_data={'totals': True}, color='channel', orientation='h', text_auto='.0%',
    )

    return fig


def revenue_time_series(df):
    col = df.columns[0]
    fig = px.line(df.reset_index(), x='dates', y=col)
    return fig


def get_optimal_mix_chart(df):
    df = df.rename('totals').astype(int).to_frame()
    df.index = df.index.rename('channel')

    fig = px.bar(df.reset_index(), x='totals', y='channel', text='totals', color='channel', orientation='h')
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

def get_marginal_roas_card(df):
    avg = df['ROAS'].mean()

    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="number",
        value=avg,
        number={"font":{"size":35}},
        domain={'row': 0, 'column': 0}
    ))
    fig.update_layout(
        grid={'rows': 1, 'columns': 1, 'pattern': "independent"},
        template={'data': {'indicator': [{
            'mode': "number",
        }]}
        },
        height=450,
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
    # Output('optimal_mix-chart', 'figure'),
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
    # fig4 = get_optimal_mix_chart(cost_totals)
    fig5 = get_return_ad_spend_card(return_on_spend)
    fig6 = get_total_ad_spend_card(cost_totals.sum().astype(int))
    fig7 = get_total_revenue_card(target_total)

    return fig5, fig6, fig7

@app.callback(
    Output('time-step-control', 'value'),
    CycleBreakerInput('budget-control', 'value'),
    Input('latest-daily-spending', 'value'),
    State('time-step-control', 'value'),
)
def update_celsius(budget, daily, time_step):
    if (budget and daily and time_step) is None:
        raise PreventUpdate()

    time.sleep(1)

    if (budget > (time_step * daily)*1.2) | (budget < (time_step * daily)*0.8):
        return int(budget / daily)
    else:
        raise PreventUpdate()

@app.callback(
    Output('budget-control', 'value'),
    Input('time-step-control', 'value'),
    Input('latest-daily-spending', 'value'),
    State('budget-control', 'value'),
)
def update_fahrenheit(time_step, daily, budget):
    if (time_step and budget and daily) is None:
        raise PreventUpdate()

    time.sleep(1)

    if (budget > (time_step * daily)*1.2) | (budget < (time_step * daily)*0.8):
        return int(time_step * daily)
    else:
        raise PreventUpdate()

@app.callback(
    Output('optimal-mix-chart', 'children'),
    Input('latest-daily-spending', 'value'),
    Input('rerun-optimiser', 'n_clicks'),
    State('budget-control', 'value'),
    State('time-step-control', 'value'),
    State('sample-data', 'data'),
)
def get_prescription(budget_per_day, n_clicks, budget, time_steps, data):
    df = pd.DataFrame.from_dict(data)

    # Long function call
    mmm_cache = 'mmm_test_cache'
    prediction_cache = 'mmm_predictions.csv'

    train_data, test_data, dates_train, dates_test = train_test_split(df, test_size=10)
    train_data, test_data, target_scaler, media_scaler, extra_features_scaler = preprocess_data(train_data, test_data)
    [X_media_test, X_extra_features_test, y_test] = test_data

    mmm = utils.load_model(DATA_PATH.joinpath(mmm_cache))
    # Get optimal mix
    X_media_test_unscaled = media_scaler.inverse_transform(X_media_test) # your *current budget here (+/- 20%).

    if not time_steps:
        time_steps = X_media_test_unscaled.shape[0]
    if not budget:
        budget = budget_per_day * time_steps

    jnp_mean = X_media_test_unscaled.mean()

    prices = jnp.broadcast_to(np.array([1]), (mmm.n_media_channels))

    solution = optimize_media.find_optimal_budgets(
        n_time_periods=time_steps,
        media_mix_model=mmm,
        budget=budget/jnp_mean,
        prices=prices,
        bounds_lower_pct=.3,
        bounds_upper_pct=.3,
    )
    if solution[0]['success']:
        optimal_budgets = pd.DataFrame((solution[0]['x']) * jnp_mean, index=mmm.media_names).squeeze()
        fig = dcc.Graph(
            id='optimal-mix-fig',
            figure=ad_spend_per_channel(optimal_budgets),
            style={'height': '50%', 'width': '100%'},
            config={'displayModeBar': False, 'displaylogo': False},
        )

        return fig
    else:
        print('Optimisation failed')


@app.callback(
    Output('revenue-chart', 'figure'),
    Output('marginal-roas-card', 'children'),
    Output('latest-daily-spending', 'value'),
    Output('budget-control', 'value'),
    Output('time-step-control', 'value'),
    Input('sample-data', 'data'),
    Input('train-model', 'n_clicks'),
)
def perform_training(data, train):
    if train is None:
        pass

    df = pd.DataFrame.from_dict(data)

    # Long function call
    mmm_cache = 'mmm_test_cache'
    prediction_cache = 'mmm_predictions.csv'

    train_data, test_data, dates_train, dates_test = train_test_split(df, test_size=10)
    train_data, test_data, target_scaler, media_scaler, extra_features_scaler = preprocess_data(train_data, test_data)
    [X_media_test, X_extra_features_test, y_test] = test_data

    if train:
        mmm, all_data = train_model(train_data, test_data, dates_train, dates_test, target_scaler)
        # Save mmm and csv file
        utils.save_model(mmm, DATA_PATH.joinpath(mmm_cache))
        all_data.to_csv(DATA_PATH.joinpath(prediction_cache), index=True)
    else:
        mmm = utils.load_model(DATA_PATH.joinpath(mmm_cache))
        all_data = (
            pd.read_csv(DATA_PATH.joinpath(prediction_cache), parse_dates=['dates'])
            .set_index('dates')
            .round(3)
        )

    optimal_model = mmm.model_name + ' ' + str(mmm._degrees_seasonality)
    optimal_prediction = all_data.get([optimal_model])

    roas_avg, roas_marginal = calculate_ROAS(mmm, test_data, target_scaler, optimal_prediction)
    marginal_return_ad_spend = (
        roas_marginal.mean(axis=0).rename_axis('channel').rename('ROAS').to_frame()
        .assign(measure='Marginal Return on Ad Spend')
    )

    fig3 = revenue_time_series(optimal_prediction)

    fig2 = dcc.Graph(
        id='marginal-roas-fig',
        figure=get_marginal_roas_card(marginal_return_ad_spend),
        style={'height': '100%', 'width': '100%'},
        config={'displayModeBar': False, 'displaylogo': False},
    ),

    X_media_test_unscaled = media_scaler.inverse_transform(X_media_test)


    budget_per_day = int(X_media_test_unscaled.mean(axis=0).sum())
    default_time_steps = X_media_test_unscaled.shape[0]
    default_budget = budget_per_day * default_time_steps

    return fig3, fig2, budget_per_day, default_budget, default_time_steps



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
    channel_totals.index = [f'channel_{str(int(v)-1)}'for (k, v) in channel_totals.index.str.split('_')]

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
