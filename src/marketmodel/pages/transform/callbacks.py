from pathlib import Path
import jax
import functools
from typing import Optional
import jax.numpy as jnp
import numpyro
import math

from operator import attrgetter

from itertools import product
from dash import Dash, html, dcc, Input, Output, State, no_update, dash_table
from dash.exceptions import PreventUpdate

from numpyro.diagnostics import summary
from jax.tree_util import tree_flatten, tree_map
import dash_bootstrap_components as dbc
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_percentage_error

from lightweight_mmm import preprocessing, lightweight_mmm, plot, optimize_media
from lightweight_mmm import utils

from marketmodel.pages.transform import cards
from marketmodel.dash_config import app


DATA_PATH = Path(__file__).parents[4].joinpath('data')

_PALETTE = px.colors.qualitative.Plotly


@functools.partial(jax.jit, static_argnames=("media_mix_model"))
def _make_single_prediction(media_mix_model: lightweight_mmm.LightweightMMM,
                            mock_media: jnp.ndarray,
                            extra_features: Optional[jnp.ndarray],
                            seed: Optional[int]
                            ) -> jnp.ndarray:
  """Makes a prediction of a single row.
  Serves as a helper function for making predictions individually for each media
  channel and one row at a time. It is meant to be used vmaped otherwise it can
  be slow as it's meant to be used for plotting curve responses only. Use
  lightweight_mmm.LightweightMMM for regular predict functionality.
  Args:
    media_mix_model: Media mix model to use for getting the predictions.
    mock_media: Mock media for this iteration of predictions.
    extra_features: Extra features to use for predictions.
    seed: Seed to use for PRNGKey during sampling. For replicability run
      this function and any other function that gets predictions with the same
      seed.
  Returns:
    A point estimate for the given data.
  """
  return media_mix_model.predict(
      media=jnp.expand_dims(mock_media, axis=0),
      extra_features=extra_features,
      seed=seed).mean(axis=0)


@functools.partial(
    jax.jit,
    static_argnames=("media_mix_model", "target_scaler"))
def _generate_diagonal_predictions(
    media_mix_model: lightweight_mmm.LightweightMMM,
    media_values: jnp.ndarray,
    extra_features: Optional[jnp.ndarray],
    target_scaler: Optional[preprocessing.CustomScaler],
    prediction_offset: jnp.ndarray,
    seed: Optional[int]):
  """Generates predictions for one value per channel leaving the rest to zero.
  This function does the following steps:
    - Vmaps the single prediction function on axis=0 of the media arg.
    - Diagonalizes the media input values so that each value is represented
      along side zeros on for the rest of the channels.
    - Generate predictions.
    - Unscale prediction if target_scaler is given.
  Args:
    media_mix_model: Media mix model to use for plotting the response curves.
    media_values: Media values.
    extra_features: Extra features values.
    target_scaler: Scaler used for scaling the target, to unscaled values and
      plot in the original scale.
    prediction_offset: The value of a prediction of an all zero media input.
    seed: Seed to use for PRNGKey during sampling. For replicability run
      this function and any other function that gets predictions with the same
      seed.
  Returns:
    The predictions for the given data.
  """
  make_predictions = jax.vmap(fun=_make_single_prediction,
                              in_axes=(None, 0, None, None))
  diagonal = jnp.eye(media_values.shape[0])
  if media_values.ndim == 2:  # Only two since we only provide one row
    diagonal = jnp.expand_dims(diagonal, axis=-1)
    media_values = jnp.expand_dims(media_values, axis=0)
  diag_media_values = diagonal * media_values
  predictions = make_predictions(
      media_mix_model,
      diag_media_values,
      extra_features,
      seed) - prediction_offset
  predictions = jnp.squeeze(predictions)
  if target_scaler:
    predictions = target_scaler.inverse_transform(predictions)
  if predictions.ndim == 2:
    predictions = jnp.sum(predictions, axis=-1)
  return predictions



def train_test_split(df, mapping, test_size=None):
    if test_size is None:
        test_size = 0
    # Split and scale data.
    data_size = len(df)
    split_point = data_size - test_size
    print(f'Splitting at data_size ({data_size}) less test_size ({test_size} = {split_point})')
    # Media data
    dates = df.get(mapping['date'])
    # df = df.drop(['dates'], axis=1)

    media = df.get(mapping['media'])
    cost = df.get(mapping['cost'])
    extra_features = df.get(mapping.get('extra_features'))
    target = df.get(mapping['target'])

    X_media_train = media.to_numpy()[:split_point, :]
    X_cost_train = cost.to_numpy()[:split_point, :]
    y_train = target.to_numpy()[:split_point]
    dates_train = dates.iloc[:split_point].to_numpy()

    X_media_test = media.to_numpy()[split_point:, :]
    y_test = target.to_numpy()[split_point:]
    dates_test = dates.iloc[split_point:].to_numpy()

    if not extra_features.empty:
        X_extra_features_train = extra_features.to_numpy()[:split_point, :]
        X_extra_features_test = extra_features.to_numpy()[split_point:, :]
    else:
        X_extra_features_train = None
        X_extra_features_test = None

    return [X_media_train, X_cost_train, X_extra_features_train, y_train], [X_media_test, X_extra_features_test, y_test], dates_train, dates_test


def preprocess_data(train, test):
    [X_media_train, X_cost_train, X_extra_features_train, y_train], [X_media_test, X_extra_features_test, y_test] = train, test

    media_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
    cost_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
    extra_features_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
    target_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)

    X_media_train = media_scaler.fit_transform(X_media_train)
    X_cost_train = cost_scaler.fit_transform(X_cost_train)

    if X_extra_features_train is not None:
        X_extra_features_train = extra_features_scaler.fit_transform(X_extra_features_train)
    else:
        X_extra_features_train = None

    y_train = target_scaler.fit_transform(y_train)

    X_media_test = media_scaler.transform(X_media_test)
    if X_extra_features_test is not None:
        X_extra_features_test = extra_features_scaler.transform(X_extra_features_test)
    else:
        X_extra_features_test = None
    # y_test = target_scaler.transform(y_test)

    train = [X_media_train, X_cost_train, X_extra_features_train, y_train]
    test = [X_media_test, X_extra_features_test, y_test]

    return train, test, target_scaler, media_scaler, extra_features_scaler


def scale_test_set(test, data_scaler):
    test_scaled = data_scaler.transform(test.to_numpy())
    return test_scaled



def plotly_plot_media_posteriors1(mmm):
    n_media_channels = np.shape(mmm.trace["coef_media"])[1]
    media_channel_posteriors = mmm.trace["coef_media"]
    quantiles = (0.05, 0.5, 0.95)

    fig = make_subplots(
        rows=n_media_channels, cols=1,
        row_heights=[1] * 5,
        specs=[
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
        ],
        subplot_titles=[f'Media Channel {i}' for i in range(n_media_channels)],
    )

    for idx, channel_i in enumerate(range(n_media_channels)):
        group_label = f'media_channel_{idx}'
        y = media_channel_posteriors[:, channel_i]

        distplfig = ff.create_distplot([np.array(y)], [group_label], bin_size=.05, fill='tozeroy', show_hist=False, show_rug=False)

        fig.add_trace(
            distplfig.data[0],
            row=idx+1, col=1,
        )
    fig.update_layout(barmode='overlay', title_text='Posterior Distributions')
    fig.update_xaxes(title='Coefficient (channel strength)')
    fig.update_yaxes(title='Probability Density')
    fig.write_html('kde_plots.html')
    return fig


def plotly_plot_media_posteriors2(mmm):
    n_media_channels = np.shape(mmm.trace["coef_media"])[1]
    media_channel_posteriors = mmm.trace["coef_media"]
    quantiles = (0.05, 0.5, 0.95)

    fig = go.Figure()

    for idx, channel_i in enumerate(range(n_media_channels)):
        group_label = f'media_channel_{idx}'
        y = media_channel_posteriors[:, channel_i]

        distplfig = go.Violin(x=y)

        fig.add_trace(distplfig)
    fig.update_traces(orientation='h', side='positive', width=3, points=False)
    fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False, title_text='Posterior Distributions')
    fig.write_html('kde_plots.html')
    return fig


def plotly_plot_media_posteriors3(mmm):
    n_media_channels = np.shape(mmm.trace["coef_media"])[1]
    media_channel_posteriors = mmm.trace["coef_media"]
    quantiles = (0.05, 0.5, 0.95)

    fig = make_subplots(
        rows=n_media_channels, cols=1,
        row_heights=[1] * 5,
        specs=[
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
        ],
        subplot_titles=[f'Media Channel {i}' for i in range(n_media_channels)],
    )

    for idx, channel_i in enumerate(range(n_media_channels)):
        group_label = f'media_channel_{idx}'
        y = media_channel_posteriors[:, channel_i]

        distplfig = go.Violin(x=y, name=group_label)

        fig.add_trace(
            distplfig,
            row=idx+1, col=1,
        )
    fig.update_traces(orientation='h', side='positive', width=3, points=False)
    fig.update_layout(barmode='overlay', title_text='Posterior Distributions')
    fig.update_xaxes(title='Coefficient (channel strength)')
    fig.update_yaxes(title='Probability Density')
    fig.write_html('kde_plots.html')
    return fig



def plotly_plot_media_posteriors4(mmm):
    n_media_channels = np.shape(mmm.trace["coef_media"])[1]
    media_channel_posteriors = mmm.trace["coef_media"]
    quantiles = (0.05, 0.5, 0.95)

    fig = make_subplots(
        rows=n_media_channels, cols=1,
        row_heights=[1] * n_media_channels,
        specs=[[{"type": "xy"}]] * n_media_channels,
        subplot_titles=[f'Media Channel {i}' for i in range(n_media_channels)],
        shared_yaxes=True,
        vertical_spacing=0.1,
    )

    for idx, channel_i in enumerate(range(n_media_channels)):
        group_label = f'Media Channel {idx}'
        y = media_channel_posteriors[:, channel_i]

        distplfig = go.Violin(x=y, name=group_label)

        fig.add_trace(
            distplfig,
            row=idx+1, col=1,
        )
    fig.update_layout(height=1000, barmode='overlay', title_text='Posterior Distributions')
    fig.update_xaxes(title='Coefficient (channel strength)')
    fig.update_yaxes(title='Probability Density')
    # fig.write_html('kde_plots.html')
    return fig


def train_model(train, test, dates_train, dates_test, target_scaler):
    [X_media_train, X_cost_train, X_extra_features_train, y_train], [X_media_test, X_extra_features_test, y_test] = train, test

    # adstock_models = ["adstock", "hill_adstock", "carryover"]
    # degrees_season = [1, 2, 3]

    adstock_models = ["hill_adstock"]
    degrees_season = [1]

    numpyro.set_host_device_count(2)

    start = time.time()

    mmm_instances = {}
    predictions = {}
    models = {}
    for model_name in adstock_models:
        mmm_instances[model_name] = {}
        models[model_name] = {}
        predictions[model_name] = {}

        for degrees in degrees_season:

            mmm = lightweight_mmm.LightweightMMM(model_name)
            mmm.fit(
                media=X_media_train,
                extra_features=X_extra_features_train,
                media_prior=X_cost_train.sum(axis=0),
                target=y_train,
                number_warmup=1000,
                number_samples=1000,
                number_chains=2,
                degrees_seasonality=degrees,
            )
            prediction = mmm.predict(
                media=X_media_test,
                extra_features=X_extra_features_test,
                target_scaler=target_scaler,
            )
            p = prediction.mean(axis=0)

            mape = mean_absolute_percentage_error(y_test, p)

            mmm_instances[model_name][degrees] = mmm
            models[model_name][degrees] = mape
            predictions[model_name][degrees] = np.array(p)
            print(f"model_name={model_name} degrees={degrees} MAPE={mape} samples={p[:3]}")

    end = time.time()
    time_elapsed = (end - start) / 60
    print(f'Train and CV time elapsed: {time_elapsed} mins')

    cross_validation_summary = pd.DataFrame.from_dict(models).rename_axis('degrees')
    optimal_degrees, optimal_model = cross_validation_summary.stack().idxmin()
    optimal_mmm = mmm_instances[optimal_model][optimal_degrees]

    prediction_time_series = pd.DataFrame.from_dict(predictions).rename_axis('degrees').stack()
    prediction_time_series.index = prediction_time_series.index.rename('model_name', level=-1)
    prediction_time_series = (
        prediction_time_series
        .rename('quantity')
        .reset_index()
        .assign(model_name=lambda x: x['model_name'] + ' ' + x['degrees'].astype(str))
        .drop(['degrees'], axis=1)
        .set_index('model_name')
    )
    actuals = (
        pd.DataFrame(np.array(y_test), columns=['quantity'])
        .assign(model_name='actuals')
        # .groupby(['model_name'])
        # .agg({'quantity': 'unique'})
        .set_index('model_name')
    )
    all_data = pd.concat([actuals, prediction_time_series]).explode('quantity')
    n_repeats = len(all_data) // len(dates_test)
    all_data = (
        all_data.assign(dates=np.tile(dates_test, (n_repeats, 1)).ravel())
        .set_index('dates', append=True)
        .squeeze()
        .unstack('model_name')
    )

    return optimal_mmm, all_data

def generate_summary_table(mmm):
    sites = mmm._mcmc._states[mmm._mcmc._sample_field]

    if isinstance(sites, dict):
        state_sample_field = attrgetter(mmm._mcmc._sample_field)(mmm._mcmc._last_state)
        # XXX: there might be the case that state.z is not a dictionary but
        # its postprocessed value `sites` is a dictionary.
        # TODO: in general, when both `sites` and `state.z` are dictionaries,
        # they can have different key names, not necessary due to deterministic
        # behavior. We might revise this logic if needed in the future.
        if isinstance(state_sample_field, dict):
            sites = {
                k: v
                for k, v in mmm._mcmc._states[mmm._mcmc._sample_field].items()
                if k in state_sample_field
            }

    summary_dict = summary(sites, group_by_chain=True)
    columns = list(list(summary_dict.values())[0].keys())

    data_dict = {}
    for name, stats_dict in summary_dict.items():
        shape = stats_dict["mean"].shape
        if len(shape) == 0:
            data_dict[name] = stats_dict.values()
        else:
            for idx in product(*map(range, shape)):
                idx_str = "[{}]".format(",".join(map(str, idx)))
                data_dict[name + idx_str] = [v[idx] for v in stats_dict.values()]

    data = pd.DataFrame.from_dict(data_dict, orient='index', columns=columns).round(2)
    data.index = data.index.rename('Attribute')
    data = data.reset_index()

    table = dash_table.DataTable(
        id='mmm-summary-table',
        data=data.to_dict('records'),
        columns=[{"name": i, "id": i} for i in data.columns],
        style_cell={
            'font_size': '14px',
            'text_align': 'center'
        },
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto',
            'lineHeight': '15px'
        },
    )

    return table


def generate_time_series_plot(df):
    df = df.stack()
    df.index = df.index.rename('model_name', -1)
    df = df.rename('quantity').to_frame()

    max_y = df.quantity.max()*1.1
    fig = px.line(df.reset_index(), x='dates', y='quantity', color='model_name')
    fig.update_xaxes(
        title='Date',
        linecolor="#BCCCDC",
        showspikes=True,  # Show spike line for X-axis
        # Format spike
        spikethickness=2,
        spikedash="dot",
        spikecolor="#999999",
        spikemode="across",
    )
    fig.update_yaxes(
        title='Revenue ($)',
        # range=[0, max_y]
    )
    fig.update_layout(
        plot_bgcolor="#FFFFFF",
        hovermode='x'
    )

    return fig


def generate_response_curves(mmm, target_scaler=None, media_scaler=None):
    percentage_add = 0.2
    steps = 50
    seed = None

    media = mmm.media
    media_maxes = media.max(axis=0) * (1 + percentage_add)
    if mmm._extra_features is not None:
        extra_features = jnp.expand_dims(
            mmm._extra_features.mean(axis=0), axis=0)
    else:
        extra_features = None

    media_ranges = jnp.expand_dims(
        jnp.linspace(start=0, stop=media_maxes, num=steps), axis=0
    )
    make_predictions = jax.vmap(
        jax.vmap(_make_single_prediction,
                 in_axes=(None, 0, None, None),
                 out_axes=0),
        in_axes=(None, 0, None, None), out_axes=1
    )
    diagonal = jnp.repeat(
        jnp.eye(mmm.n_media_channels), steps,
        axis=0).reshape(mmm.n_media_channels, steps,
                        mmm.n_media_channels)

    prediction_offset = mmm.predict(
        media=jnp.zeros((1, *media.shape[1:])),
        extra_features=extra_features).mean(axis=0)

    if media.ndim == 3:
        diagonal = jnp.expand_dims(diagonal, axis=-1)
        prediction_offset = jnp.expand_dims(prediction_offset, axis=0)
    mock_media = media_ranges * diagonal
    predictions = jnp.squeeze(a=make_predictions(mmm,
                                                 mock_media,
                                                 extra_features,
                                                 seed))
    predictions = predictions - prediction_offset
    media_ranges = jnp.squeeze(media_ranges)

    if target_scaler:
        predictions = target_scaler.inverse_transform(predictions)

    if media_scaler:
        media_ranges = media_scaler.inverse_transform(media_ranges)
      #
      # if prices is not None:
      #   if media.ndim == 3:
      #     prices = jnp.expand_dims(prices, axis=-1)
      #   media_ranges *= prices

    if predictions.ndim == 3:
        media_ranges = jnp.sum(media_ranges, axis=-1)
        predictions = jnp.sum(predictions, axis=-1)

    kpi_label = "KPI" if target_scaler else "Normalized KPI"
    media_label = "Normalized Spend" if not media_scaler else "Spend"

    n_columns = 3
    n_media_channels = mmm.n_media_channels

    if n_media_channels % n_columns == 0:
        n_rows = n_media_channels // n_columns + 1
        specs = [[{"type": "xy"}] * n_columns] * (n_rows - 1)
        specs_left = n_media_channels - ((n_rows-1) * n_columns)
        specs += [[{'colspan': specs_left} if i == 0 else None for i in range(n_columns)]]
    else:
        n_rows = n_media_channels // n_columns + 2
        specs = [[{"type": "xy"}] * n_columns] * (n_rows - 1)
        specs += [[{'colspan': n_columns}, None, None]]


    fig = make_subplots(
        rows=n_rows, cols=n_columns,
        specs=specs,
        subplot_titles=[f'Response {i}' for i in range(n_media_channels)],
        vertical_spacing=0.1,
    )

    channel_i = 0
    for row in range(n_rows):
        row_no = row + 1
        for col in range(n_columns):
            col_no = col + 1

            if (channel_i == n_media_channels) & (row_no != n_rows):
                continue

            if (channel_i == n_media_channels) & (row_no == n_rows):
                for chan in range(n_media_channels):
                    group_label = f'Media Channel {chan}'
                    data = go.Scatter(
                        x=media_ranges[:, chan], y=predictions[:, chan],
                        line=dict(color=_PALETTE[chan]),
                        name=group_label,
                        legendgroup=group_label,
                        showlegend=False,
                    )
                    fig.add_trace(
                        data,
                        row=row_no, col=col_no,
                    )
                break
            else:
                group_label = f'Media Channel {channel_i}'

                data = go.Scatter(
                    x=media_ranges[:, channel_i], y=predictions[:, channel_i],
                    name=group_label,
                    legendgroup=group_label,
                    line=dict(color=_PALETTE[channel_i]),
                )

                fig.add_trace(
                    data,
                    row=row_no, col=col_no,
                )
            channel_i += 1

    fig.update_layout(
        height=1000,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="left",
            x=0.05
        ),
    )
    fig.update_xaxes(title=media_label)
    fig.update_yaxes(title=kpi_label)

    return fig


def calculate_ROAS(mmm, test, target_scaler, optimal_prediction):
    [X_media_test, X_extra_features_test, y_test] = test

    marginal_roas = {}
    channel_roas = {}
    for idx, channel in enumerate(mmm.media_names):
        y_hat_0_spend = mmm.predict(
            media=X_media_test.at[:, idx].set(0),
            extra_features=X_extra_features_test,
            target_scaler=target_scaler,
        )
        y_hat_historical = jnp.broadcast_to(optimal_prediction.to_numpy().astype(float), y_hat_0_spend.T.shape).T
        y_hat_delta = jnp.subtract(y_hat_historical, y_hat_0_spend)

        # Sum over time periods
        posterior_avg_roas = jnp.divide(y_hat_delta.sum(axis=1), X_media_test[:, idx].sum())

        y_hat_1_pct_spend = mmm.predict(
            # Increase spending of channel i by 1%
            media=X_media_test.at[:, idx].set(jnp.multiply(X_media_test[:, idx], 1.01)),
            extra_features=X_extra_features_test,
            target_scaler=target_scaler,
        )
        y_hat_marginal_delta = jnp.subtract(y_hat_1_pct_spend, y_hat_historical)

        # Sum over time periods
        posterior_marginal_roas = jnp.divide(
            y_hat_marginal_delta.sum(axis=1),
            jnp.multiply(X_media_test[:, idx].sum(), 0.01),
        )

        channel_roas[channel] = posterior_avg_roas
        marginal_roas[channel] = posterior_marginal_roas

    df1 = pd.DataFrame.from_dict(channel_roas)
    df2 = pd.DataFrame.from_dict(marginal_roas)

    return df1, df2


def plotly_plot_roas_posteriors(df):
    n_media_channels = len(df.columns)

    fig = make_subplots(
        rows=n_media_channels, cols=1,
        row_heights=[1] * n_media_channels,
        specs=[[{"type": "xy"}]] * n_media_channels,
        subplot_titles=[f'Media Channel {i}' for i in range(n_media_channels)],
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.1,
    )

    for idx, channel_i in enumerate(df.columns):
        y = df.get(channel_i)
        channel_i = channel_i.replace('_', ' ')
        group_label = f'Media {channel_i}'

        distplfig = go.Violin(x=y, name=group_label)

        fig.add_trace(
            distplfig,
            row=idx+1, col=1,
        )
    fig.update_layout(height=1000, barmode='overlay', title_text='Posterior Distributions')
    fig.update_xaxes(title='ROAS')
    fig.update_yaxes(title='Probability Density')
    # fig.write_html('kde_plots.html')
    return fig


def plotly_dot_plot(df):
    fig = px.scatter(df.reset_index(), y="ROAS", x="channel", color="measure", symbol="measure")
    fig.update_traces(marker_size=30)
    return fig


def plotly_bar_plot(df):
    fig = px.bar(df.reset_index(), x="channel", y='ROAS', color="measure", barmode="group")

    return fig


@app.callback(
    [
        Output('posterior-violin-plot', 'figure'),
        Output('response-plots', 'figure'),
        Output('avg-roas-plots', 'figure'),
        Output('marginal-roas-plots', 'figure'),
        Output('summary-roas-plot', 'figure'),
        Output('mmm-prediction-plot', 'figure'),
        Output('mmm-summary-table', 'children'),
    ],
    Input('load-data', 'n_clicks'),
    State('train_boolean', 'on'),
    State('sample-data', 'data'),
)
def load_mmm(load, train, data):
    if not load:
        raise PreventUpdate

    df = pd.DataFrame.from_dict(data)

    # Long function call
    mmm_cache = 'mmm_test_cache'
    prediction_cache = 'mmm_predictions.csv'

    train_data, test_data, dates_train, dates_test = train_test_split(df, test_size=10)
    train_data, test_data, target_scaler, media_scaler, extra_features_scaler = preprocess_data(train_data, test_data)

    if train:
        mmm, all_data = train_model(train_data, test_data, dates_train, dates_test, target_scaler)
        # Save mmm and csv file
        utils.save_model(mmm, DATA_PATH.joinpath(mmm_cache))
        all_data.to_csv(DATA_PATH.joinpath(prediction_cache), index=True)
    else:
        # Otherwise load from file
        mmm = utils.load_model(DATA_PATH.joinpath(mmm_cache))
        all_data = (
            pd.read_csv(DATA_PATH.joinpath(prediction_cache), parse_dates=['dates'])
            .set_index('dates')
            .round(3)
        )

    optimal_model = mmm.model_name + ' ' + str(mmm._degrees_seasonality)
    optimal_prediction = all_data.get([optimal_model])

    roas_avg, roas_marginal = calculate_ROAS(mmm, test_data, target_scaler, optimal_prediction)

    point_estimates = (
        pd.concat([
            roas_avg.mean(axis=0).rename_axis('channel').rename('ROAS').to_frame()
            .assign(measure='Average Return on Ad Spend'),
            roas_marginal.mean(axis=0).rename_axis('channel').rename('ROAS').to_frame()
            .assign(measure='Marginal Return on Ad Spend')
        ])
    )

    media_effect_hat, roi_hat = mmm.get_posterior_metrics()
    media_names = mmm.media_names
    # plot.plot_media_channel_posteriors(media_mix_model=mmm, channel_names=media_names)
    # Media effect quantifies the strength of the impact of this channel on revenue (but could be expensive)
    # Rule of thumb would be allocate more budget to high ROI channels and less to low ROI ones.
    # But good ROI channels might not be scalable (diminishing returns).

    # plot.plot_bars_media_metrics(metric=media_effect_hat, channel_names=media_names)
    # plot.plot_bars_media_metrics(metric=roi_hat, channel_names=media_names)
    # plot.plot_media_baseline_contribution_area_plot

    fig = plotly_plot_media_posteriors4(mmm)
    time_series = generate_time_series_plot(all_data)
    table = generate_summary_table(mmm)
    fig2 = generate_response_curves(mmm)
    fig3 = plotly_plot_roas_posteriors(roas_avg)
    fig4 = plotly_plot_roas_posteriors(roas_marginal)
    fig5 = plotly_dot_plot(point_estimates)


    return fig, fig2, fig3, fig4, fig5, time_series, table