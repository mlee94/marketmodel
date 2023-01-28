import jax.numpy as jnp
import numpyro

from dash import Dash, html, dcc, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_percentage_error

from lightweight_mmm import preprocessing, lightweight_mmm, plot, optimize_media
from lightweight_mmm import utils


def train_test_split(df, test_size=None):
    if test_size is None:
        test_size = 0
    # Split and scale data.
    data_size = len(df)
    split_point = data_size - test_size
    print(f'Splitting at data_size ({data_size}) less test_size ({test_size} = {split_point})')
    # Media data
    dates_mask = ['dates' in s for s in df.columns]
    media_mask = ['channel' in s for s in df.columns]
    cost_mask = ['cost' in s for s in df.columns]
    extra_features_mask = ['extra_feature' in s for s in df.columns]
    target_mask = ['target' in s for s in df.columns]

    train = df.iloc[:split_point:, :].to_numpy()
    test = df.iloc[split_point:, :].to_numpy()

    X_media_train = train[:, media_mask]
    X_cost_train = train[:, cost_mask]
    X_extra_features_train = train[:, extra_features_mask]
    y_train = train[:, target_mask]
    dates_train = train[:, dates_mask]

    X_media_test = test[:, media_mask]
    X_extra_features_test = test[:, extra_features_mask]
    y_test = test[:, target_mask]
    dates_test = test[:, dates_mask]

    return [X_media_train, X_cost_train, X_extra_features_train, y_train], [X_media_test, X_extra_features_test, y_test], dates_train, dates_test


def preprocess_data(train, test):
    [X_media_train, X_cost_train, X_extra_features_train, y_train], [X_media_test, X_extra_features_test, y_test] = train, test

    media_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
    cost_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
    extra_features_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
    target_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)

    X_media_train = media_scaler.fit_transform(X_media_train)
    X_cost_train = cost_scaler.fit_transform(X_cost_train)
    X_extra_features_train = extra_features_scaler.fit_transform(X_extra_features_train)

    y_train = target_scaler.fit_transform(y_train)

    X_media_test = media_scaler.transform(X_media_test)
    X_extra_features_test = extra_features_scaler.transform(X_extra_features_test)
    # y_test = target_scaler.transform(y_test)

    train = [X_media_train, X_cost_train, X_extra_features_train, y_train]
    test = [X_media_test, X_extra_features_test, y_test]

    return train, test, target_scaler


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
        group_label = f'Media Channel {idx}'
        y = media_channel_posteriors[:, channel_i]

        distplfig = go.Violin(x=y, name=group_label)

        fig.add_trace(
            distplfig,
            row=idx+1, col=1,
        )
    fig.update_layout(barmode='overlay', title_text='Posterior Distributions')
    fig.update_xaxes(title='Coefficient (channel strength)')
    fig.update_yaxes(title='Probability Density')
    fig.write_html('kde_plots.html')
    return fig


def train_model(train, test, dates_train, dates_test, target_scaler):
    [X_media_train, X_cost_train, X_extra_features_train, y_train], [X_media_test, X_extra_features_test, y_test] = train, test

    adstock_models = ["adstock", "hill_adstock", "carryover"]
    degrees_season = [1, 2, 3]
    #
    # adstock_models = ["hill_adstock"]
    # degrees_season = [1]

    numpyro.set_host_device_count(2)

    start = time.time()

    predictions = {}
    models = {}
    for model_name in adstock_models:
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

            models[model_name][degrees] = mape
            predictions[model_name][degrees] = p
            print(f"model_name={model_name} degrees={degrees} MAPE={mape} samples={p[:3]}")

    end = time.time()
    time_elapsed = end - start
    print(f'Train and CV time elapsed: {time_elapsed}')

    cross_validation_summary = pd.DataFrame.from_dict(models).rename_axis('degrees')
    optimal_degrees, optimal_model = cross_validation_summary.stack().idxmin()

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
        .groupby(['model_name'])
        .agg({'quantity': 'unique'})
    )
    all_data = pd.concat([actuals, prediction_time_series]).explode('quantity')
    n_repeats = len(all_data) // len(dates_test)
    all_data = (
        all_data.assign(dates=np.tile(dates_test, (n_repeats, 1)))
        .set_index('dates', append=True)
        .squeeze()
        .unstack('model_name')
    )

    return mmm


def call_back(data):
    df = pd.DataFrame.from_dict(data)

    train, test, dates_train, dates_test = train_test_split(df, test_size=10)
    train, test, target_scaler = preprocess_data(train, test)

    train_model = train_model(train, test, dates_train, dates_test, target_scaler)


# mmm.fit(
#     media=media_data_train,
#     media_prior=costs,
#     target=target_train,
#     extra_features=extra_features_train,
#     number_warmup=number_warmup,
#     number_samples=number_samples,
#     custom_priors={"intercept": numpyro.distributions.HalfNormal(5)})
#
# mmm.get_posterior_metrics()
