import jax.numpy as jnp
import numpyro

from dash import Dash, html, dcc, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_percentage_error

from lightweight_mmm import lightweight_mmm
from lightweight_mmm import preprocessing
from lightweight_mmm import utils


def train_test_split(df, test_size=None):
    if test_size is None:
        test_size = 0
    # Split and scale data.
    data_size = len(df)
    split_point = data_size - test_size
    print(f'Splitting at data_size ({data_size}) less test_size ({test_size} = {split_point}')
    # Media data
    train = df.iloc[:split_point, :]
    test = df.iloc[split_point:, :]

    return train, test


def preprocess_data(train):
    data_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
    data_cols = train.columns
    train_scaled = (
        pd.DataFrame(
            data=data_scaler.fit_transform(train.to_numpy()),
            columns=data_cols
        )
    )
    return train_scaled, data_scaler


def scale_test_set(test, data_scaler):
    data_cols = test.columns
    test_scaled = (
        pd.DataFrame(
            data=data_scaler.transform(test.to_numpy()),
            columns=data_cols
        )
    )
    return test_scaled


def train_model(train_scaled, test_scaled, data_scaler):
    jax_array = train_scaled.to_numpy()

    media_data_train = train_scaled.get([s for s in train_scaled.columns if 'channel' in s])
    cost_data_train = train_scaled.get([s for s in train_scaled.columns if 'cost' in s]).to_numpy()
    extra_features_train = train_scaled.get([s for s in train_scaled.columns if 'extra_feature' in s]).to_numpy()
    target_train = train_scaled.get([s for s in train_scaled.columns if 'target' in s]).to_numpy()

    media_data_test = test_scaled.get([s for s in train_scaled.columns if 'channel' in s]).to_numpy()
    extra_features_test = test_scaled.get([s for s in train_scaled.columns if 'extra_feature' in s]).to_numpy()
    target_test = test_scaled.get([s for s in train_scaled.columns if 'target' in s]).to_numpy()

    adstock_models = ["adstock", "hill_adstock", "carryover"]
    degrees_season = [1, 2, 3]
    #
    # adstock_models = ["hill_adstock"]
    # degrees_season = [1]

    for model_name in adstock_models:
        for degrees in degrees_season:

            mmm = lightweight_mmm.LightweightMMM()
            mmm.fit(
                media=media_data_train,
                extra_features=extra_features_train,
                media_prior=cost_data_train.sum(axis=0),
                target=target_train,
                number_warmup=1000,
                number_samples=1000,
                number_chains=2,
            )
            prediction = mmm.predict(
                media=media_data_test,
                extra_features=extra_features_test,
                target_scaler=data_scaler
            )
            p = prediction.mean(axis=0)

            mape = mean_absolute_percentage_error(target_test.values, p)
            print(f"model_name={model_name} degrees={degrees} MAPE={mape} samples={p[:3]}")



def call_back(data):
    df = pd.DataFrame.from_dict(data)

    train, test = train_test_split(df, test_size=10)
    train_scaled, data_scaler = preprocess_data(df)
    test_scaled = scale_test_set(test, data_scaler)
    train_model = train_model(train_scaled, test_scaled)


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
