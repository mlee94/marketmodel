import base64
import datetime
import io

from dash import Dash, html, dcc, Input, Output, State, no_update
import dash_daq as daq
from dash.exceptions import PreventUpdate
import pandas as pd

from lightweight_mmm.utils import dataframe_to_jax

from marketmodel.dash_config import app


@app.callback(
    Output('media-features', 'options'),
    Output('extra-features', 'options'),
    Output('date-feature', 'options'),
    Output('channel-features', 'options'),
    Output('target', 'options'),
    Output('cost-features', 'options'),
    Output('raw-data', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
)
def refresh_features_from_csv(contents, filename):
    if not (contents and filename):
        raise PreventUpdate

    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    all_columns = df.columns
    df.to_csv('view_mkting.csv', index=False)

    return all_columns, all_columns, all_columns, all_columns, all_columns, all_columns, df.to_dict('records')


@app.callback(
    Output('sample-data', 'data'),
    Output('feature-mapping', 'data'),
    Input('upload-file-button', 'n_clicks'),
    State('channel-features', 'value'),
    State('media-features', 'value'),
    State('extra-features', 'value'),
    State('date-feature', 'value'),
    State('target', 'value'),
    State('cost-features', 'value'),
    State('raw-data', 'data'),
)
def get_features_from_csv(n_clicks, channel, media, extra, date, target, cost, data):
    if not n_clicks:
        raise PreventUpdate

    if not extra:
        extra = ''
        select_cols = media + cost + [target]
    else:
        select_cols = media + extra + cost + [target]
    if not cost:
        cost = media



    # pivot on channel
    # df = pd.read_csv('view_mkting.csv').astype({date[0]: 'datetime64[ns]'}).set_index(date + channel)
    df = (
        # pd.DataFrame.from_dict(data)
        pd.read_csv('view_mkting.csv')
        .astype({date: 'datetime64[ns]'})
        .set_index([date, channel])
        .get(select_cols)
    )
    date_min = df.index.get_level_values(date).min()
    date_max = df.index.get_level_values(date).max()

    total_revenue = (
        df.droplevel(channel).get(target).groupby([date]).sum()
        .reindex(pd.date_range(date_min, date_max, freq='D', name=date))
        .fillna(0)
        .to_frame()
    )
    features = df.drop(target, axis=1).unstack(channel)
    features.columns = [' - '.join(i) for i in features.columns.to_flat_index()]

    combined = (
        total_revenue
        .join(features.fillna(0))
        .reset_index()
    )
    combined.to_csv('combined_mkting.csv', index=False)

    mapping = {
        'target': target,
        'date': date,
        'media': list(combined.columns[[any([x in y for x in media]) for y in combined.columns]]),
        'cost': list(combined.columns[[any([x in y for x in cost]) for y in combined.columns]]),
        'extra_features': list(combined.columns[[any([x in y for x in extra]) for y in combined.columns]]),
    }

    return combined.to_dict('records'), mapping
