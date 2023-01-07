

def train_test_split(media_data, extra_features, target, test_size, data_size):
    # Split and scale data.
    split_point = data_size - test_size
    print(f'Splitting at data_size less test_size: {split_point}')
    # Media data
    media_data_train = media_data[:split_point, ...]
    media_data_test = media_data[split_point:, ...]
    # Extra features
    extra_features_train = extra_features[:split_point, ...]
    extra_features_test = extra_features[split_point:, ...]
    # Target
    target_train = target[:split_point]

#
#
# media_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
# extra_features_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
# target_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
# cost_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
#
# media_data_train = media_scaler.fit_transform(media_data_train)
# extra_features_train = extra_features_scaler.fit_transform(extra_features_train)
# target_train = target_scaler.fit_transform(target_train)
# costs = cost_scaler.fit_transform(costs)
#
# model_choices = ['carryover', 'hill_adstock', 'adstock']
#
# mmm = lightweight_mmm.LightweightMMM(model_name=model)
# number_warmup = 10
# number_samples = 10
#
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
