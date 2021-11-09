from csv import DictReader, DictWriter
import keras
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def datetime_str_to_hours(s):
    time = datetime.datetime.strptime(s, DATETIME_FORMAT)
    return (time - datetime.datetime(1900, 1, 1)).total_seconds() / 3600.0


def row_to_input(row, next_row, include_avg_sentiment=True, include_weighted_avg_top_sentiment=True):
    """
    x:  num_ratings, num_comments, hours_since_posted, rating_balance, [avg_pol, avg_sub], [weighted_top_pol,
        weighted_top_sub]
    y:  d(rating_balance)/dt
    """
    # TODO
    return np.zeros(10), 0


def load_data(yt_categories, subreddits, normalize=True):
    with open("data.csv", "r") as f:
        reader = DictReader(f)
        id2samples = {}
        for row in reader:
            if row["category"] in yt_categories or row["category"] in subreddits:
                id2samples[row["id"]] = id2samples.setdefault(row["id"], row)

    X, Y = [], []
    for post, samples in id2samples.items():
        samples = sorted(samples, key=lambda sample: datetime_str_to_hours(sample["time_since_posted"]))
        for i in range(len(samples) - 1):
            x, y = row_to_input(samples[i], samples[i+1])
            X.append(x)
            Y.append(y)
    if normalize:
        scaler = StandardScaler().fit(X)
        print(f"Standardizing inputs using mean {scaler.mean_} and scale {scaler.scale_}")
        X = scaler.transform(X)
    return X, Y


def best_model(X_train, Y_train, X_dev, Y_dev, output_file):
    input_shape = (len(X_train[0]),)

    models = []

    # 1 hidden layer
    for hidden_size in [4, 6]:
        model = keras.Sequential()
        model.add(Dense(hidden_size, input_shape=input_shape, activation='relu'))
        model.add(Dense(1, activation=None))
        models.append(model)

    # 2 hidden layers
    for hidden_1_size in [4, 6]:
        for hidden_2_size in [2, 4]:
            model = keras.Sequential()
            model.add(Dense(hidden_1_size, input_shape=input_shape, activation='relu'))
            model.add(Dense(hidden_2_size, activation='relu'))
            model.add(Dense(1, activation=None))
            models.append(model)

    best_val_loss = np.inf
    for model in models:
        model.compile(
            loss="huber",
            optimizer="adam",
            metrics=['mse']
        )
        early_stopping_monitor = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=0,
            verbose=0,
            mode='auto',
            baseline=None,
            restore_best_weights=True
        )
        history = model.fit(
            X_train,
            Y_train,
            batch_size=len(X_train),
            validation_data=(X_dev, Y_dev),
            callbacks=[early_stopping_monitor],
            epochs=20
        )
        print(history.history['val_loss'])
        if max(history.history['val_loss']) < best_val_loss:
            model.save(output_file)
