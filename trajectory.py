from csv import DictReader, DictWriter
import keras
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def row_to_input(row, next_row, include_avg_sentiment=True, include_weighted_avg_top_sentiment=True):
    """
    x:  num_ratings, num_comments, hours_since_posted, rating_balance, [avg_pol, avg_sub], [weighted_top_pol,
        weighted_top_sub]
    y:  d(rating_balance)/dt
    """
    row["ups"] = int(row["ups"])
    row["downs"] = int(row["downs"])
    next_row["ups"] = int(next_row["ups"])
    next_row["downs"] = int(next_row["downs"])
    if '[' in row["top_avg_polarity"]:  # Reddit
        num_ratings = 1
        num_ratings_next = 1
    else:  # YouTube
        num_ratings = row["ups"] + row["downs"]
        num_ratings_next = next_row["ups"] + next_row["downs"]
    num_comments = int(row["num_comments"])
    hours_since_posted = float(row["time_since_posted"])
    try:
        rating_balance = (row["ups"] - row["downs"]) / num_ratings  # not meaningful for Reddit; downs is always 0 :(
    except ZeroDivisionError:
        return None
    if row["all_avg_polarity"] != '':
        avg_pol = float(row["all_avg_polarity"])
    else:
        avg_pol = None
    if row["all_avg_subjectivity"] != '':
        avg_sub = float(row["all_avg_subjectivity"])
    else:
        avg_sub = None
    next_rating_balance = (next_row["ups"] - next_row["downs"]) / num_ratings_next
    y = (next_rating_balance - rating_balance) / (float(next_row["time_since_posted"]) - hours_since_posted)

    # TODO: accommodate Reddit
    weighted_top_pol = None
    weighted_top_sub = None
    if include_weighted_avg_top_sentiment:
        if row["top_avg_polarity"] != '':
            weighted_top_pol = float(row["top_avg_polarity"])
        if row["top_avg_subjectivity"] != '':
            weighted_top_sub = float(row["top_avg_subjectivity"])

    return_list = [num_ratings, num_comments, hours_since_posted, rating_balance]
    if include_avg_sentiment:
        return_list.extend([avg_pol, avg_sub])
    if include_weighted_avg_top_sentiment:
        return_list.extend([weighted_top_pol, weighted_top_sub])
    return return_list, y


def load_data(categories=None, urls=None, normalize=True, include_avg_sentiment=True,
              include_weighted_avg_top_sentiment=True):
    with open("samples.csv", "r") as f:
        reader = DictReader(f)
        id2samples = {}
        for row in reader:
            if (categories is None or row["category"] in categories) and (urls is None or row["url"] in urls):
                id2samples.setdefault(row["url"], []).append(row)

    X, Y = [], []
    for post, samples in id2samples.items():
        samples = sorted(samples, key=lambda sample: sample["time_since_posted"])
        for i in range(len(samples) - 1):
            converted = row_to_input(samples[i], samples[i+1], include_avg_sentiment, include_weighted_avg_top_sentiment)
            if converted is not None:
                X.append(converted[0])
                Y.append(converted[1])
    if normalize:
        scaler = StandardScaler().fit(X)
        print(f"Standardizing inputs using mean {scaler.mean_} and scale {scaler.scale_}")
        X = scaler.transform(X)
    return np.nan_to_num(X), Y


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


if __name__ == "__main__":
    X, Y = load_data("YouTube_gaming")
