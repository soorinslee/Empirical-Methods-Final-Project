from csv import DictReader, DictWriter
import keras
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

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


def best_model(X_train, Y_train, X_dev, Y_dev, output_file, callback_monitor="val_loss"):
    input_shape = (len(X_train[0]),)

    models = []

    # 1 hidden layer
    for hidden_size in [8, 16, 24, 32]:
        model = keras.Sequential()
        model.add(Dense(hidden_size, input_shape=input_shape, activation='relu'))
        model.add(Dense(1, activation=None))
        models.append(model)

    # 2 hidden layers
    for hidden_1_size in [8, 16, 24, 32]:
        model = keras.Sequential()
        model.add(Dense(hidden_1_size, input_shape=input_shape, activation='relu'))
        model.add(Dense(1, activation='relu'))
        model.add(Dense(1, activation=None))
        models.append(model)

    best_val_loss = np.inf
    for model in models:
        for lr_exp in range(7, 14):
            lr = 10**-lr_exp
            model.compile(
                loss="huber",
                optimizer=Adam(learning_rate=lr),
                metrics=['mse'],
            )
            early_stopping_monitor = EarlyStopping(
                monitor=callback_monitor,
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
                epochs=20,
            )
            print("Validation loss by epoch: ", history.history['val_loss'])
            if max(history.history['val_loss']) < best_val_loss:
                model.save(output_file)
                best_val_loss = max(history.history['val_loss'])
                print(f"Achieved best loss so far with lr={lr} and model:")
                print(model.summary())
    return best_val_loss


if __name__ == "__main__":
    X, Y = load_data(["YouTube_gaming", "YouTube_news", "YouTube_music", "YouTube_howto"])
    Y = np.array(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    print(len(X_train), "training examples; ", len(X_test), "test examples")

    """# Hyperparameter search; save the best model
    best_model(X_train, Y_train, X_test, Y_test, "all_youtube.h5")"""

    model = keras.models.load_model("all_youtube.h5")

    """# Check predictions of the best model (sanity check)
    predictions = model.predict(X_test[:20])
    for y_, y in zip(predictions, Y_test[:20]):
        print(y_[0], y)"""

    # TODO: Statistical goodness-of-fit tests!!!
    # https://www.statsmodels.org/stable/stats.html#goodness-of-fit-tests-and-measures

    # Graph df/d(avg_rating) against avg_rating for both the train set and the test set
    for inputs in X_train, X_test:
        x_tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)
        with tf.GradientTape() as t:
            t.watch(x_tensor)
            output = model(x_tensor)

        result = output
        gradients = t.gradient(output, x_tensor)
        plt.scatter(inputs[:, 3], gradients[:, 3])
        plt.show()
        print(f"(correlation btw r and predicted r', 2-tailed p value): {pearsonr(inputs[:, 3], gradients[:, 3])}")
        for upper_cutoff in [5, 10, 100, 1000, 10000]:
            volatile_indices = [i for i in range(len(inputs)) if inputs[i][0] < upper_cutoff]
            print(f"For samples with num_ratings < {upper_cutoff}:")
            select_inputs = [inputs[i][3] for i in volatile_indices]
            select_gradients = [gradients[i][3] for i in volatile_indices]
            print("\t", pearsonr(select_inputs, select_gradients))
        for r_range in [(-3, -1), (-1, 0), (0, 1), (-1, 1), (-3, 0)]:
            typical_indices = [i for i in range(len(inputs)) if r_range[0] < inputs[i][3] < r_range[1]]
            print(f"For samples with normalized avg_rating in range {r_range}:")
            select_inputs = [inputs[i][3] for i in typical_indices]
            select_gradients = [gradients[i][3] for i in typical_indices]
            print("\t", pearsonr(select_inputs, select_gradients))
        # TODO: combine the two grouping mechanisms



    """# Graph residuals against avg_rating
    for inputs, labels in [(X_train, Y_train), (X_test, Y_test)]:
        predictions = np.squeeze(model.predict(inputs), axis=-1)
        print(labels)
        plt.scatter(inputs[:, 3], labels - predictions, marker='x')
        plt.show()"""


