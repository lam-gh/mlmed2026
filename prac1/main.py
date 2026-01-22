import pandas as pd
import numpy as np
import keras
from keras import layers
from keras import ops


def my_model():
    # input (186-length vector, 10 timesteps)
    inputs = keras.Input(shape=(10, 186))
    # conv x3
    x = layers.Conv1D(filters=32, kernel_size=5, activation="relu")(inputs)
    x = layers.Conv1D(filters=32, kernel_size=5, activation="relu")(x)
    x = layers.Conv1D(filters=32, kernel_size=5, activation="relu")(x)
    # pool
    x = layers.MaxPool1D(pool_size=5, strides=2)(x)
    # fully connected (relu)
    x = layers.Dense(32, activation="relu")(x)
    # fully connected (softmax)
    outputs = layers.Dense(32, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
    return model


def train_model(x_train, y_train, x_test, y_test, model):
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy"],
    )
    history = model.fit(
        x_train, y_train, batch_size=64, epochs=12, validation_split=0.2
    )
    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])
    return history


def main():
    df_train = pd.read_csv("../data/mitbih_train.csv", header=None)
    df_test = pd.read_csv("../data/mitbih_test.csv", header=None)

    # print(df.shape)
    # (87553, 188)

    # print(df.iloc[:, 187].astype(int).unique())
    # [0 1 2 3 4]

    x_train = df_train.iloc[:, :186]
    y_train = df_train.iloc[:, 187].astype(int)
    x_test = df_test.iloc[:, :186]
    y_test = df_test.iloc[:, :187].astype(int)

    # (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # print(x_train.info)
    # x_train = x_train.reshape(60000, 784).astype("float32") / 255
    # print(x_test.shape)
    # x_test = x_test.reshape(10000, 784).astype("float32") / 255

    # y_train: np.ndarray
    # print(y_train)

    model = my_model()
    train_model(x_train, y_train, x_test, y_test, model)


if __name__ == "__main__":
    main()
