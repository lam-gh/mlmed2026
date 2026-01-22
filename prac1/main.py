import pandas as pd
import numpy as np
import keras
from keras import layers
from keras import ops
from keras.utils import to_categorical


def my_model():
    inputs = keras.Input(shape=(186, 1))
    # conv x3
    x = layers.Conv1D(filters=32, kernel_size=5, activation="relu")(inputs)
    x = layers.Conv1D(filters=32, kernel_size=5, activation="relu")(x)
    x = layers.Conv1D(filters=32, kernel_size=5, activation="relu")(x)
    # pool
    x = layers.MaxPool1D(pool_size=5, strides=2)(x)
    # flatten
    x = layers.Flatten()(x)
    # fully connected (relu)
    x = layers.Dense(32, activation="relu")(x)
    # fully connected (softmax)
    outputs = layers.Dense(5, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="mitbih_model")
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
    df_train = pd.read_csv("../data/prac1/mitbih_train.csv", header=None).values
    df_test = pd.read_csv("../data/prac1/mitbih_test.csv", header=None).values

    # print(df_train.head())

    # print(df.shape)
    # (87553, 188)

    # print(df.iloc[:, 187].astype(int).unique())
    # [0 1 2 3 4]

    x_train = df_train[:, 1:-1]
    y_train = df_train[:, -1]
    x_test = df_test[:, 1:-1]
    y_test = df_test[:, -1]

    x_train = x_train.reshape(x_train.shape[0], 186, 1)
    x_test = x_test.reshape(x_test.shape[0], 186, 1)
    y_train = to_categorical(y_train, num_classes=5)
    y_test = to_categorical(y_test, num_classes=5)

    model = my_model()
    train_model(x_train, y_train, x_test, y_test, model)


if __name__ == "__main__":
    main()
