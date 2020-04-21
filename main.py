from preprocessing import temporalize,flatten
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from pylab import rcParams
import os

rcParams['figure.figsize'] = 15, 7
RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)

def main():
    url = "data/spx2.csv"
    df = pd.read_csv(url, index_col='Date', parse_dates=True) # os.path.join('/content/drive/My Drive/university/anomaly detection', url)
    df.drop(df.columns.difference(['Close']), 1, inplace=True)

    train, test = train_test_split(df, test_size=0.05, shuffle=False)

    scaler = StandardScaler()
    scaler = scaler.fit(train[['Close']])
    train['NClose'] = scaler.transform(train[['Close']])
    test['NClose'] = scaler.transform(test[['Close']])
    # scaler = StandardScaler()
    # scaler = scaler.fit(train[['Difference']])
    # train['NClose'] = scaler.transform(train[['Difference']])
    # test['NClose'] = scaler.transform(test[['Difference']])

    sequence_length = 30
    X_train, y_train = temporalize(train[['NClose']], train.NClose, sequence_length)
    X_test, y_test = temporalize(test[['NClose']], test.NClose, sequence_length)

    keras.backend.clear_session()

    autoencoder = keras.Sequential()
    autoencoder.add(keras.layers.LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu',
                                      return_sequences=False))
    # autoencoder.add(keras.layers.LSTM(units=32, activation='relu', return_sequences=False))
    autoencoder.add(keras.layers.Dropout(rate=0.1))
    autoencoder.add(keras.layers.RepeatVector(n=X_train.shape[1]))
    # autoencoder.add(keras.layers.LSTM(units=32, activation='relu', return_sequences=True))
    autoencoder.add(keras.layers.LSTM(units=64, return_sequences=True, activation='relu'))
    autoencoder.add(keras.layers.Dropout(rate=0.1))
    autoencoder.add(keras.layers.TimeDistributed(keras.layers.Dense(units=X_train.shape[2])))

    autoencoder.compile(loss='mae', optimizer='adam')
    history = autoencoder.fit(X_train, X_train, epochs=10, batch_size=32, validation_split=0.1, shuffle=False,
                              verbose=2).history

    plt.plot(history['loss'], linewidth=2, label='Train')
    plt.plot(history['val_loss'], linewidth=2, label='Valid')
    plt.legend(loc='upper right')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()


if __name__ == "__main__":
    main()