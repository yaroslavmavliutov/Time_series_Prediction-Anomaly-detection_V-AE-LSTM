import preprocessing as ps
import pandas as pd
from models import V_AE_LSTM
from utils import temporalize, flatten, anomaly_detector
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from pylab import rcParams

rcParams['figure.figsize'] = 15, 7
RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)

def main():

    url = "data/data.csv"
    df = pd.read_csv(url, index_col='Date', parse_dates=True)
    df = df[['Close']]

    ps.test_stationary(df['Close'])

    train, test = train_test_split(df, test_size=0.1, shuffle=False)

    scaler = StandardScaler()
    scaler = scaler.fit(train[['Close']])
    train['NClose'] = scaler.transform(train[['Close']])
    test['NClose'] = scaler.transform(test[['Close']])

    sequence_length = 100
    X_train, y_train = temporalize(train[['NClose']], train.NClose, False, sequence_length)
    X_test, y_test = temporalize(test[['NClose']], test.NClose, False, sequence_length)

    input_shape = (X_train.shape[1], X_train.shape[2],)
    intermediate_cfg = [64, 'latent', 64]
    latent_dim = 10
    model = V_AE_LSTM(input_shape, intermediate_cfg, latent_dim, 'VAE-LSTM')
    model.fit(X_train, y_train, epochs=2, batch_size=124, validation_split=None, verbose=1)

    recunstruction, prediction = model.predict(X_test)

    recunstruction = flatten(recunstruction).reshape(-1)

    res = anomaly_detector(y_test.reshape(-1,1), prediction.reshape(-1,1))


if __name__ == "__main__":
    main()