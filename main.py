import utils
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np




def main():
    url = "data/spx.csv"
    df = pd.read_csv(url, index_col='date', parse_dates=True)
    visualization(df, show=False)
    train, test = train_test_split(df, test_size=0.05, shuffle=False)

    scaler = StandardScaler()
    scaler = scaler.fit(train[['close']])
    #train.loc['close'] = scaler.transform(train[['close']])
    #test.loc['close'] = scaler.transform(test[['close']])
    train['close'] = scaler.transform(train[['close']])
    test['close'] = scaler.transform(test[['close']])
    #visualization(train, show=True)
    #visualization(test, show=True)
    x,y = data_conversion(train[['close']], test.close, sequence_length=30)



def visualization(data, show=False):
    if show == True:
        data.plot(linewidth=0.5)
        plt.show()

def data_conversion(X, y, sequence_length = 30):
    X_train_list = []
    y_train_list = []

    for i in range(len(X) - sequence_length):
        v = X.iloc[i:(i + sequence_length)].values
        X_train_list.append(v)
        y_train_list.append(y.iloc[i + sequence_length])

    X_train = np.asarray(X_train_list)
    y_train = np.asarray(y_train_list)
    print("X_train.shape:", X_train.shape)
    print("y_train.shape:", y_train.shape)

    return X_train, y_train

if __name__ == "__main__":
    main()