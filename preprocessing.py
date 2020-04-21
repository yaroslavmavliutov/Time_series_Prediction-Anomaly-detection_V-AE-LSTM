import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np

def visualization(df, show=False):
    if show == True:
        df.plot(linewidth=0.5)
        plt.show()

def isnan(df):
    if df.isna().sum().sum() > 0:
        print("Deleting null values")
        df.dropna(inplace=True)

def differencing(df, order = 1, col_name='Close', new_col_name='Difference'):
    isnan(df)
    df[new_col_name] = df[col_name] - df[col_name].shift(order)

def transformation(df, fun=np.log, col_name='Close', new_col_name='LClose'):
    isnan(df)
    df[new_col_name] = df[col_name].apply(lambda x: fun(x))

def test_stationary(series, window=100):

    plt.figure(figsize=(15, 14))

    # Rolling statistics
    plt.subplot(211)
    movingAverage = series.rolling(window=window).mean()
    movingSTD = series.rolling(window=window).std()
    plt.plot(series, color='cornflowerblue', label='Original')
    plt.plot(movingAverage, color='red', label='Rolling Mean')
    plt.plot(movingSTD, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation\n')

    # Auto-correlation
    plt.subplot(212)
    correlation = acf(series.iloc[1:])
    partial_correlation = pacf(series.iloc[1:])
    plt.plot(correlation, color='green', marker='o', linestyle='--', label='AutoCorr')
    plt.plot(partial_correlation, color='cornflowerblue', marker='.', linestyle='--', label='PAutoCorr')
    plt.legend(loc='best')
    plt.title('Autocorrelationn & Partial Autocorrelation\n')

    # plt.subplot(211)
    # plot_acf(timeseries.dropna(), title="1st Order Differencing")
    # plt.subplot(212)
    # plot_acf(timeseries.diff().diff().dropna(), title="2nd Order Differencing")

    plt.show()

    # Dickey Fuller test
    print('Results of Dickey Fuller Test:\n')
    dftest = adfuller(series.values)
    print('ADF Statistic: %f' % dftest[0])
    print('p-value: %f' % dftest[1])
    print('Critical Values:')
    for key ,value in dftest[4].items():
        print('\t%s: %.3f' % (key, value))

def temporalize(X, y, autoencoder = False, sequence_length = 30):
    '''
    :param X:
    :param y:
    :param autoencoder:
    :param sequence_length:
    :return:

    If we have an autocoder, we will try to reconstruct the signal X
    Here, we don't care about the output signal y. We try to keep all the values of the initial X,
    so we increase the number of cycles by 1 (length +=1) to save all the values

    Example:
    init_X = [1,2,3,4,5,6]
    sequence_length = 3
    X = [[1,2,3], [2,3,4],[3,4,5], [4,5,6]], y = [4,5,6,0.0] - we don't care about y, we save all init values
    X = [[1,2,3], [2,3,4],[3,4,5]], y = [4,5,6] - new signal X does not contain the last value
    '''
    X_list = []
    y_list = []
    length = len(X) - sequence_length
    if autoencoder: length +=1
    for i in range(length):
        X_list.append(X.iloc[i:(i + sequence_length)].values)
        try: y_list.append(y.iloc[i + sequence_length])
        except: y_list.append(0.0)
    return np.asarray(X_list), np.asarray(y_list)


def flatten(X):
    flattened_X = np.empty((X.shape[0]+X.shape[1]-1, X.shape[2]))
    flattened_X[0:X.shape[0]] = X[:, 0, :]
    flattened_X[X.shape[0]-1:X.shape[0]+X.shape[1]] = X[X.shape[0]-1, :, :]
    return flattened_X