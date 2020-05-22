import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
import numpy as np

def isnan(df):
    # check if dat consists nan values
    if df.isna().sum().sum() > 0:
        print("Deleting null values")
        df.dropna(inplace=True)

def differencing(df, order = 1, col_name='Close', new_col_name='Difference'):
    isnan(df)
    df[new_col_name] = df[col_name] - df[col_name].shift(order)
    isnan(df)

def transformation(df, fun=np.log, col_name='Close', new_col_name='LClose'):
    # power transformation
    isnan(df)
    df[new_col_name] = df[col_name].apply(lambda x: fun(x))
    isnan(df)


def test_stationary(timeseries, window=500):
    """
    simple statistics to check data

    rolling stat
    autocorrelation
    test Dickey-Fuller to check stationarity
    """
    plt.figure(figsize=(15, 14))

    # Rolling statistics
    plt.subplot(211)
    movingAverage = timeseries.rolling(window=window).mean()
    movingSTD = timeseries.rolling(window=window).std()
    plt.plot(timeseries, color='cornflowerblue', label='Original')
    plt.plot(movingAverage, color='red', label='Rolling Mean')
    plt.plot(movingSTD, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation\n')

    # Auto-correlation
    plt.subplot(212)
    correlation = acf(timeseries.iloc[1:])
    partial_correlation = pacf(timeseries.iloc[1:])
    plt.plot(correlation, color='green', marker='o', linestyle='--', label='AutoCorr')
    plt.plot(partial_correlation, color='cornflowerblue', marker='.', linestyle='--', label='PAutoCorr')
    plt.legend(loc='best')
    plt.title('Autocorrelationn & Partial Autocorrelation\n')
    plt.show()

    # Dickey Fuller test
    print('Results of Dickey Fuller Test:\n')
    dftest = adfuller(timeseries.values)
    print('ADF Statistic: %f' % dftest[0])
    print('p-value: %f' % dftest[1])
    print('Critical Values:')
    for key, value in dftest[4].items():
        print('\t%s: %.3f' % (key, value))