import numpy as np
import matplotlib.pyplot as plt

def temporalize(X, y, autoencoder = False, sequence_length = 30):
    '''
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
    if autoencoder: length += 1
    for i in range(length):
        X_list.append(X.iloc[i:(i + sequence_length)].values)
        try:
            y_list.append(y.iloc[i + sequence_length])
        except:
            y_list.append(0.0)
    return np.asarray(X_list), np.asarray(y_list)


def flatten(X):
    flattened_X = np.zeros((X.shape[0] + X.shape[1] - 1, X.shape[2]))
    div = np.zeros((X.shape[0] + X.shape[1] - 1, X.shape[2]))
    for idx in range(0, X.shape[0]):
        flattened_X[idx:idx + X.shape[1]] += X[idx, :, :]
        div[idx:idx + X.shape[1]] += 1
    return np.divide(flattened_X, div)

def anomaly_detector(X, reconstruction):
    """
    Args:
    X : input data
    reconstruction : reconstructed data
    """
    #scores = np.linalg.norm(X - reconstruction, axis=-1)
    scores = np.abs(X.reshape(-1) - reconstruction.reshape(-1))
    #scores, _ = stats.boxcox(scores)
    threshold_score_max = scores.mean() + 3 * scores.std()

    plt.title("Reconstruction Error")
    plt.plot(scores, label='scores')
    plt.plot([threshold_score_max]*len(scores), c='r', label='threshold')
    plt.legend(loc='best')
    plt.show()
    anomalous = np.where((scores > threshold_score_max))
    normal = np.where((scores <= threshold_score_max))

    plt.title("Anomalies")
    plt.scatter(normal, X[normal][:,-1], s=3, label='normal')
    plt.scatter(anomalous, X[anomalous][:,-1], s=5, c='r', label='anomaly')
    plt.legend(loc='best')
    plt.show()
    return scores > threshold_score_max