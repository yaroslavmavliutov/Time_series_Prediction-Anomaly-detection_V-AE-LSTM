# Time series prediction & Anomaly-detection in Keras

### Data flow

This strategy combines autoencoding and predictor.

For comparison, we have implemented two options:
 - standard autoencoder + LSTM predictor + anomaly detector;
 - variational autoencoder + LSTM predictor + anomaly detector;

It's possible to use just an autoencoder.

We use prediction/reconstruction error to define anomaly.
Anomaly detector is based on the 99.7 rule, also known as the empirical rule. Means that 99.73% of the 
values lie within three standard deviations of the mean.


### Usage

Using a combination of autoencoders&predictor
```python
from models import V_AE_LSTM

input_shape = (X_train.shape[1], X_train.shape[2],)
intermediate_cfg = [64, 'latent', 64]
latent_dim = 10
model = V_AE_LSTM(input_shape, intermediate_cfg, latent_dim, 'VAE-LSTM')
model.fit(X_train, y_train, epochs=2, batch_size=124, validation_split=None, verbose=1)
```

Using a normal autoencoder
```python
from models import LSTM_Autoencoder

input_shape=(X_train.shape[1], X_train.shape[2],)
intermediate_cfg = None #[64, 'latent', 64]
latent_dim = 10
ae = LSTM_Autoencoder(input_shape, intermediate_cfg, latent_dim)
ae.fit(X_train, epochs=10, batch_size=32, validation_split=None, verbose=1)

reconstructed = ae.reconstruct(X_test)
```

Using a variational autoencoder
```python
from models import LSTM_VAutoencoder

input_shape=(X_train.shape[1], X_train.shape[2],)
intermediate_cfg = [64, 32, 'latent', 32, 64]
latent_dim = 10
vae = LSTM_VAutoencoder(input_shape, intermediate_cfg, latent_dim)
vae.fit(X_train, epochs=10, batch_size=32, validation_split=None, verbose=1)

reconstructed = vae.reconstruct(X_test)
```
