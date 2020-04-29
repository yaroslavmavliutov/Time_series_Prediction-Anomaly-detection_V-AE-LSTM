from keras import backend as K
from keras.models import Model
from keras.layers import Input, LSTM, RepeatVector
from keras.layers.core import Dense, Lambda
from keras.layers import RepeatVector, TimeDistributed
from keras.losses import mse, binary_crossentropy


class LSTM_VAutoencoder(object):
    """
    input_shape=(X_train.shape[1], X_train.shape[2],)
    intermediate_cfg = [128, 64, 'latent', 64, 128]
    latent_dim = 32

    vae = LSTM_VAutoencoder(input_shape, intermediate_cfg, latent_dim)
    vae.fit(X_train, epochs=10, batch_size=32)
    """

    def __init__(self, input_shape, intermediate_cfg, latent_dim):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.intermediate_cfg = intermediate_cfg
        self.mu = None
        self.log_sigma = None
        self.vae = None

        if len(intermediate_cfg)<3 or 'latent' not in intermediate_cfg:
            raise ValueError("You should set intermediate_cfg list that containts number of LSTM layers and their dimensions "
                             " \n")

    def build_model(self):

        inputs = Input(shape=self.input_shape)

        if self.intermediate_cfg.index('latent') == 1:
            encoded = LSTM(self.intermediate_cfg[0])(inputs)
        else:
            encoded = LSTM(self.intermediate_cfg[0], return_sequences=True)(inputs)
            for dim in self.intermediate_cfg[1:self.intermediate_cfg.index('latent')-1]:
                encoded = LSTM(dim, return_sequences=True)(encoded)
            encoded = LSTM(self.intermediate_cfg[self.intermediate_cfg.index('latent')-1])(encoded)

        self.mu = Dense(self.latent_dim)(encoded)
        self.log_sigma = Dense(self.latent_dim)(encoded)

        z = Lambda(self.sampling, output_shape=(self.latent_dim,))([self.mu, self.log_sigma])

        decoded = RepeatVector(self.input_shape[0])(z)

        for dim in self.intermediate_cfg[self.intermediate_cfg.index('latent')+1:]:
            decoded = LSTM(dim, return_sequences=True)(decoded)

        decoder_dense = Dense(self.input_shape[1])
        decoded = TimeDistributed(decoder_dense)(decoded)

        self.vae = Model(inputs, decoded)
        self.vae.compile(optimizer='rmsprop', loss=self.vae_loss)
        self.vae.summary()

    def fit(self, X, epochs=10, batch_size=32, validation_split=None, verbose=1):
        self.build_model()
        self.vae.fit(X, X, epochs=epochs, batch_size=batch_size, validation_split=validation_split,verbose=verbose)

    def reconstruct(self, X):
        return self.vae.predict(X)

    def sampling(self, args):
        mu, log_sigma = args
        batch_size = K.shape(mu)[0]
        latent_dim = K.int_shape(mu)[1]
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.)
        return mu + K.exp(log_sigma / 2) * epsilon

    def vae_loss(self, y_true, y_pred):
        reconstruction_loss = mse(y_true, y_pred)
        reconstruction_loss = K.mean(reconstruction_loss)
        kl_loss = - 0.5 * K.sum(1 + self.log_sigma - K.square(self.mu) - K.exp(self.log_sigma), axis=-1)
        loss = K.mean(reconstruction_loss + kl_loss)
        return loss




class LSTM_Autoencoder(object):
    """
    input_shape=(X_train.shape[1], X_train.shape[2],)
    intermediate_cfg = None
    latent_dim = 32

    ae = LSTM_Autoencoder(input_shape, intermediate_cfg, latent_dim)
    ae.fit(X_train, epochs=10, batch_size=32, validation_split=None, verbose=1)
    """
    def __init__(self, input_shape, intermediate_cfg, latent_dim):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.intermediate_cfg = intermediate_cfg
        self.ae = None

        if intermediate_cfg != None and (len(intermediate_cfg) < 3 or 'latent' not in intermediate_cfg):
            raise ValueError(
                "You should set intermediate_cfg list that containts number of LSTM layers and their dimensions (or =None)"
                " \n")

    def build_model(self):

        inputs = Input(shape=self.input_shape)

        # Encoder
        if self.intermediate_cfg:
            encoded = LSTM(self.intermediate_cfg[0], return_sequences=True)(inputs)
            if self.intermediate_cfg.index('latent') > 1:
                for dim in self.intermediate_cfg[1:self.intermediate_cfg.index('latent')]:
                    encoded = LSTM(dim, return_sequences=True)(encoded)
            encoded = LSTM(self.latent_dim)(encoded)
        else:
            encoded = LSTM(self.latent_dim)(inputs)

        decoded = RepeatVector(self.input_shape[0])(encoded)

        # Decoder
        decoded = LSTM(self.latent_dim, return_sequences=True)(decoded)

        if self.intermediate_cfg:
            for dim in self.intermediate_cfg[self.intermediate_cfg.index('latent') + 1:]:
                decoded = LSTM(dim, return_sequences=True)(decoded)

        decoder_dense = Dense(self.input_shape[1])
        decoded = TimeDistributed(decoder_dense)(decoded)

        self.ae = Model(inputs, decoded)
        self.ae.compile(loss='mae', optimizer='adam')
        self.ae.summary()

    def fit(self, X, epochs=10, batch_size=32, validation_split=None, verbose=1):
        self.build_model()
        self.ae.fit(X, X, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=verbose)

    def reconstruct(self, X):
        return self.ae.predict(X)