import tensorflow as tf
# from tensorflow.keras.layers import Conv1D, Dense, Dropout, ReLU, GlobalAveragePooling1D ## LeakyReLU,
# from tensorflow.keras.initializers import Initializer
from tf_keras.layers import Conv1D, Dense, Dropout, ReLU, GlobalAveragePooling1D ## LeakyReLU,
from tf_keras.initializers import Initializer
import tf_keras

class TorchDefaultLinInit(Initializer):
    def __call__(self, shape, dtype=None, **kwargs):
        fan_in = shape[0]
        bound = tf.sqrt(1.0 / fan_in)
        return tf.random.uniform(shape, minval=-bound, maxval=bound, dtype=dtype)

class PointNetfeat(tf_keras.Model):
    '''
    PointNet (Encoder) Implementation
    1. 1x1 Conv1D layers (Literally a linear layer)
    2. Global Statistics (Mean shows superior performance than Min/Max)
    '''
    def __init__(self, dimensions, dim_reduce_factor, args):
        super(PointNetfeat, self).__init__()
        dr = args["enc_dropout"]
        self.conv2lin = args["conv2lin"]
        self.encoder_input_shapes = [dimensions, 64, int(128 / dim_reduce_factor)] # hidden dimensions
        (_, F1, F2), self.latent_dim = self.encoder_input_shapes, int(1024 / dim_reduce_factor)
        nhits = 2126
        torch_default_lin_init = TorchDefaultLinInit()

        if self.conv2lin:
            self.conv1 = Dense(F1, input_shape=(nhits, dimensions,), kernel_initializer=torch_default_lin_init, bias_initializer=torch_default_lin_init)
            self.conv2 = Dense(F2, input_shape=(F1,), kernel_initializer=torch_default_lin_init, bias_initializer=torch_default_lin_init)
            self.conv3 = Dense(self.latent_dim, input_shape=(F2,), kernel_initializer=torch_default_lin_init, bias_initializer=torch_default_lin_init)
        else:
            self.conv1 = Conv1D(F1, 1, input_shape=(None, dimensions))
            self.conv2 = Conv1D(F2, 1)
            self.conv3 = Conv1D(self.latent_dim, 1)
        
        self.dr = Dropout(dr)

        self.mean_pooling = GlobalAveragePooling1D(data_format="channels_last" if self.conv2lin else "channels_first")


    def call(self, x, training=True):
        x = tf.nn.relu(self.conv1(x))
        x = tf.nn.relu(self.dr(self.conv2(x), training=training))
        x = self.conv3(x)
        global_stats = self.mean_pooling(x)
        return global_stats


class PointClassifier(tf_keras.Model):

    def __init__(self, n_hits, dim, dim_reduce_factor, out_dim, args, **kwargs):
        '''
        Main Model
        :param n_hits: number of points per point cloud
        :param dim: total dimensions of data (3 spatial + time and/or charge)
        '''
        self.n_hits, self.dim, self.dim_reduce_factor, self.out_dim, self.args, = n_hits, dim, dim_reduce_factor, out_dim, args,
        super(PointClassifier, self).__init__(**kwargs)
        dr = args["dec_dropout"]
        # self.training = True
        self.encoder = PointNetfeat(dimensions=dim, dim_reduce_factor=dim_reduce_factor, args=args)
        self.encoder.decoder_input_shapes = self.encoder.latent_dim, int(512/dim_reduce_factor), int(128/dim_reduce_factor), 
        latent_dim, F3, F4 = self.encoder.decoder_input_shapes
        torch_default_lin_init = TorchDefaultLinInit()

        self.decoder = tf_keras.Sequential([
            tf_keras.Input(shape=(latent_dim,)),
            Dropout(dr),
            Dense(F3, kernel_initializer=torch_default_lin_init, bias_initializer=torch_default_lin_init),
            ReLU(), ## Use Relu as qkeras doesn't support LeakyReLU
            Dense(F4, kernel_initializer=torch_default_lin_init, bias_initializer=torch_default_lin_init),
            ReLU(), ## Use Relu as qkeras doesn't support LeakyReLU
            Dense(out_dim, kernel_initializer=torch_default_lin_init, bias_initializer=torch_default_lin_init),
        ])

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_hits': self.n_hits,
            'dim': self.dim,
            'dim_reduce_factor': self.dim_reduce_factor,
            'out_dim': self.out_dim,
            'args': self.args
        })
        return config

    def call(self, x, training=False):
        import logging
        # tf.print(training)
        logging.debug(training)

        x = self.encoder(x, training=training)
        x = self.decoder(x, training=training)
        return x