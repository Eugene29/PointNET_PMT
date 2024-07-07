import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling1D, Dropout
from qkeras import QDense, QConv1D, QActivation
from qkeras.quantizers import quantized_bits

def torch_default_lin_init(shape, dtype=None):
    fan_in = shape[0]
    bound = tf.sqrt(1.0 / fan_in)
    return tf.random.uniform(shape, minval=-bound, maxval=bound, dtype=dtype)

class qPointNetfeat(tf.keras.Model):
    '''
    PointNet (Encoder) Implementation
    1. 1x1 Conv1D layers (Literally a linear layer)
    2. Global Statistics (Mean shows superior performance than Min/Max)
    '''
    def __init__(self, dimensions, dim_reduce_factor, args):
        super(qPointNetfeat, self).__init__()
        dr = args["enc_dropout"]
        self.conv2lin = args["conv2lin"]
        self.encoder_input_shapes = [dimensions, 64, int(128 / dim_reduce_factor)] # hidden dimensions
        (_, F1, F2), self.latent_dim = self.encoder_input_shapes, int(1024 / dim_reduce_factor)

        if self.conv2lin:
            self.conv1 = QDense(F1, input_shape=(dimensions,), kernel_initializer=torch_default_lin_init, kernel_quantizer=quantized_bits(8, 0), bias_quantizer=quantized_bits(8, 0))
            self.conv2 = QDense(F2, input_shape=(F1,), kernel_initializer=torch_default_lin_init, kernel_quantizer=quantized_bits(8, 0), bias_quantizer=quantized_bits(8, 0))
            self.conv3 = QDense(self.latent_dim, input_shape=(F2,), kernel_initializer=torch_default_lin_init, kernel_quantizer=quantized_bits(8, 0), bias_quantizer=quantized_bits(8, 0))
        else:
            self.conv1 = QConv1D(F1, 1, input_shape=(None, dimensions))
            self.conv2 = QConv1D(F2, 1)
            self.conv3 = QConv1D(self.latent_dim, 1)
        
        self.dr1 = Dropout(dr)
        self.dr2 = Dropout(dr)

        self.act1 = QActivation("quantized_relu(8, 0)")
        self.act2 = QActivation("quantized_relu(8, 0)")

        self.mean_pooling = GlobalAveragePooling1D(data_format="channels_last" if self.conv2lin else "channels_first")

    def call(self, x, training=True):
        x = self.act1(self.dr1(self.conv1(x), training=training))
        x = self.act2(self.dr2(self.conv2(x), training=training))
        x = self.conv3(x)
        global_stats = self.mean_pooling(x)
        return global_stats

class qPointClassifier(tf.keras.Model):

    def __init__(self, n_hits, dim, dim_reduce_factor, out_dim, args, **kwargs):
        '''
        Main Model
        :param n_hits: number of points per point cloud
        :param dim: total dimensions of data (3 spatial + time and/or charge)
        '''
        super(qPointClassifier, self).__init__(**kwargs)
        self.n_hits, self.dim, self.dim_reduce_factor, self.out_dim, self.args, = n_hits, dim, dim_reduce_factor, out_dim, args,
        dr = args["dec_dropout"]
        self.training = True
        # self.n_hits = n_hits
        self.encoder = qPointNetfeat(dimensions=dim, dim_reduce_factor=dim_reduce_factor, args=args)
        self.encoder.decoder_input_shapes = self.encoder.latent_dim, int(512/dim_reduce_factor), int(128/dim_reduce_factor), 
        latent_dim, F3, F4 = self.encoder.decoder_input_shapes

        self.decoder = tf.keras.Sequential([
            tf.keras.Input(shape=(latent_dim,)),
            Dropout(dr),
            QDense(F3, kernel_initializer=torch_default_lin_init, kernel_quantizer=quantized_bits(8, 0), bias_quantizer=quantized_bits(8, 0)),
            QActivation("quantized_relu(8, 0)"),
            QDense(F4, kernel_initializer=torch_default_lin_init, kernel_quantizer=quantized_bits(8, 0), bias_quantizer=quantized_bits(8, 0)),
            QActivation("quantized_relu(8, 0)"),
            QDense(out_dim, kernel_initializer=torch_default_lin_init, kernel_quantizer=quantized_bits(8, 0), bias_quantizer=quantized_bits(8, 0)),
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

    def call(self, x, training=True):
        x = self.encoder(x, training=training)
        x = self.decoder(x, training=training)
        return x