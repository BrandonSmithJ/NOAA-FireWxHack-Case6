import tensorflow as tf 


class SymmetricPadding2D(tf.keras.layers.Layer):
    """ Allow padding with mirror of values rather than 0 """
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReplicationPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        return tf.pad(input_tensor, [[0,0], [padding_height, padding_height], [padding_width, padding_width], [0,0] ], 'SYMMETRIC')
        
        

def create_unet(patch_size, n_features, n_outputs=1, n_filters=32, **model_kwargs):
    # https://arxiv.org/pdf/1505.04597.pdf
    from tensorflow.keras import layers
    
    def conv2d_block(x, activation='relu', batchnorm=False, renorm=False, dropout=0., filters=1, block_kwargs={}, **unused):
        for _ in range(2):
            if block_kwargs.get('padding', 'same') == 'valid': x = SymmetricPadding2D()(x)
            x = layers.Conv2D(kernel_size=3, filters=filters, **block_kwargs)(x)
            if batchnorm: x = layers.BatchNormalization(renorm=renorm)(x)
            if dropout>0: x = layers.Dropout(dropout)(x)
            x = layers.Activation(activation)(x)    
        return x

    inputs  = x = tf.keras.Input(shape=(patch_size, patch_size, n_features))
    filters = list(range(5)) 
    kwargs  = {
        'padding'            : 'valid',
        'kernel_initializer' : 'he_normal',
    }

    # Contracting Path
    blocks = []
    for expon in filters:
        x = block = conv2d_block(x, filters=n_filters * 2 ** expon, block_kwargs=kwargs, **model_kwargs)
        x = layers.MaxPooling2D(2)(x)
        blocks.append(block)

    filters = filters[:-1][::-1]
    x       = blocks.pop(-1)

    # Expansive path
    for expon in filters:
        x = block = layers.Conv2DTranspose(n_filters * 2 ** expon, kernel_size=2, strides=2, **kwargs)(x)
        x = layers.concatenate([block, blocks.pop(-1)])
        x = conv2d_block(x, filters=n_filters * 2 ** expon, block_kwargs=kwargs, **model_kwargs)

    outputs = layers.Conv2D(n_outputs, 1, activation='linear')(x)
    model   = tf.keras.Model(inputs, outputs, name='UNet')
    return model