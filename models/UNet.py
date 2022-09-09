from tensorflow.keras import Input, Model, layers, regularizers


def create_unet(patch_size, n_features, n_outputs=1, n_filters=32, **model_kwargs):
    # https://arxiv.org/pdf/1505.04597.pdf
    
    def conv2d_block(x, activation='relu', batchnorm=False, renorm=False, dropout=0., filters=1, block_kwargs={}, **unused):
        for _ in range(2):
            x = layers.Conv2D(kernel_size=3, filters=filters, **block_kwargs)(x)
            if batchnorm: x = layers.BatchNormalization(renorm=renorm)(x)
            if dropout>0: x = layers.Dropout(dropout)(x)
            x = layers.Activation(activation)(x)    
        return x

    inputs  = x = Input(shape=(patch_size, patch_size, n_features))
    filters = list(range(5)) 
    kwargs  = {
        'padding'            : 'same',
        'kernel_initializer' : 'he_normal',
        'kernel_regularizer' : regularizers.L2(1e-4),
        'bias_regularizer'   : regularizers.L2(1e-4),
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
    model   = Model(inputs, outputs, name='UNet')
    return model