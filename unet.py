import tensorflow as tf
import tensorflow.keras.layers as layers

def conv_block(x, kernelsize, filters, dropout, batchnorm=False):
    conv = layers.Conv2D(filters, (kernelsize, kernelsize), kernel_initializer='he_normal', padding="same")(x)
    if batchnorm:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)
    conv = layers.Conv2D(filters, (kernelsize, kernelsize), kernel_initializer='he_normal', padding="same")(conv)
    if batchnorm:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)
    return conv

def encoder_block(x, num_filters, dropout, batchnorm):
    x = conv_block(x, kernelsize=3, filters=num_filters, dropout=dropout, batchnorm=batchnorm)
    x = conv_block(x, kernelsize=3, filters=num_filters, dropout=dropout, batchnorm=batchnorm)

    p = layers.MaxPool2D((2, 2))(x)
    return x, p

def unet(input_shape, num_classes=1, dropout=0.3, batchnorm=True):
    """ Inputs """
    inputs = layers.Input(input_shape, name="input_layer")

    """ Encoder """
    e1, p1 = encoder_block(inputs, 64, dropout=dropout, batchnorm=batchnorm)
    e2, p2 = encoder_block(p1, 128, dropout=dropout, batchnorm=batchnorm)
    e3, p3 = encoder_block(p2, 256, dropout=dropout, batchnorm=batchnorm)
    e4, p4 = encoder_block(p3, 512, dropout=dropout, batchnorm=batchnorm)

    """ Bottleneck """
    b = conv_block(p4, kernelsize=3, filters=1024, dropout=dropout, batchnorm=batchnorm)
    b = conv_block(b, kernelsize=3, filters=1024, dropout=dropout, batchnorm=batchnorm)

    """ Decoder 4 """
    d4 = layers.UpSampling2D((2, 2), interpolation="bilinear")(b)
    d4 = conv_block(d4, filters=512, kernelsize=3, dropout=dropout, batchnorm=batchnorm)
    d4 = layers.Concatenate()([d4, e4])  # Concatenate with encoder block e4
    d4 = conv_block(d4, filters=512, kernelsize=3, dropout=dropout, batchnorm=batchnorm)

    """ Decoder 3 """
    d3 = layers.UpSampling2D((2, 2), interpolation="bilinear")(d4)
    d3 = conv_block(d3, filters=256, kernelsize=3, dropout=dropout, batchnorm=batchnorm)
    d3 = layers.Concatenate()([d3, e3])  # Concatenate with encoder block e3
    d3 = conv_block(d3, filters=256, kernelsize=3, dropout=dropout, batchnorm=batchnorm)

    """ Decoder 2 """
    d2 = layers.UpSampling2D((2, 2), interpolation="bilinear")(d3)
    d2 = conv_block(d2, filters=128, kernelsize=3, dropout=dropout, batchnorm=batchnorm)
    d2 = layers.Concatenate()([d2, e2])  # Concatenate with encoder block e2
    d2 = conv_block(d2, filters=128, kernelsize=3, dropout=dropout, batchnorm=batchnorm)

    """ Decoder 1 """
    d1 = layers.UpSampling2D((2, 2), interpolation="bilinear")(d2)
    d1 = conv_block(d1, filters=64, kernelsize=3, dropout=dropout, batchnorm=batchnorm)
    d1 = layers.Concatenate()([d1, e1])  # Concatenate with encoder block e1
    d1 = conv_block(d1, filters=64, kernelsize=3, dropout=dropout, batchnorm=batchnorm)

    """ Final Output """
    outputs = layers.Conv2D(num_classes, kernel_size=1, padding="same")(d1)
    outputs = layers.Activation("sigmoid")(outputs)

    model = tf.keras.Model(inputs, outputs)
    return model


if __name__ == "__main__":
    input_shape = (256, 256, 3)
    model = unet(input_shape)
    model.summary()