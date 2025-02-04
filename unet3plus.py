import tensorflow as tf
import tensorflow.keras.layers as layers

def conv_block(x, kernelsize, filters, dropout, batchnorm=False):
    conv = layers.Conv2D(filters, (kernelsize, kernelsize), kernel_initializer='he_normal', padding="same")(x)
    if batchnorm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)
    conv = layers.Conv2D(filters, (kernelsize, kernelsize), kernel_initializer='he_normal', padding="same")(conv)
    if batchnorm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)
    return conv

def encoder_block(x, num_filters, dropout, batchnorm):
    x = conv_block(x, kernelsize=3, filters=num_filters, dropout=dropout, batchnorm=batchnorm)
    x = conv_block(x, kernelsize=3, filters=num_filters, dropout=dropout, batchnorm=batchnorm)

    p = layers.MaxPool2D((2, 2))(x)
    return x, p

def unet3plus(input_shape, num_classes=1, dropout=0.3, batchnorm=True):
    """ Inputs """
    inputs = layers.Input(input_shape, name="input_layer")

    """ Encoder """
    e1, p1 = encoder_block(inputs, 64, dropout=dropout, batchnorm=batchnorm)
    e2, p2 = encoder_block(p1, 128, dropout=dropout, batchnorm=batchnorm)
    e3, p3 = encoder_block(p2, 256, dropout=dropout, batchnorm=batchnorm)
    e4, p4 = encoder_block(p3, 512, dropout=dropout, batchnorm=batchnorm)

    """ Bottleneck """
    e5 = conv_block(p4, kernelsize=3, filters=1024, dropout=dropout, batchnorm=batchnorm)
    e5 = conv_block(e5, kernelsize=3, filters=1024, dropout=dropout, batchnorm=batchnorm)

    """ Decoder 4 """
    e1_d4 = layers.MaxPool2D((8, 8))(e1)
    e1_d4 = conv_block(e1_d4, kernelsize=3, filters=64, dropout=dropout, batchnorm=batchnorm)

    e2_d4 = layers.MaxPool2D((4, 4))(e2)
    e2_d4 = conv_block(e2_d4, kernelsize=3, filters=64, dropout=dropout, batchnorm=batchnorm)

    e3_d4 = layers.MaxPool2D((2, 2))(e3)
    e3_d4 = conv_block(e3_d4, kernelsize=3, filters=64, dropout=dropout, batchnorm=batchnorm)

    e4_d4 = conv_block(e4, kernelsize=3, filters=64, dropout=dropout, batchnorm=batchnorm)

    e5_d4 = layers.UpSampling2D((2, 2), interpolation="bilinear")(e5)
    e5_d4 = conv_block(e5_d4, kernelsize=3, filters=64, dropout=dropout, batchnorm=batchnorm)

    d4 = layers.Concatenate()([e1_d4, e2_d4, e3_d4, e4_d4, e5_d4])
    d4 = conv_block(d4, kernelsize=3, filters=64*5, dropout=dropout, batchnorm=batchnorm)

    """ Decoder 3 """
    e1_d3 = layers.MaxPool2D((4, 4))(e1)
    e1_d3 = conv_block(e1_d3, kernelsize=3, filters=64, dropout=dropout, batchnorm=batchnorm)

    e2_d3 = layers.MaxPool2D((2, 2))(e2)
    e2_d3 = conv_block(e2_d3, kernelsize=3, filters=64, dropout=dropout, batchnorm=batchnorm)

    e3_d3 = conv_block(e3, kernelsize=3, filters=64, dropout=dropout, batchnorm=batchnorm)

    d4_d3 = layers.UpSampling2D((2, 2), interpolation="bilinear")(d4)
    d4_d3 = conv_block(d4_d3, kernelsize=3, filters=64, dropout=dropout, batchnorm=batchnorm)

    e5_d3 = layers.UpSampling2D((4, 4), interpolation="bilinear")(e5)
    e5_d3 = conv_block(e5_d3, kernelsize=3, filters=64, dropout=dropout, batchnorm=batchnorm)

    d3 = layers.Concatenate()([e1_d3, e2_d3, e3_d3, d4_d3, e5_d3])
    d3 = conv_block(d3, kernelsize=3, filters=64*5, dropout=dropout, batchnorm=batchnorm)

    """ Decoder 2 """
    e1_d2 = layers.MaxPool2D((2, 2))(e1)
    e1_d2 = conv_block(e1_d2, kernelsize=3, filters=64, dropout=dropout, batchnorm=batchnorm)

    e2_d2 = conv_block(e2, kernelsize=3, filters=64, dropout=dropout, batchnorm=batchnorm)

    d3_d2 = layers.UpSampling2D((2, 2), interpolation="bilinear")(d3)
    d3_d2 = conv_block(d3_d2, kernelsize=3, filters=64, dropout=dropout, batchnorm=batchnorm)

    d4_d2 = layers.UpSampling2D((4, 4), interpolation="bilinear")(d4)
    d4_d2 = conv_block(d4_d2, kernelsize=3, filters=64, dropout=dropout, batchnorm=batchnorm)

    e5_d2 = layers.UpSampling2D((8, 8), interpolation="bilinear")(e5)
    e5_d2 = conv_block(e5_d2, kernelsize=3, filters=64, dropout=dropout, batchnorm=batchnorm)

    d2 = layers.Concatenate()([e1_d2, e2_d2, d3_d2, d4_d2, e5_d2])
    d2 = conv_block(d2, kernelsize=3, filters=64*5, dropout=dropout, batchnorm=batchnorm)

    """ Decoder 1 """
    e1_d1 = conv_block(e1, kernelsize=3, filters=64, dropout=dropout, batchnorm=batchnorm)

    d2_d1 = layers.UpSampling2D((2, 2), interpolation="bilinear")(d2)
    d2_d1 = conv_block(d2_d1, kernelsize=3, filters=64, dropout=dropout, batchnorm=batchnorm)

    d3_d1 = layers.UpSampling2D((4, 4), interpolation="bilinear")(d3)
    d3_d1 = conv_block(d3_d1, kernelsize=3, filters=64, dropout=dropout, batchnorm=batchnorm)

    d4_d1 = layers.UpSampling2D((8, 8), interpolation="bilinear")(d4)
    d4_d1 = conv_block(d4_d1, kernelsize=3, filters=64, dropout=dropout, batchnorm=batchnorm)

    e5_d1 = layers.UpSampling2D((16, 16), interpolation="bilinear")(e5)
    e5_d1 = conv_block(e5_d1, kernelsize=3, filters=64, dropout=dropout, batchnorm=batchnorm)

    d1 = layers.Concatenate()([e1_d1, d2_d1, d3_d1, d4_d1, e5_d1])
    d1 = conv_block(d1, kernelsize=3, filters=64*5, dropout=dropout, batchnorm=batchnorm)

    """ Final Output """
    # No deep supervision, just a single output
    y1 = layers.Conv2D(num_classes, kernel_size=1, padding="same")(d1)
    y1 = layers.Activation("sigmoid")(y1)

    outputs = [y1]

    model = tf.keras.Model(inputs, outputs)
    return model


if __name__ == "__main__":
    input_shape = (256, 256, 3)
    model = unet3plus(input_shape)
    model.summary()