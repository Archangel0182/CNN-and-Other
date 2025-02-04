import tensorflow as tf
import tensorflow.keras.layers as layers

def gatingsignal(input, out_size, batchnorm=False):
    """ Gating signal for attention unit """
    x = layers.Conv2D(out_size, (1, 1), padding='same')(input)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def attention_block(x, gating, inter_shape):
    """ Attention block based on soft attention """
    shape_x = tf.keras.backend.int_shape(x)
    shape_g = tf.keras.backend.int_shape(gating)
    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), kernel_initializer='he_normal', padding='same')(x)
    shape_theta_x = tf.keras.backend.int_shape(theta_x)
    phi_g = layers.Conv2D(inter_shape, (1, 1), kernel_initializer='he_normal', padding='same')(gating)
    upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3), strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]), kernel_initializer='he_normal', padding='same')(phi_g)
    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation("relu")(concat_xg)
    psi = layers.Conv2D(1, (1, 1), kernel_initializer='he_normal', padding='same')(act_xg)
    sigmoid_xg = layers.Activation("sigmoid")(psi)
    shape_sigmoid = tf.keras.backend.int_shape(sigmoid_xg)
    upsample_psi = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)
    upsample_psi = layers.Lambda(lambda x, repnum: tf.keras.backend.repeat_elements(x, repnum, axis=3), arguments={'repnum': shape_x[3]})(upsample_psi)
    y = layers.multiply([upsample_psi, x])
    result = layers.Conv2D(shape_x[3], (1, 1), kernel_initializer='he_normal', padding='same')(y)
    attenblock = layers.BatchNormalization()(result)
    return attenblock

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

    """ Decoder 4 with Attention Gates """
    e1_d4 = layers.MaxPool2D((8, 8))(e1)
    e1_d4 = conv_block(e1_d4, kernelsize=3, filters=64, dropout=dropout, batchnorm=batchnorm)
    e1_d4 = attention_block(e1_d4, gatingsignal(e5, 64), 64)

    e2_d4 = layers.MaxPool2D((4, 4))(e2)
    e2_d4 = conv_block(e2_d4, kernelsize=3, filters=64, dropout=dropout, batchnorm=batchnorm)
    e2_d4 = attention_block(e2_d4, gatingsignal(e5, 64), 64)

    e3_d4 = layers.MaxPool2D((2, 2))(e3)
    e3_d4 = conv_block(e3_d4, kernelsize=3, filters=64, dropout=dropout, batchnorm=batchnorm)
    e3_d4 = attention_block(e3_d4, gatingsignal(e5, 64), 64)

    e4_d4 = conv_block(e4, kernelsize=3, filters=64, dropout=dropout, batchnorm=batchnorm)
    e4_d4 = attention_block(e4_d4, gatingsignal(e5, 64), 64)

    e5_d4 = layers.UpSampling2D((2, 2), interpolation="bilinear")(e5)
    e5_d4 = conv_block(e5_d4, kernelsize=3, filters=64, dropout=dropout, batchnorm=batchnorm)
    e5_d4 = attention_block(e5_d4, gatingsignal(e5, 64), 64)

    d4 = layers.Concatenate()([e1_d4, e2_d4, e3_d4, e4_d4, e5_d4])
    d4 = conv_block(d4, kernelsize=3, filters=64*5, dropout=dropout, batchnorm=batchnorm)

    """ Decoder 3 with Attention Gates """
    e1_d3 = layers.MaxPool2D((4, 4))(e1)
    e1_d3 = conv_block(e1_d3, kernelsize=3, filters=64, dropout=dropout, batchnorm=batchnorm)
    e1_d3 = attention_block(e1_d3, gatingsignal(e5, 64), 64)

    e2_d3 = layers.MaxPool2D((2, 2))(e2)
    e2_d3 = conv_block(e2_d3, kernelsize=3, filters=64, dropout=dropout, batchnorm=batchnorm)
    e2_d3 = attention_block(e2_d3, gatingsignal(e5, 64), 64)

    e3_d3 = conv_block(e3, kernelsize=3, filters=64, dropout=dropout, batchnorm=batchnorm)
    e3_d3 = attention_block(e3_d3, gatingsignal(e5, 64), 64)

    d4_d3 = layers.UpSampling2D((2, 2), interpolation="bilinear")(d4)
    d4_d3 = conv_block(d4_d3, kernelsize=3, filters=64, dropout=dropout, batchnorm=batchnorm)
    d4_d3 = attention_block(d4_d3, gatingsignal(e5, 64), 64)

    e5_d3 = layers.UpSampling2D((4, 4), interpolation="bilinear")(e5)
    e5_d3 = conv_block(e5_d3, kernelsize=3, filters=64, dropout=dropout, batchnorm=batchnorm)
    e5_d3 = attention_block(e5_d3, gatingsignal(e5, 64), 64)

    d3 = layers.Concatenate()([e1_d3, e2_d3, e3_d3, d4_d3, e5_d3])
    d3 = conv_block(d3, kernelsize=3, filters=64*5, dropout=dropout, batchnorm=batchnorm)

    """ Final Output Layer """
    y1 = layers.Conv2D(num_classes, kernel_size=1, padding="same")(d3)
    y1 = layers.Activation("sigmoid")(y1)

    outputs = [y1]

    model = tf.keras.Model(inputs, outputs)
    return model


if __name__ == "__main__":
    input_shape = (256, 256, 3)
    model = unet3plus(input_shape)
    model.summary()