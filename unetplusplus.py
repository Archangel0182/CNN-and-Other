import tensorflow as tf
import tensorflow.keras.layers as layers


def conv_block(x, kernelsize, filters, dropout, batchnorm=False):
    conv = layers.Conv2D(filters, (kernelsize, kernelsize), kernel_initializer='he_normal',  padding="same")(x)
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


def unet_plus_plus(input_shape, num_classes=1, dropout=0.3, batchnorm=True):
    filters = [64, 128, 256, 512, 1024]

    inputs = layers.Input(input_shape, name="input_layer")

    # ──────────────────────────────────────────
    # ENCODER  (column j=0)
    # ──────────────────────────────────────────
    x = [[None] * 5 for _ in range(5)]   # x[depth][dense_step]

    x[0][0] = conv_block(inputs, kernelsize=3, filters=filters[0], dropout=dropout, batchnorm=batchnorm)
    x[1][0] = conv_block(layers.MaxPool2D((2, 2))(x[0][0]), kernelsize=3, filters=filters[1], dropout=dropout, batchnorm=batchnorm)
    x[2][0] = conv_block(layers.MaxPool2D((2, 2))(x[1][0]), kernelsize=3, filters=filters[2], dropout=dropout, batchnorm=batchnorm)
    x[3][0] = conv_block(layers.MaxPool2D((2, 2))(x[2][0]), kernelsize=3, filters=filters[3], dropout=dropout, batchnorm=batchnorm)

    # ──────────────────────────────────────────
    # BOTTLENECK  (depth 4, j=0)
    # ──────────────────────────────────────────
    x[4][0] = conv_block(layers.MaxPool2D((2, 2))(x[3][0]), kernelsize=3, filters=filters[4], dropout=dropout, batchnorm=batchnorm)
    x[4][0] = conv_block(x[4][0], kernelsize=3, filters=filters[4], dropout=dropout, batchnorm=batchnorm)

    # ── j = 1 ──────────────────────────────────
    # x[0][1]: upsample x[1][0] → concat with x[0][0]
    up = layers.UpSampling2D((2, 2), interpolation="bilinear")(x[1][0])
    up = conv_block(up, kernelsize=3, filters=filters[0], dropout=dropout, batchnorm=batchnorm)
    x[0][1] = conv_block(layers.Concatenate()([x[0][0], up]), kernelsize=3, filters=filters[0], dropout=dropout, batchnorm=batchnorm)

    # x[1][1]: upsample x[2][0] → concat with x[1][0]
    up = layers.UpSampling2D((2, 2), interpolation="bilinear")(x[2][0])
    up = conv_block(up, kernelsize=3, filters=filters[1], dropout=dropout, batchnorm=batchnorm)
    x[1][1] = conv_block(layers.Concatenate()([x[1][0], up]), kernelsize=3, filters=filters[1], dropout=dropout, batchnorm=batchnorm)

    # x[2][1]: upsample x[3][0] → concat with x[2][0]
    up = layers.UpSampling2D((2, 2), interpolation="bilinear")(x[3][0])
    up = conv_block(up, kernelsize=3, filters=filters[2], dropout=dropout, batchnorm=batchnorm)
    x[2][1] = conv_block(layers.Concatenate()([x[2][0], up]), kernelsize=3, filters=filters[2], dropout=dropout, batchnorm=batchnorm)

    # x[3][1]: upsample x[4][0] → concat with x[3][0]
    up = layers.UpSampling2D((2, 2), interpolation="bilinear")(x[4][0])
    up = conv_block(up, kernelsize=3, filters=filters[3], dropout=dropout, batchnorm=batchnorm)
    x[3][1] = conv_block(layers.Concatenate()([x[3][0], up]), kernelsize=3, filters=filters[3], dropout=dropout, batchnorm=batchnorm)

    # ── j = 2 ──────────────────────────────────
    # x[0][2]: upsample x[1][1] → concat with x[0][0], x[0][1]
    up = layers.UpSampling2D((2, 2), interpolation="bilinear")(x[1][1])
    up = conv_block(up, kernelsize=3, filters=filters[0], dropout=dropout, batchnorm=batchnorm)
    x[0][2] = conv_block(layers.Concatenate()([x[0][0], x[0][1], up]), kernelsize=3, filters=filters[0], dropout=dropout, batchnorm=batchnorm)

    # x[1][2]: upsample x[2][1] → concat with x[1][0], x[1][1]
    up = layers.UpSampling2D((2, 2), interpolation="bilinear")(x[2][1])
    up = conv_block(up, kernelsize=3, filters=filters[1], dropout=dropout, batchnorm=batchnorm)
    x[1][2] = conv_block(layers.Concatenate()([x[1][0], x[1][1], up]), kernelsize=3, filters=filters[1], dropout=dropout, batchnorm=batchnorm)

    # x[2][2]: upsample x[3][1] → concat with x[2][0], x[2][1]
    up = layers.UpSampling2D((2, 2), interpolation="bilinear")(x[3][1])
    up = conv_block(up, kernelsize=3, filters=filters[2], dropout=dropout, batchnorm=batchnorm)
    x[2][2] = conv_block(layers.Concatenate()([x[2][0], x[2][1], up]), kernelsize=3, filters=filters[2], dropout=dropout, batchnorm=batchnorm)

    # ── j = 3 ──────────────────────────────────
    # x[0][3]: upsample x[1][2] → concat with x[0][0], x[0][1], x[0][2]
    up = layers.UpSampling2D((2, 2), interpolation="bilinear")(x[1][2])
    up = conv_block(up, kernelsize=3, filters=filters[0], dropout=dropout, batchnorm=batchnorm)
    x[0][3] = conv_block(layers.Concatenate()([x[0][0], x[0][1], x[0][2], up]), kernelsize=3, filters=filters[0], dropout=dropout, batchnorm=batchnorm)

    # x[1][3]: upsample x[2][2] → concat with x[1][0], x[1][1], x[1][2]
    up = layers.UpSampling2D((2, 2), interpolation="bilinear")(x[2][2])
    up = conv_block(up, kernelsize=3, filters=filters[1], dropout=dropout, batchnorm=batchnorm)
    x[1][3] = conv_block(layers.Concatenate()([x[1][0], x[1][1], x[1][2], up]), kernelsize=3, filters=filters[1], dropout=dropout, batchnorm=batchnorm)

    # ── j = 4 ──────────────────────────────────
    # x[0][4]: upsample x[1][3] → concat with x[0][0..3]
    up = layers.UpSampling2D((2, 2), interpolation="bilinear")(x[1][3])
    up = conv_block(up, kernelsize=3, filters=filters[0], dropout=dropout, batchnorm=batchnorm)
    x[0][4] = conv_block(layers.Concatenate()([x[0][0], x[0][1], x[0][2], x[0][3], up]), kernelsize=3, filters=filters[0], dropout=dropout, batchnorm=batchnorm)

    # ──────────────────────────────────────────
    # OUTPUT  (from the final dense node x[0][4])
    # ──────────────────────────────────────────
    outputs = layers.Conv2D(num_classes, kernel_size=1, padding="same")(x[0][4])
    
    if num_classes == 1:
        outputs = layers.Activation("sigmoid")(outputs)
    else:
        outputs = layers.Activation("softmax")(outputs)

    model = tf.keras.Model(inputs, outputs, name="UNetPlusPlus")
    return model


if __name__ == "__main__":
    input_shape = (256, 256, 3)
    model = unet_plus_plus(input_shape)
    model.summary()