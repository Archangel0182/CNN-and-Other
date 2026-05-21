import tensorflow as tf
import tensorflow.keras.layers as layers


def conv_block(x, kernelsize, filters, dropout, batchnorm=False):
    x = layers.Conv2D(filters, (kernelsize, kernelsize), kernel_initializer="he_normal", padding="same")(x)

    if batchnorm:
        x = layers.BatchNormalization(axis=3)(x)

    x = layers.Activation("relu")(x)

    if dropout > 0:
        x = layers.Dropout(dropout)(x)

    x = layers.Conv2D(filters, (kernelsize, kernelsize), kernel_initializer="he_normal", padding="same")(x)

    if batchnorm:
        x = layers.BatchNormalization(axis=3)(x)

    x = layers.Activation("relu")(x)

    return x


# ─────────────────────────────────────────────────────────────────────────────
#  Residual Conv Block
# ─────────────────────────────────────────────────────────────────────────────
def residual_conv_block(x, filters, dropout, batchnorm):
    shortcut = x

    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(x)

    if batchnorm:
        x = layers.BatchNormalization(axis=3)(x)

    x = layers.Activation("relu")(x)

    if dropout > 0:
        x = layers.Dropout(dropout)(x)

    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(x)

    if batchnorm:
        x = layers.BatchNormalization(axis=3)(x)

    # safer dynamic check
    if shortcut.shape[-1] is None or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same")(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)

    return x


# ─────────────────────────────────────────────────────────────────────────────
#  Patch Embedding
#  Image → Patch Tokens
# ─────────────────────────────────────────────────────────────────────────────
def patch_embedding(x, patch_size, embed_dim, grid_h, grid_w):
    x = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, padding="valid")(x)
    x = layers.Reshape((grid_h * grid_w, embed_dim))(x)
    return x


# ─────────────────────────────────────────────────────────────────────────────
#  Transformer Encoder Block (Pre-LN ViT Block)
# ─────────────────────────────────────────────────────────────────────────────
def transformer_block(x, embed_dim, num_heads, mlp_dim, dropout):

    # ── Multi-Head Self-Attention ──────────────────────────────────────────
    shortcut = x
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout)(x, x)
    x = layers.Add()([shortcut, x])

    # ── MLP ────────────────────────────────────────────────────────────────
    shortcut = x
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(mlp_dim, activation="gelu")(x)

    if dropout > 0:
        x = layers.Dropout(dropout)(x)

    x = layers.Dense(embed_dim)(x)

    if dropout > 0:
        x = layers.Dropout(dropout)(x)

    x = layers.Add()([shortcut, x])
    return x


# ─────────────────────────────────────────────────────────────────────────────
#  Skip Projection Block
# ─────────────────────────────────────────────────────────────────────────────
def skip_projection_block(x, filters, num_upsample_steps, dropout, batchnorm, grid_h, grid_w):
    embed_dim = x.shape[-1]

    # Tokens → spatial map
    x = layers.Reshape((grid_h, grid_w, embed_dim))(x)

    # Project channels
    x = layers.Conv2D(filters, kernel_size=1, padding="same")(x)

    # Initial residual refinement
    x = residual_conv_block(x, filters, dropout, batchnorm)

    # Progressive upsampling
    for _ in range(num_upsample_steps):
        x = layers.UpSampling2D((2, 2), interpolation="bilinear")(x)

        x = residual_conv_block(x, filters, dropout, batchnorm)

    return x


# ─────────────────────────────────────────────────────────────────────────────
#  UNETR Decoder Block
# ─────────────────────────────────────────────────────────────────────────────
def unetr_decoder_block(x, skip, filters, dropout, batchnorm):
    x = layers.UpSampling2D((2, 2), interpolation="bilinear")(x)

    if skip is not None:
        x = layers.Concatenate()([x, skip])

    x = residual_conv_block(x, filters, dropout, batchnorm)
    return x


# ─────────────────────────────────────────────────────────────────────────────
#  UNETR (2D Adaptation)
# ─────────────────────────────────────────────────────────────────────────────
def unetr(input_shape, num_classes=1, dropout=0.1, batchnorm=True, patch_size=16, embed_dim=768, num_heads=12, mlp_dim=3072, num_transformer_layers=12):
    H, W, C = input_shape
    assert H % patch_size == 0 and W % patch_size == 0, \
        "Input dimensions must be divisible by patch_size."

    # Patch grid dimensions
    grid_h = H // patch_size
    grid_w = W // patch_size

    num_patches = grid_h * grid_w

    # ────────────────────────────────────────────────────────────────────────
    # Input
    # ────────────────────────────────────────────────────────────────────────
    inputs = layers.Input(shape=input_shape, name="input_layer")

    # ────────────────────────────────────────────────────────────────────────
    # CNN Stem (High-Resolution Local Features)
    # ────────────────────────────────────────────────────────────────────────
    enc0 = conv_block(inputs, kernelsize=3, filters=64, dropout=dropout, batchnorm=batchnorm)

    # ────────────────────────────────────────────────────────────────────────
    # Patch Embedding
    # ────────────────────────────────────────────────────────────────────────
    tokens = patch_embedding(inputs, patch_size, embed_dim, grid_h, grid_w)

    # ────────────────────────────────────────────────────────────────────────
    # Positional Embeddings
    # ────────────────────────────────────────────────────────────────────────
    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embed = layers.Embedding(input_dim=num_patches, output_dim=embed_dim, name="pos_encoding")(positions)
    tokens = tokens + pos_embed

    # ────────────────────────────────────────────────────────────────────────
    # Transformer Encoder
    # ────────────────────────────────────────────────────────────────────────
    transformer_skips = []

    skip_indices = [
        num_transformer_layers // 4,
        num_transformer_layers // 2,
        3 * num_transformer_layers // 4,
        num_transformer_layers
    ]

    x = tokens

    for i in range(1, num_transformer_layers + 1):
        x = transformer_block(x, embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout)

        if i in skip_indices:
            transformer_skips.append(x)

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    transformer_skips[-1] = x

    # z3, z6, z9, z12
    z3, z6, z9, z12 = transformer_skips

    # ────────────────────────────────────────────────────────────────────────
    # Skip Projection
    # ────────────────────────────────────────────────────────────────────────
    z3 = skip_projection_block(z3, filters=64, num_upsample_steps=3, dropout=dropout, batchnorm=batchnorm, grid_h=grid_h, grid_w=grid_w)
    z6 = skip_projection_block(z6, filters=128, num_upsample_steps=2, dropout=dropout, batchnorm=batchnorm, grid_h=grid_h, grid_w=grid_w)
    z9 = skip_projection_block(z9, filters=256, num_upsample_steps=1, dropout=dropout, batchnorm=batchnorm, grid_h=grid_h, grid_w=grid_w)
    z12 = skip_projection_block(z12, filters=512, num_upsample_steps=0, dropout=dropout, batchnorm=batchnorm, grid_h=grid_h, grid_w=grid_w)

    # ────────────────────────────────────────────────────────────────────────
    # Decoder
    # ────────────────────────────────────────────────────────────────────────
    d = unetr_decoder_block(z12, skip=z9, filters=256, dropout=dropout, batchnorm=batchnorm)
    d = unetr_decoder_block(d, skip=z6, filters=128, dropout=dropout, batchnorm=batchnorm)
    d = unetr_decoder_block(d, skip=z3, filters=64, dropout=dropout, batchnorm=batchnorm)
    d = unetr_decoder_block(d, skip=enc0, filters=64, dropout=dropout, batchnorm=batchnorm)

    # ────────────────────────────────────────────────────────────────────────
    # Output Head
    # ────────────────────────────────────────────────────────────────────────
    outputs = layers.Conv2D(num_classes, kernel_size=1, padding="same")(d)

    if num_classes == 1:
        outputs = layers.Activation("sigmoid")(outputs)
    else:
        outputs = layers.Activation("softmax")(outputs)

    # ────────────────────────────────────────────────────────────────────────
    # Model
    # ────────────────────────────────────────────────────────────────────────
    model = tf.keras.Model(inputs, outputs, name="UNETR")

    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    input_shape = (256, 256, 3)
    model = unetr(input_shape)
    model.summary()