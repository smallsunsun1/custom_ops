import tensorflow as tf
from tensorflow import keras


def batch_normalization(x, is_train=True):
    layer = keras.layers.BatchNormalization()
    y = layer(x, is_train)
    print(layer.updates)
    for ele in layer.updates:
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ele)
    return y


def batch_conv(inp, filters):
    """
        inp 的shape为[B, H, W, channels]
        filters 的shape为[B, kernel_size, kernel_size, channels, out_channels]
    """
    filters = tf.transpose(filters, perm=[1, 2, 0, 3, 4])
    filters_shape = tf.shape(filters)
    filters = tf.reshape(filters,
                         [filters_shape[0], filters_shape[1], filters_shape[2] * filters_shape[3], filters_shape[4]])
    inp_r = tf.transpose(inp, [1, 2, 0, 3])
    inp_shape = tf.shape(inp_r)
    inp_r = tf.reshape(inp_r, [1, inp_shape[0], inp_shape[1], inp_shape[2] * inp_shape[3]])
    padding = 'VALID'
    out = tf.nn.depthwise_conv2d(inp_r, filter=filters, strides=[1, 1, 1, 1], padding=padding)
    out = tf.reshape(out, [inp_shape[0] - filters_shape[0] + 1, inp_shape[1] - filters_shape[1] + 1, inp_shape[2],
                           inp_shape[3], filters_shape[4]])
    out = tf.transpose(out, [2, 0, 1, 3, 4])
    out = tf.reduce_sum(out, axis=3)
    return out


def carafe(feature_map, cm, upsample_scale, k_encoder, kernel_size):
    """implementation os ICCV 2019 oral presentation CARAFE module"""
    f1 = keras.layers.Conv2D(cm, (1, 1), padding="valid")(feature_map)
    encode_feature = keras.layers.Conv2D(upsample_scale * upsample_scale * kernel_size * kernel_size,
                                         (k_encoder, k_encoder), padding="same")(f1)
    encode_feature = tf.nn.depth_to_space(encode_feature, upsample_scale)
    encode_feature = tf.nn.softmax(encode_feature, axis=-1)
    """encode_feature [B x (h x scale) x (w x scale) x (kernel_size * kernel_size)]"""
    extract_feature = tf.image.extract_image_patches(feature_map, [1, kernel_size, kernel_size, 1],
                                                     strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding="SAME")
    """extract feature [B x h x w x (channel x kernel_size x kernel_size)]"""
    extract_feature = keras.layers.UpSampling2D((upsample_scale, upsample_scale))(extract_feature)
    extract_feature_shape = tf.shape(extract_feature)
    B = extract_feature_shape[0]
    H = extract_feature_shape[1]
    W = extract_feature_shape[2]
    block_size = kernel_size * kernel_size
    extract_feature = tf.reshape(extract_feature, [B, H, W, block_size, -1])
    extract_feature = tf.transpose(extract_feature, [0, 1, 2, 4, 3])
    """extract feature [B x (h x scale) x (w x scale) x channel x (kernel_size x kernel_size)]"""
    encode_feature = tf.expand_dims(encode_feature, axis=-1)
    upsample_feature = tf.matmul(extract_feature, encode_feature)
    upsample_feature = tf.squeeze(upsample_feature, axis=-1)
    return upsample_feature



if __name__ == "__main__":
    tf.enable_eager_execution()
    a = tf.ones(shape=[2, 64, 64, 128])
    print(carafe(a, 32, 3, 3, 3))

    # batch = tf.range(5)
    # h = tf.range(10)
    # w = tf.range(9)
    # res = tf.meshgrid(h, batch, w)
    # res = tf.stack([res[1], res[0], res[2]], axis=-1)
    # data = tf.ones(shape=[20, 10, 9, 64])
    # output = tf.gather_nd(data, res)
    # print(output)

    # a = tf.reshape(tf.range(3), [1, 1, 1, 3])
    # b = keras.layers.UpSampling2D()(a)
    # print(b)
    # b = tf.tile(a, [1, 1, 1, 3])
    # c = tf.nn.depth_to_space(b, 3)
    # print(a)
    # print(c)
