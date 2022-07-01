from keras.models import Model
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, GlobalAveragePooling2D,Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.python import keras
from keras import layers
from keras import initializers
from keras import backend as K

class ResNet50V2():
    @staticmethod
    def build(height = 224, width = 224, depth = 3, classes = 1000):
        bn_axis = 3
        input_shape = (height, width, depth)
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            bn_axis = 1

        img_input = Input(shape = input_shape)
        x = ZeroPadding2D(padding=((3, 3), (3, 3)), name = 'conv1_pad')(img_input)
        x = Conv2D(64, 7, strides = 2, name = "conv1_conv")(x)
        x = ZeroPadding2D(padding = ((1, 1), (1, 1)), name = "pool1_pad")(x)
        x = MaxPooling2D(3, strides = 2, name = "pool1_pool")(x)

        preact = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv2_block1_preact_bn")(x)
        preact = Activation("relu", name = "conv2_block1_preact_relu")(preact)

        shortcut = Conv2D(4 * 64, 1, strides = 1, name = "conv2_block1_0_conv")(preact)
        x = Conv2D(64, 1, strides = 1, use_bias = False, name = "conv2_block1_1_conv")(preact)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv2_block1_1_bn")(x)
        x = Activation("relu",  name = "conv2_block1_1_relu")(x)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name = "conv2_block1_2_pad")(x)
        x = Conv2D(64, 3, strides = 1, use_bias = False, name = "conv2_block1_2_conv")(x)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv2_block1_2_bn")(x)
        x = Activation("relu", name = "conv2_block1_2_relu")(x)

        x = Conv2D(4 * 64, 1, name = "conv2_block1_3_conv")(x)
        x = Add(name = "conv2_block1_out")([shortcut, x])

        preact = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv2_block2_preact_bn")(x)
        preact = Activation("relu", name = "conv2_block2_preact_relu")(preact)

        shortcut = x
        x = Conv2D(64, 1, strides = 1, use_bias = False, name = "conv2_block2_1_conv")(preact)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv2_block2_1_bn")(x)
        x = Activation("relu",  name = "conv2_block2_1_relu")(x)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name = "conv2_block2_2_pad")(x)
        x = Conv2D(64, 3, strides = 1, use_bias = False, name = "conv2_block2_2_conv")(x)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv2_block2_2_bn")(x)
        x = Activation("relu", name = "conv2_block2_2_relu")(x)

        x = Conv2D(4 * 64, 1, name = "conv2_block2_3_conv")(x)
        x = Add(name = "conv2_block2_out")([shortcut, x])

        preact = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv2_block3_preact_bn")(x)
        preact = Activation("relu", name = "conv2_block3_preact_relu")(preact)

        shortcut = MaxPooling2D(1, strides = 2)(x)
        x = Conv2D(64, 1, strides = 1, use_bias = False, name = "conv2_block3_1_conv")(preact)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv2_block3_1_bn")(x)
        x = Activation("relu",  name = "conv2_block3_1_relu")(x)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name = "conv2_block3_2_pad")(x)
        x = Conv2D(64, 3, strides = 2, use_bias = False, name = "conv2_block3_2_conv")(x)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv2_block3_2_bn")(x)
        x = Activation("relu", name = "conv2_block3_2_relu")(x)

        x = Conv2D(4 * 64, 1, name = "conv2_block3_3_conv")(x)
        x = Add(name = "conv2_block3_out")([shortcut, x])

        preact = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv3_block1_preact_bn")(x)
        preact = Activation("relu", name = "conv3_block1_preact_relu")(preact)

        shortcut = Conv2D(4 * 128, 1, strides = 1, name = "conv3_block1_0_conv")(preact)
        x = Conv2D(128, 1, strides = 1, use_bias = False, name = "conv3_block1_1_conv")(preact)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv3_block1_1_bn")(x)
        x = Activation("relu",  name = "conv3_block1_1_relu")(x)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name = "conv3_block1_2_pad")(x)
        x = Conv2D(128, 3, strides = 1, use_bias = False, name = "conv3_block1_2_conv")(x)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv3_block1_2_bn")(x)
        x = Activation("relu", name = "conv3_block1_2_relu")(x)

        x = Conv2D(4 * 128, 1, name = "conv3_block1_3_conv")(x)
        x = Add(name = "conv3_block1_out")([shortcut, x])

        preact = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv3_block2_preact_bn")(x)
        preact = Activation("relu", name = "conv3_block2_preact_relu")(preact)

        shortcut = x
        x = Conv2D(128, 1, strides = 1, use_bias = False, name = "conv3_block2_1_conv")(preact)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv3_block2_1_bn")(x)
        x = Activation("relu",  name = "conv3_block2_1_relu")(x)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name = "conv3_block2_2_pad")(x)
        x = Conv2D(128, 3, strides = 1, use_bias = False, name = "conv3_block2_2_conv")(x)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv3_block2_2_bn")(x)
        x = Activation("relu", name = "conv3_block2_2_relu")(x)

        x = Conv2D(4 * 128, 1, name = "conv3_block2_3_conv")(x)
        x = Add(name = "conv3_block2_out")([shortcut, x])

        preact = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv3_block3_preact_bn")(x)
        preact = Activation("relu", name = "conv3_block3_preact_relu")(preact)

        shortcut = x
        x = Conv2D(128, 1, strides = 1, use_bias = False, name = "conv3_block3_1_conv")(preact)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv3_block3_1_bn")(x)
        x = Activation("relu",  name = "conv3_block3_1_relu")(x)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name = "conv3_block3_2_pad")(x)
        x = Conv2D(128, 3, strides = 1, use_bias = False, name = "conv3_block3_2_conv")(x)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv3_block3_2_bn")(x)
        x = Activation("relu", name = "conv3_block3_2_relu")(x)

        x = Conv2D(4 * 128, 1, name = "conv3_block_3_conv")(x)
        x = Add(name = "conv3_block3_out")([shortcut, x])

        preact = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv3_block4_preact_bn")(x)
        preact = Activation("relu", name = "conv3_block4_preact_relu")(preact)

        shortcut = MaxPooling2D(1, strides = 2)(x)
        x = Conv2D(128, 1, strides = 1, use_bias = False, name = "conv3_block4_1_conv")(preact)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv3_block4_1_bn")(x)
        x = Activation("relu",  name = "conv3_block4_1_relu")(x)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name = "conv3_block4_2_pad")(x)
        x = Conv2D(128, 3, strides = 2, use_bias = False, name = "conv3_block4_2_conv")(x)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv3_block4_2_bn")(x)
        x = Activation("relu", name = "conv3_block4_2_relu")(x)

        x = Conv2D(4 * 128, 1, name = "conv3_block4_3_conv")(x)
        x = Add(name = "conv3_block4_out")([shortcut, x])

        preact = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv4_block1_preact_bn")(x)
        preact = Activation("relu", name = "conv4_block1_preact_relu")(preact)

        shortcut = Conv2D(4 * 256, 1, strides = 1, name = "conv4_block1_0_conv")(preact)
        x = Conv2D(256, 1, strides = 1, use_bias = False, name = "conv4_block1_1_conv")(preact)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv4_block1_1_bn")(x)
        x = Activation("relu",  name = "conv4_block1_1_relu")(x)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name = "conv4_block1_2_pad")(x)
        x = Conv2D(256, 3, strides = 1, use_bias = False, name = "conv4_block1_2_conv")(x)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv4_block1_2_bn")(x)
        x = Activation("relu", name = "conv4_block1_2_relu")(x)

        x = Conv2D(4 * 256, 1, name = "conv4_block1_3_conv")(x)
        x = Add(name = "conv4_block1_out")([shortcut, x])

        preact = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv4_block2_preact_bn")(x)
        preact = Activation("relu", name = "conv4_block2_preact_relu")(preact)

        shortcut = x
        x = Conv2D(256, 1, strides = 1, use_bias = False, name = "conv4_block2_1_conv")(preact)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv4_block2_1_bn")(x)
        x = Activation("relu",  name = "conv4_block2_1_relu")(x)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name = "conv4_block2_2_pad")(x)
        x = Conv2D(256, 3, strides = 1, use_bias = False, name = "conv4_block2_2_conv")(x)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv4_block2_2_bn")(x)
        x = Activation("relu", name = "conv4_block2_2_relu")(x)

        x = Conv2D(4 * 256, 1, name = "conv4_block2_3_conv")(x)
        x = Add(name = "conv4_block2_out")([shortcut, x])

        preact = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv4_block3_preact_bn")(x)
        preact = Activation("relu", name = "conv4_block3_preact_relu")(preact)

        shortcut = x
        x = Conv2D(256, 1, strides = 1, use_bias = False, name = "conv4_block3_1_conv")(preact)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv4_block3_1_bn")(x)
        x = Activation("relu",  name = "conv4_block3_1_relu")(x)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name = "conv4_block3_2_pad")(x)
        x = Conv2D(256, 3, strides = 1, use_bias = False, name = "conv4_block3_2_conv")(x)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv4_block3_2_bn")(x)
        x = Activation("relu", name = "conv4_block3_2_relu")(x)

        x = Conv2D(4 * 256, 1, name = "conv4_block3_3_conv")(x)
        x = Add(name = "conv4_block3_out")([shortcut, x])

        preact = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv4_block4_preact_bn")(x)
        preact = Activation("relu", name = "conv4_block4_preact_relu")(preact)

        shortcut = x
        x = Conv2D(256, 1, strides = 1, use_bias = False, name = "conv4_block4_1_conv")(preact)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv4_block4_1_bn")(x)
        x = Activation("relu",  name = "conv4_block4_1_relu")(x)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name = "conv4_block4_2_pad")(x)
        x = Conv2D(256, 3, strides = 1, use_bias = False, name = "conv4_block4_2_conv")(x)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv4_block4_2_bn")(x)
        x = Activation("relu", name = "conv4_block4_2_relu")(x)

        x = Conv2D(4 * 256, 1, name = "conv4_block4_3_conv")(x)
        x = Add(name = "conv4_block4_out")([shortcut, x])

        preact = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv4_block5_preact_bn")(x)
        preact = Activation("relu", name = "conv4_block5_preact_relu")(preact)

        shortcut = x
        x = Conv2D(256, 1, strides = 1, use_bias = False, name = "conv4_block5_1_conv")(preact)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv4_block5_1_bn")(x)
        x = Activation("relu",  name = "conv4_block5_1_relu")(x)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name = "conv4_block5_2_pad")(x)
        x = Conv2D(256, 3, strides = 1, use_bias = False, name = "conv4_block5_2_conv")(x)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv4_block5_2_bn")(x)
        x = Activation("relu", name = "conv4_block5_2_relu")(x)

        x = Conv2D(4 * 256, 1, name = "conv4_block5_3_conv")(x)
        x = Add(name = "conv4_block5_out")([shortcut, x])

        preact = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv4_block6_preact_bn")(x)
        preact = Activation("relu", name = "conv4_block6_preact_relu")(preact)

        shortcut = MaxPooling2D(1, strides = 2)(x)
        x = Conv2D(256, 1, strides = 1, use_bias = False, name = "conv4_block6_1_conv")(preact)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv4_block6_1_bn")(x)
        x = Activation("relu",  name = "conv4_block6_1_relu")(x)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name = "conv4_block6_2_pad")(x)
        x = Conv2D(256, 3, strides = 2, use_bias = False, name = "conv4_block6_2_conv")(x)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv4_block6_2_bn")(x)
        x = Activation("relu", name = "conv4_block6_2_relu")(x)

        x = Conv2D(4 * 256, 1, name = "conv4_block6_3_conv")(x)
        x = Add(name = "conv4_block6_out")([shortcut, x])

        preact = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv5_block1_preact_bn")(x)
        preact = Activation("relu", name = "conv5_block1_preact_relu")(preact)

        shortcut = Conv2D(4 * 512, 1, strides = 1, name = "conv5_block1_0_conv")(preact)
        x = Conv2D(512, 1, strides = 1, use_bias = False, name = "conv5_block1_1_conv")(preact)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv5_block1_1_bn")(x)
        x = Activation("relu",  name = "conv5_block1_1_relu")(x)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name = "conv5_block1_2_pad")(x)
        x = Conv2D(512, 3, strides = 1, use_bias = False, name = "conv5_block1_2_conv")(x)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv5_block1_2_bn")(x)
        x = Activation("relu", name = "conv5_block1_2_relu")(x)

        x = Conv2D(4 * 512, 1, name = "conv5_block1_3_conv")(x)
        x = Add(name = "conv5_block1_out")([shortcut, x])

        preact = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv5_block2_preact_bn")(x)
        preact = Activation("relu", name = "conv5_block2_preact_relu")(preact)

        shortcut = x
        x = Conv2D(512, 1, strides = 1, use_bias = False, name = "conv5_block2_1_conv")(preact)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv5_block2_1_bn")(x)
        x = Activation("relu",  name = "conv5_block2_1_relu")(x)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name = "conv5_block2_2_pad")(x)
        x = Conv2D(512, 3, strides = 1, use_bias = False, name = "conv5_block2_2_conv")(x)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv5_block2_2_bn")(x)
        x = Activation("relu", name = "conv5_block2_2_relu")(x)

        x = Conv2D(4 * 512, 1, name = "conv5_block2_3_conv")(x)
        x = Add(name = "conv5_block2_out")([shortcut, x])

        preact = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv5_block3_preact_bn")(x)
        preact = Activation("relu", name = "conv5_block3_preact_relu")(preact)

        shortcut = x
        x = Conv2D(512, 1, strides = 1, use_bias = False, name = "conv5_block3_1_conv")(preact)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv5_block3_1_bn")(x)
        x = Activation("relu",  name = "conv5_block3_1_relu")(x)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name = "conv5_block3_2_pad")(x)
        x = Conv2D(512, 3, strides = 1, use_bias = False, name = "conv5_block3_2_conv")(x)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv5_block3_2_bn")(x)
        x = Activation("relu", name = "conv5_block3_2_relu")(x)

        x = Conv2D(4 * 512, 1, name = "conv5_block3_3_conv")(x)
        x = Add(name = "conv5_block3_out")([shortcut, x])


        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "post_bn")(x)
        x = Activation("relu", name = "post_relu")(x)

        x = GlobalAveragePooling2D(name = "avg_pool")(x)
        x = Dense(classes, activation = "softmax", name = "probs")(x)

        model = Model(img_input, x, name = "ResNet50V2")
        return model