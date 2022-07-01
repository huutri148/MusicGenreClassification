import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation, \
    GlobalAveragePooling2D
from tensorflow.keras.utils import plot_model


# Inherit Layer, establish resnet18 and 34 convolutional layer modules
class CellBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(CellBlock, self).__init__()

        self.conv1 = Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = BatchNormalization()
        self.relu = Activation('relu')

        self.conv2 = Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = BatchNormalization()

        if stride != 1:
            self.residual = Conv2D(filter_num, (1, 1), strides=stride)
        else:
            self.residual = lambda x: x

    def call(self, inputs, training=None):

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        r = self.residual(inputs)

        x = layers.add([x, r])
        output = tf.nn.relu(x)

        return output


# Inherit Model, create resnet18 and 34
class ResNet(models.Model):
    def __init__(self, layers_dims, nb_classes):
        super(ResNet, self).__init__()

        self.stem = Sequential([
            Conv2D(64, (7, 7), strides=(2, 2), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D((3, 3), strides=(2, 2), padding='same')
        ])  # Start module

        self.layer1 = self.build_cellblock(64, layers_dims[0])
        self.layer2 = self.build_cellblock(128, layers_dims[1], stride=2)
        self.layer3 = self.build_cellblock(256, layers_dims[2], stride=2)
        self.layer4 = self.build_cellblock(512, layers_dims[3], stride=2)

        self.avgpool = GlobalAveragePooling2D()
        self.fc = Dense(nb_classes, activation='softmax')

    def call(self, inputs, training=None):
        x = self.stem(inputs)
        # print(x.shape)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x

    def build_cellblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        res_blocks.add(CellBlock(filter_num, stride))  # The first block stride of each layer may be non-1

        for _ in range(1, blocks):  # How many blocks each layer consists of
            res_blocks.add(CellBlock(filter_num, stride=1))

        return res_blocks


def build_ResNet(NetName, nb_classes):
    ResNet_Config = {'ResNet18': [2, 2, 2, 2],
                     'ResNet34': [3, 4, 6, 3]}
    model = ResNet(ResNet_Config[NetName], nb_classes)
    model.build(input_shape=(None, 130, 13, 1))
    return model


if __name__ == '__main__':
    model = build_ResNet('ResNet18', 9)
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # model.save_weights("res18.h5")
    # model.load_weights("res18.h5")
    model.summary()
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True, rankdir='TB',
               expand_nested=False, dpi=96)
    model.summary()