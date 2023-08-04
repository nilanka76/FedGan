
D_LEARNING_RATE = 0.0001    # Discriminater learning rate
G_LEARNING_RATE = 0.0001    # Generater learning rate
BATCH_SIZE = 16     # batch size
PATCH_NUM = 64     # patch per imag
PATCH_SHAPE = [BATCH_SIZE, 64, 64, 3]       # pathc size
BATCH_SHAPE = [BATCH_SIZE, 256, 256, 3]     # bathc size

#---------------------------------------Model----------------------------------------


import tensorflow as tf
from keras import Model, Sequential
import numpy as np
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, Activation, Flatten, Dense
# from utils import *

class generatorNet(Model):
    def __init__(self):
        super(generatorNet, self).__init__()
        # generator_cov Module
        self.cov1 = self.build_generator(part='cov', i=1)
        self.cov2 = self.build_generator(part='cov', i=2)
        self.cov3 = self.build_generator(part='cov', i=3)
        # res Module
        self.res_cov1_1 = self.res_block(1, 1)
        self.res_cov1_2 = self.res_block(1, 2)
        self.res_cov2_1 = self.res_block(2, 1)
        self.res_cov2_2 = self.res_block(2, 2)
        self.res_cov3_1 = self.res_block(3, 1)
        self.res_cov3_2 = self.res_block(3, 2)
        # generator_decov Module
        self.decov1 = self.build_generator(part='decov', i=1)
        self.decov2 = self.build_generator(part='decov', i=2)
        self.decov3 = self.build_generator(part='decov', i=3)

    def build_generator(self, part, i=1):
        if part == 'cov':
            if i == 1:
                parm = [32, 1]
            elif i == 2:
                parm = [64, 3]
            elif i == 3:
                parm = [128, 3]
            else:
                print('error')
            model = Sequential()
            model.add(Conv2D(parm[0], parm[1], padding='SAME', name='g_cov' + str(i)))
            model.add(BatchNormalization())
            model.add(LeakyReLU(alpha=0.2))
            return model

        elif part == 'decov':
            if i == 1:
                parm = [64, 3]
            elif i == 2:
                parm = [32, 3]
            elif i == 3:
                parm = [3, 3]
            else:
                print('error')
            model = Sequential()
            model.add(Conv2D(parm[0], parm[1], padding='SAME', name='g_decov' + str(i)))
            model.add(BatchNormalization())
            if i != 3:
                model.add(LeakyReLU(alpha=0.2))
            else:
                model.add(Activation(tf.nn.tanh))
            return model
        else:
            print('error')

    def res_block(self, n_res, n_cov):
        model = Sequential()
        model.add(Conv2D(128, 3, padding='SAME', name='res_cov' + str(n_cov) + '_' + str(n_res)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        return model

    def call(self, input):

        # generator_cov
        x1 = self.cov1(input)
        x = self.cov2(x1)
        x = self.cov3(x)

        # res_block_1
        y = self.res_cov1_1(x)
        y = self.res_cov1_2(y)
        x = tf.add(x, y)
        # res_block_2
        y = self.res_cov2_1(x)
        y = self.res_cov2_2(y)
        x = tf.add(x, y)
        # res_block_3
        y = self.res_cov3_1(x)
        y = self.res_cov3_2(y)
        x = tf.add(x, y)

        # generator_decov
        if input.shape[1] is not None:
            x = tf.image.resize(x, (input.shape[1] // 2, input.shape[2] // 2))
        else:
            x = tf.image.resize(x, (64, 64))
        x = self.decov1(x)
        if input.shape[1] is not None:
            x = tf.image.resize(x, (input.shape[1], input.shape[2]))
        else:
            x = tf.image.resize(x, (128, 128))
        x = self.decov2(x)
        x = tf.multiply(x1, x)
        x = self.decov3(x)
        x = tf.add(input, x)
        return x


class discriminatorNet(Model):
    def __init__(self):
        super(discriminatorNet, self).__init__()
        # discriminator Module
        self.discriminator = self.build_discriminator()

    def build_discriminator(self):
        model = Sequential()
        # 1st cov
        model.add(Conv2D(48, 4, padding='SAME', strides=2, name='d_cov1'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        # 2nd cov
        model.add(Conv2D(96, 4, padding='SAME', strides=2, name='d_cov2'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        # 3rd cov
        model.add(Conv2D(192, 4, padding='SAME', strides=2, name='d_cov3'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        # 4th cov
        model.add(Conv2D(384, 4, padding='SAME', name='d_cov4'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        # 5th cov
        model.add(Conv2D(1, 4, padding='SAME', name='d_cov5'))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(1))
        return model

    def call(self, input):
        x = self.discriminator(input)
        return x
    
if __name__ == "__main__":
    generator = generatorNet()
    generator2 = generatorNet()
    generator.build(input_shape=(None, None, None, BATCH_SHAPE[3]))
    generator2.build(input_shape=(None, None, None, BATCH_SHAPE[3]))


    discriminator = discriminatorNet()
    discriminator.build(input_shape=(None, PATCH_SHAPE[1], PATCH_SHAPE[2], PATCH_SHAPE[3]))

    g_optimizer = tf.keras.optimizers.Adam(learning_rate=G_LEARNING_RATE, name="g_optimizer")
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=D_LEARNING_RATE, name="d_optimizer")


    generator.load_weights("/root/ml/Simulations/generator_weights.h5")
    discriminator.load_weights("/root/ml/Simulations/discriminator_weights.h5")
    

    generator.summary()

    g = generator.get_weights()
    d = discriminator.get_weights()
    a = [g,d]
    def initialize_model1(weights):
        # Transfer Learning
        generator.set_weights(weights[0])
        g = generator.get_weights()
        g = [layer.tolist() for layer in g]  # Convert ndarray to list
        print("Generator Initialized with:", g[0])

        discriminator.set_weights(weights[1])
        d = discriminator.get_weights()
        d = [layer.tolist() for layer in d]  # Convert ndarray to list
        print("Discriminator Initialized with:", d[0])

        return [g, d]
    w = initialize_model1([g,d])

    def initialize_model2(model_weights):
    # Transfer Learning
        print("inside ", len(model_weights[0]))
        generator_weights = model_weights[0]
        # print(len(generator_weights[0]))
        generator_weights = [np.array(layer) for layer in generator_weights] # Convert list to ndarray


        discriminator_weights = model_weights[1]
        discriminator_weights = [np.array(layer) for layer in discriminator_weights]

        generator.set_weights(generator_weights)
        discriminator.set_weights(discriminator_weights)  # Convert list to ndarray
        
        return generator, discriminator
    initialize_model2(a)
    g, d = initialize_model2(w)

    print(type(g.get_weights()))