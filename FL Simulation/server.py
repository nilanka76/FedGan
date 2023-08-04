from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from model import generatorNet, discriminatorNet

app = Flask(__name__)

D_LEARNING_RATE = 0.0001    # Discriminator learning rate
G_LEARNING_RATE = 0.0001  
BATCH_SIZE = 16 
PATCH_NUM = 64  
PATCH_SHAPE = [BATCH_SIZE, 64, 64, 3] 
BATCH_SHAPE = [BATCH_SIZE, 256, 256, 3] 


# Global model weights
global_model_weights_generator = None
global_model_weights_discriminator = None
G_model_weight_list_generator = []
G_model_weight_list_discriminator = []
num_parties = 1

generator = generatorNet()
generator.build(input_shape=(None, None, None, BATCH_SHAPE[3]))

discriminator = discriminatorNet()
discriminator.build(input_shape=(None, PATCH_SHAPE[1], PATCH_SHAPE[2], PATCH_SHAPE[3]))

g_optimizer = tf.keras.optimizers.Adam(learning_rate=G_LEARNING_RATE, name="g_optimizer")
d_optimizer = tf.keras.optimizers.Adam(learning_rate=D_LEARNING_RATE, name="d_optimizer")

max_ssmi = 0
max_psnr = 0
min_mse = 0

generator.load_weights("./generator_weights.h5")
discriminator.load_weights("./discriminator_weights.h5")
gen_weights, dis_weights =  generator.get_weights(), discriminator.get_weights()

# print(gen_weights[0][0], type(gen_weights[0][0]), dis_weights[0][0] , type(dis_weights[0][0]))

def initialize_model_array_to_list(weights):
    # Transfer Learning
    generator.set_weights(weights[0])
    g = generator.get_weights()
    g = [layer.tolist() for layer in g]  # Convert ndarray to list
    print(f"Generator Initialized with: ... {type(g[0][0])}" )

    discriminator.set_weights(weights[1])
    d = discriminator.get_weights()
    d = [layer.tolist() for layer in d]  # Convert ndarray to list
    print(f"Discriminator Initialized with: ... {type(d[0][0])}")

    return [g, d]


def aggregate(global_model_weights, model_weights_list):

    num_parties = len(model_weights_list)
    averaged_weights = global_model_weights.copy()

    # EDIT : converted average weight to ndarray
    averaged_weights = np.array(averaged_weights)
    
    for weights in model_weights_list:
        # Perform aggregation (e.g., averaging)
        for layer in range(len(weights)):
            averaged_weights[layer] += np.array(weights[layer]) / num_parties
    
    print("Returned avg weights", averaged_weights)
    # Return the updated global model weights
    return averaged_weights.tolist()


@app.route('/', methods=['POST'])
def home():
    print("Server is Running")

@app.route('/update_model', methods=['POST'])
def update_model():
    global global_model_weights_generator
    global G_model_weight_list_generator

    global global_model_weights_discriminator
    global G_model_weight_list_discriminator

    # Get model weights from the party
    model_weights = request.json['model_weights']
    G_model_weight_list_generator.append(model_weights[0])
    G_model_weight_list_discriminator.append(model_weights[1])

    # # Performs aggrigation to update global model weights
    if len(G_model_weight_list_generator) == num_parties:
        # global_model_weights_generator = np.array(global_model_weights_generator)
        # G_model_weight_list_generator = np.array(G_model_weight_list_generator)
        # print(global_model_weights_generator.shape,G_model_weight_list_generator.shape)
        global_model_weights_generator = aggregate(global_model_weights_generator, G_model_weight_list_generator)
        global_model_weights_discriminator = aggregate(global_model_weights_discriminator, G_model_weight_list_discriminator)

        G_model_weight_list_generator = []
        G_model_weight_list_discriminator = []

    # print(f"generator weights : {type(global_model_weights_generator)}")
    # print(f"discriminator weights : {type(global_model_weights_discriminator)}")    
    print(type(G_model_weight_list_generator))
    print(type(G_model_weight_list_discriminator))
    
    print('Model updated successfully!')
    return 'Model updated successfully!'
@app.route('/get_global_weights', methods=['GET'])
def get_global_weights():
    global global_model_weights_generator
    global global_model_weights_discriminator
    print("default global_model_weights_generator : ", global_model_weights_generator)
    if global_model_weights_generator is None:
        # global_model_weights_generator = initialize_model_array_to_list([gen_weights, dis_weights]) 
        global_model_weights_generator = gen_weights
    if global_model_weights_discriminator is None:
        global_model_weights_discriminator = dis_weights
    # print("here ",np.array(gen_weights).shape)
    
    response = {'updated_model_weights': initialize_model_array_to_list([gen_weights, dis_weights])}
    json_response = jsonify(response)
    return json_response


if __name__ == '__main__':
    app.run()
